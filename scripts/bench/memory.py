#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
memory.py - peak-memory harness for one (variant, sample) cell at a time.

Every memory branch in notes/memory-reduction-plan.md is accepted on a number,
not on "it got better", and those numbers are only comparable if they come from
the same harness, run one process at a time (a 16 GB host OOMs and corrupts
peaks otherwise). This is that harness: it runs transcribe-cli serially, records
peak RSS, the platform's own footprint number, and the scheduler scratch the
run committed, and appends one CSV row per cell.

Usage:
    # one variant, one sample
    uv run scripts/bench/memory.py --models granite-4.0-1b-speech --samples jfk

    # a family sweep at the plan's reference length, warm (2 runs, keep the 2nd)
    uv run scripts/bench/memory.py \\
        --models granite-4.0-1b-speech,granite-speech-4.1-2b,canary-qwen-2.5b \\
        --samples dots-full --repeat 2

    # duration scaling (generates a cut with ffmpeg if missing)
    uv run scripts/bench/memory.py --models sensevoice --samples dots-full \\
        --duration 1200

    # A/B a branch-specific switch; --env for library env vars, `--` for CLI flags
    uv run scripts/bench/memory.py --models granite-4.0-1b-speech --samples jfk \\
        --label pack-gate-up-copy --env TRANSCRIBE_NO_GATE_UP_ALIAS=1
    uv run scripts/bench/memory.py --models qwen3-asr-0.6b --samples jfk \\
        --backend cpu -- --n-ctx 4096

Everything after `--` is passed straight to transcribe-cli, so branch-specific
CLI switches work without teaching this script about them.

Output: reports/memory/<machine-slug>/<label>.csv, one row per cell:

    variant,quant,backend,sample,audio_s,model_mb,peak_rss_mb,
    peak_footprint_mb,peak_anon_mb,scratch_mb,load_ms,wall_s,rc,
    transcript_sha256,notes

Columns that the host cannot measure are left empty rather than faked:
peak_footprint_mb is macOS (and Linux where /usr/bin/time -v reports it),
peak_anon_mb is Linux only (sampled RssAnon - what the Android OOM killer
charges). scratch_mb is the sum of the per-backend scheduler buffer high-water
marks (transcribe-backend.h, wave 0 instrumentation); it is the portable number
for graph cost, since RSS under-reports GPU-backed scratch on Metal.

`--report last` (default) keeps the last repeat of a cell, `min` keeps the
smallest peak - use `min` for noisy hosts, and never mix the two in one table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import detect_machine, get_git_sha, resolve_model_tokens  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CLI = REPO / "build" / "bin" / "transcribe-cli"
DEFAULT_QUANTS = ["q8_0"]

CSV_FIELDS = [
    "variant",
    "quant",
    "backend",
    "sample",
    "audio_s",
    "model_mb",
    "peak_rss_mb",
    "peak_footprint_mb",
    "peak_anon_mb",
    "scratch_mb",
    "load_ms",
    "wall_s",
    "rc",
    "transcript_sha256",
    "notes",
]


# -----------------------------------------------------------------------------
# samples
# -----------------------------------------------------------------------------


def audio_seconds(path: Path) -> int | None:
    """Duration in seconds, via ffprobe (or a WAV header read as a fallback)."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return round(float(proc.stdout.strip()))
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        pass
    return wav_seconds(path)


def wav_seconds(path: Path) -> int | None:
    """Duration of a PCM WAV from its header, without ffprobe."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos = 12
    byte_rate = 0
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        chunk_size = int.from_bytes(data[pos + 4:pos + 8], "little")
        if chunk_id == b"fmt " and pos + 16 <= len(data):
            byte_rate = int.from_bytes(data[pos + 16:pos + 20], "little")
        if chunk_id == b"data" and byte_rate:
            return round(chunk_size / byte_rate)
        pos += 8 + chunk_size + (chunk_size & 1)
    return None


def resolve_sample(repo: Path, stem: str, duration: int | None) -> tuple[str, Path]:
    """(label, path). --duration cuts (and caches) a prefix of the sample."""
    path = repo / "samples" / f"{stem}.wav"
    if not path.exists() and not path.suffix:
        path = repo / "samples" / f"{stem}"
    if not path.exists():
        raise SystemExit(f"error: sample not found: {path}")
    if duration is None:
        return stem, path

    total = audio_seconds(path)
    if total is not None and duration >= total:
        return stem, path

    cache_dir = repo / "tmp" / "memory-samples"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cut = cache_dir / f"{stem}-{duration}s.wav"
    if not cut.exists():
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-t", str(duration),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(cut)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(f"error: ffmpeg could not cut {path} to {duration}s: {proc.stderr.strip()}")
    return f"{stem}@{duration}s", cut


# -----------------------------------------------------------------------------
# measurement
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class Cell:
    variant: str
    quant: str
    backend: str
    sample: str
    audio_s: int | None = None
    model_mb: int = 0
    peak_rss_mb: int = 0
    peak_footprint_mb: int = 0
    peak_anon_mb: int = 0
    scratch_mb: int = 0
    load_ms: int = 0
    wall_s: float = 0.0
    rc: int = 0
    transcript_sha256: str = ""
    notes: list[str] = field(default_factory=list)

    def row(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "quant": self.quant,
            "backend": self.backend,
            "sample": self.sample,
            "audio_s": self.audio_s if self.audio_s is not None else "",
            "model_mb": self.model_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "peak_footprint_mb": self.peak_footprint_mb or "",
            "peak_anon_mb": self.peak_anon_mb or "",
            "scratch_mb": self.scratch_mb or "",
            "load_ms": self.load_ms or "",
            "wall_s": f"{self.wall_s:.2f}",
            "rc": self.rc,
            "transcript_sha256": self.transcript_sha256,
            "notes": ";".join(self.notes),
        }


class AnonSampler:
    """Sample a process's RssAnon on Linux (no equivalent on macOS).

    peak RSS already comes from the kernel's high-water mark; anonymous is the
    share an OOM killer charges, and it only exists as a live counter.
    """

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.peak_kb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> AnonSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        status = Path(f"/proc/{self.pid}/status")
        while not self._stop.is_set():
            try:
                for line in status.read_text().splitlines():
                    if line.startswith("RssAnon:"):
                        self.peak_kb = max(self.peak_kb, int(line.split()[1]))
                        break
            except OSError:
                return
            self._stop.wait(0.05)


def time_argv() -> tuple[list[str], str]:
    """The extern time(1) wrapper and how to read its output.

    BSD/macOS `/usr/bin/time -l` reports "maximum resident set size" and "peak
    memory footprint"; GNU `/usr/bin/time -v` reports a resident high-water mark
    and its own minor/major page faults. Whichever is installed, use it.
    """
    if platform.system() == "Darwin":
        return ["/usr/bin/time", "-l"], "bsd"
    if Path("/usr/bin/time").exists():
        return ["/usr/bin/time", "-v"], "gnu"
    return [], "none"


TIME_RE = {
    "rss": re.compile(r"^\s*(\d+)\s+maximum resident set size", re.MULTILINE),
    "footprint": re.compile(r"^\s*(\d+)\s+peak memory footprint", re.MULTILINE),
    "gnu_rss": re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)", re.MULTILINE),
    "real": re.compile(r"^\s*(\d+\.\d+)\s+real", re.MULTILINE),
}

SCRATCH_RE = re.compile(r"sched buffer (\S+) -> ([\d.]+) MB")
BACKEND_RE = re.compile(r"using (\S+) backend: (\S+)")
TIMINGS_RE = re.compile(r"timings: ([^\n]*)")
MODEL_MB_RE = re.compile(r"weights (?:memory-mapped|loaded|streamed): ([\d.]+) MiB")


def run_cell(
    cli: Path,
    gguf: Path,
    sample: Path,
    repeated: int,
    env_overrides: dict[str, str],
    cli_args: list[str],
) -> tuple[Cell, str]:
    """Run one (model, sample) cell `repeated` times; return the reported cell."""
    variant, quant = model_identity(gguf)
    backend = "auto"
    for i, arg in enumerate(cli_args):
        if arg in ("--backend", "-b") and i + 1 < len(cli_args):
            backend = cli_args[i + 1]

    cell = Cell(variant=variant, quant=quant, backend=backend, sample=sample.name)
    cell.audio_s = audio_seconds(sample)
    cell.model_mb = round(gguf.stat().st_size / (1024 * 1024)) if gguf.exists() else 0

    env = dict(os.environ)
    env.update(env_overrides)

    rows: list[Cell] = []
    transcript = ""
    for _ in range(max(1, repeated)):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out:
            out_path = Path(out.name)
        try:
            argv, mode = time_argv()
            argv = [*argv, str(cli), "-m", str(gguf), "-o", str(out_path), *cli_args, str(sample)]

            t0 = time.monotonic()
            proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    text=True, env=env)
            sampler: AnonSampler | None = None
            if mode != "bsd" and platform.system() == "Linux":
                sampler = AnonSampler(proc.pid)
                sampler.__enter__()
            stderr = proc.communicate()[1] or ""
            if sampler is not None:
                sampler.__exit__()
            wall = time.monotonic() - t0

            run = Cell(variant=variant, quant=quant, backend=backend, sample=sample.name)
            run.audio_s = cell.audio_s
            run.model_mb = cell.model_mb
            run.wall_s = wall
            run.rc = proc.returncode

            if mode == "bsd":
                run.peak_rss_mb = to_mb(TIME_RE["rss"].search(stderr))
                run.peak_footprint_mb = to_mb(TIME_RE["footprint"].search(stderr))
            elif mode == "gnu":
                m = TIME_RE["gnu_rss"].search(stderr)
                run.peak_rss_mb = round(int(m.group(1)) / 1024) if m else 0
            if sampler is not None and sampler.peak_kb:
                run.peak_anon_mb = round(sampler.peak_kb / 1024)

            scratch: dict[str, float] = {}
            for backend_name, mb in SCRATCH_RE.findall(stderr):
                scratch[backend_name] = max(scratch.get(backend_name, 0.0), float(mb))
            run.scratch_mb = round(sum(scratch.values()))

            m = TIMINGS_RE.search(stderr)
            if m:
                lm = re.search(r"load=([\d.]+)", m.group(1))
                run.load_ms = round(float(lm.group(1))) if lm else 0
            else:
                mm = MODEL_MB_RE.search(stderr)
                if mm:
                    run.model_mb = round(float(mm.group(1)))

            # The CLI's --backend value may be "auto": report what was chosen.
            mb_match = BACKEND_RE.search(stderr)
            if mb_match:
                run.backend = mb_match.group(2)

            notes = [f"{name}={mb:.0f}MB" for name, mb in sorted(scratch.items())]
            if "gate+up aliased" in stderr:
                notes.append("gate_up=aliased")
            elif "gate+up packed by copying" in stderr:
                notes.append("gate_up=copied")
            if "repeating" in stderr:
                notes.append("repetition-guard")
            if run.rc != 0:
                notes.append(f"rc={run.rc}")
            run.notes = notes

            try:
                transcript = out_path.read_text()
            except OSError:
                transcript = ""
            run.transcript_sha256 = hashlib.sha256(transcript.encode()).hexdigest()[:16]
            rows.append(run)
        finally:
            out_path.unlink(missing_ok=True)

    keep = rows[-1]
    keep.notes = list(rows[-1].notes)
    return keep, transcript


def to_mb(match: re.Match[str] | None) -> int:
    return round(int(match.group(1)) / (1024 * 1024)) if match else 0


def model_identity(gguf: Path) -> tuple[str, str]:
    variant = gguf.parent.name if gguf.parent.name else gguf.stem
    m = re.search(r"-(F32|F16|BF16|Q[0-9A-Z_]+)\.gguf$", gguf.name)
    return variant, (m.group(1) if m else "?")


# -----------------------------------------------------------------------------
# reporting
# -----------------------------------------------------------------------------


def merge_row(previous: list[dict[str, str]], row: dict[str, object], report: str) -> list[dict[str, str]]:
    """Replace a cell's earlier row. `min` keeps the smaller peak of the two."""
    out = [r for r in previous
           if not (r["variant"], r["quant"], r["backend"], r["sample"])
           == (row["variant"], row["quant"], row["backend"], row["sample"])]
    for old in previous:
        same = (old["variant"], old["quant"], old["backend"], old["sample"]) == (
            row["variant"], row["quant"], row["backend"], row["sample"])
        if not same:
            continue
        if report == "min" and int(old["peak_rss_mb"] or 0) and int(old["peak_rss_mb"] or 0) < int(row["peak_rss_mb"] or 0):
            out.append(old)
            return out
    out.append({k: str(v) for k, v in row.items()})
    return out


def print_table(rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    header = ["variant", "quant", "backend", "sample", "audio_s", "model_mb",
              "rss_mb", "footprint_mb", "anon_mb", "scratch_mb", "load_ms", "wall_s", "rc", "notes"]
    widths = [len(h) for h in header]
    body = []
    for r in rows:
        line = [r["variant"], r["quant"], r["backend"], r["sample"], r["audio_s"], r["model_mb"],
                r["peak_rss_mb"], r["peak_footprint_mb"], r["peak_anon_mb"], r["scratch_mb"],
                r["load_ms"], r["wall_s"], r["rc"], r["notes"]]
        body.append(line)
        widths = [max(w, len(str(v))) for w, v in zip(widths, line)]

    def fmt(line: list[str]) -> str:
        cells = []
        for w, v in zip(widths, line):
            s = str(v)
            cells.append(s.rjust(w) if w == len(str(v)) and s.replace(".", "").isdigit() else s.ljust(w))
        return "  ".join(cells).rstrip()

    print(fmt(header))
    for line in body:
        print(fmt(line))


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", help="variant dirs under models/, *.gguf paths, or comma-separated (default: all with the chosen quants)")
    p.add_argument("--samples", default="dots-full", help="comma-separated sample stems (default: dots-full)")
    p.add_argument("--quants", default=",".join(DEFAULT_QUANTS), help="comma-separated quant filter for --models dirs")
    p.add_argument("--duration", type=int, default=None,
                   help="cut the sample to N seconds (cached under tmp/memory-samples/)")
    p.add_argument("--repeat", type=int, default=1,
                   help="runs per cell; the last one is reported (2 for a warm number)")
    p.add_argument("--report", choices=("last", "min"), default="last",
                   help="which repeat to keep across invocations of the same cell")
    p.add_argument("--backend", default=None, help="transcribe-cli --backend value (default: auto)")
    p.add_argument("--label", default=None, help="CSV name (default: <git sha>)")
    p.add_argument("--out-dir", type=Path, default=None, help="output root (default: reports/memory)")
    p.add_argument("--cli", type=Path, default=DEFAULT_CLI, help=f"transcribe-cli path (default: {DEFAULT_CLI})")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="library env var for the run; repeatable")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args, cli_args = p.parse_known_args()
    if cli_args and cli_args[0] == "--":
        cli_args = cli_args[1:]
    if args.backend:
        cli_args = ["--backend", args.backend, *cli_args]

    cli = args.cli if args.cli.is_absolute() else (REPO / args.cli)
    if not cli.exists():
        print(f"error: {cli} not found; build it with:\n  cmake --build build --target transcribe-cli",
              file=sys.stderr)
        return 2

    env_overrides: dict[str, str] = {}
    for item in args.env:
        if "=" not in item:
            print(f"error: --env wants KEY=VALUE, got {item!r}", file=sys.stderr)
            return 2
        key, value = item.split("=", 1)
        env_overrides[key] = value

    models = resolve_model_tokens(REPO, [t for t in args.models.split(",") if t] or None,
                                 {q.strip().lower() for q in args.quants.split(",") if q.strip()})
    samples = [resolve_sample(REPO, s.strip(), args.duration)
               for s in args.samples.split(",") if s.strip()]
    if not models:
        print("error: no models matched", file=sys.stderr)
        return 2

    machine = detect_machine()
    label = args.label or get_git_sha(REPO)
    out_dir = (args.out_dir or (REPO / "reports" / "memory")) / machine["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{label}.csv"

    print(f"memory.py: {len(models)} model(s) x {len(samples)} sample(s) on {machine['cpu_model']} ({machine['os']})")
    print(f"  binary {cli.relative_to(REPO) if cli.is_relative_to(REPO) else cli}")
    print(f"  serialized: one transcribe-cli at a time (concurrent runs corrupt peaks)")
    if args.dry_run:
        for variant, quant, path in models:
            for name, sample in samples:
                print(f"  {variant} {quant} {name} -> {path.relative_to(REPO)}")
        return 0

    rows: list[dict[str, str]] = []
    if out_csv.exists():
        with out_csv.open(newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]

    try:
        for variant, quant, path in models:
            for name, sample in samples:
                cell, _ = run_cell(cli, path, sample, args.repeat, env_overrides, cli_args)
                cell.variant = variant
                cell.quant = quant
                cell.sample = name
                rows = merge_row(rows, cell.row(), args.report)
                print(f"  {variant:32s} {quant:7s} {name:16s} rss {cell.peak_rss_mb:6d} MB"
                      f"  footprint {cell.peak_footprint_mb or 0:6d} MB"
                      f"  scratch {cell.scratch_mb or 0:5d} MB  {cell.wall_s:7.2f}s  rc {cell.rc}")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\ninterrupted; writing what we have", file=sys.stderr)

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        rows.sort(key=lambda r: (r["variant"], r["sample"], r["quant"]))
        writer.writerows(rows)

    print()
    print_table(rows)
    print(f"\nwrote {out_csv.relative_to(REPO) if out_csv.is_relative_to(REPO) else out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
