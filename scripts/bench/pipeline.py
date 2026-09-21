"""Benchmark one installed GGUF through the public API; never download models.

Run with uv run --no-project scripts/bench/pipeline.py --help.
Use --library to select the exact native DLL/shared library under investigation.
"""
from __future__ import annotations

import argparse
import array
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time
import wave


# --- per-stage streaming metrics ---------------------------------------------
#
# The native library emits one line per streaming chunk from
# emit_streaming_chunk (src/arch/parakeet/model.cpp):
#
#   parakeet stream chunk 7: total=12.3 ms  graph_build=0.4 ms  sched_alloc=0.1 ms
#   graph_compute=9.8 ms  readback=0.2 ms  cache_rot=1.1 ms  decoder=0.5 ms
#   other=0.2 ms (backend=CUDA0, threads=6, T_q=17, T_cache=70, kv_mode=1, n_layers=24)
#
# transcribe::log_msg applies no level filter (src/transcribe-log.h): the line
# is delivered to whatever sink is installed, so tapping the sink in-process
# recovers the full pipeline breakdown without a native rebuild. The tap also
# forwards every line to stderr, so the raw case log keeps the original
# evidence. Both trees share this harness verbatim, which is what makes the
# per-stage numbers directly comparable between fork and upstream.
STAGES = ("total", "graph_build", "sched_alloc", "graph_compute", "readback", "cache_rot", "decoder", "other")
_STAGE_LINE = re.compile(
    r"^parakeet stream chunk (?P<chunk>\d+): "
    r"total=(?P<total>[\d.]+) ms\s+graph_build=(?P<graph_build>[\d.]+) ms\s+"
    r"sched_alloc=(?P<sched_alloc>[\d.]+) ms\s+graph_compute=(?P<graph_compute>[\d.]+) ms\s+"
    r"readback=(?P<readback>[\d.]+) ms\s+cache_rot=(?P<cache_rot>[\d.]+) ms\s+"
    r"decoder=(?P<decoder>[\d.]+) ms\s+other=(?P<other>[\d.]+) ms\s+"
    r"\(backend=(?P<backend>[^,]+), threads=(?P<threads>\d+), T_q=(?P<T_q>-?\d+), "
    r"T_cache=(?P<T_cache>-?\d+), kv_mode=(?P<kv_mode>\d+), n_layers=(?P<n_layers>\d+)\)")


class StageCapture:
    """Thread-safe log sink that segments per-chunk stage records by run.

    The callback contract allows invocation from ggml worker threads, so the
    record list is guarded by a lock. Rows are attributed to the run that is
    currently executing, which is what lets the "discard run 1, mean of runs
    2-3" policy be applied to stage metrics as well as wall time.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._run = 0
        self.rows = {}
        self.geometry = None

    def begin_run(self, index):
        with self._lock:
            self._run = index
            self.rows.setdefault(index, [])

    def __call__(self, level, message):
        print(message, file=sys.stderr, flush=True)
        match = _STAGE_LINE.match(message)
        if match is None:
            return
        values = match.groupdict()
        row = {stage: float(values[stage]) for stage in STAGES}
        with self._lock:
            self.rows.setdefault(self._run, []).append(row)
            self.geometry = {"backend": values["backend"].strip(), "threads": int(values["threads"]),
                             "T_q": int(values["T_q"]), "T_cache": int(values["T_cache"]),
                             "kv_mode": int(values["kv_mode"]), "n_layers": int(values["n_layers"])}


def _percentile(values, quantile):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * quantile))]


def stage_summary(rows):
    """Aggregate parsed chunk rows into per-stage mean/p50/p95/max/sum."""
    if not rows:
        return None
    return {"chunks": len(rows),
            "stages_ms": {stage: {"mean": statistics.mean(row[stage] for row in rows),
                                  "p50": _percentile([row[stage] for row in rows], 0.50),
                                  "p95": _percentile([row[stage] for row in rows], 0.95),
                                  "max": max(row[stage] for row in rows),
                                  "sum": sum(row[stage] for row in rows)}
                          for stage in STAGES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--bindings", type=Path, help="Alternate bindings/python/src for an upstream ABI")
    parser.add_argument("--stream-chunk-ms", type=int, default=0)
    parser.add_argument("--att-right", type=int)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    language = None if args.language == "auto" else args.language
    if args.stream_chunk_ms < 0 or args.threads < 0:
        parser.error("chunk size and threads must be nonnegative")
    os.environ["TRANSCRIBE_LIBRARY"] = str(args.library.resolve())
    sys.path.insert(0, str(args.bindings or Path(__file__).resolve().parents[2] / "bindings/python/src"))
    import transcribe_cpp as tc

    # Install before the model is created: the native contract requires the
    # log sink to be set once at startup, before threads or models exist.
    capture = StageCapture()
    tc.set_log_callback(capture)

    start = time.perf_counter()
    with wave.open(str(args.wav), "rb") as wav:
        if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (1, 16000, 2):
            parser.error("WAV must be mono 16 kHz PCM16 (use a samples/ fixture)")
        pcm16 = array.array("h", wav.readframes(wav.getnframes()))
        if sys.byteorder != "little":
            pcm16.byteswap()
    pcm = array.array("f", (x / 32768.0 for x in pcm16))
    wav_ms = (time.perf_counter() - start) * 1000
    if not pcm:
        parser.error("empty WAV")
    report = dict(schema_version=1, library=str(args.library.resolve()),
                  library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                  architecture_modules={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in sorted(args.library.parent.glob("*transcribe-arch-*"))
                                        if path.suffix in (".dll", ".so", ".dylib")},
                  # The native identity string opens with a newline: it is built
                  # as a blank-line-prefixed banner. Strip it here so a stored
                  # identity never begins with whitespace, which would break the
                  # alignment of any report that prints it next to a label.
                  build_id=(tc._lib.transcribe_build_id().decode()
                            if hasattr(tc._lib, "transcribe_build_id")
                            else "upstream commit " + tc._lib.transcribe_version_commit().decode()).strip(),
                  model=str(args.model.resolve()), model_bytes=args.model.stat().st_size,
                  wav=str(args.wav.resolve()), wav_sha256=hashlib.sha256(args.wav.read_bytes()).hexdigest(),
                  audio_ms=len(pcm) / 16,
                  wav_read_ms=wav_ms, backend_requested=args.backend,
                  threads=args.threads, stream_chunk_ms=args.stream_chunk_ms,
                  att_right=args.att_right, language=args.language,
                  environment={k: v for k, v in os.environ.items()
                               if k.startswith(("TRANSCRIBE_", "GGML_"))}, runs=[])
    start = time.perf_counter()
    with tc.Model(args.model, backend=args.backend) as model:
        report.update(load_ms=(time.perf_counter() - start) * 1000,
                      backend=model.backend, variant=model.variant)
        with model.session(n_threads=args.threads) as session:
            for index in range(3):
                capture.begin_run(index)
                start = time.perf_counter()
                row = dict(run=index + 1, excluded_warmup=index == 0)
                if args.stream_chunk_ms:
                    extension = (tc.ParakeetStreamOptions(att_context_right=args.att_right)
                                 if args.att_right is not None else None)
                    with session.stream(language=language, family=extension) as stream:
                        row["begin_ms"] = (time.perf_counter() - start) * 1000
                        feeds = []
                        first_text_ms = None
                        for offset in range(0, len(pcm), args.stream_chunk_ms * 16):
                            tick = time.perf_counter()
                            update = stream.feed(pcm[offset:offset + args.stream_chunk_ms * 16])
                            feeds.append((time.perf_counter() - tick) * 1000)
                            if first_text_ms is None and (update.committed_changed or update.tentative_changed):
                                first_text_ms = (time.perf_counter() - start) * 1000
                        tick = time.perf_counter()
                        stream.finalize()
                        row["finalize_ms"] = (time.perf_counter() - tick) * 1000
                        row.update(feed_ms=feeds, first_text_compute_ms=first_text_ms,
                                   feed_max_ms=max(feeds), feed_p95_ms=sorted(feeds)[int((len(feeds)-1)*.95)])
                        result = stream.snapshot()
                else:
                    result = session.run(pcm, language=language)
                row.update(wall_ms=(time.perf_counter() - start) * 1000,
                           timings=dataclasses.asdict(result.timings), text=result.text)
                row["rtf"] = row["wall_ms"] / report["audio_ms"]
                report["runs"].append(row)
                print(f"run={index + 1} excluded_warmup={index == 0} wall_ms={row['wall_ms']:.1f} rtf={row['rtf']:.3f}", file=sys.stderr, flush=True)
    warm = report["runs"][1:]
    report["warm_mean_ms"] = statistics.mean(r["wall_ms"] for r in warm)
    report["warm_mean_rtf"] = report["warm_mean_ms"] / report["audio_ms"]
    report["warm_mean_timings"] = {key: statistics.mean(r["timings"][key] for r in warm)
                                   for key in warm[0]["timings"] if key != "load_ms"}
    # Stage metrics follow the same policy as wall time: run 1 is warm-up, the
    # reported figures aggregate runs 2 and 3 only.
    stage_rows = [row for index in (1, 2) for row in capture.rows.get(index, [])]
    summary = stage_summary(stage_rows)
    if summary:
        summary["source"] = "in-process tap on the native log sink (emit_streaming_chunk)"
        summary["geometry"] = capture.geometry
        summary["chunks_per_run"] = {f"run{index + 1}": len(capture.rows.get(index, [])) for index in range(3)}
        report["stage_metrics"] = summary
    elif capture.rows:
        report["stage_metrics"] = {"chunks": 0,
                                   "note": "no per-chunk stage lines were emitted by this configuration"}
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
