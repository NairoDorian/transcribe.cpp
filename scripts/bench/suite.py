"""Installed-model regression suite: CPU and CUDA, 1 warm-up + 2 measured runs.

No downloads. Each case runs in a separate process with a timeout and raw logs.
Use --only to select a subset. --baseline compares like-for-like case averages.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys


def run_case(cmd, log, timeout):
    # uv launches a child interpreter. Terminate our whole process tree on a
    # timeout so a stranded inference cannot contaminate subsequent timings.
    with subprocess.Popen(cmd, stdout=log, stderr=log,
                          start_new_session=os.name != "nt") as process:
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=log, stderr=log, check=False)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, cmd)

PROFILES = {
    "granite": ("models--handy-computer--granite-speech-4.1-2b-gguf", ["*Q4_K_M.gguf"]),
    "qwen": ("models--handy-computer--Qwen3-ASR-1.7B-gguf", ["*.gguf"]),
    "nemotron": ("models--handy-computer--nemotron-3.5-asr-streaming-0.6b-gguf", ["*Q6_K.gguf", "*Q8_0.gguf"]),
    "parakeet": ("models--handy-computer--parakeet-tdt-0.6b-v3-gguf", ["*Q4_K_M.gguf", "*Q8_0.gguf"]),
}

# Streaming feed sizes per profile, in milliseconds of PCM per feed; 0 means
# batch only. Coverage is governed by the model's declared capability, not by
# preference: the runtime derives caps.supports_streaming from encoder
# attention geometry (src/arch/parakeet/model.cpp) and gates stream_begin on it
# (src/transcribe.cpp), so an offline checkpoint such as parakeet-tdt-0.6b-v3
# has no streaming path to measure at all — asking for one fails rather than
# producing a number. Add a profile here only when its GGUF streams.
STREAM_MODES = {"nemotron": [0, 16]}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", required=True, type=Path)
    p.add_argument("--bindings", type=Path)
    p.add_argument("--wav", required=True, type=Path)
    p.add_argument("--cache", type=Path, default=Path(os.environ.get("HF_HUB_CACHE", Path.home()/".cache/huggingface/hub")))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--only", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--max-regression", type=float, default=0.15)
    args = p.parse_args()
    if args.timeout <= 0 or args.max_regression < 0 or args.threads < 0:
        p.error("timeout must be positive; regression budget and threads must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, machine=platform.platform(), python=sys.version,
                  policy="3 runs: discard run 1; arithmetic mean of runs 2 and 3",
                  cases=[], missing=[], failures=[])
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else {}
    old_cases = {case["case"]: case for case in baseline.get("cases", [])}
    for name in args.only:
        repo, patterns = PROFILES[name]
        selected = {}
        repositories = [repo]
        if name == "qwen":
            repositories.append("models--handy-computer--Qwen3-ASR-0.6B-gguf")
        for pattern in patterns:
            for repository in repositories:
                for file in sorted((args.cache/repository/"snapshots").glob("*/"+pattern)):
                    if file.is_file() and file.stat().st_size > 1024:
                        selected[file.name] = file
        if not selected:
            report["missing"].append(name)
        for filename, model in selected.items():
            modes = STREAM_MODES.get(name, [0])
            for backend in ("cpu", "cuda"):
                for chunk in modes:
                    case = f"{model.stem}.{backend}.{'stream' if chunk else 'batch'}"
                    output = args.output/(case+".json")
                    cmd = ["uv", "run", "--no-project", str(Path(__file__).with_name("pipeline.py")),
                           "--model", str(model), "--wav", str(args.wav.resolve()),
                           "--library", str(args.library.resolve()), "--backend", backend,
                           "--language", "auto" if name == "nemotron" else "en",
                           "--threads", str(args.threads), "--output", str(output)]
                    if args.bindings:
                        cmd += ["--bindings", str(args.bindings.resolve())]
                    if chunk:
                        cmd += ["--stream-chunk-ms", str(chunk), "--att-right", "6"]
                    print(case, flush=True)
                    try:
                        with (args.output/(case+".log")).open("w", encoding="utf-8") as log:
                            run_case(cmd, log, args.timeout)
                        data = json.loads(output.read_text(encoding="utf-8"))
                        if len(data["runs"]) != 3:
                            raise ValueError("expected exactly three runs")
                        if any(run["text"] != data["runs"][1]["text"] for run in data["runs"][2:]):
                            raise ValueError("measured runs produced different transcripts")
                        row = dict(case=case, warm_mean_ms=data["warm_mean_ms"],
                                   warm_mean_rtf=data["warm_mean_rtf"], timings=data["warm_mean_timings"],
                                   # Absent when the case ran through an upstream ABI whose
                                   # pipeline.py predates per-stage capture; the report
                                   # renders the column blank rather than failing.
                                   stage_metrics=data.get("stage_metrics"),
                                   build_id=data["build_id"], backend=data["backend"],
                                   model_bytes=data["model_bytes"], audio_ms=data["audio_ms"],
                                   language=data["language"], threads=data["threads"],
                                   wav_sha256=data["wav_sha256"],
                                   stream_chunk_ms=data["stream_chunk_ms"], att_right=data["att_right"],
                                   text=data["runs"][-1]["text"], raw=str(output))
                        if case in old_cases:
                            old = old_cases[case]
                            for key in ("backend", "model_bytes", "audio_ms", "language", "threads"):
                                if old.get(key) != row[key]:
                                    raise ValueError(f"baseline mismatch for {key}")
                            for key in ("wav_sha256", "stream_chunk_ms", "att_right"):
                                if key in old and old[key] != row[key]:
                                    raise ValueError(f"baseline mismatch for {key}")
                            row["ratio_to_baseline"] = row["warm_mean_ms"]/old["warm_mean_ms"]
                            row["transcript_matches_baseline"] = row["text"] == old["text"]
                            if row["ratio_to_baseline"] > 1+args.max_regression:
                                report["failures"].append(dict(case=case, error="timing regression"))
                            if not row["transcript_matches_baseline"]:
                                report["failures"].append(dict(case=case, error="transcript changed; review required"))
                        report["cases"].append(row)
                    except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
                        report["failures"].append(dict(case=case, error=str(error)))
                    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    # Human-readable forms of the same data. A reporter failure must not hide
    # the measurements, so the summary above is already durable.
    try:
        subprocess.run(["uv", "run", "--no-project", str(Path(__file__).with_name("report.py")),
                        "--summary", str(args.output/"summary.json")], check=True)
    except (subprocess.SubprocessError, OSError) as error:
        print(f"report.py failed: {error}", file=sys.stderr)
    return 1 if report["failures"] or not report["cases"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
