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
import statistics
import sys
import time
import wave


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
                  build_id=(tc._lib.transcribe_build_id().decode()
                            if hasattr(tc._lib, "transcribe_build_id")
                            else "upstream commit " + tc._lib.transcribe_version_commit().decode()),
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
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
