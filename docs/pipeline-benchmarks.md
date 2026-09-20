# CPU and CUDA pipeline regression benchmarks

These local benchmarks never download model weights. They use installed GGUFs
and a mono, 16 kHz, PCM16 WAV. Every case loads a model once, executes exactly
three runs on the same session, retains run 1 as excluded warm-up evidence, and
scores the arithmetic mean of runs 2 and 3. Cold model load is separate.

```powershell
# One model, one backend; repeat with --backend cuda.
uv run --no-project scripts/bench/pipeline.py --model <model.gguf> `
  --library <absolute-path-to-transcribe.dll> --wav samples/jfk.wav `
  --backend cpu --output build/bench/single.json
# Nemotron streaming: append --stream-chunk-ms 16 --att-right 6

# Installed subset, BOTH CPU and CUDA; no downloads.
uv run --no-project scripts/bench/suite.py --library <transcribe.dll> `
  --wav samples/jfk.wav --output build/bench/current
# Optional: --only nemotron parakeet
# Regression gate: --baseline build/bench/baseline/summary.json
```

Default profiles: Granite Speech 4.1 2B Q4, installed Qwen3-ASR 1.7B/0.6B,
Nemotron 3.5 Q6/Q8, Parakeet TDT 0.6B v3 Q4/Q8. Missing families are listed;
missing CUDA is a failure, never a CPU substitution. A case has a timeout and
its own process/log. Timeout cleanup also kills the owned uv child interpreter.
Use the single-model runner for any explicit model path or alternate quant.
Its language default is `auto`; pass `--language en` for an English constraint.
The subset uses `auto` for monolingual Nemotron (upstream rejects explicit
language hints there) and `en` for the other profiles.

JSON includes the exact library/build identity, model size, WAV SHA256,
language, threads, backend, model load, WAV conversion, wall time, native
mel/encoder/decoder stages, all three transcripts, and streaming begin/feed/
finalize durations. RTF is compute time / audio duration (lower is better).
`first_text_compute_ms` is unpaced replay compute time, not microphone latency.
Native timings on an upstream build retain that build's accounting, including
any known omissions; compare wall time when timer definitions differ.

The default regression budget is 15%, configurable with `--max-regression`.
The suite also rejects mismatched available configuration fingerprints or
changed transcripts. It records failures without hiding slow cases. For an
older ABI use `--bindings <matching-repo>/bindings/python/src`; never disable
ABI checking. Do not mix GGML modules from different source versions in release
or reference builds.

Keep the machine idle and on the same power profile. Run suites sequentially,
with no builds, other inference, or graphics workloads. If conditions change,
retain the contaminated report and rerun the complete three-run cases. One
short fixture is a regression smoke test, not a WER or long-stream acceptance
test. Add representative short/long/silent/multilingual WAVs before release.

ZER0 has the equivalent app replay runner in `scripts/bench-stt.ts`; its
`docs/STT_BENCHMARKS.md` covers native-library selection and live pipeline logs.
The app replay covers its loaded engine and settings but bypasses microphone,
VAD, denoise, UI and clipboard. Live pipeline logs measure those separately.
