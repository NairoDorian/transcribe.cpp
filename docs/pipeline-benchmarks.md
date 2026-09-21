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

## Rendering a summary

```powershell
uv run --no-project scripts/bench/report.py --summary build/bench/current/summary.json
```

Writes `report.md` and `report.csv` beside the summary. The Markdown table is
for pasting into a discussion; the CSV is for diffing two runs column by column.
The per-chunk stage columns come from the streaming chunk records and are blank
for batch cases, which have no per-chunk pipeline to break down.

## Comparing two libraries

A suite measures one library, so it cannot answer "is this tree slower than
that one?". Use the paired runner for that, which alternates the arms within
each repetition and rotates which arm starts:

```powershell
uv run --no-project scripts/bench/compare.py `
  --wav samples/jfk.wav --model <installed.gguf> --backend cpu `
  --arm ref=../transcribe_benchmarks/build/bench-native/install/bin/transcribe.dll `
  --arm fork=build/bench-native/install/bin/transcribe.dll
# Repeatable --arm; the first is the baseline. --reps (min 2), --stream-chunk-ms,
# --att-right, --threads, --language, --backend, --output.
```

This is not a convenience wrapper around the suite; it exists because a suite's
cross-arm verdicts are not trustworthy. A suite runs one arm's cases back to
back, so machine drift lands entirely inside whichever arm ran second. Measured
on this fixture: Nemotron Q6 CPU batch read 1090.3 ms (reference) against
1208.4 ms (fork) in a suite — an apparent 10.8% regression — while the same
pair interleaved read 937.2 against 809.3, the fork 13.6% faster. The same
library moved 1090 → 937 ms (18%) between the two runs. Use `suite.py
--baseline` to track one tree over time and `compare.py` for any claim about
one tree against another.

`compare.py` loads each arm through `--bindings`, defaulting to the bindings of
the checkout that owns that library, because generated bindings are ABI
specific: a foreign library fails at import time on the first symbol it does
not export. That is a hard error, never something to silently time; the arm's
log is printed when it fails. The verdict is conservative by construction — a
difference counts only when it exceeds the larger arm's own within-arm spread,
the noise floor that case actually exhibits rather than a fixed guess. Anything
smaller is reported as within-noise and must not justify a change.

Always keep the arms on the same WAV, weights, language, backend and build
profile. A reported `spread` far larger than its neighbours means the machine
was not idle; discard that run.

Some optimizations here are gated at run time rather than build time, so they
can be compared without rebuilding between arms. `--arm-env NAME=KEY=VALUE`
sets an environment variable for one arm only, and the gate each arm ran under
is recorded in the JSON:

```powershell
uv run --no-project scripts/bench/compare.py `
  --wav samples/jfk.wav --model <installed.gguf> --backend cuda `
  --arm off=build/bench-native/install/bin/transcribe.dll `
  --arm on=build/bench-native/install/bin/transcribe.dll `
  --arm-env on=TRANSCRIBE_GRAPH_OPTIMIZER=1
```

A gate compared this way proves nothing unless it actually fired: if the pass
finds nothing to remove it is a literal no-op, and the two arms ran identical
work, so any difference between them is noise. Confirm the gate's own log line
is present in the `on` arm — and absent from `off` — before reading the
verdict. `TRANSCRIBE_GRAPH_OPTIMIZER` reports itself only when it changed
something (`graph optimizer: nodes=N->M`), so its silence is the signal that
the case is not a valid test of it. Use `--reps` at least 3 for a decision;
with two repetitions the spread is the difference of two samples and the
resulting noise floor is far too small.

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
