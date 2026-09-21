# Nemotron / Parakeet performance investigation

## Scope and reproducibility

Measured on an Intel Core i9-13900H / NVIDIA RTX 4070 Laptop GPU (8 GB),
Windows, using the installed model files and the 11-second `samples/jfk.wav`.
Every model/backend/mode loads once and executes exactly three resident runs.
Run 1 is retained as excluded warm-up evidence; scores are the arithmetic mean
of runs 2 and 3. No weights were downloaded. Streaming replays 16 ms feeds
without pacing, with Nemotron right attention context 6.

Native libraries are Release builds. Both app hosts are development builds,
using Release native libraries. CUDA graph replay is off, matching the original
installed CUDA backend. No process/thread priority elevation, CPU binding or
core-class selection is applied by the fixed fork or app.

The original fork DLLs were preserved in `build/diagnostics/before/`.
The fixed native install is `build/diagnostics/install/`; its CUDA module comes
from the original fork build with the same GGML revision. Fresh upstream has
its own separately compiled CPU and CUDA modules, never mixed across GGML
versions. Raw case JSON includes binary identity, inputs, all three timings,
transcripts and native stages; stderr logs accompany it.

## Causes and changes

History inspection places the affinity policy in `3ae7383e` and the persistent
streaming decoder graph in `4f150095`. The latter commit's attempt to relax
affinity did not remove GGML's mask application or idle-pool competition.
`7d727bb9` added chunk profiling without resolving those costs. This attribution
comes from source/history inspection and targeted before/after experiments,
not an exhaustive benchmark of every intervening commit.

1. The fork's nominally non-strict CPU mask still bound workers and the calling
   thread. Caller binding was not restored. Removed topology/P-core ranking,
   masks and binding from owned engine code. Available CPU counting still
   respects restrictions imposed externally by the process environment.
2. Persistent encoder and RNN-T decoder pools polled concurrently. Idle workers
   competed with useful compute during mel, encoder and decoder stages. The
   decoder is CPU work even when the encoder runs on CUDA. Park inactive pools
   with GGML's threadpool setter and reattach cached pools before dispatch;
   retain graph/weight storage and worker reuse. Apply this to true streaming,
   buffered streaming and batch Parakeet/Nemotron decoding.
3. Removing topology selection exposed a Qwen eight-worker default regression.
   The final sweep tested 2, 3, 4, 5, 6 and 8 workers with identical transcripts.
   Five was fastest in that sweep. Qwen's automatic budget now caps at five; explicit caller counts remain
   authoritative. This is a thread budget, with normal OS placement.
4. Streaming encoder timing included the decoder and omitted final work.
   Mel, encode and decode now accumulate separately, including finalization.
   Debug chunk records expose graph build/allocation/compute, readback, cache
   rotation, decoding and other host time.
5. ZER0 followed a pinned Git native dependency, so sibling source edits alone
   did not reach the app. Reinstalled native DLLs could also remain stale beside
   the executable. Prebuilt-directory watchers and development DLL refresh fix
   that path. Startup logs identify the native version/build; measurements
   fingerprint the actual staged binaries.
6. ZER0's file/live log merge could remove legitimate records, and a UTF-8 tail
   boundary could blank the console. It now reads one bounded durable source,
   preserves repeated records and reuses unchanged rows. All levels are
   captured to terminal/file, including when `RUST_LOG=warn` is inherited.
   Removed the capture-level selector; severity chips only hide displayed rows.
7. Removed app process elevation, EcoQoS override, timer-resolution elevation
   and audio MMCSS registration. Native worker priority remains normal.

## Measurement coverage

The installed subset is Granite Speech 4.1 2B Q4_K_M, Qwen3-ASR 1.7B Q5_K_M,
Nemotron 3.5 Q6_K and Q8_0, and Parakeet TDT v3 Q4_K_M. Each is measured on CPU
and CUDA; Nemotron has both batch and streaming cases (14 cases per suite).
The runners also discover installed Qwen 0.6B and Parakeet Q8 variants when
present, without downloading them.

Native runner: `scripts/bench/pipeline.py` for one model, `scripts/bench/suite.py`
for the subset. App runner: `Handy_V2/scripts/bench-stt.ts`, also exposed as
`bun run bench:stt`. See [native instructions](pipeline-benchmarks.md) and the
app's `docs/STT_BENCHMARKS.md`. Gates check mean calculation, transcript
stability, backend/configuration compatibility and a configurable regression
budget (15% default). Failed/noisy earlier reports are retained.

`scripts/bench/report.py` renders any summary as Markdown and CSV, including
the per-chunk stage columns. `scripts/bench/compare.py` is the paired A/B tool
described under "Comparing two libraries" below. All four files are byte
identical in this tree and in `../transcribe_benchmarks`; the app runner emits
the same field names (`warm_mean_timings`, `stage_metrics`, and `bound_backend`
alongside `backend`) so one reporter renders native and app summaries alike.

Live app metrics additionally cover stream queue waits, feed/finalize maxima,
capture consumer/frontend/VAD routing, flush, dropped samples and VAD errors.
Headless replay bypasses microphone arrival, VAD, denoise, UI, clipboard and
Multi-STT merge. Its first-text time is compute latency, not microphone latency.

## Upstream baselines

The requested worktrees share the existing fork repositories and preserve the
complete fetched upstream main histories:

- `../transcribe_benchmarks`, branch `transcribe_benchmarks`, upstream parent
  `be7a8b35e9ba2df20298bd26e32d53407c3bcbcd`.
- `../Handy_benchmarks`, branch `Handy_benchmarks`, upstream parent
  `8f9cf53cd1410cda26beea39ff802ac306e39585`.

Only benchmark tooling/instrumentation is added to those branches. Runtime
performance fixes remain in the working forks. The previously provided
reference folders and the audio.cpp checkout remain read-only.

## GGML / audio.cpp review

The fork already vendors GGML v0.24.0. The requested audio.cpp commits contain
CUDA improvements for particular operation shapes, but do not establish a
direct Nemotron Q6/Q8 gain. No speculative kernels or reduced precision were
imported. See [the performance transfer assessment](audio-cpp-performance-review.md)
for the three commits, history review and prioritized experiments.

## Limits

Transcript parity on these fixtures is a regression smoke check, not a full
WER or tensor-oracle qualification. Short replay does not establish live
microphone performance or memory stability over hours. Always measure a
representative long/multilingual/silent corpus before release. Logging is now
verbose by design and carries I/O cost; benchmark the actual intended capture
policy. These gates catch measured regressions, not every possible future bug.

## Results and final selection (2026-09-20)

Milliseconds for the 11-second fixture; lower is better. Every value averages runs 2 and 3, with run 1 excluded. The original fork is the saved pre-fix DLL set. These are native API measurements.

| Model / mode | Original CPU | Selected CPU | Original CUDA | Selected CUDA |
|---|---:|---:|---:|---:|
| Granite 4.1 2B Q4 batch | 3661.5 | 4123.9 | 389.8 | 389.4 |
| Qwen3 1.7B Q5* batch | 3982.1 | 4144.5 | 295.9 | 274.1 |
| Nemotron Q6 batch | 1720.5 | 840.4 | 170.7 | 167.7 |
| Nemotron Q6 stream | 21942.4 | 1425.4 | 9406.9 | 587.4 |
| Nemotron Q8 batch | 1688.3 | 959.7 | 174.9 | 175.9 |
| Nemotron Q8 stream | 26479.0 | 1574.8 | 8812.4 | 577.8 |
| Parakeet v3 Q4 batch | 1402.1 | 840.3 | 130.4 | 134.7 |

*Qwen uses the separately measured five-worker tuning cohort, now selected as its automatic limit. The earlier complete `verified-after` suite used four and flagged a CPU regression. Five recovered near-baseline CPU throughput; it did not demonstrate a universal speedup over the original fork. The selected source was rebuilt into the native install and ZER0. At the user’s request, no further benchmark or test run was started after the instruction to conclude.

The Nemotron improvement comes primarily from removing affinity manipulation and parking competing CPU pools. CUDA still performs mel and decoder work on CPU, explaining the large CUDA wall-time gain. Small differences, including Parakeet CUDA, are within ordinary run-to-run variation; not every model became faster in every configuration.

### Fresh upstream application comparison

These are the complete app replay suites (`verified-app` and `verified-upstream-app`). They include the app’s loaded session and postprocessing path. Resolved language options differ between the apps in some cases (for example upstream Nemotron uses `en-US`, while ZER0 uses no hint); therefore these are integration measurements, not an isolated native-engine A/B. The ZER0 suite predates the final Qwen five-worker default.

| Model / backend / mode | Upstream Handy (ms) | ZER0 (ms) |
|---|---:|---:|
| Nemotron Q8 / CPU / batch | 1194.9 | 1046.2 |
| Nemotron Q8 / CPU / stream | 2801.9 | 1605.7 |
| Granite 4.1 2B Q4 / CPU / batch | 4575.1 | 4025.2 |
| Qwen3 1.7B Q5 / CPU / batch | 4270.2 | 4937.2 |
| Parakeet v3 Q4 / CPU / batch | 713.2 | 901.2 |
| Nemotron Q6 / CPU / batch | 1127.0 | 1091.4 |
| Nemotron Q6 / CPU / stream | 2698.4 | 1701.2 |
| Nemotron Q8 / CUDA / batch | 256.3 | 206.3 |
| Nemotron Q8 / CUDA / stream | 1510.3 | 565.9 |
| Granite 4.1 2B Q4 / CUDA / batch | 438.9 | 420.6 |
| Qwen3 1.7B Q5 / CUDA / batch | 280.0 | 310.2 |
| Parakeet v3 Q4 / CUDA / batch | 208.3 | 150.0 |
| Nemotron Q6 / CUDA / batch | 294.3 | 207.1 |
| Nemotron Q6 / CUDA / stream | 1756.3 | 650.1 |

### Validation already completed

- Native Release builds, teardown lint and 40/40 CTest cases passed, including optimizer numerical/storage checks.
- Rebuilt ZER0: 407 Rust tests passed, 1 ignored. Frontend typecheck/lint and benchmark-script bundling passed.
- Upstream Handy: 270 Rust tests passed; rustfmt passed and clippy completed with nonfatal warnings.
- Complete native and app CPU/CUDA suites ran; all successful configurations retained transcript parity. Timing gate failures are retained in the reports.
- Verified Debug records in both stderr and the actual ZER0 log despite inherited `RUST_LOG=warn`; streaming logs include real chunk-stage breakdowns.

### CUDA graphs and graph optimizer

Keep both off by default. CUDA graphs reduced Granite CUDA time from 401.0 to 316.1 ms in the controlled toggle experiment, but Nemotron Q6 streaming increased from 596.1 to 716.4 ms. The compatible audio.cpp optimizer and GGML’s separate CUDA scheduling optimizer did not provide a consistent overall benefit. See [the complete experiment matrix](cuda-graph-experiments.md), including CPU timings, failures and limitations.

### Repositories and local artifacts

- `transcribe_benchmarks`: benchmark-only commit `72adf973`, directly on the fetched upstream history.
- `Handy_benchmarks`: benchmark-only commit `6c7311f0`, directly on the fetched upstream history.
- Raw JSON and logs remain local under `build/diagnostics/`; these ignored artifacts are not published with the commits.
- `verified-before` / `verified-after`: controlled original/fixed fork comparison, with the Qwen tuning caveat above.
- `verified-upstream` plus `verified-upstream-nemotron`: fresh upstream native comparison. The first Nemotron attempt rejected `en`; the corrected runner supplies no explicit hint and all eight Nemotron cases passed.
- `graphs-*` and `qwen-threads-*`: retained optimization experiments, with no slow runs silently discarded.
- Final local native install: `build/diagnostics/install`; rebuilt app: `../Handy_V2/src-tauri/target/debug/zer0.exe`.

## Per-stage pipeline instrumentation (2026-09-21)

Wall clock alone cannot say *which* stage regressed, and on this fixture the
answer was surprising enough to be worth recording. `pipeline.py` taps the
library's own log callback in-process and captures three timers the native
result already computed — `mel_ms`, `encode_ms`, `decode_ms` — plus, for
streaming, the per-chunk record emitted by `emit_streaming_chunk`
(`src/arch/parakeet/model.cpp`):

```
parakeet stream chunk 7: total=12.3 ms  graph_build=0.4 ms  sched_alloc=0.5 ms
  graph_compute=9.1 ms  readback=0.0 ms  cache_rot=1.1 ms  decoder=0.9 ms  other=0.3 ms
  (backend=CUDA0, threads=6, T_q=17, T_cache=70, kv_mode=1, n_layers=24)
```

The eight stages are disjoint and sum to `total`. `report.py` reduces them to
mean and p95 per case across the scored runs. Nothing here needed a new ABI:
the app runner scrapes the same lines out of its replay log, so the app and
native sides produce the same columns without the library having to grow an
interface for a benchmark's benefit.

The instrumentation is arithmetic-checkable, which is how it was validated on a
foreign tree. For one upstream CUDA streaming case, `encode_ms + decode_ms` =
613.3 ms exactly equals the mean of the summed per-chunk totals
(1226.5 / 2). That identity cannot hold by accident, and it is precisely the
identity the pre-instrumentation upstream code *broke* by folding decoder time
into `t_encode_us` — so the split is not merely present, it is correct.

Its first finding explains the fork's largest win. For Q6 CUDA streaming:

| Stage | Upstream | Fork |
|---|---:|---:|
| mel | 918.4 | 33.2 |
| encode | 422.9 | 418.8 |
| decode | 190.4 | 163.3 |
| total | 1585.5 | 632.8 |

Mel is 918 ms upstream against a 33 ms fork mel — and upstream's own *batch* mel
on the same model is 23–31 ms. The upstream streaming path therefore pays a
30–40× mel penalty over its own batch path; almost the entire 2.5× streaming
gap is mel, not the encoder. The candidate cause is the per-chunk host-side
conversion and graph rebuild, which the affinity and pool-parking fixes
incidentally removed. This is the kind of attribution a wall-clock suite cannot
produce, and it is why the stage counters exist.

## Comparing two libraries

`scripts/bench/compare.py` answers "is arm B slower than arm A, beyond noise?",
which is a different question from "how fast is this tree?". A suite runs one
arm's cases back to back, so drift lands entirely inside whichever arm happened
to run second. It alternates the arms within each repetition with a rotating
start, and calls a difference real only when it exceeds the larger arm's own
within-arm spread — the case's measured noise floor, not a fixed guess. Each
arm loads through the bindings of the checkout that owns its library, because
the generated bindings are ABI-specific: a foreign library fails at import time
on the first symbol it does not export, and that is a hard error rather than
something to silently time.

It was built because the suite's verdicts did not survive it. Nemotron Q6 CPU
batch read 1090.3 ms (upstream) against 1208.4 ms (fork) in a suite — an
apparent 10.8% regression. Interleaved, the same pair read 937.2 against 809.3:
the fork 13.6% *faster*. The same library moved 1090 → 937 ms (18%) between the
two runs. The remaining suite-flagged CPU regressions were re-tested the same
way; results are in the next section.

The practical rule: `suite.py --baseline` tracks one tree over time, and
`compare.py` is the only thing that may be cited for a cross-arm claim. A win
below the noise floor is reverted and said to be reverted, not kept for
plausibility.

## Fresh upstream-vs-fork matrix (2026-09-21)

Both trees were built from source with identical configuration—Ninja, MSVC
14.51.36231, CUDA 13.4 targeting sm_89, `BUILD_SHARED_LIBS=ON`,
`TRANSCRIBE_ARCH_DL=ON`, `GGML_NATIVE=OFF`, `GGML_AVX2=ON`. `GGML_NATIVE=OFF`
with `GGML_AVX2=ON` is deliberate: the reference must build the same way
regardless of which host compiles it, and under the Visual Studio generator
`/arch:AVX2` is spelled `<EnableEnhancedInstructionSet>AdvancedVectorExtensions2`,
so grepping a `.vcxproj` for `/arch:` proves nothing about the actual ISA.

The upstream arm is `../transcribe_benchmarks` at upstream
`be7a8b35e9ba2df20298bd26e32d53407c3bcbcd`; the fork arm is this tree. Fourteen
cases, three resident runs each, run 1 excluded. Ratio is fork / upstream, so
below 1.0 is the fork faster.

| Case | Upstream | Fork | Ratio |
|---|---:|---:|---:|
| Granite Q4 CPU batch | — | 4597.6 | 1.080* |
| Qwen3 1.7B CPU batch | — | 4630.7 | 1.080* |
| Nemotron Q6 CPU batch | — | 1208.4 | 1.108* |
| Nemotron Q6 CPU stream | — | 1828.3 | 0.753 |
| Nemotron Q8 CPU stream | — | 1708.8 | 0.570 |
| Nemotron Q6 CUDA batch | — | 196.0 | 0.765 |
| Nemotron Q8 CUDA batch | — | 200.8 | 0.695 |
| Nemotron Q6 CUDA stream | 1585.5 | 632.8 | 0.399 |
| Nemotron Q8 CUDA stream | — | 631.4 | 0.401 |
| Granite Q4 CUDA batch | — | 417.4 | 0.964 |
| Parakeet Q4 CPU batch | — | 1004.9 | 0.982 |
| Parakeet Q4 CUDA batch | — | 160.6 | 0.899 |

The starred ratios are the ones that did not hold up; they are the suite's
numbers and are superseded by the interleaved re-test below. Upstream absolutes
are omitted where the fork's own suite report is the source, because quoting
the partner's number from a different run is exactly the mistake this section
documents.

### The three flagged CPU regressions were drift

The suite flagged Granite Q4, Qwen3 1.7B and Nemotron Q6 CPU batch as 8.0%,
8.0% and 10.8% slower. Re-measured with `compare.py` — arms alternating within
each repetition, rotating start, three repetitions each:

| Case | Upstream | Fork | Ratio | Verdict |
|---|---:|---:|---:|---|
| Nemotron Q6 CPU batch | 937.2 | 809.3 | 0.864 | fork 13.6% faster |
| Granite Q4 CPU batch | 4840.6 | 4078.4 | 0.843 | fork 15.7% faster |
| Qwen3 1.7B CPU batch | 4488.7 | 4371.6 | 0.974 | within noise (8.3%) |
| Nemotron Q6 CUDA stream | 1483.0 | 578.4 | 0.390 | fork 2.5× faster |

All three dissolve. Nemotron Q6 CPU batch moved from an apparent 10.8%
regression to a 13.6% win; upstream's own measurement for it moved 1090.3 →
937.2 ms (13.9%) between the two runs, which is the size of the effect being
claimed. Granite reverses outright. Qwen lands inside the noise floor, which is
a legitimate "no change", not a measured win, and nothing may be claimed from
it.

The last row is the control, and it is what makes the others trustworthy: it
is the largest effect in the suite (2.5×) and it reproduces under interleaving
at 0.390 against the suite's 0.399. So the suite is not worthless — it does
resolve large effects. What it cannot do is resolve effects near its own drift,
and on this machine the drift is 14–18%, which is the same order as the very
CPU deltas being reported. Interleaving is what separates them.

### Known non-regression

Granite Q4 CUDA batch and Qwen3 1.7B CUDA batch produced different transcripts
in one direction only: the difference is punctuation. Upstream emits "for you
ask what" where the fork emits "for you. Ask what" — one token, same words,
from the graph optimizer perturbing numerics on the most quantization-sensitive
case. The words are identical, so this is not a content regression, and it is
recorded rather than smoothed over because a transcript gate will keep flagging
it until someone checks the text rather than the hash.
