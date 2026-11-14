# VoxLock voice isolation: review and implementation plan

Date: 2026-10-16. Status: researched proposal; no VoxLock model was downloaded,
executed, converted or integrated into either application. This plan complements
the [VAD/frontend roadmap](2026-10-three-tier-super-vad-and-firered-architecture-plan.md)
and [paired-project contract](../zer0-engine-integration.md).

## Decision

VoxLock merits an optional target-speaker-isolation experiment for dictation
near other talkers, TV or shared-office speech. It produces a cleaned waveform;
it is neither a VAD nor a turn-completion detector. RNNoise mainly addresses
noise, while voice isolation addresses which speaker should reach ASR.

The preferred first ZER0 mode is explicit enrollment using a clean voice sample,
with isolation off by default. Retain a normal dictation mode and raw recording
for controlled recovery/comparison where the user's recording settings allow.
Keep profile selection in the app and reusable signal/model processing in the
engine. One isolator per source serves all ASR models in Multi-STT.

Do not market the enrollment as authentication or a guarantee that only one
person can be transcribed. The wrong sample can select the wrong talker. This
is an audio quality feature, with measurable rejection/leakage tradeoffs.

## 1. Audited source and artifact provenance

- Source: `arbazsiddiqui/voxlock` at
  `08ed37cbb79f449ca6b499bc23fb14138b207bfd`.
- HF repository inspected at `894e78845eca66a7691d330665c60a7c821b8e2d`.
- The source package actually downloads revision
  `a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03`. Resolve that difference explicitly;
  the compared ONNX artifact hashes match in these two metadata snapshots.
- [Source lock](../research/vad-source-lock-2026-10-16.json) records artifacts,
  full checksums and byte sizes from the public API; file bytes were not fetched.
- Model card declares Apache-2.0 and English; source NOTICE names WeSep/REAL-TSE
  and WeSpeaker origins. Retain required notices when porting. The demo's Krisp
  audio is identified as CC BY-NC 4.0; do not bundle those recordings as ordinary
  commercial app assets. Model licensing and evaluation-data licensing differ.

[Model card](https://huggingface.co/arbazsiddiqui/voxlock),
[pinned source](https://github.com/arbazsiddiqui/voxlock/tree/08ed37cbb79f449ca6b499bc23fb14138b207bfd),
[NOTICE](https://github.com/arbazsiddiqui/voxlock/blob/08ed37cbb79f449ca6b499bc23fb14138b207bfd/NOTICE).

## 2. Why automatic enrollment is not the initial dictation default

File mode chooses a loud three-second region from the first fifteen seconds.
That selection uses later audio and can select a loud bystander; it does not
establish the identity of the microphone owner.

Live mode passes through audio until it accumulates approximately three seconds
of speech, then warms recurrent state on recent audio and transitions to the
isolated stream. A short command can finish before isolation starts. A published
offline WER obtained with a selected sample is therefore not the result of
immediate, causal automatic enrollment. [Enrollment implementation](https://github.com/arbazsiddiqui/voxlock/blob/08ed37cbb79f449ca6b499bc23fb14138b207bfd/voxlock/core.py).

Plan a separate enrollment action: capture a clean sample, let the user audition
it, compute reusable model conditioning outside the hotkey path, associate it
with its model/preprocessing version, and allow deletion/re-enrollment. Keep it
local; do not put raw samples or voice conditioning into routine logs. If no
valid profile is selected, show ordinary mode or unavailable status rather than
silently promising isolation. Optional automatic enrollment can be evaluated
later with explicit pre-lock bypass status and wrong-speaker tests.

## 3. Actual signal and model contract

From `core.py`, `export_onnx.py` and the model config:

| Component | Reference behavior to preserve |
| --- | --- |
| Model audio | 16 kHz mono float32; normalize once at the app boundary |
| Analysis | 512-sample window, 128-sample hop: 32 ms / 8 ms |
| Boundaries | Reflect padding at both ends; leading padding removed from output |
| Enrollment | Separate network returns enrollment magnitude and frame-level speaker context, not just one generic speaker ID vector |
| Chunk input | Waveform of `K * 128 + 384` samples plus enrollment conditioning and `h/c` |
| Chunk output | `K` windowed 512-sample frames plus updated recurrent state |
| Recurrent state | Each of h/c has shape `(6, 32, 256)` in this export; together 384 KiB float32, excluding all other working buffers |
| Synthesis | Continuous overlap-add with squared-window normalization; preserve incomplete overlap across pushes |
| Output scaling | Reference applies `GAIN = -0.5`; omit it and gain/polarity differ |
| Final output | Flush and trim against real source length; file wrapper explicitly slices to input length |

[Export contract](https://github.com/arbazsiddiqui/voxlock/blob/08ed37cbb79f449ca6b499bc23fb14138b207bfd/export_onnx.py).

The advertised 24 ms is algorithmic streaming delay, not the full cost of
resampling, collecting a 40/80/160 ms chunk, queueing, inference and endpointing.
Measure each separately with actual source samples and a monotonic clock.
Do not run the heavy model once per 16 ms Earshot frame merely because that is
the recorder's current callback size.

## 4. Precision, packaging and evidence quality

These are public metadata file sizes for enrollment + chunk models, not peak
RAM or proposed GGUF sizes. Binary units explain the rounded download labels.

| Reference precision | Combined bytes | Approx. MiB | Author-reported Krisp WER |
| --- | --- | --- | --- |
| FP32 | 99,866,472 | 95.24 | 4.95% |
| FP16 | 50,598,514 | 48.25 | FP32 score reused from waveform agreement; not a separate full WER sweep |
| INT8 | 79,083,600 | 75.42 | 5.11% |
| INT8-full | 70,215,223 | 66.96 | 5.85% |

The headline WER is author-reported on 265 Krisp recordings with Parakeet TDT
0.6B v2 and the published normalization. The default INT8 checkpoint has a
different score. The source's paired confidence interval against Krisp Default
crosses zero, so the reported ranking does not establish a decisive universal
advantage. Leakage is counted during background-only spans; also measure
overlap contamination and target deletions. [Benchmark method](https://github.com/arbazsiddiqui/voxlock/blob/08ed37cbb79f449ca6b499bc23fb14138b207bfd/bench/krisp_bench.py),
[per-method uncertainty](https://github.com/arbazsiddiqui/voxlock/blob/08ed37cbb79f449ca6b499bc23fb14138b207bfd/bench/results/TABLE_krisp.md).

The author's INT8 streaming RTF is 0.69 with 40 ms chunks on four M2 Pro threads.
That is approximately 27.6 ms compute per 40 ms audio in that test, a substantial
budget beside ASR. It is not a Windows measurement. Require a full concurrent
pipeline test on the actual supported CPU; standalone RTF below 1 is necessary
but insufficient. [Published precision/performance comparison](https://pypi.org/project/voxlock/).

## 5. Source review: integration issues to resolve

1. `VoxLock.__init__` can fall back to FP32 filenames when the requested chunk
   artifact is absent. ZER0 must verify the exact model pair/hash and report
   load errors instead of silently changing its resource budget.
2. `Stream.flush()` returns empty while the initial head has at most 256 samples;
   add tiny-input vectors. General streamed flush output must be bounded by the
   remaining real sample count; file mode's final slicing is part of parity.
3. The Pipecat adapter's `stop()` clears the stream, and its enable control only
   changes a flag. It does not expose a complete pending PCM, resampler and
   separator-tail drain path. Do not copy that lifecycle into final dictation.
4. Repeated NumPy concatenations allocate. They are a Python oracle design,
   not an allocation-free capture implementation. Native rings/pools need
   waveform and state parity tests rather than line-by-line translation.
5. Preserve state through real silence. An early VAD that removes background or
   pause frames changes the separator's history. Do not put destructive VAD
   before enrollment/separation.
6. Pre-enrolled mode and live self-enrollment are different lifecycle contracts.
   Test source/device/profile switches, reset, cancel, start after idle, and
   failure midway through a chunk. No old conditioning or tail may leak into
   the next source epoch.

[Stream implementation](https://github.com/arbazsiddiqui/voxlock/blob/08ed37cbb79f449ca6b499bc23fb14138b207bfd/voxlock/core.py),
[adapter lifecycle](https://github.com/arbazsiddiqui/voxlock/blob/08ed37cbb79f449ca6b499bc23fb14138b207bfd/voxlock/pipecat.py).
These are source-level review findings; no model execution has confirmed their
audible impact in ZER0.

### Wrapper counterexamples now reproduced without weights

An identity-network stub exercised the pinned wrapper's bookkeeping with
NumPy 2.5.3. This executes the reviewed Stream/LiveStream definitions, not ONNX
or the neural separator; the stub's output is deliberately constructed to
reconstruct the input after overlap-add and gain correction.

| Input length | Push output | Flush output | Total output |
| --- | --- | --- | --- |
| 1 through 256 samples, selected boundary cases | 0 | 0 | 0 |
| 257 | 0 | 640 | 640 |
| 512 | 128 | 768 | 896 |
| 640 | 256 | 768 | 1024 |
| 48,000 | 47,616 | 768 | 48,384 |

For tested inputs >=257, trimming to real input length reconstructed the stub
signal within 4.5e-8 maximum absolute error. This confirms wrapper length and
alignment behavior, not neural waveform fidelity or ASR quality. Inputs under
16 ms are an API boundary case; this does not imply that ordinary 200 ms words
are discarded by the same branch.

In a forced live switch after one 640-sample block, all 640 samples were already
passed through, 384 warm-up samples still needed discarding, yet flush returned
768 samples without applying that remaining skip. A product adapter must apply
the warm-up suppression to flush as well as push and cap total real output by
its source ledger. The forced switch deliberately bypassed normal enrollment
duration; this isolates a reachable lifecycle branch without assessing how
the live sampler selects a real speaker.

Reproduce with the reviewed upstream core.py at the pinned source revision:

```powershell
uv run --no-project --with numpy==2.5.3 scripts/vad/probe_voxlock_bookkeeping.py --source <local-voxlock>/voxlock/core.py
```

The utility verifies the reviewed source hash and runs no ONNX/model imports.
It records all 14 tested lengths and the forced switch under ignored build/.
Do not copy reference overshoot into the engine or quietly change an oracle
expected output: record raw-reference behavior and corrected product behavior
as separate vectors. A reused Stream after flush also lacks a full reset;
create a fresh stream/context per epoch rather than treating it like Silero's
automatic restart contract.

### Thread and lifetime policy for feasibility tests

VoxLock's default threads=0 lets ORT choose, and enrollment and chunk models
are distinct sessions. ORT documents per-session pools and spinning; default
pool settings can compete with ASR rather than provide free parallelism.
Pin ORT/provider/version, compare explicit 1/2/4 intra-op threads and sequential
execution, and measure pool/spinning behavior with the actual concurrent load.
Enrollment and profile loading stay off the recorder consumer/Stop handshake.
Budget native model-lock waiting separately; cancel must not free active
conditioning or stale results into a new epoch. [ORT threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

Dynamic quantization has its own runtime overhead and quality tradeoff; do not
derive throughput from file size or translate ONNX INT8 scores to GGML quants.
Pin preprocessing/graph optimization and verify the chosen precision instead
of silently falling back. [ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

## 6. Placement in the paired pipeline

```text
device mono PCM + original source positions
  -> one rate converter to 16 kHz
  -> optional pre-enrolled VoxLock, once per source
  -> continuous isolated PCM + original-time map
       -> acoustic VAD + retention policy -> one/many ASR models
       -> recent continuous turn audio -> optional Smart Turn at pauses
       -> isolated tap / target speech clock, with declared semantics
```

Begin with RNNoise off for VoxLock trials. Compare no enhancement, RNNoise only,
VoxLock only, then RNNoise-before-VoxLock only if needed. That fourth path
requires native -> 48 kHz RNNoise -> 16 kHz isolation; do not implement a
16 -> 48 -> 16 kHz round trip by accident. Stacking models is not an assumed
quality gain. Recalibrate downstream VAD thresholds on each selected path.

Preserve existing raw/native/denoised FFT taps; add an explicitly named isolated
tap only when enabled and observed. The speaking indicator and WPM should
represent the chosen target policy, while raw diagnostics remain distinct.
Imported files and multi-speaker meeting transcription must opt into a profile;
silently removing non-enrolled meeting speakers would be incorrect.

## 7. Implementation stages

| Stage | Concrete deliverable | Exit gate |
| --- | --- | --- |
| V0: intake | Follow `porting-1-intake`; pin code/weights/conditioning/frontend/license, inventory GGML ops and both subgraphs | New audio-output role/lifecycle agreed; reference dtype selected; no ASR/VAD API abuse |
| V1: oracle | Pinned ONNX/PyTorch comparison environment; enroll outputs, per-chunk h/c, windowed frames, final waveform and real-length manifest | Ragged/chunked/full-file contract documented, including current tiny-input/flush differences |
| V2: product feasibility | Offline replay plus paced 40/80/160 ms chunks, explicit samples; actual ZER0 ASR families; raw/RNNoise baselines | Target leakage improves without unacceptable caller deletion, clean-speech or short-command regressions; concurrent CPU/latency budgets pass |
| V3: native reference port | Follow convert/C++ stages for `voxlock`; enrollment and separator subgraphs, reference dtype, exact synthesis/scaling | Tensor/state and waveform parity against V1; no partial port presented as a shipping frontend |
| V4: paired opt-in app | One engine source context, warm model/conditioning, optional profile UI/storage, source map, diagnostics and fail/reset/drain policy | Hotkey unchanged when off; manual stop preserves tail; cancel releases leases; 1/9-model fan-out shares processed PCM |
| V5: quantization/release | Per-precision quality/performance, packaging and native-build provenance; reciprocal docs | Full held-out and sustained-load checks pass; download sizes/CPU requirements accurate; default remains off until an explicit product decision |

Use ONNX as the initial reference/evaluation runtime outside the shipping app.
The preferred production direction is a native engine audio-enhancement role
or equivalent typed frontend interface, with opaque source context and owned
conditioning handles. Current roles are ASR/diarize/langid/VAD; a waveform
transform does not fit their existing output contracts. Reuse model-loading,
capability and binding patterns, but specify a separate audio data plane.

A production ONNX Runtime dependency is an alternative only after comparing
package size, maintenance, CPU performance and time to a validated GGML port.
It is not introduced by this plan. Reusing Silero's LSTM math may help an
inventory, but does not prove support for VoxLock's band communication,
conditioning, normalization and complex masks. Avoid new GGML operators until
the porting review establishes a need.

## 8. Acceptance matrix and rollback

- **Waveform/state:** lengths 0, 1, 127, 128, 255, 256, 257, 511, 512 and
  non-hop-aligned long streams; file/stream parity at 16 ms, 40/80/160 ms and
  randomized delivery; no NaNs; sign/gain, overlap envelope and final word kept.
- **Speaker selection:** correct/wrong/noisy/short enrollment, louder bystander,
  target silent, overlapping speakers, same household voices, headset/speaker
  changes, quiet/whispered target and French/English/multilingual commands.
- **Quality:** target WER/CER, short-command exact match, first/last-word errors,
  substitutions/deletions separately, background-only leakage and overlap
  contamination. Use a held-out speaker/device split and paired uncertainty.
- **Latency/load:** warm/cold setup, first processed audio, first text and
  stop-to-final p95/p99; CPU/RSS and queue depth under 1 and 9 ASR consumers;
  sustained 20-minute runs. Candidate chunk/thread settings are experiments.
- **Provisional product gate:** added warm frontend availability p95 <=100 ms
  relative to the same raw path is a starting target for dictation, not a
  measured result. If it fails, keep the feature experimental or limit it to
  a mode that explicitly accepts its latency. Do not hide delay in VAD timers.
- **Lifecycle:** stop during partial chunk drains through ASR finalize exactly
  once; cancel drops the current epoch; disable/profile switch cannot replay
  old samples; missing model/profile and overruns have visible bounded outcomes.
- **Rollback:** feature off uses the recorded normal-mode pipeline and settings;
  removing a model/profile leaves no mandatory runtime/model load. Preserve the
  existing raw-recording policy rather than forcing voice-sample retention.

No benchmark in this review establishes a ZER0 quality gain. The next action is
V0/V1 plus a small feasibility replay, after the shared ownership/timing work in
the main roadmap. A large native port should follow evidence of product value.

Use the [shared frontend contract](2026-10-audio-frontend-contract.md) for
positioned gaps, final source watermarks, model serialization and preregistered
quality/load gates. Semantic completion never substitutes for ASR text drain.
