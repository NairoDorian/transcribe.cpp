# VAD, endpointing, and audio preprocessing integration plan

Revision 9, 2026-10-16. Status: implementation plan; the three-tier pipeline,
RNNoise migration and VoxLock integration have not been implemented. This
revision supersedes the previous eight revisions and their illustrative APIs.

> **Update 2026-11-14.** This plan's engineering contract still stands, but
> the tier-1 model choice is revised by the
> [November master plan](2026-11-fork-master-plan.md) §B.0: on the same TEN
> VAD test set audio.cpp measured frame F1 Silero 0.935, Earshot 0.926,
> FireRed 0.917, with Earshot ~14× faster than FireRed. FireRed Stream-VAD is
> therefore an evaluated option (master plan §B.5), not the default; Earshot
> fidelity and per-ISA kernels (§B.1–B.3) come first. Both Silero and Earshot
> are now on `main`.

Read [the paired-project contract](../zer0-engine-integration.md) before working
on either repository. ZER0 is the reference consumer for this fork; engine
changes must be checked against its actual Rust call sites and capture behavior.
Keep the engine reusable for other consumers, including a future Android client.
No Android checkout or battery measurements were examined for this revision.

## Recommended direction after the deeper review

Prioritize ownership and source-time correctness, then buffer/queue latency,
before adding models. Use one acoustic detector as the production baseline.
Treat FireRed as a challenger to Earshot/Silero, Smart Turn as optional endpoint
advice, and VoxLock as optional enrolled-speaker waveform isolation. A fixed
three-VAD cascade is not the default architecture.

The new [evidence review](../research/2026-10-vad-preprocessing-evidence.md)
contains pinned sources, additional local mismatches and a reproducible framing
probe. The separate [VoxLock review and staged implementation plan](2026-10-voxlock-voice-isolation-plan.md)
covers its model/role requirements, enrollment, CPU cost and reference defects.
No new model-quality or end-to-end performance result is claimed.

The [frontend contracts and acceptance gates](2026-10-audio-frontend-contract.md)
now define the source/retained/text boundaries, capture gaps, stop handshake,
existing Silero API semantics and preregistered experiment criteria. Read that
contract before implementing P0-P6; it strengthens and qualifies the shorthand
deadline and lifecycle language in earlier revisions.

## 1. Verified baseline and branch provenance

The 2026-10-16 audit inspected these separate states. These are provenance pins,
not instructions to keep future development on old revisions.

| Surface | Audited state | Meaning |
| --- | --- | --- |
| Fork main | `692350de` | Embedded Earshot, ASR streaming VAD defaults, model composites, architecture plugins |
| Upstream VAD | `de75ca13` | Standalone Silero VAD role, probabilities, offline segmentation, bindings and tests |
| Upstream main | `b61b7f93` | Contains VAD through `c63b18e2` / PR #201, merged 2026-10-10 |
| Local VAD integration | `codex/vad-integration`, based on fork main | Upstream VAD merge prepared without committing; inspect Git status before assuming it is landed |
| ZER0 | `Handy_Multi_STT`, HEAD `51a69d55` | Capture uses Rust Earshot and optional Rust RNNoise |
| ZER0 dependency | Working `Cargo.lock`: `468e8776`; HEAD lock: `ba949120` | The dependency/API update is uncommitted; a built binary can use a prebuilt override instead |

This integration preserves fork Earshot in `src/transcribe-earshot.{h,cpp}` and
the upstream role dispatcher in `src/transcribe-vad.{h,cpp}`. Both public APIs
remain distinct. Rust retains `VoiceActivityDetector` through `earshot.rs` and
adds `VadSession` through `vad.rs`. Renaming these internal files is merge
reconciliation, not adoption of a new capture pipeline by ZER0.

### What exists and what does not

| Capability | Verified current behavior | Remaining work |
| --- | --- | --- |
| Capture VAD | ZER0's Rust Earshot, 256 samples / 16 ms | Explicitly assign one owner per audio source |
| Embedded engine VAD | Native ASR streams default `enable_vad=true` | ZER0 uses defaults, so Rust filtering can be followed by another Earshot evaluation per model |
| Standalone model VAD | Upstream Silero, 512 samples / 32 ms, GGML; CPU default | Integrate with app deliberately; it is not auto-inserted before ASR |
| Partial-frame buffering | Silero role retains remainder and flushes final padded frame | Fork embedded Earshot still lacks carry-over between arbitrary ASR feed sizes |
| Live endpoint events | Latest Silero API returns streaming probabilities | START/END iterator was removed during upstream review; design live policy separately |
| Pre-roll / onset | Rust `SmoothedVad`, preallocated circular storage | Engine `vad_prefill_ms` currently does not implement an audio pre-roll buffer |
| Speech duration | Rust `SpeechClock`, monotonic with deferred short-gap accounting | Embedded C++ counter includes hangover; do not substitute it for WPM |
| RNNoise | Rust `nnnoiseless` 0.5.2, optional, before Earshot | No engine denoiser or migration is present |
| FireRed / Smart Turn | No runtime implementations in the inspected projects | Intake, reference contracts, implementation and validation are future work |

The old claim that ZER0 does not use engine VAD is false: its native stream
builders use `StreamOptions { family, ..Default::default() }`, whose VAD default
is true. Its slider and toggle currently control the Rust capture detector;
the app does not use engine VAD telemetry for its speech UI.

The old claim of eliminating 62.5-125 allocations/second from `SmoothedVad` is
also obsolete: the Rust ring already removed those allocations. Remaining
costs include `FrameResampler::process_chunk` allocating output and copying
it again, and `StreamRouter::feed` cloning PCM separately for each model.

## 2. Ownership and proposed pipeline

One preprocessing context belongs to an audio source, independently of ASR
models. Its output can be shared by all models in Multi-STT. Do not instantiate
the entire frontend once per ASR stream or silently enable several detectors.

```text
device-rate mono PCM + original sample position + source epoch
  -> selected path: direct / RNNoise at 48 kHz / enrolled VoxLock at 16 kHz
  -> continuous 16 kHz source audio + valid-sample/time mapping
       -> acoustic scores -> pre-roll/onset/retention -> one/many ASR sessions
       -> continuous recent-turn ring -> optional semantic advice at pauses
       -> declared raw/processed taps and distinct speech-time metrics
```

Keep the existing ASR input contract: 16 kHz mono finite float32 PCM. A proposed
native-rate preprocessing interface must be separate; do not force already
resampled callers through a 16 -> 48 -> 16 kHz RNNoise round trip. The context
must also work without an ASR model for the live VAD test and history analysis.

Capture, device selection, hotkeys, UI event throttling, history persistence,
LLM orchestration, clipboard and text insertion stay in the app. Shared DSP,
framing, optional suppression, activity scoring and tested audio policies can
move into the engine. Publish raw scores, voiced decisions, retained audio,
speech duration and endpoint decisions as different facts.

### Optional three-tier experiment

| Tier | Proposed function | Acceptance condition |
| --- | --- | --- |
| 0: RMS + Earshot | Cheap activity sentry | No irreversible loss of quiet speech; retain pre-roll before rejecting audio |
| 1: FireRed Stream-VAD | Candidate acoustic verification, or alternative to Earshot | Improves false activations/misses and downstream WER within a measured latency budget |
| 2: Smart Turn + bounded timing policy | Distinguish completed turns from hesitation | Runs at candidate pauses; fragments, lists and code still reach a hard endpoint deadline |

This is a candidate cascade, not three models that must always run. Compare
Earshot-only, Silero-only, FireRed-only and cascaded policies. A later detector
cannot recover speech an earlier stage discarded. Define recurrent-state and
time advancement through skipped regions; do not concatenate unrelated spans
and pretend that they are a continuous acoustic history.

### Detector ownership rollout

First centralize and explicitly set the current native stream options at all
app call sites; retain a named legacy configuration for replay comparisons.
Then compare CaptureOwned (Rust retention, native `enable_vad=false`) against
EngineOwned (untrimmed capture, one source-level engine activity policy).
Do not change the library's global defaults merely to repair this consumer.
CaptureOwned disables the engine's quiet-feed bypass as well as Earshot work,
so its compute/quality tradeoff must be measured per family.

Choose and document exactly what the VAD toggle disables: acoustic retention,
the RMS shortcut and all per-ASR VAD options. RNNoise suppression, its optional
probability gate and VoxLock are separately named controls; no hidden gate may
be presented as disabled. Diagnostics may run shadow scoring for experiments,
but only one policy owns production retention and the published speech clock.

## 3. Compatibility requirements from ZER0

Read the actual constants and state machines in the peer checkout, not just
this snapshot. At audit time:

| Policy | Current app contract |
| --- | --- |
| Earshot frame | 256 samples at 16 kHz |
| Pre-roll | 450 ms, rounded up to 29 frames / 464 ms |
| Onset confirmation | 60 ms, rounded up to 4 frames / 64 ms |
| Offline hangover | 450 ms, rounded up to 29 frames / 464 ms |
| Streaming hangover | 1650 ms, rounded up to 104 frames / 1664 ms |
| Hysteresis | Entry = configured threshold; exit = max(threshold - 0.15, 0.01) |
| Energy pre-gate | -45 dBFS, only while not already voiced; skips Earshot inference, not RNNoise |
| Speech clock | Confirm onset; defer short silence and count it only when speech resumes; never rewind published time |
| Very short capture | Ordinary dictation currently checks 200 ms measured speech before final output |

The embedded detector is not policy-identical: its exit threshold is
`max(0.70 * entry, 0.05)`, and hangover uses floor division by 16. At entry
0.8 the app exits at 0.65 while C++ exits at 0.56; 450 ms rounds to 464 ms in
the app versus 448 ms in C++. Include low/default/high slider values and
non-multiple frame durations in parity vectors; testing only entry 0.5 hides
the hysteresis disagreement. The C++ partial-frame feed limitation is a VAD
accounting gap, not proof that ASR discards every remainder.

Preserve microphone channel handling, raw/native/denoised/processed FFT taps,
live threshold updates, denoise settings, and batch/stream audio agreement.
History currently re-analyzes using Rust Earshot at 0.5; that is not identical
to the live clock. Unify it only through an explicit metric migration.

Imported files currently decode, downmix, resample and segment near quiet
boundaries outside the microphone chain. Decide whether file preprocessing is
opt-in; do not silently apply microphone suppression to all imports.

R2T2 opts out of the engine silence shortcut because removing silence changed
its input and previously broke transcription. Preserve this family contract.
Keep original source positions and retained-audio positions separate so history,
Multi-STT chunk slicing and timestamps cannot confuse their two timelines.

## 4. RNNoise and resampling workstream

Current RNNoise uses 480-sample / 10 ms frames at 48 kHz. Defaults are disabled,
strength 1.0, probability gate 0 (off), grace 200 ms. The gate is distinct from
Earshot's threshold and from utterance endpointing.

Preserve the delayed dry side of the wet/dry mix, finite bounded PCM, the final
silent-frame flush, resampler tails, recording resets and tap semantics.
Strength zero is not a CPU bypass: the network and any enabled probability gate
still run. Account for probability/audio alignment around word boundaries.

The deterministic app-source probe measured first output at 70 ms for a
16 kHz denoise path versus 20 ms direct, using synthetic 10 ms deliveries.
This is collection/framing behavior, not compute speed or first-word latency.
See the evidence review for all rates, impulse alignment and padded lengths.
Treat requested FFT block size, filter delay and output framing as separate
optimization variables. Do not start by attributing the delay to Rust.

Work in this order:

1. Replace allocating Rubato `process` calls and the second output copy with
   reusable buffers via `process_into_buffer`. Measure collection delay and
   filter delay separately; current blocks are not zero-latency.
2. Reduce per-model PCM cloning with shared immutable blocks or a bounded pool.
   Backpressure must not block the platform capture callback or silently drop
   speech. Retain queue-delay and overrun diagnostics.
3. Benchmark current Rust RNNoise against a CPU-native candidate using pinned
   revisions and matched weights where possible. Changing model weights is a
   quality experiment, not an implementation-only speed comparison.
4. Evaluate a separate optional native-rate engine frontend only after these
   measurements. Document model/code licensing, packaging and runtime dispatch.

A 48 kHz device avoids the input resampling stage of the current noise chain.
Do not hard-gate audio before RNNoise without proving recurrent continuity,
quiet-speech recall, boundary preservation and reset behavior. The existing
post-denoiser energy gate does not save RNNoise compute.

### Live toggles and final tails are part of preprocessing

The current recorder resets both chains on a denoise toggle, potentially
discarding buffered source samples and filter tails. Define a transition at a
source position: drain the old path, compensate the new path's delay, and emit
each real span once. Track source epoch and policy generation; do not count
synthetic flush/padding frames as speech or replay old tails after a toggle.
Test off->on->off at every phase of a 10/16 ms frame and input FFT block.

## 5. Model contracts to verify before porting

Follow the repository's ordered porting stages. Pin reference source, checkpoint,
dtype, frontend, reset behavior and representative audio before writing kernels.

### FireRed

The authors' 97.57% F1 / 99.60% AUC figures are for non-streaming FireRedVAD on
FLEURS-VAD-102. They are not a measurement of Stream-VAD, this GGML port, the
cascade, or ZER0 transcription. The upstream Stream-VAD reference lists 567,937
parameters; any proposed 1.2 MB artifact needs an explicit dtype and conversion.

The frontend uses 80-bin Kaldi FBank, 25 ms windows, 10 ms hops, CMVN and
`snip_edges=True`. An isolated 2560-sample block produces
`1 + floor((2560 - 400) / 160) = 14` complete frames, not 16. Specify streaming
overlap, initial alignment, final padding and timestamp mapping explicitly.
Read checkpoint lookback/lookahead orders and strides instead of assuming a
causal configuration. Zero neural lookahead does not remove collection delay.

The upstream sample wrapper retains neural caches but recreates OnlineFbank
on every feature extraction. One 2560-sample call yields 14 frames; two
independent 1280-sample calls yield 12. Establish a continuous frontend oracle
with persistent overlap/global hop phase, and document its difference from
upstream chunk-local extraction before writing kernels. If the checkpoint has
`N2 > 0`, test the model's `T > 1` lookahead branch too. Do not hide these
differences inside a broad numerical tolerance.

The existing Kaldi frontend is a reuse candidate. Matching window, scaling,
pre-emphasis, CMVN and padding is a validation task; code reuse alone does not
prove numerical parity or allow sharing another model's FFT output.

### Smart Turn

The authors use a Whisper Tiny encoder, attention pooling and a classifier.
Tiny has d_model=384, four encoder layers and six heads; the previous draft's
six-layer/four-head specification was wrong. Reference code uses the most recent
eight seconds, left-pads short audio, normalizes through its feature extractor,
and consumes an 80x800 log-mel input. Validate the actual checkpoint and tensors.

Pin v3.2 deliberately: the repository README names v3.2 while its standalone
example still names a v3.1 file; current Pipecat uses v3.2 CPU. Validate against
one exact frontend/checkpoint pair. Preserve continuous pauses in its input;
do not feed the VAD-compacted ASR timeline. Waveform normalization and global
mel clipping also make naive incremental feature reuse inexact.

Smart Turn predicts turn completion, not merely a terminal pitch fall, and its
authors caution against very short context. Short commands, lists, code,
multilingual fragments and trailing words such as "to" or "the" need bounded
timing rules even when the classifier or lexical heuristic says "incomplete".

## 6. Endpoint policy: explicit deadlines

Separate raw voiced time, retained-audio hangover, pause detection, semantic
hesitation and the boundary-decision deadline. ASR drain/finalize and text
publication have separate budgets. Do not use one counter for these quantities
or promise complete text within the endpoint timer.

The candidate configurable curve remains: spoken duration <=1.5 s -> 350 ms
pause tolerance; 3.5 s -> 650 ms; 7 s -> 950 ms; >=12 s -> 1200 ms. These are
starting experiment values, not validated product defaults. Linear/smoothstep
interpolation must be monotonic and clamped; reject invalid/unsorted points.

For a raw-command/fragment mode, use an absolute deadline that semantic or
lexical hints cannot extend. A conversational mode may allow a separately
bounded hesitation extension. The old pseudocode added 600 ms before checking
its supposed unconditional 350 ms ceiling; it did not implement that promise.
Specify timer start, retrigger, equality and final-flush behavior in tests.

Ordinary push-to-talk/manual-stop dictation does not wait for semantic
approval after the user releases the key. Auto endpointing applies only to a
mode that requests it. Detect candidate pauses from raw activity before the
1650 ms retained-audio hangover; otherwise adding a fast classifier cannot
remove the earlier hold. Record `last_voiced_source_sample`, candidate start,
classifier start/result and absolute deadline separately.

Use at most one semantic inference in flight and one replaceable pending
snapshot. Match epoch, policy/profile generation, candidate-pause ID and speech
revision; reject a result if new candidate speech occurred after its observed
source end. Continuing confirmed silence may advance the PCM cursor without
invalidating the snapshot. An equality check against the latest PCM cursor
would otherwise reject every asynchronous result. Inference error/timeout
falls back to the bounded acoustic policy. A late classifier may not extend
the boundary deadline or reopen a finalized turn.

At the deadline, issue a boundary/drain decision; do not force text ownership.
Multi-STT must still satisfy its own text-present and audio-drained close gate,
or retire/fall back on timeout. Preserve the identical retained sample sequence
seen by primary ASR and ChunkTap. See the shared frontend contract for the
current gate constants and the recorder's pause/acknowledge/drain ordering.

Lexical rules must be language-aware hints, not grammatical guarantees. A user
may intentionally dictate any token. Quiet-boundary snapping needs retained
audio/context, explicit search bounds and timing-map updates; it cannot promise
that no word ending is ever clipped.

## 7. API and lifecycle constraints

- Reuse the new `TRANSCRIBE_ROLE_VAD` / `VadOps` family boundary where applicable;
  do not invent a second incompatible model-loading API for FireRed.
- Keep preprocessing source-owned and semantic endpoint models optional. Specify
  how they coexist with existing ASR sessions and per-model compute locks.
- Extend existing C structs by appending fields or use a separately versioned
  extension. Never insert a new energy field before existing `vad_threshold`
  or remove `vad_hangover_ms`: `struct_size` cannot repair shifted offsets.
- The integration bumps the internal architecture plugin ABI from 2 to 3
  for the new `VadOps` pointer. Rebuild old plugins; keep this separate from
  the public C ABI layout rules.
- Keep C ABI exception guards, ownership out-params and safe teardown contracts.
  Regenerate Python/TS/Rust ABI layers and review the Swift hash after changes.
- Define lifetimes, buffer ownership, cancellation, reset and stream-finalization
  ordering. A live diagnostic must not require an ASR lease or model load.
- Publish score, raw voiced state, retained state, monotonic speech duration,
  source positions and endpoint decision/reason separately. Throttle app events.

### Proposed frontend contract, before committing to C structs

Use an opaque source context independently of any ASR lease. Feed arbitrary
finite native-rate mono blocks; report consumed source samples and emit
immutable blocks with source span(s), output offset/rate, valid length,
padding/delay, epoch and policy generation. Retained audio needs a piecewise
source map when silence is removed; a single constant timestamp offset is
insufficient. The continuous semantic branch must not inherit that compaction.

Specify running/draining/finished/cancelled/error states. Finish is idempotent,
emits real tails once, and precedes ASR finalize. Reset starts a new epoch;
cancel does not release another source's buffers. No callback may retain a
borrowed buffer after return unless ownership was explicitly transferred.
Define size queries, partial consumption, error behavior, thread affinity and
reentrant callback policy before exposing bindings. These are proposed
contracts, not exported APIs added by this plan.

The existing standalone Silero feed/flush is not a ready-made retention
context: feed returns raw probabilities; offline segmentation defaults drop
segments <=250 ms; flush resets and permits a new stream on the next feed.
Error handling distinguishes invalid-input preservation from processing-error
reset. The adapter must implement mode-specific threshold/retention and own
its real sample count/epoch. Native temporary vectors and Rust result copies
also mean that migration alone does not prove zero allocations.

Capture loss requires positioned gap metadata, not just a dropped-sample total.
Extend the source ledger before claiming exact timelines through overruns.
Likewise, per-model compute serialization and cooperative cancellation need
measured wait/abort bounds; an opaque handle does not create realtime safety.

Do not batch every detector onto a shared LCM frame: 160- and 512-sample
cadences align only every 160 ms. Separate small framers can read one source
ring as soon as each analyzer's input is available.

## 8. Implementation sequence and acceptance gates

- [x] Audit both local projects and identify implicit duplicate VAD.
- [x] Reconcile upstream Silero source names with fork Earshot in the isolated
  VAD integration worktree; retain old public APIs. Merge is uncommitted.
- [x] Verify CPU integration builds with full and empty model sets, ABI
  generators and targeted fixture tests. See the dated
  [verification log](2026-10-vad-integration-verification.md) for passed checks
  and the remaining GPU, plugin, model and app acceptance work.
- [ ] Make VAD ownership explicit in every ZER0 stream builder, including
  benchmark/replay paths. A disable toggle must have a documented whole-path
  meaning. Compare output before changing any default.
- [ ] Remove resampler allocations and reduce audio fan-out copies, as separate
  measured changes. Do not repeat already-completed smoothing-ring work.
- [ ] Implement shared framing, pre-roll, onset and speech-clock contracts with
  regression vectors from the app; migrate UI/history only after parity passes.
- [ ] Complete FireRed intake/reference/port gates; compare against Earshot and
  Silero before enabling a cascade.
- [ ] Complete Smart Turn intake/reference/port gates; add fragment-safe deadline
  policy and multi-language endpoint tests.
- [ ] Evaluate optional engine RNNoise/frontend with CPU, quality and packaging
  evidence; update both projects' integration notes after adoption.

For each accepted behavior/performance change, capture:

| Area | Required evidence |
| --- | --- |
| PCM correctness | Ragged feed sizes, 16/44.1/48 kHz, final tails, reset, finite input, channel selection and sample-count/timestamp mapping |
| Activity | Quiet voices, whispering, room noise, keyboard/clicks, zero input, VAD off, live threshold changes, no duplicate pre-roll |
| Endpointing | Short isolated words, lists/code, hesitation, long turns, no punctuation, hard deadline, cancel and stop during pending onset |
| ASR quality | Paired raw/processed audio, per-family WER/CER, first/last words and R2T2 silence regression |
| App parity | Live test, FFT taps, speaking indicator, monotonic WPM clock, history, files, Live Mode and Multi-STT chunk ownership |
| Performance | Allocation count, CPU time, p50/p95 feed/queue latency, first text, stop-to-final text, memory, sustained overruns; hardware/build/revision pinned |

Distinguish model compute time from audio collection, resampler delay and
endpoint waiting. Existing Apple M4 Silero timings are not Windows or Android
measurements. No mobile battery savings, FireRed speedups, universal accuracy,
or zero-hallucination guarantees are established by this plan.

## 9. Concrete work packages and stop/go decisions

| Package | Deliverable | Gate before proceeding |
| --- | --- | --- |
| P0: frozen paired baseline | Source/build pins, centralized explicit options, raw replay manifest and sample/clock trace | Reproduce current behavior and explain each active gate; no dependency pin refresh as a side effect |
| P1: ownership | CaptureOwned/EngineOwned replay variants and off-toggle matrix | One retention owner selected per source; per-family transcripts, short-command behavior and quiet-skip cost compared |
| P2: framing/lifecycle | Partial-frame carry, valid-length/source mapping, drain/reset/toggle vectors, reusable resampler buffers | Ragged vs contiguous input agrees at the declared frontend contract; real tails retained once; no speech-time inflation from padding |
| P3: fan-out/overload | Shared immutable audio, bounded per-slot audio-time/byte backlog, sequence-aware finalize/cancel | Slow secondary model cannot grow memory indefinitely or block capture; no silent PCM loss |
| P4: acoustic model trials | Calibrated Earshot/Silero/RNNoise-score baselines; then FireRed reference/port if justified | Improve the paired error/CPU tradeoff on held-out data; a three-detector cascade must beat simpler alternatives |
| P5: semantic endpointing | Mode-specific hard deadline, latest-only worker, stale-result tests | Explicit stop remains immediate; resumed speech never commits an obsolete snapshot; timeout behavior bounded |
| V0-V2: VoxLock feasibility | Enrollment/oracle/replay from the dedicated plan | Show useful competing-speaker suppression with acceptable target-word loss and concurrent CPU/latency before a large port |
| P6 / V3-V5: native frontend adoption | Selected RNNoise or VoxLock engine implementation, optional packaging and paired app migration | Model/waveform/state checks, actual app build, sustained workload, rollback and all affected binding/ABI checks pass |

Keep model trials separate from buffer optimizations so a measured gain can be
attributed. Start a modest replay set before any expensive full evaluation;
reserve speaker/device-disjoint held-out recordings for acceptance. Declare
thresholds and allowed regressions before inspecting that split.

Candidate product targets are provisional, not measured promises: added warm
frontend audio-availability p95 <=100 ms for optional isolation; no silent
overruns; no new clipping of curated command boundaries; no unbounded queue.
Record baseline and confidence intervals for WER/CER rather than choosing an
arbitrary universal accuracy number. If isolation needs a higher-delay mode,
keep that explicit instead of relaxing VAD timers to disguise it.

Benchmark 1 and 9 ASR consumers on the actual Windows CPU, with frontend,
semantic worker, FFT diagnostics and UI active as applicable. Include p99 and
maxima, cold/warm start, cancellation under backlog and 20-minute sustained
runs. A single model's RTF <1 does not establish whole-pipeline capacity.

Activity/endpoint ground truth, ASR word error and perceptual enhancement are
different metrics. Report false activations/hour, missed quiet speech, target
deletions and first/last-word loss alongside throughput; for VoxLock include
background-only leakage and overlap contamination. Do not tune on held-out
data or quote a generous timing collar without reporting uncollared errors.

## 10. Primary references

- [Upstream VAD API](https://github.com/handy-computer/transcribe.cpp/blob/de75ca13/docs/vad.md)
- [Upstream Silero family/validation](https://github.com/handy-computer/transcribe.cpp/blob/de75ca13/docs/porting/families/silero_vad.md)
- [FireRed evaluation](https://github.com/FireRedTeam/FireRedVAD#evaluation)
- [FireRed frontend](https://github.com/FireRedTeam/FireRedVAD/blob/main/fireredvad/core/audio_feat.py)
- [FireRed streaming reference](https://github.com/FireRedTeam/FireRedVAD/blob/main/fireredvad/stream_vad.py)
- [Smart Turn training/model](https://github.com/pipecat-ai/smart-turn/blob/main/train.py)
- [Smart Turn inference](https://github.com/pipecat-ai/smart-turn/blob/main/inference.py)
- [Whisper Tiny configuration](https://huggingface.co/openai/whisper-tiny/blob/main/config.json)
- [RNNoise](https://github.com/xiph/rnnoise), [nnnoiseless](https://github.com/jneem/nnnoiseless)
- [Rubato reusable output buffers](https://docs.rs/rubato/5.0.1/rubato/trait.Resampler.html#tymethod.process_into_buffer)

Re-resolve and pin the source/checkpoint revisions during intake; these moving
source links are orientation, not a reproducible oracle packet.

Revision 8 adds pinned-source review and deterministic framing evidence. See
[the evidence review](../research/2026-10-vad-preprocessing-evidence.md) and
[VoxLock implementation track](2026-10-voxlock-voice-isolation-plan.md); previous
CPU integration tests in the verification log were not re-run for this docs
and standalone-probe change. No engine/app audio behavior changed here.

Revision 9 resolves endpoint-versus-text ambiguity, adds implementable source
gap/stop/role contracts and explicit acceptance criteria, and promotes VoxLock
wrapper edge cases from source review to a reproducible identity-stub probe.
This does not establish neural-model quality or end-to-end app performance.
