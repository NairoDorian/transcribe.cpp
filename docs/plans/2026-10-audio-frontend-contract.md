# Audio frontend contracts and acceptance gates

Revision 1, 2026-10-16. Proposed design requirements for the
[revision 9 roadmap](2026-10-three-tier-super-vad-and-firered-architecture-plan.md)
and [VoxLock track](2026-10-voxlock-voice-isolation-plan.md). No new exported
API, shipping preset or audio behavior is implemented by this document.

## 1. Four boundaries, four clocks

| Boundary | Authoritative coordinate | Must not be substituted with |
| --- | --- | --- |
| Physical source | Native mono frame index, including known dropped spans; source epoch | Number of samples successfully read from the ring |
| Continuous prepared PCM | 16 kHz sample index plus delay/valid-length mapping | Wall time at callback arrival |
| Retained ASR PCM | Compacted sample index and piecewise source spans | Continuous source time or speech-duration/WPM |
| Text/chunk ownership | Family text publication and coordinator chunk/byte ownership | `audio_committed_ms` alone, a VAD score or semantic completion |

Store 64-bit sample totals. Derive time from accumulated totals with rational
conversion; never sum rounded per-call milliseconds. A hypothetical converter
that floors every independent 147-sample input at 44.1 kHz emits only 15,900
samples after 300 calls instead of the correct 16,000. Rubato already carries
phase; preserve that property when replacing its adapter.

Every output block needs epoch, policy generation, source span(s), rate,
absolute output offset and valid sample count; padding and unavailable source
regions must be distinguishable. Use half-open ranges. A silence-compacted
output requires a piecewise map. Profile/model identity must accompany an
isolated recording; this is provenance, not a speaker-authentication claim.

## 2. Capture discontinuities are not silence

Current `write_input_to_ring` counts dropped mono frames in an atomic total.
`observe_overrun` logs/counts the loss while recording continues; the float
ring does not carry exact gap positions. A total is useful diagnostics but
cannot reconstruct which source spans a neural model missed.

Before claiming exact mapping, add a bounded metadata/descriptor mechanism
that stamps each block's absolute device frame position before attempting the
ring write. Account for channel selection/downmix in mono-frame units. Keep
the platform callback allocation/lock/log free and define metadata-overflow
behavior too. Reconcile audio and descriptors atomically by sequence.

On a gap, advance the physical-source ledger, mark missing audio unknown and
invalidate semantic evidence that crosses it. Define model-specific state
recovery: VAD/denoiser/separator may need a new continuity epoch; ASR needs an
explicit segment/failure policy that preserves already published text. Do not
invent silence to make counters line up, count the gap as voiced time, or let
a compute stall satisfy an acoustic pause timer. Test forced overruns and
device changes separately from intentional user pauses.

## 3. Preserve ZER0's stop handshake

The recorder's existing Stop path requests a callback pause, waits for boundary
acknowledgement while draining, drains the ring again, observes overruns, then
finishes the recording. Its current acknowledgement timeout is two seconds.
A heavy frontend cannot run inside that handshake's capture callback.

Required order for a worker-based frontend:

1. Freeze the source boundary with the existing transport handshake.
2. Drain captured packets through that boundary to the source frontend queue.
3. Process/drain its real resampler, enhancement, overlap and VAD tails once.
4. Publish prepared/retained audio through a final sequence watermark to both
   StreamRouter and ChunkTap, using the same retained sequence.
5. Finalize ASR only after that watermark is consumed; then settle text/chunks.

Manual stop starts draining immediately and never waits for Smart Turn.
Acknowledgement timeout, missing packet, worker failure and cancellation have
distinct outcomes; a successful acknowledgement is not proof that downstream
inference finished. Cancel invalidates the epoch and may discard pending work;
it must not send a stale final response into a new recording.

Frontend states are Running -> Draining -> Finished, with Cancelled/Error exits.
After Finished, feed is an error until explicit reset/new context. Repeated
finish emits no duplicate audio. Keep this frontend contract separate from
Silero's public flush API, which resets and permits the next feed to start a
new stream automatically.

## 4. Semantic endpointing proposes a boundary, not text

ZER0 `multi_stt_stream::break_outcome` closes only with the chunk's text present
and drain lag <= `min(500 ms, configured pause)`. Otherwise it waits, then
retires after pause + 2500 ms. These are current implementation facts, not
new recommended defaults. `docs/MULTI_STT_STREAMING.md` explains the word-loss
and duplication failures this gate prevents.

Define three independent budgets: pause-to-boundary decision, boundary-to-ASR
drain/finalize, and text-to-merge/publication. A 350 ms endpoint target is never
a promise that complete text is available in 350 ms. At a hard endpoint
deadline, stop waiting for semantic advice and issue the boundary/drain
decision; never force a successful text commit while the ASR owns more audio.

Integrate Smart Turn through a boundary proposal containing epoch, turn
revision, observed source end, proposed sample boundary and reason. The
coordinator maps it to retained audio and still enforces ownership. Match the
candidate-pause ID and speech revision, not exact equality with the latest PCM
cursor: continued silence advances that cursor legitimately. New candidate
speech after the snapshot invalidates completion, including speech pending
onset confirmation. Test resume during queued frontend work too. Late or
stale proposals do nothing. If text/drain times out, use the existing
retire/fallback path or an explicitly reviewed successor; never borrow the
next chunk's text to satisfy the deadline.

ChunkTap receives the same retained samples as the primary stream today and
uses a generation token. Preserve that equality after enhancement: using raw
audio for extras and isolated audio for the primary invalidates the shared
span contract. Semantic context remains a separate continuous-audio branch.

## 5. Reuse existing VAD APIs without importing their wrong semantics

| Existing behavior | Integration requirement |
| --- | --- |
| Silero feed returns probabilities only; segmentation params apply to offline run | The streaming slider controls the external policy; changing offline `VadOptions` cannot change feed decisions |
| Offline defaults drop segments <=250 ms | Never use defaults to silently filter short dictation/history; choose and test an explicit mode policy |
| Invalid input preserves pending/result state; processing error or abort resets it | Report error class, invalidate downstream evidence as needed, and never publish an older result as a new score |
| Empty valid feed replaces the last result with an empty one; reset retains the last result | Copy results under the model lock and key telemetry by call/epoch; an accessor alone does not prove new work occurred |
| Flush scores one padded tail, resets frame count, and the next feed starts at frame zero | Track real source length outside frame count and increment the adapter's epoch; repeated flush does not duplicate audio |
| Probability pointer is session-owned; Rust copies it to a Vec under lock | Retain owned copies or a new explicit lifetime contract, never borrowed pointers across async tasks |
| One compute in flight per loaded model; Rust locks per call, ASR streaming also holds a lifetime lease | Serialize role operations, avoid same-model enrollment during streaming, budget lock waiting and cancellation separately |

The C layer relies on the caller's serialization contract; binding locks are
not a universal native scheduler. Profile warmup/offline diagnostics must not
hold the realtime model's lock unboundedly. Abort callbacks only request
cooperative cancellation at defined polling points; they do not preempt a
mutex acquisition or arbitrary inference instantly. Record the longest observed
uninterruptible stage and prevent model/profile destruction until work exits.

Native VAD is not currently allocation-free: feed creates temporary joined and
remainder vectors, flush creates a padded vector, and Rust copies probabilities.
Measure these call sizes before promising allocation savings from migration.
Consider reusable pending storage and caller-owned result buffers as a separate
compatible optimization after parity; do not run this API in the CPAL callback.

## 6. Implementable task order and measurable gates

| Task | Owning layer / first files | Required evidence |
| --- | --- | --- |
| C0: explicit options and modes | App `managers/transcription.rs`, settings/commands; engine Rust options | All three inspected builders plus replay/benchmark callers inventoried; current defaults reproduced before switching ownership |
| C1: source descriptor / gaps | App recorder transport, resampler adapter | Native 16/44.1/48 kHz, selected channel, ragged calls, forced overflow and reset; exact source ledger and visible loss |
| C2: source-context lifecycle | Engine frontend boundary + app recorder Stop/finish | Toggle every frame/block phase, stop/cancel under backlog; real tails once, no padding counted as speech |
| C3: score/retention adapter | Engine VAD API + app SmoothedVad/SpeechClock | Short commands, threshold 0.05/0.5/0.8/0.95, ceil duration parity, partial frames and invalid/error recovery |
| C4: paired fan-out | StreamRouter, ChunkTap, Multi-STT coordinator | Same retained samples/order for primary/extras; bounded audio-time/byte queues; no control-message starvation |
| C5: optional models | Dedicated model plans after C0-C4 | Acoustic/semantic/isolation ablations separated; no new dependency/default justified solely by a model card |

Quality gates to preregister before viewing held-out results: clean/target WER
noninferiority margin +0.3 percentage points, judged by a one-sided 95% paired
speaker/session confidence bound; no newly missing words on a curated command
boundary suite; for a useful VoxLock noisy-speaker preset, aim for >=50%
background-leakage reduction where the baseline leakage is nonzero while also
passing the target-word gate. These are proposed product criteria, not observed
results or universal model guarantees. An underpowered confidence interval is
inconclusive, not a pass; a zero-leak baseline cannot establish reduction.

Keep the existing provisional added-frontend availability p95 <=100 ms target
for warm dictation. Measure p99/maxima, actual source-to-text/stop-to-text and
sustained queue depth under the selected ASR load. A 9-model stress run that
already overruns without a frontend cannot demonstrate 9-model realtime
support by comparison alone. Record the supported load explicitly.

No acceptance job may report success merely because an optional probe returned
early. The app WAV probe skips without its input flag; the real Silero smoke
returns 77 without model paths. Required audio/model hashes, exercised case
counts and non-skip evidence belong in the result manifest. Fixture-only,
identity-stub, real-checkpoint and end-to-end app results remain separate.

## Source map

Engine: `include/transcribe/vad.h`, `src/transcribe-vad.cpp`,
`src/transcribe-session-core.h`, `docs/roles.md`, Rust `model.rs`, `session.rs`
and `vad.rs`. App: recorder Stop/overrun paths, `audio/chunk_tap.rs`,
`multi_stt_stream.rs::break_outcome`, `docs/MULTI_STT_STREAMING.md`,
`tests/vad_speech_clock_probe.rs`. Locate the peer with the
[paired-project contract](../zer0-engine-integration.md); recheck working-file
hashes rather than assuming a branch name fixes the source.
