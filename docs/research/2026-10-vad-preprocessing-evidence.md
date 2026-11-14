# VAD and preprocessing evidence review

Research date: 2026-10-16. Scope: the local ZER0 and transcribe.cpp pair,
upstream implementations and public model metadata. No new speech-model
quality benchmark was run. Distinguish code observations, deterministic framing
probes, author-reported results and proposed acceptance thresholds below.

Use this with the [active plan](../plans/2026-10-three-tier-super-vad-and-firered-architecture-plan.md)
and [VoxLock review/implementation plan](../plans/2026-10-voxlock-voice-isolation-plan.md).
The [source lock](vad-source-lock-2026-10-16.json) records full revisions and
download metadata; it is not an oracle packet or proof of checkpoint parity.

## 1. Reproducible source selection

| Source | Inspected revision / package |
| --- | --- |
| RNNoise default branch `main` | `70f1d256acd4b34a572f999a05c87bf00b67730d` |
| nnnoiseless actually used by ZER0 | 0.5.2, crate VCS `924a2dd143ccad7bce9e5bda061b60ca32911a67` |
| nnnoiseless current main, orientation only | `49fa1945e997f0fee4814a5038703fb40aaf756c` |
| Rubato actually used by ZER0 | 5.0.1, VCS `eb9190855133fd40d54b17f8d17065ef98ab4db0` |
| FireRedVAD code | `c30ec49e8cc69642b0ee65362eba11b9d11c6e54` |
| FireRed model repository | `7990aaccc6b7aec1e527743bd30201f2c4a03b8c` |
| Smart Turn code | `4786657e242dfe77dd138699ac564ee074a2a543` |
| Smart Turn model repository | `f766f81d3cfdf7737ac64aad813d91bbfd56bf93` |
| Pipecat integration | `f4d1dea8f3f2555074b9a37e97f85c6c43355cb5` |
| Silero upstream reference | `1e261b036686cd0017d500ee96acd1c4ba572a9d` |
| VoxLock source | `08ed37cbb79f449ca6b499bc23fb14138b207bfd` |
| VoxLock model repository latest inspected | `894e78845eca66a7691d330665c60a7c821b8e2d` |
| VoxLock package's actual model pin | `a50f4e1d109301d4b6f61dec57ebb1a61e7d8f03` |

RNNoise's `master` URL returns an older implementation; the default branch is
`main`. Do not benchmark whichever moving URL happens to resolve and call it
the same model as nnnoiseless 0.5.2. Package source, model weights and native
implementation need independent pins.

FireRed Stream-VAD's model artifact is 2,283,513 bytes in this snapshot, not a
verified 1.2 MB GGUF. Smart Turn v3.2 CPU ONNX is 8,679,182 bytes and GPU ONNX is
32,411,198 bytes. ONNX dynamic INT8, GGML Q8 and rounded FP16 are different
quantization contracts; size labels cannot establish numerical equivalence.

## 2. Additional mismatches in the local pair

These observations come from the working sources, not upstream performance
claims. Recheck dirty-file hashes before reproducing them.

| Policy | ZER0 Rust path | Embedded C++ Earshot path | Consequence |
| --- | --- | --- | --- |
| Exit threshold | `max(entry - 0.15, 0.01)` | `max(entry * 0.70, 0.05)` | Both give 0.35 at entry 0.5, hiding disagreement; at 0.8 they give 0.65 vs 0.56 |
| Hangover rounding | Ceiling to 16 ms frames | `max(1, ms / 16)` floor | 450 ms becomes 464 vs 448 ms; 1650 becomes 1664 vs 1648 ms, before transition-state details |
| Onset/pre-roll | SmoothedVad confirms onset and keeps a ring | `prefill_ms` constructor argument does not buffer PCM | Replacing a wrapper is not policy parity |
| Partial ASR feeds | Recorder normally emits complete 256-sample blocks | Feed loop scores only complete 256-sample frames in that call | Remainders are not carried into the next VAD update; ASR still receives them unless the quiet shortcut applies |
| Quiet shortcut | Capture retention policy | Conditional whole-feed bypass | Engine VAD is not a second generic frame cutter |

The engine shortcut additionally requires no family-buffered work, no tentative
text, low RMS, VAD not speaking and family permission. It advances session
cursors while bypassing the family's feed hook. `enable_vad=false` disables
both this Earshot evaluation and that shortcut; do not describe it as changing
only diagnostic telemetry. R2T2's opt-out remains necessary.

Sources: `src/transcribe-asr.cpp`, `src/transcribe-earshot.cpp`; paired app
`audio_toolkit/vad/{mod,earshot,smoothed}.rs` and `audio/recorder.rs`.

## 3. Empirical framing probe: collection delay is real

The probe compiles the app's existing `resampler.rs` and `denoise.rs` with
Rubato 5.0.1 and nnnoiseless 0.5.2 in a separate scratch crate. Input is two
seconds of otherwise zero PCM with a 0.5 impulse at 100 ms. Deliveries are
synthetic 10 ms blocks. Denoising uses strength zero and probability gate zero;
the network still executes, while the mix exposes the delayed dry path.

| Input rate | Path | First output callback after input collected | Impulse displacement vs 100 ms | Output samples after finish |
| --- | --- | --- | --- | --- |
| 16 kHz | Direct | 20 ms | 0 ms | 32,000 |
| 16 kHz | Denoise, strength 0 | 70 ms | 20.688 ms | 33,024 |
| 44.1 kHz | Direct | 30 ms | 5 ms | 32,512 |
| 44.1 kHz | Denoise, strength 0 | 50 ms | 16 ms | 32,768 |
| 48 kHz | Direct | 30 ms | 2.688 ms | 32,256 |
| 48 kHz | Denoise, strength 0 | 30 ms | 12.688 ms | 32,512 |

These are deterministic buffer/signal observations on this source snapshot,
not wall-clock inference latency, microphone latency, onset confirmation,
speech quality or time to first text. First output can contain only silence.
Peak displacement and first emission measure different things.

The current 1024-sample input block alone needs 64 ms at 16 kHz before the
16 -> 48 kHz converter runs. Smaller requested blocks are experiments, not
automatic fixes: rate ratios and the FFT configuration can yield zero output
on an early call, and actual filter delay must be queried.

`finish()` emits filter delay and padded frames. The table's sample counts are
not source durations: 33,024 samples must not become 2064 ms of real speech
for a 2000 ms source. The partial-chunk path currently emits its full padded
resampler output before checking the drain target. A new frontend needs valid
sample counts, padding metadata and a declared trim/alignment policy; preserve
tail bursts instead of blindly truncating the first `input * ratio` samples.

Reproduce from the VAD engine worktree:

```powershell
uv run scripts/vad/probe_zer0_frontend.py --zer0-root C:/Users/Z/Downloads/PROJECTS/Handy_V2
```

The utility uses cached Cargo dependencies offline and writes its CSV, build
log, crate lock and source hashes to ignored `build/zer0-frontend-probe/`.
It does not build ZER0, record a microphone, download weights or alter either
audio implementation. Changing the paired source intentionally changes the
probe input; retain provenance when comparing results.

## 4. FireRed: separate frontend continuity from neural continuity

At the pinned revision, `detect_chunk` retains neural caches but `AudioFeat`
constructs a new `OnlineFbank` inside each extraction and `reset()` is empty.
Thus cache reuse does not prove continuous feature framing. `detect_frame`
expects a 400-sample window; treating consecutive 400-sample calls as 10 ms
hops would also be wrong. [Frontend](https://github.com/FireRedTeam/FireRedVAD/blob/c30ec49e8cc69642b0ee65362eba11b9d11c6e54/fireredvad/core/audio_feat.py),
[stream wrapper](https://github.com/FireRedTeam/FireRedVAD/blob/c30ec49e8cc69642b0ee65362eba11b9d11c6e54/fireredvad/stream_vad.py).

For snipped 400/160-sample framing, one 2560-sample block yields 14 frames;
two independent 1280-sample blocks yield 12 total. This arithmetic proves a
framing discrepancy, not a measured WER regression. Define a full continuous
reference and an upstream-call-compatible reference, document their difference,
and choose the intended oracle before porting. Keep the required 240-sample
window overlap, global hop phase, CMVN/scaling and final-edge policy explicit.

The neural implementation has a conditional lookahead path when `N2 > 0`
and `T > 1`. Extract the actual checkpoint arguments before claiming
chunk invariance or zero neural lookahead. [Model](https://github.com/FireRedTeam/FireRedVAD/blob/c30ec49e8cc69642b0ee65362eba11b9d11c6e54/fireredvad/core/detect_model.py).

Do not stack FireRed's smoothing/start/min-silence defaults on top of the
app's 1650 ms hold and then add a semantic timeout. Score generation and final
retention/endpoint policy need separate owners. Use raw probabilities for the
controlled detector comparison; evaluate postprocessors separately.

## 5. Smart Turn: version, context and scheduling

The pinned repository README describes v3.2 while `inference.py` still names
`smart-turn-v3.1.onnx`; current Pipecat selects `smart-turn-v3.2-cpu.onnx`.
Select one exact checkpoint plus frontend and verify shapes/weights. Do not
use an outdated filename as the download contract. [Example](https://github.com/pipecat-ai/smart-turn/blob/4786657e242dfe77dd138699ac564ee074a2a543/inference.py),
[Pipecat implementation](https://github.com/pipecat-ai/pipecat/blob/f4d1dea8f3f2555074b9a37e97f85c6c43355cb5/src/pipecat/audio/turn/smart_turn/local_smart_turn_v3.py).

Keep the recent turn's continuous audio, including pauses; VAD-compacted ASR
audio destroys timing cues. Do not include earlier turns. A ring for eight
seconds of 16 kHz float32 is 512,000 bytes; this is a storage calculation,
not the model's total memory requirement.

Current Pipecat normalizes the padded waveform before centered STFT/mel and
applies a window-global log-mel floor. Updating the audio window changes those
statistics, so reusing previous normalized mel columns verbatim is not exact.
A noncausal encoder also cannot inherit a streaming ASR cache by name alone.
[Feature implementation](https://github.com/pipecat-ai/pipecat/blob/f4d1dea8f3f2555074b9a37e97f85c6c43355cb5/src/pipecat/audio/turn/smart_turn/_whisper_features.py).

Proposed worker contract: one inference in flight, at most one replaceable
pending snapshot; every result carries recording epoch, turn revision and
the last source sample it saw. Resumed speech invalidates an older completion.
Manual stop does not wait for a semantic model. A hard endpoint deadline must
win over late, failed or repeatedly "incomplete" results. This is a proposed
integration design, not a claim that Pipecat already implements this contract.

## 6. RNNoise: preserve the right baseline

The installed Rust API expects float samples scaled like signed 16-bit PCM,
not normalized floats. ZER0 already scales them. A migration must not scale
twice or omit scaling. Probe strength 0/0.5/1, probability gate off/on, impulses,
tail speech, reset and both sides of live toggles.

Current upstream RNNoise `main` synthesizes a stored `delayed_X` frame and
updates that storage after synthesis, unlike the installed legacy Rust path.
Its README also documents a changed model format and runtime CPU dispatch.
Re-measure delay, dry alignment and flush length for the chosen candidate;
do not copy the app's one-frame flush assumption. [Denoiser](https://github.com/xiph/rnnoise/blob/70f1d256acd4b34a572f999a05c87bf00b67730d/src/denoise.c),
[build/model notes](https://github.com/xiph/rnnoise/blob/70f1d256acd4b34a572f999a05c87bf00b67730d/README).

Treat newer RNNoise with different weights as a quality-plus-performance
candidate, separate from a same-weights C/Rust comparison. Its probability
score can be a low-cost activity experiment when denoising already runs;
enabling the whole denoiser only to get that score is a different cost.

For raw versus denoised VAD experiments, align scores with their source spans.
Do not map the most recently returned 10 ms probability onto every 16 ms output
frame as though it were a calibrated score for that entire frame. The current
callback does exactly that coarse association; sample-span metadata is needed
before relying on the probability as the primary activity signal.

## 7. Buffers, queues and live transitions

ZER0's StreamRouter uses unbounded channels and allocates a PCM Vec per active
model; workers coalesce queued frames into another growing Vec. At full-rate
16 ms delivery, 9 models imply up to 562.5 PCM allocations/s and 576,000 PCM
bytes/s of fan-out. A stalled 60 s backlog can retain about 34.56 MB of PCM,
excluding allocator/channel overhead. These are upper-path calculations for
retained full-rate audio, not measured production allocation counts.

Use shared immutable blocks or a bounded pool with a separately bounded queue.
Arc sharing alone does not bound backlog. Define per-slot audio-time and byte
limits, finalize sequence watermarks and a reserved control path. Never silently
drop arbitrary PCM or let a slow secondary model block source capture. A slot
that cannot keep up must be explicitly stopped/failed or use a declared replay
policy; retained source audio and UI status must make that outcome visible.

The recorder currently resets both resampling chains on a denoise toggle.
Unemitted input, filter tails and an unfinished output frame can be discarded.
That is a concrete transition to test, not proof of an audible failure. A new
frontend must drain up to a source boundary, align the replacement path and
emit each real span once. Crossfade alone cannot repair missing timestamps.

Do not force all analyzers onto a common LCM-sized block: 160-sample FireRed
hops and 512-sample Silero frames have a 2560-sample LCM, or 160 ms. Independent
small adapters over one source ring avoid introducing that batching delay.

## 8. Evidence needed before a default changes

Compare one change at a time, using the same source audio, source-time map,
ASR family/quant, endpoint policy and hardware. Detector F1 alone is insufficient.

1. Freeze the current behavior and explicit ownership variants; record raw
   detector scores separately from their postprocessors.
2. Compare direct, existing RNNoise and lower-block/different-resampler trials;
   keep gates off until alignment and waveform contracts pass.
3. Compare Earshot, Silero and an RNNoise-score policy on the calibrated split.
   Only then evaluate FireRed and cascades.
4. Add Smart Turn as endpoint advice with a bounded deadline; keep acoustic
   retention unchanged during that experiment.
5. Add VoxLock as its own optional enrolled-speaker trial, initially with
   RNNoise off; do not assume several enhancement models improve each other.

Keep a speaker/device-disjoint held-out set. Include short commands, names,
numbers, code, French/English and supported multilingual speech, quiet/whispered
voices, keyboard/room noise, TV speech, a louder competing speaker, overlap and
source switches. Report corpus and per-utterance WER/CER, caller deletions,
first/last-word loss, false activations/hour, background-only word leakage and
premature endpoints. Report boundary errors without a forgiveness collar too.

Latency comes from monotonic capture/queue/compute/text timestamps, while audio
positions come from sample counts. Report p50/p95/p99 and maxima, cold/warm
states, 1 vs 9 ASR consumers, total CPU/memory and sustained overruns. Bootstrap
paired results by speaker/session; do not tune thresholds on the held-out set.
No numeric quality improvement or mobile battery result is established here.

## 9. Additional local-code review: release criteria need strengthening

`docs/roles.md` requires caller-serialized model compute. Rust `with_compute`
locks the model before checking a lifetime ASR streaming lease; abort cannot
instantly cancel a blocked mutex acquisition. Do not run enrollment, history
analysis or a large offline request on a shared realtime model without a
scheduling/resource policy. No worst-case wait has been measured here.

Standalone VAD still allocates temporary joined/remainder vectors and copies
probabilities into Rust results. Its streaming API does not apply offline
segmentation settings; offline defaults discard <=250 ms segments. Invalid
input preserves state/results while processing failure resets them; flush
starts a new frame-zero epoch. These are concrete adapter requirements, not
reasons to replace the tested API blindly.

Recorder overrun totals lack exact gap positions; an epoch/sample descriptor
is required before promising source-time correctness through missing capture.
The existing stop boundary handshake must precede frontend tail drain and ASR
finalize. `ChunkTap` and primary stream must keep identical retained audio.
`multi_stt_stream::break_outcome` also requires text present and bounded drain
lag; a semantic-complete result cannot force a successful chunk close.

The WAV speech-clock probe returns early when no input is configured, and the
real Silero smoke returns 77 without a checkpoint path. A green test command
is therefore not proof that required real-audio cases ran. Acceptance records
need artifact hashes, expected/exercised case counts and explicit skip status.
See [the implementation contract](../plans/2026-10-audio-frontend-contract.md).

## 10. VoxLock wrapper probe: stronger evidence with a narrower claim

The new identity-stub probe reproduces 14 input-length cases and a forced live
switch on the reviewed source hash. <=256 samples produce no output; a
640-sample stream produces 1024 including flush; a 48,000-sample stream produces
48,384. Trimming the tested >=257 inputs restores the identity-stub waveform
within 4.5e-8 max absolute error. A forced enrollment switch also leaves
384 warm-up samples to skip while flush returns 768 without that suppression.

These are executed wrapper observations, not neural inference, enrollment
selection accuracy, runtime speed or ASR quality. The
[VoxLock plan](../plans/2026-10-voxlock-voice-isolation-plan.md) records the
experiment, constraints and corrected-product test requirements. Public ORT
[threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
and [quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
docs further support pinning pool/provider/precision settings for feasibility
tests instead of extrapolating from download size or single-model RTF.
