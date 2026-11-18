# ZER0 and transcribe.cpp: paired development contract

Last audited: 2026-10-16. This is durable project context, not a claim that two
working trees, branch tips, dependency locks or built binaries are identical.

ZER0 (`Handy_V2`) and this transcribe.cpp fork are complementary projects. The
engine supplies speech inference and reusable audio primitives; ZER0 supplies
the desktop capture pipeline, settings, diagnostics and user workflows. For a
change spanning their boundary, inspect both implementations locally before
deciding where it belongs. Neither project's prose overrides the other's code.

## Find the complementary checkout

| Project | Usual local path | First reads |
| --- | --- | --- |
| Engine | `C:/Users/Z/Downloads/PROJECTS/transcribe-fork` | `AGENTS.md`, `CONTRIBUTING.md`, this note |
| ZER0 app | `C:/Users/Z/Downloads/PROJECTS/Handy_V2` | `AGENTS.md`, `docs/TRANSCRIBE_CPP_INTEGRATION.md`, `docs/PERFORMANCE.md` |

Try the sibling `../Handy_V2` in a normal checkout. In a managed worktree it may
not be a sibling: use the local path above or an explicit user-provided path.
Run `git worktree list` and `git status --short --branch` in both repositories.
Find the worktree for the intended feature branch instead of assuming the
canonical folder is the active implementation. The VAD work uses
`codex/vad-integration`; main and benchmark checkouts may have different code.
Do not switch, reset, merge or overwrite the peer's unrelated changes just to
inspect it. If the peer is unavailable, state what could not be verified.

The app's reciprocal note links back to this contract and the
[current VAD/preprocessing plan](plans/2026-10-three-tier-super-vad-and-firered-architecture-plan.md).

## Trace the boundary in both directions

| Concern | Engine sources | ZER0 sources |
| --- | --- | --- |
| Public PCM/API contract | `include/transcribe.h`, `include/transcribe/vad.h` | `src-tauri/src/managers/transcription.rs`, `src-tauri/Cargo.toml`, `src-tauri/Cargo.lock` |
| Stream defaults, lifecycle and telemetry | `src/transcribe-asr.cpp`, `bindings/rust/transcribe-cpp/src/streaming.rs`, `session.rs` | StreamOptions builders, workers and StreamRouter in `managers/transcription.rs` |
| Embedded Earshot and standalone model VAD | `src/transcribe-earshot*.*`, `src/transcribe-vad-gate.cpp`, `src/transcribe-vad.*`, `src/arch/silero_vad/` | `audio_toolkit/vad/{mod,earshot,smoothed}.rs`, `managers/audio.rs` |
| Suppression, rate conversion and tails | `src/transcribe-rnnoise.*`, `src/transcribe-denoise.cpp`; ASR itself still expects 16 kHz | `audio_toolkit/audio/{denoise,resampler,recorder}.rs` |
| Speech time and endpoint/chunk ownership | VAD telemetry and family stream cursors | `SpeechClock`, `multi_stt_stream.rs`, `multi_streaming.rs`, `live_mode.rs`, `file_transcription.rs` |
| Capture diagnostics and persistence | Preserve API observations and source positions | `live_fft/`, `managers/history.rs`, `settings.rs`, `shortcut/mod.rs` |
| Dependency/build provenance | ABI digest, generated bindings, model composite/plugin options | `scripts/check-transcribe-deps.ts`, `scripts/tauri-runner.ts`, `native_build_identity()` |

If filenames move, locate their symbols with `rg`; do not infer that the feature
was removed. Before Silero integration the fork's Earshot files were also called
`src/transcribe-vad.*`; upstream introduced a different role in those same names.

## Rules for changes affecting both projects

1. Establish the actual consumer: branch/dirty state, Cargo git revision, model
   features and native build identity. `TRANSCRIBE_DIR` or
   `TRANSCRIBE_PREBUILT_DIR` can make the loaded library differ from Cargo.lock.
   A local engine edit is not automatically used by ZER0. Its runner may update
   git pins before a build; account for that during a controlled comparison.
2. Trace one piece of audio from capture to feed/finalize and back to displayed
   text/telemetry. Include off/error/cancel paths, native vs batch, imported
   files, diagnostics without ASR, and Multi-STT sharing.
3. Assign each transform one owner per audio source. Do not silently duplicate
   resampling, denoising or VAD in each model. Keep raw, retained and source-time
   coordinates distinct and preserve reset/tail behavior.
4. Keep implementation choices behind a reusable engine boundary when they are
   audio/model primitives; keep capture devices, UI, history and text insertion
   in the app. Moving code to C++ is not performance evidence.
5. Preserve C ABI layout/lifetimes and regenerate bindings together. Audit real
   Rust builders when adding a defaulted field: omission can enable behavior.
6. Update both integration notes and the active plan when ownership/defaults
   change. Separate implemented, proposed, validated and published states. Date
   factual snapshots and pin measurements; never claim parity from API names.
7. Run the checks appropriate to each changed layer. Engine fixture/API tests
   do not measure microphone buffering, RNNoise, UI or app queue delay; headless
   replay may deliberately bypass capture. Verify the real path before claiming
   an end-to-end improvement. No commit/push/PR is implied by a paired change.

## Baseline traps future agents must recheck

- Since ZER0 `a12ca902` (2026-11-17) the app's capture runs this engine's
  `VadGate` (wrapped as `SpeechGate`) and `Denoiser`; the Rust `earshot` /
  `nnnoiseless` copies, `Hysteresis` and `SmoothedVad` are gone from the app.
  Every native stream it opens passes `enable_vad = false` (live builder,
  `benchmark_stream`, `replay_stream_once`), so the activity gate is no longer
  a second detector in series. A change to `transcribe_vad_gate_*` or
  `transcribe_denoise_*` now changes ZER0's capture: check its gate tests
  (`audio_toolkit/vad/gate.rs`), the recorder tests and
  `tests/vad_speech_clock_probe.rs` against the new build.
- ZER0 feeds the gate exactly one 256-sample frame per `feed` and reads
  `voiced` / `last_score` / `audio`; it does not read the gate's own speech
  clock (its `SpeechClock` counts `voiced` with a per-recording pause hold) and
  never calls `finish` (no partial frame). Since 2026-11-18 (pin `8b27bc6b`)
  it passes **no** `exit_threshold`: the gate derives `max(t - 0.15, 0.01)`
  and re-derives it on every `set_threshold`, which is what ZER0's VAD slider
  needs mid-recording. Until then it passed that value explicitly, and an
  explicit exit is the caller's fixed edge across threshold changes
  (`1f6df3c4`), so the slider moved only the entry edge. A change to
  `derived_exit` changes ZER0: its `gate.rs` test
  `the_exit_edge_follows_every_threshold_change` checks every verdict against
  the rule.
- Architecture plugins: ZER0 builds no `arch-dl` posture. Its
  `arch_plugins.rs` registers, in order, `<exe>/arch`, `<exe>`,
  `<exe>/plugins`, `<app data>/plugins`, `<app data>/arch`; since
  2026-11-18 it no longer registers its models folder, which would have
  undone the model-folder refusal of `502ce99c` (registered directories are
  searched first).
- **Fixed after `8b27bc6b` (2026-11-18), found by ZER0's A/B of `cfb07a12` →
  `8b27bc6b`:** the Parakeet GPU decoder (`1f6df3c4`) was on for every Vulkan
  device, and on an integrated GPU (Intel Iris Xe) it is slower than the host
  decoder: Nemotron batch 490 → 540 ms (decode 107 → 158), Parakeet Redux
  263 → 286, v3 317 → 331, Ultra 429 → 442. `session_device_decoder`
  (`src/arch/parakeet/model.cpp`) now leaves any
  `GGML_BACKEND_DEVICE_TYPE_IGPU` device on the host decoder unless
  `TRANSCRIBE_PARAKEET_DEVICE_DECODER=1`, and logs that at DEBUG. Engine
  check on the same laptop (fleurs-en, decoder stage, warm runs): Nemotron
  Iris Xe default 92 ms vs 156 forced onto the GPU and 89 host; the TDT
  models 36-42 vs 57-69; on the 35 s `dots.wav` the host still wins
  (Nemotron 415 vs 708 ms, TDT 192-219 vs 244-286); the RTX's Vulkan
  default still runs the GPU decoder (21-27 vs 37-49 host); transcripts
  identical in every arm. The choice is per session from that model's own
  device, so a Multi-STT set with one model on CUDA and one on the iGPU
  gets each its own decoder. The app
  needs only the pin bump. Also measured by ZER0: the step pipeline is
  neutral on Vulkan (R2T2 streaming: event waits move into `tensor_get`, the
  step loop is unchanged). Full tables in ZER0's `docs/STT_BENCHMARKS.md`.
- ZER0's `build.rs` copies `share/licenses/transcribe/*` from the install
  prefix (the parent of `DEP_TRANSCRIBE_CPP_INCLUDE_DIR`) into its bundle's
  resources. Keep that install path, or tell the app when it moves.
- RNNoise is optional, 48 kHz / 10 ms, framed by the app. Its probability gate
  is different from acoustic frame retention and semantic turn completion.
- Correction to the 2026-11-14 measurement: at `cfb07a12` a headless Nemotron
  replay of the quiet FLEURS French clip transcribes with the stream gate on
  and off; the empty transcript was measured on the earlier engine.
- Resampler output and per-model stream copies remain separate measurable
  costs.
- The embedded engine speech counter includes hangover; the app clock uses
  deferred short-gap accounting. A common field name does not mean equal WPM.
- Upstream's new Silero role buffers partial frames and segments offline; it
  does not auto-wire the app or implement the proposed FireRed/Smart Turn cascade.
- R2T2 has an explicit silence-skip opt-out. Preprocessing must preserve that
  family constraint and compare transcripts before changing audio retention.

## Scope of the VAD branch

`codex/vad-integration` starts from fork main `692350de` and prepares upstream
VAD `de75ca13` as an uncommitted merge. Fork main is therefore already included.
Latest upstream main was inspected at `b61b7f93`; unrelated later upstream
features are not silently treated as part of this VAD-only merge. Recheck remote
tips and compare histories before a future synchronization.

The [September plan](plans/2026-09-native-earshot-vad-and-zer0-integration-plan.md)
is historical. The [revised VAD plan](plans/2026-10-three-tier-super-vad-and-firered-architecture-plan.md)
defines remaining work; the feature's presence in a plan is not implementation.

The dated [integration verification log](plans/2026-10-vad-integration-verification.md)
records the resolved merge, plugin ABI change, passed checks and remaining
acceptance work. It does not certify end-to-end app adoption.

## Research follow-up: 2026-10-16

The [deeper evidence review](research/2026-10-vad-preprocessing-evidence.md) adds
source pins, a paired-app framing probe, hysteresis/rounding differences,
continuous frontend requirements, queue bounds and live-toggle/tail contracts.
Read the [VoxLock review and implementation plan](plans/2026-10-voxlock-voice-isolation-plan.md)
for optional pre-enrolled voice isolation. It is an audio waveform transform,
not another VAD, and no VoxLock implementation or quality result exists in the
paired projects yet. Keep enhancement per source and inspect both repositories
before adding profiles, model roles, native dependencies or processed taps.

The revision 9 follow-up adds the engine's
`docs/plans/2026-10-audio-frontend-contract.md`: read it for physical/prepared/
retained/text timelines, capture gap positions, the recorder Stop handshake,
model locks and skip-aware acceptance. Preserve Multi-STT's text/drain gate
when adding semantic endpointing. The VoxLock plan now includes an executed
identity-stub wrapper probe; this is not a real-model quality result.
