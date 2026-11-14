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
| Embedded Earshot and standalone model VAD | `src/transcribe-earshot.*`, `src/transcribe-vad.*`, `src/arch/silero_vad/` | `audio_toolkit/vad/{mod,earshot,smoothed}.rs`, `managers/audio.rs` |
| Suppression, rate conversion and tails | Proposed optional frontend; ASR itself still expects 16 kHz | `audio_toolkit/audio/{denoise,resampler,recorder}.rs` |
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

- ZER0 filters with Rust Earshot, but native ASR streams inherit engine VAD on
  through StreamOptions defaults. The app toggle/slider do not synchronize that
  second detector, and the app does not consume its VAD telemetry for the UI.
- RNNoise is currently app-side, optional, 48 kHz / 10 ms. Its probability gate
  is different from acoustic frame retention and semantic turn completion.
- SmoothedVad already has a preallocated ring. The old proposed allocation
  savings are not work still to do. Resampler output and per-model stream copies
  remain separate measurable costs.
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
