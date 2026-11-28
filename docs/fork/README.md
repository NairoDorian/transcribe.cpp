# This fork of transcribe.cpp

`NairoDorian/transcribe.cpp` tracks
[`handy-computer/transcribe.cpp`](https://github.com/handy-computer/transcribe.cpp)
and adds what a real-time dictation engine needs: an audio front end (VAD,
speech gate, noise suppression) that needs no model files, GPU-resident decoding,
lower per-step latency, a language allowlist, two model ports, and a leaner,
cached build. This page lists what differs from upstream and where each piece is
documented. The dated history is in [changelog.md](changelog.md); the
latest whole-fork audit is [audit-2026-11-25.md](audit-2026-11-25.md)
(previous: [audit-2026-11-17.md](audit-2026-11-17.md)).

| | |
|---|---|
| Upstream base | 0.4.2, merged through `a0729fa5` (2026-11-25) |
| ggml | master `d9fe7bff` (v1.0.0 + one examples-only commit) + upstream patches `00xx` + fork patches `01xx` ([patches/ggml](../../patches/ggml)) |
| Divergence | 122 fork commits and 35 merges (16 of them from upstream); the public C ABI is additive only (`struct_size`-versioned fields, new entry points) |
| Consumers | ZER0 (`Handy_V2`, desktop Multi-STT on CUDA, tracks `main`), FreeSpeech (Android arm64, Vulkan, pinned revision) |
| Contract with ZER0 | [../zer0-engine-integration.md](../zer0-engine-integration.md) |
| Roadmap | [../plans/2026-11-fork-master-plan.md](../plans/2026-11-fork-master-plan.md) |

## Fork vs upstream at a glance

Measured 2026-11-17 against upstream at the merge base, built with its own
defaults (RTX 4070 Laptop, i9-13900H, interleaved runs, mean of 3). Details
and method are in [audit-2026-11-17.md](audit-2026-11-17.md); the transcript
column is updated by the 21-model A/B of [audit-2026-11-25.md](audit-2026-11-25.md).

| Workload | Fork speed-up | Transcripts |
|---|---:|---|
| Parakeet TDT 0.6B v3, CUDA, offline | 2.9-3.3x | identical |
| Nemotron 3.5 0.6B, CUDA, offline | 3.5-4.0x | identical |
| Granite Speech 4.1 2B, CUDA (wall clock) | 1.6-1.9x | identical |
| Qwen3-ASR 1.7B, CUDA | 1.06-1.21x | identical |
| Nemotron 3.5 streaming, 160 ms chunks, CUDA / CPU | 1.4-2.5x / 1.3-1.4x | identical, or one comma (on CPU: CPU_REPACK, see the 2026-11-25 audit) |
| Parakeet / Nemotron, CPU, offline | 1.13-1.28x | identical on `jfk` and `dots`; a word or a comma on some other clips (CPU_REPACK) |
| Qwen3-ASR 1.7B, CPU | parity | identical |

## Behaviour that differs from upstream by default

An upstream caller that changes nothing sees these differences:

| Area | Upstream | This fork | Turn off |
|---|---|---|---|
| Streaming silence | every slice reaches the model | `transcribe_stream_params::enable_vad` (default on): Earshot + an energy floor skip silence the model holds nothing for; skipped time is mapped back into every published timestamp | `enable_vad = false`, `TRANSCRIBE_DISABLE_ACTIVITY_GATE=1` |
| CUDA graphs | off | on (`GGML_CUDA_GRAPHS=ON`) | `GGML_CUDA_DISABLE_GRAPHS=1` |
| Parakeet TDT / RNN-T decoder on a discrete CUDA, ROCm or Vulkan GPU | host CPU | the model's GPU (byte-identical transcripts); an integrated GPU keeps the host decoder | `TRANSCRIBE_PARAKEET_DEVICE_DECODER=0` (`=1` forces it onto an integrated GPU) |
| Greedy LLM steps (Granite, Qwen3-ASR, R2T2) on a GPU | launch, wait, read, repeat | pipelined: step i+1 launches while step i runs (same tokens) | `TRANSCRIBE_STEP_PIPELINE=0` |
| Conformer attention on CUDA (Canary, Cohere, Sortformer, GigaAM) | flash kernel, which CUDA lacks for these shapes, so the op ran on the CPU | probed once per shape; the manual path stays on the GPU | `TRANSCRIBE_FORCE_FLASH=1` |
| Parakeet-family encoder on a CPU primary | weights in the file layout | ggml's repacked GEMM (CPU_REPACK), about 1.7x faster; not bit-exact, so a word or a comma can differ from upstream | `TRANSCRIBE_NO_CPU_REPACK=1` |
| GPU memory between runs (Parakeet, Granite, Whisper, SenseVoice, Canary, ...) | the scheduler (and KV caches) stay allocated and only grow | released after every run and streaming finalize; a transcript no longer depends on what the session ran before (upstream's Granite sizes its KV window from the longest earlier clip) | — |
| `transcribe_version()` | `"0.4.2"` | `"0.4.2 <commit> <branch> <build-time> <backend>"` (the leading field is unchanged) | — |

Every switch is listed in [../environment-variables.md](../environment-variables.md).

## Public API the fork adds

All additions are additive: new entry points, and trailing struct fields
gated by `struct_size`, so a caller built against upstream's header keeps
working. Rust wrappers live in `bindings/rust/transcribe-cpp/src/`
(`earshot.rs`, `denoise.rs`, `arch.rs`); Python and TypeScript get the raw FFI.

| Area | C API (`include/transcribe.h` unless noted) | Rust |
|---|---|---|
| Earshot frame VAD | `transcribe_vad_init / _free / _predict_frame / _process_frame / _set_threshold / _reset` | `VoiceActivityDetector` |
| Batch scoring | `transcribe_vad_score_frames(pcm, n, n_threads, out)`: one score per 256-sample frame, the same values a fresh frame-by-frame run gives | `score_frames` |
| Speech gate | `transcribe_vad_gate_params_init / _init / _free / _reset / _set_threshold / _set_hangover_ms / _feed / _finish / _get_tail`. The update struct returns the retained audio, valid until the next call | `VadGate`, `VadGateParams` |
| Noise suppression | `transcribe_denoise_params_init / _init / _free / _reset / _set_params / _process_frame / _flush`, 480-sample frames at 48 kHz (`TRANSCRIBE_DENOISE_*` macros) | `Denoiser`, `DenoiseParams` |
| Stream VAD | `transcribe_stream_params::{enable_vad, vad_threshold, vad_prefill_ms (reserved, no effect), vad_hangover_ms}`; `transcribe_stream_update::{vad_speaking, vad_speech_ms, vad_last_score, audio_level_dbfs}`; `transcribe_stream_set_vad_threshold` | `StreamOptions` / `StreamUpdate` fields, `Stream::set_vad_threshold` |
| Language allowlist | `transcribe_run_params::{n_allowed_languages, allowed_languages, allowlist_letters}`, `TRANSCRIBE_FEATURE_LANGUAGE_ALLOWLIST` | `RunOptions::{allowed_languages, allowlist_letters}` |
| R2T2 streaming | `include/transcribe/r2t2.h`: `transcribe_r2t2_stream_ext` (kind `TRANSCRIBE_EXT_KIND_R2T2_STREAM`), a continuous chunk-size control | raw FFI only |
| Architecture plugins | `transcribe_register_arch_dir`, `transcribe_load_arch_plugin`; plugin ABI `include/transcribe-plugin.h` (`TRANSCRIBE_ARCH_PLUGIN_ABI_VERSION` 5) | `register_arch_dir`, `load_arch_plugin` (feature `arch-dl`) |
| Provenance | `transcribe_build_id()`; `transcribe_version()` carries commit, branch, time and backend | `version()` (no `build_id` wrapper) |

## What the fork adds

### Audio front end (no model to load)

- **Earshot VAD**: pykeio/earshot 1.2.2, ported with SIMD kernels (baseline,
  AVX2, AVX-512 VNNI) and bundled weights. The scores are bit-identical across
  thread counts and streaming vs batch, and between the baseline and AVX2 sets
  (AVX-512 is designed to match but untested here).
  `transcribe_vad_*` scores frame by frame, and `transcribe_vad_score_frames`
  scores whole clips in batch.
- **Speech gate** `transcribe_vad_gate_*`: ZER0's capture VAD, natively. It
  adds hysteresis, onset debounce, a pre-roll ring that never releases a frame
  twice, a hangover, a speech clock and a tail report.
- **Noise suppression** `transcribe_denoise_*`: classic RNNoise (the
  nnnoiseless 0.5.2 algorithm), about 1.4x faster on x86-64. It adds ZER0's
  wet/dry mix and speech-probability gate.
- **Stream VAD** (see the table above): with `enable_vad` (the default) every
  stream update carries telemetry (`vad_speaking`, `vad_speech_ms`,
  `vad_last_score`, `audio_level_dbfs`), and
  `transcribe_stream_set_vad_threshold` retunes it mid-stream. With
  `enable_vad = false` (how ZER0 opens its streams) the detector does not run
  and the fields stay idle.

Contracts: [../vad.md](../vad.md) and
[../plans/2026-10-audio-frontend-contract.md](../plans/2026-10-audio-frontend-contract.md).

### Models

- **parakeet-ultra / parakeet-redux**: ports of the moondream models, including
  the native ternary weight type TQ1_G128 (1.75 bpw, ggml patch `0102`).
  Model cards are in [../models/](../models/); the record is
  [../porting/parakeet-ultra-redux-master-plan.md](../porting/parakeet-ultra-redux-master-plan.md).
- **Confucius4-R2T2**: a `qwen3_asr` variant with native streaming and a
  continuous chunk-size control
  ([include/transcribe/r2t2.h](../../include/transcribe/r2t2.h)). Ticks cost
  what they add, thanks to encoder-prefix, mel and KV-cache continuation. See
  [../porting/families/confucius4_r2t2.md](../porting/families/confucius4_r2t2.md).

### Language allowlist

`transcribe_run_params::allowed_languages` names "the languages I speak".
Mid-sentence switching is allowed, and the model is held to their union both in
its language decision and in the letters it may write.
`allowlist_letters = ANY` restricts the decision only. Detect support with
`TRANSCRIBE_FEATURE_LANGUAGE_ALLOWLIST`. Five families have it: Parakeet
(Nemotron 3.5 through its locale tags; the multilingual TDT models v3, ultra and
redux through the alphabet only), Qwen3-ASR (R2T2 included), Granite, Whisper
and SenseVoice. The others (Canary, Cohere, Voxtral, Moonshine, ...) do not: a
non-empty list there is
`TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE`, never silently ignored.

### Speed: GPU

- Parakeet's predictor and joint run on the device: F32 weight copies, argmax
  on the device, pinned staging. Decode is 2.7x (v3) to 5.2x (Nemotron) faster
  on CUDA.
- `causal_lm::StepPipeline` overlaps greedy steps. The GPU gap between tokens
  drops from about 0.8 ms to about 13 us on Windows CUDA.
- R2T2 decode is about 1.29x faster: KV trim and MMVQ rows (patch `0103`),
  packed QKV and encoder flash (both opt-in).
- The conformer flash probe and Granite's dequantize-once `proj.query` removed
  the last CPU nodes the local models had on CUDA. Check with
  `GGML_SCHED_DEBUG` split counts.

Not bit-exact on GPUs: attention computed on the device instead of on the
CPU (the flash probe), Granite's direct depthwise convolution, and the im2col
`conv0` of a causal Parakeet stem each move a word or a comma against
upstream's CUDA text. All are on by default. Each has a switch that gives
upstream's text: `TRANSCRIBE_FORCE_FLASH=1`, `TRANSCRIBE_CONV_NO_DIRECT_DW=1`,
`TRANSCRIBE_CONV_DIRECT_CONV0=1` (2026-11-26 CUDA A/B,
[audit-2026-11-25.md](audit-2026-11-25.md)).

### Speed: CPU

Bit-exact (same transcripts as upstream):

- A persistent ggml threadpool per CPU backend (OpenMP stays off), parked while
  its session is idle, so a Multi-STT host's idle models do not spin. Pools
  start paused, so a GPU session's CPU fallback pool never spins at all. The
  default thread count (the usable cores, capped at 8) is upstream's.
- The CPU barrier yields its core only when the pool's threads outnumber the
  CPUs the process may use (ggml patch `0104`). Upstream's patch `0001`
  yields after every 4096 spins, which on a machine loaded by other work
  loses a time slice per graph node: SenseVoice took 21-28 s instead of
  3-6 s next to 12 busy processes. Unloaded runs are unchanged.
- A frame-batched RNN-T joint. Transposed predictor/joint and CTC-head weights
  let the matvecs vectorize while each output keeps its summation order.
- Kaldi fbank sums each mel band over its nonzero span only, and the mel power
  spectrum uses SIMD.
- Granite's depthwise convolution is one direct op instead of im2col plus a
  matmul (2.2 s less on a 29 s clip's CPU encode). Its encoder, projector and
  prefill outputs are bit-identical to upstream's on CPU (2026-11-25 tensor
  dumps).
- SenseVoice's and Fun-ASR-Nano's SAN-M depthwise convolution takes the same
  direct op on the CPU: encode 14-18% and 24-25% faster, every dumped stage
  byte-identical to the im2col path (2026-11-26). GPUs keep im2col, where the
  op is not bit-exact; `TRANSCRIBE_CONV_DIRECT_DW=1` opts them in.
- Moonshine keeps both attention caches per head and already padded, so
  decoding no longer re-copies and pads them at every token: decode 2.3x
  faster on CPU, 1.6x on CUDA, output byte-identical. Moonshine streaming, whose
  heads need no padding, hands its cache views to flash attention without
  a per-token copy: decode 12-20% faster.
- Canary's pre-encode depthwise convolutions take the direct op on the CPU
  too (they were 13-17% of its CPU graph as im2col): encode 7-12% faster,
  every dumped tensor byte-identical. GPUs keep im2col.

Not bit-exact, on by default:

- The Parakeet-family encoder runs ggml's repacked GEMM (CPU_REPACK, about
  1.7x faster). A word or a comma can differ from upstream, but FLEURS-en WER
  does not move (CTC 5.46% vs 5.46%, v3 4.66% vs 4.63% without it, 150
  utterances; [audit-2026-11-25.md](audit-2026-11-25.md)).
  `TRANSCRIBE_NO_CPU_REPACK=1` turns it off.

### Memory

- GPU buffers are released after every transcription and streaming finalize.
- CUDA pool-trim and graph-evict hooks (patch `0101`) drive one reclaim-and-retry
  on an allocation failure. It reclaims only the caller's own backend, never a
  sibling model's.

### Build and packaging

- **Model sets**: `TRANSCRIBE_MODEL_SET=full | minimal-multilingual | none |
  custom`; `custom` takes the list in `TRANSCRIBE_MODELS="parakeet;granite"`
  (source folder names; an unknown one is a configure error, and a list
  without `custom` is ignored with a warning). The Rust build treats a
  `TRANSCRIBE_MODELS` list on its own as `custom`.
  `minimal-multilingual` builds `parakeet`, `granite` and `qwen3_asr`.
  Excluded families keep their C ABI symbols as safe stubs
  (`src/transcribe-stubs.cpp`), so bindings always link. The Rust feature
  `minimal-multilingual` selects that set.
- **Architecture plugins** (`TRANSCRIBE_ARCH_DL=ON`, Rust feature `arch-dl`):
  a model-free core plus one `transcribe-arch-<arch>.dll|.so|.dylib` per family.
  `<arch>` is the GGUF's `general.architecture` (for example `granite_speech`,
  `cohere_asr`), not the source folder name. Search order, first match wins:
  1. directories passed to `transcribe_register_arch_dir`;
  2. `TRANSCRIBE_ARCH_DIR`;
  3. the library's `arch/` folder, then its own folder;
  4. the model's folder and its `arch/`, only with
     `TRANSCRIBE_ARCH_SEARCH_MODEL_DIR=1` (a plugin is native code, and model
     files are often downloaded).

  The current directory is never searched. On Windows a plugin's own imports
  resolve only from its folder, the application folder and System32.
  `transcribe_load_arch_plugin` loads one file explicitly.
- **Build provenance**: `transcribe_build_id()` returns
  `"transcribe-build-id: <version> <commit> <branch> <date> <backend>"`, also
  recoverable with `strings` from the binary alone.
- **Rust `build.rs`**: a persistent native build cache keyed by target,
  features, model set, `TRANSCRIBE_MODELS`, CMake args, CUDA arch and a
  source fingerprint. The newest `TRANSCRIBE_CACHE_KEEP` (default 6) entries
  are kept. It also adds Ninja with automatic MSVC environment bootstrap,
  ccache/sccache, CUDA arch auto-probe for dev builds, and a short build path
  on Windows. `TRANSCRIBE_FORCE_REBUILD=1` bypasses the cache.
- Windows: a UTF-8 active-code-page manifest on the executables, and UTF-8
  console output in the CLI. Build guide: [../build-windows.md](../build-windows.md);
  Android: [../tools/android-build.md](../tools/android-build.md).

### Fixes upstream does not have (selected)

- Whisper's CPU static decode: the cross-attention mask is uploaded every step,
  because the allocator reuses set-once inputs (`55504bde`). The batched
  enc-dec loop (Canary, Cohere, Moonshine) got the same fix.
- The conformer `rel_shift` diagonal view corrupted CUDA streaming; it is
  materialized again (`edaacaff`).
- A double `pad_causal` in the conv2/conv5 pre-encode depthwise paths
  (`8e413fef`).
- Streaming text never commits or shows half a UTF-8 character.
- The silence fast path no longer shortens the caller's timeline, and its
  energy check is per 16 ms window, so a quiet onset inside a large feed cannot
  be skipped (see the audit).

### Third-party code the fork adds

Earshot (MIT), RNNoise via nnnoiseless (BSD-3-Clause), cJSON (MIT) and ports
from audio.cpp (Apache-2.0) are compiled into libtranscribe. Their texts live
in [`../../licenses/`](../../licenses/) and `src/third_party/cJSON/LICENSE`, are
listed in [`../../THIRD-PARTY-LICENSES.md`](../../THIRD-PARTY-LICENSES.md), and
`cmake --install` stages them as `share/licenses/transcribe/LICENSE.<name>`.
Ship them with any binary.

### Groundwork not yet wired in

- `src/transcribe-chunking.*`: silence-aligned long-form chunk planning and
  timeline rebasing (master plan B.4). Only its unit test calls it today.
- `TRANSCRIBE_GRAPH_OPTIMIZER=1`: the audio.cpp redundant-copy pass. Opt-in,
  because it showed no consistent wall-time win
  ([../cuda-graph-experiments.md](../cuda-graph-experiments.md)).

## How changes are verified

- `cmake --build build --target transcribe-cli`, `ctest` (73 tests), `cargo test`
  (79), `scripts/ci/clang-format.sh --check`, and
  `cmake -DSRC_DIR=src -P tests/lint_teardown.cmake`.
- **Transcripts, not just tests.** A change that can touch numerics is A/B'd
  against the previous build on FLEURS and the `samples/` clips, and must come
  out byte-identical. Options that move transcripts at all stay opt-in: packed
  QKV, encoder flash, native Parakeet decoder weights, prior-transcript
  drafts. One exception is on by default: R2T2 streaming verifies last tick's
  held-back tail as a draft. Its acceptance is exact-greedy, but the
  multi-column verify graph is not bit-identical to plain steps
  (`TRANSCRIBE_R2T2_NO_DRAFT=1` is the parity arm). The method is in
  [../plans/2026-11-fork-master-plan.md](../plans/2026-11-fork-master-plan.md).
- **Timings are paired.** `scripts/bench/compare.py` interleaves arms; see
  [../pipeline-benchmarks.md](../pipeline-benchmarks.md). Back-to-back suite
  runs drift by up to 18 % and are never cited as an A/B.

## Document map

| Kind | Where |
|---|---|
| Current behaviour and contracts | `docs/*.md`, `docs/models/`, `docs/porting/families/`, this folder |
| Plans (each opens with its date and status) | [../plans/](../plans/) |
| Dated measurement records (true when written; superseded decisions are marked) | [records/](records/), [../cuda-graph-experiments.md](../cuda-graph-experiments.md), [../nemotron-performance-investigation.md](../nemotron-performance-investigation.md), [../porting/](../porting/) `*-2026-09-26.md` |
