# Environment Variables

A single reference for the environment variables transcribe.cpp recognizes,
split by audience. The library follows one parse convention for boolean
toggles: a flag is **on** when it is set, non-empty, and its first character is
not `0` (so `FOO=1`, `FOO=on`, `FOO=yes` are on; `FOO=0`, `FOO=` and unset are
off). Path-valued vars are off when unset or empty.

## Tier 1 — runtime configuration

Compiled into every build, including release. These are operational knobs, not
build-gated. Note that `TRANSCRIBE_DUMP_DIR` — while it ships — is still a
tensor-dumping debug/validation hook: enabling it does per-tensor device→host
copies and writes raw activations to disk, so it carries I/O cost and a
data-leak surface. Leave it unset in production unless you are deliberately
collecting dumps.

The two `TRANSCRIBE_TEST_*_THROW` fault-injection hooks also ship in release
binaries on purpose: wheel clean-install CI proves exception containment
against the exact artifact users install (see `AGENTS.md`, "C ABI Exception
Discipline"). Unlike the boolean convention above, **any** non-empty value
arms them (including `0`); unset or empty is inert. Leave them unset outside
of tests.

| Variable | Effect |
| --- | --- |
| `TRANSCRIBE_BACKENDS=<list>` | Restrict which backends may register: a comma-separated list of `cpu`, `metal`, `vulkan`, `cuda`, `rocm`, `other`, `all` (case-insensitive). An excluded backend never runs any code (no module load, no driver init). It can only narrow the mask a host passes to `transcribe_init_backends_ex()`, applies even if the host never calls it, and CPU is always kept. Unknown names are dropped and logged as an error naming what is still allowed, so a typo narrows rather than widens (`vulcan` means CPU-only). Unset or empty means `all`. Read once per process. |
| `TRANSCRIBE_NO_MMAP` | Copy model weights into memory instead of memory-mapping the model file, for every model load in the process (same as `transcribe_model_load_params::no_mmap`, CLI `--no-mmap`). Read at each load. |
| `TRANSCRIBE_NO_GATE_UP_ALIAS` | Disable the load-time gate+up fusion shortcut that points one packed FFN weight tensor at the existing adjacent gate and up weights instead of copying them into a new buffer (granite, canary-qwen, qwen3-asr, voxtral, voxtral-realtime, moss, funasr-nano). Forces the copying path, i.e. the pre-mmapping memory cost. Read at each load; for A/B measurement only. |
| `TRANSCRIBE_NO_FLASH` | Disable flash attention on encoder and decoder (forces the manual F32 path). |
| `TRANSCRIBE_FORCE_FLASH` | Force flash attention on. Wins over `TRANSCRIBE_NO_FLASH` if both are set. Also bypasses the per-backend kernel probe that sends conformer attention to the manual path where the backend has no fused kernel for the shape (CUDA: a per-head rel-pos mask in parakeet / canary / cohere / sortformer, head_dim 48 in gigaam), so the scheduler then runs that attention on the CPU; for A/B only. |
| `TRANSCRIBE_PARAKEET_DEVICE_DECODER=0\|1` | Parakeet TDT / RNN-T: `0` keeps the predictor + joint on the CPU; unset runs them on the model's GPU for CUDA, ROCm and Vulkan; `1` also allows the other GPU backends (Metal, SYCL, ...). Transcripts match the CPU decoder; token confidences may differ in the third decimal. Read when a session first decodes. |
| `TRANSCRIBE_PARAKEET_DEVICE_DECODER_WEIGHTS=native` | Parakeet GPU decoder: multiply with the model's own (quantized) predictor and joint tensors instead of F32 copies. Faster, but the quantized matmul shifts logits enough to change a few transcripts in a hundred, so it is opt-in. |
| `TRANSCRIBE_PARAKEET_DEVICE_JOINT_W=<n>` | Parakeet GPU decoder: frames scored per joint pass (1..256; default 1 for TDT, 16 for RNN-T). For experiments. |
| `TRANSCRIBE_STEP_PIPELINE=0` | Greedy LLM decode (granite, qwen3_asr / R2T2): run the plain step loop instead of the pipelined one, which launches step i+1 while step i runs and feeds the token back on the device. Same tokens either way; for A/B. The CPU backend always uses the plain loop. |
| `TRANSCRIBE_CONV_DIRECT_DW` / `TRANSCRIBE_CONV_NO_DIRECT_DW` | Force the depthwise-conv dispatch to the direct `conv_2d_dw` path / the im2col path, overriding the per-family backend default. |
| `TRANSCRIBE_CONV_DIRECT_PW` / `TRANSCRIBE_CONV_NO_DIRECT_PW` | Force the pointwise-conv dispatch to direct `mul_mat` / im2col, overriding the backend default. |
| `TRANSCRIBE_NO_REPETITION_GUARD` | Turn off the stop for greedy decodes that start repeating themselves, and the trim of a repeating tail at a budget stop (see [`input-limits.md`](input-limits.md)). For byte-exact reference parity; a looping decode then runs to its budget and keeps its repeats. Read once per process. |
| `TRANSCRIBE_EARSHOT_ISA=baseline\|avx2` | Cap the Earshot kernel set picked by cpuid (default: the widest the CPU runs, AVX-512 > AVX2 > baseline). Baseline and AVX2 give bit-identical scores (AVX-512 untested here); for benchmarks and tests. |
| `TRANSCRIBE_DISABLE_ACTIVITY_GATE` | Streaming: never skip a slice through the silence fast path (`transcribe_stream_params::enable_vad` keeps the VAD telemetry either way). The fast path skips a slice only when the family holds nothing, the stream VAD is not speaking and no 16 ms window of the slice is above -42 dBFS; skipped time is mapped back into published timestamps. Read once per process. |
| `TRANSCRIBE_ENCODER_FLASH=0\|1` | Encoder-only flash-attention override, applied after `TRANSCRIBE_NO_FLASH` / `TRANSCRIBE_FORCE_FLASH`. Qwen3-ASR / R2T2 keep encoder flash off by default because it moved 9/100 FLEURS transcripts on CUDA. |
| `TRANSCRIBE_QKV_PACK` | Causal-LM decoders: fuse the Q, K and V projections into one packed matmul. About 2 % faster decode, but it rounds differently (3/100 FLEURS transcripts flipped), so it is opt-in. |
| `TRANSCRIBE_SPEC_PRIOR_TEXT=<text>` | Qwen3-ASR (experimental): use another model's transcript of the same audio as a speculative-decode draft. Only this model's own argmax is committed, so it changes speed, not the token set; the multi-column verify graph is not bit-identical to plain stepping. Read as UTF-8 on Windows. |
| `TRANSCRIBE_R2T2_NO_DRAFT` | R2T2 streaming: re-derive the held-back tail by plain steps instead of verifying last tick's tail as a draft. The A/B arm for text parity. |
| `TRANSCRIBE_R2T2_NO_MEL_CACHE` / `_NO_ENC_CACHE` / `_NO_KV_REUSE` | R2T2 streaming: turn off the incremental mel, the encoder prefix cache or the KV-cache continuation (each makes a tick re-do that work in full). For A/B and bisecting. |
| `TRANSCRIBE_R2T2_TRACE` | R2T2 streaming: log each tick's generated text and held-back tail (makes draft acceptance measurable). |
| `TRANSCRIBE_QWEN3_ASR_ENC_NO_WINDOW` | Qwen3-ASR encoder: one full-attention block instead of windowed (`cu_seqlens`) attention. Diagnostic; the R2T2 encoder prefix cache needs the windowed form. |
| `TRANSCRIBE_RNNT_BATCH_W=<n>` | Parakeet CPU decoder: frames per batched joint pass (`<= 1` forces the serial loop). Defaults: 16 for RNN-T, 1 for TDT. Ignored while tensor dumps are on. |
| `TRANSCRIBE_RNNT_BATCH_CHECK` | Parakeet CPU decoder: cross-check every batched joint column against the serial n=1 GEMV and log mismatches. Debug; disables the GPU decoder. |
| `TRANSCRIBE_CONV_DIRECT_CONV0` / `TRANSCRIBE_CONV_NO_DIRECT_CONV0` | Parakeet non-causal pre-encode: force conv0 to the direct `conv_2d` path / to im2col (default direct except on Metal). |
| `TRANSCRIBE_DISABLE_STATIC_DECODE` | Cohere on a CPU primary: build a per-step decoder graph sized to the current KV length instead of the static step graph. Ignored on GPUs. |
| `TRANSCRIBE_DISABLE_MEL_SIMD` | Compute the mel power spectrum with the scalar loop instead of the SIMD one (the SIMD path fuses a multiply-add, so features can differ in the last bit). For A/B. |
| `TRANSCRIBE_NO_CPU_REPACK` | CPU primary: keep weights in their file layout instead of the CPU backend's repacked GEMM layout (e.g. `q4_K_8x8`). For A/B; repacking is several times faster on x86. |
| `TRANSCRIBE_TERNARY_RUNTIME=q4_0\|q2_0\|native` | Layout TQ1_G128 ternary weights (parakeet-redux) run in. Defaults: `q2_0` on CUDA (fastest and least VRAM), `q4_0` elsewhere; `native` keeps TQ1_G128. |
| `TRANSCRIBE_DISABLE_POOL_RECLAIM` | On a device allocation failure, fail at once instead of trimming this backend's idle CUDA pool and graph cache and retrying once. |
| `TRANSCRIBE_GRAPH_OPTIMIZER` | Run the audio.cpp redundant-copy pass on inference graphs before allocation (opt-in: no consistent wall-time win, see [`cuda-graph-experiments.md`](cuda-graph-experiments.md)). Reports itself only when it changed a graph. |
| `TRANSCRIBE_ARCH_DIR=<dir>` | Architecture-plugin builds (`TRANSCRIBE_ARCH_DL=ON`): a plugin directory searched after the ones registered with `transcribe_register_arch_dir`. Full order: registered dirs, this, the library's `arch/` then its own folder, then (opt-in) the model's folder. The current directory is never searched. |
| `TRANSCRIBE_ARCH_SEARCH_MODEL_DIR` | Architecture-plugin builds: also search the model file's folder (and its `arch/`), last. Off by default because a plugin is native code and model files are often downloaded. |
| `GGML_CUDA_DISABLE_GRAPHS` | ggml: turn off CUDA graph capture (on by default in this fork's builds). Any value disables it, `0` included (ggml checks presence, not the boolean convention above). |
| `TRANSCRIBE_DUMP_DIR=<dir>` | Enable the per-stage tensor dumper; writes `<name>.f32` + `<name>.json` per dumped tensor into `<dir>`. The basis for the numerical-comparison harness (`scripts/compare_tensors.py`). |
| `TRANSCRIBE_PERF_DEBUG` | Print a per-stage timing breakdown to stderr (DEBUG log) on the families that profile (`cohere`, `granite`, `canary`, `canary_qwen`, `moonshine`, `moonshine_streaming`, `moss`, `qwen3_asr`, `whisper`). For whisper, a value containing `cpu` or `all` additionally prints the CPU sub-section breakdown. A value containing `nodes` times every graph node of the first scheduler seen and prints per-op and per-node tables at exit (it defeats fusion, so read it as relative attribution). |
| `TRANSCRIBE_VOXTRAL_REALTIME_STREAM_TIMING` | Print a per-component streaming wall-time breakdown at stream finalize (voxtral_realtime). |
| `TRANSCRIBE_MOONSHINE_STREAMING_ENC_WINDOW=<frames>` | Encoder frames emitted per window by the moonshine_streaming one-shot / batch encoder (default 1024, ~20 s). Windowing bounds encoder memory; it is mathematically exact (each window carries the full sliding-window context) and float-level differences stay at the backend's reduction-order noise. Set small to exercise the windowed path on short clips. Read at each run. |
| `TRANSCRIBE_TEST_DEV_INIT_THROW=<match>` | Fault injection: backend device init (`ggml_backend_dev_init`) throws for devices whose name contains `<match>` (`*` matches every device). Exercises throw → skip → CPU-fallback in backend probing; an explicit backend request fails with `TRANSCRIBE_ERR_BACKEND`. Used by `backend_init_throw_unit` and `scripts/ci/vulkan_degradation_check.py`. |
| `TRANSCRIBE_TEST_TEARDOWN_THROW` | Fault injection: any non-empty value injects a throw after each real free inside the `transcribe::safe_*` teardown wrappers, proving containment without leaking the handle. Used by `teardown_safety_unit`. |

## Tier 2 — validation hooks

Numerical-parity hooks for porting and validation. **Compiled in only when the
library is built with `-DTRANSCRIBE_ENABLE_VALIDATION_HOOKS=ON`** (see the CMake
option of the same name, or the `validation` preset). Release/wheel builds leave
them out entirely — the env-var reads, ref-mel loaders, and dump scaffolding are
all behind the compile guard, so these names do not even appear in a release
binary. In a build *without* the flag, setting any of these has no effect;
`scripts/validate.py` hard-fails `--mel-from-ref` against such a build rather
than silently falling back.

| Variable | Effect |
| --- | --- |
| `TRANSCRIBE_MEL_FROM_REF=<dir>` | Inject a reference log-mel from `<dir>` instead of computing it, so encoder drift can be isolated from frontend drift. Each family reads its own dump filename/layout from the directory (`enc.mel.in.f32` for whisper / voxtral_realtime, `frontend.mel.out.f32` for gigaam, `mel.in.f32` for medasr). Driven by `scripts/validate.py --mel-from-ref`. |
| `TRANSCRIBE_DUMP_ALL_BLOCKS` | Dump every encoder block output (not just mid/last) for a layer-by-layer divergence bisect (parakeet, gigaam). Requires `TRANSCRIBE_DUMP_DIR`. |
| `TRANSCRIBE_DUMP_SUB_BLOCKS=<csv>` | Dump intermediate sub-layer activations (ff1/attn/conv/ff2) for the listed block indices, e.g. `0,12,23` (parakeet). Requires `TRANSCRIBE_DUMP_DIR`. |

Build a validation-capable `build/` with:

```bash
cmake --preset validation        # configures build/ with the hooks ON
cmake --build build --target transcribe-cli
```

When **toggling** `TRANSCRIBE_ENABLE_VALIDATION_HOOKS` on an existing `build/`,
do a clean build of the target — an incremental build can leave a stale object
that did not pick up the changed compile definition:

```bash
cmake --build build --clean-first --target transcribe-cli
```

(`scripts/validate.py` guards against the OFF case: it hard-fails
`--mel-from-ref` when `build/CMakeCache.txt` has the flag off. Longer term, a
dedicated validation build directory would avoid the toggle entirely.)

## Build time — Rust `transcribe-cpp-sys`

Read by `bindings/rust/sys/build.rs` when Cargo builds the native library.
Each one is a `rerun-if-env-changed` trigger.

| Variable | Effect |
| --- | --- |
| `TRANSCRIBE_DIR` (alias `TRANSCRIBE_PREBUILT_DIR`) | Link an existing `cmake --install` prefix instead of building. Cargo features are then inert. |
| `TRANSCRIBE_CMAKE_ARGS` / `CMAKE_ARGS` | Extra CMake configure arguments, applied after the feature-derived ones (a user `-D` wins). Part of the cache key. |
| `TRANSCRIBE_MODEL_SET` | `full` (default), `minimal-multilingual`, `none` or `custom`. Part of the cache key. |
| `TRANSCRIBE_MODELS` | Explicit family list for `custom` (e.g. `parakeet;granite`). Part of the cache key. |
| `TRANSCRIBE_CUDA_ARCHITECTURES` / `CMAKE_CUDA_ARCHITECTURES` | CUDA targets: an explicit list, `auto` (probe this GPU with `nvidia-smi`) or `default` (the distribution set). Unset: `auto` for debug builds, the distribution set for release. |
| `TRANSCRIBE_CACHE_DIR` | Root of the persistent native-build cache (default `%LOCALAPPDATA%\handy\transcribe_cpp_cache`, or `$XDG_CACHE_HOME`/`~/.cache` `/handy/transcribe_cpp_cache`). Give each build lane its own. |
| `TRANSCRIBE_CACHE_KEEP=<n>` | Cache entries kept per root after a save, newest first (default 6). Only directories named like a cache key are removed. |
| `TRANSCRIBE_FORCE_REBUILD` | Ignore a cache hit and build from source (the result is still cached). |
| `TRANSCRIBE_NO_CCACHE` / `TRANSCRIBE_CCACHE_PATH` | Disable / point at ccache or sccache (otherwise found on `PATH` or in `~/.cargo/bin`). sccache never wraps `nvcc` on Windows. |
| `TRANSCRIBE_NO_NINJA` / `TRANSCRIBE_NINJA_PATH` | Disable / point at Ninja (otherwise `PATH` or Visual Studio's bundled copy; the MSVC environment is bootstrapped from `vcvars64.bat` when `cl.exe` is not on `PATH`). |

## Test & tooling

Not part of the library runtime.

**Real-model test model paths.** Built only with
`-DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON`; each test skips (exit code 77) when
its var is unset. Convention: `TRANSCRIBE_<FAMILY>_GGUF`.

| Variable | Test(s) |
| --- | --- |
| `TRANSCRIBE_PARAKEET_GGUF` | `parakeet_real_smoke`, `decoder_smoke` |
| `TRANSCRIBE_PARAKEET_UNIFIED_GGUF` | `parakeet_buffered_stream_eos_smoke`, `stream_offline_interleave_smoke` |
| `TRANSCRIBE_GIGAAM_GGUF` | `gigaam_workspace_release_smoke` |
| `TRANSCRIBE_MULTITALKER_BUNDLE_GGUF` | `parakeet_multitalker_e2e_smoke` |
| `TRANSCRIBE_SORTFORMER_GGUF` | `sortformer_diarize_unit`, `cli_diarize_smoke` |
| `TRANSCRIBE_ECAPA_TDNN_GGUF` | `ecapa_tdnn_real_smoke`, `cli_langid_smoke` (its real-model check) |
| `TRANSCRIBE_COHERE_GGUF` | `cohere_real_smoke`, `cohere_e2e_smoke` |
| `TRANSCRIBE_GRANITE5_CTC_GGUF` | `granite5_ctc_real_smoke`, `granite5_ctc_e2e_smoke` |
| `TRANSCRIBE_WHISPER_GGUF` | `whisper_e2e_smoke`, `whisper_tokenize_parity` |
| `TRANSCRIBE_QWEN3_ASR_GGUF` (+ `_0_6B_GGUF` / `_1_7B_GGUF`) | qwen3_asr smokes / parity |
| `TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF` | moonshine_streaming smokes, `stream_offline_interleave_smoke` |
| `TRANSCRIBE_VOXTRAL_REALTIME_GGUF` | `voxtral_realtime_real_smoke`, `stream_offline_interleave_smoke` |
| `TRANSCRIBE_WHISPER_BIN_*` | whisper.cpp `.bin` parser/e2e fixtures |

Other test/tooling vars:

- `TRANSCRIBE_TEST_AUDIO` — override the `jfk.wav` path for ad-hoc e2e runs.
- `TRANSCRIBE_TEST_SAMPLES_DIR` / `TRANSCRIBE_TEST_FIXTURES_DIR` — sample and
  fixture directories, passed at configure time as compile definitions (not
  runtime env vars).
- Python tooling: `TRANSCRIBE_MODELS_DIR`, `TRANSCRIBE_DUMP_DIR` (shared with
  the library), `HF_TOKEN`, and the binding-level `TRANSCRIBE_BACKEND` /
  `TRANSCRIBE_LIBRARY`. See [`tools/conversion.md`](tools/conversion.md) and
  [`tools/validate.md`](tools/validate.md).
