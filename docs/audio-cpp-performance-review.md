# GGML and audio.cpp performance review

Reviewed 2026-09-19. Read-only audio.cpp snapshot:
`a7b58a6d3d6ae4143c485266b1c6c09898ad8c72` (677 commits reachable from main).
The history was searched for performance, CUDA, memory, graph/cache, threading,
batching and ASR changes; the relevant implementation diffs were inspected.
This is a prioritized transfer assessment, not a claim that every historical
model implementation was numerically validated.

## GGML v0.24.0

The fork already pins release commit `456172ec733a135778adcd32d00e576a58232e45`
in `ggml/UPSTREAM`, with two recorded downstream patches. The
[release](https://github.com/ggml-org/ggml/releases/tag/v0.24.0) includes CUDA
correctness fixes and quantized matvec improvements; those are already in the
vendored source. HIP/RDNA-specific tuning does not imply an RTX 4070 gain.

Relevant integration checks:

- Conformer uses GGML's scheduler, flash attention and direct depthwise
  convolution. Its precision wrappers still request F32 accumulation where
  required; the deprecated setter remains implemented in GGML. Moving to
  `ggml_prec_set_acc` is API maintenance, not evidence for lowering precision.
- GPU paths still do CPU mel and RNN-T/TDT decoding. Pinning the calling thread
  and keeping competing CPU pools polling can dominate CUDA wall time. The
  current fix removes affinity manipulation and parks inactive pools.
- Graph allocation uses scheduler/gallocr where lifetime reuse is needed;
  persistent KV/state tensors require storage across graph dispatches. Do not
  replace their allocation indiscriminately.
- The installed CUDA backend used for the regression baseline was built with
  `GGML_CUDA_GRAPHS=OFF`. CUDA graph replay is therefore a separate controlled
  experiment, not a benefit available merely from upgrading GGML. Reuse graph
  shapes/storage and measure recapture, allocation, VRAM and transcript parity
  before enabling it in an app build.

## The three requested commits

| Commit | What it adds | Relevance to this STT path |
|---|---|---|
| [07490d8](https://github.com/0xShug0/audio.cpp/commit/07490d8357f0d66e500056dd1451ee9034e85bac) | Explicit operation-lowering selectors and new operation APIs | Selectors alone do no work. These APIs/op enums are downstream additions, not drop-in v0.24 interfaces. |
| [62c664a](https://github.com/0xShug0/audio.cpp/commit/62c664af9c14f6aa0f6b30f65229be0854c4f0c5) | CUDA implementations for selected concat, im2col, norm, 3D-convolution and matmul paths | Potentially useful for matching shapes. Much is video/3D-convolution specific. NVFP4 activation specialization is not a Q6/Q8 Nemotron optimization. |
| [82b4dc3](https://github.com/0xShug0/audio.cpp/commit/82b4dc3a3ce7b4261bcd3eb261f77d4544e9ef8a) | Opt-in SSM gate fusion and tiled F32 3x3 im2col | The selected models do not use the SSM scan/gate chain. Tiled im2col requires at least 32 input channels and output width 32; Nemotron's first pre-encode convolution has one input channel and later depthwise convolutions already use direct kernels. No demonstrated win here. |

These commits can improve matching CUDA workloads, but importing their entire
GGML fork would change APIs, op numbering and backend assumptions. No kernels
were copied without matching-shape measurements and CPU/CUDA numerical tests.
Any future import belongs in `patches/ggml/` with an upstream plan, followed by
the repository sync process; never hand-edit the vendored tree.

## Performance work worth evaluating next

| Priority | audio.cpp evidence | Assessment for transcribe.cpp |
|---|---|---|
| High | [4d383be](https://github.com/0xShug0/audio.cpp/commit/4d383be1bff107e823ffc19120dcb6c78d493c0f) graph-cache eviction and allocation retry | The fork already has an export/trim patch. Audit per-family graph ownership and evict on destruction/pressure, not every warm chunk. Measure VRAM across repeated and concurrent sessions. |
| High | [670d00a](https://github.com/0xShug0/audio.cpp/commit/670d00a023128e65dd474fd1db6c603e8cd3be9b), [b46fe6b](https://github.com/0xShug0/audio.cpp/commit/b46fe6b) packed QKV and bounded/chunked prefill | Profile Qwen/Granite prefill and persistent decode graph inputs. Batch projections or bound prefill workspace only with long-context parity and memory evidence. |
| Medium | [98c8d32](https://github.com/0xShug0/audio.cpp/commit/98c8d327b2e805fcb252754ec8103dd6e940afdf) Q8 matvec/reshape/residual fusion | Candidate for matching autoregressive decoder graphs. It is not a general encoder GEMM or Q6 improvement. Requires shape/use-count/alias checks and a backend-op test before benchmarking. |
| Medium | [78d4770](https://github.com/0xShug0/audio.cpp/commit/78d47706c30ef215ba9ad3559baff309efeb5260) keep inputs on the compute stream | General lesson: avoid per-token host transfers and synchronization. Its HIP Fish-TTS sampler is not directly portable to the CPU RNN-T predictor. Use stage logs to quantify transfer cost first. |
| Medium | [1da6ebf](https://github.com/0xShug0/audio.cpp/commit/1da6ebf7e099d69dcc430302ed3d9d280611d5cb) size Parakeet graphs for actual chunk dispatch | Relevant to long-file graph reserve costs and ceilings, rather than the measured short-stream regression. Test bounded windows with long recordings and nondefault chunk limits. |
| Already substantially present | [3081c63](https://github.com/0xShug0/audio.cpp/commit/3081c631bb2a1cd2f757e1a55dab809cbcd5371b) direct depthwise lowering | Parakeet already selects direct depthwise kernels. Preserve F32 kernel conversion and backend-specific fallback rules. |
| Audit, not blind replacement | [e9a36d4](https://github.com/0xShug0/audio.cpp/commit/e9a36d40659c914d85776f46bbd9fb6e3a74c535) gallocr reuse | Useful warning against allocating all temporary tensors at once. Persisted constants/KV must survive reuse; verify input re-feeding on every cached execution. |
| Cold-load only | [99c3725](https://github.com/0xShug0/audio.cpp/commit/99c3725dc5a09ad7cc01edd44aa3cb46b9385101) Windows memory mapping | Evaluate against transcribe's own loader and model-load metric. Cannot explain or fix warmed per-chunk inference. |

Other reviewed families' changes (SSM/video, TTS CFG, LoRA, phonemizers,
Metal-specific rounding, HIP-only layouts, Turing SASS and Vulkan fixes) do not
provide a substantiated direct gain for these four CPU/RTX STT profiles.
Treat all candidate speedups as hypotheses until the three-run suites and
transcript/numerical gates pass on the actual hardware.

## Graph optimizer experiment

The subsequently requested experiment adapts the compatible subset of
`src/framework/runtime/graph_optimizer.cpp` from audio.cpp (introduced in
`3815635`) into `src/transcribe-graph-opt.h`. Enable it only explicitly with
`TRANSCRIBE_GRAPH_OPTIMIZER=1`. It folds redundant materializations and RHS
broadcast repeats before scheduler allocation in Parakeet, Qwen and Granite.
It preserves protected outputs, storage views and write destinations. It does
not import audio.cpp's unsupported two-sided/unary broadcasting extensions or
direct-CPU metadata-node pruning into our scheduler path. Graph reconstruction
uses the public v0.24 API, not audio.cpp's downstream node-count setter.

The adapted file retains Apache-2.0 attribution and license text. Unit tests
compare actual CPU graph results and check storage/output preservation. The
CPU/CUDA benchmark matrix compares the pass off/on with CUDA graphs off/on;
the option remains experimental until those measurements justify a default.
