# Parakeet optimization round — 2026-09-26

Findings and changes from the session that ported `moondream/parakeet-ultra` /
`moondream/parakeet-redux` and then optimized them on CPU, CUDA, Vulkan and Android.
Method reference: `docs/tools/profiling-nsys.md`. Port record:
`docs/porting/parakeet-ultra-redux-master-plan.md`.

Machine: Windows 11, RTX 4070 Laptop (8 GB, driver 617.14, CUDA 13.4), laptop x86
CPU with AVX2. Clip: `samples/german.wav` (29.3 s). Timings: `transcribe-bench`, warm,
mean of 3–5 iterations. WER: FLEURS-fr test, 676 utterances, greedy, CUDA/CPU batch 1.

## 1. Results

### Before → after (total ms for the 29.3 s clip; × = realtime)

| Model / backend | Before | After | Speed-up |
|---|---|---|---|
| ultra Q8_0 — CUDA | 259 ms (110×) | **133 ms (212×)** | 1.95× |
| ultra Q4_K_M — CUDA | ~260 ms | **110 ms (253×)** | 2.4× |
| redux — CUDA | 258 ms (111×) | **109 ms (256×)** | 2.4× |
| ultra Q8_0 — Vulkan | 152 ms (178×) | 130 ms (190×) | ~1.15× (noise-level) |
| redux — Vulkan | 166 ms (163×) | 136 ms (196×) | 1.2× |
| ultra Q4_K_M — CPU | 1.88 s (15.6×) | **1.13 s (26×)** | 1.7× |
| ultra Q8_0 — CPU | 1.92 s | 1.92 s | — (no x86 Q8_0 repack kernel) |
| redux — CPU | 4.54 s (6.5×) | **1.40 s (21×)** | 3.2× |

### Accuracy after all changes (FLEURS-fr WER)

| Path | Before | After |
|---|---|---|
| redux, CUDA (Q2_0 layout + manual attention) | 8.32 % | 8.34 % |
| redux, CPU (Q4_0 layout + CPU_REPACK) | 8.32 % | 8.34 % |
| ultra Q8_0, CUDA | 4.62 % | 4.63 % |
| ultra Q4_K_M, CPU (CPU_REPACK) | 4.98 % | 4.99 % |

All within the bootstrap CIs (±0.7 pp). `validate.py compare` stays 18/18 with exact
transcripts for both models; the redux encoder output moved closer to the reference
(max |Δ| 9.5e-3 → 5.9e-3) because the Q4_0 path quantizes activations per 32 (Q8_0)
instead of per 256 (Q8_K).

## 2. Finding 1 — CUDA was idle 75 % of the time (nsys)

Symptom: CUDA took the same time with F16, Q8_0, Q4_K and ternary weights, and 2.3×
longer than Vulkan in the encoder (194 vs 84 ms); decoder equal on all backends.

Nsight Systems (`nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none … transcribe-bench`):

* Σ kernel time ≈ 57 ms per iteration vs ≈ 250 ms wall → GPU busy ~25 %.
* `cudaStreamSynchronize`: ~740 per iteration (~70 ms), `cudaMemcpyAsync`: ~420.
* Top kernels were ordinary (MMQ Q8_0 28 %, cuBLAS 11 %, subsampling conv2d 8 %).

`GGML_SCHED_DEBUG=2` then showed 97 graph splits with two ops on the CPU in each of the
24 conformer blocks:

1. **`FLASH_ATTN_EXT`** — the rel-pos attention passes a per-head mask
   `[T_k, T_q, n_head]`; CUDA's flash attention rejects `mask->ne[2] != 1`
   (`ggml-cuda/fattn.cu`, `ggml_cuda_get_best_fattn_kernel`). Vulkan supports it.
2. **`SIGMOID`** of the conv module's GLU, applied to a strided view (half of a
   tensor); CUDA unary kernels require contiguous input.

Fixes (graph level, no ggml change):

* `conformer::flash_supports_rel_pos_mask(backend, head_dim, n_head)` probes
  `ggml_backend_dev_supports_op` with the exact FA shape once per backend and the
  parakeet encoder uses the manual `mul_mat + soft_max` attention where the fused op
  is not native (CUDA today). `TRANSCRIBE_FORCE_FLASH=1` overrides.
* GLU: `sigmoid` on the whole contiguous tensor, the value half viewed from the
  result (one extra elementwise op, no CPU fallback).

Result: 1 split on CUDA, Vulkan and CPU; CUDA encoder 194 → 66 ms (Q8_0), 40 ms (redux).
The CUDA kernel for per-head masks would be the next step if manual attention ever
shows up on the profile (it does not today).

## 3. Finding 2 — ternary on CPU was decode-bound, not bandwidth-bound

The native `TQ1_G128` CPU kernel decoded packed trits inside `vec_dot`, which ggml
calls once per (weight row × audio frame): each weight was decoded ~370× per encoder
call. Faster SIMD decode (TQ1 → Q2_0 re-layout, new AVX2 Q2_0 kernel) only got to
2.8 s. The fix was structural:

* **Lossless re-layouts at load** (`ggml_tq1_g128_to_q4_0`, `ggml_tq1_g128_to_q2_0`):
  ternary `w = s·(c−1)` is exactly Q4_0 with `q = c+7, d = s` and exactly Q2_0 with
  `q = c, d = s`. The file stays 1.75 bpw; memory is 4.5 or 2.25 bpw.
  `load_common::retype_ternary_for_runtime` picks per backend
  (`ternary_runtime_default`), `TRANSCRIBE_TERNARY_RUNTIME=q4_0|q2_0|native` overrides.
* **CPU_REPACK GEMM** (`load_common::alloc_cpu_repack_weights`): on a CPU primary the
  encoder's linear + 2-D pointwise weights go to ggml's CPU_REPACK buffer when ggml has
  a repacked kernel for their type on this ISA (x86 AVX2: Q4_0, Q4_K, IQ4_NL, MXFP4;
  ARM dotprod/i8mm also Q5_K, Q6_K, Q8_0). Decoding happens once per tile. Only
  tensors consumed directly as MUL_MAT src0 are eligible (predictor/joint are read on
  the host). The conformer now skips its no-op `reshape_2d` for 2-D pointwise kernels
  so they stay eligible. `TRANSCRIBE_NO_CPU_REPACK=1` disables.

Measured layouts (encoder ms):

| Backend | Q4_0 | Q2_0 | native TQ1_G128 | default |
|---|---|---|---|---|
| CPU | **1327** (repack) | ~2700 | ~4400 | Q4_0 |
| CUDA | 52 | **38** | 47 | Q2_0 (also least VRAM) |
| Vulkan | **66** | 133 | 114 | Q4_0 |
| Metal | — | — | — | Q4_0 (unmeasured; most mature) |

## 4. Other changes and bugs found

* **Vulkan batched crash (pre-existing):** the im2col pointwise-conv branch (Vulkan's
  default) builds its GLU with `view_4d(…, 1, 1)`, pinning batch = 1 →
  `GGML_ASSERT(ggml_can_repeat(b, a))` in batch mode for every dense parakeet on
  Vulkan. Batched graphs (B > 1) now take the batch-aware direct path.
* **New AVX2 `ggml_vec_dot_q2_0_q8_0`** (x86 had only the scalar fallback) — benefits
  any Q2_0 model.
* **Batch parity:** byte-exact on CPU (flash and manual attention) and Vulkan for both
  models. On CUDA, batched vs serial can flip borderline words (1–2 of 8 clips for
  redux, also with the old flash path; ultra 0–1 of 8): kernel selection depends on the
  batch shape. CPU with the same graph is exact, so the logic is correct.
* **Android:** new `android-arm64` CMake preset — runtime-dispatched CPU variants
  (ARMv8.0 … ARMv9.2 SME, dotprod, i8mm) + Vulkan; full build verified with NDK 30.
  See `docs/tools/android-build.md`.

## 5. Where the time goes now and what is next

* **Decoder ≈ 57 ms on every backend** (TDT greedy loop; identical on CPU/CUDA/Vulkan,
  i.e. host-bound). It is now the largest share on GPUs (~45 % of CUDA total). Next
  candidates: keep the predictor/joint step on the device with a persistent graph,
  batch the joint over duration candidates, avoid per-step host reads.
* **Subsampling `conv2d_kernel`** ≈ 4.6 ms per call on CUDA (direct conv on a large
  first layer); an implicit-GEMM or im2col+GEMM path is worth a profile.
* **Vulkan Q4_K_M encoder** is 2× slower than Q8_0 (135 vs 65 ms) — check its matmul
  path/splits with the same method.
* **Metal** has no measurements; run §1–2 of the profiling guide on an Apple machine.
* **CUDA flash attention with per-head masks** (ggml patch) only if manual attention
  becomes visible in the profile.
