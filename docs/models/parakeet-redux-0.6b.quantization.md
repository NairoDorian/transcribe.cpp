# Parakeet Redux 0.6B — native ternary conversion

How the GGUF files for [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux)
were produced without leaving the ternary format, the `TQ1_G128` type and kernels that
make that possible, how they were checked, and how to reproduce them.

## 1. Source checkpoint

| | |
|---|---|
| Repository | `moondream/parakeet-redux` |
| Revision | `2bf128600aac4b16946f7ed8372e56117fe5e23b` |
| Files used | `config.json`, `model.safetensors` (177,774,490 B), `ternary.json`, `tokenizer.json` |
| Ternary format | `thrush-ternary-v2` (declared in `ternary.json`) |
| Ternary modules | 264 = 24 layers × {feed_forward1/2 . linear1/2, self_attn . q/k/v/o/relative_k proj, conv . pointwise_conv1/2} — 603,979,776 weights |
| Dense tensors | subsampling convs, depthwise convs, norms, BN stats, LSTM predictor, joint, embeddings — 23,290,911 weights (F16 matrices, F32 vectors) |

`thrush-ternary-v2` stores each module as `<m>.qweight` (uint8 `[out, ceil(in/5)]`,
five base-3 digits per byte, least-significant digit first) and `<m>.scales`
(fp16 `[out, in/128]`); the weight is `scale[r, c // 128] · (code[r, c] − 1)`.

## 2. Why a new ggml type

ggml's existing ternary types, `TQ1_0` (1.69 bpw) and `TQ2_0` (2.06 bpw), carry **one
scale per 256 weights**. Redux carries one per **128**, and adjacent scales differ
(only 0.09 % of 128-group pairs are equal; median difference 19 %). Storing redux in
TQ1_0/TQ2_0 would change the model's weights. CUDA also has no TQ1_0/TQ2_0 support at
all in the vendored ggml. Expanding the codes back to F16/Q8/Q4 (what an earlier draft
of this port did) loses the format's whole point: the result is 7× larger than the
checkpoint.

So transcribe.cpp adds **`GGML_TYPE_TQ1_G128`** (id 96), shipped as the downstream ggml
patch `patches/ggml/0003-tq1_g128-ternary.patch`:

```c
// one block = 256 weights = two independent 128-weight groups, 56 bytes (1.75 bpw)
typedef struct {
    uint8_t   qs[48];   // group g: qs[24g .. 24g+23]
    uint8_t   qh[4];    // group g: qh[2g .. 2g+1]
    ggml_half d[2];     // group g scale (copied from ternary.json scales)
} block_tq1_g128;
```

Inside a group the codes follow TQ1_0's layout at half size: 16 bytes × 5 trits
(elements 0–79, stride 16), 8 bytes × 5 trits (80–119, stride 8), 2 bytes × 4 trits
(120–127, stride 2). Each byte stores its trits most-significant first as
`ceil(v · 256 / 243)`, so trit *n* decodes branch-free as `((uint8_t)(q · 3ⁿ) · 3) >> 8`,
the same trick TQ1_0 uses. The id is parked far above upstream's range (43–95 unused) so a
future ggml sync can never assign the same number to a different format — a collision
that has already bitten other ggml forks.

### Kernels

| Backend | What runs |
|---|---|
| CPU (all) | reference `quantize`/`dequantize`, generic dot product vs Q8_K activations |
| CPU x86-64 | AVX2 dot product: widen 16 code bytes, extract trits with `mullo`/mask/`×3`/`>>8`, `madd` with int8 activations, Q8_K `bsums` supply the `Σq` correction |
| CPU ARM64 / Android | NEON: `vmulq_u8`/`vmull_u8` trit extraction, `vmull_s8` + `vpadalq_s16` accumulation; plain ARMv8 (verified bit-exact against the scalar reference under SIMDe; ggml-cpu cross-compiles for arm64-v8a with NDK 30) |
| CUDA | `dequantize` → F16 + cuBLAS tensor-core GEMM for encoder-sized batches; native `vec_dot_tq1_g128_q8_1` mat-vec (`dp4a`) for batches ≤ 8 |
| Vulkan | dedicated mat-vec shader, branch in the shared tiled mat-mul shader, coopmat2 decode, dequant and get_rows shaders |
| Metal | no native TQ1_G128 kernel; by default the weights run as Q4_0 on Metal's own kernels (see below) |

`test-backend-ops` registers the type (full mul_mat shape sweep), and
`tests/ternary_tq1_g128_unit.cpp` checks lossless packing, the CPU kernel against an
integer reference, and `mul_mat` on every backend in the build.

### Runtime layouts (what actually runs by default)

The native kernels above decode trits inside the dot product, which ggml's CPU
`mul_mat` calls once per (weight row × audio frame) — ~370 decodes per weight per
encoder call. Measurement showed it is faster to re-lay the weights out **losslessly at
load time** into a type with mature GEMM kernels, keeping the 1.75-bpw file:

* ternary `w = s·(c−1)` is exactly **Q4_0** with `q = c + 7`, `d = s` (four 32-blocks
  per group) — `ggml_tq1_g128_to_q4_0`;
* and exactly **Q2_0** with `q = c`, `d = s` (two 64-blocks per group) —
  `ggml_tq1_g128_to_q2_0`.

`load_common::retype_ternary_for_runtime` picks per backend (encoder ms on the 29.3 s
clip, RTX 4070 Laptop):

| Backend | Q4_0 | Q2_0 | native TQ1_G128 | default |
|---|---|---|---|---|
| CPU (x86 AVX2) | **1327** (CPU_REPACK GEMM) | ~2700 | ~4400 | Q4_0 |
| CUDA | 52 | **38** | 47 | Q2_0 |
| Vulkan | **66** | 133 | 114 | Q4_0 |
| Metal | — | — | — | Q4_0 (unmeasured) |

`TRANSCRIBE_TERNARY_RUNTIME=q4_0|q2_0|native` overrides. The patch also adds an AVX2
`ggml_vec_dot_q2_0_q8_0` (x86 previously used the scalar fallback). Accuracy with the
default layouts was re-measured: FLEURS-fr 9.93 % (CPU) / 9.94 % (CUDA) vs 9.91 %
before — unchanged.

## 3. Conversion (`scripts/convert-parakeet.py`)

Same Hugging Face → NeMo rename and tokenizer rebuild as parakeet-ultra (see that
model's QUANTIZATION.md), plus, for every ternary module:

1. decode the thrush bytes to codes {0,1,2} (`scripts/lib/ternary.py
   unpack_thrush_codes`), check the zero-code fraction against `ternary.json`;
2. re-lay-out codes + the original fp16 scales into `TQ1_G128` blocks
   (`pack_tq1_g128`, byte-identical to ggml's `ggml_tq1_g128_pack_codes`);
3. write pointwise-conv kernels 2-D `[in, out]` like linears (quant blocks run along
   the input axis; the parakeet loader accepts quantized pointwise kernels in that
   shape and runs them as `mul_mat`).

Dense tensors keep their upstream dtype (F16 matrices stay F16; 1-D tensors and the
attention position biases are F32, as the loader requires). Result: **179,312,288 B**,
all 697 tensors, 264 of them `TQ1_G128`.

## 4. Dense-part variants

`tools/transcribe-quantize` passes `TQ1_G128` tensors through byte-for-byte for every
preset and only re-types the dense ones, so the ternary part is identical in every file.

| Variant | Dense linear weights | Size | Decision |
|---|---|---:|---|
| TQ1_F16 | F16 | 179,312,288 B | published — bit-exact |
| TQ1_Q8_0 | Q8_0 | 159,121,504 B | published — recommended |
| TQ1_Q4_K | Q4_K on the 2 tensors whose rows allow it, Q8_0 on the other 7 | 156,696,672 B | published — smallest |

Byte budget of TQ1_F16: 132.1 MB ternary, 45.3 MB F16 matrices, 1.6 MB F32. The
presets re-type only the 9 dense matrices (subsampling projection, embedding, LSTM,
joint): 45.3 → 23.7 MB at Q8_0. `transcribe-quantize` stores convolution kernels as
F32 in quantized presets (+1.4 MB). K-quants need rows that are a multiple of 256, so
the Q4_K_M preset can only apply Q4_K to 2 of those 9 matrices (the 640-wide predictor
and joint rows fall back to Q8_0) — hence only 2.4 MB between the Q8_0 and Q4
variants. WER is equal within noise for all three
(9.91 / 9.92 / 9.80 %, below).

```bash
build/bin/transcribe-quantize parakeet-redux-0.6b-TQ1_F16.gguf parakeet-redux-0.6b-TQ1_Q8_0.gguf --quant Q8_0
build/bin/transcribe-quantize parakeet-redux-0.6b-TQ1_F16.gguf parakeet-redux-0.6b-TQ1_Q4_K.gguf --quant Q4_K_M
```

## 5. Validation

1. Reference dumps from transformers `ParakeetForTDT` (fp32, eager attention) with the
   ternary weights dequantized exactly: `scripts/dump_reference_parakeet_hf.py`.
2. `scripts/validate.py compare --family parakeet --variant parakeet-redux-0.6b --gguf
   …-TQ1_F16.gguf`: **18/18 tensors within tolerance**, transcript exact (encoder output
   max |Δ| 9.5e-3; parakeet-ultra's standard Q8_0 reaches 2.9e-2 and passes 16/18).
3. `transcribe_ternary_tq1_g128_unit` — all checks pass; `mul_mat` max relative error vs
   the float reference:

   | Backend | 1–3 tokens (mat-vec) | 64–138 tokens (mat-mul) |
   |---|---|---|
   | CPU (AVX2) | 8.8e-4 – 1.1e-3 | 1.6e-3 |
   | CUDA (RTX 4070) | 7.0e-4 – 8.6e-4 (native mmvq) | 2.8e-4 – 6.5e-4 (dequant + cuBLAS) |
   | Vulkan (RTX 4070, coopmat2) | 1.5e-8 – 2.2e-8 | 5.5e-4 – 6.5e-4 |

   End to end, JFK and German transcripts are identical on CPU, CUDA and Vulkan.
4. FLEURS-fr WER:

   | File | FLEURS-fr WER | 95 % CI |
   |---|---:|---|
   | TQ1_F16 | 9.91 % | 9.18 – 10.66 |
   | TQ1_Q8_0 | 9.92 % | 9.20 – 10.65 |
   | TQ1_Q4_K | 9.80 % | 9.07 – 10.53 |

   Same recipe as parakeet-ultra's card (FLEURS French test, 676 utterances, greedy, no LM,
   CUDA, batch 1). The three files are indistinguishable; the gap to parakeet-ultra
   (6.42 %) is the model's own ternary compression, not the conversion — the C++ output
   matches Moondream's weights run in transformers tensor for tensor.

## 6. Reproduce

```bash
git clone https://github.com/NairoDorian/transcribe.cpp && cd transcribe.cpp
cmake -B build -DTRANSCRIBE_BUILD_TOOLS=ON && cmake --build build --config Release

uv run --project scripts/envs/parakeet scripts/convert-parakeet.py \
    moondream/parakeet-redux --revision 2bf128600aac4b16946f7ed8372e56117fe5e23b
# -> models/parakeet-redux-0.6b/parakeet-redux-0.6b-TQ1_F16.gguf (179,312,288 B)

# kernel tests (add -DTRANSCRIBE_CUDA=ON / -DTRANSCRIBE_VULKAN=ON to cover GPUs)
build/bin/Release/transcribe_ternary_tq1_g128_unit
uv run scripts/lib/test_ternary.py
```

## Checksums

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `parakeet-redux-0.6b-TQ1_F16.gguf` | 179,312,288 | `98f34a4dee8c5cf82a251281717851feda84a3291052e819809706c76bbb758f` |
| `parakeet-redux-0.6b-TQ1_Q8_0.gguf` | 159,121,504 | `74f43ba852479e86e29df92cdbc89aa8215c7e8070f711be424ff466415b6184` |
| `parakeet-redux-0.6b-TQ1_Q4_K.gguf` | 156,696,672 | `24a8b9af6ab1fd05eb33d5e8fc5b00c7459af0108ac397a9635267ea4e814374` |
