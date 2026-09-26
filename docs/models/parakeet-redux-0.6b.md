# Parakeet Redux 0.6B — native ternary GGUF for transcribe.cpp

GGUF files of [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux)
for [transcribe.cpp](https://github.com/NairoDorian/transcribe.cpp), **keeping the model
ternary**: the encoder weights stay {−1, 0, +1} codes with one scale per 128 weights
(1.75 bits/weight) and run on dedicated ternary kernels. Nothing is dequantized on disk.

Parakeet Redux is Moondream's ternary sibling of
[parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra), itself a post-training
of NVIDIA's [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3):
FastConformer encoder + TDT transducer decoder, 25 European languages, 16 kHz mono input,
punctuated and cased output with optional token/word timestamps. Not a streaming model;
does not translate.

- **Parent model:** [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux) @ `2bf128600aac4b16946f7ed8372e56117fe5e23b`
- **Weights:** 604 M ternary (the 264 encoder linear / pointwise-conv matrices) + 23 M dense
- **Languages (25):** bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, ru, sk, sl, es, sv, uk — automatic language detection
- **License:** CC-BY-4.0 (inherited; see [Attribution](#attribution))
- **Format details, kernels, and how to reproduce:** [QUANTIZATION.md](QUANTIZATION.md)

## Files

The ternary encoder weights (604 M, 1.75 bits/weight) are **bit-identical** to
Moondream's in all three files; they only differ in how the 23 M dense (never-ternary)
parameters are stored. For reference the parent is `model.safetensors` 177.8 MB +
`tokenizer.json` 1.2 MB; each GGUF is a single self-contained file including the
tokenizer.

| File | Ternary part | Dense part | Size | Notes |
|---|---|---|---:|---|
| `parakeet-redux-0.6b-TQ1_F16.gguf` | TQ1_G128 (exact) | F16 (exact) | 179.3 MB | Bit-exact copy of the parent's weights |
| `parakeet-redux-0.6b-TQ1_Q8_0.gguf` | TQ1_G128 (exact) | Q8_0 | 159.1 MB | **Recommended** — same accuracy, 11 % smaller than the parent |
| `parakeet-redux-0.6b-TQ1_Q4_K.gguf` | TQ1_G128 (exact) | Q4_K where rows allow, else Q8_0 | 156.7 MB | Smallest; same measured accuracy |

Why TQ1_F16 is 0.9 % larger than `model.safetensors`: it also embeds the tokenizer, and
its fixed 256-weight blocks pack trits per 128-weight group (26 bytes) where Moondream
packs them across whole rows. Against the parent's weights + tokenizer it is 0.17 %
larger.

## Requirements

`TQ1_G128` is a ggml type added by transcribe.cpp (downstream patch
`patches/ggml/0003-tq1_g128-ternary.patch`). You need a transcribe.cpp build that
includes it; stock llama.cpp / ggml cannot read these files.

| Backend | Ternary kernel |
|---|---|
| CPU x86-64 | AVX2 dot product (scalar fallback without AVX2) |
| CPU ARM64 / Android | NEON dot product (ARMv8, no dot-product extension needed) |
| CUDA | native mat-vec kernel (batch ≤ 8) + dequantize-to-F16 tensor-core GEMM (encoder) |
| Vulkan | native mat-vec shader, tiled mat-mul (incl. coopmat / coopmat2), dequant, get_rows |
| Metal | runs the ternary matmuls on the CPU backend (unified memory); a Metal kernel is future work |

## Usage

```bash
cmake -B build && cmake --build build --config Release --target transcribe-cli
build/bin/transcribe-cli -m parakeet-redux-0.6b-TQ1_F16.gguf audio.wav
build/bin/transcribe-cli -m parakeet-redux-0.6b-TQ1_F16.gguf -l de --timestamps word audio.wav
```

## Validation

- **Numerical parity** against the reference (transformers `ParakeetForTDT`, fp32, with
  Moondream's ternary weights dequantized exactly) on `samples/jfk.wav`: 18/18 tensors
  within tolerance with the TQ1_F16 file, transcript identical (for comparison, the
  standard Q8_0 of parakeet-ultra passes 16/18 of the same tolerances).
- **Kernels**: packing is lossless (random codes and scales round-trip exactly); the CPU
  kernel matches an integer reference; `mul_mat` on CPU / CUDA / Vulkan matches a float
  reference (`tests/ternary_tq1_g128_unit.cpp`).
- **Accuracy:**

| File | FLEURS-fr WER | 95 % CI |
|---|---:|---|
| TQ1_F16 | 9.91 % | 9.18 – 10.66 |
| TQ1_Q8_0 | 9.92 % | 9.20 – 10.65 |
| TQ1_Q4_K | 9.80 % | 9.07 – 10.53 |

Same recipe as parakeet-ultra's card (FLEURS French test, 676 utterances, greedy, no LM,
CUDA, batch 1). The three files are indistinguishable; the gap to parakeet-ultra
(6.42 %) is the model's own ternary compression, not the conversion — the C++ output
matches Moondream's weights run in transformers tensor for tensor.
- **Speed** (RTX 4070 Laptop / same laptop CPU, TQ1_F16): CUDA 72× realtime, Vulkan
  131× realtime (after the one-time shader compile), CPU 6× realtime — on CPU 1.8× faster
  than the same weights expanded to F32, because it reads 7× fewer weight bytes.

## Differences from the parent checkpoint

- The 6 `vad_head.*` tensors (Photon runtime's VAD head, ≈213 K parameters) are omitted.
- The ternary codes are re-laid-out from Moondream's per-row base-3 packing
  (`thrush-ternary-v2`) into `TQ1_G128` blocks; codes and FP16 scales are copied exactly.

## Attribution

Model weights: © Moondream, released under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) as
[moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux), derived
from NVIDIA's [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(CC-BY-4.0). This repository changes only the file format (GGUF, ternary blocks
re-laid-out losslessly) and, for the optional variants, the precision of the dense
tensors, as described in QUANTIZATION.md.
