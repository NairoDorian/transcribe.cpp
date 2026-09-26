# Parakeet Ultra 0.6B — GGUF for transcribe.cpp

GGUF quantizations of [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra)
for [transcribe.cpp](https://github.com/NairoDorian/transcribe.cpp).

Parakeet Ultra is Moondream's full-precision post-training of NVIDIA's
[parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3): the same
FastConformer encoder (24 layers, d_model 1024) and TDT transducer decoder, the same
25-language 8192-piece vocabulary, with better accuracy than v3 across English,
European languages, noisy audio and long-form speech (upstream figures below). It
takes 16 kHz mono audio and returns a punctuated, cased transcript with optional
token/word timestamps. It is not a streaming model and does not translate.

- **Parent model:** [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) @ `73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b`
- **Architecture:** FastConformer-TDT, 627 M parameters (`parakeet` in transcribe.cpp)
- **Languages (25):** bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, ru, sk, sl, es, sv, uk — automatic language detection
- **License:** CC-BY-4.0 (inherited from the parent model; see [Attribution](#attribution))
- **How these files were made and how to reproduce them:** [QUANTIZATION.md](QUANTIZATION.md)

## Files

Every file is at most the size of the parent checkpoint (`model.safetensors`, 1.26 GB).
Pick one; they are alternatives, not parts.

| File | Type | Size | Notes |
|---|---|---:|---|
| `parakeet-ultra-0.6b-F16.gguf` | F16 | 1.20 GB | Exact copy of the parent's F16 weights |
| `parakeet-ultra-0.6b-Q8_0.gguf` | Q8_0 | 705 MB | **Recommended default** — indistinguishable from F16 in our checks |
| `parakeet-ultra-0.6b-Q6_K.gguf` | Q6_K | 582 MB | |
| `parakeet-ultra-0.6b-Q5_K_M.gguf` | Q5_K_M | 523 MB | |
| `parakeet-ultra-0.6b-Q4_K_M.gguf` | Q4_K_M | 463 MB | Smallest; for memory-constrained devices (phones) |

The F16 file is 516 KB larger than the parent's `model.safetensors` because the GGUF
also embeds the tokenizer (upstream ships it separately as `tokenizer.json`); it is
smaller than the parent's weights + tokenizer together. SHA-256 sums are listed in
[QUANTIZATION.md](QUANTIZATION.md#checksums).

## Usage

```bash
# build transcribe.cpp (CPU; add -DTRANSCRIBE_CUDA=ON / -DTRANSCRIBE_VULKAN=ON for GPUs)
cmake -B build && cmake --build build --config Release --target transcribe-cli

# transcribe (language is auto-detected; -l de etc. pins it)
build/bin/transcribe-cli -m parakeet-ultra-0.6b-Q8_0.gguf audio.wav
build/bin/transcribe-cli -m parakeet-ultra-0.6b-Q8_0.gguf -l de --timestamps word audio.wav
```

The model runs on the stock transcribe.cpp parakeet runtime (CPU, CUDA, Vulkan, Metal,
Android/ARM64); no special build flags are needed. The GGUF is structurally identical
to NVIDIA's v3 GGUF (same 697 tensors, tokenizer and hyper-parameters), so any
application that already runs parakeet-tdt-0.6b-v3 through transcribe.cpp can load it.

## Validation

- **Numerical parity** against the reference implementation (Hugging Face
  `transformers` `ParakeetForTDT`, fp32, eager attention) on `samples/jfk.wav`:
  18/18 dumped tensors within tolerance (encoder output max |Δ| 3.0e-4, joint log-probs
  max |Δ| 1.1e-2), transcript identical.
- **Every file** (F16 → Q4_K_M) produces the reference transcript on English (JFK) and
  German test clips; batch decoding (batch 2/4/8) is byte-identical to serial decoding.
- **Accuracy** (measured here):

| File | FLEURS-fr WER | 95 % CI |
|---|---:|---|
| F16 | 4.65 % | 4.22 – 5.13 |
| Q8_0 | 4.62 % | 4.21 – 5.11 |
| Q6_K | 4.66 % | 4.24 – 5.15 |
| Q5_K_M | 4.73 % | 4.29 – 5.21 |
| Q4_K_M | 4.98 % | 4.54 – 5.46 |

FLEURS French test split, all 676 utterances, greedy decoding, no LM, CUDA backend,
batch 1, language hint `fr`; bootstrap 95 % confidence intervals. F16 → Q5_K_M are
indistinguishable; Q4_K_M costs ≈ 0.3 pp.

Upstream-reported accuracy (Moondream's own evaluation, not re-measured here):
Open ASR Leaderboard 7-set average 5.80 % WER (v3: 6.26 %); FLEURS 25-language average
9.55 % (v3: 11.62 %); MUSAN noise 5.82 % (v3: 6.72 %); TED-LIUM long-form 1.94 %
(v3: 2.71 %).

## Speed

Measured with `transcribe-bench` on a 29.3 s clip (`samples/german.wav`), warm, mean of
3 iterations, RTX 4070 Laptop GPU (8 GB) and its laptop x86 CPU (AVX2), transcribe.cpp
after the 2026-09-26 optimization round (see
`docs/porting/parakeet-optimization-2026-09-26.md` in the repo). "×" = times faster than
realtime.

| File | CUDA | Vulkan | CPU |
|---|---|---|---|
| Q8_0 | 133 ms — **212×** | 130 ms — **190×** | 1.92 s — **15×** |
| Q4_K_M | 110 ms — **253×** | 200 ms — **137×** | 1.13 s — **26×** |

On CPU, K-quants and Q4_0 use ggml's repacked GEMM kernels (x86 AVX2: Q4_K; ARM
dotprod/i8mm: also Q5_K, Q6_K, Q8_0), which is why Q4_K_M is the fastest CPU file here.
Accuracy of the optimized paths was re-measured (FLEURS-fr: Q8_0 on CUDA 4.63 %,
Q4_K_M on CPU 4.99 %) and is unchanged.

## Differences from the parent checkpoint

- The 6 `vad_head.*` tensors (≈213 K parameters) are omitted. They belong to
  Moondream's Photon runtime's voice-activity head and are not part of speech
  recognition.
- Decoding is greedy TDT with at most 10 symbols per frame, exactly as in NeMo /
  transformers. No external language model.

## Attribution

Model weights: © Moondream, released under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) as
[moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra), a
post-training of NVIDIA's [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(CC-BY-4.0). This repository changes only the file format (GGUF) and the numeric
precision of the weights as described in QUANTIZATION.md.

> **Correction (2026-09-26):** WER figures published earlier were ~1.6–1.8 pp too high. 61 of the FLEURS-fr reference transcripts in the manifest had been stored mojibake-encoded (`é` → `Ã©`, Windows cp1252 default in the manifest builder), so correct hypotheses scored as errors. The references were repaired and every report re-scored; the numbers here are the corrected ones. The scripts now always pass an explicit encoding (`scripts/ci/check_text_encoding.py`).
