# Parakeet Ultra 0.6B — conversion and quantization

How the GGUF files for [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra)
were produced, how they were checked, and how to reproduce them bit-for-bit.

## 1. Source checkpoint

| | |
|---|---|
| Repository | `moondream/parakeet-ultra` |
| Revision | `73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b` |
| Files used | `config.json`, `model.safetensors` (1,255,353,386 B), `tokenizer.json` |
| Tensors | 729: 356 F16 weight matrices, 349 F32 norms/biases/BN statistics, 24 I64 BN counters |
| Format | Hugging Face `transformers` `ParakeetForTDT` (no NeMo `.nemo` archive exists) |

## 2. Conversion (`scripts/convert-parakeet.py`)

transcribe.cpp's parakeet converter was written for NeMo checkpoints. For Hugging Face
checkpoints it:

1. **Renames tensors back to NeMo names** — the inverse of transformers'
   `convert_nemo_to_hf` rename (e.g. `self_attn.q_proj` → `self_attn.linear_q`,
   `relative_k_proj` → `linear_pos`, `bias_u` → `pos_bias_u`, `conv.norm` →
   `conv.batch_norm`, `decoder.lstm` → `decoder.prediction.dec_rnn.lstm`,
   `encoder_projector` → `joint.enc`, `decoder.decoder_projector` → `joint.pred`,
   `joint.head` → `joint.joint_net.2`). Every HF tensor has exactly one NeMo
   counterpart with the same shape; unknown names are a hard error. The rest of the
   converter (the tensor table, LSTM bias fusion `b_ih + b_hh`, BN statistics, shape
   checks) then runs unchanged — the same code path that produces NVIDIA's v3 GGUF.
2. **Synthesizes the NeMo config** from `config.json`. The frontend is not described in
   the HF repo at all; the values transformers' `ParakeetFeatureExtractor` uses (25 ms /
   10 ms Hann windows, 512-point FFT, 128 Slaney mel bins to 8 kHz, per-feature
   normalization, pre-emphasis **0.97**) are exactly v3's and are written as such.
   (Some community conversions of this model wrote pre-emphasis 0.0; that is wrong.)
3. **Rebuilds the tokenizer** from `tokenizer.json` (no SentencePiece `.model` ships):
   8192 BPE pieces in id order, SentencePiece-BPE scores (0 for the 274 user-defined
   pieces, `274 − id` for merged pieces) plus the `<blank>` token 8192. The result is
   byte-identical to the tokenizer arrays in NVIDIA's v3 GGUF.
4. **Drops** the 6 `vad_head.*` tensors (Photon VAD head, not used for ASR) and the 24
   BN `num_batches_tracked` counters.

Output: a 697-tensor GGUF whose tensor names, shapes, tokenizer and hyper-parameter KVs
equal NVIDIA's `parakeet-tdt-0.6b-v3` GGUF; only identity metadata differs.

The converter writes an F32 GGUF (the repository's reference dtype, used for numerical
validation). It is an intermediate and is not published: it is larger than the source.

## 3. Quantization

`tools/transcribe-quantize` (via `scripts/quantize-all.py`) derives each file from the
F32 GGUF with the transcribe.cpp tensor policy: linear/attention/feed-forward weights
take the preset type; convolution kernels stay F16; norms, biases and BN statistics stay
F32. No importance matrix is used.

| File | Linear weights | Size |
|---|---|---:|
| F16 | F16 (lossless w.r.t. the source, which is F16) | 1,255,869,984 B |
| Q8_0 | Q8_0 | 739,508,704 B |
| Q6_K | Q6_K | 610,342,368 B |
| Q5_K_M | Q5_K (Q6_K for sensitive tensors) | 548,946,400 B |
| Q4_K_M | Q4_K (Q6_K for sensitive tensors) | 485,425,632 B |

## 4. Validation

1. **Reference dumps** — `scripts/dump_reference_parakeet_hf.py` runs transformers'
   `ParakeetForTDT` in fp32 with eager attention and dumps mel, subsampling, positional
   encoding, blocks 0/12/23, encoder output, first predictor step, LSTM states and
   joint log-probs, plus a greedy transcript using the exact C++ TDT loop.
2. **C++ comparison** — `scripts/validate.py compare --family parakeet --variant
   parakeet-ultra-0.6b`: **18/18 tensors within tolerance** (tests/tolerances/parakeet.json),
   transcript exact:

   | tensor | max \|Δ\| |
   |---|---|
   | enc.mel.in | 3.0e-4 |
   | enc.block.12.out | 1.8e-1 (values up to ±470) |
   | enc.final | 3.0e-4 |
   | dec.joint.0 | 1.1e-2 |

3. **Smoke tests** — each file transcribes English (JFK) and German clips identically;
   batched decoding (2/4/8) is byte-identical to serial.
4. **WER** — FLEURS French test (676 utterances):

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

## 5. Reproduce

```bash
git clone https://github.com/NairoDorian/transcribe.cpp && cd transcribe.cpp
cmake -B build -DTRANSCRIBE_BUILD_TOOLS=ON && cmake --build build --config Release

# convert (Python env with torch/transformers/gguf: scripts/envs/parakeet)
uv run --project scripts/envs/parakeet scripts/convert-parakeet.py \
    moondream/parakeet-ultra --revision 73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b
# -> models/parakeet-ultra-0.6b/parakeet-ultra-0.6b-F32.gguf (intermediate)

# quantize to the published set
uv run scripts/quantize-all.py models/parakeet-ultra-0.6b/parakeet-ultra-0.6b-F32.gguf

# optional: reference dumps + C++ parity check
uv run scripts/validate.py ref     --family parakeet --variant parakeet-ultra-0.6b
uv run scripts/validate.py cpp     --family parakeet --variant parakeet-ultra-0.6b
uv run scripts/validate.py compare --family parakeet --variant parakeet-ultra-0.6b
```

## Checksums

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `parakeet-ultra-0.6b-F16.gguf` | 1,255,869,984 | `06d3d511e03b2f36aac831f11ce05d088fd071e0fa5685dc70cda1cc7a4a04e4` |
| `parakeet-ultra-0.6b-Q8_0.gguf` | 739,508,704 | `283562ac9b513f39244fe23c6632738c167d32731a5f4693319a10ca498550a8` |
| `parakeet-ultra-0.6b-Q6_K.gguf` | 610,342,368 | `e1c6c0860397473dc4b7831e2da70fa53f5d472d1791ae174982897d3d60ab6a` |
| `parakeet-ultra-0.6b-Q5_K_M.gguf` | 548,946,400 | `2a943b4574664abc96b2dbb2b46cd149bc162fda41080febbf5f949db73ad055` |
| `parakeet-ultra-0.6b-Q4_K_M.gguf` | 485,425,632 | `1865a03092b566251a9a0a7cc1036872821225047759e1f73fa476694bb453f5` |
