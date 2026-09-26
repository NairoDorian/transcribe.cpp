# Port record: `moondream/parakeet-ultra` and `moondream/parakeet-redux`

**Status (2026-09-26): ported, validated, published.** This file is the engineering
record of the port — what was built, what was verified and how, what went wrong on the
way, and what is still open. User-facing documentation lives in:

| | Model doc (HF README body) | Conversion / quantization write-up (HF `QUANTIZATION.md`) |
|---|---|---|
| Ultra | `docs/models/parakeet-ultra-0.6b.md` | `docs/models/parakeet-ultra-0.6b.quantization.md` |
| Redux | `docs/models/parakeet-redux-0.6b.md` | `docs/models/parakeet-redux-0.6b.quantization.md` |

Published: `Nairod785/parakeet-ultra-gguf`, `Nairod785/parakeet-redux-gguf`.

An earlier version of this file (v5.0, "Grand Master Plan") contained several factual
errors and an obsolete roadmap; §5 lists what was wrong so nobody re-imports it from
history.

## 1. The models

| | Ultra | Redux |
|---|---|---|
| Upstream | `moondream/parakeet-ultra` @ `73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b` | `moondream/parakeet-redux` @ `2bf128600aac4b16946f7ed8372e56117fe5e23b` |
| Format | transformers `ParakeetForTDT` safetensors (no `.nemo`) | same + `ternary.json` (`thrush-ternary-v2`) |
| Size | 1,255,353,386 B | 177,774,490 B |
| Architecture | FastConformer-TDT, identical to `nvidia/parakeet-tdt-0.6b-v3` (24 × 1024, 8 heads, conv k=9, 2 × 640 LSTM, durations 0–4) | same; 264 encoder matrices ternary with one fp16 scale per 128 weights |
| Extra tensors | `vad_head.*` (6, Photon-only VAD) — dropped | same |
| Tokenizer | `tokenizer.json`, byte-identical vocab to v3's SentencePiece | identical to Ultra's |

## 2. What was built

1. **Reference dumper** `scripts/dump_reference_parakeet_hf.py` — transformers
   `ParakeetForTDT`, fp32, eager attention; manual safetensors load (drops `vad_head`,
   dequantizes redux's ternary weights exactly for the fp32 reference); zeros-start
   greedy TDT with the exact C++ advance rule; log-softmax joint dump. Driven by
   `scripts/validate.py ref` through the golden manifests
   (`tests/golden/parakeet/parakeet-{ultra,redux}-0.6b.manifest.json`, revision pinned
   via `reference.dump_args`).
2. **Converter** (`scripts/convert-parakeet.py`) — HF safetensors path: HF → NeMo tensor
   rename, synthesized NeMo config, tokenizer rebuilt from `tokenizer.json`
   (`extract_tokenizer_hf`), variant profiles `parakeet-ultra-0.6b` /
   `parakeet-redux-0.6b`, repo-slug aliases, `--revision`. For redux, ternary modules
   are repacked losslessly into `TQ1_G128` and dense tensors keep their upstream dtype
   (`keep_source_dtypes`), output `parakeet-redux-0.6b-TQ1_F16.gguf`.
3. **Native ternary type** `GGML_TYPE_TQ1_G128` (id 96) — downstream ggml patch
   `patches/ggml/0003-tq1_g128-ternary.patch` (verified to apply to a pristine export
   of the vendored tree): type traits, reference quant/dequant, lossless packer, CPU
   kernels (generic, AVX2, NEON), CUDA (dequant → cuBLAS, native mmvq), Vulkan
   (mat-vec, tiled mat-mul + coopmat2, dequant, get_rows), `test-backend-ops`
   registration.
4. **transcribe.cpp runtime** — `TQ1_G128` in the shared linear allowlist
   (`src/transcribe-weights-util.h`); parakeet loads quantized pointwise kernels 2-D
   (`GET_PW`); conformer routes quantized pointwise kernels to the direct `mul_mat`
   path on every backend (Vulkan's im2col default can't take them);
   `transcribe-quantize` passes `TQ1_G128` through byte-for-byte.
5. **Tests** — `tests/ternary_tq1_g128_unit.cpp` (packing, CPU kernel vs integer
   reference, `mul_mat` on every backend), `scripts/lib/test_ternary.py` (9 checks incl.
   the Python packer), NEON kernel run bit-exactly under SIMDe on x86, Android arm64
   cross-compile of ggml-cpu (NDK 30).
6. **Catalog / docs** — `catalog/parakeet-{ultra,redux}-0.6b.json`,
   `scripts/hf_cards/parakeet-{ultra,redux}-0.6b.yaml`, model docs above, family
   capability table (`docs/porting/families/parakeet.md`), quantization tool doc
   (`docs/tools/quantization.md`, "Native ternary weights"), README model list.
7. `scripts/validate.py find_cli` finds Visual Studio multi-config builds.

## 3. Verification summary

| Check | Ultra | Redux (native TQ1_G128) |
|---|---|---|
| `validate.py compare` vs HF fp32 reference (jfk) | 18/18, transcript exact (F32 GGUF) | 18/18, transcript exact |
| Structural equality with NVIDIA's v3 GGUF | 697 tensors, tokenizer, KVs identical | same tensor set |
| Smoke: jfk + German, every published file | pass | pass on CPU, CUDA, Vulkan |
| Batch parity 2/4/8 vs serial | pass | pass |
| Kernel unit test | — | CPU / CUDA / Vulkan pass; NEON bit-exact (SIMDe) |
| FLEURS-fr WER (676 utts) | see model doc | see model doc |

For scale: ultra **Q8_0** only passes 16/18 of the same tensor tolerances, while
native ternary redux passes 18/18 — the ternary kernels are not a precision
compromise.

## 4. Decisions

- **Reference framework = transformers**, not NeMo (no `.nemo` exists).
- **Pre-emphasis 0.97.** Some community GGUFs of Ultra wrote 0.0; wrong.
- **Dither** is metadata only (1e-5 as v3); neither side applies it.
- **Ternary stays ternary.** A first draft dequantized redux to dense F32/F16/Q* —
  correct but up to 14× larger than the checkpoint. Rejected: published files must not
  exceed the upstream checkpoint's size, and the ternary format is the point of redux.
- **New type instead of TQ1_0/TQ2_0.** Those carry one scale per 256 weights; redux has
  one per 128 and adjacent scales differ (0.09 % equal, median 19 % apart), so they
  cannot represent redux exactly. CUDA also lacked any TQ support.
- **Type id 96**, not the next free id: forks that took the next id (e.g. I2_S/I8_S in
  other projects) collided with later upstream types; 43–95 stay unassigned.
- **No published F32** for either model (larger than the source). F32 remains the
  converter's intermediate for Ultra and the validation dtype.
- **Redux dense-part variants** (Q8_0 / Q4 on the 23 M dense weights) ship only if
  they measure equal on WER and save meaningful bytes; outcome in the redux model doc.

## 5. Errors found in the previous plan (v5.0) and handoff notes

- HF BN tensors are `conv.norm.*` (not `conv.batch_norm.*`); pointwise convs are
  `[O, I, 1]`; `decoder_projector` lives under `decoder.`.
- `predictor.vocab` is 8193 (includes blank), not 8192.
- Ternary padding is per row (`ceil(in/5)` bytes), not per 128-group.
- The WER "target 5.80 % on LibriSpeech test-clean" was Moondream's Open ASR
  Leaderboard 7-set average — not a LibriSpeech number and not a gate.
- The dumper would have crashed on redux (2-D ternary weight into a Conv1d slot).
- The `zero_fraction` check was presented as validating the ternary unpack; it is
  permutation-invariant and cannot catch a digit-order error. Digit order was verified
  separately (LSB-first: correlation with Ultra's dense weights 0.60–0.83 vs 0.11–0.17).
- Redux revision `9ad64b97…` was dead (404); re-pinned to `2bf12860…`.
- "C++ emits sub-block taps only under a flag": false — `enc.block.0.{ff1,attn,conv,ff2}`
  are dumped by default; the reference doesn't emit them, compare reports
  `MISSING-right` (non-failing).
- Community-GGUF claims (§3 of v5.0) were never verified and are not relied on.

## 6. Open items

- **Commit and push** the fork (converter, ggml patch, loader, tests, docs) so the
  reproduction instructions in the HF cards resolve.
- Metal kernel for `TQ1_G128` (today: CPU fallback over unified memory).
- A dot-product (`vdotq_s32`) NEON variant and an x86 AVX-512/VNNI variant would be
  faster; current kernels are correct and portable.
- ZER0 (Handy_V2): must ship a transcribe library built with patch 0003 before listing
  redux; ultra works with any current build.
- Stage 6/7 (publication benchmarks on the reference rigs, full WER sweeps across
  FLEURS languages) not run.

## 7. ZER0 (Handy_V2) catalog entries

Ready to paste into `src-tauri/src/catalog/catalog.json` (`models` array). Revisions and
hashes are the published Hub state. ZER0's own ranking fields (`speed_score`,
`accuracy_score`, `recommended`, `recommended_rank`) are left for ZER0 to set.
**Redux requires a transcribe library built from a fork commit that includes
`patches/ggml/0003-tq1_g128-ternary.patch`**; older builds reject its GGUFs
(unknown tensor type 96). Ultra loads in any current build.

```json
[
  {
    "id": "Nairod785/parakeet-ultra-gguf",
    "revision": "906393870d67d285b9a2201b750f101ef11e59ef",
    "slug": "parakeet-ultra-0.6b",
    "name": "Parakeet Ultra 0.6B",
    "architecture": "parakeet",
    "family": "parakeet",
    "parameters": "0.6B",
    "description": "25-language speech-to-text with auto language detection, token-level timestamps.",
    "base_model": "moondream/parakeet-ultra",
    "license": "cc-by-4.0",
    "language_count": 25,
    "languages": [
      "bg",
      "hr",
      "cs",
      "da",
      "nl",
      "en",
      "et",
      "fi",
      "fr",
      "de",
      "el",
      "hu",
      "it",
      "lv",
      "lt",
      "mt",
      "pl",
      "pt",
      "ro",
      "ru",
      "sk",
      "sl",
      "es",
      "sv",
      "uk"
    ],
    "capabilities": {
      "streaming": false,
      "translate": false,
      "lang_detect": true,
      "timestamps": "token"
    },
    "files": [
      {
        "filename": "parakeet-ultra-0.6b-Q4_K_M.gguf",
        "quant": "Q4_K_M",
        "size_bytes": 485425632,
        "sha256": "1865a03092b566251a9a0a7cc1036872821225047759e1f73fa476694bb453f5"
      },
      {
        "filename": "parakeet-ultra-0.6b-Q5_K_M.gguf",
        "quant": "Q5_K_M",
        "size_bytes": 548946400,
        "sha256": "2a943b4574664abc96b2dbb2b46cd149bc162fda41080febbf5f949db73ad055"
      },
      {
        "filename": "parakeet-ultra-0.6b-Q6_K.gguf",
        "quant": "Q6_K",
        "size_bytes": 610342368,
        "sha256": "e1c6c0860397473dc4b7831e2da70fa53f5d472d1791ae174982897d3d60ab6a"
      },
      {
        "filename": "parakeet-ultra-0.6b-Q8_0.gguf",
        "quant": "Q8_0",
        "size_bytes": 739508704,
        "sha256": "283562ac9b513f39244fe23c6632738c167d32731a5f4693319a10ca498550a8"
      },
      {
        "filename": "parakeet-ultra-0.6b-F16.gguf",
        "quant": "F16",
        "size_bytes": 1255869984,
        "sha256": "06d3d511e03b2f36aac831f11ce05d088fd071e0fa5685dc70cda1cc7a4a04e4"
      }
    ],
    "default_quant": "Q8_0"
  },
  {
    "id": "Nairod785/parakeet-redux-gguf",
    "revision": "ff6b0dd210b7f5462e5f451b1e00b0bdc05d2bac",
    "slug": "parakeet-redux-0.6b",
    "name": "Parakeet Redux 0.6B (native ternary)",
    "architecture": "parakeet",
    "family": "parakeet",
    "parameters": "0.6B",
    "description": "25-language speech-to-text with auto language detection, token-level timestamps. Native ternary weights (TQ1_G128).",
    "base_model": "moondream/parakeet-redux",
    "license": "cc-by-4.0",
    "language_count": 25,
    "languages": [
      "bg",
      "hr",
      "cs",
      "da",
      "nl",
      "en",
      "et",
      "fi",
      "fr",
      "de",
      "el",
      "hu",
      "it",
      "lv",
      "lt",
      "mt",
      "pl",
      "pt",
      "ro",
      "ru",
      "sk",
      "sl",
      "es",
      "sv",
      "uk"
    ],
    "capabilities": {
      "streaming": false,
      "translate": false,
      "lang_detect": true,
      "timestamps": "token"
    },
    "files": [
      {
        "filename": "parakeet-redux-0.6b-TQ1_Q4_K.gguf",
        "quant": "TQ1_Q4_K",
        "size_bytes": 156696672,
        "sha256": "24a8b9af6ab1fd05eb33d5e8fc5b00c7459af0108ac397a9635267ea4e814374"
      },
      {
        "filename": "parakeet-redux-0.6b-TQ1_Q8_0.gguf",
        "quant": "TQ1_Q8_0",
        "size_bytes": 159121504,
        "sha256": "74f43ba852479e86e29df92cdbc89aa8215c7e8070f711be424ff466415b6184"
      },
      {
        "filename": "parakeet-redux-0.6b-TQ1_F16.gguf",
        "quant": "TQ1_F16",
        "size_bytes": 179312288,
        "sha256": "98f34a4dee8c5cf82a251281717851feda84a3291052e819809706c76bbb758f"
      }
    ],
    "default_quant": "TQ1_Q8_0"
  }
]
```
