# Confucius4-R2T2 on CUDA — optimization record (2026-09-26)

Model: `r2t2-q4_k_m.gguf` (arm M, 1.187 GB; see
[confucius4_r2t2-quantization.md](families/confucius4_r2t2-quantization.md)).
Rig: RTX 4070 Laptop GPU (8 GB), `transcribe-bench --backend cuda`, warm, mean of 3.
Method: [profiling-nsys.md](../tools/profiling-nsys.md) (phase timings, then
`GGML_SCHED_DEBUG` splits, then nsys).

## Findings

| Lever | Effect | Status |
|---|---|---|
| Arm M instead of Q8_0 | decode ~1.6× faster (decode is weight-bandwidth bound) | shipped model |
| `GGML_CUDA_GRAPHS=ON` | +15 % decode; the per-token graph is small and launch-bound | **default ON** for CUDA builds (top-level `CMakeLists.txt`, Rust `-sys` crate) |
| Tied `embed_tokens` at F16 (622 MB) | the logits matmul alone costs ~2.45 ms/token | arm M stores it Q2_K |
| Cross-model draft prior (`TRANSCRIBE_SPEC_PRIOR_TEXT`) | see below | prototype, opt-in |

End to end, arm M with CUDA graphs runs at ~64× realtime on the 29 s German clip.

Before this change the CMake presets turned graphs on, but the plain CMake
default and the Rust crate used by ZER0 did not. ggml's own default is OFF
("llama.cpp only"). The top-level default is set without `FORCE`, so
`-DGGML_CUDA_GRAPHS=OFF` still wins.

## Cross-model speculative drafting (prototype)

R2T2 already verifies up to `QWEN3_ASR_SPEC_K_MAX` (8) drafted tokens per decode
step. Normally it drafts from its own n-gram history. The prototype adds a second
draft source: the transcript of a faster model on the same audio (for example
parakeet-ultra, ~250× realtime on CUDA). The environment variable
`TRANSCRIBE_SPEC_PRIOR_TEXT` holds that text.

How the draft is chosen:

- **Tokenizing the prior:** it is tokenized with R2T2's own tokenizer. Ids outside
  the decoder vocabulary are dropped.
- **Alignment:** a cursor into the prior is matched by bigram on the last two
  committed tokens. It falls back to a unigram match, searching a 24-token window.
- **Precedence:** seed > prior > own 1-gram.

**Why the output is unchanged:** verification is greedy. R2T2 still decides every
token and the prior only proposes. A different output can only come from
batched-verify numerics (the same effect as ordinary self-drafting).

**Reading the variable on Windows:** it goes through `transcribe::env::utf8()`,
which calls `GetEnvironmentVariableW`. `getenv` returns the ANSI code page there
and mangled accented prompts into invalid token ids.

Results:

- **29 s German clip:** 118× realtime with the prior.
- **64 FLEURS-fr clips:** decode is 2.35× faster; 52 of 64 outputs are
  byte-identical. WER is 6.26 % plain vs 6.90 % with the prior, and the 95 %
  confidence intervals overlap.
- **First 184 FLEURS-fr clips** (the 676-clip run was stopped early to save time): decode is 2.17× faster; 162 of 184 outputs are byte-identical. WER is 6.66 % [5.56, 7.84] plain vs 6.94 % [5.86, 8.11] with the prior, so the intervals overlap.

It stays opt-in until a larger run shows the ~0.3 pp WER gap is noise.

## Encoding bug found on the way

Some FLEURS-fr WER figures were ~1.6–1.8 pp too high. 61 reference transcripts in
`samples/wer/fleurs-fr.64.manifest.jsonl` were mojibake (`é` stored as `Ã©`).
Separately, the bench scripts decoded subprocess output with the Windows ANSI
code page.

Both are fixed:

- The manifest was repaired.
- Every text-mode `open()`, `read_text`/`write_text` and `subprocess(text=True)`
  under `scripts/` now states `encoding="utf-8"`.
- `scripts/ci/check_text_encoding.py` enforces this (`--fix` rewrites in place).

The parakeet ultra/redux WER tables were re-scored with the repaired references.
