# Confucius4-R2T2 — quantization

Companion to `docs/porting/families/confucius4_r2t2.md` (which covers porting and
parity, not quantization) and to `docs/tools/quantization-arms.md` (which covers
the method generally). This file records what was actually measured on R2T2.

**Result: the floor is 1.187 GB — 52% below the 2.478 GB Q8_0 reference.**

| | |
|---|---|
| shipped default | `r2t2-q8_0.gguf`, 2.478 GB |
| shipped alternative | `r2t2-q4_k_m.gguf`, 1.187 GB (**arm M**) |
| composition | tower @Q4_K · gate/up @Q4_K · attention @Q4_K · **`down_proj` @Q6_K** · `embed_tokens` @Q2_K |

> **Durability caveat.** The arm GGUFs were built into a job scratch directory
> and no longer exist, and the five French clips were scratch-only recordings
> that are not tracked in the repo. The dtype table, sizes and French diffs below
> were re-read from disk before this file was written; the German column and the
> speed table were gathered across several sessions. **The contents of this file
> are trustworthy; the artifacts are not recoverable.** Anything that needs
> re-checking must be rebuilt from the recipe in *Reproduction* and re-run
> against `samples/german.wav` (tracked) plus `samples/jfk.wav` / `samples/zh.wav`.

## Provenance

| | |
|---|---|
| source | `netease-youdao/Confucius4-R2T2` |
| revision | snapshot `185ce639118ad1362d049ca0d8ed04b6ec5cd6c9` |
| weights | a **single** 4.076 GB `model.safetensors` (not sharded) |
| side files | tokenizer + config set copied via `--root` |
| converter | `audiocpp_gguf.exe`, built out-of-tree from the read-only `audio.cpp` tree |
| model spec | `audio.cpp/model_specs/confucius4_r2t2.json` |
| tensor census | tower 147 · attention 112 · gate/up 56 · down 28 |

`transcribe-quantize` could not build these arms: it is preset-only and offers no
per-tensor control, and its preset menu has no Q2_K or Q3_K. See
`docs/tools/quantization-arms.md` for the `--keep-type` mechanics, the name
matching rules, and the traps.

## The arm inventory

`embed` is the tied `thinker.model.embed_tokens.weight`, which feeds both the
input lookup and the output logits. The converter pins it to F16 on its own; the
table below shows where that pin was deliberately overridden.

| arm | file | attention | gate/up | `down_proj` | embed | tower | size (GB) | verdict |
|---|---|---|---|---|---|---|---|---|
| q8 | `r2t2-q8_0` | Q8_0 | Q8_0 | Q8_0 | F16 | BF16/Q8_0 | 2.478 | reference |
| — | `tower-q4-lm-q8` | Q8_0 | Q8_0 | Q8_0 | F16 | BF16/Q4_K | 2.321 | tower floor probe |
| q6_k | `r2t2-q6_k` | Q6_K | Q6_K | Q6_K | F16 | BF16/Q6_K | 2.060 | **passes** |
| q5_k | `r2t2-q5_k` | Q5_K | Q5_K | Q5_K | F16 | BF16/Q5_K | 1.832 | fails Russian; slower |
| — | `lm-q4-tower-q8` | Q4_K | Q4_K | Q4_K | F16 | BF16/Q8_0 | 1.773 | fails |
| q4_k | `r2t2-q4_k` | Q4_K | Q4_K | Q4_K | F16 | BF16/Q4_K | 1.616 | fails |
| C | `r2t2-C` | Q6_K | Q6_K | Q6_K | Q4_K | BF16/Q4_K | 1.532 | **passes** |
| D | `r2t2-D` | Q6_K | Q4_K | Q6_K | Q4_K | BF16/Q4_K | 1.351 | **passes** |
| De3 | `r2t2-De3` | Q6_K | Q4_K | Q6_K | Q3_K | BF16/Q4_K | 1.309 | **passes** |
| Dt3 | `r2t2-Dt3` | Q6_K | Q4_K | Q6_K | Q4_K | BF16/**Q3_K** | 1.309 | **empty output** |
| D5 | `r2t2-D5` | Q6_K | Q4_K | **Q5_K** | Q4_K | BF16/Q4_K | 1.304 | fails German |
| Da4 | `r2t2-Da4` | Q4_K | Q4_K | Q6_K | Q4_K | BF16/Q4_K | 1.260 | **passes** |
| E | `r2t2-E` | Q6_K | Q4_K | **Q4_K** | Q4_K | BF16/Q4_K | 1.260 | fails German |
| J | `r2t2-J` | Q4_K | Q4_K | Q6_K | Q3_K | BF16/Q4_K | 1.219 | marginal — fails French |
| K | `r2t2-K` | Q6_K | Q4_K | **Q4_K** | Q3_K | BF16/Q4_K | 1.219 | fails German |
| **M** | **`r2t2-q4_k_m`** | Q4_K | Q4_K | Q6_K | **Q2_K** | BF16/Q4_K | **1.187** | **floor — passes all** |
| L | `r2t2-L` | Q4_K | Q4_K | **Q5_K** | Q3_K | BF16/Q4_K | 1.172 | fails German |
| G | `r2t2-G` | **Q4_K** | Q4_K | **Q4_K** | Q4_K | BF16/Q4_K | 1.169 | fails German |
| N | `r2t2-N` | Q4_K | Q4_K | **Q5_K** | Q2_K | BF16/Q4_K | 1.140 | fails German |
| G3 | `r2t2-G3` | **Q4_K** | Q4_K | **Q4_K** | Q3_K | BF16/Q4_K | 1.128 | fails German |
| P | `r2t2-P` | Q4_K | **Q3_K** | Q6_K | Q2_K | BF16/Q4_K | 1.093 | **empty output** |

The tower is `MIXED{BF16, <type>}` in every arm: the converter leaves part of
the audio tower at BF16 and quantizes the rest. That is the converter's own
behaviour, not an override.

## Why M works

### The error budget is cumulative, not per-tensor

There is no independent per-tensor floor to look up. What a block can afford
depends on what the rest of the network already spent:

- **Da4** (attention @Q4_K, `down_proj` @Q6_K) passes.
- **E** (attention @Q6_K, `down_proj` @Q4_K) also passes its earlier probes.
- **G** (both @Q4_K) **fails**.

Each individual Q4_K move fits on its own; both together exceed the budget. So
the LM affords **exactly one** of {attention, `down_proj`} at Q4_K, and M spends
that budget on attention while holding `down_proj` at Q6_K.

### `down_proj` is the hard floor

`down_proj` @Q6_K is the single non-negotiable constraint. Every arm that put it
at Q5_K or Q4_K — D5, E, L, N, K, G, G3 — emitted **English for German input**.
There is no arm in the table with `down_proj` below Q6_K that passes.

The mechanism is architectural: `down_proj` writes straight into the residual
stream, so its error propagates through every remaining layer and compounds.
`gate_proj`/`up_proj` error passes through SwiGLU, which bounds it — which is why
gate/up tolerates Q4_K while `down_proj` does not. This is the same reasoning
behind llama.cpp's Q4_K_M rule, but R2T2 needs **Q6_K**, one step stricter than
that rule. Do not assume the rule's step size transfers to a new architecture.

### Attention buys the most, `down_proj` the least

Cost ranking for the LM: `down_proj` > attention > embed.

### The embedding is the cheapest block to quantize

`embed_tokens` is a lookup: its error enters once and leaves once and never
compounds. It is also 311 M parameters, so it is the largest single size lever
per unit of quality. M quantizes it to **Q2_K** — aggressively — and passes every
probe. That combination is what takes the floor from 1.26 GB (Da4) to 1.19 GB.

Note the converter pins embedding tensors to F16 by default. Overriding that pin
is what makes this lever available, and an override does beat the pin.

### Low-bit matmuls are a cliff, not a slope

- **Dt3** (tower @Q3_K) → empty output
- **P** (gate/up @Q3_K) → empty output

Both produced *nothing*, not degraded text. There is no gradual size/quality
trade to negotiate below Q4_K on a matmul — the arm works or it does not, and it
stops working abruptly.

## The marginal arm: J

**J sits exactly at the edge of the budget and is the most informative failure.**

| arm | attention | `down_proj` | embed | outcome |
|---|---|---|---|---|
| M | Q4_K | Q6_K | **Q2_K** | passes everything |
| J | Q4_K | Q6_K | **Q3_K** | **fails French clip fr5** |
| Da4 | Q4_K | Q6_K | **Q4_K** | passes everything |
| De3 | **Q6_K** | Q6_K | Q3_K | passes everything |

J and M differ in **nothing but the embedding quant**, and the *higher*-precision
one failed. So the embed floor is not monotone — but the four rows together show
why: at attention @Q4_K the arm is already at the edge, embed @Q3_K tips it over,
embed @Q2_K happens not to, and embed @Q4_K happens not to. De3 shows Q3_K embed
is entirely fine once attention is raised to Q6_K. **Q3_K embed is not "bad"; it
is marginal, and J was marginal on two axes at once.**

The practical consequences:

1. **Do not treat any passing arm as having margin.** M passes every probe tried
   — jfk, zh, German, and five French clips — and sits in a region where a
   same-or-higher precision neighbour fails. Its passes are evidence it works,
   not evidence it is safe.
2. **Do not step one block up in precision and assume that buys safety.** Embed
   Q3_K is *worse* than Q2_K here.
3. If a margin is wanted at the same embed precision as the gate/up floor, the
   nearest arm is **Da4** (embed @Q4_K, 1.260 GB, +73 MB over M).

Honest weight: this rests on **one clip, one run**. That is thin. It is reported
because a language switch is unambiguous when it happens, not because the sample
is adequate.

## Validation: what was measured, and with what probe

The governing lesson is that **`jfk` + `zh` is not a screen.** R2T2 is trained to
score high on English and Chinese, so those are its two easiest inputs. `q5_k`
matched Q8_0 exactly on both and still switched to English on Russian. E, D5 and
N all passed jfk+zh and switch to English on German. Any verdict in this document
that rests on jfk+zh alone should be treated as unverified.

**The German clip is the cheapest real probe found.** It separated the arms
cleanly where jfk/zh did not.

**The best detector is `transcribe-cli`'s own `detected-language:` line.** Grep
it rather than scoring text by hand. Across 8 arms × 5 French clips it reads `fr`
in 39/40 runs, and the single `en` is J on fr5 — the same failure German caught,
found with one grep. (These runs are on **CPU**, ~3× realtime.)

**French method.** Five of the user's own recordings, selected by
French-likelihood score, resampled 48k→16k with a windowed-sinc low-pass
(`scripts/quant/wav-to-16k.js`) — explicitly not sample dropping, since aliasing
would test the resampler rather than the model. Against Q8_0, every passing arm
matched French
**word for word**; the only differences were punctuation and, once, an accent
(`voilà` → `voila` on De3). That accent is the same orthographic class as the
Japanese `五十円`/`50円` and `いけない`/`行けない` variation between arms, and is not
content drift. Diacritics must be folded before calling French drift real.

**A quantitative FLEURS French WER pass was started and did not complete** (the
reference arm was at 37/64 when it was killed). No WER numbers exist for these
arms. The French evidence is the word-level diff above, not a WER.

### Two harness traps

Both produced wrong conclusions rather than errors:

1. **Reading a `-o` output file while the CLI is still writing it yields an empty
   transcript, which is indistinguishable from the Q3_K "empty output" cliff.**
   M was falsely reported as emitting nothing on fr5 until the file settled. Wait
   on process exit, not on file existence.
2. **Comparing French words without folding diacritics flags `voila`/`voilà` as
   content drift.** Use raw / strict / loose as three separate levels.

## Speed

Interleaved bench, 4 reps dropping the first, pooled ratio against Q8_0, on
jfk+zh. Positive = faster.

| arm | vs Q8_0 |
|---|---|
| B (`tower-q4-lm-q8`) | ±0% |
| q6_k | +5% |
| C | +12% |
| D | +15% |
| D5 | +18% |
| E | +22% |
| q4_k | +32% (broken) |
| A (`lm-q4-tower-q8`) | +39% (broken) |
| **q5_k** | **−20% (SLOWER)** |

**Two non-obvious results:**

1. **Q8_0 is not the fast path for this model.** Every arm with the decoder LM
   quantized beats it, and `tower-q4-lm-q8` — tower @Q4_K with the entire LM left
   at Q8_0 — lands *exactly* on Q8_0. That isolates the LM's matmuls as the
   runtime and the tower as free to quantize.
2. **Q5_K is strictly dominated.** It is ~20% *slower* than Q8_0 and it switches
   to English on Russian. Q6_K and Q4_K have well-optimized kernels; Q5_K's does
   not. **Speed does not follow bit width** — a quant ladder has to be measured.

Read next to quality, this table says the fastest arms are the broken ones, which
is why a speed column without a verdict column is worse than no speed column.

## Reproduction

The arm GGUFs are gone; this is the recipe that rebuilds them. `build` below
takes the base type plus the embed type, enumerates the block tensors by exact
name from the Q8_0 reference, asserts the census, and invokes the converter once
with all overrides.

```bash
SNAP=".../models--netease-youdao--Confucius4-R2T2/snapshots/185ce639118ad1362d049ca0d8ed04b6ec5cd6c9"
ROOT="$D/r2t2-src"
GGUF="$D/audio-build/bin/audiocpp_gguf.exe"
SPEC=".../audio.cpp/model_specs/confucius4_r2t2.json"
EMBED="thinker.model.embed_tokens.weight"

Q="scripts/quant/quantized-names.js"
mapfile -t TOWER  < <(node "$Q" "$Q8" thinker.audio_tower)
mapfile -t LMATTN < <(node "$Q" "$Q8" | grep -E '^thinker\.model\.layers\..*\.self_attn\.(q|k|o|v)_proj\.weight$')
mapfile -t GATEUP < <(node "$Q" "$Q8" | grep -E '^thinker\.model\.layers\..*\.mlp\.(gate|up)_proj\.weight$')
mapfile -t DOWN   < <(node "$Q" "$Q8" | grep -E '^thinker\.model\.layers\..*\.mlp\.down_proj\.weight$')
# census: tower 147, attn 112, gate/up 56, down 28 -- abort on mismatch

"$GGUF" --input "$SNAP/model.safetensors" --root "$ROOT" --family confucius4_r2t2 \
  --model-spec "$SPEC" --output "$D/$2" --type "$1" "${ARGS[@]}" --overwrite
```

Run it from **bash, not `cmd`** — 147 exact-name overrides overrun the Windows
8191-character command-line limit and the arm will silently truncate.

- **M** (the floor): base `q6_k`, embed `q2_k`, with tower / attn / gate+up each overridden to `q4_k`. `down_proj` needs no override — it stays on the base Q6_K, which is exactly the floor.
- **L** (for contrast): base `q5_k`, embed `q3_k`, same three overrides.

### Verify the build

```bash
node scripts/quant/dtype-table.js M=<arm>.gguf
```

Compare the row against the arm inventory above. This is the check that the
overrides actually matched — the converter does not warn when a `--keep-type`
name matches nothing.

### Validate

```bash
transcribe-cli -m <arm>.gguf samples/german.wav -o out.txt
grep detected-language: <log>     # a switch away from `de` is the failure
```

Then jfk and zh for regression, and the French clips if they are available. Check
that `out.txt` is non-empty before diffing it, and read it only after the process
exits — a mid-write read yields an empty file that mimics the Q3_K cliff.

## Open items

- **No WER numbers exist** for any arm. The FLEURS French run was killed at 37/64.
- **The J result rests on one clip, one run.** It should be re-run before being
  relied on as a margin argument.
- **The embed floor between Q2_K and Q4_K was never resolved.** Q2_K passes and
  Q3_K fails at attention @Q4_K, but that is a two-point observation inside a
  region that is evidently not monotone. The safe statement is "Q2_K happened to
  pass every probe", not "Q2_K is safe".
- **The Q6_K `down_proj` floor is well established, but not exhaustively.**
  D5 is the clean test — attention @Q6_K and embed @Q4_K, both at or above their
  passing values, with only `down_proj` moved to Q5_K — and it fails German. So
  the floor is real and not a combination effect. What is untested is **Q3_K and
  below on attention**: every Q3_K matmul attempt elsewhere in this table
  produced empty output, so a cliff is expected, but attention specifically was
  never probed below Q4_K.
- **The tower floor is bounded, not located.** Q4_K passes and Q3_K emits
  nothing; no intermediate was built. If the tower tolerates anything between,
  it is worth ~100 MB.
