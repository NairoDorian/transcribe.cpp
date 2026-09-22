# Per-tensor quantization arms

`docs/tools/quantization.md` covers `transcribe-quantize`, which applies a
**preset** to a whole file according to a bucket policy. That is the right tool
for shipping a model and it is the tool that should stay the default.

This document covers the other thing: building **arms** — variants that differ
from each other in exactly one block, so that the effect of quantizing that
block can be attributed. A preset cannot express "everything at Q6_K except
`mlp.down_proj`, which stays at Q4_K", so the arms were built with the external
`audiocpp_gguf` converter from the read-only `audio.cpp` tree.

The tooling here is a means, not a policy. Per-tensor control is what you reach
for when a preset's floor is too conservative and you want to know *which* block
is actually paying for it.

## Why the bundled quantizer can't do this

| | `transcribe-quantize` | `audiocpp_gguf` |
|---|---|---|
| unit of control | whole file | whole file **plus** per-tensor overrides |
| selection | bucket policy (`Linear`, `Embed`, `ConvPw`, `Conv`, `Norm`) | exact source tensor name, or trailing-`*` prefix |
| type menu | F16, Q4_0/1, Q5_0/1, Q8_0, Q6_K, Q5_K_M, Q4_K_M | orig, f16, bf16, q8_0, q2_k…q6_k |
| accepts non-allowlisted types | no — loader allowlist enforced | yes; output is not guaranteed loadable |
| output guarantee | loads in `transcribe` | **none** — verify before trusting |

The last row matters. `transcribe-quantize` only emits types the loader
allowlist accepts (`kQuantLinearTypes`), so its output is loadable by
construction. `audiocpp_gguf` will happily write a package the loader rejects.
Every arm must therefore be opened and run, not just built.

## `--keep-type` semantics

The whole method rests on this flag, and its behaviour is not obvious:

- **`--type <t>` is all-or-nothing per file.** It sets the default for every
  tensor the converter is willing to quantize.
- **`--keep-type <name>=<t>` overrides one tensor.** The base `--type` plus N
  overrides *is* the arm — there is no other axis.
- **Names are matched against the SOURCE safetensors tensor name**, not the
  name the tensor ends up with in the GGUF. Getting this backwards silently
  produces an arm with no overrides applied and the wrong size.
- **A pattern is either an exact name or a trailing-`*` prefix.** There is no
  mid-string wildcard. `thinker.model.layers.*.mlp.down_proj.weight` does *not*
  work; you must enumerate the 28 names.
- **First matching rule wins.** Order is significant when rules overlap.
- **An override to a quantized type THROWS rather than falling back** if the
  tensor is not 2D float or `shape.back() % blck_size != 0`. This is the trap
  behind prefix rules: a `*-proj.weight`-style prefix that also catches a 1D
  bias aborts the build instead of quietly skipping it. Enumerate exactly.
- **An override beats the `use_f16_lookup` heuristic.** The converter pins
  embedding-like tensors (`embed`, `codebook`) to F16 on its own. An explicit
  override wins over that pin — which is the only reason the tied embedding is
  available as a size lever at all.

## Enumerating names, and checking the census

Because there is no mid-string wildcard, protecting half a model means listing
its matmuls by exact name. Do not retype them. Dump them from a package the
converter already got right — the Q8_0 reference — by reading the GGUF header
directly.

`scripts/quant/quantized-names.js` does this: it reads the tensor-info table,
prints the names whose dtype is quantized, and takes an optional scope prefix to
filter.

The build script must then **assert a census** before invoking the converter —
a count per block, aborting on mismatch. The failure mode this prevents is a
build that succeeds while quantizing the wrong set of tensors, which looks fine
until you notice the size is wrong. For R2T2 the census is tower 147, attention
112, gate/up 56, down 28.

## Command-line length

147 exact-name overrides do not fit in the Windows `cmd` limit (8191
characters). Drive the converter from bash, not `cmd.exe`, or the arm silently
truncates and comes out wrong. This is a real limit, not a style preference.

## Verify what you built, not what you meant to build

After building an arm, read its dtype table back off disk with
`scripts/quant/dtype-table.js`: one representative tensor per block, asserting
uniformity within the block and flagging `MIXED` where the tower legitimately
mixes. Never infer an arm's composition from the build script's intent — the
flags are easy to get subtly wrong and the build does not complain.

The unit of a documented arm is the **dtype table**, not the command line that
produced it. Two arms can have identical command lines and different contents if
a name failed to match.

## Validation protocol

### `jfk` + `zh` is not a screen

This is the most expensive lesson here. R2T2 is trained to score high on English
and Chinese, so those are its two *easiest* inputs. An arm can match the
reference exactly on both and still be broken.

Concretely: `q5_k` matched Q8_0 exactly on jfk+zh and switched to English on
Russian. E, D5 and N all passed jfk+zh and switched to English on German — three
arms that were reported as passing on the strength of jfk+zh alone.

**A probe only tests what the model is bad at.** Pick a clip in a language the
model was *not* optimized for. For R2T2 that is German.

### Use the model's own language detection

`transcribe-cli` logs `detected-language:`. Grep that instead of scoring the
transcript with a heuristic — it is the model's own answer about which language
it thinks it is transcribing, which is precisely the failure being tested, and
it is one grep. Across 8 arms × 5 French clips it read `fr` in 39/40 runs, and
the single `en` was the real failure.

Note the CLI runs these on **CPU** (`backend: CPU`), roughly 3× realtime, so a
sweep is cheap but not instant.

### Compare against the highest-precision reference, not against each other

Arms should be diffed against F16 or Q8_0, never ranked against one another —
two arms agreeing on wrong output is not evidence of anything.

Textual comparison needs **three levels**, because they mean different things
(`scripts/quant/fr-diff.js` implements them):

| level | normalization | what a difference means |
|---|---|---|
| `raw` | none | punctuation and casing the model chose differently — benign |
| `strict` | lowercase, punctuation stripped | orthography, including accents (`voilà` → `voila`) |
| `loose` | strict + diacritics folded (NFD, drop `\p{Mn}`) | **content** drift — a dropped or substituted word |

Only `loose` answers the question. Without the diacritic fold, `voila`/`voilà`
is flagged as content drift when it is the same word. That is the same
orthographic class as Japanese `五十円`/`50円` and `いけない`/`行けない` — different
renderings of identical content, not a quality difference.

### Empty output is a failure, not a pass

A quantized matmul can fail hard and produce *nothing*. A harness that only
diffs text will not notice; a harness that reads the exit code will see success.
Check that output is non-empty before comparing it.

### Read output files only after the process exits

**A harness trap that produced a wrong published conclusion.** Polling for an
output file and diffing it while the CLI is still writing yields an **empty
transcript, which is indistinguishable from the "empty output" cliff.** An arm
was falsely reported as emitting nothing until the file settled. Wait on process
exit, not on file existence.

## Speed

Interleave arms rather than running all reps of one and then all reps of the
next — thermal and cache state drift over a sweep and will look like a real
difference. Four reps, drop the first, average the remaining three. Revert any
win below the noise floor; a change that cannot be distinguished from noise is
not a win.

Speed is only meaningful next to quality. The two fastest arms in the R2T2
ladder are both broken, so a speed table without a quality column is worse than
no table.

## What generalizes beyond R2T2

1. **Quantization error is cumulative, not per-tensor.** There is no independent
   per-tensor floor to look up. Block A at Q4_K may be fine alone and fatal in
   combination with block B at Q4_K, because the budget is spent network-wide.
2. **Blocks differ in cost per bit, and the ordering is architectural.** What
   matters is whether a block's error writes into the residual stream or passes
   through a bounded nonlinearity. `down_proj` writes into the residual, so its
   error compounds through every remaining layer — the strictest floor.
   `gate_proj`/`up_proj` pass through SwiGLU, which bounds the error — a looser
   floor. Embeddings are a lookup: error enters once and leaves once and does
   not compound at all — the loosest floor. This is why llama.cpp's Q4_K_M rule
   protects `down_proj`, and it is worth checking that rule's step size against
   your own model rather than assuming it transfers.
3. **Low-bit matmuls are a cliff, not a slope.** Q3_K on a matmul produced empty
   output in every attempt. There is no gradual degradation to trade against
   size — the arm works or it does not, and the transition is fast.
4. **Floors can be non-monotone.** A *higher*-precision block can fail where a
   lower-precision one passes. Do not assume that stepping one block up in
   precision buys safety, and do not treat a single passing arm as having
   margin.
5. **A tied embedding is a real size lever, and an unusual one.** When the
   embedding matrix also serves as the output projection head, quantizing it
   changes the logits directly. The converter's automatic F16 pinning of
   embedding tensors must be overridden explicitly to exploit this.
6. **A quant can preserve text on one language and switch language on another.**
   Quality is not a scalar. Text similarity on the training distribution tells
   you almost nothing about behaviour off it.
7. **Better precision is not always slower in the ways you expect.** Optimizer
   coverage per type dominates: Q5_K was *slower* than Q8_0 and Q6_K because its
   kernel is less well optimized. A quant ladder must be measured, not derived
   from bit width.
