# Per-tensor quantization helpers

Small Node scripts used to build and validate per-tensor quantization arms —
variants that differ from each other in exactly one block, so the cost of
quantizing that block can be attributed. Method:
`docs/tools/quantization-arms.md`. Worked results:
`docs/porting/families/confucius4_r2t2-quantization.md`.

They live in the repo rather than in a scratch directory because the docs'
reproduction recipe depends on them and a recipe that points at a deleted
directory is not a recipe.

These complement `scripts/convert-*.py` (per-family conversion) and the bundled
`transcribe-quantize` (preset quantization, `docs/tools/quantization.md`). They
do not convert or quantize anything themselves — they read GGUF headers and
resample audio.

## GGUF header readers

Both parse the header directly (KV table, then tensor-info table) rather than
going through a library, so they work on packages the loader would reject.

### `quantized-names.js <file.gguf> [scope-prefix]`

Prints the names of tensors whose dtype is quantized, optionally filtered to a
name prefix. Exists because `--keep-type` has no mid-string wildcard — a pattern
is an exact name or a trailing-`*` prefix — so protecting half a model means
listing its matmuls by exact name. Dump that list from a package the converter
already got right instead of retyping it.

```bash
# every quantized matmul under the audio tower
node scripts/quant/quantized-names.js r2t2-q8_0.gguf thinker.audio_tower

# one block at a time, by shape
node scripts/quant/quantized-names.js r2t2-q8_0.gguf \
  | grep -E '^thinker\.model\.layers\..*\.mlp\.down_proj\.weight$'
```

### `dtype-table.js <tag>=<path.gguf> [...]`

Prints one row per package with the dtype of each block, asserting that every
tensor within a block agrees and printing `MIXED{...}` when it does not.

**This is the artifact of record for an arm.** An arm is defined by its dtype
table, not by the command line that produced it — two arms can have identical
command lines and different contents when a `--keep-type` name fails to match,
and the build does not complain. Read the table off disk rather than trusting
the build script's intent.

```bash
node scripts/quant/dtype-table.js \
  q8=build/diagnostics/r2t2-reference/published/r2t2-q8_0.gguf \
  M=/path/to/r2t2-q4_k_m.gguf
```

The block patterns in `BLOCKS` are R2T2-specific (`thinker.model.layers.*`).
Edit them for another family.

## Audio

### `wav-to-16k.js <in.wav> <out.wav>`

Converts any PCM or float WAV to the 16 kHz mono 16-bit PCM that
`transcribe-cli` accepts. Written for 48 kHz 32-bit `WAVE_FORMAT_EXTENSIBLE`
recordings.

Resampling is windowed-sinc low-pass then decimate — **not** sample dropping.
Dropping every third sample aliases 8–24 kHz content back into the speech band,
which would make the transcript a test of the resampler rather than of the model.

## Validation

### `fr-diff.js <dir> <arm> [arm...]`

Compares each arm's transcript against the `q8-<clip>` reference in `<dir>` at
three levels, because they mean different things:

| level | normalization | a difference means |
|---|---|---|
| `raw` | none | punctuation and casing — benign |
| `strict` | lowercase, punctuation stripped | orthography, incl. accents (`voilà` → `voila`) |
| `loose` | strict + diacritics folded | **content** drift — a dropped or substituted word |

Only `loose` answers the question. Without the diacritic fold, `voila`/`voilà`
reads as content drift when it is the same word — the same class as the Japanese
`五十円`/`50円` and `いけない`/`行けない` variation between arms. It also flags a
switch to English by counting English vs French function words, which is the
failure mode the German and French probes were built to catch.

Expects transcripts named `<arm>-<clip>.txt` for clips `fr1`–`fr5`. The clip
names and the EN/FR word lists are French-specific; the three-level comparison
is not.

## Traps these exist to avoid

- **Reading a `-o` output file while the CLI is still writing it** yields an
  empty transcript, which is indistinguishable from the "empty output" failure
  of a low-bit matmul. Wait on process exit, not on file existence.
- **Driving the converter from `cmd.exe`.** 147 exact-name overrides overrun the
  8191-character command-line limit and the arm silently truncates. Use bash.
- **A `--keep-type` prefix that catches a 1D bias** throws instead of falling
  back. Enumerate exact names.
