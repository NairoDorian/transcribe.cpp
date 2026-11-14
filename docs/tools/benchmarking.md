# Benchmarking

`scripts/bench/run.py` drives `transcribe-bench` across a matrix of
(backend, variant, quant, sample) cells, auto-detects the host, and
aggregates results into per-(variant, backend) JSON reports.

A **variant** is the per-checkpoint slug that names the directory under
`models/`, e.g. `Qwen3-ASR-0.6B`, `Qwen3-ASR-1.7B`, `parakeet-tdt-0.6b-v3`.
Different sizes of the same model family (0.6B vs 1.7B) are separate
variants and are never collapsed — each gets its own output file.

## What it measures

`transcribe-bench` times per-stage latency (mel frontend, encoder,
decoder) and overall RTFx on a fixed audio sample, with configurable
warmup + measurement iterations. It loads the model once per cell and
runs the full stack — not just matmul microbenchmarks.

## CLI

```bash
uv run scripts/bench/run.py                                                # full matrix, all backends
uv run scripts/bench/run.py --models Qwen3-ASR-0.6B                        # one variant
uv run scripts/bench/run.py --models Qwen/Qwen3-ASR-0.6B                   # HF slug form (org prefix stripped)
uv run scripts/bench/run.py --models Qwen3-ASR-0.6B,Qwen3-ASR-1.7B         # multiple variants
uv run scripts/bench/run.py --models parakeet-tdt-0.6b-v3 --quants F16
uv run scripts/bench/run.py --samples jfk --iters 20 --warmup 5
uv run scripts/bench/run.py --backends metal,cpu,vulkan
uv run scripts/bench/run.py --backends cpu --name pre-refactor
uv run scripts/bench/run.py --models models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf --samples jfk
```

### `--models` tokens

Each comma-separated token in `--models` resolves to one of three forms:

| Form              | Example                            | Behavior                                                              |
|-------------------|------------------------------------|-----------------------------------------------------------------------|
| Dir slug          | `Qwen3-ASR-0.6B`                   | All GGUFs under `models/<slug>/`, filtered by `--quants`              |
| HF slug           | `Qwen/Qwen3-ASR-0.6B`              | Everything before the last `/` is stripped; otherwise same as above   |
| `.gguf` path      | `models/.../Qwen3-ASR-0.6B-BF16.gguf` | That exact file only; `--quants` is ignored for this token          |

When `--models` is omitted, every variant dir under `models/*/` with at
least one matching GGUF is benched.

### Flags

| Flag           | Purpose                                                              |
|----------------|----------------------------------------------------------------------|
| `--models`     | Variant slugs (short or HF-style) or `.gguf` paths (default: all)    |
| `--quants`     | Quant presets to include (default `f16,q8_0,q4_k_m`)                 |
| `--samples`    | Sample stems under `samples/` (default `jfk,dots`)                   |
| `--iters`      | Measured iterations per cell (default `2`)                           |
| `--warmup`     | Warmup iterations per cell (default `1`)                             |
| `--backends`   | `metal,cpu,vulkan` or `all` (default: auto-detect)                   |
| `--name`       | Stable label for named baselines (otherwise timestamp)               |
| `--bench-bin`  | Legacy override: single-backend, explicit binary path                |
| `--out-dir`    | Override reports output root                                         |
| `--dry-run`    | Print selected backends + matrix without running                     |

## Backend resolution

Each run passes `--backend <name>` to the bench binary so the library
selector is exercised end-to-end (not just the binary's default).

| Backend | Binary                                   | Flag                |
|---------|------------------------------------------|---------------------|
| metal   | `build/bin/transcribe-bench`             | `--backend metal`   |
| cpu     | `build/bin/transcribe-bench`             | `--backend cpu`     |
| vulkan  | `build-vulkan/bin/transcribe-bench`      | `--backend vulkan`  |

Auto-detection (when `--backends` is unset or `all`):

- `metal` — `build/bin/transcribe-bench` exists **and** `sys.platform == 'darwin'`
- `cpu` — `build/bin/transcribe-bench` exists
- `vulkan` — `build-vulkan/bin/transcribe-bench` exists

## Output

One aggregated JSON per (variant, backend):

```text
reports/perf/<machine-slug>/<timestamp-or-name>_<variant>_<backend>.json
```

`<machine-slug>` is derived from hostname + CPU model so reports from
different machines don't overwrite each other. `<variant>` is the
directory slug, e.g. `Qwen3-ASR-0.6B_metal.json`.

## Comparing runs

`scripts/bench/compare.py` diffs two report files and prints a
per-cell delta table. Use `--name pre-refactor` / `--name post-refactor`
for stable named baselines that overwrite on re-run, then compare the
named pair.

## Publication profiles and size caps

`uv run scripts/bench/run.py --profile` runs the exact matrix the catalog
requires, as declared in `catalog/_benchmark_profiles.json`. The profile owns
quants, samples, iteration counts, and the machine targets.

A target may declare `max_model_gib`. That makes the target a *weak reference
machine*: it is only required to run variants whose reference publication
footprint fits, and a larger model is published on the strong reference alone.
The footprint is the published Q8_0 size in GiB (a variant with no Q8_0 falls
back to its largest published quant), so the comparison is in binary units and
includes the runtime margin implicitly through the chosen cap.

The cap is requirement policy, not a measurement filter: an out-of-scope cell
is neither demanded by `scripts/catalog/check.py`, nor run by
`scripts/bench/run.py --profile`, nor ingested by
`scripts/catalog/ingest_perf.py`. Because an unexpected row fails the gate as
`speed_extra`, lowering a cap later also means deleting any rows already
collected for the newly excluded models, not just re-running.

Current policy: `m4` (a base Apple M4) and `ryzen-4750u` are capped at 5 GiB.
`m4-max` has no cap and publishes every model.

## Quant coverage

The default `--quants f16,q8_0,q4_k_m` covers the canonical tiers.
Add legacy quants or K-quants as they stabilize in
[`quantization.md`](quantization.md). The variant must have a GGUF
present in `models/<variant>/` for each listed quant — missing files
are skipped with a warning, not an error. Some models ship `BF16`
instead of `F16` (e.g. Qwen3-ASR); pass `--quants bf16,q8_0,q4_k_m`
for those.
