# transcribe.cpp Agent Conventions

Read `CONTRIBUTING.md` for contribution policy, review gates, and coding
style. This file is only the local command/automation convention sheet for
coding agents.

## Python

- ALWAYS use `uv run` for every Python invocation. Never bare `python`,
  `python3`, or `pip`.
- Use `uv pip` for packages, `uv sync` for envs, and `uv run` for scripts.
- Per-family reference environments live under `scripts/envs/<family>/` and are
  invoked as:

```bash
uv run --project scripts/envs/<family> scripts/<script>.py ...
```

- **Backlog (deferred, not started):** every env under `scripts/envs/` should
  eventually get its dependencies bumped to latest — and where the env uses
  PyTorch, to the nightly CUDA index matching the local toolkit (the parakeet
  port env is the reference for this: `https://download.pytorch.org/whl/nightly/cu134`
  for a CUDA 13.4 host). Do this as a dedicated refresh pass later; never mix
  fleet-wide dep bumps into a behavior-change commit.

## Build

- After C++ changes, run:

```bash
cmake --build build --target transcribe-cli
```

## Formatting

- Format our C/C++ before committing. The formatter is pinned and fetched via
  `uvx`, so do not rely on a system clang-format:

```bash
scripts/ci/clang-format.sh            # format our tree in place (default)
scripts/ci/clang-format.sh --check    # verify, no changes
```

- Scope is our code only. Vendored trees (`ggml/`, `src/third_party/`) and
  verbatim upstream copies (`src/transcribe-unicode-data.cpp`) are never
  formatted. CI gates our C/C++ in
  `.github/workflows/clang-format.yml`.

## C ABI Exception Discipline

No C++ exception may escape a public entry point.

- A new public entry point must either route through an `api_guard_*`
  wrapper (`src/transcribe.cpp`) or be nothrow by construction. Device and
  registry queries are not pure reads; guard them.
- Entry points with ownership out-params enforce "non-OK => `*out == NULL`,
  nothing leaked" in their forwarders on every error return.
- Teardown never uses raw `ggml_backend_free` / `ggml_backend_buffer_free` /
  `ggml_backend_sched_free` in library code: use `transcribe::safe_*` from
  `src/transcribe-backend.h`. `tests/lint_teardown.cmake` fails CI on
  violations.
- Host log callbacks are contained at the emission site
  (`transcribe_log_invoke`).
- The `TRANSCRIBE_TEST_*` fault hooks (`_DEV_INIT_THROW`, `_TEARDOWN_THROW`)
  intentionally ship in release artifacts for wheel clean-install CI.
  Present-but-empty values are inert.

## Verification

- End-to-end numerical checks:

```bash
uv run scripts/validate.py all --family <f> [--variant <v>]
```

`--variant` is required when the family has multiple manifests.

- Manual tensor debugging:

```bash
uv run scripts/compare_tensors.py ...
```

- Cheap metadata/config gates before expensive numerical work:

```bash
uv run scripts/preflight.py --family <f> [--variant <v>]
```

- Never suppress test failures without root cause analysis.

## Porting a New Model

Use the `porting-*` skills in `.claude/skills/`. Stage skills are independent
and run in order:

```text
porting-1-intake -> porting-2-oracle -> porting-3-convert -> porting-4-cpp
-> porting-5-quants -> porting-6-bench -> porting-7-wer -> porting-8-ship
```

## GGML Dependency Sync

Keep `ggml` updated regularly against upstream HEAD (`ggml-org/ggml`).

```bash
# Preview bump
scripts/sync-ggml.sh master --dry-run

# Perform sync
scripts/sync-ggml.sh master

# Post-sync verification
cmake --build build --target transcribe-cli
cmake -DSRC_DIR=src -P tests/lint_teardown.cmake
```

- Pinned upstream commit is recorded in `ggml/UPSTREAM`.
- Downstream patches live in `patches/ggml/`. Never edit `ggml/` manually.

## Git Hygiene

- Do not commit, push, create pull requests, or comment on pull requests unless
  the user explicitly asks for that action.
- Do not reformat unrelated code in the same change as a behavior fix.
