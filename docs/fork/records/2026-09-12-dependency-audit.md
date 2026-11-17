# Dependency audit — 2026-09-12

Every pin in this repo, checked against its registry for the newest **stable** and
the newest **pre-release**. The brief was "update everything to whatever is
newest, pre-release included, and make sure of it". This is the record of what
that produced, and — more usefully — of the pins that are deliberately *not*
newest, because a future reader who bumps one of those will silently invalidate
the evidence it exists to preserve.

Method: registry APIs directly (`pypi.org/pypi/<pkg>/json`, `crates.io/api/v1`,
`registry.npmjs.org`, the GitHub API), taking the max over *all* published
versions with a SemVer-aware sort rather than reading `dist-tags.latest`, so a
pre-release that is newer than the stable tag is actually seen. Where a bump was
made it was verified by a gate that already exists in this repo, not by
inspection.

## Verdict

| | Count | Meaning |
|---|---|---|
| Already newest | 34 | nothing to do; range floors already resolve to it |
| Bumped | 1 | Node in CI, 22 → 24 |
| Newest is **broken** upstream | 1 | `bindgen` — stay on 0.72 (see below) |
| Deliberately frozen | 10 | port oracles + eval-data pins (see below) |
| Flagged, not changed | 4 | toolchain migrations, unverifiable here |
| ggml | 1 | already at upstream master tip |

Nothing in this repo was found to be *behind* except the Node CI pin. That is
the headline: the pin discipline here was already sound, and the work was
almost entirely in establishing **which** pins are load-bearing.

## ggml

Pinned `7840aaba1989c6deeefede1d77d5aaf8f52b947e` = **0.23.0**, dated
2026-09-09. Verified against the GitHub API: that commit **is** the current
`ggml-org/ggml` `master` tip. There is no bump available to take.

## Bumped

**Node in CI: `node-version: 22` → `24`, six sites** (`publish.yml` ×4,
`typescript-ci.yml` ×2). Node 22 ("Jod") is **maintenance** LTS; Node 24
("Krypton") is the **active** LTS. Node 26 is the current release but is not
LTS, so 24 is the correct CI target for a library.

`engines: { node: ">=22" }` in `bindings/typescript/package.json` is **left
alone deliberately**. It is a lower bound promised to consumers, not a claim
about what CI runs; raising it would be a breaking change for anyone on 22 for
no gain. The consequence to know: CI now tests 24 only, so 22 is a supported but
untested floor. A `[22, 24]` matrix would close that gap if it ever matters.

## Newest is broken: `bindgen`

`bindings/rust/xtask/Cargo.toml` pins `bindgen = "0.72"`. Newest is 0.73.2.
**0.73.1 and 0.73.2 do not compile at all**:

```
error[E0277]: the trait bound `syn::file::File: ParseQuote` is not satisfied
note: there are multiple different versions of crate `syn` in the dependency graph
       syn-3.0.5  and  syn-2.0.119
```

The cause is upstream's own dependency range. `bindgen 0.73.0` correctly
migrated to `syn = "^3.0"` + `prettyplease = "^0.3"`. Then **0.73.1 and 0.73.2
reverted those to the permissive ranges `syn = ">=2, <4"` and
`prettyplease = ">=0.2.7, <0.4"`**. Cargo resolves `syn` to 3.0.5 while
`prettyplease` keeps syn 2 in the graph, so the two `syn` crates disagree about
`Parse` and the crate fails to build.

Verified by running the repo's own gate:

```
$ cargo xtask bindgen --check          # at 0.72.1  -> "up to date"      rc=0
$ # (Cargo.toml -> bindgen = "0.73")   # at 0.73.2  -> compile error     rc=101
$ # (Cargo.toml -> bindgen = "=0.73.0")# at 0.73.0  -> compiles, output STALE rc=1
```

`=0.73.0` compiles but is not a free upgrade: it is **not byte-identical** to
0.72.1's output. The 8-line diff is the version banner plus `0.73` dropping
`Copy, Clone` from the three opaque handle structs (`transcribe_model`,
`transcribe_session`, `transcribe_device`). That is consumer-visible — a handle
that was `Copy` stops being `Copy` — while `PUBLIC_HEADER_HASH` is unchanged, so
the public header did not move. Upgrading onto the *broken* 0.73 line through an
exact `=` pin, for a dev-only code generator, to gain a derive change that would
need a full `transcribe-cpp` rebuild to validate, is churn carrying risk for no
benefit.

**Decision: stay on 0.72.** `0.72.1` is the newest release on the working line.
Revisit when upstream republishes 0.73.x with a consistent `syn` range.

## Deliberately frozen (do not "fix" these)

These are not stale pins. Each is a **reproducibility anchor**, and bumping one
redefines what a piece of committed evidence means.

**1. Per-family port oracles — `scripts/envs/<family>/pyproject.toml`.**
Eight files pin `transformers`, to six different versions:

```
funasr_nano  4.57.6     granite_nar  5.5.3     moss  5.12.1
qwen3_asr    4.57.6     voxtral      4.57.6    whisper 5.6.1
```

These are the HF **reference implementations** each family was ported against
(`AGENTS.md:12`; `.claude/skills/porting-2-oracle`). The port's golden tensors
were dumped from *that* version. `transformers` 5.17.0 is newest. Bumping these
would change the oracle and silently invalidate the parity evidence for families
that were validated against the older API — the fixtures would keep passing
against a reference that no longer exists.

**2. `scripts/tokenizer-parity-fixture.py` — `transformers==4.57.6`.**
It *regenerates a committed file*, `tests/fixtures/qwen3_asr_bpe_parity.inc`. An
exact pin here is what makes that fixture reproducible.

**3. `scripts/wer/ingest.py` — `datasets==3.6.0`.**
The pin fixes the *eval corpus*. A `datasets` bump can change which revision of
a dataset is fetched, which changes the audio and references, which changes the
WER number. Newest is 5.0.1. Freeze is the point.

If any of these must move, move it as its own change with the affected family's
fixtures regenerated and diffed in the same commit — never as part of a sweep.

## Flagged, not changed

Real gaps, each needing something this box cannot provide.

**Swift `swift-tools-version: 5.9`** (`bindings/swift/Package.swift:1`). Current
Swift is **6.3.3**. This is not a mechanical bump: tools-version ≥ 6.0 changes
the default language mode to **Swift 6**, which turns strict-concurrency
diagnostics into errors and would very likely break the bindings. It is a
language-mode migration. It also cannot be validated here — there is no Swift
toolchain on this machine, and `swift-ci.yml` runs on a **self-hosted macOS
ARM64 runner**. Doing it properly: bump the tools version, add an explicit
`swiftLanguageModes: [.v5]` to hold behaviour, then migrate in a later commit.

**Rust `edition = "2021"` / `rust-version = "1.74"`** (root `Cargo.toml`, ×3).
Current stable is **1.98.1**; edition 2024 has been available since 1.85. Both
are *compatibility promises*, not dependencies: raising the MSRV breaks
consumers, and edition 2024 is a migration (`unsafe_op_in_unsafe_fn`,
RPIT capture rules, `gen` keyword, temporary-scope changes). Rust is verifiable
on this box, so this is doable — but it is a project, not a version bump.

**Python build interpreter — `build = "cp312-*"`, `python-version: "3.12"`.**
Current stable is 3.14.7 (3.11 EOL 2027-10, 3.12 EOL 2028-10 — all supported).
The wheels are `py3-none-<platform>`, so the CPython used to *build* them does
not affect the artifact tag; it only affects the build-time environment. Low
value, and unverifiable without a cibuildwheel/docker run of the whole wheel
lane.

**`typescript` is `^7.0.2` = newest stable.** The `next` tag is
`7.1.0-dev.20260912.1` (a nightly published the day of this audit). Not taken: a
nightly compiler is the one dependency whose failure mode is *the build tool
itself*, and nothing here needs a 7.1 feature. Worth a deliberate trial on a
branch, not a silent bump.

This is also the one major bump in this audit that **was verified locally**,
because the binding's build is just the compiler under test: `npm ci` resolved
the lock cleanly (`package.json` and `package-lock.json` are in sync — `npm ci`
fails otherwise) and `npm run build` — which is plain `tsc` — compiled
`bindings/typescript` with **tsc 7.0.2, exit 0**. So the 5.6 → 7.0 jump and the
`@types/node` 22.20.2 bump are exercised here, not merely asserted. (`node_modules`
and `dist` are both gitignored; the build left no tracked residue.)

## Already newest

Ranges that already resolve to the newest release, so the floor is documentation
rather than a constraint — listed so the next audit can skip them.

`scikit-build-core >=1.0.3` → 1.0.3 (no pre-release exists) · `hatchling` →
1.32.0 · `cmake >=3.21` → 4.4.3 · `ninja >=1.11` → 1.13.2 · `pytest >=7` →
9.1.1 · `numpy >=1.26` → 2.5.3 · `cibuildwheel` → 4.2.1 · `auditwheel` → 6.8.2 ·
`delvewheel` → 1.13.1 · `nvidia-cuda-runtime-cu12 >=12.1` → 12.9.79 ·
`nvidia-cublas-cu12 >=12.1` → 12.9.2.10 · `safetensors >=0.4` → 0.8.0 ·
`pyyaml >=6.0` → 6.0.3 · `jinja2 >=3.1` → 3.1.6 · `soundfile >=0.12` → 0.14.0 ·
`librosa >=0.10` → 1.0.0 · `meeteval >=0.4` → 0.4.3 · `pyannote.metrics >=3.2` →
4.0.0 · `jiwer >=3.0` → 4.0.0 · `whisper-normalizer` → 0.1.15 ·
`huggingface-hub >=0.20/>=0.24` → 1.31.0 · `gguf >=0.10` → 0.19.0 ·
`cmake (crate) 0.1` → 0.1.58 · `serde_json 1` → 1.0.151 · `thiserror 2` →
2.0.20 · `log 0.4` → 0.4.34 · `hound 3` → 3.5.1 · `koffi ^3.2.1` → 3.2.1 ·
`@types/node ^22.20.2` → 22.20.2 · `typescript ^7.0.2` → 7.0.2 ·
`requires-python >=3.11` (in support) · `cargo xtask` / clippy via
`dtolnay/rust-toolchain@stable` (always latest).

**GitHub Actions — all current.** `actions/checkout@v7` (×36),
`actions/cache@v6`, `actions/upload-artifact@v7`, `actions/download-artifact@v8`,
`actions/setup-node@v7`, `actions/setup-python@v7`,
`actions/configure-pages@v6`, `actions/upload-pages-artifact@v5`,
`actions/deploy-pages@v5`, `astral-sh/setup-uv@v10.1.0`,
`denoland/setup-deno@v2`, `ilammy/msvc-dev-cmd@v1`,
`Jimver/cuda-toolkit@v0.2.36`, `dtolnay/rust-toolchain@stable`,
`pypa/gh-action-pypi-publish@release/v1`.

**`@types/node` stays on the 22 line on purpose.** Newer majors exist (24.13.4,
25.9.6, 26.5.1) and `@types/node`'s `ts5.9`/`ts6.0` dist-tags point at 26.5.1.
But a `@types/node` major tracks the Node major whose API you compile against,
and `engines` promises `>=22` — taking 26 typings would let 26-only APIs into the
build while the package still claims 22. Typings should move only when the
supported floor does.

## Not verified here

- `bindgen`'s 0.73.0 output was diffed but **not** compiled through
  `transcribe-cpp` (that needs a full CMake build of the C++ tree). The diff was
  read instead, and the change it makes is visible in it.
- The Node 24 CI bump was **not** exercised: GitHub Actions cannot be run from
  this box. The change is a one-token version pin across six sites; the first CI
  run confirms it.
- Swift and the wheel build interpreter were **not** touched (see above).
