# What transcribe.cpp can learn from audio.cpp and speech.cpp

A survey of two sibling projects, filtered for what is actually transplantable
into an ASR-only library, with a prioritised adoption plan.

Sources: `STT_BRAIN_TTS/audio.cpp` @ `78d4770` (upstream `5bea9c7`),
`Unified_Audio.cpp/speech.cpp` @ `fbd84fed` (the fusion project), and this repo
@ `f8e97c9`. Companion to `PASSOVER.md`, which is this fork's audit register for
the 2026-08 performance series.

**Scope rule.** This repo is an ASR/diarization library. Nothing here proposes
importing TTS, voice cloning, source separation, codecs, vocoders, music or
MIDI. Where a sibling idea is only expressed through those subsystems, the
*pattern* is called out and the subsystem is left behind.

---

## 1. Verdict in one page

1. **On dependencies you are already ahead — and the margin is wider than it
   first looks.** `audio.cpp` vendors ggml **0.12.0**
   (`external/ggml/CMakeLists.txt:6-9`) with no `UPSTREAM` record *and no record
   of its own patches*. That last part is not an absence: its vendored tree is a
   **silently hand-patched fork**. It adds three CUDA backend extension procs
   that upstream ggml does not have — `ggml_backend_cuda_trim_pools`,
   `ggml_backend_cuda_set_stream_priority`, `ggml_backend_cuda_clear_graph`
   (`external/ggml/include/ggml-cuda.h:26,30,34`; bodies at
   `src/ggml-cuda/ggml-cuda.cu:5056,5071,5079`; registered in the backend proc
   table at `:5905,5908,5911`). Nothing in that tree says so. Re-vendoring ggml
   there would delete all three without a warning, and every `trim_pools` call
   site would keep compiling — `get_proc_address` returns `nullptr` — and quietly
   become a no-op. That is the exact failure mode this repo's
   `UPSTREAM` + `patches/ggml/` recipe exists to make impossible. Your pin is
   `7840aaba` = **v0.23.0** (`ggml/UPSTREAM`), which is *current upstream
   `ggml-org/ggml` master tip* — verified against `git ls-remote`. There is no
   ggml bump available to take. The dependency lesson is therefore about
   **discipline around the pin**, not catching up.

2. **The single most valuable transferable artefact is `graph_optimizer.cpp`.**
   `audio.cpp`'s `src/framework/runtime/graph_optimizer.cpp` (~589 lines) is a
   model-agnostic pass over a bare `ggml_cgraph&`: it folds broadcast `repeat`
   into consumers, elides identity materialisations and no-op/metadata-only
   nodes, takes a per-backend policy, has an env kill-switch, and returns a
   report with counts. Your measured CPU profile is `MUL_MAT 66%, ADD 9%,
   MUL 4.9%, CONT 4.6%` — `CONT`/`ADD`/`MUL` are exactly what such a pass
   removes. This is the one item with a plausible *measured* win.

3. **The second most valuable is the long-form audio layer.**
   `audio/chunking.{h,cpp}` (~705 lines) covers chunk planning (fixed / quiet
   energy / VAD-driven), overlap-add, and — critically — three
   `append_chunk_word_timestamps` overloads that rebase chunk-local word times
   into the parent timeline. You have `docs/input-limits.md` and a three-bucket
   limits contract, but no planner/rebaser of this shape.

4. **The fusion project's most expensive lesson is a warning, not a gift.**
   `speech.cpp` attempted a two-parent unification and, per its own tracker,
   executed **zero deletions** while producing an inert `SharedWeightRegistry`,
   an unreachable batched-decode path, and — in `capi/audiocpp.h` — a *third*
   façade. Its review concluded: *"Both parents stay live"*, *"One ABI, now"*,
   and *"Build the ASR runtime layer first, then port"* — with the risk
   *"porting before the runtime layer"* rated High and annotated **"(it
   happened)"**. Read its doctrine; do not attempt its programme.

5. **One structural idea from the fusion corpus is unambiguously worth it:**
   the *ASR runtime layer*. This repo has **five field-identical KV caches**
   (`canary.h:40`, `cohere.h:39`, `moonshine.h:49`,
   `moonshine_streaming.h:45`, `whisper.h:39`) and every family owns its own
   runner loop. Consolidating that is the difference the fusion project spent
   a year discovering it should have built first.

6. **Licensing is a gate, not a footnote.** `audio.cpp` is **Apache-2.0**
   (`LICENSE:15`; "Copyright 2026 ShugoAI LLC"). This repo is **MIT**. The
   fusion project's own plan flags the discrepancy and leaves it unresolved —
   its README wrongly calls all three projects MIT. Studying audio.cpp is
   unconstrained. **Copying its code is not**: Apache-2.0 imposes notice,
   `NOTICE`-file and state-changes obligations that a bare MIT header does not
   satisfy. See §7.2.

---

## 2. The three reframes

The brief assumed audio.cpp is the modern one to catch up to. That is only
true on some axes, and backwards on the most important one.

| Axis | audio.cpp | transcribe.cpp | Who leads |
|---|---|---|---|
| ggml version | 0.12.0, no upstream SHA | **0.23.0 = upstream master tip** | **you** |
| ggml patch discipline | **3 unrecorded CUDA patches**, no `UPSTREAM` | `UPSTREAM` + `patches/ggml/` + `sync-ggml.sh` + a reproducibility gate | **you** |
| Public ABI | **none** — C++ library + HTTP server only; `extern "C"` in 4 internal shims | 75+ entry points, `struct_size` discipline, abihash, 6 bindings | **you** |
| Streaming contract | `start/process/finalize/reset`, no cancellation protocol | 4-state dispatcher machine, revisions, committed text, abort | **you** |
| Input limits | **no contract** | 3 buckets, `INPUT_TOO_LONG`/`OUTPUT_TRUNCATED`, `was_truncated()` | **you** |
| Numeric validation | per-PR evidence | golden manifests + tolerances + `validate.py` + `preflight.py` | **you** |
| Build modernity | CMake **3.20**, C11/C++17, no formatter gate, no presets file | CMake floor **3.16** declared (but 3.21 elsewhere) | audio.cpp |
| Test execution in CI | **runs `ctest` on Linux/macOS/Windows** | tests exist; wheels+CI are broad | audio.cpp (parity) |
| Declarative model metadata | `model_specs/*.json` (74), schema-gated, drives CLI/server/packaging | `docs/models/` cards + `docs/porting/families/*.md` (prose) | audio.cpp |
| Graph-level optimisation | `graph_optimizer.cpp` | none | **audio.cpp** |
| Long-audio chunk planning | `audio/chunking.*` | none of this shape | **audio.cpp** |
| Multi-backend execution | single-backend graphs | `BackendPlan` + `BackendKind` | you |

So the honest framing: **audio.cpp is a design donor, not a version to catch up
to.** Its value is three or four self-contained subsystems, not a posture.

---

## 3. Doctrine worth adopting (from the fusion corpus)

These are cheap, textual, and prevent expensive classes of mistake. They come
from `speech.cpp`'s planning documents, which are the most valuable thing that
project produced.

### 3.1 The Reciprocity Rule

> *"Neither codebase's **code** is canonical. Each codebase's **contracts**
> are. Before speech.cpp absorbs a family from one side, the receiving side must
> first satisfy the contract the donating side enforced on it. Absorbing code
> onto a weaker contract is not fusion — it is a silent downgrade wearing a
> merge commit."*

Applied here: when taking anything from audio.cpp, port the *behaviour* into
this repo's contracts (`api_guard_*`, `safe_*` teardown, `struct_size`,
`lint_teardown.cmake`), not the code's structure. Never import an unguarded path.

### 3.2 Laws that bind this repo directly

- **L2/L3 — additive before destructive; never rename an upstream-owned file.**
  Your ownership map is: `ggml/` is *generated* (the patch stack is the
  invariant); everything under `upstream/*` is *additive only*; fork-owned paths
  (`include/transcribe-plugin.h`, `src/transcribe-plugin-entry.cpp`,
  `src/transcribe-stubs.cpp`, `cmake/transcribe-build-info.h.in`) are free to
  restructure.
- **L9 — exception containment is a build-time property.** You already do this
  (`api_guard_*`, `lint_teardown.cmake`). Keep it lint-enforced, never
  review-enforced.
- **L11 — never truncate silently.** Three separate silent-truncation defects
  were found across the two parents. Your `docs/input-limits.md` contract
  already encodes the fix; the risk is regressing it while touching streaming.
- **L12 — measure the real flow, then read the code.** This is the same lesson
  as your own `PASSOVER.md` §8, learned independently: the multi-STT serializer
  looked like a 55% win on a synthetic stand-in and was a 38–45% *regression* on
  the real dictation flow.
- **L6 — specification beats inspection.** Their worked example is a unified mel
  frontend that would have silently broken 11 of 18 families over an
  `n_fft/2` vs `n_fft/2 + 1` drift. There are 19 families here, each with its
  own frontend. Treat any "unify the mel" proposal as guilty until specified.
- **D24 — GGUF `general.architecture` strings are immutable ABI.** They reached
  this after a family-id aliasing defect (`sense_asr`/`sensevoice`,
  `parakeet_tdt`/`parakeet`). If you ever add aliases, resolve them at the
  registry *display* level, never by renaming the architecture string.

### 3.3 The upstream-sync discipline you are closest to already having

The fusion project's ggml work produced two ideas that fit your tree exactly:

- **The reproducibility criterion: `sync + patches == committed tree`.**
  After a re-sync, `scripts/sync-ggml.sh` must report zero changed paths and
  `git diff --ignore-cr-at-eol -- ggml` must be empty. You have the script and
  the patch stack; what you lack is the *verification step* that proves the
  invariant on every run.
- **A short SHA is not a fetchable ref.** Their `AGENTS.md` states it because it
  cost them a session. Yours records a SHA in `ggml/UPSTREAM`; also record the
  version string and date so a human can tell 0.23.0 from a future 0.24.0
  without reading `ggml/CMakeLists.txt`.
- **An `upstream-merge-dryrun` CI job.** Their 6-job CI matrix includes one;
  it is cheap and it catches a stale merge-base before it becomes a phantom
  "N commits behind" count.

---

## 4. Design assets from audio.cpp, ranked by transplantability

Line counts are from the reconnaissance pass; paths are authoritative.

### 4.1 Immediate — self-contained, no registry coupling

| # | Asset | Path | Why it transfers |
|---|---|---|---|
| A1 | **Graph optimiser** | `src/framework/runtime/graph_optimizer.cpp` (~589), `include/engine/framework/runtime/graph_optimizer.h` | Pure `ggml_cgraph&` in, report out. Per-backend policy + env kill-switch. Backend- and model-agnostic. **Targets your `CONT`/`ADD`/`MUL` share directly.** |
| A2 | **Long-audio chunk planning** | `include/engine/framework/audio/chunking.h` (~153), `src/framework/audio/chunking.cpp` (~705) | `plan_audio_chunks`, `plan_vad_audio_chunks` (segment list *or* live VAD session), `plan_quiet_energy_audio_chunks`, triangular/linear fade windows, `overlap_add_planar_chunk`, `append_chunk_word_timestamps` ×3 (rebase into parent timeline). |
| A3 | **Energy activity (non-ML VAD)** | `src/framework/audio/activity.{h,cpp}` (~76) | dBFS + window + margin speech-region finder. 76 lines; useful as the cheap tier under a streaming VAD. |
| A4 | **Kaldi log-mel + LFR** | `src/framework/audio/kaldi_fbank.{h,cpp}` (~392) | Snip-edges, Hamming/Povey, `lfr_m=7`/`lfr_n=6`, CMVN, cached filterbank. Relevant to the Zipformer/Parakeet/Nemotron/SenseVoice lineages — use as a **parity oracle**, not a replacement. |
| A5 | **Torchaudio-parity resampling** | `src/framework/audio/resampling.{h,cpp}` (~511) | Explicit float32/float64 *kernel* and *accumulation* modes → bit-exact against a PyTorch reference. The mode-selection idea is the transferable part. |

### 4.2 Needs adaptation

| # | Asset | Note |
|---|---|---|
| A6 | **Streaming diarization frame accounting** | `community_models/sortformer_diar/{stream_schedule,aosc_state}.*` — `SortformerV2StreamWindow` + scheduler with `emitted/consumed/produced` frame counters, reflect-left/zero-right padding, `final_flush`. A drift-free mel/encoder bookkeeping scheme; the *idea* generalises to any bounded-context streaming ASR. |
| A7 | **`set_backend_stream_priority` / `trim_backend_pools`** | `src/framework/core/backend.cpp` (~768). CUDA stream priority for realtime work; dropping cached idle CUDA/HIP pool memory on alloc-failure retry. **Directly relevant to the 8 GB-VRAM / 4-model Handy case.** |
| A8 | **Graph capacity controller** | `GraphCapacityController` with `Fixed / Tiered / Grow / Double / Unsupported` arena-growth policies — trade memory for avoiding re-alloc between requests. |
| A9 | **`CapacityError` taxonomy** | `runtime/errors.h`: "cannot be served *at this size*" as a **client** error, distinct from an internal fault. Complements your `INPUT_TOO_LONG` story for long audio. |
| A10 | **Warmbench case format** | `tests/core/audio_task_warm_bench.h` (~310) + `tests/warmbench.py`. Case file = `{warmup, requests[{audio,start_sec,end_sec,language,expected_fragments}]}` — timing *and* correctness in one artefact. Their `cd2fe10` fixed warming in the wrong mode (`warmup_case` streaming flag): warm in the mode you measure. |
| A11 | **Declarative model specs** | `model_specs/*.json` (74) + `src/framework/model_spec/*` (~1849). The transferable idea is `dependencies[]` with `required_when[]` — e.g. "attach the forced aligner when `return_timestamps=true`", "attach bundled VAD when `audio_chunk_mode=vad`". **This repo has no `model_specs/` directory at all**; adopting the *concept* is a larger project than it looks. |
| A12 | **Catalog/registry sync gate** | `tools/check_loader_catalog_sync.py` — fails if an installable package advertises a family the CLI does not expose, or if a live HF snapshot source outlives a commented-out loader. |

### 4.3 Take the idea, not the code

- **`host_ops` registry** (`framework/runtime/host_ops.{h,cpp}`, ~401):
  `CTCDecoder`, `BeamSearch`, `MonotonicAlignment` as *named* host ops. In
  audio.cpp only one model consumes it — it is a reference layer, not a path.
- **Session/task/streaming interfaces** (`runtime/session.h`, ~375):
  `VoiceTaskKind`, `RunMode`, `StreamEvent`, `TaskResult{speech_segments,
  speaker_turns, word_timestamps}`. Your C ABI already expresses this better;
  read it for vocabulary, do not adopt the C++ vtable.
- **Pull-based ingress** (`app/streaming/*`, ~352):
  `AudioChunkReader = function<bool(int64_t max_samples, vector<float>&)>`
  returning false at EOF, expected to block. A clean backpressure story, but it
  is an *app-facing* design; your contract is the session state machine.

---

## 5. Where this repo is already the teacher

For completeness, since the question was "can transcribe learn from audio and
speech" — the reciprocal answer is also yes, and the fusion project says so
explicitly. It adopted from transcribe.cpp:

- the C ABI with `struct_size` + `*_init()` + `copy_out_prefix()`;
- `api_guard_status/value/void` at every boundary, as a *build-time* property;
- `safe_*` teardown + the `lint_teardown.cmake` gate — audio.cpp had ~170 raw
  `ggml_backend_*_free` sites across 78 files outside its ported subtree;
- the streaming state machine, which their risk register says to port **verbatim**;
- the three-bucket input-limits contract;
- the golden-manifest validation methodology, which they identified as a gap
  across **all** their families.

Their `Appendix E` (24-row architecture comparison matrix) is the densest
artefact in the corpus and is worth reading once as an outside view of this
repo's design.

---

## 6. Prioritised plan

Ordered by (value ÷ risk), with the verification each tier needs. Your
`PASSOVER.md` rule applies throughout: **measure before and after; revert
anything inside the noise floor.**

### Tier 0 — Documentation and metadata truth (no runtime effect)

Status as of this writing: **landed**, **open** (safe, just not done), or
**flagged** (a product/identity decision that is yours, not a mechanical fix).

Every row below was **re-verified against the tree on 2026-09-12** rather than
trusted from this table — all six landed, at the evidence sites named.

| # | Item | Evidence | Status |
|---|---|---|---|
| T0.1 | Record ggml **version + date**, not just SHA, in `ggml/UPSTREAM`; have `sync-ggml.sh` write them | `ggml/UPSTREAM` recorded `7840aaba` only | **landed** — `sync-ggml.sh` now reads `GGML_VERSION_*` out of the *staged* tree and the commit date out of the clone, and regenerates `ggml/UPSTREAM` with `version: 0.23.0` / `date: 2026-09-09` |
| T0.2 | Fix "**18 architectures**" where the list has **19** (`medasr` added last) | `CMakeLists.txt:285`, `Cargo.toml:79` | **landed** — both now point at `_all_families` as the authority instead of restating a count that drifts with every new family |
| T0.3 | Document `TRANSCRIBE_MODEL_SET=none` | legal at `src/CMakeLists.txt:77,91`, absent from the option block | **landed** — documented in the option comment and added to `set_property(... PROPERTY STRINGS ...)` so the GUI dropdown offers it |
| T0.4 | Fix package metadata that still points at `handy-computer/transcribe.cpp` | `pyproject.toml:43-45`, `Cargo.toml:20-21`, `bindings/typescript/package.json`, `bindings/swift/Package.swift:12` | **landed** — see the note below. `Package.swift` turned out to have no URL to change (line 12 is a comment; the mirror-repo `binaryTarget` is a manual post-release step and `publish.yml:150` only *echoes* the checksum) |
| T0.5 | Replace `Development Status :: 1 - Planning` on a shipped 0.2.3 | `pyproject.toml:35` + 2 more | **landed** — all three are now `4 - Beta`, which is the honest claim for a pre-1.0 library (not `5 - Production/Stable`: 0.2.x reserves the right to break the API) |

On T0.4, the flag resolved on technical grounds rather than preference. The
release machinery is **already fork-agnostic** — every upload in `publish.yml`
passes `--repo "$GITHUB_REPOSITORY"` (`:394,395,595,610,618,641,668,674`), so a
wheel or crate released from this tree is released *from this tree's repo*.
Static metadata naming `handy-computer` therefore described a repository that
does not contain the artefact being published, and would route its bug reports
to a project that cannot act on them. Changed in `pyproject.toml`,
`bindings/python/pyproject.toml`, `bindings/python-native-cu12/pyproject.toml`,
`Cargo.toml`, `bindings/rust/transcribe-cpp/Cargo.toml`, and the two `src/lib.rs`
doc links.

**Not** changed, deliberately: the `handy-computer/*-gguf` references in
`scripts/hf_cards/*.yaml`, `scripts/audit_gguf_metadata.py` and the CI canary
fetchers. Those name real Hugging Face repositories owned by upstream — they are
model provenance, not this project's identity, and rewriting them would point at
HF repos that do not exist. 81 files mention `handy-computer`; only the ones that
are *this project's own published identity* were in scope.
| T0.6 | Remove the 0-byte `ggml/.gitmodules` residue from the vendor step | `ggml/.gitmodules` | **landed** — added to `EXCLUDES` in `sync-ggml.sh` and deleted from the tree, so nothing downstream reads it as "ggml is a submodule" |

### Tier 1 — Build, CI and dependency hygiene (verifiable by build + CI)

Re-verified on 2026-09-12: T1.1, T1.2, T1.3, T1.5, T1.6, T1.7 all landed (T1.4
deliberately rejected). The action pins have moved on since this table was
written — `actions/checkout` is now `@v7` at all 36 sites, and `setup-node` /
`setup-python` are uniformly `@v7`, so the T1.2 inconsistency is gone and gone
uniformly.

A full dependency sweep followed this tier; its results, including which pins
are deliberately *not* newest and why, are in **`docs/dependency-audit.md`**.
The short version: only the Node CI pin was behind, `bindgen` 0.73.x is broken
upstream, and the per-family `transformers` pins are port oracles that must not
be bumped.

| # | Item | Evidence | Risk | Status |
|---|---|---|---|---|
| T1.1 | Raise the CMake floor `3.16` → **`3.21`**, matching `CMakePresets.json:3` and `pyproject.toml:62`; 3.16 currently contradicts both. Plain minimum, *not* a `3.21...4.0` range — neither sibling declaration uses one, so a range would introduce a third convention to fix a two-way disagreement | `CMakeLists.txt:1` | low | **landed** — configure-verified on CMake 4.4.2 (clean, no policy warnings) plus a full `transcribe-cli` build |
| T1.2 | Bump `actions/checkout@v4` → `@v6` at the **7 stragglers**; `setup-node@v4` → `v5` | `publish.yml` ×4+4, `swift-ci.yml` ×3, `typescript-ci.yml` ×2 | low | **landed** for checkout — 7 stragglers bumped, tree now **34/34 at v6**. `setup-node` left at `v4`: it is *uniform* across the tree, so unlike checkout there is no inconsistency to fix |
| T1.3 | Raise `requires-python` to `>=3.11` (3.9 EOL'd 2025-10; repo scripts already require 3.11 via PEP-723) and extend classifiers | `pyproject.toml:27`, `bindings/python/pyproject.toml:12,39-43` | low — but a packaging decision | **landed** — all three manifests at `>=3.11`; classifiers now `3.11`–`3.14` (3.14.7 is current stable as of this writing; 3.15 is still at rc and is therefore *not* claimed). `bindings/python-native-cu12/pyproject.toml` was not in this row's evidence but carried the identical stale metadata, so it was done with the rest — leaving it would have produced exactly the drift this row is about |
| T1.4 | Replace the global `CMAKE_CXX_FLAGS` string surgery for `/EHs` with a target/`add_compile_options` form | `CMakeLists.txt:544-552` | medium | **rejected, deliberately** — the comment at that site says "Apply before ggml and src are added", because exceptions thrown across ggml's C-ABI frames must unwind to this library's public-entry guards. A target-scoped rewrite would silently break that. Leave it |
| T1.5 | Add the **`sync + patches == committed tree`** verification to `sync-ggml.sh` | from §3.3 | low | **landed** — and it earned its keep on the first run (see below) |
| T1.6 | Add an **`upstream-merge-dryrun`** CI job | from §3.3 | low | **landed** — `.github/workflows/upstream-drift.yml`: a scheduled+PR `ggml-recipe` job (`sync-ggml.sh --check`, verified green for real) and an `upstream-merge-dryrun` job. See the note below on why the first version's hard check was dead code |
| T1.7 | Add `target_compile_features(... cxx_std_17)` so public headers advertise the standard | zero occurrences today | low | **landed** — `src/CMakeLists.txt`, immediately before `target_compile_definitions(transcribe ...)`, `PUBLIC` so it propagates to the C++ tests/examples and travels with the target if it is ever exported |

On T1.6: the first version of this job carried a hard failure on
`git merge-base --is-ancestor "$mb" upstream/main`, where `$mb` is the merge-base
of HEAD and upstream. That check **can never fire**. `git merge-base A B` is
*computed*, so the commit it returns is an ancestor of `B` by construction —
testing its ancestry against `B` is a tautology. It would have been a green tick
that verified nothing.

The condition actually worth failing on is the one §7.1 warns about: upstream's
*content* present in the tree while its *commit* is not an ancestor, which is
what content-copying across remotes produces. `git cherry` detects it by
patch-id, and the replacement was verified by constructing the failure: a
synthetic repo where the same patch is applied to both sides as separate commits
reports `1 commit(s) — 0 new, 1 already present by patch-id` and exits 1, while
a genuinely new upstream commit reports `1 new, 0 already present` and passes.
Both merge outcomes (clean, and conflict-with-details-block) were exercised
against real git too. A gate that has never been seen to fail is not a gate.

On T1.5: the first implementation compared the re-vendored tree against `git
status -- ggml`, i.e. tree-versus-HEAD. That conflates *"the tree matches its
recipe"* with *"the tree matches the last commit"* and would re-report every
already-committed recipe change forever. It also fired immediately — because the
very change under test (the `UPSTREAM` rewrite and the `.gitmodules` deletion)
showed up as a diff. Rewritten to `diff -rq` the staged tree against the live one
**before** the swap, so the invariant is a claim about the recipe: same pin + any
difference = error (downgraded to a warning by `--force`); different pin = a
reported pin bump. Verified by re-running at the pinned SHA:

```
sync-ggml: recipe reproduced ggml/ exactly (sync + patches == tree).
```

That check now means a hand-edit applied to `ggml/` without a matching patch in
`patches/ggml/` cannot survive a re-vendor unnoticed — the exact silent loss the
vendoring scheme exists to prevent.

### Tier 2 — Measured wins (needs the benchmark protocol)

| # | Item | Why it might win | How to verify |
|---|---|---|---|
| T2.1 | **Port the graph optimiser (A1)** | `CONT 4.6% + ADD 9% + MUL 4.9%` of encoder time are exactly its targets | **measured and declined** — the premise is wrong; see the census below |
| T2.2 | **Lift the `default_n_threads` cap** | `int cap = 8` at `src/transcribe-batch-util.cpp:406` caps a 16-P-core desktop at 8; `make_threadpool_params` (`:417`) then masks the pool to `performance_cpu_ids(n_threads)` | **Not measurable on the current box** (>8 P-cores required). Do it with a reason, or not at all |
| T2.3 | **Unify the 5 KV caches + decode loops** | Removes five copies of the same layout and the per-family runner loops; this is the fusion project's concluded "build first" layer | Structural + byte-identical transcripts across every family with a local GGUF |
| T2.4 | **Adopt the chunk planner/rebaser (A2)** | Closes the long-form gap; complements `docs/input-limits.md` | Word-timestamp rebasing is directly diffable |
| T2.5 | **`trim_backend_pools` retry path (A7)** | 4 models on an 8 GB card is a real Handy scenario | Handy's real multi-STT flow, not a synthetic one. **Requires a `patches/ggml/` patch first** — see the note below |

### T2.1 is declined on evidence, not on effort

The Tier-2 entry above assumed A1's targets and this repo's hot kernels were the
same set. A node-kind census over the four families with local GGUFs
(`Qwen3-ASR-1.7B`, `granite-speech-4.1-2b`,
`nemotron-3.5-asr-streaming-0.6b`, `parakeet-tdt-0.6b-v3`) says they are not:

| node kind | count | of total | what A1 does with it |
|---|---:|---:|---|
| `CONT` | 809 | 6.6 % | **nothing** |
| `IDENTITY` | 7 | 0.06 % | elides |
| `repeat` | 1 | 0.01 % | folds into consumers |
| (all kinds) | ~12 233 | 100 % | |

A1's three transforms are broadcast-`repeat` folding, identity materialisation
elision, and no-op/metadata-only node elision. On these graphs that is a
**~8-node addressable set out of ~12 233 (0.065 %)** — and the census was taken
across four families, so this is not one architecture's quirk.

The plan's premise was the error. `CONT 4.6% + ADD 9% + MUL 4.9%` is *encoder
time* — kernel time. A graph rewriter cannot delete a copy or a matmul the model
requires; it deletes nodes that were never needed. Those are different sets, and
this repo's graphs are already free of the second one: 809 `CONT` nodes are
necessary data movement, not redundancy. A pass that removed 0.065 % of nodes
could not move a 4.6 % share, so there is no measurement to run — the census is
the measurement, and it says no.

What this does **not** say: that A1 is worthless in general. audio.cpp's graphs
are built by a different frontend and may carry redundancy this one does not.
The finding is that *transplanting it here* has no target to act on.

Method and reproducibility: the census came from an instrumented build counting
`ggml_cgraph` node kinds at graph-build time. There is no committed tool for it
(the numbers are a one-off), so a re-run means re-instrumenting
`ggml_graph_*` construction — the GGUF family list above is what was measured,
and all four are available in the local Hugging Face cache.

### T2.5 is not a port of library code — it is a port of *ggml* code

`trim_backend_pools` is a two-line function in audio.cpp
(`src/framework/core/backend.cpp:333`) that does nothing but resolve a backend
proc by name and call it. All the substance is on the **ggml** side, in the three
procs audio.cpp added to its own vendored CUDA backend. ggml 0.23.0 has none of
them: its proc table exports only `comm_init`, `comm_free`,
`comm_allreduce_tensor`, `register_host_buffer`, `unregister_host_buffer` and
`get_features` (`ggml/src/ggml-cuda/ggml-cuda.cu:5670-5689`).

So T2.5 is really: **add a `patches/ggml/` patch that restores the three procs
against 0.23.0's internals, then add the retry helper on this side.** The bodies
do not transfer verbatim — between 0.12 and 0.23 the CUDA context changed
(`pools[…]`, `stream_priority`, and the `USE_CUDA_GRAPH` graph cache are all
different shapes now). This is a good test of the vendoring discipline, because
it is the *first* patch in `patches/ggml/` that is a feature rather than a fix,
and it must round-trip through `sync-ggml.sh` like any other.

The box can measure it: CUDA 13.3 toolkit and a 4070 Laptop with 8188 MiB are
both present, and the scenario is the 4-model Handy flow.

**Outcome — written, not yet measured.** The patch exists as
`patches/ggml/0002-export-cuda-pool-trim-and-graph-evict.patch` (+102 lines
across `include/ggml-cuda.h`, `src/ggml-cuda/common.cuh`,
`src/ggml-cuda/ggml-cuda.cu`), and the transcribe side as
`trim_backend_pools` / `evict_backend_graph_cache` /
`alloc_ctx_tensors_with_reclaim` in `src/transcribe-backend.{h,cpp}`.

**The retry path is an API, not yet a wired-up call site — so nothing reclaims
anything in production yet.** That is the honest state and the reason there is
no measurement below to report. `alloc_ctx_tensors_with_reclaim` takes the set
of *other* backends to reclaim from, and no caller in the tree supplies one:
there is no process-wide registry of live backends to draw it from (only
per-model lists — `plan.scheduler_list`, `append_accel_backends`), and the
allocation that actually fails under memory pressure is the per-arch weights
buffer in `src/arch/*/model.cpp`, ~20 separate sites that each hold only their
own plan. Finishing T2.5 therefore means adding a live-backend registry and
then adopting the helper at those sites — new global state with its own
lifetime and teardown ordering, in the neighbourhood of `cleanup_gpu()`. That
is a design change the user owns, so it is left here as the explicit next step
rather than guessed at.

Two things the 0.23.0 internals changed since audio.cpp's 0.12, both found by
reading the vendored tree rather than assuming the sibling's shape transferred:

1. **`ggml_cuda_pool` has no `clear()` at all** — only `alloc`/`free`. So the
   patch adds the virtual and implements it for *both* classes. Implementing it
   for `pool_leg` alone would have been a silent no-op on any machine where
   `new_pool_for_device` picks VMM (which is what the 4070 does).
2. **`ggml_cuda_graph_get_key(cgraph)` is `cgraph->nodes[0]`**
   (`ggml-cuda.cu:2629`), and the cache is
   `std::unordered_map<const void *, std::unique_ptr<ggml_cuda_graph>>`. The
   erase key the patch uses is the same expression, so it addresses the entry
   `cuda_graph()` looks up.

Worth recording, because it bounds the claim: **0.23.0 already evicts graphs on
its own.** `cuda_graph()` (`common.cuh:1475`) sweeps every 5 s (`:1479`) and
drops entries unused for ≥ 10 s (`:1482`). So `clear_graph`'s value is not that
it makes the memory returnable — it is that it returns it *now*, on a path that
has already failed to allocate and cannot wait out a sweep. The pool half has no
upstream equivalent at all: nothing reclaims a pool's idle buffers before
teardown.

Verified: the patch applies cleanly to a fresh upstream tree
(`sync-ggml.sh --dry-run`), and `ggml-cuda.cu` **compiles** with it — object
mtime 17:51:52 against source edits at 17:44:44, with 65 TUs rebuilt from the
`common.cuh` change. `USE_CUDA_GRAPH` is undefined under the `build-cuda`
preset (`GGML_CUDA_GRAPHS` defaults OFF), so that build exercises the `#else`
branch; the `#ifdef` branch is verified by inspection against upstream's own
key derivation and map type, not by compilation.

The transcribe side is compile-and-link verified too: in the same `transcribe`
target, `transcribe-backend.obj` rebuilt at 18:22:42 against source at 17:46:02
(and again after the 18:01:57 clang-format pass), and the link produced
`build-cuda/src/Release/transcribe.lib` at 18:24:21 with the target exiting 0 —
so the header's declarations and the .cpp definitions agree, which a
compile-only check of each file separately would not have shown.

**Adding the patch also edits the recipe, not just the tree.** A patch reaches
the vendored tree through `sync-ggml.sh`, which writes `ggml/UPSTREAM`'s
`patches:` list from a glob of `patches/ggml/*.patch`. Applying the patch
directly (as the compile check above did) puts its content in the tree while
leaving the list one entry short, and the venue that catches this is
`sync-ggml.sh --check` — which regenerates from the recipe and diffs the result
against `HEAD:ggml`, deliberately **not** against the working copy (`sync-ggml.sh`
lines 158-168), so it cannot be satisfied by fixing up the tree alone. The
committed set must therefore carry the regenerated `ggml/UPSTREAM` alongside the
patch and the three patched files. Until that commit lands, `--check` is *red by
design*: it is comparing `HEAD`, which does not yet have the patch. Green means
"the recipe in the commit reproduces the tree in the commit", so it is only
reachable through a commit, never from a dirty working tree.

**Not yet done:** the actual measurement. Per §8 this needs Handy's real 4-model
multi-STT flow, 4 runs dropping the first — and it is a *memory* claim, so the
metric is whether the fifth load succeeds where it previously failed, not a
timing delta. Until that runs, this is an implemented capability, not a win.

### Tier 3 — Larger programmes (propose, do not start)

- Declarative `model_specs/` with `dependencies`/`required_when` (A11).
- A family-id alias registry with canonical ids + a unit gate (from D24's lesson).
- Golden manifests for **all** families — the fusion project identified this as
  their largest validation gap, and this repo's `validate.py` cannot run on a
  box without oracles.

---

## 7. Do-not-do list

### 7.1 Programme-level

- **Do not attempt a speech.cpp-style unification here.** Their own tracker:
  ~12–12.5 months for 3–4 engineers, zero deletions executed, and a default-OFF
  gateway (`SPEECHCPP_ENABLE_UNIFIED_ABI`) that meant nothing in CI exercised
  the merged subsystem. Their conclusion is *"Both parents stay live."*
- **Do not add a second public façade.** `capi/audiocpp.h` became a third
  façade over the two they already had (their F14: façade accretion).
- **Do not gate a subsystem default-OFF and call it integrated.** Their C8: zero
  of transcribe.cpp's 51 tests ran in their CI.
- **Do not rename an upstream-owned file, and do not rename a GGUF architecture
  string** (D24). Additive aliases only.
- **Do not content-copy across remotes in place of merging.** It leaves the
  merge-base stale and git then reports applied commits as "behind" forever.

### 7.2 Code-provenance

`audio.cpp` is Apache-2.0; this repo is MIT. Anything copied — not merely
learned from — needs: the Apache-2.0 text, retention of copyright/NOTICE
notices, and a statement of changes. `THIRD-PARTY-LICENSES.md` is the natural
home. Prefer clean-room reimplementation from the described behaviour for
anything smaller than a module.

### 7.3 Verification integrity

- `validate.py`, `preflight.py` and `compare_tensors.py` **cannot run without
  reference oracles**. Do not present a green build as numerical validation.
- `cleanup_gpu()` (the end-of-run scheduler/KV free) is off-limits — ruled so
  three times, per `PASSOVER.md`.
- The multi-STT serializer was reverted. Do not re-add it; do not set
  `TRANSCRIBE_MULTI_STT_SERIALIZE` in Handy (it is inert).

---

## 8. Verification protocol for anything above

Per `PASSOVER.md` and the established method:

1. Run 4 times, **drop the first** (warm-up), average the remaining 3.
2. **Interleave** before/after arms within each round — total-runtime noise is
   ±5–20% and swamps small wins otherwise.
3. Prefer **stage counters** (`mel=`, `encode=`, `decode=` on stderr) over wall
   clock.
4. Prove output is unchanged with `--timestamps token`, filtering the
   non-deterministic `realtime:` line.
5. Anything under the noise floor: **revert it and say so**, rather than keeping
   the complexity.
6. Every optimisation gets a **kill-switch env var**.
7. `scripts/ci/clang-format.sh --check-diff` for our tree only (the whole-tree
   gate has 3 pre-existing offenders from `f5cb30a`).

---

## 9. The one-paragraph answer

transcribe.cpp should take **three things** from audio.cpp — its ggml graph
optimiser, its long-audio chunk planner/rebaser, and its backend pool/stream
tuning — and **one thing** from speech.cpp: the doctrine that contracts travel
and code does not. It should take nothing else, because on the axes that matter
most (ggml currency, ABI, streaming semantics, input limits, numerical
validation) it is already the more disciplined of the three, and the fusion
project's year of work is best read as an expensive confirmation of that.
