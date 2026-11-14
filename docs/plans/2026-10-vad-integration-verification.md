# VAD integration verification log

Date: 2026-10-16. This records local integration checks, not model publication
acceptance or an end-to-end ZER0 performance improvement.

## Branch and scope

- Worktree: `C:/Users/Z/.codex/worktrees/vad-integration/transcribe-fork`.
- Branch: `codex/vad-integration`, based on fork main / origin main `692350de`.
- Incoming branch: upstream VAD `de75ca13`.
- Merge conflicts are resolved. The merge and documentation remain uncommitted;
  `HEAD` is still `692350de` and `MERGE_HEAD` is `de75ca13`. No push or PR.
- Latest inspected upstream main is `b61b7f93`. Later unrelated upstream
  features are outside this VAD integration.
- Main and benchmark working trees were not edited. Their pre-existing changes
  remain separate. ZER0's existing `Handy_Multi_STT` working tree received only
  the paired-development instructions, integration note and historical-plan
  corrections; its existing application changes and dependency pins were kept.

## Integration decisions

Preserve the fork's embedded Earshot API and the upstream standalone Silero API.
Move the former implementation to `src/transcribe-earshot.*` and its Rust wrapper
to `earshot.rs`; retain the upstream role dispatcher in `transcribe-vad.*` and
the new Rust `VadSession` in `vad.rs`. Both are built together.

Resolve model-registry and loader references conditionally, and only register
Silero model tests when that family is enabled. Preserve the fork's model-set
and CLI streaming options. Regenerate the combined ABI bindings/hash rather
than choosing either side's generated files. The resulting public ABI digest
is `0a075c496509af7a`.

The internal architecture plugin ABI increases from 2 to 3 because `Arch` gains
a `VadOps` pointer. Old plugins must be rebuilt; the version check prevents an
old, shorter interface object from being read as the new layout. This is
distinct from the public C API's append-only compatibility rules.

Both September plan copies now identify their compiler observations as
archived, withdraw unsupported SIMD/RNNoise/resampler speedups, and describe the
app energy pre-gate as a score/inference policy rather than a denoiser bypass or
an unconditional PCM drop. UI parity remains an acceptance goal.

## Checks completed

Host: Windows x64, Visual Studio 18 2026 generator, Release, CPU backend;
CUDA/Vulkan disabled. These results apply to this working-tree state.

| Check | Result |
| --- | --- |
| `cmake --build build --config Release --target transcribe-cli` with `TRANSCRIBE_MODEL_SET=full` | Passed |
| CLI build in `build-none` with `TRANSCRIBE_MODEL_SET=none` | Passed; excluded-family references link correctly |
| Targeted CTest suite | 5/5 passed: API smoke, role resolution, VAD segmentation, stream dispatch, Silero synthetic smoke |
| Rust `cargo test -p transcribe-cpp --lib --test no_model --test vad` | 20/20 passed: 7 library, 12 no-model, 1 toy VAD test |
| Python and TypeScript ABI generator checks; Swift ABI hash check | Passed |
| Rust formatting | `cargo fmt --all --check` passed |
| Pinned C/C++ formatter | `scripts/ci/clang-format.sh --check-diff main` passed on 25 changed files |
| Teardown lint | `cmake -DSRC_DIR=src -P tests/lint_teardown.cmake` passed |
| Git whitespace / conflict checks | Working and staged diffs clean; no unresolved index entries |
| Paired Markdown notes | Local links, code fences and reciprocal agent instructions checked |

The Rust checks used the newly installed native build at `build/install` through
`TRANSCRIBE_DIR`, so they exercised this integration rather than an unrelated
cached native library. No new GGML or workspace Cargo.lock changes were needed.

Re-run the focused native suite with:

```powershell
ctest --test-dir build -C Release --output-on-failure -R '^transcribe_(api_smoke|stream_dispatch_unit|role_resolve_unit|vad_segments_unit|silero_vad_smoke)$'
```

Use the repository's `rtk` command wrapper where configured. Python tooling
must continue to use `uv run`.

## Remaining acceptance work

The full CTest matrix, GPU builds, dynamic-plugin runtime, every custom model
combination, real Silero checkpoint numerics/WER, Swift/TypeScript runtime
tests and a ZER0 build against this native integration were not exercised.
Do not infer those results from fixture or ABI checks.

No ZER0 audio behavior was changed: duplicate Rust/engine VAD ownership remains
to be fixed, and RNNoise remains app-side. FireRed, Smart Turn, engine pre-roll,
speech-clock parity and the optional engine preprocessing frontend remain
planned. No latency, battery, clipping or hallucination improvement is claimed.

Continue from the [revised implementation plan](2026-10-three-tier-super-vad-and-firered-architecture-plan.md)
and [paired-project contract](../zer0-engine-integration.md). Inspect current
Git state and both local projects before continuing; this is a dated snapshot.

## Revision 8 research/probe follow-up, 2026-10-16

Expanded the main roadmap with source-level mismatches, continuous framing,
valid-length/source-time contracts, live-toggle draining, bounded fan-out,
stale semantic-result handling and concrete work-package gates. Added the
[deeper evidence review](../research/2026-10-vad-preprocessing-evidence.md),
[source/metadata lock](../research/vad-source-lock-2026-10-16.json) and
[VoxLock review/implementation plan](2026-10-voxlock-voice-isolation-plan.md).
Updated both paired-project notes.

Executed the saved `scripts/vad/probe_zer0_frontend.py` against snapshots of
ZER0's actual resampler/denoiser modules with cached dependencies offline.
The saved utility reproduced all six framing/impulse rows in the evidence
review. Scratch CSV, build log, Cargo.lock and source-hash provenance are in
`build/zer0-frontend-probe/`. Rust template formatting and Python syntax/help
checks passed. Checked new/changed Markdown links/fences, source-lock JSON and
Git whitespace; these checks do not validate audio-model quality.

VoxLock code/config and public model metadata were reviewed without downloading
or executing weights. Its package-pinned and latest inspected ONNX checksums
match. No VoxLock port, enrollment feature, production runtime dependency or
audio default was introduced. No new full app/C++/GPU/WER run was performed for
this documentation and standalone-probe follow-up. Earlier integration-test
results above retain their original scope.

## Revision 9 contract and wrapper audit, 2026-10-16

Reviewed native VAD feed/flush/error/ownership behavior, Rust model locks and
ASR leases, recorder pause/acknowledge/drain and capture-overrun handling,
ChunkTap, Multi-STT close gates and existing probe skip behavior. Consulted
official ORT threading/quantization docs and pinned upstream wrapper/export
code. Added the [shared implementation contract](2026-10-audio-frontend-contract.md)
and reinforced both model plans and reciprocal project notes.

Executed the saved VoxLock bookkeeping utility with NumPy 2.5.3 and the pinned
reviewed source hash. Fourteen input lengths and a forced live switch reproduced
short-head loss, flush overshoot and pending warm-up suppression. The network
was an identity stub; no ONNX Runtime, model weights, real enrollment selection,
neural waveform quality, WER or compute performance was exercised. Full details
and scope are in the VoxLock plan and source lock. Scratch output is
`build/voxlock-bookkeeping-probe.json`.

Verified audited source hashes remained unchanged, Python syntax, local Markdown
links/fences, JSON metadata and Git whitespace. No C++/app audio behavior changed
and earlier integration tests were not re-run for this plan/probe-only update.
New numeric quality criteria are provisional preregistration targets, not
experimental results. Changes remain on the uncommitted VAD integration branch.

## Landed on main, 2026-11-14

The integration above was recorded on this branch as merge `703841ac` and is
now on `main` through the upstream sync `26764bf6` (upstream 0.4.2, which
contains this VAD work as squash commit `c63b18e2`). The reconciliation was
re-applied by 3-way merging this branch's result against upstream (base
`c63b18e2`), so the Earshot/Silero split, the plugin ABI 3 bump and the
conditional registrations are unchanged; later upstream changes (mmap
weights, VAD fixture smoke under `weights_mmap_smoke`) came on top.

Checks on the merged `main` (Windows, MSVC 14.52, CPU, full model set): full
build including tests, CTest 66/66 (incl. `silero_vad_smoke`,
`vad_segments_unit`, `weights_mmap_smoke`), Rust lib/no_model/vad/serde
28/28, ABI generators + Swift pin, teardown lint, clang-format diff gate.
Still not exercised, as before: real Silero checkpoint WER, GPU VAD, ZER0
running against the native VAD.

What to do next lives in
[the November master plan](2026-11-fork-master-plan.md), section B.
