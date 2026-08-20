# PASSOVER — audit of the 2026-08-20 optimization series

**Status:** the audit requested by the previous passover is **done**. This file
is now the record of what was checked, what was wrong, and what was changed.

**Repo:** fork of `handy-computer/transcribe.cpp`, remote `origin` =
`github.com/NairoDorian/transcribe.cpp`, branch `main`.

---

## 0. What happened

Four commits landed in one session (`8fecde4`, `e90b0e7`, `f0b1cba`,
`e602020`, base `16f579b`) — perf-core thread default, granite direct
depthwise conv, frame-batched RNN-T joint, qwen3 spec decode, `--multi` bench
harness, node profiler. The author flagged them as measured-but-rushed and
asked for a line-by-line correctness audit rather than more benchmarking.

That audit found **six defects**. Two were real bugs, one a silent performance
regression, one a latent crash, one an overstated correctness claim that
measurement falsified, one cosmetic. All are fixed. Transcripts are byte-identical before and after across 9 model×clip
combinations, so nothing in the fix set changes output on x86/CPU or CUDA.

---

## 1. THE ONE THING NOT TO UNDO — the multi-STT serializer

Unchanged from the previous passover, and still the most valuable thing here.

`8fecde4` added a per-device mutex (`MultiSttRunGuard`) serializing concurrent
`transcribe_run` calls; `f0b1cba` removed it. **Keep it removed.**

The author benchmarked 4 offline models over a 29.3 s clip and measured
serialized 2326 ms vs concurrent 5030 ms — a "55% win". That scenario does not
exist in the product. Measuring the real flow inverted it:

| scenario | concurrent | serialized | |
|---|---|---|---|
| 3 models, 5 s (real flow, streaming primary) | **440 ms** | 606 ms | +37.8% worse |
| 4 models, 5 s (real flow, offline primary) | **437 ms** | 636 ms | +45.4% worse |
| 4 models, 29.3 s (synthetic) | 5030 ms | **2326 ms** | −53.8% |

Mechanism: solo runs on the 5 s clip were granite 219 + qwen3 194 + canary
166 = **579 ms**. Serialized measured **606 ms** — the sum plus lock overhead,
confirming the lock worked. Concurrent measured **440 ms**, *24% below the
sum*, because one model's host-side work (mel, host decode loops, D2H
readback) overlaps another's GPU work. Serialization forfeits that overlap.
The long-form "win" was never throughput: the concurrent **min** (2323 ms)
already equalled the serialized **median** (2326 ms) — the gap was variance
from VRAM pressure with 4 weight sets on an 8 GB card.

If you think you have a reason to re-add device-level scheduling, the bar is:
measure the *real* flow (3 models, ~5 s, streaming primary). Verify
`git diff 16f579b HEAD -- src/transcribe.cpp` is still empty.

Handy's real Multi-STT flow, for reference: the primary is a **streaming**
model that finishes during recording; models 2/3/4 are preloaded while the
user speaks; at finalize only those 3 run, **concurrently**, over
**dictation-length audio (~5 s)**.

---

## 2. Defects found and fixed

### 2.1 Windows `performance_cpu_count()` dropped the last core — REAL BUG

`GetLogicalProcessorInformationEx` returns variable-length records where
`Size` is the stride. The shipped walk bounded itself with

```c
while (off + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= len + sizeof(GROUP_AFFINITY))
```

`sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)` is **80** on x64 — the union
is sized by its `GROUP_RELATIONSHIP` member — while a `RelationProcessorCore`
record is **48**. The condition reduces to `off <= len - 64`, so with N
48-byte records the walk stops at `i <= N-2`: **the last core is never
visited**. Verified by compiling a probe against the real Win32 headers:
`shipped-loop entries visited = 13 (expected 14)`.

It was invisible on the dev box only by luck — the dropped record is an
E-core, so the P-core count was still right. **On any homogeneous CPU (all
AMD, all non-hybrid Intel) it silently resolved one fewer thread than
intended.**

Fixed: bound by `len`, validate `Size` against a named minimum-record
constant. Probe with the corrected loop visits 14/14.

### 2.2 `parallel_for_all` lost 70% of its workers — SILENT REGRESSION

The previous passover claimed `parallel_for_all` "intentionally still uses the
logical count via `default_n_threads(/*cap=*/0)`". It does not. `cap <= 0`
disables only the **cap**, not the new performance-core restriction:

```c
n = performance_cpu_count();                 // 6 on the dev box
if (n > 0) n = min(n, usable_cpu_count());
if (cap > 0 && n > cap) n = cap;             // cap == 0 → no-op
```

So the batch host-parallel pool went from **20 workers to 6**. That pool hands
items out of an atomic counter with **no barrier**, so SMT siblings and
E-cores add real throughput there — the exact opposite of ggml's
barrier-synchronized op split that motivated the perf-core default.

Fixed: `parallel_for_all` calls `usable_cpu_count()` directly. The
`default_n_threads` header doc now states the perf-core semantics explicitly
and warns that `cap <= 0` does **not** restore the logical count.

### 2.3 granite/granite_nar copied the wrong Metal policy — MISSED WIN

The new `detect_conv_dw_direct` used `backend_default = !is_metal` for the
**in-block** depthwise site. That is the **pre_encode** policy. Six other
conformer families use `true` unconditionally at the in-block site (parakeet,
canary, canary_qwen, gigaam, medasr, sortformer — cohere is the one exception,
see below), and the vendored ggml Metal backend **does** implement the op —
`ggml-metal-device.m`:

```c
case GGML_OP_CONV_2D_DW:
    return op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
```

which is exactly what granite's branch builds (kernel cast to F32, f32 data,
f32 out). So granite forfeited the −19% encode win on Apple silicon for no
reason.

Root cause: a **stale** claim in the "CANONICAL Metal conv-quirk note"
(`conformer.cpp`) that `CONV_2D_DW` is unsupported on Metal. That note has
been corrected, with an explicit warning not to copy the pre_encode
`!is_metal` default to a 1-D (KH=1) in-block site.

Fixed: both granite families default to `true`. Because the policy is no
longer backend-dependent, the whole `backend_name` parameter added by the
series was removed — both `encoder.h` files are now byte-identical to
`16f579b`, and the `= ""` default-argument footgun the previous passover
worried about is gone by construction rather than by convention.

**This is the one change made without hardware to verify it on**, so here is
the evidence in full. Metal's `supports_op` returns true for this op with
these dtypes (`ggml-metal-device.m`). Six other families already run the same
op with the same dtypes and the same KH=1 shape there in production. And the
kernel *variant* matches too: Metal selects its pipeline with
`use_tiled = (nb12 < nb10)` (`ggml-metal-ops.cpp`), and granite's
`reshape_4d(x, T, 1, inner_dim, 1)` over a contiguous `[T, inner_dim]` leaves
`nb12 = 4T > nb10 = 4`, exactly as parakeet's in-block reshape does — so both
land on the non-tiled `kernel_conv_2d_dw_f32_f32`. This is not "a supported
op", it is the identical already-exercised kernel. Worst case remains a
scheduler CPU fallback (perf, not correctness), and
`TRANSCRIBE_CONV_NO_DIRECT_DW=1` is the kill switch.

Note the landscape is not uniform: **cohere** opts out of direct dw at *both*
its sites on CPU as well as Metal, for F16-kernel/measured reasons of its own.
Do not read that as evidence about the op.

### 2.4 `dec.joint.0` debug dump read a null pointer — LATENT CRASH

Both dump sites passed `logits.data()`. On the batched path `logits` is never
resized (only `joint_step` sizes it, and it is skipped), so `logits.data()` is
`nullptr` and the dump read `joint_n` floats from it. It was saved *only* by
`resolve_joint_batch_w()` forcing `W = 1` under `debug::enabled()` — in a
different function. Raising the batch default or reordering that guard would
have turned it into a crash.

Fixed: both dumps use `frame_logits` (identical value today, no coupling).

### 2.5 The `e602020` mistake, repeated — COSMETIC

`8fecde4`/`e90b0e7` inserted `resolve_joint_batch_w` *between*
`resolve_decode_threads`'s doc comment and its function, orphaning it — the
same detached-doc-comment defect that `e602020` was written to fix for
`run_one_inner`. Reattached. Its stale "(min(8, usable cpus))" text was
corrected too.

### 2.6 `TRANSCRIBE_RNNT_BATCH_CHECK` was unusable as shipped

The previous passover says of this hook: *"that hook is permanent, use it."*
As shipped, using it emitted **~500 `[error]` lines per run** — one per
consumed column — because ~4000 logits drift by ≤3e-5 from the tinyBLAS
kernel switch, which is the **documented, expected** behavior. Every one of
those lines also showed `serial_argmax == batch_argmax`, i.e. no divergence.

Fixed: a shared `joint_batch_check()` now logs at ERROR **only on an argmax
flip** (the thing that changes a decode decision) and at DEBUG for drift. It
also checks the TDT duration head's argmax, which the old TDT branch did not
look at at all. Re-run on jfk: **0 error lines**, drift still visible at
DEBUG.

---

## 3. Audited and found CORRECT (no change made)

### 3.1 Frame-batched RNN-T joint — the correctness argument holds

Re-derived from the code rather than trusted:

- `decoder_out` contents change **only** when `predictor_step_ggml` runs,
  which happens only when `predictor_dirty` is set — at exactly the one site
  that also clears `win_valid`. The `else` branch reads
  `next_state.h.back().data()`, whose contents the `std::swap` cannot have
  changed, because the swap always co-occurs with `predictor_dirty = true`.
- `step` is **monotonically non-decreasing** on every path (blank `+1`, TDT
  `+= duration` with `duration >= 0`, the `max_symbols` `+1`, and the
  `is_blank` fast-forward `+1`), so `step < win_base` is unreachable and
  `step >= win_base + W` catches every forward jump, TDT skips included.
- Bounds are exact, not lucky: the loop guard caps `step <= T_enc-1`, so a
  window based there reads through row `T_enc-1+W`;
  `precompute_enc_proj_ggml` sizes its output to exactly `T_enc*joint_h` and
  the pad grows it to `(T_enc+W-1)*joint_h`. No truncation, no overread.
- `ggml_tensor_overhead() * 16` covers the 8 nodes actually built; the joint
  weights live in `HostJoint`'s context, not this one.

**`decode_rnnt_greedy_streaming` was left serial, and that is right.** The
graphs are built per decode call; on a per-feed chunk of a handful of frames
the extra `ggml_init` + backend alloc + graph build would cost more than the
dispatches it saves. Do not "optimize" it without measuring the real Handy
streaming flow — that is the §1 lesson.

### 3.2 qwen3 speculative decode — KEEP (the previous passover suggested cutting it)

It was framed as "~250 lines of dead-by-default complexity", to be deleted on
the same logic that killed the serializer. That framing is wrong:

- `transcribe_run_params::spec_k_drafts` and
  `transcribe_capabilities::supports_spec_decode` are **pre-existing public
  ABI** (commit `de857f4`), documented in `include/transcribe.h`, and exposed
  by the CLI as `--spec-k-drafts`.
- `voxtral_realtime` already implements the same mechanism. qwen3_asr is the
  **second implementor of an established contract**, not a private
  experiment.
- Deleting it would make `--spec-k-drafts` silently no-op on qwen3 while
  working on voxtral — the inconsistency users actually hit.

The serializer was categorically different: a *new*, undocumented behavior
change that harmed the default path. This is default-OFF and inert.

**But its documented losslessness was overstated, and is now corrected.** The
code claimed in three places that greedy acceptance keeps the committed
sequence "bit-identical to plain stepping". Measured on
`samples/whole-earth.wav` (Qwen3-ASR-0.6B Q4_K_M, CPU, 255 generated tokens):

| k | tokens/run | output |
|---|---|---|
| 0 | 1.00 | `…stay foolish." And I have always wished…` |
| 1 | 1.06 | `…stay foolish," and I have always wished…` |
| 2, 3, 4 | up to 1.08 | identical to k=1 |

jfk and german are identical at every k; only the long clip exposes it.

The acceptance rule is **not** the culprit and is exactly right: k=1 and k=4
accept different numbers of drafts (1.06 vs 1.08 tokens/run) yet produce
byte-identical output, and both differ from k=0 at the same single token. If
acceptance were wrong the divergence would be k-dependent. The cause is that
every k >= 1 routes through `build_verify_graph` (T = k+1 >= 2 columns)
instead of `build_step_graph` (T = 1), and a multi-column `mul_mat` dispatches
a different kernel than the n=1 GEMV under `GGML_LLAMAFILE=ON` — the same
numerics caveat the RNN-T batched joint documents, which here flips one
near-tie argmax.

`include/transcribe.h` was already honest (it documents `spec_k_drafts == 0`
as the setting "for byte-equal reproduction of pre-spec behavior"), so no ABI
promise was broken and the feature is default-OFF. The three code comments
that claimed bit-identity now state the real property: drafting introduces no
token the model did not predict, which is not the same as byte-equality with
k=0.

Both subtleties the previous passover flagged were verified:

- **KV headroom guard is correct.** Loop entry requires
  `generated_ids.size() < max_new`, so `cur_past <= T_prompt + max_new - 2`,
  hence `cur_past + T_verify <= T_prompt + max_new + k - 1 < max_n_kv`. The
  window guard can never truncate generation early, so the path stays
  lossless in the tail case.
- **`last_pos_by_tok` update order is correct.** `all_ids.size() == cur_past+1`
  is invariant; the backfill writes index `cur_past+1+j` at position
  `cur_past+1+j`, and `j + 1 < n_commit` correctly leaves the tail token
  unpinned so the next iteration's lookup finds an earlier occurrence.

Known inefficiency, left alone because the path is off by default: the verify
loop refills and re-uploads the entire `[max_n_kv, T_verify]` mask every
iteration, where only the new columns change.

### 3.3 granite_nar conv — shapes verified analogous

The previous passover called this "the single most likely place for a real bug
in the series" (no local GGUF to test with). It is correct. `conv_module`'s
shape flow is identical to granite's: pw1 → `[2*inner_dim, T]`, GLU →
`[inner_dim, T]`, `cont(permute)` → `[T, inner_dim]`, so
`reshape_4d(x, T, 1, inner_dim, 1)` is the right `[W, H, C, N]`, and
`reshape_3d` back is right. All `build_encoder_graph` call sites in both
families were covered.

### 3.4 nemotron language caps — verified against the real GGUF

Dumped `nemotron-3.5-asr-streaming-0.6b-F16.gguf` directly:

```
auto_id = 101
n_locales = 121   n_general.languages = 32
in general.languages but NOT in dictionary: []      <- superset CONFIRMED
locales mapping to auto_id: ['auto']                <- string filter == index filter
```

Overwriting loses nothing, and 121 − 1 = **120** advertised, matching what
Handy's log showed. Stronger: since `resolve_prompt_id` returns `-1` for
anything not in the dictionary, **merging would have been the bug** — it would
advertise languages the model must reject. Non-prompt variants are guarded by
`has_prompt` and untouched.

### 3.5 `--multi` CLI harness — no leaks

`free_all()` is called on every early return, and `slots` is pre-sized with
null members so partial teardown is safe.

### 3.6 Linux/macOS `performance_cpu_count()` paths — reviewed, not executed

Still compiled only on MSVC here. Reviewed for the `fast.empty()` semantics
the previous passover worried about: they are consistent. If
`/sys/devices/cpu_core/cpus` is absent and no CPU exposes `cpu_capacity`,
`best_cap` stays `-1` and `fast` is cleared, which correctly means "no class
info, every usable CPU counts". If affinity excludes every fast-class CPU the
function returns 0 and the caller falls back to `usable_cpu_count()` — a
sane degradation. Still unexecuted; treat as review-grade, not tested.

---

## 4. Open questions deliberately NOT acted on

- **Seven families hand-roll the fallback `default_n_threads()` gives them.**
  `canary`, `canary_qwen`, `cohere`, `funasr_nano`, `granite`, `qwen3_asr`,
  `voxtral` and `whisper` all do
  `n = cc->n_threads; if (n <= 0) n = default_n_threads();` and pass the
  result to `parallel_for_all`, each item's mel pinned to 1 thread. That is
  the §2.2 defect spelled explicitly: a barrier-free work-stealing pool sized
  by the barrier-oriented count. `8fecde4` silently took them from 8 workers
  to 6.

  **Deliberately not changed.** Deleting the hand-rolled fallback (letting
  `parallel_for_all` apply its own, now-correct default) would take them to
  20, which is not a restore of anything — base was 8. And
  `parallel_for_all` already clamps to `min(n, n_threads)`, so it only bites
  on batches larger than 6 utterances. Changing eight families' batch
  threading on an unmeasured hunch is precisely the §1 mistake. Measure a
  large batch first; if it wins, the fix is a three-line deletion per family,
  not an addition.

- **The cap of 8 in `default_n_threads`.** A 16-P-core desktop still gets 8.
  Almost certainly leaving performance on the table, but it is also what stops
  a 64-core server spawning 64 threads on a tiny graph, and it cannot be
  measured on this box. Raising it affects every family — do it only with
  numbers, and say so loudly.
- **A small joint window for the streaming decode path (W=4?).** See §3.1 for
  why it is probably a loss. Needs the real Handy streaming flow to settle.
- **Node profiler.** Now claims a single scheduler via a CAS and warns if a
  second appears, so the unsynchronized global has exactly one writer. In a
  concurrent host the report therefore covers one session, not the process.

---

## 5. Things that are off-limits

- **`cleanup_gpu()` / per-run `safe_sched_free`** (fork commit `c9bd43b`). It
  looks like an obvious performance target. It is not. It exists to stop CUDA
  galloc accumulation with 4 models resident on an 8 GB card, and the user has
  ruled it off-limits **three** times now. Do not touch it, and do not
  benchmark it as a way to argue for touching it.
- **`ggml/` and `src/third_party/`** — vendored, never formatted or edited.
- Families with no local GGUF (`granite_nar`, `cohere`, `moss`, `medasr`, …):
  prefer reasoning and `--check`-style hooks over speculative edits.

---

## 6. Conventions (read `AGENTS.md` first — it is canonical)

- **Python is always `uv run`.** Never bare `python`/`pip`.
- **Build:** `cmake --build build --target transcribe-cli --config Release`.
  A CUDA tree exists at `build-cuda` (`CMAKE_CUDA_ARCHITECTURES=89`). Both
  have `GGML_LLAMAFILE=ON`, which is why the batched joint's kernel switch
  shows up at all.
- **Format:** `scripts/ci/clang-format.sh` (writes) / `--check`. Pinned via
  `uvx`; never a system clang-format.
- **Teardown lint:** `cmake -DSRC_DIR=$PWD/src -P tests/lint_teardown.cmake`.
- **C ABI:** no C++ exception escapes a public entry point; teardown uses
  `transcribe::safe_*`.
- **Public ABI changes are expensive** — `include/transcribe.abihash` gates
  five bindings plus a pinned Swift header hash. Avoid.
- **Do not commit/push/PR unless the user explicitly asks.**

---

## 7. Verification tools

- `TRANSCRIBE_RNNT_BATCH_CHECK=1` — batched vs serial joint. **Now errors only
  on an argmax flip**; drift is DEBUG. A clean run prints zero errors.
- `TRANSCRIBE_RNNT_BATCH_W=1` — kill switch back to the serial joint.
- `TRANSCRIBE_CONV_NO_DIRECT_DW=1` — kill switch back to im2col depthwise.
- `TRANSCRIBE_PERF_DEBUG=1` — per-family stage breakdowns; `=nodes` — the
  cross-family per-op/per-node table.
- `--timestamps token` — the byte-diff harness for decode changes. Filter the
  non-deterministic `realtime:` / `timings:` / `[debug]` lines and diff the
  payload lines. This is what proved the fix set output-neutral across
  nemotron/parakeet/granite × jfk/german/whole-earth.
- `uv run scripts/validate.py all --family <f>` and `uv run scripts/preflight.py`
  — the real numerical gates. **They need reference oracles that are not on
  this machine.** If you can obtain them, running them is worth more than
  anything either of these two sessions did.

**Benchmark methodology the user requires:** 4+ runs, drop the first,
average/min the rest, **interleave the arms within each round**, prefer the
library's stage counters over wall clock, and **revert any "win" inside the
noise floor**. The box is an i9-13900H (6 P + 8 E, 20 logical) with ~11%
background load; single measurements swing 2–5×.

---

## 8. The lesson, for whoever is next

The previous passover's most valuable content was §4 — the record of a
confident, well-benchmarked change that was simply wrong because it was
measured against the wrong scenario. This audit adds a second, quieter
failure mode worth the same attention:

**Every defect in §2 was invisible on the machine it was written on.** The
Windows core walk dropped an E-core, so the P-core answer stayed right. The
`parallel_for_all` regression only slowed a path nobody timed. The Metal
policy only costs hardware that isn't here. The null dump was masked by a
guard in another function. None of them would have been caught by more
benchmarking — only by reading.

So: when a change's correctness depends on a platform, a code path, or a
guard that your test run does not exercise, that is precisely where to look
hardest. And when you copy a policy from a neighbouring call site, check what
that policy was *for* — §2.3 is an entire missed optimization that came from
copying the right-looking line from the wrong site.
