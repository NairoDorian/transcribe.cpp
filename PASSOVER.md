# PASSOVER — audit of the 2026-08-20 optimization series

**To:** the next AI agent working on this repo
**From:** the agent that wrote commits `8fecde4`, `e90b0e7`, `f0b1cba`, `e602020`
**Repo:** `C:\Users\Z\Downloads\PROJECTS\transcribe-fork` (fork of
`handy-computer/transcribe.cpp`, remote `origin` =
`github.com/NairoDorian/transcribe.cpp`, branch `main`)

---

## 0. Your task, in one paragraph

Four commits landed on `main` in one session. They were measured, but they
were written fast and under measurement pressure. **Your job is not to add
features and not to re-run the benchmarks. Your job is to READ these diffs
line by line, file by file, and think hard about whether each change is
actually correct and actually an improvement** — including the ones I was
confident about. Where you find a bug, a latent hazard, a wrong abstraction,
or something that is merely more complex than it needs to be, **fix it**.
The one thing you must NOT do is re-introduce the multi-STT run serializer
(see §4 — it was measured, it was harmful, and it is gone deliberately).

Bias your effort toward *thinking*, not toward tooling. I already spent this
session's budget on measurement; the marginal value now is in careful reading.
The last commit in the series (`f0b1cba`) is proof that this matters: it
reverted a 121-line feature from the first commit that benchmarks had
"proven" was a 55% win and that turned out to be a 38–45% regression on the
workload that actually matters. **Assume there is at least one more mistake
of that kind still in the tree, and go looking for it.**

---

## 1. The commits under audit

Run `git log --oneline 16f579b..HEAD`. Base for all diffs is `16f579b`.

| commit | title | net |
|---|---|---|
| `8fecde4` | perf(*): multi-STT device serializer, perf-core threads, granite direct dw conv | 14 files, +1088/−69 |
| `e90b0e7` | perf(parakeet): frame-batched joint for the RNN-T greedy decode | 1 file, +271/−7 |
| `f0b1cba` | revert(multi-stt): drop the run serializer | 2 files, +16/−126 |
| `e602020` | style(transcribe): reattach run_one_inner doc comment | 1 file, −1 |

**Net effect of the whole series** (`git diff 16f579b..HEAD --stat`) — 14
files, +1248/−76:

```
examples/cli/main.cpp               | 206 +++++      <- --multi bench harness (new)
src/arch/granite/encoder.cpp        |  49 +++        <- direct depthwise conv
src/arch/granite/encoder.h          |   3 +-        <- backend_name param
src/arch/granite/model.cpp          |   6 +-        <- pass backend name
src/arch/granite_nar/encoder.cpp    |  50 ++++      <- same conv change (UNVERIFIED)
src/arch/granite_nar/encoder.h      |   3 +-
src/arch/granite_nar/model.cpp      |   3 +-
src/arch/parakeet/decoder.cpp       | 278 +++++      <- frame-batched RNN-T joint
src/arch/parakeet/model.cpp         |  22 +++        <- nemotron language caps fix
src/arch/qwen3_asr/capabilities.cpp |   5 +          <- supports_spec_decode = true
src/arch/qwen3_asr/decoder.cpp      |  78 ++++      <- build_verify_graph (new)
src/arch/qwen3_asr/decoder.h        |  29 +++        <- VerifyBuild struct
src/arch/qwen3_asr/model.cpp        | 250 +++++      <- spec-decode loop (default OFF)
src/transcribe-batch-util.cpp       | 342 +++++      <- perf-core threads + node profiler
```

`src/transcribe.cpp` appears in the individual commits but is **byte-identical
to `16f579b`** in the net — that is the point of `f0b1cba` + `e602020`. Verify
with `git diff 16f579b HEAD -- src/transcribe.cpp` (expect empty output). If
that ever becomes non-empty, someone re-added the serializer.

---

## 2. Repo conventions you must follow

Read `AGENTS.md` first — it is the canonical instruction file (`CLAUDE.md`
just points at it). The load-bearing rules for this work:

- **Python is always `uv run`.** Never bare `python`/`pip`. (This box also has
  a working venv at `.venv/Scripts/python.exe` used for scratch scripts.)
- **Build:** `cmake --build build --target transcribe-cli --config Release`.
  A second CUDA build tree exists: `build-cuda` (already configured,
  `CMAKE_CUDA_ARCHITECTURES=89` for the RTX 4070). Both have
  `GGML_LLAMAFILE=ON` — this matters, see §5.
- **Format before committing:** `scripts/ci/clang-format.sh` (writes) /
  `--check` (verifies). It is pinned via `uvx`; do not use a system
  clang-format. Vendored trees (`ggml/`, `src/third_party/`) are never
  formatted and should not be edited.
- **C ABI exception discipline:** no C++ exception may escape a public entry
  point; new public entry points route through an `api_guard_*` wrapper or
  are nothrow by construction. Teardown uses `transcribe::safe_*`, never raw
  `ggml_backend_free` etc. — `tests/lint_teardown.cmake` enforces this.
- **Do not commit/push/PR unless the user explicitly asks.** (They did ask,
  for these four.)
- **Public ABI changes are expensive.** `include/transcribe.abihash` gates
  five language bindings plus a pinned Swift header hash; changing the public
  header means regenerating FFI (`bindings/python/_generate/generate.py`,
  `cargo xtask bindgen`) and consciously bumping the Swift pin. Avoid.

---

## 3. Environment on this machine (so you don't rediscover it)

- **CPU:** i9-13900H — 6 P-cores + 8 E-cores = 14 physical, 20 logical. This
  hybrid topology is the *reason* for the thread change in §5.1. **The machine
  is noisy** (background load ~11%, a `ProcessGovernor` process runs): single
  measurements swing 2–5×. Any benchmark needs interleaved arms and
  min/median over many rounds. Do not trust a single number.
- **GPU:** RTX 4070 Laptop, **8 GB** — this is small relative to the 4-model
  multi-STT set (~4.4 GB of weights resident), which is exactly why the
  long-form concurrency case thrashed (§4).
- **Models** (already downloaded, in the HF cache, not the app dir):
  `C:\Users\Z\.cache\huggingface\hub\models--handy-computer--<name>-gguf\snapshots\<sha>\*.gguf`
  Available: nemotron-3.5-asr-streaming-0.6b (Q8_0), granite-speech-4.1-2b
  (Q4_K_M/Q5_K_M/Q6_K), Qwen3-ASR-1.7B (Q4_K_M…BF16), Qwen3-ASR-0.6B,
  canary-1b-v2 (Q4_K_M/Q5_K_M), parakeet-tdt-0.6b-v3, parakeet-tdt-1.1b,
  parakeet-unified-en-0.6b, nemotron-speech-streaming-en-0.6b.
  **No granite_nar and no cohere/moss/etc. GGUF locally** — that gap matters
  in §5.2.
- **The downstream product** is Handy_V2 (`..\Handy_V2`, branch
  `Handy_Multi_STT`), a Tauri app. It consumes this repo as a git dependency
  pinned in `src-tauri/Cargo.lock`; `bun run tauri dev` runs
  `scripts/check-transcribe-deps.ts`, which auto-detects a new `main` tip and
  `cargo update`s to it. So **anything you push to `main` lands in the user's
  app on their next dev run.** Be careful.
- Handy's real Multi-STT flow (confirmed from its runtime log — memorize this,
  it is what §4 turns on): the **primary is a streaming model** that
  transcribes live *during* recording and is already finished when recording
  stops; models 2/3/4 are **preloaded while the user speaks**; at finalize
  only those 3 run, **concurrently**, over **dictation-length audio (~5 s)**.

---

## 4. THE ONE THING NOT TO UNDO — the multi-STT serializer

`8fecde4` added a per-device mutex (`MultiSttRunGuard`) that serialized
concurrent `transcribe_run`/`transcribe_run_batch` calls, gated on
`TRANSCRIBE_MULTI_STT_SERIALIZE`. `f0b1cba` removed it entirely. **Keep it
removed.** Here is the full reasoning so you don't have to re-derive it:

I benchmarked **4 offline models over a 29.3 s clip** and measured serialized
= 2326 ms median vs concurrent = 5030 ms — a "55% win". That scenario does
not exist in the product. Measuring the real flow (§3) inverted it:

| scenario | concurrent | serialized | |
|---|---|---|---|
| 3 models, 5 s (real flow, streaming primary) | **440 ms** | 606 ms | +37.8% worse |
| 4 models, 5 s (real flow, offline primary) | **437 ms** | 636 ms | +45.4% worse |
| 4 models, 29.3 s (my synthetic) | 5030 ms | **2326 ms** | −53.8% |

The mechanism, which is the part worth internalizing: solo runs on the 5 s
clip were granite 219 ms + qwen3 194 ms + canary 166 ms = **579 ms**. The
serialized arm measured **606 ms** — the sum plus lock overhead, confirming
the lock did exactly what it claimed. The concurrent arm measured **440 ms**,
*24% below the sum*, because one model's host-side work (mel, host decode
loops, D2H readback) overlaps another's GPU work. Serialization forfeits that
overlap. And the long-form "win" was never throughput: there the concurrent
**min** (2323 ms) already equalled the serialized **median** (2326 ms) — the
gap was variance from VRAM pressure with 4 weight sets resident on an 8 GB
card, not speed.

A `transcribe_run_serialization_set/get` public ABI pair was also drafted to
make it host-controllable and was deliberately **not shipped** (§2: ABI
additions are permanent, and this one would have been a footgun).

**Conclusion the user endorsed:** Handy's existing architecture is correct;
the library should not second-guess the host's scheduling. The portable wins
are the per-model ones. If you think you have found a reason to re-add
device-level scheduling, the bar is: measure the *real* flow (3 models, ~5 s,
streaming primary), not a synthetic one — and read the header comment in
`examples/cli/main.cpp` (`multi_main`), which records these numbers so this
does not get re-litigated from first principles.

---

## 5. What to audit, change by change

Below is every surviving change with what I believe, what I verified, and —
most importantly — **what I am unsure about**. Treat the "suspicion" lines as
your starting worklist, not as a complete list of what's wrong.

### 5.1 Perf-core thread default — `src/transcribe-batch-util.cpp`

`default_n_threads()` previously returned `min(8, logical CPUs)`. It now
returns `min(8, performance_cpu_count())`, where the new
`performance_cpu_count()` counts **one entry per physical core of the fastest
core class only**:
- Windows: `GetLogicalProcessorInformationEx(RelationProcessorCore)`, highest
  `EfficiencyClass` among cores in the process affinity mask, two passes.
- macOS: `sysctlbyname("hw.perflevel0.physicalcpu")`, falling back to
  `hw.physicalcpu`.
- Linux: `/sys/devices/cpu_core/cpus` (Intel hybrid) or per-CPU
  `cpu_capacity` (ARM big.LITTLE) to pick the fast class, then collapse SMT
  siblings via `topology/core_id` × `physical_package_id`, honoring
  `sched_getaffinity`.
- Returns 0 when unavailable → caller falls back to the old
  `usable_cpu_count()`. Always clamped by the affinity mask and the cap, so
  **the resolved count can never exceed the old default.**

*Why:* ggml's CPU backend splits each op's rows evenly and joins on a spin
barrier, so every thread waits for the slowest. Proven by pinning on this box:
6 threads on P-cores ran the parakeet encoder in **1.9 s**, the same 6 on
E-cores in **5.4 s**, 12 threads on the 6 P-cores (SMT) in **2.9 s**.
Resolved 8→6 here; parakeet-v3 encode 2623→1610 ms (−39%, 21-round
interleaved min).

**Audit this hardest — it is the riskiest change in the series**, because it
is the only one whose platform code is *mostly unexecuted on this machine*.
Specific suspicions:
1. **The Windows buffer walk is convoluted.** The loop condition is
   `while (off + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= len + sizeof(GROUP_AFFINITY))`
   with a second bounds check inside (`off >= len || e->Size == 0 || off + e->Size > len`).
   That outer condition is nearly meaningless and the real guard is the inner
   one. It is *safe* as far as I can tell but it is ugly and hard to prove.
   **Consider rewriting it as a clean `while (off < len)` with the inner
   `e->Size` validation.** Convince yourself of the bounds first.
2. **The macOS and Linux paths have never been compiled**, let alone run —
   this session only built MSVC/Windows. At minimum, eyeball them for
   compile errors (headers, types, `CPU_SETSIZE` loops) and for the
   `fast.empty()` semantics: an empty `fast` set is overloaded to mean "no
   class info, every usable CPU counts", and I want a second opinion that
   every path sets it consistently. The `best_cap`/`fast` interaction in the
   ARM branch is the fiddliest part.
3. **The cap of 8 was left in place deliberately** (it makes the change
   strictly non-increasing vs the old default, so it can't regress a machine
   I couldn't test). But it means a 16-P-core desktop still gets 8. That is
   almost certainly leaving performance on the table. I could not measure it
   here. **Think about whether the cap should be raised or removed** — but
   note it is also what keeps a 64-core server from spawning 64 threads on a
   tiny graph. If you change it, say so loudly; it affects every family.
4. `parallel_for_all` intentionally still uses the *logical* count
   (`default_n_threads(/*cap=*/0)`) because it is task-parallel, not
   barrier-synchronized. Check I didn't break that distinction.

### 5.2 Granite direct depthwise conv — `src/arch/granite{,_nar}/`

`granite_conv_module` lowered its k=15 depthwise conv through
`conv_1d_dw_f32`, whose `B==1` path uses im2col (15× scratch expansion) plus a
degenerate per-channel `[1 × k]` matmul. Replaced with the single-op
`ggml_conv_2d_dw_direct` (H=1) lowering that the shared conformer
`conv_module` already uses for parakeet/canary — same f32 kernel cast, same
`TRANSCRIBE_CONV_DIRECT_DW` / `TRANSCRIBE_CONV_NO_DIRECT_DW` overrides, same
Metal opt-out. Dispatch resolved per load from the bound backend name,
threaded through `build_encoder_graph` as a new `backend_name` parameter.
Measured: granite encode CPU 6706→5406 ms (−19.4%), CUDA 468→380 ms (−18.9%),
transcripts byte-identical.

**Suspicions:**
1. **`granite_nar` is UNVERIFIED.** It got the mechanically identical change,
   it compiles, but there is no granite_nar GGUF on this machine so its
   output was never checked. **Read that diff especially carefully** and
   confirm the shapes really are analogous (`inner_dim`, the pad, the
   reshape back to 3D). This is the single most likely place for a real bug
   in the series.
2. The new `backend_name` parameter defaults to `""` in both headers. An
   empty string is not Metal, so it resolves to *direct* — meaning a caller
   that forgets to pass it silently gets the direct path on Metal too, which
   is precisely the case the opt-out exists for. There is only one call site
   each today and both pass `cm->backend.c_str()`, so it is fine now, but
   **consider whether the default should be the safe one** (or whether the
   parameter should be non-defaulted so a new call site must decide).
3. There is now near-duplicate `detect_conv_dw_direct()` in *both*
   `granite/encoder.cpp` and `granite_nar/encoder.cpp`, and a third
   equivalent in the shared conformer. **Consider hoisting one helper into
   `src/conformer/conformer.h`** next to `resolve_conv_direct` /
   `detect_direct_pw`, which is where it belongs.

### 5.3 Frame-batched RNN-T joint — `src/arch/parakeet/decoder.cpp` (`e90b0e7`)

The greedy loop evaluated `joint(enc_proj[frame], pred_state)` as one ggml
dispatch per frame (~0.5 ms/call, ×425 frames). `JointGraphBatch` evaluates a
**W-frame window in one dispatch** — the pred side is constant between
emissions, so `pred_proj` is computed once and broadcast. The window is
invalidated on every emission and recomputed at the current frame with the new
state, so the sequence of `(frame, state)` joint evaluations, and therefore
every decode decision, is identical to serial by construction.
`enc_proj_all` is zero-padded by `W-1` frames so a window based near `T_enc`
stays in bounds (padded columns are never consumed).

Defaults are per-head and measured, not symmetric: **RNN-T W=16**, **TDT
serial (W=1)** — the TDT duration head already skips frames, so windows get
invalidated after ~2 consumed columns and batching measured **3–4× slower**
there. `TRANSCRIBE_RNNT_BATCH_W` overrides both (≤1 = serial kill switch,
clamped 256); debug dumping forces serial; a batch-graph build failure falls
back to serial silently.

Measured: nemotron decode CPU 279→162 ms (−42%), CUDA 247→126 ms (−49%),
total on CUDA −21%.

**Numerics — read this carefully.** With `GGML_LLAMAFILE=ON` the W-column
`mul_mat` may take a different kernel (tinyBLAS) than the n=1 GEMV, shifting
logits by accumulation order: **bit-identical** on parakeet-v3's joint shape,
**≤1.5e-5** max element drift on nemotron's. That is ~6× below the family's
own ~1e-4 envelope vs the NeMo reference (see the `decoder.cpp` file header),
and I verified **zero argmax flips over ~3450 tokens across 9 samples / 7
languages** including a 2067-token long-form, with byte-identical token
payloads (ids, timestamps, confidences). `TRANSCRIBE_RNNT_BATCH_CHECK=1`
re-runs the serial joint per consumed column and reports any byte difference
plus both argmaxes — that hook is permanent, use it.

**Suspicions:**
1. **This is the change most likely to bite in a way tests didn't catch**,
   because "decisions are identical by construction" is an *argument*, not a
   proof. Re-derive it yourself from the code. Pay particular attention to:
   the `max_symbols` stuck-frame path (does the window stay valid when
   `step` advances without an emission?), and the TDT `duration == 0` +
   `is_local` fast-forward branch which mutates `iter`/`step` in ways the
   window-validity predicate must respect. TDT defaults to serial so a bug
   there is latent, not active — but latent bugs behind an env var are worse
   than active ones.
2. `decode_rnnt_greedy_streaming` was **deliberately left serial and
   untouched** (per-feed windows are a handful of frames; it is the
   latency-critical dictation path). Confirm that was the right call — but
   note the streaming path is what Handy's primary model actually uses, so
   if there IS a win there it matters more than the offline one I optimized.
   Think about whether a small window (W=4?) would help the streaming
   finalize without hurting per-feed latency.
3. The `win_valid`/`win_base` state is duplicated between the TDT and RNN-T
   loops with slightly different surrounding code. Consider whether it can be
   factored without obscuring the invalidation logic (readability of this
   loop matters more than DRY — use judgment).

### 5.4 Nemotron language caps — `src/arch/parakeet/model.cpp`

Prompt-conditioned variants resolve language hints by **exact string lookup**
in the GGUF prompt dictionary, which carries short aliases (`"fr"`, `"en"`)
next to BCP-47 locales (`"fr-FR"`). `caps.languages` only carried
`general.languages` (region-qualified only), so `-l fr` was rejected by the
library's language gate and capability-driven hosts silently dropped the
user's hint and fell back to auto-detect. `load()` now overwrites
`caps.languages` with the dictionary locales (minus the `"auto"` slot) when
`has_prompt`. Verified working in the product (Handy's log now shows the full
120-entry list).

**Suspicions:** this one I am fairly confident in, but check (a) that
overwriting rather than merging can't *lose* a language that was in
`general.languages` but not in the dictionary — I believe the dictionary is a
superset, **verify that assumption** against the GGUF; (b) the interaction
with `supports_language_detect` and the `"auto"` slot; (c) that non-prompt
parakeet variants are genuinely untouched.

### 5.5 Qwen3 speculative decode — `src/arch/qwen3_asr/` (**default OFF**)

A 1-gram-lookup spec-decode path ported from `voxtral_realtime`:
`build_verify_graph` runs `T = k+1` positions through
`causal_lm::block_step_n` with per-position argmax; drafts come from a 1-gram
last-position map; greedy acceptance commits the longest confirmed prefix,
which keeps the token sequence bit-identical to plain stepping (verified for
k=0..3). `caps.supports_spec_decode` is now advertised.

**It is default-OFF because it measured as a LOSS**: on CPU the T=2 verify
costs ~1.5× a single-token step (it leaves the matvec fast path) and at the
observed ~1.1 tokens/run acceptance it loses; on CUDA it is break-even.

**Suspicions:** this is ~250 lines of complexity in `model.cpp` paying for
nothing today. **Seriously consider whether it should be deleted**, on the
same logic that killed the serializer — dead-by-default complexity in a
correctness-critical decode loop is a liability. Arguments to keep: it is
genuinely lossless, it is properly gated, and it may win on long repetitive
dictation (untested). Arguments to cut: nobody can enable it from Handy, and
it makes the step loop much harder to read. **Make a call and justify it.**
If you keep it, at least check the `max_n_kv` headroom fallback (`k_drafts`
forced to 0 when the context clamp removes the draft slack) and the
`last_pos_by_tok` update ordering, which is subtle and which I had to fix
once already (the first version pinned a token's own tail position and got
1.02 tokens/run instead of 1.11).

### 5.6 CLI `--multi` harness + node profiler

- `examples/cli/main.cpp`: `--multi m1,m2,… [--multi-serial] --repeat N`
  loads N models/sessions and transcribes one wav with all of them
  concurrently (or serially), printing per-model wall/mel/encode/decode and
  per-round wall. Round 0 is cold. Its header comment records the §4 numbers.
  This is a dev harness, not a product feature — check it doesn't leak
  sessions/models on the error paths (`free_all` is called on every early
  return, but re-verify).
- `src/transcribe-batch-util.cpp`: `TRANSCRIBE_PERF_DEBUG=nodes` attaches a
  scheduler eval callback at `configure_sched_n_threads` (the choke point
  every family crosses before compute) and prints a per-op/per-node table at
  exit via `atexit`. **Known limitation, stated in its comment: the
  accumulator is a process global with no locking — single-session
  diagnostics only.** In a concurrent host (i.e. Handy's multi-STT) enabling
  it would race. It is inert unless requested, but consider whether "inert
  unless requested" is a strong enough guard for a data race, or whether it
  should refuse to attach when a second scheduler shows up.

---

## 6. Things I deliberately did NOT touch (leave them alone unless you have proof)

- **`cleanup_gpu()` / per-run `safe_sched_free`** (fork commit `c9bd43b`).
  This frees the scheduler after every run. It looks like an obvious
  performance target — it is not. It exists to stop CUDA galloc accumulation
  with 4 models resident on an 8 GB card, and **the user has explicitly ruled
  it off-limits twice.** Do not touch it, do not benchmark it as a way to
  argue for touching it.
- **`ggml/` and `src/third_party/`** — vendored, synced from upstream, never
  formatted or edited here.
- The `granite_nar`, `cohere`, `moss`, `medasr`, … families beyond the conv
  change: no local GGUFs, so anything you change there is unverifiable on
  this machine. Prefer reasoning + `--check`-style hooks over speculative
  edits.

---

## 7. Verification tools that already exist (use them instead of rebuilding)

- `TRANSCRIBE_RNNT_BATCH_CHECK=1` — batched vs serial joint logits, per
  consumed column, with both argmaxes on mismatch.
- `TRANSCRIBE_RNNT_BATCH_W=1` — kill switch back to the serial joint.
- `TRANSCRIBE_CONV_NO_DIRECT_DW=1` — kill switch back to im2col depthwise.
- `TRANSCRIBE_PERF_DEBUG=1` — per-family stage breakdowns (several families
  print detailed tables); `=nodes` — the cross-family per-op/per-node table.
- `--timestamps token` — the byte-diff harness for decode changes. Filter the
  non-deterministic `realtime:` / `timings:` / `[debug] decoder` lines before
  diffing; compare the `[ t0 -> t1 ] p=… text` payload lines.
- `uv run scripts/validate.py all --family <f> [--variant <v>]` and
  `uv run scripts/preflight.py` — the repo's real numerical gates. **They
  need reference oracles that are not on this machine**, which is why this
  session leaned on transcript byte-diffing instead. If you can obtain the
  oracles, running these is worth more than anything I did.

**Benchmark methodology the user requires** (they were explicit, twice): run
4+ times, drop the first, average/min the rest, **interleave the arms within
each round**, prefer the library's stage counters over wall clock, and
**revert any "win" that lands inside the noise floor**. On this machine the
noise floor is large — see §3.

---

## 8. Suggested order of work

1. Read §4 so you don't undo the revert. Confirm
   `git diff 16f579b HEAD -- src/transcribe.cpp` is empty.
2. Read `AGENTS.md` and `CONTRIBUTING.md`.
3. Audit in descending risk order: **5.1 (thread detection, unexecuted
   platform code) → 5.2.1 (granite_nar, unverified) → 5.3 (decode
   correctness argument) → 5.5 (decide: keep or delete) → 5.4 → 5.6.**
4. For each: read the diff, reason about it, and either fix it or write down
   why it is fine. Prefer small, well-argued commits over one big one.
5. Re-run `scripts/ci/clang-format.sh --check` and both builds before
   committing. Do not push unless asked.

Write your own findings back into this file (or replace it) for whoever comes
after you. The most valuable thing in it is not the list of changes — it is
§4, the record of a confident, well-benchmarked change that was simply wrong
because it was measured against the wrong scenario.
