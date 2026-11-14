# Fork master plan — November 2026

Status: plan, 2026-11-14. Branch `codex/vad-integration` (to be merged into
`main`, then deleted). Supersedes the model choices in
`2026-10-three-tier-super-vad-and-firered-architecture-plan.md` where the two
disagree (see §B.0); every other October plan stays valid and is linked from
here.

Consumers this plan is written for:

- **ZER0** (`C:/Users/Z/Downloads/PROJECTS/Handy_V2`) — desktop dictation,
  Multi-STT (Nemotron 3.5 streaming primary + Granite 4.1 2B + Qwen3-ASR 1.7B
  + Canary 1B v2) on CUDA, tracks fork `main` automatically
  (`scripts/check-transcribe-deps.ts`).
- **FreeSpeech** (`C:/Users/Z/Downloads/PROJECTS/Android_FreeSpeech`) — Android
  arm64 keyboard/recognizer, `shared` + `dynamic-backends` + Vulkan, pinned to
  an exact fork revision (`ba949120`, 2026-09-26).

Sources read for this plan are listed in the [source ledger](#source-ledger).

---

## 0. Where the fork stands (2026-11-14)

Landed on `main` in this sync (three commits):

| Commit | What |
|---|---|
| `26764bf6` | Manual merge of upstream `handy-computer/transcribe.cpp` 0.4.1 → **0.4.2** (28 commits): Silero VAD role API (#201), memory-mapped weights (#195), long-form memory work (#199 conformer causal pre-encode without copies, #208 GigaAM windows, #222 moonshine-streaming bounded encoder + 81.9 s cap, #224 granite-NAR pooled BPE head), Phonon-2 (#213), NeMo padding-frame drop (#252), CBLAS link check, x86 XSAVE ISA scoring patch. |
| `f843ffcd` | ggml **v0.26.0 → v1.0.0** (`417a024b`) with the patch series refreshed. |
| `5c1aa6b2` | Dependency refresh: Rust, npm (TS nightly), Python build/test deps, all 23 `scripts/envs` lockfiles, CI action pins, `dr_wav.h` v0.14.7. |

Merge decisions that later work must respect (full list in the merge
commit message):

1. **Qwen3-ASR encoder stays the fork's**: windowed `cu_seqlens` attention
   (needed by the R2T2 encoder prefix cache), encoder flash **off** by default
   (it flipped 9/100 FLEURS transcripts on CUDA). Upstream #224's separate
   subsample graph and flash-on default were *not* taken — see §D.2.
2. **Conformer causal pre-encode is upstream's** (#199: symmetric p=2 +
   in-place tail zeroing, no pad copies). Nemotron offline and streaming
   transcripts were byte-identical before/after. This also removes the reason
   for the fork's `detect_direct_conv0_in_pre_encode` CUDA special case (§C.4).
3. **One weight loader**: upstream `WeightStorage`/`load_weights` (mmap where
   the device can use file bytes) + the fork's OOM reclaim retry inside
   `allocate_copied_weights` + whole-tensor staging for TQ1_G128 layout
   conversion and CPU_REPACK destinations (`src/transcribe-load-common.cpp`
   `stream_tensor_data`).
4. **VAD**: Silero = upstream role API (`include/transcribe/vad.h`,
   `src/transcribe-vad.*`, Rust `vad.rs`); the fork's Earshot moved to
   `src/transcribe-earshot.*` / Rust `earshot.rs` with its existing
   `transcribe_vad_init / predict_frame / process_frame` C API. Architecture
   plugin ABI 2 → 3.
5. **ggml patches**: upstream owns `00xx`, the fork owns `01xx`
   (`0101` CUDA pool trim + graph evict, `0102` TQ1_G128 ternary, `0103` CUDA
   decode KV-trim + MMVQ rows). Refresh notes for v1.0 are in §H.1.

Verification on this machine (Windows 11, MSVC 14.52, CPU): full build incl.
tests, CTest 66/66, Rust 28/28, ABI/bindgen/Swift drift gates, teardown lint,
pinned clang-format diff gate; transcript A/B vs the pre-merge build 23/25
byte-identical + Nemotron streaming 4/4 (the 2 diffs are #252 punctuation on a
35 s clip). ggml 1.0, CUDA and Android results: §0.1.

### 0.1 Results recorded for the ggml 1.0 bump

CPU (Windows, MSVC 14.52): CTest 66/66 (incl. `ternary_tq1_g128_unit`,
`threadpool_oversubscription`), Rust 28/28. Transcript A/B against the
pre-merge build: 25 offline cases + 4 Nemotron streaming cases; beyond the two
known #252 differences, ggml 1.0 changed exactly two punctuation marks —
Qwen3-ASR on `dots.wav` (`whatever—because` → `whatever. Because`) and
Nemotron streaming on `german.wav` (`Meer` → `Meer,`). That is rounding-level
drift from ggml's reworked CPU kernels, not a regression; it is recorded so a
later bisect over 0.27/0.28/0.29/1.0 has a starting point if it ever matters.

CUDA (sm_89, RTX 4070 Laptop, CUDA 13.5) vs the pre-merge CUDA build: Nemotron
offline and streaming, Granite 4.1 2B and parakeet-redux byte-identical;
Qwen3-ASR changed one comma on two clips (deterministic run to run). The ggml
1.0 bump moves Granite's GLU sigmoid back to the GPU (§C.1): graph splits
38 → 6, CPU sigmoids 16 → 0, **encode 2.05–2.13× faster** (461.7 → 216.6 ms
on `german.wav`, 497.1 → 241.9 ms on the 35 s `dots.wav`; interleaved, 4
rounds, first dropped). Nemotron 3.5 and Qwen3-ASR timings within noise.
Parakeet v3 encode is ~7 % slower on the 35 s clip — see §D.4, it is
upstream's long-audio attention policy, not ggml.

Android: the `android-arm64` preset (NDK r30) cross-compiles all seven CPU
ISA variants (armv8.0 … armv9.2), Vulkan and `libtranscribe` against ggml
1.0. (The Vulkan shader generator is a host-MSVC sub-build: run the build
from a VS developer environment, and delete
`build-android-arm64/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix` after an
MSVC toolset update.)

---

## 1. Ground rules for every workstream

- **Drift-free defaults.** A speed or memory change ships on by default only if
  transcripts stay byte-identical on the A/B set (`--timestamps token`, text
  from `-o`, never merged stdout/stderr); otherwise it is opt-in until a WER
  run justifies it. Accuracy changes (like #252) are called out in the commit.
- **Measure properly**: 4 runs, drop the first, interleave arms,
  `scripts/bench/compare.py` for any cross-arm claim; stage timers over wall
  clock.
- **ABI is append-only.** New run/stream fields go at the end of their struct
  with `struct_size` gating; regenerate Python/TS/Rust/Swift in the same commit.
  When upstream has an open PR for the same field (e.g. #168), use its exact
  names and semantics so the next upstream merge is a no-op.
- **Never mix** dependency bumps, reformatting or ggml syncs into a behavior
  commit.
- **Consumer check** for anything ZER0/FreeSpeech call: grep the Rust call
  sites in both apps before changing a default (`docs/zer0-engine-integration.md`).

---

## A. Constrained automatic language detection — everywhere it fits

Upstream PR #168 (open) adds a language **allowlist** for Nemotron automatic
detection: the RNN-T decoder masks the language-tag tokens that are not in the
list at every selection step, including later streaming steps; an empty list
changes nothing; an explicit `language` wins. This plan generalises that idea
to every family that can honour it, with one public contract.

Why it matters for ZER0/FreeSpeech: a dictation user speaks a known small set
of languages. Unconstrained detection is where the worst failures come from —
a French sentence decoded as Spanish/Portuguese/Catalan, short German phrases
as Dutch, Russian vs Ukrainian, Japanese vs Chinese. Forcing a single
`language` hint is the other failure: ZER0's Multi-STT "empty second column"
trap (a hint one model cannot honour) and code-switching users. An allowlist
removes the first class of errors without forcing the second.

### A.0 Public contract

- C ABI: append to `struct transcribe_run_params` (after `prefix`), exactly as
  #168 names them:

  ```c
  int32_t              n_allowed_languages;
  const char * const * allowed_languages;   /* borrowed for the call; copied for a stream */
  ```

  Semantics (identical to #168): empty list = unrestricted (**byte-identical**
  to today); ignored when `language` is set; unknown code →
  `TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE`; NULL list with n>0 or a NULL/empty
  element → `TRANSCRIBE_ERR_INVALID_ARG`; duplicates count once; streams copy
  the list at `transcribe_stream_begin` and apply it to every step.
- Discovery: new `TRANSCRIBE_FEATURE_LANGUAGE_ALLOWLIST` for
  `transcribe_model_supports()` (no capability-struct churn). Families that
  cannot honour it return `TRANSCRIBE_ERR_UNSUPPORTED` when a non-empty list is
  passed — never silently ignore it.
- Result: `transcribe_detected_language()` already exists; add
  `allowed_mass` (share of the unrestricted language distribution inside the
  allowlist, the same meaning LangID already reports) where the family can
  compute it, so hosts can flag "speech probably outside the allowed set".
- Bindings: Rust `RunOptions.allowed_languages: Vec<String>` and the same on
  `StreamOptions` (keep the CStrings alive for the stream, as #168's
  `build_run_params` does); Python/TS/Swift regenerated.

### A.1 One code normaliser for hints *and* allowlists

`src/transcribe-lang.{h,cpp}` (new): trims, lower-cases, maps `_`→`-`, reduces
regional tags to the primary subtag unless the model advertises the regional
one (`zh-CN`/`zh_cn`/`ZH-ch` → `zh`, `pt-BR` → `pt`), and applies the alias
table models actually disagree on (`he`/`iw`, `jv`/`jw`, `nb`/`no`, `fil`/`tl`,
`yue` vs `zh`). Every family's `language` resolution and the allowlist go
through it. Source of truth for the regional/alias rules: NVIDIA
NeMo-Speech.cpp `salm_policy.cpp` `language_name()` as adopted in audio.cpp
#1190. Unit-tested table, no model needed.

### A.2 Family matrix

| Family (fork) | How it picks the language today | Constraint mechanism | Streaming | Size |
|---|---|---|---|---|
| **parakeet: Nemotron 3.5 streaming** (+ any variant with locale-tag tokens) | RNN-T emits a locale tag token | **#168 token mask** at every argmax: serial greedy, the fork's frame-batched joint window (`JointGraphBatch`, W=16 — mask the per-frame argmax inside the window), and the streaming emit path | yes, every step | S |
| **whisper** (multilingual) | detect pass: argmax over `lang_token_ids` (`src/arch/whisper/model.cpp` ~1783) | restrict the argmax/softmax to the allowed ids; report `allowed_mass` | per 30 s chunk | S |
| **sensevoice** | language code token at the LID query position | mask non-allowed language tokens at that position | n/a | S |
| **qwen3_asr** incl. **R2T2** | generates `language {Name}<asr_text>` | constrained prefix: score each allowed `language X<asr_text>` continuation (one batched teacher-forced pass, or a trie-constrained greedy over the name tokens), then decode after the forced prefix. R2T2: decide on the first committed tick ≥ N ms, then lock for the stream (opt-in re-decide at a VAD boundary) | lock-after-decide | M |
| **voxtral** | optional `lang:xx` prompt before `[TRANSCRIBE]` (`build_transcription_prompt`); auto = omitted, nothing emitted | **LangID-gated** (A.3): pick the best allowed language, inject it as the model's own `lang:xx` conditioning | — | S |
| **voxtral_realtime** | same family of prompt conditioning | A.3, decided once at stream start (re-decide at a VAD boundary, opt-in) | lock | S |
| **funasr_nano** (MLT) | optional `语音转写成<lang>` prompt; auto = omitted | A.3 → forced-language prompt | — | S |
| **parakeet-tdt v3 / primeline / ultra / redux** | implicit — no language tag in the output stream (verify by dumping raw token ids on the FLEURS clips; if tags do appear, they take the #168 mask) | otherwise not maskable → **LangID** (A.3) only to *report*/warn; they need no hint | — | S |
| **canary / cohere** | need an explicit source language | **LangID-gated auto**: pick the best allowed language with the LangID role, pass it as `language` | per segment | M |
| **granite** (4.x multilingual) | prompt-driven, language implicit | A.3 to report/warn; inject a language hint only if the chat template has one | — | S |
| canary_qwen | English only | n/a | — | XS |
| moonshine (per-language), gigaam, medasr, granite5_ctc | single language | n/a (`UNSUPPORTED` only if the list excludes their language) | — | XS |

### A.3 LangID-gated auto mode (engine orchestration)

Upstream already shipped the LangID role (`include/transcribe/langid.h`,
`ecapa_tdnn` arch) and its params already take an **allowed set** and report
`allowed_mass`. Add an opt-in session-level attachment
(`transcribe_session_set_langid(session, langid_model)`, append-only API) so an
ASR session with `language == NULL` and a non-empty allowlist, whose family
cannot detect natively (canary, cohere, voxtral, funasr_nano), runs the attached LangID on
the first ≤ 30 s (or the first VAD segment) restricted to the allowlist and
proceeds as if that language had been passed. This makes "auto" exist for
families that never had it, and makes Canary 1B v2 in ZER0's Multi-STT stop
needing a hard-coded hint.

### A.4 Rollout order and gates

1. A.0 + A.1 + Nemotron (port #168 onto the fork, including the batched-joint
   window) + Rust binding → ZER0 can ship "languages I speak" for the primary.
2. Whisper, SenseVoice.
3. LangID-gated auto (Canary, Cohere, Granite).
4. Qwen3-ASR / R2T2 constrained prefix.

Gates, per family: (a) empty list → byte-identical transcripts on the A/B set;
(b) `parakeet_language_mask_unit`-style unit test of the masked argmax;
(c) FLEURS local clips (`samples/fleurs-{de,en,es,fr,id,ja,ru,zh}.wav`) with a
two-language allowlist that contains the spoken language: zero wrong-language
outputs, WER not worse than the forced-language run; (d) out-of-set audio
behaviour documented (it is forced into the nearest allowed language; hosts
read `allowed_mass`).

---

## B. VAD

### B.0 What changed since the October plan

`main` now carries **two** VAD engines, both built:

- **Silero v6.2** through upstream's VAD role API (`transcribe_vad_session_*`,
  model file, segments + probabilities, streaming feed/flush).
- **Earshot** (fork) through the embedded frame API
  (`transcribe_vad_init / predict_frame / process_frame`, bundled weights).

The October three-tier plan premised "FireRed Stream-VAD replaces Silero
(97.57 % F1)". audio.cpp has since measured all three on the same footing
(TEN VAD test set, 30 labelled clips, frame F1 at each model's defaults,
#1095): **Silero 0.935, Earshot 0.926, MarbleNet 0.919, FireRed 0.917,
PulseVAD 0.914** — and on 43.7 min of audio Earshot runs at 5,421× real time
(4 threads) vs FireRed 392× and Silero 134×. So: Earshot stays the always-on
tier, Silero stays the accurate offline/segmenter option, and FireRed becomes
an *evaluated option*, not the plan's default (§B.5).

### B.1 Earshot fidelity (audio.cpp #1095)

Port the segmenter fixes into `src/transcribe-earshot.cpp`:
- keep a segment still open when audio ends (`keep_trailing_segment`, default
  true — upstream Earshot drops the last words of an utterance);
- count every frame (upstream's iterator skips its counter on frames that emit
  a segment, drifting later segments 16 ms earlier per segment);
- the streaming `SpeechStart` fires on the same rule `finish()` uses (a stream
  ending exactly at the minimum duration produced an end without a start);
- Silero-style option names (`threshold`, `neg_threshold`,
  `min_speech_duration_ms`, `min_silence_duration_ms`, `speech_pad_ms`) plus
  `smoothing_factor`, `start_delay_ms`.
Gate: parity with the pinned Rust reference (`pykeio/earshot@ffd32d4`, max
per-frame score diff ≤ 1e-4) and the frame-30..50 stream-end sweep test.

### B.2 Earshot as a VAD-role model

Register Earshot as a VAD-role architecture with bundled weights (no file to
load), so `transcribe_vad_session_init` serves both engines and hosts use one
API. Keep the frame API as a thin compatibility layer for ZER0 until it
migrates (§F.3).

### B.3 Earshot speed: per-ISA kernels (audio.cpp #1102)

- Kernel table compiled three times (baseline SSE2/NEON/scalar, AVX2,
  AVX-512 F+BW+VNNI), picked at runtime by cpuid, overridable for tests.
- minGRU matrix products batched across frames offline as an int16 GEMM
  (`vpdpwssd` / `pmaddwd`; NEON `smlal`), scan split across threads in groups
  of 8 hidden units.
- FFT: bit-reversal fused with the first radix-4 stage, paired stages, h=4/8 in
  registers, vectorised split + power.
- **Deterministic math** (Cephes `log`/`exp`, fixed mel summation order):
  bit-identical scores across ISAs, compilers, thread counts and streaming vs
  offline. Test hashes one clip's scores against a committed value on every
  kernel set CI can run.
Reported: 3–5× faster single-thread, 7–8× threaded; NEON baseline ~10,600×
real time single-thread. This is the Android battery item (§E.3).

### B.4 Any VAD family for long-form chunking (audio.cpp #1096)

`src/transcribe-chunking.*` plans long-audio chunks at silences. Let the chunk
planner take any VAD-role model (Silero, Earshot, later FireRed) through one
`load_vad_model`-style resolver, defaulting to the current behaviour.

### B.5 FireRed VAD / Stream-VAD — evaluate, don't assume (audio.cpp #846)

Port as an opt-in VAD-role family only if it wins on *our* data: DFSMN,
Kaldi fbank with an opt-in sparse mel projection, causal Stream-VAD with
explicit cache reset; F32 only (F16/Q8 gave no speed/VRAM benefit and Q8
drifted). Acceptance: beats Earshot on a ZER0 dictation set on onset clipping
and hangover, not on frame F1 alone.

### B.6 Benchmark harness

Extend `scripts/vad/bench.py` with the TEN VAD test-set frame-F1 table
(10 ms resolution, defaults per model) so the fork reproduces audio.cpp's
numbers, plus CPU-ms per hour of audio per ISA.

### B.7 The rest of the three-tier plan

Smart Turn v3.2, the lexical trailing guard and the adaptive hangover curve
remain as written in the October plan, after B.1–B.3. The paired-project
contract (`docs/plans/2026-10-audio-frontend-contract.md`) and the VoxLock
review stay authoritative for frontend ordering.

---

## C. CUDA: keep the encoders on the GPU, and keep them deterministic

ZER0 runs four models concurrently on one 8 GB RTX 4070; every op that falls
back to the CPU is a host round-trip per block. Measure with
`GGML_SCHED_DEBUG=2` and count `## SPLIT` lines and CPU-assigned nodes.

### C.1 GLU sigmoid on a contiguous input (upstream PR #228, open)

ggml-cuda only runs unary ops on contiguous tensors; the GLU's value half is a
strided view, so every SIGMOID goes to the CPU. #228 measured graph splits
196 → 100 (Cohere) and 97 → 49 (Parakeet v3), encode 1.12–1.17× faster on
upstream code (this fork's Parakeet path was already contiguous; its Granite
encoders were not — §0.1).
**ggml 1.0 removed that restriction**: `ggml_backend_cuda_device_supports_op`
no longer requires a contiguous source for unary ops (the old
`return ggml_is_contiguous(op->src[0])` and its "should become
contiguous_rows" TODO are gone). So the bump itself keeps the GLU sigmoid on
the GPU for every conformer family; see §0.1 for the measured split counts.
Remaining value of #228 is only the shared `glu()` helper (deduplicating four
copies); take it only if the contiguous copy measures neutral on CUDA,
Vulkan and Metal.

### C.2 Per-head relative-position mask vs CUDA flash attention (upstream PR #233, open)

ggml-cuda flash attention rejects masks with `ne[2] != 1` (still true in ggml
1.0, `fattn.cu`), so per-head rel-pos masks push FLASH_ATTN_EXT to the CPU.
The fork already disables encoder flash for Parakeet where the backend can't
fuse the mask (`conf::flash_supports_rel_pos_mask`); extend the same policy to
Cohere, Canary and Sortformer, resolved once at load as #233 does, and unify
the fork's helper with #233's `per_head_mask_supported(BackendKind)`.

### C.3 Reduction race and concurrency determinism (audio.cpp #837)

llama.cpp #26385 fixed a shared-memory race in kernels that call
`block_reduce` twice (softmax max+sum, group norm): two sessions in one
process could produce different bytes. **Verified present in the vendored
ggml 1.0** (`softmax.cu` and `norm.cu` carry the `see #26385` barriers and
separate buffers), so no patch is needed. What remains is the guard: a test
that runs two sessions concurrently in one process and requires byte-identical
output to each run alone — that is ZER0's Multi-STT shape.

### C.4 Re-measure after the pre-encode change

With upstream's copy-free causal pre-encode, conv0's input is contiguous on
CUDA again; re-benchmark `direct_conv0_in_pre_encode` on CUDA and drop the
fork's CPU-only special case if direct wins.

---

## D. Memory and long-form

### D.1 Memory-mapped weights (upstream #195, now on)

CPU and Metal map GGUF bytes instead of copying them; CUDA/Vulkan discrete
still copy. For ZER0 this changes CPU RSS accounting (shared, evictable pages)
and makes `no_mmap` a knob worth exposing in the Rust options. For Android
(§E.6) it is the biggest RSS lever under the low-memory killer.

### D.2 Qwen3-ASR memory without losing R2T2

Upstream #224 bounds Qwen3 memory with a separate conv-subsample graph run in
passes of ≤ 512 chunks and encoder flash on by default. The fork kept its own
bounded subsample (32-chunk groups) and windowed attention. Next steps:
(a) port the multi-pass subsample *structure* onto the fork's windowed encoder
(no numeric change expected; gate byte-identical); (b) evaluate flash only
above a T_enc threshold where the score matrix dominates (> ~2 min), keeping
short dictation drift-free; (c) re-read the R2T2 CUDA optimization record in
`docs/` before touching the encoder prefix cache.

### D.3 Parakeet graph hygiene (audio.cpp #1155)

Upload the common attention key-mask row and expand it on the backend instead
of filling a square host matrix; size encoder graph metadata by layer count;
release one-shot producer workspaces after use. Byte-identical gate.

### D.4 Parakeet long-audio attention window (upstream #199, now on)

Full-attention Parakeet checkpoints (`att_context -1/-1`, e.g. TDT 0.6B v3)
now run NeMo's documented long-audio window `[256, 256]` by default
(`src/arch/parakeet/weights.cpp`). Up to 256 encoder frames (20.5 s) the band
covers everything and the graph is the full-attention one (bit-identical);
above ~46 s the bounded-window graph keeps time and scratch linear. In
between it takes the dense local graph: on the 35 s `dots.wav` that costs
~7 % encode time on CUDA (193 → 207 ms) and changes attention numerics.
For ZER0 (dictation mostly < 20 s) nothing changes; for longer dictation,
measure the 20–46 s band and consider either lowering the windowed-graph
threshold for Regular-local runs or a session option to keep full attention
up to a length cap.

---

## E. Android ARM (FreeSpeech)

### E.1 Move FreeSpeech to the new engine

FreeSpeech pins `ba949120`. After this sync is validated on device, bump the
pin to the new `main` SHA (one line in `Android_FreeSpeech/Cargo.toml`), run the
Gradle build with the `android-arm64` preset settings, and smoke-test CPU and
Vulkan on at least one Snapdragon (Adreno) and one Exynos/Tensor (Mali/Xclipse)
device. Watch: plugin ABI 3, mmap weights, the #252 and #199 output changes.

### E.2 ggml 1.0 CPU variants on ARM

The preset builds every Android CPU variant (`GGML_CPU_ALL_VARIANTS`) and
loads the best at runtime. Since v0.26, ggml added SVE kernels (v0.29) and
KleidiAI SME2 k-quant matmuls (v0.27). Measure enabling `GGML_CPU_KLEIDIAI`
for the Android preset on an SME2/i8mm phone; confirm the TQ1_G128 → Q4_0
runtime layout still lands on the repacked dotprod/i8mm GEMM (patch `0102`).

### E.3 Always-on VAD on NEON

B.3's NEON kernel set + B.2's role API give FreeSpeech the same Earshot gate
ZER0 has, with no Rust audio stack.

### E.4 Threads on big.LITTLE

`performance_cpu_count()` (fork) reads `cpu_capacity`/`cpu_core` on Linux;
validate it on Android (sysfs layout varies by vendor) and A/B 4/6/8 threads
for Nemotron and Parakeet v3.

### E.5 Cross-compile lane

Add a CI job that only cross-compiles the `android-arm64` preset (no device),
so ggml syncs can't silently break the Android build.

### E.6 mmap on Android

Weights mapped from app-private storage (not the APK) cut RSS and survive
memory pressure better; measure peak RSS for Nemotron Q8_0 with and without
`no_mmap`.

---

## F. ZER0 (Handy_V2) integration

### F.1 What this sync changes for ZER0

- Rust: the fork's Earshot wrapper is now `transcribe_cpp::earshot`;
  `transcribe_cpp::vad` is Silero's `VadSession`. ZER0 uses the `earshot`
  crate directly today, so nothing breaks; the native path is §F.4.
- Weights are memory-mapped on CPU paths; CUDA unchanged.
- Parakeet/Canary offline transcripts can differ slightly (#252 padding frame),
  and full-attention Parakeet checkpoints use local attention above 20.5 s
  (#199, §D.4).
- Dynamic architecture plugins must be rebuilt (plugin ABI 3); ZER0 compiles
  architectures in, so this does not apply to its default build.

### F.2 Allowlist in the app

Expose A.0 as a "languages I speak" setting and pass it to all Multi-STT
models (Nemotron natively; Canary through A.3). This replaces the single
language hint that caused the empty-column trap.

### F.3 One VAD owner

Today the Rust `earshot` crate gates audio and the engine never sees silence.
Once B.1–B.2 are bit-identical to the crate, move gating into the engine
(as the October plan intends) so desktop and Android share one state machine.

---

## G. Decoder robustness

- **Repetition guard**: the fork's guard covers canary, canary_qwen, cohere,
  funasr_nano, granite, moonshine, moonshine_streaming, moss, qwen3_asr,
  voxtral. Evaluate NeMo-Speech.cpp LoopGuard's *replay* rule (sustained replay
  of an earlier emitted span, audio.cpp #1198) as a second detector, and cover
  whisper and voxtral_realtime.
- **Regional language hints**: §A.1 fixes `zh-CN`-style hints for every
  family, not only Nemotron.

---

## H. Housekeeping

### H.1 ggml patch maintenance (notes from the v1.0 refresh)

- `0002-backend-reg-filter`: context moved in `ggml_backend_load_all_from_path`.
- `0101`: v1.0 added `invalidate_cuda_graphs()` to the legacy pool; the fork's
  exported `clear()` now syncs, frees and invalidates like the pool's own OOM
  path, and `ggml_backend_cuda_trim_pools()` drops the backend's captured graphs
  after trimming (a replayed graph skips the ops that would re-map a trimmed
  VMM tail).
- `0102`: `GGML_TYPE_F8_E4M3` took id 43; TQ1_G128 stays parked at 96. ggml
  now ships its own AVX2 `Q2_0` dot product, so the patch's duplicate was
  dropped. Re-added the TQ1_G128 MMVQ case, the coopmat2 dequant functions
  (new `readonly` block style) and the backend-ops test entry.
- `0103`: `K->ne[1]` became `n_kv_cols` in the FA mask-scan condition.

### H.2 Pending local items

- `cmake/transcribe_utf8.manifest` + its CMake wiring (UTF-8 process code page
  for the Windows executables) is still uncommitted local work; finish it as
  its own commit.
- `main`'s working tree also holds two untracked drafts: an older v6.0.0 copy
  of the three-tier VAD plan (superseded by revision 9 on this branch — delete
  it before merging, or the merge refuses to overwrite it) and the Nemotron-3
  diarization port draft (separate work, not part of this plan).

---

## Milestones

| | Contents | Exit gate |
|---|---|---|
| **M0** (done) | upstream 0.4.2 merge, ggml 1.0, deps | ctest, Rust, A/B transcripts |
| **M1** | A.0, A.1, Nemotron allowlist, Rust/ZER0 setting | byte-identical with empty list; FLEURS allowlist gate |
| **M2** | B.1–B.3 Earshot fidelity, role API, per-ISA kernels | Rust-reference parity, kernel hash |
| **M3** | C.1–C.3 CUDA residency + determinism test | split counts down, Multi-STT bytes stable |
| **M4** | A for Whisper, SenseVoice, LangID-gated auto, Qwen3/R2T2 | per-family gates |
| **M5** | E.1–E.6 Android refresh | device smoke, RSS, thread A/B |
| **M6** | B.5 FireRed evaluation, D.2–D.4, G | data-driven |

---

## Source ledger

| Source | State (2026-11-14) | Used in |
|---|---|---|
| [handy-computer/transcribe.cpp#168](https://github.com/handy-computer/transcribe.cpp/pull/168) constrain Nemotron automatic language detection | open | A (core design) |
| [handy-computer/transcribe.cpp#228](https://github.com/handy-computer/transcribe.cpp/pull/228) GLU sigmoid contiguous on CUDA | open | C.1 |
| [handy-computer/transcribe.cpp#233](https://github.com/handy-computer/transcribe.cpp/pull/233) conformer flash per-head mask on CUDA/ROCm | open | C.2 |
| [handy-computer/transcribe.cpp#162](https://github.com/handy-computer/transcribe.cpp/pull/162) reduce memory usage across models | open, superseded upstream by #195/#199/#208/#222/#224 (merged) | D (fork import resolved in the merge) |
| [0xShug0/audio.cpp#1095](https://github.com/0xShug0/audio.cpp/pull/1095) Earshot VAD native port | merged (v0.9.3) | B.0, B.1, B.6 |
| [0xShug0/audio.cpp#1102](https://github.com/0xShug0/audio.cpp/pull/1102) Earshot per-ISA SIMD kernels | merged (v0.9.4) | B.3, E.3 |
| [0xShug0/audio.cpp#846](https://github.com/0xShug0/audio.cpp/pull/846) FireRed VAD + Stream-VAD | merged (v0.9.2) | B.5 |
| [0xShug0/audio.cpp#837](https://github.com/0xShug0/audio.cpp/pull/837) CUDA block_reduce race backport | merged (v0.9.2); fix already in ggml 1.0 | C.3 |
| [audio.cpp releases (VAD)](https://github.com/0xShug0/audio.cpp/releases?q=VAD): #1096 any-VAD chunking, #1190 regional language hints, #1198 repetition guard, #1155 Parakeet lifecycle cleanup, #962 NEON library | merged | B.4, A.1, G, D.3, E |
