# Fork changelog

What the fork changed, by period and theme, newest first. Each entry names
the commits that carry the detail; `git show <sha>` has the measurements. Most
upstream merges are not listed (`git log --merges upstream/main..HEAD`); the
ones listed carry fork-side merge decisions. Overview: [README.md](README.md).

## 2026-11-28 — GPU decoder and streaming speed (Parakeet family, Nemotron, R2T2)

- The Parakeet device decoder (TDT and RNN-T loops, offline and streaming)
  computes an emitted token's confidence while the next joint step runs. That
  confidence is a softmax plus entropy over every token class (8193 for the
  multilingual models, ~150 us), and it used to run after each step's
  synchronize with the GPU idle. An event marks the end of the logits copy, so
  the host waits for that copy alone. Same values, off the critical path.
  Decode on CUDA, `dots`: Parakeet Ultra 125.5 -> 69.7 ms, Nemotron 3.5 Q8_0
  166.1 -> 104.8 ms, Parakeet v3 85.7 -> 61.5 ms (`jfk` 23-35% too); Vulkan
  Ultra 130 -> 92 ms. Every token's id, time and confidence is unchanged
  (8 clips x 3 models on CUDA, 4 clips x 2 on Vulkan).
- Cache-aware Parakeet streaming (Nemotron 3.5): the per-layer cache rotation
  after every encoder step (K, V and conv caches, 72 copies for 24 layers)
  is queued as device-to-device copies with one synchronize, instead of a
  synchronize per copy. A `dots` stream at 160 ms chunks makes 3601 -> 1329
  stream synchronizations; partials, final text and every token's time and
  confidence are unchanged (6 clips on CUDA, 3 on Vulkan).
- R2T2 streaming: a tick that prefills from scratch (every tick until the
  first encoder window completes, so every tick of an utterance under ~8 s)
  no longer clears the whole KV cache first. That clear wrote 235 MB
  (2048 positions x 28 layers) with the host waiting, about 0.9 ms on the
  RTX 4070 per tick; `jfk` at 160 ms went from 12.2 GB of memsets to the
  one 235 MB clear at allocation. Masked rows only have to be finite,
  which the KV-continuation path already relied on. Every partial,
  committed and final text is unchanged (9 clips at 80 ms, 160 ms and
  offline plus `dots` on CUDA; 3 clips on Vulkan; `jfk` on CPU).

## 2026-11-26 — audit follow-ups

Continues [audit-2026-11-25.md](audit-2026-11-25.md) (items 14-19 and the CUDA A/B there):

Commits: `3db20e2f` SAN-M direct conv, `90a6fc89` alphabet filter, `edc09283`
license texts, `709b0ca4` conv0 switches, `97820460` paused pools, `9bc76e39`
ggml patch 0104.

- SenseVoice and Fun-ASR-Nano on the CPU: a single utterance's SAN-M
  depthwise convolution takes the direct op, as batches already did, instead
  of im2col plus a matmul. Every dumped stage is byte-identical to the im2col
  path's (4 clips per model). Encode is 14-18% faster for SenseVoice and
  24-25% for Fun-ASR-Nano, whose compute buffer shrinks by a third. GPUs keep
  im2col: the direct op is faster there too, but not bit-identical, and on
  CUDA it changed 3 of 47 SenseVoice and 8 of 47 Fun-ASR-Nano transcripts
  (Vulkan: 0 and 1). `TRANSCRIBE_CONV_DIRECT_DW=1` takes it on a GPU anyway.
- Language allowlist: the alphabet filter judged a character cut at a token's
  end by the smallest code point it could become. Lead byte 0xD4 starts
  Cyrillic or Armenian letters and 0xD6 Armenian letters or Hebrew points, so
  an Armenian allowlist blocked the capitals U+0531-053F whenever the
  tokenizer split them, and a Hebrew one the vowel points. Such a prefix now
  passes, as a cut Latin letter already did.
- The native Python wheels, the npm platform packages and the Swift
  xcframework carry all eight license texts of `THIRD-PARTY-LICENSES.md`.
  They lacked cJSON, Earshot, RNNoise and (all but the default wheel)
  audio.cpp.
- Moonshine: both attention caches store K/V per head with the head size
  already padded (cross: written once per utterance; self: each row as its
  token decodes), and flash attention reads them through strided views
  instead of copies re-made and re-padded at every decoded token. Decode is
  2.3x faster on CPU and 1.6x on CUDA (`dots`: 451 -> 194 ms CPU, 181 -> 113
  ms CUDA). Every dumped tensor and all 37 test transcripts are
  byte-identical, single and batched, on both backends.
- Moonshine streaming: with no head padding to apply (head size 40), flash
  attention reads the K/V cache views directly instead of a copy made at
  every decoded token. Decode 12-20% faster on CPU and CUDA (`dots`: 386 ->
  308 ms CPU, 393 -> 324 ms CUDA); offline (batch 1 and 4) and streaming
  output byte-identical on both backends.
- MedASR single runs honor `n_threads`. Only its batch path applied the
  session's thread count; a single run kept the CPU backend's default, so
  encode took the same time at 1, 2 and 8 threads (now 5.3 / 2.5 / 1.4 s on
  `dots`). Transcripts unchanged (11 clips, at 4 and 8 threads).
- Canary on the CPU: the pre-encode's stride-2 depthwise convs take the
  direct op instead of im2col (13-17% of the CPU graph). Every dumped
  tensor is byte-identical on 3 clips; encode 7-12% faster (4 threads,
  machine partly loaded). GPUs keep im2col; CUDA transcripts unchanged.
- `TRANSCRIBE_CONV_DIRECT_CONV0` / `_NO_DIRECT_CONV0` now also apply to a
  causal Parakeet stem (Nemotron), which ignored them; its default is
  unchanged.
- ggml patch `0104`: the CPU barrier yields only when the threads outnumber
  the CPUs the process may use. Patch `0001`'s unconditional yield cost a
  time slice per graph node when other processes loaded the CPU (SenseVoice
  next to 12 busy processes: 21-28 s, now 3-6 s; unloaded runs and the
  2-CPU deadlock test unchanged). Found by the speech.cpp hand-off's
  audio.cpp measurements.
- CPU threadpools start paused. A GPU session's CPU fallback pool used to
  busy-poll about 7 cores for a few hundred ms after it was created, and
  that slowed the host-side mel of the next runs: Granite 5 CTC on CUDA ran
  2-6x slower than upstream on back-to-back runs. It now matches upstream,
  four CUDA runs use 1.5 s of CPU time instead of 7.9 s, and transcripts are
  unchanged.

CUDA A/B against upstream `a0729fa5`, 22 models x 11 clips: 186 of 215
comparable pairs identical. The 29 differences come from three
documented GPU-side changes (on-device attention, Granite's direct depthwise
conv, Nemotron's causal `conv0`), each with a switch that gives upstream's
text exactly, plus the fork's Qwen3 encoder and Whisper's random seed on
silence. Streaming matches upstream partial for partial.

## 2026-11-25 — second whole-fork audit

Record: [audit-2026-11-25.md](audit-2026-11-25.md), commit `dc9788ee`. A 21-model A/B against
upstream `a0729fa5`, clang-tidy over every fork-changed C/C++ file, and code
review. Fixes:

- Rust `build.rs` defines `TRANSCRIBE_MODEL_SET` / `TRANSCRIBE_MODELS` on
  every configure. Before, unsetting one kept the old set in the CMake cache,
  and the library built from it was cached under the new key. A family list on
  its own now selects `custom`. CMake warns about a list without `custom` and
  rejects an unknown family name.
- Plugin loader: `dlopen` takes the absolute path, as `LoadLibraryExW` does,
  so a bare name is not looked up on `LD_LIBRARY_PATH`. Paths follow the UTF-8
  contract on Windows.
- `transcribe_vad_predict_frame` / `_process_frame` refuse a non-finite
  frame: a NaN used to poison the detector's recurrent state.
- The `transcribe_get_whisper_chunk_trace` stub, in builds without Whisper,
  now follows the real accessor's contract.
- The public C++23 requirement is private; the public headers are C.
- Windows shared builds with Ninja and sccache: per-language compiler
  launchers keep ggml's global one from wrapping the resource compiler,
  which broke `transcribe-version.rc`.
- Tests: three unit tests are gated on the families they link, so plugin and
  minimal builds compile their test suites. The threadpool oversubscription
  test runs serially with a 60 s watchdog.
- Docs: the stream VAD telemetry and the frame VAD API are documented as they
  behave; CPU_REPACK is documented as not bit-exact, and its WER
  is measured (FLEURS-en, 150 utterances: no change).
  Also corrected: allowlist coverage, test counts, and the stale VAD-branch
  and master-plan items.

Follow-ups the same day:

- Test setup for builds without every model family: fixture tests gate on the
  new `TRANSCRIBE_LOADABLE_MODELS` (compiled-in families, or the plugins of an
  `ARCH_DL` build), `weights_mmap_smoke` runs the cases its families allow and
  names the rest, and the two tests that drive ggml state directly skip a
  shared core with embedded ggml. An `ARCH_DL` + minimal tree now passes
  52/52 (was 50/56), a static minimal tree 64/64.
- The two merged local branches (`feat/vad-frontend`,
  `codex/upstream-sync-2026-10-06`) were deleted.

Finding: upstream's Granite sizes its KV window from the longest clip a
session ran before, so its transcript of the same audio depends on session
history. The fork's per-run release removed that, which is why batch A/Bs show
Granite differences that vanish clip by clip. Also: ggml's CPU text is not
thread-count invariant for every graph (Granite 4.0 1B on `fleurs-fr`:
"semblait" at 8 threads, "semblerait" at 3), so A/Bs pin `--threads`.

## 2026-11-25 — upstream merge: Parakeet phrase boosting

Upstream through `cfa237c2`, three commits:

- Upstream #221: Parakeet's TDT and RNN-T decoders take
  `transcribe_run_params::vocabulary` as phrase boosting
  ([../prompting.md](../prompting.md#parakeet-vocabulary)), offline and in
  batches. Streaming and CTC heads ignore the terms; streaming says so with a
  WARN.
- Upstream #264: Windows shared builds find their DLLs under Cargo 1.100's
  build-dir layout.
- Upstream #262: ggml 1.0.0. The fork has vendored it since 2026-11-14 (same
  `417a024b`, same refreshed patch 0002), so `ggml/` and the patch series
  stayed as they were.

- Upstream `a0729fa5` (merged the same day): `TRANSCRIBE_USE_SYSTEM_BLAS`
  now defaults to OFF everywhere except Apple, which always uses
  Accelerate. This changes nothing on the fork's Windows builds, where
  CMake finds no BLAS.

Merge decisions on the fork side:

- The boost composes with the language allowlist. Both the model's own
  argmax and the re-ranked one go through `argmax_range_masked`, so a blocked
  locale tag stays unreachable after the re-rank.
- The frame-batched joint keeps reading its window (`frame_logits`), not
  the serial-only `logits` that upstream's line reads.
- The GPU decoder honours the vocabulary as well. From the automaton's root,
  the GPU's argmax is already the biased answer, because a term's first piece
  earns no bonus. So only a token emitted inside a partial match fetches its
  column's logits and is re-ranked on the host. That costs one extra
  synchronize per such token, and nothing when there are no terms.

Verification:

- Without terms, transcripts are byte-identical to the pre-merge build on
  CPU and CUDA. That covers Parakeet v3, Nemotron 3.5, Ultra, Redux TQ1 and
  CTC on 16 clips, plus Nemotron streaming at 160 ms.
- With `QuirkQuidQuill,RAPZ,Unifund` on `samples/product-names.wav`, both
  heads write the terms as given, and the GPU decoder's transcript equals the
  host decoder's on the same CUDA encoder output.
- One upstream caveat seen on CUDA: Nemotron wrote a late "Quill" as "Quir".
  The model started the term and the boost pulled the next piece, and then
  the model ended the word.

ZER0 effect: Parakeet models now report `Feature::Vocabulary`, so ZER0's
batch runs send the custom words to Parakeet, Nemotron, Ultra and Redux and
skip the fuzzy post-correction for those runs. Live streams still send no
vocabulary.

## 2026-11-18 — Parakeet decoder stays on the host for integrated GPUs

- The Parakeet TDT / RNN-T GPU decoder (`1f6df3c4`) no longer runs by
  default on an integrated GPU (ggml's `IGPU` device type), where it was
  slower than the host decoder (ZER0's A/B on Intel Iris Xe via Vulkan:
  Nemotron decode 107 → 158 ms, Parakeet Redux, v3 and Ultra 3-9 % slower in
  total). `TRANSCRIBE_PARAKEET_DEVICE_DECODER=1` still forces it. Discrete
  CUDA, ROCm and Vulkan GPUs are unchanged. Verified on the same laptop:
  Nemotron decode on the Iris Xe 156 → 92 ms, the RTX 4070 still on its GPU
  decoder, transcripts identical.

## 2026-11-17 — fork audit checkpoint

The whole-fork review in [audit-2026-11-17.md](audit-2026-11-17.md):

- The silence fast path judges energy per 16 ms window instead of over the
  caller's slice, so a word onset inside a long quiet slice can no longer
  be dropped.
- A NaN VAD threshold is rejected.
- Architecture plugins load only from host-controlled locations. The model's
  folder is opt-in (`TRANSCRIBE_ARCH_SEARCH_MODEL_DIR=1`), and the current
  directory is never searched.
- The Rust build cache keys on `TRANSCRIBE_MODELS`, drops its key-less
  fallback, and evicts old entries (`TRANSCRIBE_CACHE_KEEP`).
- Earshot, RNNoise, cJSON and audio.cpp attributions: license texts in
  `licenses/`, `THIRD-PARTY-LICENSES.md`, and installed with the library.
- `transcribe_vad_score_frames` no longer copies the whole clip.
- `vad_prefill_ms` is documented as what it is (reserved, no effect).
- The docs were reorganized under `docs/fork/`, and every runtime and build
  switch the fork reads is listed in `docs/environment-variables.md`.

## 2026-11-17 — GPU decode latency

- `1f6df3c4`: Parakeet's TDT / RNN-T predictor and joint run on the GPU.
  Decode is 2.7-5.2x faster on CUDA, and transcripts are byte-identical.
  - Conformer attention no longer falls back to the CPU on CUDA (Canary,
    Cohere, Sortformer, GigaAM).
  - Bit-exact host matvec layouts for the GigaAM, Parakeet CTC and Kaldi-fbank
    paths.
  - Streaming fixes: the silence fast path records skipped gaps and maps
    family time back to caller time, and text never splits a UTF-8 character.
  - The batched enc-dec loop uploads its cross mask every step.
- `3065929e`, `a13f2f45`: the Granite encoder is leaner. Its last CPU node
  (`proj.query`) is dequantized once at load, and Granite's decode timing now
  includes the token loop. `causal_lm::StepPipeline` overlaps greedy steps for
  Granite, Qwen3-ASR and R2T2, and sessions keep their pipeline buffers.
- `157c95c2`: merged upstream `adfc5021` (Rust log-callback panic guard).

## 2026-11-14 — upstream 0.4.2, ggml 1.0, language allowlist, audio front end

- `26764bf6`: merged upstream 0.4.1 → 0.4.2 (Silero VAD role API, mmap
  weights, long-form memory work). The fork keeps its Qwen3-ASR encoder,
  upstream's conformer pre-encode and a single weight loader; the reasons are
  in the [master plan](../plans/2026-11-fork-master-plan.md) §0.
- `f843ffcd`: ggml v1.0.0 with the patch series refreshed (`00xx` upstream,
  `01xx` fork).
- `5c1aa6b2`: dependency refresh.
- `2713c149`, `699021fb`, `e5bd2af0`: the language allowlist ("the languages
  I speak") on every multilingual family, plus `allowlist_letters`.
- `55504bde`: Whisper's CPU decode uploads its cross-attention mask every
  step.
- `abf9afef`: a UTF-8 code-page manifest for the Windows executables.
- `c4463aec`, `a50b3bf7`, `b6e0c294`: the SIMD Earshot engine (the network
  scores every frame), the capture-side speech gate, batch scoring, and native
  RNNoise with ZER0's denoise wrapper.

## 2026-10 — upstream syncs

- `01a4e9cf`, `468e8776`: ggml v0.26.0, and reproducible patch application on
  Windows.
- `692350de`: merged orukeet, parakeet attention, Rust serde and async TS.

## 2026-09-26 — Parakeet ultra/redux, R2T2 on CUDA

- `81022f38`, `0611d23f`: native ternary TQ1_G128 (1.75 bpw) and the
  parakeet-ultra / parakeet-redux ports.
- `c20c4bfa`: Parakeet graphs stay on the GPU, plus repacked CPU GEMM, ternary
  runtime layouts and an Android preset.
- `77895e09`: CUDA graphs on by default (+15 % R2T2 decode).
- `8e50b61f`, `145c96a6`: R2T2 decode ~1.29x faster: KV trim, MMVQ rows,
  packed QKV and encoder flash (both opt-in, drift-free default).
- `f2de3264`, `ba949120`: the silence fast path no longer drops speech on R2T2
  streams; R2T2 opts out of it and always runs its final pass.

## 2026-09-21 — Confucius4-R2T2

- `2d6d2271`, `09f554e0`: the R2T2 package runs natively, with streaming
  driven by a continuous chunk-size control.
- `47d8c5da`, `6b5f590b`, `ecfe327d`: a buffered backlog folds into one tick;
  a tick costs what it adds (encoder prefix and mel caches); the KV cache
  continues across ticks.
- `81c1afb0`, `d22a2718`: paired A/B bench runner, per-stage capture, shared
  reporter.
- `02022386`: allocation retry reclaims only the caller's own backend.

## 2026-09-09 → 09-20 — plugins, VAD, memory

- `f5cb30af` and follow-ups (`a33f7afe`, `ca662d28`, `74ea3ef8`, `30a8b643`,
  `574b0647`): architecture plugins (`TRANSCRIBE_ARCH_DL`), model-set presets
  and ABI stubs.
- `b796cf79`: CUDA pool-trim and graph-evict hooks (patch `0101`).
- `edaacaff`: the conformer `rel_shift` is materialized again; the diagonal
  view corrupted CUDA streaming.
- `8e413fef`: removed a double `pad_causal` in the conv2/conv5 pre-encode
  depthwise paths.
- `6c767184`, `cffa320b`: native Earshot VAD plus the stream activity gate.
- `4f150095`, `28a3f985`: removed the affinity pinning that livelocked the
  threadpool; idle STT workers are parked; streaming decoder graph cache.
- `3ae7383e`: P-core thread binding, since replaced by plain normal-priority
  pools.

## 2026-08 — memory, threads, build tooling

- `197833ed`, `c9bd43b2`: free GPU buffers after every transcription.
- `8fecde41`, `e90b0e75`, `ba0ad93c`, `1cb4a7ca`: thread default, frame-batched
  RNN-T joint, Granite direct depthwise conv. The Multi-STT device serializer
  was measured and reverted (`f0b1cba1`); the audit is in
  [records/2026-08-20-optimization-audit.md](records/2026-08-20-optimization-audit.md).
- `6f42c20e`, `911d3dd6`, `98e439f3`, `a5a2db3e`: persistent Rust build cache,
  build provenance, CUDA arch auto-probe, ccache, and Ninja with MSVC
  bootstrap.
