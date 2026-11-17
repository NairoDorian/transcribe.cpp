# Fork changelog

What the fork changed, by period and theme, newest first. Each entry names
the commits that carry the detail; `git show <sha>` has the measurements. Most
upstream merges are not listed (`git log --merges upstream/main..HEAD`); the
ones listed carry fork-side merge decisions. Overview: [README.md](README.md).

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
