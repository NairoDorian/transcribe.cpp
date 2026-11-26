# Handoff: what the retired speech.cpp leaves for this fork (2026-11-25)

> **For the agent working on this fork.** Written by the session that retired speech.cpp, at the user's request. Nothing else in this repository was touched: this is one new file. Every item says how to verify it before adopting it. The source archive (`../Unified_Audio.cpp/_archive/`) is about to be deleted, so the code worth keeping is pasted below.

## Context, in five lines

- **speech.cpp is retired.** It tried to merge this fork and audio.cpp into one C++ engine.
- **ZER0 (`../Handy_V2`) becomes FreeSpeech.** It keeps `transcribe-cpp` for speech-to-text and adds text-to-speech through a new crate, `../audiocpp-rs` (github.com/NairoDorian/audiocpp-rs). That crate builds audio.cpp v0.9.6 into a shared `audiocpp.dll` that keeps its ggml private. See `../Handy_V2/docs/2026-11-tts-audiocpp-and-freespeech.md`.
- **Coexistence rule for ZER0:** at most one *static* ggml per binary. `transcribe-cpp` may stay static; `audiocpp` is always shared.
- **audio.cpp has a fork too**, `../audiocpp-fork` (github.com/NairoDorian/audio.cpp), for pull requests to 0xShug0/audio.cpp. The two forks are separate workstreams.
- **This machine is shared by several agent sessions.** On 2026-11-25 a benchmark here and audio.cpp builds there disturbed each other: the same synthesis varied 20×. Check the CPU load before quoting any timing, and say so when it was not idle.

## Items, in priority order

### 1. SANM: the direct depthwise conv for single utterances (performance; small)

`src/sanm/sanm.cpp` `fsmn_branch()` uses `ggml_conv_2d_dw_direct` only when `B > 1`. The single-utterance case (`B == 1`, the common one for SenseVoice and FunASR-nano) still goes through `conf::conv_1d_dw_f32`, the im2col path, which builds a `[kernel, T, d_model]` scratch: 11× the activation for the default SAN-M kernel of 11. Every other encoder in the fork (parakeet, canary, canary_qwen, cohere, granite, granite_nar, gigaam, medasr, sortformer) already takes the direct path behind the `TRANSCRIBE_CONV_NO_DIRECT_DW` kill switch.

speech.cpp made this change on its copy of the runtime (commit `27e0962b`, 2026-08-20). The commit message claims a speed and memory win, but **it was never measured**. The diff, against code that is still identical here:

```diff
@@ fsmn_branch(...)
     ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v_in));  // [T, d_model, B]
 
+    // Direct single-op depthwise via ggml_conv_2d_dw_direct (H=1) for the
+    // batched case (B>1) and, by default, the single-shot case (B==1). For
+    // B==1 this avoids conv_1d_dw_f32's im2col path, which materializes a
+    // [kernel, T, d_model] scratch (11x for the default SAN-M kernel=11).
+    // TRANSCRIBE_CONV_NO_DIRECT_DW=1 opts back into the im2col path.
+    static const bool conv_dw_direct =
+        conf::resolve_conv_direct("TRANSCRIBE_CONV_DIRECT_DW",
+                                  "TRANSCRIBE_CONV_NO_DIRECT_DW",
+                                  /*backend_default=*/true);
     ggml_tensor * fsmn;
-    if (B > 1) {
-        ggml_tensor * knl  = ggml_reshape_4d(ctx, fsmn_w, kernel, 1, 1, d_model);
+    if (B > 1 || conv_dw_direct) {
+        // Cast to F32 for ggml_conv_2d_dw_direct (it misbehaves for non-f32
+        // kernels; the depthwise kernel is tiny so the cast is negligible).
+        ggml_tensor * knl  = fsmn_w;
+        if (knl->type != GGML_TYPE_F32) {
+            knl = ggml_cast(ctx, knl, GGML_TYPE_F32);
+        }
+        knl                = ggml_reshape_4d(ctx, knl, kernel, 1, 1, d_model);
         ggml_tensor * data = ggml_reshape_4d(ctx, v_t, v_t->ne[0], 1, d_model, B);
         fsmn               = ggml_conv_2d_dw_direct(ctx, knl, data, 1, 1, padding, 0, 1, 1);
         fsmn               = ggml_reshape_3d(ctx, fsmn, fsmn->ne[0], d_model, B);
     } else {
-        // Single-shot: im2col path.
+        // im2col fallback (TRANSCRIBE_CONV_NO_DIRECT_DW=1).
         fsmn = conf::conv_1d_dw_f32(ctx, fsmn_w, v_t, /*stride=*/1, /*padding=*/padding, /*dilation=*/1);
```

`conf::resolve_conv_direct` is declared in `src/conformer/conformer.h`, which `sanm.cpp` already includes.

**Before adopting:**
- **Parity:** identical transcripts on the SenseVoice and FunASR-nano fixtures, the CJK FLEURS clips included (`samples/fleurs-zh.wav`, `fleurs-ja.wav`), and `validate.py` within tolerance.
- **Speed and memory:** the fork's benchmark protocol (≥ 3 runs, run 1 = warm-up, mean of runs 2..N), with RTF and peak memory, on CPU first, then Vulkan and CUDA.
- **Kill switch:** keep `TRANSCRIBE_CONV_NO_DIRECT_DW` working.

### 2. SANM batched path: a non-F32 kernel into `ggml_conv_2d_dw_direct` (possible bug; verify)

The same change casts the kernel to F32 first, with the comment "it misbehaves for non-f32 kernels". The fork's existing batched path (`B > 1`) passes `fsmn_w` uncast. If a SenseVoice or FunASR-nano GGUF stores `fsmn_w` as F16, batched runs may be wrong.

**To check:**
1. Inspect `fsmn_w`'s type in the published GGUFs.
2. If any is not F32, compare `transcribe_run_batch` against single runs on the same clips.

If they differ, the cast above fixes both paths. If every published GGUF keeps F32, the cast is still a cheap guard.

### 3. Windows shared builds fail when sccache is on PATH (bug; small)

**What happens:** ggml's `GGML_CCACHE` option (default ON, `ggml/CMakeLists.txt:125`) installs a *global* compiler launcher. That launcher also wraps the Windows resource compiler, and `src/CMakeLists.txt:674` compiles `transcribe-version.rc` for shared builds (`TRANSCRIBE_BUILD_SHARED` or `TRANSCRIBE_SHARED_EMBED`) on MSVC.

**Who it hits:** sccache cannot run the RC step:

```
FAILED: src/CMakeFiles/transcribe.dir/__/generated/transcribe-version.rc.res
sccache: error: failed to execute compile
sccache: caused by: Compiler not supported: "ninja: FATAL: Couldn't parse arguments. ... cmcldeps <language C, CXX or RC> ..."
```

- Seen 2026-11-25 with Ninja + MSVC 14.52 + sccache, configuring `TRANSCRIBE_SHARED_EMBED=ON` from a superbuild.
- `bindings/rust/sys/build.rs` sets the C/C++ launchers itself when it finds a cache tool, but then leaves `GGML_CCACHE` at its default, so the `shared` / `dynamic-backends` / `arch-dl` features should hit the same failure on a Windows machine with sccache.
- ZER0's default static build compiles no `.rc` and is unaffected.

**Fix:** define `GGML_CCACHE=OFF` wherever the build sets `CMAKE_C_COMPILER_LAUNCHER` / `CMAKE_CXX_COMPILER_LAUNCHER` itself: build.rs always, and the root CMakeLists for direct CMake users. A launcher set per language never reaches RC. `../audiocpp-rs/audiocpp-sys/build.rs` does exactly that for audio.cpp.

### 4. Upstream ggml: the MSVC barrier fix (coordination)

`patches/ggml/0001-fix-threadpool-oversubscription.patch` (CJ Pais, official #131) fixes ggml's CPU thread pool on MSVC:
- `ggml_thread_cpu_relax()` compiles to nothing there (MSVC defines `_M_X64`, not `__x86_64__`);
- the patch adds `YieldProcessor()`, plus a bounded spin before `SwitchToThread()`.

It is still a patch at ggml 1.0, so ggml-org/ggml has not taken it. **audio.cpp's vendored ggml lacks it too.** Proposing it to ggml-org/ggml benefits every downstream; whether that should come from this fork or from CJ is the user's call.

**In short: this fork already has the `pause` (`YieldProcessor()`, `ggml/src/ggml-cpu/ggml-cpu.c` around line 549). Keep it. It is the yield after it that needs checking.**

**Measured 2026-11-26: the yield half regresses under CPU load. Check this fork.** The audio.cpp fork session benchmarked the patch in audio.cpp:
- **Setup:** MSVC 14.52, OpenMP off, i9-13900H with 6 cores / 12 logical CPUs. Warm sessions of tiny_voca TTS and SenseVoice ASR; the arms alternate; outputs are identical.
- **The `pause` half:** neutral at 6 threads. At 12 threads it is 14–54% faster and much steadier. Unchanged under external load.
- **The bounded spin + `SwitchToThread()` half:**
  - It rescues self-oversubscription. At 24 threads on 12 CPUs, TTS drops from 165 s to 6.4 s, and ASR from more than 300 s to 45 s.
  - **It is much worse when other processes load the CPU.** With 12 busy processes at normal priority and the pool at 6 threads, TTS was 3.7× slower (13.7 s → 50.4 s), and ASR went from about 48 s to more than 180 s.
  - The waiter yields its core to the other processes' threads and loses its time slice at every barrier.
- **What audio.cpp gets:** the `pause` branch only. It is open as [0xShug0/audio.cpp#1407](https://github.com/0xShug0/audio.cpp/pull/1407), with all the numbers in `../audiocpp-fork/pr-notes/pr-3-ggml-cpu-msvc-pause.md`.

This fork forces the native pool on every platform (the "OpenMP — CENTRAL POLICY" block in `CMakeLists.txt`), so **ZER0's speech-to-text runs this yield**. Dictating while other work loads the CPU (TTS, a build, a game) is exactly the losing case.

**To check:**
1. Measure transcribe.cpp under the same load (`n_logical` busy processes; the pool at half the logical CPUs; patch 0001 against a pause-only build).
2. If it reproduces, make the yield conditional instead of unconditional: for example, only when the pool's thread count exceeds the logical CPU count, which is the oversubscription the patch was written for (the 2-vCPU CI deadlock).
3. Keep the CI deadlock covered.

The benchmark scripts are `../audiocpp-fork/pr-notes/bench/bench_ab.py` (`--burners N`) and `bench_contention.py`.

### 4b. Silero VAD: this fork has the official weights; audio.cpp did not (information)

This fork's Silero VAD is converted from the official silero-vad 6.2.3 TorchScript checkpoint (`scripts/convert-silero_vad.py`, `docs/models/silero-vad-v6.2.md`), which is correct.

audio.cpp bundles a `silero_vad_16k.safetensors` that matches no official release. Only the STFT basis matches; the conv and LSTM weights differ (snakers4/silero-vad#768). Measured 2026-11-26 on 376 s of speech: audio.cpp's VAD agreed with silero-vad on only 97.4% of the time, and found 64 instead of 79 segments on `samples/death.wav`. With the official v6.2 weights, its segments are identical to silero-vad's.

The fix is open as [0xShug0/audio.cpp#1409](https://github.com/0xShug0/audio.cpp/pull/1409) (notes in `../audiocpp-fork/pr-notes/pr-4-silero-vad-official-weights.md`). Until it lands, ZER0's two engines can disagree on speech segments for the same audio: this fork's VAD is the reference one.

### 5. Read audio.cpp for optimisations on the families both engines run (research)

audio.cpp now implements many of the same STT families on its own runtime: parakeet_tdt, qwen3_asr, cohere_asr, canary_asr, gigaam_asr, moonshine_asr (streaming), sense_asr, fun_asr_nano, nemotron_asr / nemotron_transcribe, voxtral_realtime, granite5asr, sortformer_diar. Its recent work on those may hold ideas worth porting. It may also be slower than the fork somewhere, which is good to know too.

| Commit | Change |
|---|---|
| #869 | Parakeet: fewer Vulkan decoder transfers, lower encoder allocation peaks |
| #1155, #1157 | Parakeet: graph and resource lifecycle; subsampling on Metal |
| #1117 | Parakeet offline chunking with stereo / non-16 kHz input (a correctness fix) |
| #1156 | Qwen3-ASR performance controls and a `cuda_perf` preset |
| #1215, #1213, #1214 | Nemotron: fewer transfers; shared RoPE encoder execution and host CTC decoding |
| #1335 | Granite 5 encoder graph reuse and batched GPU block attention |
| #793, #684 | Voxtral Realtime: repeated transcriptions; never ending a delta inside a UTF-8 character |

Only port what a measurement backs (same GGUF family and quant, same clip, same threads). The `audiocpp` crate or audio.cpp's `audiocpp_cli` gives the audio.cpp side of a comparison.

## Already in the fork: no action

Checked on 2026-11-25 against `60644e11`, so these are not redone:

- the Parakeet frame-batched joint window (`TRANSCRIBE_RNNT_BATCH_W`);
- the mel band-skip (`fb_begin_` / `fb_end_`);
- the direct depthwise conv in the granite, granite_nar, canary_qwen and cohere encoders;
- Voxtral not advertising `INITIAL_PROMPT`;
- the S3 conformer memory work.

Also, the fork's tests do not rely on `assert()`, which Release compiles out (a speech.cpp test-integrity problem that does not exist here), and the official commits #221, #262 and #264 are merged.

Everything else in the archive was speech.cpp-specific and does not apply here:
- its engine ports of these same families into audio.cpp's framework (candidates for audio.cpp pull requests, handled in `../audiocpp-fork`);
- a ggml patch stack that merged both engines' ggml.
