# Confucius4-R2T2

Status: loader adaptation implemented and compiling; streaming and application
integration are not implemented yet.

## Current implementation findings — 2026-09-20

The user requires a **native transcribe.cpp port**, usable by the application;
an audio.cpp executable, server, DLL or alternate runtime is not acceptable.
Use the sibling code as implementation reference only. Inference must use
transcribe's GGML backend selection, model/session ownership and streaming API.

The network matches the existing Qwen3-ASR audio encoder and causal decoder:
24 encoder layers, 1024 encoder width, 2048 projected audio width, 28 decoder
layers, 16 query heads, 8 KV heads and 128 head width. The pinned publisher
configuration uses `thinker_config` and declares BF16. Existing native Qwen
graphs are the reuse target; R2T2's additional work is the streaming state
machine, prompt-prefix handling and text post-processing, not a new backend.

The published Q8 file actually declares `general.architecture=audiocpp`, with
`audiocpp.model_spec.family=confucius4_r2t2`. It contains 707 tensors with native
publisher names and embedded configuration/tokenizer files. This describes
its **file format**, not a runtime we intend to use. The existing transcribe
loader cannot consume that schema unchanged. A native, family-specific loader
adapter or a transcribe-format conversion is required before using the verified
URL in the application. No external audio.cpp inference dependency is planned.

Initial source preparation now exists: the R2T2 text post-processing code was
adapted with Apache attribution, tensor-name mappings were generated from the
existing Qwen converter contract, and an 80–2000 ms stream-extension structure
was drafted. These are scaffolding, not connected or validated model support.
They have not yet passed a build as part of the model runtime.

The compatible audio.cpp graph-optimizer subset is already in
`src/transcribe-graph-opt.h` and used before scheduler allocation by the Qwen,
Granite and Parakeet paths. It preserves outputs, views and writable storage.
`TRANSCRIBE_GRAPH_OPTIMIZER=1` enables it. Earlier measurements did not establish
a universal gain, so importing it does not justify enabling it unconditionally.
The sibling's capacity-bucketed encoder/prefill cache is a separate optimization
and has **not** been ported. It requires valid-prefix masks and KV reset checks.

Remaining implementation: native package adaptation/conversion; streaming
begin/feed/finalize/reset; token rollback and UTF-8-safe stable output; exact
chunk-duration propagation through C/Rust and both application streaming paths;
catalogue entry with verified artifact identity; numeric slider/settings;
CPU/CUDA transcript and tail/reset validation. No claim of a working R2T2
download, native stream, slider, or model speedup is made at this checkpoint.

The four earlier task commit descriptions were expanded locally while preserving
their trees and parents. The history rewrite is not pushed yet; application
dependency pins must be updated to the final engine commit before publication.

## Identity and references

- Family: `confucius4_r2t2`; variant: `confucius4-r2t2-1.7b`.
- Publisher: [NetEase Youdao](https://huggingface.co/netease-youdao/Confucius4-R2T2), checkpoint `185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`.
- Reference dtype: BF16, verified from 707 safetensors tensor headers.
- Weight license: NetEase Model Use License Agreement, not Qwen's Apache license.
- Canonical streaming reference: [publisher implementation](https://github.com/netease-youdao/Confucius4-R2T2/blob/80c22e6140bcb9166fb9906798894fc8b18c8309/r2t2/r2t2_asr.py).
- Secondary implementation: [audio.cpp a7b58a6](https://github.com/0xShug0/audio.cpp/commit/a7b58a6d3d6ae4143c485266b1c6c09898ad8c72).
- Acceptance dataset: LibriSpeech test-clean; measured reference WER pending.

## Capability validation

| Capability | Target | Status |
|---|---|---|
| Explicit language transcription | MUST PASS | TODO |
| Auto/no-hint transcription | MUST PASS | TODO |
| Offline batch | MUST PASS | TODO |
| Native streaming, stable deltas and authoritative final text | MUST PASS | TODO |
| Chunk sizes 80 through 2000 ms, including both endpoints | MUST PASS — user request | TODO |
| Context and hotwords | OUT OF SCOPE — user deferred on 2026-09-20 to prioritize streaming | TODO |

## Download identity

The secondary implementation publishes a standalone Q8 file at:

[r2t2-q8_0.gguf](https://huggingface.co/davidxifeng/Confucius4-R2T2-gguf/resolve/a8e6b385d7df7eae9519363e07034a209004797a/r2t2-q8_0.gguf)

Verified HTTP HEAD 200 on 2026-09-20. Content length: 2,477,512,064 bytes.
Hugging Face LFS SHA-256:
`19f5ccd624484bcb5d44301437de41560b0ecc40c430e8850dfeefefbe82ccf5`.
The downloaded file was subsequently SHA-256 verified locally on 2026-09-20;
its hash matches the LFS digest above. It is retained under ignored
`build/diagnostics/r2t2-reference/published` in the engine workspace.

The same revision has `r2t2-f16.gguf` (4,092,155,264 bytes), SHA-256
`d1b531ceaf5640d98352d3a9180238d99d36d393e160afd4692031077e7bae2c`.
These files use audio.cpp packaging. Do not expose them as working transcribe
downloads until metadata/tensor adaptation and actual inference are validated.
An ordinary Qwen catalogue alias does not implement R2T2 streaming semantics.

## Streaming control contract

Expose a per-model integer chunk-size control from **80 to 2000 ms**, with a
1 ms step and direct numeric entry. Use 320 ms initially, matching the sibling
implementation; make 80 ms directly selectable. Persist the exact value and
validate it again in the native extension. Do not map this model through the
existing four Parakeet/Nemotron presets or silently round to those presets.
At 16 kHz, an integer millisecond corresponds to exactly 16 samples.

Label it **Streaming chunk size**. Chunk duration is not guaranteed end-to-end
latency: include queue wait, first committed text, per-feed p95/max and final
flush timings in diagnostics. Record the requested and resolved chunk duration.
Changing a setting applies to the next stream, not halfway through an active
decoder state. The capture path must forward small frames without waiting for
VAD silence before a native streaming dispatch.

Benchmark CPU and CUDA with exactly three runs per loaded configuration;
discard the first, retain raw results, and average runs two and three. Exercise
80/160/320/640/1280/2000 ms plus an irregular value to catch hidden quantization.

## Porting risks and remaining work

The publisher's pinned streaming reference explicitly requires vLLM. The
sibling implementation's MPS golden-generation script refers to a different
reference environment, so it cannot establish our Windows oracle by itself.
Establish a reproducible reference before claiming parity.

Preserve token rollback, UTF-8 boundaries, punctuation/repetition repair,
language-tag handling, pipe truncation, final tail and reset behavior. The
model re-encodes accumulated audio; capacity-bucketed encoder and prefill graph
reuse must refill masks and clear stale KV. Padding may change floating-point
reduction order; check both numerics and transcript behavior across bucket
growth and shrink. Context/hotword UI is intentionally deferred.

## Commands and artifacts

```powershell
uv run scripts/intake.py inspect --repo netease-youdao/Confucius4-R2T2 --family confucius4_r2t2 --variant confucius4-r2t2-1.7b --out reports/porting/confucius4_r2t2/confucius4-r2t2-1.7b/intake.json
uv run scripts/preflight.py --family confucius4_r2t2 --variant confucius4-r2t2-1.7b --gate A
```

Initial Gate A: WARN, not a numerical pass. Tokenizer alignment passes;
dtype declaration lookup misses nested `thinker_config.dtype`, and missing
frontend normalization metadata is interpreted as `none` by preflight even
though Whisper applies log-mel clamping/scaling. GGUF capability checks await
conversion. The golden manifest remains explicitly marked as a skeleton.

## Quantization

See `docs/porting/families/confucius4_r2t2-quantization.md` for the measured
precision ladder: which blocks tolerate which type, the 1.187 GB floor
(`r2t2-q4_k_m.gguf`, 52% below Q8_0), the Q6_K floor on `mlp.down_proj`, and the
build recipe. `docs/tools/quantization-arms.md` covers the per-tensor method
generally.

## Local reference checkpoint

The pinned publisher safetensors checkpoint has been downloaded under ignored
`build/diagnostics/r2t2-reference/checkpoint`. The existing author-Qwen dumper
successfully ran BF16 CPU inference with this checkpoint using Transformers
4.57.6 / qwen-asr 0.0.6 / Torch 2.11.0+cpu. It produced 13 tensor dumps and the
expected JFK transcription (29 generated tokens) under
`build/validate/confucius4_r2t2/confucius4-r2t2-1.7b/jfk/decode/ref`.
This establishes offline network bring-up only: it is not a streaming oracle,
a WER result, or a C++ parity pass. The existing dumper lacks the newer RMS/p99
sidecar fields, which must be added to the dedicated adapter before completing
Stage 2. No supported-model claim follows from this result.
