# Moonshine Streaming

Useful Sensors' Moonshine Streaming family ported to transcribe.cpp.
An English-only encoder-decoder transformer designed for streaming use:
ergodic encoder with sliding-window self-attention, 50 Hz time-domain
frontend (CMVN + asinh + linear + two causal stride-2 convs, no STFT),
and a learned-positional adapter between encoder and decoder. Distinct
from the non-streaming [Moonshine](moonshine.md) family — different
encoder, different frontend, different tokenizer hash, separate HF
`model_type`.

For the architecture deep-dive, validation contract, and porting notes,
see the family doc at
[`docs/porting/families/moonshine_streaming.md`](../porting/families/moonshine_streaming.md).

## Choosing a variant

- **Smallest footprint.** `moonshine-streaming-tiny` decodes well above
  realtime on Apple Silicon and Vulkan-class GPUs; CPU is also viable
  for live use.
- **Better accuracy, modest cost.** `moonshine-streaming-small` roughly
  halves WER vs tiny.
- **Best accuracy in the family.** `moonshine-streaming-medium` has the
  most accuracy headroom. Keep an eye on decode latency; the 14-layer
  decoder dominates wall time on long utterances.
- **Non-streaming, batch-only workloads.** Use the
  [`moonshine`](moonshine.md) family instead — it's smaller for the
  same WER on offline audio because it doesn't carry the streaming
  encoder's sliding-window machinery.
- **Non-English audio.** Not supported on this family (English-only,
  no language detection, no translation).

> **Streaming runtime status.** Real-time streaming is implemented and
> validated. Feed audio incrementally through the
> `transcribe_stream_begin` / `transcribe_stream_feed` /
> `transcribe_stream_finalize` API (CLI: `--stream-chunk-ms`), or run a
> single one-shot pass over the full encoder for offline transcription.
> The streaming path uses ~240 ms cumulative encoder right-context, an
> 80 ms feed cadence, and a 20 ms natural emit unit.

## All variants

WER is on LibriSpeech test-clean for the **Q8_0** preset, measured by
transcribe.cpp's WER pipeline with greedy decode (`num_beams=1`,
`do_sample=False`). See each per-variant doc for the full F32 / F16 /
Q8_0 matrix and the comparison to Useful Sensors' Open ASR Leaderboard
numbers — across the family, our F32 and Q8_0 land within 0.1pp of the
HF reference scored on the same manifest, and the small residual to the
upstream-reported numbers is a scoring / text-normalization difference,
not a numerical drift in the port. The K-tier presets (Q6_K / Q5_K_M /
Q4_K_M) are not currently shipped for this family.

<!-- catalog:family variants=moonshine-streaming-tiny,moonshine-streaming-small,moonshine-streaming-medium -->
| Variant                      | Params | Languages | Q8_0 size | Benchmark                    |  Q8_0 | Capabilities | Doc |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `moonshine-streaming-tiny`   |    44M | en        |     50 MB | LibriSpeech test-clean (WER) | 4.52% | streaming    | [moonshine-streaming-tiny.md](moonshine-streaming-tiny.md) |
| `moonshine-streaming-small`  |   140M | en        |    199 MB | LibriSpeech test-clean (WER) | 2.54% | streaming    | [moonshine-streaming-small.md](moonshine-streaming-small.md) |
| `moonshine-streaming-medium` |   266M | en        |    296 MB | LibriSpeech test-clean (WER) | 2.16% | streaming    | [moonshine-streaming-medium.md](moonshine-streaming-medium.md) |
<!-- /catalog -->

Pre-built GGUFs for every variant and quant are hosted under
[`handy-computer` on Hugging Face](https://huggingface.co/handy-computer);
each per-variant doc has direct download links.

## Input limits

**81.9 s of audio per decode** (all variants). The encoder adapter adds a
learned position embedding to every encoder frame from a 4,096-entry table, and
one encoder frame is 20 ms; the reference implementation has no defined output
past it either. `transcribe_capabilities::max_audio_ms` reports 81920.

- **Offline** (`transcribe_run`, `transcribe_run_batch`): longer input is
  rejected up front with `TRANSCRIBE_ERR_INPUT_TOO_LONG` (per utterance in a
  batch). Split long recordings before transcribing them.
- **Streaming**: a stream that runs past 81.9 s keeps its transcript up to the
  cap, ignores later audio, sets `transcribe_was_truncated()` and logs one
  `WARN`; feeds and finalize still return `TRANSCRIBE_OK`. Start a new stream
  (for example at a pause) to keep transcribing.

Within that window the decoder can still stop early: a transcript that reaches
the generation budget is returned with `TRANSCRIBE_ERR_OUTPUT_TRUNCATED`
offline (partial text retained), or flagged via `transcribe_was_truncated()`
when streaming. See the [input-length contract](../input-limits.md).

## Quick start

Pick a variant and run:

```bash
cmake -B build
cmake --build build

build/bin/transcribe-cli \
  -m models/moonshine-streaming-tiny/moonshine-streaming-tiny-Q8_0.gguf \
  samples/jfk.wav
```

The repo doesn't ship the GGUFs — pull them from the corresponding
`handy-computer/<variant>-gguf` repo on Hugging Face, or convert from
the upstream Useful Sensors checkpoint via the per-variant doc's
reproduction section.

## Capabilities

All Moonshine Streaming variants support:

- **Transcription** of 16 kHz mono WAV input directly from raw PCM
  through the time-domain frontend.
- **Single-utterance (one-shot) decode** for offline audio.
- **Real-time streaming** via `transcribe_stream_begin` /
  `transcribe_stream_feed` / `transcribe_stream_finalize` (CLI:
  `--stream-chunk-ms`) — ~240 ms cumulative encoder right-context, 80 ms
  feed cadence, 20 ms natural emit unit.

What's not supported (consistent across the family): translation,
language detection, multilingual transcription (English only),
timestamps, VAD, speaker diarization. See the family doc for the full
runtime contract.
