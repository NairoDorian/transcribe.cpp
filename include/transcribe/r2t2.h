/*
 * include/transcribe/r2t2.h - Confucius4-R2T2 streaming extension.
 *
 * Includes transcribe.h; safe to include in C or C++ TUs. Holds the
 * streaming-operating-point extension, its kind constant, and its init
 * function.
 *
 * Confucius4-R2T2 is a variant of the qwen3_asr family, not a family of its
 * own: same audio encoder and causal decoder, dispatching to the same
 * architecture. The extension is therefore accepted only by a session whose
 * loaded model is that variant, and the probe is
 * transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_STREAM,
 * TRANSCRIBE_EXT_KIND_R2T2_STREAM) before pointing
 * transcribe_stream_params::family at the struct below.
 *
 * Probe before use. On any other model the kind is rejected and passing it
 * is an error, not a no-op.
 */
#ifndef TRANSCRIBE_R2T2_H
#define TRANSCRIBE_R2T2_H

#include "transcribe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* R2T2 native decoding cadence, independent of PCM feed packet duration.
 *
 * This is a cadence, not a window: the variant re-encodes the audio
 * accumulated so far on each tick and commits the longest stable prefix, so
 * the value bounds how often that happens and therefore how soon text can
 * first appear. It is not end-to-end latency, which also includes queue wait,
 * the feed call's own duration and the final flush.
 *
 * Every integer millisecond in [80, 2000] is accepted, both endpoints
 * included, with a 1 ms step and no rounding to a preset. 16 kHz audio makes
 * one millisecond exactly 16 samples, so the value maps to a sample count
 * with no quantization. Out-of-range values are rejected, not clamped.
 *
 * Default is 320 ms. The value is copied at stream_begin and cannot change an
 * active stream; a new value applies to the next stream.
 */
#define TRANSCRIBE_EXT_KIND_R2T2_STREAM 0x32543252u

struct transcribe_r2t2_stream_ext {
    struct transcribe_ext ext;
    uint32_t              chunk_size_ms;
};

/* Fills ext.size, ext.kind and chunk_size_ms = 320. */
TRANSCRIBE_API void transcribe_r2t2_stream_ext_init(struct transcribe_r2t2_stream_ext * ext);

#ifdef __cplusplus
}
#endif

#endif /* TRANSCRIBE_R2T2_H */
