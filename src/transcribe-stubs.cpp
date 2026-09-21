// transcribe-stubs.cpp - C ABI extension initializers and stubs for models
// excluded from custom or minimal builds.
//
// Ensures full public C ABI symbol stability: even when a model architecture
// (such as Whisper or Sortformer) is not compiled into the library, callers
// of its family-specific extension initializers or query functions link
// without unresolved external symbols.

#include "transcribe.h"
#include "transcribe/moonshine_streaming.h"
#include "transcribe/parakeet.h"
#include "transcribe/r2t2.h"
#include "transcribe/sortformer.h"
#include "transcribe/voxtral_realtime.h"
#include "transcribe/whisper.h"

#include <cstring>

#if !defined(TRANSCRIBE_ENABLE_ARCH_WHISPER)
extern "C" {

TRANSCRIBE_API void transcribe_whisper_run_ext_init(struct transcribe_whisper_run_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size                = sizeof(*p);
    p->ext.kind                = TRANSCRIBE_EXT_KIND_WHISPER_RUN;
    p->prompt_condition        = TRANSCRIBE_WHISPER_PROMPT_FIRST_SEGMENT;
    p->max_prev_context_tokens = 223;
    p->temperature_inc         = 0.2f;
    p->compression_ratio_thold = 2.4f;
    p->logprob_thold           = -1.0f;
    p->no_speech_thold         = 0.6f;
    p->max_initial_timestamp   = 1.0f;
}

TRANSCRIBE_API void transcribe_whisper_chunk_trace_init(struct transcribe_whisper_chunk_trace * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->struct_size = sizeof(*p);
}

TRANSCRIBE_API int transcribe_get_whisper_chunk_count(const struct transcribe_session * /*session*/) {
    return 0;
}

TRANSCRIBE_API transcribe_status
transcribe_get_whisper_chunk_trace(const struct transcribe_session * /*session*/,
                                   int /*i*/,
                                   struct transcribe_whisper_chunk_trace * /*out_trace*/) {
    return TRANSCRIBE_ERR_UNSUPPORTED_ARCH;
}

}  // extern "C"
#endif

#if !defined(TRANSCRIBE_ENABLE_ARCH_VOXTRAL_REALTIME)
extern "C" {

TRANSCRIBE_API void transcribe_voxtral_realtime_stream_ext_init(struct transcribe_voxtral_realtime_stream_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size               = sizeof(*p);
    p->ext.kind               = TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM;
    p->num_delay_tokens       = -1;
    p->min_decode_interval_ms = -1;
}

}  // extern "C"
#endif

#if !defined(TRANSCRIBE_ENABLE_ARCH_MOONSHINE_STREAMING)
extern "C" {

TRANSCRIBE_API void transcribe_moonshine_streaming_stream_ext_init(
    struct transcribe_moonshine_streaming_stream_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size               = sizeof(*p);
    p->ext.kind               = TRANSCRIBE_EXT_KIND_MOONSHINE_STREAMING_STREAM;
    p->min_decode_interval_ms = -1;
}

}  // extern "C"
#endif

#if !defined(TRANSCRIBE_ENABLE_ARCH_PARAKEET)
extern "C" {

TRANSCRIBE_API void transcribe_parakeet_stream_ext_init(struct transcribe_parakeet_stream_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size          = sizeof(*p);
    p->ext.kind          = TRANSCRIBE_EXT_KIND_PARAKEET_STREAM;
    p->att_context_right = -1;
}

TRANSCRIBE_API void transcribe_parakeet_buffered_stream_ext_init(struct transcribe_parakeet_buffered_stream_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size = sizeof(*p);
    p->ext.kind = TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM;
    p->left_ms  = -1;
    p->chunk_ms = -1;
    p->right_ms = -1;
}

}  // extern "C"
#endif

#if !defined(TRANSCRIBE_ENABLE_ARCH_SORTFORMER)
extern "C" {

TRANSCRIBE_API void transcribe_sortformer_stream_ext_init(struct transcribe_sortformer_stream_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size = sizeof(*p);
    p->ext.kind = TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM;
    p->preset   = TRANSCRIBE_SORTFORMER_PRESET_DEFAULT;
}

}  // extern "C"
#endif

#if !defined(TRANSCRIBE_ENABLE_ARCH_QWEN3_ASR)
extern "C" {

TRANSCRIBE_API void transcribe_r2t2_stream_ext_init(struct transcribe_r2t2_stream_ext * p) {
    if (p == nullptr) {
        return;
    }
    std::memset(p, 0, sizeof(*p));
    p->ext.size      = sizeof(*p);
    p->ext.kind      = TRANSCRIBE_EXT_KIND_R2T2_STREAM;
    // Spelled out rather than shared with the family: this function exists
    // precisely for builds that exclude the family, so it cannot reach that
    // family's constant. The C header documents 320 ms as the contract
    // default, and this is that value. The two initializers are therefore an
    // untested duplicate of one number, and a change to one must change the
    // other; no current test compares them, because the two definitions never
    // coexist in one binary.
    p->chunk_size_ms = 320;
}

}  // extern "C"
#endif
