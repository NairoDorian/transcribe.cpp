// arch/qwen3_asr/qwen3_asr.h - Qwen3-ASR model and context types.
//
// INTERNAL to src/arch/qwen3_asr/. Concrete transcribe_model /
// transcribe_session subclasses for the Qwen3-ASR family (audio-LLM: audio
// encoder + Qwen3 causal LM with audio-token injection).

#pragma once

#include "causal_lm/causal_lm.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe-backend.h"
#include "transcribe-mel.h"
#include "transcribe-model.h"
#include "transcribe-session.h"
#include "transcribe-tokenizer.h"
#include "weights.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_backend_sched;
typedef struct ggml_backend *        ggml_backend_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend_sched *  ggml_backend_sched_t;

namespace transcribe::qwen3_asr {

void apply_family_invariants(transcribe_model & model);

// Encode "language {Name}<asr_text>" for the given BCP-47 code: the token-id
// sequence the chat template seeds the assistant turn with on a forced hint.
// Declared here (not in model.cpp's anon namespace) so the BPE parity test can
// verify the full prefix against the HF reference.
//
// Returns:
//   TRANSCRIBE_OK                     on success, out_ids populated.
//   TRANSCRIBE_ERR_INVALID_ARG        bcp47 null or empty.
//   TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE  bcp47 not in the publisher's
//                                     support list (the static map
//                                     in model.cpp is the source of
//                                     truth; drift vs. caps.languages
//                                     surfaces here).
//   TRANSCRIBE_ERR_GGUF               tokenizer has no encoder
//                                     (missing merges) or vocab
//                                     missing <asr_text> special.
transcribe_status encode_language_prefix(const transcribe::Tokenizer & tok,
                                         const char *                  bcp47,
                                         std::vector<int32_t> &        out_ids);

// Reverse of the map encode_language_prefix reads: canonical publisher name
// ("English") -> the BCP-47 code the public API reports ("en"). Returns
// nullptr when the name is unknown (a model-invented language, or one the
// frozen table does not cover).
//
// Declared here so the streaming path reports detected_language in the same
// spelling the offline path does. The offline path reverse-maps through the
// same table; two copies of the mapping is exactly how the two paths would
// silently start disagreeing about which language was detected.
const char * bcp47_for_publisher_name(const std::string & name);

// ---------------------------------------------------------------------------
// Shared decode pass (offline run + R2T2 streaming)
// ---------------------------------------------------------------------------

// What one decode pass produced. `raw_text` is the detokenized generation with
// the trailing EOS already removed and nothing else stripped: the caller
// applies its own output parsing, because offline and streaming parse
// differently (offline strips the "language X<asr_text>" envelope once;
// streaming re-parses the whole accumulated text every chunk).
struct DecodePassResult {
    std::string raw_text;
    bool        truncated   = false;
    int         n_generated = 0;
};

// Run one full decode pass over `pcm`: mel -> audio encoder -> prefill ->
// greedy step loop. `suffix_ids` (may be null) is appended after the assistant
// header. `max_new_tokens` is the generation budget.
//
// Writes only session scratch (mel_buf, enc_host, t_* timers) — never result
// state, and never frees the scheduler or KV cache. Defined in model.cpp and
// declared here so the streaming path reuses the offline implementation
// instead of duplicating it.
transcribe_status run_decode_pass(transcribe_session *          session,
                                  const float *                 pcm,
                                  int                           n_samples,
                                  const transcribe_run_params * params,
                                  const std::vector<int32_t> *  suffix_ids,
                                  int                           max_new_tokens,
                                  DecodePassResult *            out);

// ---------------------------------------------------------------------------
// Model / Context
// ---------------------------------------------------------------------------

// Qwen3 chat-template special-token ids, resolved through the loaded tokenizer
// at load time so a future vocab reorder fails loudly. Reference ids on the
// 0.6B/1.7B vocab are stable but never hardcoded at runtime.
struct ChatTokens {
    int32_t im_start       = -1;
    int32_t im_end         = -1;
    int32_t newline        = -1;
    int32_t role_system    = -1;
    int32_t role_user      = -1;
    int32_t role_assistant = -1;
};

// Per-stream state for the Confucius4-R2T2 variant. Mirrors the reference's
// `ASRStreamingState` field set (audio.cpp R2T2ASRSession, session.h:102-120);
// the places this deliberately differs from it are noted at their fields.
//
// Declared here, beside the session that owns it, rather than in
// r2t2-stream.h: it is a session member and r2t2-stream.h includes this header
// for the hook signatures, so defining it there would be a cycle.
//
// Lifetime: zeroed by stream_reset / stream_begin. Holds no GPU resources —
// every graph and buffer it uses belongs to the session and is rebuilt per
// chunk by run_decode_pass — so resetting it is free and a stream that fails
// mid-way cannot leak device memory.
struct R2T2StreamState {
    // Cadence latched from the stream extension at stream_begin. Because the
    // model re-encodes the whole accumulated buffer on each tick, this bounds
    // how often the expensive path runs, not how much audio the model sees —
    // it is a decode interval, not a window. Latched rather than re-read so a
    // caller mutating its ext struct mid-stream cannot change an active
    // stream's cadence.
    uint32_t chunk_size_ms = 0;

    // Frames per decode tick. Latched on the first feed rather than derived
    // from chunk_size_ms each tick, because it is the value chunk slicing
    // actually counts in and the two must not drift.
    int64_t chunk_size_samples = 0;

    // Audio accepted but not yet forming a whole tick. Decouples the caller's
    // feed granularity from the decode cadence, so any feed size — including
    // one larger than a whole chunk — drives the same cadence.
    std::vector<float> buffer;

    // Every sample ever accepted, never trimmed and never padded: the model's
    // whole input on every tick. The reference re-encodes this in full instead
    // of keeping an encoder cache, which is what makes the cadence the only
    // streaming knob and makes per-tick cost grow with the utterance.
    std::vector<float> audio_accum;

    // Transcription text accumulated so far, tag envelope included — the
    // reference's `raw_decoded`, and the prompt continuation for the next
    // tick. Held as text rather than token ids because the text pipeline
    // rewrites it (punctuation, spacing, envelope re-attachment) between
    // ticks, so any token list would go stale the moment one of those fired.
    std::string raw_decoded;

    // Current transcript, the reference's `text_`: committed prefix plus
    // unstable tail. Published as the session's full_text.
    std::string text;

    // Language in the canonical publisher spelling ("Chinese", not "zh"),
    // because every r2t2:: text-layer function takes that form. Empty until
    // the model reports one.
    std::string language;

    // Caller's language hint, canonical spelling; empty == auto-detect. The
    // reference re-derives this from the request each tick; the hooks here are
    // called without the original stream params, so it has to be latched.
    std::string force_language;

    // Token ids seeding the assistant turn when a language is forced:
    // encode("language {Name}") + the <asr_text> special id, from
    // encode_language_prefix. Empty in auto-detect mode, where the model emits
    // the envelope itself. Precomputed at stream_begin because resolving it
    // per tick would re-encode the same string for no reason.
    std::vector<int32_t> prompt_seed_ids;

    // Decode ticks that reached the commit path. A tick that ends before the
    // language tag appears does NOT advance this (matching the reference), so
    // in auto-detect mode the lead-in is counted in tag arrivals, not in
    // chunks of audio.
    int64_t chunk_id = 0;

    // Samples fed to the stream, and samples covered by a completed tick. Both
    // are published as the update's audio cursors; the difference is what the
    // caller sees as buffered.
    int64_t audio_input_samples     = 0;
    int64_t audio_committed_samples = 0;

    // Byte length of the stable prefix of `text` (the reference's
    // `fixed_text`), published so the dispatcher can commit up to it and
    // derive tentative_text. Recomputed every tick; the dispatcher ignores a
    // value that does not advance, which is how the append-only guarantee is
    // enforced on the shared side.
    size_t committed_bytes = 0;
};

struct QwenAsrModel final : public transcribe_model {
    Tokenizer      tok;
    QwenAsrHParams hparams;
    QwenAsrWeights weights;
    ggml_context * ctx_meta = nullptr;

    transcribe::BackendPlan                    plan;
    ggml_backend_buffer_t                      backend_buffer = nullptr;
    transcribe::causal_lm::PackedGateUpHandles packed_gate_up;

    std::optional<transcribe::MelFrontend> mel;

    // Jinja chat template string from the GGUF KV (empty == none).
    std::string chat_template;

    // Resolved chat-template token ids (see resolve_chat_tokens in model.cpp).
    ChatTokens chat_tokens;

    QwenAsrModel() = default;
    ~QwenAsrModel() override;

    const transcribe::Tokenizer * tokenizer() const override { return &tok; }
};

struct QwenAsrSession final : public transcribe_session {
    transcribe::causal_lm::KvCache kv_cache;

    // Batched KV cache for offline transcribe_run_batch (n_batch slabs).
    // Allocated/resized lazily by run_batch; freed in the destructor.
    transcribe::causal_lm::KvCache kv_cache_batch;
    int                            kv_batch_cap   = 0;  // allocated n_batch
    int                            kv_batch_n_ctx = 0;  // allocated n_ctx

    std::vector<float> mel_buf;
    std::vector<float> enc_host;  // audio encoder output, pre-injection

    bool encoder_use_flash = true;
    bool decoder_use_flash = true;

    // Confucius4-R2T2 streaming state. Reset by stream_begin/stream_reset;
    // unused by every other variant of this family (they reject the stream
    // extension and advertise supports_streaming = false).
    R2T2StreamState r2t2;

    QwenAsrSession() = default;
    ~QwenAsrSession() override;
};

}  // namespace transcribe::qwen3_asr
