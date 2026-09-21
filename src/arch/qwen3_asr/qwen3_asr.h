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
    std::string          raw_text;
    bool                 truncated   = false;
    int                  n_generated = 0;
    // The generated token ids, trailing EOS stripped — the same sequence
    // `raw_text` was decoded from. Diagnostic (the streaming trace hook);
    // offline ignores it.
    std::vector<int32_t> gen_ids;
};

// ---------------------------------------------------------------------------
// Encoder prefix cache (R2T2 streaming)
// ---------------------------------------------------------------------------

// Streaming re-runs the whole decode pass over the accumulated buffer every
// tick, so its encoder cost grows with the stream: measured on R2T2, 115 ms of
// a 265 ms tick at 54 s of audio and ~470 ms at 81 s. Encoder attention is
// windowed (encoder.h), which makes the encoder output for one *complete*
// window a function of that window's own mel frames only — per-chunk conv, a
// positional table indexed inside the chunk, masked attention — so those rows
// are final the moment the window fills, and only the trailing window (plus any
// partial chunk) has to be encoded again. This struct is that carry-over.
//
// It is a cache, never an approximation, and the argument has two halves that
// fail independently — which is worth spelling out, because the first version
// of this cache got the second one wrong and shipped a silently truncated
// transcript.
//
//   (a) The mel the rows were derived from. The front-end normalizes per
//       utterance over the whole buffer (`clamp(global_max - 8)`, then
//       `(x + 4) / 4`), so a new global maximum rewrites the level of every
//       earlier frame. run_decode_pass memcmps the cached prefix against the
//       frames this pass just produced and falls back to a full encode when
//       they differ (mel_prefix_matches). Reused rows are therefore rows of
//       this utterance's mel, not of an earlier one.
//
//   (b) *Where* those frames sit in the tail that gets encoded. On a hit the
//       pass encodes only the tail, so the tail's frame offset has to reach
//       the encoder intact: the mel buffer is [n_mels, n_frames] row-major
//       with the frame count as its row stride, and the frames from offset f
//       on are a column slice of it — not contiguous, and not a pointer
//       offset. Getting this wrong keeps every shape and every count correct
//       (they come from lengths, not from the layout) and feeds the encoder
//       entirely plausible garbage, so nothing downstream can notice. The
//       offset is a parameter of pack_mel_chunks for exactly this reason, and
//       tests/qwen3_asr_mel_pack_unit.cpp pins it.
//
// With both halves holding, a hit can only remove work and the streaming
// transcript is identical with the cache on and off; TRANSCRIBE_R2T2_NO_ENC_CACHE=1
// is the A/B that shows it.
//
// Owned by R2T2StreamState and reused across ticks; `clear()` it when the
// stream restarts. run_decode_pass(a) reads rows/mel/tokens/frames when they
// describe the same mel and (b) rewrites all of them for the next tick.
//
// Alignment is a precondition, not a runtime check: `tokens` is always a whole
// number of attention windows and `frames` the mel frames those windows cover,
// so the tail starts on a chunk boundary too (a window is 8 whole chunks).
// Everything downstream — chunk grid, positional table, window partition —
// then reproduces the full pass's rows for those positions.
struct EncoderPrefixCache {
    // [d_enc, tokens] — the window-aligned prefix of the pass's enc_host.
    std::vector<float> rows;
    // [n_mels, frames] — the mel `rows` was computed from.
    std::vector<float> mel;
    int32_t            n_mels = 0;
    int32_t            frames = 0;  // mel frames covered by `rows`
    int32_t            tokens = 0;  // after-CNN tokens covered by `rows` (whole windows)

    // What the last pass did, for the trace and the perf breakdown: how many
    // of its tokens came from the cache and how many it encoded.
    int32_t reused_tokens  = 0;
    int32_t encoded_tokens = 0;

    void clear() {
        rows.clear();
        mel.clear();
        n_mels = frames = tokens = 0;
        reused_tokens = encoded_tokens = 0;
    }
};

// Run one full decode pass over `pcm`: mel -> audio encoder -> prefill ->
// greedy step loop. `suffix_ids` (may be null) is appended after the assistant
// header. `max_new_tokens` is the generation budget.
//
// `enc_cache` (may be null) is the streaming encoder prefix cache described
// above. Offline callers pass null. A non-null cache is consulted and updated
// in place; see EncoderPrefixCache for the exactness argument. The environment
// variable TRANSCRIBE_R2T2_NO_ENC_CACHE=1 disables its use without changing the
// call, for the A/B that shows both paths produce identical text.
//
// `draft_seed` (may be null) is an external speculation seed: a guess at the
// tokens this pass will emit first, derived by the caller from the previous
// pass (R2T2 streaming hands over the tokens its last tick deliberately held
// back, which are what the next tick re-derives — see r2t2-stream.cpp). It is
// used as the draft for the FIRST verify run only; every later run in the same
// pass falls back to the 1-gram lookup, because a run only continues past a
// mismatch, and past the seed there is no such guess left. Acceptance is the
// same exact-greedy rule as the family's own drafting, so no token is committed
// that plain stepping would not have produced; the numerics caveat on that loop
// (no byte-equality with drafts disabled) applies here too. Ignored when empty,
// and dropped (the pass degrades to plain stepping) when the KV window has no
// room for the draft columns.
//
// `mel_stream`, when non-null and supported by the frontend, switches the mel
// extraction to the incremental path: the pass recomputes only the frames this
// tick added and reuses the rest from the caller's state. The result is
// bit-identical to the batch extraction (tests/mel_unit.cpp is the gate), so it
// changes cost, not values. Pass the same state object across a stream's ticks
// and clear it when the stream restarts; leave null for a one-shot pass.
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
                                  DecodePassResult *            out,
                                  const std::vector<int32_t> *  draft_seed = nullptr,
                                  EncoderPrefixCache *          enc_cache  = nullptr,
                                  transcribe::MelStreamState *  mel_stream = nullptr);

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
    // whole input on every tick. The reference re-encodes this in full; here
    // `enc_cache` carries the finished windows over, so a tick only encodes
    // what it added (see EncoderPrefixCache). The buffer itself still holds the
    // whole stream, because the mel front-end's per-utterance normalization and
    // the LM prompt's audio-token prefix are both defined over all of it.
    std::vector<float> audio_accum;

    // Encoder rows whose windows are already final, and the mel they came
    // from. A hit is bit-exact and a miss falls back to a full encode, so this
    // changes per-tick cost only, never output; cleared with the rest of the
    // stream state.
    EncoderPrefixCache enc_cache;

    // Mel front-end state: the raw log-mel of every frame this stream has
    // already produced, so a tick re-runs the STFT + filterbank only over the
    // frames it added. That cost is otherwise O(stream length) — the front-end
    // is a pure function of (config, audio), so re-feeding the whole
    // `audio_accum` re-derives every frame of the utterance on every tick.
    // Values are unchanged (compute_incremental is bit-identical to the batch
    // path), so this is a pure cost fix and is cleared with the rest of the
    // stream state.
    transcribe::MelStreamState mel_stream;

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

    // Speculation seed for the NEXT tick: the token-space rollback tail of the
    // last tick — exactly the tokens that tick refused to commit and the next
    // one therefore re-derives. Measured on this checkpoint it is reproduced
    // nearly verbatim (~4.5 of 5 tokens, jfk/zh-long, 80-320 ms), so running
    // the greedy step loop as a verify pass over it costs one graph run where
    // plain stepping costs ~5.5. Empty when there is no hold-back (k == 0) or
    // before the first commit path. See decode_tick for how it is rebuilt and
    // TRANSCRIBE_R2T2_NO_DRAFT for the switch that disables it.
    std::vector<int32_t> draft_tail;

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
