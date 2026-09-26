// arch/qwen3_asr/r2t2-stream.cpp - Confucius4-R2T2 streaming surface.
//
// The R2T2 variant streams by re-encoding the audio accumulated so far, not by
// advancing a KV ring: the publisher's decoder is trained on whole utterances
// and its streaming mode re-runs the encoder over the growing buffer at a fixed
// cadence, committing the longest stable prefix of each re-decode. That is why
// the only streaming knob is a cadence in milliseconds rather than a window
// size, and why nothing here keeps an incremental encoder cache.
//
// This file owns the extension surface (kind, defaults, validation), the
// per-stream state machine, and the hooks. It is deliberately separate from
// model.cpp: the offline path there is shared by every variant of this family,
// and nothing in this file may change what an offline run does.
//
// Ported from audio.cpp's community_models/confucius4_r2t2/session.cpp
// (decode_stream_chunk, build_stream_prefix, decode_rollback_prefix,
// publish_stream_delta). The mapping onto this API is:
//
//   reference                        -> here
//   raw_decoded_ (text)              -> R2T2StreamState::raw_decoded
//   text_ (display transcript)       -> R2T2StreamState::text == session->full_text
//   fixed_text (stable prefix)       -> R2T2StreamState::committed_bytes: the
//                                       byte length of fixed_text inside text_
//   publish_stream_delta()           -> the dispatcher's
//                                       FamilyRawByteCommit candidate, see
//                                       stream_stable_prefix_impl_for_arch
//
// The reference counts published progress in code points and slices
// fixed_text to find the new ones. Here the dispatcher owns that job: it
// commits up to the byte offset this file publishes and derives
// tentative_text from the remainder, which gets the append-only guarantee from
// one shared implementation instead of a second one per family.
#include "r2t2-stream.h"

#include "r2t2-package.h"
#include "r2t2-text.h"
#include "transcribe-debug.h"
#include "transcribe-env.h"
#include "transcribe-log.h"
// The public surface this file implements: the ext struct, its kind, and the
// contract the validation below enforces. Included rather than restated so a
// change to the header cannot leave the runtime validating a different range
// than the one callers read.
#include "transcribe/r2t2.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace transcribe::qwen3_asr {

namespace {

// ---------------------------------------------------------------------------
// Reference streaming configuration
// ---------------------------------------------------------------------------
//
// These are the reference implementation's `StreamConfig` defaults
// (session.h), held as constants rather than exposed as extension fields
// because the porting contract fixes them: the offline and streaming paths
// must produce the same text, and every one of these values was tuned against
// the publisher's checkpoint. A future variant that needs different values
// gets its own constant block, not a runtime knob.

// Generation budget per tick. Not a truncation limit in practice: a chunk of
// audio produces far fewer than 32 tokens, and the reference treats a tick that
// runs to the budget as ordinary (see run_decode_pass's `truncated`).
inline constexpr int k_max_new_tokens = 32;

// Ticks at the start of a stream that decode but commit nothing. The first
// re-decode of a short prefix is the least reliable, and in auto-detect mode
// the language tag has not appeared yet; both are handled by simply not
// building a continuation prompt until this many ticks have passed.
inline constexpr int64_t k_unfixed_chunk_num = 2;

// Tokens held back from the end of the re-decoded text when computing the
// stable prefix. The last few tokens of an autoregressive re-decode over a
// growing buffer are the most likely to change on the next tick, so they are
// never committed. Held back in TOKEN space, which is why the prefix is
// computed by decoding a truncated token list rather than by slicing text.
inline constexpr int64_t k_unfixed_token_num = 5;

// Reference default; when true, a stable prefix ending in sentence
// punctuation is committed in full (k = 0) instead of holding back
// k_unfixed_token_num tokens. False on this checkpoint, so the hold-back
// always applies — kept as a named constant so the branch below reads like the
// reference it came from rather than being silently dropped.
inline constexpr bool k_rollback_punctuation = false;

// Speculation on the hold-back: the tokens a tick refuses to commit are the
// tokens the next tick re-derives, so they are handed to the next decode pass
// as its draft. Measured on this checkpoint that draft is reproduced almost
// verbatim — ~4.5 of the 5 held-back tokens, at 80 and 320 ms, on English and
// Chinese samples — which turns ~5.5 plain steps per tick into ~1.1 verify
// passes over the same tokens. The family's acceptance rule is exact-greedy (a
// draft is committed only where the verify pass's own argmax equals it), so no
// token enters the transcript that plain stepping would not have produced; what
// it does change is the graph shape the step loop runs (one pass over
// k_unfixed_token_num + 1 columns instead of a sequence of single-column
// steps), which is not bit-identical — the caveat documented at length on the
// spec branch of run_decode_pass, and the reason for the kill switch below.
//
// TRANSCRIBE_R2T2_NO_DRAFT=1 disables it: the stream then re-derives the
// hold-back by plain stepping, the pre-speculation behaviour. Keep the switch
// (rather than making this a build-time choice) because it is also the A/B arm
// for text parity: `TRANSCRIBE_R2T2_NO_DRAFT=1` versus unset is a byte-exact
// comparison of the two paths on any sample.
inline bool draft_enabled() {
    return !transcribe::env::flag("TRANSCRIBE_R2T2_NO_DRAFT");
}

// ---------------------------------------------------------------------------
// Text/token helpers
// ---------------------------------------------------------------------------

// Encode `text` to token ids, splitting out the literal "<asr_text>" tag.
//
// Tokenizer::encode runs BPE over the whole string and does not recognise
// special tokens inside it (see its header contract): a literal "<asr_text>"
// would be encoded piece-wise into nonsense instead of resolving to the single
// special id. Since the streaming continuation prompt is built from text that
// contains that tag, the tag has to be split out and its id appended by hand —
// the same hand-insertion encode_language_prefix does for the language seed.
//
// Returns TRANSCRIBE_ERR_GGUF when the vocab has no such special, which is the
// same failure encode_language_prefix reports for a missing tag.
transcribe_status encode_with_tag(const Tokenizer & tok, const std::string & text, std::vector<int32_t> & out_ids) {
    out_ids.clear();
    if (text.empty()) {
        return TRANSCRIBE_OK;
    }
    if (!tok.has_encoder()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "r2t2 stream: tokenizer missing encoder (merges unavailable)");
        return TRANSCRIBE_ERR_GGUF;
    }
    const std::string tag    = r2t2::kAsrTextTag;
    const int         tag_id = tok.find(tag);
    if (tag_id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "r2t2 stream: tokenizer vocab missing <asr_text> special token");
        return TRANSCRIBE_ERR_GGUF;
    }

    std::vector<int32_t> segment;
    size_t               pos = 0;
    while (true) {
        const size_t      hit  = text.find(tag, pos);
        const std::string part = (hit == std::string::npos) ? text.substr(pos) : text.substr(pos, hit - pos);
        if (!part.empty()) {
            // encode() clears its output, so encode into a scratch vector and
            // append — passing out_ids directly would drop every segment but
            // the last.
            if (const transcribe_status st = tok.encode(part, segment); st != TRANSCRIBE_OK) {
                return st;
            }
            out_ids.insert(out_ids.end(), segment.begin(), segment.end());
        }
        if (hit == std::string::npos) {
            break;
        }
        out_ids.push_back(tag_id);
        pos = hit + tag.size();
    }
    return TRANSCRIBE_OK;
}

// Reference decode_rollback_prefix: decode ids[0 : size-k], growing k until the
// decoded text contains no U+FFFD. A truncated token list can end mid-way
// through a multi-byte character, and committing half a character would put a
// replacement glyph in the transcript permanently (commitments are
// append-only). Growing k drops whole tokens from the end until the remainder
// decodes cleanly.
std::string decode_rollback_prefix(const Tokenizer & tok, const std::vector<int32_t> & ids, int64_t rollback) {
    int64_t k = rollback;
    while (true) {
        const int64_t end_index = std::max<int64_t>(0, static_cast<int64_t>(ids.size()) - k);
        std::string   prefix;
        if (end_index > 0) {
            prefix = r2t2::sanitize_utf8_lossy(tok.decode(ids.data(), static_cast<int>(end_index)));
        }
        if (prefix.find(r2t2::kReplacementChar) == std::string::npos) {
            return prefix;
        }
        if (end_index == 0) {
            return {};
        }
        ++k;
    }
}

// Reference build_stream_prefix: the text continuation seeded into the prompt
// for this tick, derived from the text decoded so far.
//
// Empty for the first k_unfixed_chunk_num ticks — there is nothing stable
// enough to condition on yet — and empty whenever the accumulated text is
// empty, which is also what makes auto-detect work: with no seed the model
// emits the "language X<asr_text>" envelope itself on the tick that starts
// producing a transcript.
std::string build_stream_prefix(const Tokenizer &   tok,
                                const std::string & raw_decoded,
                                int64_t             chunk_id,
                                bool                final_flush) {
    if (chunk_id < k_unfixed_chunk_num) {
        return {};
    }
    const std::string raw_truncated = r2t2::truncate_at_pipe(raw_decoded);
    const auto        ids           = [&] {
        std::vector<int32_t> v;
        std::ignore = encode_with_tag(tok, raw_truncated, v);
        return v;
    }();
    if (ids.empty()) {
        return {};
    }
    if (final_flush) {
        // The reference's finish_streaming_transcribe uses a fixed rollback
        // without the replacement-character loop and never rolls back past the
        // first token (max(1, ...)): on the last tick the whole utterance is
        // already available, so there is no next tick to protect the prefix
        // from, and holding back a token would drop it from the final text.
        const int64_t end_index = std::max<int64_t>(1, static_cast<int64_t>(ids.size()) - k_unfixed_token_num);
        return r2t2::truncate_at_pipe(r2t2::sanitize_utf8_lossy(tok.decode(ids.data(), static_cast<int>(end_index))));
    }
    int64_t k = k_unfixed_token_num;
    if (k_rollback_punctuation && r2t2::ends_with_rollback_punctuation(raw_truncated)) {
        k = 0;
    }
    return r2t2::truncate_at_pipe(decode_rollback_prefix(tok, ids, k));
}

// Byte length of the longest common prefix of `a` and `b`, floored to a UTF-8
// boundary of `a`.
//
// Why this instead of just fixed_text.size(): the dispatcher only commits a
// candidate that is a byte prefix of full_text, because that is what makes
// "committed bytes are never rewritten" true. The reference derives fixed_text
// from raw_decoded_ (envelope included) and text_ from the parsed form, which
// agree in practice but are produced by different pipelines, and a disagreement
// would otherwise make the candidate inert — every tick silently committing
// nothing, which a caller sees as a stream that never produces text. Measuring
// the agreement keeps the invariant true by construction: in the normal case
// this returns fixed_text.size() exactly as the reference would, and in the
// abnormal case it commits only what the two agree on instead of stalling.
size_t agreed_prefix_bytes(const std::string & a, const std::string & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t       i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    while (i > 0 && (static_cast<unsigned char>(a[i]) & 0xC0u) == 0x80u) {
        --i;
    }
    return i;
}

// ---------------------------------------------------------------------------
// One decode tick
// ---------------------------------------------------------------------------

struct TickOutcome {
    bool        produced_text = false;  // text_ holds a transcript this tick
    std::string fixed_text;             // stable prefix of text_, tag-stripped
};

// Reference decode_stream_chunk + the text repair above it. Re-encodes the
// whole accumulated audio, re-decodes the prompt with the current stable
// prefix as a continuation, then rebuilds raw_decoded_/text_/fixed_text.
//
// `params` is deliberately null: the only field run_decode_pass reads from it
// is spec_k_drafts, and this path drafts from `draft_tail` instead — the
// previous tick's own hold-back, which is a far better guess than the 1-gram
// lookup because the tokens it aims at are ones the model produced a moment ago
// (see the note above draft_enabled). The family default is what a null params
// selects, and the seed overrides it in any case.
transcribe_status decode_tick(QwenAsrSession * cc, QwenAsrModel * cm, bool final_flush, TickOutcome * out) {
    R2T2StreamState & st = cc->r2t2;

    const std::string prefix = build_stream_prefix(cm->tok, st.raw_decoded, st.chunk_id, final_flush);

    // Continuation prompt: the forced-language seed (empty in auto-detect),
    // then the stable prefix as text. Both are token-level appendages after
    // the assistant header, which is exactly the slot run_decode_pass exposes
    // as `suffix_ids`.
    std::vector<int32_t> suffix_ids = st.prompt_seed_ids;
    if (!prefix.empty()) {
        std::vector<int32_t> prefix_ids;
        if (const transcribe_status st_enc = encode_with_tag(cm->tok, prefix, prefix_ids); st_enc != TRANSCRIBE_OK) {
            return st_enc;
        }
        suffix_ids.insert(suffix_ids.end(), prefix_ids.begin(), prefix_ids.end());
    }

    DecodePassResult        pass;
    const transcribe_status decode_status = run_decode_pass(
        cc, st.audio_accum.data(), static_cast<int>(st.audio_accum.size()),
        /*params=*/nullptr, suffix_ids.empty() ? nullptr : &suffix_ids, k_max_new_tokens, &pass,
        draft_enabled() && !st.draft_tail.empty() ? &st.draft_tail : nullptr, &st.enc_cache, &st.mel_stream,
        /*kv_reuse=*/true);
    if (decode_status != TRANSCRIBE_OK) {
        return decode_status;
    }

    // TRANSCRIBE_R2T2_TRACE: one line per tick with the generated ids and the
    // rollback tail. The tail is computed below for the next tick's seed; the
    // log line is what makes the draft's acceptance measurable from outside
    // (`tail[i]` versus `gen[i+1]`) rather than only inferable from timings.
    const bool trace = transcribe::env::flag("TRANSCRIBE_R2T2_TRACE");

    std::string generated = r2t2::normalize_punct_by_context(pass.raw_text);
    generated             = r2t2::sanitize_utf8_lossy(generated);
    // Drop replacement characters outright, mirroring the reference's
    // .replace('�', ''): a glyph the model hallucinated from a partial
    // token must not enter raw_decoded_, because raw_decoded_ is re-encoded
    // into the next prompt and would feed the error forward.
    for (size_t pos = 0; (pos = generated.find(r2t2::kReplacementChar, pos)) != std::string::npos;) {
        generated.erase(pos, std::char_traits<char>::length(r2t2::kReplacementChar));
    }

    st.raw_decoded = prefix + generated;

    std::string detected;
    if (st.force_language.empty()) {
        detected = r2t2::parse_language_output(st.raw_decoded, std::string()).language;
    }
    if (st.force_language == "Chinese" || detected == "Chinese") {
        st.raw_decoded = r2t2::remove_spaces_between_chinese(st.raw_decoded);
    }

    const auto parsed = r2t2::parse_asr_output(st.raw_decoded, st.force_language);
    // Re-attach the envelope the parse stripped, so the next tick's
    // continuation prompt still carries the language tag. Without this the
    // re-decoded prompt would lose the tag after the first tick and the model
    // would start emitting a second envelope.
    if (r2t2::contains_asr_text_tag(st.raw_decoded)) {
        st.raw_decoded = r2t2::text_before_asr_tag(st.raw_decoded) + r2t2::kAsrTextTag + parsed.text;
    } else {
        st.raw_decoded = parsed.text;
    }
    st.raw_decoded = r2t2::truncate_at_pipe(st.raw_decoded);

    // Stable prefix, in token space: hold back k_unfixed_token_num tokens from
    // the end, or nothing at all when the text ends in punctuation (with
    // k_rollback_punctuation off, the hold-back always applies) or when the tag
    // has only just been emitted and there is no transcript after it yet.
    std::vector<int32_t> current_ids;
    if (const transcribe_status st_enc = encode_with_tag(cm->tok, st.raw_decoded, current_ids);
        st_enc != TRANSCRIBE_OK) {
        return st_enc;
    }
    int64_t k = k_unfixed_token_num;
    if (k_rollback_punctuation && r2t2::ends_with_rollback_punctuation(st.raw_decoded)) {
        k = 0;
    }
    if (r2t2::contains_asr_text_tag(st.raw_decoded) && r2t2::text_after_asr_tag(st.raw_decoded).empty()) {
        k = 0;
    }
    std::string fixed_text = decode_rollback_prefix(cm->tok, current_ids, k);
    if (r2t2::contains_asr_text_tag(fixed_text)) {
        fixed_text = r2t2::text_after_asr_tag(fixed_text);
    } else if (st.force_language.empty()) {
        // In auto-detect mode everything before the tag is metadata, not
        // transcript. The rollback can strip the tag itself off the end, so
        // gate on the tag being absent from the PREFIX rather than from the
        // rebuilt raw_decoded_ above.
        fixed_text.clear();
    }
    fixed_text = r2t2::truncate_at_pipe(fixed_text);

    // The next tick's speculation seed: the tokens held back just above, which
    // are the ones the next tick's decode re-derives. Rebuilt from the token
    // list the prefix was cut from, so it is the same window `k` describes and
    // not a re-encoding that could split differently. Empty when nothing is
    // held back (k == 0), which correctly disables drafting for a tick that
    // committed its whole tail: there is then no window the next tick repeats.
    const size_t tail_from = (k > 0 && static_cast<size_t>(k) < current_ids.size()) ?
                                 current_ids.size() - static_cast<size_t>(k) :
                                 current_ids.size();

    if (trace) {
        const auto join = [](const std::vector<int32_t> & v, size_t from) {
            std::string s;
            for (size_t i = from; i < v.size(); ++i) {
                s += std::to_string(v[i]);
                s += ' ';
            }
            return s;
        };
        log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "r2t2 trace: tick=%lld fixed=%zu gen=[ %s] tail=[ %s]",
                static_cast<long long>(st.chunk_id), fixed_text.size(), join(pass.gen_ids, 0).c_str(),
                join(current_ids, tail_from).c_str());
    }
    st.draft_tail.assign(current_ids.begin() + static_cast<ptrdiff_t>(tail_from), current_ids.end());

    if (!r2t2::contains_asr_text_tag(st.raw_decoded) && st.force_language.empty()) {
        // Auto-detect and the model has not committed to a language yet:
        // there is no transcript, so nothing to display and nothing to commit.
        // chunk_id deliberately does NOT advance here (matching the reference),
        // which keeps build_stream_prefix returning empty for the ticks that
        // have not produced an envelope yet.
        st.text.clear();
        // No commit means no commit boundary, so the window above is not one
        // the next tick repeats: drop the seed rather than aim the next pass at
        // a boundary that did not move.
        st.draft_tail.clear();
        out->produced_text = false;
        out->fixed_text.clear();
        return TRANSCRIBE_OK;
    }

    st.language = parsed.language;
    st.text     = r2t2::truncate_at_pipe(parsed.text);
    ++st.chunk_id;

    out->produced_text = true;
    out->fixed_text    = std::move(fixed_text);
    return TRANSCRIBE_OK;
}

// Publish the tick's output to the session: full_text, the family-published
// commit boundary, and has_result.
void publish_state(QwenAsrSession * cc, const TickOutcome & tick) {
    R2T2StreamState & st = cc->r2t2;

    cc->full_text   = st.text;
    cc->raw_text    = st.text;
    cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    // Only claim a result once a tick has actually produced transcript text,
    // and then stay claimed: a later tick can legitimately publish an empty
    // text_ (auto-detect mode clearing the display until the language tag
    // appears), and dropping has_result there would make the dispatcher report
    // the stream as having gone from a transcript back to no transcript at all.
    // has_result is itself per-utterance state — the dispatcher clears it
    // before stream_begin — so it is the sticky flag, with no second copy to
    // keep in sync.
    cc->has_result  = cc->has_result || tick.produced_text;

    // Report the detected language in the same BCP-47 spelling the offline path
    // uses, and — like offline — only when the caller gave no hint: the field
    // reports what the model chose, not what the caller asked for. The internal
    // `language` stays canonical because the r2t2:: text layer matches on it.
    if (st.force_language.empty() && !st.language.empty()) {
        if (const char * bcp47 = bcp47_for_publisher_name(st.language); bcp47 != nullptr) {
            cc->detected_language = bcp47;
        }
    }

    // The candidate must index full_text, so it is measured against it rather
    // than taken from fixed_text directly (see agreed_prefix_bytes).
    st.committed_bytes                = agreed_prefix_bytes(st.text, tick.fixed_text);
    cc->stream_family_committed_bytes = st.committed_bytes;

    // One segment per stream, refreshed in place so a caller polling
    // transcribe_segments during a stream sees the current text rather than the
    // previous tick's. Words/tokens are intentionally not populated: the
    // reference's output carries no per-word alignment for this family.
    cc->segments.clear();
    if (cc->has_result) {
        transcribe_session::SegmentEntry seg{};
        seg.text  = st.text;
        seg.t0_ms = 0;
        seg.t1_ms = st.audio_input_samples * 1000 / 16000;
        cc->segments.push_back(std::move(seg));
    }
}

// Fill the audio cursors. `audio_committed_samples` is what has been *decoded*,
// not what has been committed as text: a decode tick consumes its audio even
// when the tick commits nothing, and reporting otherwise would make the
// buffered_ms a caller renders look like audio is stuck.
void fill_update(const QwenAsrSession * cc, transcribe_stream_update * update, bool changed, bool is_final) {
    if (update == nullptr) {
        return;
    }
    const R2T2StreamState & st = cc->r2t2;
    update->result_changed     = update->result_changed || changed;
    update->revision           = cc->stream_revision;
    update->input_received_ms  = st.audio_input_samples / 16;
    update->audio_committed_ms = st.audio_committed_samples / 16;
    update->buffered_ms        = (st.audio_input_samples - st.audio_committed_samples) / 16;
    update->is_final           = is_final;
}

// Keep the session-level audio cursors the dispatcher's silence fast path
// reads. That path skips a frame (it never reaches the family) only when
// stream_audio_committed_us == stream_audio_input_us, i.e. when the family has
// nothing buffered. R2T2 used to leave both at zero, so the equality held
// permanently: whenever tentative text happened to be empty at a pause, the
// quiet frames after it were skipped, ticks stopped, and the words after the
// pause were never decoded — which, depending on how the chunk size lined up
// with the pause, looked like a 3-5x "faster" stream at some cadences.
//
// `fed_samples` is what this call handed the family (0 on finalize). Input
// advances by it on the dispatcher's own timeline, which also counts the
// frames the fast path skipped; committed trails input by exactly the audio
// still waiting in st.buffer, so the two are equal only when nothing is
// buffered.
void sync_session_cursors(QwenAsrSession * cc, int64_t fed_samples) {
    const R2T2StreamState & st = cc->r2t2;
    cc->stream_audio_input_us += fed_samples * 1000000LL / 16000LL;
    const int64_t buffered_us     = static_cast<int64_t>(st.buffer.size()) * 1000000LL / 16000LL;
    cc->stream_audio_committed_us = cc->stream_audio_input_us - buffered_us;
}

// True when the accumulated audio is long enough for the mel front-end to
// produce frames.
//
// Needed because a caller may finalize a stream that never received usable
// audio (feed nothing, or feed a few samples), and the decoder cannot run on
// that. Decoding anyway would surface the front-end's INVALID_ARG as a failed
// stream, when the correct answer is an empty transcript — the stream simply
// had no speech in it. MelFrontend::compute requires at least two frames.
bool has_decodable_audio(const QwenAsrModel * cm, const std::vector<float> & audio) {
    if (!cm->mel.has_value() || audio.empty()) {
        return false;
    }
    return cm->mel->n_frames_for(audio.size()) >= 2;
}

// Advance the stream by consuming the buffered audio. Shared by feed
// (non-final) and finalize (final, and drains a partial trailing chunk).
//
// One tick takes every whole chunk that is buffered, not one chunk per tick.
// A tick re-encodes and re-prefills the whole accumulated utterance, so its
// cost is fixed by how much audio it covers and barely moves with how many
// chunks it advances; running a tick per buffered chunk therefore pays that
// cost once per chunk to derive text that the tick over the last of them
// supersedes outright (each tick re-decodes from the same held-back stable
// prefix, so an intermediate one is a strictly weaker version of the next).
// Taking the backlog in one tick is the same text for a fraction of the work,
// and it is what lets a caller whose cadence the machine cannot sustain fall
// behind by a bounded lag — one tick's worth — instead of by a lag that grows
// with the utterance, which is what "the stream is slow" looks like from the
// outside: text arrives far behind the speaker and stops late after they do.
transcribe_status drain_chunks(QwenAsrSession * cc, bool final_flush, transcribe_stream_update * update) {
    auto *            cm      = static_cast<QwenAsrModel *>(cc->model);
    R2T2StreamState & st      = cc->r2t2;
    const size_t      chunk   = static_cast<size_t>(st.chunk_size_samples);
    bool              changed = false;
    transcribe_status status  = TRANSCRIBE_OK;

    while (!st.buffer.empty()) {
        const size_t whole = (st.buffer.size() / chunk) * chunk;
        // A non-final tick needs a whole chunk before it can advance at all; a
        // final flush takes whatever is left, partial chunk included, because
        // the utterance has ended and the remaining samples are all there is.
        if (whole == 0 && !final_flush) {
            break;
        }
        const size_t take = final_flush ? st.buffer.size() : whole;

        st.audio_accum.insert(st.audio_accum.end(), st.buffer.begin(),
                              st.buffer.begin() + static_cast<ptrdiff_t>(take));
        st.buffer.erase(st.buffer.begin(), st.buffer.begin() + static_cast<ptrdiff_t>(take));
        st.audio_committed_samples += static_cast<int64_t>(take);

        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }

        // Too little audio for the front-end to produce frames (only reachable
        // on a final flush of a nearly-empty stream). The audio is already
        // accounted as consumed; there is simply nothing to decode.
        if (!has_decodable_audio(cm, st.audio_accum)) {
            continue;
        }

        const int64_t t_tick_start = ggml_time_us();
        TickOutcome   tick;
        status = decode_tick(cc, cm, final_flush, &tick);
        if (status != TRANSCRIBE_OK) {
            return status;
        }
        const int64_t tick_us = ggml_time_us() - t_tick_start;

        const std::string prev_text = st.text;
        publish_state(cc, tick);
        changed = changed || st.text != prev_text;

        // Per-tick diagnostics: this is the shape a streaming regression shows
        // up in (cost per tick growing faster than the audio does, or text
        // that stops advancing because the committed prefix stalled).
        log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                "r2t2 stream: tick=%lld audio=%.2fs fixed=%zu/%zu text=%zu bytes "
                "lang=%s commit=%zu%% tick_us=%lld%s",
                static_cast<long long>(st.chunk_id), static_cast<double>(st.audio_accum.size()) / 16000.0,
                tick.fixed_text.size(), st.text.size(), st.raw_decoded.size(),
                st.language.empty() ? "(auto)" : st.language.c_str(),
                st.text.empty() ? 0u : static_cast<size_t>(100 * st.committed_bytes / st.text.size()),
                static_cast<long long>(tick_us), final_flush ? " final" : "");

        // The loop needs no break of its own: it ends when the buffer holds
        // less than a whole chunk (non-final) or is empty (final flush).
    }

    if (update != nullptr) {
        update->result_changed = update->result_changed || changed;
    }
    return TRANSCRIBE_OK;
}

}  // namespace

// ---------------------------------------------------------------------------
// Extension surface
// ---------------------------------------------------------------------------

// The cadence bounds live in r2t2-package.h: both endpoints are directly
// selectable and every integer in between is legal, because at 16 kHz one
// millisecond is exactly 16 samples, so the caller's value maps to a sample
// count with no rounding and no hidden quantization to a preset. They are the
// R2T2 contract's numbers (docs/porting/families/confucius4_r2t2.md, "Streaming
// control contract"), not a tuning choice made here.

// Resolve a caller's stream extension into the cadence to run at. Pure: it
// reads the extension and writes nothing else, so it is safe from both
// stream_validate (before the result snapshot is cleared) and stream_begin.
//
// An absent extension is not an error — it selects the default. A present one
// with the wrong kind or a short struct is rejected by transcribe_ext_check
// with the generic BAD_STRUCT_SIZE / INVALID_ARG the other families return.
transcribe_status resolve_r2t2_stream_ext(const struct transcribe_stream_params * stream_params,
                                          uint32_t *                              out_chunk_ms) {
    uint32_t               chunk_ms = k_r2t2_chunk_ms_default;
    const transcribe_ext * fam      = (stream_params != nullptr) ? stream_params->family : nullptr;
    if (const transcribe_status st =
            transcribe_ext_check(fam, TRANSCRIBE_EXT_KIND_R2T2_STREAM, sizeof(transcribe_r2t2_stream_ext));
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (fam != nullptr) {
        const auto * rx = reinterpret_cast<const transcribe_r2t2_stream_ext *>(fam);
        // Out of range is rejected rather than clamped. A caller that asked
        // for 40 ms or 5000 ms has a bug or a stale preset, and silently
        // running a different cadence than the one recorded in diagnostics
        // would make the complaint impossible to reproduce.
        if (rx->chunk_size_ms < k_r2t2_chunk_ms_min || rx->chunk_size_ms > k_r2t2_chunk_ms_max) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "r2t2 stream: chunk_size_ms=%u is outside the supported %u-%u ms range",
                    rx->chunk_size_ms, k_r2t2_chunk_ms_min, k_r2t2_chunk_ms_max);
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        chunk_ms = rx->chunk_size_ms;
    }
    if (out_chunk_ms != nullptr) {
        *out_chunk_ms = chunk_ms;
    }
    return TRANSCRIBE_OK;
}

namespace {

// True when this model is the Confucius4-R2T2 package. Keyed on the variant the
// loader reported, which prepare_r2t2_metadata sets, rather than on the
// filename or the tensor layout: the variant string is what the rest of the
// runtime already uses to distinguish packages of one arch.
bool is_r2t2_model(const QwenAsrModel * cm) {
    return cm != nullptr && cm->variant == k_r2t2_variant;
}

}  // namespace

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

transcribe_status r2t2_stream_validate(const transcribe_session *       ctx,
                                       const transcribe_run_params *    run_params,
                                       const transcribe_stream_params * stream_params) {
    (void) run_params;
    if (ctx == nullptr || ctx->model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const auto * cm = static_cast<const QwenAsrModel *>(ctx->model);
    if (!is_r2t2_model(cm)) {
        // Defense in depth: the dispatcher only reaches this hook when
        // accepts_ext_kind admitted the stream extension, which is already
        // gated on the package marker.
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    if (cm->plan.scheduler_list.empty() || !cm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // Pure: vets the extension value (and rejects an out-of-range cadence)
    // without touching the stream state, so a caller typo cannot destroy the
    // previous transcript.
    uint32_t chunk_ms = 0;
    return resolve_r2t2_stream_ext(stream_params, &chunk_ms);
}

transcribe_status r2t2_stream_begin(transcribe_session *             ctx,
                                    const transcribe_run_params *    run_params,
                                    const transcribe_stream_params * stream_params) {
    if (ctx == nullptr || ctx->model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cm = static_cast<QwenAsrModel *>(ctx->model);
    auto * cc = static_cast<QwenAsrSession *>(ctx);
    if (!is_r2t2_model(cm)) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    if (cm->plan.scheduler_list.empty() || !cm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Streaming may run without ever going through run(), so init the dumper
    // here (same reason the parakeet hooks do).
    transcribe::debug::init();

    uint32_t chunk_ms = k_r2t2_chunk_ms_default;
    if (const transcribe_status st = resolve_r2t2_stream_ext(stream_params, &chunk_ms); st != TRANSCRIBE_OK) {
        return st;
    }

    R2T2StreamState & st  = cc->r2t2;
    st                    = R2T2StreamState{};
    st.chunk_size_ms      = chunk_ms;
    // Frames per tick, latched now: 16 kHz is the only rate this pipeline
    // accepts, and chunk boundaries are counted in frames, so the divisor must
    // not be re-derived from anything that can change mid-stream.
    st.chunk_size_samples = std::max<int64_t>(1, std::llround(static_cast<double>(chunk_ms) * 16000.0 / 1000.0));

    // Language hint. The caller's BCP-47 code is resolved to the canonical name
    // the text layer uses, because every r2t2:: function matches on "Chinese" /
    // "English" rather than on "zh" / "en" — a code passed through unresolved
    // would silently become an unsupported-language request inside
    // parse_asr_output.
    if (run_params != nullptr && run_params->language != nullptr && run_params->language[0] != '\0') {
        st.force_language = r2t2::resolve_language(run_params->language);
        if (const transcribe_status enc = encode_language_prefix(cm->tok, run_params->language, st.prompt_seed_ids);
            enc != TRANSCRIBE_OK) {
            return enc;
        }
    }

    // A stream is a fresh session-level result. The dispatcher has already
    // cleared its snapshot; clearing here too keeps the family's own state
    // consistent with what a caller re-reading immediately after begin sees.
    cc->stream_family_committed_bytes = 0;

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "r2t2 stream: begin chunk_size_ms=%u (%lld samples) language=%s",
            st.chunk_size_ms, static_cast<long long>(st.chunk_size_samples),
            st.force_language.empty() ? "(auto)" : st.force_language.c_str());
    return TRANSCRIBE_OK;
}

transcribe_status r2t2_stream_feed(transcribe_session *       ctx,
                                   const float *              pcm,
                                   int                        n_samples,
                                   transcribe_stream_update * update) {
    if (ctx == nullptr || ctx->model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<QwenAsrSession *>(ctx);
    auto * cm = static_cast<QwenAsrModel *>(ctx->model);
    if (!is_r2t2_model(cm)) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }

    R2T2StreamState & st = cc->r2t2;
    st.audio_input_samples += n_samples;
    st.buffer.insert(st.buffer.end(), pcm, pcm + n_samples);

    // The dispatcher's VAD fast path already handles the silence case; this
    // path just consumes audio and only pays for a decode when a whole tick is
    // buffered.
    const transcribe_status status = drain_chunks(cc, /*final_flush=*/false, update);
    // Even on failure: the samples were received, and a cursor pair left equal
    // over a non-empty buffer is exactly what let the fast path drop speech.
    sync_session_cursors(cc, n_samples);
    if (status != TRANSCRIBE_OK) {
        return status;
    }

    fill_update(cc, update, /*changed=*/false, /*is_final=*/false);
    return TRANSCRIBE_OK;
}

transcribe_status r2t2_stream_finalize(transcribe_session * ctx, transcribe_stream_update * update) {
    if (ctx == nullptr || ctx->model == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<QwenAsrSession *>(ctx);
    auto * cm = static_cast<QwenAsrModel *>(ctx->model);
    if (!is_r2t2_model(cm)) {
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }

    R2T2StreamState & st = cc->r2t2;

    // Flush the trailing partial chunk through one final decode. Skipping it
    // would drop the tail of every utterance whose length is not a whole
    // number of chunks, which is nearly all of them.
    const transcribe_status status = drain_chunks(cc, /*final_flush=*/true, update);
    sync_session_cursors(cc, 0);
    if (status != TRANSCRIBE_OK) {
        return status;
    }

    // A stream that never produced a decodable tick (too little audio) still
    // has to leave a well-formed empty result rather than a stale one.
    if (!cc->has_result) {
        cc->full_text   = st.text;
        cc->raw_text    = st.text;
        cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
        cc->has_result  = true;
        cc->segments.clear();
        transcribe_session::SegmentEntry seg{};
        seg.text  = st.text;
        seg.t0_ms = 0;
        seg.t1_ms = st.audio_input_samples * 1000 / 16000;
        cc->segments.push_back(std::move(seg));
    }

    // The whole remaining text is committed by the dispatcher's
    // finalize_committed_text, so publishing the full length here makes the
    // final feed's tentative_text drain to nothing exactly once.
    cc->stream_family_committed_bytes = st.text.size();

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "r2t2 stream: finalize ticks=%lld audio=%.2fs text=%zu bytes committed=%zu",
            static_cast<long long>(st.chunk_id), static_cast<double>(st.audio_accum.size()) / 16000.0, st.text.size(),
            cc->stream_family_committed_bytes);

    fill_update(cc, update, /*changed=*/false, /*is_final=*/true);
    return TRANSCRIBE_OK;
}

void r2t2_stream_reset(transcribe_session * ctx) {
    if (ctx == nullptr) {
        return;
    }
    auto * cc                         = static_cast<QwenAsrSession *>(ctx);
    // Assignment, not a field-by-field clear: the state is plain data with no
    // GPU handles, and re-assigning a fresh instance cannot miss a field the
    // way a hand-written clear list can (the reason clear_result's own list is
    // documented field by field).
    cc->r2t2                          = R2T2StreamState{};
    // The KV cache and scheduler are session scratch shared with the offline
    // path; run_decode_pass resizes and clears the cache per pass, so releasing
    // them here would only cost the next stream a re-allocation.
    cc->stream_family_committed_bytes = 0;
}

bool r2t2_accepts_ext_kind(const transcribe_model * model, transcribe_ext_slot slot, uint32_t kind) {
    if (model == nullptr || slot != TRANSCRIBE_EXT_SLOT_STREAM) {
        return false;
    }
    // Per-variant: only the R2T2 package streams. The other qwen3_asr variants
    // have no streaming surface at all and must report false rather than
    // NULL'ing the hook (transcribe-arch.h: NULL is reserved for "no extension
    // surface at all").
    return is_r2t2_model(static_cast<const QwenAsrModel *>(model)) && kind == TRANSCRIBE_EXT_KIND_R2T2_STREAM;
}

}  // namespace transcribe::qwen3_asr

// Public streaming-extension initializer (C ABI). Mirrors the per-family
// pattern used by parakeet / sortformer / voxtral_realtime: the definition
// deliberately carries no TRANSCRIBE_API, because in this source set that
// macro expands to dllimport (this file is not defining the core library) and
// the symbol is exported by the module's own export list instead.
extern "C" void transcribe_r2t2_stream_ext_init(struct transcribe_r2t2_stream_ext * ext) {
    if (ext == nullptr) {
        return;
    }
    std::memset(ext, 0, sizeof(*ext));
    ext->ext.size      = sizeof(*ext);
    ext->ext.kind      = TRANSCRIBE_EXT_KIND_R2T2_STREAM;
    ext->chunk_size_ms = transcribe::qwen3_asr::k_r2t2_chunk_ms_default;
}
