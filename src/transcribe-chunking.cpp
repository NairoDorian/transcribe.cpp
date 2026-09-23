// transcribe-chunking.cpp - Long-audio chunk planning and timeline rebasing.

#include "transcribe-chunking.h"

#include "transcribe-activity.h"
#include "transcribe-log.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace transcribe::chunking {

std::vector<AudioChunkSpan> plan_audio_chunks(int64_t input_samples, const AudioChunkSpec & spec) {
    std::vector<AudioChunkSpan> spans;
    if (input_samples <= 0 || spec.chunk_samples <= 0) {
        return spans;
    }

    const int64_t hop     = (spec.hop_samples > 0) ? spec.hop_samples : spec.chunk_samples;
    const int64_t overlap = std::max<int64_t>(0, spec.chunk_samples - hop);

    int64_t start = 0;
    int64_t index = 0;

    while (start < input_samples) {
        AudioChunkSpan span;
        span.index         = index;
        span.start_sample  = start;
        span.valid_samples = std::min<int64_t>(spec.chunk_samples, input_samples - start);

        // Compute keep window:
        // First chunk keeps from 0 to hop + overlap / 2.
        // Intermediate chunks keep from start + overlap / 2 to start + hop + overlap / 2.
        // Last chunk keeps up to input_samples.
        if (index == 0) {
            span.keep_start_sample = 0;
        } else {
            span.keep_start_sample = start + overlap / 2;
        }

        if (start + hop >= input_samples) {
            span.keep_end_sample = input_samples;
        } else {
            span.keep_end_sample = start + hop + overlap / 2;
        }

        spans.push_back(span);
        index++;
        start += hop;
    }

    return spans;
}

std::vector<AudioChunkSpan> plan_quiet_energy_audio_chunks(const float * pcm,
                                                           int64_t       n_samples,
                                                           int64_t       target_chunk_samples,
                                                           int64_t       search_window_samples,
                                                           float         dbfs_threshold) {
    std::vector<AudioChunkSpan> spans;
    if (pcm == nullptr || n_samples <= 0 || target_chunk_samples <= 0) {
        return spans;
    }

    if (n_samples <= target_chunk_samples) {
        AudioChunkSpan span;
        span.index             = 0;
        span.start_sample      = 0;
        span.valid_samples     = n_samples;
        span.keep_start_sample = 0;
        span.keep_end_sample   = n_samples;
        spans.push_back(span);
        return spans;
    }

    const double  threshold_energy = std::pow(10.0, static_cast<double>(dbfs_threshold) / 10.0);
    const int64_t frame_size       = 800;  // 50ms at 16 kHz
    const int64_t frame_step       = 400;  // 25ms step

    int64_t chunk_start = 0;
    int64_t index       = 0;

    while (chunk_start < n_samples) {
        const int64_t remaining = n_samples - chunk_start;
        if (remaining <= target_chunk_samples) {
            AudioChunkSpan span;
            span.index             = index++;
            span.start_sample      = chunk_start;
            span.valid_samples     = remaining;
            span.keep_start_sample = chunk_start;
            span.keep_end_sample   = n_samples;
            spans.push_back(span);
            break;
        }

        // Search for natural quiet cut point in [chunk_start + target_chunk_samples - search_window_samples, chunk_start + target_chunk_samples]
        const int64_t search_start =
            std::max<int64_t>(chunk_start + frame_size, chunk_start + target_chunk_samples - search_window_samples);
        const int64_t search_end = std::min<int64_t>(n_samples - frame_size, chunk_start + target_chunk_samples);

        int64_t best_cut_sample = chunk_start + target_chunk_samples;
        double  min_energy      = std::numeric_limits<double>::infinity();
        bool    found_silence   = false;

        for (int64_t s = search_start; s + frame_size <= search_end; s += frame_step) {
            double sum_sq = 0.0;
            for (int64_t k = 0; k < frame_size; ++k) {
                const double v = pcm[s + k];
                sum_sq += v * v;
            }
            const double energy = sum_sq / static_cast<double>(frame_size);
            if (energy < threshold_energy) {
                // Found silence: cut in the middle of this quiet frame
                best_cut_sample = s + frame_size / 2;
                found_silence   = true;
                break;
            }
            if (energy < min_energy) {
                min_energy      = energy;
                best_cut_sample = s + frame_size / 2;
            }
        }

        // If no silence was hit, best_cut_sample holds the minimum energy dip in the window
        (void) found_silence;

        AudioChunkSpan span;
        span.index             = index++;
        span.start_sample      = chunk_start;
        span.valid_samples     = best_cut_sample - chunk_start;
        span.keep_start_sample = chunk_start;
        span.keep_end_sample   = best_cut_sample;
        spans.push_back(span);

        chunk_start = best_cut_sample;
    }

    return spans;
}

void rebase_tokens(std::vector<transcribe_session::TokenEntry> & tokens,
                   int64_t                                       offset_ms,
                   int                                           seg_offset,
                   int                                           word_offset) {
    for (auto & t : tokens) {
        t.t0_ms += offset_ms;
        t.t1_ms += offset_ms;
        t.seg_index += seg_offset;
        if (t.word_index >= 0) {
            t.word_index += word_offset;
        }
    }
}

void rebase_words(std::vector<transcribe_session::WordEntry> & words,
                  int64_t                                      offset_ms,
                  int                                          seg_offset,
                  int                                          token_offset) {
    for (auto & w : words) {
        w.t0_ms += offset_ms;
        w.t1_ms += offset_ms;
        w.seg_index += seg_offset;
        w.first_token += token_offset;
    }
}

void rebase_segments(std::vector<transcribe_session::SegmentEntry> & segments,
                     int64_t                                         offset_ms,
                     int                                             word_offset,
                     int                                             token_offset) {
    for (auto & s : segments) {
        s.t0_ms += offset_ms;
        s.t1_ms += offset_ms;
        s.first_word += word_offset;
        s.first_token += token_offset;
    }
}

void rebase_speaker_segments(std::vector<transcribe_session::SpeakerSegmentEntry> & spk_segments, int64_t offset_ms) {
    for (auto & spk : spk_segments) {
        spk.t0_ms += offset_ms;
        spk.t1_ms += offset_ms;
    }
}

transcribe_session::ResultSet merge_chunk_results(const std::vector<transcribe_session::ResultSet> & chunk_results,
                                                  const std::vector<int64_t> & chunk_start_samples,
                                                  int                          sample_rate) {
    transcribe_session::ResultSet merged;
    if (chunk_results.empty() || sample_rate <= 0) {
        return merged;
    }

    for (size_t i = 0; i < chunk_results.size(); ++i) {
        const auto & cr = chunk_results[i];
        if (cr.status != TRANSCRIBE_OK) {
            merged.status = cr.status;
        }
        if (!cr.has_result) {
            continue;
        }

        const int64_t start_samp = (i < chunk_start_samples.size()) ? chunk_start_samples[i] : 0;
        const int64_t offset_ms  = (start_samp * 1000) / sample_rate;

        const int base_token = static_cast<int>(merged.tokens.size());
        const int base_word  = static_cast<int>(merged.words.size());
        const int base_seg   = static_cast<int>(merged.segments.size());

        // Append and rebase tokens
        for (auto tok : cr.tokens) {
            tok.t0_ms += offset_ms;
            tok.t1_ms += offset_ms;
            tok.seg_index += base_seg;
            if (tok.word_index >= 0) {
                tok.word_index += base_word;
            }
            merged.tokens.push_back(std::move(tok));
        }

        // Append and rebase words
        for (auto w : cr.words) {
            w.t0_ms += offset_ms;
            w.t1_ms += offset_ms;
            w.seg_index += base_seg;
            w.first_token += base_token;
            merged.words.push_back(std::move(w));
        }

        // Append and rebase segments
        for (auto seg : cr.segments) {
            seg.t0_ms += offset_ms;
            seg.t1_ms += offset_ms;
            seg.first_word += base_word;
            seg.first_token += base_token;
            merged.segments.push_back(std::move(seg));
        }

        // Append and rebase speaker segments
        for (auto spk : cr.speaker_segments) {
            spk.t0_ms += offset_ms;
            spk.t1_ms += offset_ms;
            merged.speaker_segments.push_back(std::move(spk));
        }

        // Concatenate text
        if (!cr.full_text.empty()) {
            if (!merged.full_text.empty() && merged.full_text.back() != ' ' && cr.full_text.front() != ' ') {
                merged.full_text += " ";
            }
            merged.full_text += cr.full_text;
        }
        if (!cr.raw_text.empty()) {
            if (!merged.raw_text.empty() && merged.raw_text.back() != ' ' && cr.raw_text.front() != ' ') {
                merged.raw_text += " ";
            }
            merged.raw_text += cr.raw_text;
        }

        if (merged.detected_language.empty() && !cr.detected_language.empty()) {
            merged.detected_language = cr.detected_language;
        }

        merged.result_kind = std::max(merged.result_kind, cr.result_kind);
        merged.has_result  = true;
        merged.t_mel_us += cr.t_mel_us;
        merged.t_encode_us += cr.t_encode_us;
        merged.t_decode_us += cr.t_decode_us;
    }

    return merged;
}

}  // namespace transcribe::chunking
