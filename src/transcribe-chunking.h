// transcribe-chunking.h - Long-audio chunk planning and timeline rebasing.
//
// INTERNAL, C++17. Not part of the public ABI.
//
// Splits long audio into manageable chunks at natural silence boundaries,
// and stitches chunk-local token/word/segment timestamps back to the global timeline.

#pragma once

#include "transcribe-session.h"
#include "transcribe.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transcribe::chunking {

enum class AudioChunkPadMode {
    Zero,
    Reflect,
};

struct AudioChunkSpec {
    int64_t           chunk_samples             = 480000;  // Default 30s at 16 kHz
    int64_t           hop_samples               = 448000;  // Default 28s hop (2s overlap)
    AudioChunkPadMode pad_mode                  = AudioChunkPadMode::Zero;
    int64_t           reflect_min_valid_samples = 0;
};

struct AudioChunkSpan {
    int64_t index             = 0;
    int64_t start_sample      = 0;  // Start sample in the source stream
    int64_t valid_samples     = 0;  // Number of valid samples in this chunk
    int64_t keep_start_sample = 0;  // Global sample where this chunk's output starts
    int64_t keep_end_sample   = 0;  // Global sample where this chunk's output ends
};

// Plan uniform overlapping chunks across an input of length `input_samples`.
std::vector<AudioChunkSpan> plan_audio_chunks(int64_t input_samples, const AudioChunkSpec & spec);

// Plan variable-length chunks cut at silence / quiet energy boundaries.
// Searches for the minimum energy or silence frame in the window [target_chunk_samples - search_window_samples, target_chunk_samples].
// If no silence is found, cuts at the minimum energy dip in the search window.
std::vector<AudioChunkSpan> plan_quiet_energy_audio_chunks(const float * pcm,
                                                           int64_t       n_samples,
                                                           int64_t       target_chunk_samples  = 480000,
                                                           int64_t       search_window_samples = 32000,
                                                           float         dbfs_threshold        = -35.0f);

// Rebase chunk-local timestamps and level indices in place.
void rebase_tokens(std::vector<transcribe_session::TokenEntry> & tokens,
                   int64_t                                       offset_ms,
                   int                                           seg_offset,
                   int                                           word_offset);

void rebase_words(std::vector<transcribe_session::WordEntry> & words,
                  int64_t                                      offset_ms,
                  int                                          seg_offset,
                  int                                          token_offset);

void rebase_segments(std::vector<transcribe_session::SegmentEntry> & segments,
                     int64_t                                         offset_ms,
                     int                                             word_offset,
                     int                                             token_offset);

void rebase_speaker_segments(std::vector<transcribe_session::SpeakerSegmentEntry> & spk_segments, int64_t offset_ms);

// Merge multiple chunk-level ResultSets into a unified, timeline-consistent ResultSet.
transcribe_session::ResultSet merge_chunk_results(const std::vector<transcribe_session::ResultSet> & chunk_results,
                                                  const std::vector<int64_t> & chunk_start_samples,
                                                  int                          sample_rate = 16000);

}  // namespace transcribe::chunking
