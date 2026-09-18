// transcribe-activity.h - native fast energy activity detection.
//
// Fast non-ML audio activity finder (RMS energy in dBFS).
// Zero heap allocations on the hot path; suitable for sub-millisecond
// scanning of 16 kHz mono PCM frames to filter pure silence intervals
// before mel spectrogram or neural network processing.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace transcribe {

struct AudioActivityRegion {
    bool   has_activity = false;
    size_t start_sample = 0;
    size_t end_sample   = 0;
};

// Check if a 16 kHz mono PCM buffer contains speech energy above threshold_dbfs.
// Defaults to -42.0 dBFS, which cleanly distinguishes background room silence
// from conversational speech.
bool is_audio_active(const float * pcm, size_t n_samples, float threshold_dbfs = -42.0f);

// Find active speech region within a PCM buffer, using sliding window and margin.
AudioActivityRegion find_audio_activity_region(const float * pcm,
                                              size_t        n_samples,
                                              float         threshold_dbfs = -42.0f,
                                              float         window_seconds = 0.030f,
                                              float         margin_seconds = 0.100f,
                                              int           sample_rate_hz = 16000);

// Environment kill-switch check: TRANSCRIBE_DISABLE_ACTIVITY_GATE=1
bool is_activity_gate_disabled();

} // namespace transcribe
