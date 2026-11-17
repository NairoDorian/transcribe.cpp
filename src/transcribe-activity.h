// transcribe-activity.h - native fast energy activity detection.
//
// Fast non-ML audio activity check (RMS energy in dBFS) used by the stream
// dispatcher's silence fast path. Zero heap allocations.

#pragma once

#include <cstddef>

namespace transcribe {

// True when any 16 ms window (256 samples at 16 kHz) of the buffer carries
// mean energy above threshold_dbfs; a buffer shorter than one window is
// measured whole. Windowed, not averaged over the buffer, so the answer does
// not depend on the caller's slice size: a 30 ms consonant inside a 160 ms
// slice reads ~7 dB quieter as a slice mean and would be mistaken for
// silence. Defaults to -42.0 dBFS, which separates background room silence
// from conversational speech.
bool is_audio_active(const float * pcm, size_t n_samples, float threshold_dbfs = -42.0f);

// Environment kill-switch check: TRANSCRIBE_DISABLE_ACTIVITY_GATE=1
bool is_activity_gate_disabled();

}  // namespace transcribe
