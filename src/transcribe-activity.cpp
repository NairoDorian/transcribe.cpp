// transcribe-activity.cpp - native fast energy activity detection.

#include "transcribe-activity.h"

#include "transcribe-env.h"

#include <cmath>

namespace transcribe {

namespace {

constexpr size_t k_window = 256;  // 16 ms at 16 kHz, the VAD frame

double sum_squares(const float * pcm, size_t n) {
    // Four accumulators: independent chains the compiler can keep in flight.
    double a0 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
    double a3 = 0.0;
    size_t i  = 0;
    for (; i + 3 < n; i += 4) {
        a0 += static_cast<double>(pcm[i]) * pcm[i];
        a1 += static_cast<double>(pcm[i + 1]) * pcm[i + 1];
        a2 += static_cast<double>(pcm[i + 2]) * pcm[i + 2];
        a3 += static_cast<double>(pcm[i + 3]) * pcm[i + 3];
    }
    for (; i < n; ++i) {
        a0 += static_cast<double>(pcm[i]) * pcm[i];
    }
    return (a0 + a1) + (a2 + a3);
}

}  // namespace

bool is_activity_gate_disabled() {
    static const bool disabled = env::flag("TRANSCRIBE_DISABLE_ACTIVITY_GATE");
    return disabled;
}

bool is_audio_active(const float * pcm, size_t n_samples, float threshold_dbfs) {
    if (pcm == nullptr || n_samples == 0) {
        return false;
    }
    // energy = 10^(dBFS / 10); compared as a window sum to skip the divide.
    const double threshold_energy = std::pow(10.0, static_cast<double>(threshold_dbfs) / 10.0);
    if (n_samples <= k_window) {
        return sum_squares(pcm, n_samples) > threshold_energy * static_cast<double>(n_samples);
    }
    const double window_threshold = threshold_energy * static_cast<double>(k_window);
    size_t       start            = 0;
    for (; start + k_window <= n_samples; start += k_window) {
        if (sum_squares(pcm + start, k_window) > window_threshold) {
            return true;
        }
    }
    // A ragged tail is measured as the buffer's last whole window (overlapping
    // the previous one), so a few trailing samples are never judged alone.
    return start < n_samples && sum_squares(pcm + n_samples - k_window, k_window) > window_threshold;
}

}  // namespace transcribe
