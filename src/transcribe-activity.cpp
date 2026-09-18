// transcribe-activity.cpp - native fast energy activity detection.

#include "transcribe-activity.h"
#include "transcribe-env.h"

#include <algorithm>
#include <cmath>

namespace transcribe {

bool is_activity_gate_disabled() {
    static const bool disabled = env::flag("TRANSCRIBE_DISABLE_ACTIVITY_GATE");
    return disabled;
}

bool is_audio_active(const float * pcm, size_t n_samples, float threshold_dbfs) {
    if (pcm == nullptr || n_samples == 0) {
        return false;
    }

    // Convert dBFS threshold to linear energy (power) threshold:
    // energy = 10^(threshold_dbfs / 10)
    const double threshold_energy = std::pow(10.0, static_cast<double>(threshold_dbfs) / 10.0);

    double sum_sq = 0.0;
    size_t i      = 0;

    // 4-way unrolled accumulator for fast scalar processing
    for (; i + 3 < n_samples; i += 4) {
        const double s0 = pcm[i];
        const double s1 = pcm[i + 1];
        const double s2 = pcm[i + 2];
        const double s3 = pcm[i + 3];
        sum_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
    }
    for (; i < n_samples; ++i) {
        const double s = pcm[i];
        sum_sq += s * s;
    }

    const double mean_energy = sum_sq / static_cast<double>(n_samples);
    return mean_energy > threshold_energy;
}

AudioActivityRegion find_audio_activity_region(const float * pcm,
                                              size_t        n_samples,
                                              float         threshold_dbfs,
                                              float         window_seconds,
                                              float         margin_seconds,
                                              int           sample_rate_hz) {
    AudioActivityRegion region;
    if (pcm == nullptr || n_samples == 0 || sample_rate_hz <= 0) {
        return region;
    }

    const size_t window_frames = std::max<size_t>(1, static_cast<size_t>(std::llround(window_seconds * sample_rate_hz)));
    const size_t margin_frames = static_cast<size_t>(std::llround(margin_seconds * sample_rate_hz));
    const double threshold_energy = std::pow(10.0, static_cast<double>(threshold_dbfs) / 10.0);

    size_t first_active = n_samples;
    size_t last_active  = 0;

    for (size_t win_start = 0; win_start < n_samples; win_start += window_frames) {
        const size_t win_end = std::min(n_samples, win_start + window_frames);
        const size_t count   = win_end - win_start;
        if (count == 0) {
            continue;
        }

        double sum_sq = 0.0;
        for (size_t i = win_start; i < win_end; ++i) {
            const double s = pcm[i];
            sum_sq += s * s;
        }
        const double mean_energy = sum_sq / static_cast<double>(count);
        if (mean_energy > threshold_energy) {
            first_active = std::min(first_active, win_start);
            last_active  = std::max(last_active, win_end);
        }
    }

    if (first_active >= last_active) {
        return region;
    }

    region.has_activity = true;
    region.start_sample = (first_active > margin_frames) ? (first_active - margin_frames) : 0;
    region.end_sample   = std::min(n_samples, last_active + margin_frames);
    return region;
}

} // namespace transcribe
