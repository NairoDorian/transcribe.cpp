// transcribe-earshot.cpp - Earshot voice activity detector: the frame-level
// speech gate (VoiceActivityDetector, used by the ASR stream activity gate)
// and the transcribe_vad_* frame C API, on top of the SIMD scoring engine in
// transcribe-earshot-engine.*.
#include "transcribe-earshot.h"

#include "transcribe-earshot-engine.h"
#include "transcribe-log.h"
#include "transcribe-session.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

namespace transcribe {

struct VoiceActivityDetector::Impl {
    earshot::Detector detector;

    void reset() noexcept { detector.reset(); }

    float predict_f32(std::span<const float, 256> frame) noexcept { return detector.predict(frame.data()); }
};

namespace {

// The accepted threshold range; anything else (NaN included, which compares
// false both ways and would leave the gate unable to ever enter speech) takes
// the default.
float sanitize_threshold(float threshold) noexcept {
    return threshold >= 0.05f && threshold <= 0.95f ? threshold : 0.50f;
}

}  // namespace

VoiceActivityDetector::VoiceActivityDetector(float threshold, uint32_t /*prefill_ms*/, uint32_t hangover_ms) :
    impl_(new Impl()),
    threshold_(sanitize_threshold(threshold)),
    enter_threshold_(threshold_),
    exit_threshold_(std::max(0.05f, threshold_ * 0.70f)),
    hangover_frames_(std::max(1u, hangover_ms / 16u)) {}

VoiceActivityDetector::~VoiceActivityDetector() {
    delete impl_;
}

VoiceActivityDetector::VoiceActivityDetector(VoiceActivityDetector && other) noexcept :
    impl_(other.impl_),
    threshold_(other.threshold_),
    enter_threshold_(other.enter_threshold_),
    exit_threshold_(other.exit_threshold_),
    in_speech_(other.in_speech_),
    accumulated_speech_ms_(other.accumulated_speech_ms_),
    last_score_(other.last_score_),
    last_level_dbfs_(other.last_level_dbfs_),
    hangover_frames_(other.hangover_frames_),
    hangover_rem_(other.hangover_rem_) {
    other.impl_ = nullptr;
}

VoiceActivityDetector & VoiceActivityDetector::operator=(VoiceActivityDetector && other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_                  = other.impl_;
        threshold_             = other.threshold_;
        enter_threshold_       = other.enter_threshold_;
        exit_threshold_        = other.exit_threshold_;
        in_speech_             = other.in_speech_;
        accumulated_speech_ms_ = other.accumulated_speech_ms_;
        last_score_            = other.last_score_;
        last_level_dbfs_       = other.last_level_dbfs_;
        hangover_frames_       = other.hangover_frames_;
        hangover_rem_          = other.hangover_rem_;
        other.impl_            = nullptr;
    }
    return *this;
}

float VoiceActivityDetector::predict_f32(FrameSpan frame) noexcept {
    if (!impl_) {
        return 0.0f;
    }
    // Branchless clamp [-1.0, 1.0]
    alignas(32) float clamped[FRAME_SAMPLES];
    for (size_t i = 0; i < FRAME_SAMPLES; ++i) {
        clamped[i] = std::clamp(frame[i], -1.0f, 1.0f);
    }
    return impl_->predict_f32(std::span<const float, 256>(clamped));
}

std::expected<bool, VadError> VoiceActivityDetector::process_frame(FrameSpan frame, float * out_score) noexcept {
    if (!impl_) {
        return std::unexpected(VadError::Uninitialized);
    }
    for (size_t i = 0; i < FRAME_SAMPLES; ++i) {
        if (!std::isfinite(frame[i])) {
            return std::unexpected(VadError::InvalidInput);
        }
    }

    // Compute RMS dBFS
    double sum_sq = 0.0;
    for (size_t i = 0; i < FRAME_SAMPLES; ++i) {
        sum_sq += static_cast<double>(frame[i]) * static_cast<double>(frame[i]);
    }
    double mean_sq   = sum_sq / 256.0;
    last_level_dbfs_ = (mean_sq > 1e-12) ? static_cast<float>(10.0 * std::log10(mean_sq)) : -120.0f;

    // The network runs on every frame. It used to be skipped below -45 dBFS
    // while not in speech, which saved little and cost a lot: Earshot
    // normalizes the log-mel features per frame, so it detects quiet speech,
    // but a quiet microphone never reached it (a -62 dBFS clip streamed to an
    // empty transcript), and the skipped frames left its recurrent state stale.
    const float score = predict_f32(frame);
    last_score_       = score;
    if (out_score) {
        *out_score = score;
    }

    // Dual-threshold hysteresis gate
    if (!in_speech_) {
        if (score >= enter_threshold_) {
            in_speech_    = true;
            hangover_rem_ = hangover_frames_;
            accumulated_speech_ms_ += 16;
        }
    } else {
        if (score >= exit_threshold_) {
            hangover_rem_ = hangover_frames_;
            accumulated_speech_ms_ += 16;
        } else if (hangover_rem_ > 0) {
            --hangover_rem_;
            accumulated_speech_ms_ += 16;
        } else {
            in_speech_ = false;
        }
    }
    return in_speech_;
}

void VoiceActivityDetector::reset() noexcept {
    if (impl_) {
        impl_->reset();
    }
    in_speech_             = false;
    accumulated_speech_ms_ = 0;
    last_score_            = 0.0f;
    last_level_dbfs_       = -100.0f;
    hangover_rem_          = 0;
}

void VoiceActivityDetector::set_threshold(float threshold) noexcept {
    threshold_       = std::isnan(threshold) ? threshold_ : std::clamp(threshold, 0.05f, 0.95f);
    enter_threshold_ = threshold_;
    exit_threshold_  = std::max(0.05f, threshold_ * 0.70f);
}

}  // namespace transcribe

// ---------------------------------------------------------------------------
// Public C ABI Export Functions
// ---------------------------------------------------------------------------
extern "C" {

struct transcribe_vad * transcribe_vad_init(float threshold) {
    try {
        return new transcribe_vad(threshold);
    } catch (...) {
        return nullptr;
    }
}

void transcribe_vad_free(struct transcribe_vad * vad) {
    delete vad;
}

float transcribe_vad_predict_frame(struct transcribe_vad * vad, const float * frame_256) {
    if (!vad || !frame_256) {
        return 0.0f;
    }
    // The clamp below does not catch NaN (std::clamp passes it through), and
    // one NaN reaching the network would poison its recurrent state for every
    // later frame. Refuse the frame instead, as process_frame does.
    for (size_t i = 0; i < transcribe::VoiceActivityDetector::FRAME_SAMPLES; ++i) {
        if (!std::isfinite(frame_256[i])) {
            return 0.0f;
        }
    }
    return vad->detector.predict_f32(std::span<const float, 256>(frame_256, 256));
}

bool transcribe_vad_process_frame(struct transcribe_vad * vad, const float * frame_256, float * out_score) {
    if (!vad || !frame_256) {
        if (out_score) {
            *out_score = 0.0f;
        }
        return false;
    }
    auto res = vad->detector.process_frame(std::span<const float, 256>(frame_256, 256), out_score);
    if (!res.has_value() && out_score) {
        *out_score = 0.0f;  // a non-finite sample: the frame was not scored
    }
    return res.value_or(false);
}

void transcribe_vad_set_threshold(struct transcribe_vad * vad, float threshold) {
    if (vad) {
        vad->detector.set_threshold(threshold);
    }
}

void transcribe_vad_reset(struct transcribe_vad * vad) {
    if (vad) {
        vad->detector.reset();
    }
}

transcribe_status transcribe_stream_set_vad_threshold(struct transcribe_session * session, float threshold) {
    if (!session) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (!(threshold >= 0.05f && threshold <= 0.95f)) {  // NaN fails both comparisons
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    session->stream_vad.set_threshold(threshold);
    session->vad_threshold = threshold;
    return TRANSCRIBE_OK;
}
}
