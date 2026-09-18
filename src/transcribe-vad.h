// transcribe-vad.h - native Earshot minGRU voice activity detection.
#pragma once

#include "transcribe.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

namespace transcribe {

struct VadTelemetry {
    bool     is_speaking      = false;
    uint64_t speech_ms        = 0;
    float    last_score       = 0.0f;
    float    audio_level_dbfs = -100.0f;
};

enum class VadError : uint8_t {
    InvalidInput  = 1,
    Uninitialized = 2,
};

class VoiceActivityDetector {
  public:
    static constexpr size_t FRAME_SAMPLES = 256;  // 16 ms @ 16 kHz
    using FrameSpan                       = std::span<const float, FRAME_SAMPLES>;

    explicit VoiceActivityDetector(float threshold = 0.50f, uint32_t prefill_ms = 450, uint32_t hangover_ms = 1200);
    ~VoiceActivityDetector();

    VoiceActivityDetector(const VoiceActivityDetector &)             = delete;
    VoiceActivityDetector & operator=(const VoiceActivityDetector &) = delete;
    VoiceActivityDetector(VoiceActivityDetector && other) noexcept;
    VoiceActivityDetector & operator=(VoiceActivityDetector && other) noexcept;

    // Process a 256-sample frame using C++23 std::span and std::expected:
    std::expected<bool, VadError> process_frame(FrameSpan frame, float * out_score = nullptr) noexcept;

    // Raw model score prediction for a 256-sample frame:
    float predict_f32(FrameSpan frame) noexcept;

    void reset() noexcept;
    void set_threshold(float threshold) noexcept;

    [[nodiscard]] VadTelemetry telemetry() const noexcept {
        return VadTelemetry{
            .is_speaking      = in_speech_,
            .speech_ms        = accumulated_speech_ms_,
            .last_score       = last_score_,
            .audio_level_dbfs = last_level_dbfs_,
        };
    }

    [[nodiscard]] bool is_speaking() const noexcept { return in_speech_; }

    [[nodiscard]] uint64_t speech_ms() const noexcept { return accumulated_speech_ms_; }

    [[nodiscard]] float last_score() const noexcept { return last_score_; }

    [[nodiscard]] float audio_level_dbfs() const noexcept { return last_level_dbfs_; }

  private:
    struct Impl;
    Impl * impl_ = nullptr;

    float    threshold_             = 0.50f;
    float    enter_threshold_       = 0.50f;
    float    exit_threshold_        = 0.35f;
    bool     in_speech_             = false;
    uint64_t accumulated_speech_ms_ = 0;
    float    last_score_            = 0.0f;
    float    last_level_dbfs_       = -100.0f;
    uint32_t hangover_frames_       = 75;  // 1200 ms / 16 ms
    uint32_t hangover_rem_          = 0;
};

}  // namespace transcribe

struct transcribe_vad {
    transcribe::VoiceActivityDetector detector;

    explicit transcribe_vad(float threshold) : detector(threshold) {}
};
