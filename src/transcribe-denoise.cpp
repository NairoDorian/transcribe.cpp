// transcribe-denoise.cpp - capture-side noise suppression (transcribe_denoise_*).
//
// The native counterpart of ZER0's denoise_frame
// (Handy_V2/src-tauri/src/audio_toolkit/audio/denoise.rs): scale to the int16
// range RNNoise expects, run the network (transcribe-rnnoise.*, a port of the
// nnnoiseless crate ZER0 runs), mix wet and dry with the dry side delayed the
// frame RNNoise's output lags, apply the speech-probability gate, scale back
// and clamp. Resampling to and from 48 kHz stays with the caller.
#include "transcribe-abi.h"
#include "transcribe-rnnoise.h"
#include "transcribe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <new>

namespace {

constexpr int      k_frame     = transcribe::rnnoise::k_frame_size;
constexpr uint32_t k_frame_ms  = 10;
constexpr float    k_i16       = 32767.0f;
// Fraction of the remaining distance the gate's gain covers per frame: about
// three frames from open to closed and back.
constexpr float    k_gate_slew = 0.5f;

static_assert(k_frame == TRANSCRIBE_DENOISE_FRAME_SAMPLES, "RNNoise frame size");

bool params_ok(const transcribe_denoise_params & p) {
    return std::isfinite(p.strength) && p.strength >= 0.0f && p.strength <= 1.0f && std::isfinite(p.vad_threshold) &&
           p.vad_threshold >= 0.0f && p.vad_threshold <= 1.0f && p.vad_grace_ms <= TRANSCRIBE_DENOISE_MAX_GRACE_MS;
}

transcribe_status read_params(const transcribe_denoise_params * params, transcribe_denoise_params & p) {
    transcribe_denoise_params_init(&p);
    if (params != nullptr) {
        if (const auto st = transcribe::check_input_struct_size(
                params->struct_size, TRANSCRIBE_FIELD_END(transcribe_denoise_params, vad_grace_ms));
            st != TRANSCRIBE_OK) {
            return st;
        }
        transcribe::copy_out_prefix(&p, params, params->struct_size, sizeof(p));
        p.struct_size = sizeof(p);
    }
    return params_ok(p) ? TRANSCRIBE_OK : TRANSCRIBE_ERR_INVALID_ARG;
}

}  // namespace

struct transcribe_denoise {
    transcribe::rnnoise::Denoiser net;
    transcribe_denoise_params     params{};
    std::array<float, k_frame>    scaled_in{};
    std::array<float, k_frame>    scaled_out{};
    std::array<float, k_frame>    dry_delay{};
    float                         gain       = 1.0f;
    uint32_t                      grace_left = 0;

    void reset() {
        net.reset();
        dry_delay.fill(0.0f);
        gain       = 1.0f;
        grace_left = 0;
    }

    // ZER0's Gate::update.
    float gate(float prob) {
        float target = 0.0f;
        if (params.vad_threshold <= 0.0f) {
            target = 1.0f;
        } else if (prob >= params.vad_threshold) {
            grace_left = params.vad_grace_ms / k_frame_ms;
            target     = 1.0f;
        } else if (grace_left > 0) {
            --grace_left;
            target = 1.0f;
        }
        gain += (target - gain) * k_gate_slew;
        if (std::abs(gain - target) < 1e-3f) {
            gain = target;
        }
        return gain;
    }

    // ZER0's denoise_frame. `in` may alias `out`.
    float process(const float * in, float * out) {
        for (int i = 0; i < k_frame; ++i) {
            scaled_in[i] = std::clamp(in[i] * k_i16, -32768.0f, 32767.0f);
        }
        const float prob = net.process_frame(scaled_in.data(), scaled_out.data());
        const float g    = gate(prob);
        const float wet  = params.strength;
        const float dry  = 1.0f - wet;
        for (int i = 0; i < k_frame; ++i) {
            out[i] = std::clamp(((dry_delay[i] * dry + scaled_out[i] * wet) / k_i16) * g, -1.0f, 1.0f);
        }
        dry_delay = scaled_in;
        return prob;
    }
};

extern "C" {

void transcribe_denoise_params_init(transcribe_denoise_params * p) {
    transcribe::init_sized(p);
    if (p != nullptr) {
        p->strength      = 1.0f;
        p->vad_threshold = 0.0f;
        p->vad_grace_ms  = 200;
    }
}

transcribe_status transcribe_denoise_init(const transcribe_denoise_params * params, transcribe_denoise ** out_denoise) {
    if (out_denoise == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out_denoise = nullptr;
    transcribe_denoise_params p;
    if (const auto st = read_params(params, p); st != TRANSCRIBE_OK) {
        return st;
    }
    try {
        auto * d     = new transcribe_denoise();
        d->params    = p;
        *out_denoise = d;
        return TRANSCRIBE_OK;
    } catch (const std::bad_alloc &) {
        return TRANSCRIBE_ERR_OOM;
    }
}

void transcribe_denoise_free(transcribe_denoise * denoise) {
    delete denoise;
}

void transcribe_denoise_reset(transcribe_denoise * denoise) {
    if (denoise != nullptr) {
        denoise->reset();
    }
}

transcribe_status transcribe_denoise_set_params(transcribe_denoise *              denoise,
                                                const transcribe_denoise_params * params) {
    if (denoise == nullptr || params == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    transcribe_denoise_params p;
    if (const auto st = read_params(params, p); st != TRANSCRIBE_OK) {
        return st;
    }
    denoise->params = p;
    return TRANSCRIBE_OK;
}

transcribe_status transcribe_denoise_process_frame(transcribe_denoise * denoise,
                                                   const float *        in,
                                                   float *              out,
                                                   float *              out_speech_prob) {
    if (denoise == nullptr || in == nullptr || out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    for (int i = 0; i < k_frame; ++i) {
        if (!std::isfinite(in[i])) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
    }
    const float prob = denoise->process(in, out);
    if (out_speech_prob != nullptr) {
        *out_speech_prob = prob;
    }
    return TRANSCRIBE_OK;
}

transcribe_status transcribe_denoise_flush(transcribe_denoise * denoise, float * out, float * out_speech_prob) {
    static const std::array<float, k_frame> silence{};
    return transcribe_denoise_process_frame(denoise, silence.data(), out, out_speech_prob);
}

}  // extern "C"
