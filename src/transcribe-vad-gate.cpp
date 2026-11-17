// transcribe-vad-gate.cpp - capture-side speech gate (transcribe_vad_gate_*)
// and batch Earshot scoring (transcribe_vad_score_frames).
//
// The gate is the native counterpart of ZER0's Rust capture VAD
// (Handy_V2/src-tauri/src/audio_toolkit/vad/{earshot,smoothed,mod}.rs and the
// recorder's SpeechClock), state machine for state machine, so the app can
// drop that stack without changing what it releases:
//   - EarshotVad: clamp to [-1, 1], score, two-threshold Hysteresis;
//   - SmoothedVad: onset debounce, pre-roll ring that never releases a frame
//     twice, hangover tail, tail report;
//   - SpeechClock: voiced time with deferred short pauses.
// The one deliberate difference: EarshotVad skipped the network (score 0) for
// frames below -45 dBFS while not in speech; here every frame is scored, so a
// quiet microphone can still reach speech (see transcribe-earshot.cpp).
#include "transcribe-abi.h"
#include "transcribe-earshot-engine.h"
#include "transcribe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace {

constexpr int64_t  k_frame    = transcribe::earshot::k_frame_samples;  // 16 ms
constexpr uint32_t k_frame_ms = 16;

// Whole detector frames covering `ms`, rounded up so a duration is never
// shortened (ZER0's frames_for_duration_ms).
uint32_t frames_for_ms(uint32_t ms) {
    const uint64_t num = static_cast<uint64_t>(ms) * 16000u;
    const uint64_t den = static_cast<uint64_t>(k_frame) * 1000u;
    return static_cast<uint32_t>((num + den - 1) / den);
}

bool threshold_ok(float t) {
    return std::isfinite(t) && t >= 0.05f && t <= 0.95f;
}

float derived_exit(float threshold) {
    return std::max(threshold - 0.15f, 0.01f);
}

}  // namespace

struct transcribe_vad_gate {
    struct Slot {
        bool    emitted = false;
        bool    voiced  = false;
        int32_t n_valid = 0;  // real samples (the finish frame is zero-padded)
    };

    transcribe::earshot::Detector detector;

    // Hysteresis.
    float threshold      = 0.5f;
    float exit           = 0.35f;
    bool  exit_explicit  = false;
    float exit_requested = -1.0f;  // the caller's exit_threshold when explicit
    bool  gate_open      = false;

    // The exit edge for `t`: the requested one, never above the entry edge,
    // or the derived default. Recomputed from the request on every threshold
    // change, so lowering then raising the threshold restores the caller's exit.
    float exit_for(float t) const { return exit_explicit ? std::min(exit_requested, t) : derived_exit(t); }

    // Smoother.
    uint32_t           onset_frames    = 4;
    uint32_t           preroll_frames  = 29;
    uint32_t           hangover_frames = 29;
    uint32_t           capacity        = 31;  // preroll + 2
    std::vector<float> ring;
    std::vector<Slot>  slots;
    uint32_t           head             = 0;
    uint32_t           buffered         = 0;
    uint32_t           onset_counter    = 0;
    uint32_t           hangover_counter = 0;
    bool               in_speech        = false;

    // Speech clock.
    uint32_t hold_ms            = 500;
    bool     clock_speaking     = false;
    uint32_t clock_onset        = 0;
    uint64_t pending_silence_ms = 0;
    uint64_t speech_ms          = 0;

    // Input framing and per-call output.
    std::array<float, k_frame> partial{};
    int64_t                    partial_n = 0;
    std::vector<float>         out;
    int32_t                    call_frames = 0;
    bool                       started     = false;
    bool                       ended       = false;
    bool                       last_voiced = false;
    float                      last_score  = 0.0f;
    float                      level_dbfs  = -120.0f;

    explicit transcribe_vad_gate(const transcribe_vad_gate_params & p) {
        threshold       = p.threshold;
        exit_explicit   = p.exit_threshold >= 0.0f;
        exit_requested  = p.exit_threshold;
        exit            = exit_for(p.threshold);
        onset_frames    = frames_for_ms(p.onset_ms);
        preroll_frames  = frames_for_ms(p.preroll_ms);
        hangover_frames = frames_for_ms(p.hangover_ms);
        hold_ms         = p.speech_hold_ms;
        capacity        = preroll_frames + 2;
        ring.assign(static_cast<size_t>(capacity) * k_frame, 0.0f);
        slots.assign(capacity, Slot{});
        out.reserve(static_cast<size_t>(preroll_frames + 2) * k_frame);
    }

    void reset() {
        detector.reset();
        gate_open          = false;
        head               = 0;
        buffered           = 0;
        onset_counter      = 0;
        hangover_counter   = 0;
        in_speech          = false;
        clock_speaking     = false;
        clock_onset        = 0;
        pending_silence_ms = 0;
        speech_ms          = 0;
        partial_n          = 0;
        out.clear();
        call_frames = 0;
        started     = false;
        ended       = false;
        last_voiced = false;
        last_score  = 0.0f;
        level_dbfs  = -120.0f;
    }

    void begin_call() {
        out.clear();
        call_frames = 0;
        started     = false;
        ended       = false;
    }

    void release(const float * samples, int32_t n) { out.insert(out.end(), samples, samples + n); }

    void mark_last_emitted() {
        if (buffered > 0) {
            slots[(head + buffered - 1) % capacity].emitted = true;
        }
    }

    void clock_tick(bool voiced) {
        if (voiced) {
            if (clock_speaking) {
                // Crossing a gap shorter than the hold: bill it with this frame.
                speech_ms += k_frame_ms + pending_silence_ms;
                pending_silence_ms = 0;
            } else if (++clock_onset >= std::max<uint32_t>(1, onset_frames)) {
                clock_speaking = true;
                // Every frame that established the onset counts.
                speech_ms += static_cast<uint64_t>(k_frame_ms) * clock_onset;
                clock_onset        = 0;
                pending_silence_ms = 0;
            }
        } else {
            clock_onset = 0;
            if (clock_speaking) {
                pending_silence_ms += k_frame_ms;
                if (pending_silence_ms >= hold_ms) {
                    clock_speaking     = false;
                    pending_silence_ms = 0;
                }
            }
        }
    }

    // One 256-sample frame; n_valid < 256 only for the zero-padded last frame.
    void process_frame(const float * frame, int32_t n_valid) {
        ++call_frames;
        alignas(64) float clamped[k_frame];
        double            sum_sq = 0.0;
        for (int64_t i = 0; i < k_frame; ++i) {
            clamped[i] = std::clamp(frame[i], -1.0f, 1.0f);
            sum_sq += static_cast<double>(frame[i]) * static_cast<double>(frame[i]);
        }
        const double mean_sq = sum_sq / static_cast<double>(k_frame);
        level_dbfs           = mean_sq > 1e-12 ? static_cast<float>(10.0 * std::log10(mean_sq)) : -120.0f;
        last_score           = detector.predict(clamped);

        gate_open         = gate_open ? last_score >= exit : last_score >= threshold;
        const bool voiced = gate_open;
        last_voiced       = voiced;

        // Buffer every scored frame for a possible pre-roll.
        const uint32_t write = (head + buffered) % capacity;
        std::memcpy(ring.data() + static_cast<size_t>(write) * k_frame, frame, k_frame * sizeof(float));
        slots[write] = Slot{ false, voiced, n_valid };
        ++buffered;
        while (buffered > preroll_frames + 1) {
            head = (head + 1) % capacity;
            --buffered;
        }

        if (!in_speech && voiced) {
            if (++onset_counter >= onset_frames) {
                in_speech        = true;
                started          = true;
                hangover_counter = hangover_frames;
                onset_counter    = 0;
                // Pre-roll plus the current frame, skipping slots already
                // released (a previous tail when the pause was shorter than
                // the pre-roll); the current frame is never one of them.
                for (uint32_t i = 0; i < buffered; ++i) {
                    Slot & s = slots[(head + i) % capacity];
                    if (s.emitted) {
                        continue;
                    }
                    release(ring.data() + static_cast<size_t>((head + i) % capacity) * k_frame, s.n_valid);
                    s.emitted = true;
                }
            }
        } else if (in_speech && voiced) {
            hangover_counter = hangover_frames;
            mark_last_emitted();
            release(frame, n_valid);
        } else if (in_speech) {
            if (hangover_counter > 0) {
                --hangover_counter;
                mark_last_emitted();
                release(frame, n_valid);
            } else {
                in_speech = false;
                ended     = true;
            }
        } else {
            onset_counter = 0;  // a broken onset starts over
        }
        clock_tick(voiced);
    }

    void fill(transcribe_vad_gate_update * u) const {
        if (u == nullptr) {
            return;
        }
        u->n_frames       = call_frames;
        u->speech         = in_speech;
        u->voiced         = last_voiced;
        u->speech_started = started;
        u->speech_ended   = ended;
        u->clock_speaking = clock_speaking;
        u->last_score     = last_score;
        u->level_dbfs     = level_dbfs;
        u->speech_ms      = speech_ms;
        u->audio          = out.empty() ? nullptr : out.data();
        u->n_audio        = static_cast<int64_t>(out.size());
    }
};

extern "C" {

void transcribe_vad_gate_params_init(transcribe_vad_gate_params * p) {
    transcribe::init_sized(p);
    if (p != nullptr) {
        p->threshold      = 0.5f;
        p->exit_threshold = -1.0f;
        p->onset_ms       = 60;
        p->preroll_ms     = 450;
        p->hangover_ms    = 450;
        p->speech_hold_ms = 500;
    }
}

void transcribe_vad_gate_update_init(transcribe_vad_gate_update * u) {
    transcribe::init_sized(u);
}

void transcribe_vad_gate_tail_init(transcribe_vad_gate_tail * t) {
    transcribe::init_sized(t);
}

transcribe_status transcribe_vad_gate_init(const transcribe_vad_gate_params * params, transcribe_vad_gate ** out_gate) {
    if (out_gate == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    *out_gate = nullptr;
    transcribe_vad_gate_params p;
    transcribe_vad_gate_params_init(&p);
    if (params != nullptr) {
        if (const auto st = transcribe::check_input_struct_size(
                params->struct_size, TRANSCRIBE_FIELD_END(transcribe_vad_gate_params, speech_hold_ms));
            st != TRANSCRIBE_OK) {
            return st;
        }
        transcribe::copy_out_prefix(&p, params, params->struct_size, sizeof(p));
        p.struct_size = sizeof(p);
    }
    if (!threshold_ok(p.threshold) || !std::isfinite(p.exit_threshold) || p.exit_threshold > 1.0f) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    try {
        *out_gate = new transcribe_vad_gate(p);
        return TRANSCRIBE_OK;
    } catch (const std::bad_alloc &) {
        return TRANSCRIBE_ERR_OOM;
    }
}

void transcribe_vad_gate_free(transcribe_vad_gate * gate) {
    delete gate;
}

void transcribe_vad_gate_reset(transcribe_vad_gate * gate) {
    if (gate != nullptr) {
        gate->reset();
    }
}

transcribe_status transcribe_vad_gate_set_threshold(transcribe_vad_gate * gate, float threshold) {
    if (gate == nullptr || !threshold_ok(threshold)) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // Like swapping ZER0's Hysteresis: new edges, the gate closes, and the
    // next frame clearing the new threshold re-enters. Detector and smoother
    // state are kept, so this is safe mid-recording.
    gate->threshold = threshold;
    gate->exit      = gate->exit_for(threshold);
    gate->gate_open = false;
    return TRANSCRIBE_OK;
}

transcribe_status transcribe_vad_gate_set_hangover_ms(transcribe_vad_gate * gate, uint32_t hangover_ms) {
    if (gate == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    gate->hangover_frames = frames_for_ms(hangover_ms);
    return TRANSCRIBE_OK;
}

transcribe_status transcribe_vad_gate_feed(transcribe_vad_gate *        gate,
                                           const float *                pcm,
                                           int64_t                      n_samples,
                                           transcribe_vad_gate_update * update) {
    if (gate == nullptr || n_samples < 0 || (pcm == nullptr && n_samples > 0)) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (update != nullptr) {
        if (const auto st = transcribe::check_struct_size(update->struct_size,
                                                          TRANSCRIBE_FIELD_END(transcribe_vad_gate_update, n_audio));
            st != TRANSCRIBE_OK) {
            return st;
        }
    }
    for (int64_t i = 0; i < n_samples; ++i) {
        if (!std::isfinite(pcm[i])) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
    }
    try {
        gate->begin_call();
        int64_t i = 0;
        if (gate->partial_n > 0) {
            const int64_t take = std::min(k_frame - gate->partial_n, n_samples);
            std::memcpy(gate->partial.data() + gate->partial_n, pcm, static_cast<size_t>(take) * sizeof(float));
            gate->partial_n += take;
            i += take;
            if (gate->partial_n == k_frame) {
                gate->process_frame(gate->partial.data(), static_cast<int32_t>(k_frame));
                gate->partial_n = 0;
            }
        }
        for (; i + k_frame <= n_samples; i += k_frame) {
            gate->process_frame(pcm + i, static_cast<int32_t>(k_frame));
        }
        if (i < n_samples) {
            std::memcpy(gate->partial.data() + gate->partial_n, pcm + i,
                        static_cast<size_t>(n_samples - i) * sizeof(float));
            gate->partial_n += n_samples - i;
        }
    } catch (const std::bad_alloc &) {
        return TRANSCRIBE_ERR_OOM;
    }
    gate->fill(update);
    return TRANSCRIBE_OK;
}

transcribe_status transcribe_vad_gate_finish(transcribe_vad_gate * gate, transcribe_vad_gate_update * update) {
    if (gate == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (update != nullptr) {
        if (const auto st = transcribe::check_struct_size(update->struct_size,
                                                          TRANSCRIBE_FIELD_END(transcribe_vad_gate_update, n_audio));
            st != TRANSCRIBE_OK) {
            return st;
        }
    }
    try {
        gate->begin_call();
        if (gate->partial_n > 0) {
            std::fill(gate->partial.begin() + gate->partial_n, gate->partial.end(), 0.0f);
            gate->process_frame(gate->partial.data(), static_cast<int32_t>(gate->partial_n));
            gate->partial_n = 0;
        }
    } catch (const std::bad_alloc &) {
        return TRANSCRIBE_ERR_OOM;
    }
    gate->fill(update);
    return TRANSCRIBE_OK;
}

transcribe_status transcribe_vad_gate_get_tail(const transcribe_vad_gate * gate, transcribe_vad_gate_tail * out) {
    if (gate == nullptr || out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (const auto st = transcribe::check_struct_size(out->struct_size,
                                                      TRANSCRIBE_FIELD_END(transcribe_vad_gate_tail, hangover_counter));
        st != TRANSCRIBE_OK) {
        return st;
    }
    // The trailing run of withheld frames; interior ones (before released
    // speech) are not counted.
    int32_t withheld = 0;
    int32_t voiced   = 0;
    for (uint32_t i = gate->buffered; i-- > 0;) {
        const auto & s = gate->slots[(gate->head + i) % gate->capacity];
        if (s.emitted) {
            break;
        }
        ++withheld;
        voiced += s.voiced ? 1 : 0;
    }
    out->withheld_frames        = withheld;
    out->withheld_voiced_frames = voiced;
    out->speech                 = gate->in_speech;
    out->onset_counter          = static_cast<int32_t>(gate->onset_counter);
    out->hangover_counter       = static_cast<int32_t>(gate->hangover_counter);
    return TRANSCRIBE_OK;
}

transcribe_status transcribe_vad_score_frames(const float * pcm,
                                              int64_t       n_samples,
                                              int32_t       n_threads,
                                              float *       out_scores) {
    if (n_samples < 0 || (n_samples > 0 && (pcm == nullptr || out_scores == nullptr))) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    bool in_range = true;
    for (int64_t i = 0; i < n_samples; ++i) {
        if (!std::isfinite(pcm[i])) {
            return TRANSCRIBE_ERR_INVALID_ARG;
        }
        in_range = in_range && pcm[i] >= -1.0f && pcm[i] <= 1.0f;
    }
    try {
        // A clamped copy only when a sample needs it: normalized capture audio
        // (the usual input) is scored in place instead of duplicating the clip.
        std::vector<float> clamped;
        const float *      samples = pcm;
        if (!in_range) {
            clamped.resize(static_cast<size_t>(n_samples));
            for (int64_t i = 0; i < n_samples; ++i) {
                clamped[static_cast<size_t>(i)] = std::clamp(pcm[i], -1.0f, 1.0f);
            }
            samples = clamped.data();
        }
        transcribe::earshot::score_frames(samples, n_samples, n_threads, out_scores);
    } catch (const std::bad_alloc &) {
        return TRANSCRIBE_ERR_OOM;
    } catch (...) {
        return TRANSCRIBE_ERR_OOM;  // a worker thread could not start
    }
    return TRANSCRIBE_OK;
}

}  // extern "C"
