// Noise suppression (transcribe_denoise_*) through the public C API: the
// wrapper around RNNoise that replaces ZER0's denoise_frame (wet/dry mix with
// a delayed dry side, probability gate, scaling, clamping).
#include "transcribe-rnnoise.h"
#include "transcribe.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int N = TRANSCRIBE_DENOISE_FRAME_SAMPLES;

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// Voiced-like harmonics over noise, in [-1, 1], `frames` frames long.
std::vector<float> make_signal(int frames, uint32_t seed = 7u) {
    std::vector<float> s(static_cast<size_t>(frames) * N);
    uint32_t           r = seed;
    for (size_t n = 0; n < s.size(); ++n) {
        r               = r * 1664525u + 1013904223u;
        const float  wn = static_cast<float>(r >> 8) / 8388608.0f - 1.0f;
        const double t  = static_cast<double>(n) / 48000.0;
        double       v  = 0.0;
        for (int h = 1; h <= 6; ++h) {
            v += std::sin(2.0 * 3.14159265358979323846 * 150.0 * h * t) / h;
        }
        s[n] = static_cast<float>(0.1 * v) + 0.05f * wn;
    }
    return s;
}

transcribe_denoise * make(float strength, float vad_threshold, uint32_t grace_ms) {
    transcribe_denoise_params p;
    transcribe_denoise_params_init(&p);
    p.strength             = strength;
    p.vad_threshold        = vad_threshold;
    p.vad_grace_ms         = grace_ms;
    transcribe_denoise * d = nullptr;
    CHECK(transcribe_denoise_init(&p, &d) == TRANSCRIBE_OK);
    return d;
}

std::vector<float> run(transcribe_denoise * d, const std::vector<float> & in, std::vector<float> * probs = nullptr) {
    std::vector<float> out(in.size());
    for (size_t i = 0; i < in.size(); i += N) {
        float p = -1.0f;
        CHECK(transcribe_denoise_process_frame(d, in.data() + i, out.data() + i, &p) == TRANSCRIBE_OK);
        if (probs) {
            probs->push_back(p);
        }
    }
    return out;
}

void test_defaults_and_validation() {
    transcribe_denoise_params p;
    transcribe_denoise_params_init(&p);
    CHECK(p.struct_size == sizeof(p));
    CHECK(p.strength == 1.0f && p.vad_threshold == 0.0f && p.vad_grace_ms == 200);

    transcribe_denoise * d = nullptr;
    CHECK(transcribe_denoise_init(nullptr, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    for (const float bad : { -0.1f, 1.1f, NAN }) {
        transcribe_denoise_params q = p;
        q.strength                  = bad;
        CHECK(transcribe_denoise_init(&q, &d) == TRANSCRIBE_ERR_INVALID_ARG);
        CHECK(d == nullptr);
        q               = p;
        q.vad_threshold = bad;
        CHECK(transcribe_denoise_init(&q, &d) == TRANSCRIBE_ERR_INVALID_ARG);
    }
    transcribe_denoise_params q = p;
    q.vad_grace_ms              = TRANSCRIBE_DENOISE_MAX_GRACE_MS + 1;
    CHECK(transcribe_denoise_init(&q, &d) == TRANSCRIBE_ERR_INVALID_ARG);
    q             = p;
    q.struct_size = 0;
    CHECK(transcribe_denoise_init(&q, &d) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);

    CHECK(transcribe_denoise_init(nullptr, &d) == TRANSCRIBE_OK);
    float frame[N] = {};
    frame[17]      = NAN;
    CHECK(transcribe_denoise_process_frame(d, frame, frame, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_denoise_process_frame(d, nullptr, frame, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_denoise_process_frame(nullptr, frame, frame, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_denoise_set_params(d, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    q          = p;
    q.strength = 2.0f;
    CHECK(transcribe_denoise_set_params(d, &q) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_denoise_free(d);
    transcribe_denoise_free(nullptr);
    transcribe_denoise_reset(nullptr);
}

// Strength 1, no gate: the network's output scaled back and clamped.
void test_full_strength_is_the_network() {
    const auto                    sig = make_signal(60);
    transcribe_denoise *          d   = make(1.0f, 0.0f, 200);
    std::vector<float>            probs;
    const auto                    out = run(d, sig, &probs);
    transcribe::rnnoise::Denoiser net;
    float                         scaled[N];
    float                         ref[N];
    for (size_t f = 0; f * N < sig.size(); ++f) {
        for (int i = 0; i < N; ++i) {
            scaled[i] = std::fmin(std::fmax(sig[f * N + i] * 32767.0f, -32768.0f), 32767.0f);
        }
        const float prob = net.process_frame(scaled, ref);
        CHECK(prob == probs[f]);
        for (int i = 0; i < N; ++i) {
            const float want = std::fmin(std::fmax(ref[i] / 32767.0f, -1.0f), 1.0f);
            CHECK(out[f * N + i] == want);
        }
    }
    transcribe_denoise_free(d);
}

// Strength 0: the input, one frame late (ZER0's
// strength_zero_passes_the_input_through_one_frame_late).
void test_strength_zero_is_delayed_input() {
    const auto           sig = make_signal(10);
    transcribe_denoise * d   = make(0.0f, 0.0f, 200);
    const auto           out = run(d, sig);
    for (int i = 0; i < N; ++i) {
        CHECK(out[i] == 0.0f);
    }
    for (size_t i = N; i < sig.size(); ++i) {
        CHECK(std::abs(out[i] - sig[i - N]) < 1e-4f);
    }
    transcribe_denoise_free(d);
}

// Gate: threshold 1 is never reached by a real frame, so after the grace
// period the gain slews shut; speech probability 0 (silence) the same.
void test_gate_grace_and_slew() {
    const auto           sig  = make_signal(40);
    transcribe_denoise * d    = make(1.0f, 1.0f, 30);
    const auto           out  = run(d, sig);
    // No frame reached the threshold, so no grace: the gain halves from the
    // first frame and snaps to 0 within ~10 frames.
    double               tail = 0.0;
    for (size_t i = 15 * N; i < out.size(); ++i) {
        tail += std::abs(out[i]);
    }
    CHECK(tail == 0.0);

    // A threshold the frames clear keeps everything open.
    transcribe_denoise * open = make(1.0f, 0.0f, 0);
    transcribe_denoise * low  = make(1.0f, 1e-6f, 0);
    std::vector<float>   probs;
    const auto           a         = run(open, sig, &probs);
    const auto           b         = run(low, sig);
    bool                 all_clear = true;
    for (float p : probs) {
        all_clear = all_clear && p >= 1e-6f;
    }
    if (all_clear) {
        CHECK(a == b);
    }

    // Live params: switching the gate off reopens it (slewing back).
    transcribe_denoise_params p;
    transcribe_denoise_params_init(&p);
    CHECK(transcribe_denoise_set_params(d, &p) == TRANSCRIBE_OK);
    const auto again = run(d, sig);
    double     late  = 0.0;
    for (size_t i = 15 * N; i < again.size(); ++i) {
        late += std::abs(again[i]);
    }
    CHECK(late > 0.0);
    transcribe_denoise_free(d);
    transcribe_denoise_free(open);
    transcribe_denoise_free(low);
}

void test_flush_reset_alias_bounds() {
    const auto           sig = make_signal(30);
    transcribe_denoise * d   = make(1.0f, 0.0f, 200);
    const auto           out = run(d, sig);
    float                last[N];
    float                prob = -1.0f;
    CHECK(transcribe_denoise_flush(d, last, &prob) == TRANSCRIBE_OK);
    CHECK(prob >= 0.0f && prob <= 1.0f);  // the window still spans the last real frame
    double e = 0.0;
    for (float v : last) {
        e += std::abs(v);
    }
    CHECK(e > 0.0);  // the last real frame came out

    // Reset reproduces; in place equals out of place.
    transcribe_denoise_reset(d);
    std::vector<float> buf = sig;
    for (size_t i = 0; i < buf.size(); i += N) {
        CHECK(transcribe_denoise_process_frame(d, buf.data() + i, buf.data() + i, nullptr) == TRANSCRIBE_OK);
    }
    CHECK(buf == out);

    // Full-scale noise stays in [-1, 1].
    transcribe_denoise_reset(d);
    std::vector<float> loud(static_cast<size_t>(N) * 50);
    uint32_t           r = 1u;
    for (float & v : loud) {
        r = r * 1664525u + 1013904223u;
        v = (r & 0x80000000u) ? 1.0f : -1.0f;
    }
    for (float v : run(d, loud)) {
        CHECK(std::isfinite(v) && v >= -1.0f && v <= 1.0f);
    }
    transcribe_denoise_free(d);
}

}  // namespace

int main() {
    test_defaults_and_validation();
    test_full_strength_is_the_network();
    test_strength_zero_is_delayed_input();
    test_gate_grace_and_slew();
    test_flush_reset_alias_bounds();
    if (g_failures == 0) {
        std::printf("denoise_unit: OK\n");
    }
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
