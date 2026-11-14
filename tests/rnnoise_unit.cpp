// RNNoise port (src/transcribe-rnnoise.*): the 960-point real FFT against a
// naive DFT, its inverse, and denoiser invariants (silence in, silence out;
// reset reproduces; in-place equals out-of-place; bounded output; speech
// probability in [0, 1]); and the output against nnnoiseless 0.5.2, the crate
// ZER0 runs (tests/fixtures/rnnoise).
#include "transcribe-rnnoise.h"
#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace rn = transcribe::rnnoise;

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// Deterministic noise in [-1, 1).
struct Lcg {
    uint32_t s = 12345u;

    float next() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>(s >> 8) / 8388608.0f - 1.0f;
    }
};

void test_rfft_matches_dft() {
    std::vector<float> x(rn::k_window);
    Lcg                rng;
    for (float & v : x) {
        v = 1000.0f * rng.next();
    }
    std::array<std::complex<float>, rn::k_freq_size> got;
    rn::detail::rfft960(x.data(), got);
    double max_err = 0.0;
    double max_mag = 0.0;
    for (int k = 0; k < rn::k_freq_size; ++k) {
        std::complex<double> ref = 0.0;
        for (int n = 0; n < rn::k_window; ++n) {
            const double a = -2.0 * 3.14159265358979323846 * k * n / rn::k_window;
            ref += static_cast<double>(x[n]) * std::complex<double>(std::cos(a), std::sin(a));
        }
        max_err = std::max(max_err, std::abs(ref - std::complex<double>(got[k])));
        max_mag = std::max(max_mag, std::abs(ref));
    }
    std::printf("rfft960: max |err| %.3g of max |X| %.3g\n", max_err, max_mag);
    CHECK(max_err < 1e-5 * max_mag);

    // Inverse: x = irfft(rfft(x)) / 960.
    std::vector<float> back(rn::k_window);
    rn::detail::irfft960(got, back.data());
    double max_back = 0.0;
    for (int n = 0; n < rn::k_window; ++n) {
        max_back = std::max(max_back, std::abs(static_cast<double>(back[n]) / rn::k_window - x[n]));
    }
    std::printf("irfft960 round trip: max |err| %.3g\n", max_back);
    CHECK(max_back < 1e-3);
}

std::vector<float> run(rn::Denoiser & d, const std::vector<float> & in, std::vector<float> * probs = nullptr) {
    std::vector<float> out(in.size());
    for (size_t i = 0; i + rn::k_frame_size <= in.size(); i += rn::k_frame_size) {
        const float p = d.process_frame(in.data() + i, out.data() + i);
        if (probs) {
            probs->push_back(p);
        }
    }
    return out;
}

void test_silence() {
    rn::Denoiser             d;
    const std::vector<float> zeros(rn::k_frame_size * 50, 0.0f);
    std::vector<float>       probs;
    const auto               out = run(d, zeros, &probs);
    for (float v : out) {
        CHECK(v == 0.0f);
    }
    for (float p : probs) {
        CHECK(p == 0.0f);
    }
}

// Pink-ish noise plus a few voiced-like harmonics, at int16 scale.
std::vector<float> make_signal(size_t frames) {
    std::vector<float> s(frames * rn::k_frame_size);
    Lcg                rng;
    float              lp = 0.0f;
    for (size_t n = 0; n < s.size(); ++n) {
        lp             = 0.95f * lp + 0.05f * rng.next();
        const double t = static_cast<double>(n) / 48000.0;
        const double f = 140.0 + 20.0 * std::sin(2.0 * 3.14159265358979323846 * 0.7 * t);
        double       v = 0.0;
        for (int h = 1; h <= 8; ++h) {
            v += std::sin(2.0 * 3.14159265358979323846 * f * h * t) / h;
        }
        const double env = (n / 24000) % 2 == 0 ? 1.0 : 0.05;  // 0.5 s on / off
        s[n]             = static_cast<float>(3000.0 * env * v + 4000.0 * lp);
    }
    return s;
}

void test_reset_inplace_bounds() {
    const auto         sig = make_signal(200);
    rn::Denoiser       d;
    std::vector<float> probs;
    const auto         first = run(d, sig, &probs);
    for (float p : probs) {
        CHECK(p >= 0.0f && p <= 1.0f);
    }
    double e_in = 0.0, e_out = 0.0;
    for (size_t i = 0; i < sig.size(); ++i) {
        CHECK(std::isfinite(first[i]));
        e_in += static_cast<double>(sig[i]) * sig[i];
        e_out += static_cast<double>(first[i]) * first[i];
    }
    // Suppression never adds energy overall.
    CHECK(e_out <= e_in * 1.05);

    d.reset();
    CHECK(run(d, sig) == first);

    // In place.
    rn::Denoiser       d2;
    std::vector<float> buf = sig;
    for (size_t i = 0; i < buf.size(); i += rn::k_frame_size) {
        d2.process_frame(buf.data() + i, buf.data() + i);
    }
    CHECK(buf == first);

    // Moved-from denoisers stay safe.
    rn::Denoiser d3                    = std::move(d2);
    float        tmp[rn::k_frame_size] = {};
    CHECK(d2.process_frame(tmp, tmp) == 0.0f);  // NOLINT(bugprone-use-after-move)
    (void) d3;
}

template <typename T> std::vector<T> read_raw(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return {};
    }
    f.seekg(0, std::ios::end);
    const auto bytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<T> v(bytes / sizeof(T));
    f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    return v;
}

// The golden input: a LibriSpeech clip (the Earshot fixture) held 3x to
// 48 kHz at int16 scale, plus low-passed noise. `--dump-input <path>` writes
// it for the nnnoiseless harness (tests/fixtures/rnnoise/README.md).
std::vector<float> golden_input() {
    std::vector<float> pcm;
    std::string        err;
    const std::string  path = std::string(TRANSCRIBE_EARSHOT_FIXTURES) + "/ls-6930-75918-0000.wav";
    if (!transcribe_cli::load_wav_mono_16k(path, pcm, err)) {
        std::fprintf(stderr, "cannot load %s: %s\n", path.c_str(), err.c_str());
        return {};
    }
    std::vector<float> x(pcm.size() * 3 / rn::k_frame_size * rn::k_frame_size);
    Lcg                rng;
    float              lp = 0.0f;
    for (size_t n = 0; n < x.size(); ++n) {
        lp   = 0.9f * lp + 0.1f * rng.next();
        x[n] = pcm[n / 3] * 32767.0f + 3000.0f * lp;
    }
    return x;
}

// nnnoiseless 0.5.2 on golden_input(): its output rounded to int16, and its
// per-frame speech probabilities.
void test_golden(const std::vector<float> & in) {
    const std::string dir   = TRANSCRIBE_RNNOISE_FIXTURES;
    const auto        ref   = read_raw<int16_t>(dir + "/nnnoiseless.out.i16");
    const auto        rprob = read_raw<float>(dir + "/nnnoiseless.vad.f32");
    if (in.empty() || ref.size() != in.size() || rprob.size() != in.size() / rn::k_frame_size) {
        std::fprintf(stderr, "golden fixtures missing or inconsistent in %s\n", dir.c_str());
        ++g_failures;
        return;
    }
    const bool         wide = rn::detail::set_wide_kernels(true);  // the default where available
    rn::Denoiser       d;
    std::vector<float> probs;
    const auto         out = run(d, in, &probs);
    double             num = 0.0, den = 0.0, max_abs = 0.0, max_prob = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double o = std::clamp(std::nearbyint(static_cast<double>(out[i])), -32768.0, 32767.0);
        const double e = o - ref[i];
        num += e * e;
        den += static_cast<double>(ref[i]) * ref[i];
        max_abs = std::max(max_abs, std::abs(e));
    }
    for (size_t f = 0; f < probs.size(); ++f) {
        max_prob = std::max(max_prob, std::abs(static_cast<double>(probs[f]) - rprob[f]));
    }
    const double snr = 10.0 * std::log10(den / std::max(num, 1e-30));
    std::printf("golden vs nnnoiseless: SNR %.1f dB, max |diff| %.0f LSB, max |vad diff| %.3g over %zu frames\n", snr,
                max_abs, max_prob, probs.size());
    // Float rounding differences only (another FFT): far below audible.
    CHECK(snr > 50.0);
    CHECK(max_prob < 2e-2);

    // 8-wide and 4-wide kernels: the same bits.
    rn::detail::set_wide_kernels(false);
    rn::Denoiser       d4;
    std::vector<float> probs4;
    CHECK(run(d4, in, &probs4) == out);
    CHECK(probs4 == probs);
    rn::detail::set_wide_kernels(wide);
    std::printf("kernels: %s by default, 4-wide matches bit for bit\n", wide ? "8-wide" : "4-wide");
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<float> golden = golden_input();
    if (argc == 3 && std::strcmp(argv[1], "--dump-input") == 0) {
        std::ofstream f(argv[2], std::ios::binary);
        f.write(reinterpret_cast<const char *>(golden.data()),
                static_cast<std::streamsize>(golden.size() * sizeof(float)));
        return f ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    test_rfft_matches_dft();
    test_silence();
    test_reset_inplace_bounds();
    test_golden(golden);
    if (g_failures == 0) {
        std::printf("rnnoise_unit: OK\n");
    }
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
