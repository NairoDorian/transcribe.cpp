// mel_unit.cpp - unit tests for the precomputed buffers in
// transcribe::MelFrontend.
//
// Validates the constructor outputs (the symmetric-Hann window
// zero-padded to n_fft, and the librosa Slaney mel filterbank)
// in isolation, before the full pipeline runs. Catches the kinds of
// off-by-one bugs that would silently bias the entire frontend:
//
//   - periodic vs symmetric Hann (the most common subtle window bug)
//   - off-by-one in the win_length / n_fft zero-pad placement
//   - wrong mel scale formula (HTK vs Slaney)
//   - missing Slaney area normalization (would scale by ~50x)
//
// All reference values are bit-precise constants captured from
// librosa 0.11 and the symmetric-hann formula. Regenerate via the
// preflight script if librosa updates and the tolerances drift.

#include "transcribe-mel.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(actual, expected, tol)                                                                          \
    do {                                                                                                           \
        const double _a = static_cast<double>(actual);                                                             \
        const double _e = static_cast<double>(expected);                                                           \
        const double _d = std::fabs(_a - _e);                                                                      \
        if (_d > (tol)) {                                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s = %.17g, expected %.17g, diff %.6g > %.6g\n", __FILE__, __LINE__, \
                         #actual, _a, _e, _d, (tol));                                                              \
            ++g_failures;                                                                                          \
        }                                                                                                          \
    } while (0)

// Real Parakeet 0.6B frontend config (v2 + v3 share these values).
transcribe::MelConfig parakeet_config() {
    transcribe::MelConfig cfg;
    cfg.sample_rate  = 16000;
    cfg.num_mels     = 128;
    cfg.n_fft        = 512;
    cfg.win_length   = 400;
    cfg.hop_length   = 160;
    cfg.pre_emphasis = 0.97f;
    cfg.f_min        = 0.0f;
    cfg.f_max        = 8000.0f;
    return cfg;
}

void test_window() {
    transcribe::MelFrontend mf(parakeet_config());
    const auto &            w = mf.window();

    CHECK(w.size() == 512);

    // First and last 56 entries are the zero pad: (n_fft - win_length) / 2
    // on each side. The 57th entry from each end is the first non-zero
    // sample of the symmetric Hann.
    for (int i = 0; i < 56; ++i) {
        CHECK(w[i] == 0.0);
    }
    for (int i = 0; i < 56; ++i) {
        CHECK(w[511 - i] == 0.0);
    }

    // Symmetric Hann formula 0.5 - 0.5*cos(2*pi*k/(N-1)) with N=400.
    // The boundary samples at the edge of the Hann are tiny but non-zero.
    // Reference values captured from numpy and librosa, machine-precision.
    CHECK_NEAR(w[56], 0.0, 1e-15);
    CHECK_NEAR(w[57], 6.199333200590518e-05, 1e-15);
    CHECK_NEAR(w[58], 0.0002479579553307798, 1e-15);
    CHECK_NEAR(w[59], 0.0005578477557081074, 1e-15);
    CHECK_NEAR(w[60], 0.0009915858887327156, 1e-15);

    // Peak region. For a symmetric N=400 Hann the peak straddles indices
    // 199 and 200 (in window space) = 255 and 256 in the padded buffer,
    // both equal to ~0.99998 (NOT 1.0 - that would be the periodic form).
    CHECK_NEAR(w[254], 0.9998605186060137, 1e-15);
    CHECK_NEAR(w[255], 0.9999845014267927, 1e-15);
    CHECK_NEAR(w[256], 0.9999845014267927, 1e-15);
    CHECK_NEAR(w[257], 0.9998605186060137, 1e-15);
    CHECK_NEAR(w[258], 0.9996125837088883, 1e-15);

    // Sum of a symmetric Hann window of length 400 is exactly
    // (N - 1) / 2 = 199.5. Off by anything > a few ULPs means the
    // formula is wrong.
    double sum = 0.0;
    for (double v : w) {
        sum += v;
    }
    CHECK_NEAR(sum, 199.5, 1e-12);

    // Sum of squares: closed form for a symmetric Hann length N is
    // (3N - 4) / 8. For N=400: (1200 - 4) / 8 = 149.5. But the
    // librosa-style symmetric form (k / (N-1)) gives a slightly
    // different value; reference captured from numpy directly:
    double sumsq = 0.0;
    for (double v : w) {
        sumsq += v * v;
    }
    CHECK_NEAR(sumsq, 149.625, 1e-12);
}

void test_mel_filterbank() {
    transcribe::MelFrontend mf(parakeet_config());
    const auto &            fb     = mf.filterbank();
    const int               n_freq = mf.n_freq();
    const int               n_mels = mf.num_mels();

    CHECK(static_cast<int>(fb.size()) == n_mels * n_freq);
    CHECK(n_freq == 257);
    CHECK(n_mels == 128);

    auto get = [&](int m, int k) -> float {
        return fb[static_cast<size_t>(m) * n_freq + k];
    };

    // Total filterbank sum. Captured from librosa 0.11 with the
    // exact (sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000,
    // norm='slaney') call.
    double total = 0.0;
    for (float v : fb) {
        total += static_cast<double>(v);
    }
    CHECK_NEAR(total, 4.090487480163574, 5e-6);

    // Per-row spot checks. Slaney filters are sparse at low mel
    // bins (some span only 1-3 freq bins) and progressively wider
    // at the top.

    // Mel bin 0: only fft bin 1 has a nonzero weight.
    for (int k = 0; k < n_freq; ++k) {
        if (k == 1) {
            CHECK_NEAR(get(0, k), 0.02837754227221012, 5e-7);
        } else {
            CHECK(get(0, k) == 0.0f);
        }
    }

    // Mel bin 5: triangular filter spanning fft bins 4 and 5.
    CHECK_NEAR(get(5, 4), 0.014789481647312641, 5e-7);
    CHECK_NEAR(get(5, 5), 0.013588061556220055, 5e-7);
    CHECK(get(5, 3) == 0.0f);
    CHECK(get(5, 6) == 0.0f);

    // Mel bin 64: middle of the bank.
    CHECK_NEAR(get(64, 55), 0.018818283453584, 5e-7);

    // Mel bin 100: 6 nonzero bins centered at fft bin 130.
    CHECK_NEAR(get(100, 130), 0.009136910550296, 5e-7);

    // Mel bin 127 (top): widest filter, peak at fft bin 250.
    CHECK_NEAR(get(127, 250), 0.005223188549280, 5e-7);

    // Per-row sums (Slaney area-normalized triangles, so each row's
    // sum is 2/(width) * triangle area = roughly constant per
    // octave; reference values from librosa).
    double row_0_sum   = 0.0;
    double row_64_sum  = 0.0;
    double row_127_sum = 0.0;
    for (int k = 0; k < n_freq; ++k) {
        row_0_sum += get(0, k);
        row_64_sum += get(64, k);
        row_127_sum += get(127, k);
    }
    CHECK_NEAR(row_0_sum, 0.028377542272210, 5e-7);
    CHECK_NEAR(row_64_sum, 0.030684133991599, 5e-6);
    CHECK_NEAR(row_127_sum, 0.031943686306477, 5e-6);

    // Total nonzero count: librosa Slaney with these params gives
    // exactly 504 non-zero entries out of 32896. A wrong formula
    // would either widen the filters (more nonzeros) or break the
    // mel scale (different distribution).
    int nonzero = 0;
    for (float v : fb) {
        if (v != 0.0f) {
            ++nonzero;
        }
    }
    CHECK(nonzero == 504);
}

void test_n_frames_for() {
    transcribe::MelFrontend mf(parakeet_config());
    // jfk.wav is 176000 samples = 11 s @ 16 kHz.
    // n_frames = 176000 / 160 + 1 = 1101.
    CHECK(mf.n_frames_for(176000) == 1101);
    // Another spot check.
    CHECK(mf.n_frames_for(16000) == 101);
    // Empty audio still returns the +1.
    CHECK(mf.n_frames_for(0) == 1);
}

// ---------------------------------------------------------------------
// Incremental (streaming) extraction: the equality gate.
//
// compute_incremental() claims to be compute() restricted to the frames
// that changed -- not an approximation of it. Everything downstream of the
// frontend (the encoder prefix cache in arch/qwen3_asr, which reuses
// encoder rows across R2T2 ticks) is built on that claim, and a silent
// divergence here would show up as transcript drift far from the cause. So
// the claim is tested directly: feed the same audio in growing prefixes and
// require every emitted buffer to be bit-identical to compute()'s over the
// same prefix.
//
// Bit-identical (`==` on the floats), not near-equal: both paths run the
// same FusedFrameStepper::frame on the same padded values, and the clamp
// level is the same maximum, so anything other than exact equality means a
// frame is being reused that should have been recomputed.
// ---------------------------------------------------------------------

// Qwen3-ASR / Whisper-style frontend: this is what R2T2 streams through.
transcribe::MelConfig qwen3_asr_config() {
    transcribe::MelConfig cfg;
    cfg.sample_rate  = 16000;
    cfg.num_mels     = 128;
    cfg.n_fft        = 400;
    cfg.win_length   = 400;
    cfg.hop_length   = 160;
    cfg.pre_emphasis = 0.0f;
    cfg.f_min        = 0.0f;
    cfg.f_max        = 8000.0f;
    cfg.pad_mode     = "reflect";
    cfg.window_type  = "hann_periodic";
    cfg.normalize    = "per_utterance";
    return cfg;
}

// Deterministic pseudo-random audio in [-1, 1). A LCG rather than
// sin()+rand() so the test is reproducible across machines and libm
// versions -- the point is to exercise the arithmetic, not to be pretty.
// Mixed amplitude is deliberate: a signal whose level wanders is what makes
// the per-utterance maximum move, which is the case the running-maximum
// bookkeeping exists for.
std::vector<float> make_audio(size_t n) {
    std::vector<float> pcm(n);
    unsigned long long s    = 0x2545F4914F6CDD1DULL;
    auto               next = [&]() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((s >> 11) & 0xFFFFFFFFULL) / 2147483648.0 - 1.0;
    };
    double env = 0.05;
    for (size_t i = 0; i < n; ++i) {
        // A slow envelope, so loud passages arrive late and the global max
        // rises after many frames are already final.
        if ((i % 4096) == 0) {
            env = 0.02 + 0.9 * std::fabs(next());
        }
        pcm[i] = static_cast<float>(env * next());
    }
    return pcm;
}

// Feed `pcm` in growing prefixes and compare against the batch result.
// `steps` is the prefix length in samples for each call; the last entry
// must be pcm.size().
void check_incremental_equivalence(const char *                  tag,
                                   const transcribe::MelConfig & cfg,
                                   const std::vector<float> &    pcm,
                                   const std::vector<size_t> &   steps) {
    transcribe::MelFrontend    mf(cfg);
    transcribe::MelStreamState st;

    if (!mf.supports_incremental()) {
        std::fprintf(stderr, "FAIL %s: config should support incremental extraction\n", tag);
        ++g_failures;
        return;
    }

    int compared = 0;
    for (size_t n : steps) {
        std::vector<float> inc;
        int                inc_mels = 0, inc_frames = 0;
        const auto         rc = mf.compute_incremental(st, pcm.data(), n, inc, inc_mels, inc_frames);
        if (rc != TRANSCRIBE_OK) {
            std::fprintf(stderr, "FAIL %s: compute_incremental returned %d at n=%zu\n", tag, static_cast<int>(rc), n);
            ++g_failures;
            return;
        }

        std::vector<float> batch;
        int                b_mels = 0, b_frames = 0;
        if (mf.compute(pcm.data(), n, batch, b_mels, b_frames) != TRANSCRIBE_OK) {
            continue;  // prefix too short for the batch path; nothing to compare
        }

        if (inc_frames != b_frames || inc_mels != b_mels) {
            std::fprintf(stderr, "FAIL %s: shape at n=%zu: incremental [%d,%d] vs batch [%d,%d]\n", tag, n, inc_mels,
                         inc_frames, b_mels, b_frames);
            ++g_failures;
            return;
        }
        if (inc.size() != batch.size()) {
            std::fprintf(stderr, "FAIL %s: size at n=%zu: %zu vs %zu\n", tag, n, inc.size(), batch.size());
            ++g_failures;
            return;
        }
        for (size_t i = 0; i < inc.size(); ++i) {
            if (inc[i] != batch[i]) {
                const int t = static_cast<int>(i % static_cast<size_t>(inc_frames));
                const int m = static_cast<int>(i / static_cast<size_t>(inc_frames));
                std::fprintf(stderr, "FAIL %s: n=%zu mel[%d][%d]: %.9g (incremental) != %.9g (batch)\n", tag, n, m, t,
                             static_cast<double>(inc[i]), static_cast<double>(batch[i]));
                ++g_failures;
                return;
            }
        }
        ++compared;
    }
    // A gate that silently compared nothing would pass for the wrong reason.
    if (compared < 4) {
        std::fprintf(stderr, "FAIL %s: only %d prefixes compared\n", tag, compared);
        ++g_failures;
    }
}

void test_incremental_equivalence() {
    // Long enough to cross many frames and several envelope changes; no
    // network, so the cost is the batch re-derivation, which is the point
    // (it is the work the incremental path avoids).
    const auto pcm = make_audio(300000);  // 18.75 s @ 16 kHz

    // Regular 80 ms ticks (1280 samples = 8 hops), which is the cadence the
    // 80 ms R2T2 mode streams at.
    {
        std::vector<size_t> steps;
        for (size_t n = 1280; n <= pcm.size(); n += 1280) {
            steps.push_back(n);
        }
        if (steps.empty() || steps.back() != pcm.size()) {
            steps.push_back(pcm.size());
        }
        transcribe::MelConfig cfg = qwen3_asr_config();
        check_incremental_equivalence("qwen3_asr/80ms", cfg, pcm, steps);
    }

    // Irregular prefixes: audio does not arrive on a frame boundary, and a
    // stream that starts late (or an early flush) feeds a prefix that is not
    // a multiple of hop. The increments here are deliberately not multiples
    // of the 160-sample hop, so the prefix length walks through every phase
    // of the frame grid. Every prefix must be exact, not just the ones that
    // land on a hop.
    //
    // The list is kept short on purpose: each entry costs a full batch
    // extraction (that is the reference the incremental result is checked
    // against), so the useful thing is a handful of awkward prefixes, not
    // thousands of redundant ones.
    {
        std::vector<size_t> steps;
        unsigned long long  s = 12345;
        size_t              n = 1280;
        while (n < 40000) {
            steps.push_back(n);
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            n += 161 + static_cast<size_t>((s >> 33) % 4000);
        }
        steps.push_back(pcm.size());
        transcribe::MelConfig cfg = qwen3_asr_config();
        check_incremental_equivalence("qwen3_asr/irregular", cfg, pcm, steps);
    }

    // normalize="global" (Voxtral Realtime): no running maximum at all, the
    // clamp level is a constant, so this isolates the frame-reuse logic from
    // the level bookkeeping.
    {
        std::vector<size_t> steps;
        for (size_t n = 3200; n <= pcm.size(); n += 3200) {
            steps.push_back(n);
        }
        if (steps.back() != pcm.size()) {
            steps.push_back(pcm.size());
        }
        transcribe::MelConfig cfg = qwen3_asr_config();
        cfg.normalize             = "global";
        cfg.global_log_mel_max    = 1.5f;
        check_incremental_equivalence("global", cfg, pcm, steps);
    }

    // normalize="none": emits every raw frame plus the zeroed trailing
    // center-pad column, so the emitted count differs from the Whisper
    // modes and the last column is written by the mask rather than by the
    // STFT. Also exercises the non-whisper log() branch of the frame body.
    {
        std::vector<size_t> steps;
        for (size_t n = 1600; n <= pcm.size(); n += 6400) {
            steps.push_back(n);
        }
        if (steps.back() != pcm.size()) {
            steps.push_back(pcm.size());
        }
        transcribe::MelConfig cfg = qwen3_asr_config();
        cfg.normalize             = "none";
        check_incremental_equivalence("none", cfg, pcm, steps);
    }

    // pad_mode="constant": zero padding, so every frame is final as soon as
    // its window fits and the provisional tail is empty. Same values must
    // come out as the batch path's.
    {
        std::vector<size_t> steps;
        for (size_t n = 1280; n <= pcm.size(); n += 5120) {
            steps.push_back(n);
        }
        if (steps.back() != pcm.size()) {
            steps.push_back(pcm.size());
        }
        transcribe::MelConfig cfg = qwen3_asr_config();
        cfg.pad_mode              = "constant";
        check_incremental_equivalence("constant-pad", cfg, pcm, steps);
    }
}

void test_incremental_rejects_unsupported() {
    transcribe::MelFrontend mf(parakeet_config());  // per_feature, pow2 n_fft
    CHECK(!mf.supports_incremental());

    transcribe::MelConfig cfg = parakeet_config();
    cfg.normalize             = "per_feature";
    cfg.n_fft                 = 400;
    cfg.win_length            = 400;
    cfg.window_type           = "hann_periodic";
    transcribe::MelFrontend mf2(cfg);
    CHECK(!mf2.supports_incremental());

    // pad_mode="none" has its own frame grid; not offered.
    transcribe::MelConfig cfg3 = qwen3_asr_config();
    cfg3.pad_mode              = "none";
    transcribe::MelFrontend mf3(cfg3);
    CHECK(!mf3.supports_incremental());

    // The call must refuse rather than silently fall back, so a caller that
    // forgets to check cannot get a wrong answer.
    transcribe::MelStreamState st;
    std::vector<float>         out;
    int                        m = 0, f = 0;
    const std::vector<float>   pcm(16000, 0.1f);
    CHECK(mf.compute_incremental(st, pcm.data(), pcm.size(), out, m, f) == TRANSCRIBE_ERR_NOT_IMPLEMENTED);
}

// A state reused across a config change (or a restarted stream) must not mix
// frames from two frontends; it restarts instead. Distinct configs whose
// frames differ (different n_fft) must both come out equal to their batch
// results from the same state object.
void test_incremental_state_reuse() {
    const auto pcm = make_audio(40000);

    transcribe::MelConfig cfg_a = qwen3_asr_config();
    transcribe::MelConfig cfg_b = qwen3_asr_config();
    cfg_b.n_fft                 = 200;
    cfg_b.win_length            = 200;
    cfg_b.normalize             = "global";

    transcribe::MelFrontend    a(cfg_a);
    transcribe::MelFrontend    b(cfg_b);
    transcribe::MelStreamState st;

    for (int pass = 0; pass < 2; ++pass) {
        const transcribe::MelFrontend & mf = (pass == 0) ? a : b;
        for (size_t n = 8000; n <= pcm.size(); n += 8000) {
            std::vector<float> inc, batch;
            int                im = 0, ifr = 0, bm = 0, bfr = 0;
            CHECK(mf.compute_incremental(st, pcm.data(), n, inc, im, ifr) == TRANSCRIBE_OK);
            CHECK(mf.compute(pcm.data(), n, batch, bm, bfr) == TRANSCRIBE_OK);
            CHECK(im == bm && ifr == bfr && inc.size() == batch.size());
            for (size_t i = 0; i < inc.size() && i < batch.size(); ++i) {
                CHECK(inc[i] == batch[i]);
            }
        }
    }
}

}  // namespace

int main() {
    test_window();
    test_mel_filterbank();
    test_n_frames_for();
    test_incremental_equivalence();
    test_incremental_rejects_unsupported();
    test_incremental_state_reuse();

    if (g_failures > 0) {
        std::fprintf(stderr, "mel_unit: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "mel_unit: ok\n");
    return EXIT_SUCCESS;
}
