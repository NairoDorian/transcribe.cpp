// transcribe-rnnoise.cpp - classic RNNoise, ported from nnnoiseless 0.5.2
// (src/{lib,denoise,features,pitch,rnn,util}.rs, BSD-3-Clause; itself a port
// of xiph/rnnoise). See transcribe-rnnoise.h.
//
// Faithfulness notes: loops keep nnnoiseless's bounds and summation order,
// including two quirks that change numbers (pitch_xcorr's zip truncation at
// the end of the search range, and band gains above bin 400 forced to zero);
// floats round differently only where the FFT does (a mixed-radix 480-point
// FFT here, rustfft there).
#include "transcribe-rnnoise.h"

#include "rnnoise-weights.inl"
#include "transcribe-earshot-engine.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// Four-float SIMD for the FFT, the network and the pitch correlation (see F4).
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    define TRANSCRIBE_RNNOISE_SSE2 1
#    include <emmintrin.h>
#elif defined(__ARM_NEON) || defined(_M_ARM64)
#    define TRANSCRIBE_RNNOISE_NEON 1
#    include <arm_neon.h>
#endif

// x86-64: eight-float kernels for the network and the pitch correlation,
// compiled for AVX2 and picked at run time.
// clang-format off
#if defined(TRANSCRIBE_RNNOISE_SSE2) && (defined(__x86_64__) || defined(_M_X64))
#    define TRANSCRIBE_RNNOISE_AVX 1
#    include <immintrin.h>
#    if defined(__clang__)
#        define TRANSCRIBE_RNNOISE_PUSH_AVX2 _Pragma("clang attribute push(__attribute__((target(\"avx2\"))), apply_to = function)")
#        define TRANSCRIBE_RNNOISE_POP_TARGET _Pragma("clang attribute pop")
#    elif defined(__GNUC__)
#        define TRANSCRIBE_RNNOISE_PUSH_AVX2 _Pragma("GCC push_options") _Pragma("GCC target(\"avx2\")")
#        define TRANSCRIBE_RNNOISE_POP_TARGET _Pragma("GCC pop_options")
#    else
#        define TRANSCRIBE_RNNOISE_PUSH_AVX2  // MSVC compiles AVX intrinsics without target flags
#        define TRANSCRIBE_RNNOISE_POP_TARGET
#    endif
#endif
// clang-format on

namespace transcribe::rnnoise {
namespace {

constexpr int k_frame          = k_frame_size;  // 480
constexpr int k_win            = k_window;      // 960
constexpr int k_freq           = k_freq_size;   // 481
constexpr int k_pitch_min      = 60;
constexpr int k_pitch_max      = 768;
constexpr int k_pitch_frame    = 960;
constexpr int k_pitch_buf      = k_pitch_max + k_pitch_frame;  // 1728
constexpr int k_bands          = 22;
constexpr int k_ceps_mem       = 8;
constexpr int k_delta_ceps     = 6;
constexpr int k_features       = k_bands + 3 * k_delta_ceps + 2;  // 42
constexpr int k_shift          = 2;                               // FRAME_SIZE_SHIFT
constexpr int k_eband[k_bands] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100 };

// ---------------------------------------------------------------- complex

struct C {
    float r;
    float i;
};

inline C operator+(C a, C b) {
    return { a.r + b.r, a.i + b.i };
}

inline C operator-(C a, C b) {
    return { a.r - b.r, a.i - b.i };
}

inline C operator*(C a, C b) {
    return { a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r };
}

inline C conj(C a) {
    return { a.r, -a.i };
}

// ---------------------------------------------------------------- SIMD

// Four floats per operation: SSE2 on x86-64, NEON on ARM, plain arrays
// otherwise. No fused multiply-add, so the three give the same results.

struct F4 {
#if defined(TRANSCRIBE_RNNOISE_SSE2)
    __m128 v;

    F4() = default;

    explicit F4(__m128 x) : v(x) {}

    F4(float x) : v(_mm_set1_ps(x)) {}  // NOLINT(google-explicit-constructor): splat constants

    static F4 load(const float * p) { return F4(_mm_loadu_ps(p)); }

    void store(float * p) const { _mm_storeu_ps(p, v); }
#elif defined(TRANSCRIBE_RNNOISE_NEON)
    float32x4_t v;

    F4() = default;

    explicit F4(float32x4_t x) : v(x) {}

    F4(float x) : v(vdupq_n_f32(x)) {}  // NOLINT(google-explicit-constructor): splat constants

    static F4 load(const float * p) { return F4(vld1q_f32(p)); }

    void store(float * p) const { vst1q_f32(p, v); }
#else
    float v[4];

    F4() = default;

    F4(float x) : v{ x, x, x, x } {}  // NOLINT(google-explicit-constructor): splat constants

    static F4 load(const float * p) {
        F4 r;
        std::memcpy(r.v, p, sizeof(r.v));
        return r;
    }

    void store(float * p) const { std::memcpy(p, v, sizeof(v)); }
#endif
};

#if defined(TRANSCRIBE_RNNOISE_SSE2)
inline F4 operator+(F4 a, F4 b) {
    return F4(_mm_add_ps(a.v, b.v));
}

inline F4 operator-(F4 a, F4 b) {
    return F4(_mm_sub_ps(a.v, b.v));
}

inline F4 operator*(F4 a, F4 b) {
    return F4(_mm_mul_ps(a.v, b.v));
}

inline void transpose(F4 & a, F4 & b, F4 & c, F4 & d) {
    _MM_TRANSPOSE4_PS(a.v, b.v, c.v, d.v);
}
#elif defined(TRANSCRIBE_RNNOISE_NEON)
inline F4 operator+(F4 a, F4 b) {
    return F4(vaddq_f32(a.v, b.v));
}

inline F4 operator-(F4 a, F4 b) {
    return F4(vsubq_f32(a.v, b.v));
}

inline F4 operator*(F4 a, F4 b) {
    return F4(vmulq_f32(a.v, b.v));
}

inline void transpose(F4 & a, F4 & b, F4 & c, F4 & d) {
    const float32x4x2_t ab = vtrnq_f32(a.v, b.v);
    const float32x4x2_t cd = vtrnq_f32(c.v, d.v);
    a.v                    = vcombine_f32(vget_low_f32(ab.val[0]), vget_low_f32(cd.val[0]));
    b.v                    = vcombine_f32(vget_low_f32(ab.val[1]), vget_low_f32(cd.val[1]));
    c.v                    = vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0]));
    d.v                    = vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1]));
}
#else
template <typename Op> inline F4 lanes(F4 a, F4 b, Op op) {
    F4 r;
    for (int l = 0; l < 4; ++l) {
        r.v[l] = op(a.v[l], b.v[l]);
    }
    return r;
}

inline F4 operator+(F4 a, F4 b) {
    return lanes(a, b, [](float x, float y) { return x + y; });
}

inline F4 operator-(F4 a, F4 b) {
    return lanes(a, b, [](float x, float y) { return x - y; });
}

inline F4 operator*(F4 a, F4 b) {
    return lanes(a, b, [](float x, float y) { return x * y; });
}

inline void transpose(F4 & a, F4 & b, F4 & c, F4 & d) {
    F4 * rows[4] = { &a, &b, &c, &d };
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            std::swap(rows[i]->v[j], rows[j]->v[i]);
        }
    }
}
#endif

#if defined(TRANSCRIBE_RNNOISE_AVX)
// Whether the 8-wide kernels run: on by default where the CPU has AVX2.
// They add the same terms in the same order as the 4-wide ones (multiply,
// then add; no FMA), so the output does not depend on it.
std::atomic<bool> & wide_kernels() {
    static std::atomic<bool> on{ earshot::isa_supported(earshot::Isa::Avx2) };
    return on;
}
#endif

// ---------------------------------------------------------------- FFT

// 480-point complex FFT on split (re, im) arrays: a Stockham autosort with
// radices 4, 4, 2, 3, 5, four lanes at a time. Stage (r, m, s) treats the data
// as s interleaved sequences of length n = r m, element p of sequence q at
// [q + s p], and writes y[q + s (r p + j)] = DFT_r(x[q + s (p + k m)])_j
// times w_n^(j p); after the last stage the result is in natural order. The
// first stage (s = 1) runs four p at a time and transposes; the others run
// four q at a time (s is a multiple of four from there on).
struct Fft {
    static constexpr int N        = 480;
    static constexpr int k_stages = 5;
    static constexpr int radix[5] = { 4, 4, 2, 3, 5 };
    std::vector<float>   tw_re[k_stages];  // [(j - 1) * m + p]
    std::vector<float>   tw_im[k_stages];
    C                    post[k_freq];     // exp(-2 pi i k / 960)

    Fft() {
        const double pi = 3.14159265358979323846;
        int          n  = N;
        for (int st = 0; st < k_stages; ++st) {
            const int r = radix[st];
            const int m = n / r;
            tw_re[st].resize(static_cast<size_t>((r - 1) * m));
            tw_im[st].resize(static_cast<size_t>((r - 1) * m));
            for (int j = 1; j < r; ++j) {
                for (int p = 0; p < m; ++p) {
                    const double a                                  = -2.0 * pi * j * p / n;
                    tw_re[st][static_cast<size_t>((j - 1) * m + p)] = static_cast<float>(std::cos(a));
                    tw_im[st][static_cast<size_t>((j - 1) * m + p)] = static_cast<float>(std::sin(a));
                }
            }
            n = m;
        }
        for (int k = 0; k < k_freq; ++k) {
            const double a = -2.0 * pi * k / k_win;
            post[k]        = { static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a)) };
        }
    }
};

const Fft & fft() {
    static const Fft f;
    return f;
}

// In-place forward DFT of r points, a[k] -> b[j] = sum_k a[k] e^(-2 pi i j k / r).
template <int R> inline void butterfly(F4 * re, F4 * im);

template <> inline void butterfly<2>(F4 * re, F4 * im) {
    const F4 r0 = re[0] + re[1];
    const F4 i0 = im[0] + im[1];
    re[1]       = re[0] - re[1];
    im[1]       = im[0] - im[1];
    re[0]       = r0;
    im[0]       = i0;
}

template <> inline void butterfly<3>(F4 * re, F4 * im) {
    const F4 s   = 0.86602540378443864676f;  // sin(2 pi / 3)
    const F4 t1r = re[1] + re[2];
    const F4 t1i = im[1] + im[2];
    const F4 t2r = (re[1] - re[2]) * s;
    const F4 t2i = (im[1] - im[2]) * s;
    const F4 m1r = re[0] - t1r * 0.5f;
    const F4 m1i = im[0] - t1i * 0.5f;
    re[0]        = re[0] + t1r;
    im[0]        = im[0] + t1i;
    re[1]        = m1r + t2i;
    im[1]        = m1i - t2r;
    re[2]        = m1r - t2i;
    im[2]        = m1i + t2r;
}

template <> inline void butterfly<4>(F4 * re, F4 * im) {
    const F4 t0r = re[0] + re[2];
    const F4 t0i = im[0] + im[2];
    const F4 t1r = re[0] - re[2];
    const F4 t1i = im[0] - im[2];
    const F4 t2r = re[1] + re[3];
    const F4 t2i = im[1] + im[3];
    const F4 t3r = re[1] - re[3];
    const F4 t3i = im[1] - im[3];
    re[0]        = t0r + t2r;
    im[0]        = t0i + t2i;
    re[2]        = t0r - t2r;
    im[2]        = t0i - t2i;
    re[1]        = t1r + t3i;  // t1 - i t3
    im[1]        = t1i - t3r;
    re[3]        = t1r - t3i;  // t1 + i t3
    im[3]        = t1i + t3r;
}

template <> inline void butterfly<5>(F4 * re, F4 * im) {
    const F4 c1  = 0.30901699437494742410f;   // cos(2 pi / 5)
    const F4 c2  = -0.80901699437494742410f;  // cos(4 pi / 5)
    const F4 s1  = 0.95105651629515357212f;   // sin(2 pi / 5)
    const F4 s2  = 0.58778525229247312917f;   // sin(4 pi / 5)
    const F4 t1r = re[1] + re[4];
    const F4 t1i = im[1] + im[4];
    const F4 t2r = re[2] + re[3];
    const F4 t2i = im[2] + im[3];
    const F4 t3r = re[1] - re[4];
    const F4 t3i = im[1] - im[4];
    const F4 t4r = re[2] - re[3];
    const F4 t4i = im[2] - im[3];
    const F4 m1r = re[0] + t1r * c1 + t2r * c2;
    const F4 m1i = im[0] + t1i * c1 + t2i * c2;
    const F4 m2r = re[0] + t1r * c2 + t2r * c1;
    const F4 m2i = im[0] + t1i * c2 + t2i * c1;
    const F4 n1r = t3r * s1 + t4r * s2;
    const F4 n1i = t3i * s1 + t4i * s2;
    const F4 n2r = t3r * s2 - t4r * s1;
    const F4 n2i = t3i * s2 - t4i * s1;
    re[0]        = re[0] + t1r + t2r;
    im[0]        = im[0] + t1i + t2i;
    re[1]        = m1r + n1i;  // m1 - i n1
    im[1]        = m1i - n1r;
    re[4]        = m1r - n1i;  // m1 + i n1
    im[4]        = m1i + n1r;
    re[2]        = m2r + n2i;  // m2 - i n2
    im[2]        = m2i - n2r;
    re[3]        = m2r - n2i;  // m2 + i n2
    im[3]        = m2i + n2r;
}

// Stage with s a multiple of four: four sequences (q) per step.
template <int R>
void fft_stage(const float * xr,
               const float * xi,
               float *       yr,
               float *       yi,
               int           m,
               int           s,
               const float * twr,
               const float * twi) {
    for (int p = 0; p < m; ++p) {
        F4 wr[R];
        F4 wi[R];
        for (int j = 1; j < R; ++j) {
            wr[j] = twr[(j - 1) * m + p];
            wi[j] = twi[(j - 1) * m + p];
        }
        for (int q = 0; q < s; q += 4) {
            F4 ar[R];
            F4 ai[R];
            for (int k = 0; k < R; ++k) {
                ar[k] = F4::load(xr + q + s * (p + k * m));
                ai[k] = F4::load(xi + q + s * (p + k * m));
            }
            butterfly<R>(ar, ai);
            ar[0].store(yr + q + s * (R * p));
            ai[0].store(yi + q + s * (R * p));
            for (int j = 1; j < R; ++j) {
                (ar[j] * wr[j] - ai[j] * wi[j]).store(yr + q + s * (R * p + j));
                (ar[j] * wi[j] + ai[j] * wr[j]).store(yi + q + s * (R * p + j));
            }
        }
    }
}

// First stage (s = 1), radix 4: four p per step, outputs transposed into
// place (y[4 p + j]).
void fft_stage_first4(const float * xr,
                      const float * xi,
                      float *       yr,
                      float *       yi,
                      int           m,
                      const float * twr,
                      const float * twi) {
    for (int p = 0; p < m; p += 4) {
        F4 ar[4];
        F4 ai[4];
        for (int k = 0; k < 4; ++k) {
            ar[k] = F4::load(xr + p + k * m);
            ai[k] = F4::load(xi + p + k * m);
        }
        butterfly<4>(ar, ai);
        for (int j = 1; j < 4; ++j) {
            const F4 wr = F4::load(twr + (j - 1) * m + p);
            const F4 wi = F4::load(twi + (j - 1) * m + p);
            const F4 r  = ar[j] * wr - ai[j] * wi;
            ai[j]       = ar[j] * wi + ai[j] * wr;
            ar[j]       = r;
        }
        transpose(ar[0], ar[1], ar[2], ar[3]);
        transpose(ai[0], ai[1], ai[2], ai[3]);
        for (int t = 0; t < 4; ++t) {
            ar[t].store(yr + 4 * (p + t));
            ai[t].store(yi + 4 * (p + t));
        }
    }
}

// Forward, unnormalized, in place on (re, im).
void fft480(float * re, float * im) {
    const Fft &       t = fft();
    alignas(16) float br[Fft::N];
    alignas(16) float bi[Fft::N];
    float *           xr = re;
    float *           xi = im;
    float *           yr = br;
    float *           yi = bi;
    int               m  = Fft::N;
    int               s  = 1;
    for (int st = 0; st < Fft::k_stages; ++st) {
        const int     r   = Fft::radix[st];
        const float * twr = t.tw_re[st].data();
        const float * twi = t.tw_im[st].data();
        m /= r;
        if (st == 0) {
            fft_stage_first4(xr, xi, yr, yi, m, twr, twi);
        } else if (r == 2) {
            fft_stage<2>(xr, xi, yr, yi, m, s, twr, twi);
        } else if (r == 3) {
            fft_stage<3>(xr, xi, yr, yi, m, s, twr, twi);
        } else if (r == 4) {
            fft_stage<4>(xr, xi, yr, yi, m, s, twr, twi);
        } else {
            fft_stage<5>(xr, xi, yr, yi, m, s, twr, twi);
        }
        s *= r;
        std::swap(xr, yr);
        std::swap(xi, yi);
    }
    if (xr != re) {
        std::memcpy(re, xr, sizeof(br));
        std::memcpy(im, xi, sizeof(bi));
    }
}

// Unnormalized 960-point real DFT from one 480-point complex FFT of the
// (even, odd) sample pairs.
void rfft(const float * in, C * x) {
    const Fft &       t = fft();
    alignas(16) float zr[Fft::N];
    alignas(16) float zi[Fft::N];
    for (int n = 0; n < Fft::N; ++n) {
        zr[n] = in[2 * n];
        zi[n] = in[2 * n + 1];
    }
    fft480(zr, zi);
    for (int k = 0; k <= Fft::N; ++k) {
        const int ka = k % Fft::N;
        const int kb = (Fft::N - k) % Fft::N;
        const C   e  = { (zr[ka] + zr[kb]) * 0.5f, (zi[ka] - zi[kb]) * 0.5f };
        const C   d  = { (zr[ka] - zr[kb]) * 0.5f, (zi[ka] + zi[kb]) * 0.5f };
        const C   o  = { d.i, -d.r };  // d / i
        x[k]         = e + t.post[k] * o;
    }
}

// Unnormalized inverse over the Hermitian extension X[k + 480] = conj(X[480 - k]),
// as conj(FFT(conj(c))).
void irfft(const C * x, float * out) {
    const Fft &       t = fft();
    alignas(16) float cr[Fft::N];
    alignas(16) float ci[Fft::N];
    for (int k = 0; k < Fft::N; ++k) {
        const C xk = x[k];
        const C xn = conj(x[Fft::N - k]);
        const C a  = xk + xn;
        const C b  = (xk - xn) * conj(t.post[k]);
        cr[k]      = a.r - b.i;     // a + i b
        ci[k]      = -(a.i + b.r);  // conjugated
    }
    fft480(cr, ci);
    for (int n = 0; n < Fft::N; ++n) {
        out[2 * n]     = cr[n];
        out[2 * n + 1] = -ci[n];
    }
}

// ---------------------------------------------------------------- tables

constexpr int k_band_bins = k_eband[k_bands - 1] << k_shift;  // 400: bins above get no band

struct Common {
    float window[k_win];
    float dct_table[k_bands * k_bands];
    float wnorm;
    // Per bin below k_band_bins: its band and the interpolation weights
    // (1 - frac, frac) toward that band and the next, frac = j / band_size.
    int   band_of[k_band_bins];
    float lo_w[k_band_bins];
    float hi_w[k_band_bins];

    Common() {
        const double pi = 3.14159265358979323846;
        for (int i = 0; i < k_frame; ++i) {
            const double s        = std::sin(0.5 * pi * (i + 0.5) / k_frame);
            const float  w        = static_cast<float>(std::sin(0.5 * pi * s * s));
            window[i]             = w;
            window[k_win - i - 1] = w;
        }
        float sum = 0.0f;
        for (float w : window) {
            sum += w * w;
        }
        wnorm = 1.0f / sum;
        for (int i = 0; i < k_bands; ++i) {
            for (int j = 0; j < k_bands; ++j) {
                dct_table[i * k_bands + j] = static_cast<float>(std::cos((i + 0.5) * j * pi / k_bands));
                if (j == 0) {
                    dct_table[i * k_bands + j] *= std::sqrt(0.5f);
                }
            }
        }
        for (int i = 0; i < k_bands - 1; ++i) {
            const int band_size = (k_eband[i + 1] - k_eband[i]) << k_shift;
            for (int j = 0; j < band_size; ++j) {
                const float frac = static_cast<float>(j) / static_cast<float>(band_size);
                const int   idx  = (k_eband[i] << k_shift) + j;
                band_of[idx]     = i;
                lo_w[idx]        = 1.0f - frac;
                hi_w[idx]        = frac;
            }
        }
    }
};

const Common & common() {
    static const Common c;
    return c;
}

void dct(float * out, const float * x) {
    const Common & c = common();
    for (int i = 0; i < k_bands; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < k_bands; ++j) {
            sum += x[j] * c.dct_table[j * k_bands + i];
        }
        out[i] = static_cast<float>(static_cast<double>(sum) * std::sqrt(2.0 / k_bands));
    }
}

// Bins in ascending order: the same sequence of additions as nnnoiseless's
// band-by-band loop.
void compute_band_corr(float * out, const C * x, const C * p) {
    const Common & c = common();
    std::fill(out, out + k_bands, 0.0f);
    for (int idx = 0; idx < k_band_bins; ++idx) {
        const int   i    = c.band_of[idx];
        const float corr = x[idx].r * p[idx].r + x[idx].i * p[idx].i;
        out[i] += c.lo_w[idx] * corr;
        out[i + 1] += c.hi_w[idx] * corr;
    }
    out[0] *= 2.0f;
    out[k_bands - 1] *= 2.0f;
}

void interp_band_gain(float * out, const float * band_e) {
    const Common & c = common();
    for (int idx = 0; idx < k_band_bins; ++idx) {
        const int i = c.band_of[idx];
        out[idx]    = c.lo_w[idx] * band_e[i] + c.hi_w[idx] * band_e[i + 1];
    }
    std::fill(out + k_band_bins, out + k_freq, 0.0f);
}

// ---------------------------------------------------------------- math

constexpr float k_tansig[201] = {
    0.000000f, 0.039979f, 0.079830f, 0.119427f, 0.158649f, 0.197375f, 0.235496f, 0.272905f, 0.309507f, 0.345214f,
    0.379949f, 0.413644f, 0.446244f, 0.477700f, 0.507977f, 0.537050f, 0.564900f, 0.591519f, 0.616909f, 0.641077f,
    0.664037f, 0.685809f, 0.706419f, 0.725897f, 0.744277f, 0.761594f, 0.777888f, 0.793199f, 0.807569f, 0.821040f,
    0.833655f, 0.845456f, 0.856485f, 0.866784f, 0.876393f, 0.885352f, 0.893698f, 0.901468f, 0.908698f, 0.915420f,
    0.921669f, 0.927473f, 0.932862f, 0.937863f, 0.942503f, 0.946806f, 0.950795f, 0.954492f, 0.957917f, 0.961090f,
    0.964028f, 0.966747f, 0.969265f, 0.971594f, 0.973749f, 0.975743f, 0.977587f, 0.979293f, 0.980869f, 0.982327f,
    0.983675f, 0.984921f, 0.986072f, 0.987136f, 0.988119f, 0.989027f, 0.989867f, 0.990642f, 0.991359f, 0.992020f,
    0.992631f, 0.993196f, 0.993718f, 0.994199f, 0.994644f, 0.995055f, 0.995434f, 0.995784f, 0.996108f, 0.996407f,
    0.996682f, 0.996937f, 0.997172f, 0.997389f, 0.997590f, 0.997775f, 0.997946f, 0.998104f, 0.998249f, 0.998384f,
    0.998508f, 0.998623f, 0.998728f, 0.998826f, 0.998916f, 0.999000f, 0.999076f, 0.999147f, 0.999213f, 0.999273f,
    0.999329f, 0.999381f, 0.999428f, 0.999472f, 0.999513f, 0.999550f, 0.999585f, 0.999617f, 0.999646f, 0.999673f,
    0.999699f, 0.999722f, 0.999743f, 0.999763f, 0.999781f, 0.999798f, 0.999813f, 0.999828f, 0.999841f, 0.999853f,
    0.999865f, 0.999875f, 0.999885f, 0.999893f, 0.999902f, 0.999909f, 0.999916f, 0.999923f, 0.999929f, 0.999934f,
    0.999939f, 0.999944f, 0.999948f, 0.999952f, 0.999956f, 0.999959f, 0.999962f, 0.999965f, 0.999968f, 0.999970f,
    0.999973f, 0.999975f, 0.999977f, 0.999978f, 0.999980f, 0.999982f, 0.999983f, 0.999984f, 0.999986f, 0.999987f,
    0.999988f, 0.999989f, 0.999990f, 0.999990f, 0.999991f, 0.999992f, 0.999992f, 0.999993f, 0.999994f, 0.999994f,
    0.999994f, 0.999995f, 0.999995f, 0.999996f, 0.999996f, 0.999996f, 0.999997f, 0.999997f, 0.999997f, 0.999997f,
    0.999997f, 0.999998f, 0.999998f, 0.999998f, 0.999998f, 0.999998f, 0.999998f, 0.999999f, 0.999999f, 0.999999f,
    0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f, 0.999999f,
    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
    1.000000f,
};

float tansig_approx(float x) {
    // Tests reversed to catch NaNs.
    if (!(x < 8.0f)) {
        return 1.0f;
    }
    if (!(x > -8.0f)) {
        return -1.0f;
    }
    float sign = 1.0f;
    if (x < 0.0f) {
        x    = -x;
        sign = -1.0f;
    }
    // floor() of a value in [0.5, 200.5): truncation, which is one
    // instruction where floor() can be a library call.
    const int   ii = static_cast<int>(0.5f + 25.0f * x);
    const float i  = static_cast<float>(ii);
    x -= 0.04f * i;
    const float y  = k_tansig[ii];
    const float dy = 1.0f - y * y;
    return sign * (y + x * dy * (1.0f - y * x));
}

float sigmoid_approx(float x) {
    return 0.5f + 0.5f * tansig_approx(0.5f * x);
}

// Four partial sums (one per lane) added in order: nnnoiseless's s0..s3.
float inner_prod(const float * xs, const float * ys, int n) {
    const int n_4 = n - n % 4;
    F4        acc = 0.0f;
    for (int i = 0; i < n_4; i += 4) {
        acc = acc + F4::load(xs + i) * F4::load(ys + i);
    }
    alignas(16) float s[4];
    acc.store(s);
    float sum = s[0] + s[1] + s[2] + s[3];
    for (int i = n_4; i < n; ++i) {
        sum += xs[i] * ys[i];
    }
    return sum;
}

// xcorr[i] = sum_j xs[j] * ys[i + j] for i < n_xcorr. nnnoiseless takes lags
// four at a time and, per group, sums j in order over min(xs, ys[i + 4..]) / 4
// chunks of four and then the xs tail: a Rust zip that drops the last chunks
// when ys runs short. Here each group is one F4 whose lanes sum the same
// terms in the same order; four groups run side by side, because the adds'
// latency, not the loads, is the bottleneck.
int xcorr_chunked(int n_xs, int n_ys, int i) {
    return 4 * std::min(n_xs / 4, (n_ys - (i + 4)) / 4);
}

// Lags from `i` (a multiple of four) on.
void pitch_xcorr4(const float * xs, int n_xs, const float * ys, int n_ys, float * xcorr, int n_xcorr, int i) {
    const int  xcorr_len_4 = n_xcorr - n_xcorr % 4;
    const int  xs_len_4    = n_xs - n_xs % 4;
    const auto chunked     = [&](int lag) {
        return xcorr_chunked(n_xs, n_ys, lag);
    };
    for (; i + 16 <= xcorr_len_4 && chunked(i) == chunked(i + 12); i += 16) {
        const int  n    = chunked(i);
        F4         c0   = 0.0f;
        F4         c1   = 0.0f;
        F4         c2   = 0.0f;
        F4         c3   = 0.0f;
        const auto step = [&](int j) {
            const F4 x = xs[j];
            c0         = c0 + x * F4::load(ys + i + j);
            c1         = c1 + x * F4::load(ys + i + 4 + j);
            c2         = c2 + x * F4::load(ys + i + 8 + j);
            c3         = c3 + x * F4::load(ys + i + 12 + j);
        };
        for (int j = 0; j < n; ++j) {
            step(j);
        }
        for (int j = xs_len_4; j < n_xs; ++j) {
            step(j);
        }
        c0.store(xcorr + i);
        c1.store(xcorr + i + 4);
        c2.store(xcorr + i + 8);
        c3.store(xcorr + i + 12);
    }
    for (; i < xcorr_len_4; i += 4) {
        const int n = chunked(i);
        F4        c = 0.0f;
        for (int j = 0; j < n; ++j) {
            c = c + F4(xs[j]) * F4::load(ys + i + j);
        }
        for (int j = xs_len_4; j < n_xs; ++j) {
            c = c + F4(xs[j]) * F4::load(ys + i + j);
        }
        c.store(xcorr + i);
    }
    for (i = xcorr_len_4; i < n_xcorr; ++i) {
        float     sum = 0.0f;
        const int n   = std::min(n_xs, n_ys - i);
        for (int j = 0; j < n; ++j) {
            sum += xs[j] * ys[i + j];
        }
        xcorr[i] = sum;
    }
}

#if defined(TRANSCRIBE_RNNOISE_AVX)
TRANSCRIBE_RNNOISE_PUSH_AVX2

// 32 lags per block (two groups of four per register), the rest 4-wide.
void pitch_xcorr8(const float * xs, int n_xs, const float * ys, int n_ys, float * xcorr, int n_xcorr) {
    const int xcorr_len_4 = n_xcorr - n_xcorr % 4;
    const int xs_len_4    = n_xs - n_xs % 4;
    int       i           = 0;
    for (; i + 32 <= xcorr_len_4 && xcorr_chunked(n_xs, n_ys, i) == xcorr_chunked(n_xs, n_ys, i + 28); i += 32) {
        const int  n    = xcorr_chunked(n_xs, n_ys, i);
        __m256     c0   = _mm256_setzero_ps();
        __m256     c1   = _mm256_setzero_ps();
        __m256     c2   = _mm256_setzero_ps();
        __m256     c3   = _mm256_setzero_ps();
        const auto step = [&](int j) {
            const __m256  x = _mm256_set1_ps(xs[j]);
            const float * y = ys + i + j;
            c0              = _mm256_add_ps(c0, _mm256_mul_ps(x, _mm256_loadu_ps(y)));
            c1              = _mm256_add_ps(c1, _mm256_mul_ps(x, _mm256_loadu_ps(y + 8)));
            c2              = _mm256_add_ps(c2, _mm256_mul_ps(x, _mm256_loadu_ps(y + 16)));
            c3              = _mm256_add_ps(c3, _mm256_mul_ps(x, _mm256_loadu_ps(y + 24)));
        };
        for (int j = 0; j < n; ++j) {
            step(j);
        }
        for (int j = xs_len_4; j < n_xs; ++j) {
            step(j);
        }
        _mm256_storeu_ps(xcorr + i, c0);
        _mm256_storeu_ps(xcorr + i + 8, c1);
        _mm256_storeu_ps(xcorr + i + 16, c2);
        _mm256_storeu_ps(xcorr + i + 24, c3);
    }
    pitch_xcorr4(xs, n_xs, ys, n_ys, xcorr, n_xcorr, i);
}

TRANSCRIBE_RNNOISE_POP_TARGET
#endif

void pitch_xcorr(const float * xs, int n_xs, const float * ys, int n_ys, float * xcorr, int n_xcorr) {
#if defined(TRANSCRIBE_RNNOISE_AVX)
    if (wide_kernels().load(std::memory_order_relaxed)) {
        pitch_xcorr8(xs, n_xs, ys, n_ys, xcorr, n_xcorr);
        return;
    }
#endif
    pitch_xcorr4(xs, n_xs, ys, n_ys, xcorr, n_xcorr, 0);
}

void find_best_pitch(const float * xcorr, int n_xcorr, const float * ys, int len, int & best, int & second) {
    float best_num = -1.0f, second_num = -1.0f, best_den = 0.0f, second_den = 0.0f;
    best         = 0;
    second       = 1;
    float y_norm = 1.0f;
    for (int j = 0; j < len; ++j) {
        y_norm += ys[j] * ys[j];
    }
    for (int i = 0; i < n_xcorr; ++i) {
        const float corr = xcorr[i];
        if (corr > 0.0f) {
            const float num = corr * corr;
            if (num * second_den > second_num * y_norm) {
                if (num * best_den > best_num * y_norm) {
                    second_num = best_num;
                    second_den = best_den;
                    second     = best;
                    best_num   = num;
                    best_den   = y_norm;
                    best       = i;
                } else {
                    second_num = num;
                    second_den = y_norm;
                    second     = i;
                }
            }
        }
        y_norm += ys[i + len] * ys[i + len] - ys[i] * ys[i];
        y_norm = std::max(y_norm, 1.0f);
    }
}

void lpc(float * lpc_out, int p, const float * ac) {
    float error = ac[0];
    std::fill(lpc_out, lpc_out + p, 0.0f);
    if (ac[0] == 0.0f) {
        return;
    }
    for (int i = 0; i < p; ++i) {
        float rr = 0.0f;
        for (int j = 0; j < i; ++j) {
            rr += lpc_out[j] * ac[i - j];
        }
        rr += ac[i + 1];
        const float r = -rr / error;
        lpc_out[i]    = r;
        for (int j = 0; j < (i + 1) / 2; ++j) {
            const float t1     = lpc_out[j];
            const float t2     = lpc_out[i - 1 - j];
            lpc_out[j]         = t1 + r * t2;
            lpc_out[i - 1 - j] = t2 + r * t1;
        }
        error = error - r * r * error;
        if (error < 0.001f * ac[0]) {
            return;
        }
    }
}

void fir5_in_place(float * xs, int n, const float * num) {
    float m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0;
    for (int i = 0; i < n; ++i) {
        const float x   = xs[i];
        const float out = x + num[0] * m0 + num[1] * m1 + num[2] * m2 + num[3] * m3 + num[4] * m4;
        m4              = m3;
        m3              = m2;
        m2              = m1;
        m1              = m0;
        m0              = x;
        xs[i]           = out;
    }
}

void celt_autocorr(const float * x, int n, float * ac, int n_ac) {
    const int lag    = n_ac - 1;
    const int fast_n = n - lag;
    pitch_xcorr(x, fast_n, x, n, ac, n_ac);
    for (int k = 0; k < n_ac; ++k) {
        float d = 0.0f;
        for (int i = k + fast_n; i < n; ++i) {
            d += x[i] * x[i - k];
        }
        ac[k] += d;
    }
}

void pitch_downsample(const float * x, int n_x, float * x_lp) {
    const int n_lp = n_x / 2;
    for (int i = 1; i < n_lp; ++i) {
        x_lp[i] = ((x[2 * i - 1] + x[2 * i + 1]) / 2.0f + x[2 * i]) / 2.0f;
    }
    x_lp[0] = (x[1] / 2.0f + x[0]) / 2.0f;
    float ac[5];
    celt_autocorr(x_lp, n_lp, ac, 5);
    ac[0] *= 1.0001f;  // noise floor -40 dB
    for (int i = 1; i < 5; ++i) {
        ac[i] -= ac[i] * (0.008f * static_cast<float>(i)) * (0.008f * static_cast<float>(i));
    }
    float lpc_c[4];
    lpc(lpc_c, 4, ac);
    float tmp = 1.0f;
    for (float & c : lpc_c) {
        tmp *= 0.9f;
        c *= tmp;
    }
    const float lpc2[5] = { lpc_c[0] + 0.8f, lpc_c[1] + 0.8f * lpc_c[0], lpc_c[2] + 0.8f * lpc_c[1],
                            lpc_c[3] + 0.8f * lpc_c[2], 0.8f * lpc_c[3] };
    fir5_in_place(x_lp, n_lp, lpc2);
}

float pitch_gain(float xy, float xx, float yy) {
    return xy / std::sqrt(1.0f + xx * yy);
}

constexpr int k_second_check[16] = { 0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2 };

// ---------------------------------------------------------------- network

struct Dense {
    int                nb_inputs  = 0;
    int                nb_neurons = 0;
    int                activation = 0;  // 0 tanh, 1 sigmoid, 2 relu
    std::vector<float> weights;         // [input][neuron]
    std::vector<float> bias;
};

struct Gru {
    int                nb_inputs  = 0;
    int                nb_neurons = 0;
    int                activation = 0;
    std::vector<float> input_weights;      // [input][3 * neurons]
    std::vector<float> recurrent_weights;  // [neuron][3 * neurons]
    std::vector<float> bias;               // [3 * neurons]
};

struct Model {
    Dense input_dense;
    Gru   vad_gru;
    Gru   noise_gru;
    Gru   denoise_gru;
    Dense denoise_output;
    Dense vad_output;

    Model() {
        const int8_t * p   = RNNOISE_WEIGHTS_BIN;
        const auto     arr = [&p](std::vector<float> & out, size_t n) {
            out.resize(n);
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<float>(*p++);
            }
        };
        const auto dense = [&](Dense & d) {
            d.nb_inputs  = *p++;
            d.nb_neurons = *p++;
            d.activation = *p++;
            arr(d.weights, static_cast<size_t>(d.nb_inputs) * d.nb_neurons);
            arr(d.bias, static_cast<size_t>(d.nb_neurons));
        };
        const auto gru = [&](Gru & g) {
            g.nb_inputs  = *p++;
            g.nb_neurons = *p++;
            g.activation = *p++;
            arr(g.input_weights, 3u * g.nb_inputs * g.nb_neurons);
            arr(g.recurrent_weights, 3u * g.nb_neurons * g.nb_neurons);
            arr(g.bias, 3u * g.nb_neurons);
        };
        dense(input_dense);
        gru(vad_gru);
        gru(noise_gru);
        gru(denoise_gru);
        dense(denoise_output);
        dense(vad_output);
    }
};

const Model & model() {
    static const Model m;
    return m;
}

constexpr float k_weights_scale = 1.0f / 256.0f;
constexpr int   k_max_neurons   = 128;

float activate(int activation, float x) {
    switch (activation) {
        case 1:
            return sigmoid_approx(x);
        case 2:
            return std::max(x, 0.0f);
        default:
            return tansig_approx(x);
    }
}

// out[j] += sum over inputs i, in order, of w[i * stride + offset + j] * in[i]:
// 16 outputs held in registers across all inputs, then 4, then single ones.
// Each output keeps nnnoiseless's summation order.
void mul_add4(float * out, int n_out, const float * w, int stride, int offset, const float * in, int n_in) {
    const float * base = w + offset;
    int           j    = 0;
    for (; j + 16 <= n_out; j += 16) {
        F4            a0  = F4::load(out + j);
        F4            a1  = F4::load(out + j + 4);
        F4            a2  = F4::load(out + j + 8);
        F4            a3  = F4::load(out + j + 12);
        const float * col = base + j;
        for (int i = 0; i < n_in; ++i, col += stride) {
            const F4 x = in[i];
            a0         = a0 + F4::load(col) * x;
            a1         = a1 + F4::load(col + 4) * x;
            a2         = a2 + F4::load(col + 8) * x;
            a3         = a3 + F4::load(col + 12) * x;
        }
        a0.store(out + j);
        a1.store(out + j + 4);
        a2.store(out + j + 8);
        a3.store(out + j + 12);
    }
    for (; j + 4 <= n_out; j += 4) {
        F4            a   = F4::load(out + j);
        const float * col = base + j;
        for (int i = 0; i < n_in; ++i, col += stride) {
            a = a + F4::load(col) * F4(in[i]);
        }
        a.store(out + j);
    }
    for (; j < n_out; ++j) {
        float         a   = out[j];
        const float * col = base + j;
        for (int i = 0; i < n_in; ++i, col += stride) {
            a += *col * in[i];
        }
        out[j] = a;
    }
}

#if defined(TRANSCRIBE_RNNOISE_AVX)
TRANSCRIBE_RNNOISE_PUSH_AVX2

// 32 outputs per block, then 8, the rest 4-wide.
void mul_add8(float * out, int n_out, const float * w, int stride, int offset, const float * in, int n_in) {
    const float * base = w + offset;
    int           j    = 0;
    for (; j + 32 <= n_out; j += 32) {
        __m256        a0  = _mm256_loadu_ps(out + j);
        __m256        a1  = _mm256_loadu_ps(out + j + 8);
        __m256        a2  = _mm256_loadu_ps(out + j + 16);
        __m256        a3  = _mm256_loadu_ps(out + j + 24);
        const float * col = base + j;
        for (int i = 0; i < n_in; ++i, col += stride) {
            const __m256 x = _mm256_set1_ps(in[i]);
            a0             = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(col), x));
            a1             = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(col + 8), x));
            a2             = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_loadu_ps(col + 16), x));
            a3             = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_loadu_ps(col + 24), x));
        }
        _mm256_storeu_ps(out + j, a0);
        _mm256_storeu_ps(out + j + 8, a1);
        _mm256_storeu_ps(out + j + 16, a2);
        _mm256_storeu_ps(out + j + 24, a3);
    }
    for (; j + 8 <= n_out; j += 8) {
        __m256        a   = _mm256_loadu_ps(out + j);
        const float * col = base + j;
        for (int i = 0; i < n_in; ++i, col += stride) {
            a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_loadu_ps(col), _mm256_set1_ps(in[i])));
        }
        _mm256_storeu_ps(out + j, a);
    }
    if (j < n_out) {
        mul_add4(out + j, n_out - j, w, stride, offset + j, in, n_in);
    }
}

TRANSCRIBE_RNNOISE_POP_TARGET
#endif

void mul_add(float * out, int n_out, const float * w, int stride, int offset, const float * in, int n_in) {
#if defined(TRANSCRIBE_RNNOISE_AVX)
    if (wide_kernels().load(std::memory_order_relaxed)) {
        mul_add8(out, n_out, w, stride, offset, in, n_in);
        return;
    }
#endif
    mul_add4(out, n_out, w, stride, offset, in, n_in);
}

void dense_compute(const Dense & d, float * out, const float * in) {
    std::copy(d.bias.begin(), d.bias.end(), out);
    mul_add(out, d.nb_neurons, d.weights.data(), d.nb_neurons, 0, in, d.nb_inputs);
    for (int j = 0; j < d.nb_neurons; ++j) {
        out[j] = activate(d.activation, out[j] * k_weights_scale);
    }
}

// Gate sums in nnnoiseless's order (bias, input terms, recurrent terms), with
// the input terms of all three gates in one pass.
void gru_compute(const Gru & g, float * state, const float * in) {
    const int n      = g.nb_neurons;
    const int stride = 3 * n;
    float     zrh[3 * k_max_neurons];
    float     rs[k_max_neurons];
    float *   z = zrh;
    float *   r = zrh + n;
    float *   h = zrh + 2 * n;
    std::copy_n(g.bias.data(), stride, zrh);
    mul_add(zrh, stride, g.input_weights.data(), stride, 0, in, g.nb_inputs);
    mul_add(zrh, 2 * n, g.recurrent_weights.data(), stride, 0, state, n);
    for (int j = 0; j < n; ++j) {
        z[j]  = sigmoid_approx(k_weights_scale * z[j]);
        rs[j] = state[j] * sigmoid_approx(k_weights_scale * r[j]);
    }
    mul_add(h, n, g.recurrent_weights.data(), stride, 2 * n, rs, n);
    for (int j = 0; j < n; ++j) {
        const float hj = activate(g.activation, k_weights_scale * h[j]);
        state[j]       = z[j] * state[j] + (1.0f - z[j]) * hj;
    }
}

}  // namespace

// ---------------------------------------------------------------- state

struct Denoiser::State {
    // Features (DenoiseFeatures).
    float input_mem[k_pitch_buf]                        = {};
    float cepstral_mem[k_ceps_mem][k_bands]             = {};
    int   mem_id                                        = 0;
    float mem_hp_x[2]                                   = {};
    float synthesis_mem[k_frame]                        = {};
    float window_buf[k_win]                             = {};
    C     x[k_freq]                                     = {};
    C     p[k_freq]                                     = {};
    float ex[k_bands]                                   = {};
    float ep[k_bands]                                   = {};
    float exp[k_bands]                                  = {};
    float features[k_features]                          = {};
    // Pitch.
    int   last_period                                   = 0;
    float last_gain                                     = 0.0f;
    float pitch_buf[k_pitch_buf / 2]                    = {};
    float scratch[k_pitch_max + 1]                      = {};
    float scratch2[k_pitch_frame / 4]                   = {};
    float scratch3[(k_pitch_max - 3 * k_pitch_min) / 2] = {};
    // Network.
    float vad_state[k_max_neurons]                      = {};
    float noise_state[k_max_neurons]                    = {};
    float denoise_state[k_max_neurons]                  = {};
    float lastg[k_bands]                                = {};

    void shift_and_filter_input(const float * input) {
        constexpr int new_idx = k_pitch_buf - k_frame;
        std::memmove(input_mem, input_mem + k_frame, new_idx * sizeof(float));
        // BIQUAD_HP: y = x + mem0; mem0 = mem1 + (b0 x - a0 y); mem1 = b1 x - a1 y; in f64.
        const double a0 = -1.99599f, a1 = 0.99600f, b0 = -2.0, b1 = 1.0;
        for (int i = 0; i < k_frame; ++i) {
            const double x64       = input[i];
            const double y64       = x64 + mem_hp_x[0];
            mem_hp_x[0]            = static_cast<float>(mem_hp_x[1] + (b0 * x64 - a0 * y64));
            mem_hp_x[1]            = static_cast<float>(b1 * x64 - a1 * y64);
            input_mem[new_idx + i] = static_cast<float>(y64);
        }
    }

    void transform_input(int lag, C * out, float * e) {
        const Common & c     = common();
        const float *  input = input_mem + (k_pitch_buf - (k_win + lag));
        for (int i = 0; i < k_win; ++i) {
            window_buf[i] = input[i] * c.window[i];
        }
        rfft(window_buf, out);
        for (int k = 0; k < k_freq; ++k) {
            out[k] = { out[k].r * c.wnorm, out[k].i * c.wnorm };
        }
        compute_band_corr(e, out, out);
    }

    int pitch_search() {
        const float * x_lp      = pitch_buf + k_pitch_max / 2;
        const float * y         = pitch_buf;
        constexpr int len       = k_pitch_frame;
        constexpr int max_pitch = k_pitch_max - 3 * k_pitch_min;
        float *       x_lp4     = scratch2;
        float *       y_lp4     = scratch;
        float *       xcorr     = scratch3;
        for (int j = 0; j < len / 4; ++j) {
            x_lp4[j] = x_lp[2 * j];
        }
        for (int j = 0; j < len / 4 + max_pitch / 4; ++j) {
            y_lp4[j] = y[2 * j];
        }
        pitch_xcorr(x_lp4, len / 4, y_lp4, len / 4 + max_pitch / 4, xcorr, max_pitch / 4);
        int best = 0, second = 1;
        find_best_pitch(xcorr, max_pitch / 4, y_lp4, len / 4, best, second);
        for (int i = 0; i < max_pitch / 2; ++i) {
            xcorr[i] = 0.0f;
            if (std::abs(i - 2 * best) > 2 && std::abs(i - 2 * second) > 2) {
                continue;
            }
            xcorr[i] = std::max(inner_prod(x_lp, y + i, len / 2), -1.0f);
        }
        int unused = 0;
        find_best_pitch(xcorr, max_pitch / 2, y, len / 2, best, unused);
        int offset = 0;
        if (best > 0 && best < max_pitch / 2 - 1) {
            const float a = xcorr[best - 1], b = xcorr[best], c = xcorr[best + 1];
            if (c - a > 0.7f * (b - a)) {
                offset = 1;
            } else if (a - c > 0.7f * (b - c)) {
                offset = -1;
            }
        }
        return 2 * best - offset;
    }

    int remove_doubling(int pitch_idx, float & gain_out) {
        const float * x          = pitch_buf;
        constexpr int min_period = k_pitch_min / 2;
        constexpr int max_period = k_pitch_max / 2;
        constexpr int n          = k_pitch_frame / 2;
        const int     t0         = std::min(pitch_idx / 2, max_period - 1);
        const int     prev       = last_period / 2;
        float *       yy_lookup  = scratch;
        int           t          = t0;
        const float * xm         = x + max_period;
        const float   xx         = inner_prod(xm, xm, n);
        float         xy         = inner_prod(xm, x + (max_period - t0), n);
        yy_lookup[0]             = xx;
        float yy                 = xx;
        for (int i = 1; i <= max_period; ++i) {
            yy += x[max_period - i] * x[max_period - i] - x[max_period + n - i] * x[max_period + n - i];
            yy_lookup[i] = std::max(yy, 0.0f);
        }
        yy                  = yy_lookup[t0];
        float       best_xy = xy;
        float       best_yy = yy;
        const float g0      = pitch_gain(xy, xx, yy);
        float       g       = g0;
        for (int k = 2; k <= 15; ++k) {
            const int t1 = (2 * t0 + k) / (2 * k);
            if (t1 < min_period) {
                break;
            }
            int t1b = 0;
            if (k == 2) {
                t1b = t1 + t0 > max_period ? t0 : t0 + t1;
            } else {
                t1b = (2 * k_second_check[k] * t0 + k) / (2 * k);
            }
            xy               = inner_prod(xm, x + (max_period - t1), n);
            const float xy2  = inner_prod(xm, x + (max_period - t1b), n);
            xy               = (xy + xy2) / 2.0f;
            yy               = (yy_lookup[t1] + yy_lookup[t1b]) / 2.0f;
            const float g1   = pitch_gain(xy, xx, yy);
            float       cont = 0.0f;
            if (std::abs(t1 - prev) <= 1) {
                cont = last_gain;
            } else if (std::abs(t1 - prev) <= 2 && 5 * k * k < t0) {
                cont = last_gain / 2.0f;
            }
            // nnnoiseless's order: the middle branch can never run.
            float thresh = 0.0f;
            if (t1 < 3 * min_period) {
                thresh = std::max(0.85f * g0 - cont, 0.4f);
            } else if (t1 < 2 * min_period) {
                thresh = std::max(0.9f * g0 - cont, 0.5f);
            } else {
                thresh = std::max(0.7f * g0 - cont, 0.3f);
            }
            if (g1 > thresh) {
                best_xy = xy;
                best_yy = yy;
                t       = t1;
                g       = g1;
            }
        }
        best_xy  = std::max(best_xy, 0.0f);
        float pg = best_yy <= best_xy ? 1.0f : best_xy / (best_yy + 1.0f);
        float xc[3];
        for (int k = 0; k < 3; ++k) {
            xc[k] = inner_prod(xm, x + (max_period - (t + k - 1)), n);
        }
        int offset = 0;
        if (xc[2] - xc[0] > 0.7f * (xc[1] - xc[0])) {
            offset = 1;
        } else if (xc[0] - xc[2] > 0.7f * (xc[1] - xc[2])) {
            offset = -1;
        }
        pg       = std::min(pg, g);
        gain_out = pg;
        return std::max(2 * t + offset, k_pitch_min);
    }

    int find_pitch() {
        pitch_downsample(input_mem, k_pitch_buf, pitch_buf);
        const int idx    = k_pitch_max - pitch_search();
        float     gain   = 0.0f;
        const int period = remove_doubling(idx, gain);
        last_period      = period;
        last_gain        = gain;
        return period;
    }

    // True when the frame was nearly silent (features zeroed).
    bool compute_frame_features() {
        float ly[k_bands];
        float tmp[k_bands];
        transform_input(0, x, ex);
        const int pitch_idx = find_pitch();
        transform_input(pitch_idx, p, ep);
        compute_band_corr(exp, x, p);
        for (int i = 0; i < k_bands; ++i) {
            exp[i] /= std::sqrt(0.001f + ex[i] * ep[i]);
        }
        dct(tmp, exp);
        for (int i = 0; i < k_delta_ceps; ++i) {
            features[k_bands + 2 * k_delta_ceps + i] = tmp[i];
        }
        features[k_bands + 2 * k_delta_ceps] -= 1.3f;
        features[k_bands + 2 * k_delta_ceps + 1] -= 0.9f;
        features[k_bands + 3 * k_delta_ceps] = 0.01f * (static_cast<float>(pitch_idx) - 300.0f);
        float log_max                        = -2.0f;
        float follow                         = -2.0f;
        float e                              = 0.0f;
        for (int i = 0; i < k_bands; ++i) {
            ly[i]   = std::max(std::max(std::log10(1e-2f + ex[i]), log_max - 7.0f), follow - 1.5f);
            log_max = std::max(log_max, ly[i]);
            follow  = std::max(follow - 1.5f, ly[i]);
            e += ex[i];
        }
        if (e < 0.04f) {
            std::fill(features, features + k_features, 0.0f);
            return true;
        }
        dct(features, ly);
        features[0] -= 12.0f;
        features[1] -= 4.0f;
        const int c0 = mem_id;
        const int c1 = mem_id < 1 ? k_ceps_mem + mem_id - 1 : mem_id - 1;
        const int c2 = mem_id < 2 ? k_ceps_mem + mem_id - 2 : mem_id - 2;
        std::copy_n(features, k_bands, cepstral_mem[c0]);
        ++mem_id;
        for (int i = 0; i < k_delta_ceps; ++i) {
            features[i]           = cepstral_mem[c0][i] + cepstral_mem[c1][i] + cepstral_mem[c2][i];
            features[k_bands + i] = cepstral_mem[c0][i] - cepstral_mem[c2][i];
            features[k_bands + k_delta_ceps + i] =
                cepstral_mem[c0][i] - 2.0f * cepstral_mem[c1][i] + cepstral_mem[c2][i];
        }
        float spec_variability = 0.0f;
        if (mem_id == k_ceps_mem) {
            mem_id = 0;
        }
        for (int i = 0; i < k_ceps_mem; ++i) {
            float min_dist = 1e15f;
            for (int j = 0; j < k_ceps_mem; ++j) {
                float dist = 0.0f;
                for (int k = 0; k < k_bands; ++k) {
                    const float d = cepstral_mem[i][k] - cepstral_mem[j][k];
                    dist += d * d;
                }
                if (j != i) {
                    min_dist = std::min(min_dist, dist);
                }
            }
            spec_variability += min_dist;
        }
        features[k_bands + 3 * k_delta_ceps + 1] = spec_variability / static_cast<float>(k_ceps_mem) - 2.1f;
        return false;
    }

    void pitch_filter(const float * gain) {
        float r[k_bands];
        float rf[k_freq];
        for (int i = 0; i < k_bands; ++i) {
            if (exp[i] > gain[i]) {
                r[i] = 1.0f;
            } else {
                const float exp_sq = exp[i] * exp[i];
                const float g_sq   = gain[i] * gain[i];
                r[i]               = exp_sq * (1.0f - g_sq) / (0.001f + g_sq * (1.0f - exp_sq));
            }
            r[i] = std::sqrt(std::clamp(r[i], 0.0f, 1.0f));
            r[i] *= std::sqrt(ex[i] / (1e-8f + ep[i]));
        }
        interp_band_gain(rf, r);
        for (int k = 0; k < k_freq; ++k) {
            x[k] = { x[k].r + p[k].r * rf[k], x[k].i + p[k].i * rf[k] };
        }
        float new_e[k_bands];
        compute_band_corr(new_e, x, x);
        for (int i = 0; i < k_bands; ++i) {
            r[i] = std::sqrt(ex[i] / (1e-8f + new_e[i]));
        }
        interp_band_gain(rf, r);
        for (int k = 0; k < k_freq; ++k) {
            x[k] = { x[k].r * rf[k], x[k].i * rf[k] };
        }
    }

    void frame_synthesis(float * out) {
        const Common & c = common();
        irfft(x, window_buf);
        for (int i = 0; i < k_win; ++i) {
            window_buf[i] = window_buf[i] / 2.0f * c.window[i];
        }
        for (int i = 0; i < k_frame; ++i) {
            out[i]           = window_buf[i] + synthesis_mem[i];
            synthesis_mem[i] = window_buf[k_frame + i];
        }
    }

    float rnn_compute(float * gains) {
        const Model & m                              = model();
        float         buf[k_max_neurons * 3]         = {};
        float         denoise_buf[k_max_neurons * 3] = {};
        float         vad[1]                         = { 0.0f };
        const int     nd                             = m.input_dense.nb_neurons;
        const int     nv                             = m.vad_gru.nb_neurons;
        const int     nn                             = m.noise_gru.nb_neurons;
        dense_compute(m.input_dense, buf, features);
        gru_compute(m.vad_gru, vad_state, buf);
        dense_compute(m.vad_output, vad, vad_state);
        std::copy_n(vad_state, nv, buf + nd);
        std::copy_n(features, k_features, buf + nd + nv);
        gru_compute(m.noise_gru, noise_state, buf);
        std::copy_n(vad_state, nv, denoise_buf);
        std::copy_n(noise_state, nn, denoise_buf + nv);
        std::copy_n(features, k_features, denoise_buf + nv + nn);
        gru_compute(m.denoise_gru, denoise_state, denoise_buf);
        dense_compute(m.denoise_output, gains, denoise_state);
        return vad[0];
    }

    float process_frame(const float * input, float * output) {
        float g[k_bands] = {};
        float gf[k_freq];
        float vad_prob = 0.0f;
        shift_and_filter_input(input);
        const bool silence = compute_frame_features();
        if (!silence) {
            vad_prob = rnn_compute(g);
            pitch_filter(g);
            for (int i = 0; i < k_bands; ++i) {
                g[i]     = std::max(g[i], 0.6f * lastg[i]);
                lastg[i] = g[i];
            }
            interp_band_gain(gf, g);
            for (int k = 0; k < k_freq; ++k) {
                x[k] = { x[k].r * gf[k], x[k].i * gf[k] };
            }
        }
        frame_synthesis(output);
        return vad_prob;
    }
};

Denoiser::Denoiser() : state_(std::make_unique<State>()) {
    (void) model();
    (void) common();
    (void) fft();
}

Denoiser::~Denoiser()                                = default;
Denoiser::Denoiser(Denoiser &&) noexcept             = default;
Denoiser & Denoiser::operator=(Denoiser &&) noexcept = default;

void Denoiser::reset() noexcept {
    if (state_) {
        *state_ = State{};
    }
}

float Denoiser::process_frame(const float * input, float * output) noexcept {
    if (!state_ || input == nullptr || output == nullptr) {
        return 0.0f;
    }
    float in_copy[k_frame];
    std::memcpy(in_copy, input, sizeof(in_copy));  // input and output may alias
    return state_->process_frame(in_copy, output);
}

namespace detail {

bool set_wide_kernels(bool enable) noexcept {
#if defined(TRANSCRIBE_RNNOISE_AVX)
    const bool on = enable && earshot::isa_supported(earshot::Isa::Avx2);
    wide_kernels().store(on, std::memory_order_relaxed);
    return on;
#else
    (void) enable;
    return false;
#endif
}

void rfft960(const float * in, std::array<std::complex<float>, k_freq_size> & out) noexcept {
    C x[k_freq];
    rfft(in, x);
    for (int k = 0; k < k_freq; ++k) {
        out[k] = { x[k].r, x[k].i };
    }
}

void irfft960(const std::array<std::complex<float>, k_freq_size> & in, float * out) noexcept {
    C x[k_freq];
    for (int k = 0; k < k_freq; ++k) {
        x[k] = { in[k].real(), in[k].imag() };
    }
    irfft(x, out);
}

}  // namespace detail

}  // namespace transcribe::rnnoise
