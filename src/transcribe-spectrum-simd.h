// transcribe-spectrum-simd.h - SIMD complex de-interleaving and spectrum utilities.
//
// Algorithmic SIMD patterns for audio STFT feature extraction:
//   - Complex De-interleaving & Power Spectrum (re^2 + im^2):
//       * x86_64 AVX2 + FMA: _mm256_shuffle_ps + _mm256_permute4x64_pd + _mm256_fmadd_ps
//       * ARM64 NEON: vld2q_f32 native 2-way de-interleaving + vmlaq_f32
//       * Fallback: clean scalar C++ loop
//   - Fast Reciprocal Square Root (Magnitude Spectrum = sqrt(re^2 + im^2)):
//       * x86_64 AVX2: _mm256_rsqrt_ps + 1-step Newton-Raphson refinement
//       * ARM64 NEON: vrsqrteq_f32 + vrsqrtsq_f32 step
//       * Fallback: std::sqrt
//   - Double to Float Power Spectrum:
//       * x86_64 AVX2: _mm256_shuffle_pd + _mm256_permute4x64_pd + _mm256_cvtpd_ps

#pragma once

#include <cmath>
#include <cstddef>

#if defined(__AVX2__)
#    include <immintrin.h>
#endif

#if defined(__ARM_NEON)
#    include <arm_neon.h>
#endif

namespace transcribe {

// Compute power spectrum (re^2 + im^2) * scale from interleaved complex floats.
// raw_complex: interleaved [re0, im0, re1, im1, ...] of length 2 * n_complex.
// power_out: output array of length n_complex.
inline void compute_power_spectrum_f32(const float * raw_complex,
                                       float *       power_out,
                                       size_t        n_complex,
                                       float         scale = 1.0f) noexcept {
    size_t i = 0;
#if defined(__AVX2__)
    auto reorderLanes = [](__m256 v) noexcept -> __m256 {
        return _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(v), _MM_SHUFFLE(3, 1, 2, 0)));
    };
    const __m256 vscale    = _mm256_set1_ps(scale);
    const bool   has_scale = (scale != 1.0f);

    size_t n_vec16 = (n_complex / 16) * 16;
    for (; i < n_vec16; i += 16) {
        __m256 cA0 = _mm256_loadu_ps(raw_complex + 2 * i);
        __m256 cB0 = _mm256_loadu_ps(raw_complex + 2 * i + 8);
        __m256 re0 = _mm256_shuffle_ps(cA0, cB0, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im0 = _mm256_shuffle_ps(cA0, cB0, _MM_SHUFFLE(3, 1, 3, 1));
        __m256 p0  = _mm256_fmadd_ps(re0, re0, _mm256_mul_ps(im0, im0));
        p0         = reorderLanes(p0);
        if (has_scale) {
            p0 = _mm256_mul_ps(p0, vscale);
        }
        _mm256_storeu_ps(power_out + i, p0);

        __m256 cA1 = _mm256_loadu_ps(raw_complex + 2 * i + 16);
        __m256 cB1 = _mm256_loadu_ps(raw_complex + 2 * i + 24);
        __m256 re1 = _mm256_shuffle_ps(cA1, cB1, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im1 = _mm256_shuffle_ps(cA1, cB1, _MM_SHUFFLE(3, 1, 3, 1));
        __m256 p1  = _mm256_fmadd_ps(re1, re1, _mm256_mul_ps(im1, im1));
        p1         = reorderLanes(p1);
        if (has_scale) {
            p1 = _mm256_mul_ps(p1, vscale);
        }
        _mm256_storeu_ps(power_out + i + 8, p1);
    }
    for (; i + 7 < n_complex; i += 8) {
        __m256 cA = _mm256_loadu_ps(raw_complex + 2 * i);
        __m256 cB = _mm256_loadu_ps(raw_complex + 2 * i + 8);
        __m256 re = _mm256_shuffle_ps(cA, cB, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(cA, cB, _MM_SHUFFLE(3, 1, 3, 1));
        __m256 p  = _mm256_fmadd_ps(re, re, _mm256_mul_ps(im, im));
        p         = reorderLanes(p);
        if (has_scale) {
            p = _mm256_mul_ps(p, vscale);
        }
        _mm256_storeu_ps(power_out + i, p);
    }
#elif defined(__ARM_NEON)
    const float32x4_t vscale    = vdupq_n_f32(scale);
    const bool        has_scale = (scale != 1.0f);
    for (; i + 3 < n_complex; i += 4) {
        float32x4x2_t c   = vld2q_f32(raw_complex + 2 * i);
        float32x4_t   pwr = vmlaq_f32(vmulq_f32(c.val[0], c.val[0]), c.val[1], c.val[1]);
        if (has_scale) {
            pwr = vmulq_f32(pwr, vscale);
        }
        vst1q_f32(power_out + i, pwr);
    }
#endif
    for (; i < n_complex; ++i) {
        const float r      = raw_complex[2 * i];
        const float im_val = raw_complex[2 * i + 1];
        power_out[i]       = (r * r + im_val * im_val) * scale;
    }
}

// Compute magnitude spectrum sqrt(re^2 + im^2) * scale from interleaved complex floats.
// Uses fast reciprocal square root with 1-step Newton-Raphson approximation.
inline void compute_magnitude_spectrum_f32(const float * raw_complex,
                                           float *       mag_out,
                                           size_t        n_complex,
                                           float         scale = 1.0f) noexcept {
    size_t i = 0;
#if defined(__AVX2__)
    auto reorderLanes = [](__m256 v) noexcept -> __m256 {
        return _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(v), _MM_SHUFFLE(3, 1, 2, 0)));
    };
    auto fast_sqrt_ps = [](__m256 x) noexcept -> __m256 {
        __m256 xc    = _mm256_max_ps(x, _mm256_set1_ps(1e-30f));
        __m256 r     = _mm256_rsqrt_ps(xc);
        __m256 half  = _mm256_set1_ps(0.5f);
        __m256 three = _mm256_set1_ps(3.0f);
        r            = _mm256_mul_ps(_mm256_mul_ps(half, r), _mm256_fnmadd_ps(_mm256_mul_ps(xc, r), r, three));
        return _mm256_mul_ps(xc, r);
    };
    const __m256 vscale    = _mm256_set1_ps(scale);
    const bool   has_scale = (scale != 1.0f);

    size_t n_vec16 = (n_complex / 16) * 16;
    for (; i < n_vec16; i += 16) {
        __m256 cA0 = _mm256_loadu_ps(raw_complex + 2 * i);
        __m256 cB0 = _mm256_loadu_ps(raw_complex + 2 * i + 8);
        __m256 re0 = _mm256_shuffle_ps(cA0, cB0, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im0 = _mm256_shuffle_ps(cA0, cB0, _MM_SHUFFLE(3, 1, 3, 1));
        __m256 m0  = fast_sqrt_ps(_mm256_fmadd_ps(re0, re0, _mm256_mul_ps(im0, im0)));
        m0         = reorderLanes(m0);
        if (has_scale) {
            m0 = _mm256_mul_ps(m0, vscale);
        }
        _mm256_storeu_ps(mag_out + i, m0);

        __m256 cA1 = _mm256_loadu_ps(raw_complex + 2 * i + 16);
        __m256 cB1 = _mm256_loadu_ps(raw_complex + 2 * i + 24);
        __m256 re1 = _mm256_shuffle_ps(cA1, cB1, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im1 = _mm256_shuffle_ps(cA1, cB1, _MM_SHUFFLE(3, 1, 3, 1));
        __m256 m1  = fast_sqrt_ps(_mm256_fmadd_ps(re1, re1, _mm256_mul_ps(im1, im1)));
        m1         = reorderLanes(m1);
        if (has_scale) {
            m1 = _mm256_mul_ps(m1, vscale);
        }
        _mm256_storeu_ps(mag_out + i + 8, m1);
    }
    for (; i + 7 < n_complex; i += 8) {
        __m256 cA = _mm256_loadu_ps(raw_complex + 2 * i);
        __m256 cB = _mm256_loadu_ps(raw_complex + 2 * i + 8);
        __m256 re = _mm256_shuffle_ps(cA, cB, _MM_SHUFFLE(2, 0, 2, 0));
        __m256 im = _mm256_shuffle_ps(cA, cB, _MM_SHUFFLE(3, 1, 3, 1));
        __m256 m  = fast_sqrt_ps(_mm256_fmadd_ps(re, re, _mm256_mul_ps(im, im)));
        m         = reorderLanes(m);
        if (has_scale) {
            m = _mm256_mul_ps(m, vscale);
        }
        _mm256_storeu_ps(mag_out + i, m);
    }
#elif defined(__ARM_NEON)
    const float32x4_t vscale    = vdupq_n_f32(scale);
    const bool        has_scale = (scale != 1.0f);
    for (; i + 3 < n_complex; i += 4) {
        float32x4x2_t c   = vld2q_f32(raw_complex + 2 * i);
        float32x4_t   pwr = vmlaq_f32(vmulq_f32(c.val[0], c.val[0]), c.val[1], c.val[1]);
        pwr               = vmaxq_f32(pwr, vdupq_n_f32(1e-30f));
        // vrsqrteq_f32 + vrsqrtsq_f32 for reciprocal square root with Newton-Raphson
        float32x4_t r     = vrsqrteq_f32(pwr);
        r                 = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(pwr, r), r));
        float32x4_t mag   = vmulq_f32(pwr, r);
        if (has_scale) {
            mag = vmulq_f32(mag, vscale);
        }
        vst1q_f32(mag_out + i, mag);
    }
#endif
    for (; i < n_complex; ++i) {
        const float r      = raw_complex[2 * i];
        const float im_val = raw_complex[2 * i + 1];
        mag_out[i]         = std::sqrt(r * r + im_val * im_val) * scale;
    }
}

// Compute power spectrum (re^2 + im^2) * scale from interleaved complex doubles to float output.
// raw_complex: interleaved [re0, im0, re1, im1, ...] of length 2 * n_complex doubles.
// power_out: output array of length n_complex floats.
inline void compute_power_spectrum_f64_to_f32(const double * raw_complex,
                                              float *        power_out,
                                              size_t         n_complex,
                                              float          scale = 1.0f) noexcept {
    size_t i = 0;
#if defined(__AVX2__)
    const __m256d vscale_d  = _mm256_set1_pd(static_cast<double>(scale));
    const bool    has_scale = (scale != 1.0f);
    for (; i + 3 < n_complex; i += 4) {
        __m256d cA    = _mm256_loadu_pd(raw_complex + 2 * i);
        __m256d cB    = _mm256_loadu_pd(raw_complex + 2 * i + 4);
        __m256d re    = _mm256_shuffle_pd(cA, cB, 0b0000);
        __m256d im    = _mm256_shuffle_pd(cA, cB, 0b1111);
        __m256d pwr_d = _mm256_fmadd_pd(re, re, _mm256_mul_pd(im, im));
        pwr_d         = _mm256_permute4x64_pd(pwr_d, _MM_SHUFFLE(3, 1, 2, 0));
        if (has_scale) {
            pwr_d = _mm256_mul_pd(pwr_d, vscale_d);
        }
        __m128 pwr_f = _mm256_cvtpd_ps(pwr_d);
        _mm_storeu_ps(power_out + i, pwr_f);
    }
#endif
    for (; i < n_complex; ++i) {
        const double r      = raw_complex[2 * i];
        const double im_val = raw_complex[2 * i + 1];
        power_out[i]        = static_cast<float>((r * r + im_val * im_val) * static_cast<double>(scale));
    }
}

}  // namespace transcribe
