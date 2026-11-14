// transcribe-rnnoise.h - classic RNNoise noise suppression (xiph/rnnoise,
// BSD-3-Clause), ported from nnnoiseless 0.5.2 - the crate ZER0 runs - with
// its bundled model, so nothing is loaded from disk.
//
// One 10 ms frame of 48 kHz mono audio per call, in the int16 sample range
// ([-32768, 32767] as floats); the output is one frame behind the input
// (overlap-add synthesis). Same algorithm and model as nnnoiseless: a
// high-pass biquad, 22 band energies and pitch correlations from a 960-point
// FFT, a pitch search on 2x / 4x downsampled audio, 42 features into a dense
// layer and three GRUs (24 / 48 / 96 units, int8 weights), band gains with a
// pitch filter, and windowed overlap-add. Speed comes from float-converted
// weights in loops that keep each output's summation order (vectorizable
// without changing results) and a 960-point real FFT computed as a 480-point
// mixed-radix complex FFT.
#pragma once

#include <array>
#include <complex>
#include <memory>

namespace transcribe::rnnoise {

inline constexpr int k_frame_size = 480;  // 10 ms at 48 kHz
inline constexpr int k_window     = 960;
inline constexpr int k_freq_size  = 481;

class Denoiser {
  public:
    Denoiser();
    ~Denoiser();
    Denoiser(Denoiser &&) noexcept;
    Denoiser & operator=(Denoiser &&) noexcept;
    Denoiser(const Denoiser &)             = delete;
    Denoiser & operator=(const Denoiser &) = delete;

    void  reset() noexcept;
    // input / output: k_frame_size samples each (may alias). Returns RNNoise's
    // speech probability for the frame (0 for a silent frame).
    float process_frame(const float * input, float * output) noexcept;

  private:
    struct State;
    std::unique_ptr<State> state_;
};

namespace detail {
// Exposed for the unit test: unnormalized 960-point real DFT and its inverse
// (x[n] = sum_k X[k] e^{+2 pi i k n / 960} over the Hermitian extension).
void rfft960(const float * in, std::array<std::complex<float>, k_freq_size> & out) noexcept;
void irfft960(const std::array<std::complex<float>, k_freq_size> & in, float * out) noexcept;
// Switches the network and pitch kernels between 8-wide (x86-64 with AVX2;
// the default there) and 4-wide, process-wide; returns whether 8-wide is now
// on. Both give bit-identical output - the unit test checks that.
bool set_wide_kernels(bool enable) noexcept;
}  // namespace detail

}  // namespace transcribe::rnnoise
