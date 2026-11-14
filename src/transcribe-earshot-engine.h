// transcribe-earshot-engine.h - Earshot 1.2.2 VAD scoring engine.
//
// The network (pykeio/earshot, MIT OR Apache-2.0) scores 16 ms frames of
// 16 kHz mono audio: pre-emphasis + Hann window over the last three frames,
// a 1024-point power spectrum, 40 log-mel bands, a three-layer separable conv
// stack and two int16 minGRU layers. Weights are bundled (earshot-weights.inl),
// so nothing is loaded from disk.
//
// Kernels come in three instruction sets picked at run time: a baseline
// (SSE2 on x86-64, NEON on ARM64, scalar elsewhere), AVX2, and AVX-512
// (F + BW + VNNI). Every set applies the same single-precision operations in
// the same order and uses Cephes log/exp instead of the C library, so scores
// are bit-identical across sets, compilers, thread counts and streaming vs
// batch. They match the Rust crate within ~1e-4 per frame (tested against its
// recorded scores). Ported from audio.cpp #1095 / #1102.
#pragma once

#include <cstdint>
#include <memory>

namespace transcribe::earshot {

inline constexpr int32_t k_sample_rate   = 16000;
inline constexpr int64_t k_frame_samples = 256;  // 16 ms

// Kernel set. Auto picks the widest one this build and CPU support; the
// TRANSCRIBE_EARSHOT_ISA environment variable (baseline | avx2 | avx512) caps
// Auto for benchmarks and tests.
enum class Isa : uint8_t { Auto, Baseline, Avx2, Avx512 };

// Whether this build and CPU can run `isa` (Auto and Baseline always can).
bool         isa_supported(Isa isa) noexcept;
// Auto -> the widest supported set (after the environment cap); an explicit
// set this build or CPU cannot run falls back to Baseline.
Isa          resolve_isa(Isa requested) noexcept;
// "auto", "baseline", "avx2" or "avx512".
const char * isa_name(Isa isa) noexcept;

// Streaming detector: one 256-sample frame per call, recurrent state carried
// between calls. Mirrors upstream earshot::Detector. Samples are expected in
// [-1, 1]; callers clamp.
class Detector {
  public:
    explicit Detector(Isa isa = Isa::Auto);
    ~Detector();
    Detector(Detector &&) noexcept;
    Detector & operator=(Detector &&) noexcept;
    Detector(const Detector &)             = delete;
    Detector & operator=(const Detector &) = delete;

    void  reset() noexcept;
    float predict(const float * frame) noexcept;  // k_frame_samples samples

    Isa isa() const noexcept { return isa_; }

  private:
    struct State;
    std::unique_ptr<State> state_;
    Isa                    isa_;
};

// Score every complete frame of `n_samples` samples into out[0 .. n_samples /
// k_frame_samples). Equivalent to feeding the frames to a fresh Detector, with
// the front end, the conv stack and the matrix products spread over
// `n_threads` (<= 1: the calling thread only). Throws only std::bad_alloc /
// std::system_error.
void score_frames(const float * samples, int64_t n_samples, int n_threads, float * out, Isa isa = Isa::Auto);

}  // namespace transcribe::earshot
