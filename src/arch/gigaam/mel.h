// arch/gigaam/mel.h - GigaAM log-mel feature extractor.
//
// Mirrors torchaudio.transforms.MelSpectrogram(...) + log(clamp(x, 1e-9))
// as configured by gigaam.preprocess.FeatureExtractor: 16 kHz mono, 64
// HTK mel bins, n_fft = win_length = 320, hop = 160, center=False, no
// pre-emphasis, periodic Hann window, power=2.

#pragma once

#include "transcribe.h"

#include <cstddef>
#include <vector>

namespace transcribe::gigaam {

struct GigaamHParams;
struct GigaamWeights;

// Loads the HTK mel filterbank + periodic Hann window baked into the GGUF at
// load time (hp drives the scalar shapes). We use the baked buffers rather than
// recomputing from htk_hz_to_mel / Hann: torchaudio uses Float32 throughout, so
// an fp64 C++ recomputation drifts at high-freq bin boundaries.
struct GigaamMelFrontend {
    int                n_freq;  // n_fft/2 + 1
    int                n_mels;
    int                n_fft;
    int                win_length;
    int                hop_length;
    std::vector<float> hann;    // [win_length] periodic
    std::vector<float> mel_fb;  // [n_mels, n_freq] row-major HTK

    // Nonzero support of each mel_fb row: band m covers bins
    // [fb_begin[m], fb_end[m]). An HTK triangular filterbank touches only
    // ~2*n_freq/n_mels bins per band (5 of 161 here), so the no-BLAS matmul
    // walks just this span. Skipped terms are exactly 0.0f * power added to a
    // sequential f32 accumulator, which leaves it unchanged — bit-identical
    // to the dense loop. Filled by init(); [n_mels] each.
    std::vector<int> fb_begin;
    std::vector<int> fb_end;

    void init(const GigaamHParams & hp, const GigaamWeights & w);

    // n_frames for a given audio length under center=False.
    // = floor((n_samples - win_length) / hop) + 1, or 0 if too short.
    int n_frames_for(size_t n_samples) const;

    // Compute log-mel. Output is row-major [n_mels, n_frames].
    // n_threads controls frame FFT parallelism; <= 0 uses the affinity-aware
    // project default.
    transcribe_status compute(const float *        pcm,
                              size_t               n_samples,
                              std::vector<float> & out_mel,
                              int &                out_n_frames,
                              int                  n_threads = 0) const;
};

}  // namespace transcribe::gigaam
