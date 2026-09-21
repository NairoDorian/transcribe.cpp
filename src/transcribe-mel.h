// transcribe-mel.h - native C++ log-mel feature extractor.
//
// PRIVATE header (src/, not exported). Pure CPU, no ggml dependency:
// audio in, std::vector<float> mel out; the family encoder builder
// copies the result into a ggml_tensor at the backend boundary.
//
// Matches NeMo's AudioToMelSpectrogramPreprocessor / FilterbankFeatures
// as exported in nemo128.onnx. The per-NeMo numerical constants
// (log_eps = 2^-24, unbiased variance, reflect padding, fp64 STFT,
// mag_power = 2.0) are hardcoded in the .cpp rather than in the GGUF
// schema; they become per-family knobs only when a second family needs
// different values.
//
// Construction is one-shot: the constructor precomputes the hann window
// (zero-padded from win_length to n_fft) and the Slaney-normalized mel
// filterbank. compute() is then a function of (config, audio) only.

#pragma once

#include "transcribe.h"

#include <cstddef>
#include <string>
#include <vector>

namespace transcribe {

// Frontend configuration. Mirrors the stt.frontend.* KV the loader
// reads out of ParakeetHParams::fe_*. Family-agnostic: SenseVoice
// will fill the same struct with different values when it lands.
//
// The defaults are the real Parakeet 0.6B values (matching
// nemo128.onnx). Constructing a default-initialized MelConfig and
// calling MelFrontend(cfg) gives a working extractor for v2 / v3.
struct MelConfig {
    int   sample_rate  = 16000;
    int   num_mels     = 128;
    int   n_fft        = 512;
    int   win_length   = 400;
    int   hop_length   = 160;
    float pre_emphasis = 0.97f;  // 0.0 disables
    float f_min        = 0.0f;
    float f_max        = 8000.0f;

    // STFT input padding mode:
    //   "reflect"  — symmetric reflect-pad by n_fft/2 on both sides
    //                (NeMo default). Frame count: floor(n / hop) + 1.
    //   "constant" — zero-pad by n_fft/2 on both sides.
    //   "none"     — PyTorch center=False: no input padding, window is
    //                left-aligned in the n_fft buffer instead of centered,
    //                frame count = (n_samples - win_length) / hop + 1.
    //                Used by LASR / MedASR.
    std::string pad_mode = "reflect";

    // STFT window shape:
    //   "hann_symmetric" — torch.hann_window(N, periodic=False):
    //                      cos(2*pi*k / (N-1)). Default; used by NeMo
    //                      and Cohere.
    //   "hann_periodic"  — torch.hann_window(N, periodic=True):
    //                      cos(2*pi*k / N). Used by Whisper (and
    //                      Qwen3-ASR's Whisper frontend).
    std::string window_type = "hann_symmetric";

    // Normalization mode:
    //   "per_feature"   — NeMo: per-mel-bin zero-mean / unit-variance
    //                     (unbiased); default.
    //   "per_utterance" — Whisper: log10 base, global clamp to
    //                     max - 8.0, then (x + 4) / 4. Also drops the
    //                     trailing center-pad STFT frame so the output
    //                     has exactly `n_samples / hop_length` frames.
    //   "none"          — emit raw log-mel as-is (NeMo's "NA"/no-op
    //                     normalize; used by streaming-trained variants
    //                     whose feature normalisation is baked into
    //                     training rather than applied at inference).
    //   "global"        — Voxtral Realtime streaming log-mel: like
    //                      "per_utterance" but the log clamp floor uses a
    //                      FIXED max (global_log_mel_max) instead of the
    //                      per-utterance maximum, so each frame is
    //                      causal/streaming-safe. Still drops the trailing
    //                      center-pad STFT frame.
    std::string normalize = "per_feature";

    // Fixed log-mel maximum for normalize == "global" (Voxtral Realtime
    // global_log_mel_max). Unused by other normalize modes.
    float global_log_mel_max = 1.5f;

    // Optional checkpoint-provided mel filterbank [num_mels * (n_fft/2+1)]
    // row-major, Slaney-normalised. When non-empty, used instead of
    // computing from scratch. Set by the loader when the GGUF contains
    // frontend.mel_filterbank.
    std::vector<float> filterbank;

    // When > 0 AND normalize="none", emit log(max(power, log_clamp_min))
    // instead of NeMo's log(power + kLogEps). LASR / MedASR use this with
    // log_clamp_min = 1e-5; NeMo's frontends leave it at 0.0.
    float log_clamp_min = 0.0f;

    // Optional checkpoint-provided window [win_length]. When non-empty,
    // used instead of computing a periodic Hann window.
    std::vector<float> window;

    // NeMo seq-len semantics for the centered (reflect/constant) paths.
    // Newer NeMo AudioToMelSpectrogramPreprocessor computes the feature
    // length as ceil(n_samples / hop) rather than floor(n_samples / hop) + 1.
    // The two agree except when n_samples is an exact multiple of hop, where
    // ceil drops the extra trailing center frame. Sortformer's preprocessor
    // uses the ceil form; leave false for the legacy +1 families.
    bool nemo_seq_len_ceil = false;
};

// Streaming state for MelFrontend::compute_incremental(). Opaque to
// callers except for clear(): it holds the raw (pre-normalization)
// log-mel of every frame the stream has produced, in a fixed-stride
// [n_mels, capacity] layout, plus the bookkeeping that decides which
// frames are final. One state belongs to one stream; clear() resets it,
// and a config change is detected and restarts it.
struct MelStreamState {
    std::vector<float> raw;             // [n_mels, capacity], log10 max(power, 1e-10)
    int                capacity   = 0;  // column stride of `raw`
    int                n_mels     = 0;  // latched config identity
    int                n_fft      = 0;
    int                hop        = 0;
    int                win        = 0;
    int                stable     = 0;    // frames [0, stable) are final and counted
    int                emitted    = 0;    // frames emitted by the previous call
    double             stable_max = 0.0;  // max over the raw frames [0, stable)

    void clear() {
        std::vector<float>().swap(raw);
        capacity = n_mels = n_fft = hop = win = stable = emitted = 0;
        stable_max                                               = 0.0;
    }
};

// Pure C++ log-mel extractor. Construct once, call compute() any
// number of times. Thread-safety: const after construction; multiple
// threads may call compute() concurrently.
class MelFrontend {
  public:
    explicit MelFrontend(const MelConfig & cfg);

    // Run the full pipeline. pcm must be 16 kHz mono float32 in
    // [-1, 1]; n_samples is the number of input samples. The output
    // mel buffer is resized to num_mels * n_frames in row-major
    // [num_mels, n_frames] layout.
    //
    // n_threads controls STFT parallelism. 0 (default) auto-detects
    // via std::thread::hardware_concurrency() capped at 8. The STFT
    // loop is the dominant cost of compute(); the filterbank matmul
    // is handed to cblas_sgemm and is already multi-threaded on
    // Accelerate / OpenBLAS.
    //
    // out_frames overrides how many leading frames the per_utterance /
    // global normalize modes emit. 0 (default) keeps each mode's own
    // rule, which drops the trailing center-pad frame. granite5_ctc
    // needs 2*ceil(floor(n/hop)/2) frames, which equals n_frames-1 when
    // floor(n/hop) is even and n_frames when it is odd, so the count is
    // a per-utterance property and cannot live in MelConfig. The
    // per-utterance max is taken over exactly the emitted frames, which
    // is what the reference's `mel[..., :num_frames].amax()` does.
    // Ignored by per_feature / none.
    //
    // Returns:
    //   TRANSCRIBE_OK              normal success.
    //   TRANSCRIBE_ERR_INVALID_ARG pcm is null, or n_samples is too
    //                              short to produce >= 2 frames
    //                              (per-feature normalize divides by
    //                              n_frames - 1, which would NaN).
    transcribe_status compute(const float *        pcm,
                              size_t               n_samples,
                              std::vector<float> & out_mel,
                              int &                out_n_mels,
                              int &                out_n_frames,
                              int                  n_threads  = 0,
                              int                  out_frames = 0) const;

    // Number of mel bins (matches MelConfig::num_mels).
    int num_mels() const { return cfg_.num_mels; }

    // Frame count for a given audio length, before calling compute().
    // Matches NeMo: floor(n_samples / hop_length) + 1.
    int n_frames_for(size_t n_samples) const;

    // How many of those frames are *final*: their STFT window lies entirely
    // inside the audio, so extending the buffer cannot change them.
    //
    // compute() emits frames with a centered window: frame t is built from
    // samples [t*hop - n_fft/2, t*hop + n_fft/2]. Whatever the buffer does not
    // contain yet comes from the pad, and with pad_mode="reflect" the pad is
    // the signal reflected at the *current* end — so the last frame or two of
    // every buffer are provisional and move as soon as more samples arrive.
    // (pad_mode="constant" pads with zeros, which never move, and "none"
    // left-aligns the window so it never reads ahead; both make every frame
    // final.) Streaming consumers that carry features across calls — the
    // encoder prefix cache in arch/qwen3_asr, which reuses encoder rows across
    // R2T2 ticks — must cut at this count, not at n_frames_for().
    int final_frame_count(size_t n_samples) const;

    // ------------------------------------------------------------------
    // Incremental (streaming) extraction
    // ------------------------------------------------------------------
    // compute() is a pure function of (config, audio), so a streaming
    // caller that appends audio and re-calls it re-derives the whole
    // utterance every time. The frame loop is the dominant cost of that
    // (STFT + filterbank per frame), and in an R2T2 tick it is the
    // difference between a few tens of microseconds and tens of
    // milliseconds once the utterance is past ~30 s: measured on this
    // machine, 0.6 ms at 1 s of audio but 26 ms at 35 s, growing without
    // bound. compute_incremental() keeps the frames it already computed
    // and pays only for what changed.
    //
    // It is not an approximation. Every frame is an independent function
    // of a bounded input span (frame t reads padded[t*hop, t*hop+n_fft)),
    // so a frame that is *final* in the sense of final_frame_count() has
    // a value no longer buffer can change, and is computed exactly once.
    // The frames that are not final -- at most two, the ones whose
    // centered window still runs off the end -- are recomputed on every
    // call, exactly as compute() would. The only cross-frame coupling is
    // the clamp level: per_utterance takes it from the maximum over the
    // emitted frames, which is tracked as a running maximum (a rise
    // re-derives the level of every frame, again exactly as compute()
    // would), and "global" takes it from a constant. The emitted frames
    // and the arithmetic are the same expressions as compute()'s.
    // tests/mel_unit.cpp holds the equality gate: over a growing buffer,
    // every call must be bit-identical to compute() over the same prefix.
    //
    // Supported configs are the ones whose normalization is per-frame plus a
    // scalar level: normalize "per_utterance", "global" or "none", any
    // padding mode except "none", and a non-pow2 n_fft (the fused per-frame
    // path). supports_incremental() reports it; when false the caller must
    // fall back to compute(), whose contract is unchanged, and
    // compute_incremental() returns TRANSCRIBE_ERR_NOT_IMPLEMENTED rather
    // than quietly doing the slow thing.
    bool supports_incremental() const;

    // Append `n_samples` of pcm (the whole buffer so far, not the delta)
    // and emit the normalized mel of its emitted frames. The result is
    // identical to compute(pcm, n_samples, ..., out_frames = 0).
    //
    // `state` is the caller's, so a stream resets by clearing it and no
    // two streams share an allocation.
    transcribe_status compute_incremental(MelStreamState &     state,
                                          const float *        pcm,
                                          size_t               n_samples,
                                          std::vector<float> & out_mel,
                                          int &                out_n_mels,
                                          int &                out_n_frames,
                                          int                  n_threads = 0) const;

    // Read-only accessors for unit tests. Not part of the runtime
    // API; the goal is to validate the precomputed buffers in
    // isolation without running the full pipeline.
    const std::vector<double> & window() const { return window_; }

    const std::vector<float> & filterbank() const { return mel_fb_; }

    const MelConfig & config() const { return cfg_; }

    int n_freq() const { return n_freq_; }

  private:
    MelConfig           cfg_;
    int                 n_freq_;  // n_fft/2 + 1
    std::vector<double> window_;  // [n_fft], periodic hann zero-padded
    std::vector<float>  mel_fb_;  // [num_mels * n_freq] row-major, Slaney

    // Per-band nonzero support of mel_fb_: band m occupies bins
    // [fb_begin_[m], fb_end_[m]). A Slaney/whisper filterbank is
    // triangular, so each band touches only ~2*n_freq/num_mels bins out
    // of n_freq (e.g. 5 of 201 for whisper's 80x201). The scalar matmul
    // paths iterate only this span; the skipped terms are exactly
    // 0.0f * power, and adding 0.0 to the fp64 accumulator is exact, so
    // the result is bit-identical to the dense loop. An empty band gets
    // begin == end (sum stays 0.0, same as the dense loop). Sized
    // [num_mels]; built once in the constructor.
    std::vector<int> fb_begin_;
    std::vector<int> fb_end_;

    // sin/cos LUT for the mixed-radix FFT. Sized to n_fft so that every
    // recursion-level N (n_fft, n_fft/2, n_fft/4, ..., odd leaf) divides
    // the LUT exactly and lookups never fall back to live std::cos/sin.
    // Stored as fp32 to match the FFT precision and avoid a per-load
    // cast in the inner butterfly. Only populated when n_fft is non-pow2
    // (the only path that uses the LUT — pow2 sizes use fft_radix2 /
    // vDSP, which carry their own twiddle factors). Empty when unused.
    std::vector<float> cos_lut_;
    std::vector<float> sin_lut_;
};

}  // namespace transcribe
