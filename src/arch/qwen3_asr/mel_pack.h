// mel_pack.h — host-side mel framing for the Qwen3-ASR encoder, and the
// bookkeeping the R2T2 encoder prefix cache needs to carry mel across ticks.
//
// PRIVATE header (src/, not exported). Pure host code: no ggml, no backend,
// no model. That is deliberate — every function here is a total function of
// its arguments, so it is tested directly by
// tests/qwen3_asr_mel_pack_unit.cpp rather than through a run.
//
// The layout these functions agree on is the one MelFrontend::compute()
// emits and the encoder graph consumes: mel is [n_mels, n_frames] row-major,
// i.e. one flat run of frames per mel bin, so the row stride IS the frame
// count. Every bug in this file's history has been a disagreement about that
// stride, which is why the frame offsets below are frames and never pointer
// arithmetic.

#pragma once

#include "encoder.h"

#include <cstdint>
#include <vector>

namespace transcribe::qwen3_asr {

// Pack [n_mels, n_mel_frames] mel into the encoder's batched chunk layout
// [mel_per_chunk, n_mels, 1, n_chunks]. The last chunk holds
// `last_chunk_real_mel` real frames and is zero-padded out to mel_per_chunk.
//
// `frame0` is the mel frame the packing starts at. Offline it is 0; the R2T2
// streaming path passes the first frame its tail covers, so that a tick which
// reuses cached encoder rows encodes only the frames it has not encoded yet.
//
// frame0 must be a whole number of chunks (a multiple of t.mel_per_chunk):
// the returned layout has no way to express a start offset inside the first
// chunk, and callers rely on the tail's chunk grid lining up with the full
// pass's. It is a *frame* offset, not a pointer offset — the rows are still
// walked with the whole buffer's stride, because the frames from frame0 on
// are a column slice of a row-major matrix and are not contiguous.
void pack_mel_chunks(const float *         mel,  // [n_mels, n_mel_frames]
                     int                   n_mels,
                     int                   n_mel_frames,
                     const EncoderTiming & t,
                     std::vector<float> &  out,
                     int                   frame0 = 0);

// True when the first `cached_frames` frames of `cached` are bit-identical to
// the first `cached_frames` frames of `live`. `cached` is stored compactly as
// [n_mels, cached_frames]; `live` is [n_mels, live_frames].
//
// The row strides differ, so this cannot be a memcmp of two flat runs: at a
// different frame count the same flat index addresses a different
// (mel bin, frame) cell and the comparison would report a difference that is
// not there. It walks rows and compares each frame-wise.
//
// The R2T2 encoder cache uses this as its soundness test. The front-end
// normalizes per utterance over the whole buffer, so a new global maximum
// re-levels every earlier frame; the memcmp is what catches that, and a false
// result means "re-encode in full", never "assume equal".
bool mel_prefix_matches(const std::vector<float> & cached,
                        int32_t                    cached_frames,
                        const float *              live,
                        int32_t                    live_frames,
                        int32_t                    n_mels);

// Copy the first `frames` frames of a [n_mels, live_frames] mel into `dst`,
// stored compactly as [n_mels, frames] — the frame-aligned view
// mel_prefix_matches() expects on the next call.
void copy_mel_prefix(const float * live, int32_t live_frames, int32_t n_mels, int32_t frames, std::vector<float> & dst);

}  // namespace transcribe::qwen3_asr
