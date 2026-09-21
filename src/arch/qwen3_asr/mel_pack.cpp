// mel_pack.cpp — see mel_pack.h. Host-only, no backend dependency.

#include "mel_pack.h"

#include <cstring>

namespace transcribe::qwen3_asr {

void pack_mel_chunks(const float *         mel,
                     int                   n_mels,
                     int                   n_mel_frames,
                     const EncoderTiming & t,
                     std::vector<float> &  out,
                     int                   frame0) {
    const size_t per_chunk_elems = static_cast<size_t>(t.mel_per_chunk) * static_cast<size_t>(n_mels);
    out.assign(per_chunk_elems * static_cast<size_t>(t.n_chunks), 0.0f);
    if (mel == nullptr || n_mels <= 0 || n_mel_frames <= 0 || frame0 < 0 || t.mel_per_chunk <= 0) {
        return;
    }

    for (int c = 0; c < t.n_chunks; ++c) {
        const int     tail  = (c == t.n_chunks - 1) ? t.last_chunk_real_mel : t.mel_per_chunk;
        // A chunk whose frames run past the buffer contributes nothing: the
        // output is already zero-filled, which is the same zero pad the whole
        // -buffer path produces for its short last chunk.
        const int64_t start = static_cast<int64_t>(frame0) + static_cast<int64_t>(c) * t.mel_per_chunk;
        if (start >= n_mel_frames) {
            continue;
        }
        const int avail = static_cast<int>(n_mel_frames - start);
        const int run   = avail < tail ? avail : tail;
        if (run <= 0) {
            continue;
        }
        for (int m = 0; m < n_mels; ++m) {
            const float * src =
                mel + static_cast<size_t>(m) * static_cast<size_t>(n_mel_frames) + static_cast<size_t>(start);
            float * dst = out.data() + static_cast<size_t>(c) * per_chunk_elems +
                          static_cast<size_t>(m) * static_cast<size_t>(t.mel_per_chunk);
            std::memcpy(dst, src, static_cast<size_t>(run) * sizeof(float));
        }
    }
}

bool mel_prefix_matches(const std::vector<float> & cached,
                        int32_t                    cached_frames,
                        const float *              live,
                        int32_t                    live_frames,
                        int32_t                    n_mels) {
    if (live == nullptr || n_mels <= 0 || cached_frames <= 0 || cached_frames > live_frames ||
        cached.size() != static_cast<size_t>(n_mels) * static_cast<size_t>(cached_frames)) {
        return false;
    }
    for (int32_t m = 0; m < n_mels; ++m) {
        if (std::memcmp(cached.data() + static_cast<size_t>(m) * static_cast<size_t>(cached_frames),
                        live + static_cast<size_t>(m) * static_cast<size_t>(live_frames),
                        static_cast<size_t>(cached_frames) * sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

void copy_mel_prefix(const float *        live,
                     int32_t              live_frames,
                     int32_t              n_mels,
                     int32_t              frames,
                     std::vector<float> & dst) {
    dst.resize(static_cast<size_t>(n_mels) * static_cast<size_t>(frames));
    if (live == nullptr || n_mels <= 0 || frames <= 0 || frames > live_frames) {
        return;
    }
    for (int32_t m = 0; m < n_mels; ++m) {
        std::memcpy(dst.data() + static_cast<size_t>(m) * static_cast<size_t>(frames),
                    live + static_cast<size_t>(m) * static_cast<size_t>(live_frames),
                    static_cast<size_t>(frames) * sizeof(float));
    }
}

}  // namespace transcribe::qwen3_asr
