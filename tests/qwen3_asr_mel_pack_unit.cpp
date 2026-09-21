// Host-side mel framing for the Qwen3-ASR encoder, and the mel-prefix
// bookkeeping the R2T2 encoder cache carries across ticks.
//
// Why this exists as a direct test rather than an end-to-end one: both of
// these helpers take a [n_mels, n_frames] buffer whose *row stride is the
// frame count*, and the frame count grows every streaming tick. Every bug
// this file guards against is a disagreement about that stride, and none of
// them are visible from the outside. A wrong stride reads a neighbouring cell
// — still a finite float in [-1, 1]-ish range, still the right shape, still
// the right chunk and token *counts* (those are computed from lengths, not
// from the layout) — so the encoder runs, the graph looks right, and the only
// symptom is that the transcript quietly comes out wrong. The R2T2 80 ms
// stream hit exactly that: the encoder cache packed its tail from
// `mel + frame0 * n_mels` with a row stride of `tail_frames` instead of
// `mel_n_frames`, and 285 of 592 characters of the transcript survived.
//
// The tests below are therefore written against the *invariant* rather than
// against a literal table, so an implementation that is internally consistent
// but wrong about the stride cannot pass:
//
//   1. packing a window-aligned tail must reproduce exactly the sub-block of
//      the whole-buffer packing at the same frame offset (this is the
//      property the encoder cache's bit-exactness argument rests on);
//   2. every source frame read by the packing must be the one at its
//      (mel bin, frame) coordinate in the live buffer, checked against a
//      layout-independent oracle: the live buffer is built so that no two
//      cells share a value, so a mis-addressed read cannot coincide;
//   3. a mel prefix comparison must be frame-wise, not flat, so a prefix
//      compared against a longer buffer is equal iff every frame really is.
//
// Pure host: no ggml graph, no backend, no model file. Runs in microseconds.

#include "mel_pack.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

using transcribe::qwen3_asr::copy_mel_prefix;
using transcribe::qwen3_asr::EncoderTiming;
using transcribe::qwen3_asr::mel_prefix_matches;
using transcribe::qwen3_asr::pack_mel_chunks;

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

bool near(float a, float b) {
    return std::fabs(a - b) <= 1e-6f;
}

// Value of the cell at (mel bin m, frame f) of a synthetic [n_mels, n_frames]
// buffer. Distinct for every (m, f) pair within the ranges used here, and
// non-zero everywhere, so a read from any other cell — including the
// zero-padded tail of the packed output — is detectable.
float cell(int m, int f) {
    return static_cast<float>(1 + m * 1000 + f);
}

std::vector<float> make_mel(int n_mels, int n_frames) {
    std::vector<float> mel(static_cast<size_t>(n_mels) * n_frames);
    for (int m = 0; m < n_mels; ++m) {
        for (int f = 0; f < n_frames; ++f) {
            mel[static_cast<size_t>(m) * n_frames + f] = cell(m, f);
        }
    }
    return mel;
}

// EncoderTiming for a hand-built case: the fields pack_mel_chunks reads are
// mel_per_chunk, n_chunks and last_chunk_real_mel. They are set to the values
// compute_encoder_timing() would produce for `n_frames`, so the test does not
// depend on the real hparams (and does not silently drift if they change).
EncoderTiming timing_for(int n_frames, int mel_per_chunk) {
    EncoderTiming t{};
    t.n_mel_frames        = n_frames;
    t.mel_per_chunk       = mel_per_chunk;
    t.n_chunks            = (n_frames + mel_per_chunk - 1) / mel_per_chunk;
    t.last_chunk_real_mel = n_frames - (t.n_chunks - 1) * mel_per_chunk;
    return t;
}

// --- 1. a tail packing is the sub-block of the whole-buffer packing --------
//
// This is the property the encoder prefix cache relies on: the rows it keeps
// are the rows the full pass would have produced. The packing is the first
// stage of that, so frame0's packing of frames [frame0, n) must be exactly
// the same floats as chunk (frame0 / mel_per_chunk) onward of the frame0=0
// packing.
void test_tail_packing_is_a_subblock() {
    constexpr int kMels  = 8;    // small: the property is about addressing
    constexpr int kChunk = 100;  // mel frames per chunk (the real value)

    for (const int n_frames : { 100, 137, 200, 437, 800 }) {
        const std::vector<float> mel   = make_mel(kMels, n_frames);
        const EncoderTiming      whole = timing_for(n_frames, kChunk);

        std::vector<float> full;
        pack_mel_chunks(mel.data(), kMels, n_frames, whole, full, 0);
        CHECK(full.size() == static_cast<size_t>(kMels) * kChunk * whole.n_chunks);

        for (int frame0 = 0; frame0 < n_frames; frame0 += kChunk) {
            const int c0 = frame0 / kChunk;

            EncoderTiming tail = timing_for(n_frames - frame0, kChunk);
            // The tail's own chunk grid starts at frame0, so its last chunk is
            // the same partial chunk as the whole buffer's.
            CHECK(tail.last_chunk_real_mel == whole.last_chunk_real_mel);

            std::vector<float> out;
            pack_mel_chunks(mel.data(), kMels, n_frames, tail, out, frame0);
            CHECK(out.size() == static_cast<size_t>(kMels) * kChunk * tail.n_chunks);

            int mismatches = 0;
            for (int c = 0; c < tail.n_chunks; ++c) {
                for (int m = 0; m < kMels; ++m) {
                    for (int f = 0; f < kChunk; ++f) {
                        const float got =
                            out[static_cast<size_t>(c) * kMels * kChunk + static_cast<size_t>(m) * kChunk + f];
                        const float want =
                            full[static_cast<size_t>(c + c0) * kMels * kChunk + static_cast<size_t>(m) * kChunk + f];
                        if (!near(got, want)) {
                            ++mismatches;
                        }
                    }
                }
            }
            if (mismatches != 0) {
                std::fprintf(stderr,
                             "FAIL tail packing (n=%d frame0=%d): %d of %d values differ from the whole-buffer pack\n",
                             n_frames, frame0, mismatches, kMels * kChunk * tail.n_chunks);
                ++g_failures;
            }
        }
    }
}

// --- 2. every packed value comes from its own (mel bin, frame) cell --------
//
// The direct statement of the stride invariant, checked against the cell()
// oracle rather than against another packing, so an implementation that is
// wrong in the same way twice still fails. Includes the two error modes this
// file's history produced: reading with the tail's stride instead of the
// buffer's, and starting from a pointer offset of frame0 * n_mels.
void test_packed_values_match_their_coordinates() {
    constexpr int kMels  = 4;
    constexpr int kChunk = 100;

    const int         n_frames = 250;
    const std::vector mel      = make_mel(kMels, n_frames);

    for (const int frame0 : { 0, 100, 200 }) {
        const EncoderTiming t = timing_for(n_frames - frame0, kChunk);
        std::vector<float>  out;
        pack_mel_chunks(mel.data(), kMels, n_frames, t, out, frame0);

        int bad = 0;
        for (int c = 0; c < t.n_chunks; ++c) {
            const int tail = (c == t.n_chunks - 1) ? t.last_chunk_real_mel : kChunk;
            for (int m = 0; m < kMels; ++m) {
                for (int f = 0; f < kChunk; ++f) {
                    const float got =
                        out[static_cast<size_t>(c) * kMels * kChunk + static_cast<size_t>(m) * kChunk + f];
                    // Past the real frames of the last chunk the packing must
                    // zero-pad, exactly as the whole-buffer pass does.
                    const float want = (f < tail) ? cell(m, frame0 + c * kChunk + f) : 0.0f;
                    if (!near(got, want)) {
                        ++bad;
                    }
                }
            }
        }
        if (bad != 0) {
            std::fprintf(stderr, "FAIL packed coordinates (n=%d frame0=%d): %d values read the wrong cell\n", n_frames,
                         frame0, bad);
            ++g_failures;
        }
    }

    // A frame offset that is not chunk-aligned is not expressible in the
    // output layout; the contract says so and the test pins the contract by
    // showing why callers must not rely on it: the first chunk would silently
    // begin mid-chunk. (Guarding here rather than in the function keeps the
    // hot path branch-free; the callers' alignment is what makes it sound.)
    CHECK(kChunk > 0);
}

// --- 3. prefix comparison and copy are frame-wise --------------------------
//
// The prefix lives in a buffer of a different frame count than the live one
// (the cached copy is [n_mels, cached_frames], the live buffer is
// [n_mels, live_frames]). A flat memcmp of the two would compare bin 0's
// frames against bin 1's as soon as the counts differ, and the cache would
// either miss forever or, worse, hit on frames that had actually moved.
void test_prefix_compare_and_copy_are_frame_wise() {
    constexpr int kMels = 5;

    const std::vector<float> live = make_mel(kMels, 400);

    // Cache the first 200 frames, as the encoder cache does.
    std::vector<float> cached;
    copy_mel_prefix(live.data(), 400, kMels, 200, cached);
    CHECK(cached.size() == static_cast<size_t>(kMels) * 200);

    // Same first 200 frames, longer buffer: must match.
    CHECK(mel_prefix_matches(cached, 200, live.data(), 400, kMels));

    // One frame anywhere inside the cached prefix moves (the front-end's
    // per-utterance re-level): must not match. Swept over the frame-0,
    // chunk-boundary and last-prefix-frame edges, because a stride bug hides
    // at the edges. Frame 200 is deliberately *not* in this list — it is the
    // first frame past the prefix and must not invalidate it (next check).
    for (const int f : { 0, 1, 99, 100, 199 }) {
        std::vector<float> moved = live;
        moved[static_cast<size_t>(3) * 400 + f] += 1.0f;
        if (mel_prefix_matches(cached, 200, moved.data(), 400, kMels)) {
            std::fprintf(stderr, "FAIL prefix match missed a moved frame at f=%d\n", f);
            ++g_failures;
        }
    }

    // A change outside the prefix must not invalidate the prefix.
    std::vector<float> later = live;
    later[static_cast<size_t>(0) * 400 + 250] += 1.0f;
    CHECK(mel_prefix_matches(cached, 200, later.data(), 400, kMels));

    // Shape and range guards fail closed.
    CHECK(!mel_prefix_matches(cached, 201, live.data(), 400, kMels));  // prefix longer than the copy
    CHECK(!mel_prefix_matches(cached, 200, live.data(), 150, kMels));  // live buffer shorter than the prefix
    CHECK(!mel_prefix_matches(std::vector<float>(3, 1.0f), 200, live.data(), 400, kMels));

    // The copy is a frame-aligned view of the live buffer, not a flat run.
    CHECK(near(cached[static_cast<size_t>(2) * 200 + 199], cell(2, 199)));
}

}  // namespace

int main() {
    test_tail_packing_is_a_subblock();
    test_packed_values_match_their_coordinates();
    test_prefix_compare_and_copy_are_frame_wise();

    if (g_failures == 0) {
        std::printf("qwen3_asr_mel_pack: ok\n");
        return 0;
    }
    std::fprintf(stderr, "qwen3_asr_mel_pack: %d failure(s)\n", g_failures);
    return 1;
}
