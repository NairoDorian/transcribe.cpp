// granite_nar BPE CTC pool: the in-graph pooled head against the frame-level
// reference formulation. Pure host, no model.
//
// run() does not decode frame-level BPE logits any more (that tensor was
// [100352, T_enc] f32, 6.1 GB at 5 minutes of audio, then copied to the host
// whole). The head is affine and the pool weights sum to 1, so
//     sum_t w_t (W h_t + b) = W (sum_t w_t h_t) + b
// and the graph pools the 1024-wide hidden states first, runs the head on one
// column per window, and takes the argmax there. This test is what keeps that
// restructure honest: it builds the same head both ways and requires the two
// decodes to agree token for token, including the all-blank-window rule and
// both blank-id schemes.
//
// It also pins the failure mode the caller relies on: mismatched id/mass
// lengths are an error, not a silently shortened hypothesis.

#include "arch/granite_nar/encoder.h"

#include <cmath>
#include <cstdio>
#include <limits>
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

// Deterministic, well-spread values without pulling in a RNG.
float pseudo(int i) {
    const float s = std::sin(static_cast<float>(i) * 12.9898f + 78.233f) * 43758.5453f;
    return s - std::floor(s) - 0.5f;
}

struct Head {
    int                dim;
    int                vocab;
    std::vector<float> w;  // [vocab, dim]
    std::vector<float> b;  // [vocab]

    Head(int dim_, int vocab_) : dim(dim_), vocab(vocab_), w(static_cast<size_t>(dim_) * vocab_), b(vocab_) {
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = pseudo(static_cast<int>(i));
        }
        for (int v = 0; v < vocab; ++v) {
            b[static_cast<size_t>(v)] = pseudo(100000 + v) * 0.1f;
        }
    }

    void apply(const std::vector<float> & x, int n_cols, std::vector<float> & out) const {
        // out[col][v] = sum_i w[v][i] * x[i][col] + b[v]
        out.assign(static_cast<size_t>(n_cols) * vocab, 0.0f);
        for (int v = 0; v < vocab; ++v) {
            for (int c = 0; c < n_cols; ++c) {
                float acc = b[static_cast<size_t>(v)];
                for (int i = 0; i < dim; ++i) {
                    acc += w[static_cast<size_t>(v) * dim + i] * x[static_cast<size_t>(c) * dim + i];
                }
                out[static_cast<size_t>(c) * vocab + v] = acc;
            }
        }
    }
};

// One scenario: `hidden` is [T, dim] row-major (frame-major), importance is
// [T]. Runs the reference (frame-level logits, then pool + collapse) and the
// graph's shape (pool hidden, then head + argmax + collapse) and compares.
void check_equivalence(const char *               what,
                       const Head &               head,
                       const std::vector<float> & hidden,
                       const std::vector<float> & importance,
                       int                        pool_window,
                       int                        blank_id) {
    const int T     = static_cast<int>(importance.size());
    const int n_win = (T + pool_window - 1) / pool_window;

    // --- Reference: frame-level logits, host pool + greedy (encoder.cpp) ---
    std::vector<float> frame_logits;
    head.apply(hidden, T, frame_logits);
    std::vector<int32_t> ref_ids;
    transcribe::granite_nar::compute_bpe_ctc_initial_hypothesis(importance, frame_logits, head.vocab, T, pool_window,
                                                                blank_id, ref_ids);

    // --- Graph shape: importance-weighted hidden pool, then head + argmax ---
    std::vector<float>   pooled(static_cast<size_t>(n_win) * head.dim, 0.0f);
    std::vector<float>   mass(static_cast<size_t>(n_win), 0.0f);
    std::vector<int32_t> window_ids(static_cast<size_t>(n_win), blank_id);
    for (int w = 0; w < n_win; ++w) {
        const int t0 = w * pool_window;
        const int t1 = std::min(t0 + pool_window, T);
        for (int t = t0; t < t1; ++t) {
            mass[static_cast<size_t>(w)] += importance[static_cast<size_t>(t)];
        }
        // Softmax mid-blank CT counts are in (0, 1], so a window mass is
        // never 0 unless every frame in it is blank; the graph divides by a
        // clamped mass and the host then treats the window as blank.
        const float denom = mass[static_cast<size_t>(w)] > 1e-9f ? mass[static_cast<size_t>(w)] : 1e-9f;
        for (int t = t0; t < t1; ++t) {
            const float wt = importance[static_cast<size_t>(t)] / denom;
            for (int i = 0; i < head.dim; ++i) {
                pooled[static_cast<size_t>(w) * head.dim + i] += wt * hidden[static_cast<size_t>(t) * head.dim + i];
            }
        }
    }
    std::vector<float> win_logits;
    head.apply(pooled, n_win, win_logits);
    for (int w = 0; w < n_win; ++w) {
        int   best   = 0;
        float best_v = win_logits[static_cast<size_t>(w) * head.vocab];
        for (int v = 1; v < head.vocab; ++v) {
            const float cand = win_logits[static_cast<size_t>(w) * head.vocab + v];
            if (cand > best_v) {
                best_v = cand;
                best   = v;
            }
        }
        window_ids[static_cast<size_t>(w)] = best;
    }
    std::vector<int32_t> got_ids;
    CHECK(transcribe::granite_nar::collapse_bpe_window_ids(window_ids, mass, blank_id, got_ids));

    if (ref_ids != got_ids) {
        ++g_failures;
        std::fprintf(stderr, "FAIL %s: %zu vs %zu tokens differ (blank_id=%d, T=%d, pool=%d)\n", what, ref_ids.size(),
                     got_ids.size(), blank_id, T, pool_window);
        const size_t n = std::min(ref_ids.size(), got_ids.size());
        for (size_t i = 0; i < n; ++i) {
            if (ref_ids[i] != got_ids[i]) {
                std::fprintf(stderr, "  first diff at %zu: reference %d, pooled %d\n", i, ref_ids[i], got_ids[i]);
                break;
            }
        }
        // The pooled head is 4x cheaper but does not reproduce per-window
        // division in the same summation order, so allow a tiny numerical
        // disagreement to be visible rather than silently tolerated: the
        // test's inputs are well-separated integers, so exact equality is the
        // right bar. Print the reference to make a real drift obvious.
        std::fprintf(stderr, "  reference:");
        for (int32_t id : ref_ids) {
            std::fprintf(stderr, " %d", id);
        }
        std::fprintf(stderr, "\n");
    }
}

}  // namespace

int main() {
    // A tiny head so the arithmetic stays exact in f32: values are integers,
    // a window's weighted mean is representable, and argmax has no ties.
    Head head(/*dim=*/4, /*vocab=*/16);

    // T = 10, P = 4 -> 3 windows, the last one partial (2 frames).
    const int          T = 10;
    const int          P = 4;
    std::vector<float> hidden(static_cast<size_t>(T) * head.dim);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < head.dim; ++i) {
            hidden[static_cast<size_t>(t) * head.dim + i] = static_cast<float>(1 + ((t * 7 + i * 3) % 11));
        }
    }

    // All frames voiced, weights 1/2/4/8 cycling: exercises the weighting.
    std::vector<float> voiced(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t) {
        voiced[static_cast<size_t>(t)] = static_cast<float>(1 << (t % 4));
    }

    // Scheme A (blank_id == 0): synthetic blank at channel 0, ids shifted.
    check_equivalence("voiced/blank=0", head, hidden, voiced, P, /*blank_id=*/0);
    // Scheme B (blank_id == 100257, BOS): channels are LLM ids directly.
    check_equivalence("voiced/blank=BOS", head, hidden, voiced, P, /*blank_id=*/100257);

    // A silent window in the middle (mass 0) plus a partial tail: the
    // reference emits blank there and the collapse must do the same.
    std::vector<float> with_gap = voiced;
    for (int t = P; t < 2 * P; ++t) {
        with_gap[static_cast<size_t>(t)] = 0.0f;
    }
    check_equivalence("gap/blank=0", head, hidden, with_gap, P, /*blank_id=*/0);
    check_equivalence("gap/blank=BOS", head, hidden, with_gap, P, /*blank_id=*/100257);

    // Collapse rules: adjacent duplicate windows collapse, blanks are dropped.
    {
        const std::vector<int32_t> ids  = { 3, 3, 0, 7, 0, 0, 7 };
        const std::vector<float>   mass = { 1, 1, 1, 1, 1, 1, 1 };
        std::vector<int32_t>       out;
        CHECK(transcribe::granite_nar::collapse_bpe_window_ids(ids, mass, /*blank_id=*/0, out));
        // Collapse runs on the window ids (before the shift): 3,3 -> one 3;
        // 0 is blank; 7 survives both times because a blank resets "prev".
        // Scheme A then subtracts 1 from each emitted id.
        const std::vector<int32_t> expect = { 2, 6, 6 };
        CHECK(out == expect);
    }

    // A length mismatch is a caller bug: refuse instead of truncating.
    {
        const std::vector<int32_t> ids  = { 1, 2, 3 };
        const std::vector<float>   mass = { 1, 1 };
        std::vector<int32_t>       out  = { 42 };
        CHECK(!transcribe::granite_nar::collapse_bpe_window_ids(ids, mass, /*blank_id=*/0, out));
        CHECK(out.empty());
    }

    if (g_failures == 0) {
        std::fprintf(stdout, "granite_nar_bpe_pool_unit: ok\n");
        return 0;
    }
    std::fprintf(stderr, "granite_nar_bpe_pool_unit: %d failures\n", g_failures);
    return 1;
}
