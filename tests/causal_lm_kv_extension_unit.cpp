// Extending a KV cache computes the same rows a full prefill would.
//
// The R2T2 stream re-prefills its prompt on every tick, and after the encoder
// cache has carried the finished audio over, almost all of that prompt is
// already sitting in the KV cache from the previous tick. `BlockOpts::
// kv_write_off` lets a block *extend* that cache instead of rebuilding it:
// write rows [P, P + T_seq), attend over [0, P + T_seq).
//
// The claim is causal and does not depend on the shape of the graph: K/V of
// position j is a function of positions [0, j] only, so row j is the same
// whether the token that produced it was the (j+1)-th of one long prefill or
// the (j-P+1)-th of a continuation after P rows that were computed earlier.
// What *can* differ is the last bits: a differently-shaped graph accumulates
// the same sums in a different order, which is why this is the one extension
// here that is not byte-identical by construction and why the streaming
// caller keeps a kill switch. On this CPU path the agreement comes out exact
// (diff 0) -- both graphs reduce the same 24-wide mask row in the same order
// -- but that is an artifact of the CPU kernels, so the check below is written
// against a float tolerance rather than against zero.
//
// This test pins the property that has to hold anyway — the rows must match to
// float tolerance, and the three ways of getting the *geometry* wrong (a write
// at the wrong offset, a read window that does not cover the reused rows, a
// mask that leaks the future) must not survive it.
//
// Three variants per KV type, all from the same weights and the same input:
//   A  single-shot prefill of all N positions                (the reference)
//   B  prefill [0, P) into a fresh cache, then extend by [P, N) via n_past=P
//   C  B's geometry with a mask that hides nothing            (negative control)
// B must agree with A on both the returned hidden state and the KV rows it
// wrote; C must *disagree*, which is what makes an agreement in B mean
// something. Run once with F32 KV through the non-flash attention branch and
// once with F16 KV through flash, because both branches take the n_kv_read
// changes.

#include "causal_lm/causal_lm.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
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

// Deterministic and small: the test compares two graphs' arithmetic, so the
// values only have to be non-degenerate.
struct Lcg {
    uint32_t s = 0x2545f491u;

    float next() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xffffu) / 32768.0f - 1.0f;
    }

    void fill(std::vector<float> & v, float scale = 1.0f) {
        for (float & f : v) {
            f = next() * scale;
        }
    }
};

struct Shape {
    int hidden     = 64;
    int n_heads    = 8;
    int n_kv_heads = 2;
    int head_dim   = 8;
    int inter      = 128;
    int n_ctx      = 64;
};

struct Stats {
    float max_abs_diff = 0.0f;
    float scale        = 0.0f;  // max |reference| over the compared elements
};

void accumulate(Stats & st, const std::vector<float> & got, const std::vector<float> & want, size_t from, size_t to) {
    for (size_t i = from; i < to; ++i) {
        st.max_abs_diff = std::max(st.max_abs_diff, std::fabs(got[i] - want[i]));
        st.scale        = std::max(st.scale, std::fabs(want[i]));
    }
}

// Row r of `got` holds a different absolute position than row r of `want` when
// the two tensors start at different offsets -- the extension's row r is the
// full prefill's row first_row + r. Comparing them at the same index instead
// compares two different tokens and reports a diff the size of the signal,
// which is what the first version of this test did.
void accumulate_rows(Stats &                    st,
                     const std::vector<float> & got,
                     int                        got_row,
                     const std::vector<float> & want,
                     int                        want_row,
                     int                        rows,
                     int                        width) {
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < width; ++i) {
            const size_t gi = static_cast<size_t>(got_row + r) * width + i;
            const size_t wi = static_cast<size_t>(want_row + r) * width + i;
            st.max_abs_diff = std::max(st.max_abs_diff, std::fabs(got[gi] - want[wi]));
            st.scale        = std::max(st.scale, std::fabs(want[wi]));
        }
    }
}

// Everything one variant needs, kept in the single shared context so the
// weights are literally the same tensors in every graph.
struct Graph {
    ggml_cgraph * graph = nullptr;
    // Non-const because building the graph forward from it is what gives the
    // graph its nodes; nothing writes through it after that.
    ggml_tensor * out   = nullptr;
};

struct Variant {
    transcribe::causal_lm::KvCache cache;
    Graph                          first;  // prefill [0, P)
    Graph                          rest;   // the part under test
    std::vector<float>             out;    // [hidden, N - P) hidden state
};

}  // namespace

int main() {
    using namespace transcribe::causal_lm;

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "SKIP: could not initialize CPU backend\n");
        return 77;
    }

    const Shape sh{};
    const int   N      = 24;  // total positions
    const int   P      = 16;  // rows the cache already holds when the extension runs
    const int   kv_dim = sh.n_kv_heads * sh.head_dim;
    const int   T_rest = N - P;

    for (const ggml_type kv_type : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        const bool   flash     = (kv_type == GGML_TYPE_F16);  // the branch the type selects
        const size_t ctx_bytes = 64u * 1024u * 1024u;

        ggml_init_params ip{};
        ip.mem_size        = ctx_bytes;
        ip.no_alloc        = true;
        ggml_context * ctx = ggml_init(ip);
        CHECK(ctx != nullptr);
        if (ctx == nullptr) {
            ggml_backend_free(backend);
            return EXIT_FAILURE;
        }

        // --- weights (shared by every graph) -------------------------------
        Lcg                rng;
        std::vector<float> w_attn_norm(sh.hidden), w_ffn_norm(sh.hidden);
        std::vector<float> w_q(static_cast<size_t>(sh.hidden) * sh.n_heads * sh.head_dim);
        std::vector<float> w_k(static_cast<size_t>(sh.hidden) * sh.n_kv_heads * sh.head_dim);
        std::vector<float> w_v(static_cast<size_t>(sh.hidden) * sh.n_kv_heads * sh.head_dim);
        std::vector<float> w_o(static_cast<size_t>(sh.n_heads * sh.head_dim) * sh.hidden);
        std::vector<float> w_gu(static_cast<size_t>(sh.hidden) * 2 * sh.inter);
        std::vector<float> w_down(static_cast<size_t>(sh.inter) * sh.hidden);
        for (std::vector<float> * w : { &w_attn_norm, &w_q, &w_k, &w_v, &w_o, &w_gu, &w_down }) {
            rng.fill(*w, 0.15f);
        }
        // RMSNorm weights near 1: the block is a scale-free test otherwise.
        for (float & f : w_attn_norm) {
            f = 1.0f + 0.05f * f;
        }
        for (float & f : w_ffn_norm) {
            f = 1.0f + 0.05f * f;
        }

        ggml_tensor * t_attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sh.hidden);
        ggml_tensor * t_ffn_norm  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sh.hidden);
        ggml_tensor * t_q         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.hidden, sh.n_heads * sh.head_dim);
        ggml_tensor * t_k         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.hidden, sh.n_kv_heads * sh.head_dim);
        ggml_tensor * t_v         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.hidden, sh.n_kv_heads * sh.head_dim);
        ggml_tensor * t_o         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.n_heads * sh.head_dim, sh.hidden);
        ggml_tensor * t_gu        = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.hidden, 2 * sh.inter);
        ggml_tensor * t_down      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.inter, sh.hidden);

        BlockView view;
        view.norm_attn_w   = t_attn_norm;
        view.norm_ffn_w    = t_ffn_norm;
        view.attn_q_w      = t_q;
        view.attn_k_w      = t_k;
        view.attn_v_w      = t_v;
        view.attn_o_w      = t_o;
        view.ffn_gate_up_w = t_gu;
        view.ffn_down_w    = t_down;

        BlockParams params;
        params.n_heads      = sh.n_heads;
        params.n_kv_heads   = sh.n_kv_heads;
        params.head_dim     = sh.head_dim;
        params.max_position = sh.n_ctx;
        params.rms_eps      = 1e-5f;
        params.rope_theta   = 10000.0f;

        // --- the input, shared by every graph ------------------------------
        ggml_tensor *      x_all = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.hidden, N);
        std::vector<float> x_data(static_cast<size_t>(sh.hidden) * N);
        rng.fill(x_data);

        auto positions_tensor = [&](int from, int count) {
            ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, count);
            return std::pair<ggml_tensor *, std::vector<int32_t>>{ t, [&] {
                                                                      std::vector<int32_t> v(
                                                                          static_cast<size_t>(count));
                                                                      for (int i = 0; i < count; ++i) {
                                                                          v[static_cast<size_t>(i)] = from + i;
                                                                      }
                                                                      return v;
                                                                  }() };
        };
        auto mask_tensor = [&](int kv_extent, int n_query, int n_past, bool leaky) {
            std::vector<ggml_fp16_t> m(static_cast<size_t>(kv_extent) * n_query);
            fill_prefill_chunk_mask(m.data(), kv_extent, n_query, n_past);
            if (leaky) {
                for (ggml_fp16_t & f : m) {
                    if (ggml_fp16_to_fp32(f) < -1e4f) {
                        f = ggml_fp32_to_fp16(0.0f);
                    }
                }
            }
            return m;
        };

        // Variant A: one prefill of everything.
        auto          pos_a    = positions_tensor(0, N);
        const auto    mask_a   = mask_tensor(N, N, 0, /*leaky=*/false);
        ggml_tensor * t_mask_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N, N);

        Variant A, B, C;
        CHECK(kv_init(A.cache, backend, sh.n_ctx, sh.n_kv_heads, sh.head_dim, /*n_layer=*/1, kv_type));
        CHECK(kv_init(B.cache, backend, sh.n_ctx, sh.n_kv_heads, sh.head_dim, 1, kv_type));
        CHECK(kv_init(C.cache, backend, sh.n_ctx, sh.n_kv_heads, sh.head_dim, 1, kv_type));

        // A
        {
            A.first.graph = ggml_new_graph_custom(ctx, 4096, false);
            BlockOpts opts{};
            opts.use_flash    = flash;
            opts.kv_write_off = 0;
            A.first.out       = block_prefill(ctx, A.first.graph, x_all, view, params, A.cache,
                                              /*layer_idx=*/0, N, t_mask_a, pos_a.first, opts);
            // block_prefill expands only its own KV-write subgraph; without
            // expanding from the returned node the graph would be empty and
            // the compute a no-op that "succeeds", leaving every read at zero.
            ggml_build_forward_expand(A.first.graph, A.first.out);
        }
        // B/C: the shared "prefill [0, P)" half.
        auto          pos_0    = positions_tensor(0, P);
        const auto    mask_0   = mask_tensor(P, P, 0, false);
        ggml_tensor * t_mask_0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, P, P);

        // The extension half: mask width N (the reused rows are visible), and
        // the queries are the tail of the same input.
        auto          pos_p    = positions_tensor(P, T_rest);
        const auto    mask_b   = mask_tensor(N, T_rest, P, /*leaky=*/false);
        const auto    mask_c   = mask_tensor(N, T_rest, P, /*leaky=*/true);
        ggml_tensor * t_mask_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N, T_rest);
        ggml_tensor * t_mask_c = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, N, T_rest);

        // Each graph has to be handed a tensor whose second axis *is* its
        // T_seq: ggml reads the width off the tensor, not off the argument, so
        // passing the whole x_all (N columns) to the P-position prefill builds
        // a [.., N] projection and then asserts in reshape_4d. Q tail likewise.
        const size_t  row_bytes = ggml_type_size(GGML_TYPE_F32) * sh.hidden;
        ggml_tensor * x_head    = ggml_view_2d(ctx, x_all, sh.hidden, P, row_bytes, 0);
        ggml_tensor * x_tail    = ggml_view_2d(ctx, x_all, sh.hidden, T_rest, row_bytes, row_bytes * P);

        for (Variant * v : { &B, &C }) {
            v->first.graph = ggml_new_graph_custom(ctx, 4096, false);
            BlockOpts o{};
            o.use_flash    = flash;
            o.kv_write_off = 0;
            v->first.out =
                block_prefill(ctx, v->first.graph, x_head, view, params, v->cache, 0, P, t_mask_0, pos_0.first, o);
            ggml_build_forward_expand(v->first.graph, v->first.out);

            v->rest.graph = ggml_new_graph_custom(ctx, 4096, false);
            BlockOpts e{};
            e.use_flash     = flash;
            e.kv_write_off  = P;
            const bool is_c = (v == &C);
            v->rest.out     = block_prefill(ctx, v->rest.graph, x_tail, view, params, v->cache, 0, T_rest,
                                            is_c ? t_mask_c : t_mask_b, pos_p.first, e);
            ggml_build_forward_expand(v->rest.graph, v->rest.out);
        }

        // --- allocate, upload, compute -------------------------------------
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        CHECK(buf != nullptr);
        if (buf == nullptr) {
            ggml_backend_free(backend);
            return EXIT_FAILURE;
        }
        ggml_backend_tensor_set(t_attn_norm, w_attn_norm.data(), 0, w_attn_norm.size() * sizeof(float));
        ggml_backend_tensor_set(t_ffn_norm, w_ffn_norm.data(), 0, w_ffn_norm.size() * sizeof(float));
        ggml_backend_tensor_set(t_q, w_q.data(), 0, w_q.size() * sizeof(float));
        ggml_backend_tensor_set(t_k, w_k.data(), 0, w_k.size() * sizeof(float));
        ggml_backend_tensor_set(t_v, w_v.data(), 0, w_v.size() * sizeof(float));
        ggml_backend_tensor_set(t_o, w_o.data(), 0, w_o.size() * sizeof(float));
        ggml_backend_tensor_set(t_gu, w_gu.data(), 0, w_gu.size() * sizeof(float));
        ggml_backend_tensor_set(t_down, w_down.data(), 0, w_down.size() * sizeof(float));
        ggml_backend_tensor_set(x_all, x_data.data(), 0, x_data.size() * sizeof(float));
        ggml_backend_tensor_set(t_mask_a, mask_a.data(), 0, mask_a.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_mask_0, mask_0.data(), 0, mask_0.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_mask_b, mask_b.data(), 0, mask_b.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(t_mask_c, mask_c.data(), 0, mask_c.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(pos_a.first, pos_a.second.data(), 0, pos_a.second.size() * sizeof(int32_t));
        ggml_backend_tensor_set(pos_0.first, pos_0.second.data(), 0, pos_0.second.size() * sizeof(int32_t));
        ggml_backend_tensor_set(pos_p.first, pos_p.second.data(), 0, pos_p.second.size() * sizeof(int32_t));

        auto run = [&](Graph & g) {
            CHECK(ggml_backend_graph_compute(backend, g.graph) == GGML_STATUS_SUCCESS);
        };
        run(A.first);
        run(B.first);
        run(C.first);
        run(B.rest);
        run(C.rest);

        // --- A's reference rows, and the extension's -----------------------
        std::vector<float> out_a(static_cast<size_t>(sh.hidden) * N);
        ggml_backend_tensor_get(A.first.out, out_a.data(), 0, out_a.size() * sizeof(float));

        for (Variant * v : { &B, &C }) {
            v->out.assign(static_cast<size_t>(sh.hidden) * T_rest, 0.0f);
            ggml_backend_tensor_get(v->rest.out, v->out.data(), 0, v->out.size() * sizeof(float));
        }

        auto compare_hidden = [&](const Variant & v) {
            // Row q of the extension is absolute position P + q, so it is
            // compared against that row of the full prefill -- not against
            // row q, which is a different token at a different position.
            Stats st;
            accumulate_rows(st, v.out, 0, out_a, P, T_rest, sh.hidden);
            return st;
        };

        // The KV rows the extension wrote must match the reference's rows too:
        // the hidden state is read back after the last block, so a wrong write
        // offset can still leave the *output* right and only corrupt the cache
        // for whoever runs next. Read through the cache's own element size --
        // F16 rows are half as wide, so striding them by sizeof(float) would
        // read the wrong region entirely (and, both caches holding the same
        // bytes, would compare that wrong region with itself and report 0).
        auto kv_rows = [&](const KvCache & cache, int from, int count) {
            const size_t         elem = ggml_element_size(cache.self_k);
            const size_t         n    = static_cast<size_t>(kv_dim) * count;
            const size_t         off  = static_cast<size_t>(from) * static_cast<size_t>(kv_dim) * elem;
            std::vector<uint8_t> raw(n * elem);
            ggml_backend_tensor_get(cache.self_k, raw.data(), off, raw.size());

            std::vector<float> rows(n);
            if (cache.self_k->type == GGML_TYPE_F16) {
                const auto * h = reinterpret_cast<const ggml_fp16_t *>(raw.data());
                for (size_t i = 0; i < n; ++i) {
                    rows[i] = ggml_fp16_to_fp32(h[i]);
                }
            } else {
                std::memcpy(rows.data(), raw.data(), raw.size());
            }
            return rows;
        };

        const Stats kvB = [&] {
            const std::vector<float> ref = kv_rows(A.cache, P, T_rest);
            const std::vector<float> got = kv_rows(B.cache, P, T_rest);
            Stats                    st;
            accumulate(st, got, ref, 0, ref.size());
            return st;
        }();
        const Stats reusedB = [&] {
            const std::vector<float> ref = kv_rows(A.cache, 0, P);
            const std::vector<float> got = kv_rows(B.cache, 0, P);
            Stats                    st;
            accumulate(st, got, ref, 0, ref.size());
            return st;
        }();

        const Stats hidB = compare_hidden(B);
        const Stats hidC = compare_hidden(C);

        // Float tolerance, relative to the magnitude of the reference rows:
        // the two graphs are mathematically identical, so the only difference
        // is summation order.
        const float tol = 1e-4f * std::max(1.0f, hidB.scale);
        std::printf(
            "kv=%s flash=%d  hidden: diff=%.3e (scale %.3f, tol %.1e)  "
            "kv[P,%d): diff=%.3e  kv[0,%d): diff=%.3e  leaky-control: diff=%.3e\n",
            kv_type == GGML_TYPE_F16 ? "f16" : "f32", flash ? 1 : 0, hidB.max_abs_diff, hidB.scale, tol, N,
            kvB.max_abs_diff, P, reusedB.max_abs_diff, hidC.max_abs_diff);

        CHECK(hidB.scale > 0.01f);           // non-degenerate reference
        CHECK(hidB.max_abs_diff <= tol);     // the extension reproduces the full prefill
        CHECK(kvB.max_abs_diff <= tol);      // ...in the rows it wrote...
        CHECK(reusedB.max_abs_diff <= tol);  // ...and without disturbing the reused ones
        // The control has to fail loudly, or an agreement above proves nothing:
        // a mask that leaks the next token changes the row it leaked into.
        CHECK(hidC.max_abs_diff > 1e-3f);

        for (Variant * v : { &A, &B, &C }) {
            v->cache.free();
        }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    if (g_failures > 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        ggml_backend_free(backend);
        return EXIT_FAILURE;
    }
    ggml_backend_free(backend);
    std::printf("causal_lm_kv_extension: ok\n");
    return EXIT_SUCCESS;
}
