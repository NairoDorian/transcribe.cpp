// Numeric checks for GGML_TYPE_TQ1_G128 (patches/ggml/0003), the group-128
// ternary type parakeet-redux ships in.
//
//  1. Packing is lossless: ggml_tq1_g128_pack_codes() followed by the type's
//     to_float gives exactly scale * (code - 1) for every element, including
//     each 128-group's own scale. Checked with random codes, not just the
//     codes a real checkpoint happens to contain.
//  2. The CPU vec_dot (the kernel ggml_mul_mat uses) equals an integer
//     reference on the same Q8_K activations: sum((code - 1) * q) is integral,
//     so anything beyond float rounding of the final scale products is a bug.
//  3. ggml_mul_mat on every registered backend (CPU, CUDA, Vulkan, ...) agrees
//     with a float reference built from the dequantized weights, for decode-
//     sized (1 column) and encoder-sized (many columns) inputs.

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" void ggml_tq1_g128_pack_codes(const uint8_t * codes, const ggml_fp16_t * scales, void * y, int64_t k);

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

struct Ternary {
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<uint8_t>     codes;   // rows * cols, values 0..2
    std::vector<ggml_fp16_t> scales;  // rows * cols / 128
    std::vector<uint8_t>     packed;  // ggml_row_size(TQ1_G128, cols) * rows
};

Ternary make_ternary(int64_t rows, int64_t cols, std::mt19937 & rng) {
    Ternary t;
    t.rows = rows;
    t.cols = cols;
    t.codes.resize(rows * cols);
    t.scales.resize(rows * cols / 128);
    std::uniform_int_distribution<int> code(0, 2);
    std::uniform_real_distribution<float> scale(1e-4f, 0.5f);
    for (auto & c : t.codes) {
        c = (uint8_t) code(rng);
    }
    for (auto & s : t.scales) {
        s = ggml_fp32_to_fp16(scale(rng));
    }
    const size_t row_size = ggml_row_size(GGML_TYPE_TQ1_G128, cols);
    t.packed.resize(row_size * rows);
    for (int64_t r = 0; r < rows; ++r) {
        ggml_tq1_g128_pack_codes(t.codes.data() + r * cols, t.scales.data() + r * cols / 128,
                                 t.packed.data() + r * row_size, cols);
    }
    return t;
}

std::vector<float> dequantize(const Ternary & t) {
    const auto * traits = ggml_get_type_traits(GGML_TYPE_TQ1_G128);
    const size_t row_size = ggml_row_size(GGML_TYPE_TQ1_G128, t.cols);
    std::vector<float> w(t.rows * t.cols);
    for (int64_t r = 0; r < t.rows; ++r) {
        traits->to_float(t.packed.data() + r * row_size, w.data() + r * t.cols, t.cols);
    }
    return w;
}

void test_pack_roundtrip(std::mt19937 & rng) {
    const Ternary t = make_ternary(7, 1024, rng);
    const std::vector<float> w = dequantize(t);
    int64_t bad = 0;
    for (int64_t i = 0; i < t.rows * t.cols; ++i) {
        const float want = ggml_fp16_to_fp32(t.scales[i / 128]) * (float) ((int) t.codes[i] - 1);
        bad += (w[i] != want);
    }
    check(bad == 0, "pack/dequant round trip is exact (" + std::to_string(bad) + " mismatches)");
    check(ggml_row_size(GGML_TYPE_TQ1_G128, 1024) == 4 * 56, "row size is 56 bytes per 256 weights (1.75 bpw)");
}

void test_vec_dot(std::mt19937 & rng) {
    const int64_t n = 4096;
    const Ternary t = make_ternary(3, n, rng);
    const auto * cpu = ggml_get_type_traits_cpu(GGML_TYPE_TQ1_G128);
    check(cpu->vec_dot_type == GGML_TYPE_Q8_K, "vec_dot_type is Q8_K");

    std::normal_distribution<float> act(0.0f, 1.0f);
    std::vector<float> x(n);
    for (auto & v : x) {
        v = act(rng);
    }
    // Quantize activations once and read the int8 values + scale back out of
    // the Q8_K blocks through the type's own to_float.
    std::vector<uint8_t> xq(ggml_row_size(GGML_TYPE_Q8_K, n));
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), xq.data(), n);
    // Q8_K has no to_float; decode its blocks (ggml-common.h block_q8_K).
    struct q8k {
        float   d;
        int8_t  qs[256];
        int16_t bsums[16];
    };
    static_assert(sizeof(q8k) == 292, "block_q8_K layout");
    check(ggml_row_size(GGML_TYPE_Q8_K, 256) == sizeof(q8k), "block_q8_K size");
    std::vector<float> xd(n);
    for (int64_t b = 0; b < n / 256; ++b) {
        q8k blk;
        std::memcpy(&blk, xq.data() + b * sizeof(q8k), sizeof(q8k));
        for (int j = 0; j < 256; ++j) {
            xd[b * 256 + j] = blk.d * blk.qs[j];
        }
    }

    const size_t row_size = ggml_row_size(GGML_TYPE_TQ1_G128, n);
    for (int64_t r = 0; r < t.rows; ++r) {
        float got = 0.0f;
        cpu->vec_dot((int) n, &got, 0, t.packed.data() + r * row_size, 0, xq.data(), 0, 1);
        double want = 0.0;
        for (int64_t c = 0; c < n; ++c) {
            want += (double) ggml_fp16_to_fp32(t.scales[(r * n + c) / 128]) * ((int) t.codes[r * n + c] - 1) * xd[c];
        }
        const double err = std::fabs(got - want) / (std::fabs(want) + 1e-3);
        check(err < 1e-4, "vec_dot row " + std::to_string(r) + ": got " + std::to_string(got) + " want " +
                              std::to_string(want));
    }
}

// W [cols x rows] (ggml ne0 = cols) times X [cols x n_tok] on one backend.
std::vector<float> run_mul_mat(ggml_backend_t backend, const Ternary & t, const std::vector<float> & x, int64_t n_tok) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ1_G128, t.cols, t.rows);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t.cols, n_tok);
    ggml_tensor * y = ggml_mul_mat(ctx, w, a);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(w, t.packed.data(), 0, t.packed.size());
    ggml_backend_tensor_set(a, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> out(t.rows * n_tok);
    ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

void test_backends(std::mt19937 & rng) {
    const Ternary t = make_ternary(96, 1024, rng);
    const std::vector<float> w = dequantize(t);
    std::normal_distribution<float> act(0.0f, 1.0f);

    for (size_t d = 0; d < ggml_backend_dev_count(); ++d) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        const std::string name = ggml_backend_dev_name(dev);
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            continue;
        }
        for (int64_t n_tok : {1, 3, 64, 138}) {
            std::vector<float> x(t.cols * n_tok);
            for (auto & v : x) {
                v = act(rng);
            }
            // Probe support first so an unsupported backend reports instead of aborting.
            ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
            ggml_context * pctx = ggml_init(ip);
            ggml_tensor * pw = ggml_new_tensor_2d(pctx, GGML_TYPE_TQ1_G128, t.cols, t.rows);
            ggml_tensor * pa = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, t.cols, n_tok);
            const bool supported = ggml_backend_dev_supports_op(dev, ggml_mul_mat(pctx, pw, pa));
            ggml_free(pctx);
            check(supported, name + ": supports MUL_MAT with TQ1_G128 (n_tok=" + std::to_string(n_tok) + ")");
            if (!supported) {
                continue;
            }
            const std::vector<float> got = run_mul_mat(backend, t, x, n_tok);
            // Reference: dequantized weights in double. Backends quantize the
            // activations (Q8_K / Q8_1) or use f16 GEMMs, so compare with a
            // relative tolerance on the row-vector norm.
            double max_rel = 0.0;
            for (int64_t j = 0; j < n_tok; ++j) {
                for (int64_t r = 0; r < t.rows; ++r) {
                    double ref = 0.0, norm = 0.0;
                    for (int64_t c = 0; c < t.cols; ++c) {
                        const double p = (double) w[r * t.cols + c] * x[j * t.cols + c];
                        ref += p;
                        norm += std::fabs(p);
                    }
                    max_rel = std::max(max_rel, std::fabs(got[j * t.rows + r] - ref) / (norm + 1e-6));
                }
            }
            std::printf("  %-12s n_tok=%-4lld max rel err %.3e\n", name.c_str(), (long long) n_tok, max_rel);
            check(max_rel < 2e-2, name + ": mul_mat matches reference (n_tok=" + std::to_string(n_tok) + ")");
        }
        ggml_backend_free(backend);
    }
}

}  // namespace

int main() {
    ggml_backend_load_all();
    ggml_cpu_init();
    std::mt19937 rng(1234);
    std::fprintf(stderr, "[pack]\n");
    test_pack_roundtrip(rng);
    std::fprintf(stderr, "[vec_dot]\n");
    test_vec_dot(rng);
    std::fprintf(stderr, "[backends]\n");
    test_backends(rng);
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("ternary_tq1_g128_unit: all checks passed\n");
    return 0;
}
