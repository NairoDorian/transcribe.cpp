// pack_gate_up_unit.cpp - unit test for causal_lm::pack_gate_up.
//
// The packing step has three behaviours and no fixture pins them:
//
//   - alias:    gate and up already sit back to back in one backend buffer, so
//               the fused tensor points at gate's bytes and nothing is
//               duplicated (PackedGateUpHandles::buffer stays null, and the
//               weights buffer is what the fused tensor lives in).
//   - copy:     anything else (a gap between gate and up, non-adjacent
//               buffers) is copied into a buffer owned by the handles.
//   - mismatch: gate and up do not add up to the fused shape -> TRANSCRIBE_ERR_GGUF,
//               reported the same way whichever path the block would have taken.
//
// It also covers the mixed case, where some blocks alias and others copy in
// the same call, because the two storage kinds have to coexist in one call and
// the handles must free only what they allocated.
//
// Tensors are built by hand on the CPU backend rather than through a GGUF:
// adjacency is a property of how tensors land in a buffer, which a synthetic
// catalog controls exactly, and no model is needed.

#include "causal_lm/causal_lm.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Bail macro for setup prerequisites: log and return rather than crash on a
// null deref.
#define REQUIRE(cond)                                                                  \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "SETUP FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            return EXIT_FAILURE;                                                       \
        }                                                                              \
    } while (0)

constexpr int k_hidden       = 4;
constexpr int k_intermediate = 8;

// setenv/unsetenv do not exist on MSVC (same helper shape as
// weights_mmap_smoke). A null value unsets.
void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct Pair {
    ggml_tensor * gate  = nullptr;
    ggml_tensor * up    = nullptr;
    ggml_tensor * fused = nullptr;
};

// Fill a [hidden, intermediate] F32 tensor with a value derived from `seed` and
// the element index, so a wrong source tensor (or a shifted copy) is detected
// by the byte comparison rather than by luck.
void fill(ggml_tensor * t, int seed) {
    std::vector<float> data(static_cast<size_t>(ggml_nelements(t)));
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(seed) * 1000.0f + static_cast<float>(i);
    }
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

std::vector<uint8_t> bytes_of(const ggml_tensor * t) {
    std::vector<uint8_t> out(ggml_nbytes(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size());
    return out;
}

// The fused tensor must be gate's bytes followed by up's, whichever storage it
// ended up with.
void check_fused_bytes(const Pair & p) {
    const std::vector<uint8_t> gate = bytes_of(p.gate);
    const std::vector<uint8_t> up   = bytes_of(p.up);
    std::vector<uint8_t>       want = gate;
    want.insert(want.end(), up.begin(), up.end());
    const std::vector<uint8_t> got = bytes_of(p.fused);
    CHECK(want.size() == got.size());
    CHECK(want.size() == got.size() && std::memcmp(want.data(), got.data(), got.size()) == 0);
    CHECK(ggml_nbytes(p.fused) == ggml_nbytes(p.gate) + ggml_nbytes(p.up));
}

}  // namespace

int main() {
    using transcribe::causal_lm::GateUpEntry;
    using transcribe::causal_lm::pack_gate_up;
    using transcribe::causal_lm::PackedGateUpHandles;

    ggml_backend_t cpu_be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu_be == nullptr) {
        std::fprintf(stderr, "SKIP: could not init CPU backend\n");
        return 77;
    }

    // ---------------------------------------------------------------
    // Case 1: adjacent gate/up in one buffer -> aliased, no new buffer.
    // ---------------------------------------------------------------
    {
        ggml_init_params params = { 2 * ggml_tensor_overhead() + 256, nullptr, true };
        ggml_context *   ctx    = ggml_init(params);
        REQUIRE(ctx != nullptr);

        Pair p;
        p.gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(p.gate != nullptr);
        p.up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(p.up != nullptr);

        ggml_backend_buffer_t weights = ggml_backend_alloc_ctx_tensors(ctx, cpu_be);
        REQUIRE(weights != nullptr);
        fill(p.gate, 1);
        fill(p.up, 2);

        // alloc_ctx_tensors lays tensors out in catalog order, so up starts
        // where gate ends.
        REQUIRE(static_cast<const char *>(p.gate->data) + ggml_nbytes(p.gate) == p.up->data);

        const std::vector<uint8_t> gate_bytes_before = bytes_of(p.gate);
        const std::vector<uint8_t> up_bytes_before   = bytes_of(p.up);

        std::vector<GateUpEntry> entries = {
            { p.gate, p.up, &p.fused }
        };
        PackedGateUpHandles handles;

        set_env("TRANSCRIBE_NO_GATE_UP_ALIAS", nullptr);
        const transcribe_status st = pack_gate_up(cpu_be, k_hidden, k_intermediate, entries, handles, "unit");
        CHECK(st == TRANSCRIBE_OK);
        CHECK(p.fused != nullptr);
        CHECK(handles.buffer == nullptr);
        if (p.fused != nullptr) {
            CHECK(p.fused->data == p.gate->data);
            CHECK(p.fused->buffer == p.gate->buffer);
            check_fused_bytes(p);
        }

        // free() must not free the weights buffer the tensors point into: the
        // aliased path owns nothing. Everything the handles owned (the packed
        // ctx, holding `fused`'s metadata) is gone, so only the weights-side
        // pointers may be touched from here on.
        handles.free();
        CHECK(bytes_of(p.gate) == gate_bytes_before);
        CHECK(bytes_of(p.up) == up_bytes_before);

        ggml_backend_buffer_free(weights);
        ggml_free(ctx);
    }

    // ---------------------------------------------------------------
    // Case 2: a gap between gate and up -> copied.
    // ---------------------------------------------------------------
    {
        ggml_init_params params = { 3 * ggml_tensor_overhead() + 256, nullptr, true };
        ggml_context *   ctx    = ggml_init(params);
        REQUIRE(ctx != nullptr);

        Pair p;
        p.gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(p.gate != nullptr);
        // Sits between the two in catalog order, so up does not follow gate.
        ggml_tensor * filler = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(filler != nullptr);
        p.up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(p.up != nullptr);

        ggml_backend_buffer_t weights = ggml_backend_alloc_ctx_tensors(ctx, cpu_be);
        REQUIRE(weights != nullptr);
        fill(p.gate, 3);
        fill(p.up, 4);

        REQUIRE(static_cast<const char *>(p.gate->data) + ggml_nbytes(p.gate) != p.up->data);

        std::vector<GateUpEntry> entries = {
            { p.gate, p.up, &p.fused }
        };
        PackedGateUpHandles handles;

        set_env("TRANSCRIBE_NO_GATE_UP_ALIAS", nullptr);
        const transcribe_status st = pack_gate_up(cpu_be, k_hidden, k_intermediate, entries, handles, "unit");
        CHECK(st == TRANSCRIBE_OK);
        CHECK(p.fused != nullptr);
        CHECK(handles.buffer != nullptr);
        if (p.fused != nullptr) {
            CHECK(p.fused->data != p.gate->data);
            check_fused_bytes(p);
        }

        handles.free();
        ggml_backend_buffer_free(weights);
        ggml_free(ctx);
    }

    // ---------------------------------------------------------------
    // Case 3: mixed - one adjacent block and one gapped block in one call.
    // ---------------------------------------------------------------
    {
        ggml_init_params params = { 6 * ggml_tensor_overhead() + 256, nullptr, true };
        ggml_context *   ctx    = ggml_init(params);
        REQUIRE(ctx != nullptr);

        // Catalog order: gate0, up0, gate1, filler, up1.
        Pair a, b;
        a.gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(a.gate != nullptr);
        a.up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(a.up != nullptr);
        b.gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(b.gate != nullptr);
        ggml_tensor * filler = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(filler != nullptr);
        b.up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(b.up != nullptr);

        ggml_backend_buffer_t weights = ggml_backend_alloc_ctx_tensors(ctx, cpu_be);
        REQUIRE(weights != nullptr);
        fill(a.gate, 5);
        fill(a.up, 6);
        fill(b.gate, 7);
        fill(b.up, 8);

        REQUIRE(static_cast<const char *>(a.gate->data) + ggml_nbytes(a.gate) == a.up->data);
        REQUIRE(static_cast<const char *>(b.gate->data) + ggml_nbytes(b.gate) != b.up->data);

        const std::vector<uint8_t> gate_before   = bytes_of(a.gate);
        const std::vector<uint8_t> b_gate_before = bytes_of(b.gate);

        std::vector<GateUpEntry> entries = {
            { a.gate, a.up, &a.fused },
            { b.gate, b.up, &b.fused }
        };
        PackedGateUpHandles handles;

        set_env("TRANSCRIBE_NO_GATE_UP_ALIAS", nullptr);
        const transcribe_status st = pack_gate_up(cpu_be, k_hidden, k_intermediate, entries, handles, "unit");
        CHECK(st == TRANSCRIBE_OK);
        CHECK(a.fused != nullptr);
        CHECK(b.fused != nullptr);
        // The buffer is allocated for the copied block only.
        CHECK(handles.buffer != nullptr);
        if (a.fused != nullptr && b.fused != nullptr) {
            CHECK(a.fused->data == a.gate->data);
            CHECK(a.fused->buffer == a.gate->buffer);
            CHECK(b.fused->data != b.gate->data);
            CHECK(b.fused->buffer == handles.buffer);
            check_fused_bytes(a);
            check_fused_bytes(b);
        }

        handles.free();
        // The aliased block's bytes were never ours, so they survive the handles;
        // the copied block's buffer went with them. Only the weights-side
        // pointers are legal to touch now (the packed ctx held `a.fused`).
        CHECK(bytes_of(a.gate) == gate_before);
        CHECK(bytes_of(b.gate) == b_gate_before);
        ggml_backend_buffer_free(weights);
        ggml_free(ctx);
    }

    // ---------------------------------------------------------------
    // Case 4: TRANSCRIBE_NO_GATE_UP_ALIAS forces the copy path.
    // ---------------------------------------------------------------
    {
        ggml_init_params params = { 2 * ggml_tensor_overhead() + 256, nullptr, true };
        ggml_context *   ctx    = ggml_init(params);
        REQUIRE(ctx != nullptr);

        Pair p;
        p.gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(p.gate != nullptr);
        p.up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(p.up != nullptr);

        ggml_backend_buffer_t weights = ggml_backend_alloc_ctx_tensors(ctx, cpu_be);
        REQUIRE(weights != nullptr);
        fill(p.gate, 9);
        fill(p.up, 10);

        std::vector<GateUpEntry> entries = {
            { p.gate, p.up, &p.fused }
        };
        PackedGateUpHandles handles;

        set_env("TRANSCRIBE_NO_GATE_UP_ALIAS", "1");
        const transcribe_status st = pack_gate_up(cpu_be, k_hidden, k_intermediate, entries, handles, "unit");
        set_env("TRANSCRIBE_NO_GATE_UP_ALIAS", nullptr);
        CHECK(st == TRANSCRIBE_OK);
        CHECK(handles.buffer != nullptr);
        if (p.fused != nullptr) {
            CHECK(p.fused->data != p.gate->data);
            check_fused_bytes(p);
        }

        handles.free();
        ggml_backend_buffer_free(weights);
        ggml_free(ctx);
    }

    // ---------------------------------------------------------------
    // Case 5: gate+up do not add up to the fused shape -> error, not a
    // silently short tensor (the aliased path would read past up otherwise).
    // ---------------------------------------------------------------
    {
        ggml_init_params params = { 2 * ggml_tensor_overhead() + 256, nullptr, true };
        ggml_context *   ctx    = ggml_init(params);
        REQUIRE(ctx != nullptr);

        Pair p;
        // gate/up are [hidden, intermediate], but the caller claims a fused
        // tensor of 2 * (intermediate + 1) rows.
        p.gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(p.gate != nullptr);
        p.up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_hidden, k_intermediate);
        REQUIRE(p.up != nullptr);

        ggml_backend_buffer_t weights = ggml_backend_alloc_ctx_tensors(ctx, cpu_be);
        REQUIRE(weights != nullptr);
        fill(p.gate, 11);
        fill(p.up, 12);

        std::vector<GateUpEntry> entries = {
            { p.gate, p.up, &p.fused }
        };
        PackedGateUpHandles handles;

        set_env("TRANSCRIBE_NO_GATE_UP_ALIAS", nullptr);
        const transcribe_status st = pack_gate_up(cpu_be, k_hidden, k_intermediate + 1, entries, handles, "unit");
        CHECK(st == TRANSCRIBE_ERR_GGUF);

        handles.free();
        ggml_backend_buffer_free(weights);
        ggml_free(ctx);
    }

    ggml_backend_free(cpu_be);

    if (g_failures > 0) {
        std::fprintf(stderr, "pack_gate_up_unit: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "pack_gate_up_unit: ok\n");
    return EXIT_SUCCESS;
}
