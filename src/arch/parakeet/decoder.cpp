// arch/parakeet/decoder.cpp - Parakeet TDT decoder implementation.
//
// See decoder.h for the API contract. The predictor LSTM, encoder
// projection, and joint network all run as ggml backend graphs on a
// single shared CPU backend + persistent threadpool (owned by PredGraph).
// Decode weights are uploaded to resident ggml tensors at load and the
// host fp32 mirrors then freed.
//
// Numerical note: ggml's reduction order differs slightly from a naive
// host loop, so per-step values drift ~1e-7 vs the reference; over an
// utterance this stays within ~1e-4 and the greedy transcript is
// unchanged.

#include "decoder.h"

#include "parakeet.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-env.h"
#include "transcribe-log.h"
#include "weights.h"

#include <limits>

// ggml-backend.h, not ggml-cpu.h: under GGML_BACKEND_DL the CPU backend
// is a loadable module, so backend-specific entry points must be reached
// through the registry (ggml_backend_init_by_type / get_proc_address),
// never by direct link-time reference.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"  // ggml_threadpool_params_default (rest via registry)
#include "ggml.h"

#include <thread>

#if TRANSCRIBE_HAS_BLAS
#    ifdef __APPLE__
#        include <Accelerate/Accelerate.h>
#    else
#        include <cblas.h>
#    endif
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace transcribe::parakeet {

int argmax_range_masked(const float * data, int n, const std::vector<uint8_t> * blocked);

// ---------------------------------------------------------------------------
// Host weight extraction
// ---------------------------------------------------------------------------

namespace {

// Read all elements of a ggml_tensor into a host fp32 vector,
// dequantizing via ggml_get_type_traits()->to_float. Source dtype
// support is whatever ggml registers a to_float for (F32, F16, BF16, all
// Q*/IQ*). ggml_backend_tensor_get is universal (memcpy on host buffers,
// readback on discrete GPUs); we stage the raw bytes then materialize
// fp32 in `out`. Returns false on missing-trait / size errors.
bool read_tensor_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (t == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: read_tensor_to_f32: null tensor");
        return false;
    }
    const size_t nbytes = ggml_nbytes(t);
    if (nbytes == 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: tensor \"%s\" has 0 bytes", t->name);
        return false;
    }
    const int64_t nelem = ggml_nelements(t);
    if (nelem <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: tensor \"%s\" has nelem=%lld", t->name,
                static_cast<long long>(nelem));
        return false;
    }

    out.resize(static_cast<size_t>(nelem));

    // F32 fast path. type_traits[GGML_TYPE_F32] has no to_float (identity
    // is left null upstream), so short-circuit rather than dispatch.
    if (t->type == GGML_TYPE_F32) {
        if (nbytes != static_cast<size_t>(nelem) * sizeof(float)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "parakeet decoder: tensor \"%s\" f32 nbytes %zu "
                    "!= nelem*sizeof(float) %zu",
                    t->name, nbytes, static_cast<size_t>(nelem) * sizeof(float));
            return false;
        }
        ggml_backend_tensor_get(t, out.data(), 0, nbytes);
        return true;
    }

    // Quantized / non-fp32 path: stage the raw bytes off the backend,
    // then walk the type's to_float to materialize fp32.
    const auto * tt = ggml_get_type_traits(t->type);
    if (tt == nullptr || tt->to_float == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: tensor \"%s\" type %s has no to_float", t->name,
                ggml_type_name(t->type));
        return false;
    }

    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    tt->to_float(raw.data(), out.data(), nelem);
    return true;
}

// Make the joint network's weights resident as fp32 ggml tensors on the
// model: enc / pred / out projections. Built once at load. out_w is
// dequantized from the model tensor src_out_w; the rest come from the
// host mirrors, freed here once uploaded. On failure frees partial state
// and returns false (w_ready stays false → hard decode error).
bool build_joint_weight(HostJoint & j, const ggml_tensor * src_out_w) {
    const int joint_h = j.joint_h;
    const int joint_n = j.joint_n;

    auto fail = [&]() -> bool {
        if (j.w_buf != nullptr) {
            safe_buffer_free(j.w_buf);
            j.w_buf = nullptr;
        }
        if (j.w_ctx != nullptr) {
            ggml_free(j.w_ctx);
            j.w_ctx = nullptr;
        }
        if (j.w_backend != nullptr) {
            safe_backend_free(j.w_backend);
            j.w_backend = nullptr;
        }
        j.g_enc_w  = nullptr;
        j.g_enc_b  = nullptr;
        j.g_pred_w = nullptr;
        j.g_pred_b = nullptr;
        j.gw_w     = nullptr;
        j.gw_b     = nullptr;
        j.w_ready  = false;
        return false;
    };

    // CPU backend used ONLY to allocate the resident weight buffer —
    // never graph_compute'd, so it carries no per-step state.
    j.w_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (j.w_backend == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml CPU backend init failed");
        return fail();
    }

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 8;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    j.w_ctx       = ggml_init(ip);
    if (j.w_ctx == nullptr) {
        return fail();
    }

    j.g_enc_w  = ggml_new_tensor_2d(j.w_ctx, GGML_TYPE_F32, j.d_enc, joint_h);
    j.g_enc_b  = ggml_new_tensor_1d(j.w_ctx, GGML_TYPE_F32, joint_h);
    j.g_pred_w = ggml_new_tensor_2d(j.w_ctx, GGML_TYPE_F32, j.pred_hidden, joint_h);
    j.g_pred_b = ggml_new_tensor_1d(j.w_ctx, GGML_TYPE_F32, joint_h);
    j.gw_w     = ggml_new_tensor_2d(j.w_ctx, GGML_TYPE_F32, joint_h, joint_n);
    j.gw_b     = ggml_new_tensor_1d(j.w_ctx, GGML_TYPE_F32, joint_n);

    j.w_buf = ggml_backend_alloc_ctx_tensors(j.w_ctx, j.w_backend);
    if (j.w_buf == nullptr) {
        return fail();
    }

    // out_w: dequantize the model tensor to fp32 once.
    {
        std::vector<float> tmp;
        if (!read_tensor_to_f32(src_out_w, tmp)) {
            return fail();
        }
        ggml_backend_tensor_set(j.gw_w, tmp.data(), 0, tmp.size() * sizeof(float));
    }
    // The rest come from the host mirrors.
    ggml_backend_tensor_set(j.g_enc_w, j.enc_w.data(), 0, j.enc_w.size() * sizeof(float));
    ggml_backend_tensor_set(j.g_enc_b, j.enc_b.data(), 0, j.enc_b.size() * sizeof(float));
    ggml_backend_tensor_set(j.g_pred_w, j.pred_w.data(), 0, j.pred_w.size() * sizeof(float));
    ggml_backend_tensor_set(j.g_pred_b, j.pred_b.data(), 0, j.pred_b.size() * sizeof(float));
    ggml_backend_tensor_set(j.gw_b, j.out_b.data(), 0, j.out_b.size() * sizeof(float));

    // Host mirrors are now resident in ggml — release them (load-time scratch).
    std::vector<float>().swap(j.enc_w);
    std::vector<float>().swap(j.enc_b);
    std::vector<float>().swap(j.pred_w);
    std::vector<float>().swap(j.pred_b);
    std::vector<float>().swap(j.out_b);

    j.w_ready = true;
    return true;
}

// Make the predictor LSTM weights resident as fp32 ggml tensors for the
// per-call PredGraph. Built once at load. ne is [pred_hidden, 4*pred_hidden]
// for Wx/Wh (row-major [4*H, H] host bytes as a mul_mat operand) and
// [4*pred_hidden] for the bias. On failure frees partial state and returns
// false (lstm_ready stays false → hard decode error).
bool build_pred_weights(HostPredictor & p) {
    const int H      = p.pred_hidden;
    const int four_H = 4 * H;
    const int L      = static_cast<int>(p.lstm.size());
    if (L == 0 || H == 0) {
        return false;
    }

    auto fail = [&]() -> bool {
        if (p.lstm_w_buf != nullptr) {
            safe_buffer_free(p.lstm_w_buf);
            p.lstm_w_buf = nullptr;
        }
        if (p.lstm_w_ctx != nullptr) {
            ggml_free(p.lstm_w_ctx);
            p.lstm_w_ctx = nullptr;
        }
        if (p.lstm_w_backend != nullptr) {
            safe_backend_free(p.lstm_w_backend);
            p.lstm_w_backend = nullptr;
        }
        for (auto & lh : p.lstm) {
            lh.g_Wx = nullptr;
            lh.g_Wh = nullptr;
            lh.g_b  = nullptr;
        }
        p.lstm_ready = false;
        return false;
    };

    // Alloc-only backend (never graph_compute'd), mirroring HostJoint::w_backend.
    p.lstm_w_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (p.lstm_w_backend == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml CPU backend init failed (pred)");
        return fail();
    }

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * static_cast<size_t>(L * 3 + 1);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    p.lstm_w_ctx  = ggml_init(ip);
    if (p.lstm_w_ctx == nullptr) {
        return fail();
    }

    for (int l = 0; l < L; ++l) {
        auto & lh = p.lstm[l];
        lh.g_Wx   = ggml_new_tensor_2d(p.lstm_w_ctx, GGML_TYPE_F32, H, four_H);
        lh.g_Wh   = ggml_new_tensor_2d(p.lstm_w_ctx, GGML_TYPE_F32, H, four_H);
        lh.g_b    = ggml_new_tensor_1d(p.lstm_w_ctx, GGML_TYPE_F32, four_H);
    }

    p.lstm_w_buf = ggml_backend_alloc_ctx_tensors(p.lstm_w_ctx, p.lstm_w_backend);
    if (p.lstm_w_buf == nullptr) {
        return fail();
    }

    for (int l = 0; l < L; ++l) {
        auto & lh = p.lstm[l];
        ggml_backend_tensor_set(lh.g_Wx, lh.Wx.data(), 0, lh.Wx.size() * sizeof(float));
        ggml_backend_tensor_set(lh.g_Wh, lh.Wh.data(), 0, lh.Wh.size() * sizeof(float));
        ggml_backend_tensor_set(lh.g_b, lh.b.data(), 0, lh.b.size() * sizeof(float));
        // Host mirrors now resident in ggml — release them.
        std::vector<float>().swap(lh.Wx);
        std::vector<float>().swap(lh.Wh);
        std::vector<float>().swap(lh.b);
    }

    p.lstm_ready = true;
    return true;
}

}  // namespace

// Per-call joint compute graph (the mutable half of the joint network):
//   pred_proj = pred_w @ pred_in + pred_b      [joint_h]
//   summed    = enc_proj + pred_proj           [joint_h]
//   activated = activation(summed)             [joint_h]
//   logits    = out_w @ activated + out_b      [joint_n]
// Built on a BORROWED backend (PredGraph's shared pool) so the joint runs
// on the same threadpool as the pred LSTM and enc_proj. Inputs are the
// predictor output and this frame's precomputed enc projection; output is
// the logits. Built fresh per decode call (reentrant); weights are the
// shared resident HostJoint tensors.
namespace {

struct JointGraph {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;  // BORROWED (PredGraph's); NOT freed here
    ggml_backend_buffer_t buf     = nullptr;
    ggml_cgraph *         graph   = nullptr;
    ggml_tensor *         pred_in = nullptr;  // [pred_hidden] fp32 input (decoder out)
    ggml_tensor *         enc_in  = nullptr;  // [joint_h] fp32 input (enc_proj frame)
    ggml_tensor *         logits  = nullptr;  // [joint_n] fp32 output
    bool                  ready   = false;

    JointGraph() = default;

    ~JointGraph() {
        if (buf != nullptr) {
            safe_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        // backend is borrowed from PredGraph — do NOT free here.
    }

    JointGraph(const JointGraph &)             = delete;
    JointGraph & operator=(const JointGraph &) = delete;
};

// Build the full joint graph on the shared `backend` (PredGraph's pool).
// Returns false if the resident weights are absent or any ggml step
// fails (→ hard decode error).
bool build_joint_graph(JointGraph & g, const HostJoint & j, ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    if (!j.w_ready || j.gw_w == nullptr || j.gw_b == nullptr || j.g_pred_w == nullptr || j.g_pred_b == nullptr) {
        return false;
    }

    g.backend = backend;  // borrowed

    auto fail = [&]() -> bool {
        if (g.buf != nullptr) {
            safe_buffer_free(g.buf);
            g.buf = nullptr;
        }
        if (g.ctx != nullptr) {
            ggml_free(g.ctx);
            g.ctx = nullptr;
        }
        g.pred_in = nullptr;
        g.enc_in  = nullptr;
        g.logits  = nullptr;
        g.graph   = nullptr;
        g.ready   = false;
        return false;
    };

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx         = ggml_init(ip);
    if (g.ctx == nullptr) {
        return fail();
    }

    g.pred_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, j.pred_hidden);
    ggml_set_input(g.pred_in);
    g.enc_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, j.joint_h);
    ggml_set_input(g.enc_in);

    // pred_proj = pred_w @ pred_in + pred_b   [joint_h]
    ggml_tensor * pred_proj = ggml_add(g.ctx, ggml_mul_mat(g.ctx, j.g_pred_w, g.pred_in), j.g_pred_b);
    // summed = enc_proj + pred_proj           [joint_h]
    ggml_tensor * summed    = ggml_add(g.ctx, g.enc_in, pred_proj);
    // activation — loader allow-list guarantees exactly one of relu/sigmoid/tanh.
    ggml_tensor * activated;
    if (j.activation == "relu") {
        activated = ggml_relu(g.ctx, summed);
    } else if (j.activation == "sigmoid") {
        activated = ggml_sigmoid(g.ctx, summed);
    } else {  // "tanh"
        activated = ggml_tanh(g.ctx, summed);
    }
    // logits = out_w @ activated + out_b      [joint_n]
    ggml_tensor * mm = ggml_mul_mat(g.ctx, j.gw_w, activated);
    g.logits         = ggml_add(g.ctx, mm, j.gw_b);
    ggml_set_output(g.logits);

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, g.backend);
    if (g.buf == nullptr) {
        return fail();
    }

    g.graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.graph, g.logits);
    g.ready = true;
    return true;
}

// ---------------------------------------------------------------------------
// Batched joint graph (frame-windowed greedy decode)
// ---------------------------------------------------------------------------
//
// The greedy loops evaluate joint(enc_proj[frame], pred_state) once per
// frame, and the predictor state only changes when a non-blank token is
// emitted. Between emissions the pred side is a constant, so the joint for a
// window of W consecutive frames collapses into ONE graph dispatch:
//
//   pred_proj = pred_w @ pred_in + pred_b            [joint_h]      (once)
//   summed    = enc_in + pred_proj                   [joint_h, W]   (broadcast)
//   act       = relu/sigmoid/tanh(summed)
//   logits    = out_w @ act + out_b                  [joint_n, W]
//
// vs W separate dispatches of the same math. This amortizes the per-call
// graph dispatch + threadpool wake + readback overhead (~0.5 ms per frame
// measured on an i9-13900H, x425 frames = the bulk of nemotron-3.5's
// 200-290 ms joint time; W=16 lands at 60-90 ms). Columns after the first
// non-blank are discarded and recomputed with the new state, which
// reproduces the serial decode DECISIONS exactly. Numerics: the W-column
// mul_mat may take a different kernel than the n=1 GEMV (GGML_LLAMAFILE
// tinyBLAS), shifting logits by accumulation order — observed <= 1.5e-5 on
// nemotron (argmax stable; well under the family's ~1e-4 accuracy envelope
// vs the NeMo reference, see the file header) and bit-identical on
// parakeet-v3's joint shape. TRANSCRIBE_RNNT_BATCH_CHECK=1 re-runs the
// serial joint per consumed column and compares the two (see
// joint_batch_check: an argmax flip is an error, drift alone is DEBUG).
// Enabled by default only for the RNN-T loop; the TDT loop
// defaults to serial (its duration head skips frames, so windows are
// mostly wasted — measured slower). TRANSCRIBE_RNNT_BATCH_W overrides.
struct JointGraphBatch {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;  // BORROWED (PredGraph's); NOT freed here
    ggml_backend_buffer_t buf     = nullptr;
    ggml_cgraph *         graph   = nullptr;
    ggml_tensor *         pred_in = nullptr;  // [pred_hidden] fp32 input (decoder out)
    ggml_tensor *         enc_in  = nullptr;  // [joint_h, W] fp32 input (enc_proj window)
    ggml_tensor *         logits  = nullptr;  // [joint_n, W] fp32 output
    int                   W       = 0;
    bool                  ready   = false;

    JointGraphBatch() = default;

    ~JointGraphBatch() {
        if (buf != nullptr) {
            safe_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        // backend is borrowed from PredGraph — do NOT free here.
    }

    JointGraphBatch(const JointGraphBatch &)             = delete;
    JointGraphBatch & operator=(const JointGraphBatch &) = delete;
};

// Build the W-frame joint window graph on the shared `backend`. Mirrors
// build_joint_graph exactly except for the window axis.
bool build_joint_graph_batch(JointGraphBatch & g, const HostJoint & j, ggml_backend_t backend, int W) {
    if (backend == nullptr || W < 2) {
        return false;
    }
    if (!j.w_ready || j.gw_w == nullptr || j.gw_b == nullptr || j.g_pred_w == nullptr || j.g_pred_b == nullptr) {
        return false;
    }

    g.backend = backend;  // borrowed
    g.W       = W;

    auto fail = [&]() -> bool {
        if (g.buf != nullptr) {
            safe_buffer_free(g.buf);
            g.buf = nullptr;
        }
        if (g.ctx != nullptr) {
            ggml_free(g.ctx);
            g.ctx = nullptr;
        }
        g.pred_in = nullptr;
        g.enc_in  = nullptr;
        g.logits  = nullptr;
        g.graph   = nullptr;
        g.ready   = false;
        return false;
    };

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx         = ggml_init(ip);
    if (g.ctx == nullptr) {
        return fail();
    }

    g.pred_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, j.pred_hidden);
    ggml_set_input(g.pred_in);
    g.enc_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, j.joint_h, W);
    ggml_set_input(g.enc_in);

    // pred_proj = pred_w @ pred_in + pred_b   [joint_h]   (once per window)
    ggml_tensor * pred_proj = ggml_add(g.ctx, ggml_mul_mat(g.ctx, j.g_pred_w, g.pred_in), j.g_pred_b);
    // summed = enc window + pred_proj broadcast across columns  [joint_h, W]
    ggml_tensor * summed    = ggml_add(g.ctx, g.enc_in, pred_proj);
    ggml_tensor * activated;
    if (j.activation == "relu") {
        activated = ggml_relu(g.ctx, summed);
    } else if (j.activation == "sigmoid") {
        activated = ggml_sigmoid(g.ctx, summed);
    } else {  // "tanh"
        activated = ggml_tanh(g.ctx, summed);
    }
    // logits = out_w @ activated + out_b      [joint_n, W]
    ggml_tensor * mm = ggml_mul_mat(g.ctx, j.gw_w, activated);
    g.logits         = ggml_add(g.ctx, mm, j.gw_b);
    ggml_set_output(g.logits);

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, g.backend);
    if (g.buf == nullptr) {
        return fail();
    }

    g.graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.graph, g.logits);
    g.ready = true;
    return true;
}

// CPU-backend threadpool entry points, reached through the registry so
// the library stays DL-safe (under GGML_BACKEND_DL these symbols are not
// directly linkable). Resolved via ggml_backend_reg_get_proc_address.
typedef ggml_threadpool_t (*pfn_threadpool_new)(ggml_threadpool_params *);
typedef void (*pfn_threadpool_free)(ggml_threadpool_t);
typedef void (*pfn_set_threadpool)(ggml_backend_t, ggml_threadpool_t);

static void * cpu_backend_proc(ggml_backend_t backend, const char * name) {
    ggml_backend_dev_t dev = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
    ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg != nullptr ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

// Free a CPU-backend threadpool via the registry. Resolving the free fn needs
// the backend alive, so callers must invoke this BEFORE freeing the backend.
static void free_cpu_threadpool(ggml_backend_t backend, ggml_threadpool_t tp) {
    if (tp == nullptr || backend == nullptr) {
        return;
    }
    if (auto fn = (pfn_threadpool_free) cpu_backend_proc(backend, "ggml_threadpool_free")) {
        fn(tp);
    }
}

// Per-call predictor LSTM graph (the mutable half of the predictor):
// a single per-step graph built fresh per decode call around the
// model-resident HostPredictor::lstm weights, recomputed in place each
// step. Inputs are the embedding x and per-layer previous (h, c); outputs
// are the new (h, c) per layer. Reentrant: every concurrent decode owns
// its own PredGraph (backend + threadpool + I/O tensors).
struct PredGraph {
    ggml_context *             ctx     = nullptr;
    ggml_backend_t             backend = nullptr;
    ggml_backend_buffer_t      buf     = nullptr;
    ggml_cgraph *              graph   = nullptr;
    ggml_threadpool_t          tp      = nullptr;
    ggml_tensor *              x       = nullptr;  // [H] input embedding
    std::vector<ggml_tensor *> ph;                 // [H] prev hidden per layer (input)
    std::vector<ggml_tensor *> pc;                 // [H] prev cell   per layer (input)
    std::vector<ggml_tensor *> nh;                 // [H] new  hidden per layer (output)
    std::vector<ggml_tensor *> nc;                 // [H] new  cell   per layer (output)
    int                        H     = 0;
    int                        L     = 0;
    bool                       ready = false;

    PredGraph() = default;

    ~PredGraph() {
        if (buf != nullptr) {
            safe_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        free_cpu_threadpool(backend, tp);  // before safe_backend_free(backend)
        if (backend != nullptr) {
            safe_backend_free(backend);
        }
    }

    PredGraph(const PredGraph &)             = delete;
    PredGraph & operator=(const PredGraph &) = delete;
};

}  // namespace

struct ParakeetStreamingDecoderGraph {
    std::unique_ptr<PredGraph>  pg;
    std::unique_ptr<JointGraph> jg;
    int                         nt    = 0;
    bool                        ready = false;

    void reset() {
        jg.reset();
        pg.reset();
        ready = false;
        nt    = 0;
    }
};

ParakeetStreamingDecoderGraph * parakeet_streaming_decoder_graph_new() {
    return new ParakeetStreamingDecoderGraph();
}

void parakeet_streaming_decoder_graph_free(ParakeetStreamingDecoderGraph * g) {
    delete g;
}

namespace {

// Build the per-call LSTM graph around the resident p.lstm[*].g_Wx/g_Wh/g_b.
// Returns false if the resident weights are absent or any ggml step fails
// (→ hard decode error). n_threads is the resolved (>0) thread count.
bool build_pred_graph(PredGraph & g, const HostPredictor & p, int n_threads) {
    if (!p.lstm_ready) {
        return false;
    }
    const int H = p.pred_hidden;
    const int L = static_cast<int>(p.lstm.size());
    if (H == 0 || L == 0) {
        return false;
    }

    auto fail = [&]() -> bool {
        if (g.buf != nullptr) {
            safe_buffer_free(g.buf);
            g.buf = nullptr;
        }
        if (g.ctx != nullptr) {
            ggml_free(g.ctx);
            g.ctx = nullptr;
        }
        free_cpu_threadpool(g.backend, g.tp);
        g.tp = nullptr;  // before freeing backend
        if (g.backend != nullptr) {
            safe_backend_free(g.backend);
            g.backend = nullptr;
        }
        g.x     = nullptr;
        g.graph = nullptr;
        g.ready = false;
        g.ph.clear();
        g.pc.clear();
        g.nh.clear();
        g.nc.clear();
        return false;
    };

    g.H = H;
    g.L = L;

    g.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (g.backend == nullptr) {
        return fail();
    }
    if (ggml_backend_dev_t dev = ggml_backend_get_device(g.backend)) {
        auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads");
        if (set_n_threads != nullptr) {
            set_n_threads(g.backend, std::max(1, n_threads));
        }
    }
    // Persistent threadpool so the per-step graph_compute reuses workers instead
    // of spawning a transient pool each call. ggml's default hybrid-polling
    // (poll=50) keeps the workers hot between the sub-millisecond per-step
    // dispatches; poll=0 (park-immediately) was measured to regress pred to
    // 70-110 ms because parked workers are slow to reschedule.
    {
        auto tp_new = (pfn_threadpool_new) cpu_backend_proc(g.backend, "ggml_threadpool_new");
        auto tp_set = (pfn_set_threadpool) cpu_backend_proc(g.backend, "ggml_backend_cpu_set_threadpool");
        if (tp_new != nullptr && tp_set != nullptr) {
            ggml_threadpool_params tpp = transcribe::make_threadpool_params(std::max(1, n_threads));
            g.tp                       = tp_new(&tpp);
            if (g.tp != nullptr) {
                tp_set(g.backend, g.tp);
            }
        }
        // If unresolved (non-CPU exotic backend), g.tp stays null and each
        // graph_compute spawns a transient pool — correct, just less tuned.
    }

    ggml_init_params ip{};
    // ~17 op nodes/layer + (1 + 2L) input tensors; generous headroom.
    ip.mem_size   = ggml_tensor_overhead() * static_cast<size_t>(32 * L + 16) + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx         = ggml_init(ip);
    if (g.ctx == nullptr) {
        return fail();
    }

    g.x = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, H);
    ggml_set_input(g.x);
    g.ph.assign(static_cast<size_t>(L), nullptr);
    g.pc.assign(static_cast<size_t>(L), nullptr);
    g.nh.assign(static_cast<size_t>(L), nullptr);
    g.nc.assign(static_cast<size_t>(L), nullptr);
    for (int l = 0; l < L; ++l) {
        g.ph[l] = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, H);
        ggml_set_input(g.ph[l]);
        g.pc[l] = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, H);
        ggml_set_input(g.pc[l]);
    }

    // gates = Wx@x + Wh@h_prev + b; split [i,f,g,o]; c' = f*c + i*g; h' = o*tanh(c').
    // Gate order [i, f, g, o] (PyTorch standard).
    ggml_tensor * in = g.x;
    for (int l = 0; l < L; ++l) {
        const auto &  lh = p.lstm[static_cast<size_t>(l)];
        ggml_tensor * gates =
            ggml_add(g.ctx, ggml_add(g.ctx, ggml_mul_mat(g.ctx, lh.g_Wx, in), ggml_mul_mat(g.ctx, lh.g_Wh, g.ph[l])),
                     lh.g_b);  // [4H]
        auto part = [&](int k) {
            return ggml_view_1d(g.ctx, gates, H,
                                static_cast<size_t>(k) * static_cast<size_t>(H) * ggml_element_size(gates));
        };
        ggml_tensor * i_ = ggml_sigmoid(g.ctx, part(0));
        ggml_tensor * f_ = ggml_sigmoid(g.ctx, part(1));
        ggml_tensor * gg = ggml_tanh(g.ctx, part(2));
        ggml_tensor * o_ = ggml_sigmoid(g.ctx, part(3));
        g.nc[l]          = ggml_add(g.ctx, ggml_mul(g.ctx, f_, g.pc[l]), ggml_mul(g.ctx, i_, gg));
        ggml_set_output(g.nc[l]);
        g.nh[l] = ggml_mul(g.ctx, o_, ggml_tanh(g.ctx, g.nc[l]));
        ggml_set_output(g.nh[l]);
        in = g.nh[l];
    }

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, g.backend);
    if (g.buf == nullptr) {
        return fail();
    }

    g.graph = ggml_new_graph(g.ctx);
    for (int l = 0; l < L; ++l) {
        ggml_build_forward_expand(g.graph, g.nh[l]);
        ggml_build_forward_expand(g.graph, g.nc[l]);
    }
    g.ready = true;
    return true;
}

}  // namespace

HostJoint::~HostJoint() {
    if (w_buf != nullptr) {
        safe_buffer_free(w_buf);
    }
    if (w_ctx != nullptr) {
        ggml_free(w_ctx);
    }
    if (w_backend != nullptr) {
        safe_backend_free(w_backend);
    }
}

HostPredictor::~HostPredictor() {
    if (lstm_w_buf != nullptr) {
        safe_buffer_free(lstm_w_buf);
    }
    if (lstm_w_ctx != nullptr) {
        ggml_free(lstm_w_ctx);
    }
    if (lstm_w_backend != nullptr) {
        safe_backend_free(lstm_w_backend);
    }
}

// Joint-window width for the frame-batched greedy decode. `default_w` is the
// per-head default: 16 for the RNN-T loop (one joint eval per frame; window
// amortizes ~3x — see JointGraphBatch), 1 (serial) for the TDT loop, whose
// duration head already skips frames so a window mostly computes columns an
// emission then invalidates (measured 3-4x SLOWER at W=32 than serial).
// TRANSCRIBE_RNNT_BATCH_W overrides both loops (<= 1 forces the serial
// one-frame-per-dispatch path — the kill switch; clamped to 256). Debug
// tensor dumping forces serial so the dec.joint.0 dump and its log_softmax
// normalization stay byte-stable.
static int resolve_joint_batch_w(int default_w) {
    int w = default_w;
    if (const char * v = transcribe::env::str("TRANSCRIBE_RNNT_BATCH_W")) {
        w = std::atoi(v);
    }
    if (transcribe::debug::enabled()) {
        w = 1;
    }
    return std::min(std::max(w, 1), 256);
}

static bool joint_batch_check_enabled() {
    return transcribe::env::flag("TRANSCRIBE_RNNT_BATCH_CHECK");
}

// Resolve a decode thread count: n_threads <= 0 means "auto" →
// default_n_threads(), matching the encoder.
static int resolve_decode_threads(int n_threads) {
    return n_threads > 0 ? n_threads : transcribe::default_n_threads();
}

transcribe_status build_host_decoder_weights(const ParakeetModel & model, HostDecoderWeights & out) {
    const ParakeetHParams & hp = model.hparams;
    const ParakeetWeights & w  = model.weights;

    // Head-kind dispatch. CTC skips predictor/joint entirely; RNNT and
    // TDT share the predictor + joint mirror code below.
    switch (hp.head_kind) {
        case HeadKind::TDT:
            out.head_kind = HostHeadKind::TDT;
            break;
        case HeadKind::RNNT:
            out.head_kind = HostHeadKind::RNNT;
            break;
        case HeadKind::CTC:
            out.head_kind = HostHeadKind::CTC;
            break;
    }

    if (hp.head_kind == HeadKind::CTC) {
        // CTC head mirror. Source weight ne [1, d_model, n_classes]
        // (PyTorch [n_classes, d_model, 1]); bytes are already row-major
        // [n_classes, d_model] as the host decoder consumes them. Bias is
        // [n_classes].
        out.ctc_head.n_classes = hp.head_ctc_n_classes;
        out.ctc_head.blank_id  = hp.head_ctc_n_classes - 1;  // NeMo convention
        out.ctc_head.d_enc     = hp.enc_d_model;

        if (!read_tensor_to_f32(w.ctc_head.weight, out.ctc_head.weight)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (!read_tensor_to_f32(w.ctc_head.bias, out.ctc_head.bias)) {
            return TRANSCRIBE_ERR_GGUF;
        }

        const size_t expected_w = static_cast<size_t>(out.ctc_head.n_classes) * static_cast<size_t>(out.ctc_head.d_enc);
        if (out.ctc_head.weight.size() != expected_w) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "parakeet decoder: head.ctc.weight size %zu "
                    "!= n_classes*d_enc %zu",
                    out.ctc_head.weight.size(), expected_w);
            return TRANSCRIBE_ERR_GGUF;
        }
        if (static_cast<int>(out.ctc_head.bias.size()) != out.ctc_head.n_classes) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "parakeet decoder: head.ctc.bias size %zu "
                    "!= n_classes %d",
                    out.ctc_head.bias.size(), out.ctc_head.n_classes);
            return TRANSCRIBE_ERR_GGUF;
        }
#if !TRANSCRIBE_HAS_BLAS
        {
            const int n_cls = out.ctc_head.n_classes;
            const int d     = out.ctc_head.d_enc;
            out.ctc_head.weight_t.resize(out.ctc_head.weight.size());
            for (int r = 0; r < n_cls; ++r) {
                for (int c = 0; c < d; ++c) {
                    out.ctc_head.weight_t[static_cast<size_t>(c) * n_cls + r] =
                        out.ctc_head.weight[static_cast<size_t>(r) * d + c];
                }
            }
        }
#endif

        out.tdt_durations.clear();
        out.tdt_max_symbols = 0;
        out.blank_id        = out.ctc_head.blank_id;
        out.n_vocab         = out.ctc_head.n_classes - 1;
        return TRANSCRIBE_OK;
    }

    // ----- Predictor mirror (TDT, RNNT) -----
    out.predictor.pred_hidden = hp.pred_hidden;
    out.predictor.pred_vocab  = hp.pred_vocab;

    if (!read_tensor_to_f32(w.predictor.embed_w, out.predictor.embed_w)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    // Sanity-check the flattened size against the catalog shape.
    {
        const size_t expected = static_cast<size_t>(hp.pred_vocab) * static_cast<size_t>(hp.pred_hidden);
        if (out.predictor.embed_w.size() != expected) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "parakeet decoder: pred.embed.weight size %zu "
                    "!= pred_vocab*pred_hidden %zu",
                    out.predictor.embed_w.size(), expected);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    out.predictor.lstm.assign(hp.pred_n_layers, HostLstmLayer{});
    for (int i = 0; i < hp.pred_n_layers; ++i) {
        const auto & lw = w.predictor.lstm[i];
        auto &       lh = out.predictor.lstm[i];
        if (!read_tensor_to_f32(lw.Wx, lh.Wx)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (!read_tensor_to_f32(lw.Wh, lh.Wh)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (!read_tensor_to_f32(lw.b, lh.b)) {
            return TRANSCRIBE_ERR_GGUF;
        }

        const size_t gates    = static_cast<size_t>(4 * hp.pred_hidden);
        const size_t mat_size = gates * static_cast<size_t>(hp.pred_hidden);
        if (lh.Wx.size() != mat_size || lh.Wh.size() != mat_size) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: pred.lstm.%d Wx/Wh wrong size", i);
            return TRANSCRIBE_ERR_GGUF;
        }
        if (lh.b.size() != gates) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: pred.lstm.%d bias wrong size", i);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // Make the predictor LSTM weights resident. Fatal: a failure means
    // the model cannot decode — fail fast at load.
    if (!build_pred_weights(out.predictor)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: predictor ggml weight build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // ----- Joint mirror -----
    out.joint.d_enc       = hp.enc_d_model;
    out.joint.pred_hidden = hp.pred_hidden;
    out.joint.joint_h     = hp.joint_hidden;
    out.joint.joint_n     = hp.joint_n_classes();
    out.joint.activation  = hp.joint_activation;

    // enc/pred/out_b are mirrored to host fp32; out_w is NOT mirrored
    // here (build_joint_weight dequantizes it from the model tensor).
    if (!read_tensor_to_f32(w.joint.enc_w, out.joint.enc_w)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!read_tensor_to_f32(w.joint.enc_b, out.joint.enc_b)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!read_tensor_to_f32(w.joint.pred_w, out.joint.pred_w)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!read_tensor_to_f32(w.joint.pred_b, out.joint.pred_b)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!read_tensor_to_f32(w.joint.out_b, out.joint.out_b)) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // Make the joint weights resident as fp32 ggml tensors. Fatal: a
    // failure means the model cannot decode — fail fast at load.
    if (w.joint.out_w == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: joint out_w missing");
        return TRANSCRIBE_ERR_GGUF;
    }
    const int ne0 = static_cast<int>(w.joint.out_w->ne[0]);
    const int ne1 = static_cast<int>(w.joint.out_w->ne[1]);
    if (ne0 != out.joint.joint_h || ne1 != out.joint.joint_n) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: out_w ne [%d,%d] != [joint_h=%d, joint_n=%d]", ne0, ne1,
                out.joint.joint_h, out.joint.joint_n);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!build_joint_weight(out.joint, w.joint.out_w)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: joint ggml weight build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // ----- TDT params -----
    out.tdt_durations            = hp.tdt_durations;
    out.tdt_max_symbols          = hp.tdt_max_symbols;
    out.tdt_global_symbol_budget = hp.tdt_global_symbol_budget;
    out.n_vocab                  = hp.pred_vocab - 1;  // raw SP vocab size
    out.blank_id                 = hp.pred_vocab - 1;  // blank lives at vocab_size

    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// LSTM state
// ---------------------------------------------------------------------------

void LstmState::reset(int n_layers, int pred_hidden) {
    h.resize(static_cast<size_t>(n_layers));
    c.resize(static_cast<size_t>(n_layers));
    for (int layer = 0; layer < n_layers; ++layer) {
        h[static_cast<size_t>(layer)].assign(static_cast<size_t>(pred_hidden), 0.0f);
        c[static_cast<size_t>(layer)].assign(static_cast<size_t>(pred_hidden), 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

namespace {

// One predictor step on the ggml graph: reads prev_state, writes the new
// step's (h, c) into new_state, returns a borrowed pointer into
// new_state.h.back(). Feeds prev state + embedding into the resident
// per-call graph, computes, reads new state back into the host LstmState.
// Caller guarantees g.ready and that new_state is sized to (L, H). Returns
// nullptr if the graph compute fails.
const float * predictor_step_ggml(const HostPredictor & predictor,
                                  PredGraph &           g,
                                  int                   last_token,
                                  const LstmState &     prev_state,
                                  LstmState &           new_state,
                                  std::vector<float> &  scratch_x) {
    const int H = predictor.pred_hidden;
    if (static_cast<int>(scratch_x.size()) < H) {
        scratch_x.resize(static_cast<size_t>(H));
    }

    // Embed lookup (or start-state zeros).
    if (last_token < 0) {
        std::fill_n(scratch_x.data(), H, 0.0f);
    } else {
        const size_t row_off = static_cast<size_t>(last_token) * static_cast<size_t>(H);
        std::memcpy(scratch_x.data(), predictor.embed_w.data() + row_off, static_cast<size_t>(H) * sizeof(float));
    }

    const size_t hb = static_cast<size_t>(H) * sizeof(float);
    ggml_backend_tensor_set(g.x, scratch_x.data(), 0, hb);
    for (int l = 0; l < g.L; ++l) {
        ggml_backend_tensor_set(g.ph[l], prev_state.h[static_cast<size_t>(l)].data(), 0, hb);
        ggml_backend_tensor_set(g.pc[l], prev_state.c[static_cast<size_t>(l)].data(), 0, hb);
    }

    if (const ggml_status gs = ggml_backend_graph_compute(g.backend, g.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: predictor compute failed (%d)", static_cast<int>(gs));
        return nullptr;
    }

    for (int l = 0; l < g.L; ++l) {
        ggml_backend_tensor_get(g.nh[l], new_state.h[static_cast<size_t>(l)].data(), 0, hb);
        ggml_backend_tensor_get(g.nc[l], new_state.c[static_cast<size_t>(l)].data(), 0, hb);
    }
    return new_state.h.back().data();
}

// Run the joint network for one decode step.
//
// `enc_proj` is the precomputed encoder projection for this frame
// (enc_w @ enc_frame + enc_b, joint_h floats); the caller batches all
// T_enc projections via one GEMM (precompute_enc_proj_ggml) before the
// loop. The graph computes:
//   pred_proj = pred_w @ pred_state + pred_b     [joint_h]
//   summed    = enc_proj + pred_proj             [joint_h]
//   activated = activation(summed)               [joint_h]
//   logits    = out_w @ activated   + out_b      [joint_n]
// Activation is one of {relu, sigmoid, tanh} (loader allow-list). Returns
// false if the graph compute fails.
bool joint_step(const HostJoint &    j,
                const JointGraph &   g,
                const float *        enc_proj,
                const float *        pred_state,
                std::vector<float> & out_logits) {
    if (static_cast<int>(out_logits.size()) < j.joint_n) {
        out_logits.resize(static_cast<size_t>(j.joint_n));
    }

    // Full joint on the shared decoder pool, one graph, one dispatch.
    ggml_backend_tensor_set(g.pred_in, pred_state, 0, static_cast<size_t>(j.pred_hidden) * sizeof(float));
    ggml_backend_tensor_set(g.enc_in, enc_proj, 0, static_cast<size_t>(j.joint_h) * sizeof(float));
    if (const ggml_status gs = ggml_backend_graph_compute(g.backend, g.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: joint compute failed (%d)", static_cast<int>(gs));
        return false;
    }
    ggml_backend_tensor_get(g.logits, out_logits.data(), 0, static_cast<size_t>(j.joint_n) * sizeof(float));

    // log_softmax over the full joint output matches NeMo's CPU-inference
    // representation of joint_after_projection. It is a uniform per-row
    // shift, so it leaves token/duration argmax AND token_confidence
    // (which re-softmaxes the token sub-range) invariant — the decode
    // output is bit-identical with or without it. It is only needed to
    // make the dec.joint.0 dump element-wise comparable to NeMo's
    // normalized reference, so skip it unless dumping.
    if (transcribe::debug::enabled()) {
        float max_v = out_logits[0];
        for (int i = 1; i < j.joint_n; ++i) {
            if (out_logits[i] > max_v) {
                max_v = out_logits[i];
            }
        }
        double sum = 0.0;
        for (int i = 0; i < j.joint_n; ++i) {
            sum += std::exp(static_cast<double>(out_logits[i] - max_v));
        }
        const float log_sum = static_cast<float>(std::log(sum)) + max_v;
        for (int i = 0; i < j.joint_n; ++i) {
            out_logits[i] -= log_sum;
        }
    }
    return true;
}

// Compute the per-utterance encoder projection out[T, joint_h] =
// enc_out[T, d_enc] @ enc_w^T + enc_b as a single GEMM on `backend`
// (the shared decoder pool). Weight + bias are the model-resident
// j.g_enc_w / j.g_enc_b; only the input and result are allocated here.
// At n = T the matmul is a real GEMM (tinyBLAS engages with
// GGML_LLAMAFILE=ON). Returns false on ggml failure (→ hard decode error).
bool precompute_enc_proj_ggml(const HostJoint &    j,
                              ggml_backend_t       backend,
                              const float *        enc_out,
                              int                  T,
                              int                  d_enc,
                              std::vector<float> & out) {
    if (backend == nullptr || j.g_enc_w == nullptr || j.g_enc_b == nullptr) {
        return false;
    }
    const int joint_h = j.joint_h;
    out.resize(static_cast<size_t>(T) * static_cast<size_t>(joint_h));

    ggml_init_params ip{};
    ip.mem_size        = ggml_tensor_overhead() * 6 + ggml_graph_overhead();
    ip.mem_buffer      = nullptr;
    ip.no_alloc        = true;
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_enc, T);
    ggml_set_input(in);
    ggml_tensor * mm  = ggml_mul_mat(ctx, j.g_enc_w, in);  // [joint_h, T]
    ggml_tensor * res = ggml_add(ctx, mm, j.g_enc_b);      // + [joint_h] (broadcast over T)
    ggml_set_output(res);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(in, enc_out, 0, static_cast<size_t>(T) * static_cast<size_t>(d_enc) * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, res);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        safe_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(res, out.data(), 0, static_cast<size_t>(T) * static_cast<size_t>(joint_h) * sizeof(float));

    safe_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Argmax over a contiguous fp32 range. Returns the index of the
// largest value; ties go to the first occurrence (matches the numpy
// argmax convention). The decoder uses this for both the token and
// duration argmaxes.
int argmax_range(const float * data, int n) {
    int   best_i = 0;
    float best_v = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_v) {
            best_v = data[i];
            best_i = i;
        }
    }
    return best_i;
}

// Batched-vs-serial joint verification (TRANSCRIBE_RNNT_BATCH_CHECK=1).
//
// The W-column mul_mat can dispatch a different kernel than the n=1 GEMV
// (GGML_LLAMAFILE tinyBLAS), so last-bit drift between the two is EXPECTED —
// ~1e-5 on nemotron's joint shape, roughly an order of magnitude under the
// family's ~1e-4 accuracy envelope vs the NeMo reference (see the file
// header) — and it shows up on nearly every column. Only a differing ARGMAX
// changes a decode decision, so that alone is an error; the drift itself is
// a DEBUG-level observation. Reporting drift as an error would bury the one
// signal worth grepping for under hundreds of expected-noise lines.
//
// `n_dur > 0` additionally checks the TDT duration head, whose argmax is the
// loop's second decision.
static void joint_batch_check(const char *  tag,
                              const float * serial,
                              const float * batch,
                              int           n_logits,
                              int           n_token_cls,
                              int           n_dur,
                              int           step,
                              int           iter) {
    if (std::memcmp(serial, batch, static_cast<size_t>(n_logits) * sizeof(float)) == 0) {
        return;
    }
    float max_d  = 0.0f;
    int   n_diff = 0;
    for (int i = 0; i < n_logits; ++i) {
        const float d = std::fabs(serial[i] - batch[i]);
        if (d > 0.0f) {
            ++n_diff;
        }
        max_d = std::max(max_d, d);
    }
    const int  tok_s   = argmax_range(serial, n_token_cls);
    const int  tok_b   = argmax_range(batch, n_token_cls);
    const int  dur_s   = n_dur > 0 ? argmax_range(serial + n_token_cls, n_dur) : 0;
    const int  dur_b   = n_dur > 0 ? argmax_range(batch + n_token_cls, n_dur) : 0;
    const bool flipped = (tok_s != tok_b) || (dur_s != dur_b);
    log_msg(flipped ? TRANSCRIBE_LOG_LEVEL_ERROR : TRANSCRIBE_LOG_LEVEL_DEBUG,
            "%s: batched joint %s at step=%d iter=%d n_diff=%d max=%g token=%d/%d dur=%d/%d", tag,
            flipped ? "ARGMAX FLIP (decode diverges)" : "drift (argmax stable)", step, iter, n_diff, max_d, tok_s,
            tok_b, dur_s, dur_b);
}

// The vocabulary's re-rank of a step whose unbiased argmax emitted a token:
// the bonuses go on a copy of the logits, blank is excluded, and the
// language allowlist's blocked tokens stay unreachable (argmax_range_masked).
// The model's own logits are left untouched for the confidence that follows.
static int boost_rerank(const float *                   token_logits,
                        int                             n_token_cls,
                        int                             blank_id,
                        const std::vector<uint8_t> *    blocked_tokens,
                        const transcribe::PhraseBoost & boost,
                        int                             boost_state,
                        std::vector<float> &            scratch) {
    scratch.assign(token_logits, token_logits + n_token_cls);
    boost.apply(boost_state, scratch.data(), n_token_cls, blank_id);
    if (blank_id >= 0 && blank_id < n_token_cls) {
        scratch[static_cast<size_t>(blank_id)] = -std::numeric_limits<float>::infinity();
    }
    return argmax_range_masked(scratch.data(), n_token_cls, blocked_tokens);
}

// Token argmax, biased toward the vocabulary when there is one.
//
// The model alone decides between blank and a token: only when its own
// argmax emits something does the vocabulary re-rank the non-blank tokens.
// Biasing can then change which word is written but never whether one is,
// so it cannot invent speech on silence or noise. With no terms this is
// argmax_range_masked, bit for bit.
static int biased_argmax(const float *                   token_logits,
                         int                             n_token_cls,
                         int                             blank_id,
                         const std::vector<uint8_t> *    blocked_tokens,
                         const transcribe::PhraseBoost * boost,
                         int                             boost_state,
                         std::vector<float> &            scratch) {
    const int unbiased = argmax_range_masked(token_logits, n_token_cls, blocked_tokens);
    if (boost == nullptr || boost->empty() || unbiased == blank_id) {
        return unbiased;
    }
    return boost_rerank(token_logits, n_token_cls, blank_id, blocked_tokens, *boost, boost_state, scratch);
}

// Entropy-based confidence over a token-logit slice. Mirrors the
// reference ParakeetTDT.decode_greedy path:
//
//     token_probs = softmax(token_logits)
//     entropy     = -sum(p * log(p + 1e-10))
//     max_entropy = log(vocab_size + 1)
//     confidence  = 1 - entropy / max_entropy
//
// In our terms `vocab_size + 1 == pred_vocab == n_token_classes`.
// The result lives in [0, 1] modulo the +1e-10 epsilon (which
// matches the reference verbatim, including its slight
// negative-bias for nearly-uniform distributions).
float token_confidence(const float * token_logits, int n_token_classes, std::vector<float> & scratch_probs) {
    // Numerically stable softmax: subtract max before exp.
    float max_logit = token_logits[0];
    for (int i = 1; i < n_token_classes; ++i) {
        if (token_logits[i] > max_logit) {
            max_logit = token_logits[i];
        }
    }
    if (static_cast<int>(scratch_probs.size()) < n_token_classes) {
        scratch_probs.resize(static_cast<size_t>(n_token_classes));
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_token_classes; ++i) {
        const float e                         = std::exp(token_logits[i] - max_logit);
        scratch_probs[static_cast<size_t>(i)] = e;
        sum_exp += static_cast<double>(e);
    }
    const float inv_sum = static_cast<float>(1.0 / sum_exp);
    double      entropy = 0.0;
    for (int i = 0; i < n_token_classes; ++i) {
        const float p = scratch_probs[static_cast<size_t>(i)] * inv_sum;
        // +1e-10 matches the reference.
        entropy -= static_cast<double>(p) * std::log(static_cast<double>(p) + 1e-10);
    }
    const double max_entropy = std::log(static_cast<double>(n_token_classes));
    if (max_entropy <= 0.0) {
        return 1.0f;
    }
    return static_cast<float>(1.0 - entropy / max_entropy);
}

}  // namespace

int argmax_range_masked(const float * data, int n, const std::vector<uint8_t> * blocked) {
    if (blocked == nullptr || blocked->empty()) {
        return argmax_range(data, n);
    }
    const int n_mask = static_cast<int>(blocked->size());
    int       best_i = -1;
    float     best_v = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (i < n_mask && (*blocked)[static_cast<size_t>(i)] != 0) {
            continue;
        }
        if (best_i < 0 || data[i] > best_v) {
            best_v = data[i];
            best_i = i;
        }
    }
    // Every id blocked cannot happen (blank is never a locale tag); keep the
    // plain argmax as the defined answer rather than an out-of-range id.
    return best_i < 0 ? argmax_range(data, n) : best_i;
}

// ---------------------------------------------------------------------------
// Device-resident decode (GPU primary)
// ---------------------------------------------------------------------------
//
// The CPU decoder above runs one sub-millisecond CPU graph per step over
// ~35 MB of F32 weights; on a GPU primary the encoder finishes long before it
// (measured 2026-11-17, RTX 4070, warm: parakeet v3 jfk encode 34 ms / decode
// 47 ms, nemotron 38 / 140 ms). This path runs the same predictor and joint
// math on the model's GPU device, with everything that changes between steps
// kept there:
//   - weights: F32 copies of the resident CPU tensors (the same values);
//   - LSTM state in two slots S[0] / S[1]: a predictor step reads the
//     committed slot and writes the other, an emission flips `committed` —
//     the CPU loops' state / next_state swap, so a streaming chunk that
//     re-runs the predictor from the committed state gets the same input;
//   - P = pred_w @ h + pred_b, written by the predictor step (the CPU joint
//     recomputes it every frame: same values, now once per emission);
//   - the encoder projection of the whole utterance (one GEMM), from which a
//     joint step gathers its frames by index.
// A predictor step uploads one embedding row; a joint step uploads its frame
// indices and downloads the logits. Everything runs on the decoder's own
// backend and stream (async set / compute / get, one synchronize per joint
// step), so persistent graphs replay as CUDA graphs. The decision logic
// (argmax, masks, confidence, stepping) is the CPU loops', on host logits.
// TRANSCRIBE_PARAKEET_DEVICE_DECODER=0 keeps the decoder on the CPU.

// Widest joint window the device loops take (TRANSCRIBE_PARAKEET_DEVICE_JOINT_W).
constexpr int k_max_device_window = 256;

struct ParakeetDeviceDecoder {
    ggml_backend_t backend = nullptr;
    int            H       = 0;  // predictor hidden
    int            L       = 0;  // LSTM layers
    int            joint_h = 0;
    int            joint_n = 0;
    int            d_enc   = 0;
    std::string    activation;
    const float *  embed      = nullptr;  // borrowed host embedding [pred_vocab, H] (model-owned)
    int            pred_vocab = 0;

    // Weights, the two LSTM state slots, P and the predictor input: fixed for
    // the decoder's lifetime.
    ggml_context *             w_ctx = nullptr;
    ggml_backend_buffer_t      w_buf = nullptr;
    std::vector<ggml_tensor *> Wx, Wh, Wb;
    ggml_tensor *              pred_w = nullptr;
    ggml_tensor *              pred_b = nullptr;
    ggml_tensor *              enc_w  = nullptr;
    ggml_tensor *              enc_b  = nullptr;
    ggml_tensor *              out_w  = nullptr;
    ggml_tensor *              out_b  = nullptr;
    std::vector<ggml_tensor *> Sh[2], Sc[2];
    ggml_tensor *              P           = nullptr;  // [joint_h]
    ggml_tensor *              x           = nullptr;  // [H] embedding row
    ggml_tensor *              bias        = nullptr;  // [n_token_cls] 0 / -inf: the run's blocked tokens
    bool                       bias_active = false;
    int                        n_token_cls = 0;
    int                        n_dur       = 0;  // TDT duration classes, 0 for RNN-T

    // Pinned host staging for every per-step copy (pageable memory makes an
    // async CUDA copy wait for the stream). Falls back to heap memory when the
    // device has no host buffer type.
    ggml_backend_buffer_t host_buf = nullptr;
    std::vector<uint8_t>  host_fallback;
    float *               x_pin     = nullptr;  // [H]
    int32_t *             idx_pin   = nullptr;  // [k_max_window]
    int32_t *             tok_pin   = nullptr;  // [k_max_window]
    int32_t *             dur_pin   = nullptr;  // [k_max_window]
    float *               col_pin   = nullptr;  // [n_token_cls] one column's token logits
    float *               state_pin = nullptr;  // [2 * L * H] h then c per layer

    // Predictor step graphs: pg[i] reads slot i, writes slot 1 - i and P.
    ggml_context *        pg_ctx[2] = { nullptr, nullptr };
    ggml_backend_buffer_t pg_buf[2] = { nullptr, nullptr };
    ggml_cgraph *         pg[2]     = { nullptr, nullptr };

    // Encoder projection + joint graphs, sized for `cap` frames and a window
    // of W frames; rebuilt when an utterance needs more.
    int                   cap       = 0;
    int                   W         = 0;
    ggml_context *        ej_ctx    = nullptr;
    ggml_backend_buffer_t ej_buf    = nullptr;
    ggml_tensor *         enc_in    = nullptr;  // [d_enc, cap]
    ggml_tensor *         enc_proj  = nullptr;  // [joint_h, cap] leaf, written by enc_graph
    ggml_cgraph *         enc_graph = nullptr;

    // Joint graphs over 1 frame (j1) and over the window (jw): logits
    // [joint_n, n], and the token / duration argmax per frame [n] (I32).
    struct Joint {
        ggml_tensor * idx    = nullptr;
        ggml_tensor * logits = nullptr;
        ggml_tensor * tok    = nullptr;
        ggml_tensor * dur    = nullptr;  // null for RNN-T
        ggml_cgraph * graph  = nullptr;
    };

    Joint j1;
    Joint jw;

    int                T         = 0;  // frames of the current utterance / chunk
    int                committed = 0;
    std::vector<float> zeros;
    const Joint *      last_joint = nullptr;  // the graph whose outputs are current

    void free_capacity() noexcept {
        if (ej_buf != nullptr) {
            safe_buffer_free(ej_buf);
            ej_buf = nullptr;
        }
        if (ej_ctx != nullptr) {
            ggml_free(ej_ctx);
            ej_ctx = nullptr;
        }
        enc_in = enc_proj = nullptr;
        enc_graph         = nullptr;
        j1                = Joint{};
        jw                = Joint{};
        last_joint        = nullptr;
        cap = W = 0;
    }

    ~ParakeetDeviceDecoder() {
        free_capacity();
        if (host_buf != nullptr) {
            safe_buffer_free(host_buf);
        }
        for (int i = 0; i < 2; ++i) {
            if (pg_buf[i] != nullptr) {
                safe_buffer_free(pg_buf[i]);
            }
            if (pg_ctx[i] != nullptr) {
                ggml_free(pg_ctx[i]);
            }
        }
        if (w_buf != nullptr) {
            safe_buffer_free(w_buf);
        }
        if (w_ctx != nullptr) {
            ggml_free(w_ctx);
        }
        if (backend != nullptr) {
            safe_backend_free(backend);
        }
    }
};

namespace {

ggml_tensor * joint_activation(ggml_context * ctx, const std::string & activation, ggml_tensor * t) {
    if (activation == "relu") {
        return ggml_relu(ctx, t);
    }
    if (activation == "sigmoid") {
        return ggml_sigmoid(ctx, t);
    }
    return ggml_tanh(ctx, t);  // "tanh" (loader allow-list)
}

// Predictor step graph for slot `i`: the CPU build_pred_graph math (same op
// order), reading slot i, then copies of every layer's (h, c) into slot 1 - i
// and P = pred_w @ h_last + pred_b. The copies are expanded last, after every
// read of slot i.
bool build_device_pred_graph(ParakeetDeviceDecoder & d, int i) {
    ggml_init_params ip{};
    ip.mem_size        = ggml_tensor_overhead() * static_cast<size_t>(32 * d.L + 16) + ggml_graph_overhead();
    ip.no_alloc        = true;
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return false;
    }
    d.pg_ctx[i]                  = ctx;
    const int                  H = d.H;
    std::vector<ggml_tensor *> outs;
    ggml_tensor *              in = d.x;
    for (int l = 0; l < d.L; ++l) {
        ggml_tensor * gates = ggml_add(
            ctx, ggml_add(ctx, ggml_mul_mat(ctx, d.Wx[l], in), ggml_mul_mat(ctx, d.Wh[l], d.Sh[i][l])), d.Wb[l]);
        auto part = [&](int k) {
            return ggml_view_1d(ctx, gates, H,
                                static_cast<size_t>(k) * static_cast<size_t>(H) * ggml_element_size(gates));
        };
        ggml_tensor * i_ = ggml_sigmoid(ctx, part(0));
        ggml_tensor * f_ = ggml_sigmoid(ctx, part(1));
        ggml_tensor * gg = ggml_tanh(ctx, part(2));
        ggml_tensor * o_ = ggml_sigmoid(ctx, part(3));
        ggml_tensor * nc = ggml_add(ctx, ggml_mul(ctx, f_, d.Sc[i][l]), ggml_mul(ctx, i_, gg));
        ggml_tensor * nh = ggml_mul(ctx, o_, ggml_tanh(ctx, nc));
        outs.push_back(ggml_cpy(ctx, nh, d.Sh[1 - i][l]));
        outs.push_back(ggml_cpy(ctx, nc, d.Sc[1 - i][l]));
        in = nh;
    }
    outs.push_back(ggml_cpy(ctx, ggml_add(ctx, ggml_mul_mat(ctx, d.pred_w, in), d.pred_b), d.P));

    d.pg[i] = ggml_new_graph(ctx);
    for (ggml_tensor * t : outs) {
        ggml_build_forward_expand(d.pg[i], t);
    }
    d.pg_buf[i] = ggml_backend_alloc_ctx_tensors(ctx, d.backend);
    return d.pg_buf[i] != nullptr;
}

// Joint graph over `n` gathered frames: the CPU joint math with P for the
// predictor term, then the decisions on the device: the token argmax over the
// token logits plus `bias` (0, or -inf for the run's blocked tokens:
// argmax_range_masked), and for TDT the duration argmax. A step then downloads
// a few int32s instead of the logits; the token logits of a frame are fetched
// only when it emits (its confidence).
ParakeetDeviceDecoder::Joint build_device_joint(ParakeetDeviceDecoder & d, ggml_context * ctx, int n) {
    ParakeetDeviceDecoder::Joint j;
    j.idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_input(j.idx);
    ggml_tensor * cols   = ggml_get_rows(ctx, d.enc_proj, j.idx);  // [joint_h, n]
    ggml_tensor * summed = ggml_add(ctx, cols, d.P);
    ggml_tensor * act    = joint_activation(ctx, d.activation, summed);
    j.logits             = ggml_add(ctx, ggml_mul_mat(ctx, d.out_w, act), d.out_b);  // [joint_n, n]
    ggml_set_output(j.logits);

    ggml_tensor * tok_logits = ggml_view_2d(ctx, j.logits, d.n_token_cls, n, j.logits->nb[1], 0);
    j.tok                    = ggml_argmax(ctx, ggml_add(ctx, tok_logits, d.bias));  // [n]
    ggml_set_output(j.tok);
    if (d.n_dur > 0) {
        ggml_tensor * dur_logits = ggml_view_2d(ctx, j.logits, d.n_dur, n, j.logits->nb[1],
                                                static_cast<size_t>(d.n_token_cls) * ggml_element_size(j.logits));
        // Always copied out: the view starts n_token_cls floats into the
        // logits, an offset Vulkan's argmax rejects (storage buffer offset
        // alignment), and for n > 1 it is strided as well.
        j.dur                    = ggml_argmax(ctx, ggml_cont(ctx, dur_logits));
        ggml_set_output(j.dur);
    }
    j.graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(j.graph, j.tok);
    if (j.dur != nullptr) {
        ggml_build_forward_expand(j.graph, j.dur);
    }
    return j;
}

// Size the encoder-projection and joint graphs for `T` frames and a joint
// window of `W` (>= 1). Capacity grows geometrically, so an utterance only
// rebuilds when it is longer than every earlier one.
bool device_ensure_capacity(ParakeetDeviceDecoder & d, int T, int W) {
    if (d.ej_ctx != nullptr && T <= d.cap && W == d.W) {
        return true;
    }
    const int new_cap = T <= d.cap ? d.cap : std::max({ T, d.cap * 2, 64 });  // a new W alone keeps the size
    d.free_capacity();

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 48 + 3 * ggml_graph_overhead();
    ip.no_alloc = true;
    d.ej_ctx    = ggml_init(ip);
    if (d.ej_ctx == nullptr) {
        return false;
    }
    ggml_context * ctx = d.ej_ctx;
    d.enc_in           = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.d_enc, new_cap);
    ggml_set_input(d.enc_in);
    d.enc_proj         = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.joint_h, new_cap);
    // enc_proj = enc_w @ enc_in + enc_b (precompute_enc_proj_ggml's math),
    // copied into the leaf the joint graphs read.
    ggml_tensor * proj = ggml_add(ctx, ggml_mul_mat(ctx, d.enc_w, d.enc_in), d.enc_b);
    d.enc_graph        = ggml_new_graph(ctx);
    ggml_build_forward_expand(d.enc_graph, ggml_cpy(ctx, proj, d.enc_proj));

    d.j1 = build_device_joint(d, ctx, 1);
    if (W > 1) {
        d.jw = build_device_joint(d, ctx, W);
    }
    d.ej_buf = ggml_backend_alloc_ctx_tensors(ctx, d.backend);
    if (d.ej_buf == nullptr) {
        d.free_capacity();
        return false;
    }
    // Frames past an utterance's end are never gathered; keep them finite.
    ggml_backend_buffer_clear(d.ej_buf, 0);
    d.cap = new_cap;
    d.W   = W;
    return true;
}

// Upload `T` encoder frames and project them (async; the first joint step's
// synchronize completes it). `enc_out` must stay valid until then.
bool device_begin_frames(ParakeetDeviceDecoder & d, const float * enc_out, int T, int W) {
    if (!device_ensure_capacity(d, T, W)) {
        return false;
    }
    d.T          = T;
    d.last_joint = nullptr;
    ggml_backend_tensor_set_async(d.backend, d.enc_in, enc_out, 0,
                                  static_cast<size_t>(T) * static_cast<size_t>(d.d_enc) * sizeof(float));
    return ggml_backend_graph_compute_async(d.backend, d.enc_graph) == GGML_STATUS_SUCCESS;
}

// The run's blocked tokens as the argmax bias (argmax_range_masked's mask).
// Uploads only when a mask is set or one has to be cleared; synchronizes then,
// so the host vector may be reused.
void device_set_mask(ParakeetDeviceDecoder & d, const std::vector<uint8_t> * blocked) {
    const bool active = blocked != nullptr && !blocked->empty();
    if (!active && !d.bias_active) {
        return;
    }
    std::vector<float> bias(static_cast<size_t>(d.n_token_cls), 0.0f);
    if (active) {
        const size_t n = std::min(blocked->size(), bias.size());
        for (size_t i = 0; i < n; ++i) {
            if ((*blocked)[i] != 0) {
                bias[i] = -INFINITY;
            }
        }
    }
    ggml_backend_tensor_set_async(d.backend, d.bias, bias.data(), 0, bias.size() * sizeof(float));
    ggml_backend_synchronize(d.backend);
    d.bias_active = active;
}

// Committed slot <- the given state (zeros when null).
void device_load_state(ParakeetDeviceDecoder & d, const LstmState * state) {
    d.committed     = 0;
    const size_t H  = static_cast<size_t>(d.H);
    const size_t hb = H * sizeof(float);
    for (int l = 0; l < d.L; ++l) {
        float * h = d.state_pin + static_cast<size_t>(2 * l) * H;
        float * c = h + H;
        std::memcpy(h, state != nullptr ? state->h[static_cast<size_t>(l)].data() : d.zeros.data(), hb);
        std::memcpy(c, state != nullptr ? state->c[static_cast<size_t>(l)].data() : d.zeros.data(), hb);
        ggml_backend_tensor_set_async(d.backend, d.Sh[0][l], h, 0, hb);
        ggml_backend_tensor_set_async(d.backend, d.Sc[0][l], c, 0, hb);
    }
}

// The committed slot -> state (synchronizes).
void device_store_state(ParakeetDeviceDecoder & d, LstmState & state) {
    const size_t H  = static_cast<size_t>(d.H);
    const size_t hb = H * sizeof(float);
    for (int l = 0; l < d.L; ++l) {
        float * h = d.state_pin + static_cast<size_t>(2 * l) * H;
        ggml_backend_tensor_get_async(d.backend, d.Sh[d.committed][l], h, 0, hb);
        ggml_backend_tensor_get_async(d.backend, d.Sc[d.committed][l], h + H, 0, hb);
    }
    ggml_backend_synchronize(d.backend);
    for (int l = 0; l < d.L; ++l) {
        const float * h = d.state_pin + static_cast<size_t>(2 * l) * H;
        std::memcpy(state.h[static_cast<size_t>(l)].data(), h, hb);
        std::memcpy(state.c[static_cast<size_t>(l)].data(), h + H, hb);
    }
}

// One predictor step from the committed slot (embedding of last_token, zeros
// for the start state). Async: the next joint step's synchronize completes it.
bool device_pred(ParakeetDeviceDecoder & d, int last_token) {
    if (last_token < 0) {
        std::fill_n(d.x_pin, d.H, 0.0f);
    } else {
        std::memcpy(d.x_pin, d.embed + static_cast<size_t>(last_token) * static_cast<size_t>(d.H),
                    static_cast<size_t>(d.H) * sizeof(float));
    }
    ggml_backend_tensor_set_async(d.backend, d.x, d.x_pin, 0, static_cast<size_t>(d.H) * sizeof(float));
    return ggml_backend_graph_compute_async(d.backend, d.pg[d.committed]) == GGML_STATUS_SUCCESS;
}

// Joint decisions for frames step .. step + n - 1 (n = 1 or the window W;
// frames past the end repeat the last one and are never consumed):
// d.tok_pin[k] / d.dur_pin[k] hold frame step + k's token (-1 if every token
// was blocked) and duration argmax until the next call. False on a compute
// failure.
bool device_joint(ParakeetDeviceDecoder & d, int step, int n) {
    for (int k = 0; k < n; ++k) {
        d.idx_pin[k] = std::min(step + k, d.T - 1);
    }
    const ParakeetDeviceDecoder::Joint & j = n > 1 ? d.jw : d.j1;
    ggml_backend_tensor_set_async(d.backend, j.idx, d.idx_pin, 0, static_cast<size_t>(n) * sizeof(int32_t));
    if (ggml_backend_graph_compute_async(d.backend, j.graph) != GGML_STATUS_SUCCESS) {
        return false;
    }
    ggml_backend_tensor_get_async(d.backend, j.tok, d.tok_pin, 0, static_cast<size_t>(n) * sizeof(int32_t));
    if (j.dur != nullptr) {
        ggml_backend_tensor_get_async(d.backend, j.dur, d.dur_pin, 0, static_cast<size_t>(n) * sizeof(int32_t));
    }
    ggml_backend_synchronize(d.backend);
    d.last_joint = &j;
    return true;
}

// Queue the copy of column k's token logits (of the last joint call) into
// d.col_pin, for the emitted token's confidence. No synchronize: the next
// device_joint (or device_finish) completes it, and it reads the logits before
// any later graph on the stream overwrites them.
void device_fetch_token_logits(ParakeetDeviceDecoder & d, int k) {
    const ParakeetDeviceDecoder::Joint & j = *d.last_joint;
    ggml_backend_tensor_get_async(d.backend, j.logits, d.col_pin, static_cast<size_t>(k) * j.logits->nb[1],
                                  static_cast<size_t>(d.n_token_cls) * sizeof(float));
}

// Joint window width for the device loops: the CPU default, unless
// TRANSCRIBE_PARAKEET_DEVICE_JOINT_W overrides it (<= 1 = one frame per step,
// clamped to 256).
int resolve_device_joint_w(int default_w) {
    int w = default_w;
    if (const char * v = transcribe::env::str("TRANSCRIBE_PARAKEET_DEVICE_JOINT_W")) {
        w = std::atoi(v);
    }
    return std::min(std::max(w, 1), k_max_device_window);
}

// Whether a decode call may use `device`: given, and not a debugging mode that
// reads the CPU loop's intermediate state (tensor dumps, the batched-joint
// cross-check).
bool device_usable(const ParakeetDeviceDecoder * device) {
    return device != nullptr && !transcribe::debug::enabled() && !joint_batch_check_enabled();
}

// The vocabulary's say in a device step's token. The device's argmax is
// already biased_argmax's answer on a blank, and from the automaton's root
// (where PhraseBoost::apply adds nothing: a term's first piece earns no
// bonus). Only inside a partial match does a non-blank step need the re-rank:
// then this column's logits are fetched now, re-ranked on the host exactly as
// the CPU loop does, and left in d.col_pin for the confidence (*logits_ready).
int device_boosted_token(ParakeetDeviceDecoder &         d,
                         int                             col,
                         int                             pred_token,
                         int                             blank_id,
                         const std::vector<uint8_t> *    blocked_tokens,
                         const transcribe::PhraseBoost * boost,
                         int                             boost_state,
                         std::vector<float> &            scratch,
                         bool *                          logits_ready) {
    *logits_ready = false;
    if (boost == nullptr || boost->empty() || boost_state == transcribe::PhraseBoost::root || pred_token == blank_id) {
        return pred_token;
    }
    device_fetch_token_logits(d, col);
    ggml_backend_synchronize(d.backend);
    *logits_ready = true;
    return boost_rerank(d.col_pin, d.n_token_cls, blank_id, blocked_tokens, *boost, boost_state, scratch);
}

// TDT greedy decode on the device: decode_tdt_greedy's loop, line for line,
// with device steps in place of the CPU graphs.
transcribe_status decode_tdt_greedy_device(const HostDecoderWeights &      w,
                                           ParakeetDeviceDecoder &         d,
                                           const float *                   enc_out,
                                           int                             T_enc,
                                           std::vector<TdtToken> &         out_tokens,
                                           const std::vector<uint8_t> *    blocked_tokens,
                                           const transcribe::PhraseBoost * boost) {
    const int n_token_cls = w.predictor.pred_vocab;
    const int n_dur       = static_cast<int>(w.tdt_durations.size());
    const int blank_id    = w.blank_id;
    const int batch_W     = resolve_device_joint_w(/*default_w=*/1);

    const int64_t t_begin = ggml_time_us();
    if (!device_begin_frames(d, enc_out, T_enc, batch_W)) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    device_load_state(d, nullptr);
    device_set_mask(d, blocked_tokens);

    std::vector<float> scratch_probs;
    int                win_base  = 0;
    bool               win_valid = false;
    int                pending   = -1;  // out_tokens index whose confidence waits for its logits copy

    // Vocabulary biasing state (see decode_tdt_greedy); untouched with no terms.
    int                boost_state = transcribe::PhraseBoost::root;
    std::vector<float> boosted;

    int       last_token      = -1;
    int       step            = 0;
    int       new_symbols     = 0;
    const int max_iters       = 16 * T_enc + 1024;
    int       iter            = 0;
    bool      predictor_dirty = true;

    const bool global_budget   = w.tdt_global_symbol_budget;
    int64_t    steps_remaining = static_cast<int64_t>(w.tdt_max_symbols) * T_enc;

    while (step < T_enc && iter < max_iters && (!global_budget || steps_remaining > 0)) {
        ++iter;
        if (predictor_dirty) {
            if (!device_pred(d, last_token)) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            predictor_dirty = false;
        }

        int col = 0;  // this frame's column in the joint output
        if (batch_W > 1) {
            if (!win_valid || step < win_base || step >= win_base + batch_W) {
                if (!device_joint(d, step, batch_W)) {
                    return TRANSCRIBE_ERR_BACKEND;
                }
                win_base  = step;
                win_valid = true;
            }
            col = step - win_base;
        } else if (!device_joint(d, step, 1)) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        if (pending >= 0) {  // the previous emission's logits arrived with that synchronize
            out_tokens[static_cast<size_t>(pending)].p = token_confidence(d.col_pin, n_token_cls, scratch_probs);
            pending                                    = -1;
        }

        // Every token blocked (-1) is argmax_range_masked's plain-argmax
        // fallback: leave that rare case to the CPU decode.
        int       pred_token = d.tok_pin[col];
        const int decision   = d.dur_pin[col];
        if (pred_token < 0 || decision < 0 || decision >= n_dur) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        const int duration = w.tdt_durations[static_cast<size_t>(decision)];

        bool logits_ready = false;
        pred_token = device_boosted_token(d, col, pred_token, blank_id, blocked_tokens, boost, boost_state, boosted,
                                          &logits_ready);

        const bool is_blank = (pred_token == blank_id);
        if (!is_blank) {
            TdtToken tok;
            tok.id              = pred_token;
            tok.step_at_emit    = step;
            tok.duration_frames = duration;
            out_tokens.push_back(tok);
            if (logits_ready) {
                out_tokens.back().p = token_confidence(d.col_pin, n_token_cls, scratch_probs);
            } else {
                device_fetch_token_logits(d, col);  // confidence: filled in after the next synchronize
                pending = static_cast<int>(out_tokens.size()) - 1;
            }

            last_token = pred_token;
            d.committed ^= 1;  // commit (the CPU loop's state swap)
            predictor_dirty = true;
            win_valid       = false;
            if (boost != nullptr) {
                boost_state = boost->next(boost_state, pred_token);
            }
        }

        if (global_budget) {
            step += (is_blank && duration == 0) ? 1 : duration;
            --steps_remaining;
            continue;
        }
        step += duration;
        new_symbols += 1;
        if (duration != 0) {
            new_symbols = 0;
        } else if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
            step += 1;
            new_symbols = 0;
        } else if (is_blank && w.tdt_max_symbols > 0) {
            const int skip = w.tdt_max_symbols - new_symbols;
            if (skip > 0 && iter + skip < max_iters) {
                iter += skip;
                step += 1;
                new_symbols = 0;
            }
        }
    }
    if (pending >= 0) {  // the last emission's logits
        ggml_backend_synchronize(d.backend);
        out_tokens[static_cast<size_t>(pending)].p = token_confidence(d.col_pin, n_token_cls, scratch_probs);
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
            "decoder (device): %d iters, %d tokens, T_enc=%d  total=%.1f ms  per_iter=%.0f us", iter,
            static_cast<int>(out_tokens.size()), T_enc, (ggml_time_us() - t_begin) / 1000.0,
            static_cast<double>(ggml_time_us() - t_begin) / std::max(iter, 1));
    if (iter >= max_iters && !global_budget) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder (device): hit iteration cap (%d)", max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }
    return TRANSCRIBE_OK;
}

// RNN-T greedy decode on the device, offline (state from the start state) or
// streaming (state_io / last_token_io carried across chunks, tokens at
// frame_offset + step): decode_rnnt_greedy's and
// decode_rnnt_greedy_streaming's loop, line for line, with device steps.
// `boost` is the offline vocabulary (streaming passes none).
transcribe_status decode_rnnt_greedy_device(const HostDecoderWeights &      w,
                                            ParakeetDeviceDecoder &         d,
                                            const float *                   enc_out,
                                            int                             T_enc,
                                            LstmState *                     state_io,
                                            int *                           last_token_io,
                                            int                             frame_offset,
                                            int                             batch_W,
                                            std::vector<TdtToken> &         out_tokens,
                                            const std::vector<uint8_t> *    blocked_tokens,
                                            const transcribe::PhraseBoost * boost = nullptr) {
    const int n_token_cls = w.predictor.pred_vocab;
    const int blank_id    = w.blank_id;

    const int64_t t_begin = ggml_time_us();
    if (!device_begin_frames(d, enc_out, T_enc, batch_W)) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    device_load_state(d, state_io);
    device_set_mask(d, blocked_tokens);

    std::vector<float> scratch_probs;
    int                win_base  = 0;
    bool               win_valid = false;
    int                pending   = -1;  // out_tokens index whose confidence waits for its logits copy

    int                boost_state = transcribe::PhraseBoost::root;
    std::vector<float> boosted;

    int       last_token      = last_token_io != nullptr ? *last_token_io : -1;
    int       step            = 0;
    int       new_symbols     = 0;
    const int max_iters       = 16 * T_enc + 1024;
    int       iter            = 0;
    bool      predictor_dirty = true;

    while (step < T_enc && iter < max_iters) {
        ++iter;
        if (predictor_dirty) {
            if (!device_pred(d, last_token)) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            predictor_dirty = false;
        }

        int col = 0;  // this frame's column in the joint output
        if (batch_W > 1) {
            if (!win_valid || step < win_base || step >= win_base + batch_W) {
                if (!device_joint(d, step, batch_W)) {
                    return TRANSCRIBE_ERR_BACKEND;
                }
                win_base  = step;
                win_valid = true;
            }
            col = step - win_base;
        } else if (!device_joint(d, step, 1)) {
            return TRANSCRIBE_ERR_BACKEND;
        }

        if (pending >= 0) {  // the previous emission's logits arrived with that synchronize
            out_tokens[static_cast<size_t>(pending)].p = token_confidence(d.col_pin, n_token_cls, scratch_probs);
            pending                                    = -1;
        }
        int pred_token = d.tok_pin[col];
        if (pred_token < 0) {
            return TRANSCRIBE_ERR_BACKEND;  // every token blocked: the CPU decode's fallback
        }
        bool logits_ready = false;
        pred_token = device_boosted_token(d, col, pred_token, blank_id, blocked_tokens, boost, boost_state, boosted,
                                          &logits_ready);
        const bool is_blank = (pred_token == blank_id);
        if (is_blank) {
            step += 1;
            new_symbols = 0;
        } else {
            TdtToken tok;
            tok.id              = pred_token;
            tok.step_at_emit    = frame_offset + step;
            tok.duration_frames = 1;
            out_tokens.push_back(tok);
            if (logits_ready) {
                out_tokens.back().p = token_confidence(d.col_pin, n_token_cls, scratch_probs);
            } else {
                device_fetch_token_logits(d, col);  // confidence: filled in after the next synchronize
                pending = static_cast<int>(out_tokens.size()) - 1;
            }

            last_token = pred_token;
            d.committed ^= 1;  // commit (the CPU loop's state swap)
            predictor_dirty = true;
            win_valid       = false;
            if (boost != nullptr) {
                boost_state = boost->next(boost_state, pred_token);
            }

            new_symbols += 1;
            if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
                step += 1;
                new_symbols = 0;
            }
        }
    }
    if (pending >= 0) {  // the last emission's logits
        ggml_backend_synchronize(d.backend);
        out_tokens[static_cast<size_t>(pending)].p = token_confidence(d.col_pin, n_token_cls, scratch_probs);
    }

    if (iter >= max_iters) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder (device rnnt): hit iteration cap (%d)", max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }
    if (state_io != nullptr) {
        device_store_state(d, *state_io);
    }
    if (last_token_io != nullptr) {
        *last_token_io = last_token;
    }
    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "decoder (device rnnt): %d iters, %d tokens, T_enc=%d  total=%.1f ms", iter,
            static_cast<int>(out_tokens.size()), T_enc, (ggml_time_us() - t_begin) / 1000.0);
    return TRANSCRIBE_OK;
}

}  // namespace

ParakeetDeviceDecoder * parakeet_device_decoder_new(const HostDecoderWeights & w,
                                                    ggml_backend_dev_t         dev,
                                                    const ParakeetWeights *    model_weights) {
    if (dev == nullptr || w.head_kind == HostHeadKind::CTC || !w.predictor.lstm_ready || !w.joint.w_ready ||
        w.predictor.lstm.empty()) {
        return nullptr;
    }
    auto d     = std::make_unique<ParakeetDeviceDecoder>();
    d->backend = ggml_backend_dev_init(dev, nullptr);
    if (d->backend == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet decoder: cannot open %s for the device decoder",
                ggml_backend_dev_name(dev));
        return nullptr;
    }
    const HostPredictor & p = w.predictor;
    const HostJoint &     j = w.joint;
    d->H                    = p.pred_hidden;
    d->L                    = static_cast<int>(p.lstm.size());
    d->joint_h              = j.joint_h;
    d->joint_n              = j.joint_n;
    d->d_enc                = j.d_enc;
    d->activation           = j.activation;
    d->embed                = p.embed_w.data();
    d->pred_vocab           = p.pred_vocab;
    d->n_token_cls          = p.pred_vocab;
    d->n_dur                = w.head_kind == HostHeadKind::TDT ? static_cast<int>(w.tdt_durations.size()) : 0;
    if (d->n_token_cls + d->n_dur != d->joint_n || (w.head_kind == HostHeadKind::TDT && d->n_dur == 0)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet decoder: joint width %d != %d tokens + %d durations; CPU decoder",
                d->joint_n, d->n_token_cls, d->n_dur);
        return nullptr;
    }
    d->zeros.assign(static_cast<size_t>(d->H), 0.0f);

    // Pinned staging: x, frame indices, token / duration decisions, one
    // column of token logits, the LSTM state.
    {
        const size_t f    = sizeof(float);
        const size_t i32  = sizeof(int32_t);
        const size_t n_x  = static_cast<size_t>(d->H);
        const size_t n_c  = static_cast<size_t>(d->n_token_cls);
        const size_t n_s  = static_cast<size_t>(2 * d->L * d->H);
        const size_t w_i  = static_cast<size_t>(k_max_device_window);
        const size_t sz   = n_x * f + 3 * w_i * i32 + n_c * f + n_s * f + 64;
        uint8_t *    base = nullptr;
        if (ggml_backend_buffer_type_t hbt = ggml_backend_dev_host_buffer_type(dev)) {
            d->host_buf = ggml_backend_buft_alloc_buffer(hbt, sz);
        }
        if (d->host_buf != nullptr) {
            base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(d->host_buf));
        } else {
            d->host_fallback.assign(sz, 0);
            base = d->host_fallback.data();
        }
        auto take = [&base](size_t bytes) {
            uint8_t * p = base;
            base += (bytes + 15) & ~static_cast<size_t>(15);
            return p;
        };
        d->x_pin     = reinterpret_cast<float *>(take(n_x * f));
        d->idx_pin   = reinterpret_cast<int32_t *>(take(w_i * i32));
        d->tok_pin   = reinterpret_cast<int32_t *>(take(w_i * i32));
        d->dur_pin   = reinterpret_cast<int32_t *>(take(w_i * i32));
        d->col_pin   = reinterpret_cast<float *>(take(n_c * f));
        d->state_pin = reinterpret_cast<float *>(take(n_s * f));
        std::fill_n(d->dur_pin, w_i, 0);
    }

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * static_cast<size_t>(8 * d->L + 16);
    ip.no_alloc = true;
    d->w_ctx    = ggml_init(ip);
    if (d->w_ctx == nullptr) {
        return nullptr;
    }
    // Weights. Default: F32 copies of the resident CPU tensors, the values
    // the CPU decoder multiplies with. TRANSCRIBE_PARAKEET_DEVICE_DECODER_WEIGHTS
    // =native instead borrows the model's own tensors (the GGUF's storage,
    // e.g. Q8_0) when they already live on this device: no copy and ~4x fewer
    // bytes per step, but a quantized matmul also quantizes the activation,
    // so the logits move further from the CPU decoder's than summation order
    // alone does.
    const char * wmode  = transcribe::env::str("TRANSCRIBE_PARAKEET_DEVICE_DECODER_WEIGHTS");
    bool         native = model_weights != nullptr && wmode != nullptr && std::strcmp(wmode, "native") == 0;
    auto         on_dev = [dev](const ggml_tensor * t) {
        return t != nullptr && t->buffer != nullptr &&
               ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer)) == dev &&
               !ggml_backend_buffer_is_host(t->buffer);
    };
    auto same_shape = [](const ggml_tensor * a, const ggml_tensor * b) {
        return a != nullptr && b != nullptr && ggml_are_same_shape(a, b);
    };
    if (native) {
        const ParakeetPredictor & mp = model_weights->predictor;
        const ParakeetJoint &     mj = model_weights->joint;
        native                       = mp.lstm.size() == p.lstm.size();
        for (size_t l = 0; native && l < p.lstm.size(); ++l) {
            native = on_dev(mp.lstm[l].Wx) && on_dev(mp.lstm[l].Wh) && on_dev(mp.lstm[l].b) &&
                     same_shape(mp.lstm[l].Wx, p.lstm[l].g_Wx) && same_shape(mp.lstm[l].Wh, p.lstm[l].g_Wh) &&
                     same_shape(mp.lstm[l].b, p.lstm[l].g_b);
        }
        const std::pair<const ggml_tensor *, const ggml_tensor *> pairs[] = {
            { mj.pred_w, j.g_pred_w },
            { mj.pred_b, j.g_pred_b },
            { mj.enc_w,  j.g_enc_w  },
            { mj.enc_b,  j.g_enc_b  },
            { mj.out_w,  j.gw_w     },
            { mj.out_b,  j.gw_b     },
        };
        for (const auto & pr : pairs) {
            native = native && on_dev(pr.first) && same_shape(pr.first, pr.second);
        }
        if (!native) {
            log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                    "parakeet decoder: the model's decoder weights are not on %s in the expected shapes; "
                    "using F32 copies",
                    ggml_backend_dev_name(dev));
        }
    }
    auto like = [&](const ggml_tensor * src) {
        return ggml_new_tensor(d->w_ctx, src->type, ggml_n_dims(src), src->ne);
    };
    if (native) {
        const ParakeetPredictor & mp = model_weights->predictor;
        const ParakeetJoint &     mj = model_weights->joint;
        for (const auto & ml : mp.lstm) {
            d->Wx.push_back(ml.Wx);
            d->Wh.push_back(ml.Wh);
            d->Wb.push_back(ml.b);
        }
        d->pred_w = mj.pred_w;
        d->pred_b = mj.pred_b;
        d->enc_w  = mj.enc_w;
        d->enc_b  = mj.enc_b;
        d->out_w  = mj.out_w;
        d->out_b  = mj.out_b;
    } else {
        for (const HostLstmLayer & lh : p.lstm) {
            d->Wx.push_back(like(lh.g_Wx));
            d->Wh.push_back(like(lh.g_Wh));
            d->Wb.push_back(like(lh.g_b));
        }
        d->pred_w = like(j.g_pred_w);
        d->pred_b = like(j.g_pred_b);
        d->enc_w  = like(j.g_enc_w);
        d->enc_b  = like(j.g_enc_b);
        d->out_w  = like(j.gw_w);
        d->out_b  = like(j.gw_b);
    }
    for (int s = 0; s < 2; ++s) {
        for (int l = 0; l < d->L; ++l) {
            d->Sh[s].push_back(ggml_new_tensor_1d(d->w_ctx, GGML_TYPE_F32, d->H));
            d->Sc[s].push_back(ggml_new_tensor_1d(d->w_ctx, GGML_TYPE_F32, d->H));
        }
    }
    d->P = ggml_new_tensor_1d(d->w_ctx, GGML_TYPE_F32, d->joint_h);
    d->x = ggml_new_tensor_1d(d->w_ctx, GGML_TYPE_F32, d->H);
    ggml_set_input(d->x);
    d->bias = ggml_new_tensor_1d(d->w_ctx, GGML_TYPE_F32, d->n_token_cls);

    d->w_buf = ggml_backend_alloc_ctx_tensors(d->w_ctx, d->backend);
    if (d->w_buf == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet decoder: %s has no room for the device decoder",
                ggml_backend_dev_name(dev));
        return nullptr;
    }
    ggml_backend_buffer_clear(d->w_buf, 0);
    if (!native) {
        for (int l = 0; l < d->L; ++l) {
            ggml_backend_tensor_copy(p.lstm[static_cast<size_t>(l)].g_Wx, d->Wx[l]);
            ggml_backend_tensor_copy(p.lstm[static_cast<size_t>(l)].g_Wh, d->Wh[l]);
            ggml_backend_tensor_copy(p.lstm[static_cast<size_t>(l)].g_b, d->Wb[l]);
        }
        ggml_backend_tensor_copy(j.g_pred_w, d->pred_w);
        ggml_backend_tensor_copy(j.g_pred_b, d->pred_b);
        ggml_backend_tensor_copy(j.g_enc_w, d->enc_w);
        ggml_backend_tensor_copy(j.g_enc_b, d->enc_b);
        ggml_backend_tensor_copy(j.gw_w, d->out_w);
        ggml_backend_tensor_copy(j.gw_b, d->out_b);
    }

    for (int i = 0; i < 2; ++i) {
        if (!build_device_pred_graph(*d, i)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet decoder: device predictor graph build failed");
            return nullptr;
        }
    }
    // Every op must run on the device itself (no CPU fallback inside these
    // graphs: they are computed directly on the backend, not via a scheduler).
    for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < ggml_graph_n_nodes(d->pg[i]); ++k) {
            if (!ggml_backend_supports_op(d->backend, ggml_graph_node(d->pg[i], k))) {
                log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet decoder: %s cannot run %s; decoding on the CPU",
                        ggml_backend_dev_name(dev), ggml_op_desc(ggml_graph_node(d->pg[i], k)));
                return nullptr;
            }
        }
    }
    if (!device_ensure_capacity(*d, 64, 1)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet decoder: device joint graph build failed");
        return nullptr;
    }
    for (ggml_cgraph * g : { d->enc_graph, d->j1.graph }) {
        for (int k = 0; k < ggml_graph_n_nodes(g); ++k) {
            if (!ggml_backend_supports_op(d->backend, ggml_graph_node(g, k))) {
                log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet decoder: %s cannot run %s; decoding on the CPU",
                        ggml_backend_dev_name(dev), ggml_op_desc(ggml_graph_node(g, k)));
                return nullptr;
            }
        }
    }
    log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "parakeet decoder: predictor and joint on %s (%s weights, %.1f MiB)",
            ggml_backend_dev_name(dev), native ? "model" : "F32",
            static_cast<double>(ggml_backend_buffer_get_size(d->w_buf)) / (1024.0 * 1024.0));
    return d.release();
}

void parakeet_device_decoder_free(ParakeetDeviceDecoder * d) noexcept {
    delete d;
}

void parakeet_device_decoder_trim(ParakeetDeviceDecoder * d) noexcept {
    // 2048 encoder frames ~ 164 s at 80 ms: ~13 MB of frame buffers kept for
    // the common case, released for long-form inputs.
    if (d != nullptr && d->cap > 2048) {
        ggml_backend_synchronize(d->backend);
        d->free_capacity();
    }
}

// ---------------------------------------------------------------------------
// Greedy decode driver
// ---------------------------------------------------------------------------

transcribe_status decode_tdt_greedy(const HostDecoderWeights &      w,
                                    const float *                   enc_out,
                                    int                             T_enc,
                                    int                             d_enc,
                                    int                             n_threads,
                                    std::vector<TdtToken> &         out_tokens,
                                    const std::vector<uint8_t> *    blocked_tokens,
                                    ParakeetDeviceDecoder *         device,
                                    const transcribe::PhraseBoost * boost) {
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.joint.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder: enc d_model mismatch (got %d, "
                "expected %d)",
                d_enc, w.joint.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (device_usable(device)) {
        const size_t            n_before = out_tokens.size();
        const transcribe_status dst =
            decode_tdt_greedy_device(w, *device, enc_out, T_enc, out_tokens, blocked_tokens, boost);
        if (dst == TRANSCRIBE_OK) {
            return dst;
        }
        out_tokens.resize(n_before);
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet decoder: device decode failed (%d); decoding on the CPU",
                static_cast<int>(dst));
    }

    const int nt          = resolve_decode_threads(n_threads);
    const int n_layers    = static_cast<int>(w.predictor.lstm.size());
    const int H           = w.predictor.pred_hidden;
    const int n_token_cls = w.predictor.pred_vocab;  // == vocab_size + 1
    const int n_dur       = static_cast<int>(w.tdt_durations.size());
    const int blank_id    = w.blank_id;

    // All-ggml decode on ONE shared threadpool: PredGraph owns the
    // backend + pool; enc_proj and the joint graph borrow it. pg is
    // declared first so it is destroyed LAST (jg's buffer lives on pg's
    // backend). Built per decode call, reentrant. A build failure is a
    // hard decode error.
    PredGraph  pg;
    JointGraph jg;
    build_pred_graph(pg, w.predictor, nt);
    if (pg.ready) {
        build_joint_graph(jg, w.joint, pg.backend);
    }
    if (!pg.ready || !jg.ready) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml decode graph build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    // Frame-batched joint window (see JointGraphBatch). A batch-graph build
    // failure silently falls back to the serial path — never a decode error.
    const int       batch_W = resolve_joint_batch_w(/*default_w=*/1);
    JointGraphBatch jgb;
    if (batch_W > 1) {
        build_joint_graph_batch(jgb, w.joint, pg.backend, batch_W);
    }
    const bool use_batch   = jgb.ready;
    const bool batch_check = use_batch && joint_batch_check_enabled();

    // Two LSTM states, both pre-sized: `state` is the committed
    // state we read from each iteration; `next_state` is where the
    // predictor writes the new step's outputs. On a non-blank
    // emission we swap; on a blank emission we just discard the
    // contents of `next_state` (overwritten on the next iteration).
    LstmState state;
    LstmState next_state;
    state.reset(n_layers, H);
    next_state.reset(n_layers, H);

    // Precompute encoder projections for all T_enc frames (decode-state
    // independent; one sgemm before the loop). See precompute_enc_proj_ggml.
    const int          joint_h = w.joint.joint_h;
    std::vector<float> enc_proj_all;
    const int64_t      t_enc_proj_start = ggml_time_us();
    if (!precompute_enc_proj_ggml(w.joint, pg.backend, enc_out, T_enc, d_enc, enc_proj_all)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: enc_proj graph failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    if (use_batch) {
        // Zero-pad W-1 trailing frames so a window based near the last frame
        // stays in bounds; the padded columns are never consumed.
        enc_proj_all.resize(static_cast<size_t>(T_enc + batch_W - 1) * static_cast<size_t>(joint_h), 0.0f);
    }
    const int64_t t_enc_proj_us = ggml_time_us() - t_enc_proj_start;

    // Per-call scratch reused across every decode step.
    std::vector<float> scratch_x;
    std::vector<float> scratch_probs;
    std::vector<float> logits;

    // Batched-window state: logits_w holds the current window's [joint_n, W]
    // readback; win_base is its first frame; win_valid drops on every state
    // change (non-blank emission) so stale columns are never consumed.
    std::vector<float> logits_w;
    if (use_batch) {
        logits_w.resize(static_cast<size_t>(w.joint.joint_n) * static_cast<size_t>(batch_W));
    }
    int  win_base  = 0;
    bool win_valid = false;

    int last_token  = -1;  // sentinel: no previous token (start state)
    int step        = 0;
    int new_symbols = 0;

    // Runaway-protection cap; legitimate transcriptions never approach it.
    const int max_iters = 16 * T_enc + 1024;
    int       iter      = 0;
    int64_t   t_pred_us = 0, t_joint_us = 0, t_conf_us = 0;

    // On a blank emission (last_token, state) are preserved, so the
    // predictor would write the same next_state next iteration. Skip the
    // LSTM unroll and reuse the previous decoder_out; recompute on a
    // non-blank emission (state.swap + last_token change). Bit-exact with
    // the unconditional path.
    bool predictor_dirty = true;

    // Vocabulary biasing state; untouched when no terms were given.
    int                boost_state = transcribe::PhraseBoost::root;
    std::vector<float> boosted;

    // See ParakeetHParams::tdt_global_symbol_budget.
    const bool global_budget   = w.tdt_global_symbol_budget;
    int64_t    steps_remaining = static_cast<int64_t>(w.tdt_max_symbols) * T_enc;

    while (step < T_enc && iter < max_iters && (!global_budget || steps_remaining > 0)) {
        ++iter;

        // ----- Predictor (one LSTM step) -----
        const int64_t t0 = ggml_time_us();
        const float * decoder_out;
        if (predictor_dirty) {
            decoder_out = predictor_step_ggml(w.predictor, pg, last_token, state, next_state, scratch_x);
            if (decoder_out == nullptr) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            predictor_dirty = false;
        } else {
            decoder_out = next_state.h.back().data();
        }
        const int64_t t1 = ggml_time_us();

        // ----- Joint (using precomputed encoder projection) -----
        const float * frame_logits;
        if (use_batch) {
            if (!win_valid || step < win_base || step >= win_base + batch_W) {
                ggml_backend_tensor_set(jgb.pred_in, decoder_out, 0,
                                        static_cast<size_t>(w.joint.pred_hidden) * sizeof(float));
                ggml_backend_tensor_set(jgb.enc_in,
                                        enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h),
                                        0, static_cast<size_t>(joint_h) * static_cast<size_t>(batch_W) * sizeof(float));
                if (ggml_backend_graph_compute(jgb.backend, jgb.graph) != GGML_STATUS_SUCCESS) {
                    return TRANSCRIBE_ERR_BACKEND;
                }
                ggml_backend_tensor_get(jgb.logits, logits_w.data(), 0, logits_w.size() * sizeof(float));
                win_base  = step;
                win_valid = true;
            }
            frame_logits =
                logits_w.data() + static_cast<size_t>(step - win_base) * static_cast<size_t>(w.joint.joint_n);
            if (batch_check) {
                const float * enc_proj = enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h);
                if (!joint_step(w.joint, jg, enc_proj, decoder_out, logits)) {
                    return TRANSCRIBE_ERR_BACKEND;
                }
                joint_batch_check("parakeet decoder (tdt)", logits.data(), frame_logits, w.joint.joint_n, n_token_cls,
                                  n_dur, step, iter);
            }
        } else {
            const float * enc_proj = enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h);
            if (!joint_step(w.joint, jg, enc_proj, decoder_out, logits)) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            frame_logits = logits.data();
        }
        const int64_t t2 = ggml_time_us();
        t_pred_us += t1 - t0;
        t_joint_us += t2 - t1;

        // ----- Argmax (token + duration) -----
        const float * token_logits    = frame_logits;
        const float * duration_logits = frame_logits + n_token_cls;

        const int pred_token =
            biased_argmax(token_logits, n_token_cls, blank_id, blocked_tokens, boost, boost_state, boosted);
        const int decision = argmax_range(duration_logits, n_dur);
        const int duration = w.tdt_durations[static_cast<size_t>(decision)];

        // Optional dump of the first decode step (start state, encoder
        // frame 0): embed input, per-layer LSTM h, joint logits. Mirrors
        // the Python reference dumper's decode subcommand.
        if (iter == 1 && transcribe::debug::enabled()) {
            // Predictor scratch_x at start: zeros (vector of length H).
            const long long s_h = H;
            transcribe::debug::dump_host_f32("dec.embed.0", scratch_x.data(), s_h, &s_h, 1, "decoder.embed");

            for (int layer = 0; layer < n_layers; ++layer) {
                char name_h[64];
                char name_c[64];
                std::snprintf(name_h, sizeof(name_h), "dec.lstm.%d.h.0", layer);
                std::snprintf(name_c, sizeof(name_c), "dec.lstm.%d.c.0", layer);
                transcribe::debug::dump_host_f32(name_h, next_state.h[layer].data(), s_h, &s_h, 1, "decoder.lstm");
                transcribe::debug::dump_host_f32(name_c, next_state.c[layer].data(), s_h, &s_h, 1, "decoder.lstm");
            }

            const long long s_n = w.joint.joint_n;
            transcribe::debug::dump_host_f32("dec.joint.0", frame_logits, s_n, &s_n, 1, "decoder.joint");
        }

        // ----- TDT emit + state advance -----
        const bool is_blank = (pred_token == blank_id);
        if (!is_blank) {
            const int64_t tc0 = ggml_time_us();
            const float   p   = token_confidence(token_logits, n_token_cls, scratch_probs);
            t_conf_us += ggml_time_us() - tc0;
            TdtToken tok;
            tok.id              = pred_token;
            tok.p               = p;
            tok.step_at_emit    = step;
            tok.duration_frames = duration;
            out_tokens.push_back(tok);

            last_token = pred_token;
            std::swap(state, next_state);  // commit
            predictor_dirty = true;
            win_valid       = false;       // pred side changed; window logits are stale
            if (boost != nullptr) {
                boost_state = boost->next(boost_state, pred_token);
            }
        }

        if (global_budget) {
            step += (is_blank && duration == 0) ? 1 : duration;
            --steps_remaining;
            continue;
        }

        // Step / stuck advance. Matches the reference:
        //   step += duration
        //   new_symbols += 1
        //   if duration != 0: new_symbols = 0
        //   elif max_symbols and new_symbols >= max_symbols:
        //       step += 1; new_symbols = 0
        step += duration;
        new_symbols += 1;
        if (duration != 0) {
            new_symbols = 0;
        } else if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
            step += 1;
            new_symbols = 0;
        } else if (is_blank && w.tdt_max_symbols > 0) {
            // A blank with duration 0 leaves (step, last_token, state)
            // unchanged, so every following iteration is identical until
            // max_symbols forces the step to advance. Fast-forward those
            // repeated no-op loops without changing the observable result.
            const int skip = w.tdt_max_symbols - new_symbols;
            if (skip > 0 && iter + skip < max_iters) {
                iter += skip;
                step += 1;
                new_symbols = 0;
            }
        }
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
            "decoder: %d iters, %d tokens, T_enc=%d  "
            "enc_proj=%.1f ms  pred=%.1f ms  joint=%.1f ms  conf=%.1f ms  "
            "total=%.1f ms  per_iter=%.0f us",
            iter, static_cast<int>(out_tokens.size()), T_enc, t_enc_proj_us / 1000.0, t_pred_us / 1000.0,
            t_joint_us / 1000.0, t_conf_us / 1000.0, (t_enc_proj_us + t_pred_us + t_joint_us + t_conf_us) / 1000.0,
            static_cast<double>(t_pred_us + t_joint_us) / std::max(iter, 1));

    if (iter >= max_iters && !global_budget) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder: hit iteration cap (%d) — pathological "
                "logits or loop bug",
                max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// RNNT greedy decode
// ---------------------------------------------------------------------------
//
// Same predictor + joint forward as TDT, but the joint emits exactly
// `vocab+1` logits (no duration extras). Step rule:
//
//   per iteration:
//     run predictor (one LSTM step)
//     run joint -> token_logits[vocab+1]
//     argmax -> pred_token
//     if pred_token == blank:
//         step += 1
//         new_symbols = 0
//     else:
//         emit token, swap predictor state, last_token = pred_token
//         new_symbols += 1
//         if max_symbols and new_symbols >= max_symbols:
//             step += 1; new_symbols = 0   # break out of stuck-on-frame loop
//
// Mirrors NeMo's `RNNTGreedyDecodeInfer.step_per_frame` (without
// duration). Matches the reference dump points (`dec.embed.0`,
// `dec.lstm.{layer}.{h,c}.0`, `dec.joint.0`) emitted on iter 1.

transcribe_status decode_rnnt_greedy(const HostDecoderWeights &      w,
                                     const float *                   enc_out,
                                     int                             T_enc,
                                     int                             d_enc,
                                     int                             n_threads,
                                     std::vector<TdtToken> &         out_tokens,
                                     const std::vector<uint8_t> *    blocked_tokens,
                                     ParakeetDeviceDecoder *         device,
                                     const transcribe::PhraseBoost * boost) {
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.joint.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (rnnt): enc d_model mismatch (got %d, "
                "expected %d)",
                d_enc, w.joint.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (device_usable(device)) {
        const size_t            n_before = out_tokens.size();
        const transcribe_status dst      = decode_rnnt_greedy_device(
            w, *device, enc_out, T_enc, /*state_io=*/nullptr, /*last_token_io=*/nullptr, /*frame_offset=*/0,
            resolve_device_joint_w(/*default_w=*/16), out_tokens, blocked_tokens, boost);
        if (dst == TRANSCRIBE_OK) {
            return dst;
        }
        out_tokens.resize(n_before);
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "parakeet decoder (rnnt): device decode failed (%d); decoding on the CPU",
                static_cast<int>(dst));
    }

    const int nt          = resolve_decode_threads(n_threads);
    const int n_layers    = static_cast<int>(w.predictor.lstm.size());
    const int H           = w.predictor.pred_hidden;
    const int n_token_cls = w.predictor.pred_vocab;  // == vocab + 1
    const int blank_id    = w.blank_id;

    // Per-call decode graphs (see decode_tdt_greedy for the lifecycle).
    PredGraph  pg;
    JointGraph jg;
    build_pred_graph(pg, w.predictor, nt);
    if (pg.ready) {
        build_joint_graph(jg, w.joint, pg.backend);
    }
    if (!pg.ready || !jg.ready) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml decode graph build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    // Frame-batched joint window (see JointGraphBatch); build failure falls
    // back to serial.
    const int       batch_W = resolve_joint_batch_w(/*default_w=*/16);
    JointGraphBatch jgb;
    if (batch_W > 1) {
        build_joint_graph_batch(jgb, w.joint, pg.backend, batch_W);
    }
    const bool use_batch   = jgb.ready;
    const bool batch_check = use_batch && joint_batch_check_enabled();

    LstmState state;
    LstmState next_state;
    state.reset(n_layers, H);
    next_state.reset(n_layers, H);

    // Precompute encoder projections for all T_enc frames (same as TDT).
    const int          joint_h = w.joint.joint_h;
    std::vector<float> enc_proj_all;
    const int64_t      t_enc_proj_start = ggml_time_us();
    if (!precompute_enc_proj_ggml(w.joint, pg.backend, enc_out, T_enc, d_enc, enc_proj_all)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: enc_proj graph failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    if (use_batch) {
        enc_proj_all.resize(static_cast<size_t>(T_enc + batch_W - 1) * static_cast<size_t>(joint_h), 0.0f);
    }
    const int64_t t_enc_proj_us = ggml_time_us() - t_enc_proj_start;

    std::vector<float> scratch_x;
    std::vector<float> scratch_probs;
    std::vector<float> logits;

    std::vector<float> logits_w;
    if (use_batch) {
        logits_w.resize(static_cast<size_t>(w.joint.joint_n) * static_cast<size_t>(batch_W));
    }
    int  win_base  = 0;
    bool win_valid = false;

    int last_token  = -1;
    int step        = 0;
    int new_symbols = 0;

    const int max_iters = 16 * T_enc + 1024;
    int       iter      = 0;
    int64_t   t_pred_us = 0, t_joint_us = 0, t_conf_us = 0;

    // Predictor-step cache (see decode_tdt_greedy). Bit-exact with the
    // unconditional path; elides 60-80% of predictor work on a typical
    // RNN-T iters/token ratio.
    bool predictor_dirty = true;

    int                boost_state = transcribe::PhraseBoost::root;
    std::vector<float> boosted;

    while (step < T_enc && iter < max_iters) {
        ++iter;

        const int64_t t0 = ggml_time_us();
        const float * decoder_out;
        if (predictor_dirty) {
            decoder_out = predictor_step_ggml(w.predictor, pg, last_token, state, next_state, scratch_x);
            if (decoder_out == nullptr) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            predictor_dirty = false;
        } else {
            decoder_out = next_state.h.back().data();
        }
        const int64_t t1 = ggml_time_us();

        const float * frame_logits;
        if (use_batch) {
            if (!win_valid || step < win_base || step >= win_base + batch_W) {
                ggml_backend_tensor_set(jgb.pred_in, decoder_out, 0,
                                        static_cast<size_t>(w.joint.pred_hidden) * sizeof(float));
                ggml_backend_tensor_set(jgb.enc_in,
                                        enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h),
                                        0, static_cast<size_t>(joint_h) * static_cast<size_t>(batch_W) * sizeof(float));
                if (ggml_backend_graph_compute(jgb.backend, jgb.graph) != GGML_STATUS_SUCCESS) {
                    return TRANSCRIBE_ERR_BACKEND;
                }
                ggml_backend_tensor_get(jgb.logits, logits_w.data(), 0, logits_w.size() * sizeof(float));
                win_base  = step;
                win_valid = true;
            }
            frame_logits =
                logits_w.data() + static_cast<size_t>(step - win_base) * static_cast<size_t>(w.joint.joint_n);
            if (batch_check) {
                const float * enc_proj = enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h);
                if (!joint_step(w.joint, jg, enc_proj, decoder_out, logits)) {
                    return TRANSCRIBE_ERR_BACKEND;
                }
                joint_batch_check("parakeet decoder (rnnt)", logits.data(), frame_logits, w.joint.joint_n, n_token_cls,
                                  /*n_dur=*/0, step, iter);
            }
        } else {
            const float * enc_proj = enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h);
            if (!joint_step(w.joint, jg, enc_proj, decoder_out, logits)) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            frame_logits = logits.data();
        }
        const int64_t t2 = ggml_time_us();
        t_pred_us += t1 - t0;
        t_joint_us += t2 - t1;

        // RNNT joint output is just `n_token_cls` floats (no duration extras).
        // The same line serves the serial joint and the frame-batched window.
        const float * token_logits = frame_logits;
        const int     pred_token =
            biased_argmax(token_logits, n_token_cls, blank_id, blocked_tokens, boost, boost_state, boosted);

        if (iter == 1 && transcribe::debug::enabled()) {
            const long long s_h = H;
            transcribe::debug::dump_host_f32("dec.embed.0", scratch_x.data(), s_h, &s_h, 1, "decoder.embed");
            for (int layer = 0; layer < n_layers; ++layer) {
                char name_h[64];
                char name_c[64];
                std::snprintf(name_h, sizeof(name_h), "dec.lstm.%d.h.0", layer);
                std::snprintf(name_c, sizeof(name_c), "dec.lstm.%d.c.0", layer);
                transcribe::debug::dump_host_f32(name_h, next_state.h[layer].data(), s_h, &s_h, 1, "decoder.lstm");
                transcribe::debug::dump_host_f32(name_c, next_state.c[layer].data(), s_h, &s_h, 1, "decoder.lstm");
            }
            const long long s_n = w.joint.joint_n;
            transcribe::debug::dump_host_f32("dec.joint.0", frame_logits, s_n, &s_n, 1, "decoder.joint");
        }

        const bool is_blank = (pred_token == blank_id);
        if (is_blank) {
            step += 1;
            new_symbols = 0;
        } else {
            const int64_t tc0 = ggml_time_us();
            const float   p   = token_confidence(token_logits, n_token_cls, scratch_probs);
            t_conf_us += ggml_time_us() - tc0;
            TdtToken tok;
            tok.id              = pred_token;
            tok.p               = p;
            tok.step_at_emit    = step;
            tok.duration_frames = 1;
            out_tokens.push_back(tok);

            last_token = pred_token;
            std::swap(state, next_state);
            predictor_dirty = true;
            win_valid       = false;  // pred side changed; window logits are stale
            if (boost != nullptr) {
                boost_state = boost->next(boost_state, pred_token);
            }

            new_symbols += 1;
            if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
                step += 1;
                new_symbols = 0;
            }
        }
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
            "decoder (rnnt): %d iters, %d tokens, T_enc=%d  "
            "enc_proj=%.1f ms  pred=%.1f ms  joint=%.1f ms  conf=%.1f ms  "
            "total=%.1f ms",
            iter, static_cast<int>(out_tokens.size()), T_enc, t_enc_proj_us / 1000.0, t_pred_us / 1000.0,
            t_joint_us / 1000.0, t_conf_us / 1000.0, (t_enc_proj_us + t_pred_us + t_joint_us + t_conf_us) / 1000.0);

    if (iter >= max_iters) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (rnnt): hit iteration cap (%d) — "
                "pathological logits or loop bug",
                max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// RNNT greedy decode — streaming
// ---------------------------------------------------------------------------
//
// Same algorithm as decode_rnnt_greedy but with externally-managed
// LSTM state (state_io) and previous token (last_token_io). The
// chunk's encoder frames are decoded in stream-wide coordinates
// (step_at_emit = frame_offset + local_step). No timing log.
transcribe_status decode_rnnt_greedy_streaming(const HostDecoderWeights &      w,
                                               const float *                   enc_out,
                                               int                             T_enc_new,
                                               int                             d_enc,
                                               LstmState &                     state_io,
                                               int &                           last_token_io,
                                               int                             frame_offset,
                                               int                             n_threads,
                                               std::vector<TdtToken> &         out_tokens,
                                               ParakeetStreamingDecoderGraph * dec_graph,
                                               const std::vector<uint8_t> *    blocked_tokens,
                                               ParakeetDeviceDecoder *         device) {
    if (enc_out == nullptr || T_enc_new <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // Pure RNN-T greedy only: a TDT head's durations would be silently
    // ignored and a CTC head carries no predictor state. Fail loud rather
    // than mis-decode a hypothetical future TDT/CTC streaming model.
    if (w.head_kind != HostHeadKind::RNNT) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (rnnt-stream): streaming decode "
                "requires an RNN-T head; this model's head is not "
                "RNN-T");
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    if (d_enc != w.joint.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (rnnt-stream): enc d_model mismatch "
                "(got %d, expected %d)",
                d_enc, w.joint.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int nt          = resolve_decode_threads(n_threads);
    const int n_layers    = static_cast<int>(w.predictor.lstm.size());
    const int H           = w.predictor.pred_hidden;
    const int n_token_cls = w.predictor.pred_vocab;
    const int blank_id    = w.blank_id;

    if (device_usable(device)) {
        // Same state_io validation as the CPU path below, then the chunk on
        // the device; state_io / last_token_io are only written on success,
        // so a failure falls back to the CPU decode from the same state.
        bool degenerate =
            static_cast<int>(state_io.h.size()) != n_layers || static_cast<int>(state_io.c.size()) != n_layers;
        for (int l = 0; !degenerate && l < n_layers; ++l) {
            degenerate = static_cast<int>(state_io.h[l].size()) != H || static_cast<int>(state_io.c[l].size()) != H;
        }
        if (degenerate) {
            state_io.reset(n_layers, H);
            last_token_io = -1;
        }
        LstmState               state      = state_io;
        int                     last_token = last_token_io;
        const size_t            n_before   = out_tokens.size();
        const transcribe_status dst =
            decode_rnnt_greedy_device(w, *device, enc_out, T_enc_new, &state, &last_token, frame_offset,
                                      resolve_device_joint_w(/*default_w=*/16), out_tokens, blocked_tokens);
        if (dst == TRANSCRIBE_OK) {
            state_io      = std::move(state);
            last_token_io = last_token;
            return dst;
        }
        out_tokens.resize(n_before);
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                "parakeet decoder (rnnt-stream): device decode failed (%d); decoding on the CPU",
                static_cast<int>(dst));
    }

    // Reuse persistent decode graphs if provided, otherwise fall back to local per-call graphs.
    PredGraph *  p_pg = nullptr;
    JointGraph * p_jg = nullptr;
    PredGraph    local_pg;
    JointGraph   local_jg;

    if (dec_graph != nullptr) {
        if (!dec_graph->ready || dec_graph->nt != nt || !dec_graph->pg || !dec_graph->jg) {
            dec_graph->reset();
            dec_graph->pg = std::make_unique<PredGraph>();
            dec_graph->jg = std::make_unique<JointGraph>();
            build_pred_graph(*dec_graph->pg, w.predictor, nt);
            if (dec_graph->pg->ready) {
                build_joint_graph(*dec_graph->jg, w.joint, dec_graph->pg->backend);
            }
            if (dec_graph->pg->ready && dec_graph->jg->ready) {
                dec_graph->nt    = nt;
                dec_graph->ready = true;
            }
        }
        if (dec_graph->ready) {
            p_pg = dec_graph->pg.get();
            p_jg = dec_graph->jg.get();
        }
    }

    if (p_pg == nullptr || p_jg == nullptr) {
        build_pred_graph(local_pg, w.predictor, nt);
        if (local_pg.ready) {
            build_joint_graph(local_jg, w.joint, local_pg.backend);
        }
        if (!local_pg.ready || !local_jg.ready) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml decode graph build failed");
            return TRANSCRIBE_ERR_BACKEND;
        }
        p_pg = &local_pg;
        p_jg = &local_jg;
    }

    PredGraph &  pg = *p_pg;
    JointGraph & jg = *p_jg;

    // The cached decoder survives chunk boundaries. Park its polling workers
    // while mel/encoder work runs; otherwise two CPU pools fight for the same
    // cores. Retain the graph, weights and workers, and resume on dispatch.
    struct pool_lease {
        ggml_backend_t     backend;
        pfn_set_threadpool set_pool;

        ~pool_lease() {
            if (set_pool != nullptr) {
                set_pool(backend, nullptr);
            }
        }
    } lease{ pg.backend,
             reinterpret_cast<pfn_set_threadpool>(cpu_backend_proc(pg.backend, "ggml_backend_cpu_set_threadpool")) };

    if (lease.set_pool != nullptr && pg.tp != nullptr) {
        lease.set_pool(pg.backend, pg.tp);
    }

    // Validate state_io shape; reset if degenerate.
    if (static_cast<int>(state_io.h.size()) != n_layers || static_cast<int>(state_io.c.size()) != n_layers) {
        state_io.reset(n_layers, H);
        last_token_io = -1;
    }
    for (int l = 0; l < n_layers; ++l) {
        if (static_cast<int>(state_io.h[l].size()) != H || static_cast<int>(state_io.c[l].size()) != H) {
            state_io.reset(n_layers, H);
            last_token_io = -1;
            break;
        }
    }

    // Precompute encoder projections for this chunk only.
    const int          joint_h = w.joint.joint_h;
    std::vector<float> enc_proj_all;
    if (!precompute_enc_proj_ggml(w.joint, pg.backend, enc_out, T_enc_new, d_enc, enc_proj_all)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder (rnnt-stream): enc_proj graph failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Working state. On return, `state` (committed after the last
    // non-blank emission, or unchanged for an all-blank chunk) and
    // last_token are copied back into state_io / last_token_io.
    LstmState state = state_io;
    LstmState next_state;
    next_state.reset(n_layers, H);

    std::vector<float> scratch_x;
    std::vector<float> scratch_probs;
    std::vector<float> logits;

    int last_token  = last_token_io;
    int step        = 0;
    int new_symbols = 0;

    const int max_iters = 16 * T_enc_new + 1024;
    int       iter      = 0;

    bool predictor_dirty = true;

    while (step < T_enc_new && iter < max_iters) {
        ++iter;

        const float * decoder_out;
        if (predictor_dirty) {
            decoder_out = predictor_step_ggml(w.predictor, pg, last_token, state, next_state, scratch_x);
            if (decoder_out == nullptr) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            predictor_dirty = false;
        } else {
            decoder_out = next_state.h.back().data();
        }

        const float * enc_proj = enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h);
        if (!joint_step(w.joint, jg, enc_proj, decoder_out, logits)) {
            return TRANSCRIBE_ERR_BACKEND;
        }

        const float * token_logits = logits.data();
        const int     pred_token   = argmax_range_masked(token_logits, n_token_cls, blocked_tokens);

        const bool is_blank = (pred_token == blank_id);
        if (is_blank) {
            step += 1;
            new_symbols = 0;
        } else {
            const float p = token_confidence(token_logits, n_token_cls, scratch_probs);
            TdtToken    tok;
            tok.id              = pred_token;
            tok.p               = p;
            tok.step_at_emit    = frame_offset + step;
            tok.duration_frames = 1;
            out_tokens.push_back(tok);

            last_token = pred_token;
            std::swap(state, next_state);
            predictor_dirty = true;

            new_symbols += 1;
            if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
                step += 1;
                new_symbols = 0;
            }
        }
    }

    if (iter >= max_iters) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder (rnnt-stream): hit iteration cap (%d)", max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Commit final state. After the loop, `state` is the post-last-
    // non-blank state (the predictor's "current" since the last emit).
    state_io      = std::move(state);
    last_token_io = last_token;
    return TRANSCRIBE_OK;
}

// CTC greedy decode.
//
// Per-frame: logits[t] = W @ enc[t] + b -> log_softmax -> argmax.
// Collapse: drop adjacent duplicates ("aaab" -> "ab"), then drop blanks
// (standard CTC greedy, cf. NeMo GreedyCTCInfer).
transcribe_status decode_ctc_greedy(const HostDecoderWeights &   w,
                                    const float *                enc_out,
                                    int                          T_enc,
                                    int                          d_enc,
                                    int                          n_threads,
                                    std::vector<TdtToken> &      out_tokens,
                                    const std::vector<uint8_t> * blocked_tokens) {
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.ctc_head.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (ctc): enc d_model mismatch "
                "(got %d, expected %d)",
                d_enc, w.ctc_head.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int n_classes = w.ctc_head.n_classes;
    const int blank_id  = w.ctc_head.blank_id;

    // Compute T_enc frames of logits: [T_enc, n_classes] = enc_out[T_enc, d_enc] @ W^T + b.
    std::vector<float> logits_all(static_cast<size_t>(T_enc) * static_cast<size_t>(n_classes));

    const int64_t t_proj_start = ggml_time_us();
#if TRANSCRIBE_HAS_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T_enc, n_classes, d_enc, 1.0f, enc_out, d_enc,
                w.ctc_head.weight.data(), d_enc, 0.0f, logits_all.data(), n_classes);
#else
    // No BLAS: project the frames in parallel over parallel_for_all, four
    // frames per pass over the weight. Each class sums W[r][0]*x[0],
    // W[r][1]*x[1], ... from 0 in that order, as a row-by-row dot would;
    // walking the transposed weight keeps that order per class
    // (bit-identical) while the inner loop runs across classes, which
    // compilers vectorize without reordering a sum (a dot product's
    // reduction they do not, short of fast-math). Sharing each weight load
    // across four frames keeps the pass from being bound by weight traffic.
    // Bias folded in below.
    {
        constexpr int kFrames  = 4;
        const int     nt       = resolve_decode_threads(n_threads);
        const float * Wt       = w.ctc_head.weight_t.data();
        const int     n_blocks = (T_enc + kFrames - 1) / kFrames;
        transcribe::parallel_for_all(n_blocks, nt, [&](int blk) {
            const int     t0 = blk * kFrames;
            const int     nb = std::min(kFrames, T_enc - t0);
            const float * x[kFrames];
            float *       y[kFrames];
            for (int b = 0; b < nb; ++b) {
                x[b] = enc_out + static_cast<size_t>(t0 + b) * static_cast<size_t>(d_enc);
                y[b] = logits_all.data() + static_cast<size_t>(t0 + b) * static_cast<size_t>(n_classes);
                std::fill(y[b], y[b] + n_classes, 0.0f);
            }
            for (int c = 0; c < d_enc; ++c) {
                const float * __restrict col = Wt + static_cast<size_t>(c) * static_cast<size_t>(n_classes);
                if (nb == kFrames) {
                    const float x0        = x[0][c];
                    const float x1        = x[1][c];
                    const float x2        = x[2][c];
                    const float x3        = x[3][c];
                    float * __restrict y0 = y[0];
                    float * __restrict y1 = y[1];
                    float * __restrict y2 = y[2];
                    float * __restrict y3 = y[3];
                    for (int r = 0; r < n_classes; ++r) {
                        const float wr = col[r];
                        y0[r] += wr * x0;
                        y1[r] += wr * x1;
                        y2[r] += wr * x2;
                        y3[r] += wr * x3;
                    }
                } else {
                    for (int b = 0; b < nb; ++b) {
                        const float xb        = x[b][c];
                        float * __restrict yb = y[b];
                        for (int r = 0; r < n_classes; ++r) {
                            yb[r] += col[r] * xb;
                        }
                    }
                }
            }
            return true;
        });
    }
#endif
    for (int t = 0; t < T_enc; ++t) {
        float * row = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
        for (int c = 0; c < n_classes; ++c) {
            row[c] += w.ctc_head.bias[static_cast<size_t>(c)];
        }
    }
    const int64_t t_proj_us = ggml_time_us() - t_proj_start;

    // Per-frame log-softmax in place. The reference dumps log-probs,
    // not raw logits.
    for (int t = 0; t < T_enc; ++t) {
        float * row   = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
        float   max_v = row[0];
        for (int c = 1; c < n_classes; ++c) {
            if (row[c] > max_v) {
                max_v = row[c];
            }
        }
        double sum = 0.0;
        for (int c = 0; c < n_classes; ++c) {
            sum += std::exp(static_cast<double>(row[c] - max_v));
        }
        const float log_sum = static_cast<float>(std::log(sum)) + max_v;
        for (int c = 0; c < n_classes; ++c) {
            row[c] -= log_sum;
        }
    }

    // Optional dumps. Frame 0 is shape [n_classes]; full is
    // [T_enc, n_classes] in row-major (== ggml ne fast-to-slow
    // [n_classes, T_enc]).
    if (transcribe::debug::enabled()) {
        const long long s_n = n_classes;
        transcribe::debug::dump_host_f32("dec.ctc.logprobs.0", logits_all.data(), s_n, &s_n, 1,
                                         "decoder.ctc.logprobs.0");
        const long long shape_full[2] = { T_enc, n_classes };
        transcribe::debug::dump_host_f32("dec.ctc.logprobs", logits_all.data(),
                                         static_cast<long long>(logits_all.size()), shape_full, 2,
                                         "decoder.ctc.logprobs");
    }

    // Greedy collapse: drop runs of identical labels, then drop blanks.
    // Per-emit `step_at_emit` is the encoder frame at which the label
    // first changed; duration_frames = 1.
    int prev_label = -1;
    for (int t = 0; t < T_enc; ++t) {
        const float * row   = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
        const int     label = argmax_range_masked(row, n_classes, blocked_tokens);
        if (label == prev_label) {
            continue;  // run-length collapse
        }
        prev_label = label;
        if (label == blank_id) {
            continue;
        }
        TdtToken tok;
        tok.id              = label;
        // CTC greedy confidence = exp(top log-prob); this is the
        // model's own probability for the chosen label, in [0, 1].
        tok.p               = std::exp(row[label]);
        tok.step_at_emit    = t;
        tok.duration_frames = 1;
        out_tokens.push_back(tok);
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "decoder (ctc): T_enc=%d  proj=%.1f ms  emitted=%d", T_enc, t_proj_us / 1000.0,
            static_cast<int>(out_tokens.size()));

    return TRANSCRIBE_OK;
}

}  // namespace transcribe::parakeet
