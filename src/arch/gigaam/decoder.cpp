// arch/gigaam/decoder.cpp - GigaAM RNN-T / CTC greedy decode on host.
//
// The decoder is tiny (1 LSTM layer of 320, single joint linear) so per-step
// compute fits in CPU cache; running on host avoids backend dispatch overhead
// for the inner symbol-loop. PyTorch nn.LSTM stores gate weights concatenated
// [4*hidden, hidden] in (i, f, g, o) order; the converter emits Wx/Wh in that
// layout and collapses bias_ih + bias_hh into one [4*hidden] bias, which the
// host LSTM step reads directly.

#include "decoder.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "gigaam.h"
#include "transcribe-debug.h"
#include "transcribe-log.h"
#include "weights.h"

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
#include <cstring>
#include <vector>

namespace transcribe::gigaam {

namespace {

// Copy a ggml tensor's data into a host f32 vector, dequantizing if needed.
// F32 takes the fast path; F16/BF16/Q*/IQ* stage raw bytes off the backend and
// walk ggml's to_float trait. Runs once at load time, so the per-step hot path
// pays nothing.
void readback_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (t == nullptr) {
        out.clear();
        return;
    }
    const size_t n = static_cast<size_t>(ggml_nelements(t));
    out.assign(n, 0.0f);
    if (n == 0) {
        return;
    }

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        return;
    }

    const auto * tt = ggml_get_type_traits(t->type);
    if (tt == nullptr || tt->to_float == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "gigaam decoder: tensor \"%s\" type %s has no to_float", t->name,
                ggml_type_name(t->type));
        out.clear();
        return;
    }
    const size_t         nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    tt->to_float(raw.data(), out.data(), static_cast<int64_t>(n));
}

inline float sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

#if TRANSCRIBE_HAS_BLAS
// y = W @ x + b, where W is [out, in] row-major (numpy shape), x is
// [in], b is [out]. b may be nullptr to skip the add.
void matvec_add(const float * W, const float * b, const float * x, int out, int in_dim, float * y) {
    if (b != nullptr) {
        std::memcpy(y, b, static_cast<size_t>(out) * sizeof(float));
    } else {
        std::fill(y, y + out, 0.0f);
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out, in_dim, 1.0f, W, in_dim, x, 1, 1.0f, y, 1);
}
#else
// y = W @ x + b with W stored transposed: Wt is [in, out] row-major, x is
// [in], b is [out] or nullptr. Each y[i] still adds b[i] (or 0), then
// W[i][0]*x[0], W[i][1]*x[1], ... in that order, one multiply and one add
// per term, exactly like a row-by-row dot product, so the result is the
// same bit for bit. The inner loop runs across outputs instead of along a
// dot product, so compilers vectorize it without reordering any sum.
void matvec_t_add(const float * Wt, const float * b, const float * x, int out, int in_dim, float * y_out) {
    float * __restrict y = y_out;
    if (b != nullptr) {
        std::memcpy(y, b, static_cast<size_t>(out) * sizeof(float));
    } else {
        std::fill(y, y + out, 0.0f);
    }
    for (int j = 0; j < in_dim; ++j) {
        const float xj               = x[j];
        const float * __restrict col = Wt + static_cast<size_t>(j) * out;
        for (int i = 0; i < out; ++i) {
            y[i] += col[i] * xj;
        }
    }
}

// Row-major [rows, cols] to [cols, rows]. Empty when the size is not
// rows * cols.
std::vector<float> transposed(const std::vector<float> & m, int rows, int cols) {
    std::vector<float> t;
    if (rows <= 0 || cols <= 0 || m.size() != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
        return t;
    }
    t.resize(m.size());
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            t[static_cast<size_t>(c) * rows + r] = m[static_cast<size_t>(r) * cols + c];
        }
    }
    return t;
}
#endif

// y = W @ x + b for a matrix kept for the active path: BLAS reads the
// row-major copy, the portable loop the transposed one.
inline void linear(const std::vector<float> & W,
                   const std::vector<float> & Wt,
                   const float *              b,
                   const float *              x,
                   int                        out,
                   int                        in_dim,
                   float *                    y) {
#if TRANSCRIBE_HAS_BLAS
    (void) Wt;
    matvec_add(W.data(), b, x, out, in_dim, y);
#else
    (void) W;
    matvec_t_add(Wt.data(), b, x, out, in_dim, y);
#endif
}

}  // namespace

// Build host buffers for the predictor + joint weights. Reads back from
// the backend exactly once at load time; the decode loop then iterates
// the host floats.
transcribe_status build_host_decoder_weights(const GigaamWeights & w,
                                             const GigaamHParams & hp,
                                             HostDecoderWeights &  out) {
    if (hp.head_kind == HeadKind::CTC) {
        readback_f32(w.ctc_head.weight, out.ctc_w);
        readback_f32(w.ctc_head.bias, out.ctc_b);
#if !TRANSCRIBE_HAS_BLAS
        // decode_ctc_greedy validates ctc_w's size, which this copy shares.
        out.ctc_w_t = transposed(out.ctc_w, hp.head_n_classes, hp.enc_d_model);
#endif
        return TRANSCRIBE_OK;
    }
    // RNN-T.
    readback_f32(w.predictor.embed_w, out.pred_embed);
    out.lstm_Wx.resize(hp.pred_n_layers);
    out.lstm_Wh.resize(hp.pred_n_layers);
    out.lstm_b.resize(hp.pred_n_layers);
    for (int i = 0; i < hp.pred_n_layers; ++i) {
        readback_f32(w.predictor.lstm[i].Wx, out.lstm_Wx[i]);
        readback_f32(w.predictor.lstm[i].Wh, out.lstm_Wh[i]);
        readback_f32(w.predictor.lstm[i].b, out.lstm_b[i]);
    }
    readback_f32(w.joint.enc_w, out.joint_enc_w);
    readback_f32(w.joint.enc_b, out.joint_enc_b);
    readback_f32(w.joint.pred_w, out.joint_pred_w);
    readback_f32(w.joint.pred_b, out.joint_pred_b);
    readback_f32(w.joint.out_w, out.joint_out_w);
    readback_f32(w.joint.out_b, out.joint_out_b);
#if !TRANSCRIBE_HAS_BLAS
    const int H       = hp.pred_hidden;
    const int joint_h = hp.joint_hidden;
    out.lstm_Wx_t.resize(hp.pred_n_layers);
    out.lstm_Wh_t.resize(hp.pred_n_layers);
    bool shapes_ok = true;
    for (int i = 0; i < hp.pred_n_layers; ++i) {
        out.lstm_Wx_t[i] = transposed(out.lstm_Wx[i], 4 * H, H);
        out.lstm_Wh_t[i] = transposed(out.lstm_Wh[i], 4 * H, H);
        shapes_ok        = shapes_ok && !out.lstm_Wx_t[i].empty() && !out.lstm_Wh_t[i].empty();
    }
    out.joint_enc_w_t  = transposed(out.joint_enc_w, joint_h, hp.enc_d_model);
    out.joint_pred_w_t = transposed(out.joint_pred_w, joint_h, H);
    out.joint_out_w_t  = transposed(out.joint_out_w, hp.joint_n_classes, joint_h);
    if (!shapes_ok || out.joint_enc_w_t.empty() || out.joint_pred_w_t.empty() || out.joint_out_w_t.empty()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "gigaam decoder: predictor/joint tensor sizes do not match the hparams");
        return TRANSCRIBE_ERR_GGUF;
    }
#endif
    return TRANSCRIBE_OK;
}

// One LSTM step (single batch). Returns the new hidden vector (= LSTM
// output) into `h_out`, and the new cell state into `c_out`. Inputs:
//   x:      [pred_hidden] embedding (or zero on the first step)
//   h_prev: [pred_hidden]
//   c_prev: [pred_hidden]
//   host:   layer 0's Wx, Wh ([4*pred_hidden, pred_hidden] row-major, or
//           their transposes without BLAS) and b ([4*pred_hidden], already
//           collapsed bias_ih + bias_hh)
// Gate order is PyTorch (i, f, g, o), each block size `pred_hidden`.
static void lstm_step(const float *              x,
                      const float *              h_prev,
                      const float *              c_prev,
                      const HostDecoderWeights & host,
                      int                        H,
                      float *                    h_out,
                      float *                    c_out,
                      float *                    scratch_gates) {
    // scratch_gates: [4*H]. Compute Wx @ x + Wh @ h_prev + b in place.
    const float * b = host.lstm_b[0].data();
#if TRANSCRIBE_HAS_BLAS
    std::memcpy(scratch_gates, b, static_cast<size_t>(4 * H) * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 4 * H, H, 1.0f, host.lstm_Wx[0].data(), H, x, 1, 1.0f, scratch_gates, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 4 * H, H, 1.0f, host.lstm_Wh[0].data(), H, h_prev, 1, 1.0f, scratch_gates,
                1);
#else
    // Per gate: b, then + (Wx[i][j]*x[j] + Wh[i][j]*h[j]) for j = 0, 1, ...;
    // the same terms in the same order as a row-by-row loop (see
    // matvec_t_add), here walking the transposed weights.
    float * __restrict gates = scratch_gates;
    std::memcpy(gates, b, static_cast<size_t>(4 * H) * sizeof(float));
    for (int j = 0; j < H; ++j) {
        const float xj              = x[j];
        const float hj              = h_prev[j];
        const float * __restrict wx = host.lstm_Wx_t[0].data() + static_cast<size_t>(j) * 4 * H;
        const float * __restrict wh = host.lstm_Wh_t[0].data() + static_cast<size_t>(j) * 4 * H;
        for (int i = 0; i < 4 * H; ++i) {
            gates[i] += wx[i] * xj + wh[i] * hj;
        }
    }
#endif

    const float * gi = scratch_gates;
    const float * gf = scratch_gates + H;
    const float * gg = scratch_gates + 2 * H;
    const float * go = scratch_gates + 3 * H;

    for (int j = 0; j < H; ++j) {
        const float i_t   = sigmoidf(gi[j]);
        const float f_t   = sigmoidf(gf[j]);
        const float g_t   = std::tanh(gg[j]);
        const float o_t   = sigmoidf(go[j]);
        const float c_new = f_t * c_prev[j] + i_t * g_t;
        c_out[j]          = c_new;
        h_out[j]          = o_t * std::tanh(c_new);
    }
}

// One joint forward pass for batch=1, using a precomputed encoder
// projection (enc_proj = enc_w @ f_t + enc_b, hoisted out of the time
// loop in `decode_rnnt_greedy`):
//   pred_proj = pred_w @ g_t + pred_b          [joint_hidden]
//   logits    = out_w @ relu(enc_proj + pred_proj) + out_b   [n_classes]
//
// Returns the argmax token id.
static int joint_argmax(const HostDecoderWeights & h,
                        const float *              enc_proj,  // [joint_h]
                        const float *              g_t,       // [pred_hidden]
                        int                        pred_d,
                        int                        joint_h,
                        int                        n_classes,
                        std::vector<float> &       scratch_join,  // size joint_h
                        std::vector<float> &       scratch_logits) {
    scratch_join.assign(joint_h, 0.0f);
    scratch_logits.assign(n_classes, 0.0f);

    // pred_proj into scratch_join, then fuse `+ enc_proj` and ReLU.
    linear(h.joint_pred_w, h.joint_pred_w_t, h.joint_pred_b.data(), g_t, joint_h, pred_d, scratch_join.data());
    for (int i = 0; i < joint_h; ++i) {
        float v         = scratch_join[i] + enc_proj[i];
        scratch_join[i] = v > 0.0f ? v : 0.0f;
    }
    // logits = out_w @ relu + out_b. The reference applies log_softmax
    // but argmax is invariant under it, so we skip.
    linear(h.joint_out_w, h.joint_out_w_t, h.joint_out_b.data(), scratch_join.data(), n_classes, joint_h,
           scratch_logits.data());

    int   best_idx = 0;
    float best_val = scratch_logits[0];
    for (int i = 1; i < n_classes; ++i) {
        if (scratch_logits[i] > best_val) {
            best_val = scratch_logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

// Score consecutive encoder frames while the predictor state is fixed.
// A blank does not update that state, so the scores stay valid through the
// first non-blank token. One matrix-vector operation projects the predictor,
// and one matrix operation projects all joint rows.
static void joint_argmax_span(const HostDecoderWeights & h,
                              const float *              enc_proj,
                              const float *              g_t,
                              int                        n_frames,
                              int                        pred_d,
                              int                        joint_h,
                              int                        n_classes,
                              std::vector<float> &       pred_proj,
                              std::vector<float> &       join,
                              std::vector<float> &       logits,
                              std::vector<int> &         out_tokens) {
    pred_proj.assign(joint_h, 0.0f);
    join.resize(static_cast<size_t>(n_frames) * joint_h);
    logits.resize(static_cast<size_t>(n_frames) * n_classes);
    out_tokens.resize(n_frames);

    linear(h.joint_pred_w, h.joint_pred_w_t, h.joint_pred_b.data(), g_t, joint_h, pred_d, pred_proj.data());
    for (int t = 0; t < n_frames; ++t) {
        const float * enc_row  = enc_proj + static_cast<size_t>(t) * joint_h;
        float *       join_row = join.data() + static_cast<size_t>(t) * joint_h;
        for (int j = 0; j < joint_h; ++j) {
            const float value = enc_row[j] + pred_proj[j];
            join_row[j]       = value > 0.0f ? value : 0.0f;
        }
    }

#if TRANSCRIBE_HAS_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_frames, n_classes, joint_h, 1.0f, join.data(), joint_h,
                h.joint_out_w.data(), joint_h, 0.0f, logits.data(), n_classes);
#else
    for (int t = 0; t < n_frames; ++t) {
        matvec_t_add(h.joint_out_w_t.data(), nullptr, join.data() + static_cast<size_t>(t) * joint_h, n_classes,
                     joint_h, logits.data() + static_cast<size_t>(t) * n_classes);
    }
#endif

    // Add the output bias during argmax instead of writing it to every row.
    for (int t = 0; t < n_frames; ++t) {
        float * row      = logits.data() + static_cast<size_t>(t) * n_classes;
        int     best_idx = 0;
        float   best_val = row[0] + h.joint_out_b[0];
        for (int i = 1; i < n_classes; ++i) {
            const float value = row[i] + h.joint_out_b[i];
            if (value > best_val) {
                best_val = value;
                best_idx = i;
            }
        }
        out_tokens[t] = best_idx;
    }
}

// Run RNN-T greedy decode against the encoder output. `encoded` is a
// host-side buffer laid out [T_enc * d_model] in T-major order (each
// contiguous d_model-sized chunk is one time step's encoder vector) —
// matches the rnnt.encoded reference dump.
//
// Returns the emitted token sequence (excluding blanks) and the
// per-token encoder frame indices.
transcribe_status decode_rnnt_greedy(const HostDecoderWeights & host,
                                     const GigaamHParams &      hp,
                                     const float *              encoded,
                                     int                        T_enc,
                                     int                        max_symbols_per_step,
                                     std::vector<int> &         out_tokens,
                                     std::vector<int> &         out_frames) {
    out_tokens.clear();
    out_frames.clear();

    const int H        = hp.pred_hidden;
    const int enc_d    = hp.enc_d_model;
    const int joint_h  = hp.joint_hidden;
    const int n_class  = hp.joint_n_classes;
    const int blank_id = n_class - 1;

    // Precompute encoder projection for every frame in one BLAS sgemm
    // (or a fallback loop). The joint's enc_w @ f_t + enc_b is independent
    // of decode state, so hoisting it out of the inner loop eliminates
    // O(iters) redundant matvecs and gives BLAS a single large matrix.
    std::vector<float> enc_proj_all(static_cast<size_t>(T_enc) * static_cast<size_t>(joint_h));
#if TRANSCRIBE_HAS_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T_enc, joint_h, enc_d, 1.0f, encoded, enc_d,
                host.joint_enc_w.data(), enc_d, 0.0f, enc_proj_all.data(), joint_h);
#else
    for (int t = 0; t < T_enc; ++t) {
        const float * frame = encoded + static_cast<size_t>(t) * enc_d;
        float *       proj  = enc_proj_all.data() + static_cast<size_t>(t) * joint_h;
        matvec_t_add(host.joint_enc_w_t.data(), nullptr, frame, joint_h, enc_d, proj);
    }
#endif
    for (int t = 0; t < T_enc; ++t) {
        float * proj = enc_proj_all.data() + static_cast<size_t>(t) * joint_h;
        for (int j = 0; j < joint_h; ++j) {
            proj[j] += host.joint_enc_b[j];
        }
    }

    std::vector<float> h_cur(H, 0.0f);
    std::vector<float> c_cur(H, 0.0f);
    std::vector<float> h_next(H, 0.0f);
    std::vector<float> c_next(H, 0.0f);
    std::vector<float> gates(4 * H, 0.0f);
    std::vector<float> pred_in(H, 0.0f);  // embed lookup output
    std::vector<float> jh(joint_h, 0.0f);
    std::vector<float> logits(n_class, 0.0f);
    std::vector<float> span_pred_proj;
    std::vector<float> span_join;
    std::vector<float> span_logits;
    std::vector<int>   span_tokens;

    bool fresh = true;  // first step uses zero input + zero state (reference predict(None, None))

    // Predictor-dirty cache: on a blank emission `(prev_token, h_cur, c_cur)`
    // are unchanged, so the next LSTM step would deterministically produce
    // the same `h_next`. Skip the unroll and reuse `h_next`; recompute on
    // emit (where state and last_token change). Bit-exact with the
    // unconditional path.
    bool predictor_dirty = true;

    // Speculate over a short span. Consume all leading blank frames, but stop
    // at the first token because it changes the predictor state. The span
    // starts short after a token (a character model emits densely, so the
    // rest of a long span is scored for nothing) and doubles across blank
    // runs. A frame's scores do not depend on the span, so neither does the
    // result.
    constexpr int k_span_min = 4;
    constexpr int k_span_max = 32;
    int           lookahead  = k_span_min;
    int           t          = 0;
    while (t < T_enc) {
        if (predictor_dirty) {
            if (fresh) {
                std::fill(pred_in.begin(), pred_in.end(), 0.0f);
            } else {
                const int prev = out_tokens.back();
                std::memcpy(pred_in.data(), host.pred_embed.data() + static_cast<size_t>(prev) * H, H * sizeof(float));
            }
            lstm_step(pred_in.data(), h_cur.data(), c_cur.data(), host, H, h_next.data(), c_next.data(), gates.data());
            predictor_dirty = false;
        }

        const int span = std::min(lookahead, T_enc - t);
        joint_argmax_span(host, enc_proj_all.data() + static_cast<size_t>(t) * joint_h, h_next.data(), span, H, joint_h,
                          n_class, span_pred_proj, span_join, span_logits, span_tokens);

        int offset = 0;
        while (offset < span && span_tokens[offset] == blank_id) {
            ++offset;
        }
        if (offset == span) {
            t += span;
            lookahead = std::min(2 * lookahead, k_span_max);
            continue;
        }
        lookahead = k_span_min;

        // Commit the first token and candidate predictor state at its frame.
        t += offset;
        int tok = span_tokens[offset];
        out_tokens.push_back(tok);
        out_frames.push_back(t);
        h_cur           = h_next;
        c_cur           = c_next;
        fresh           = false;
        predictor_dirty = true;

        // A frame may emit more than one token. Continue with scalar scoring
        // on this frame until blank or max_symbols_per_step advances time.
        int sym_count = 1;
        while (max_symbols_per_step <= 0 || sym_count < max_symbols_per_step) {
            const int prev = out_tokens.back();
            std::memcpy(pred_in.data(), host.pred_embed.data() + static_cast<size_t>(prev) * H, H * sizeof(float));
            lstm_step(pred_in.data(), h_cur.data(), c_cur.data(), host, H, h_next.data(), c_next.data(), gates.data());
            predictor_dirty     = false;
            const float * enc_p = enc_proj_all.data() + static_cast<size_t>(t) * joint_h;
            tok                 = joint_argmax(host, enc_p, h_next.data(), H, joint_h, n_class, jh, logits);
            if (tok == blank_id) {
                // A blank advances time without committing the candidate
                // predictor state. The next frame can reuse h_next.
                break;
            }
            // A token commits the candidate state and invalidates the cache.
            out_tokens.push_back(tok);
            out_frames.push_back(t);
            h_cur           = h_next;
            c_cur           = c_next;
            predictor_dirty = true;
            ++sym_count;
        }
        ++t;
    }

    return TRANSCRIBE_OK;
}

// CTC greedy decode. The 1×1 Conv1d head is a per-frame linear:
//   logits[t, :] = ctc_w @ encoded[t, :] + ctc_b
// Layout of host.ctc_w after readback: ggml ne=[1, d_model, n_classes]
// is byte-equivalent to a row-major [n_classes, d_model] matrix (the
// k=1 axis collapses), which is what linear() expects.
transcribe_status decode_ctc_greedy(const HostDecoderWeights & host,
                                    const GigaamHParams &      hp,
                                    const float *              encoded,
                                    int                        T_enc,
                                    std::vector<int> &         out_tokens,
                                    std::vector<int> &         out_frames) {
    out_tokens.clear();
    out_frames.clear();

    const int d_model   = hp.enc_d_model;
    const int n_classes = hp.head_n_classes;
    const int blank_id  = n_classes - 1;

    if (encoded == nullptr || T_enc <= 0 || d_model <= 0 || n_classes <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (static_cast<int>(host.ctc_w.size()) != n_classes * d_model ||
        static_cast<int>(host.ctc_b.size()) != n_classes) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "gigaam decoder (ctc): host head shape mismatch "
                "(ctc_w=%zu expected=%d, ctc_b=%zu expected=%d)",
                host.ctc_w.size(), n_classes * d_model, host.ctc_b.size(), n_classes);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // logits[T_enc, n_classes] row-major.
    std::vector<float> logits(static_cast<size_t>(T_enc) * static_cast<size_t>(n_classes), 0.0f);
    for (int t = 0; t < T_enc; ++t) {
        const float * frame = encoded + static_cast<size_t>(t) * d_model;
        float *       row   = logits.data() + static_cast<size_t>(t) * n_classes;
        linear(host.ctc_w, host.ctc_w_t, host.ctc_b.data(), frame, n_classes, d_model, row);
    }

    // Raw logits before log_softmax, [T_enc, n_classes] row-major.
    if (transcribe::debug::enabled()) {
        const long long shape[2] = { T_enc, n_classes };
        transcribe::debug::dump_host_f32("ctc.logits.raw", logits.data(), static_cast<long long>(logits.size()), shape,
                                         2, "decoder.ctc.logits_raw");
    }

    // Per-frame log_softmax in place.
    for (int t = 0; t < T_enc; ++t) {
        float * row   = logits.data() + static_cast<size_t>(t) * n_classes;
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

    if (transcribe::debug::enabled()) {
        const long long shape[2] = { T_enc, n_classes };
        transcribe::debug::dump_host_f32("ctc.log_probs", logits.data(), static_cast<long long>(logits.size()), shape,
                                         2, "decoder.ctc.log_probs");
    }

    // Greedy collapse: drop runs of the same label, then drop blanks.
    int prev_label = -1;
    for (int t = 0; t < T_enc; ++t) {
        const float * row    = logits.data() + static_cast<size_t>(t) * n_classes;
        int           best   = 0;
        float         best_v = row[0];
        for (int c = 1; c < n_classes; ++c) {
            if (row[c] > best_v) {
                best_v = row[c];
                best   = c;
            }
        }
        if (best == prev_label) {
            continue;  // run-length collapse
        }
        prev_label = best;
        if (best == blank_id) {
            continue;
        }
        out_tokens.push_back(best);
        out_frames.push_back(t);
    }

    return TRANSCRIBE_OK;
}

}  // namespace transcribe::gigaam
