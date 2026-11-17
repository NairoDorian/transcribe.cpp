// src/granite_conformer/shaw_attn.cpp - shared Shaw block-local attention.
//
// See shaw_attn.h for the contract.

#include "shaw_attn.h"

#include "conformer/conformer.h"
#include "ggml.h"
#include "transcribe-log.h"

#include <cmath>
#include <cstdio>

namespace transcribe::granite_conformer {

namespace {

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * gamma, ggml_tensor * beta, float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y               = ggml_mul(ctx, y, gamma);
    if (beta != nullptr) {
        y = ggml_add(ctx, y, beta);
    }
    return y;
}

}  // namespace

ggml_tensor * shaw_block_attn(ggml_context *          ctx,
                              ggml_tensor *           x,
                              ggml_tensor *           zero_pad,
                              ggml_tensor *           dists,
                              ggml_tensor *           pad_mask_3d,
                              const ShawAttnWeights & w,
                              int                     n_heads,
                              int                     head_dim,
                              int                     context_size,
                              int                     num_blocks,
                              int                     T_enc,
                              float                   layer_norm_eps,
                              ggml_tensor *           pos_rows) {
    const int64_t inner_dim = static_cast<int64_t>(n_heads) * head_dim;
    const int64_t T_pad     = static_cast<int64_t>(context_size) * num_blocks;
    const float   scale     = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Offline batch: x ne=[d_model, T_enc, B]. Block-local attention treats
    // each time-block independently, so B utterances fold into the block
    // axis: num_blocks_eff = num_blocks * B (memory order [.., T_pad, B]
    // splits cleanly into (ctx, num_blocks*B)). The caller tiles the per-block
    // pad mask across B and sizes zero_pad with the batch.
    const int64_t B              = x->ne[2];
    const int64_t num_blocks_eff = static_cast<int64_t>(num_blocks) * B;

    // Pre-LayerNorm on the T_enc input.
    ggml_tensor * h = layer_norm(ctx, x, w.norm_attn_w, w.norm_attn_b, layer_norm_eps);

    // Zero-pad along the time axis when T_enc is not block-aligned. The
    // pad buffer is a caller-owned graph input; nullptr is a programmer
    // error.
    if (T_pad > T_enc) {
        if (zero_pad == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "granite_conformer::shaw_block_attn: "
                    "zero_pad is null but T_pad > T_enc");
            return nullptr;
        }
        h = ggml_concat(ctx, h, zero_pad, /*dim=*/1);
    }
    // h ne = [d_model, T_pad]

    // QKV projections. q has no bias; kv is fused (2*inner_dim) with no
    // bias; out has bias.
    ggml_tensor * q  = ggml_mul_mat(ctx, w.attn_q_w, h);   // [inner_dim, T_pad]
    ggml_tensor * kv = ggml_mul_mat(ctx, w.attn_kv_w, h);  // [2*inner_dim, T_pad]

    // Block-local form. Target ne layout [head_dim, context_size, n_heads,
    // num_blocks_eff] so that ggml_mul_mat(k, q) does per-(head, block,
    // utterance) batched attention. ne[2] of q/kv carries the utterance batch
    // (1 for single-shot, == B otherwise); its [.., T_pad, B] memory order
    // splits cleanly into (context_size, num_blocks_eff).
    auto reshape_qkv = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * r = ggml_reshape_4d(ctx, t, head_dim, n_heads, context_size, num_blocks_eff);
        r               = ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));
        return r;
    };
    // k and v are read straight out of the fused kv rows ([2*inner_dim,
    // T_pad, B], contiguous) as [head_dim, n_heads, context_size,
    // num_blocks_eff] views, so each takes one copy into its final layout
    // instead of a de-interleaving copy first (copies move values exactly).
    auto kv_half = [&](size_t offset) -> ggml_tensor * {
        return ggml_view_4d(ctx, kv, head_dim, n_heads, context_size, num_blocks_eff, head_dim * ggml_element_size(kv),
                            kv->nb[1], context_size * kv->nb[1], offset);
    };
    q                 = reshape_qkv(q);
    ggml_tensor * k   = ggml_cont(ctx, ggml_permute(ctx, kv_half(0), 0, 2, 1, 3));
    // V is only ever used transposed: [context_size, head_dim, n_heads,
    // num_blocks_eff], ne[0] = key position, for out = V^T @ attn below.
    ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, kv_half(inner_dim * ggml_element_size(kv)), 1, 2, 0, 3));

    // QK^T: ggml_mul_mat(K, Q), both [head_dim, context_size, n_heads,
    // num_blocks] -> [k=context_size, q=context_size, n_heads, num_blocks].
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

    // Shaw positional bias, in kq's axis order. See the header for why
    // there are two forms.
    ggml_tensor * pos_attn = nullptr;
    if (pos_rows != nullptr) {
        // Skew. rel_pos_emb rows for the 2*context_size - 1 distinct
        // relative offsets, then one mul_mat against q:
        //   bd[d, q, h, blk] = sum_i q[i, q, h, blk] * e_sub[i, d]
        // rel_shift rotates that into out[k, q] = bd[k - q + ctx - 1, q],
        // i.e. the bias for key k against query q. Output already matches
        // kq's [k, q, n_heads, num_blocks_eff] layout, so no permute.
        ggml_tensor * e_sub = ggml_get_rows(ctx, w.attn_rel_pos_emb, pos_rows);
        pos_attn            = ggml_mul_mat(ctx, e_sub, q);
        pos_attn            = transcribe::conformer::rel_shift(ctx, pos_attn);
        // rel_shift keeps ne[0] = 2*ctx - 1; the trailing columns are the
        // out-of-block offsets plus the zero column the trick injects.
        // Contiguous-rows, which ggml_add accepts on CPU and Metal.
        pos_attn = ggml_view_4d(ctx, pos_attn, context_size, context_size, n_heads, num_blocks_eff, pos_attn->nb[1],
                                pos_attn->nb[2], pos_attn->nb[3], /*offset=*/0);
    } else {
        // Direct. rel_pos_emb [head_dim, 2*max_pos_emb+1], dists
        // [context_size, context_size] int32. Materialise the per-(c, r)
        // lookup as [head_dim, context_size*context_size], then reshape to
        // [head_dim, r=context_size, c=context_size].
        ggml_tensor * dists_flat = ggml_reshape_1d(ctx, dists, static_cast<int64_t>(context_size) * context_size);
        ggml_tensor * rel_lookup = ggml_get_rows(ctx, w.attn_rel_pos_emb, dists_flat);
        rel_lookup               = ggml_reshape_3d(ctx, rel_lookup, head_dim, context_size, context_size);

        // pos_attn[h, b, c, r] = sum_d q[h, b, c, d] * rel_lookup[c, r, d].
        // Permute q to put c on ne[2] for batched mul_mat against
        // rel_lookup; rel_lookup's ne[3]=1 broadcasts across num_blocks.
        ggml_tensor * q_perm = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        pos_attn             = ggml_mul_mat(ctx, rel_lookup, q_perm);

        // Align pos_attn's axis order to kq's before the add.
        pos_attn = ggml_cont(ctx, ggml_permute(ctx, pos_attn, 0, 2, 1, 3));
    }

    // softmax((kq + pos_attn) * scale + pad_mask) over the k axis (ne[0]).
    // The mask is per block ([ctx, ctx, num_blocks_eff], tiled across the
    // batch by the caller) and broadcasts across n_heads through ne[2] = 1.
    // soft_max_ext applies the scale and the mask in the same order as a
    // separate scale then add, and the mask holds only 0 and -inf, so the
    // scores are the same even where a backend fuses the multiply-add.
    ggml_tensor * scores      = ggml_add(ctx, kq, pos_attn);
    ggml_tensor * pad_mask_4d = ggml_reshape_4d(ctx, pad_mask_3d, context_size, context_size, 1, num_blocks_eff);
    ggml_tensor * attn        = ggml_soft_max_ext(ctx, scores, pad_mask_4d, scale, 0.0f);

    // out = V^T @ attn → [head_dim, q, n_heads, num_blocks].
    ggml_tensor * out = ggml_mul_mat(ctx, v_t, attn);

    // Reshape back to [inner_dim, T_pad, B].
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    out = ggml_reshape_3d(ctx, out, inner_dim, T_pad, B);

    // Slice off pad rows BEFORE out_proj. This mirrors the reference's
    // `out = self.to_out(out[:, :num_features, :])` order; it keeps
    // pad-row garbage from leaking through the post-attention depthwise
    // conv. With one utterance the leading rows are already contiguous, so
    // the view feeds out_proj directly.
    if (T_pad > T_enc) {
        out = ggml_view_3d(ctx, out, inner_dim, T_enc, B, out->nb[1], out->nb[2], 0);
        if (!ggml_is_contiguous(out)) {
            out = ggml_cont(ctx, out);
        }
    }

    // Final out_proj (with bias). Operates at T_enc.
    out = ggml_mul_mat(ctx, w.attn_out_w, out);
    if (w.attn_out_b != nullptr) {
        out = ggml_add(ctx, out, w.attn_out_b);
    }
    // out ne = [d_model, T_enc]
    return out;
}

}  // namespace transcribe::granite_conformer
