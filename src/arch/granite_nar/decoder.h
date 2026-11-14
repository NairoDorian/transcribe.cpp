// arch/granite_nar/decoder.h - bidirectional Granite-4 LM as a NAR editor.
//
// Reference: NLENARDecoder in the NLE HF repo's modeling_nle.py.
//
// Unlike the AR Granite Speech family, the NAR variant:
//   - sets `is_causal=False` on every decoder layer (BIDIRECTIONAL),
//   - does NOT run an autoregressive token loop — the entire transcript
//     is produced from one forward pass,
//   - does NOT use a KV cache (single-pass; no prefill/step split),
//   - feeds a flat embed sequence directly via `inputs_embeds`:
//       cat([proj_out / embedding_multiplier,
//            embed_tokens(text_with_eos_slots)])
//     where `text_with_eos_slots` is the initial-hypothesis BPE id
//     sequence with eos_id inserted between every pair of tokens (and
//     bracketing the sequence) — length 2n+1 for n input tokens. The
//     final transcript comes from argmax over the text-portion logits
//     followed by repeat-collapse + eos drop.
//
// The block math is identical to AR Granite-4: pre-RMSNorm GQA, NeoX
// RoPE, scaled-by-attention_multiplier softmax, residual_multiplier
// scaling on each residual, SwiGLU MLP. embedding_multiplier (×12) is
// applied AFTER the audio scatter, logits_scaling (÷8) divides the
// lm_head output, lm_head is tied to embed_tokens.
//
// This header is INTERNAL to src/arch/granite_nar/.

#pragma once

#include "ggml.h"
#include "weights.h"

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace transcribe::granite_nar {

struct DecoderDumps {
    ggml_tensor * flat_embeds = nullptr;  // [hidden, T_total] post-scatter,
                                          //  pre-embedding_multiplier
    ggml_tensor * text_logits = nullptr;  // [vocab, n_text] over text part only
};

struct ForwardBuild {
    // Graph inputs (caller uploads at compute time).
    ggml_tensor * audio_in     = nullptr;  // [hidden, n_audio_tokens] f32
                                           //  (already / embedding_multiplier)
    ggml_tensor * text_ids_in  = nullptr;  // [n_text] i32
    ggml_tensor * positions_in = nullptr;  // [T_total] i32 RoPE positions

    // Output (lm_head sliced over the text portion). Exactly one of the two
    // forms is built (see build_forward_graph's full_logits):
    ggml_tensor *              out = nullptr;  // [vocab, n_text] full logits
    std::vector<ggml_tensor *> text_ids;       // argmax per text row, in tiles
                                               //  of kTextTileRows, each [n] i32

    DecoderDumps  dumps{};
    ggml_cgraph * graph = nullptr;

    int n_audio_tokens = 0;
    int n_text         = 0;
    int T_total        = 0;
};

// Text rows per LM-head tile in the argmax form: bounds the
// [vocab, rows] f32 logits to ~103 MB (100352 x 256).
constexpr int kTextTileRows = 256;

// Build the bidirectional editor forward graph. n_text is the length of
// the post-insertion-slots text sequence (2 * n_hyp + 1).
//
// use_flash: attention through flash_attn_ext (GQA-native, no
// [T_total, T_total, n_heads] scores) instead of mul_mat + soft_max_ext.
// full_logits: emit the whole [vocab, n_text] `dec.text_logits` (needed for
// tensor dumps). Otherwise the LM head runs in tiles of kTextTileRows with
// the argmax in-graph and only ids reach the host; the transcript is the
// same because argmax is invariant to the positive logits_scaling.
ForwardBuild build_forward_graph(ggml_context *            ctx,
                                 const GraniteNarWeights & weights,
                                 const GraniteNarHParams & hp,
                                 int                       n_audio_tokens,
                                 int                       n_text,
                                 bool                      use_flash,
                                 bool                      full_logits);

// Host-side helper: insert eos_id between every adjacent pair (and at
// the head/tail) of `hyp_ids`. Length grows from n to 2n + 1.
void add_insertion_slots(const std::vector<int32_t> & hyp_ids, int32_t eos_id, std::vector<int32_t> & out);

// Host-side argmax+collapse decoder: walks the text-portion logits row
// by row, picks the argmax per row, collapses consecutive duplicates,
// drops eos_id. Returns the final token id sequence.
void argmax_collapse_drop_eos(const std::vector<float> & text_logits,
                              int                        vocab,
                              int                        n_text,
                              int32_t                    eos_id,
                              std::vector<int32_t> &     out_ids);

// The collapse half of argmax_collapse_drop_eos, over per-row argmax ids.
void collapse_drop_eos(const std::vector<int32_t> & row_ids, int32_t eos_id, std::vector<int32_t> & out_ids);

}  // namespace transcribe::granite_nar
