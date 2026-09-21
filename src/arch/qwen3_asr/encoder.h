// arch/qwen3_asr/encoder.h - Qwen3-ASR audio encoder graph builder.
//
// The encoder is a bidirectional 18-layer transformer on top of a 3x
// Conv2d subsampler. Modeled directly off
// qwen_asr.core.transformers_backend.modeling_qwen3_asr
// .Qwen3ASRAudioEncoder.
//
// Attention: bidirectional WITHIN fixed-size windows of the pad-trimmed
// sequence, never across them — the reference's `cu_seqlens` chunking
// (modeling_qwen3_asr.py Qwen3ASRAudioEncoder.forward builds cu_chunk_lens
// from `aftercnn_lens` in blocks of
// `padded_mask_after_cnn.shape[-1] * (n_window_infer // (n_window*2))`,
// i.e. n_window_infer mel frames per window, and every encoder layer takes
// those cumulative lengths). Both of the reference's real attention paths
// honour it: FA2 receives cu_seqlens as varlen, the eager/sdpa path is
// handed a block-diagonal additive mask by _prepare_attention_mask. The
// vLLM port (core/vllm_backend/qwen3_asr.py) and the audio.cpp port
// (community_models/confucius4_r2t2/audio_encoder.cpp) build the same
// block-diagonal mask.
//
// Window geometry for R2T2 (n_window=50, n_window_infer=800): 13 after-CNN
// tokens per 100-frame chunk * (800 // 100) = 104 tokens = 8 chunks = 8 s.
// An utterance shorter than one window is a single block, so the windowed
// mask is the identity on it — identical numerics, identical graph.
//
// The window is what makes a streaming tick's cost independent of how much
// audio precedes it: a block's output depends on its own window only, so
// under windowed attention the encoder is reusable prefix-wise in a way
// full bidirectional attention can never be.
//
// Shape conventions (ggml fast-to-slow ne[]):
//
//   mel_in      : [mel_per_chunk,          n_mels,       1, n_chunks]
//                 per-chunk batched input. Chunks with fewer real mel
//                 frames than mel_per_chunk are zero-padded by the caller.
//   after 3x conv: [F_ds=n_mels/8, T_ds=mel_per_chunk/8, downsample_hidden, n_chunks]
//                 (ggml conv_2d convention swaps PyTorch's H/W, which is
//                 a no-op for the square kernel + symmetric stride.)
//   after conv_out + PE: [d_model, per_chunk_aftercnn, n_chunks]
//   after pad-select: [d_model, T_enc]  where T_enc =
//                 (n_chunks-1) * per_chunk_aftercnn + last_chunk_aftercnn.
//                 The last chunk's aftercnn trailing pad rows are dropped
//                 in the graph (matches reference's
//                 `padded_embed[padded_mask_after_cnn]` selection).
//   output      : [output_dim, T_enc]

#pragma once

#include "ggml.h"
#include "weights.h"

#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;

namespace transcribe::qwen3_asr {

// Per-utterance timing metadata. Computed host-side from the mel length
// and encoder hparams; drives graph shape and mask construction.
struct EncoderTiming {
    int32_t n_mel_frames        = 0;  // input mel frame count
    int32_t mel_per_chunk       = 0;  // n_window * 2
    int32_t n_chunks            = 0;  // ceil(n_mel_frames / mel_per_chunk)
    int32_t last_chunk_real_mel = 0;  // real (unpadded) mel count in the last chunk
    int32_t per_chunk_aftercnn  = 0;  // aftercnn(mel_per_chunk) — same for every full chunk
    int32_t last_chunk_aftercnn = 0;  // aftercnn(last_chunk_real_mel)
    int32_t T_enc_padded        = 0;  // n_chunks * per_chunk_aftercnn (graph sequence length)
    int32_t T_enc               = 0;  // (n_chunks-1)*per + last_after (real, ragged)
    int32_t aftercnn_lens_total = 0;  // aftercnn(n_mel_frames) — matches T_enc for single-utterance inputs
    // Attention window in after-CNN tokens: the reference's
    // `padded_mask_after_cnn.shape[-1] * (n_window_infer // (n_window*2))`,
    // where the first factor is the widest chunk's after-CNN token count.
    // 0 (or >= T_enc) means the whole utterance is one window, in which case
    // no mask tensor is built and the graph is exactly what it was before
    // windows existed. See the header note for why this is windowed at all.
    int32_t window_tokens       = 0;
};

// Apply the 3x stride-2 pad-1 kernel-3 downsampling formula three times.
int32_t aftercnn_len(int32_t mel_len);

EncoderTiming compute_encoder_timing(int32_t n_mel_frames, const QwenAsrHParams & hp);

// Fill one [T_kv, T_q] f16 slab of the windowed-attention mask, row-major
// (row * T_q + col), in the additive form ggml_flash_attn_ext and
// ggml_soft_max_ext take: 0.0 inside a window, -inf between windows.
//
// `valid` is the row count that is real audio; rows at or past it are graph
// padding (the batched path's unused chunk slots) and attend only
// themselves, so they cannot leak into a real row's output. `valid` must
// equal T_q for the single-utterance graph. Windows are laid down from row 0
// in `window_tokens` strides with a short final window, matching the
// reference's cu_chunk_lens construction.
void fill_encoder_window_mask(ggml_fp16_t * dst, int32_t T_kv, int32_t T_q, int32_t window_tokens, int32_t valid);

// True when encoder attention is windowed (the default). The environment
// variable TRANSCRIBE_QWEN3_ASR_ENC_NO_WINDOW=1 restores full bidirectional
// attention, for the A/B measurement that compares the two on the same
// audio; it is not a supported configuration.
bool encoder_window_attention_enabled();

// Sinusoidal position table, row-major "[length, d_model]" layout.
// Matches SinusoidsPositionEmbedding: the first d_model/2 channels are
// sin(p * inv_ts[k]), the second d_model/2 are cos(p * inv_ts[k]).
std::vector<float> build_sinusoid_pe(int32_t d_model, int32_t length, double max_timescale = 10000.0);

struct EncoderDumps {
    ggml_tensor * mel_in         = nullptr;  // graph input
    ggml_tensor * subsample_out  = nullptr;  // post conv_out linear, pre-PE
    ggml_tensor * pos_add_out    = nullptr;  // post-PE, flattened and pad-trimmed to [d_model, T_enc]
    ggml_tensor * block_0_out    = nullptr;
    ggml_tensor * block_last_out = nullptr;
    ggml_tensor * ln_post_out    = nullptr;
    ggml_tensor * proj_out       = nullptr;
};

struct EncoderBuild {
    ggml_tensor * mel_in     = nullptr;  // [mel_per_chunk, n_mels, 1, n_chunks]
    ggml_tensor * pos_emb_in = nullptr;  // [d_model, per_chunk_aftercnn]
    // Block-diagonal window mask, [T_enc, T_enc] f16, or null when the
    // utterance fits in one window. The caller uploads it after the
    // scheduler has allocated the graph (see fill_encoder_window_mask).
    ggml_tensor * mask_in    = nullptr;
    ggml_tensor * out        = nullptr;  // [output_dim, T_enc]
    EncoderDumps  dumps{};
    ggml_cgraph * graph = nullptr;
    EncoderTiming timing{};
};

EncoderBuild build_encoder_graph(ggml_context *         ctx,
                                 const QwenAsrWeights & weights,
                                 const QwenAsrHParams & hp,
                                 const EncoderTiming &  timing,
                                 bool                   use_flash = false);

// ---------------------------------------------------------------------------
// Batched encoder (offline transcribe_run_batch)
// ---------------------------------------------------------------------------

struct EncoderBuildBatched {
    ggml_tensor * mel_in     = nullptr;  // [mel_per_chunk, n_mels, 1, B*n_chunks_max]
    ggml_tensor * pos_emb_in = nullptr;  // [d_model, per_chunk_aftercnn]
    // Window mask, [T_pad_max, T_pad_max, 1, B] f16: one slab per utterance,
    // built over that utterance's T_enc[b] real rows. Null when every
    // utterance in the batch fits in a single window.
    ggml_tensor * mask_in    = nullptr;
    ggml_tensor * out        = nullptr;  // [output_dim, T_pad_max, B]
    ggml_cgraph * graph      = nullptr;

    int n_batch       = 0;
    int n_chunks_max  = 0;
    int T_per_chunk   = 0;
    int T_pad_max     = 0;  // n_chunks_max * per_chunk_aftercnn
    int window_tokens = 0;  // after-CNN tokens per attention window (0 = one window)
};

// Build one encoder graph that processes B utterances in parallel on the
// batch axis ne[2]. All utterances share the same per_chunk_aftercnn (it
// depends only on enc_n_window), so they pack cleanly: utterance b's chunks
// occupy N-indices [b*n_chunks_max, b*n_chunks_max + n_chunks[b]) of mel_in,
// zero-padded to n_chunks_max. The conv subsampler is per-chunk (no cross-
// utterance leak); the 18 blocks attend per-utterance over pad-trimmed rows
// (the bounded chunked subsample drops padded rows before the blocks, so
// no explicit mask tensor is needed). The real rows of
// utterance b are the first T_enc[b] rows of its [T_pad_max] section, so the
// caller slices out[:, 0:T_enc[b], b]. Real-row outputs are bit-identical to
// the single-shot encoder (same per-chunk conv, same masked attention).
//
// Pass use_flash=false to match the single-shot non-flash default exactly.
EncoderBuildBatched build_encoder_graph_batched(ggml_context *         ctx,
                                                const QwenAsrWeights & weights,
                                                const QwenAsrHParams & hp,
                                                int                    n_chunks_max,
                                                int                    n_batch,
                                                bool                   use_flash = false);

}  // namespace transcribe::qwen3_asr
