#pragma once
#include "transcribe.h"
#include <string>
#include <utility>
#include <vector>
struct gguf_context;
struct ggml_context;
struct ggml_tensor;
namespace transcribe::qwen3_asr {
/// Value prepare_r2t2_metadata() writes into `stt.variant`, and the variant the
/// loader must report for such a file. One spelling in one place: the adapter
/// synthesises it after the loader has already snapshotted stt.variant, so the
/// loader has to know it independently rather than read it back.
inline constexpr const char * k_r2t2_variant = "confucius4-r2t2-1.7b";

/// True when this GGUF is the sibling's Confucius4-R2T2 package rather than a
/// native qwen3_asr file. Keyed on the packaging marker, not on `stt.variant`:
/// the variant string is itself produced by prepare_r2t2_metadata, so it
/// cannot be the thing that decides whether to run it.
bool is_r2t2_package(const gguf_context * meta);
transcribe_status prepare_r2t2_metadata(gguf_context * meta);
std::vector<std::pair<ggml_tensor *, std::string>> rename_r2t2_tensors(ggml_context * ctx);

/// Rebuild a tensor catalog in which every BF16 entry is planned as F32, every
/// other entry keeping the type the file declares, and return it via *out_ctx.
///
/// Why this exists: the published package keeps its unquantized weights (all
/// 363 of the convs, norms, biases and the audio-tower projections) in BF16 —
/// the publisher's reference dtype — while quantizing only the matmuls. This
/// family's graph computes in F32 and cannot consume a BF16 weight in the
/// slots that feed elementwise ops: ggml dispatches (BF16, F32) but not
/// (F32, BF16), and `ggml_mul(y, gamma)` / `ggml_add(x, bias)` put the f32
/// activation in src0, so a BF16 operand there aborts rather than converting.
/// Planning those tensors as F32 lets stream_tensor_data() convert each one
/// through f32 as it uploads, which is lossless (BF16 is a strict subset of
/// F32) and costs ~9 MB across the whole file.
///
/// The source context is left untouched; the caller owns the result and is
/// responsible for freeing whichever context it replaces.
transcribe_status plan_r2t2_dtypes(ggml_context * src_ctx, ggml_context ** out_ctx);
}  // namespace transcribe::qwen3_asr
