// arch/qwen3_asr/r2t2-stream.h - Confucius4-R2T2 streaming hook surface.
//
// INTERNAL to src/arch/qwen3_asr/. Declares the hooks model.cpp installs into
// the qwen3_asr Arch table. The algorithm lives in r2t2-stream.cpp; the state
// they carry is R2T2StreamState, declared beside the session that owns it in
// qwen3_asr.h (including it from there would be a cycle).
#pragma once

#include "qwen3_asr.h"
#include "transcribe-arch.h"

namespace transcribe::qwen3_asr {

// Streaming hooks for the R2T2 variant. Reachable only when the loaded model
// IS that variant: accepts_ext_kind gates the stream extension on the package
// marker, and the dispatcher refuses to begin a stream unless the model
// advertises supports_streaming, which only the R2T2 loader sets.
//
// The hooks are the required begin/feed/finalize triple plus the optional
// validate/reset, so the contract the dispatcher enforces is documented once
// in transcribe-arch.h and not restated here.
transcribe_status r2t2_stream_validate(const transcribe_session *       ctx,
                                       const transcribe_run_params *    run_params,
                                       const transcribe_stream_params * stream_params);
transcribe_status r2t2_stream_begin(transcribe_session *             ctx,
                                    const transcribe_run_params *    run_params,
                                    const transcribe_stream_params * stream_params);
transcribe_status r2t2_stream_feed(transcribe_session *       ctx,
                                   const float *              pcm,
                                   int                        n_samples,
                                   transcribe_stream_update * update);
transcribe_status r2t2_stream_finalize(transcribe_session * ctx, transcribe_stream_update * update);
void              r2t2_stream_reset(transcribe_session * ctx);

bool r2t2_accepts_ext_kind(const transcribe_model * model, transcribe_ext_slot slot, uint32_t kind);

}  // namespace transcribe::qwen3_asr
