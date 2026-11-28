// transcribe-flash-policy.h - shared flash-attention policy helpers.
//
// INTERNAL, C++17. Not part of the public ABI.
//
// Flash-attention support in ggml is per-backend and per-head-dim, so the
// default on/off decision is per-family (it depends on head dim and the
// backend it was loaded on). What lives here are the cross-family concerns:
// honoring the TRANSCRIBE_NO_FLASH / TRANSCRIBE_FORCE_FLASH global debug
// overrides, and handing attention its inputs.

#pragma once

#include "ggml.h"

namespace transcribe::flash {

// Apply the TRANSCRIBE_NO_FLASH / TRANSCRIBE_FORCE_FLASH environment
// overrides to a pair of encoder/decoder flash flags. Either override forces
// both flags in the same direction; if both are set, FORCE wins. The flags
// are in/out: updated in-place if an override is set, left untouched
// otherwise. TRANSCRIBE_ENCODER_FLASH=0|1 then sets the encoder flag alone.
void apply_env_overrides(bool & encoder_use_flash, bool & decoder_use_flash);

// An attention input (Q, K or V) after its permute into the
// [head_dim, T, heads(, B)] layout attention takes. ggml_flash_attn_ext reads
// its inputs through their strides on every backend, so the permuted view goes
// in as is and flash attention sees exactly the values a copy would hold; the
// manual path's matmuls still get the contiguous copy they always had. On a
// decode step (T = 1) the permute only moves a size-1 axis, so that copy was a
// byte-for-byte duplicate of its source.
inline ggml_tensor * attn_input(ggml_context * ctx, ggml_tensor * permuted, bool use_flash) {
    return use_flash ? permuted : ggml_cont(ctx, permuted);
}

}  // namespace transcribe::flash
