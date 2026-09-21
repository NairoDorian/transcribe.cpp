// transcribe-loader.cpp - implementation of the GGUF loader.
//
// See transcribe-loader.h for the contract. This file is intentionally
// architecture-agnostic: it knows nothing about Parakeet, Whisper, or any
// other family. It only reads enough KV to identify which family handler
// the per-arch dispatch should hand the file to.

#include "transcribe-loader.h"

#include "gguf.h"
#include "transcribe-meta.h"
#include "transcribe-path.h"

#include <cstring>
#include <string>

namespace transcribe {

Loader::~Loader() {
    if (gguf_ != nullptr) {
        gguf_free(gguf_);
        gguf_ = nullptr;
    }
}

gguf_context * Loader::release_gguf() {
    gguf_context * out = gguf_;
    gguf_              = nullptr;
    return out;
}

// Resolution of foreign GGUF packaging onto a family this repo implements.
//
// A GGUF produced by a sibling runtime can declare that runtime's own
// architecture string and name the equivalent transcribe family in a sidecar
// key instead. Resolving that has to happen here, in the core, and before
// dispatch, for a structural reason: `general.architecture` is the *only*
// thing that selects a family, and under TRANSCRIBE_ARCH_DL the family's
// plugin is not loaded until after that selection — so the plugin cannot be
// asked what it would accept. The check therefore has to precede the read at
// the top of open().
//
// The table is deliberately explicit rather than a naming convention. Each row
// is a claim that this repository has verified a specific sibling packaging
// maps onto a specific family; that claim belongs written down once, here,
// rather than inferred from a string at runtime.
//
// Scope: this rewrites the architecture KV only. Adapting the remainder of a
// package — injecting metadata the family's hparam reader expects, renaming
// tensors — is family knowledge and stays in the family's own load(), which
// has the gguf_context and the ggml_context to do it properly.
namespace {
struct ForeignPackaging {
    const char * packaging;  // value of general.architecture
    const char * family_key; // sidecar KV naming the family
    const char * family_id;  // value of that KV this repo implements
    const char * arch;       // transcribe family to dispatch to
};

constexpr ForeignPackaging kForeignPackaging[] = {
    // Confucius4-R2T2, packaged by audio.cpp: same encoder/decoder graph as
    // qwen3_asr (see docs/porting/families/confucius4_r2t2.md). The family
    // adapts its own metadata and tensor names in load(); all this does is
    // make the file dispatchable.
    { "audiocpp", "audiocpp.model_spec.family", "confucius4_r2t2", "qwen3_asr" },
};

bool resolve_foreign_packaging(gguf_context * g) {
    const int64_t arch_key = gguf_find_key(g, "general.architecture");
    if (arch_key < 0 || gguf_get_kv_type(g, arch_key) != GGUF_TYPE_STRING) {
        return false;
    }
    const char * declared = gguf_get_val_str(g, arch_key);

    for (const ForeignPackaging & row : kForeignPackaging) {
        if (std::strcmp(declared, row.packaging) != 0) {
            continue;
        }
        const int64_t fam_key = gguf_find_key(g, row.family_key);
        if (fam_key < 0 || gguf_get_kv_type(g, fam_key) != GGUF_TYPE_STRING) {
            continue;
        }
        // Copied before the write below: gguf_set_val_str() removes the key it
        // replaces, which frees the storage this pointer refers to.
        const std::string family_id = gguf_get_val_str(g, fam_key);
        if (family_id != row.family_id) {
            continue;
        }
        gguf_set_val_str(g, "general.architecture", row.arch);
        return true;
    }
    return false;
}
}  // namespace

transcribe_status Loader::open(const char * path) {
    if (path == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    path_ = path;

    // Distinguishes "the file is not at this path" (-> FILE_NOT_FOUND)
    // from every other reason gguf_init_from_file might return nullptr
    // (-> ERR_GGUF). Exact semantics in transcribe-path.h.
    if (!path_is_present(path)) {
        return TRANSCRIBE_ERR_FILE_NOT_FOUND;
    }

    // Header-only inspection: do not allocate ggml tensors. With ctx=null
    // gguf_init_from_file skips the entire tensor-allocation block; with
    // no_alloc=true any future code path that does pass a ctx will not
    // read the data blob.
    gguf_init_params init_params{};
    init_params.no_alloc = true;
    init_params.ctx      = nullptr;

    gguf_ = gguf_init_from_file(path, init_params);
    if (gguf_ == nullptr) {
        // ggml has already logged a structured error via GGML_LOG_ERROR
        // (corrupt magic, version mismatch, truncated header, IO error
        // after the existence check, etc.). All of those collapse to a
        // single public status.
        return TRANSCRIBE_ERR_GGUF;
    }

    // Rewrite a sibling-runtime architecture string onto the family this repo
    // implements, so everything below (and the dispatch that follows) sees an
    // ordinary transcribe GGUF. No-op for every file we produce ourselves.
    resolve_foreign_packaging(gguf_);

    // general.architecture is required. Without it the dispatch layer
    // has nothing to look up in the registry. Both Absent and BadType
    // are fatal: a missing arch and an arch with the wrong GGUF type
    // both leave us with no family to dispatch to.
    switch (read_string_kv(gguf_, "general.architecture", arch_)) {
        case KvResult::Absent:
        case KvResult::BadType:
            return TRANSCRIBE_ERR_GGUF;
        case KvResult::Ok:
            break;
    }

    // stt.variant is optional in 0.x. Per-family handlers default it
    // when absent (e.g. parakeet -> tdt-0.6b-v2). A present-but-wrong-
    // type variant is still a converter bug, though, so BadType is
    // fatal here while Absent silently leaves variant_ empty.
    switch (read_string_kv(gguf_, "stt.variant", variant_)) {
        case KvResult::Absent:
            // variant_ remains empty; family handler will default it.
            break;
        case KvResult::BadType:
            return TRANSCRIBE_ERR_GGUF;
        case KvResult::Ok:
            break;
    }

    // Copy every scalar-string KV into the metadata map. This mirrors
    // llama.cpp's generic metadata accessor: the public API exposes one
    // keyed getter (transcribe_model_meta_val_str) instead of a typed
    // accessor per field, so adding a new string KV in the converter needs
    // no API change. Non-string KVs (hparams, arrays such as the token
    // list) are intentionally skipped — this is identity/display metadata.
    const int64_t n_kv = gguf_get_n_kv(gguf_);
    for (int64_t i = 0; i < n_kv; ++i) {
        if (gguf_get_kv_type(gguf_, i) != GGUF_TYPE_STRING) {
            continue;
        }
        meta_.emplace(gguf_get_key(gguf_, i), gguf_get_val_str(gguf_, i));
    }

    return TRANSCRIBE_OK;
}

}  // namespace transcribe
