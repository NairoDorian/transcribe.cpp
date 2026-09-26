// arch/qwen3_asr/model.cpp - Qwen3-ASR family handler.

#include "causal_lm/causal_lm.h"
#include "decoder.h"
#include "encoder.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "mel_pack.h"
#include "qwen3_asr.h"
#include "r2t2-package.h"
#include "r2t2-stream.h"
#include "transcribe-arch.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-decode-budget.h"
#include "transcribe-env.h"
#include "transcribe-flash-policy.h"
#include "transcribe-graph-opt.h"
#include "transcribe-load-common.h"
#include "transcribe-loader.h"
#include "transcribe-log.h"
#include "transcribe-mel.h"
#include "transcribe-meta.h"
#include "weights.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace transcribe::qwen3_asr {

extern const Arch arch;

static_assert(std::is_base_of_v<transcribe_model, QwenAsrModel>);
static_assert(std::is_base_of_v<transcribe_session, QwenAsrSession>);

QwenAsrSession::~QwenAsrSession() {
    kv_cache.free();
    kv_cache_batch.free();
}

QwenAsrModel::~QwenAsrModel() {
    if (ctx_meta != nullptr) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    if (backend_buffer != nullptr) {
        safe_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }
    packed_gate_up.free();
    for (auto it = plan.scheduler_list.rbegin(); it != plan.scheduler_list.rend(); ++it) {
        safe_backend_free(*it);
    }
    plan.scheduler_list.clear();
    plan.primary      = nullptr;
    plan.primary_kind = transcribe::BackendKind::Unknown;
}

namespace {

constexpr const char k_default_variant[] = "qwen3-asr";

// Input-length contract (see docs/input-limits.md). Qwen3-ASR is a
// hard-context-cap family: audio tokens + chat prompt + generation share the
// Qwen3 decoder's context window (dec_max_position_embeddings), clamped to that
// ceiling. Over-length input is rejected with TRANSCRIBE_ERR_INPUT_TOO_LONG; a
// transcript that fills the generation budget before end-of-stream is flagged
// via transcribe_was_truncated().

// Generation reserve: what the input gate keeps free, and the decode-budget floor.
constexpr int k_gen_reserve = 256;

// Effective decoder context ceiling, in tokens: the model's trained maximum,
// optionally lowered — never raised — by the caller's session n_ctx knob.
int qwen3_context_ceiling(int32_t n_ctx_knob, const QwenAsrHParams & hp) {
    int ceiling = hp.dec_max_position_embeddings;
    if (n_ctx_knob > 0 && n_ctx_knob < ceiling) {
        ceiling = n_ctx_knob;
    }
    return ceiling;
}

// Advisory transcribe_capabilities::max_audio_ms: the longest audio whose
// audio tokens plus a representative prompt and the generation reserve fit
// the context ceiling. The audio encoder downsamples mel frames 8x (three
// stride-2 convs, see aftercnn_len); inverting that gives ms. Returns 0
// ("unknown / unbounded") if the rate constants are missing. Note: even
// within this bound a long transcript may truncate at the generation budget
// (transcribe_was_truncated) — max_audio_ms is the input bound.
int64_t qwen3_max_audio_ms(const QwenAsrHParams & hp) {
    if (hp.dec_max_position_embeddings <= 0 || hp.fe_hop_length <= 0 || hp.fe_sample_rate <= 0) {
        return 0;
    }
    constexpr int k_prompt_overhead = 48;  // chat affixes; advisory
    const int     max_audio_tokens  = hp.dec_max_position_embeddings - k_prompt_overhead - k_gen_reserve;
    if (max_audio_tokens <= 0) {
        return 0;
    }
    // audio_tokens ≈ mel_frames / 8 ; mel_frames = ms * sr / (hop * 1000)
    //   => ms ≈ audio_tokens * 8 * hop * 1000 / sr
    const int64_t mel_frames = static_cast<int64_t>(max_audio_tokens) * 8;
    return mel_frames * hp.fe_hop_length * 1000 / hp.fe_sample_rate;
}

// Forward declarations for helpers defined further down in this file.
transcribe_status resolve_chat_tokens(const transcribe::Tokenizer & tok, ChatTokens & out);

transcribe_status load(Loader & loader, const transcribe_model_load_params * params, transcribe_model ** out_model) {
    const int64_t t_load_start = ggml_time_us();

    auto m       = std::make_unique<QwenAsrModel>();
    m->arch      = &arch;
    m->t_load_us = 0;
    m->variant   = loader.variant().empty() ? k_default_variant : loader.variant();
    m->backend.clear();

    apply_family_invariants(*m);
    m->caps.n_languages = 0;
    m->caps.languages   = nullptr;

    // Confucius4-R2T2 arrives as a sibling-runtime package: encoder, decoder and
    // tokenizer are this family's graph, but the hparams, the capability flags
    // and the entire vocabulary live in JSON sidecars embedded in the GGUF
    // rather than in KVs this family reads. Adapt before the first metadata read
    // below, because synthesising exactly those KVs is what the adapter does —
    // and before read_languages_kv, which needs the general.languages it adds.
    // (The core already rewrote general.architecture, or we would not have been
    // dispatched here at all; see resolve_foreign_packaging in
    // transcribe-loader.cpp.)
    //
    // The variant is assigned here rather than taken from loader.variant(),
    // which was snapshotted in Loader::open() before this ran and so reports the
    // family default for a package that carries no stt.variant KV at all.
    if (is_r2t2_package(loader.gguf())) {
        if (const transcribe_status st = prepare_r2t2_metadata(loader.gguf()); st != TRANSCRIBE_OK) {
            return st;
        }
        m->variant = k_r2t2_variant;
    }

    if (const transcribe_status st = read_capability_kv(loader.gguf(), m->caps); st != TRANSCRIBE_OK) {
        return st;
    }
    // Streaming capability is derived from the package, not read from a KV: the
    // GGUF carries no streaming capability tag, and the R2T2 hooks in
    // r2t2-stream.cpp gate themselves on the same package marker. Setting it
    // only here (rather than unconditionally for the family) is what keeps a
    // native Qwen3-ASR file from advertising a stream API it has no hooks for.
    if (m->variant == k_r2t2_variant) {
        m->caps.supports_streaming = true;
    }
    if (const transcribe_status st = read_languages_kv(loader.gguf(), *m); st != TRANSCRIBE_OK) {
        return st;
    }

    // Tokenizer (byte-level BPE; loader handles the "gpt2" model tag).
    if (const transcribe_status st = m->tok.load(loader.gguf()); st != TRANSCRIBE_OK) {
        return st;
    }

    // Chat template (read here so its absence surfaces at load, not mid-decode).
    (void) read_optional_string_kv(loader.gguf(), "tokenizer.chat_template", "qwen3_asr", "", m->chat_template);

    // Resolve chat-template special-token ids at load so a vocab drift surfaces
    // here instead of silently producing a wrong prompt at decode time.
    if (const transcribe_status st = resolve_chat_tokens(m->tok, m->chat_tokens); st != TRANSCRIBE_OK) {
        return st;
    }

    if (const transcribe_status st = read_qwen3_asr_hparams(loader.gguf(), m->hparams); st != TRANSCRIBE_OK) {
        return st;
    }

    // Publish the input-length ceiling now that the decoder context window
    // and frontend rate are known.
    m->caps.max_audio_ms = qwen3_max_audio_ms(m->hparams);

    // Basis for transcribe_session_get_limits: the same constants
    // qwen3_max_audio_ms uses, so the limit recomputes at a lowered n_ctx.
    if (m->hparams.dec_max_position_embeddings > 0 && m->hparams.fe_hop_length > 0 && m->hparams.fe_sample_rate > 0) {
        m->limits.has_context_cap    = true;
        m->limits.model_max_ctx      = m->hparams.dec_max_position_embeddings;
        m->limits.prompt_overhead    = 48;
        m->limits.gen_reserve        = k_gen_reserve;
        // audio_tokens ≈ mel_frames / 8 ; mel_frames = ms*sr/(hop*1000)
        m->limits.ms_per_audio_token = 8.0 * m->hparams.fe_hop_length * 1000.0 / m->hparams.fe_sample_rate;
        m->limits.kv_elems_per_ctx_token =
            (int64_t) m->hparams.dec_n_kv_heads * m->hparams.dec_head_dim * m->hparams.dec_n_layers * 2;
    }

    m->hparams.vocab_size   = m->tok.n_tokens();
    m->hparams.bos_token_id = m->tok.bos_id();
    m->hparams.eos_token_id = m->tok.eos_id();

    if (m->hparams.vocab_size != m->hparams.dec_vocab_size) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr: tokenizer vocab (%d) != decoder vocab_size (%d)",
                m->hparams.vocab_size, m->hparams.dec_vocab_size);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (m->hparams.eos_token_id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr: GGUF tokenizer has no eos_token_id");
        return TRANSCRIBE_ERR_GGUF;
    }

    // Mel frontend (Whisper-style 128-bin log-mel at 16 kHz, 30 s max).
    {
        transcribe::MelConfig cfg{};
        cfg.sample_rate  = m->hparams.fe_sample_rate;
        cfg.num_mels     = m->hparams.fe_num_mels;
        cfg.n_fft        = m->hparams.fe_n_fft;
        cfg.win_length   = m->hparams.fe_win_length;
        cfg.hop_length   = m->hparams.fe_hop_length;
        cfg.pre_emphasis = m->hparams.fe_pre_emphasis;
        cfg.f_min        = m->hparams.fe_f_min;
        cfg.f_max        = m->hparams.fe_f_max;
        cfg.pad_mode     = m->hparams.fe_pad_mode;
        cfg.window_type  = m->hparams.fe_window;     // "hann_periodic"
        cfg.normalize    = m->hparams.fe_normalize;  // "per_utterance" (Whisper)

        // Optional filterbank + window buffers baked by the converter; if
        // present MelFrontend uses them instead of reconstructing from hparams.
        {
            using R               = transcribe::load_common::ReadF32Result;
            const size_t fb_elems = static_cast<size_t>(cfg.num_mels) * static_cast<size_t>(cfg.n_fft / 2 + 1);
            const auto   fb_rc    = transcribe::load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(), "frontend.mel_filterbank", fb_elems, "qwen3_asr", cfg.filterbank);
            if (fb_rc != R::Ok && fb_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
            const size_t win_elems = static_cast<size_t>(cfg.win_length);
            const auto   win_rc    = transcribe::load_common::read_f32_tensor_checked(
                loader.gguf(), loader.path(), "frontend.window", win_elems, "qwen3_asr", cfg.window);
            if (win_rc != R::Ok && win_rc != R::Absent) {
                return TRANSCRIBE_ERR_GGUF;
            }
        }

        m->mel.emplace(cfg);
    }

    // Reopen with no_alloc to build the tensor catalog.
    gguf_init_params init_params{};
    init_params.no_alloc = true;
    init_params.ctx      = &m->ctx_meta;

    gguf_context * gguf_data = gguf_init_from_file(loader.path().c_str(), init_params);
    if (gguf_data == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // gguf_init_from_file gave ctx_meta the dtypes the file declares. For the
    // R2T2 package those include BF16 on every unquantized weight, which this
    // family's F32 graph cannot take (see plan_r2t2_dtypes). Rebuild the
    // catalog with those planned as F32 so that streaming performs the
    // conversion; every other package is unaffected and keeps this context.
    if (is_r2t2_package(loader.gguf())) {
        ggml_context *          normalized = nullptr;
        const transcribe_status st         = plan_r2t2_dtypes(m->ctx_meta, &normalized);
        if (st != TRANSCRIBE_OK) {
            gguf_free(gguf_data);
            return st;
        }
        // Nothing has been allocated against ctx_meta yet — it is metadata
        // only at this point — so discarding it costs a few hundred KB.
        ggml_free(m->ctx_meta);
        m->ctx_meta = normalized;
    }

    // Ordering in this block is load-bearing. stream_tensor_data() below
    // resolves every tensor in ctx_meta against the GGUF's own tensor table
    // *by name*, so it has to run while ctx_meta still carries the names as
    // written in the file. Only once the bytes are in place do we translate
    // those names into this family's contract, which is what the weight catalog
    // then binds against. Hence: open -> plan -> allocate -> stream -> rename ->
    // catalog. Renaming any earlier would make every streaming lookup miss.

    // Backend plan.
    const transcribe_backend_request backend_req = (params != nullptr) ? params->backend : TRANSCRIBE_BACKEND_AUTO;
    if (const transcribe_status st = transcribe::load_common::init_backends(
            backend_req, (params != nullptr) ? params->device : nullptr, "qwen3_asr", m->plan);
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    m->backend         = ggml_backend_name(m->plan.primary);
    m->primary_backend = m->plan.primary;

    ggml_backend_buffer_t weights_buffer = alloc_ctx_tensors_with_reclaim(m->plan.primary, m->ctx_meta);
    if (weights_buffer == nullptr) {
        gguf_free(gguf_data);
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr: alloc_ctx_tensors_with_reclaim failed");
        return TRANSCRIBE_ERR_OOM;
    }
    m->backend_buffer = weights_buffer;
    ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    if (const transcribe_status st =
            transcribe::load_common::stream_tensor_data(loader.path(), gguf_data, m->ctx_meta, "qwen3_asr");
        st != TRANSCRIBE_OK) {
        gguf_free(gguf_data);
        return st;
    }
    gguf_free(gguf_data);

    // Translate the file's tensor names into this family's contract. For a
    // foreign package (Confucius4-R2T2) the file carries the sibling runtime's
    // names and this matters; for every GGUF we publish ourselves the mapping is
    // the identity and this is a no-op. Safe to do after streaming because
    // ggml_set_name only rewrites the tensor's name field, not its data or its
    // placement in the buffer.
    if (is_r2t2_package(loader.gguf())) {
        (void) rename_r2t2_tensors(m->ctx_meta);
    }

    // Bind the catalog last, now that tensor names are the ones the GET_*
    // helpers look for. A malformed file therefore pays one weights allocation
    // before its missing tensor is reported; that is the price of having a
    // single ordering rather than two variants of this catalog build (before
    // and after allocation) which would drift apart.
    if (const transcribe_status st = build_qwen3_asr_weights(m->ctx_meta, m->hparams, m->weights);
        st != TRANSCRIBE_OK) {
        return st;
    }

    // Pack gate+up into a separate session + backend buffer so the FFN
    // can run a single mul_mat instead of two. ctx_meta is sized
    // exactly for GGUF file tensors with no headroom, so packed
    // tensors live in their own context owned by `causal_lm::pack_gate_up`.
    {
        std::vector<transcribe::causal_lm::GateUpEntry> entries;
        entries.reserve(m->weights.dec_blocks.size());
        for (auto & b : m->weights.dec_blocks) {
            entries.push_back({ b.ffn_gate_w, b.ffn_up_w, &b.ffn_gate_up_w });
        }
        if (const transcribe_status st =
                transcribe::causal_lm::pack_gate_up(m->plan.primary, m->hparams.dec_hidden, m->hparams.dec_intermediate,
                                                    entries, m->packed_gate_up, "qwen3_asr");
            st != TRANSCRIBE_OK) {
            m->packed_gate_up.free();
            return st;
        }
    }

    m->t_load_us = ggml_time_us() - t_load_start;
    *out_model   = m.release();
    return TRANSCRIBE_OK;
}

transcribe_status init_context(transcribe_model *                model,
                               const transcribe_session_params * params,
                               transcribe_session **             out_ctx) {
    if (model->arch != &arch) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto cc       = std::make_unique<QwenAsrSession>();
    cc->model     = model;
    // Single-stream quantized decoding is bandwidth-bound. A conservative
    // worker budget avoids synchronization overhead without constraining OS
    // placement. Explicit caller thread counts remain authoritative.
    cc->n_threads = params->n_threads > 0 ? params->n_threads : transcribe::default_n_threads(5);
    cc->kv_type   = params->kv_type;
    cc->n_ctx     = transcribe_session_params_n_ctx(params);

    cc->encoder_use_flash = false;
    cc->decoder_use_flash = true;
    transcribe::flash::apply_env_overrides(cc->encoder_use_flash, cc->decoder_use_flash);

    // Pre-allocate KV cache at context creation so the first run
    // doesn't pay the allocation cost inside the decode phase.
    auto * cm = static_cast<QwenAsrModel *>(model);
    {
        ggml_type kv_type = GGML_TYPE_F16;
        if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) {
            kv_type = GGML_TYPE_F32;
        }
        int n_ctx_ceiling = qwen3_context_ceiling(cc->n_ctx, cm->hparams);
        int initial_n_ctx = std::min(n_ctx_ceiling, 2048);
        if (!transcribe::causal_lm::kv_init(cc->kv_cache, cm->plan.primary,
                                            /*n_ctx=*/initial_n_ctx, cm->hparams.dec_n_kv_heads,
                                            cm->hparams.dec_head_dim, cm->hparams.dec_n_layers, kv_type)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr init_context: KV cache allocation failed "
                                "(n_ctx=%d, %d kv-heads x %d head-dim x %d layers) — "
                                "out of memory.",
                                initial_n_ctx, cm->hparams.dec_n_kv_heads, cm->hparams.dec_head_dim,
                                cm->hparams.dec_n_layers);
            return TRANSCRIBE_ERR_OOM;
        }
    }

    *out_ctx = cc.release();
    return TRANSCRIBE_OK;
}

// Resolve the chat-template piece strings against the loaded tokenizer.
// Hard-fails with TRANSCRIBE_ERR_GGUF on a missing piece (a vocab reorder
// surfaces here at load). The newline is stored in its GPT-2 byte-level form
// (\n → U+010A "Ċ", \xC4\x8A); roles and <|im_*|> tokens are verbatim.
transcribe_status resolve_chat_tokens(const transcribe::Tokenizer & tok, ChatTokens & out) {
    struct PieceSlot {
        const char * piece;
        int32_t *    slot;
    };

    const PieceSlot pieces[] = {
        { "<|im_start|>", &out.im_start       },
        { "<|im_end|>",   &out.im_end         },
        { "\xC4\x8A",     &out.newline        }, // "Ċ" = byte-level \n
        { "system",       &out.role_system    },
        { "user",         &out.role_user      },
        { "assistant",    &out.role_assistant },
    };
    for (const auto & p : pieces) {
        const int id = tok.find(p.piece);
        if (id < 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr: chat-template piece \"%s\" not in tokenizer", p.piece);
            return TRANSCRIBE_ERR_GGUF;
        }
        *p.slot = id;
    }
    return TRANSCRIBE_OK;
}

// Build the prompt token sequence + audio-position list, mirroring the
// Qwen3-ASR chat template at the token level:
//
//   <|im_start|>system\n<|im_end|>\n
//   <|im_start|>user\n<|audio_start|><|audio_pad|>*T_enc<|audio_end|><|im_end|>\n
//   <|im_start|>assistant\n[language {Name}<asr_text>]?
//
// System prompt is empty. A non-null `lang_prefix_ids` (resolved via
// encode_language_prefix) is appended after the trailing newline to force an
// output language; kept out of here so this stays a pure token-id assembler.
void build_prompt_tokens(const QwenAsrHParams &       hp,
                         const ChatTokens &           ct,
                         int                          T_enc,
                         const std::vector<int32_t> * lang_prefix_ids,
                         std::vector<int32_t> &       out_ids,
                         std::vector<int64_t> &       out_audio_positions) {
    out_ids.clear();
    out_audio_positions.clear();

    out_ids.push_back(ct.im_start);
    out_ids.push_back(ct.role_system);
    out_ids.push_back(ct.newline);
    out_ids.push_back(ct.im_end);
    out_ids.push_back(ct.newline);

    out_ids.push_back(ct.im_start);
    out_ids.push_back(ct.role_user);
    out_ids.push_back(ct.newline);

    out_ids.push_back(hp.audio_start_token_id);
    const int64_t audio_start_pos = static_cast<int64_t>(out_ids.size());
    for (int i = 0; i < T_enc; ++i) {
        out_ids.push_back(hp.audio_token_id);
        out_audio_positions.push_back(audio_start_pos + i);
    }
    out_ids.push_back(hp.audio_end_token_id);

    out_ids.push_back(ct.im_end);
    out_ids.push_back(ct.newline);

    out_ids.push_back(ct.im_start);
    out_ids.push_back(ct.role_assistant);
    out_ids.push_back(ct.newline);

    if (lang_prefix_ids != nullptr && !lang_prefix_ids->empty()) {
        out_ids.insert(out_ids.end(), lang_prefix_ids->begin(), lang_prefix_ids->end());
    }
}

}  // namespace

// below; encode_language_prefix matches the qwen3_asr.h declaration.)

// BCP-47 → publisher canonical name ("English", "Chinese", ...), which the
// prompt renders instead of the code. Frozen per release
// (qwen_asr.inference.utils.SUPPORTED_LANGUAGES); this table is the update
// point for a future variant.
struct LangNameEntry {
    const char * bcp47;
    const char * pub_name;
};

constexpr LangNameEntry k_qwen3_asr_language_names[] = {
    { "zh",  "Chinese"    },
    { "en",  "English"    },
    { "yue", "Cantonese"  },
    { "ar",  "Arabic"     },
    { "de",  "German"     },
    { "fr",  "French"     },
    { "es",  "Spanish"    },
    { "pt",  "Portuguese" },
    { "id",  "Indonesian" },
    { "it",  "Italian"    },
    { "ko",  "Korean"     },
    { "ru",  "Russian"    },
    { "th",  "Thai"       },
    { "vi",  "Vietnamese" },
    { "ja",  "Japanese"   },
    { "tr",  "Turkish"    },
    { "hi",  "Hindi"      },
    { "ms",  "Malay"      },
    { "nl",  "Dutch"      },
    { "sv",  "Swedish"    },
    { "da",  "Danish"     },
    { "fi",  "Finnish"    },
    { "pl",  "Polish"     },
    { "cs",  "Czech"      },
    { "fil", "Filipino"   },
    { "fa",  "Persian"    },
    { "el",  "Greek"      },
    { "ro",  "Romanian"   },
    { "hu",  "Hungarian"  },
    { "mk",  "Macedonian" },
};

// Resolve a caller-supplied BCP-47 code to the token-id sequence the chat
// template expects: BPE("language {Name}") + the `<asr_text>` special id.
// encode() doesn't split special tokens out, so we look up the <asr_text> id
// directly from the vocab (never hardcoded) and append it by hand. The
// dispatcher validates `bcp47` against caps.languages first, so an unknown
// code here means converter/map drift — surface as UNSUPPORTED_LANGUAGE.
const char * bcp47_for_publisher_name(const std::string & name) {
    for (const auto & e : k_qwen3_asr_language_names) {
        if (name == e.pub_name) {
            return e.bcp47;
        }
    }
    return nullptr;
}

transcribe_status encode_language_prefix(const transcribe::Tokenizer & tok,
                                         const char *                  bcp47,
                                         std::vector<int32_t> &        out_ids) {
    out_ids.clear();
    if (bcp47 == nullptr || bcp47[0] == '\0') {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const char * pub_name = nullptr;
    for (const auto & e : k_qwen3_asr_language_names) {
        if (std::strcmp(e.bcp47, bcp47) == 0) {
            pub_name = e.pub_name;
            break;
        }
    }
    if (pub_name == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "qwen3_asr: no canonical publisher name for "
                "language=\"%s\"",
                bcp47);
        return TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE;
    }
    if (!tok.has_encoder()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "qwen3_asr: tokenizer missing encoder (merges "
                "unavailable); cannot render language hint");
        return TRANSCRIBE_ERR_GGUF;
    }
    const int asr_text_id = tok.find("<asr_text>");
    if (asr_text_id < 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "qwen3_asr: tokenizer vocab missing <asr_text> "
                "special token");
        return TRANSCRIBE_ERR_GGUF;
    }
    std::string text = "language ";
    text += pub_name;
    if (const transcribe_status st = tok.encode(text, out_ids); st != TRANSCRIBE_OK) {
        return st;
    }
    out_ids.push_back(asr_text_id);
    return TRANSCRIBE_OK;
}

namespace {  // reopen anon for the rest of the file's helpers.

// mel framing + the encoder cache's mel bookkeeping live in mel_pack.h so the
// unit test can reach them without a model, a backend or a run; they are pure
// functions of the [n_mels, n_frames] layout below and have no state of their
// own. Use-qualified rather than `using` so the call sites read as the header
// they come from.
using transcribe::qwen3_asr::copy_mel_prefix;
using transcribe::qwen3_asr::mel_prefix_matches;
using transcribe::qwen3_asr::pack_mel_chunks;

}  // namespace

// One complete decode pass: mel -> audio encoder -> prefill -> greedy step loop
// -> detokenized text. Shared verbatim by the offline run() below and by the
// Confucius4-R2T2 streaming path (r2t2-stream.cpp), which calls it once per
// committed audio chunk. Keeping one implementation is what makes the two
// paths numerically identical by construction rather than by review: the
// streaming re-decode is the *same* graph sequence offline runs, over a longer
// buffer.
//
// `suffix_ids`, when non-null, is appended to the prompt after the assistant
// header. Offline that is the "language {Name}<asr_text>" seed; for R2T2
// streaming it is that seed plus the already-decoded continuation text, which
// is exactly the reference's `prompt_raw_ + prefix`.
//
// `max_new_tokens` is the greedy generation budget: > 0 is an absolute cap
// (streaming's per-tick budget); 0 sizes the budget from the audio length
// (offline, floor k_gen_reserve — see docs/input-limits.md).
//
// This function writes ONLY session scratch (mel_buf, enc_host, t_* timers) and
// the abort flag. It does not touch full_text / segments / has_result / the
// detected language, and it does not free the scheduler or KV cache: the caller
// owns result state and GPU lifetime, which is what lets the streaming path
// keep the KV cache alive across chunks instead of reallocating per chunk.
transcribe_status run_decode_pass(transcribe_session *          session,
                                  const float *                 pcm,
                                  int                           n_samples,
                                  const transcribe_run_params * params,
                                  const std::vector<int32_t> *  suffix_ids,
                                  int                           max_new_tokens,
                                  DecodePassResult *            out,
                                  const std::vector<int32_t> *  draft_seed,
                                  EncoderPrefixCache *          enc_cache,
                                  transcribe::MelStreamState *  mel_stream,
                                  bool                          kv_reuse) {
    if (session == nullptr || pcm == nullptr || n_samples <= 0 || out == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (max_new_tokens < 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<QwenAsrSession *>(session);
    auto * cm = static_cast<QwenAsrModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Pre-run abort check (Qwen3-ASR is single-shot, so this is the only
    // observation point). The streaming caller polls between chunks.
    if (cc->poll_abort()) {
        return TRANSCRIBE_ERR_ABORTED;
    }

    transcribe::debug::init();

    // Mel front-end.
    if (!cm->mel.has_value()) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: model has no MelFrontend");
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    const int64_t     t_mel_start  = ggml_time_us();
    int               mel_n_mels   = 0;
    int               mel_n_frames = 0;
    // Streaming passes in a MelStreamState, so the STFT + filterbank runs
    // only over the frames this tick added instead of re-deriving the whole
    // utterance (a 320 ms tick at 35 s of audio otherwise spends ~26 ms of an
    // 80 ms budget here). The two paths are bit-identical -- that is the
    // contract compute_incremental() is tested against -- so the encoder
    // prefix cache below sees the same mel either way and its reuse decision
    // is unaffected. A model whose frontend cannot do it incrementally (or a
    // one-shot pass, which has no state to carry) uses compute() unchanged.
    transcribe_status mst          = TRANSCRIBE_OK;
    MelStreamState *  stream =
        (mel_stream != nullptr && !transcribe::env::flag("TRANSCRIBE_R2T2_NO_MEL_CACHE")) ? mel_stream : nullptr;
    if (stream != nullptr && cm->mel->supports_incremental()) {
        mst = cm->mel->compute_incremental(*stream, pcm, static_cast<size_t>(n_samples), cc->mel_buf, mel_n_mels,
                                           mel_n_frames, cc->n_threads);
    } else {
        mst =
            cm->mel->compute(pcm, static_cast<size_t>(n_samples), cc->mel_buf, mel_n_mels, mel_n_frames, cc->n_threads);
    }
    if (mst != TRANSCRIBE_OK) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: MelFrontend::compute failed (%s)",
                transcribe_status_string(mst));
        return mst;
    }
    cc->t_mel_us = ggml_time_us() - t_mel_start;

    // Dump the post-frontend mel in the reference's contract shape
    // [n_mels, T_mel]. The batched graph input is a reshaped view of
    // the same data, so the comparison point lives on the host.
    if (transcribe::debug::enabled()) {
        const long long shape[2] = { mel_n_mels, mel_n_frames };
        transcribe::debug::dump_host_f32("enc.mel.in", cc->mel_buf.data(), static_cast<long long>(cc->mel_buf.size()),
                                         shape, 2, "frontend.mel.norm");
    }

    // Compute encoder timing + reject unsupported shapes.
    EncoderTiming timing = compute_encoder_timing(mel_n_frames, cm->hparams);
    if (timing.n_chunks <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "qwen3_asr run: encoder timing is degenerate "
                "(n_mel_frames=%d)",
                mel_n_frames);
        return TRANSCRIBE_ERR_GGUF;
    }

    // ------------------------------------------------------------------
    // Encoder prefix reuse (R2T2 streaming — see EncoderPrefixCache).
    //
    // Decide, before anything is built, whether this pass encodes the whole
    // utterance or only its trailing windows. Offline (enc_cache == null) and
    // on any doubt this is the full encode it has always been.
    // ------------------------------------------------------------------
    EncoderPrefixCache * cache =
        (enc_cache != nullptr && !transcribe::env::flag("TRANSCRIBE_R2T2_NO_ENC_CACHE")) ? enc_cache : nullptr;

    // One window is `window_tokens` after-CNN tokens = window_tokens /
    // per_chunk_aftercnn whole chunks = that many times mel_per_chunk mel
    // frames. The cache is aligned to that grid and only to it: a partial
    // window's rows still move when more audio lands inside it.
    const int32_t tokens_per_window = timing.window_tokens;
    const int32_t frames_per_window = (tokens_per_window > 0 && timing.per_chunk_aftercnn > 0) ?
                                          (tokens_per_window / timing.per_chunk_aftercnn) * timing.mel_per_chunk :
                                          0;

    // Every clause here is a precondition of the *bit-exactness* claim, so it
    // fails closed: the grid must exist and attention must be windowed (under
    // global attention every row depends on the whole utterance), the cached
    // rows and mel must be self-consistent and cover whole windows, the mel
    // must not have shrunk, and — the real test — the frames the cached rows
    // were computed from must be byte-identical to the ones this pass just
    // produced. The front-end normalizes per utterance over the whole buffer,
    // so a new global maximum rewrites the level of every earlier frame; that
    // memcmp catches it and the pass re-encodes in full, exactly as before.
    int32_t cached_tokens = 0;
    if (cache != nullptr && frames_per_window > 0 && encoder_window_attention_enabled() && cache->tokens > 0 &&
        cache->n_mels == mel_n_mels && cache->tokens % tokens_per_window == 0 &&
        cache->frames == (cache->tokens / tokens_per_window) * frames_per_window && cache->frames <= mel_n_frames &&
        cache->rows.size() == static_cast<size_t>(cache->tokens) * static_cast<size_t>(cm->hparams.enc_output_dim) &&
        mel_prefix_matches(cache->mel, cache->frames, cc->mel_buf.data(), mel_n_frames, mel_n_mels)) {
        cached_tokens = cache->tokens;
    }
    const bool cache_hit = cached_tokens > 0;

    // Why a prefix that looked usable did not survive. A miss is expected
    // sometimes (the front-end's per-utterance normalization rewrites earlier
    // frames whenever a new global maximum arrives) but a permanent miss means
    // the cache is dead weight, so the reason has to be visible rather than
    // inferred from timings: this reports how many frames moved and which,
    // which separates "one new maximum re-levelled the buffer" from "the frame
    // grid itself shifted".
    if (cache != nullptr && !cache_hit && cache->tokens > 0) {
        int32_t moved = 0, first_moved = -1, last_moved = -1;
        if (cache->mel.size() == static_cast<size_t>(cache->n_mels) * cache->frames && cache->n_mels == mel_n_mels &&
            cache->frames <= mel_n_frames) {
            for (int32_t f = 0; f < cache->frames; ++f) {
                bool differs = false;
                for (int32_t m = 0; m < cache->n_mels && !differs; ++m) {
                    differs = cache->mel[static_cast<size_t>(m) * cache->frames + f] !=
                              cc->mel_buf[static_cast<size_t>(m) * mel_n_frames + f];
                }
                if (differs) {
                    if (first_moved < 0) {
                        first_moved = f;
                    }
                    last_moved = f;
                    ++moved;
                }
            }
        }
        log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                "qwen3_asr run: encoder cache miss — cached %d tok / %d frames, mel %d x %d vs %d x %d, "
                "%d of %d cached frames moved (first %d, last %d)",
                cache->tokens, cache->frames, cache->n_mels, cache->frames, mel_n_mels, mel_n_frames, moved,
                cache->frames, first_moved, last_moved);
    }

    // Encode mel frames [tail_frame0, mel_n_frames) and prepend the cached rows
    // for everything before it. The tail starts on a chunk *and* window
    // boundary, so its own chunk packing, positional table and window partition
    // reproduce the full pass's rows for those positions token for token.
    int32_t       tail_frame0 = 0;
    int32_t       tail_frames = mel_n_frames;
    EncoderTiming enc_timing  = timing;
    if (cache_hit) {
        const int32_t       cand_frame0 = (cached_tokens / tokens_per_window) * frames_per_window;
        const EncoderTiming cand        = compute_encoder_timing(mel_n_frames - cand_frame0, cm->hparams);
        if (cand.T_enc == timing.T_enc - cached_tokens) {
            tail_frame0 = cand_frame0;
            tail_frames = mel_n_frames - cand_frame0;
            enc_timing  = cand;
        } else {
            // aftercnn_len is additive over chunks, so this cannot fire; if it
            // ever does the prefix is not a whole number of windows and the
            // only safe answer is the full encode.
            log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "qwen3_asr run: encoder cache dropped — tail timing %d != %d - %d",
                    cand.T_enc, timing.T_enc, cached_tokens);
            cached_tokens = 0;
        }
    }
    const bool enc_tail = tail_frames > 0;

    // Reset per-call compute state. The phase timers below break out
    // per-run cost (graph build, sched alloc, uploads, prefill compute)
    // that the public transcribe_timings (mel/encode/decode) doesn't
    // expose, for the TRANSCRIBE_PERF_DEBUG breakdown.
    const int64_t t_enc_build_start    = ggml_time_us();
    int64_t       t_enc_build_us       = 0;
    int64_t       t_enc_d2h_us         = 0;
    int64_t       t_prefill_build_us   = 0;
    int64_t       t_prefill_compute_us = 0;
    int64_t       t_prefill_logits_us  = 0;
    int64_t       t_step_loop_us       = 0;
    int           n_steps              = 0;

    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }

    {
        ggml_init_params ip{};
        ip.mem_size     = 16 * 1024 * 1024;
        ip.mem_buffer   = nullptr;
        ip.no_alloc     = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx == nullptr) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: ggml_init for compute_ctx failed");
            return TRANSCRIBE_ERR_OOM;
        }
    }

    // Build encoder graph. Skipped entirely when the cache already covers the
    // whole utterance (the mel length landed exactly on a window boundary),
    // which is the one case where a tick needs no encoder at all.
    EncoderBuild eb;
    if (enc_tail) {
        eb = build_encoder_graph(cc->compute_ctx, cm->weights, cm->hparams, enc_timing, cc->encoder_use_flash);
        if (eb.graph == nullptr || eb.out == nullptr) {
            return TRANSCRIBE_ERR_GGUF;
        }

        // Allocate + compute encoder graph.
        if (cc->sched == nullptr) {
            cc->sched = ggml_backend_sched_new(cm->plan.scheduler_list.data(), nullptr,
                                               static_cast<int>(cm->plan.scheduler_list.size()), 16384, false, true);
            if (cc->sched == nullptr) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: ggml_backend_sched_new failed");
                return TRANSCRIBE_ERR_BACKEND;
            }
        }
        ggml_backend_sched_reset(cc->sched);
        if (!alloc_inference_graph(cc->sched, eb.graph)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr run: encoder graph allocation failed — out of memory.");
            return TRANSCRIBE_ERR_OOM;
        }

        // Pack + upload mel. On a cache hit this is the tail's frames only,
        // starting on a whole chunk. The start frame goes to pack_mel_chunks as
        // a frame offset — the mel buffer's row stride is the *whole* frame
        // count, so the tail is a column slice of it and its rows must still be
        // walked with mel_n_frames.
        std::vector<float> mel_batched;
        pack_mel_chunks(cc->mel_buf.data(), mel_n_mels, mel_n_frames, enc_timing, mel_batched, tail_frame0);
        ggml_backend_tensor_set(eb.mel_in, mel_batched.data(), 0, mel_batched.size() * sizeof(float));

        // Positional embedding.
        {
            std::vector<float> pe = build_sinusoid_pe(cm->hparams.enc_d_model, enc_timing.per_chunk_aftercnn);
            ggml_backend_tensor_set(eb.pos_emb_in, pe.data(), 0, pe.size() * sizeof(float));
        }

        // Windowed-attention mask (null when the sequence fits in one window).
        // Deterministic in T_enc, so the streaming path pays only a fill of
        // T_enc^2 fp16 values per tick. The tail's windows are laid down from
        // its own row 0, and its row 0 is a global window boundary, so they
        // coincide with the windows the full pass would have used.
        if (eb.mask_in != nullptr) {
            std::vector<ggml_fp16_t> mask(static_cast<size_t>(enc_timing.T_enc) *
                                          static_cast<size_t>(enc_timing.T_enc));
            fill_encoder_window_mask(mask.data(), enc_timing.T_enc, enc_timing.T_enc, enc_timing.window_tokens,
                                     enc_timing.T_enc);
            ggml_backend_tensor_set(eb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }

        transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);

        const int64_t t_enc_start = ggml_time_us();
        t_enc_build_us            = t_enc_start - t_enc_build_start;
        if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, eb.graph); gs != GGML_STATUS_SUCCESS) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: encoder graph compute failed (%d)",
                    static_cast<int>(gs));
            return TRANSCRIBE_ERR_BACKEND;
        }
        cc->t_encode_us = ggml_time_us() - t_enc_start;
    } else {
        cc->t_encode_us = 0;
        t_enc_build_us  = 0;
    }

    // Dump encoder intermediates. With a cache hit these are the tail's, not
    // the utterance's: the debug contract is "what this graph computed".
    auto try_dump = [](const char * name, ggml_tensor * t, const char * stage) {
        if (t != nullptr) {
            transcribe::debug::dump_tensor(name, t, stage);
        }
    };
    try_dump("enc.subsample.out", eb.dumps.subsample_out, "enc.subsample");
    try_dump("enc.pos_add.out", eb.dumps.pos_add_out, "enc.pos_add");
    try_dump("enc.block.0.out", eb.dumps.block_0_out, "enc.block.0");
    {
        char bname[64];
        std::snprintf(bname, sizeof(bname), "enc.block.%d.out", cm->hparams.enc_n_layers - 1);
        try_dump(bname, eb.dumps.block_last_out, "enc.block.last");
    }
    try_dump("enc.ln_post.out", eb.dumps.ln_post_out, "enc.ln_post");
    try_dump("enc.proj.out", eb.dumps.proj_out, "enc.proj");

    // Read encoder output to host for the LM prefill. The graph already
    // dropped the aftercnn pad rows (see encoder.cpp), so what it produces is
    // exactly [d_enc, T_enc] — the reference's
    // `padded_embed[padded_mask_after_cnn]` shape. On a cache hit the first
    // `cached_tokens` columns were not computed by this pass at all: they are
    // copied from the cache, which the hit test proved describes this mel.
    const int d_enc = static_cast<int>(cm->hparams.enc_output_dim);
    const int T_enc = static_cast<int>(enc_timing.T_enc) + cached_tokens;
    if (T_enc != timing.T_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: encoder cache arithmetic is inconsistent (%d + %d != %d)",
                static_cast<int>(enc_timing.T_enc), cached_tokens, timing.T_enc);
        return TRANSCRIBE_ERR_GGUF;
    }
    cc->enc_host.resize(static_cast<size_t>(d_enc) * static_cast<size_t>(T_enc));
    const int64_t t_d2h_start = ggml_time_us();
    if (cached_tokens > 0) {
        std::memcpy(cc->enc_host.data(), cache->rows.data(),
                    static_cast<size_t>(cached_tokens) * static_cast<size_t>(d_enc) * sizeof(float));
    }
    if (enc_tail) {
        ggml_backend_tensor_get(eb.out,
                                cc->enc_host.data() + static_cast<size_t>(cached_tokens) * static_cast<size_t>(d_enc),
                                0, static_cast<size_t>(enc_timing.T_enc) * static_cast<size_t>(d_enc) * sizeof(float));
    }
    t_enc_d2h_us = ggml_time_us() - t_d2h_start;

    // Hand the next tick everything that is final now: every *complete* window
    // this pass covered (the cached prefix, plus whatever the tail filled)
    // whose mel frames the front-end can vouch for. Complete windows are
    // exactly the rows more audio cannot change, and the mel they were computed
    // from goes with them so the next pass can prove it.
    //
    // The frame cut is the front-end's `final_frame_count`, not the mel length:
    // a centered reflect-padded STFT builds the last frame or two of every
    // buffer from the end padding, so those frames move the moment more samples
    // arrive, and a window containing one is not final no matter how complete
    // its chunk count looks. Cutting at a whole window keeps the prefix on the
    // grid the encoder's attention windows are laid on.
    if (cache != nullptr) {
        const int32_t final_frames  = cm->mel->final_frame_count(static_cast<size_t>(n_samples));
        const int32_t final_windows = (frames_per_window > 0) ? std::max(0, final_frames / frames_per_window) : 0;
        int32_t       complete      = (frames_per_window > 0) ? (T_enc / tokens_per_window) * tokens_per_window : 0;
        complete                    = std::min(complete, final_windows * tokens_per_window);
        if (complete <= 0) {
            cache->clear();
        } else {
            const int32_t keep_rows = std::min(cached_tokens, complete);
            const int32_t new_rows  = complete - keep_rows;
            const int32_t frames    = (complete / tokens_per_window) * frames_per_window;
            const size_t  want_mel  = static_cast<size_t>(mel_n_mels) * static_cast<size_t>(frames);

            cache->rows.resize(static_cast<size_t>(complete) * static_cast<size_t>(d_enc));
            if (new_rows > 0) {
                std::memcpy(cache->rows.data() + static_cast<size_t>(keep_rows) * static_cast<size_t>(d_enc),
                            cc->enc_host.data() + static_cast<size_t>(keep_rows) * static_cast<size_t>(d_enc),
                            static_cast<size_t>(new_rows) * static_cast<size_t>(d_enc) * sizeof(float));
            }
            if (!cache_hit || cache->n_mels != mel_n_mels || cache->frames != frames || cache->mel.size() != want_mel) {
                copy_mel_prefix(cc->mel_buf.data(), mel_n_frames, mel_n_mels, frames, cache->mel);
                cache->n_mels = mel_n_mels;
            }
            cache->frames = frames;
            cache->tokens = complete;
        }
        cache->reused_tokens  = cached_tokens;
        cache->encoded_tokens = T_enc - cached_tokens;
    }

    // Decode phase begins. t_dec_start covers prompt + KV init + prefill
    // build/compute + step loop (prefill is part of "decode" to users).
    const int64_t t_dec_start           = ggml_time_us();
    const int64_t t_prefill_build_start = t_dec_start;

    // Prompt construction.
    std::vector<int32_t> prompt_ids;
    std::vector<int64_t> audio_positions;
    build_prompt_tokens(cm->hparams, cm->chat_tokens, T_enc, suffix_ids, prompt_ids, audio_positions);
    const int T_prompt   = static_cast<int>(prompt_ids.size());
    const int prefix_len = audio_positions.empty() ? 0 : static_cast<int>(audio_positions.front());
    const int suffix_len = T_prompt - prefix_len - T_enc;
    (void) audio_positions;

    // Input-length gate: audio + prompt + generation must fit the decoder
    // context window. Reject an over-length clip here, before prefill/decode.
    const int ceiling = qwen3_context_ceiling(cc->n_ctx, cm->hparams);
    if (T_prompt + k_gen_reserve > ceiling) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                            "qwen3_asr run: input too long — %d audio + %d prompt tokens "
                            "leave no room for output within the %d-token context (need %d). "
                            "Shorten the audio (see transcribe_capabilities.max_audio_ms) or "
                            "split it into segments.",
                            T_enc, prefix_len + suffix_len, ceiling, T_prompt + k_gen_reserve);
        return TRANSCRIBE_ERR_INPUT_TOO_LONG;
    }

    // Decode budget: streaming passes an absolute per-tick cap (the reference
    // StreamConfig value); the offline pass passes 0 and sizes from the audio
    // length, floor k_gen_reserve, clamped to the context (docs/input-limits.md).
    const int max_new =
        max_new_tokens > 0 ?
            max_new_tokens :
            transcribe::pick_decode_budget(transcribe::predict_transcript_tokens(T_enc, cm->limits.ms_per_audio_token),
                                           k_gen_reserve, T_prompt, ceiling);

    // KV cache init (grow-to-fit, clamped to the context ceiling). Size to
    // hold prompt + decode budget, rounded up to a power of two (the step
    // graph's flash-attn path wants pow2 attention width). A pre-allocated
    // smaller cache is freed and re-allocated.
    int want_n_ctx = 1024;
    while (want_n_ctx < T_prompt + max_new) {
        want_n_ctx *= 2;
    }
    if (want_n_ctx > ceiling) {
        want_n_ctx = ceiling;
    }

    // ------------------------------------------------------------------
    // Cross-tick KV continuation (R2T2 streaming).
    //
    // `cached_tokens` (above) is the whole-window-aligned prefix of encoder
    // rows the *previous* pass had — and on a hit this pass proves those rows
    // are bit-identical to the ones it just produced (mel prefix memcmp, rows
    // copied verbatim). Their token ids, embeddings and therefore their
    // attention keys/values are therefore the same ones the previous pass
    // wrote into the KV cache at the same positions, and so is everything
    // before them: within a stream the text prefix is fixed, and K/V of
    // position j is a function of positions [0, j] alone. Those positions do
    // not have to be re-prefilled at all.
    //
    // What is NOT reused: the suffix (the accumulated transcript, which shifts
    // right as the audio block grows) and the audio rows newer than the cache.
    // So a tick prefills its own delta instead of the whole utterance.
    //
    // Every clause fails closed. `kv_realloc` matters because the reuse lives
    // in the session's KV cache: a cache about to be grown is about to lose its
    // rows, and kv_cache.n — the previous pass's high-water mark, always at
    // least its T_prompt — is what proves the rest are still there (a fresh or
    // cleared cache has n == 0, so this covers those too). The exactness
    // argument is a *mathematical* one (causality), not bit-equality: a
    // differently-shaped graph can accumulate the same sums in a different
    // order, so the reused rows can differ from a from-scratch prefill in the
    // last bits. That makes this the one cache here whose output is not
    // byte-identical by construction, which is why it has its own kill switch
    // and its own A/B (TRANSCRIBE_R2T2_NO_KV_REUSE=1 must leave the transcript
    // alone).
    //
    // The reused rows are always rows the previous *prefill* wrote, never rows
    // the step loop did: those start at the previous T_prompt, which is strictly
    // above the boundary here, because kv_reuse_len <= prefix_len + T_enc - 1.
    // That matters — the step loop's rows were computed one token at a time
    // against a mask of its own shape, and its drafting can also leave rejected
    // rows behind.
    // ------------------------------------------------------------------
    const bool kv_realloc   = (cc->kv_cache.ctx == nullptr || cc->kv_cache.n_ctx < want_n_ctx);
    int        kv_reuse_len = 0;
    // cached_tokens is re-read, not just cache_hit: the hit is latched before
    // the window-alignment fallback above can zero it, and a latched-with-zero
    // hit would leave only the text prefix to reuse, which is a real (if tiny)
    // saving but not what "the encoder cache carried the audio over" means.
    // dumps_on is on the list because a continuation carries only the prompt's
    // tail: the dump tensors would be shaped and named for a whole prefill
    // while holding a slice of one, which is worse than no dump at all.
    if (kv_reuse && !transcribe::env::flag("TRANSCRIBE_R2T2_NO_KV_REUSE") && cache_hit && cached_tokens > 0 &&
        !kv_realloc && !transcribe::debug::enabled()) {
        // At least one audio row has to remain: the graph's audio block is an
        // input tensor and a zero-row block is not a shape it can carry. That
        // also keeps the boundary strictly inside the prompt.
        kv_reuse_len = prefix_len + std::min<int32_t>(cached_tokens, T_enc - 1);
        if (cc->kv_cache.n < kv_reuse_len) {
            kv_reuse_len = 0;
        }
    }

    if (cc->kv_cache.ctx != nullptr && cc->kv_cache.n_ctx < want_n_ctx) {
        cc->kv_cache.free();
    }
    if (cc->kv_cache.ctx == nullptr) {
        ggml_type kv_type = GGML_TYPE_F16;
        if (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) {
            kv_type = GGML_TYPE_F32;
        }
        if (!transcribe::causal_lm::kv_init(cc->kv_cache, cm->plan.primary, want_n_ctx, cm->hparams.dec_n_kv_heads,
                                            cm->hparams.dec_head_dim, cm->hparams.dec_n_layers, kv_type)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr run: KV cache allocation failed (n_ctx=%d, "
                                "%d kv-heads x %d head-dim x %d layers) — out of memory. "
                                "Lower transcribe_session_params.n_ctx or shorten the audio.",
                                want_n_ctx, cm->hparams.dec_n_kv_heads, cm->hparams.dec_head_dim,
                                cm->hparams.dec_n_layers);
            return TRANSCRIBE_ERR_OOM;
        }
    } else if (kv_reuse_len == 0) {
        // Clear stale positions for a fresh prefill. Skipped when the pass is
        // extending the cache: the rows [0, kv_reuse_len) are the whole point,
        // and everything past them is overwritten by the prefill's own writes
        // (the mask never lets a query see past `head`, so rows the step loop
        // left beyond the prompt cannot leak in).
        if (cc->kv_cache.buffer != nullptr) {
            ggml_backend_buffer_clear(cc->kv_cache.buffer, 0);
        }
        cc->kv_cache.n    = 0;
        cc->kv_cache.head = 0;
    }

    // Free GPU buffers (scheduler galloc + KV cache) after each transcription to
    // prevent memory accumulation across repeated runs. The session persists
    // across calls (e.g. Multi-STT extra models with multi_stt_keep_extra_models_loaded),
    // so releasing here lets CUDA's caching allocator reuse freed blocks on the
    // next run() rather than growing the cache (GPU memory leak on Windows).
    auto cleanup_gpu = [&]() {
        cc->kv_cache.free();
        if (cc->sched != nullptr) {
            safe_sched_free(cc->sched);
            cc->sched = nullptr;
        }
    };

    // Prefill graph. slice_last false: last block's FFN + final norm run on
    // every position (needed for dump parity). true: slice to just the final
    // position before the last FFN (llama.cpp's inp_out_ids trick, ~25 ms).
    //
    // Geometry of the tail with a KV continuation: drop the audio rows the
    // cache already holds (whole rows, so the graph's audio input stays one
    // contiguous [d_enc, T_enc_tail] slab) and keep the entire suffix, whose
    // absolute positions move with the audio block and which is therefore never
    // reusable. With kv_reuse_len == 0 these collapse to the original
    // prefix_len / T_enc / T_prompt.
    const bool   dumps_on   = transcribe::debug::enabled();
    const bool   slice_last = !dumps_on;
    // Rows the cache already holds. kv_reuse_len counts the text prefix first,
    // so the audio rows it covers are the difference — and only when it is
    // set at all: with reuse off (kv_reuse_len == 0) nothing is skipped, and
    // `kv_reuse_len - prefix_len` would be negative and lengthen the tail past
    // the encoder output.
    const int    audio_skip = kv_reuse_len > 0 ? (kv_reuse_len - prefix_len) : 0;
    const int    T_enc_tail = T_enc - audio_skip;
    // The graph carries the prompt minus what the cache holds — the same
    // quantity build_prefill_graph derives internally as T_prompt - n_past, and
    // it has to be that, not `T_enc_tail + suffix_len`: on a tick that does not
    // reuse, the audio tail is the whole audio block and the two agree, but the
    // graph's token axis is what the uploads below must fill, and a short
    // upload leaves the rest of input_ids as whatever the buffer held — which
    // get_rows then reads as a token id.
    const int    T_graph    = T_prompt - kv_reuse_len;
    PrefillBuild pb = build_prefill_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, T_prompt, T_enc_tail,
                                          kv_reuse_len > 0 ? 0 : prefix_len, suffix_len,
                                          /*use_flash=*/cc->decoder_use_flash, slice_last, /*kv_batch_slot=*/0,
                                          /*kv_n_batch=*/1, /*n_past=*/kv_reuse_len);
    if (pb.graph == nullptr || pb.out == nullptr) {
        cleanup_gpu();
        return TRANSCRIBE_ERR_GGUF;
    }
    if (pb.T_graph != T_graph) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                            "qwen3_asr run: prefill geometry %d != %d (T_prompt=%d, n_past=%d)", pb.T_graph, T_graph,
                            T_prompt, kv_reuse_len);
        cleanup_gpu();
        return TRANSCRIBE_ERR_GGUF;
    }

    // Allocate + compute prefill on the same scheduler.
    ggml_backend_sched_reset(cc->sched);
    if (!alloc_inference_graph(cc->sched, pb.graph)) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                            "qwen3_asr run: prefill graph allocation failed (T_prompt=%d) — "
                            "out of memory. Lower transcribe_session_params.n_ctx or shorten "
                            "the audio.",
                            T_prompt);
        cleanup_gpu();
        return TRANSCRIBE_ERR_OOM;
    }

    // Upload prefill inputs. The graph's token axis is its own tail
    // (input_ids_in / positions_in are T_graph wide), while the mask spans the
    // whole decoder context so its queries can see the reused keys.
    ggml_backend_tensor_set(pb.input_ids_in, prompt_ids.data() + (T_prompt - T_graph), 0,
                            static_cast<size_t>(T_graph) * sizeof(int32_t));
    ggml_backend_tensor_set(pb.enc_out_in, cc->enc_host.data() + static_cast<size_t>(audio_skip) * d_enc, 0,
                            static_cast<size_t>(T_enc_tail) * d_enc * sizeof(float));

    {
        // Absolute positions: the tail continues the prompt, so RoPE sees the
        // same angles a full prefill would have produced.
        std::vector<int32_t> positions(T_graph);
        for (int i = 0; i < T_graph; ++i) {
            positions[i] = kv_reuse_len + i;
        }
        ggml_backend_tensor_set(pb.positions_in, positions.data(), 0, positions.size() * sizeof(int32_t));
    }

    {
        // Causal mask in F16 (matches pb.mask_in): row q of the graph's token
        // axis keeps columns [0, n_past + q], -inf past that. With n_past == 0
        // that is the plain causal triangle it has always been; with n_past > 0
        // it is the trapezoid that lets the tail attend the reused prefix.
        std::vector<ggml_fp16_t> mask(static_cast<size_t>(T_prompt) * T_graph);
        causal_lm::fill_prefill_chunk_mask(mask.data(), T_prompt, T_graph, kv_reuse_len);
        ggml_backend_tensor_set(pb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    const int64_t t_prefill_compute_start = ggml_time_us();
    t_prefill_build_us                    = t_prefill_compute_start - t_prefill_build_start;
    if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, pb.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr run: prefill graph compute failed (%d)", static_cast<int>(gs));
        cleanup_gpu();
        return TRANSCRIBE_ERR_BACKEND;
    }
    t_prefill_compute_us = ggml_time_us() - t_prefill_compute_start;

    cc->kv_cache.n    = T_prompt;
    cc->kv_cache.head = T_prompt;

    // Dump dec.* intermediates.
    try_dump("dec.token_emb", pb.dumps.token_emb, "dec.token_emb");
    try_dump("dec.audio_injected", pb.dumps.audio_injected, "dec.audio_injected");
    try_dump("dec.block.0.out", pb.dumps.block_0_out, "dec.block.0");
    {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "dec.block.%d.out", cm->hparams.dec_n_layers - 1);
        try_dump(nm, pb.dumps.block_last_out, "dec.block.last");
    }
    try_dump("dec.out_before_head", pb.dumps.out_before_head, "dec.out_before_head");
    try_dump("dec.logits_raw", pb.dumps.logits_raw, "dec.logits_raw");

    // Read prefill logits + first argmax.
    const int64_t      t_prefill_logits_start = ggml_time_us();
    const int          vocab                  = cm->hparams.dec_vocab_size;
    std::vector<float> logits(vocab);
    ggml_backend_tensor_get(pb.out, logits.data(), 0, logits.size() * sizeof(float));

    auto argmax = [&](const std::vector<float> & v) -> int32_t {
        int32_t best   = 0;
        float   best_v = v[0];
        for (int32_t i = 1; i < static_cast<int32_t>(v.size()); ++i) {
            if (v[i] > best_v) {
                best_v = v[i];
                best   = i;
            }
        }
        return best;
    };

    std::vector<int32_t> generated_ids;
    int32_t              next_tok = argmax(logits);
    generated_ids.push_back(next_tok);
    t_prefill_logits_us = ggml_time_us() - t_prefill_logits_start;

    // Step loop.
    const int32_t eos_id   = cm->hparams.eos_token_id;
    int           cur_past = T_prompt;

    // params->spec_k_drafts: -1 = family default (=0, disabled), 0 =
    // disabled, 1..QWEN3_ASR_SPEC_K_MAX = explicit draft length for the
    // 1-gram-lookup speculative decode below. Greedy acceptance commits only
    // tokens the verify pass itself predicted, so drafting adds no
    // approximation — but see the numerics note on the loop for why that is
    // still not byte-equality with k=0. It defaults OFF: measured on short
    // clips the
    // 1-gram acceptance (~1.1 tokens/verify) does not amortize the verify
    // pass, which costs ~1.5x a single step on CPU (T=2 leaves the matvec
    // fast path) and break-even on CUDA. Worth re-testing per workload via
    // --spec-k-drafts; repetitive long-form dictation accepts more.
    constexpr int QWEN3_ASR_SPEC_K_MAX = 8;
    int           k_drafts             = 0;
    if (params != nullptr &&
        params->struct_size >= offsetof(transcribe_run_params, spec_k_drafts) + sizeof(params->spec_k_drafts)) {
        const int requested = params->spec_k_drafts;
        if (requested == 0) {
            k_drafts = 0;
        } else if (requested > 0) {
            k_drafts = std::min(requested, QWEN3_ASR_SPEC_K_MAX);
        }
        // requested == -1 keeps the family default; requested < -1 falls
        // through to the family default (matches the silent-ignore semantics).
    }

    // External seed (streaming): a caller-supplied guess at this pass's first
    // tokens. It sets the draft length itself, because the verify graph needs
    // one column per seeded token, and it takes precedence over the run params
    // — a caller that computed a seed has evidence the 1-gram lookup cannot
    // match, and the seed costs nothing when it misses (its columns ride along
    // with the mandatory one).
    //
    // The seed is indexed by generation position: seed[0] is a guess at the
    // token the PREFILL just produced (so it is not a draft column — the
    // mandatory column already carries it), seed[1] a guess at the token the
    // first column predicts, and so on. The loop keeps that index moving as it
    // commits, so a seed longer than one run keeps aiming at the right position
    // in later runs instead of restarting at the prompt boundary.
    const bool seeded = draft_seed != nullptr && !draft_seed->empty();
    if (seeded) {
        k_drafts = std::min(static_cast<int>(draft_seed->size()), QWEN3_ASR_SPEC_K_MAX);
    }

    // Prior-transcript drafting (experimental, TRANSCRIBE_SPEC_PRIOR_TEXT): a
    // transcript of the same audio from another model (e.g. parakeet in a
    // multi-STT setup) is tokenized once and used as a draft corpus aligned by
    // a moving cursor. Same exact-greedy acceptance as the 1-gram lookup — a
    // token is committed only when this model's own argmax produced it — so
    // the prior can only change speed, never which tokens can be emitted
    // (numerics caveat of the verify graph below still applies).
    std::vector<int32_t> prior_ids;
    if (const std::string pt = transcribe::env::utf8("TRANSCRIBE_SPEC_PRIOR_TEXT"); !pt.empty()) {
        if (cm->tok.encode(pt, prior_ids) != TRANSCRIBE_OK) {
            prior_ids.clear();
        }
        // Never feed an id the embedding table does not have.
        const int32_t n_vocab = cm->hparams.dec_vocab_size;
        prior_ids.erase(std::remove_if(prior_ids.begin(), prior_ids.end(),
                                       [n_vocab](int32_t id) { return id < 0 || id >= n_vocab; }),
                        prior_ids.end());
        if (!prior_ids.empty() && k_drafts == 0) {
            k_drafts = QWEN3_ASR_SPEC_K_MAX;
        }
    }

    // Build the step graph ONCE and reuse every step, sized for the actual
    // workload (T_prompt written + up to max_new generated). Metal's flash-attn
    // kernels dispatch ~30% faster (M4 Max) when K/V ne[1] is a power of 2, so
    // round up; floor of 1024 (smaller just hits the slow-misaligned branch).
    // The verify graph writes up to k_drafts KV rows past the last committed
    // position, so reserve that slack too.
    int max_n_kv = 1024;
    while (max_n_kv < T_prompt + max_new + k_drafts) {
        max_n_kv *= 2;
    }
    if (max_n_kv > cc->kv_cache.n_ctx) {
        max_n_kv = cc->kv_cache.n_ctx;
    }
    // The spec path needs headroom for its draft columns inside the KV
    // window; if the context clamp removed it, fall back to plain stepping.
    if (k_drafts > 0 && T_prompt + max_new + k_drafts > max_n_kv) {
        k_drafts = 0;
    }
    const int64_t t_step_build_start = ggml_time_us();
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    {
        ggml_init_params ip{};
        ip.mem_size     = 8 * 1024 * 1024;
        ip.mem_buffer   = nullptr;
        ip.no_alloc     = true;
        cc->compute_ctx = ggml_init(ip);
        if (cc->compute_ctx == nullptr) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr step: compute context allocation failed — "
                                "out of memory.");
            cleanup_gpu();
            return TRANSCRIBE_ERR_OOM;
        }
    }
    StepBuild   sb{};
    VerifyBuild vb{};
    if (k_drafts == 0) {
        sb = build_step_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, max_n_kv,
                              /*use_flash=*/cc->decoder_use_flash);
        if (sb.graph == nullptr || sb.out == nullptr) {
            cleanup_gpu();
            return TRANSCRIBE_ERR_GGUF;
        }
    } else {
        vb = build_verify_graph(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache, /*T_verify=*/k_drafts + 1,
                                max_n_kv, /*use_flash=*/cc->decoder_use_flash);
        if (vb.graph == nullptr || vb.out == nullptr) {
            cleanup_gpu();
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!alloc_inference_graph(cc->sched, k_drafts == 0 ? sb.graph : vb.graph)) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                            "qwen3_asr step: decode graph allocation failed — out of memory. "
                            "Lower transcribe_session_params.n_ctx or shorten the audio.");
        cleanup_gpu();
        return TRANSCRIBE_ERR_OOM;
    }
    const int64_t t_step_build_once_us = ggml_time_us() - t_step_build_start;

    // Mask buffer reused host-side across steps. Starts all -inf; per step we
    // zero positions [0, cur_past] (attend) and leave the rest -inf.
    const ggml_fp16_t        mask_zero    = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t        mask_neg_inf = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> step_mask(max_n_kv, mask_neg_inf);

    // Per-step timers.
    int64_t       t_step_set_us     = 0;
    int64_t       t_step_comp_us    = 0;
    int64_t       t_step_get_us     = 0;
    const int64_t t_step_loop_start = ggml_time_us();
    int           n_graph_runs      = 0;
    if (k_drafts == 0) {
        while (next_tok != eos_id && static_cast<int32_t>(generated_ids.size()) < max_new && cur_past + 1 <= max_n_kv) {
            const int64_t t_set0 = ggml_time_us();

            ggml_backend_tensor_set(sb.input_id_in, &next_tok, 0, sizeof(int32_t));
            const int32_t pos_val = cur_past;
            ggml_backend_tensor_set(sb.position_in, &pos_val, 0, sizeof(int32_t));
            const int64_t kv_idx_val = cur_past;
            ggml_backend_tensor_set(sb.kv_idx_in, &kv_idx_val, 0, sizeof(int64_t));

            // Mask: mark the newly-added position as attendable. Positions
            // [0, cur_past) were zeroed in prior iterations; just set the
            // new one. (On iter 0, zero everything in [0, cur_past].)
            if (cur_past == T_prompt) {
                std::fill(step_mask.begin(), step_mask.begin() + cur_past + 1, mask_zero);
            } else {
                step_mask[cur_past] = mask_zero;
            }
            ggml_backend_tensor_set(sb.mask_in, step_mask.data(), 0,
                                    static_cast<size_t>(max_n_kv) * sizeof(ggml_fp16_t));

            const int64_t t_set1 = ggml_time_us();
            t_step_set_us += t_set1 - t_set0;

            if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, sb.graph);
                gs != GGML_STATUS_SUCCESS) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr step: graph compute failed (%d)", static_cast<int>(gs));
                cleanup_gpu();
                return TRANSCRIBE_ERR_BACKEND;
            }
            const int64_t t_comp1 = ggml_time_us();
            t_step_comp_us += t_comp1 - t_set1;
            ++n_graph_runs;

            int32_t argmax_tok = 0;
            ggml_backend_tensor_get(sb.out, &argmax_tok, 0, sizeof(int32_t));
            next_tok = argmax_tok;
            generated_ids.push_back(next_tok);
            cur_past += 1;
            cc->kv_cache.n    = cur_past + 1;
            cc->kv_cache.head = cur_past + 1;
            t_step_get_us += ggml_time_us() - t_comp1;
        }
    } else {
        // ---- 1-gram-lookup speculative decode (mechanism as in ----
        // ---- arch/voxtral_realtime) ----
        //
        // NUMERICS — read before trusting this to reproduce k=0. The
        // acceptance rule is exact: a draft is committed only when the verify
        // pass's own argmax at the previous column equals it, so no token is
        // committed that plain stepping would not have produced. What is NOT
        // guaranteed is byte-equality with spec_k_drafts == 0, because this
        // path runs build_verify_graph (T = k+1 >= 2 columns) instead of
        // build_step_graph (T = 1), and a multi-column mul_mat dispatches a
        // different kernel than the n=1 GEMV under GGML_LLAMAFILE. The drift
        // is enough to flip a near-tie argmax: on whole-earth.wav
        // (Qwen3-ASR-0.6B Q4_K_M, CPU, 255 tokens) every k >= 1 differs from
        // k = 0 at exactly one token. That the divergence is identical for
        // k = 1 and k = 4 — which accept different numbers of drafts, 1.06 vs
        // 1.08 tokens/run — is what isolates the cause to the graph rather
        // than the acceptance rule. transcribe.h documents k == 0 as the
        // byte-equal setting; this is that caveat, made concrete.
        //
        // all_ids[p] = token at absolute position p (prompt + committed).
        // last_pos_by_tok[t] = latest position whose token is t. A draft is
        // the sequence that followed the previous occurrence of next_tok;
        // when there is none, repeat next_tok (a miss only wastes the draft
        // columns, whose compute rides along with the mandatory column 0).
        //
        // KV hygiene: every verify writes rows [cur_past, cur_past+K]; a
        // rejected draft leaves stale rows ABOVE the last committed
        // position, but the next iteration's window starts at the new
        // cur_past and rewrites each row before any mask column exposes it,
        // so a stale row is never read.
        const int T_verify = k_drafts + 1;

        std::vector<int32_t> all_ids;
        all_ids.reserve(static_cast<size_t>(T_prompt) + max_new);
        all_ids.insert(all_ids.end(), prompt_ids.begin(), prompt_ids.end());
        all_ids.push_back(next_tok);

        std::unordered_map<int32_t, int> last_pos_by_tok;
        last_pos_by_tok.reserve(static_cast<size_t>(T_prompt) + max_new);
        for (int p = 0; p < T_prompt; ++p) {
            last_pos_by_tok[prompt_ids[static_cast<size_t>(p)]] = p;
        }

        std::vector<int32_t>     in_ids(T_verify, 0);
        std::vector<int32_t>     positions(T_verify, 0);
        std::vector<int64_t>     kv_idxs(T_verify, 0);
        std::vector<ggml_fp16_t> verify_mask(static_cast<size_t>(max_n_kv) * T_verify, mask_neg_inf);
        std::vector<int32_t>     predicted(T_verify, 0);
        // Where the next token to be fed sits in the seed's frame: 0 on the
        // first run (the prefill produced the token the seed starts at), and
        // advanced by every commitment so later runs keep the same alignment.
        int                      seed_pos     = 0;
        // Prior-transcript cursor: prior_ids index just after the last
        // committed token's aligned position. The lookup only searches a short
        // window ahead so one divergent token cannot jump the alignment far.
        int                      prior_cursor = 0;
        int                      prior_hits   = 0;
        constexpr int            kPriorWindow = 24;

        while (next_tok != eos_id && static_cast<int32_t>(generated_ids.size()) < max_new &&
               cur_past + T_verify <= max_n_kv) {
            const int64_t t_set0 = ggml_time_us();

            const auto it           = last_pos_by_tok.find(next_tok);
            const int  draft_origin = (it != last_pos_by_tok.end()) ? it->second : -1;
            int        prior_origin = -1;
            if (!prior_ids.empty()) {
                // Align on the last two committed tokens (bigram) — frequent
                // tokens like "," or " die" make a single-token match
                // ambiguous and jump the cursor — and fall back to the first
                // single-token match. The window reaches a few tokens back so
                // a cursor that jumped ahead can recover.
                const int32_t prev_tok = all_ids.size() >= 2 ? all_ids[all_ids.size() - 2] : -1;
                const int     beg      = std::max(1, prior_cursor - 4);
                const int     end      = std::min(static_cast<int>(prior_ids.size()), prior_cursor + kPriorWindow);
                int           uni      = -1;
                for (int p = beg - 1; p < end; ++p) {
                    if (prior_ids[static_cast<size_t>(p)] != next_tok) {
                        continue;
                    }
                    if (p > 0 && prior_ids[static_cast<size_t>(p - 1)] == prev_tok) {
                        prior_origin = p;
                        break;
                    }
                    if (uni < 0 && p >= prior_cursor - 1) {
                        uni = p;
                    }
                }
                if (prior_origin < 0) {
                    prior_origin = uni;
                }
            }

            in_ids[0]    = next_tok;
            positions[0] = cur_past;
            kv_idxs[0]   = cur_past;
            for (int c = 1; c < T_verify; ++c) {
                // The seeded columns come first and the 1-gram lookup fills
                // whatever the seed does not cover: outside the seed's window
                // there is no better guess, and a column rides along with the
                // mandatory one, so guessing is free and only the acceptance
                // chain decides what is committed.
                const int  seed_index = seed_pos + c;
                const int  src        = (draft_origin >= 0) ? (draft_origin + c) : -1;
                const bool from_seed  = seeded && seed_index < static_cast<int>(draft_seed->size());
                const int  prior_src  = prior_origin >= 0 ? prior_origin + c : -1;
                const bool from_prior = prior_src >= 0 && prior_src < static_cast<int>(prior_ids.size());
                in_ids[c] = from_seed  ? (*draft_seed)[static_cast<size_t>(seed_index)] :
                            from_prior ? prior_ids[static_cast<size_t>(prior_src)] :
                            (src >= 0 && src < static_cast<int>(all_ids.size())) ? all_ids[static_cast<size_t>(src)] :
                                                                                   next_tok;
                positions[c] = cur_past + c;
                kv_idxs[c]   = cur_past + c;
            }

            // Per-column causal mask: column c (absolute position
            // cur_past + c) attends to slots [0, cur_past + c].
            std::fill(verify_mask.begin(), verify_mask.end(), mask_neg_inf);
            for (int c = 0; c < T_verify; ++c) {
                std::fill(verify_mask.begin() + static_cast<size_t>(c) * max_n_kv,
                          verify_mask.begin() + static_cast<size_t>(c) * max_n_kv + cur_past + c + 1, mask_zero);
            }

            ggml_backend_tensor_set(vb.input_ids_in, in_ids.data(), 0, static_cast<size_t>(T_verify) * sizeof(int32_t));
            ggml_backend_tensor_set(vb.positions_in, positions.data(), 0,
                                    static_cast<size_t>(T_verify) * sizeof(int32_t));
            ggml_backend_tensor_set(vb.kv_idx_in, kv_idxs.data(), 0, static_cast<size_t>(T_verify) * sizeof(int64_t));
            ggml_backend_tensor_set(vb.mask_in, verify_mask.data(), 0, verify_mask.size() * sizeof(ggml_fp16_t));

            const int64_t t_set1 = ggml_time_us();
            t_step_set_us += t_set1 - t_set0;

            if (const ggml_status gs = ggml_backend_sched_graph_compute(cc->sched, vb.graph);
                gs != GGML_STATUS_SUCCESS) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "qwen3_asr verify: graph compute failed (%d)",
                        static_cast<int>(gs));
                cleanup_gpu();
                return TRANSCRIBE_ERR_BACKEND;
            }
            const int64_t t_comp1 = ggml_time_us();
            t_step_comp_us += t_comp1 - t_set1;
            ++n_graph_runs;

            ggml_backend_tensor_get(vb.out, predicted.data(), 0, static_cast<size_t>(T_verify) * sizeof(int32_t));

            // Commit predicted[0] (the mandatory step) plus every draft the
            // model confirmed: predicted[i] is valid iff in_ids[i] matched
            // the previous column's prediction, i.e. accept while
            // predicted[i-1] == in_ids[i].
            //
            // Map update order matters: a token's position may only enter the
            // map once it is a PREVIOUS occurrence — next_tok is pinned here
            // (after this iteration's lookup), and the last committed token
            // (the next iteration's next_tok) is deliberately NOT pinned so
            // that iteration's lookup can find an earlier occurrence instead
            // of its own tail position.
            last_pos_by_tok[next_tok] = cur_past;
            int n_commit              = 0;
            for (int i = 0; i < T_verify; ++i) {
                if (i > 0 && predicted[i - 1] != in_ids[i]) {
                    break;
                }
                const int32_t tok = predicted[i];
                next_tok          = tok;
                generated_ids.push_back(tok);
                all_ids.push_back(tok);
                ++n_commit;
                if (tok == eos_id || static_cast<int32_t>(generated_ids.size()) >= max_new) {
                    break;
                }
            }
            for (int j = 0; j + 1 < n_commit; ++j) {
                last_pos_by_tok[all_ids[all_ids.size() - static_cast<size_t>(n_commit) + j]] = cur_past + 1 + j;
            }

            cur_past += n_commit;
            seed_pos += n_commit;
            if (prior_origin >= 0) {
                prior_cursor = prior_origin + n_commit;
                prior_hits += n_commit - 1;
            }
            cc->kv_cache.n    = cur_past + 1;
            cc->kv_cache.head = cur_past + 1;
            t_step_get_us += ggml_time_us() - t_comp1;
        }
    }
    t_step_loop_us = ggml_time_us() - t_step_loop_start;
    n_steps        = static_cast<int>(generated_ids.size()) - 1;

    // Decode stopped at EOS (complete) or the generation budget / context width
    // (truncated). The caller decides what a truncated decode means: offline it
    // is a WARN plus TRANSCRIBE_ERR_OUTPUT_TRUNCATED, streaming it is the
    // expected end of a chunk that still has audio behind it, so it is NOT
    // logged or flagged here.
    const bool truncated = next_tok != eos_id;

    // Map granular counters to the debug-print shape. With graph reuse all
    // per-step overhead collapses to tensor_set; build/alloc/ctx_reset are
    // amortized in t_step_build_once_us.
    int64_t t_step_ctx_us   = 0;
    int64_t t_step_build_us = t_step_build_once_us;
    int64_t t_step_alloc_us = 0;

    // Strip trailing EOS if present (match the reference transcript).
    if (!generated_ids.empty() && generated_ids.back() == eos_id) {
        generated_ids.pop_back();
    }

    // Decode generated ids to text (Tokenizer::decode handles the "gpt2"
    // byte-level inversion natively).
    std::string raw_text = cm->tok.decode(generated_ids.data(), static_cast<int>(generated_ids.size()));
    cc->t_decode_us      = ggml_time_us() - t_dec_start;

    // Optional perf breakdown (finer split than the public mel/encode/decode
    // timings), gated on env var.
    if (transcribe::env::flag("TRANSCRIBE_PERF_DEBUG")) {
        const double ms     = 1.0 / 1000.0;
        const double sum_ms = (cc->t_mel_us + t_enc_build_us + cc->t_encode_us + t_enc_d2h_us + t_prefill_build_us +
                               t_prefill_compute_us + t_prefill_logits_us + t_step_loop_us) *
                              ms;
        const double per_step_ms = (n_steps > 0) ? (t_step_loop_us * ms / n_steps) : 0.0;
        log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                "qwen3_asr perf breakdown:\n"
                "  mel              %8.2f ms\n"
                "  enc_build        %8.2f ms  (graph + sched + uploads)\n"
                "  enc_compute      %8.2f ms\n"
                "  enc_d2h          %8.2f ms  (%d floats)\n"
                "  enc_cache        %8d     tokens reused, %d encoded\n"
                "  prefill_build    %8.2f ms  (kv_init + prompt + graph + sched + uploads)\n"
                "  prefill_compute  %8.2f ms  (T_prompt=%d, %d from the KV cache, %d prefilled)\n"
                "  prefill_logits   %8.2f ms  (readback + argmax, vocab=%d)\n"
                "  step_loop        %8.2f ms  (%d steps, %.2f ms/step)\n"
                "    ctx_reset    %8.2f ms  (%.3f ms/step)\n"
                "    graph_build  %8.2f ms  (%.3f ms/step)\n"
                "    sched_alloc  %8.2f ms  (%.3f ms/step)\n"
                "    tensor_set   %8.2f ms  (%.3f ms/step)\n"
                "    compute      %8.2f ms  (%.3f ms/step)\n"
                "    tensor_get   %8.2f ms  (%.3f ms/step)\n"
                "  ---\n"
                "  sum              %8.2f ms",
                cc->t_mel_us * ms, t_enc_build_us * ms, cc->t_encode_us * ms, t_enc_d2h_us * ms,
                static_cast<int>(cc->enc_host.size()), cached_tokens, T_enc - cached_tokens, t_prefill_build_us * ms,
                t_prefill_compute_us * ms, T_prompt, kv_reuse_len, T_graph, t_prefill_logits_us * ms, vocab,
                t_step_loop_us * ms, n_steps, per_step_ms, t_step_ctx_us * ms,
                (n_steps > 0) ? (t_step_ctx_us * ms / n_steps) : 0.0, t_step_build_us * ms,
                (n_steps > 0) ? (t_step_build_us * ms / n_steps) : 0.0, t_step_alloc_us * ms,
                (n_steps > 0) ? (t_step_alloc_us * ms / n_steps) : 0.0, t_step_set_us * ms,
                (n_steps > 0) ? (t_step_set_us * ms / n_steps) : 0.0, t_step_comp_us * ms,
                (n_steps > 0) ? (t_step_comp_us * ms / n_steps) : 0.0, t_step_get_us * ms,
                (n_steps > 0) ? (t_step_get_us * ms / n_steps) : 0.0, sum_ms);
        if (k_drafts > 0 && n_graph_runs > 0) {
            log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "  spec: k=%d  graph_runs=%d  tokens/run=%.2f", k_drafts, n_graph_runs,
                    static_cast<double>(n_steps + 1) / n_graph_runs);
        }
    }

    out->raw_text    = std::move(raw_text);
    out->truncated   = truncated;
    out->n_generated = static_cast<int>(generated_ids.size());
    out->gen_ids     = generated_ids;
    return TRANSCRIBE_OK;
}

namespace {  // reopen: everything below is file-local, as above the core.

// Offline single-shot transcription: resolve the language hint, run one decode
// pass over the whole clip, then envelope the text into the session result.
//
// All of the numeric work lives in run_decode_pass above; what stays here is
// everything that is specific to being *offline* — the language hint becoming a
// prompt seed, the auto-detect language readback, the single segment, and the
// per-run GPU teardown. Streaming reuses run_decode_pass and supplies its own
// answer to each of those.
transcribe_status run(transcribe_session *          session,
                      const float *                 pcm,
                      int                           n_samples,
                      const transcribe_run_params * params) {
    if (session == nullptr || pcm == nullptr || n_samples <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    auto * cc = static_cast<QwenAsrSession *>(session);
    auto * cm = static_cast<QwenAsrModel *>(cc->model);
    if (cm == nullptr) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Language hint. Null/empty == auto-detect (the LM emits its own
    // "language X<asr_text>" prefix, stripped by the parser below). A non-null
    // code is resolved to "language {Name}<asr_text>" tokens that seed the
    // assistant turn; a resolve failure surfaces as UNSUPPORTED_LANGUAGE.
    std::vector<int32_t>         lang_prefix_ids;
    const std::vector<int32_t> * lang_prefix_ptr = nullptr;
    if (params != nullptr && params->language != nullptr && params->language[0] != '\0') {
        if (const transcribe_status st = encode_language_prefix(cm->tok, params->language, lang_prefix_ids);
            st != TRANSCRIBE_OK) {
            return st;
        }
        lang_prefix_ptr = &lang_prefix_ids;
    }

    // Free GPU buffers (scheduler galloc + KV cache) after the run to prevent
    // memory accumulation across repeated calls. The session persists across
    // calls (e.g. Multi-STT extra models with multi_stt_keep_extra_models_loaded),
    // so releasing here lets CUDA's caching allocator reuse freed blocks on the
    // next run() rather than growing the cache (GPU memory leak on Windows).
    // run_decode_pass deliberately does not do this: the streaming path keeps
    // the scheduler and KV cache alive across chunks. Note this runs on the
    // error paths too, which is why it is a scope guard rather than a call
    // before each return.
    struct GpuGuard {
        QwenAsrSession * cc;

        ~GpuGuard() {
            cc->kv_cache.free();
            if (cc->sched != nullptr) {
                safe_sched_free(cc->sched);
                cc->sched = nullptr;
            }
        }
    } gpu_guard{ cc };

    DecodePassResult        pass;
    const transcribe_status st =
        run_decode_pass(session, pcm, n_samples, params, lang_prefix_ptr, /*max_new_tokens=*/0, &pass);
    if (st != TRANSCRIBE_OK) {
        return st;
    }

    cc->raw_text      = pass.raw_text;  // pre-envelope text, via transcribe_raw_text
    cc->was_truncated = pass.truncated;
    if (pass.truncated) {
        transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                            "qwen3_asr run: output truncated at %d tokens — decode reached the "
                            "generation budget before end-of-stream; the transcript may be "
                            "incomplete.",
                            pass.n_generated);
    }

    // Parse Qwen3-ASR output: auto-detect emits "language X<asr_text>text"
    // (strip prefix); forced emits "text" (we already seeded the prefix). The
    // split is unconditional — absent prefix leaves transcript_text = raw_text.
    std::string transcript_text = pass.raw_text;
    if (auto sep = pass.raw_text.find("<asr_text>"); sep != std::string::npos) {
        // Auto-detect path: surface the model-picked language name as
        // detected_language (reverse-mapped to BCP-47), but only when the
        // caller did NOT supply a hint (the field reports what the model told
        // us, not what we told it).
        if (lang_prefix_ptr == nullptr) {
            constexpr const char k_prefix[] = "language ";
            const size_t         name_start = pass.raw_text.find(k_prefix);
            if (name_start != std::string::npos && name_start < sep) {
                const size_t ns   = name_start + (sizeof(k_prefix) - 1);
                std::string  name = pass.raw_text.substr(ns, sep - ns);
                while (!name.empty() && (name.back() == ' ' || name.back() == '\t' || name.back() == '\n')) {
                    name.pop_back();
                }
                if (const char * bcp47 = bcp47_for_publisher_name(name); bcp47 != nullptr) {
                    cc->detected_language = bcp47;
                }
            }
        }
        transcript_text = pass.raw_text.substr(sep + std::strlen("<asr_text>"));
    }

    // Write full_text + a single segment (no timestamps; TIMESTAMPS_NONE).
    cc->full_text   = transcript_text;
    cc->result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    cc->has_result  = true;
    transcribe_session::SegmentEntry seg{};
    seg.text  = transcript_text;
    seg.t0_ms = 0;
    seg.t1_ms = static_cast<int64_t>(n_samples) * 1000 / static_cast<int64_t>(cm->hparams.fe_sample_rate);
    cc->segments.push_back(std::move(seg));

    // A truncated decode returns OUTPUT_TRUNCATED; the partial transcript above
    // stays readable (like an aborted run).
    return pass.truncated ? TRANSCRIBE_ERR_OUTPUT_TRUNCATED : TRANSCRIBE_OK;
}

// ===========================================================================
// Offline batched decode (transcribe_run_batch)
// ===========================================================================
//
// Prefill each utterance serially into its OWN slab of a batched KV cache
// (byte-identical to single-shot prefill, so it inherits correctness), then
// batch only the autoregressive step loop. Encoder + prefill stay
// per-utterance; the batch-axis math lives in the batched step graph.

namespace {

// Apply the session thread count to every backend behind the scheduler. The
// setting persists across sched_reset, so callers only need this once.
void apply_sched_threads(QwenAsrSession * cc) {
    transcribe::configure_sched_n_threads(cc->sched, cc->n_threads);
}

// Fresh per-utterance graph arena. Frees any prior compute_ctx and inits a
// no_alloc metadata context of `mb` MiB.
transcribe_status reset_compute_ctx(QwenAsrSession * cc, int mb) {
    if (cc->compute_ctx != nullptr) {
        ggml_free(cc->compute_ctx);
        cc->compute_ctx = nullptr;
    }
    ggml_init_params ip{};
    ip.mem_size     = static_cast<size_t>(mb) * 1024 * 1024;
    ip.mem_buffer   = nullptr;
    ip.no_alloc     = true;
    cc->compute_ctx = ggml_init(ip);
    return cc->compute_ctx != nullptr ? TRANSCRIBE_OK : TRANSCRIBE_ERR_OOM;
}

// Batched encoder: mel (parallel) + one encoder graph over all B utterances
// on the batch axis. Fills enc_hosts[b] = [d_enc, T_enc[b]], T_enc[b], valid[b].
// Real-row outputs are bit-identical to encode_one per utterance.
transcribe_status encode_all_batched(QwenAsrSession *                  cc,
                                     QwenAsrModel *                    cm,
                                     const float * const *             pcm,
                                     const int *                       n_samples,
                                     int                               n,
                                     std::vector<char> &               valid,
                                     std::vector<std::vector<float>> & enc_hosts,
                                     std::vector<int> &                T_enc_out,
                                     int64_t &                         mel_us,
                                     int64_t &                         enc_us) {
    const int n_mels = cm->hparams.enc_num_mel_bins;

    // Mel (parallel across utterances).
    std::vector<std::vector<float>> mels(n);
    std::vector<int>                mel_nf(n, 0);
    int                             n_threads = cc->n_threads;
    if (n_threads <= 0) {
        n_threads = transcribe::default_n_threads();
    }
    const int64_t t_mel0 = ggml_time_us();
    transcribe::parallel_for_all(n, n_threads, [&](int b) {
        if (pcm[b] == nullptr || n_samples[b] <= 0) {
            return true;  // valid stays 0
        }
        int nm = 0, nf = 0;
        if (cm->mel->compute(pcm[b], static_cast<size_t>(n_samples[b]), mels[b], nm, nf, /*n_threads=*/1) !=
            TRANSCRIBE_OK) {
            return true;  // leave invalid
        }
        mel_nf[b] = nf;
        valid[b]  = 1;
        return true;
    });
    mel_us += ggml_time_us() - t_mel0;

    // Per-utterance timing + packing geometry.
    std::vector<EncoderTiming> timings(n);
    int                        n_chunks_max = 1;
    bool                       any          = false;
    for (int b = 0; b < n; ++b) {
        if (!valid[b]) {
            continue;
        }
        timings[b] = compute_encoder_timing(mel_nf[b], cm->hparams);
        if (timings[b].n_chunks <= 0) {
            valid[b] = 0;
            continue;
        }
        n_chunks_max = std::max(n_chunks_max, timings[b].n_chunks);
        T_enc_out[b] = timings[b].T_enc;
        any          = true;
    }
    if (!any) {
        return TRANSCRIBE_OK;  // caller emits per-row errors
    }

    // Build batched encoder graph.
    if (const transcribe_status st = reset_compute_ctx(cc, 16); st != TRANSCRIBE_OK) {
        return st;
    }
    EncoderBuildBatched eb =
        build_encoder_graph_batched(cc->compute_ctx, cm->weights, cm->hparams, n_chunks_max, n, cc->encoder_use_flash);
    if (eb.graph == nullptr || eb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    if (cc->sched == nullptr) {
        cc->sched = ggml_backend_sched_new(cm->plan.scheduler_list.data(), nullptr,
                                           static_cast<int>(cm->plan.scheduler_list.size()), 16384, false, true);
        if (cc->sched == nullptr) {
            return TRANSCRIBE_ERR_BACKEND;
        }
    }
    ggml_backend_sched_reset(cc->sched);
    if (!alloc_inference_graph(cc->sched, eb.graph)) {
        return TRANSCRIBE_ERR_OOM;
    }

    const int    mel_per_chunk   = cm->hparams.enc_n_window * 2;
    const int    T_per_chunk     = eb.T_per_chunk;
    const int    T_pad_max       = eb.T_pad_max;
    const size_t per_chunk_elems = static_cast<size_t>(mel_per_chunk) * n_mels;

    // Pack mel: [mel_per_chunk, n_mels, 1, n*n_chunks_max]; utterance b's
    // chunk c at N = b*n_chunks_max + c, zero-padded beyond n_chunks[b].
    {
        std::vector<float> packed(per_chunk_elems * static_cast<size_t>(n) * n_chunks_max, 0.0f);
        for (int b = 0; b < n; ++b) {
            if (!valid[b]) {
                continue;
            }
            const EncoderTiming & t = timings[b];
            for (int c = 0; c < t.n_chunks; ++c) {
                const int    tail = (c == t.n_chunks - 1) ? t.last_chunk_real_mel : mel_per_chunk;
                const size_t nidx = static_cast<size_t>(b) * n_chunks_max + c;
                for (int m = 0; m < n_mels; ++m) {
                    const float * src =
                        mels[b].data() + static_cast<size_t>(m) * mel_nf[b] + static_cast<size_t>(c) * mel_per_chunk;
                    float * dst = packed.data() + nidx * per_chunk_elems + static_cast<size_t>(m) * mel_per_chunk;
                    std::memcpy(dst, src, static_cast<size_t>(tail) * sizeof(float));
                }
            }
        }
        ggml_backend_tensor_set(eb.mel_in, packed.data(), 0, packed.size() * sizeof(float));
    }

    // Positional embedding (shared across chunks/utterances).
    {
        std::vector<float> pe = build_sinusoid_pe(cm->hparams.enc_d_model, T_per_chunk);
        ggml_backend_tensor_set(eb.pos_emb_in, pe.data(), 0, pe.size() * sizeof(float));
    }

    // NOTE: the key-pad mask is not needed — the bounded chunked subsample
    // trims each chunk's own conv padding before the attention blocks. What
    // is uploaded here is the *window* mask (one slab per utterance), which
    // is a different thing: it bounds attention to the reference's
    // n_window_infer-sized blocks and isolates each utterance's unused chunk
    // slots. Null when the whole batch fits in one window.
    if (eb.mask_in != nullptr) {
        const size_t             slab = static_cast<size_t>(T_pad_max) * static_cast<size_t>(T_pad_max);
        std::vector<ggml_fp16_t> masks(slab * static_cast<size_t>(n));
        for (int b = 0; b < n; ++b) {
            fill_encoder_window_mask(masks.data() + static_cast<size_t>(b) * slab, T_pad_max, T_pad_max,
                                     eb.window_tokens, T_enc_out[b]);
        }
        ggml_backend_tensor_set(eb.mask_in, masks.data(), 0, masks.size() * sizeof(ggml_fp16_t));
    }
    apply_sched_threads(cc);

    const int64_t t_enc0 = ggml_time_us();
    if (ggml_backend_sched_graph_compute(cc->sched, eb.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_BACKEND;
    }
    enc_us += ggml_time_us() - t_enc0;

    // Readback [d_enc, T_pad_max, n]; slice each utterance's first T_enc[b] rows.
    const int          d_enc = static_cast<int>(eb.out->ne[0]);
    std::vector<float> out_all(static_cast<size_t>(d_enc) * T_pad_max * n);
    ggml_backend_tensor_get(eb.out, out_all.data(), 0, out_all.size() * sizeof(float));
    for (int b = 0; b < n; ++b) {
        if (!valid[b]) {
            continue;
        }
        const int te = T_enc_out[b];
        enc_hosts[b].resize(static_cast<size_t>(d_enc) * te);
        const float * src = out_all.data() + static_cast<size_t>(b) * T_pad_max * d_enc;
        std::memcpy(enc_hosts[b].data(), src, static_cast<size_t>(d_enc) * te * sizeof(float));
    }
    return TRANSCRIBE_OK;
}

// Batched prefill: one graph processes all B prompts, writing each utterance's
// KV into its slab and returning the first generated token per utterance. The
// caller must have allocated cc->kv_cache_batch with n_ctx >= max T_prompt and
// computed prompt_ids[b] / T_prompt[b] / prefix_len. Collapses B per-utterance
// prefill graph-builds + sched-allocs into one.
transcribe_status prefill_all_batched(QwenAsrSession *                          cc,
                                      QwenAsrModel *                            cm,
                                      const std::vector<char> &                 valid,
                                      const std::vector<std::vector<int32_t>> & prompt_ids,
                                      const std::vector<int> &                  T_prompt,
                                      int                                       prefix_len,
                                      const std::vector<std::vector<float>> &   enc_hosts,
                                      const std::vector<int> &                  T_enc,
                                      int                                       n,
                                      std::vector<int32_t> &                    first_tok_out) {
    int T_prompt_max = 0, T_enc_max = 0;
    for (int b = 0; b < n; ++b) {
        if (!valid[b]) {
            continue;
        }
        T_prompt_max = std::max(T_prompt_max, T_prompt[b]);
        T_enc_max    = std::max(T_enc_max, T_enc[b]);
    }
    if (T_prompt_max == 0) {
        return TRANSCRIBE_OK;
    }
    T_enc_max = std::max(1, T_enc_max);

    if (const transcribe_status st = reset_compute_ctx(cc, 32); st != TRANSCRIBE_OK) {
        return st;
    }
    PrefillBuildBatched pb = build_prefill_graph_batched(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache_batch,
                                                         T_prompt_max, T_enc_max, n, cc->decoder_use_flash);
    if (pb.graph == nullptr || pb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }

    ggml_backend_sched_reset(cc->sched);
    if (!alloc_inference_graph(cc->sched, pb.graph)) {
        return TRANSCRIBE_ERR_OOM;
    }

    const int d_enc = cm->hparams.enc_output_dim;

    // input_ids [T_prompt_max, n] (pad 0).
    {
        std::vector<int32_t> ids(static_cast<size_t>(T_prompt_max) * n, 0);
        for (int b = 0; b < n; ++b) {
            if (!valid[b]) {
                continue;
            }
            std::memcpy(ids.data() + static_cast<size_t>(b) * T_prompt_max, prompt_ids[b].data(),
                        static_cast<size_t>(T_prompt[b]) * sizeof(int32_t));
        }
        ggml_backend_tensor_set(pb.input_ids_in, ids.data(), 0, ids.size() * sizeof(int32_t));
    }
    // Audio injection by elementwise blend (see decoder.h): audio_dense holds
    // each utterance's enc_out embeds scattered into their prompt positions,
    // keep is 0 there and 1 elsewhere. (d_enc == dec_hidden, enforced.)
    {
        std::vector<float> audio_dense(static_cast<size_t>(d_enc) * T_prompt_max * n, 0.0f);
        std::vector<float> keep(static_cast<size_t>(T_prompt_max) * n, 1.0f);
        for (int b = 0; b < n; ++b) {
            const int te = valid[b] ? T_enc[b] : 0;
            // enc_hosts[b] is [d_enc, te] column-major; audio token j lands at
            // prompt position prefix_len+j, flat column b*T_prompt_max+pos.
            for (int j = 0; j < te; ++j) {
                const size_t dst_col = static_cast<size_t>(b) * T_prompt_max + (prefix_len + j);
                std::memcpy(audio_dense.data() + dst_col * d_enc, enc_hosts[b].data() + static_cast<size_t>(j) * d_enc,
                            static_cast<size_t>(d_enc) * sizeof(float));
                keep[dst_col] = 0.0f;
            }
        }
        ggml_backend_tensor_set(pb.audio_dense_in, audio_dense.data(), 0, audio_dense.size() * sizeof(float));
        ggml_backend_tensor_set(pb.keep_mask_in, keep.data(), 0, keep.size() * sizeof(float));
    }
    // positions [T_prompt_max] 0..T-1 (shared).
    {
        std::vector<int32_t> pos(T_prompt_max);
        for (int t = 0; t < T_prompt_max; ++t) {
            pos[t] = t;
        }
        ggml_backend_tensor_set(pb.positions_in, pos.data(), 0, pos.size() * sizeof(int32_t));
    }
    // causal mask [T_prompt_max, T_prompt_max] f16 (shared): m[q*T+k]=0 if k<=q.
    {
        const ggml_fp16_t        mz = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t        mn = ggml_fp32_to_fp16(-INFINITY);
        std::vector<ggml_fp16_t> mask(static_cast<size_t>(T_prompt_max) * T_prompt_max, mn);
        for (int q = 0; q < T_prompt_max; ++q) {
            std::fill(mask.begin() + static_cast<size_t>(q) * T_prompt_max,
                      mask.begin() + static_cast<size_t>(q) * T_prompt_max + q + 1, mz);
        }
        ggml_backend_tensor_set(pb.mask_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }
    // kv_idx [T_prompt_max, n] i64: idx[t,b] = t.
    {
        std::vector<int64_t> kidx(static_cast<size_t>(T_prompt_max) * n);
        for (int b = 0; b < n; ++b) {
            for (int t = 0; t < T_prompt_max; ++t) {
                kidx[static_cast<size_t>(b) * T_prompt_max + t] = t;
            }
        }
        ggml_backend_tensor_set(pb.kv_idx_in, kidx.data(), 0, kidx.size() * sizeof(int64_t));
    }
    // last_idx [1, n] i32: each utterance's last real position.
    {
        std::vector<int32_t> lidx(n, 0);
        for (int b = 0; b < n; ++b) {
            lidx[b] = valid[b] ? (T_prompt[b] - 1) : 0;
        }
        ggml_backend_tensor_set(pb.last_idx_in, lidx.data(), 0, lidx.size() * sizeof(int32_t));
    }

    apply_sched_threads(cc);
    if (ggml_backend_sched_graph_compute(cc->sched, pb.graph) != GGML_STATUS_SUCCESS) {
        return TRANSCRIBE_ERR_BACKEND;
    }

    std::vector<int32_t> amax(n, 0);
    ggml_backend_tensor_get(pb.out, amax.data(), 0, amax.size() * sizeof(int32_t));
    for (int b = 0; b < n; ++b) {
        first_tok_out[b] = amax[b];
    }
    return TRANSCRIBE_OK;
}

// Decode one utterance's generated token ids into a ResultSet (strip EOS,
// detokenize, parse the Qwen3-ASR "language X<asr_text>…" envelope). Mirrors
// run()'s output-parsing tail; `lang_prefix_ptr` non-null means the caller
// forced a language (so we don't surface a detected one).
transcribe_session::ResultSet finalize_utterance(QwenAsrModel *               cm,
                                                 std::vector<int32_t>         generated_ids,
                                                 const std::vector<int32_t> * lang_prefix_ptr,
                                                 int                          n_samples) {
    transcribe_session::ResultSet rs;
    const int32_t                 eos_id = cm->hparams.eos_token_id;
    if (!generated_ids.empty() && generated_ids.back() == eos_id) {
        generated_ids.pop_back();
    }

    std::string raw_text = cm->tok.decode(generated_ids.data(), static_cast<int>(generated_ids.size()));

    std::string transcript_text = raw_text;
    if (auto sep = raw_text.find("<asr_text>"); sep != std::string::npos) {
        if (lang_prefix_ptr == nullptr) {
            constexpr const char k_prefix[] = "language ";
            const size_t         name_start = raw_text.find(k_prefix);
            if (name_start != std::string::npos && name_start < sep) {
                const size_t ns   = name_start + (sizeof(k_prefix) - 1);
                std::string  name = raw_text.substr(ns, sep - ns);
                while (!name.empty() && (name.back() == ' ' || name.back() == '\t' || name.back() == '\n')) {
                    name.pop_back();
                }
                for (const auto & e : k_qwen3_asr_language_names) {
                    if (name == e.pub_name) {
                        rs.detected_language = e.bcp47;
                        break;
                    }
                }
            }
        }
        transcript_text = raw_text.substr(sep + std::strlen("<asr_text>"));
    }

    rs.full_text   = transcript_text;
    rs.raw_text    = raw_text;
    rs.result_kind = TRANSCRIBE_TIMESTAMPS_NONE;
    rs.has_result  = true;
    rs.status      = TRANSCRIBE_OK;
    transcribe_session::SegmentEntry seg{};
    seg.text  = transcript_text;
    seg.t0_ms = 0;
    seg.t1_ms = static_cast<int64_t>(n_samples) * 1000 / static_cast<int64_t>(cm->hparams.fe_sample_rate);
    rs.segments.push_back(std::move(seg));
    return rs;
}

// Serial fallback: run() per utterance, capturing each into batch_results.
// Used when batched decode is unavailable (flash off) or as the safe path.
transcribe_status run_batch_serial(QwenAsrSession *              cc,
                                   const float * const *         pcm,
                                   const int *                   n_samples,
                                   int                           n,
                                   const transcribe_run_params * params) {
    for (int i = 0; i < n; ++i) {
        if (cc->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        const transcribe_status st = (pcm[i] == nullptr || n_samples[i] <= 0) ? TRANSCRIBE_ERR_INVALID_ARG :
                                                                                run(cc, pcm[i], n_samples[i], params);
        if (st == TRANSCRIBE_OK) {
            cc->batch_results.push_back(cc->capture_result(st));
        } else {
            transcribe_session::ResultSet rs;
            rs.status = st;
            cc->batch_results.push_back(std::move(rs));
        }
    }
    return TRANSCRIBE_OK;
}

}  // namespace

transcribe_status run_batch(transcribe_session *          session,
                            const float * const *         pcm,
                            const int *                   n_samples,
                            int                           n,
                            const transcribe_run_params * params) {
    if (session == nullptr || pcm == nullptr || n_samples == nullptr || n <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    auto * cc = static_cast<QwenAsrSession *>(session);
    auto * cm = static_cast<QwenAsrModel *>(cc->model);
    if (cm == nullptr || cm->plan.scheduler_list.empty()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (!cm->mel.has_value()) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    // Batched decode requires the flash-attention step path and dump-free
    // operation. Fall back to the serial loop otherwise (same results).
    if (!cc->decoder_use_flash || transcribe::debug::enabled() || n == 1) {
        return run_batch_serial(cc, pcm, n_samples, n, params);
    }

    transcribe::debug::init();

    // Shared language hint (v1: one run_params across the batch).
    std::vector<int32_t>         lang_prefix_ids;
    const std::vector<int32_t> * lang_prefix_ptr = nullptr;
    if (params != nullptr && params->language != nullptr && params->language[0] != '\0') {
        if (encode_language_prefix(cm->tok, params->language, lang_prefix_ids) != TRANSCRIBE_OK) {
            return TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE;
        }
        lang_prefix_ptr = &lang_prefix_ids;
    }

    // Pass 1: per-utterance encoder + prefill into KV slabs.
    std::vector<std::vector<int32_t>> generated(n);
    std::vector<int>                  T_prompt(n, 0);
    std::vector<int32_t>              next_tok(n, 0);
    std::vector<int>                  n_past(n, 0);
    std::vector<char>                 valid(n, 0);  // utterance produced a usable prefill
    int64_t                           mel_us = 0, enc_us = 0;

    // Encode every utterance first so we can size the batched cache to the
    // real max prompt length before allocating it.
    std::vector<std::vector<float>> enc_hosts(n);
    std::vector<int>                T_enc(n, 0);
    const int64_t                   t_encpass0 = ggml_time_us();
    if (const transcribe_status st =
            encode_all_batched(cc, cm, pcm, n_samples, n, valid, enc_hosts, T_enc, mel_us, enc_us);
        st != TRANSCRIBE_OK) {
        return st;
    }
    const int64_t enc_pass_us = ggml_time_us() - t_encpass0;

    // Prompt length bound → max_n_kv and batched-cache n_ctx. Build and keep
    // each utterance's prompt token ids for the batched prefill.
    int                               max_T_prompt = 0;
    int                               max_T_enc    = 0;
    int                               prefix_len   = 0;
    // Per-utterance terminal status for rejected rows. Defaults to INVALID_ARG;
    // over-length rows below are upgraded to INPUT_TOO_LONG.
    const int                         ceiling      = qwen3_context_ceiling(cc->n_ctx, cm->hparams);
    std::vector<transcribe_status>    fail_status(n, TRANSCRIBE_ERR_INVALID_ARG);
    std::vector<std::vector<int32_t>> prompt_ids(n);
    for (int b = 0; b < n; ++b) {
        if (!valid[b]) {
            continue;
        }
        std::vector<int64_t> ap;
        build_prompt_tokens(cm->hparams, cm->chat_tokens, T_enc[b], lang_prefix_ptr, prompt_ids[b], ap);
        T_prompt[b] = static_cast<int>(prompt_ids[b].size());
        prefix_len  = ap.empty() ? 0 : static_cast<int>(ap.front());
        // Same gate as single-shot run(); the rest of the batch still runs.
        if (T_prompt[b] + k_gen_reserve > ceiling) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr run_batch: utterance %d input too long — %d audio + "
                                "%d prompt tokens exceed the %d-token context. See "
                                "transcribe_capabilities.max_audio_ms.",
                                b, T_enc[b], T_prompt[b] - T_enc[b], ceiling);
            valid[b]       = 0;
            fail_status[b] = TRANSCRIBE_ERR_INPUT_TOO_LONG;
            continue;
        }
        max_T_prompt = std::max(max_T_prompt, T_prompt[b]);
        max_T_enc    = std::max(max_T_enc, T_enc[b]);
    }
    if (max_T_prompt == 0) {
        // No usable utterance — emit per-row errors and return.
        for (int b = 0; b < n; ++b) {
            transcribe_session::ResultSet rs;
            rs.status = fail_status[b];
            cc->batch_results.push_back(std::move(rs));
        }
        return TRANSCRIBE_OK;
    }
    const int max_new =
        transcribe::pick_decode_budget(transcribe::predict_transcript_tokens(max_T_enc, cm->limits.ms_per_audio_token),
                                       k_gen_reserve, max_T_prompt, ceiling);
    int max_n_kv = 1024;
    while (max_n_kv < max_T_prompt + max_new) {
        max_n_kv *= 2;
    }
    // Clamp the pow2 round-up to the ceiling (the per-utterance gate guarantees
    // every valid row still fits).
    if (max_n_kv > ceiling) {
        max_n_kv = ceiling;
    }

    // Allocate / reuse the batched KV cache (n_ctx == max_n_kv, n slabs).
    ggml_type kv_type = (cc->kv_type == TRANSCRIBE_KV_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
    if (cc->kv_cache_batch.self_k == nullptr || cc->kv_batch_cap != n || cc->kv_batch_n_ctx != max_n_kv) {
        cc->kv_cache_batch.free();
        if (!transcribe::causal_lm::kv_init_batched(cc->kv_cache_batch, cm->plan.primary, max_n_kv,
                                                    cm->hparams.dec_n_kv_heads, cm->hparams.dec_head_dim,
                                                    cm->hparams.dec_n_layers, n, kv_type)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "qwen3_asr run_batch: batched KV cache allocation failed "
                                "(n_ctx=%d x %d utterances) — out of memory. Lower "
                                "transcribe_session_params.n_ctx or the batch size.",
                                max_n_kv, n);
            return TRANSCRIBE_ERR_OOM;
        }
        cc->kv_batch_cap   = n;
        cc->kv_batch_n_ctx = max_n_kv;
    } else if (cc->kv_cache_batch.buffer != nullptr) {
        ggml_backend_buffer_clear(cc->kv_cache_batch.buffer, 0);
    }

    const int64_t t_prefpass0 = ggml_time_us();
    {
        std::vector<int32_t> first_tok(n, 0);
        if (const transcribe_status st =
                prefill_all_batched(cc, cm, valid, prompt_ids, T_prompt, prefix_len, enc_hosts, T_enc, n, first_tok);
            st != TRANSCRIBE_OK) {
            return st;
        }
        for (int b = 0; b < n; ++b) {
            if (!valid[b]) {
                continue;
            }
            n_past[b]   = T_prompt[b];
            next_tok[b] = first_tok[b];
            generated[b].push_back(first_tok[b]);
        }
    }
    const int64_t prefill_pass_us = ggml_time_us() - t_prefpass0;

    // Pass 2: batched step loop (shared causal_lm driver).
    const int32_t eos_id = cm->hparams.eos_token_id;

    if (const transcribe_status st = reset_compute_ctx(cc, 16); st != TRANSCRIBE_OK) {
        return st;
    }
    StepBuildBatched sb = build_step_graph_batched(cc->compute_ctx, cm->weights, cm->hparams, cc->kv_cache_batch,
                                                   max_n_kv, n, cc->decoder_use_flash);
    if (sb.graph == nullptr || sb.out == nullptr) {
        return TRANSCRIBE_ERR_GGUF;
    }
    ggml_backend_sched_reset(cc->sched);
    if (!alloc_inference_graph(cc->sched, sb.graph)) {
        return TRANSCRIBE_ERR_OOM;
    }

    transcribe::causal_lm::StepBatchedIO io{};
    io.input_ids = sb.input_ids_in;
    io.positions = sb.position_in;
    io.kv_idx    = sb.kv_idx_in;
    io.mask      = sb.mask_in;
    io.argmax    = sb.out;
    io.graph     = sb.graph;

    transcribe::causal_lm::StepBatchedState step_state;
    step_state.valid    = valid;
    step_state.next_tok = next_tok;
    step_state.n_past   = n_past;

    transcribe::causal_lm::StepLoopStats step_stats;
    std::vector<char>                    truncated;
    if (const transcribe_status st = transcribe::causal_lm::run_batched_step_loop(
            cc, cc->sched, io, n, max_n_kv, eos_id, max_new, step_state, generated, &step_stats, &truncated);
        st != TRANSCRIBE_OK) {
        return st;
    }
    const int64_t step_us = step_stats.step_us;
    const int     n_steps = step_stats.n_steps;

    // Capture per-utterance results.
    const int valid_count = std::max(1, static_cast<int>(std::count(valid.begin(), valid.end(), char(1))));
    for (int b = 0; b < n; ++b) {
        if (!valid[b]) {
            transcribe_session::ResultSet rs;
            rs.status = fail_status[b];
            cc->batch_results.push_back(std::move(rs));
            continue;
        }
        transcribe_session::ResultSet rs = finalize_utterance(cm, generated[b], lang_prefix_ptr, n_samples[b]);
        // Per-utterance truncation parity with the single-shot path.
        if (b < static_cast<int>(truncated.size()) && truncated[b]) {
            cc->was_truncated = true;
            rs.status         = TRANSCRIBE_ERR_OUTPUT_TRUNCATED;
        }
        rs.t_mel_us    = mel_us / valid_count;
        rs.t_encode_us = enc_us / valid_count;
        rs.t_decode_us = step_us / valid_count;
        cc->batch_results.push_back(std::move(rs));
    }

    if (transcribe::env::flag("TRANSCRIBE_PERF_DEBUG")) {
        log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
                "qwen3_asr run_batch: n=%d valid=%d max_n_kv=%d steps=%d (all phases batched x%d)\n"
                "  enc_pass=%.1fms (mel=%.1f parallel + enc_compute=%.1f, 1 graph)\n"
                "  prefill_pass=%.1fms (1 batched graph: build/sched/compute/readback)\n"
                "  step_loop=%.1fms (%.2fms/step)",
                n, valid_count, max_n_kv, n_steps, valid_count, enc_pass_us / 1000.0, mel_us / 1000.0, enc_us / 1000.0,
                prefill_pass_us / 1000.0, step_us / 1000.0, n_steps > 0 ? step_us / 1000.0 / n_steps : 0.0);
    }

    return TRANSCRIBE_OK;
}

}  // namespace

extern const Arch arch = {
    /* .name             = */ "qwen3_asr",
    /* .load             = */ load,
    /* .init_context     = */ init_context,
    /* .run              = */ run,
    /* .run_batch        = */ run_batch,
    // Streaming is the Confucius4-R2T2 variant's surface (r2t2-stream.cpp).
    // The hooks are installed unconditionally and gate themselves on the
    // package marker, so a non-R2T2 qwen3_asr model reports
    // supports_streaming = false and rejects the stream extension rather than
    // exposing a surface it cannot serve. The triple is all-or-nothing: the
    // dispatcher returns NOT_IMPLEMENTED if any of the three is NULL, so they
    // are installed together.
    /* .stream_validate  = */ r2t2_stream_validate,
    /* .stream_begin     = */ r2t2_stream_begin,
    /* .stream_feed      = */ r2t2_stream_feed,
    /* .stream_finalize  = */ r2t2_stream_finalize,
    /* .stream_reset     = */ r2t2_stream_reset,
    /* .accepts_ext_kind = */ r2t2_accepts_ext_kind,
};

}  // namespace transcribe::qwen3_asr
