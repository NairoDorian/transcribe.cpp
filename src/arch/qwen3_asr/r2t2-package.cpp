// Native transcribe loader for the standalone R2T2 GGUF package.
// This reads embedded data only; it does not link or execute audio.cpp.
#include "r2t2-package.h"
#include "ggml.h"
#include "gguf.h"
#include "transcribe-log.h"
#include "third_party/cJSON/cJSON.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

namespace transcribe::qwen3_asr {
namespace {
using Json = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;
const cJSON * field(const cJSON * obj, const char * name) {
    auto * v = cJSON_GetObjectItemCaseSensitive(obj, name);
    if (!v) throw std::runtime_error(std::string("R2T2 missing field: ") + name);
    return v;
}
uint32_t number(const cJSON * obj, const char * name) {
    const auto * v = field(obj, name);
    if (!cJSON_IsNumber(v) || !std::isfinite(v->valuedouble) || v->valuedouble < 0 ||
        v->valuedouble > UINT32_MAX || std::floor(v->valuedouble) != v->valuedouble)
        throw std::runtime_error(std::string("R2T2 invalid integer: ") + name);
    return static_cast<uint32_t>(v->valuedouble);
}
const char * str(const cJSON * v) {
    if (!cJSON_IsString(v) || !v->valuestring) throw std::runtime_error("R2T2 expected JSON string");
    return v->valuestring;
}
int64_t array_key(gguf_context * g, const char * name, gguf_type type) {
    const auto k = gguf_find_key(g, name);
    if (k < 0 || gguf_get_kv_type(g,k) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g,k) != type)
        throw std::runtime_error(std::string("R2T2 invalid package array: ") + name);
    return k;
}
Json sidecar(gguf_context * g, const char * name) {
    auto nk=array_key(g,"audiocpp.embedded_files.names",GGUF_TYPE_STRING);
    auto ok=array_key(g,"audiocpp.embedded_files.offsets",GGUF_TYPE_UINT64);
    auto dk=array_key(g,"audiocpp.embedded_files.data",GGUF_TYPE_UINT8);
    auto n=gguf_get_arr_n(g,nk);
    if (gguf_get_arr_n(g,ok)!=n+1) throw std::runtime_error("R2T2 invalid sidecar offsets");
    auto * offsets=static_cast<const uint64_t *>(gguf_get_arr_data(g,ok));
    auto * data=static_cast<const char *>(gguf_get_arr_data(g,dk));
    auto bytes=gguf_get_arr_n(g,dk);
    for (size_t i=0;i<n;++i) {
        if (std::strcmp(gguf_get_arr_str(g,nk,i),name)!=0) continue;
        if (offsets[i]>offsets[i+1] || offsets[i+1]>bytes || offsets[i+1]-offsets[i]>64*1024*1024)
            throw std::runtime_error("R2T2 invalid sidecar range");
        Json j(cJSON_ParseWithLength(data+offsets[i],static_cast<size_t>(offsets[i+1]-offsets[i])),cJSON_Delete);
        if (!j) throw std::runtime_error(std::string("R2T2 invalid JSON: ")+name);
        return j;
    }
    throw std::runtime_error(std::string("R2T2 missing sidecar: ")+name);
}
void strings(gguf_context * g, const char * key, const std::vector<std::string> & values) {
    std::vector<const char *> ptrs; ptrs.reserve(values.size());
    for (const auto & s:values) ptrs.push_back(s.c_str());
    gguf_set_arr_str(g,key,ptrs.data(),ptrs.size());
}
#include "r2t2-tensor-map.inc"
}

bool is_r2t2_package(const gguf_context * g) {
    if (g == nullptr) return false;
    const int64_t k = gguf_find_key(g, "audiocpp.model_spec.family");
    if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_STRING) return false;
    return std::strcmp(gguf_get_val_str(g, k), "confucius4_r2t2") == 0;
}

transcribe_status prepare_r2t2_metadata(gguf_context * g) {
    try {
        auto config=sidecar(g,"config.json"); auto tok=sidecar(g,"tokenizer.json");
        auto * thinker=cJSON_GetObjectItemCaseSensitive(config.get(),"thinker_config");
        if (!thinker) thinker=config.get();
        auto * enc=field(thinker,"audio_config"); auto * dec=field(thinker,"text_config");
        auto u=[&](const std::string & k,uint32_t v){gguf_set_val_u32(g,k.c_str(),v);};
        auto s=[&](const std::string & k,const char * v){gguf_set_val_str(g,k.c_str(),v);};
        auto f=[&](const std::string & k,float v){gguf_set_val_f32(g,k.c_str(),v);};
        const std::pair<const char *,const char *> emap[]={
            {"n_layers","encoder_layers"},{"d_model","d_model"},{"n_heads","encoder_attention_heads"},
            {"ffn_dim","encoder_ffn_dim"},{"num_mel_bins","num_mel_bins"},{"downsample_hidden","downsample_hidden_size"},
            {"output_dim","output_dim"},{"n_window","n_window"},{"n_window_infer","n_window_infer"},{"conv_chunksize","conv_chunksize"}};
        for (auto kv:emap) u(std::string("stt.qwen3_asr.encoder.")+kv.first,number(enc,kv.second));
        u("stt.qwen3_asr.encoder.max_source_positions",cJSON_GetObjectItemCaseSensitive(enc,"max_source_positions")?number(enc,"max_source_positions"):1500);
        s("stt.qwen3_asr.encoder.activation",str(field(enc,"activation_function")));
        const std::pair<const char *,const char *> dmap[]={
            {"n_layers","num_hidden_layers"},{"hidden_size","hidden_size"},{"intermediate_size","intermediate_size"},
            {"n_heads","num_attention_heads"},{"n_kv_heads","num_key_value_heads"},{"head_dim","head_dim"},
            {"max_position_embeddings","max_position_embeddings"},{"vocab_size","vocab_size"}};
        for(auto kv:dmap) u(std::string("stt.qwen3_asr.decoder.")+kv.first,number(dec,kv.second));
        s("stt.qwen3_asr.decoder.hidden_act","silu");
        f("stt.qwen3_asr.decoder.rms_norm_eps",static_cast<float>(field(dec,"rms_norm_eps")->valuedouble));
        // Declared const because the fallback below comes from `field`, which
        // returns a const view; the direct lookup returns a mutable pointer that
        // converts to it, so one type serves both branches.
        const cJSON * rope=cJSON_GetObjectItemCaseSensitive(dec,"rope_theta");
        if (!rope) rope=field(field(dec,"rope_parameters"),"rope_theta");
        f("stt.qwen3_asr.decoder.rope_theta",static_cast<float>(rope->valuedouble));
        u("stt.qwen3_asr.decoder.rope_mrope_section_t",24);u("stt.qwen3_asr.decoder.rope_mrope_section_h",20);u("stt.qwen3_asr.decoder.rope_mrope_section_w",20);
        gguf_set_val_bool(g,"stt.qwen3_asr.decoder.rope_mrope_interleaved",true);
        auto * tied=cJSON_GetObjectItemCaseSensitive(dec,"tie_word_embeddings");
        if (tied && cJSON_IsFalse(tied)) throw std::runtime_error("R2T2 untied output projection is unsupported");
        gguf_set_val_bool(g,"stt.qwen3_asr.decoder.tie_word_embeddings",true);
        for (auto k:{"audio_token_id","audio_start_token_id","audio_end_token_id"}) u(std::string("stt.qwen3_asr.")+k,number(thinker,k));
        s("stt.variant",k_r2t2_variant);
        s("stt.frontend.type","mel"); u("stt.frontend.sample_rate",16000);u("stt.frontend.num_mels",128);
        u("stt.frontend.n_fft",400);u("stt.frontend.win_length",400);u("stt.frontend.hop_length",160);
        s("stt.frontend.window","hann_periodic");s("stt.frontend.normalize","per_utterance");
        s("stt.frontend.pad_mode","reflect");s("stt.frontend.mel_norm","slaney");
        gguf_set_val_bool(g,"stt.frontend.center",true);
        f("stt.frontend.pre_emphasis",0);f("stt.frontend.dither",0);f("stt.frontend.f_min",0);f("stt.frontend.f_max",8000);
        u("stt.frontend.chunk_length",30);u("stt.frontend.n_samples",480000);u("stt.frontend.nb_max_frames",3000);
        uint32_t vocab=number(dec,"vocab_size");
        if (vocab<1000 || vocab>1000000) throw std::runtime_error("R2T2 invalid vocabulary size");
        std::vector<std::string> tokens(vocab);std::vector<int32_t> types(vocab,1);
        for(uint32_t i=0;i<vocab;++i) tokens[i]="[PAD"+std::to_string(i)+"]";
        const auto * model=field(tok.get(),"model");
        for(auto * item=field(model,"vocab")->child;item;item=item->next) {
            if (!cJSON_IsNumber(item) || item->valuedouble<0 || item->valuedouble>=vocab || !item->string)
                throw std::runtime_error("R2T2 invalid vocabulary entry");
            tokens[static_cast<size_t>(item->valuedouble)]=item->string;
        }
        auto * added=field(tok.get(),"added_tokens");
        for(auto * item=added->child;item;item=item->next) {
            auto id=number(item,"id");if(id>=vocab) throw std::runtime_error("R2T2 added token out of range");
            tokens[id]=str(field(item,"content"));types[id]=3;
        }
        std::vector<std::string> merges;
        for(auto * item=field(model,"merges")->child;item;item=item->next) {
            if(cJSON_IsString(item)) merges.emplace_back(str(item));
            else if(cJSON_IsArray(item) && cJSON_GetArraySize(item)==2)
                merges.emplace_back(std::string(str(cJSON_GetArrayItem(item,0)))+" "+str(cJSON_GetArrayItem(item,1)));
            else throw std::runtime_error("R2T2 invalid BPE merge");
        }
        s("tokenizer.ggml.model","gpt2");s("tokenizer.ggml.pre","qwen2");
        strings(g,"tokenizer.ggml.tokens",tokens);strings(g,"tokenizer.ggml.merges",merges);
        gguf_set_arr_data(g,"tokenizer.ggml.token_type",GGUF_TYPE_INT32,types.data(),types.size());
        for(uint32_t i=0;i<vocab;++i) {
            if(tokens[i]=="<|im_end|>")u("tokenizer.ggml.eos_token_id",i);
            if(tokens[i]=="<|endoftext|>")u("tokenizer.ggml.padding_token_id",i);
        }
        strings(g,"general.languages",{"zh","en","yue","ar","de","fr","es","pt","id","it","ko","ru","th","vi","ja","tr","hi","ms","nl","sv","da","fi","pl","cs","fil","fa","el","ro","hu","mk"});
        gguf_set_val_bool(g,"stt.capability.lang_detect",true);
        return TRANSCRIBE_OK;
    } catch(const std::exception & e) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,"R2T2 package: %s",e.what());return TRANSCRIBE_ERR_GGUF;
    }
}

std::vector<std::pair<ggml_tensor *,std::string>> rename_r2t2_tensors(ggml_context * ctx) {
    std::vector<std::pair<ggml_tensor *,std::string>> old;
    for(auto * t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)) {
        auto mapped=r2t2_tensor_name(t->name);
        if(mapped!=t->name) {old.emplace_back(t,t->name);ggml_set_name(t,mapped.c_str());}
    }
    return old;
}

transcribe_status plan_r2t2_dtypes(ggml_context * src_ctx, ggml_context ** out_ctx) {
    if(src_ctx==nullptr||out_ctx==nullptr) return TRANSCRIBE_ERR_INVALID_ARG;
    *out_ctx=nullptr;

    size_t n=0;
    for(auto * t=ggml_get_first_tensor(src_ctx);t;t=ggml_get_next_tensor(src_ctx,t)) ++n;
    if(n==0) return TRANSCRIBE_ERR_GGUF;

    // Metadata-only context, same shape as the one the conv-pointwise F16
    // promotion builds: no_alloc=true, and the storage buffer is allocated
    // later by alloc_ctx_tensors_with_reclaim() over *out_ctx.
    ggml_init_params params={n*ggml_tensor_overhead()+256,nullptr,true};
    ggml_context * ctx=ggml_init(params);
    if(ctx==nullptr) return TRANSCRIBE_ERR_BACKEND;

    size_t promoted=0;
    for(auto * t=ggml_get_first_tensor(src_ctx);t;t=ggml_get_next_tensor(src_ctx,t)) {
        const bool      is_bf16 = t->type==GGML_TYPE_BF16;
        ggml_tensor *   r       = ggml_new_tensor(ctx,is_bf16?GGML_TYPE_F32:t->type,ggml_n_dims(t),t->ne);
        if(r==nullptr) {ggml_free(ctx);return TRANSCRIBE_ERR_BACKEND;}
        // Names carry through unchanged; streaming and the weight catalog both
        // resolve tensors by name, and neither cares which context holds them.
        ggml_set_name(r,t->name);
        if(is_bf16) ++promoted;
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_INFO,
            "R2T2 package: planned %zu of %zu tensors as F32 (BF16 in the file)",promoted,n);
    *out_ctx=ctx;
    return TRANSCRIBE_OK;
}
}
