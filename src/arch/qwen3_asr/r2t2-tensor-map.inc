// Derived from the existing Qwen tensor contract.
static std::string r2t2_tensor_name(const std::string & name) {
    if (name == "thinker.audio_tower.conv2d1.weight") return "enc.conv.0.weight";
    if (name == "thinker.audio_tower.conv2d1.bias") return "enc.conv.0.bias";
    if (name == "thinker.audio_tower.conv2d2.weight") return "enc.conv.1.weight";
    if (name == "thinker.audio_tower.conv2d2.bias") return "enc.conv.1.bias";
    if (name == "thinker.audio_tower.conv2d3.weight") return "enc.conv.2.weight";
    if (name == "thinker.audio_tower.conv2d3.bias") return "enc.conv.2.bias";
    if (name == "thinker.audio_tower.conv_out.weight") return "enc.conv_out.weight";
    if (name == "thinker.audio_tower.ln_post.weight") return "enc.ln_post.weight";
    if (name == "thinker.audio_tower.ln_post.bias") return "enc.ln_post.bias";
    if (name == "thinker.audio_tower.proj1.weight") return "enc.proj1.weight";
    if (name == "thinker.audio_tower.proj1.bias") return "enc.proj1.bias";
    if (name == "thinker.audio_tower.proj2.weight") return "enc.proj2.weight";
    if (name == "thinker.audio_tower.proj2.bias") return "enc.proj2.bias";
    if (name == "thinker.model.embed_tokens.weight") return "dec.token_embd.weight";
    if (name == "thinker.model.norm.weight") return "dec.output_norm.weight";
    {
        const std::string prefix = "thinker.audio_tower.layers.";
        if (name.compare(0, prefix.size(), prefix) == 0) {
            auto dot = name.find('.', prefix.size());
            if (dot != std::string::npos) {
                auto suffix = name.substr(dot + 1);
                if (suffix == "self_attn_layer_norm.weight") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".norm_attn.weight";
                if (suffix == "self_attn_layer_norm.bias") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".norm_attn.bias";
                if (suffix == "self_attn.q_proj.weight") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.q.weight";
                if (suffix == "self_attn.q_proj.bias") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.q.bias";
                if (suffix == "self_attn.k_proj.weight") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.k.weight";
                if (suffix == "self_attn.k_proj.bias") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.k.bias";
                if (suffix == "self_attn.v_proj.weight") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.v.weight";
                if (suffix == "self_attn.v_proj.bias") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.v.bias";
                if (suffix == "self_attn.out_proj.weight") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.out.weight";
                if (suffix == "self_attn.out_proj.bias") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.out.bias";
                if (suffix == "final_layer_norm.weight") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".norm_ffn.weight";
                if (suffix == "final_layer_norm.bias") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".norm_ffn.bias";
                if (suffix == "fc1.weight") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".ffn.fc1.weight";
                if (suffix == "fc1.bias") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".ffn.fc1.bias";
                if (suffix == "fc2.weight") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".ffn.fc2.weight";
                if (suffix == "fc2.bias") return "enc.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".ffn.fc2.bias";
            }
        }
    }
    {
        const std::string prefix = "thinker.model.layers.";
        if (name.compare(0, prefix.size(), prefix) == 0) {
            auto dot = name.find('.', prefix.size());
            if (dot != std::string::npos) {
                auto suffix = name.substr(dot + 1);
                if (suffix == "input_layernorm.weight") return "dec.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".norm_attn.weight";
                if (suffix == "post_attention_layernorm.weight") return "dec.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".norm_ffn.weight";
                if (suffix == "self_attn.q_proj.weight") return "dec.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.q.weight";
                if (suffix == "self_attn.k_proj.weight") return "dec.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.k.weight";
                if (suffix == "self_attn.v_proj.weight") return "dec.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.v.weight";
                if (suffix == "self_attn.o_proj.weight") return "dec.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.o.weight";
                if (suffix == "self_attn.q_norm.weight") return "dec.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.q_norm.weight";
                if (suffix == "self_attn.k_norm.weight") return "dec.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".attn.k_norm.weight";
                if (suffix == "mlp.gate_proj.weight") return "dec.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".ffn.gate.weight";
                if (suffix == "mlp.up_proj.weight") return "dec.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".ffn.up.weight";
                if (suffix == "mlp.down_proj.weight") return "dec.blocks." + name.substr(prefix.size(), dot-prefix.size()) + ".ffn.down.weight";
            }
        }
    }
    return name;
}
