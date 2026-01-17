use crate::causal_attention::CausalAttention;
use crate::engine::TokenizerAdapter;
use crate::generation::{FinishReason, GenerationConfig, GenerationOutput};
use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::moe_decoder_layer::MoEDecoderLayer;
use crate::sampler::{sample_next_token, SamplingConfig};
use crate::tensor::Tensor3;
use crate::types::{Error, Result};
use crate::weight_loader::{load_embedding, load_linear, WeightLoader};
use gllm_kernels::{linear_forward, WeightMatrix};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

use crate::awq::AwqWeight;

#[derive(Clone)]
pub struct MoEGeneratorModel {
    pub(crate) embeddings: WeightMatrix,
    pub(crate) layers: Vec<MoEDecoderLayer>,
    pub(crate) final_norm: crate::rms_norm::RmsNorm,
    pub(crate) lm_head: crate::weight_loader::LinearWeights,
    pub(crate) pad_token_id: i64,
    pub(crate) max_position_embeddings: usize,
    pub(crate) vocab_size: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    hidden_size: usize,
}

impl MoEGeneratorModel {
    pub fn new(config: ModelConfig) -> Result<Self> {
        if config.num_hidden_layers == 0 {
            return Err(Error::InvalidConfig(
                "num_hidden_layers must be greater than 0 for generator model".into(),
            ));
        }
        if config.vocab_size == 0 {
            return Err(Error::InvalidConfig(
                "vocab_size must be greater than 0 for generator model".into(),
            ));
        }

        let embeddings = WeightMatrix::zeros(config.vocab_size, config.hidden_size);
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads.unwrap_or(num_attention_heads);
        let head_dim = config
            .head_dim
            .unwrap_or_else(|| config.hidden_size / num_attention_heads);
        let rope = CausalAttention::build_rope(&config, head_dim);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(MoEDecoderLayer::new(&config, rope.clone())?);
        }

        Ok(Self {
            embeddings,
            layers,
            final_norm: crate::rms_norm::RmsNorm::new(&config),
            lm_head: crate::weight_loader::LinearWeights::zeros(
                config.vocab_size,
                config.hidden_size,
            ),
            pad_token_id: config.pad_token_id.unwrap_or(0),
            max_position_embeddings: config.max_position_embeddings,
            vocab_size: config.vocab_size,
            num_key_value_heads,
            head_dim,
            hidden_size: config.hidden_size,
        })
    }

    pub fn forward_step(&self, input_ids: &[i64], cache: &mut KVCache) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            return Err(Error::InvalidConfig(
                "Generator input ids must be non-empty".into(),
            ));
        }
        let seq_len = input_ids.len();
        let position_offset = cache.seq_len();
        let mut hidden_states = self.embed_tokens(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states =
                layer.forward_with_cache(&hidden_states, position_offset, cache, layer_idx)?;
        }

        let normed =
            self.final_norm
                .forward_3d(&hidden_states.data, 1, seq_len);
        let normed = Tensor3::new(normed, 1, seq_len, self.hidden_size)?;
        let logits = self.apply_lm_head(&normed)?;

        let start = (seq_len - 1) * self.vocab_size;
        Ok(logits[start..start + self.vocab_size].to_vec())
    }

    pub(crate) fn generate(
        &self,
        prompt_ids: Vec<i64>,
        config: &GenerationConfig,
        tokenizer: &TokenizerAdapter,
    ) -> Result<GenerationOutput> {
        if prompt_ids.is_empty() {
            return Err(Error::InvalidConfig(
                "Prompt tokens are required for generation".into(),
            ));
        }

        if self.max_position_embeddings > 0 && prompt_ids.len() > self.max_position_embeddings {
            return Err(Error::InvalidConfig(format!(
                "Prompt length {} exceeds max position {}",
                prompt_ids.len(),
                self.max_position_embeddings
            )));
        }

        let max_len = if self.max_position_embeddings > 0 {
            self.max_position_embeddings
        } else {
            prompt_ids.len().saturating_add(config.max_new_tokens)
        };
        let mut cache = KVCache::preallocate(
            self.layers.len(),
            max_len,
            1,
            self.num_key_value_heads,
            self.head_dim,
        );

        let sampling = SamplingConfig {
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            seed: None,
        };

        let mut tokens = prompt_ids.clone();
        let mut finish_reason = FinishReason::MaxTokens;
        let mut logits = self.forward_step(&prompt_ids, &mut cache)?;

        for _ in 0..config.max_new_tokens {
            if self.max_position_embeddings > 0 && cache.seq_len() >= self.max_position_embeddings {
                finish_reason = FinishReason::MaxTokens;
                break;
            }

            let next = sample_next_token(&logits, 1, self.vocab_size, &sampling);
            let next_token = next.first().copied().unwrap_or(self.pad_token_id as u32) as i64;
            tokens.push(next_token);

            if config.stop_tokens.contains(&next_token) {
                finish_reason = FinishReason::StopToken;
                break;
            }

            logits = self.forward_step(&[next_token], &mut cache)?;
        }

        Ok(GenerationOutput {
            text: tokenizer.decode(&tokens),
            tokens,
            finish_reason,
        })
    }

    pub fn load_safetensors(&mut self, safetensors_path: &Path) -> Result<()> {
        let file = File::open(safetensors_path).map_err(|err| {
            Error::LoadError(format!(
                "Failed to open SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|err| {
            Error::LoadError(format!(
                "Failed to map SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;
        let loader = WeightLoader::from_bytes(&mmap)?;

        load_embeddings_from_safetensors(&loader, &mut self.embeddings)?;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{}", layer_idx);
            if loader.has_tensor(&format!("{}.self_attn.q_proj.weight", prefix)) {
                layer.attention.q_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.q_proj.weight", prefix),
                    Some(&format!("{}.self_attn.q_proj.bias", prefix)),
                )?;
                layer.attention.k_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.k_proj.weight", prefix),
                    Some(&format!("{}.self_attn.k_proj.bias", prefix)),
                )?;
                layer.attention.v_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.v_proj.weight", prefix),
                    Some(&format!("{}.self_attn.v_proj.bias", prefix)),
                )?;
                layer.attention.o_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.o_proj.weight", prefix),
                    Some(&format!("{}.self_attn.o_proj.bias", prefix)),
                )?;
            }

            if loader.has_tensor(&format!("{}.mlp.gate.weight", prefix)) {
                let gate = load_linear(&loader, &format!("{}.mlp.gate.weight", prefix), None)?;
                layer.moe.router.set_gate(gate)?;
            }

            for (expert_idx, expert) in layer.moe.experts.iter_mut().enumerate() {
                let expert_prefix = format!("{}.mlp.experts.{}", prefix, expert_idx);
                if loader.has_tensor(&format!("{}.gate_proj.weight", expert_prefix)) {
                    expert.gate_proj = load_linear(
                        &loader,
                        &format!("{}.gate_proj.weight", expert_prefix),
                        None,
                    )?;
                    expert.up_proj = load_linear(
                        &loader,
                        &format!("{}.up_proj.weight", expert_prefix),
                        None,
                    )?;
                    expert.down_proj = load_linear(
                        &loader,
                        &format!("{}.down_proj.weight", expert_prefix),
                        None,
                    )?;
                }
            }

            if let Some(shared) = &mut layer.moe.shared_expert {
                let shared_prefix = format!("{}.mlp.shared_expert", prefix);
                if loader.has_tensor(&format!("{}.gate_proj.weight", shared_prefix)) {
                    shared.gate_proj = load_linear(
                        &loader,
                        &format!("{}.gate_proj.weight", shared_prefix),
                        None,
                    )?;
                    shared.up_proj = load_linear(
                        &loader,
                        &format!("{}.up_proj.weight", shared_prefix),
                        None,
                    )?;
                    shared.down_proj = load_linear(
                        &loader,
                        &format!("{}.down_proj.weight", shared_prefix),
                        None,
                    )?;
                }
            }

            load_rms_norm(&loader, &format!("{}.input_layernorm.weight", prefix), &mut layer.attention_norm)?;
            load_rms_norm(&loader, &format!("{}.post_attention_layernorm.weight", prefix), &mut layer.ffn_norm)?;
        }

        load_final_norm(&loader, &mut self.final_norm)?;
        load_lm_head(&loader, &mut self.lm_head, &self.embeddings)?;
        self.vocab_size = self.lm_head.weight.rows;

        Ok(())
    }

    pub fn load_gguf(&mut self, _gguf_path: &Path) -> Result<()> {
        Err(Error::InvalidConfig(
            "GGUF support is not available for MoE generator models".into(),
        ))
    }

    pub fn load_awq(&mut self, safetensors_path: &Path) -> Result<()> {
        let file = File::open(safetensors_path).map_err(|err| {
            Error::LoadError(format!(
                "Failed to open SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|err| {
            Error::LoadError(format!(
                "Failed to map SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;
        let loader = WeightLoader::from_bytes(&mmap)?;
        if !loader.is_awq_model() {
            return Err(Error::LoadError(
                "Provided file does not appear to be an AWQ model".into(),
            ));
        }

        load_embeddings_from_safetensors(&loader, &mut self.embeddings)?;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{layer_idx}");
            layer.attention.q_proj = load_awq_or_linear(&loader, &format!("{prefix}.self_attn.q_proj"))?;
            layer.attention.k_proj = load_awq_or_linear(&loader, &format!("{prefix}.self_attn.k_proj"))?;
            layer.attention.v_proj = load_awq_or_linear(&loader, &format!("{prefix}.self_attn.v_proj"))?;
            layer.attention.o_proj = load_awq_or_linear(&loader, &format!("{prefix}.self_attn.o_proj"))?;

            let gate = load_awq_or_linear(&loader, &format!("{prefix}.mlp.gate"))?;
            layer.moe.router.set_gate(gate)?;

            for (expert_idx, expert) in layer.moe.experts.iter_mut().enumerate() {
                let expert_prefix = format!("{prefix}.mlp.experts.{expert_idx}");
                expert.gate_proj = load_awq_or_linear(&loader, &format!("{expert_prefix}.gate_proj"))?;
                expert.up_proj = load_awq_or_linear(&loader, &format!("{expert_prefix}.up_proj"))?;
                expert.down_proj = load_awq_or_linear(&loader, &format!("{expert_prefix}.down_proj"))?;
            }

            if let Some(shared) = &mut layer.moe.shared_expert {
                let shared_prefix = format!("{prefix}.mlp.shared_expert");
                shared.gate_proj = load_awq_or_linear(&loader, &format!("{shared_prefix}.gate_proj"))?;
                shared.up_proj = load_awq_or_linear(&loader, &format!("{shared_prefix}.up_proj"))?;
                shared.down_proj = load_awq_or_linear(&loader, &format!("{shared_prefix}.down_proj"))?;
            }

            load_rms_norm(&loader, &format!("{prefix}.input_layernorm.weight"), &mut layer.attention_norm)?;
            load_rms_norm(&loader, &format!("{prefix}.post_attention_layernorm.weight"), &mut layer.ffn_norm)?;
        }

        load_final_norm(&loader, &mut self.final_norm)?;
        load_lm_head_awq(&loader, &mut self.lm_head)?;
        self.vocab_size = self.lm_head.weight.rows;

        Ok(())
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn embed_tokens(&self, tokens: &[i64]) -> Result<Tensor3> {
        let seq_len = tokens.len();
        let mut data = vec![0.0f32; seq_len * self.hidden_size];
        for (idx, &token) in tokens.iter().enumerate() {
            let row = safe_token_index(token, self.embeddings.rows);
            let start = idx * self.hidden_size;
            data[start..start + self.hidden_size].copy_from_slice(self.embeddings.row(row));
        }
        Tensor3::new(data, 1, seq_len, self.hidden_size)
    }

    fn apply_lm_head(&self, hidden_states: &Tensor3) -> Result<Vec<f32>> {
        let (batch, seq_len, hidden) = hidden_states.shape();
        if hidden != self.lm_head.weight.cols {
            return Err(Error::InferenceError(
                "LM head input hidden size mismatch".into(),
            ));
        }
        let rows = batch * seq_len;
        let mut logits = vec![0.0f32; rows * self.lm_head.weight.rows];
        linear_forward(
            &hidden_states.data,
            self.lm_head.weight.as_slice(),
            self.lm_head.bias.as_ref().map(|b| b.as_slice()),
            &mut logits,
            rows,
            self.lm_head.weight.cols,
            self.lm_head.weight.rows,
        );
        Ok(logits)
    }
}

fn load_embeddings_from_safetensors(loader: &WeightLoader, embeddings: &mut WeightMatrix) -> Result<()> {
    let embed_names = [
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "transformer.embedding.word_embeddings.weight",
        "embeddings.word_embeddings.weight",
    ];
    for name in embed_names {
        if loader.has_tensor(name) {
            *embeddings = load_embedding(loader, name)?;
            break;
        }
    }
    Ok(())
}

fn load_rms_norm(
    loader: &WeightLoader,
    name: &str,
    norm: &mut crate::rms_norm::RmsNorm,
) -> Result<()> {
    if loader.has_tensor(name) {
        let tensor = loader.load_tensor(name)?;
        norm.gamma = tensor.to_weight_vector()?;
    }
    Ok(())
}

fn load_final_norm(loader: &WeightLoader, norm: &mut crate::rms_norm::RmsNorm) -> Result<()> {
    let final_norm_names = ["model.norm.weight", "transformer.ln_f.weight"];
    for name in final_norm_names {
        if loader.has_tensor(name) {
            let tensor = loader.load_tensor(name)?;
            norm.gamma = tensor.to_weight_vector()?;
            break;
        }
    }
    Ok(())
}

fn load_lm_head(
    loader: &WeightLoader,
    lm_head: &mut crate::weight_loader::LinearWeights,
    embeddings: &WeightMatrix,
) -> Result<()> {
    let lm_head_names = ["lm_head.weight", "output.weight"];
    for name in lm_head_names {
        if loader.has_tensor(name) {
            *lm_head = load_linear(loader, name, None)?;
            return Ok(());
        }
    }

    *lm_head = crate::weight_loader::LinearWeights {
        weight: WeightMatrix::new(embeddings.data.clone(), embeddings.rows, embeddings.cols),
        bias: None,
    };
    Ok(())
}

fn load_lm_head_awq(
    loader: &WeightLoader,
    lm_head: &mut crate::weight_loader::LinearWeights,
) -> Result<()> {
    let prefixes = ["lm_head", "output", "model.lm_head"];
    for prefix in prefixes {
        if loader.has_tensor(&format!("{prefix}.qweight")) {
            *lm_head = load_awq_linear(loader, prefix)?;
            return Ok(());
        }
        if loader.has_tensor(&format!("{prefix}.weight")) {
            *lm_head = load_linear(loader, &format!("{prefix}.weight"), None)?;
            return Ok(());
        }
    }
    Err(Error::LoadError(
        "AWQ model is missing LM head weights".into(),
    ))
}

fn safe_token_index(id: i64, vocab: usize) -> usize {
    let fallback = if vocab == 0 { 0 } else { vocab - 1 };
    if id < 0 {
        return 0;
    }
    let idx = id as usize;
    if idx < vocab {
        idx
    } else {
        fallback
    }
}

fn load_awq_linear(loader: &WeightLoader, prefix: &str) -> Result<crate::weight_loader::LinearWeights> {
    let weight = AwqWeight::from_safetensors(loader, prefix)?;
    let [out_features, in_features] = weight.shape;
    let data = weight.dequantize();
    if data.len() != out_features * in_features {
        return Err(Error::LoadError(
            "AWQ linear weight size does not match shape".into(),
        ));
    }
    let bias_name = format!("{prefix}.bias");
    let bias = if loader.has_tensor(&bias_name) {
        Some(loader.load_tensor(&bias_name)?.to_weight_vector()?)
    } else {
        None
    };
    Ok(crate::weight_loader::LinearWeights {
        weight: WeightMatrix::new(data, out_features, in_features),
        bias,
    })
}

fn load_awq_or_linear(
    loader: &WeightLoader,
    prefix: &str,
) -> Result<crate::weight_loader::LinearWeights> {
    if loader.has_tensor(&format!("{prefix}.qweight")) {
        load_awq_linear(loader, prefix)
    } else {
        load_linear(loader, &format!("{prefix}.weight"), None)
    }
}
