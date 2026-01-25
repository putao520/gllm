use crate::causal_attention::CausalAttention;
use crate::engine::TokenizerAdapter;
use crate::generation::{FinishReason, GenerationConfig, GenerationOutput};
use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::moe_decoder_layer::MoEDecoderLayer;
use crate::moe_layer::{MoEScratchGpu, PackedExpertWeights};
use crate::sampler::{sample_next_token, SamplingConfig};
use crate::tensor::Tensor3;
use crate::types::{Error, Result};
use crate::weight_loader::{load_embedding, load_linear, WeightLoader};
use gllm_kernels::{linear_forward, BackendType, GpuTensor, KernelDispatcher, TensorDtype, WeightMatrix};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::OnceLock;

/// Global KernelDispatcher singleton for GPU-accelerated operations.
static KERNEL_DISPATCHER: OnceLock<KernelDispatcher> = OnceLock::new();

#[inline]
fn kernel_dispatcher() -> &'static KernelDispatcher {
    KERNEL_DISPATCHER.get_or_init(KernelDispatcher::new)
}

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

pub struct MoEInferenceWorkspace {
    pub hidden_states: Vec<f32>,
    pub gpu_hidden_states: Option<GpuTensor>,
    pub moe_scratch: Option<MoEScratchGpu>,
    pub packed_weights: Vec<PackedExpertWeights>,
}

impl MoEInferenceWorkspace {
    pub fn new(model: &MoEGeneratorModel, backend: BackendType) -> Result<Self> {
        let mut gpu_hidden_states = None;
        if backend != BackendType::Cpu {
            match GpuTensor::new_temp(vec![1, model.hidden_size], TensorDtype::F32, backend) {
                Ok(tensor) => gpu_hidden_states = Some(tensor),
                Err(err) => log::warn!("Failed to allocate GPU hidden states: {err}"),
            }
        }

        let mut moe_scratch = None;
        let mut packed_weights = Vec::new();
        if gpu_hidden_states.is_some() {
            match Self::init_gpu_resources(model, backend) {
                Ok((scratch, packed)) => {
                    moe_scratch = Some(scratch);
                    packed_weights = packed;
                }
                Err(err) => {
                    log::warn!("Failed to initialize MoE GPU resources, falling back to CPU: {err}");
                    gpu_hidden_states = None;
                }
            }
        }

        Ok(Self {
            hidden_states: vec![0.0f32; model.hidden_size],
            gpu_hidden_states,
            moe_scratch,
            packed_weights,
        })
    }

    pub fn uses_gpu(&self) -> bool {
        self.gpu_hidden_states.is_some()
    }

    fn init_gpu_resources(
        model: &MoEGeneratorModel,
        backend: BackendType,
    ) -> Result<(MoEScratchGpu, Vec<PackedExpertWeights>)> {
        let first_layer = model.layers.first().ok_or_else(|| {
            Error::InvalidConfig("MoE model has no layers".into())
        })?;
        let top_k = first_layer.moe.router.num_experts_per_tok();
        let intermediate_size = first_layer.moe.intermediate_size();
        let scratch = MoEScratchGpu::new_with_routing(
            model.hidden_size,
            intermediate_size,
            top_k,
            backend,
        )?;
        let mut packed_weights = Vec::with_capacity(model.layers.len());
        let backend_dispatch = kernel_dispatcher().backend_dispatched();
        for layer in &model.layers {
            match layer.moe.pack_weights(backend, backend_dispatch) {
                Ok(packed) => packed_weights.push(packed),
                Err(err) => {
                    scratch.release();
                    release_packed_weights(packed_weights);
                    return Err(err);
                }
            }
        }
        Ok((scratch, packed_weights))
    }
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

    pub fn forward_step_with_workspace(
        &self,
        input_ids: &[i64],
        cache: &mut KVCache,
        workspace: &mut MoEInferenceWorkspace,
        logits_out: &mut [f32],
    ) -> Result<()> {
        if input_ids.is_empty() {
            return Err(Error::InvalidConfig(
                "Generator input ids must be non-empty".into(),
            ));
        }
        if workspace.uses_gpu() {
            return self.forward_step_gpu_sequence(input_ids, cache, workspace, logits_out);
        }
        self.forward_step_cpu_inplace(input_ids, cache, logits_out)
    }

    pub fn forward_step(&self, input_ids: &[i64], cache: &mut KVCache) -> Result<Vec<f32>> {
        let mut logits = vec![0.0f32; self.vocab_size];
        let mut workspace = MoEInferenceWorkspace::new(self, kernel_dispatcher().backend())?;
        self.forward_step_with_workspace(input_ids, cache, &mut workspace, &mut logits)?;
        Ok(logits)
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
        let backend = kernel_dispatcher().backend();
        let mut workspace = MoEInferenceWorkspace::new(self, backend)?;
        let cache_backend = if workspace.uses_gpu() {
            backend
        } else {
            BackendType::Cpu
        };
        let mut cache = KVCache::preallocate(
            self.layers.len(),
            max_len,
            1,
            self.num_key_value_heads,
            self.head_dim,
            cache_backend,
        );

        let sampling = SamplingConfig {
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            seed: None,
        };

        let mut tokens = prompt_ids.clone();
        let mut finish_reason = FinishReason::MaxTokens;
        let mut logits = vec![0.0f32; self.vocab_size];
        self.forward_step_with_workspace(&prompt_ids, &mut cache, &mut workspace, &mut logits)?;

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

            self.forward_step_with_workspace(&[next_token], &mut cache, &mut workspace, &mut logits)?;
        }

        Ok(GenerationOutput {
            text: tokenizer.decode(&tokens),
            tokens,
            finish_reason,
        })
    }

    fn forward_step_cpu_inplace(
        &self,
        input_ids: &[i64],
        cache: &mut KVCache,
        logits_out: &mut [f32],
    ) -> Result<()> {
        let seq_len = input_ids.len();
        let position_offset = cache.seq_len();
        let mut hidden_states = self.embed_tokens(input_ids)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states =
                layer.forward_with_cache(&hidden_states, position_offset, cache, layer_idx)?;
        }

        let normed = self.final_norm.forward_3d(&hidden_states.data, 1, seq_len);
        let normed = Tensor3::new(normed, 1, seq_len, self.hidden_size)?;
        let logits = self.apply_lm_head(&normed)?;

        let start = (seq_len - 1) * self.vocab_size;
        logits_out.copy_from_slice(&logits[start..start + self.vocab_size]);
        Ok(())
    }

    fn forward_step_gpu_sequence(
        &self,
        input_ids: &[i64],
        cache: &mut KVCache,
        workspace: &mut MoEInferenceWorkspace,
        logits_out: &mut [f32],
    ) -> Result<()> {
        for &token in input_ids {
            self.forward_step_gpu(token, cache, workspace, logits_out)?;
        }
        Ok(())
    }

    fn forward_step_gpu(
        &self,
        token: i64,
        cache: &mut KVCache,
        workspace: &mut MoEInferenceWorkspace,
        logits_out: &mut [f32],
    ) -> Result<()> {
        let gpu_hidden_states = workspace.gpu_hidden_states.as_mut().ok_or_else(|| {
            Error::InferenceError("GPU hidden states are not initialized".into())
        })?;
        let moe_scratch = workspace.moe_scratch.as_mut().ok_or_else(|| {
            Error::InferenceError("MoE GPU scratch is not initialized".into())
        })?;
        if workspace.packed_weights.len() != self.layers.len() {
            return Err(Error::InferenceError(
                "Packed MoE weights are not initialized".into(),
            ));
        }

        self.embed_tokens_inplace(token, &mut workspace.hidden_states)?;
        kernel_dispatcher()
            .upload_to_tensor(&workspace.hidden_states, gpu_hidden_states)
            .map_err(Error::InferenceError)?;

        let position_offset = cache.seq_len();
        let backend = kernel_dispatcher().backend_dispatched();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            layer.forward_inplace_gpu_with_cache(
                gpu_hidden_states,
                position_offset,
                cache,
                layer_idx,
                &workspace.packed_weights[layer_idx],
                moe_scratch,
                backend,
            )?;
        }

        self.final_norm.forward_gpu_inplace(gpu_hidden_states)?;
        kernel_dispatcher()
            .linear_forward_host_io_readback(gpu_hidden_states, &mut workspace.hidden_states)
            .map_err(Error::InferenceError)?;
        self.apply_lm_head_inplace(&workspace.hidden_states, logits_out)?;

        Ok(())
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

    fn embed_tokens_inplace(&self, token: i64, output: &mut [f32]) -> Result<()> {
        let row = safe_token_index(token, self.embeddings.rows);
        output[..self.hidden_size].copy_from_slice(self.embeddings.row(row));
        Ok(())
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

    fn apply_lm_head_inplace(&self, hidden_state: &[f32], output: &mut [f32]) -> Result<()> {
        linear_forward(
            hidden_state,
            self.lm_head.weight.as_slice(),
            self.lm_head.bias.as_ref().map(|b| b.as_slice()),
            output,
            1,
            self.lm_head.weight.cols,
            self.lm_head.weight.rows,
        );
        Ok(())
    }
}

fn release_packed_weights(packed: Vec<PackedExpertWeights>) {
    for weights in packed {
        weights.all_gate.release();
        weights.all_up.release();
        weights.all_down.release();
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
        norm.set_gamma(tensor.to_weight_vector()?);
    }
    Ok(())
}

fn load_final_norm(loader: &WeightLoader, norm: &mut crate::rms_norm::RmsNorm) -> Result<()> {
    let final_norm_names = ["model.norm.weight", "transformer.ln_f.weight"];
    for name in final_norm_names {
        if loader.has_tensor(name) {
            let tensor = loader.load_tensor(name)?;
            norm.set_gamma(tensor.to_weight_vector()?);
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
