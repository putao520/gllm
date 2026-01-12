use crate::causal_attention::CausalAttention;
use crate::decoder_layer::DecoderLayer;
use crate::generation::{FinishReason, GenerationConfig, GenerationOutput};
use crate::kv_cache::KVCache;
use crate::model_config::ModelConfig;
use crate::sampler::{sample_next_token, SamplingConfig};
use crate::types::{Error, Result};
use crate::engine::TokenizerAdapter;
use crate::rms_norm::RmsNorm;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use std::path::Path;

#[derive(Clone)]
pub struct GeneratorModel<B: Backend> {
    pub(crate) embeddings: Embedding<B>,
    pub(crate) layers: Vec<DecoderLayer<B>>,
    pub(crate) final_norm: RmsNorm<B>,
    pub(crate) lm_head: Linear<B>,
    pub(crate) pad_token_id: i64,
    pub(crate) max_position_embeddings: usize,
    pub(crate) vocab_size: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) device: B::Device,
}

impl<B: Backend> GeneratorModel<B> {
    pub fn new(device: &B::Device, config: ModelConfig) -> Result<Self> {
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

        let embeddings = EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads.unwrap_or(num_attention_heads);
        let head_dim = config
            .head_dim
            .unwrap_or_else(|| config.hidden_size / num_attention_heads);
        let rope = CausalAttention::build_rope(device, &config, head_dim);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(device, &config, rope.clone())?);
        }

        let final_norm = RmsNorm::new(device, &config);
        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size).init(device);

        Ok(Self {
            embeddings,
            layers,
            final_norm,
            lm_head,
            pad_token_id: config.pad_token_id.unwrap_or(0),
            max_position_embeddings: config.max_position_embeddings,
            vocab_size: config.vocab_size,
            num_key_value_heads,
            head_dim,
            device: device.clone(),
        })
    }

    pub fn forward_step(
        &self,
        input_ids: Tensor<B, 2, Int>,
        cache: &mut KVCache<B>,
    ) -> Tensor<B, 2> {
        let [_batch_size, seq_len] = input_ids.dims();
        let position_offset = cache.seq_len();

        let mut hidden_states = self.embeddings.forward(input_ids);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states =
                layer.forward_with_cache(hidden_states, position_offset, cache, layer_idx);
        }

        let hidden_states = self.final_norm.forward(hidden_states);
        let logits = self.lm_head.forward(hidden_states);

        let [batch_size, _seq, _vocab] = logits.dims();
        let last_index = seq_len.saturating_sub(1);
        logits
            .slice([0..batch_size, last_index..(last_index + 1), 0..self.vocab_size])
            .reshape([batch_size, self.vocab_size])
    }

    pub fn generate(
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
            &self.device,
        );
        let mut tokens = prompt_ids.clone();
        let sampling = SamplingConfig {
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
        };

        let mut finish_reason = FinishReason::MaxTokens;
        let mut input_ids = prompt_ids;
        let mut logits = self.forward_step(self.tokens_to_tensor(&input_ids), &mut cache);

        for _ in 0..config.max_new_tokens {
            if self.max_position_embeddings > 0
                && cache.seq_len() >= self.max_position_embeddings
            {
                finish_reason = FinishReason::MaxTokens;
                break;
            }

            let next_tokens = sample_next_token(logits, &sampling, &self.device);
            let next_token = next_tokens.first().copied().unwrap_or(self.pad_token_id);
            tokens.push(next_token);

            if config.stop_tokens.contains(&next_token) {
                finish_reason = FinishReason::StopToken;
                break;
            }

            input_ids = vec![next_token];
            logits = self.forward_step(self.tokens_to_tensor(&input_ids), &mut cache);
        }

        let text = tokenizer.decode(&tokens);
        Ok(GenerationOutput {
            text,
            tokens,
            finish_reason,
        })
    }

    pub fn load_safetensors(&mut self, safetensors_path: &Path) -> Result<()> {
        use crate::weight_loader::{load_linear, load_embedding, WeightLoader};
        use burn::module::Param;

        let bytes = std::fs::read(safetensors_path).map_err(|err| {
            Error::LoadError(format!(
                "Failed to read SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;

        let loader = WeightLoader::from_bytes(&bytes)?;

        // Load embeddings - try different naming conventions
        let embed_names = [
            "model.embed_tokens.weight",
            "transformer.wte.weight",
            "transformer.embedding.word_embeddings.weight",
            "embeddings.word_embeddings.weight",
        ];
        for name in embed_names {
            if loader.has_tensor(name) {
                self.embeddings = load_embedding(&loader, name, &self.device)?;
                break;
            }
        }

        // Load decoder layers
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Try LLaMA-style naming first
            let prefix = format!("model.layers.{}", layer_idx);

            // Load attention weights
            if loader.has_tensor(&format!("{}.self_attn.q_proj.weight", prefix)) {
                layer.attention.q_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.q_proj.weight", prefix),
                    Some(&format!("{}.self_attn.q_proj.bias", prefix)),
                    &self.device,
                )?;
                layer.attention.k_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.k_proj.weight", prefix),
                    Some(&format!("{}.self_attn.k_proj.bias", prefix)),
                    &self.device,
                )?;
                layer.attention.v_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.v_proj.weight", prefix),
                    Some(&format!("{}.self_attn.v_proj.bias", prefix)),
                    &self.device,
                )?;
                layer.attention.o_proj = load_linear(
                    &loader,
                    &format!("{}.self_attn.o_proj.weight", prefix),
                    Some(&format!("{}.self_attn.o_proj.bias", prefix)),
                    &self.device,
                )?;
            }

            // Load FFN weights
            if loader.has_tensor(&format!("{}.mlp.gate_proj.weight", prefix)) {
                layer.gate_proj = load_linear(
                    &loader,
                    &format!("{}.mlp.gate_proj.weight", prefix),
                    None,
                    &self.device,
                )?;
                layer.up_proj = load_linear(
                    &loader,
                    &format!("{}.mlp.up_proj.weight", prefix),
                    None,
                    &self.device,
                )?;
                layer.down_proj = load_linear(
                    &loader,
                    &format!("{}.mlp.down_proj.weight", prefix),
                    None,
                    &self.device,
                )?;
            }

            // Load RMSNorm weights
            if loader.has_tensor(&format!("{}.input_layernorm.weight", prefix)) {
                let norm_tensor = loader.load_tensor(&format!("{}.input_layernorm.weight", prefix))?;
                let norm_weight = norm_tensor.to_tensor::<B, 1>(&self.device, [norm_tensor.shape[0]])?;
                layer.attention_norm.inner.gamma = Param::from_tensor(norm_weight);
            }
            if loader.has_tensor(&format!("{}.post_attention_layernorm.weight", prefix)) {
                let norm_tensor = loader.load_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?;
                let norm_weight = norm_tensor.to_tensor::<B, 1>(&self.device, [norm_tensor.shape[0]])?;
                layer.ffn_norm.inner.gamma = Param::from_tensor(norm_weight);
            }
        }

        // Load final layer norm
        let final_norm_names = [
            "model.norm.weight",
            "transformer.ln_f.weight",
            "transformer.encoder.final_layernorm.weight",
        ];
        for name in final_norm_names {
            if loader.has_tensor(name) {
                let norm_tensor = loader.load_tensor(name)?;
                let norm_weight = norm_tensor.to_tensor::<B, 1>(&self.device, [norm_tensor.shape[0]])?;
                self.final_norm.inner.gamma = Param::from_tensor(norm_weight);
                break;
            }
        }

        // Load LM head
        let lm_head_names = [
            "lm_head.weight",
            "transformer.output_layer.weight",
            "output.weight",
        ];
        for name in lm_head_names {
            if loader.has_tensor(name) {
                self.lm_head = load_linear(&loader, name, None, &self.device)?;
                break;
            }
        }

        log::info!("Successfully loaded weights from {}", safetensors_path.display());
        Ok(())
    }

    #[cfg(feature = "quantized")]
    pub fn load_gguf(&mut self, gguf_path: &Path) -> Result<()> {
        use crate::gguf::GgufLoader;
        use crate::quantized::{GgmlDType, QTensor};
        use burn::module::Param;

        let loader = GgufLoader::load(gguf_path)?;

        let supported = |dtype: GgmlDType| {
            matches!(
                dtype,
                GgmlDType::F32
                    | GgmlDType::F16
                    | GgmlDType::Q4_0
                    | GgmlDType::Q4_K_S
                    | GgmlDType::Q4_K_M
                    | GgmlDType::Q8_0
            )
        };

        let load_qtensor = |name: &str| -> Result<QTensor> {
            let info = loader.get_tensor(name).ok_or_else(|| {
                Error::LoadError(format!("GGUF tensor '{name}' not found"))
            })?;
            if !supported(info.dtype) {
                return Err(Error::LoadError(format!(
                    "GGUF tensor '{name}' uses unsupported dtype {:?}",
                    info.dtype
                )));
            }
            let data = loader.get_tensor_data(info);
            if data.is_empty() {
                return Err(Error::LoadError(format!(
                    "GGUF tensor '{name}' has no data payload"
                )));
            }
            let shape = info.dims.iter().map(|d| *d as usize).collect();
            Ok(QTensor {
                data: data.to_vec(),
                dtype: info.dtype,
                shape,
            })
        };

        let qtensor_to_linear = |qtensor: QTensor| -> Result<Linear<B>> {
            if qtensor.shape.len() != 2 {
                return Err(Error::LoadError(
                    "GGUF linear weight must be 2D".into(),
                ));
            }
            let out_features = qtensor.shape[0];
            let in_features = qtensor.shape[1];
            let data = qtensor.dequantize();
            if data.len() != out_features * in_features {
                return Err(Error::LoadError(
                    "GGUF linear weight size does not match shape".into(),
                ));
            }
            let weight = Tensor::from_data(
                TensorData::new(data, [out_features, in_features]),
                &self.device,
            )
            .transpose();
            Ok(Linear {
                weight: Param::from_tensor(weight),
                bias: None,
            })
        };

        let qtensor_to_embedding = |qtensor: QTensor| -> Result<Embedding<B>> {
            if qtensor.shape.len() != 2 {
                return Err(Error::LoadError(
                    "GGUF embedding weight must be 2D".into(),
                ));
            }
            let rows = qtensor.shape[0];
            let cols = qtensor.shape[1];
            let data = qtensor.dequantize();
            if data.len() != rows * cols {
                return Err(Error::LoadError(
                    "GGUF embedding weight size does not match shape".into(),
                ));
            }
            let weight = Tensor::from_data(TensorData::new(data, [rows, cols]), &self.device);
            Ok(Embedding {
                weight: Param::from_tensor(weight),
            })
        };

        let qtensor_to_vector = |qtensor: QTensor| -> Result<Tensor<B, 1>> {
            if qtensor.shape.len() != 1 {
                return Err(Error::LoadError(
                    "GGUF norm weight must be 1D".into(),
                ));
            }
            let size = qtensor.shape[0];
            let data = qtensor.dequantize();
            if data.len() != size {
                return Err(Error::LoadError(
                    "GGUF norm weight size does not match shape".into(),
                ));
            }
            Ok(Tensor::from_data(TensorData::new(data, [size]), &self.device))
        };

        // Embedding weights.
        self.embeddings = qtensor_to_embedding(load_qtensor("token_embd.weight")?)?;

        // Decoder layers.
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("blk.{layer_idx}");
            layer.attention.q_proj =
                qtensor_to_linear(load_qtensor(&format!("{prefix}.attn_q.weight"))?)?;
            layer.attention.k_proj =
                qtensor_to_linear(load_qtensor(&format!("{prefix}.attn_k.weight"))?)?;
            layer.attention.v_proj =
                qtensor_to_linear(load_qtensor(&format!("{prefix}.attn_v.weight"))?)?;
            layer.attention.o_proj =
                qtensor_to_linear(load_qtensor(&format!("{prefix}.attn_output.weight"))?)?;
            layer.gate_proj =
                qtensor_to_linear(load_qtensor(&format!("{prefix}.ffn_gate.weight"))?)?;
            layer.up_proj = qtensor_to_linear(load_qtensor(&format!("{prefix}.ffn_up.weight"))?)?;
            layer.down_proj =
                qtensor_to_linear(load_qtensor(&format!("{prefix}.ffn_down.weight"))?)?;

            let attn_norm = qtensor_to_vector(load_qtensor(&format!("{prefix}.attn_norm.weight"))?)?;
            layer.attention_norm.inner.gamma = Param::from_tensor(attn_norm);
            let ffn_norm = qtensor_to_vector(load_qtensor(&format!("{prefix}.ffn_norm.weight"))?)?;
            layer.ffn_norm.inner.gamma = Param::from_tensor(ffn_norm);
        }

        // Final layer norm + LM head.
        let final_norm = qtensor_to_vector(load_qtensor("output_norm.weight")?)?;
        self.final_norm.inner.gamma = Param::from_tensor(final_norm);
        self.lm_head = qtensor_to_linear(load_qtensor("output.weight")?)?;

        log::info!("Successfully loaded GGUF weights from {}", gguf_path.display());
        Ok(())
    }

    #[cfg(not(feature = "quantized"))]
    pub fn load_gguf(&mut self, _gguf_path: &Path) -> Result<()> {
        Err(Error::InvalidConfig(
            "GGUF support requires the `quantized` feature".into(),
        ))
    }

    #[cfg(feature = "quantized")]
    pub fn load_awq(&mut self, safetensors_path: &Path) -> Result<()> {
        use crate::awq::AwqWeight;
        use crate::weight_loader::{load_embedding, load_linear, WeightLoader};
        use burn::module::Param;

        let bytes = std::fs::read(safetensors_path).map_err(|err| {
            Error::LoadError(format!(
                "Failed to read SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;

        let loader = WeightLoader::from_bytes(&bytes)?;
        if !loader.is_awq_model() {
            return Err(Error::LoadError(
                "Provided file does not appear to be an AWQ model".into(),
            ));
        }

        let awq_to_linear = |weight: AwqWeight| -> Result<Linear<B>> {
            let [out_features, in_features] = weight.shape;
            let data = weight.dequantize();
            if data.len() != out_features * in_features {
                return Err(Error::LoadError(
                    "AWQ linear weight size does not match shape".into(),
                ));
            }
            let weight = Tensor::from_data(
                TensorData::new(data, [out_features, in_features]),
                &self.device,
            )
            .transpose();
            Ok(Linear {
                weight: Param::from_tensor(weight),
                bias: None,
            })
        };

        // Embeddings (unquantized in most AWQ exports).
        let embed_names = [
            "model.embed_tokens.weight",
            "transformer.wte.weight",
            "transformer.embedding.word_embeddings.weight",
            "embeddings.word_embeddings.weight",
        ];
        let mut embedding_loaded = false;
        for name in embed_names {
            if loader.has_tensor(name) {
                self.embeddings = load_embedding(&loader, name, &self.device)?;
                embedding_loaded = true;
                break;
            }
        }
        if !embedding_loaded {
            return Err(Error::LoadError(
                "AWQ model is missing embedding weights".into(),
            ));
        }

        // Decoder layers (AWQ quantized linear weights).
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("model.layers.{layer_idx}");
            layer.attention.q_proj = awq_to_linear(AwqWeight::from_safetensors(
                &loader,
                &format!("{prefix}.self_attn.q_proj"),
            )?)?;
            layer.attention.k_proj = awq_to_linear(AwqWeight::from_safetensors(
                &loader,
                &format!("{prefix}.self_attn.k_proj"),
            )?)?;
            layer.attention.v_proj = awq_to_linear(AwqWeight::from_safetensors(
                &loader,
                &format!("{prefix}.self_attn.v_proj"),
            )?)?;
            layer.attention.o_proj = awq_to_linear(AwqWeight::from_safetensors(
                &loader,
                &format!("{prefix}.self_attn.o_proj"),
            )?)?;
            layer.gate_proj = awq_to_linear(AwqWeight::from_safetensors(
                &loader,
                &format!("{prefix}.mlp.gate_proj"),
            )?)?;
            layer.up_proj = awq_to_linear(AwqWeight::from_safetensors(
                &loader,
                &format!("{prefix}.mlp.up_proj"),
            )?)?;
            layer.down_proj = awq_to_linear(AwqWeight::from_safetensors(
                &loader,
                &format!("{prefix}.mlp.down_proj"),
            )?)?;

            // Norms remain in fp16/fp32 weights.
            let attn_norm_name = format!("{prefix}.input_layernorm.weight");
            let ffn_norm_name = format!("{prefix}.post_attention_layernorm.weight");
            let attn_norm = loader.load_tensor(&attn_norm_name)?;
            let attn_weight =
                attn_norm.to_tensor::<B, 1>(&self.device, [attn_norm.shape[0]])?;
            layer.attention_norm.inner.gamma = Param::from_tensor(attn_weight);
            let ffn_norm = loader.load_tensor(&ffn_norm_name)?;
            let ffn_weight = ffn_norm.to_tensor::<B, 1>(&self.device, [ffn_norm.shape[0]])?;
            layer.ffn_norm.inner.gamma = Param::from_tensor(ffn_weight);
        }

        // Final layer norm.
        let final_norm_names = [
            "model.norm.weight",
            "transformer.ln_f.weight",
            "transformer.encoder.final_layernorm.weight",
        ];
        let mut final_loaded = false;
        for name in final_norm_names {
            if loader.has_tensor(name) {
                let norm_tensor = loader.load_tensor(name)?;
                let norm_weight =
                    norm_tensor.to_tensor::<B, 1>(&self.device, [norm_tensor.shape[0]])?;
                self.final_norm.inner.gamma = Param::from_tensor(norm_weight);
                final_loaded = true;
                break;
            }
        }
        if !final_loaded {
            return Err(Error::LoadError(
                "AWQ model is missing final norm weights".into(),
            ));
        }

        // LM head (quantized if available, otherwise fall back to fp weights).
        let lm_head_prefixes = ["lm_head", "output", "model.lm_head"];
        let mut lm_loaded = false;
        for prefix in lm_head_prefixes {
            let qweight_name = format!("{prefix}.qweight");
            let weight_name = format!("{prefix}.weight");
            if loader.has_tensor(&qweight_name) {
                self.lm_head =
                    awq_to_linear(AwqWeight::from_safetensors(&loader, prefix)?)?;
                lm_loaded = true;
                break;
            }
            if loader.has_tensor(&weight_name) {
                self.lm_head = load_linear(&loader, &weight_name, None, &self.device)?;
                lm_loaded = true;
                break;
            }
        }
        if !lm_loaded {
            return Err(Error::LoadError(
                "AWQ model is missing LM head weights".into(),
            ));
        }

        log::info!(
            "Successfully loaded AWQ weights from {}",
            safetensors_path.display()
        );
        Ok(())
    }

    #[cfg(not(feature = "quantized"))]
    pub fn load_awq(&mut self, _safetensors_path: &Path) -> Result<()> {
        Err(Error::InvalidConfig(
            "AWQ support requires the `quantized` feature".into(),
        ))
    }

    pub fn load_auto(&mut self, path: &Path) -> Result<()> {
        let ext = path.extension().and_then(|value| value.to_str()).unwrap_or("");
        if ext.eq_ignore_ascii_case("gguf") {
            return self.load_gguf(path);
        }
        if ext.eq_ignore_ascii_case("safetensors") {
            if Self::detect_awq_format(path)? {
                return self.load_awq(path);
            }
            return self.load_safetensors(path);
        }
        self.load_safetensors(path)
    }

    fn detect_awq_format(path: &Path) -> Result<bool> {
        use crate::weight_loader::WeightLoader;

        let bytes = std::fs::read(path).map_err(|err| {
            Error::LoadError(format!(
                "Failed to read SafeTensors file {}: {err}",
                path.display()
            ))
        })?;
        let loader = WeightLoader::from_bytes(&bytes)?;
        Ok(loader.is_awq_model())
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    fn tokens_to_tensor(&self, tokens: &[i64]) -> Tensor<B, 2, Int> {
        let mut data = tokens.to_vec();
        if data.is_empty() {
            data.push(self.pad_token_id);
        }
        let seq_len = data.len();
        let data = TensorData::new(data, [1, seq_len]);
        Tensor::<B, 2, Int>::from_data(data, &self.device)
    }
}
