use crate::bert_variants::BertVariant;
use crate::model_config::ModelConfig;
use crate::performance_optimizer::PerformanceOptimizer;
use crate::pooling::{DynamicPooler, PoolingConfig, PoolingStrategy};
use crate::rope::{RopeConfig, RotaryPositionEmbedding};
use crate::types::{Error, Result};
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear,
    LinearConfig,
};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

#[derive(Clone)]
pub enum HiddenAct {
    Gelu,
    Relu,
}

/// Dynamic BERT layer that can be configured based on model config.
#[derive(Clone)]
pub struct DynamicBertLayer<B: Backend> {
    pub(crate) attention: MultiHeadAttention<B>,
    pub(crate) ffn_1: Linear<B>,
    pub(crate) ffn_2: Linear<B>,
    pub(crate) attention_layernorm: LayerNorm<B>,
    pub(crate) output_layernorm: LayerNorm<B>,
    pub(crate) hidden_act: HiddenAct,
}

impl<B: Backend> DynamicBertLayer<B> {
    pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size.unwrap_or(hidden_size * 4);
        let num_heads = config.num_attention_heads;
        let dropout = config.attention_probs_dropout_prob.unwrap_or(0.1) as f64;

        let attention = MultiHeadAttentionConfig::new(hidden_size, num_heads)
            .with_dropout(dropout)
            .init(device);

        let ffn_1 = LinearConfig::new(hidden_size, intermediate_size).init(device);
        let ffn_2 = LinearConfig::new(intermediate_size, hidden_size).init(device);

        let eps = config.layer_norm_eps.unwrap_or(1e-12) as f64;
        let attention_layernorm = LayerNormConfig::new(hidden_size)
            .with_epsilon(eps)
            .init(device);
        let output_layernorm = LayerNormConfig::new(hidden_size)
            .with_epsilon(eps)
            .init(device);

        let hidden_act = match config.hidden_act.as_deref() {
            Some("gelu") | Some("gelu_new") => HiddenAct::Gelu,
            Some("relu") | Some("relu_new") => HiddenAct::Relu,
            _ => HiddenAct::Gelu,
        };

        Self {
            attention,
            ffn_1,
            ffn_2,
            attention_layernorm,
            output_layernorm,
            hidden_act,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let attention_out = self.attention.forward(MhaInput::self_attn(input.clone()));
        let attention_output = self
            .attention_layernorm
            .forward(input + attention_out.context);

        let activated = match self.hidden_act {
            HiddenAct::Gelu => Gelu::new().forward(self.ffn_1.forward(attention_output.clone())),
            HiddenAct::Relu => relu(self.ffn_1.forward(attention_output.clone())),
        };
        let ffn_output = self.ffn_2.forward(activated);

        self.output_layernorm.forward(attention_output + ffn_output)
    }
}

/// Dynamic BERT encoder with configurable layers.
#[derive(Clone)]
pub struct DynamicBertEncoder<B: Backend> {
    pub(crate) layers: Vec<DynamicBertLayer<B>>,
}

impl<B: Backend> DynamicBertEncoder<B> {
    pub fn new(device: &B::Device, config: &ModelConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(DynamicBertLayer::new(device, config));
        }
        Self { layers }
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        self.layers
            .iter()
            .fold(hidden_states, |states, layer| layer.forward(states))
    }
}

/// Position embedding type.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PositionEmbeddingType {
    /// Standard absolute position embeddings (BERT-style).
    Absolute,
    /// Rotary Position Embedding (RoPE).
    Rope,
}

/// Dynamic BERT model that can be configured based on HuggingFace config.
#[derive(Clone)]
pub struct DynamicBertModel<B: Backend> {
    pub(crate) embeddings: Embedding<B>,
    /// Absolute position embeddings (used when position_type is Absolute).
    pub(crate) position_embeddings: Option<Embedding<B>>,
    /// RoPE embeddings (used when position_type is Rope).
    pub(crate) rope: Option<RotaryPositionEmbedding<B>>,
    /// Position embedding type.
    pub(crate) position_type: PositionEmbeddingType,
    pub(crate) token_type_embeddings: Option<Embedding<B>>,
    pub(crate) encoder: DynamicBertEncoder<B>,
    pub(crate) embedding_layernorm: LayerNorm<B>,
    pub(crate) embedding_dropout: Option<Dropout>,
    pub(crate) pooler: DynamicPooler<B>,
    pub(crate) optimizer: PerformanceOptimizer,
    pub(crate) config: ModelConfig,
    pub(crate) device: B::Device,
    pub(crate) max_position_embeddings: usize,
}

impl<B: Backend> DynamicBertModel<B> {
    pub fn new(device: &B::Device, config: ModelConfig) -> Result<Self> {
        let variant = BertVariant::detect(&config);
        let optimizer = PerformanceOptimizer::from_config(&config);

        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let max_pos = config.max_position_embeddings;
        let type_vocab_size = variant.type_vocab_size(&config);

        // Detect position embedding type from config
        let position_type = match config.position_embedding_type.as_deref() {
            Some("rope") | Some("rotary") => PositionEmbeddingType::Rope,
            _ => PositionEmbeddingType::Absolute,
        };

        let embeddings = EmbeddingConfig::new(vocab_size, hidden_size).init(device);

        // Create position embeddings based on type
        let (position_embeddings, rope) = match position_type {
            PositionEmbeddingType::Absolute => {
                let pos_emb = EmbeddingConfig::new(max_pos, hidden_size).init(device);
                (Some(pos_emb), None)
            }
            PositionEmbeddingType::Rope => {
                // Extract NTK scaling factor if present
                let ntk_factor = config.rope_scaling.as_ref().and_then(|scaling| {
                    scaling.get("factor").and_then(|v| v.as_f64())
                });

                let rope_config = RopeConfig {
                    theta: config.rope_theta.unwrap_or(10000.0),
                    dim: hidden_size,
                    max_seq_len: max_pos,
                    ntk_factor,
                };
                let rope_emb = RotaryPositionEmbedding::new(device, rope_config);
                (None, Some(rope_emb))
            }
        };

        let token_type_embeddings = (type_vocab_size > 0)
            .then(|| EmbeddingConfig::new(type_vocab_size, hidden_size).init(device));

        let eps = config.layer_norm_eps.unwrap_or(1e-12) as f64;
        let embedding_layernorm = LayerNormConfig::new(hidden_size)
            .with_epsilon(eps)
            .init(device);

        let embedding_dropout = config
            .hidden_dropout_prob
            .map(|p| DropoutConfig::new(p as f64).init());

        let encoder = DynamicBertEncoder::new(device, &config);

        let pooler = DynamicPooler::new(
            variant.pooling_strategy(&config),
            PoolingConfig { normalize: false },
        );

        Ok(Self {
            embeddings,
            position_embeddings,
            rope,
            position_type,
            token_type_embeddings,
            encoder,
            embedding_layernorm,
            embedding_dropout,
            pooler,
            optimizer,
            config,
            device: device.clone(),
            max_position_embeddings: max_pos,
        })
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>> {
        let [batch_size, seq_len] = input_ids.dims();

        if seq_len > self.max_position_embeddings {
            return Err(Error::InvalidConfig(format!(
                "Sequence length {} exceeds configured maximum {}",
                seq_len, self.max_position_embeddings
            )));
        }

        self.optimizer.validate_sequence(seq_len)?;
        // Trigger optimizer logic for telemetry; currently not enforced.
        let _ = self.optimizer.optimize_batch_size(seq_len);

        // Get token embeddings
        let mut embeddings = self.embeddings.forward(input_ids);

        // Apply position embeddings based on type
        match self.position_type {
            PositionEmbeddingType::Absolute => {
                if let Some(pos_emb) = &self.position_embeddings {
                    let position_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &self.device)
                        .reshape([1, seq_len])
                        .repeat(&[batch_size, 1]);
                    embeddings = embeddings + pos_emb.forward(position_ids);
                }
            }
            PositionEmbeddingType::Rope => {
                // RoPE is applied in the attention layer, but we can optionally
                // apply it to embeddings for simplified encoder-only models
                if let Some(rope) = &self.rope {
                    embeddings = rope.apply_to_hidden_states(embeddings, 0);
                }
            }
        }

        // Add token type embeddings if present
        if let Some(token_type_embeddings) = &self.token_type_embeddings {
            let token_type_ids = Tensor::<B, 2, Int>::zeros([batch_size, seq_len], &self.device);
            embeddings = embeddings + token_type_embeddings.forward(token_type_ids);
        }

        embeddings = self.embedding_layernorm.forward(embeddings);
        if let Some(dropout) = &self.embedding_dropout {
            embeddings = dropout.forward(embeddings);
        }

        Ok(self.encoder.forward(embeddings))
    }

    pub fn pool_hidden_states(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 2> {
        self.pooler.pool(hidden_states, None)
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    pub fn load_safetensors(&mut self, safetensors_path: &Path) -> Result<()> {
        use crate::weight_loader::{load_embedding, load_layer_norm, load_linear, load_mha, WeightLoader};

        let file = File::open(safetensors_path).map_err(|err| {
            Error::LoadError(format!(
                "Failed to open SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;
        // Safety: the file is not mutated while the mmap is alive.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|err| {
            Error::LoadError(format!(
                "Failed to memory-map SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;

        let loader = WeightLoader::from_bytes(&mmap)?;
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let dropout = self.config.attention_probs_dropout_prob.unwrap_or(0.1) as f64;
        let eps = self.config.layer_norm_eps.unwrap_or(1e-12) as f64;

        // Detect model prefix (bert., roberta., distilbert., xlm-roberta., etc.)
        let prefixes = ["bert.", "roberta.", "distilbert.", "xlm-roberta.", ""];
        let model_prefix = prefixes
            .iter()
            .find(|p| {
                loader.has_tensor(&format!("{}embeddings.word_embeddings.weight", p))
            })
            .copied()
            .unwrap_or("");

        // Load word embeddings
        let word_embed_name = format!("{}embeddings.word_embeddings.weight", model_prefix);
        if loader.has_tensor(&word_embed_name) {
            self.embeddings = load_embedding(&loader, &word_embed_name, &self.device)?;
        }

        // Load position embeddings (for absolute position type)
        if self.position_type == PositionEmbeddingType::Absolute {
            let pos_embed_name = format!("{}embeddings.position_embeddings.weight", model_prefix);
            if loader.has_tensor(&pos_embed_name) {
                self.position_embeddings = Some(load_embedding(&loader, &pos_embed_name, &self.device)?);
            }
        }

        // Load token type embeddings
        let type_embed_name = format!("{}embeddings.token_type_embeddings.weight", model_prefix);
        if loader.has_tensor(&type_embed_name) {
            self.token_type_embeddings = Some(load_embedding(&loader, &type_embed_name, &self.device)?);
        }

        // Load embedding LayerNorm
        let embed_ln_weight = format!("{}embeddings.LayerNorm.weight", model_prefix);
        let embed_ln_bias = format!("{}embeddings.LayerNorm.bias", model_prefix);
        if loader.has_tensor(&embed_ln_weight) {
            self.embedding_layernorm = load_layer_norm(
                &loader,
                &embed_ln_weight,
                Some(&embed_ln_bias),
                hidden_size,
                eps,
                &self.device,
            )?;
        }

        // Load encoder layers
        for (layer_idx, layer) in self.encoder.layers.iter_mut().enumerate() {
            let layer_prefix = format!("{}encoder.layer.{}", model_prefix, layer_idx);

            // Load attention weights
            let q_weight = format!("{}.attention.self.query.weight", layer_prefix);
            let q_bias = format!("{}.attention.self.query.bias", layer_prefix);
            let k_weight = format!("{}.attention.self.key.weight", layer_prefix);
            let k_bias = format!("{}.attention.self.key.bias", layer_prefix);
            let v_weight = format!("{}.attention.self.value.weight", layer_prefix);
            let v_bias = format!("{}.attention.self.value.bias", layer_prefix);
            let o_weight = format!("{}.attention.output.dense.weight", layer_prefix);
            let o_bias = format!("{}.attention.output.dense.bias", layer_prefix);

            if loader.has_tensor(&q_weight) {
                layer.attention = load_mha(
                    &loader,
                    &q_weight,
                    Some(&q_bias),
                    &k_weight,
                    Some(&k_bias),
                    &v_weight,
                    Some(&v_bias),
                    &o_weight,
                    Some(&o_bias),
                    hidden_size,
                    num_heads,
                    dropout,
                    &self.device,
                )?;
            }

            // Load attention LayerNorm
            let attn_ln_weight = format!("{}.attention.output.LayerNorm.weight", layer_prefix);
            let attn_ln_bias = format!("{}.attention.output.LayerNorm.bias", layer_prefix);
            if loader.has_tensor(&attn_ln_weight) {
                layer.attention_layernorm = load_layer_norm(
                    &loader,
                    &attn_ln_weight,
                    Some(&attn_ln_bias),
                    hidden_size,
                    eps,
                    &self.device,
                )?;
            }

            // Load FFN weights (intermediate.dense = ffn_1, output.dense = ffn_2)
            let ffn1_weight = format!("{}.intermediate.dense.weight", layer_prefix);
            let ffn1_bias = format!("{}.intermediate.dense.bias", layer_prefix);
            if loader.has_tensor(&ffn1_weight) {
                layer.ffn_1 = load_linear(
                    &loader,
                    &ffn1_weight,
                    Some(&ffn1_bias),
                    &self.device,
                )?;
            }

            let ffn2_weight = format!("{}.output.dense.weight", layer_prefix);
            let ffn2_bias = format!("{}.output.dense.bias", layer_prefix);
            if loader.has_tensor(&ffn2_weight) {
                layer.ffn_2 = load_linear(
                    &loader,
                    &ffn2_weight,
                    Some(&ffn2_bias),
                    &self.device,
                )?;
            }

            // Load output LayerNorm
            let out_ln_weight = format!("{}.output.LayerNorm.weight", layer_prefix);
            let out_ln_bias = format!("{}.output.LayerNorm.bias", layer_prefix);
            if loader.has_tensor(&out_ln_weight) {
                layer.output_layernorm = load_layer_norm(
                    &loader,
                    &out_ln_weight,
                    Some(&out_ln_bias),
                    hidden_size,
                    eps,
                    &self.device,
                )?;
            }
        }

        log::info!("Successfully loaded BERT weights from {}", safetensors_path.display());
        Ok(())
    }

}

/// Dynamic cross-encoder model for reranking.
/// Supports both single-layer (classifier.weight) and two-layer (classifier.dense + classifier.out_proj) classifiers.
#[derive(Clone)]
pub struct DynamicCrossEncoder<B: Backend> {
    bert: DynamicBertModel<B>,
    pooler: DynamicPooler<B>,
    /// Optional intermediate dense layer (for two-layer classifiers like RoBERTa rerankers)
    classifier_dense: Option<Linear<B>>,
    /// Output projection layer
    classifier_out: Linear<B>,
    classifier_dropout: Option<Dropout>,
}

impl<B: Backend> DynamicCrossEncoder<B> {
    pub fn new(device: &B::Device, config: ModelConfig) -> Result<Self> {
        let bert = DynamicBertModel::new(device, config.clone())?;
        // Default: single-layer classifier [hidden_size, 1]
        let classifier_out = LinearConfig::new(config.hidden_size, 1).init(device);
        let classifier_dropout = config
            .classifier_dropout
            .map(|p| DropoutConfig::new(p as f64).init());

        let pooler = DynamicPooler::new(PoolingStrategy::Cls, PoolingConfig { normalize: false });

        Ok(Self {
            bert,
            pooler,
            classifier_dense: None,
            classifier_out,
            classifier_dropout,
        })
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 2>> {
        let hidden_states = self.bert.forward(input_ids)?;
        let mut pooled = self.pooler.pool(hidden_states, None);

        if let Some(dropout) = &self.classifier_dropout {
            pooled = dropout.forward(pooled);
        }

        // Apply two-layer classifier if present (classifier.dense + classifier.out_proj)
        if let Some(dense) = &self.classifier_dense {
            pooled = burn::tensor::activation::tanh(dense.forward(pooled));
        }

        Ok(self.classifier_out.forward(pooled))
    }

    pub fn forward_with_sigmoid(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 2>> {
        self.forward(input_ids).map(sigmoid)
    }

    pub fn load_safetensors(&mut self, safetensors_path: &Path) -> Result<()> {
        use crate::weight_loader::{load_linear, WeightLoader};

        // First load BERT weights
        self.bert.load_safetensors(safetensors_path)?;

        // Then load classifier weights
        let bytes = std::fs::read(safetensors_path).map_err(|err| {
            Error::LoadError(format!(
                "Failed to read SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;

        let loader = WeightLoader::from_bytes(&bytes)?;

        // Check for two-layer classifier (RoBERTa style: classifier.dense + classifier.out_proj)
        if loader.has_tensor("classifier.dense.weight") && loader.has_tensor("classifier.out_proj.weight") {
            log::info!("Loading two-layer classifier (dense + out_proj)");
            self.classifier_dense = Some(load_linear(
                &loader,
                "classifier.dense.weight",
                Some("classifier.dense.bias"),
                &self.bert.device,
            )?);
            self.classifier_out = load_linear(
                &loader,
                "classifier.out_proj.weight",
                Some("classifier.out_proj.bias"),
                &self.bert.device,
            )?;
            return Ok(());
        }

        // Check for single-layer classifier (BERT style: classifier.weight)
        if loader.has_tensor("classifier.weight") {
            log::info!("Loading single-layer classifier");
            self.classifier_out = load_linear(
                &loader,
                "classifier.weight",
                Some("classifier.bias"),
                &self.bert.device,
            )?;
            return Ok(());
        }

        log::warn!("No classifier weights found, using random initialization");
        Ok(())
    }

}
