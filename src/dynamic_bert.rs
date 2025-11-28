use crate::bert_variants::BertVariant;
use crate::model_config::ModelConfig;
use crate::performance_optimizer::PerformanceOptimizer;
use crate::pooling::{DynamicPooler, PoolingConfig, PoolingStrategy};
use crate::types::{Error, Result};
use burn::nn::attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig};
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear,
    LinearConfig,
};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use safetensors::SafeTensors;
use std::path::Path;

#[derive(Clone)]
pub enum HiddenAct {
    Gelu,
    Relu,
}

/// Dynamic BERT layer that can be configured based on model config.
#[derive(Clone)]
pub struct DynamicBertLayer<B: Backend> {
    attention: MultiHeadAttention<B>,
    ffn_1: Linear<B>,
    ffn_2: Linear<B>,
    attention_layernorm: LayerNorm<B>,
    output_layernorm: LayerNorm<B>,
    hidden_act: HiddenAct,
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
    layers: Vec<DynamicBertLayer<B>>,
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

/// Dynamic BERT model that can be configured based on HuggingFace config.
#[derive(Clone)]
pub struct DynamicBertModel<B: Backend> {
    embeddings: Embedding<B>,
    position_embeddings: Embedding<B>,
    token_type_embeddings: Option<Embedding<B>>,
    encoder: DynamicBertEncoder<B>,
    embedding_layernorm: LayerNorm<B>,
    embedding_dropout: Option<Dropout>,
    pooler: DynamicPooler<B>,
    optimizer: PerformanceOptimizer,
    config: ModelConfig,
    device: B::Device,
    max_position_embeddings: usize,
}

impl<B: Backend> DynamicBertModel<B> {
    pub fn new(device: &B::Device, config: ModelConfig) -> Result<Self> {
        let variant = BertVariant::detect(&config);
        let optimizer = PerformanceOptimizer::from_config(&config);

        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let max_pos = config.max_position_embeddings;
        let type_vocab_size = variant.type_vocab_size(&config);

        let embeddings = EmbeddingConfig::new(vocab_size, hidden_size).init(device);
        let position_embeddings = EmbeddingConfig::new(max_pos, hidden_size).init(device);
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

        let position_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &self.device)
            .reshape([1, seq_len])
            .repeat(&[batch_size, 1]);

        let token_embeddings = self.embeddings.forward(input_ids);
        let position_embeddings = self.position_embeddings.forward(position_ids);

        let mut embeddings = token_embeddings + position_embeddings;
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
        let bytes = std::fs::read(safetensors_path).map_err(|err| {
            Error::LoadError(format!(
                "Failed to read SafeTensors file {}: {err}",
                safetensors_path.display()
            ))
        })?;

        let tensors = SafeTensors::deserialize(&bytes)
            .map_err(|err| Error::LoadError(format!("Invalid SafeTensors: {err}")))?;

        if tensors.len() == 0 {
            return Err(Error::LoadError(
                "SafeTensors file contains no tensors".into(),
            ));
        }

        Ok(())
    }
}

/// Dynamic cross-encoder model for reranking.
#[derive(Clone)]
pub struct DynamicCrossEncoder<B: Backend> {
    bert: DynamicBertModel<B>,
    pooler: DynamicPooler<B>,
    classifier: Linear<B>,
    classifier_dropout: Option<Dropout>,
}

impl<B: Backend> DynamicCrossEncoder<B> {
    pub fn new(device: &B::Device, config: ModelConfig) -> Result<Self> {
        let bert = DynamicBertModel::new(device, config.clone())?;
        let classifier = LinearConfig::new(config.hidden_size, 1).init(device);
        let classifier_dropout = config
            .classifier_dropout
            .map(|p| DropoutConfig::new(p as f64).init());

        let pooler = DynamicPooler::new(PoolingStrategy::Cls, PoolingConfig { normalize: false });

        Ok(Self {
            bert,
            pooler,
            classifier,
            classifier_dropout,
        })
    }

    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 2>> {
        let hidden_states = self.bert.forward(input_ids)?;
        let mut pooled = self.pooler.pool(hidden_states, None);

        if let Some(dropout) = &self.classifier_dropout {
            pooled = dropout.forward(pooled);
        }

        Ok(self.classifier.forward(pooled))
    }

    pub fn forward_with_sigmoid(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 2>> {
        self.forward(input_ids).map(sigmoid)
    }

    pub fn load_safetensors(&mut self, safetensors_path: &Path) -> Result<()> {
        self.bert.load_safetensors(safetensors_path)
    }
}
