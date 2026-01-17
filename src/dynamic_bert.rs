use crate::bert_variants::BertVariant;
use crate::causal_attention::{CausalAttention, RotaryPositionEmbedding};
use crate::model_config::ModelConfig;
use crate::pooling::{DynamicPooler, PoolingConfig, PoolingStrategy};
use crate::tensor::{Matrix, Tensor3};
use crate::types::{Error, Result};
use crate::weight_loader::{load_embedding, load_layer_norm, load_linear, load_mha, LayerNormWeights, LinearWeights, WeightLoader};
use gllm_kernels::{gelu_inplace, layer_norm_forward, linear_forward, WeightMatrix, WeightVector};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone)]
pub enum HiddenAct {
    Gelu,
    Relu,
}

/// Position embedding type.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PositionEmbeddingType {
    Absolute,
    Rope,
}

#[derive(Clone)]
pub struct DynamicBertLayer {
    pub(crate) attention: CausalAttention,
    pub(crate) ffn_1: LinearWeights,
    pub(crate) ffn_2: LinearWeights,
    pub(crate) attention_layernorm: LayerNormWeights,
    pub(crate) output_layernorm: LayerNormWeights,
    pub(crate) hidden_act: HiddenAct,
}

impl DynamicBertLayer {
    pub fn new(config: &ModelConfig, rope: Option<Arc<RotaryPositionEmbedding>>) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size.unwrap_or(hidden_size * 4);
        let eps = config.layer_norm_eps.unwrap_or(1e-12) as f32;

        let hidden_act = match config.hidden_act.as_deref() {
            Some("relu") | Some("relu_new") => HiddenAct::Relu,
            _ => HiddenAct::Gelu,
        };

        Ok(Self {
            attention: CausalAttention::new(config, rope, false)?,
            ffn_1: LinearWeights::zeros(intermediate_size, hidden_size),
            ffn_2: LinearWeights::zeros(hidden_size, intermediate_size),
            attention_layernorm: default_layer_norm(hidden_size, eps),
            output_layernorm: default_layer_norm(hidden_size, eps),
            hidden_act,
        })
    }

    pub fn forward(&self, input: &Tensor3) -> Result<Tensor3> {
        let attn_out = self.attention.forward(input, 0)?;
        let attn_residual = add_tensors(input, &attn_out)?;
        let attn_norm = apply_layer_norm(&attn_residual, &self.attention_layernorm)?;

        let ffn_out = self.ffn_forward(&attn_norm)?;
        let ffn_residual = add_tensors(&attn_norm, &ffn_out)?;
        apply_layer_norm(&ffn_residual, &self.output_layernorm)
    }

    fn ffn_forward(&self, input: &Tensor3) -> Result<Tensor3> {
        let (batch, seq_len, hidden) = input.shape();
        let rows = batch * seq_len;
        if hidden != self.ffn_1.weight.cols {
            return Err(Error::InferenceError(
                "BERT FFN input hidden size mismatch".into(),
            ));
        }

        let mut intermediate = vec![0.0f32; rows * self.ffn_1.weight.rows];
        linear_forward(
            &input.data,
            self.ffn_1.weight.as_slice(),
            self.ffn_1.bias.as_ref().map(|b| b.as_slice()),
            &mut intermediate,
            rows,
            self.ffn_1.weight.cols,
            self.ffn_1.weight.rows,
        );
        apply_activation(&self.hidden_act, &mut intermediate);

        let mut output = vec![0.0f32; rows * self.ffn_2.weight.rows];
        linear_forward(
            &intermediate,
            self.ffn_2.weight.as_slice(),
            self.ffn_2.bias.as_ref().map(|b| b.as_slice()),
            &mut output,
            rows,
            self.ffn_2.weight.cols,
            self.ffn_2.weight.rows,
        );

        Tensor3::new(output, batch, seq_len, self.ffn_2.weight.rows)
    }
}

#[derive(Clone)]
pub struct DynamicBertEncoder {
    pub(crate) layers: Vec<DynamicBertLayer>,
}

impl DynamicBertEncoder {
    pub fn new(config: &ModelConfig, rope: Option<Arc<RotaryPositionEmbedding>>) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(DynamicBertLayer::new(config, rope.clone())?);
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, hidden_states: Tensor3) -> Result<Tensor3> {
        self.layers
            .iter()
            .try_fold(hidden_states, |states, layer| layer.forward(&states))
    }
}

#[derive(Clone)]
pub struct DynamicBertModel {
    pub(crate) embeddings: WeightMatrix,
    pub(crate) position_embeddings: Option<WeightMatrix>,
    pub(crate) token_type_embeddings: Option<WeightMatrix>,
    pub(crate) embedding_layernorm: LayerNormWeights,
    pub(crate) encoder: DynamicBertEncoder,
    pub(crate) pooler: DynamicPooler,
    pub(crate) position_type: PositionEmbeddingType,
    pub(crate) rope: Option<Arc<RotaryPositionEmbedding>>,
    pub(crate) config: ModelConfig,
    pub(crate) max_position_embeddings: usize,
    pad_id: i64,
}

impl DynamicBertModel {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let variant = BertVariant::detect(&config);
        let hidden_size = config.hidden_size;
        let max_pos = config.max_position_embeddings;
        let type_vocab_size = variant.type_vocab_size(&config);
        let pad_id = config.pad_token_id.unwrap_or(0);

        let position_type = match config.position_embedding_type.as_deref() {
            Some("rope") | Some("rotary") => PositionEmbeddingType::Rope,
            _ => PositionEmbeddingType::Absolute,
        };

        let head_dim = config
            .head_dim
            .unwrap_or_else(|| hidden_size / config.num_attention_heads);
        let rope = if position_type == PositionEmbeddingType::Rope {
            CausalAttention::build_rope(&config, head_dim)
        } else {
            None
        };

        let embeddings = WeightMatrix::zeros(config.vocab_size, hidden_size);
        let position_embeddings = (position_type == PositionEmbeddingType::Absolute)
            .then(|| WeightMatrix::zeros(max_pos, hidden_size));
        let token_type_embeddings =
            (type_vocab_size > 0).then(|| WeightMatrix::zeros(type_vocab_size, hidden_size));
        let embedding_layernorm =
            default_layer_norm(hidden_size, config.layer_norm_eps.unwrap_or(1e-12) as f32);
        let encoder = DynamicBertEncoder::new(&config, rope.clone())?;
        let pooler = DynamicPooler::new(
            variant.pooling_strategy(&config),
            PoolingConfig { normalize: true },
        );

        Ok(Self {
            embeddings,
            position_embeddings,
            token_type_embeddings,
            embedding_layernorm,
            encoder,
            pooler,
            position_type,
            rope,
            config,
            max_position_embeddings: max_pos,
            pad_id,
        })
    }

    pub fn forward(&self, tokens: &[Vec<i64>]) -> Result<Tensor3> {
        let (batch, seq_len) = sequence_shape(tokens, self.max_position_embeddings)?;
        let mut embeddings = embed_tokens(
            &self.embeddings,
            tokens,
            self.pad_id,
            batch,
            seq_len,
        )?;

        if self.position_type == PositionEmbeddingType::Absolute {
            if let Some(pos_emb) = &self.position_embeddings {
                add_position_embeddings(&mut embeddings, pos_emb)?;
            }
        }

        if let Some(token_type) = &self.token_type_embeddings {
            add_token_type_embeddings(&mut embeddings, token_type)?;
        }

        embeddings = apply_layer_norm(&embeddings, &self.embedding_layernorm)?;
        self.encoder.forward(embeddings)
    }

    pub fn pool_hidden_states(&self, hidden_states: &Tensor3, tokens: &[Vec<i64>]) -> Matrix {
        let mask = build_token_mask(tokens, hidden_states.dim1, self.pad_id);
        self.pooler.pool(hidden_states, Some(&mask))
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
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
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let dropout = self.config.attention_probs_dropout_prob.unwrap_or(0.1) as f64;
        let eps = self.config.layer_norm_eps.unwrap_or(1e-12) as f64;

        let prefixes = ["bert.", "roberta.", "distilbert.", "xlm-roberta.", ""];
        let model_prefix = prefixes
            .iter()
            .find(|p| loader.has_tensor(&format!("{}embeddings.word_embeddings.weight", p)))
            .copied()
            .unwrap_or("");

        let word_embed_name = format!("{}embeddings.word_embeddings.weight", model_prefix);
        if loader.has_tensor(&word_embed_name) {
            self.embeddings = load_embedding(&loader, &word_embed_name)?;
        }

        if self.position_type == PositionEmbeddingType::Absolute {
            let pos_embed_name = format!("{}embeddings.position_embeddings.weight", model_prefix);
            if loader.has_tensor(&pos_embed_name) {
                self.position_embeddings = Some(load_embedding(&loader, &pos_embed_name)?);
            }
        }

        let type_embed_name = format!("{}embeddings.token_type_embeddings.weight", model_prefix);
        if loader.has_tensor(&type_embed_name) {
            self.token_type_embeddings = Some(load_embedding(&loader, &type_embed_name)?);
        }

        let embed_ln_weight = format!("{}embeddings.LayerNorm.weight", model_prefix);
        let embed_ln_bias = format!("{}embeddings.LayerNorm.bias", model_prefix);
        if loader.has_tensor(&embed_ln_weight) {
            self.embedding_layernorm = load_layer_norm(
                &loader,
                &embed_ln_weight,
                Some(&embed_ln_bias),
                hidden_size,
                eps,
            )?;
        }

        for (layer_idx, layer) in self.encoder.layers.iter_mut().enumerate() {
            let layer_prefix = format!("{}encoder.layer.{}", model_prefix, layer_idx);
            let q_weight = format!("{}.attention.self.query.weight", layer_prefix);
            let q_bias = format!("{}.attention.self.query.bias", layer_prefix);
            let k_weight = format!("{}.attention.self.key.weight", layer_prefix);
            let k_bias = format!("{}.attention.self.key.bias", layer_prefix);
            let v_weight = format!("{}.attention.self.value.weight", layer_prefix);
            let v_bias = format!("{}.attention.self.value.bias", layer_prefix);
            let o_weight = format!("{}.attention.output.dense.weight", layer_prefix);
            let o_bias = format!("{}.attention.output.dense.bias", layer_prefix);

            if loader.has_tensor(&q_weight) {
                let mha = load_mha(
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
                )?;
                layer.attention.q_proj = mha.query;
                layer.attention.k_proj = mha.key;
                layer.attention.v_proj = mha.value;
                layer.attention.o_proj = mha.output;
            }

            let attn_ln_weight = format!("{}.attention.output.LayerNorm.weight", layer_prefix);
            let attn_ln_bias = format!("{}.attention.output.LayerNorm.bias", layer_prefix);
            if loader.has_tensor(&attn_ln_weight) {
                layer.attention_layernorm = load_layer_norm(
                    &loader,
                    &attn_ln_weight,
                    Some(&attn_ln_bias),
                    hidden_size,
                    eps,
                )?;
            }

            let ffn1_weight = format!("{}.intermediate.dense.weight", layer_prefix);
            let ffn1_bias = format!("{}.intermediate.dense.bias", layer_prefix);
            if loader.has_tensor(&ffn1_weight) {
                layer.ffn_1 = load_linear(&loader, &ffn1_weight, Some(&ffn1_bias))?;
            }

            let ffn2_weight = format!("{}.output.dense.weight", layer_prefix);
            let ffn2_bias = format!("{}.output.dense.bias", layer_prefix);
            if loader.has_tensor(&ffn2_weight) {
                layer.ffn_2 = load_linear(&loader, &ffn2_weight, Some(&ffn2_bias))?;
            }

            let out_ln_weight = format!("{}.output.LayerNorm.weight", layer_prefix);
            let out_ln_bias = format!("{}.output.LayerNorm.bias", layer_prefix);
            if loader.has_tensor(&out_ln_weight) {
                layer.output_layernorm = load_layer_norm(
                    &loader,
                    &out_ln_weight,
                    Some(&out_ln_bias),
                    hidden_size,
                    eps,
                )?;
            }
        }

        Ok(())
    }
}

/// Dynamic cross-encoder model for reranking.
#[derive(Clone)]
pub struct DynamicCrossEncoder {
    encoder: DynamicBertModel,
    pooler: DynamicPooler,
    classifier_dense: Option<LinearWeights>,
    classifier_out: LinearWeights,
}

impl DynamicCrossEncoder {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let encoder = DynamicBertModel::new(config.clone())?;
        let classifier_out = LinearWeights::zeros(1, config.hidden_size);
        let pooler = DynamicPooler::new(PoolingStrategy::Cls, PoolingConfig { normalize: false });

        Ok(Self {
            encoder,
            pooler,
            classifier_dense: None,
            classifier_out,
        })
    }

    pub fn score(&self, tokens: &[Vec<i64>]) -> Result<Vec<f32>> {
        let hidden = self.encoder.forward(tokens)?;
        let mask = build_token_mask(tokens, hidden.dim1, self.encoder.pad_id);
        let mut pooled = self.pooler.pool(&hidden, Some(&mask));

        if let Some(dense) = &self.classifier_dense {
            pooled = apply_linear(&pooled, dense)?;
            apply_tanh_inplace(&mut pooled.data);
        }

        let logits = apply_linear(&pooled, &self.classifier_out)?;
        Ok(logits
            .data
            .chunks(logits.cols)
            .map(|row| sigmoid(row.first().copied().unwrap_or(0.0)))
            .collect())
    }

    pub fn load_safetensors(&mut self, safetensors_path: &Path) -> Result<()> {
        self.encoder.load_safetensors(safetensors_path)?;

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

        if loader.has_tensor("classifier.dense.weight") && loader.has_tensor("classifier.out_proj.weight") {
            self.classifier_dense = Some(load_linear(
                &loader,
                "classifier.dense.weight",
                Some("classifier.dense.bias"),
            )?);
            self.classifier_out = load_linear(
                &loader,
                "classifier.out_proj.weight",
                Some("classifier.out_proj.bias"),
            )?;
            return Ok(());
        }

        if loader.has_tensor("classifier.weight") {
            self.classifier_out =
                load_linear(&loader, "classifier.weight", Some("classifier.bias"))?;
            return Ok(());
        }

        if loader.has_tensor("score.weight") {
            self.classifier_out = load_linear(&loader, "score.weight", Some("score.bias"))?;
        }

        Ok(())
    }
}

fn default_layer_norm(hidden_size: usize, eps: f32) -> LayerNormWeights {
    LayerNormWeights {
        gamma: WeightVector::ones(hidden_size),
        beta: Some(WeightVector::zeros(hidden_size)),
        eps,
    }
}

fn apply_layer_norm(input: &Tensor3, weights: &LayerNormWeights) -> Result<Tensor3> {
    let (batch, seq_len, hidden) = input.shape();
    let rows = batch * seq_len;
    if weights.gamma.len() != hidden {
        return Err(Error::InferenceError(
            "LayerNorm gamma length mismatch".into(),
        ));
    }

    let zero_beta = vec![0.0f32; weights.gamma.len()];
    let beta = weights
        .beta
        .as_ref()
        .map(|b| b.as_slice())
        .unwrap_or(zero_beta.as_slice());

    let mut output = vec![0.0f32; input.data.len()];
    layer_norm_forward(
        &input.data,
        weights.gamma.as_slice(),
        beta,
        &mut output,
        rows,
        hidden,
        weights.eps,
    );
    Tensor3::new(output, batch, seq_len, hidden)
}

fn add_tensors(lhs: &Tensor3, rhs: &Tensor3) -> Result<Tensor3> {
    if lhs.data.len() != rhs.data.len() {
        return Err(Error::InferenceError(
            "Tensor add length mismatch".into(),
        ));
    }
    let mut out = Vec::with_capacity(lhs.data.len());
    for (a, b) in lhs.data.iter().zip(rhs.data.iter()) {
        out.push(a + b);
    }
    Tensor3::new(out, lhs.dim0, lhs.dim1, lhs.dim2)
}

fn apply_activation(act: &HiddenAct, data: &mut [f32]) {
    match act {
        HiddenAct::Gelu => gelu_inplace(data),
        HiddenAct::Relu => {
            for v in data.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
    }
}

fn apply_tanh_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = v.tanh();
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sequence_shape(tokens: &[Vec<i64>], max_len: usize) -> Result<(usize, usize)> {
    if tokens.is_empty() {
        return Err(Error::InvalidConfig(
            "At least one input is required".into(),
        ));
    }
    let batch = tokens.len();
    let seq_len = tokens.iter().map(|t| t.len()).max().unwrap_or(0);
    if seq_len == 0 {
        return Err(Error::InvalidConfig(
            "input sequence length must be greater than 0".into(),
        ));
    }
    if max_len > 0 && seq_len > max_len {
        return Err(Error::InvalidConfig(format!(
            "Sequence length {} exceeds configured maximum {}",
            seq_len, max_len
        )));
    }
    Ok((batch, seq_len))
}

fn embed_tokens(
    embeddings: &WeightMatrix,
    tokens: &[Vec<i64>],
    pad_id: i64,
    batch: usize,
    seq_len: usize,
) -> Result<Tensor3> {
    let hidden = embeddings.cols;
    let mut data = vec![0.0f32; batch * seq_len * hidden];
    for (b, ids) in tokens.iter().enumerate() {
        for s in 0..seq_len {
            let id = ids.get(s).copied().unwrap_or(pad_id);
            let idx = safe_token_index(id, embeddings.rows);
            let row = embeddings.row(idx);
            let start = (b * seq_len + s) * hidden;
            data[start..start + hidden].copy_from_slice(row);
        }
    }
    Tensor3::new(data, batch, seq_len, hidden)
}

fn add_position_embeddings(embeddings: &mut Tensor3, position: &WeightMatrix) -> Result<()> {
    let (batch, seq_len, hidden) = embeddings.shape();
    if position.cols != hidden {
        return Err(Error::InferenceError(
            "Position embedding hidden size mismatch".into(),
        ));
    }
    for s in 0..seq_len {
        let pos_row = position.row(s.min(position.rows.saturating_sub(1)));
        for b in 0..batch {
            let start = (b * seq_len + s) * hidden;
            for i in 0..hidden {
                embeddings.data[start + i] += pos_row[i];
            }
        }
    }
    Ok(())
}

fn add_token_type_embeddings(embeddings: &mut Tensor3, token_type: &WeightMatrix) -> Result<()> {
    let (batch, seq_len, hidden) = embeddings.shape();
    if token_type.cols != hidden {
        return Err(Error::InferenceError(
            "Token type embedding hidden size mismatch".into(),
        ));
    }
    if token_type.rows == 0 {
        return Ok(());
    }
    let row = token_type.row(0);
    for b in 0..batch {
        for s in 0..seq_len {
            let start = (b * seq_len + s) * hidden;
            for i in 0..hidden {
                embeddings.data[start + i] += row[i];
            }
        }
    }
    Ok(())
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

fn build_token_mask(tokens: &[Vec<i64>], seq_len: usize, pad_id: i64) -> Vec<i64> {
    let mut mask = Vec::with_capacity(tokens.len() * seq_len);
    for ids in tokens {
        for &id in ids {
            mask.push(if id == pad_id { 0 } else { 1 });
        }
        if ids.len() < seq_len {
            mask.extend(std::iter::repeat(0).take(seq_len - ids.len()));
        }
    }
    mask
}

fn apply_linear(input: &Matrix, weights: &LinearWeights) -> Result<Matrix> {
    if input.cols != weights.weight.cols {
        return Err(Error::InferenceError(
            "Linear input feature mismatch".into(),
        ));
    }
    let rows = input.rows;
    let mut output = vec![0.0f32; rows * weights.weight.rows];
    linear_forward(
        &input.data,
        weights.weight.as_slice(),
        weights.bias.as_ref().map(|b| b.as_slice()),
        &mut output,
        rows,
        weights.weight.cols,
        weights.weight.rows,
    );
    Matrix::new(output, rows, weights.weight.rows)
}
