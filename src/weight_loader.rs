//! Weight loading module for SafeTensors files.
//!
//! This module provides functionality to load HuggingFace model weights
//! into gllm model structures. It handles:
//! - SafeTensors file parsing
//! - HuggingFace weight name mapping
//! - Linear weights stored as [out, in] row-major
//! - Multi-shard file loading
//! - Support for different model architectures

use crate::quantized::{NativeQLinear, QuantizedWeight};
use crate::types::{Error, Result};
use gllm_kernels::backend::Backend;
use gllm_kernels::{WeightMatrix, WeightVector};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

/// Weight loader for SafeTensors files.
pub struct WeightLoader<'a> {
    tensors: SafeTensors<'a>,
}

/// Loaded tensor data with shape and dtype information.
#[derive(Debug)]
pub struct LoadedTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Raw tensor data with shape and dtype information (no conversion).
#[derive(Debug)]
pub struct RawTensor {
    /// Raw tensor bytes as stored in safetensors.
    pub data: Vec<u8>,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Tensor dtype.
    pub dtype: Dtype,
}

impl LoadedTensor {
    /// Convert to a WeightMatrix ([rows, cols] row-major).
    pub fn to_weight_matrix(&self) -> Result<WeightMatrix> {
        if self.shape.len() != 2 {
            return Err(Error::LoadError(format!(
                "Expected 2D tensor for matrix, got {}D",
                self.shape.len()
            )));
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        if self.data.len() != rows * cols {
            return Err(Error::LoadError(
                "Matrix data length does not match shape".into(),
            ));
        }

        Ok(WeightMatrix::new(self.data.clone(), rows, cols))
    }

    /// Convert to a WeightVector (accepts 1D or 2D with a singleton dimension).
    pub fn to_weight_vector(&self) -> Result<WeightVector> {
        match self.shape.as_slice() {
            [len] => {
                if self.data.len() != *len {
                    return Err(Error::LoadError(
                        "Vector data length does not match shape".into(),
                    ));
                }
                Ok(WeightVector::new(self.data.clone()))
            }
            [1, len] | [len, 1] => {
                if self.data.len() != *len {
                    return Err(Error::LoadError(
                        "Vector data length does not match shape".into(),
                    ));
                }
                Ok(WeightVector::new(self.data.clone()))
            }
            _ => Err(Error::LoadError(format!(
                "Expected 1D tensor for vector, got shape {:?}",
                self.shape
            ))),
        }
    }
}

impl<'a> WeightLoader<'a> {
    /// Create a new weight loader from a SafeTensors file.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|e| Error::LoadError(format!("Failed to deserialize SafeTensors: {}", e)))?;

        Ok(Self { tensors })
    }

    /// Get all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.names().into_iter().map(|s| s.to_string()).collect()
    }

    /// Load a tensor by name.
    pub fn load_tensor(&self, name: &str) -> Result<LoadedTensor> {
        let tensor_view = self.tensors.tensor(name).map_err(|e| {
            Error::LoadError(format!("Failed to load tensor '{}': {}", name, e))
        })?;

        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let dtype = tensor_view.dtype();
        let raw_data = tensor_view.data();

        // Convert to f32
        let data = convert_to_f32(&raw_data, dtype)?;

        Ok(LoadedTensor { data, shape })
    }

    /// Load a raw tensor by name without dtype conversion.
    pub fn load_raw_tensor(&self, name: &str) -> Result<RawTensor> {
        let tensor_view = self.tensors.tensor(name).map_err(|e| {
            Error::LoadError(format!("Failed to load tensor '{}': {}", name, e))
        })?;
        let shape = tensor_view.shape().to_vec();
        let dtype = tensor_view.dtype();
        let data = tensor_view.data().to_vec();

        Ok(RawTensor { data, shape, dtype })
    }

    /// Check if a tensor exists.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.tensor(name).is_ok()
    }

    /// Check if this safetensors file contains AWQ-weighted tensors.
    pub fn is_awq_model(&self) -> bool {
        self.has_tensor("model.layers.0.self_attn.q_proj.qweight")
    }
}

/// Convert raw bytes to f32 based on dtype.
fn convert_to_f32(data: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            let floats: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok(floats)
        }
        Dtype::F16 => {
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();
            Ok(floats)
        }
        Dtype::BF16 => {
            let floats: Vec<f32> = data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect();
            Ok(floats)
        }
        Dtype::F64 => {
            let floats: Vec<f32> = data
                .chunks_exact(8)
                .map(|chunk| {
                    let arr: [u8; 8] = chunk.try_into().unwrap();
                    f64::from_le_bytes(arr) as f32
                })
                .collect();
            Ok(floats)
        }
        _ => Err(Error::LoadError(format!(
            "Unsupported dtype for weight loading: {:?}",
            dtype
        ))),
    }
}

/// Load Linear layer weights from SafeTensors.
///
/// HuggingFace Linear weights are stored as [out_features, in_features].
#[derive(Clone, Debug)]
pub enum LinearWeights {
    Dense {
        weight: WeightMatrix,
        bias: Option<WeightVector>,
    },
    Quantized {
        weight: NativeQLinear,
    },
}

impl LinearWeights {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::Dense {
            weight: WeightMatrix::zeros(rows, cols),
            bias: None,
        }
    }

    pub fn from_dense(weight: WeightMatrix, bias: Option<WeightVector>) -> Self {
        Self::Dense { weight, bias }
    }

    pub fn from_quantized(weight: NativeQLinear) -> Self {
        Self::Quantized { weight }
    }

    pub fn in_features(&self) -> usize {
        match self {
            Self::Dense { weight, .. } => weight.cols,
            Self::Quantized { weight } => weight.in_features(),
        }
    }

    pub fn out_features(&self) -> usize {
        match self {
            Self::Dense { weight, .. } => weight.rows,
            Self::Quantized { weight } => weight.out_features(),
        }
    }

    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        batch: usize,
        backend: &dyn Backend,
    ) -> Result<()> {
        match self {
            Self::Dense { weight, bias } => {
                if input.len() != batch * weight.cols {
                    return Err(Error::InferenceError(
                        "Linear input length mismatch".into(),
                    ));
                }
                if output.len() != batch * weight.rows {
                    return Err(Error::InferenceError(
                        "Linear output length mismatch".into(),
                    ));
                }
                gllm_kernels::linear_forward(
                    input,
                    weight.as_slice(),
                    bias.as_ref().map(|b| b.as_slice()),
                    output,
                    batch,
                    weight.cols,
                    weight.rows,
                );
                Ok(())
            }
            Self::Quantized { weight } => weight.forward(input, output, batch, backend),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LayerNormWeights {
    pub gamma: WeightVector,
    pub beta: Option<WeightVector>,
    pub eps: f32,
}

#[derive(Clone, Debug)]
pub struct MultiHeadAttentionWeights {
    pub query: LinearWeights,
    pub key: LinearWeights,
    pub value: LinearWeights,
    pub output: LinearWeights,
    pub d_model: usize,
    pub n_heads: usize,
}

/// Load Linear layer weights from SafeTensors.
pub fn load_linear(
    loader: &WeightLoader,
    weight_name: &str,
    bias_name: Option<&str>,
) -> Result<LinearWeights> {
    if loader.has_tensor(weight_name) {
        let weight_tensor = loader.load_tensor(weight_name)?;
        let weight = weight_tensor.to_weight_matrix()?;

        let bias = if let Some(bias_name) = bias_name {
            if loader.has_tensor(bias_name) {
                let bias_tensor = loader.load_tensor(bias_name)?;
                Some(bias_tensor.to_weight_vector()?)
            } else {
                None
            }
        } else {
            None
        };

        return Ok(LinearWeights::from_dense(weight, bias));
    }

    let prefix = weight_name
        .strip_suffix(".weight")
        .ok_or_else(|| Error::LoadError("Quantized weight name missing .weight suffix".into()))?;

    if loader.is_awq_model() {
        let awq = crate::awq::AwqQuantizedWeight::from_safetensors(loader, prefix)?;
        let bias = if let Some(bias_name) = bias_name {
            if loader.has_tensor(bias_name) {
                let bias_tensor = loader.load_tensor(bias_name)?;
                Some(bias_tensor.to_weight_vector()?.data)
            } else {
                None
            }
        } else {
            None
        };
        let native = NativeQLinear::new(
            QuantizedWeight::Awq {
                weight: awq.weight,
                shape: awq.shape,
            },
            bias,
        )?;
        return Ok(LinearWeights::from_quantized(native));
    }

    Err(Error::LoadError(format!(
        "Linear weight '{}' not found",
        weight_name
    )))
}

/// Load Embedding layer weights from SafeTensors.
pub fn load_embedding(loader: &WeightLoader, weight_name: &str) -> Result<WeightMatrix> {
    let weight_tensor = loader.load_tensor(weight_name)?;
    weight_tensor.to_weight_matrix()
}

/// Load LayerNorm weights from SafeTensors.
///
/// BERT uses LayerNorm with gamma (weight) and beta (bias).
pub fn load_layer_norm(
    loader: &WeightLoader,
    weight_name: &str,
    bias_name: Option<&str>,
    d_model: usize,
    epsilon: f64,
) -> Result<LayerNormWeights> {
    let gamma_tensor = loader.load_tensor(weight_name)?;
    let gamma = gamma_tensor.to_weight_vector()?;
    if gamma.len() != d_model {
        return Err(Error::LoadError(format!(
            "LayerNorm gamma length mismatch: expected {}, got {}",
            d_model,
            gamma.len()
        )));
    }

    let beta = if let Some(bias_name) = bias_name {
        if loader.has_tensor(bias_name) {
            let beta_tensor = loader.load_tensor(bias_name)?;
            let beta_vec = beta_tensor.to_weight_vector()?;
            if beta_vec.len() != d_model {
                return Err(Error::LoadError(format!(
                    "LayerNorm beta length mismatch: expected {}, got {}",
                    d_model,
                    beta_vec.len()
                )));
            }
            Some(beta_vec)
        } else {
            None
        }
    } else {
        None
    };

    Ok(LayerNormWeights {
        gamma,
        beta,
        eps: epsilon as f32,
    })
}

/// Load MultiHeadAttention weights from SafeTensors (BERT-style).
///
/// BERT attention has separate query, key, value projections.
pub fn load_mha(
    loader: &WeightLoader,
    query_weight: &str,
    query_bias: Option<&str>,
    key_weight: &str,
    key_bias: Option<&str>,
    value_weight: &str,
    value_bias: Option<&str>,
    output_weight: &str,
    output_bias: Option<&str>,
    d_model: usize,
    n_heads: usize,
    _dropout: f64,
) -> Result<MultiHeadAttentionWeights> {
    Ok(MultiHeadAttentionWeights {
        query: load_linear(loader, query_weight, query_bias)?,
        key: load_linear(loader, key_weight, key_bias)?,
        value: load_linear(loader, value_weight, value_bias)?,
        output: load_linear(loader, output_weight, output_bias)?,
        d_model,
        n_heads,
    })
}

/// Architecture-specific weight name mappings.
pub mod mappings {
    /// Weight name patterns for different model architectures.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Architecture {
        /// LLaMA, Mistral, Qwen, etc.
        Llama,
        /// GPT-2, GPT-Neo, etc.
        Gpt2,
        /// BERT, RoBERTa, etc.
        Bert,
        /// GLM models
        Glm,
    }

    impl Architecture {
        /// Detect architecture from model type string.
        pub fn from_model_type(model_type: &str) -> Self {
            let lower = model_type.to_lowercase();
            if lower.contains("llama")
                || lower.contains("mistral")
                || lower.contains("qwen")
                || lower.contains("deepseek")
                || lower.contains("mixtral") {
                Architecture::Llama
            } else if lower.contains("gpt2") || lower.contains("gpt-neo") {
                Architecture::Gpt2
            } else if lower.contains("glm") || lower.contains("chatglm") {
                Architecture::Glm
            } else {
                Architecture::Bert
            }
        }

        /// Get the embedding weight name for this architecture.
        pub fn embedding_weight(&self) -> &'static str {
            match self {
                Architecture::Llama => "model.embed_tokens.weight",
                Architecture::Gpt2 => "transformer.wte.weight",
                Architecture::Bert => "embeddings.word_embeddings.weight",
                Architecture::Glm => "transformer.embedding.word_embeddings.weight",
            }
        }

        /// Get the layer prefix for this architecture.
        pub fn layer_prefix(&self, layer_idx: usize) -> String {
            match self {
                Architecture::Llama => format!("model.layers.{}", layer_idx),
                Architecture::Gpt2 => format!("transformer.h.{}", layer_idx),
                Architecture::Bert => format!("encoder.layer.{}", layer_idx),
                Architecture::Glm => format!("transformer.encoder.layers.{}", layer_idx),
            }
        }

        /// Get attention weight names for a layer.
        pub fn attention_weights(&self, layer_prefix: &str) -> AttentionWeights {
            match self {
                Architecture::Llama => AttentionWeights {
                    q_proj_weight: format!("{}.self_attn.q_proj.weight", layer_prefix),
                    k_proj_weight: format!("{}.self_attn.k_proj.weight", layer_prefix),
                    v_proj_weight: format!("{}.self_attn.v_proj.weight", layer_prefix),
                    o_proj_weight: format!("{}.self_attn.o_proj.weight", layer_prefix),
                    q_proj_bias: None,
                    k_proj_bias: None,
                    v_proj_bias: None,
                    o_proj_bias: None,
                },
                Architecture::Gpt2 => AttentionWeights {
                    q_proj_weight: format!("{}.attn.c_attn.weight", layer_prefix),
                    k_proj_weight: format!("{}.attn.c_attn.weight", layer_prefix),
                    v_proj_weight: format!("{}.attn.c_attn.weight", layer_prefix),
                    o_proj_weight: format!("{}.attn.c_proj.weight", layer_prefix),
                    q_proj_bias: Some(format!("{}.attn.c_attn.bias", layer_prefix)),
                    k_proj_bias: Some(format!("{}.attn.c_attn.bias", layer_prefix)),
                    v_proj_bias: Some(format!("{}.attn.c_attn.bias", layer_prefix)),
                    o_proj_bias: Some(format!("{}.attn.c_proj.bias", layer_prefix)),
                },
                Architecture::Bert => AttentionWeights {
                    q_proj_weight: format!("{}.attention.self.query.weight", layer_prefix),
                    k_proj_weight: format!("{}.attention.self.key.weight", layer_prefix),
                    v_proj_weight: format!("{}.attention.self.value.weight", layer_prefix),
                    o_proj_weight: format!("{}.attention.output.dense.weight", layer_prefix),
                    q_proj_bias: Some(format!("{}.attention.self.query.bias", layer_prefix)),
                    k_proj_bias: Some(format!("{}.attention.self.key.bias", layer_prefix)),
                    v_proj_bias: Some(format!("{}.attention.self.value.bias", layer_prefix)),
                    o_proj_bias: Some(format!("{}.attention.output.dense.bias", layer_prefix)),
                },
                Architecture::Glm => AttentionWeights {
                    q_proj_weight: format!("{}.self_attention.query_key_value.weight", layer_prefix),
                    k_proj_weight: format!("{}.self_attention.query_key_value.weight", layer_prefix),
                    v_proj_weight: format!("{}.self_attention.query_key_value.weight", layer_prefix),
                    o_proj_weight: format!("{}.self_attention.dense.weight", layer_prefix),
                    q_proj_bias: Some(format!("{}.self_attention.query_key_value.bias", layer_prefix)),
                    k_proj_bias: Some(format!("{}.self_attention.query_key_value.bias", layer_prefix)),
                    v_proj_bias: Some(format!("{}.self_attention.query_key_value.bias", layer_prefix)),
                    o_proj_bias: Some(format!("{}.self_attention.dense.bias", layer_prefix)),
                },
            }
        }

        /// Get FFN weight names for a layer.
        pub fn ffn_weights(&self, layer_prefix: &str) -> FfnWeights {
            match self {
                Architecture::Llama => FfnWeights {
                    gate_proj_weight: format!("{}.mlp.gate_proj.weight", layer_prefix),
                    up_proj_weight: format!("{}.mlp.up_proj.weight", layer_prefix),
                    down_proj_weight: format!("{}.mlp.down_proj.weight", layer_prefix),
                    gate_proj_bias: None,
                    up_proj_bias: None,
                    down_proj_bias: None,
                },
                Architecture::Gpt2 => FfnWeights {
                    gate_proj_weight: format!("{}.mlp.c_fc.weight", layer_prefix),
                    up_proj_weight: format!("{}.mlp.c_fc.weight", layer_prefix),
                    down_proj_weight: format!("{}.mlp.c_proj.weight", layer_prefix),
                    gate_proj_bias: Some(format!("{}.mlp.c_fc.bias", layer_prefix)),
                    up_proj_bias: Some(format!("{}.mlp.c_fc.bias", layer_prefix)),
                    down_proj_bias: Some(format!("{}.mlp.c_proj.bias", layer_prefix)),
                },
                Architecture::Bert => FfnWeights {
                    gate_proj_weight: format!("{}.intermediate.dense.weight", layer_prefix),
                    up_proj_weight: format!("{}.intermediate.dense.weight", layer_prefix),
                    down_proj_weight: format!("{}.output.dense.weight", layer_prefix),
                    gate_proj_bias: Some(format!("{}.intermediate.dense.bias", layer_prefix)),
                    up_proj_bias: Some(format!("{}.intermediate.dense.bias", layer_prefix)),
                    down_proj_bias: Some(format!("{}.output.dense.bias", layer_prefix)),
                },
                Architecture::Glm => FfnWeights {
                    gate_proj_weight: format!("{}.mlp.dense_h_to_4h.weight", layer_prefix),
                    up_proj_weight: format!("{}.mlp.dense_h_to_4h.weight", layer_prefix),
                    down_proj_weight: format!("{}.mlp.dense_4h_to_h.weight", layer_prefix),
                    gate_proj_bias: Some(format!("{}.mlp.dense_h_to_4h.bias", layer_prefix)),
                    up_proj_bias: Some(format!("{}.mlp.dense_h_to_4h.bias", layer_prefix)),
                    down_proj_bias: Some(format!("{}.mlp.dense_4h_to_h.bias", layer_prefix)),
                },
            }
        }

        /// Get layer norm weight names.
        pub fn layer_norm_weights(&self, layer_prefix: &str) -> LayerNormWeights {
            match self {
                Architecture::Llama => LayerNormWeights {
                    attention_norm_weight: format!("{}.input_layernorm.weight", layer_prefix),
                    ffn_norm_weight: format!("{}.post_attention_layernorm.weight", layer_prefix),
                    attention_norm_bias: None,
                    ffn_norm_bias: None,
                },
                Architecture::Gpt2 => LayerNormWeights {
                    attention_norm_weight: format!("{}.ln_1.weight", layer_prefix),
                    ffn_norm_weight: format!("{}.ln_2.weight", layer_prefix),
                    attention_norm_bias: Some(format!("{}.ln_1.bias", layer_prefix)),
                    ffn_norm_bias: Some(format!("{}.ln_2.bias", layer_prefix)),
                },
                Architecture::Bert => LayerNormWeights {
                    attention_norm_weight: format!("{}.attention.output.LayerNorm.weight", layer_prefix),
                    ffn_norm_weight: format!("{}.output.LayerNorm.weight", layer_prefix),
                    attention_norm_bias: Some(format!("{}.attention.output.LayerNorm.bias", layer_prefix)),
                    ffn_norm_bias: Some(format!("{}.output.LayerNorm.bias", layer_prefix)),
                },
                Architecture::Glm => LayerNormWeights {
                    attention_norm_weight: format!("{}.input_layernorm.weight", layer_prefix),
                    ffn_norm_weight: format!("{}.post_attention_layernorm.weight", layer_prefix),
                    attention_norm_bias: Some(format!("{}.input_layernorm.bias", layer_prefix)),
                    ffn_norm_bias: Some(format!("{}.post_attention_layernorm.bias", layer_prefix)),
                },
            }
        }

        /// Get final layer norm weight name.
        pub fn final_norm_weight(&self) -> &'static str {
            match self {
                Architecture::Llama => "model.norm.weight",
                Architecture::Gpt2 => "transformer.ln_f.weight",
                Architecture::Bert => "embeddings.LayerNorm.weight",
                Architecture::Glm => "transformer.encoder.final_layernorm.weight",
            }
        }

        /// Get LM head weight name.
        pub fn lm_head_weight(&self) -> &'static str {
            match self {
                Architecture::Llama => "lm_head.weight",
                Architecture::Gpt2 => "lm_head.weight",
                Architecture::Bert => "cls.predictions.decoder.weight",
                Architecture::Glm => "transformer.output_layer.weight",
            }
        }
    }

    /// Attention layer weight names.
    pub struct AttentionWeights {
        pub q_proj_weight: String,
        pub k_proj_weight: String,
        pub v_proj_weight: String,
        pub o_proj_weight: String,
        pub q_proj_bias: Option<String>,
        pub k_proj_bias: Option<String>,
        pub v_proj_bias: Option<String>,
        pub o_proj_bias: Option<String>,
    }

    /// FFN layer weight names.
    pub struct FfnWeights {
        pub gate_proj_weight: String,
        pub up_proj_weight: String,
        pub down_proj_weight: String,
        pub gate_proj_bias: Option<String>,
        pub up_proj_bias: Option<String>,
        pub down_proj_bias: Option<String>,
    }

    /// Layer normalization weight names.
    pub struct LayerNormWeights {
        pub attention_norm_weight: String,
        pub ffn_norm_weight: String,
        pub attention_norm_bias: Option<String>,
        pub ffn_norm_bias: Option<String>,
    }
}

/// Load Engram embedding table from SafeTensors.
///
/// Engram embeddings are stored as a 2D tensor [num_buckets, embedding_dim].
/// This function returns the flattened f32 data suitable for `EngramModule::from_embeddings`.
///
/// Common weight names:
/// - `model.engram.embeddings.weight` (DeepSeek-V4+)
/// - `engram.embedding_table` (alternative naming)
/// - `conditional_memory.embeddings` (alternative naming)
pub fn load_engram_embeddings(
    loader: &WeightLoader,
    config: &crate::model_config::ModelConfig,
) -> Result<Option<Vec<f32>>> {
    // Check if Engram is enabled in config
    let (_, num_buckets, embedding_dim, _) = match config.engram_config() {
        Some(cfg) => cfg,
        None => return Ok(None), // Engram not enabled
    };

    // Try common Engram weight names
    let engram_names = [
        "model.engram.embeddings.weight",
        "model.engram.embedding_table",
        "engram.embeddings.weight",
        "engram.embedding_table",
        "conditional_memory.embeddings",
        "conditional_memory.embedding_table",
    ];

    for name in engram_names {
        if loader.has_tensor(name) {
            let tensor = loader.load_tensor(name)?;

            // Verify shape matches config
            if tensor.shape.len() != 2 {
                return Err(Error::LoadError(format!(
                    "Engram tensor '{}' has wrong dimensions: expected 2D, got {}D",
                    name, tensor.shape.len()
                )));
            }

            let [loaded_buckets, loaded_dim] = [tensor.shape[0], tensor.shape[1]];
            if loaded_buckets != num_buckets {
                log::warn!(
                    "Engram bucket count mismatch: config has {}, weights have {}. Using weights.",
                    num_buckets, loaded_buckets
                );
            }
            if loaded_dim != embedding_dim {
                log::warn!(
                    "Engram embedding dim mismatch: config has {}, weights have {}. Using weights.",
                    embedding_dim, loaded_dim
                );
            }

            log::info!(
                "Loaded Engram embeddings from '{}': [{} x {}]",
                name, loaded_buckets, loaded_dim
            );

            return Ok(Some(tensor.data));
        }
    }

    // No Engram weights found - this is OK if config enables Engram but model doesn't ship with weights
    log::debug!("No Engram weights found in model file");
    Ok(None)
}

/// Multi-shard weight loading support.
pub mod shards {
    use super::*;
    use std::fs;

    /// Index for multi-shard SafeTensors files.
    #[derive(Debug)]
    pub struct ShardIndex {
        /// Mapping from tensor name to shard file.
        pub tensor_to_shard: HashMap<String, String>,
        /// List of shard files.
        pub shard_files: Vec<String>,
    }

    impl ShardIndex {
        /// Load shard index from model.safetensors.index.json.
        pub fn from_index_file(index_path: &Path) -> Result<Self> {
            let content = fs::read_to_string(index_path).map_err(|e| {
                Error::LoadError(format!("Failed to read shard index: {}", e))
            })?;

            let index: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
                Error::LoadError(format!("Failed to parse shard index: {}", e))
            })?;

            let weight_map = index
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| Error::LoadError("Missing weight_map in index".to_string()))?;

            let mut tensor_to_shard = HashMap::new();
            let mut shard_files = Vec::new();

            for (tensor_name, shard_file) in weight_map {
                let shard = shard_file
                    .as_str()
                    .ok_or_else(|| Error::LoadError("Invalid shard file name".to_string()))?
                    .to_string();

                tensor_to_shard.insert(tensor_name.clone(), shard.clone());
                if !shard_files.contains(&shard) {
                    shard_files.push(shard);
                }
            }

            Ok(Self {
                tensor_to_shard,
                shard_files,
            })
        }

        /// Check if model uses sharded weights.
        pub fn is_sharded(model_dir: &Path) -> bool {
            model_dir.join("model.safetensors.index.json").exists()
        }

        /// Get all shard file paths.
        pub fn shard_paths(&self, model_dir: &Path) -> Vec<std::path::PathBuf> {
            self.shard_files
                .iter()
                .map(|f| model_dir.join(f))
                .collect()
        }
    }

    /// Load tensor from sharded files.
    pub fn load_tensor_from_shards(
        model_dir: &Path,
        index: &ShardIndex,
        tensor_name: &str,
    ) -> Result<LoadedTensor> {
        let shard_file = index.tensor_to_shard.get(tensor_name).ok_or_else(|| {
            Error::LoadError(format!("Tensor '{}' not found in shard index", tensor_name))
        })?;

        let shard_path = model_dir.join(shard_file);
        let bytes = fs::read(&shard_path).map_err(|e| {
            Error::LoadError(format!("Failed to read shard file '{}': {}", shard_file, e))
        })?;

        let loader = WeightLoader::from_bytes(&bytes)?;
        loader.load_tensor(tensor_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_f32() {
        let data: Vec<u8> = vec![0x00, 0x00, 0x80, 0x3f]; // 1.0f32 in little-endian
        let result = convert_to_f32(&data, Dtype::F32).unwrap();
        assert_eq!(result, vec![1.0f32]);
    }

    #[test]
    fn test_tensor_transpose() {
        // 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
        let tensor = LoadedTensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: vec![2, 3],
        };

        // After transpose should be 3x2: [[1, 4], [2, 5], [3, 6]]
        // In row-major: [1, 4, 2, 5, 3, 6]
        let mut transposed = vec![0.0f32; 6];
        let [out_features, in_features] = [2, 3];
        for i in 0..in_features {
            for o in 0..out_features {
                transposed[i * out_features + o] = tensor.data[o * in_features + i];
            }
        }

        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_architecture_detection() {
        use mappings::Architecture;

        assert_eq!(
            Architecture::from_model_type("llama"),
            Architecture::Llama
        );
        assert_eq!(
            Architecture::from_model_type("Qwen2ForCausalLM"),
            Architecture::Llama
        );
        assert_eq!(
            Architecture::from_model_type("bert"),
            Architecture::Bert
        );
        assert_eq!(
            Architecture::from_model_type("chatglm"),
            Architecture::Glm
        );
    }

    #[test]
    fn test_weight_loader_from_mmap() {
        use memmap2::Mmap;
        use safetensors::tensor::{serialize, TensorView};
        use std::io::Write;
        use tempfile::NamedTempFile;

        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let tensor = TensorView::new(Dtype::F32, vec![3], &bytes).expect("tensor view");
        let safetensors = serialize([("weight", tensor)], &None).expect("serialize safetensors");

        let mut file = NamedTempFile::new().expect("tempfile");
        file.write_all(&safetensors).expect("write safetensors");

        let file_handle = std::fs::File::open(file.path()).expect("open safetensors");
        // Safety: the file is not mutated while the mmap is alive.
        let mmap = unsafe { Mmap::map(&file_handle) }.expect("mmap safetensors");
        let loader = WeightLoader::from_bytes(&mmap).expect("weight loader");
        let loaded = loader.load_tensor("weight").expect("load tensor");

        assert_eq!(loaded.shape, vec![3]);
        assert_eq!(loaded.data, data);
    }
}
