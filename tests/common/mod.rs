use std::path::{Path, PathBuf};

use ahash::AHashMap;
use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;
use tempfile::TempDir;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

use gllm::loader::Loader;

pub struct TestModelFiles {
    _dir: TempDir,
    weights: PathBuf,
    config: PathBuf,
    tokenizer: PathBuf,
}

impl TestModelFiles {
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let dir = TempDir::new()?;
        let weights = dir.path().join("model.safetensors");
        let config = dir.path().join("config.json");
        let tokenizer = dir.path().join("tokenizer.json");

        write_config(&config)?;
        write_tokenizer(&tokenizer)?;
        write_weights(&weights)?;

        Ok(Self {
            _dir: dir,
            weights,
            config,
            tokenizer,
        })
    }

    pub fn loader(&self, alias: &str) -> Result<Loader, gllm::loader::LoaderError> {
        Loader::from_local_files(
            alias,
            vec![self.weights.clone()],
            vec![self.config.clone(), self.tokenizer.clone()],
        )
    }

}

fn write_config(path: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let value = serde_json::json!({
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 16,
        "max_position_embeddings": 16,
        "rope_theta": 10000.0,
        "torch_dtype": "float32"
    });
    std::fs::write(path, serde_json::to_vec_pretty(&value)?)?;
    Ok(())
}

fn write_tokenizer(path: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut vocab = AHashMap::new();
    vocab.insert("<unk>".to_string(), 0);
    for idx in 1..16 {
        vocab.insert(format!("tok{idx}"), idx as u32);
    }

    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("<unk>".into())
        .build()?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
    tokenizer.save(path, false)?;
    Ok(())
}

fn write_weights(path: &Path) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let hidden_size = 4usize;
    let ffn_dim = 8usize;
    let vocab_size = 16usize;

    let tensors = vec![
        (
            "model.embed_tokens.weight",
            vec![vocab_size, hidden_size],
            vec![0.01f32; vocab_size * hidden_size],
        ),
        (
            "model.layers.0.input_layernorm.weight",
            vec![hidden_size],
            vec![0.1f32; hidden_size],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![hidden_size],
            vec![0.1f32; hidden_size],
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![hidden_size, hidden_size],
            vec![0.01f32; hidden_size * hidden_size],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![hidden_size, hidden_size],
            vec![0.01f32; hidden_size * hidden_size],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            vec![hidden_size, hidden_size],
            vec![0.01f32; hidden_size * hidden_size],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![hidden_size, hidden_size],
            vec![0.01f32; hidden_size * hidden_size],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![ffn_dim, hidden_size],
            vec![0.01f32; ffn_dim * hidden_size],
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            vec![ffn_dim, hidden_size],
            vec![0.01f32; ffn_dim * hidden_size],
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![hidden_size, ffn_dim],
            vec![0.01f32; hidden_size * ffn_dim],
        ),
        (
            "model.norm.weight",
            vec![hidden_size],
            vec![0.1f32; hidden_size],
        ),
        (
            "lm_head.weight",
            vec![vocab_size, hidden_size],
            vec![0.01f32; vocab_size * hidden_size],
        ),
        (
            "score.weight",
            vec![1, hidden_size],
            vec![0.01f32; hidden_size],
        ),
    ];

    let mut tensor_data = Vec::with_capacity(tensors.len());
    for (name, shape, data) in tensors {
        tensor_data.push((name.to_string(), shape, f32_to_bytes(&data)));
    }

    let mut views = Vec::with_capacity(tensor_data.len());
    for (name, shape, bytes) in &tensor_data {
        let view = TensorView::new(Dtype::F32, shape.clone(), bytes)?;
        views.push((name.clone(), view));
    }

    serialize_to_file(views, &None, path)?;
    Ok(())
}

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for value in data {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}
