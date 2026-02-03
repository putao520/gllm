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

/// 真实模型测试文件 (使用 ~/.gllm/models 缓存)
pub struct RealModelFiles {
    cache_dir: PathBuf,
}

impl RealModelFiles {
    /// 将缓存目录名 (使用 --) 转换为 HuggingFace repo 格式 (使用 /)
    /// 例如: "microsoft--phi-4-mini-instruct" -> "microsoft/phi-4-mini-instruct"
    fn cache_name_to_repo_id(cache_name: &str) -> String {
        cache_name.replace("--", "/")
    }

    /// 将 HuggingFace repo 格式转换为缓存目录名
    /// 例如: "microsoft/phi-4-mini-instruct" -> "microsoft--phi-4-mini-instruct"
    fn repo_id_to_cache_name(repo_id: &str) -> String {
        repo_id.replace('/', "--")
    }

    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let home = std::env::var("HOME").map_err(|_| {
            Box::from("无法获取 HOME 目录") as Box<dyn std::error::Error + Send + Sync>
        })?;
        let cache_dir = PathBuf::from(home).join(".gllm/models");

        if !cache_dir.exists() {
            return Err(format!("模型缓存目录不存在: {}", cache_dir.display()).into());
        }

        Ok(Self { cache_dir })
    }

    /// 检查模型是否存在
    pub fn model_exists(&self, repo_id: &str) -> bool {
        // 首先检查旧格式（直接目录）
        let old_path = self.cache_dir.join(repo_id);
        if old_path.exists() && old_path.is_dir() {
            return true;
        }

        // 检查新格式（models--前缀）
        let new_repo_id = format!("models--{}", repo_id.replace('/', "--"));
        let new_path = self.cache_dir.join(&new_repo_id);
        if !new_path.exists() {
            return false;
        }

        // 检查是否有有效的 snapshot
        if let Ok(snapshot_dirs) = std::fs::read_dir(&new_path) {
            for snapshot_dir in snapshot_dirs.flatten() {
                let snapshot_path = snapshot_dir.path();
                if snapshot_path.join("model.safetensors").exists()
                    || snapshot_path.join("pytorch_model.bin").exists()
                {
                    return true;
                }
            }
        }

        false
    }

    /// 获取模型目录（返回实际包含模型文件的路径）
    pub fn model_path(&self, repo_id: &str) -> PathBuf {
        // 首先检查旧格式
        let old_path = self.cache_dir.join(repo_id);
        if old_path.exists() {
            return old_path;
        }

        // 检查新格式
        let new_repo_id = format!("models--{}", repo_id.replace('/', "--"));
        let new_path = self.cache_dir.join(&new_repo_id);
        if new_path.exists() {
            // 找到有效的 snapshot 目录
            if let Ok(snapshot_dirs) = std::fs::read_dir(&new_path) {
                for snapshot_dir in snapshot_dirs.flatten() {
                    let snapshot_path = snapshot_dir.path();
                    if snapshot_path.join("model.safetensors").exists()
                        || snapshot_path.join("pytorch_model.bin").exists()
                    {
                        return snapshot_path;
                    }
                }
            }
        }

        // 未找到，返回新格式路径（会报错但这是正确的错误）
        self.cache_dir.join(&new_repo_id)
    }

    /// 创建使用真实模型的 Loader
    pub fn loader(&self, repo_id: &str) -> Result<Loader, gllm::loader::LoaderError> {
        self.loader_with_manifest(repo_id, None)
    }

    /// 创建使用真实模型的 Loader（使用指定的 manifest）
    pub fn loader_with_manifest(
        &self,
        repo_id: &str,
        manifest: Option<&'static gllm::manifest::ModelManifest>,
    ) -> Result<Loader, gllm::loader::LoaderError> {
        // 首先检查旧格式
        let old_path = self.cache_dir.join(repo_id);
        let old_weights = old_path.join("model.safetensors");
        if old_weights.exists() {
            return self.loader_from_path_with_manifest(repo_id, &old_path, manifest);
        }

        // 检查新格式 (models--前缀)
        let new_repo_id = format!("models--{}", repo_id.replace('/', "--"));
        let new_path = self.cache_dir.join(&new_repo_id);
        if !new_path.exists() {
            return Err(gllm::loader::LoaderError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("模型目录不存在: {}", new_path.display()),
            )));
        }

        // 找到有效的 snapshot 目录
        let snapshot_path = self.find_snapshot_path(&new_path)?;

        // 检查权重文件
        let weights = snapshot_path.join("model.safetensors");
        if weights.exists() {
            return self.loader_from_path_with_manifest(repo_id, &snapshot_path, manifest);
        }

        // 检查 sharded weights
        let index_path = snapshot_path.join("model.safetensors.index.json");
        if index_path.exists() {
            return self.loader_from_path_with_manifest(repo_id, &snapshot_path, manifest);
        }

        // 检查 pytorch bin
        let pytorch_path = snapshot_path.join("pytorch_model.bin");
        if pytorch_path.exists() {
            return self.loader_from_path_with_manifest(repo_id, &snapshot_path, manifest);
        }

        Err(gllm::loader::LoaderError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("权重文件不存在于: {}", snapshot_path.display()),
        )))
    }

    /// 从指定路径创建 Loader
    fn loader_from_path(
        &self,
        repo_id: &str,
        model_path: &Path,
    ) -> Result<Loader, gllm::loader::LoaderError> {
        self.loader_from_path_with_manifest(repo_id, model_path, None)
    }

    /// 从指定路径创建 Loader（使用指定的 manifest）
    fn loader_from_path_with_manifest(
        &self,
        repo_id: &str,
        model_path: &Path,
        manifest: Option<&'static gllm::manifest::ModelManifest>,
    ) -> Result<Loader, gllm::loader::LoaderError> {
        // 查找 config
        let config = model_path.join("config.json");
        let config_alt = model_path.join("configuration.json");
        let config_path = if config.exists() {
            config
        } else if config_alt.exists() {
            config_alt
        } else {
            return Err(gllm::loader::LoaderError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "配置文件不存在: {} 或 {}",
                    config.display(),
                    config_alt.display()
                ),
            )));
        };

        // 查找 tokenizer
        let tokenizer = model_path.join("tokenizer.json");
        let tokenizer_config = model_path.join("tokenizer_config.json");
        let tokenizer_path = if tokenizer.exists() {
            tokenizer
        } else if tokenizer_config.exists() {
            tokenizer_config
        } else {
            return Err(gllm::loader::LoaderError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "tokenizer 文件不存在: {} 或 {}",
                    tokenizer.display(),
                    tokenizer_config.display()
                ),
            )));
        };

        // 查找权重文件
        let weights = model_path.join("model.safetensors");
        let index_path = model_path.join("model.safetensors.index.json");
        let pytorch_path = model_path.join("pytorch_model.bin");

        let weight_files = if weights.exists() {
            vec![weights]
        } else if index_path.exists() {
            // sharded safetensors - 扫描所有 model-*.safetensors 文件
            let mut shards = Vec::new();
            if let Ok(entries) = std::fs::read_dir(model_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with("model-") && name.ends_with(".safetensors") {
                            shards.push(path);
                        }
                    }
                }
            }
            // 按文件名排序确保分片顺序正确
            shards.sort();
            if shards.is_empty() {
                return Err(gllm::loader::LoaderError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("分片权重文件不存在于: {}", model_path.display()),
                )));
            }
            shards
        } else if pytorch_path.exists() {
            vec![pytorch_path]
        } else {
            return Err(gllm::loader::LoaderError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("权重文件不存在: {}", model_path.display()),
            )));
        };

        Loader::from_local_files_with_manifest(
            repo_id,
            weight_files,
            vec![config_path, tokenizer_path],
            manifest,
        )
    }

    /// 查找有效的 snapshot 路径
    fn find_snapshot_path(&self, model_dir: &Path) -> Result<PathBuf, gllm::loader::LoaderError> {
        let snapshots_path = model_dir.join("snapshots");
        if let Ok(snapshot_dirs) = std::fs::read_dir(&snapshots_path) {
            for snapshot_dir in snapshot_dirs.flatten() {
                let snapshot_path = snapshot_dir.path();
                // 检查是否有模型文件
                if snapshot_path.join("model.safetensors").exists()
                    || snapshot_path.join("pytorch_model.bin").exists()
                    || snapshot_path.join("model.safetensors.index.json").exists()
                {
                    return Ok(snapshot_path);
                }
            }
        }
        Err(gllm::loader::LoaderError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("未找到有效的 snapshot: {}", model_dir.display()),
        )))
    }

    /// 列出所有可用的模型
    pub fn list_models(&self) -> Vec<String> {
        let mut models = Vec::new();

        let entries = std::fs::read_dir(&self.cache_dir);
        if let Ok(entries) = entries {
            for entry in entries.flatten() {
                if let Ok(name) = entry.file_name().into_string() {
                    // 跳过非模型目录
                    if name.starts_with('.') || name == "cache" || name == "model" {
                        continue;
                    }

                    // 检查是否为 hf-hub 新格式 (models--org--model-name)
                    // 如果是，则扫描其中的 snapshots 子目录
                    let entry_path = entry.path();
                    if name.starts_with("models--") {
                        // 新格式: models--org--model-name/snapshots/<hash>/
                        let snapshots_path = entry_path.join("snapshots");
                        if let Ok(snapshot_dirs) = std::fs::read_dir(&snapshots_path) {
                            for snapshot_dir in snapshot_dirs.flatten() {
                                // snapshot_name 是 hash，跳过
                                // 检查是否有模型文件
                                let snapshot_path = snapshot_dir.path();
                                let has_weights = snapshot_path.join("model.safetensors").exists()
                                    || snapshot_path.join("pytorch_model.bin").exists()
                                    || snapshot_path.join("model.safetensors.index.json").exists();

                                if has_weights {
                                    // 返回不带 models-- 前缀的名字
                                    models.push(
                                        name.strip_prefix("models--").unwrap_or(&name).to_string(),
                                    );
                                    break; // 找到一个有效的 snapshot 就停止
                                }
                            }
                        }
                    } else {
                        // 旧格式: 直接是模型目录
                        models.push(name);
                    }
                }
            }
        }

        models.sort();
        models
    }
}
