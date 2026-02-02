use std::path::Path;

use gllm::loader::Loader;
use tempfile::TempDir;

fn write_config(path: &Path) {
    let value = serde_json::json!({
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "num_hidden_layers": 1,
        "vocab_size": 4,
        "max_position_embeddings": 8,
        "torch_dtype": "float32"
    });
    std::fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
}

#[test]
fn loader_accepts_configuration_json() {
    let dir = TempDir::new().expect("temp dir");
    let weights = dir.path().join("model.safetensors");
    let config = dir.path().join("configuration.json");

    std::fs::write(&weights, []).expect("write weights");
    write_config(&config);

    let loader = Loader::from_local_files(
        "qwen3-7b",
        vec![weights],
        vec![config.clone()],
    )
    .expect("loader");

    let path = loader.config_path().expect("config path");
    assert_eq!(path.file_name().unwrap(), "configuration.json");
}
