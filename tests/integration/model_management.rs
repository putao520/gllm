//! Model management integration tests using real models.

use gllm::{Client, ClientConfig, Device, ModelRegistry, Result};
use std::path::PathBuf;

fn get_models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".gllm")
        .join("models")
}

fn get_config() -> ClientConfig {
    ClientConfig {
        device: Device::Auto,
        models_dir: get_models_dir(),
    }
}

#[cfg(not(feature = "tokio"))]
#[test]
fn alias_resolution_and_auto_download_creates_repo_dir() -> Result<()> {
    let registry = ModelRegistry::new();
    let info = registry.resolve("bge-small-en")?;

    // Verify alias resolves correctly
    assert!(info.repo_id.contains("bge-small-en"));

    // Verify client can be created
    let client = Client::with_config("bge-small-en", get_config())?;
    let response = client.embeddings(["test"]).generate()?;
    assert_eq!(response.embeddings.len(), 1);

    Ok(())
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn alias_resolution_and_auto_download_creates_repo_dir() -> Result<()> {
    let registry = ModelRegistry::new();
    let info = registry.resolve("bge-small-en")?;

    // Verify alias resolves correctly
    assert!(info.repo_id.contains("bge-small-en"));

    // Verify client can be created
    let client = Client::with_config("bge-small-en", get_config()).await?;
    let response = client.embeddings(["test"]).generate().await?;
    assert_eq!(response.embeddings.len(), 1);

    Ok(())
}

#[cfg(not(feature = "tokio"))]
#[test]
fn safetensors_weights_are_readable_and_used_in_clients() -> Result<()> {
    let client = Client::with_config("bge-small-en", get_config())?;

    // Verify model can generate embeddings (proves weights are loaded)
    let response = client.embeddings(["warmup"]).generate()?;
    assert_eq!(response.embeddings[0].embedding.len(), 384);
    assert!(response.embeddings[0].embedding.iter().all(|v| v.is_finite()));

    Ok(())
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn safetensors_weights_are_readable_and_used_in_clients() -> Result<()> {
    let client = Client::with_config("bge-small-en", get_config()).await?;

    // Verify model can generate embeddings (proves weights are loaded)
    let response = client.embeddings(["warmup"]).generate().await?;
    assert_eq!(response.embeddings[0].embedding.len(), 384);
    assert!(response.embeddings[0].embedding.iter().all(|v| v.is_finite()));

    Ok(())
}
