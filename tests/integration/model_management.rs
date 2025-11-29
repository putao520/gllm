use super::common::{
    EMBEDDING_DIM, env_lock, init_test_env, is_backend_unavailable, preferred_device,
    prepare_context, write_dummy_weights,
};
use gllm::{Client, Error, Result};
use safetensors::SafeTensors;
use std::fs;

#[cfg(not(feature = "tokio"))]
#[test]
fn alias_resolution_and_auto_download_creates_repo_dir() -> Result<()> {
    let _guard = env_lock();
    init_test_env();
    let device = preferred_device();

    let (client, ctx) = match prepare_context("bge-small-en", device) {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    assert!(ctx.repo_dir.ends_with("BAAI--bge-small-en-v1.5"));
    assert!(ctx.repo_dir.is_dir());
    drop(client);

    Ok(())
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn alias_resolution_and_auto_download_creates_repo_dir() -> Result<()> {
    let _guard = env_lock();
    init_test_env();
    let device = preferred_device();

    let (client, ctx) = match prepare_context("bge-small-en", device).await {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    assert!(ctx.repo_dir.ends_with("BAAI--bge-small-en-v1.5"));
    assert!(ctx.repo_dir.is_dir());
    drop(client);

    Ok(())
}

#[cfg(not(feature = "tokio"))]
#[test]
fn safetensors_weights_are_readable_and_used_in_clients() -> Result<()> {
    let _guard = env_lock();
    init_test_env();
    let device = preferred_device();

    let (_client, ctx) = match prepare_context("bge-small-en", device) {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let weights = ctx.repo_dir.join("model.safetensors");
    write_dummy_weights(&weights);

    let data = fs::read(&weights)?;
    SafeTensors::deserialize(&data).map_err(|err| Error::LoadError(err.to_string()))?;

    let client = Client::with_config("bge-small-en", ctx.config.clone())?;
    let response = client.embeddings(["warmup"]).generate()?;
    assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);

    Ok(())
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn safetensors_weights_are_readable_and_used_in_clients() -> Result<()> {
    let _guard = env_lock();
    init_test_env();
    let device = preferred_device();

    let (_client, ctx) = match prepare_context("bge-small-en", device).await {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let weights = ctx.repo_dir.join("model.safetensors");
    write_dummy_weights(&weights);

    let data = fs::read(&weights)?;
    SafeTensors::deserialize(&data).map_err(|err| Error::LoadError(err.to_string()))?;

    let client = Client::with_config("bge-small-en", ctx.config.clone()).await?;
    let response = client.embeddings(["warmup"]).generate().await?;
    assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);

    Ok(())
}
