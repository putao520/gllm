//! Error handling integration tests using real models.

use gllm::{Client, ClientConfig, Device, Error, Result};

fn get_config() -> ClientConfig {
    ClientConfig {
        device: Device::Auto,
        ..Default::default()
    }
}

#[cfg(not(feature = "tokio"))]
#[test]
fn unknown_model_returns_not_found_error() {
    match Client::new("unknown-model") {
        Ok(_) => panic!("Expected model resolution to fail"),
        Err(Error::ModelNotFound(model)) => assert_eq!(model, "unknown-model"),
        Err(err) => panic!("Unexpected error: {err}"),
    }
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn unknown_model_returns_not_found_error() {
    match Client::new("unknown-model").await {
        Ok(_) => panic!("Expected model resolution to fail"),
        Err(Error::ModelNotFound(model)) => assert_eq!(model, "unknown-model"),
        Err(err) => panic!("Unexpected error: {err}"),
    }
}

#[cfg(not(feature = "tokio"))]
#[test]
fn embeddings_reject_empty_inputs() -> Result<()> {
    let client = Client::with_config("bge-small-en", get_config())?;
    let result = client.embeddings(Vec::<String>::new()).generate();
    assert!(matches!(result, Err(Error::InvalidConfig(_))));
    Ok(())
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn embeddings_reject_empty_inputs() -> Result<()> {
    let client = Client::with_config("bge-small-en", get_config()).await?;
    let result = client.embeddings(Vec::<String>::new()).generate().await;
    assert!(matches!(result, Err(Error::InvalidConfig(_))));
    Ok(())
}

#[cfg(not(feature = "tokio"))]
#[test]
fn rerank_rejects_empty_documents() -> Result<()> {
    let client = Client::with_config("bge-reranker-v2", get_config())?;
    let result = client.rerank("query", Vec::<String>::new()).generate();
    assert!(matches!(result, Err(Error::InvalidConfig(_))));
    Ok(())
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn rerank_rejects_empty_documents() -> Result<()> {
    let client = Client::with_config("bge-reranker-v2", get_config()).await?;
    let result = client
        .rerank("query", Vec::<String>::new())
        .generate()
        .await;
    assert!(matches!(result, Err(Error::InvalidConfig(_))));
    Ok(())
}

#[cfg(not(feature = "tokio"))]
#[test]
fn download_failures_surface_as_errors() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let config = ClientConfig {
        models_dir: temp_dir.path().to_path_buf(),
        device: Device::Auto,
    };

    // Use offline mode to force download failure
    unsafe {
        std::env::set_var("HF_HUB_OFFLINE", "1");
    }

    let result = Client::with_config("BAAI/missing-model-for-tests", config);

    unsafe {
        std::env::remove_var("HF_HUB_OFFLINE");
    }

    assert!(matches!(result, Err(Error::DownloadError(_))));
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn download_failures_surface_as_errors() {
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let config = ClientConfig {
        models_dir: temp_dir.path().to_path_buf(),
        device: Device::Auto,
    };

    // Use offline mode to force download failure
    unsafe {
        std::env::set_var("HF_HUB_OFFLINE", "1");
    }

    let result = Client::with_config("BAAI/missing-model-for-tests", config).await;

    unsafe {
        std::env::remove_var("HF_HUB_OFFLINE");
    }

    assert!(matches!(result, Err(Error::DownloadError(_))));
}
