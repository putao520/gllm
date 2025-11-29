use super::common::{
    env_lock, init_test_env, is_backend_unavailable, preferred_device, prepare_context_with_weights,
};
use gllm::{Client, ClientConfig, Device, Error, Result};

#[cfg(not(feature = "tokio"))]
#[test]
fn unknown_model_returns_not_found_error() {
    let _guard = env_lock();
    init_test_env();

    match Client::new("unknown-model") {
        Ok(_) => panic!("Expected model resolution to fail"),
        Err(Error::ModelNotFound(model)) => assert_eq!(model, "unknown-model"),
        Err(err) => panic!("Unexpected error: {err}"),
    }
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn unknown_model_returns_not_found_error() {
    let _guard = env_lock();
    init_test_env();

    match Client::new("unknown-model").await {
        Ok(_) => panic!("Expected model resolution to fail"),
        Err(Error::ModelNotFound(model)) => assert_eq!(model, "unknown-model"),
        Err(err) => panic!("Unexpected error: {err}"),
    }
}

#[cfg(not(feature = "tokio"))]
#[test]
fn embeddings_reject_empty_inputs() -> Result<()> {
    let _guard = env_lock();
    init_test_env();
    let device = preferred_device();

    let (client, _ctx) = match prepare_context_with_weights("bge-small-en", device) {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let result = client.embeddings(Vec::<String>::new()).generate();
    assert!(matches!(result, Err(Error::InvalidConfig(_))));
    Ok(())
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn embeddings_reject_empty_inputs() -> Result<()> {
    let _guard = env_lock();
    init_test_env();
    let device = preferred_device();

    let (client, _ctx) = match prepare_context_with_weights("bge-small-en", device).await {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let result = client.embeddings(Vec::<String>::new()).generate().await;
    assert!(matches!(result, Err(Error::InvalidConfig(_))));
    Ok(())
}

#[cfg(not(feature = "tokio"))]
#[test]
fn rerank_rejects_empty_documents() -> Result<()> {
    let _guard = env_lock();
    init_test_env();
    let device = preferred_device();

    let (client, _ctx) = match prepare_context_with_weights("bge-reranker-v2", device) {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let result = client.rerank("query", Vec::<String>::new()).generate();
    assert!(matches!(result, Err(Error::InvalidConfig(_))));
    Ok(())
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn rerank_rejects_empty_documents() -> Result<()> {
    let _guard = env_lock();
    init_test_env();
    let device = preferred_device();

    let (client, _ctx) = match prepare_context_with_weights("bge-reranker-v2", device).await {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let result = client.rerank("query", Vec::<String>::new()).generate().await;
    assert!(matches!(result, Err(Error::InvalidConfig(_))));
    Ok(())
}

#[cfg(not(feature = "tokio"))]
#[test]
fn download_failures_surface_as_errors() {
    let _guard = env_lock();
    init_test_env();

    unsafe {
        std::env::remove_var("GLLM_SKIP_DOWNLOAD");
        std::env::set_var("HF_HUB_OFFLINE", "1");
    }

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let mut config = ClientConfig::default();
    config.models_dir = temp_dir.path().to_path_buf();
    config.device = Device::Auto;

    let result = Client::with_config("BAAI/missing-model-for-tests", config);

    unsafe {
        std::env::set_var("GLLM_SKIP_DOWNLOAD", "1");
    }

    assert!(matches!(result, Err(Error::DownloadError(_))));
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn download_failures_surface_as_errors() {
    let _guard = env_lock();
    init_test_env();

    unsafe {
        std::env::remove_var("GLLM_SKIP_DOWNLOAD");
        std::env::set_var("HF_HUB_OFFLINE", "1");
    }

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let mut config = ClientConfig::default();
    config.models_dir = temp_dir.path().to_path_buf();
    config.device = Device::Auto;

    let result = Client::with_config("BAAI/missing-model-for-tests", config).await;

    unsafe {
        std::env::set_var("GLLM_SKIP_DOWNLOAD", "1");
    }

    assert!(matches!(result, Err(Error::DownloadError(_))));
}
