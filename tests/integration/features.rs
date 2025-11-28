#[cfg(feature = "async")]
use super::common::preferred_device;
#[cfg(feature = "async")]
use super::common::prepare_async_context_with_weights;
use super::common::{
    EMBEDDING_DIM, env_lock, init_test_env, is_backend_unavailable, prepare_context_with_weights,
};
use gllm::{Device, Result};

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_backend_executes_embeddings() -> Result<()> {
    let _guard = env_lock();
    init_test_env();

    let (client, _ctx) = match prepare_context_with_weights("bge-small-en", Device::Auto) {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let response = client.embeddings(["wgpu path"]).generate()?;
    assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);
    Ok(())
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_backend_executes_embeddings() -> Result<()> {
    let _guard = env_lock();
    init_test_env();

    let (client, _ctx) = match prepare_context_with_weights("bge-small-en", Device::Cpu) {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let response = client.embeddings(["cpu path"]).generate()?;
    assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "current_thread")]
async fn async_feature_combination_supported() -> Result<()> {
    let _guard = env_lock();
    init_test_env();
    let device = preferred_device();

    let (client, _ctx) = match prepare_async_context_with_weights("bge-small-en", device).await {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let response = client.embeddings(["async blend"]).generate().await?;
    assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);

    Ok(())
}

#[cfg(all(feature = "wgpu", feature = "cpu"))]
#[test]
fn multi_backend_outputs_share_shapes() -> Result<()> {
    let _guard = env_lock();
    init_test_env();

    let (cpu_client, _cpu_ctx) = prepare_context_with_weights("bge-small-en", Device::Cpu)?;
    let (auto_client, _gpu_ctx) = match prepare_context_with_weights("bge-small-en", Device::Auto) {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let cpu = cpu_client.embeddings(["multi-backend"]).generate()?;
    let gpu = auto_client.embeddings(["multi-backend"]).generate()?;

    assert_eq!(
        cpu.embeddings[0].embedding.len(),
        gpu.embeddings[0].embedding.len()
    );
    Ok(())
}
