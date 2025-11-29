use super::common::{
    EMBEDDING_DIM, env_lock, init_test_env, is_backend_unavailable, preferred_device,
    prepare_context_with_weights,
};
use gllm::{Device, Result};

// wgpu backend 测试（同步模式）
#[cfg(all(feature = "wgpu", not(feature = "tokio")))]
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

// wgpu backend 测试（异步模式）
#[cfg(all(feature = "wgpu", feature = "tokio"))]
#[tokio::test(flavor = "multi_thread")]
async fn wgpu_backend_executes_embeddings() -> Result<()> {
    let _guard = env_lock();
    init_test_env();

    let (client, _ctx) = match prepare_context_with_weights("bge-small-en", Device::Auto).await {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let response = client.embeddings(["wgpu path"]).generate().await?;
    assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);
    Ok(())
}

// cpu backend 测试（同步模式）
#[cfg(all(feature = "cpu", not(feature = "tokio")))]
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

// cpu backend 测试（异步模式）
#[cfg(all(feature = "cpu", feature = "tokio"))]
#[tokio::test(flavor = "multi_thread")]
async fn cpu_backend_executes_embeddings() -> Result<()> {
    let _guard = env_lock();
    init_test_env();

    let (client, _ctx) = match prepare_context_with_weights("bge-small-en", Device::Cpu).await {
        Ok(value) => value,
        Err(err) if is_backend_unavailable(&err) => {
            eprintln!("Skipping test: {err}");
            return Ok(());
        }
        Err(err) => return Err(err),
    };

    let response = client.embeddings(["cpu path"]).generate().await?;
    assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);
    Ok(())
}

// 多 backend 测试（同步模式）
#[cfg(all(feature = "wgpu", feature = "cpu", not(feature = "tokio")))]
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

// 多 backend 测试（异步模式）
#[cfg(all(feature = "wgpu", feature = "cpu", feature = "tokio"))]
#[tokio::test(flavor = "multi_thread")]
async fn multi_backend_outputs_share_shapes() -> Result<()> {
    let _guard = env_lock();
    init_test_env();

    let (cpu_client, _cpu_ctx) = prepare_context_with_weights("bge-small-en", Device::Cpu).await?;
    let (auto_client, _gpu_ctx) =
        match prepare_context_with_weights("bge-small-en", Device::Auto).await {
            Ok(value) => value,
            Err(err) if is_backend_unavailable(&err) => {
                eprintln!("Skipping test: {err}");
                return Ok(());
            }
            Err(err) => return Err(err),
        };

    let cpu = cpu_client.embeddings(["multi-backend"]).generate().await?;
    let gpu = auto_client.embeddings(["multi-backend"]).generate().await?;

    assert_eq!(
        cpu.embeddings[0].embedding.len(),
        gpu.embeddings[0].embedding.len()
    );
    Ok(())
}

// tokio 特性测试
#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn tokio_feature_works() -> Result<()> {
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

    let response = client.embeddings(["tokio async"]).generate().await?;
    assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);

    Ok(())
}
