#[cfg(feature = "async")]
use super::common::prepare_async_context_with_weights;
use super::common::{
    EMBEDDING_DIM, env_lock, init_test_env, is_backend_unavailable, preferred_device,
    prepare_context_with_weights,
};
use gllm::Result;

#[test]
fn embeddings_sync_end_to_end() -> Result<()> {
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

    let inputs = vec!["hello world", "rust embeddings"];
    let response = client.embeddings(inputs.clone()).generate()?;

    assert_eq!(response.embeddings.len(), inputs.len());
    assert!(response.usage.prompt_tokens > 0);

    for (idx, emb) in response.embeddings.iter().enumerate() {
        assert_eq!(emb.index, idx);
        assert_eq!(emb.embedding.len(), EMBEDDING_DIM);
        assert!(emb.embedding.iter().all(|v| v.is_finite()));
    }

    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test(flavor = "current_thread")]
async fn embeddings_async_end_to_end() -> Result<()> {
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

    let response = client
        .embeddings(["async pathway", "parity check"])
        .generate()
        .await?;

    assert_eq!(response.embeddings.len(), 2);
    assert!(response.usage.prompt_tokens > 0);
    assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);

    Ok(())
}
