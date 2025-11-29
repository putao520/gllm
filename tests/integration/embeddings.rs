use super::common::{
    EMBEDDING_DIM, env_lock, init_test_env, is_backend_unavailable, preferred_device,
    prepare_context_with_weights,
};
use gllm::Result;

#[cfg(not(feature = "tokio"))]
#[test]
fn embeddings_end_to_end() -> Result<()> {
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

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn embeddings_end_to_end() -> Result<()> {
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

    let inputs = vec!["hello world", "rust embeddings"];
    let response = client.embeddings(inputs.clone()).generate().await?;

    assert_eq!(response.embeddings.len(), inputs.len());
    assert!(response.usage.prompt_tokens > 0);

    for (idx, emb) in response.embeddings.iter().enumerate() {
        assert_eq!(emb.index, idx);
        assert_eq!(emb.embedding.len(), EMBEDDING_DIM);
        assert!(emb.embedding.iter().all(|v| v.is_finite()));
    }

    Ok(())
}
