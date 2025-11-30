//! Embeddings integration tests using real models.

use gllm::{Client, ClientConfig, Device, Result};

fn get_config() -> ClientConfig {
    ClientConfig {
        device: Device::Auto,
        ..Default::default()
    }
}

#[cfg(not(feature = "tokio"))]
#[test]
fn embeddings_end_to_end() -> Result<()> {
    let client = Client::with_config("bge-small-en", get_config())?;

    let inputs = vec!["hello world", "rust embeddings"];
    let response = client.embeddings(inputs.clone()).generate()?;

    assert_eq!(response.embeddings.len(), inputs.len());
    assert!(response.usage.prompt_tokens > 0);

    for (idx, emb) in response.embeddings.iter().enumerate() {
        assert_eq!(emb.index, idx);
        assert_eq!(emb.embedding.len(), 384); // bge-small-en has 384 dims
        assert!(emb.embedding.iter().all(|v| v.is_finite()));
    }

    Ok(())
}

#[cfg(feature = "tokio")]
#[tokio::test(flavor = "multi_thread")]
async fn embeddings_end_to_end() -> Result<()> {
    let client = Client::with_config("bge-small-en", get_config()).await?;

    let inputs = vec!["hello world", "rust embeddings"];
    let response = client.embeddings(inputs.clone()).generate().await?;

    assert_eq!(response.embeddings.len(), inputs.len());
    assert!(response.usage.prompt_tokens > 0);

    for (idx, emb) in response.embeddings.iter().enumerate() {
        assert_eq!(emb.index, idx);
        assert_eq!(emb.embedding.len(), 384); // bge-small-en has 384 dims
        assert!(emb.embedding.iter().all(|v| v.is_finite()));
    }

    Ok(())
}
