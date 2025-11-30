//! Feature-specific integration tests using real models.

use gllm::{Client, ClientConfig, Device, Result};

const EMBEDDING_DIM: usize = 384;

fn get_config(device: Device) -> ClientConfig {
    ClientConfig {
        device,
        ..Default::default()
    }
}

// Sync tests (when tokio feature is disabled)
#[cfg(not(feature = "tokio"))]
mod sync_tests {
    use super::*;

    #[test]
    fn auto_backend_executes_embeddings() -> Result<()> {
        let client = Client::with_config("bge-small-en", get_config(Device::Auto))?;
        let response = client.embeddings(["auto backend"]).generate()?;
        assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);
        Ok(())
    }

    #[test]
    fn cpu_backend_executes_embeddings() -> Result<()> {
        let client = Client::with_config("bge-small-en", get_config(Device::Cpu))?;
        let response = client.embeddings(["cpu path"]).generate()?;
        assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);
        Ok(())
    }

    #[test]
    fn multi_backend_outputs_share_shapes() -> Result<()> {
        let cpu_client = Client::with_config("bge-small-en", get_config(Device::Cpu))?;
        let auto_client = Client::with_config("bge-small-en", get_config(Device::Auto))?;

        let cpu = cpu_client.embeddings(["multi-backend"]).generate()?;
        let auto = auto_client.embeddings(["multi-backend"]).generate()?;

        assert_eq!(
            cpu.embeddings[0].embedding.len(),
            auto.embeddings[0].embedding.len()
        );
        Ok(())
    }
}

// Async tests (when tokio feature is enabled)
#[cfg(feature = "tokio")]
mod async_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn tokio_feature_works() -> Result<()> {
        let client = Client::with_config("bge-small-en", get_config(Device::Auto)).await?;
        let response = client.embeddings(["tokio async"]).generate().await?;
        assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn auto_backend_executes_embeddings() -> Result<()> {
        let client = Client::with_config("bge-small-en", get_config(Device::Auto)).await?;
        let response = client.embeddings(["auto backend"]).generate().await?;
        assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cpu_backend_executes_embeddings() -> Result<()> {
        let client = Client::with_config("bge-small-en", get_config(Device::Cpu)).await?;
        let response = client.embeddings(["cpu path"]).generate().await?;
        assert_eq!(response.embeddings[0].embedding.len(), EMBEDDING_DIM);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn multi_backend_outputs_share_shapes() -> Result<()> {
        let cpu_client = Client::with_config("bge-small-en", get_config(Device::Cpu)).await?;
        let auto_client = Client::with_config("bge-small-en", get_config(Device::Auto)).await?;

        let cpu = cpu_client.embeddings(["multi-backend"]).generate().await?;
        let auto = auto_client.embeddings(["multi-backend"]).generate().await?;

        assert_eq!(
            cpu.embeddings[0].embedding.len(),
            auto.embeddings[0].embedding.len()
        );
        Ok(())
    }
}
