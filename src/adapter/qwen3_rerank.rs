//! Qwen3 rerank adapter.

use gllm_kernels::backend_trait::{Backend, Element};

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelKind, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, ModelAdapter};

pub struct Qwen3RerankAdapter;

impl<B: Backend<E>, E: Element> ModelAdapter<B, E> for Qwen3RerankAdapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        // Ω1: 使用 ModelKind 而非 Model ID 来区分用途
        matches!(manifest.arch, ModelArchitecture::Qwen3)
            && matches!(manifest.kind, ModelKind::Reranker)
    }

    fn load_weights(
        &self,
        loader: &mut Loader,
        backend: &B,
    ) -> AdapterResult<AdapterWeights<B, E>> {
        let handle = loader.upload_weights(backend)?;
        Ok(AdapterWeights::new(handle))
    }
}
