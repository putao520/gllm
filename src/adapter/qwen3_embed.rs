//! Qwen3 embedding adapter.

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelKind, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, ModelAdapter};

pub struct Qwen3EmbedAdapter;

impl<B: Backend> ModelAdapter<B> for Qwen3EmbedAdapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        // Ω1: 使用 ModelKind 而非 Model ID 来区分用途
        matches!(manifest.arch, ModelArchitecture::Qwen3)
            && matches!(manifest.kind, ModelKind::Embedding)
    }

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        Ok(AdapterWeights::new(handle))
    }
}
