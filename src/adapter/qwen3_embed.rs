//! Qwen3 embedding adapter.

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, ModelAdapter};

pub struct Qwen3EmbedAdapter;

impl<B: Backend> ModelAdapter<B> for Qwen3EmbedAdapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Qwen3)
            && is_qwen3_embed_id(manifest.model_id.as_ref())
    }

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        Ok(AdapterWeights::new(handle))
    }
}

fn is_qwen3_embed_id(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    lower.contains("embed")
}
