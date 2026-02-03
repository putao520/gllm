//! Qwen3 MoE adapter.

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, ModelAdapter};

pub struct Qwen3MoEAdapter;

impl<B: Backend> ModelAdapter<B> for Qwen3MoEAdapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Qwen3MoE)
    }

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        let thinking_tensors: Vec<String> = handle
            .meta
            .keys()
            .filter(|name| name.contains("thinking_head"))
            .cloned()
            .collect();
        Ok(AdapterWeights::with_thinking_head(handle, thinking_tensors))
    }

    fn add_special_tokens(&self) -> bool {
        false
    }
}
