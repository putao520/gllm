//! Gemma2 adapter.

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, ModelAdapter};

pub struct Gemma2Adapter;

impl<B: Backend> ModelAdapter<B> for Gemma2Adapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Gemma2)
    }

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        Ok(AdapterWeights::new(handle))
    }

    fn add_special_tokens(&self) -> bool {
        false
    }
}
