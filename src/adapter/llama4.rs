//! Llama4 adapter (skeleton).

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, ModelAdapter};

pub struct Llama4Adapter;

impl<B: Backend> ModelAdapter<B> for Llama4Adapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Llama4)
    }

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        Ok(AdapterWeights::new(handle))
    }

    fn add_special_tokens(&self) -> bool {
        false
    }
}
