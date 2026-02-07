//! Mistral3 adapter.

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, ModelAdapter};

pub struct Mistral3Adapter;

impl<B: Backend> ModelAdapter<B> for Mistral3Adapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Mistral3)
    }

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        Ok(AdapterWeights::new(handle))
    }

}
