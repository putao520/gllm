//! Ministral adapter.

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, ModelAdapter};

pub struct MinistralAdapter;

impl<B: Backend> ModelAdapter<B> for MinistralAdapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Ministral)
    }

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        Ok(AdapterWeights::new(handle))
    }

}
