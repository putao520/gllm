//! Phi-4 adapter.

use gllm_kernels::backend_trait::{Backend, Element};

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, ModelAdapter};

pub struct Phi4Adapter;

impl<B: Backend<E>, E: Element> ModelAdapter<B, E> for Phi4Adapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Phi4)
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
