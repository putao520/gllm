//! GLM-4/5 adapter.

use gllm_kernels::backend_trait::{Backend, Element};

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, ModelAdapter};

pub struct Glm5Adapter;

impl<B: Backend<E>, E: Element> ModelAdapter<B, E> for Glm5Adapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(
            manifest.arch,
            ModelArchitecture::GLM4 | ModelArchitecture::GLM5
        )
    }

    fn load_weights(
        &self,
        loader: &mut Loader,
        backend: &B,
    ) -> AdapterResult<AdapterWeights<B, E>> {
        let handle = loader.upload_weights(backend)?;
        Ok(AdapterWeights::new(handle))
    }

    fn add_special_tokens(&self) -> bool {
        false
    }
}
