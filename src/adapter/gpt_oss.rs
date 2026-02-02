//! GPT-OSS adapter (skeleton).

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, ModelAdapter};

pub struct GptOssAdapter;

impl<B: Backend> ModelAdapter<B> for GptOssAdapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::GPT2Next)
    }

    fn load_weights(
        &self,
        loader: &mut Loader,
        backend: &B,
    ) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        Ok(AdapterWeights::new(handle))
    }
}
