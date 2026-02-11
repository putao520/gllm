//! XLM-R adapter (embeddings / rerank).

use gllm_kernels::backend_trait::{Backend, Element};

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, ModelAdapter};

pub struct XlmRAdapter;

impl<B: Backend<E>, E: Element> ModelAdapter<B, E> for XlmRAdapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(
            manifest.arch,
            ModelArchitecture::XlmR | ModelArchitecture::XlmRNext
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
}
