//! GLM-4/5 adapter.

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, Message, ModelAdapter};

pub struct Glm5Adapter;

impl<B: Backend> ModelAdapter<B> for Glm5Adapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::GLM4 | ModelArchitecture::GLM5)
    }

    fn load_weights(
        &self,
        loader: &mut Loader,
        backend: &B,
    ) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        Ok(AdapterWeights::new(handle))
    }

    fn apply_chat_template(&self, messages: &[Message]) -> String {
        let mut out = String::new();
        for message in messages {
            out.push_str("<|");
            out.push_str(message.role.as_str());
            out.push_str("|>\n");
            out.push_str(message.content.trim());
            out.push('\n');
        }
        out.push_str("<|assistant|>\n");
        out
    }

    fn add_special_tokens(&self) -> bool {
        false
    }
}
