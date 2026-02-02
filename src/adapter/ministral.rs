//! Ministral adapter.

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, Message, ModelAdapter};

pub struct MinistralAdapter;

impl<B: Backend> ModelAdapter<B> for MinistralAdapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Ministral)
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
        out.push_str("<s>");
        for message in messages {
            out.push_str(message.role.as_str());
            out.push_str(": ");
            out.push_str(message.content.trim());
            out.push('\n');
        }
        out.push_str("assistant: ");
        out
    }
}
