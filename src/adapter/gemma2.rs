//! Gemma2 adapter.

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, Message, ModelAdapter};

pub struct Gemma2Adapter;

impl<B: Backend> ModelAdapter<B> for Gemma2Adapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Gemma2)
    }

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        Ok(AdapterWeights::new(handle))
    }

    fn apply_chat_template(&self, messages: &[Message]) -> String {
        let mut out = String::new();
        for message in messages {
            out.push_str("<start_of_turn>");
            out.push_str(message.role.as_str());
            out.push('\n');
            out.push_str(message.content.trim());
            out.push_str("<end_of_turn>\n");
        }
        out.push_str("<start_of_turn>model\n");
        out
    }

    fn add_special_tokens(&self) -> bool {
        false
    }
}
