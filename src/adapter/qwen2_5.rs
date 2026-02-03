//! Qwen2.5 adapter (skeleton).

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, Message, ModelAdapter};

pub struct Qwen2_5Adapter;

impl<B: Backend> ModelAdapter<B> for Qwen2_5Adapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Qwen2_5)
    }

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        Ok(AdapterWeights::new(handle))
    }

    fn apply_chat_template(&self, messages: &[Message]) -> String {
        // Qwen2.5 使用类似 Qwen2 的 chat template
        let mut out = String::new();
        for message in messages {
            out.push_str("<|im_start|>");
            out.push_str(message.role.as_str());
            out.push('\n');
            out.push_str(message.content.trim());
            out.push('\n');
            out.push_str("<|im_end|>");
            out.push('\n');
        }
        out.push_str("<|im_start|>assistant\n");
        out
    }

    fn add_special_tokens(&self) -> bool {
        false
    }
}
