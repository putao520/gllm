//! Qwen3 adapter (skeleton).

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, Message, ModelAdapter};

pub struct Qwen3Adapter;

impl<B: Backend> ModelAdapter<B> for Qwen3Adapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Qwen3)
            && !is_qwen3_special_id(manifest.model_id.as_ref())
    }

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        let thinking_tensors: Vec<String> = handle
            .meta
            .keys()
            .filter(|name| name.contains("thinking_head"))
            .cloned()
            .collect();
        Ok(AdapterWeights::with_thinking_head(handle, thinking_tensors))
    }

    fn apply_chat_template(&self, messages: &[Message]) -> String {
        const IM_START: &str = "<|im_start|>";
        const IM_END: &str = "<|im_end|>";

        let mut out = String::new();
        for message in messages {
            out.push_str(IM_START);
            out.push_str(message.role.as_str());
            out.push('\n');
            out.push_str(message.content.trim());
            out.push('\n');
            out.push_str(IM_END);
            out.push('\n');
        }
        out.push_str(IM_START);
        out.push_str("assistant\n");
        out
    }

    fn add_special_tokens(&self) -> bool {
        false
    }
}

fn is_qwen3_special_id(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    lower.contains("embed") || lower.contains("rerank")
}
