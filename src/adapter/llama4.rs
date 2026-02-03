//! Llama4 adapter (skeleton).

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, Message, ModelAdapter};

pub struct Llama4Adapter;

impl<B: Backend> ModelAdapter<B> for Llama4Adapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Llama4)
    }

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        Ok(AdapterWeights::new(handle))
    }

    fn apply_chat_template(&self, messages: &[Message]) -> String {
        const BOS: &str = "<|begin_of_text|>";
        const START_HEADER: &str = "<|start_header_id|>";
        const END_HEADER: &str = "<|end_header_id|>";
        const EOT: &str = "<|eot_id|>";

        let mut out = String::new();
        out.push_str(BOS);
        for message in messages {
            out.push_str(START_HEADER);
            out.push_str(message.role.as_str());
            out.push_str(END_HEADER);
            out.push('\n');
            out.push_str(message.content.trim());
            out.push('\n');
            out.push_str(EOT);
            out.push('\n');
        }
        out.push_str(START_HEADER);
        out.push_str("assistant");
        out.push_str(END_HEADER);
        out.push('\n');
        out
    }

    fn add_special_tokens(&self) -> bool {
        false
    }
}
