//! Layer 2: Model adapters (skeleton).

use gllm_kernels::backend_trait::Backend;

use crate::manifest::ModelManifest;

pub mod gemma2;
pub mod glm5;
pub mod gpt_oss;
pub mod llama4;
pub mod ministral;
pub mod mistral3;
pub mod phi4;
pub mod qwen2_5;
pub mod qwen3;
pub mod qwen3_embed;
pub mod qwen3_moe;
pub mod qwen3_rerank;
pub mod r#trait;
pub mod xlm_r;

pub use gemma2::Gemma2Adapter;
pub use glm5::Glm5Adapter;
pub use gpt_oss::GptOssAdapter;
pub use llama4::Llama4Adapter;
pub use ministral::MinistralAdapter;
pub use mistral3::Mistral3Adapter;
pub use phi4::Phi4Adapter;
pub use qwen2_5::Qwen2_5Adapter;
pub use qwen3::Qwen3Adapter;
pub use qwen3_embed::Qwen3EmbedAdapter;
pub use qwen3_moe::Qwen3MoEAdapter;
pub use qwen3_rerank::Qwen3RerankAdapter;
pub use r#trait::{
    AdapterError, AdapterResult, AdapterWeights, Message, ModelAdapter, Role, ThinkingHead,
};
pub use xlm_r::XlmRAdapter;

static QWEN2_5: Qwen2_5Adapter = Qwen2_5Adapter;
static QWEN3: Qwen3Adapter = Qwen3Adapter;
static QWEN3_EMBED: Qwen3EmbedAdapter = Qwen3EmbedAdapter;
static QWEN3_RERANK: Qwen3RerankAdapter = Qwen3RerankAdapter;
static QWEN3_MOE: Qwen3MoEAdapter = Qwen3MoEAdapter;
static LLAMA4: Llama4Adapter = Llama4Adapter;
static GEMMA2: Gemma2Adapter = Gemma2Adapter;
static PHI4: Phi4Adapter = Phi4Adapter;
static MINISTRAL: MinistralAdapter = MinistralAdapter;
static MISTRAL3: Mistral3Adapter = Mistral3Adapter;
static XLMR: XlmRAdapter = XlmRAdapter;
static GPT_OSS: GptOssAdapter = GptOssAdapter;
static GLM5: Glm5Adapter = Glm5Adapter;

pub fn adapter_for<B: Backend>(manifest: &ModelManifest) -> Option<&'static dyn ModelAdapter<B>> {
    if <Qwen3EmbedAdapter as ModelAdapter<B>>::supports(&QWEN3_EMBED, manifest) {
        return Some(&QWEN3_EMBED);
    }
    if <Qwen3RerankAdapter as ModelAdapter<B>>::supports(&QWEN3_RERANK, manifest) {
        return Some(&QWEN3_RERANK);
    }
    if <XlmRAdapter as ModelAdapter<B>>::supports(&XLMR, manifest) {
        return Some(&XLMR);
    }
    if <Qwen3MoEAdapter as ModelAdapter<B>>::supports(&QWEN3_MOE, manifest) {
        return Some(&QWEN3_MOE);
    }
    if <Qwen3Adapter as ModelAdapter<B>>::supports(&QWEN3, manifest) {
        return Some(&QWEN3);
    }
    if <Qwen2_5Adapter as ModelAdapter<B>>::supports(&QWEN2_5, manifest) {
        return Some(&QWEN2_5);
    }
    if <Llama4Adapter as ModelAdapter<B>>::supports(&LLAMA4, manifest) {
        return Some(&LLAMA4);
    }
    if <Gemma2Adapter as ModelAdapter<B>>::supports(&GEMMA2, manifest) {
        return Some(&GEMMA2);
    }
    if <Phi4Adapter as ModelAdapter<B>>::supports(&PHI4, manifest) {
        return Some(&PHI4);
    }
    if <MinistralAdapter as ModelAdapter<B>>::supports(&MINISTRAL, manifest) {
        return Some(&MINISTRAL);
    }
    if <Mistral3Adapter as ModelAdapter<B>>::supports(&MISTRAL3, manifest) {
        return Some(&MISTRAL3);
    }
    if <Glm5Adapter as ModelAdapter<B>>::supports(&GLM5, manifest) {
        return Some(&GLM5);
    }
    if <GptOssAdapter as ModelAdapter<B>>::supports(&GPT_OSS, manifest) {
        return Some(&GPT_OSS);
    }
    None
}
