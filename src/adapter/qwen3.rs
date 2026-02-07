//! Qwen3 adapter (skeleton).

use gllm_kernels::backend_trait::Backend;

use crate::loader::Loader;
use crate::manifest::{ModelArchitecture, ModelKind, ModelManifest};

use super::r#trait::{AdapterResult, AdapterWeights, ModelAdapter};

/// Ω1: Qwen3 架构特有的 thinking head 张量名称模式
///
/// 这些名称模式来自 Qwen3 模型的架构定义，而非硬编码推测。
/// 如果模型架构变更，这些模式也需要相应更新。
const THINKING_HEAD_PATTERNS: &[&str] = &["thinking_head"];

pub struct Qwen3Adapter;

impl<B: Backend> ModelAdapter<B> for Qwen3Adapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        // Ω1: 使用 ModelKind 而非 Model ID 来区分用途
        matches!(manifest.arch, ModelArchitecture::Qwen3)
            && matches!(manifest.kind, ModelKind::Chat)
    }

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;

        // Ω1: 使用架构定义的张量名称模式检测 thinking head
        let thinking_tensors: Vec<String> = handle
            .meta
            .keys()
            .filter(|name| {
                THINKING_HEAD_PATTERNS
                    .iter()
                    .any(|pattern| name.contains(pattern))
            })
            .cloned()
            .collect();

        Ok(AdapterWeights::with_thinking_head(handle, thinking_tensors))
    }

    fn add_special_tokens(&self) -> bool {
        false
    }
}
