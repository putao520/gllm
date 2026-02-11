//! Adapter 模块桩代码 (REQ-REFACTOR-001)
//!
//! ⚠️ 过渡期代码：此模块将被 arch 模块替代
//! 当前仅保留类型定义以维持编译，功能实现将迁移到 graph 模块

use gllm_kernels::backend_trait::{Backend, BackendError, Element, TensorLookup};
use thiserror::Error;

use crate::loader::{LoaderError, WeightsHandle};

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("unsupported model architecture")]
    UnsupportedArchitecture,
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Backend(#[from] BackendError),
}

pub type AdapterResult<T> = Result<T, AdapterError>;

/// 权重句柄包装器
/// 
/// ⚠️ 过渡期：将被 OnnxGraph::initializers 替代
pub struct AdapterWeights<B: Backend<E>, E: Element> {
    pub handle: WeightsHandle<B, E>,
    pub thinking_head: Option<ThinkingHead>,
}

impl<B: Backend<E>, E: Element> std::fmt::Debug for AdapterWeights<B, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdapterWeights")
            .field("thinking_head", &self.thinking_head)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
pub struct ThinkingHead {
    pub tensors: Vec<String>,
}

impl<B: Backend<E>, E: Element> AdapterWeights<B, E> {
    pub fn new(handle: WeightsHandle<B, E>) -> Self {
        Self {
            handle,
            thinking_head: None,
        }
    }

    pub fn with_thinking_head(handle: WeightsHandle<B, E>, tensors: Vec<String>) -> Self {
        let thinking_head = if tensors.is_empty() {
            None
        } else {
            Some(ThinkingHead { tensors })
        };
        Self {
            handle,
            thinking_head,
        }
    }
}

/// 实现 gllm_kernels::TensorLookup trait
impl<B: Backend<E>, E: Element> TensorLookup<E, B> for AdapterWeights<B, E> {
    fn get_tensor(&self, name: &str) -> Option<&B::Tensor> {
        self.handle.tensor(name)
    }

    fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.handle.tensor_shape(name)
    }
}

/// 模型适配器 trait
/// 
/// ⚠️ 过渡期：将被 YAML 架构模板 + GraphOptimizer 替代
pub trait ModelAdapter<B: Backend<E>, E: Element>: Sync {
    fn supports(&self, arch: crate::manifest::ModelArchitecture) -> bool;
    fn load_weights(
        &self,
        loader: &mut crate::loader::Loader,
        backend: &B,
    ) -> AdapterResult<AdapterWeights<B, E>>;
    fn add_special_tokens(&self) -> bool {
        true
    }
}

/// F32 特化的 ModelAdapter
pub trait ModelAdapterF32<B: Backend<f32>>: ModelAdapter<B, f32> {}
impl<B: Backend<f32>, T: ModelAdapter<B, f32>> ModelAdapterF32<B> for T {}

pub type AdapterWeightsF32<B> = AdapterWeights<B, f32>;

// ========== 架构查找桩 ==========

/// 获取架构对应的适配器
/// 
/// ⚠️ 过渡期：将被 arch 模块的 YAML 模板替代
pub fn adapter_for<B: Backend<E>, E: Element>(
    _manifest: &crate::manifest::ModelManifest,
) -> Option<&'static dyn ModelAdapter<B, E>> {
    // 过渡期：返回 None，后续由 graph 模块处理
    // 完整实现需要等 arch 模块完成
    None
}
