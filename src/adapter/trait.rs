//! Layer 2: Adapter trait (stateless).

use gllm_kernels::backend_trait::{Backend, Element, TensorLookup};
use thiserror::Error;

use crate::loader::{Loader, LoaderError, WeightsHandle};
use crate::manifest::{ModelArchitecture, ModelManifest};

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error(transparent)]
    Loader(#[from] LoaderError),
    #[error("unsupported architecture: {0:?}")]
    UnsupportedArchitecture(ModelArchitecture),
}

pub type AdapterResult<T> = std::result::Result<T, AdapterError>;

#[derive(Debug, Clone)]
pub struct ThinkingHead {
    pub tensors: Vec<String>,
}

pub struct AdapterWeights<B: Backend<E>, E: Element = f32> {
    pub handle: WeightsHandle<B, E>,
    pub thinking_head: Option<ThinkingHead>,
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

impl<B: Backend<E>, E: Element> TensorLookup<E, B> for AdapterWeights<B, E> {
    fn get_tensor(&self, name: &str) -> Option<&B::Tensor> {
        self.handle.tensor(name)
    }

    fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.handle.tensor_shape(name)
    }
}

pub trait ModelAdapter<B: Backend<E>, E: Element = f32>: Sync {
    fn supports(&self, manifest: &ModelManifest) -> bool;

    fn load_weights(&self, loader: &mut Loader, backend: &B)
        -> AdapterResult<AdapterWeights<B, E>>;

    fn add_special_tokens(&self) -> bool {
        true
    }
}

/// Backward-compatible type alias for f32 adapter weights.
pub type AdapterWeightsF32<B> = AdapterWeights<B, f32>;

/// Backward-compatible type alias for f32 model adapter.
pub trait ModelAdapterF32<B: Backend<f32>>: ModelAdapter<B, f32> {}
impl<B: Backend<f32>, T: ModelAdapter<B, f32>> ModelAdapterF32<B> for T {}
