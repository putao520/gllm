//! Layer 2: Adapter trait (stateless).

use gllm_kernels::backend_trait::{Backend, TensorLookup};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThinkingHead {
    pub tensors: Vec<String>,
}

pub struct AdapterWeights<B: Backend> {
    pub handle: WeightsHandle<B>,
    pub thinking_head: Option<ThinkingHead>,
}

impl<B: Backend> AdapterWeights<B> {
    pub fn new(handle: WeightsHandle<B>) -> Self {
        Self {
            handle,
            thinking_head: None,
        }
    }

    pub fn with_thinking_head(handle: WeightsHandle<B>, tensors: Vec<String>) -> Self {
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

impl<B: Backend> TensorLookup<B> for AdapterWeights<B> {
    fn tensor_f32(&self, name: &str) -> Option<&B::Tensor<f32>> {
        self.handle.tensor_f32(name)
    }

    fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.handle.tensor_shape(name)
    }
}

pub trait ModelAdapter<B: Backend> {
    fn supports(&self, manifest: &ModelManifest) -> bool;

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>>;

    fn add_special_tokens(&self) -> bool {
        true
    }
}
