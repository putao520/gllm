mod reader;
mod slice;
mod types;

pub use reader::{GgufReader, TensorInfo};
pub use slice::TensorSlice;
pub use types::{
    tensor_nbytes, GgmlDType, GgufArray, GgufError, GgufValue, GgufValueType, GGUF_MAGIC,
    GGUF_SUPPORTED_VERSION,
};

pub type GgufLoader = GgufReader;
