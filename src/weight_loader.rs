//! Weight loader facade (re-exports loader).

pub use crate::loader::{
    ChecksumPolicy, Loader, LoaderConfig, LoaderError, ParallelPolicy, TensorInfo, UploadedTensor,
    WeightsHandle,
};
