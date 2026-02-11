use gllm_kernels::backend_trait::{BackendError, Element};
use gllm_kernels::{CpuBackend, CudaBackend};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Cuda,
    Rocm,
    Metal,
    Cpu,
}

/// Detected backend with element type parameter.
#[derive(Debug)]
pub enum DetectedBackend<E: Element = f32> {
    Cuda(Box<CudaBackend<E>>),
    Cpu(Box<CpuBackend<E>>),
}

impl<E: Element> DetectedBackend<E> {
    pub fn backend_type(&self) -> BackendType {
        match self {
            DetectedBackend::Cuda(_) => BackendType::Cuda,
            DetectedBackend::Cpu(_) => BackendType::Cpu,
        }
    }
}

/// Backward-compatible type alias for f32 backend.
pub type DetectedBackendF32 = DetectedBackend<f32>;

#[derive(Debug, Clone, Copy)]
struct BackendAvailability {
    cuda: bool,
    rocm: bool,
    metal: bool,
}

/// Detect the best available backend for f32 (default element type).
///
/// This is the primary entry point for backend detection, supporting both
/// CUDA and CPU backends.
pub fn detect_backend() -> Result<DetectedBackend<f32>, BackendError> {
    detect_f32()
}

/// Detect backend for f32 element type.
fn detect_f32() -> Result<DetectedBackend<f32>, BackendError> {
    let cuda_probe = CudaBackend::<f32>::new(0).ok();
    let availability = BackendAvailability {
        cuda: cuda_probe.is_some(),
        rocm: rocm_available(),
        metal: metal_available(),
    };
    let selected = select_backend_type(availability);
    match selected {
        BackendType::Cuda => Ok(DetectedBackend::Cuda(Box::new(
            cuda_probe.expect("cuda availability checked"),
        ))),
        BackendType::Rocm => Err(BackendError::Unimplemented("rocm backend")),
        BackendType::Metal => Err(BackendError::Unimplemented("metal backend")),
        BackendType::Cpu => Ok(DetectedBackend::Cpu(Box::new(CpuBackend::<f32>::new()))),
    }
}

/// Detect backend for a generic element type.
///
/// Supports both CUDA and CPU backends for all Element types.
/// CPU backend uses pseudo-SIMD (f32 promotion) for f16/bf16 types.
pub fn detect_backend_generic<E: Element>() -> Result<DetectedBackend<E>, BackendError> {
    // Try CUDA first (preferred for performance)
    let cuda_probe = CudaBackend::<E>::new(0).ok();
    if let Some(backend) = cuda_probe {
        return Ok(DetectedBackend::Cuda(Box::new(backend)));
    }
    // Fallback to CPU (now supports all Element types via SimdFloat)
    Ok(DetectedBackend::Cpu(Box::new(CpuBackend::<E>::new())))
}

/// Detected dtype from model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectedDtype {
    F32,
    F16,
    BF16,
}

impl DetectedDtype {
    /// Infer dtype from size in bytes.
    pub fn from_size(size: usize) -> Option<Self> {
        match size {
            4 => Some(DetectedDtype::F32),
            2 => Some(DetectedDtype::F16), // Could be BF16, but F16 is more common
            _ => None,
        }
    }

    /// Infer dtype from safetensors Dtype.
    pub fn from_safetensors_dtype(dtype: ::safetensors::Dtype) -> Option<Self> {
        match dtype {
            ::safetensors::Dtype::F32 => Some(DetectedDtype::F32),
            ::safetensors::Dtype::F16 => Some(DetectedDtype::F16),
            ::safetensors::Dtype::BF16 => Some(DetectedDtype::BF16),
            _ => None,
        }
    }
}

fn select_backend_type(availability: BackendAvailability) -> BackendType {
    if availability.cuda {
        BackendType::Cuda
    } else if availability.rocm {
        BackendType::Rocm
    } else if availability.metal {
        BackendType::Metal
    } else {
        BackendType::Cpu
    }
}

fn rocm_available() -> bool {
    false
}

fn metal_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_backend_prefers_cuda() {
        let availability = BackendAvailability {
            cuda: true,
            rocm: true,
            metal: true,
        };
        assert_eq!(select_backend_type(availability), BackendType::Cuda);
    }

    #[test]
    fn select_backend_prefers_rocm_over_metal() {
        let availability = BackendAvailability {
            cuda: false,
            rocm: true,
            metal: true,
        };
        assert_eq!(select_backend_type(availability), BackendType::Rocm);
    }

    #[test]
    fn select_backend_prefers_metal_over_cpu() {
        let availability = BackendAvailability {
            cuda: false,
            rocm: false,
            metal: true,
        };
        assert_eq!(select_backend_type(availability), BackendType::Metal);
    }

    #[test]
    fn select_backend_falls_back_to_cpu() {
        let availability = BackendAvailability {
            cuda: false,
            rocm: false,
            metal: false,
        };
        assert_eq!(select_backend_type(availability), BackendType::Cpu);
    }
}
