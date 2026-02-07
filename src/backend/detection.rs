use gllm_kernels::backend_trait::BackendError;
use gllm_kernels::{CpuBackend, CudaBackend};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Cuda,
    Rocm,
    Metal,
    Cpu,
}

#[derive(Debug)]
pub enum DetectedBackend {
    Cuda(Box<CudaBackend>),
    Cpu(Box<CpuBackend>),
}

impl DetectedBackend {
    pub fn backend_type(&self) -> BackendType {
        match self {
            DetectedBackend::Cuda(_) => BackendType::Cuda,
            DetectedBackend::Cpu(_) => BackendType::Cpu,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BackendAvailability {
    cuda: bool,
    rocm: bool,
    metal: bool,
}

pub fn detect_backend() -> Result<DetectedBackend, BackendError> {
    detect_from_system()
}

fn detect_from_system() -> Result<DetectedBackend, BackendError> {
    let cuda_probe = CudaBackend::new(0).ok();
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
        BackendType::Cpu => Ok(DetectedBackend::Cpu(Box::new(CpuBackend::new()))),
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
