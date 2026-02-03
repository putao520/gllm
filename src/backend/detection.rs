use std::env;

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
    Cuda(CudaBackend),
    Cpu(CpuBackend),
}

impl DetectedBackend {
    pub fn backend_type(&self) -> BackendType {
        match self {
            DetectedBackend::Cuda(_) => BackendType::Cuda,
            DetectedBackend::Cpu(_) => BackendType::Cpu,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendRequest {
    Cuda { ordinal: usize },
    Rocm,
    Metal,
    Cpu,
}

#[derive(Debug, Clone, Copy)]
struct BackendAvailability {
    cuda: bool,
    rocm: bool,
    metal: bool,
}

pub fn detect_backend() -> Result<DetectedBackend, BackendError> {
    if let Ok(value) = env::var("GLLM_DEVICE") {
        return detect_from_override(&value);
    }
    detect_from_system()
}

fn detect_from_override(value: &str) -> Result<DetectedBackend, BackendError> {
    let request = parse_backend_override(value)?;
    init_backend(request)
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
        BackendType::Cuda => Ok(DetectedBackend::Cuda(
            cuda_probe.expect("cuda availability checked"),
        )),
        BackendType::Rocm => Err(BackendError::Unimplemented("rocm backend")),
        BackendType::Metal => Err(BackendError::Unimplemented("metal backend")),
        BackendType::Cpu => Ok(DetectedBackend::Cpu(CpuBackend::new())),
    }
}

fn parse_backend_override(value: &str) -> Result<BackendRequest, BackendError> {
    let trimmed = value.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower == "cpu" {
        return Ok(BackendRequest::Cpu);
    }
    if lower.starts_with("cuda") {
        let ordinal = if let Some((_, idx)) = trimmed.split_once(':') {
            idx.parse::<usize>()
                .map_err(|_| BackendError::InvalidBackendOverride(trimmed.to_string()))?
        } else {
            0
        };
        return Ok(BackendRequest::Cuda { ordinal });
    }
    if lower == "rocm" {
        return Ok(BackendRequest::Rocm);
    }
    if lower == "metal" {
        return Ok(BackendRequest::Metal);
    }

    Err(BackendError::InvalidBackendOverride(trimmed.to_string()))
}

fn init_backend(request: BackendRequest) -> Result<DetectedBackend, BackendError> {
    match request {
        BackendRequest::Cuda { ordinal } => CudaBackend::new(ordinal).map(DetectedBackend::Cuda),
        BackendRequest::Cpu => Ok(DetectedBackend::Cpu(CpuBackend::new())),
        BackendRequest::Rocm => Err(BackendError::Unimplemented("rocm backend")),
        BackendRequest::Metal => Err(BackendError::Unimplemented("metal backend")),
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

    #[test]
    fn parse_backend_override_cpu() {
        assert_eq!(parse_backend_override("cpu").unwrap(), BackendRequest::Cpu);
        assert_eq!(
            parse_backend_override(" CPU ").unwrap(),
            BackendRequest::Cpu
        );
    }

    #[test]
    fn parse_backend_override_cuda_with_ordinal() {
        assert_eq!(
            parse_backend_override("cuda:2").unwrap(),
            BackendRequest::Cuda { ordinal: 2 }
        );
    }

    #[test]
    fn parse_backend_override_cuda_default_ordinal() {
        assert_eq!(
            parse_backend_override("cuda").unwrap(),
            BackendRequest::Cuda { ordinal: 0 }
        );
    }

    #[test]
    fn parse_backend_override_rocm_metal() {
        assert_eq!(
            parse_backend_override("rocm").unwrap(),
            BackendRequest::Rocm
        );
        assert_eq!(
            parse_backend_override("metal").unwrap(),
            BackendRequest::Metal
        );
    }

    #[test]
    fn parse_backend_override_invalid() {
        let err = parse_backend_override("tpu").unwrap_err();
        match err {
            BackendError::InvalidBackendOverride(value) => assert_eq!(value, "tpu"),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
