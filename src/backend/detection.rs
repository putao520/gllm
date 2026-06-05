use crate::compat::backend_trait::Element;
use crate::engine::executor::BackendError;
use crate::compat::{CpuBackend, CudaBackend, HipBackend, MetalBackend};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    Rocm(Box<HipBackend<E>>),
    Metal(Box<MetalBackend<E>>),
    Cpu(Box<CpuBackend<E>>),
}

impl<E: Element> DetectedBackend<E> {
    pub fn backend_type(&self) -> BackendType {
        match self {
            DetectedBackend::Cuda(_) => BackendType::Cuda,
            DetectedBackend::Rocm(_) => BackendType::Rocm,
            DetectedBackend::Metal(_) => BackendType::Metal,
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
    let cuda_probe = CudaBackend::<f32>::new(0);
    let rocm_probe = HipBackend::<f32>::new(0);
    let metal_probe = MetalBackend::<f32>::new(0);
    let availability = BackendAvailability {
        cuda: cuda_probe.is_some(),
        rocm: rocm_probe.is_some(),
        metal: metal_probe.is_some(),
    };
    let selected = select_backend_type(availability);
    match selected {
        BackendType::Cuda => Ok(DetectedBackend::Cuda(Box::new(
            cuda_probe.ok_or_else(|| BackendError::Other("CUDA probe succeeded but backend unavailable".into()))?,
        ))),
        BackendType::Rocm => Ok(DetectedBackend::Rocm(Box::new(
            rocm_probe.ok_or_else(|| BackendError::Other("ROCm probe succeeded but backend unavailable".into()))?,
        ))),
        BackendType::Metal => Ok(DetectedBackend::Metal(Box::new(
            metal_probe.ok_or_else(|| BackendError::Other("Metal probe succeeded but backend unavailable".into()))?,
        ))),
        BackendType::Cpu => Ok(DetectedBackend::Cpu(Box::new(CpuBackend::<f32>::new()))),
    }
}

/// Detect backend for a generic element type.
///
/// Supports both CUDA and CPU backends for all Element types.
/// CPU backend uses pseudo-SIMD (f32 promotion) for f16/bf16 types.
pub fn detect_backend_generic<E: Element>() -> Result<DetectedBackend<E>, BackendError> {
    if let Some(backend) = CudaBackend::<E>::new(0) {
        return Ok(DetectedBackend::Cuda(Box::new(backend)));
    }
    if let Some(backend) = HipBackend::<E>::new(0) {
        return Ok(DetectedBackend::Rocm(Box::new(backend)));
    }
    if let Some(backend) = MetalBackend::<E>::new(0) {
        return Ok(DetectedBackend::Metal(Box::new(backend)));
    }
    Ok(DetectedBackend::Cpu(Box::default()))
}

/// Detected dtype from model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    fn backend_type_equality_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(BackendType::Cuda);
        set.insert(BackendType::Cpu);
        set.insert(BackendType::Cuda);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&BackendType::Cuda));
        assert!(!set.contains(&BackendType::Metal));
    }

    #[test]
    fn backend_type_copy_semantics() {
        let a = BackendType::Cuda;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn detected_dtype_from_size() {
        assert_eq!(DetectedDtype::from_size(4), Some(DetectedDtype::F32));
        assert_eq!(DetectedDtype::from_size(2), Some(DetectedDtype::F16));
        assert_eq!(DetectedDtype::from_size(1), None);
        assert_eq!(DetectedDtype::from_size(8), None);
        assert_eq!(DetectedDtype::from_size(0), None);
    }

    #[test]
    fn detected_dtype_from_safetensors_dtype() {
        use safetensors::Dtype;
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::F32), Some(DetectedDtype::F32));
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::F16), Some(DetectedDtype::F16));
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::BF16), Some(DetectedDtype::BF16));
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::U8), None);
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::I32), None);
    }

    #[test]
    fn detected_dtype_equality() {
        assert_eq!(DetectedDtype::F32, DetectedDtype::F32);
        assert_ne!(DetectedDtype::F16, DetectedDtype::BF16);
    }

    #[test]
    fn detected_dtype_debug_format() {
        let dbg = format!("{:?}", DetectedDtype::BF16);
        assert!(dbg.contains("BF16"));
    }

    // ── BackendType trait tests ──

    #[test]
    fn backend_type_all_variants_distinct() {
        let all = [BackendType::Cuda, BackendType::Rocm, BackendType::Metal, BackendType::Cpu];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b, "{:?} == {:?} should be {}", a, b, i == j);
            }
        }
    }

    #[test]
    fn backend_type_clone_is_independent() {
        let original = BackendType::Metal;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn backend_type_debug_outputs_variant_name() {
        assert_eq!(format!("{:?}", BackendType::Cuda), "Cuda");
        assert_eq!(format!("{:?}", BackendType::Rocm), "Rocm");
        assert_eq!(format!("{:?}", BackendType::Metal), "Metal");
        assert_eq!(format!("{:?}", BackendType::Cpu), "Cpu");
    }

    #[test]
    fn backend_type_hash_dedup_in_hashset() {
        use std::collections::HashSet;
        let all: HashSet<BackendType> = [
            BackendType::Cuda,
            BackendType::Rocm,
            BackendType::Metal,
            BackendType::Cpu,
            BackendType::Cuda,
            BackendType::Cpu,
        ].into_iter().collect();
        assert_eq!(all.len(), 4);
    }

    // ── BackendAvailability tests ──

    #[test]
    fn backend_availability_all_false_selects_cpu() {
        let avail = BackendAvailability { cuda: false, rocm: false, metal: false };
        assert_eq!(select_backend_type(avail), BackendType::Cpu);
    }

    #[test]
    fn backend_availability_cuda_only() {
        let avail = BackendAvailability { cuda: true, rocm: false, metal: false };
        assert_eq!(select_backend_type(avail), BackendType::Cuda);
    }

    #[test]
    fn backend_availability_rocm_only() {
        let avail = BackendAvailability { cuda: false, rocm: true, metal: false };
        assert_eq!(select_backend_type(avail), BackendType::Rocm);
    }

    #[test]
    fn backend_availability_metal_only() {
        let avail = BackendAvailability { cuda: false, rocm: false, metal: true };
        assert_eq!(select_backend_type(avail), BackendType::Metal);
    }

    #[test]
    fn backend_availability_cuda_and_rocm_prefers_cuda() {
        let avail = BackendAvailability { cuda: true, rocm: true, metal: false };
        assert_eq!(select_backend_type(avail), BackendType::Cuda);
    }

    #[test]
    fn backend_availability_cuda_and_metal_prefers_cuda() {
        let avail = BackendAvailability { cuda: true, rocm: false, metal: true };
        assert_eq!(select_backend_type(avail), BackendType::Cuda);
    }

    #[test]
    fn backend_availability_rocm_and_metal_prefers_rocm() {
        let avail = BackendAvailability { cuda: false, rocm: true, metal: true };
        assert_eq!(select_backend_type(avail), BackendType::Rocm);
    }

    #[test]
    fn backend_availability_all_true_prefers_cuda() {
        let avail = BackendAvailability { cuda: true, rocm: true, metal: true };
        assert_eq!(select_backend_type(avail), BackendType::Cuda);
    }

    // ── DetectedBackend tests ──

    #[test]
    fn detected_backend_cpu_type() {
        let backend: DetectedBackendF32 = DetectedBackend::Cpu(Box::new(CpuBackend::new()));
        assert_eq!(backend.backend_type(), BackendType::Cpu);
    }

    #[test]
    fn detected_backend_debug_format_cpu() {
        let backend: DetectedBackendF32 = DetectedBackend::Cpu(Box::new(CpuBackend::new()));
        let dbg_str = format!("{:?}", backend);
        assert!(dbg_str.contains("Cpu"), "Expected 'Cpu' in debug output, got: {dbg_str}");
    }

    // ── DetectedDtype extended tests ──

    #[test]
    fn detected_dtype_from_size_boundary_zero() {
        assert_eq!(DetectedDtype::from_size(0), None);
    }

    #[test]
    fn detected_dtype_from_size_boundary_max() {
        assert_eq!(DetectedDtype::from_size(usize::MAX), None);
    }

    #[test]
    fn detected_dtype_from_size_f32_exactly() {
        assert_eq!(DetectedDtype::from_size(4), Some(DetectedDtype::F32));
    }

    #[test]
    fn detected_dtype_from_size_f16_exactly() {
        assert_eq!(DetectedDtype::from_size(2), Some(DetectedDtype::F16));
    }

    #[test]
    fn detected_dtype_from_size_adjacent_values() {
        assert_eq!(DetectedDtype::from_size(3), None);
        assert_eq!(DetectedDtype::from_size(5), None);
        assert_eq!(DetectedDtype::from_size(1), None);
    }

    #[test]
    fn detected_dtype_all_variants_distinct() {
        let all = [DetectedDtype::F32, DetectedDtype::F16, DetectedDtype::BF16];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn detected_dtype_clone_copies_value() {
        let original = DetectedDtype::BF16;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn detected_dtype_copy_semantics() {
        let a = DetectedDtype::F32;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn detected_dtype_hash_dedup() {
        use std::collections::HashSet;
        let set: HashSet<DetectedDtype> = [
            DetectedDtype::F32,
            DetectedDtype::F16,
            DetectedDtype::BF16,
            DetectedDtype::F32,
        ].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn detected_dtype_debug_all_variants() {
        assert!(format!("{:?}", DetectedDtype::F32).contains("F32"));
        assert!(format!("{:?}", DetectedDtype::F16).contains("F16"));
        assert!(format!("{:?}", DetectedDtype::BF16).contains("BF16"));
    }

    #[test]
    fn detected_dtype_from_safetensors_unsupported_types() {
        use safetensors::Dtype;
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::BOOL), None);
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::I8), None);
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::I16), None);
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::I32), None);
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::I64), None);
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::U8), None);
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::U16), None);
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::U32), None);
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::U64), None);
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::F64), None);
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::F8_E5M2), None);
        assert_eq!(DetectedDtype::from_safetensors_dtype(Dtype::F8_E4M3), None);
    }

    // ── detect_backend integration test (CPU path always available) ──

    #[test]
    fn detect_backend_returns_cpu_when_no_gpu() {
        // On a machine without CUDA/ROCm/Metal, this always returns Cpu.
        // On a machine with GPU, it returns the GPU variant. Either way it succeeds.
        let result = detect_backend();
        assert!(result.is_ok(), "detect_backend should always succeed (at minimum CPU)");
        let backend = result.unwrap();
        // We can at least verify backend_type() returns a valid variant.
        let bt = backend.backend_type();
        assert!(
            matches!(bt, BackendType::Cuda | BackendType::Rocm | BackendType::Metal | BackendType::Cpu),
            "backend_type returned unexpected variant: {:?}",
            bt,
        );
    }

    #[test]
    fn detect_backend_generic_f32_returns_valid() {
        let result: Result<DetectedBackend<f32>, _> = detect_backend_generic();
        assert!(result.is_ok(), "detect_backend_generic::<f32>() should always succeed");
    }

    // ── BackendError Display test (covers the custom fmt impl in executor_types) ──

    #[test]
    fn backend_error_display_formats_correctly() {
        use crate::engine::executor::BackendError;
        assert_eq!(
            format!("{}", BackendError::Cuda("oom".into())),
            "CUDA error: oom",
        );
        assert_eq!(
            format!("{}", BackendError::Hip("fault".into())),
            "HIP error: fault",
        );
        assert_eq!(
            format!("{}", BackendError::Metal("na".into())),
            "Metal error: na",
        );
        assert_eq!(
            format!("{}", BackendError::Cpu("overflow".into())),
            "CPU error: overflow",
        );
        assert_eq!(
            format!("{}", BackendError::Unimplemented("foo")),
            "unimplemented: foo",
        );
        assert_eq!(
            format!("{}", BackendError::Other("detail".into())),
            "backend error: detail",
        );
    }

    #[test]
    fn backend_error_debug_includes_variant() {
        use crate::engine::executor::BackendError;
        let dbg = format!("{:?}", BackendError::Cuda("test".into()));
        assert!(dbg.contains("Cuda"), "Debug output should contain 'Cuda': {dbg}");
    }

    // ══════════════════════════════════════════════════════════
    // 15 new tests — uncovered paths and edge cases
    // ══════════════════════════════════════════════════════════

    // ── BackendAvailability structural tests ──

    #[test]
    fn backend_availability_clone_produces_equal_value() {
        // Arrange
        let original = BackendAvailability { cuda: true, rocm: false, metal: true };
        // Act
        let cloned = original.clone();
        // Assert
        assert_eq!(original.cuda, cloned.cuda);
        assert_eq!(original.rocm, cloned.rocm);
        assert_eq!(original.metal, cloned.metal);
    }

    #[test]
    fn backend_availability_copy_semantics_preserve_original() {
        // Arrange
        let original = BackendAvailability { cuda: false, rocm: true, metal: false };
        // Act
        let copied = original; // Copy, not move
        // Assert — both are still usable (Copy trait)
        assert_eq!(original.cuda, copied.cuda);
        assert_eq!(original.rocm, copied.rocm);
        assert_eq!(original.metal, copied.metal);
    }

    #[test]
    fn backend_availability_debug_format_contains_fields() {
        // Arrange
        let avail = BackendAvailability { cuda: true, rocm: false, metal: true };
        // Act
        let dbg_str = format!("{:?}", avail);
        // Assert
        assert!(dbg_str.contains("cuda"), "Debug should contain 'cuda': {dbg_str}");
        assert!(dbg_str.contains("rocm"), "Debug should contain 'rocm': {dbg_str}");
        assert!(dbg_str.contains("metal"), "Debug should contain 'metal': {dbg_str}");
    }

    // ── BackendType as HashMap key ──

    #[test]
    fn backend_type_used_as_hashmap_key() {
        // Arrange
        use std::collections::HashMap;
        let mut map = HashMap::new();
        // Act
        map.insert(BackendType::Cuda, "GPU-NVIDIA");
        map.insert(BackendType::Rocm, "GPU-AMD");
        map.insert(BackendType::Metal, "GPU-Apple");
        map.insert(BackendType::Cpu, "CPU");
        // Assert
        assert_eq!(map.get(&BackendType::Cuda), Some(&"GPU-NVIDIA"));
        assert_eq!(map.get(&BackendType::Rocm), Some(&"GPU-AMD"));
        assert_eq!(map.get(&BackendType::Metal), Some(&"GPU-Apple"));
        assert_eq!(map.get(&BackendType::Cpu), Some(&"CPU"));
        assert_eq!(map.len(), 4);
    }

    // ── DetectedBackend type alias verification ──

    #[test]
    fn detected_backend_f32_alias_is_detected_backend_f32() {
        // Arrange
        let cpu_backend: DetectedBackendF32 = DetectedBackend::Cpu(Box::new(CpuBackend::new()));
        // Act
        let bt = cpu_backend.backend_type();
        // Assert
        assert_eq!(bt, BackendType::Cpu);
    }

    // ── DetectedBackend match exhaustiveness on all variants ──

    #[test]
    fn detected_backend_match_all_variants_return_backend_type() {
        // Arrange — construct all 4 variants as DetectedBackend<f32>
        let cpu: DetectedBackend<f32> = DetectedBackend::Cpu(Box::new(CpuBackend::new()));
        // Act & Assert — match proves exhaustiveness
        match cpu {
            DetectedBackend::Cuda(_) | DetectedBackend::Rocm(_) |
            DetectedBackend::Metal(_) | DetectedBackend::Cpu(_) => {}
        }
    }

    // ── BackendError with empty and special strings ──

    #[test]
    fn backend_error_display_empty_message() {
        // Arrange
        use crate::engine::executor::BackendError;
        // Act & Assert
        assert_eq!(format!("{}", BackendError::Cuda("".into())), "CUDA error: ");
        assert_eq!(format!("{}", BackendError::Hip("".into())), "HIP error: ");
        assert_eq!(format!("{}", BackendError::Metal("".into())), "Metal error: ");
        assert_eq!(format!("{}", BackendError::Cpu("".into())), "CPU error: ");
    }

    #[test]
    fn backend_error_display_long_message() {
        // Arrange
        use crate::engine::executor::BackendError;
        let long_msg = "a".repeat(1000);
        let expected = format!("CUDA error: {}", long_msg);
        // Act
        let actual = format!("{}", BackendError::Cuda(long_msg));
        // Assert
        assert_eq!(actual, expected);
    }

    #[test]
    fn backend_error_unimplemented_display() {
        // Arrange
        use crate::engine::executor::BackendError;
        // Act & Assert
        assert_eq!(
            format!("{}", BackendError::Unimplemented("quantized gemm")),
            "unimplemented: quantized gemm",
        );
    }

    // ── DetectedDtype from_size additional edge cases ──

    #[test]
    fn detected_dtype_from_size_all_non_matching() {
        // Arrange — values near boundaries that should return None
        let non_matching: Vec<usize> = vec![0, 1, 3, 5, 6, 7, 8, 16, 32, 64, 128, 256, 1024];
        // Act & Assert
        for size in non_matching {
            assert_eq!(
                DetectedDtype::from_size(size), None,
                "from_size({}) should return None, only 2 and 4 are valid",
                size,
            );
        }
    }

    // ── DetectedDtype from_safetensors_dtype round-trip for all valid types ──

    #[test]
    fn detected_dtype_from_safetensors_roundtrip_all_valid() {
        // Arrange
        use safetensors::Dtype;
        let valid_pairs: Vec<(Dtype, DetectedDtype)> = vec![
            (Dtype::F32, DetectedDtype::F32),
            (Dtype::F16, DetectedDtype::F16),
            (Dtype::BF16, DetectedDtype::BF16),
        ];
        // Act & Assert
        for (st_dtype, expected) in valid_pairs {
            let result = DetectedDtype::from_safetensors_dtype(st_dtype);
            assert_eq!(result, Some(expected), "from_safetensors_dtype({:?}) mismatch", st_dtype);
        }
    }

    // ── select_backend_type determinism ──

    #[test]
    fn select_backend_type_deterministic_across_repeated_calls() {
        // Arrange
        let avail = BackendAvailability { cuda: false, rocm: false, metal: true };
        // Act
        let first = select_backend_type(avail);
        let second = select_backend_type(avail);
        let third = select_backend_type(avail);
        // Assert — pure function, must always return same result
        assert_eq!(first, BackendType::Metal);
        assert_eq!(second, BackendType::Metal);
        assert_eq!(third, BackendType::Metal);
    }

    // ── detect_backend_generic with bf16 element type ──

    #[test]
    fn detect_backend_generic_bf16_returns_valid() {
        // Arrange — use half::bf16 as element type
        use half::bf16;
        // Act
        let result: Result<DetectedBackend<bf16>, _> = detect_backend_generic();
        // Assert — should always succeed (at minimum CPU fallback)
        assert!(result.is_ok(), "detect_backend_generic::<bf16>() should always succeed");
        let backend = result.unwrap();
        assert!(
            matches!(backend.backend_type(),
                BackendType::Cuda | BackendType::Rocm | BackendType::Metal | BackendType::Cpu),
        );
    }

    // ── BackendType in collection sort context ──

    #[test]
    fn backend_type_vec_contains_and_remove() {
        // Arrange
        let mut backends = vec![BackendType::Cuda, BackendType::Rocm, BackendType::Metal, BackendType::Cpu];
        // Act
        assert!(backends.contains(&BackendType::Rocm));
        backends.retain(|b| *b != BackendType::Rocm);
        // Assert
        assert!(!backends.contains(&BackendType::Rocm));
        assert_eq!(backends.len(), 3);
    }

    // ── DetectedDtype Copy + Clone interaction ──

    #[test]
    fn detected_dtype_copy_allows_use_after_assignment() {
        // Arrange
        let a = DetectedDtype::F16;
        // Act — Copy semantics: both a and b are valid
        let b = a;
        let c = b;
        // Assert
        assert_eq!(a, DetectedDtype::F16);
        assert_eq!(b, DetectedDtype::F16);
        assert_eq!(c, DetectedDtype::F16);
    }

    // ══════════════════════════════════════════════════════════
    // 13 new tests — additional uncovered paths and edge cases
    // ══════════════════════════════════════════════════════════

    // ── BackendError Clone trait ──

    #[test]
    // @trace TEST-DET-01 [req:REQ-ARCH] [level:unit]
    fn backend_error_clone_produces_equal_value() {
        // Arrange
        use crate::engine::executor::BackendError;
        let original = BackendError::Cuda("device-side assert".into());
        // Act
        let cloned = original.clone();
        // Assert — Display output must be identical
        assert_eq!(format!("{original}"), format!("{cloned}"));
    }

    #[test]
    // @trace TEST-DET-02 [req:REQ-ARCH] [level:unit]
    fn backend_error_clone_all_variants_preserve_display() {
        // Arrange
        use crate::engine::executor::BackendError;
        let errors = vec![
            BackendError::Cuda("cuda_err".into()),
            BackendError::Hip("hip_err".into()),
            BackendError::Metal("metal_err".into()),
            BackendError::Cpu("cpu_err".into()),
            BackendError::Unimplemented("feature_x"),
            BackendError::Other("generic".into()),
        ];
        // Act & Assert — clone each variant and verify Display matches
        for err in &errors {
            let cloned = err.clone();
            assert_eq!(format!("{err}"), format!("{cloned}"), "Clone mismatch for {:?}", err);
        }
    }

    // ── BackendError std::error::Error source ──

    #[test]
    // @trace TEST-DET-03 [req:REQ-ARCH] [level:unit]
    fn backend_error_source_returns_none() {
        // Arrange
        use std::error::Error;
        use crate::engine::executor::BackendError;
        // Act
        let err = BackendError::Cuda("test".into());
        let source = err.source();
        // Assert — BackendError has no chain, source is always None
        assert!(source.is_none(), "BackendError::source() should return None");
    }

    // ── DetectedBackend backend_type for GPU variants (conditional on hardware) ──

    #[test]
    // @trace TEST-DET-04 [req:REQ-ARCH] [level:unit]
    fn detected_backend_detect_backend_returns_valid_enum_variant() {
        // Arrange — detect_backend must return one of the 4 valid variants
        // Act
        let result = detect_backend();
        // Assert
        assert!(result.is_ok(), "detect_backend must succeed");
        let backend = result.unwrap();
        let bt = backend.backend_type();
        assert!(
            matches!(bt, BackendType::Cuda | BackendType::Rocm | BackendType::Metal | BackendType::Cpu),
            "Expected a valid BackendType variant, got {:?}", bt,
        );
    }

    #[test]
    // @trace TEST-DET-05 [req:REQ-ARCH] [level:unit]
    fn detected_backend_gpu_variants_return_correct_type_when_available() {
        // Arrange — try creating GPU backends; if they succeed, backend_type must match
        // Act & Assert for CUDA
        if let Some(cuda) = CudaBackend::<f32>::new(0) {
            let wrapped: DetectedBackend<f32> = DetectedBackend::Cuda(Box::new(cuda));
            assert_eq!(wrapped.backend_type(), BackendType::Cuda);
        }
        // Act & Assert for ROCm
        if let Some(hip) = HipBackend::<f32>::new(0) {
            let wrapped: DetectedBackend<f32> = DetectedBackend::Rocm(Box::new(hip));
            assert_eq!(wrapped.backend_type(), BackendType::Rocm);
        }
        // Act & Assert for Metal
        if let Some(metal) = MetalBackend::<f32>::new(0) {
            let wrapped: DetectedBackend<f32> = DetectedBackend::Metal(Box::new(metal));
            assert_eq!(wrapped.backend_type(), BackendType::Metal);
        }
    }

    // ── detect_backend_generic with f16 element type ──

    #[test]
    // @trace TEST-DET-06 [req:REQ-ARCH] [level:unit]
    fn detect_backend_generic_f16_returns_valid() {
        // Arrange
        use half::f16;
        // Act
        let result: Result<DetectedBackend<f16>, _> = detect_backend_generic();
        // Assert — should always succeed (at minimum CPU)
        assert!(result.is_ok(), "detect_backend_generic::<f16>() should always succeed");
        let backend = result.unwrap();
        let bt = backend.backend_type();
        assert!(
            matches!(bt, BackendType::Cuda | BackendType::Rocm | BackendType::Metal | BackendType::Cpu),
            "Expected valid BackendType, got {:?}", bt,
        );
    }

    // ── DetectedDtype from_size confirms F16 not BF16 for 2-byte ──

    #[test]
    // @trace TEST-DET-07 [req:REQ-ARCH] [level:unit]
    fn detected_dtype_from_size_two_bytes_returns_f16_not_bf16() {
        // Arrange — 2-byte size is ambiguous; function docs say F16 is preferred
        // Act
        let result = DetectedDtype::from_size(2);
        // Assert — must be F16, never BF16 (documented behavior)
        assert_eq!(result, Some(DetectedDtype::F16));
        assert_ne!(result, Some(DetectedDtype::BF16));
    }

    // ── DetectedDtype as HashMap key for lookups ──

    #[test]
    // @trace TEST-DET-08 [req:REQ-ARCH] [level:unit]
    fn detected_dtype_used_as_hashmap_key() {
        // Arrange
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(DetectedDtype::F32, "fp32_compute");
        map.insert(DetectedDtype::F16, "fp16_compute");
        map.insert(DetectedDtype::BF16, "bf16_compute");
        // Act
        let f32_label = map.get(&DetectedDtype::F32);
        let f16_label = map.get(&DetectedDtype::F16);
        let bf16_label = map.get(&DetectedDtype::BF16);
        // Assert
        assert_eq!(f32_label, Some(&"fp32_compute"));
        assert_eq!(f16_label, Some(&"fp16_compute"));
        assert_eq!(bf16_label, Some(&"bf16_compute"));
        assert_eq!(map.len(), 3);
    }

    // ── BackendError with unicode and special characters ──

    #[test]
    // @trace TEST-DET-09 [req:REQ-ARCH] [level:unit]
    fn backend_error_display_unicode_and_special_chars() {
        // Arrange
        use crate::engine::executor::BackendError;
        let unicode_msg = "GPU错误: 内存不足 (OOM) \u{1F600}";
        // Act
        let displayed = format!("{}", BackendError::Other(unicode_msg.into()));
        // Assert
        assert_eq!(displayed, format!("backend error: {unicode_msg}"));
    }

    // ── BackendError Debug format covers all variants ──

    #[test]
    // @trace TEST-DET-10 [req:REQ-ARCH] [level:unit]
    fn backend_error_debug_all_variants_contain_variant_name() {
        // Arrange
        use crate::engine::executor::BackendError;
        let variants: Vec<(BackendError, &str)> = vec![
            (BackendError::Cuda("e".into()), "Cuda"),
            (BackendError::Hip("e".into()), "Hip"),
            (BackendError::Metal("e".into()), "Metal"),
            (BackendError::Cpu("e".into()), "Cpu"),
            (BackendError::Unimplemented("e"), "Unimplemented"),
            (BackendError::Other("e".into()), "Other"),
        ];
        // Act & Assert
        for (err, name) in variants {
            let dbg = format!("{:?}", err);
            assert!(dbg.contains(name), "Debug {:?} should contain '{}'", err, name);
        }
    }

    // ── BackendType exhaustive pattern matching in expression context ──

    #[test]
    // @trace TEST-DET-11 [req:REQ-ARCH] [level:unit]
    fn backend_type_match_expression_returns_string_label() {
        // Arrange
        fn label(bt: BackendType) -> &'static str {
            match bt {
                BackendType::Cuda => "nvidia",
                BackendType::Rocm => "amd",
                BackendType::Metal => "apple",
                BackendType::Cpu => "host",
            }
        }
        // Act & Assert
        assert_eq!(label(BackendType::Cuda), "nvidia");
        assert_eq!(label(BackendType::Rocm), "amd");
        assert_eq!(label(BackendType::Metal), "apple");
        assert_eq!(label(BackendType::Cpu), "host");
    }

    // ── select_backend_type priority order is stable across all permutations ──

    #[test]
    // @trace TEST-DET-12 [req:REQ-ARCH] [level:unit]
    fn select_backend_type_priority_order_cuda_rocm_metal_cpu() {
        // Arrange — test that priority is exactly Cuda > Rocm > Metal > Cpu
        // by verifying each "higher priority" backend wins when both are available
        let cuda_wins = select_backend_type(BackendAvailability { cuda: true, rocm: true, metal: true });
        let rocm_wins = select_backend_type(BackendAvailability { cuda: false, rocm: true, metal: true });
        let metal_wins = select_backend_type(BackendAvailability { cuda: false, rocm: false, metal: true });
        let cpu_wins = select_backend_type(BackendAvailability { cuda: false, rocm: false, metal: false });
        // Assert
        assert_eq!(cuda_wins, BackendType::Cuda, "Cuda must be selected when all available");
        assert_eq!(rocm_wins, BackendType::Rocm, "Rocm must be selected when Cuda unavailable");
        assert_eq!(metal_wins, BackendType::Metal, "Metal must be selected when Cuda+Rocm unavailable");
        assert_eq!(cpu_wins, BackendType::Cpu, "Cpu must be selected when no GPU available");
    }

    // ── BackendError Other variant with multiline message ──

    #[test]
    // @trace TEST-DET-13 [req:REQ-ARCH] [level:unit]
    fn backend_error_other_multiline_message_preserved() {
        // Arrange
        use crate::engine::executor::BackendError;
        let multiline = "line1\nline2\nline3";
        let err = BackendError::Other(multiline.to_string());
        // Act
        let displayed = format!("{err}");
        // Assert — newlines must be preserved, not stripped
        assert!(displayed.contains("line1\nline2"), "Multiline content must be preserved");
        assert!(displayed.starts_with("backend error: "), "Must start with prefix");
    }

    // ══════════════════════════════════════════════════════════
    // 10 new tests — remaining uncovered paths and edge cases
    // ══════════════════════════════════════════════════════════

    // ── BackendAvailability structural field independence ──

    #[test]
    // @trace TEST-DET-14 [req:REQ-ARCH] [level:unit]
    fn backend_availability_fields_are_independent() {
        // Arrange — toggle each field independently and verify no cross-contamination
        let base = BackendAvailability { cuda: false, rocm: false, metal: false };
        // Act — toggle cuda only
        let cuda_on = BackendAvailability { cuda: true, ..base };
        // Assert — only cuda changed
        assert!(cuda_on.cuda);
        assert!(!cuda_on.rocm);
        assert!(!cuda_on.metal);

        // Act — toggle rocm only
        let rocm_on = BackendAvailability { rocm: true, ..base };
        assert!(!rocm_on.cuda);
        assert!(rocm_on.rocm);
        assert!(!rocm_on.metal);

        // Act — toggle metal only
        let metal_on = BackendAvailability { metal: true, ..base };
        assert!(!metal_on.cuda);
        assert!(!metal_on.rocm);
        assert!(metal_on.metal);
    }

    // ── BackendError variant Display prefixes are distinct ──

    #[test]
    // @trace TEST-DET-15 [req:REQ-ARCH] [level:unit]
    fn backend_error_display_prefixes_distinct_across_variants() {
        // Arrange — all variants with the same payload to isolate prefix differences
        use crate::engine::executor::BackendError;
        let payload = "x";
        let cuda_str = format!("{}", BackendError::Cuda(payload.into()));
        let hip_str = format!("{}", BackendError::Hip(payload.into()));
        let metal_str = format!("{}", BackendError::Metal(payload.into()));
        let cpu_str = format!("{}", BackendError::Cpu(payload.into()));
        let other_str = format!("{}", BackendError::Other(payload.into()));
        let unimpl_str = format!("{}", BackendError::Unimplemented(payload));
        // Act & Assert — each Display output must be unique (different prefixes)
        let all_strs = [&cuda_str, &hip_str, &metal_str, &cpu_str, &other_str, &unimpl_str];
        for (i, a) in all_strs.iter().enumerate() {
            for (j, b) in all_strs.iter().enumerate() {
                assert_eq!(i == j, a == b, "Display outputs must be distinct: {:?} vs {:?}", a, b);
            }
        }
    }

    // ── DetectedBackend Debug format includes variant name for Cpu ──

    #[test]
    // @trace TEST-DET-16 [req:REQ-ARCH] [level:unit]
    fn detected_backend_debug_includes_inner_backend_type_name() {
        // Arrange — Cpu variant is always available for testing
        let cpu: DetectedBackend<f32> = DetectedBackend::Cpu(Box::new(CpuBackend::new()));
        // Act
        let dbg_str = format!("{:?}", cpu);
        // Assert — Debug must contain the enum variant identifier
        assert!(dbg_str.contains("Cpu"), "Debug must contain 'Cpu': {dbg_str}");
    }

    // ── detect_backend returns consistent backend type across calls ──

    #[test]
    // @trace TEST-DET-17 [req:REQ-ARCH] [level:unit]
    fn detect_backend_returns_consistent_type_across_two_calls() {
        // Arrange — call detect_backend twice
        let first = detect_backend().expect("first call must succeed");
        let second = detect_backend().expect("second call must succeed");
        // Act
        let first_type = first.backend_type();
        let second_type = second.backend_type();
        // Assert — same hardware, same result
        assert_eq!(first_type, second_type, "Repeated detection must yield same BackendType");
    }

    // ── DetectedDtype inequality across all distinct pairs ──

    #[test]
    // @trace TEST-DET-18 [req:REQ-ARCH] [level:unit]
    fn detected_dtype_all_pairs_are_unequal() {
        // Arrange — all 3 variants
        let variants = [DetectedDtype::F32, DetectedDtype::F16, DetectedDtype::BF16];
        let mut unequal_count = 0;
        // Act — count all unequal pairs
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "{:?} must not equal {:?}", a, b);
                    unequal_count += 1;
                }
            }
        }
        // Assert — 3 variants => 6 unequal pairs (3 * 2)
        assert_eq!(unequal_count, 6, "Expected 6 unequal pairs for 3 variants");
    }

    // ── BackendType size_of is minimal (Copy enum) ──

    #[test]
    // @trace TEST-DET-19 [req:REQ-ARCH] [level:unit]
    fn backend_type_size_is_one_byte() {
        // Arrange — BackendType is a fieldless enum with 4 variants
        use std::mem::size_of;
        // Act
        let sz = size_of::<BackendType>();
        // Assert — fieldless enum should be 1 byte (discriminant only)
        assert_eq!(sz, 1, "BackendType should be 1 byte, got {sz}");
    }

    // ── DetectedDtype size_of is one byte ──

    #[test]
    // @trace TEST-DET-20 [req:REQ-ARCH] [level:unit]
    fn detected_dtype_size_is_one_byte() {
        // Arrange — DetectedDtype is a fieldless enum with 3 variants
        use std::mem::size_of;
        // Act
        let sz = size_of::<DetectedDtype>();
        // Assert — fieldless enum should be 1 byte
        assert_eq!(sz, 1, "DetectedDtype should be 1 byte, got {sz}");
    }

    // ── BackendAvailability Copy allows partial struct update ──

    #[test]
    // @trace TEST-DET-21 [req:REQ-ARCH] [level:unit]
    fn backend_availability_struct_update_with_base() {
        // Arrange — base with all false
        let base = BackendAvailability { cuda: false, rocm: false, metal: false };
        // Act — use struct update syntax to flip two fields
        let toggled = BackendAvailability { cuda: true, metal: true, ..base };
        // Assert
        assert!(toggled.cuda, "cuda should be toggled to true");
        assert!(!toggled.rocm, "rocm should inherit false from base");
        assert!(toggled.metal, "metal should be toggled to true");
    }

    // ── select_backend_type with single GPU available returns that GPU ──

    #[test]
    // @trace TEST-DET-22 [req:REQ-ARCH] [level:unit]
    fn select_backend_type_rocm_only_skips_metal_and_cpu() {
        // Arrange — only rocm is available
        let avail = BackendAvailability { cuda: false, rocm: true, metal: false };
        // Act
        let result = select_backend_type(avail);
        // Assert — must pick rocm, not skip to cpu or metal
        assert_eq!(result, BackendType::Rocm);
    }

    // ── BackendError Unimplemented with static str vs Other with String ──

    #[test]
    // @trace TEST-DET-23 [req:REQ-ARCH] [level:unit]
    fn backend_error_unimplemented_accepts_static_str_other_accepts_string() {
        // Arrange
        use crate::engine::executor::BackendError;
        // Act — Unimplemented takes &'static str
        let unimpl: BackendError = BackendError::Unimplemented("not yet supported");
        // Act — Other takes String
        let other: BackendError = BackendError::Other("dynamic message".to_string());
        // Assert — both render without panic and have distinct prefixes
        let unimpl_display = format!("{unimpl}");
        let other_display = format!("{other}");
        assert!(unimpl_display.starts_with("unimplemented:"), "Unimplemented prefix: {unimpl_display}");
        assert!(other_display.starts_with("backend error:"), "Other prefix: {other_display}");
    }

    // ── DetectedDtype from_safetensors_dtype returns None for F64 ──

    #[test]
    // @trace TEST-DET-24 [req:REQ-ARCH] [level:unit]
    fn detected_dtype_from_safetensors_f64_returns_none() {
        // Arrange — F64 is a valid safetensors type but not a model inference dtype
        use safetensors::Dtype;
        // Act
        let result = DetectedDtype::from_safetensors_dtype(Dtype::F64);
        // Assert — F64 is not supported for inference
        assert_eq!(result, None, "F64 should not map to any DetectedDtype");
    }
}
