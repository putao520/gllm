//! OptimizationPass trait 定义 (REQ-OPT-001)

use crate::backend::BackendType;

use super::super::types::FusedGraph;
use super::OptimizeError;

/// 优化 Pass trait (REQ-OPT-001)
pub trait OptimizationPass: Send + Sync + std::fmt::Debug {
    /// Pass 名称
    fn name(&self) -> &'static str;

    /// 执行优化
    fn run(
        &self,
        graph: FusedGraph,
        ctx: &OptimizationContext,
    ) -> Result<FusedGraph, OptimizeError>;

    /// 是否启用此 Pass
    fn enabled(&self, ctx: &OptimizationContext) -> bool {
        let _ = ctx;
        true
    }

    /// Pass 优先级（越小越先执行）
    fn priority(&self) -> i32 {
        0
    }
}

/// 优化上下文 - 提供硬件和配置信息
#[derive(Debug, Clone)]
pub struct OptimizationContext {
    /// Model geometry (provides num_heads, num_kv_heads, head_dim, max_seq_len, hidden_size).
    pub geometry: std::sync::Arc<crate::model_config::ModelGeometry>,
    /// 后端类型
    pub backend_type: BackendType,
    /// CUDA SM 版本（如 (8, 0) 表示 SM 8.0）
    pub cuda_sm_version: Option<(u32, u32)>,
    /// 可用显存（字节）
    pub available_memory: usize,
    /// 数据类型 (optimizer-local DType, distinct from gllm_kernels::types::DType)
    pub dtype: DType,
    /// 是否启用 FlashAttention
    pub enable_flash_attention: bool,
    /// 是否启用 SwiGLU 融合
    pub enable_swiglu_fusion: bool,
    /// 是否支持 VNNI 指令（x86 整数点积加速）
    pub has_vnni: bool,
    /// 是否支持原生 BF16 指令（AVX-512 BF16 / AMX）
    pub has_bf16: bool,
    /// Tensor Core 代数（0=无, 1=Volta/sm70, 2=Ampere/sm80, 3=Hopper/sm90）
    pub tensor_core_gen: u8,
    /// Warp 大小（GPU 专用，CPU 填 0）
    pub warp_size: u32,
}

impl OptimizationContext {
    /// Number of query attention heads.
    pub fn num_heads(&self) -> usize { self.geometry.num_heads }
    /// Number of KV heads.
    pub fn num_kv_heads(&self) -> usize { self.geometry.num_kv_heads }
    /// Head dimension.
    pub fn head_dim(&self) -> usize { self.geometry.head_dim }
    /// Maximum sequence length.
    pub fn max_seq_len(&self) -> usize { self.geometry.max_seq_len }
    /// Hidden layer dimension.
    pub fn hidden_size(&self) -> usize { self.geometry.hidden_size }
}

impl Default for OptimizationContext {
    fn default() -> Self {
        let geometry = std::sync::Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 4096,
            num_layers: 32,
            vocab_size: 32000,
            intermediate_size: 11008,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
        });
        Self {
            geometry,
            backend_type: BackendType::Cpu,
            cuda_sm_version: None,
            available_memory: 0,
            dtype: DType::F32,
            enable_flash_attention: true,
            enable_swiglu_fusion: true,
            has_vnni: false,
            has_bf16: false,
            tensor_core_gen: 0,
            warp_size: 0,
        }
    }
}

impl OptimizationContext {
    /// 是否支持 FlashAttention（需要 CUDA SM >= 8.0）
    pub fn supports_flash_attention(&self) -> bool {
        if !self.enable_flash_attention {
            return false;
        }
        match self.backend_type {
            BackendType::Cuda => {
                if let Some((major, _)) = self.cuda_sm_version {
                    major >= 8
                } else {
                    false
                }
            }
            BackendType::Rocm => true,
            BackendType::Metal => true,
            BackendType::Cpu => false,
        }
    }

    /// 是否支持 SwiGLU 融合
    pub fn supports_swiglu(&self) -> bool {
        self.enable_swiglu_fusion
    }

    /// 创建 CUDA 上下文
    pub fn cuda(sm_version: (u32, u32)) -> Self {
        let tensor_core_gen = if sm_version.0 >= 9 { 3 } else if sm_version.0 >= 8 { 2 } else if sm_version.0 >= 7 { 1 } else { 0 };
        Self {
            backend_type: BackendType::Cuda,
            cuda_sm_version: Some(sm_version),
            tensor_core_gen,
            warp_size: 32,
            ..Default::default()
        }
    }

    /// 创建 CPU 上下文
    pub fn cpu() -> Self {
        Self {
            backend_type: BackendType::Cpu,
            ..Default::default()
        }
    }

    /// Create from ModelGeometry + hardware info.
    pub fn from_geometry(geometry: std::sync::Arc<crate::model_config::ModelGeometry>) -> Self {
        Self {
            geometry,
            ..Default::default()
        }
    }
}

/// 数据类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    #[default]
    F32,
    F16,
    BF16,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_attention_requires_cuda_sm80() {
        let ctx = OptimizationContext::cuda((8, 0));
        assert!(ctx.supports_flash_attention());

        let ctx = OptimizationContext::cuda((7, 5));
        assert!(!ctx.supports_flash_attention());

        let ctx = OptimizationContext::cpu();
        assert!(!ctx.supports_flash_attention());
    }

    #[test]
    fn swiglu_enabled_by_default() {
        let ctx = OptimizationContext::default();
        assert!(ctx.supports_swiglu());
    }
}
