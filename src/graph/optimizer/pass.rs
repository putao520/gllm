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
    /// 后端类型
    pub backend_type: BackendType,
    /// CUDA SM 版本（如 (8, 0) 表示 SM 8.0）
    pub cuda_sm_version: Option<(u32, u32)>,
    /// 可用显存（字节）
    pub available_memory: usize,
    /// 数据类型
    pub dtype: DType,
    /// 是否启用 FlashAttention
    pub enable_flash_attention: bool,
    /// 是否启用 SwiGLU 融合
    pub enable_swiglu_fusion: bool,
    /// 最大序列长度
    pub max_seq_len: usize,
    /// 注意力头数
    pub num_heads: usize,
    /// KV 头数
    pub num_kv_heads: usize,
    /// 头维度
    pub head_dim: usize,
}

impl Default for OptimizationContext {
    fn default() -> Self {
        Self {
            backend_type: BackendType::Cpu,
            cuda_sm_version: None,
            available_memory: 0,
            dtype: DType::F32,
            enable_flash_attention: true,
            enable_swiglu_fusion: true,
            max_seq_len: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
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
            BackendType::Rocm => false,
            BackendType::Metal => false,
            BackendType::Cpu => false,
        }
    }

    /// 是否支持 SwiGLU 融合
    pub fn supports_swiglu(&self) -> bool {
        self.enable_swiglu_fusion
    }

    /// 创建 CUDA 上下文
    pub fn cuda(sm_version: (u32, u32)) -> Self {
        Self {
            backend_type: BackendType::Cuda,
            cuda_sm_version: Some(sm_version),
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
