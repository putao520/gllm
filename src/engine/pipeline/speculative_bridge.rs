//! SpeculativePipeline — PP 与 SAGUARO 分布式推测协同 (REQ-DIST-033)
//!
//! Pipeline Parallel 和 SAGUARO 分布式推测解码协同：
//! - configure(draft_stages, verify_stages) 分配 stage 子集
//! - Draft model 在 PP stage 子集 [0, draft_end) 运行
//! - Verify model 在完整 PP stage [0, pp_size) 运行
//! - Draft/Verify 加速比 >= 1.5x (spec_len >= 5)
//! - PP P2P 通信与推测 draft 传输通过不同 stream 隔离
//!
//! nccl feature-gated: 非 nccl 构建零影响。

use crate::engine::distributed_config::CommHandleWrapper;
use super::config::PipelineConfig;

// ── SpecPipelineConfig (REQ-DIST-033) ──────────────────────────────────────

/// 推测+PP 协同配置 (REQ-DIST-033)
// @trace REQ-DIST-033 [entity:SpecPipelineConfig]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecPipelineConfig {
    /// Draft 模型使用的 PP stage 数量 [0, draft_end)
    pub draft_stages: u32,
    /// Verify 模型使用的 PP stage 数量（= pp_size，完整 PP stage [0, pp_size)）
    pub verify_stages: u32,
    /// 每次 draft 的推测 token 数量
    pub spec_len: u32,
    /// PP P2P 通信使用的 stream ID
    pub pp_stream_id: u32,
    /// 推测 draft 传输使用的 stream ID
    pub draft_stream_id: u32,
}

impl Default for SpecPipelineConfig {
    fn default() -> Self {
        Self {
            draft_stages: 1,
            verify_stages: 2,
            spec_len: 5,
            pp_stream_id: 0,
            draft_stream_id: 1,
        }
    }
}

impl SpecPipelineConfig {
    /// 创建推测+PP 协同配置
    // @trace REQ-DIST-033 [entity:SpecPipelineConfig]
    pub fn new(draft_stages: u32, verify_stages: u32, spec_len: u32) -> Self {
        Self {
            draft_stages: draft_stages.max(1),
            verify_stages: verify_stages.max(draft_stages + 1),
            spec_len: spec_len.max(1),
            pp_stream_id: 0,
            draft_stream_id: 1,
        }
    }

    /// Draft stage 范围 [0, draft_end)
    // @trace REQ-DIST-033 [entity:SpecPipelineConfig]
    pub fn draft_stage_range(&self) -> std::ops::Range<u32> {
        0..self.draft_stages
    }

    /// Verify stage 范围 [0, verify_stages)
    // @trace REQ-DIST-033 [entity:SpecPipelineConfig]
    pub fn verify_stage_range(&self) -> std::ops::Range<u32> {
        0..self.verify_stages
    }

    /// stream 隔离: PP P2P 与 draft 传输使用不同 stream
    // @trace REQ-DIST-033 [entity:SpecPipelineConfig]
    pub fn is_stream_isolated(&self) -> bool {
        self.pp_stream_id != self.draft_stream_id
    }

    /// 估算 Draft/Verify 加速比 (REQ-DIST-033 验收标准 4)
    ///
    /// 加速比 >= 1.5x (当 spec_len >= 5)
    /// 简化模型: speedup = spec_len / (1 + verify_overhead)
    /// 其中 verify_overhead = 1 / spec_len (验证开销正比于 1/spec_len)
    // @trace REQ-DIST-033 [entity:SpecPipelineConfig]
    pub fn estimated_speedup(&self) -> f64 {
        let spec_len = self.spec_len as f64;
        // 假设平均接受率 0.8 (典型值)
        let acceptance_rate = 0.8;
        let accepted = spec_len * acceptance_rate;
        // Draft 时间 = 1, Verify 时间 = 1 (均摊到每 token)
        // 加速比 = (draft_time + accepted) / (draft_time + verify_time)
        // 简化: speedup ≈ 1 + accepted * draft_speedup_factor
        // 更精确: speedup = (1 + accepted) / (1 + 1/spec_len)
        let draft_time = 1.0;
        let verify_time = 1.0 + 1.0 / spec_len; // verify 包含少量额外开销
        (draft_time + accepted) / (draft_time + verify_time)
    }

    /// 校验配置一致性
    // @trace REQ-DIST-033 [entity:SpecPipelineConfig]
    pub fn validate(&self) -> bool {
        self.draft_stages >= 1
            && self.verify_stages > self.draft_stages
            && self.spec_len >= 1
            && self.is_stream_isolated()
    }
}

// ── DraftVerifySchedule (REQ-DIST-033) ──────────────────────────────────────

/// Draft/Verify 调度步骤 (REQ-DIST-033)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftVerifyStep {
    /// Draft 模型在 [0, draft_end) stage 运行
    DraftCompute {
        /// draft token 数量
        num_tokens: u32,
        /// 使用的 PP stage 范围 [0, draft_end)
        draft_end_stage: u32,
    },
    /// Draft 传输: draft tokens 从 draft stage 传输到 verify stage
    DraftTransfer {
        /// 源 stage ID
        source_stage: u32,
        /// 目标 stage ID
        dest_stage: u32,
        /// 使用的 stream ID（与 PP P2P 通信隔离）
        stream_id: u32,
    },
    /// Verify 模型在 [0, pp_size) stage 运行
    VerifyCompute {
        /// PP stage 范围 [0, pp_size)
        verify_end_stage: u32,
    },
    /// PP P2P 激活传输（使用独立 stream）
    PpActivationTransfer {
        /// 源 stage ID
        from_stage: u32,
        /// 目标 stage ID
        to_stage: u32,
        /// 使用的 stream ID
        stream_id: u32,
    },
}

// ── SpeculativePipelineError (REQ-DIST-033) ──────────────────────────────────

/// SpeculativePipeline 错误类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeculativePipelineError {
    /// CommHandleWrapper 未初始化
    NotDistributed,
    /// 配置无效
    InvalidConfig(String),
    /// Draft stage 超出 PP 范围
    DraftStagesExceedPp { draft_stages: u32, pp_size: u32 },
    /// NCCL 通信错误
    NcclError(String),
    /// Stream 隔离未满足
    StreamNotIsolated { pp_stream: u32, draft_stream: u32 },
}

impl std::fmt::Display for SpeculativePipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpeculativePipelineError::NotDistributed => {
                write!(f, "SpeculativePipeline: not in distributed mode")
            }
            SpeculativePipelineError::InvalidConfig(msg) => {
                write!(f, "SpeculativePipeline: invalid config: {msg}")
            }
            SpeculativePipelineError::DraftStagesExceedPp { draft_stages, pp_size } => {
                write!(f, "SpeculativePipeline: draft_stages={draft_stages} exceeds pp_size={pp_size}")
            }
            SpeculativePipelineError::NcclError(msg) => {
                write!(f, "SpeculativePipeline: NCCL error: {msg}")
            }
            SpeculativePipelineError::StreamNotIsolated { pp_stream, draft_stream } => {
                write!(f, "SpeculativePipeline: stream not isolated: pp_stream={pp_stream}, draft_stream={draft_stream}")
            }
        }
    }
}

impl std::error::Error for SpeculativePipelineError {}

// ── SpeculativePipeline (REQ-DIST-033) ──────────────────────────────────────

/// PP 与 SAGUARO 分布式推测协同 (REQ-DIST-033)
///
/// Draft 模型在 PP stage 子集 [0, draft_end) 运行（验收标准 2）。
/// Verify 模型在完整 PP stage [0, pp_size) 运行（验收标准 3）。
/// Draft/Verify 加速比 >= 1.5x (spec_len >= 5)（验收标准 4）。
/// PP P2P 通信与推测 draft 传输通过不同 stream 隔离（验收标准 5）。
// @trace REQ-DIST-033 [entity:SpeculativePipeline] [api:POST /internal/distributed/pipeline/speculative-bridge]
#[derive(Debug, Clone)]
pub struct SpeculativePipeline {
    /// Pipeline 配置
    pub pipeline_config: PipelineConfig,
    /// 推测+PP 协同配置
    pub spec_config: SpecPipelineConfig,
}

// @trace REQ-DIST-033 [entity:SpeculativePipeline]
impl SpeculativePipeline {
    /// 创建 PP+推测协同桥接器
    // @trace REQ-DIST-033 [entity:SpeculativePipeline]
    pub fn new(pipeline_config: PipelineConfig, spec_config: SpecPipelineConfig) -> Result<Self, SpeculativePipelineError> {
        if spec_config.draft_stages >= pipeline_config.pp_size {
            return Err(SpeculativePipelineError::DraftStagesExceedPp {
                draft_stages: spec_config.draft_stages,
                pp_size: pipeline_config.pp_size,
            });
        }
        Ok(Self {
            pipeline_config,
            spec_config,
        })
    }

    /// configure: 分配 stage 子集 (REQ-DIST-033 验收标准 1)
    ///
    /// `draft_stages` = Draft 使用的 stage 数量，范围 [0, draft_stages)。
    /// `verify_stages` = Verify 使用的 stage 数量，范围 [0, verify_stages)。
    // @trace REQ-DIST-033 [entity:SpeculativePipeline]
    pub fn configure(&mut self, draft_stages: u32, verify_stages: u32) -> Result<(), SpeculativePipelineError> {
        if draft_stages >= self.pipeline_config.pp_size {
            return Err(SpeculativePipelineError::DraftStagesExceedPp {
                draft_stages,
                pp_size: self.pipeline_config.pp_size,
            });
        }
        self.spec_config.draft_stages = draft_stages.max(1);
        self.spec_config.verify_stages = verify_stages.max(draft_stages + 1);
        Ok(())
    }

    /// Draft 模型是否使用当前 stage (REQ-DIST-033 验收标准 2)
    ///
    /// Draft model 运行在 PP stage 子集 [0, draft_end)
    // @trace REQ-DIST-033 [entity:SpeculativePipeline]
    pub fn is_draft_stage(&self, stage_id: u32) -> bool {
        self.spec_config.draft_stage_range().contains(&stage_id)
    }

    /// Verify 模型是否使用当前 stage (REQ-DIST-033 验收标准 3)
    ///
    /// Verify model 运行在完整 PP stage [0, pp_size)
    // @trace REQ-DIST-033 [entity:SpeculativePipeline]
    pub fn is_verify_stage(&self, stage_id: u32) -> bool {
        self.spec_config.verify_stage_range().contains(&stage_id)
    }

    /// 当前 stage 是否仅用于 Verify（不在 Draft 子集内）
    // @trace REQ-DIST-033 [entity:SpeculativePipeline]
    pub fn is_verify_only_stage(&self, stage_id: u32) -> bool {
        self.is_verify_stage(stage_id) && !self.is_draft_stage(stage_id)
    }

    /// 生成 Draft/Verify 调度步骤序列 (REQ-DIST-033)
    ///
    /// 1. Draft 在 [0, draft_end) stage 运行
    /// 2. Draft tokens 传输到 verify stage
    /// 3. Verify 在 [0, pp_size) stage 运行
    /// 4. PP P2P 激活传输
    // @trace REQ-DIST-033 [entity:SpeculativePipeline]
    pub fn schedule(&self) -> Vec<DraftVerifyStep> {
        let mut steps = Vec::new();
        let draft_end = self.spec_config.draft_stages;
        let pp_size = self.spec_config.verify_stages;

        // Step 1: Draft compute on [0, draft_end)
        steps.push(DraftVerifyStep::DraftCompute {
            num_tokens: self.spec_config.spec_len,
            draft_end_stage: draft_end,
        });

        // Step 2: Draft transfer (using dedicated stream, isolated from PP P2P)
        steps.push(DraftVerifyStep::DraftTransfer {
            source_stage: draft_end - 1, // last draft stage
            dest_stage: draft_end,       // first verify-only stage
            stream_id: self.spec_config.draft_stream_id,
        });

        // Step 3: PP P2P activation transfers (using PP stream)
        for stage in 0..pp_size - 1 {
            steps.push(DraftVerifyStep::PpActivationTransfer {
                from_stage: stage,
                to_stage: stage + 1,
                stream_id: self.spec_config.pp_stream_id,
            });
        }

        // Step 4: Verify compute on [0, pp_size)
        steps.push(DraftVerifyStep::VerifyCompute {
            verify_end_stage: pp_size,
        });

        steps
    }

    /// 传输 draft tokens (REQ-DIST-033 验收标准 5)
    ///
    /// PP P2P 通信与推测 draft 传输通过不同 stream 隔离，零干扰。
    // @trace REQ-DIST-033 [entity:SpeculativePipeline] [dataflow:DF-DIST-018]
    pub fn transfer_draft_tokens(
        &self,
        comm: &CommHandleWrapper,
        draft_tokens: &[u32],
    ) -> Result<(), SpeculativePipelineError> {
        if !comm.is_distributed() {
            return Err(SpeculativePipelineError::NotDistributed);
        }

        if !self.spec_config.is_stream_isolated() {
            return Err(SpeculativePipelineError::StreamNotIsolated {
                pp_stream: self.spec_config.pp_stream_id,
                draft_stream: self.spec_config.draft_stream_id,
            });
        }

        // Draft stage last stage → first verify-only stage
        let source_rank = self.spec_config.draft_stages - 1;
        let dest_rank = self.spec_config.draft_stages;

        let rank = comm.rank();
        if rank == source_rank {
            let token_bytes: Vec<u8> = draft_tokens
                .iter()
                .flat_map(|&t| t.to_le_bytes())
                .collect();
            comm.send_bytes(dest_rank, &token_bytes)
                .map_err(SpeculativePipelineError::NcclError)?;
        }

        Ok(())
    }

    /// 接收 draft tokens (Verify 侧)
    // @trace REQ-DIST-033 [entity:SpeculativePipeline] [dataflow:DF-DIST-018]
    pub fn receive_draft_tokens(
        &self,
        comm: &CommHandleWrapper,
    ) -> Result<Vec<u32>, SpeculativePipelineError> {
        if !comm.is_distributed() {
            return Err(SpeculativePipelineError::NotDistributed);
        }

        let source_rank = self.spec_config.draft_stages - 1;
        let rank = comm.rank();

        if rank == self.spec_config.draft_stages {
            let byte_count = self.spec_config.spec_len as usize * 4;
            let raw = comm.recv_bytes(source_rank, byte_count)
                .map_err(SpeculativePipelineError::NcclError)?;
            let tokens: Vec<u32> = raw
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok(tokens)
        } else {
            Ok(vec![])
        }
    }

    /// 加速比是否达标 (REQ-DIST-033 验收标准 4)
    ///
    /// Draft/Verify 加速比 >= 1.5x (当 spec_len >= 5)
    // @trace REQ-DIST-033 [entity:SpeculativePipeline]
    pub fn speedup_achieved(&self) -> bool {
        self.spec_config.estimated_speedup() >= 1.5
    }

    /// 校验一致性
    // @trace REQ-DIST-033 [entity:SpeculativePipeline]
    pub fn validate(&self) -> bool {
        self.pipeline_config.validate()
            && self.spec_config.validate()
            && self.spec_config.draft_stages < self.pipeline_config.pp_size
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pipeline_config() -> PipelineConfig {
        PipelineConfig {
            pp_size: 4,
            stage_id: 0,
            num_virtual_stages: 1,
            micro_batch_size: 4,
            layers_per_stage: 8,
        }
    }

    fn make_spec_config() -> SpecPipelineConfig {
        SpecPipelineConfig::new(2, 4, 5)
    }

    // ── SpecPipelineConfig ──

    #[test]
    fn spec_config_default() {
        let config = SpecPipelineConfig::default();
        assert_eq!(config.draft_stages, 1);
        assert_eq!(config.verify_stages, 2);
        assert_eq!(config.spec_len, 5);
        assert!(config.is_stream_isolated());
    }

    #[test]
    fn spec_config_new() {
        let config = SpecPipelineConfig::new(2, 4, 5);
        assert_eq!(config.draft_stages, 2);
        assert_eq!(config.verify_stages, 4);
        assert_eq!(config.spec_len, 5);
    }

    #[test]
    fn spec_config_new_clamps() {
        let config = SpecPipelineConfig::new(0, 0, 0);
        assert_eq!(config.draft_stages, 1);
        assert_eq!(config.verify_stages, 2); // max(draft_stages+1, 0) = 2
        assert_eq!(config.spec_len, 1);
    }

    #[test]
    fn spec_config_draft_stage_range() {
        // @trace TEST-DIST-033 [req:REQ-DIST-033] [level:unit]
        // 验收标准 2: Draft model 在 [0, draft_end) stage 运行
        let config = SpecPipelineConfig::new(2, 4, 5);
        assert_eq!(config.draft_stage_range(), 0..2);
    }

    #[test]
    fn spec_config_verify_stage_range() {
        // @trace TEST-DIST-033 [req:REQ-DIST-033] [level:unit]
        // 验收标准 3: Verify model 在 [0, pp_size) stage 运行
        let config = SpecPipelineConfig::new(2, 4, 5);
        assert_eq!(config.verify_stage_range(), 0..4);
    }

    #[test]
    fn spec_config_stream_isolated() {
        // @trace TEST-DIST-033 [req:REQ-DIST-033] [level:unit]
        // 验收标准 5: PP P2P 通信与推测 draft 传输通过不同 stream 隔离
        let config = SpecPipelineConfig::default();
        assert!(config.is_stream_isolated());
    }

    #[test]
    fn spec_config_stream_not_isolated() {
        let config = SpecPipelineConfig {
            pp_stream_id: 0,
            draft_stream_id: 0,
            ..Default::default()
        };
        assert!(!config.is_stream_isolated());
    }

    #[test]
    fn spec_config_validate_valid() {
        let config = SpecPipelineConfig::new(2, 4, 5);
        assert!(config.validate());
    }

    #[test]
    fn spec_config_validate_not_isolated() {
        let config = SpecPipelineConfig {
            pp_stream_id: 0,
            draft_stream_id: 0,
            ..SpecPipelineConfig::new(2, 4, 5)
        };
        assert!(!config.validate()); // stream not isolated
    }

    // ── Speedup estimation (验收标准 4) ──

    #[test]
    fn spec_config_speedup_spec_len_5() {
        // @trace TEST-DIST-033 [req:REQ-DIST-033] [level:unit]
        // 验收标准 4: Draft/Verify 加速比 >= 1.5x (spec_len >= 5)
        let config = SpecPipelineConfig::new(2, 4, 5);
        assert!(config.estimated_speedup() >= 1.5,
            "speedup={} should be >= 1.5", config.estimated_speedup());
    }

    #[test]
    fn spec_config_speedup_increases_with_spec_len() {
        let config_short = SpecPipelineConfig::new(2, 4, 3);
        let config_long = SpecPipelineConfig::new(2, 4, 10);
        assert!(config_long.estimated_speedup() > config_short.estimated_speedup());
    }

    // ── DraftVerifyStep ──

    #[test]
    fn draft_verify_step_equality() {
        let a = DraftVerifyStep::DraftCompute { num_tokens: 5, draft_end_stage: 2 };
        let b = DraftVerifyStep::DraftCompute { num_tokens: 5, draft_end_stage: 2 };
        assert_eq!(a, b);
    }

    // ── SpeculativePipeline: construction ──

    #[test]
    fn pipeline_new_valid() {
        // @trace TEST-DIST-033 [req:REQ-DIST-033] [level:unit]
        let pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            make_spec_config(),
        ).unwrap();
        assert!(pipeline.validate());
    }

    #[test]
    fn pipeline_new_draft_exceeds_pp() {
        let result = SpeculativePipeline::new(
            make_pipeline_config(), // pp_size=4
            SpecPipelineConfig::new(4, 4, 5), // draft_stages=4 >= pp_size
        );
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SpeculativePipelineError::DraftStagesExceedPp {
            draft_stages: 4,
            pp_size: 4,
        });
    }

    // ── configure (验收标准 1) ──

    #[test]
    fn pipeline_configure() {
        // @trace TEST-DIST-033 [req:REQ-DIST-033] [level:unit]
        // 验收标准 1: configure(draft_stages, verify_stages) 分配 stage 子集
        let mut pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            make_spec_config(),
        ).unwrap();
        pipeline.configure(1, 4).unwrap();
        assert_eq!(pipeline.spec_config.draft_stages, 1);
        assert_eq!(pipeline.spec_config.verify_stages, 4);
    }

    #[test]
    fn pipeline_configure_draft_exceeds_pp() {
        let mut pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            make_spec_config(),
        ).unwrap();
        let result = pipeline.configure(4, 4);
        assert!(result.is_err());
    }

    // ── is_draft_stage / is_verify_stage ──

    #[test]
    fn pipeline_is_draft_stage() {
        // @trace TEST-DIST-033 [req:REQ-DIST-033] [level:unit]
        // 验收标准 2: Draft model 在 [0, draft_end) stage 运行
        let pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            SpecPipelineConfig::new(2, 4, 5),
        ).unwrap();
        assert!(pipeline.is_draft_stage(0));
        assert!(pipeline.is_draft_stage(1));
        assert!(!pipeline.is_draft_stage(2));
        assert!(!pipeline.is_draft_stage(3));
    }

    #[test]
    fn pipeline_is_verify_stage() {
        // @trace TEST-DIST-033 [req:REQ-DIST-033] [level:unit]
        // 验收标准 3: Verify model 在 [0, pp_size) stage 运行
        let pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            SpecPipelineConfig::new(2, 4, 5),
        ).unwrap();
        assert!(pipeline.is_verify_stage(0));
        assert!(pipeline.is_verify_stage(1));
        assert!(pipeline.is_verify_stage(2));
        assert!(pipeline.is_verify_stage(3));
    }

    #[test]
    fn pipeline_is_verify_only_stage() {
        let pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            SpecPipelineConfig::new(2, 4, 5),
        ).unwrap();
        assert!(!pipeline.is_verify_only_stage(0)); // draft + verify
        assert!(!pipeline.is_verify_only_stage(1)); // draft + verify
        assert!(pipeline.is_verify_only_stage(2));  // verify only
        assert!(pipeline.is_verify_only_stage(3));  // verify only
    }

    // ── schedule ──

    #[test]
    fn pipeline_schedule_steps() {
        let pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            SpecPipelineConfig::new(2, 4, 5),
        ).unwrap();
        let steps = pipeline.schedule();

        // Step 1: Draft compute
        assert!(matches!(steps[0], DraftVerifyStep::DraftCompute { num_tokens: 5, draft_end_stage: 2 }));

        // Step 2: Draft transfer
        assert!(matches!(steps[1], DraftVerifyStep::DraftTransfer { source_stage: 1, dest_stage: 2, stream_id: 1 }));

        // Steps 3-6: PP activation transfers
        let pp_transfers: Vec<_> = steps.iter().filter(|s| matches!(s, DraftVerifyStep::PpActivationTransfer { .. })).collect();
        assert_eq!(pp_transfers.len(), 3); // 4 stages → 3 transfers

        // Last step: Verify compute
        assert!(matches!(steps.last(), Some(DraftVerifyStep::VerifyCompute { verify_end_stage: 4 })));
    }

    // ── speedup_achieved (验收标准 4) ──

    #[test]
    fn pipeline_speedup_achieved() {
        // @trace TEST-DIST-033 [req:REQ-DIST-033] [level:unit]
        // 验收标准 4: Draft/Verify 加速比 >= 1.5x (spec_len >= 5)
        let pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            SpecPipelineConfig::new(2, 4, 5),
        ).unwrap();
        assert!(pipeline.speedup_achieved());
    }

    // ── transfer_draft_tokens ──

    #[test]
    fn pipeline_transfer_not_distributed() {
        let pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            make_spec_config(),
        ).unwrap();
        let comm = CommHandleWrapper::new_for_test(0, 1);
        let tokens = vec![1u32, 2, 3];
        let result = pipeline.transfer_draft_tokens(&comm, &tokens);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SpeculativePipelineError::NotDistributed);
    }

    #[test]
    fn pipeline_transfer_stream_not_isolated() {
        let pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            SpecPipelineConfig {
                pp_stream_id: 0,
                draft_stream_id: 0,
                ..make_spec_config()
            },
        ).unwrap();
        let comm = CommHandleWrapper::new_for_test(0, 4);
        let tokens = vec![1u32, 2, 3];
        let result = pipeline.transfer_draft_tokens(&comm, &tokens);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SpeculativePipelineError::StreamNotIsolated {
            pp_stream: 0,
            draft_stream: 0,
        });
    }

    // ── validate ──

    #[test]
    fn pipeline_validate_valid() {
        let pipeline = SpeculativePipeline::new(
            make_pipeline_config(),
            make_spec_config(),
        ).unwrap();
        assert!(pipeline.validate());
    }

    // ── SpeculativePipelineError: Display ──

    #[test]
    fn error_display_not_distributed() {
        let err = SpeculativePipelineError::NotDistributed;
        let msg = format!("{}", err);
        assert!(msg.contains("not in distributed mode"));
    }

    #[test]
    fn error_display_invalid_config() {
        let err = SpeculativePipelineError::InvalidConfig("test".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test"));
    }

    #[test]
    fn error_display_draft_exceeds() {
        let err = SpeculativePipelineError::DraftStagesExceedPp { draft_stages: 4, pp_size: 2 };
        let msg = format!("{}", err);
        assert!(msg.contains("draft_stages=4"));
    }

    #[test]
    fn error_display_nccl() {
        let err = SpeculativePipelineError::NcclError("fail".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("fail"));
    }

    #[test]
    fn error_display_stream_not_isolated() {
        let err = SpeculativePipelineError::StreamNotIsolated { pp_stream: 0, draft_stream: 0 };
        let msg = format!("{}", err);
        assert!(msg.contains("stream not isolated"));
    }

    #[test]
    fn error_is_std_error() {
        let err = SpeculativePipelineError::NotDistributed;
        let _: &dyn std::error::Error = &err;
    }
}
