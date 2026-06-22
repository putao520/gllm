//! ActivationTransport — stage 间激活传输 (REQ-DIST-023)
//!
//! 封装 Pipeline Parallel stage 间的激活（activation）P2P 传输：
//! - Forward: Stage i → Stage i+1（前向激活传递）
//! - Backward: Stage i+1 → Stage i（反向梯度传递）
//!
//! 底层使用 CommHandleWrapper::send_f32 / recv_f32 实现 NCCL P2P 通信。
//!
//! nccl feature-gated: 非 nccl 构建零影响。

use crate::engine::distributed_config::CommHandleWrapper;
use super::topology::Topology2D;

// ── ActivationDirection (REQ-DIST-023) ─────────────────────────────────────

/// 激活传输方向 (REQ-DIST-023)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationDirection {
    /// 前向激活传递: Stage i → Stage i+1
    Forward,
    /// 反向梯度传递: Stage i+1 → Stage i
    Backward,
}

impl std::fmt::Display for ActivationDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivationDirection::Forward => write!(f, "Forward"),
            ActivationDirection::Backward => write!(f, "Backward"),
        }
    }
}

// ── ActivationTransport (REQ-DIST-023) ─────────────────────────────────────

/// Stage 间激活传输 (REQ-DIST-023)
///
/// 封装 Pipeline Parallel stage 间 P2P 激活传输逻辑。
/// Forward 方向: Stage i 发送激活到 Stage i+1。
/// Backward 方向: Stage i+1 发送梯度到 Stage i。
///
/// 传输基于 CommHandleWrapper::send_f32 / recv_f32。
/// 缓冲区大小 = micro_batch_size * hidden_size * sizeof(f32)。
// @trace REQ-DIST-023 [entity:ActivationTransport] [api:POST /internal/distributed/pipeline/activation]
#[derive(Debug, Clone)]
pub struct ActivationTransport {
    /// 2D 拓扑（TP+PP），用于推导 P2P 端点
    pub topology: Topology2D,
    /// 微批次大小（token 数）
    pub micro_batch_size: usize,
    /// 隐藏层维度
    pub hidden_size: usize,
    /// 是否启用激活量化（默认 false）
    pub quant_enabled: bool,
}

/// ActivationTransport 错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationTransportError {
    /// CommHandleWrapper 未初始化（非分布式模式）
    NotDistributed,
    /// 当前 stage 无前驱/后继（首/尾 stage）
    NoPeer { direction: ActivationDirection, rank: u32 },
    /// NCCL 通信错误
    NcclError(String),
    /// 缓冲区大小为零
    ZeroBufferSize,
}

impl std::fmt::Display for ActivationTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivationTransportError::NotDistributed => {
                write!(f, "ActivationTransport: not in distributed mode")
            }
            ActivationTransportError::NoPeer { direction, rank } => {
                write!(
                    f,
                    "ActivationTransport: no {:?} peer for rank {}",
                    direction, rank
                )
            }
            ActivationTransportError::NcclError(msg) => {
                write!(f, "ActivationTransport: NCCL error: {}", msg)
            }
            ActivationTransportError::ZeroBufferSize => {
                write!(f, "ActivationTransport: buffer size is zero")
            }
        }
    }
}

impl std::error::Error for ActivationTransportError {}

// @trace REQ-DIST-023 [entity:ActivationTransport]
impl ActivationTransport {
    /// 创建激活传输器
    // @trace REQ-DIST-023 [entity:ActivationTransport]
    pub fn new(
        topology: Topology2D,
        micro_batch_size: usize,
        hidden_size: usize,
    ) -> Result<Self, ActivationTransportError> {
        let buffer_size = micro_batch_size * hidden_size;
        if buffer_size == 0 {
            return Err(ActivationTransportError::ZeroBufferSize);
        }
        Ok(Self {
            topology,
            micro_batch_size,
            hidden_size,
            quant_enabled: false,
        })
    }

    /// 启用激活量化（builder pattern）
    // @trace REQ-DIST-023 [entity:ActivationTransport]
    pub fn with_quant_enabled(mut self, enabled: bool) -> Self {
        self.quant_enabled = enabled;
        self
    }

    /// 激活缓冲区大小（元素数）= micro_batch_size * hidden_size
    // @trace REQ-DIST-023 [entity:ActivationTransport]
    pub fn buffer_element_count(&self) -> usize {
        self.micro_batch_size * self.hidden_size
    }

    /// 激活缓冲区字节数 = micro_batch_size * hidden_size * sizeof(f32)
    // @trace REQ-DIST-023 [entity:ActivationTransport]
    pub fn buffer_byte_size(&self) -> usize {
        self.buffer_element_count() * std::mem::size_of::<f32>()
    }

    /// 发送前向激活到下一个 stage (REQ-DIST-023)
    ///
    /// Forward: Stage i → Stage i+1（同 tp_rank）。
    /// 仅 tp_rank=0 的 rank 实际发送，其他 tp_rank 由 TP broadcast 同步。
    // @trace REQ-DIST-023 [entity:ActivationTransport] [dataflow:DF-DIST-007]
    pub fn send_forward(
        &self,
        comm: &CommHandleWrapper,
        data: &[f32],
    ) -> Result<(), ActivationTransportError> {
        let rank = comm.rank();
        let next_rank = self.topology.pp_next_rank(rank)
            .ok_or_else(|| ActivationTransportError::NoPeer {
                direction: ActivationDirection::Forward,
                rank,
            })?;

        // 仅 TP master rank 执行 P2P 通信
        if !self.topology.is_tp_master(rank) {
            return Ok(());
        }

        comm.send_f32(next_rank, data)
            .map_err(ActivationTransportError::NcclError)
    }

    /// 接收前向激活从上一个 stage (REQ-DIST-023)
    ///
    /// Forward: Stage i 从 Stage i-1 接收激活（同 tp_rank）。
    /// 仅 tp_rank=0 的 rank 实际接收，其他 tp_rank 由 TP broadcast 同步。
    // @trace REQ-DIST-023 [entity:ActivationTransport] [dataflow:DF-DIST-007]
    pub fn recv_forward(
        &self,
        comm: &CommHandleWrapper,
    ) -> Result<Vec<f32>, ActivationTransportError> {
        let rank = comm.rank();
        let prev_rank = self.topology.pp_prev_rank(rank)
            .ok_or_else(|| ActivationTransportError::NoPeer {
                direction: ActivationDirection::Forward,
                rank,
            })?;

        // 仅 TP master rank 执行 P2P 通信
        if !self.topology.is_tp_master(rank) {
            return Ok(Vec::new());
        }

        let count = self.buffer_element_count();
        comm.recv_f32(prev_rank, count)
            .map_err(ActivationTransportError::NcclError)
    }

    /// 发送反向梯度到上一个 stage (REQ-DIST-023)
    ///
    /// Backward: Stage i+1 → Stage i（同 tp_rank）。
    /// 仅 tp_rank=0 的 rank 实际发送，其他 tp_rank 由 TP broadcast 同步。
    // @trace REQ-DIST-023 [entity:ActivationTransport] [dataflow:DF-DIST-008]
    pub fn send_backward(
        &self,
        comm: &CommHandleWrapper,
        data: &[f32],
    ) -> Result<(), ActivationTransportError> {
        let rank = comm.rank();
        let prev_rank = self.topology.pp_prev_rank(rank)
            .ok_or_else(|| ActivationTransportError::NoPeer {
                direction: ActivationDirection::Backward,
                rank,
            })?;

        // 仅 TP master rank 执行 P2P 通信
        if !self.topology.is_tp_master(rank) {
            return Ok(());
        }

        comm.send_f32(prev_rank, data)
            .map_err(ActivationTransportError::NcclError)
    }

    /// 接收反向梯度从下一个 stage (REQ-DIST-023)
    ///
    /// Backward: Stage i 从 Stage i+1 接收梯度（同 tp_rank）。
    /// 仅 tp_rank=0 的 rank 实际接收，其他 tp_rank 由 TP broadcast 同步。
    // @trace REQ-DIST-023 [entity:ActivationTransport] [dataflow:DF-DIST-008]
    pub fn recv_backward(
        &self,
        comm: &CommHandleWrapper,
    ) -> Result<Vec<f32>, ActivationTransportError> {
        let rank = comm.rank();
        let next_rank = self.topology.pp_next_rank(rank)
            .ok_or_else(|| ActivationTransportError::NoPeer {
                direction: ActivationDirection::Backward,
                rank,
            })?;

        // 仅 TP master rank 执行 P2P 通信
        if !self.topology.is_tp_master(rank) {
            return Ok(Vec::new());
        }

        let count = self.buffer_element_count();
        comm.recv_f32(next_rank, count)
            .map_err(ActivationTransportError::NcclError)
    }

    /// 判断当前 rank 是否为首个 PP stage（无需接收前向激活）
    // @trace REQ-DIST-023 [entity:ActivationTransport]
    pub fn is_first_stage(&self, rank: u32) -> bool {
        let (_, pp_rank) = self.topology.decompose_rank(rank);
        pp_rank == 0
    }

    /// 判断当前 rank 是否为末尾 PP stage（无需接收反向梯度）
    // @trace REQ-DIST-023 [entity:ActivationTransport]
    pub fn is_last_stage(&self, rank: u32) -> bool {
        let (_, pp_rank) = self.topology.decompose_rank(rank);
        pp_rank == self.topology.pp_size - 1
    }

    /// 校验传输器一致性
    // @trace REQ-DIST-023 [entity:ActivationTransport]
    pub fn validate(&self) -> bool {
        self.topology.validate()
            && self.micro_batch_size >= 1
            && self.hidden_size >= 1
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_topology() -> Topology2D {
        Topology2D::new(2, 2, 4).unwrap()
    }

    // ── ActivationDirection: Display ──

    #[test]
    fn direction_display() {
        assert_eq!(format!("{}", ActivationDirection::Forward), "Forward");
        assert_eq!(format!("{}", ActivationDirection::Backward), "Backward");
    }

    // ── ActivationTransport: construction ──

    #[test]
    fn new_valid() {
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        assert_eq!(transport.micro_batch_size, 4);
        assert_eq!(transport.hidden_size, 512);
        assert!(!transport.quant_enabled);
    }

    #[test]
    fn new_zero_buffer_size() {
        let topo = make_topology();
        // micro_batch_size=0 → buffer_size=0
        let result = ActivationTransport::new(topo.clone(), 0, 512);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ActivationTransportError::ZeroBufferSize);

        // hidden_size=0 → buffer_size=0
        let result = ActivationTransport::new(topo, 4, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), ActivationTransportError::ZeroBufferSize);
    }

    #[test]
    fn with_quant_enabled() {
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512)
            .unwrap()
            .with_quant_enabled(true);
        assert!(transport.quant_enabled);
    }

    // ── ActivationTransport: buffer sizes ──

    #[test]
    fn buffer_element_count() {
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        assert_eq!(transport.buffer_element_count(), 4 * 512);
    }

    #[test]
    fn buffer_byte_size() {
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        assert_eq!(transport.buffer_byte_size(), 4 * 512 * 4); // f32 = 4 bytes
    }

    // ── ActivationTransport: stage predicates ──

    #[test]
    fn is_first_stage() {
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        // rank 0 (tp=0, pp=0) and rank 2 (tp=1, pp=0) are first stages
        assert!(transport.is_first_stage(0));
        assert!(transport.is_first_stage(2));
        // rank 1 (tp=0, pp=1) and rank 3 (tp=1, pp=1) are not first stages
        assert!(!transport.is_first_stage(1));
        assert!(!transport.is_first_stage(3));
    }

    #[test]
    fn is_last_stage() {
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        // rank 1 (tp=0, pp=1) and rank 3 (tp=1, pp=1) are last stages
        assert!(transport.is_last_stage(1));
        assert!(transport.is_last_stage(3));
        // rank 0 (tp=0, pp=0) and rank 2 (tp=1, pp=0) are not last stages
        assert!(!transport.is_last_stage(0));
        assert!(!transport.is_last_stage(2));
    }

    // ── ActivationTransport: send/recv with non-distributed CommHandleWrapper ──

    #[test]
    fn send_forward_non_distributed_returns_err() {
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        // Non-distributed CommHandleWrapper (world_size=1, single node)
        let comm = CommHandleWrapper::new_for_test(0, 1);
        let data = vec![1.0f32; 4 * 512];
        let result = transport.send_forward(&comm, &data);
        // rank 0 in topology tp=2,pp=2,world=4 but comm has world=1
        // This will fail because comm.is_distributed() returns false
        assert!(result.is_err());
    }

    #[test]
    fn recv_forward_non_distributed_returns_err() {
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        let comm = CommHandleWrapper::new_for_test(0, 1);
        let result = transport.recv_forward(&comm);
        assert!(result.is_err());
    }

    // ── ActivationTransport: send/recv first/last stage ──

    #[test]
    fn send_forward_no_next_peer() {
        // Last stage has no next peer
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        // rank 1 (tp=0, pp=1) = last stage, tp_master
        let comm = CommHandleWrapper::new_for_test(1, 4);
        let data = vec![1.0f32; 4 * 512];
        let result = transport.send_forward(&comm, &data);
        // rank 1 has no next pp_rank → NoPeer error
        assert!(matches!(result, Err(ActivationTransportError::NoPeer { .. })));
    }

    #[test]
    fn recv_forward_no_prev_peer() {
        // First stage has no prev peer
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        // rank 0 (tp=0, pp=0) = first stage, tp_master
        let comm = CommHandleWrapper::new_for_test(0, 4);
        let result = transport.recv_forward(&comm);
        // rank 0 has no prev pp_rank → NoPeer error
        assert!(matches!(result, Err(ActivationTransportError::NoPeer { .. })));
    }

    #[test]
    fn send_backward_no_prev_peer() {
        // First stage has no prev peer for backward
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        let comm = CommHandleWrapper::new_for_test(0, 4);
        let data = vec![1.0f32; 4 * 512];
        let result = transport.send_backward(&comm, &data);
        assert!(matches!(result, Err(ActivationTransportError::NoPeer { .. })));
    }

    #[test]
    fn recv_backward_no_next_peer() {
        // Last stage has no next peer for backward
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        let comm = CommHandleWrapper::new_for_test(1, 4);
        let result = transport.recv_backward(&comm);
        assert!(matches!(result, Err(ActivationTransportError::NoPeer { .. })));
    }

    // ── ActivationTransport: non-TP-master skips P2P ──

    #[test]
    fn send_forward_non_tp_master_skips() {
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        // rank 2 (tp=1, pp=0) = not TP master
        let comm = CommHandleWrapper::new_for_test(2, 4);
        let data = vec![1.0f32; 4 * 512];
        let result = transport.send_forward(&comm, &data);
        // Non-TP-master skips → Ok(())
        assert!(result.is_ok());
    }

    #[test]
    fn recv_forward_non_tp_master_skips() {
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        // rank 3 (tp=1, pp=1) = not TP master
        let comm = CommHandleWrapper::new_for_test(3, 4);
        let result = transport.recv_forward(&comm);
        // Non-TP-master skips → Ok(empty vec)
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    // ── ActivationTransport: validate ──

    #[test]
    fn validate_valid() {
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        assert!(transport.validate());
    }

    // ── ActivationTransportError: Display ──

    #[test]
    fn error_display_not_distributed() {
        let err = ActivationTransportError::NotDistributed;
        let msg = format!("{}", err);
        assert!(msg.contains("not in distributed mode"));
    }

    #[test]
    fn error_display_no_peer() {
        let err = ActivationTransportError::NoPeer {
            direction: ActivationDirection::Forward,
            rank: 0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Forward"));
        assert!(msg.contains("0"));
    }

    #[test]
    fn error_display_nccl() {
        let err = ActivationTransportError::NcclError("test error".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test error"));
    }

    #[test]
    fn error_display_zero_buffer() {
        let err = ActivationTransportError::ZeroBufferSize;
        let msg = format!("{}", err);
        assert!(msg.contains("zero"));
    }

    #[test]
    fn error_is_std_error() {
        let err = ActivationTransportError::NotDistributed;
        let _: &dyn std::error::Error = &err;
    }

    // ── ActivationTransport: clone ──

    #[test]
    fn clone_independence() {
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512)
            .unwrap()
            .with_quant_enabled(true);
        let cloned = transport.clone();
        assert_eq!(cloned.micro_batch_size, 4);
        assert_eq!(cloned.hidden_size, 512);
        assert!(cloned.quant_enabled);
    }
}
