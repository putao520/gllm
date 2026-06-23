//! ActivationTransport — stage 间激活传输 (REQ-DIST-021, REQ-DIST-023)
//!
//! 封装 Pipeline Parallel stage 间的激活（activation）P2P 传输：
//! - Forward: Stage i → Stage i+1（前向激活传递）
//! - Backward: Stage i+1 → Stage i（反向梯度传递）
//!
//! 两种传输模式：
//! - 同步模式：send/recv 阻塞等待完成
//! - 异步模式：send_async/recv_async 返回 CommFuture，支持通信-计算重叠 (REQ-DIST-021)
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

// ── PipelineTransferFuture (REQ-DIST-021) ───────────────────────────────────

/// Pipeline stage 间传输 Future (REQ-DIST-021)
///
/// 封装 `gllm_nccl::CommFuture`，提供 Pipeline Parallel 传输的异步句柄。
/// 支持 `is_done()` 非阻塞轮询和 `wait()` 阻塞等待。
///
/// 用途：通信-计算重叠 (REQ-DIST-026) — 发起异步传输后立即开始本地计算，
/// 传输完成由 `is_done()` 或 `wait()` 确认。
// @trace REQ-DIST-021 [entity:PipelineTransferFuture] [api:POST /internal/distributed/pipeline/transfer-future]
pub struct PipelineTransferFuture {
    /// 底层 NCCL 通信 Future
    inner: Option<gllm_nccl::CommFuture>,
    /// 传输方向
    direction: ActivationDirection,
    /// 传输的微批次索引
    micro_batch_index: usize,
    /// 传输开始时间（用于延迟测量）
    start: std::time::Instant,
}

impl std::fmt::Debug for PipelineTransferFuture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineTransferFuture")
            .field("direction", &self.direction)
            .field("micro_batch_index", &self.micro_batch_index)
            .field("is_done", &self.is_done())
            .finish_non_exhaustive()
    }
}

impl PipelineTransferFuture {
    /// 创建已完成的 Future（同步模式 / 非 TP master rank）
    // @trace REQ-DIST-021 [entity:PipelineTransferFuture]
    pub fn completed(direction: ActivationDirection, micro_batch_index: usize) -> Self {
        Self {
            inner: None,
            direction,
            micro_batch_index,
            start: std::time::Instant::now(),
        }
    }

    /// 创建待完成的 Future（异步模式）
    // @trace REQ-DIST-021 [entity:PipelineTransferFuture]
    pub fn pending(
        future: gllm_nccl::CommFuture,
        direction: ActivationDirection,
        micro_batch_index: usize,
    ) -> Self {
        Self {
            inner: Some(future),
            direction,
            micro_batch_index,
            start: std::time::Instant::now(),
        }
    }

    /// 非阻塞轮询传输是否完成 (REQ-DIST-021 验收标准 1)
    // @trace REQ-DIST-021 [entity:PipelineTransferFuture]
    pub fn is_done(&self) -> bool {
        match &self.inner {
            None => true,
            Some(f) => f.is_done(),
        }
    }

    /// 阻塞等待传输完成 (REQ-DIST-021 验收标准 1)
    ///
    /// 返回传输耗时（微秒）。
    // @trace REQ-DIST-021 [entity:PipelineTransferFuture]
    pub fn wait(&self) -> Result<u64, ActivationTransportError> {
        match &self.inner {
            None => Ok(0),
            Some(f) => {
                f.wait()
                    .map_err(|e| ActivationTransportError::NcclError(format!("{:?}", e)))?;
                let elapsed = self.start.elapsed().as_micros() as u64;
                Ok(elapsed)
            }
        }
    }

    /// 传输方向
    // @trace REQ-DIST-021 [entity:PipelineTransferFuture]
    pub fn direction(&self) -> ActivationDirection {
        self.direction
    }

    /// 微批次索引
    // @trace REQ-DIST-021 [entity:PipelineTransferFuture]
    pub fn micro_batch_index(&self) -> usize {
        self.micro_batch_index
    }
}

// ── ActivationTransport (REQ-DIST-021, REQ-DIST-023) ────────────────────────

/// Stage 间激活传输 (REQ-DIST-021, REQ-DIST-023)
///
/// 封装 Pipeline Parallel stage 间 P2P 激活传输逻辑。
/// Forward 方向: Stage i 发送激活到 Stage i+1。
/// Backward 方向: Stage i+1 发送梯度到 Stage i。
///
/// 两种传输模式：
/// - 同步模式 (send_forward/recv_forward 等): 阻塞等待完成
/// - 异步模式 (send_forward_async/recv_forward_async 等): 返回 PipelineTransferFuture，
///   支持通信-计算重叠 (REQ-DIST-021 验收标准 1: CommHandle.send/recv 返回 Future)
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

    /// 传输延迟上界 = activation_bytes / bandwidth + latency (REQ-DIST-021 验收标准 5)
    ///
    /// 计算单 stage P2P 传输的理论延迟上界。
    /// `bandwidth_bytes_per_us`: 单向带宽（字节/微秒）
    /// `latency_us`: 固定通信延迟（微秒）
    ///
    /// 延迟 = buffer_byte_size / bandwidth + latency
    // @trace REQ-DIST-021 [entity:ActivationTransport] [dataflow:DF-DIST-007]
    pub fn transfer_latency_bound_us(
        &self,
        bandwidth_bytes_per_us: f64,
        latency_us: f64,
    ) -> f64 {
        let activation_bytes = self.buffer_byte_size() as f64;
        activation_bytes / bandwidth_bytes_per_us + latency_us
    }

    /// 量化后传输延迟上界 (REQ-DIST-021 验收标准 5 + 验收标准 6)
    ///
    /// 当启用量化传输时，通信量减少，延迟相应降低。
    /// `quant_elem_bytes`: 量化后每元素字节数（如 FP8=1, INT8=1, FP16=2）
    // @trace REQ-DIST-021 [entity:ActivationTransport] [dataflow:DF-DIST-007]
    pub fn quantized_transfer_latency_bound_us(
        &self,
        bandwidth_bytes_per_us: f64,
        latency_us: f64,
        quant_elem_bytes: usize,
    ) -> f64 {
        let quantized_bytes = self.buffer_element_count() as f64 * quant_elem_bytes as f64;
        quantized_bytes / bandwidth_bytes_per_us + latency_us
    }

    // ── Async P2P methods (REQ-DIST-021 验收标准 1) ──────────────────────

    /// 异步发送前向激活到下一个 stage (REQ-DIST-021 验收标准 1)
    ///
    /// 返回 `PipelineTransferFuture`，支持通信-计算重叠。
    /// 调用方可立即开始本地计算，之后通过 `future.is_done()` / `future.wait()` 确认传输完成。
    ///
    /// Forward: Stage i → Stage i+1（同 tp_rank）。
    /// 仅 tp_rank=0 的 rank 实际发送，其他 tp_rank 返回已完成 Future。
    // @trace REQ-DIST-021 [entity:ActivationTransport] [dataflow:DF-DIST-007]
    pub fn send_forward_async(
        &self,
        comm: &CommHandleWrapper,
        data: &[f32],
        micro_batch_index: usize,
    ) -> Result<PipelineTransferFuture, ActivationTransportError> {
        let rank = comm.rank();
        let next_rank = self.topology.pp_next_rank(rank);

        // 非 TP master rank 跳过 P2P 通信
        if !self.topology.is_tp_master(rank) {
            return Ok(PipelineTransferFuture::completed(
                ActivationDirection::Forward,
                micro_batch_index,
            ));
        }

        let next = next_rank.ok_or_else(|| ActivationTransportError::NoPeer {
            direction: ActivationDirection::Forward,
            rank,
        })?;

        let future = comm.send_f32_async(next, data)
            .map_err(|e| ActivationTransportError::NcclError(e))?;

        Ok(PipelineTransferFuture::pending(
            future,
            ActivationDirection::Forward,
            micro_batch_index,
        ))
    }

    /// 异步接收前向激活从上一个 stage (REQ-DIST-021 验收标准 1)
    ///
    /// 返回 `PipelineTransferFuture`，支持通信-计算重叠。
    /// 调用方提供预分配缓冲区接收数据。
    ///
    /// Forward: Stage i 从 Stage i-1 接收激活（同 tp_rank）。
    /// 仅 tp_rank=0 的 rank 实际接收，其他 tp_rank 返回已完成 Future。
    // @trace REQ-DIST-021 [entity:ActivationTransport] [dataflow:DF-DIST-007]
    pub fn recv_forward_async(
        &self,
        comm: &CommHandleWrapper,
        buf: &mut [f32],
        micro_batch_index: usize,
    ) -> Result<PipelineTransferFuture, ActivationTransportError> {
        let rank = comm.rank();
        let prev_rank = self.topology.pp_prev_rank(rank);

        // 非 TP master rank 跳过 P2P 通信
        if !self.topology.is_tp_master(rank) {
            return Ok(PipelineTransferFuture::completed(
                ActivationDirection::Forward,
                micro_batch_index,
            ));
        }

        let prev = prev_rank.ok_or_else(|| ActivationTransportError::NoPeer {
            direction: ActivationDirection::Forward,
            rank,
        })?;

        let future = comm.recv_f32_async(prev, buf)
            .map_err(ActivationTransportError::NcclError)?;

        Ok(PipelineTransferFuture::pending(
            future,
            ActivationDirection::Forward,
            micro_batch_index,
        ))
    }

    /// 异步发送反向梯度到上一个 stage (REQ-DIST-021 验收标准 1)
    ///
    /// 返回 `PipelineTransferFuture`，支持通信-计算重叠。
    ///
    /// Backward: Stage i+1 → Stage i（同 tp_rank）。
    /// 仅 tp_rank=0 的 rank 实际发送，其他 tp_rank 返回已完成 Future。
    // @trace REQ-DIST-021 [entity:ActivationTransport] [dataflow:DF-DIST-008]
    pub fn send_backward_async(
        &self,
        comm: &CommHandleWrapper,
        data: &[f32],
        micro_batch_index: usize,
    ) -> Result<PipelineTransferFuture, ActivationTransportError> {
        let rank = comm.rank();
        let prev_rank = self.topology.pp_prev_rank(rank);

        // 非 TP master rank 跳过 P2P 通信
        if !self.topology.is_tp_master(rank) {
            return Ok(PipelineTransferFuture::completed(
                ActivationDirection::Backward,
                micro_batch_index,
            ));
        }

        let prev = prev_rank.ok_or_else(|| ActivationTransportError::NoPeer {
            direction: ActivationDirection::Backward,
            rank,
        })?;

        let future = comm.send_f32_async(prev, data)
            .map_err(ActivationTransportError::NcclError)?;

        Ok(PipelineTransferFuture::pending(
            future,
            ActivationDirection::Backward,
            micro_batch_index,
        ))
    }

    /// 异步接收反向梯度从下一个 stage (REQ-DIST-021 验收标准 1)
    ///
    /// 返回 `PipelineTransferFuture`，支持通信-计算重叠。
    /// 调用方提供预分配缓冲区接收数据。
    ///
    /// Backward: Stage i 从 Stage i+1 接收梯度（同 tp_rank）。
    /// 仅 tp_rank=0 的 rank 实际接收，其他 tp_rank 返回已完成 Future。
    // @trace REQ-DIST-021 [entity:ActivationTransport] [dataflow:DF-DIST-008]
    pub fn recv_backward_async(
        &self,
        comm: &CommHandleWrapper,
        buf: &mut [f32],
        micro_batch_index: usize,
    ) -> Result<PipelineTransferFuture, ActivationTransportError> {
        let rank = comm.rank();
        let next_rank = self.topology.pp_next_rank(rank);

        // 非 TP master rank 跳过 P2P 通信
        if !self.topology.is_tp_master(rank) {
            return Ok(PipelineTransferFuture::completed(
                ActivationDirection::Backward,
                micro_batch_index,
            ));
        }

        let next = next_rank.ok_or_else(|| ActivationTransportError::NoPeer {
            direction: ActivationDirection::Backward,
            rank,
        })?;

        let future = comm.recv_f32_async(next, buf)
            .map_err(ActivationTransportError::NcclError)?;

        Ok(PipelineTransferFuture::pending(
            future,
            ActivationDirection::Backward,
            micro_batch_index,
        ))
    }

    /// 异步重叠调度：发起传输 + 立即开始本地计算 (REQ-DIST-021, REQ-DIST-026)
    ///
    /// 典型用法：
    /// 1. `send_forward_async()` 发起激活传输
    /// 2. 立即开始本地计算（下一个微批次的 forward/backward）
    /// 3. `transfer_future.wait()` 等待传输完成
    ///
    /// 这实现了 REQ-DIST-026 的通信-计算重叠：传输和计算在不同 SM 分区并行。
    // @trace REQ-DIST-021 [entity:ActivationTransport] [dataflow:DF-DIST-012]
    pub fn overlap_send_forward_with_compute<F, R>(
        &self,
        comm: &CommHandleWrapper,
        data: &[f32],
        micro_batch_index: usize,
        compute_fn: F,
    ) -> Result<(PipelineTransferFuture, R), ActivationTransportError>
    where
        F: FnOnce() -> R,
    {
        let transfer_future = self.send_forward_async(comm, data, micro_batch_index)?;
        let compute_result = compute_fn();
        Ok((transfer_future, compute_result))
    }

    /// 异步重叠调度：发起接收 + 立即开始本地计算 (REQ-DIST-021, REQ-DIST-026)
    // @trace REQ-DIST-021 [entity:ActivationTransport] [dataflow:DF-DIST-012]
    pub fn overlap_recv_forward_with_compute<F, R>(
        &self,
        comm: &CommHandleWrapper,
        buf: &mut [f32],
        micro_batch_index: usize,
        compute_fn: F,
    ) -> Result<(PipelineTransferFuture, R), ActivationTransportError>
    where
        F: FnOnce() -> R,
    {
        let transfer_future = self.recv_forward_async(comm, buf, micro_batch_index)?;
        let compute_result = compute_fn();
        Ok((transfer_future, compute_result))
    }

    /// 异步重叠调度：发起反向梯度发送 + 立即开始本地计算 (REQ-DIST-021, REQ-DIST-026)
    // @trace REQ-DIST-021 [entity:ActivationTransport] [dataflow:DF-DIST-012]
    pub fn overlap_send_backward_with_compute<F, R>(
        &self,
        comm: &CommHandleWrapper,
        data: &[f32],
        micro_batch_index: usize,
        compute_fn: F,
    ) -> Result<(PipelineTransferFuture, R), ActivationTransportError>
    where
        F: FnOnce() -> R,
    {
        let transfer_future = self.send_backward_async(comm, data, micro_batch_index)?;
        let compute_result = compute_fn();
        Ok((transfer_future, compute_result))
    }

    /// 异步重叠调度：发起反向梯度接收 + 立即开始本地计算 (REQ-DIST-021, REQ-DIST-026)
    // @trace REQ-DIST-021 [entity:ActivationTransport] [dataflow:DF-DIST-012]
    pub fn overlap_recv_backward_with_compute<F, R>(
        &self,
        comm: &CommHandleWrapper,
        buf: &mut [f32],
        micro_batch_index: usize,
        compute_fn: F,
    ) -> Result<(PipelineTransferFuture, R), ActivationTransportError>
    where
        F: FnOnce() -> R,
    {
        let transfer_future = self.recv_backward_async(comm, buf, micro_batch_index)?;
        let compute_result = compute_fn();
        Ok((transfer_future, compute_result))
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

    // ── PipelineTransferFuture: completed ──

    #[test]
    fn transfer_future_completed_is_done() {
        // @trace TEST-DIST-021 [req:REQ-DIST-021] [level:unit]
        let future = PipelineTransferFuture::completed(
            ActivationDirection::Forward,
            0,
        );
        assert!(future.is_done());
        assert_eq!(future.direction(), ActivationDirection::Forward);
        assert_eq!(future.micro_batch_index(), 0);
    }

    #[test]
    fn transfer_future_completed_wait_returns_zero() {
        // @trace TEST-DIST-021 [req:REQ-DIST-021] [level:unit]
        let future = PipelineTransferFuture::completed(
            ActivationDirection::Backward,
            3,
        );
        let elapsed = future.wait().unwrap();
        assert_eq!(elapsed, 0);
    }

    #[test]
    fn transfer_future_debug() {
        let future = PipelineTransferFuture::completed(
            ActivationDirection::Forward,
            1,
        );
        let debug_str = format!("{:?}", future);
        assert!(debug_str.contains("PipelineTransferFuture"));
        assert!(debug_str.contains("Forward"));
    }

    // ── ActivationTransport: async methods (REQ-DIST-021) ──

    #[test]
    fn send_forward_async_non_tp_master_returns_completed() {
        // @trace TEST-DIST-021 [req:REQ-DIST-021] [level:unit]
        // 验收标准 1: CommHandle.send/recv 返回 SendFuture/RecvFuture
        // 非 TP master rank 返回已完成 Future
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        // rank 2 (tp=1, pp=0) = not TP master
        let comm = CommHandleWrapper::new_for_test(2, 4);
        let data = vec![1.0f32; 4 * 512];
        let result = transport.send_forward_async(&comm, &data, 0);
        // Non-TP-master skips → returns completed PipelineTransferFuture
        assert!(result.is_ok());
        let future = result.unwrap();
        assert!(future.is_done());
        assert_eq!(future.direction(), ActivationDirection::Forward);
        assert_eq!(future.micro_batch_index(), 0);
    }

    #[test]
    fn recv_forward_async_non_tp_master_returns_completed() {
        // @trace TEST-DIST-021 [req:REQ-DIST-021] [level:unit]
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        // rank 3 (tp=1, pp=1) = not TP master
        let comm = CommHandleWrapper::new_for_test(3, 4);
        let mut buf = vec![0.0f32; 4 * 512];
        let result = transport.recv_forward_async(&comm, &mut buf, 1);
        assert!(result.is_ok());
        let future = result.unwrap();
        assert!(future.is_done());
        assert_eq!(future.direction(), ActivationDirection::Forward);
        assert_eq!(future.micro_batch_index(), 1);
    }

    #[test]
    fn send_forward_async_non_distributed_returns_err() {
        // Non-distributed CommHandleWrapper → error
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        let comm = CommHandleWrapper::new_for_test(0, 1); // world_size=1, non-distributed
        let data = vec![1.0f32; 4 * 512];
        let result = transport.send_forward_async(&comm, &data, 0);
        assert!(result.is_err());
    }

    #[test]
    fn send_forward_async_no_next_peer() {
        // Last stage has no next peer → NoPeer error
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        let comm = CommHandleWrapper::new_for_test(1, 4); // rank 1 = last stage, tp master
        let data = vec![1.0f32; 4 * 512];
        let result = transport.send_forward_async(&comm, &data, 0);
        assert!(matches!(result, Err(ActivationTransportError::NoPeer { .. })));
    }

    #[test]
    fn recv_forward_async_no_prev_peer() {
        // First stage has no prev peer → NoPeer error
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        let comm = CommHandleWrapper::new_for_test(0, 4); // rank 0 = first stage, tp master
        let mut buf = vec![0.0f32; 4 * 512];
        let result = transport.recv_forward_async(&comm, &mut buf, 0);
        assert!(matches!(result, Err(ActivationTransportError::NoPeer { .. })));
    }

    #[test]
    fn send_backward_async_no_prev_peer() {
        // First stage has no prev peer for backward → NoPeer error
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        let comm = CommHandleWrapper::new_for_test(0, 4); // rank 0 = first stage
        let data = vec![1.0f32; 4 * 512];
        let result = transport.send_backward_async(&comm, &data, 0);
        assert!(matches!(result, Err(ActivationTransportError::NoPeer { .. })));
    }

    #[test]
    fn recv_backward_async_no_next_peer() {
        // Last stage has no next peer for backward → NoPeer error
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        let comm = CommHandleWrapper::new_for_test(1, 4); // rank 1 = last stage
        let mut buf = vec![0.0f32; 4 * 512];
        let result = transport.recv_backward_async(&comm, &mut buf, 0);
        assert!(matches!(result, Err(ActivationTransportError::NoPeer { .. })));
    }

    // ── ActivationTransport: overlap methods (REQ-DIST-021, REQ-DIST-026) ──

    #[test]
    fn overlap_send_forward_with_compute_non_tp_master() {
        // @trace TEST-DIST-021 [req:REQ-DIST-021] [level:unit]
        // 验收标准 1: 通信-计算重叠 — 非 TP master 立即返回已完成 Future
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        let comm = CommHandleWrapper::new_for_test(2, 4); // rank 2 = non-TP-master
        let data = vec![1.0f32; 4 * 512];
        let result = transport.overlap_send_forward_with_compute(
            &comm, &data, 0, || 42,
        );
        assert!(result.is_ok());
        let (future, compute_result) = result.unwrap();
        assert!(future.is_done());
        assert_eq!(compute_result, 42);
    }

    #[test]
    fn overlap_recv_backward_with_compute_non_tp_master() {
        let topo = make_topology(); // tp=2, pp=2
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        let comm = CommHandleWrapper::new_for_test(3, 4); // rank 3 = non-TP-master
        let mut buf = vec![0.0f32; 4 * 512];
        let result = transport.overlap_recv_backward_with_compute(
            &comm, &mut buf, 0, || "computed",
        );
        assert!(result.is_ok());
        let (future, compute_result) = result.unwrap();
        assert!(future.is_done());
        assert_eq!(compute_result, "computed");
    }

    #[test]
    fn overlap_send_forward_with_compute_no_peer_error() {
        // @trace TEST-DIST-026 [req:REQ-DIST-026] [level:unit]
        // Last stage has no next peer for forward send
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        let comm = CommHandleWrapper::new_for_test(1, 4); // rank 1 = last stage, tp master
        let data = vec![1.0f32; 4 * 512];
        let result = transport.overlap_send_forward_with_compute(
            &comm, &data, 0, || 42,
        );
        assert!(matches!(result, Err(ActivationTransportError::NoPeer { .. })));
    }

    // ── REQ-DIST-021 验收标准 5: 延迟边界 ──

    #[test]
    fn transfer_latency_bound_basic() {
        // @trace TEST-DIST-021 [req:REQ-DIST-021] [level:unit]
        // 验收标准 5: 单 stage 传输延迟 < activation_bytes / bandwidth + latency
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        // buffer_byte_size = 4 * 512 * 4 = 8192 bytes
        // bandwidth = 1000 bytes/us = 1 GB/s
        // latency = 5 us
        // bound = 8192 / 1000 + 5 = 13.192 us
        let bound = transport.transfer_latency_bound_us(1000.0, 5.0);
        assert!((bound - 13.192).abs() < 1e-10);
    }

    #[test]
    fn quantized_transfer_latency_bound() {
        // @trace TEST-DIST-021 [req:REQ-DIST-021] [level:unit]
        // 验收标准 5+6: 量化后传输延迟
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        // buffer_byte_size = 8192, quant_elem_bytes = 1 (FP8)
        // quantized_bytes = 4 * 512 * 1 = 2048
        // bandwidth = 1000 bytes/us, latency = 5 us
        // bound = 2048 / 1000 + 5 = 7.048 us
        let bound = transport.quantized_transfer_latency_bound_us(1000.0, 5.0, 1);
        assert!((bound - 7.048).abs() < 1e-10);
    }

    #[test]
    fn quantized_latency_less_than_unquantized() {
        // @trace TEST-DIST-021 [req:REQ-DIST-021] [level:unit]
        let topo = make_topology();
        let transport = ActivationTransport::new(topo, 4, 512).unwrap();
        let unquantized = transport.transfer_latency_bound_us(1000.0, 5.0);
        let quantized = transport.quantized_transfer_latency_bound_us(1000.0, 5.0, 1); // FP8
        assert!(quantized < unquantized);
    }
}
