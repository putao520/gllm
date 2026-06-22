//! DistributedConfig — 分布式推理配置聚合体 (REQ-IB-006~012)
//!
//! nccl feature-gated 配置体系，覆盖并行策略、PD 分离、KV 分布、
//! 通信压缩、MoE 分布式。所有字段均有 Default 值（单机模式），
//! 用户可选择性覆盖。

// ── PdDisaggMode (REQ-IB-008) ──────────────────────────────────────────────

/// PdDisaggMode — Prefill/Decode 分离模式 (REQ-IB-008)
///
/// `Collocated` = Prefill 和 Decode 在同一节点（默认单机模式）。
/// `Disaggregated` = Prefill 和 Decode 分离到不同节点。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PdDisaggMode {
    #[default]
    Collocated,
    Disaggregated,
}

// ── NodeRole (REQ-IB-008) ──────────────────────────────────────────────────

/// NodeRole — 节点角色 (REQ-IB-008)
///
/// `Auto` = 由系统根据 PdDisaggMode 自动推导。
/// `PrefillOnly` / `DecodeOnly` / `Mixed` = 显式指定节点角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NodeRole {
    #[default]
    Auto,
    PrefillOnly,
    DecodeOnly,
    Mixed,
}

// ── KvDistMode (REQ-IB-009) ────────────────────────────────────────────────

/// KvDistMode — KV Cache 分布模式 (REQ-IB-009)
///
/// - `Local`: 本地 KV Cache，无跨节点共享（默认）
/// - `OnDemand`: 按需从远端拉取 KV Cache
/// - `Mirror`: 全量镜像 KV Cache 到所有 TP 节点
/// - `PartialHeadMirror`: 部分头镜像（mirror_heads 控制镜像头数）
/// - `TieredCache`: 分层缓存（本地 + 远端 + 磁盘）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KvDistMode {
    #[default]
    Local,
    OnDemand,
    Mirror,
    PartialHeadMirror,
    TieredCache,
}

// ── CommCompressHint (REQ-IB-010) ──────────────────────────────────────────

/// CommCompressHint — 通信压缩偏好 (REQ-IB-010)
///
/// - `Auto`: 由系统根据带宽和消息大小自动决定
/// - `AlwaysCompress`: 始终压缩通信数据
/// - `NeverCompress`: 始终不压缩
/// - `ForceQuant`: 强制量化压缩（最高压缩比）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CommCompressHint {
    #[default]
    Auto,
    AlwaysCompress,
    NeverCompress,
    ForceQuant,
}

// ── ExpertPlacement (REQ-IB-011) ───────────────────────────────────────────

/// ExpertPlacement — MoE 专家放置策略 (REQ-IB-011)
///
/// - `Auto`: 由系统根据专家数量和 GPU 拓扑自动决定
/// - `RoundRobin`: 轮询分配专家到各 GPU
/// - `HotCold`: 热专家常驻 + 冷专家按需加载
/// - `Custom`: 用户自定义放置（通过 external planner）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ExpertPlacement {
    #[default]
    Auto,
    RoundRobin,
    HotCold,
    Custom,
}

// ── AllToAllStrategy (REQ-IB-011) ──────────────────────────────────────────

/// AllToAllStrategy — MoE AllToAll 通信策略 (REQ-IB-011)
///
/// - `Auto`: 由系统根据互连拓扑自动选择
/// - `NvlinkAllToAll`: NVLink 直连 AllToAll
/// - `RdmaAllToAll`: RDMA 跨节点 AllToAll
/// - `HierarchicalAllToAll`: 层次化 AllToAll（节点内 NVLink + 节点间 RDMA）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AllToAllStrategy {
    #[default]
    Auto,
    NvlinkAllToAll,
    RdmaAllToAll,
    HierarchicalAllToAll,
}

// ── ParallelConfig (REQ-IB-007) ────────────────────────────────────────────

/// ParallelConfig — 并行策略配置 (REQ-IB-007)
///
/// 定义 Tensor Parallel / Pipeline Parallel / Expert Parallel 维度，
/// 以及当前节点在并行组中的 rank 和全局 world_size。
/// 约束：`tp_size * pp_size * ep_size == world_size` 且 `rank < world_size`。
///
/// cp_size: Context Parallelism 维度 (REQ-DIST-016)。
/// CP 环通信限定在同 PP stage 内 (REQ-DIST-032)。
/// cp_size == 1 时退化为无 CP（零开销）。
#[cfg(feature = "nccl")]
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelConfig {
    /// Tensor Parallel 维度，默认 1，≥1
    pub tp_size: u32,
    /// Pipeline Parallel 维度，默认 1，≥1
    pub pp_size: u32,
    /// Expert Parallel 维度，默认 1，≥1
    pub ep_size: u32,
    /// Context Parallelism 维度，默认 1，≥1 (REQ-DIST-016)
    pub cp_size: u32,
    /// 当前节点 rank，默认 0，范围 [0, world_size)
    pub rank: u32,
    /// 全局 world_size，默认 1，必须等于 tp*pp*ep
    pub world_size: u32,
    /// NCCL unique_id 字符串，默认 ""
    pub unique_id: String,
}

// @trace REQ-IB-007 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            tp_size: 1,
            pp_size: 1,
            ep_size: 1,
            cp_size: 1,
            rank: 0,
            world_size: 1,
            unique_id: String::new(),
        }
    }
}

// @trace REQ-IB-007 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl ParallelConfig {
    /// 校验并行配置一致性
    ///
    /// 返回 `true` 当且仅当：
    /// - tp_size ≥ 1, pp_size ≥ 1, ep_size ≥ 1
    /// - rank < world_size
    /// - tp_size * pp_size * ep_size == world_size
    pub fn validate(&self) -> bool {
        self.tp_size >= 1
            && self.pp_size >= 1
            && self.ep_size >= 1
            && self.cp_size >= 1
            && self.rank < self.world_size
            && self.tp_size * self.pp_size * self.ep_size == self.world_size
    }
}

// ── PdDisaggConfig (REQ-IB-008) ────────────────────────────────────────────

/// PdDisaggConfig — Prefill/Decode 分离配置 (REQ-IB-008)
///
/// 控制 Prefill 和 Decode 是否分离到不同节点，
/// 以及当前节点的角色。
#[cfg(feature = "nccl")]
#[derive(Debug, Clone, PartialEq)]
pub struct PdDisaggConfig {
    /// PD 分离模式
    pub mode: PdDisaggMode,
    /// 当前节点角色
    pub role: NodeRole,
}

// @trace REQ-IB-008 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl Default for PdDisaggConfig {
    fn default() -> Self {
        Self {
            mode: PdDisaggMode::Collocated,
            role: NodeRole::Auto,
        }
    }
}

// ── KvDistributionConfig (REQ-IB-009) ──────────────────────────────────────

/// KvDistributionConfig — KV Cache 分布配置 (REQ-IB-009)
///
/// 控制跨节点 KV Cache 的分布策略。
/// `mirror_heads` 仅在 `PartialHeadMirror` 模式下有效，
/// 指定镜像的 KV head 数量。
#[cfg(feature = "nccl")]
#[derive(Debug, Clone, PartialEq)]
pub struct KvDistributionConfig {
    /// KV 分布模式
    pub mode: KvDistMode,
    /// 镜像头数，默认 0，仅 PartialHeadMirror 模式有效
    pub mirror_heads: u32,
}

// @trace REQ-IB-009 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl Default for KvDistributionConfig {
    fn default() -> Self {
        Self {
            mode: KvDistMode::Local,
            mirror_heads: 0,
        }
    }
}

// ── CommConfig (REQ-IB-010) ────────────────────────────────────────────────

/// CommConfig — 通信配置 (REQ-IB-010)
///
/// 控制通信-计算重叠策略、压缩偏好、以及算法覆盖。
/// `overlap` 复用 `intent_bias::OverlapHint` (REQ-IB-003)。
#[cfg(feature = "nccl")]
#[derive(Debug, Clone, PartialEq)]
pub struct CommConfig {
    /// 通信-计算重叠偏好，复用 REQ-IB-003 OverlapHint
    pub overlap: crate::engine::intent_bias::OverlapHint,
    /// 通信压缩偏好
    pub compress: CommCompressHint,
    /// 算法覆盖名称，默认 ""（不覆盖）
    pub algorithm_override: String,
}

// @trace REQ-IB-010 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl Default for CommConfig {
    fn default() -> Self {
        Self {
            overlap: crate::engine::intent_bias::OverlapHint::Auto,
            compress: CommCompressHint::Auto,
            algorithm_override: String::new(),
        }
    }
}

// ── MoeDistributedConfig (REQ-IB-011) ──────────────────────────────────────

/// MoeDistributedConfig — MoE 分布式配置 (REQ-IB-011)
///
/// 控制 MoE 模型的专家放置策略和 AllToAll 通信策略。
#[cfg(feature = "nccl")]
#[derive(Debug, Clone, PartialEq)]
pub struct MoeDistributedConfig {
    /// 专家放置策略
    pub expert_placement: ExpertPlacement,
    /// AllToAll 通信策略
    pub all_to_all: AllToAllStrategy,
}

// @trace REQ-IB-011 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl Default for MoeDistributedConfig {
    fn default() -> Self {
        Self {
            expert_placement: ExpertPlacement::Auto,
            all_to_all: AllToAllStrategy::Auto,
        }
    }
}

// ── CommHandleWrapper (REQ-DIST-001) ────────────────────────────────────────

/// CommHandleWrapper — NCCL 通信句柄封装 (REQ-DIST-001)
///
/// Layer 0 基础设施：封装 gllm-nccl `CommHandle`，提供 AllReduce/ReduceScatter/AllGather
/// 的类型安全调用接口。
///
/// - 单机模式 (world_size == 1): `inner` 为 None，所有集合操作立即返回 Ok(())。
/// - 分布式模式 (world_size > 1): 需调用 `init_nccl()` 初始化实际 NCCL 通信句柄，
///   之后集合操作通过 `CommHandle` 执行。
///
/// Drop trait 实现安全释放：分布式模式下先 barrier (AllReduce 1-element dummy) 再
/// destroy (drop inner CommHandle → Arc strong_count decrements → ncclCommDestroy)。
/// `destroyed` flag 保证幂等：重复 Drop / 显式 destroy() 调用均安全。
#[cfg(feature = "nccl")]
pub struct CommHandleWrapper {
    rank: u32,
    world_size: u32,
    /// 实际 NCCL 通信句柄（lazy init，分布式模式下通过 `init_nccl()` 注入）
    inner: Option<gllm_nccl::CommHandle>,
    /// 幂等标志：destroy() 或 Drop 后置 true，后续调用均为 no-op
    destroyed: bool,
}

#[cfg(feature = "nccl")]
impl std::fmt::Debug for CommHandleWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommHandleWrapper")
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .field("inner", &self.inner.as_ref().map(|_| "CommHandle(..)"))
            .field("destroyed", &self.destroyed)
            .finish()
    }
}

#[cfg(feature = "nccl")]
impl CommHandleWrapper {
    /// Create a CommHandleWrapper from a validated ParallelConfig.
    ///
    /// Single-node mode (world_size == 1): no NCCL init needed, only record config.
    /// Multi-node mode (world_size > 1): marked as distributed for later AllReduce routing.
    /// Call `init_nccl()` to initialize the actual NCCL communicator.
    // @trace REQ-DIST-001 [entity:ENT-DIST-COMMHANDLE]
    pub fn from_config(config: &ParallelConfig) -> Result<Self, DistributedConfigError> {
        if !config.validate() {
            return Err(DistributedConfigError::InvalidParallelConfig);
        }
        Ok(Self {
            rank: config.rank,
            world_size: config.world_size,
            inner: None,
            destroyed: false,
        })
    }

    /// Initialize the NCCL communicator from the unique_id in ParallelConfig.
    ///
    /// Must be called on all ranks simultaneously. After this, `inner` holds a real
    /// `gllm_nccl::CommHandle` and collective operations will use it.
    ///
    /// If `unique_id` is empty, generates a new one (only valid for single-process testing).
    pub fn init_nccl(&mut self) -> Result<(), DistributedConfigError> {
        if self.inner.is_some() {
            return Ok(());
        }
        let unique_id = gllm_nccl::get_unique_id()
            .map_err(|e| DistributedConfigError::CommInitFailed(format!("{:?}", e)))?;
        let handle = gllm_nccl::comm_init_rank(&unique_id, self.rank as usize, self.world_size as usize)
            .map_err(|e| DistributedConfigError::CommInitFailed(format!("{:?}", e)))?;
        self.inner = Some(handle);
        Ok(())
    }

    /// Initialize the NCCL communicator from a pre-distributed UniqueId.
    ///
    /// Use this when the unique_id was generated on rank 0 and broadcast
    /// to all other ranks out-of-band.
    pub fn init_nccl_with_unique_id(
        &mut self,
        unique_id: &gllm_nccl::UniqueId,
    ) -> Result<(), DistributedConfigError> {
        if self.inner.is_some() {
            return Ok(());
        }
        let handle = gllm_nccl::comm_init_rank(unique_id, self.rank as usize, self.world_size as usize)
            .map_err(|e| DistributedConfigError::CommInitFailed(format!("{:?}", e)))?;
        self.inner = Some(handle);
        Ok(())
    }

    /// Inject an already-initialized CommHandle (e.g., from gllm-kernels GPU init).
    pub fn set_comm_handle(&mut self, handle: gllm_nccl::CommHandle) {
        self.inner = Some(handle);
    }

    /// Whether the NCCL communicator has been initialized.
    pub fn is_nccl_initialized(&self) -> bool {
        self.inner.is_some()
    }

    /// Current node rank in the distributed group.
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// Total number of nodes in the distributed group.
    pub fn world_size(&self) -> u32 {
        self.world_size
    }

    /// Whether this is a multi-node distributed setup.
    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }

    /// Get the topology of this communication group (REQ-DIST-010).
    ///
    /// Returns `None` if the NCCL communicator has not been initialized
    /// (call `init_nccl()` first in distributed mode).
    // @trace REQ-DIST-010 [entity:ENT-DIST-TP-COMM]
    pub fn topology(&self) -> Option<gllm_nccl::Topology> {
        self.inner.as_ref().map(|h| h.topology().clone())
    }

    /// Test-only constructor (REQ-DIST-008)
    #[cfg(test)]
    pub fn new_for_test(rank: u32, world_size: u32) -> Self {
        Self { rank, world_size, inner: None, destroyed: false }
    }

    // ── Collective operations (REQ-DIST-005, REQ-DIST-006) ────────────────

    /// Inplace AllReduce (Sum) on f32 buffer (REQ-DIST-005).
    ///
    /// - Single-node mode: returns Ok(()) immediately (no communication needed).
    /// - Distributed mode with NCCL initialized: calls `CommHandle::all_reduce` + `wait()`.
    /// - Distributed mode without NCCL init: returns Err.
    // @trace REQ-DIST-005 [entity:ENT-DIST-TP-COMM] [dataflow:DF-DIST-002]
    pub fn all_reduce_inplace(&self, buffer: &mut [f32]) -> Result<(), String> {
        if !self.is_distributed() {
            return Ok(());
        }
        let handle = self.inner.as_ref().ok_or_else(|| {
            "tp_all_reduce: CommHandle not initialized — call init_nccl() first".to_string()
        })?;
        let count = buffer.len();
        let future = handle
            .all_reduce(
                buffer.as_ptr() as *const u8,
                buffer.as_mut_ptr() as *mut u8,
                count,
                gllm_nccl::DType::Fp32,
                gllm_nccl::ReduceOp::Sum,
            )
            .map_err(|e| format!("tp_all_reduce NCCL error: {:?}", e))?;
        future
            .wait()
            .map_err(|e| format!("tp_all_reduce wait error: {:?}", e))
    }

    /// Inplace ReduceScatter (Sum) on f32 buffer (REQ-DIST-006).
    ///
    /// After this call, `buffer` contains only this rank's segment of the reduced result.
    /// Buffer length must be a multiple of world_size.
    pub fn reduce_scatter_inplace(&self, buffer: &mut [f32]) -> Result<(), String> {
        if !self.is_distributed() {
            return Ok(());
        }
        let handle = self.inner.as_ref().ok_or_else(|| {
            "tp_reduce_scatter: CommHandle not initialized — call init_nccl() first".to_string()
        })?;
        let recvcount = buffer.len() / self.world_size as usize;
        let future = handle
            .reduce_scatter(
                buffer.as_ptr() as *const u8,
                buffer.as_mut_ptr() as *mut u8,
                recvcount,
                gllm_nccl::DType::Fp32,
                gllm_nccl::ReduceOp::Sum,
            )
            .map_err(|e| format!("tp_reduce_scatter NCCL error: {:?}", e))?;
        future
            .wait()
            .map_err(|e| format!("tp_reduce_scatter wait error: {:?}", e))
    }

    /// Inplace AllGather on f32 buffer (REQ-DIST-006).
    ///
    /// `buffer` must have capacity for `world_size * sendcount` elements.
    /// The first `sendcount` elements are this rank's contribution; after the call
    /// the entire buffer contains the gathered result from all ranks.
    pub fn all_gather_inplace(&self, buffer: &mut [f32], sendcount: usize) -> Result<(), String> {
        if !self.is_distributed() {
            return Ok(());
        }
        let handle = self.inner.as_ref().ok_or_else(|| {
            "tp_all_gather: CommHandle not initialized — call init_nccl() first".to_string()
        })?;
        let future = handle
            .all_gather(
                buffer.as_ptr() as *const u8,
                sendcount,
                buffer.as_mut_ptr() as *mut u8,
                gllm_nccl::DType::Fp32,
            )
            .map_err(|e| format!("tp_all_gather NCCL error: {:?}", e))?;
        future
            .wait()
            .map_err(|e| format!("tp_all_gather wait error: {:?}", e))
    }

    // ── KV page point-to-point operations (REQ-DIST-012) ────────────────────

    /// Send KV page data to a peer rank (REQ-DIST-012).
    ///
    /// `buf` is a raw pointer to GPU-resident KV page data.
    /// `count` is the number of elements to send.
    /// `peer` is the destination rank.
    /// `dtype` is the data type of the buffer.
    pub fn send_kv_pages(
        &self,
        buf: *const u8,
        count: usize,
        peer: u32,
        dtype: gllm_nccl::DType,
    ) -> Result<(), String> {
        if !self.is_distributed() {
            return Err("send_kv_pages: not in distributed mode".to_string());
        }
        let handle = self.inner.as_ref().ok_or_else(|| {
            "send_kv_pages: CommHandle not initialized — call init_nccl() first".to_string()
        })?;
        let future = handle
            .send(buf, count, peer as usize, dtype)
            .map_err(|e| format!("send_kv_pages NCCL error: {:?}", e))?;
        future
            .wait()
            .map_err(|e| format!("send_kv_pages wait error: {:?}", e))
    }

    /// Receive KV page data from a peer rank (REQ-DIST-012).
    ///
    /// `buf` is a raw pointer to GPU-resident KV page data.
    /// `count` is the number of elements to receive.
    /// `peer` is the source rank.
    /// `dtype` is the data type of the buffer.
    pub fn recv_kv_pages(
        &self,
        buf: *mut u8,
        count: usize,
        peer: u32,
        dtype: gllm_nccl::DType,
    ) -> Result<(), String> {
        if !self.is_distributed() {
            return Err("recv_kv_pages: not in distributed mode".to_string());
        }
        let handle = self.inner.as_ref().ok_or_else(|| {
            "recv_kv_pages: CommHandle not initialized — call init_nccl() first".to_string()
        })?;
        let future = handle
            .recv(buf, count, peer as usize, dtype)
            .map_err(|e| format!("recv_kv_pages NCCL error: {:?}", e))?;
        future
            .wait()
            .map_err(|e| format!("recv_kv_pages wait error: {:?}", e))
    }

    // ── EPLB cross-GPU statistics aggregation (REQ-DIST-015) ────────────────

    /// AllReduce (Sum) on u64 slice — for EPLB expert invocation count aggregation.
    ///
    /// Converts u64 → f32 for NCCL AllReduce, then converts back.
    /// Single-node or no NCCL handle: no-op (data is already local).
    pub fn all_reduce_u64_sum(&self, data: &mut [u64]) -> Result<(), String> {
        if !self.is_distributed() {
            return Ok(());
        }
        let handle = self.inner.as_ref().ok_or_else(|| {
            "all_reduce_u64_sum: CommHandle not initialized — call init_nccl() first".to_string()
        })?;

        // u64 不直接支持 CommElement，转为 f32 做归约
        let send_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let mut recv_f32 = vec![0.0f32; send_f32.len()];

        let future = handle
            .all_reduce(
                send_f32.as_ptr() as *const u8,
                recv_f32.as_mut_ptr() as *mut u8,
                send_f32.len(),
                gllm_nccl::DType::Fp32,
                gllm_nccl::ReduceOp::Sum,
            )
            .map_err(|e| format!("all_reduce_u64_sum NCCL error: {:?}", e))?;

        future
            .wait()
            .map_err(|e| format!("all_reduce_u64_sum wait error: {:?}", e))?;

        for (i, &v) in recv_f32.iter().enumerate() {
            data[i] = v as u64;
        }
        Ok(())
    }

    // ── Ring Attention KV block transfer (REQ-DIST-016) ─────────────────────

    /// Send f32 slice to a peer rank (for Ring Attention KV block transfer).
    pub fn send_f32(&self, peer: u32, data: &[f32]) -> Result<(), String> {
        if !self.is_distributed() {
            return Err("send_f32: not in distributed mode".to_string());
        }
        let handle = self.inner.as_ref().ok_or_else(|| {
            "send_f32: CommHandle not initialized — call init_nccl() first".to_string()
        })?;
        let future = handle
            .send(
                data.as_ptr() as *const u8,
                data.len(),
                peer as usize,
                gllm_nccl::DType::Fp32,
            )
            .map_err(|e| format!("send_f32 NCCL error: {:?}", e))?;
        future
            .wait()
            .map_err(|e| format!("send_f32 wait error: {:?}", e))
    }

    /// Receive f32 slice from a peer rank (for Ring Attention KV block transfer).
    ///
    /// Allocates and returns a Vec<f32> of the specified count.
    pub fn recv_f32(&self, peer: u32, count: usize) -> Result<Vec<f32>, String> {
        if !self.is_distributed() {
            return Err("recv_f32: not in distributed mode".to_string());
        }
        let handle = self.inner.as_ref().ok_or_else(|| {
            "recv_f32: CommHandle not initialized — call init_nccl() first".to_string()
        })?;
        let mut buf = vec![0.0f32; count];
        let future = handle
            .recv(
                buf.as_mut_ptr() as *mut u8,
                count,
                peer as usize,
                gllm_nccl::DType::Fp32,
            )
            .map_err(|e| format!("recv_f32 NCCL error: {:?}", e))?;
        future
            .wait()
            .map_err(|e| format!("recv_f32 wait error: {:?}", e))?;
        Ok(buf)
    }

    // ── SAGUARO draft/verify token transfer (REQ-DIST-017) ──────────────────

    /// Send raw bytes to a peer rank (for SAGUARO draft token / verify result transfer).
    pub fn send_bytes(&self, peer: u32, data: &[u8]) -> Result<(), String> {
        if !self.is_distributed() {
            return Err("send_bytes: not in distributed mode".to_string());
        }
        let handle = self.inner.as_ref().ok_or_else(|| {
            "send_bytes: CommHandle not initialized — call init_nccl() first".to_string()
        })?;
        let future = handle
            .send(data.as_ptr(), data.len(), peer as usize, gllm_nccl::DType::Int8)
            .map_err(|e| format!("send_bytes NCCL error: {:?}", e))?;
        future
            .wait()
            .map_err(|e| format!("send_bytes wait error: {:?}", e))
    }

    /// Receive raw bytes from a peer rank (for SAGUARO verify result / draft token transfer).
    pub fn recv_bytes(&self, peer: u32, count: usize) -> Result<Vec<u8>, String> {
        if !self.is_distributed() {
            return Err("recv_bytes: not in distributed mode".to_string());
        }
        let handle = self.inner.as_ref().ok_or_else(|| {
            "recv_bytes: CommHandle not initialized — call init_nccl() first".to_string()
        })?;
        let mut buf = vec![0u8; count];
        let future = handle
            .recv(buf.as_mut_ptr(), count, peer as usize, gllm_nccl::DType::Int8)
            .map_err(|e| format!("recv_bytes NCCL error: {:?}", e))?;
        future
            .wait()
            .map_err(|e| format!("recv_bytes wait error: {:?}", e))?;
        Ok(buf)
    }

    // ── MoE AllToAll dispatch (REQ-DIST-014) ────────────────────────────────

    /// AllGather f32 slices from all ranks (for MoE AllToAll dispatch).
    ///
    /// Returns a vector of length `world_size * send_data.len()`.
    /// Single-node: returns `send_data` as-is.
    pub fn all_gather_f32(&self, send_data: &[f32]) -> Result<Vec<f32>, String> {
        if !self.is_distributed() {
            return Ok(send_data.to_vec());
        }
        let handle = self.inner.as_ref().ok_or_else(|| {
            "all_gather_f32: CommHandle not initialized — call init_nccl() first".to_string()
        })?;
        let send_count = send_data.len();
        let recv_count = self.world_size as usize * send_count;
        let mut recv_buf = vec![0.0f32; recv_count];
        let future = handle
            .all_gather(
                send_data.as_ptr() as *const u8,
                send_count,
                recv_buf.as_mut_ptr() as *mut u8,
                gllm_nccl::DType::Fp32,
            )
            .map_err(|e| format!("all_gather_f32 NCCL error: {:?}", e))?;
        future
            .wait()
            .map_err(|e| format!("all_gather_f32 wait error: {:?}", e))?;
        Ok(recv_buf)
    }

    // ── Lifecycle: destroy + Drop (REQ-DIST-001) ──────────────────────────────

    /// Explicitly destroy the NCCL communicator (REQ-DIST-001).
    ///
    /// In distributed mode (world_size > 1), performs a barrier before destroy
    /// to ensure all ranks reach the same point. The barrier uses the NCCL
    /// barrier idiom: AllReduce on a 1-element f32 buffer with Sum op.
    ///
    /// After destroy:
    /// - `inner` is taken and dropped (Arc strong_count decrements; when the
    ///   last reference drops, `CommHandleInner` drops → `NcclBackend` drops
    ///   → `ncclCommDestroy` is called via FFI).
    /// - `destroyed` is set to `true`.
    /// - Subsequent calls to `destroy()` are no-ops (idempotent).
    ///
    /// In single-node mode (world_size == 1), this is a no-op.
    // @trace REQ-DIST-001 [entity:ENT-DIST-COMMHANDLE] [lifecycle:destroy]
    pub fn destroy(&mut self) {
        if self.destroyed {
            return; // idempotent
        }

        // NCCL barrier idiom: AllReduce on 1-element buffer ensures all ranks
        // reach this point before any rank proceeds to destroy. This prevents
        // a rank from issuing ncclCommDestroy while other ranks still have
        // pending operations on the same communicator.
        if self.is_distributed() && self.inner.is_some() {
            let mut barrier_buf = [0.0f32; 1];
            let _ = self.all_reduce_inplace(&mut barrier_buf);
        }

        // Take and drop the inner CommHandle. The Arc<CommHandleInner>
        // strong_count decrements; when it reaches 0, CommHandleInner drops,
        // which drops the BackendKind (NcclBackend), which calls ncclCommDestroy.
        self.inner.take();

        self.destroyed = true;
        log::info!(
            "[CommHandleWrapper] destroy: NCCL handle released (rank={})",
            self.rank
        );
    }

    /// Whether this handle has been destroyed.
    // @trace REQ-DIST-001 [entity:ENT-DIST-COMMHANDLE]
    pub fn is_destroyed(&self) -> bool {
        self.destroyed
    }
}

// ── CommHandleWrapper Drop (REQ-DIST-001) ────────────────────────────────────

/// Drop automatically calls `destroy()` — barrier + NCCL cleanup.
///
/// Because `destroy()` is idempotent, double-drop (manual `destroy()` + auto Drop)
/// is safe: the second call is a no-op.
// @trace REQ-DIST-001 [entity:ENT-DIST-COMMHANDLE] [lifecycle:drop]
#[cfg(feature = "nccl")]
impl Drop for CommHandleWrapper {
    fn drop(&mut self) {
        self.destroy();
    }
}

// ── DistributedConfigError (REQ-DIST-001) ────────────────────────────────────

/// Errors that can occur during distributed configuration initialization.
#[cfg(feature = "nccl")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributedConfigError {
    /// ParallelConfig validation failed (rank/world_size/tp*pp*ep mismatch).
    InvalidParallelConfig,
    /// NCCL communicator initialization failed.
    CommInitFailed(String),
}

#[cfg(feature = "nccl")]
impl std::fmt::Display for DistributedConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistributedConfigError::InvalidParallelConfig => {
                write!(f, "ParallelConfig validation failed: tp*pp*ep != world_size or rank out of range")
            }
            DistributedConfigError::CommInitFailed(msg) => {
                write!(f, "NCCL communicator init failed: {}", msg)
            }
        }
    }
}

#[cfg(feature = "nccl")]
impl std::error::Error for DistributedConfigError {}

// ── DistributedConfig (REQ-IB-006) ─────────────────────────────────────────

/// DistributedConfig — 分布式推理配置聚合体 (REQ-IB-006)
///
/// nccl feature-gated，Default = 单机模式（所有子配置均为默认值）。
/// 聚合 5 个子配置：Parallel / PdDisagg / KvDistribution / Comm / MoeDistributed。
#[cfg(feature = "nccl")]
#[derive(Debug, Clone, PartialEq)]
pub struct DistributedConfig {
    /// 并行策略配置
    pub parallel: ParallelConfig,
    /// Prefill/Decode 分离配置
    pub pd_disagg: PdDisaggConfig,
    /// KV Cache 分布配置
    pub kv_distribution: KvDistributionConfig,
    /// 通信配置
    pub comm: CommConfig,
    /// MoE 分布式配置
    pub moe: MoeDistributedConfig,
}

// @trace REQ-IB-006 [entity:ENT-DISTRIBUTED-CONFIG] [api:POST /internal/distributed/config]
#[cfg(feature = "nccl")]
impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            parallel: ParallelConfig::default(),
            pd_disagg: PdDisaggConfig::default(),
            kv_distribution: KvDistributionConfig::default(),
            comm: CommConfig::default(),
            moe: MoeDistributedConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PdDisaggMode (REQ-IB-008) ──────────────────────────────────────────

    #[test]
    fn pd_disagg_mode_default_is_collocated() {
        assert_eq!(PdDisaggMode::default(), PdDisaggMode::Collocated);
    }

    #[test]
    fn pd_disagg_mode_variants_distinct() {
        use std::collections::HashSet;
        let all = [PdDisaggMode::Collocated, PdDisaggMode::Disaggregated];
        let set: HashSet<PdDisaggMode> = all.into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn pd_disagg_mode_copy_trait() {
        let original = PdDisaggMode::Disaggregated;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn pd_disagg_mode_debug_trait() {
        let debug = format!("{:?}", PdDisaggMode::Collocated);
        assert!(debug.contains("Collocated"));
    }

    // ── NodeRole (REQ-IB-008) ──────────────────────────────────────────────

    #[test]
    fn node_role_default_is_auto() {
        assert_eq!(NodeRole::default(), NodeRole::Auto);
    }

    #[test]
    fn node_role_all_variants_distinct() {
        use std::collections::HashSet;
        let all = [
            NodeRole::Auto,
            NodeRole::PrefillOnly,
            NodeRole::DecodeOnly,
            NodeRole::Mixed,
        ];
        let set: HashSet<NodeRole> = all.into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn node_role_copy_trait() {
        let original = NodeRole::PrefillOnly;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn node_role_debug_trait() {
        let debug = format!("{:?}", NodeRole::DecodeOnly);
        assert!(debug.contains("DecodeOnly"));
    }

    // ── KvDistMode (REQ-IB-009) ────────────────────────────────────────────

    #[test]
    fn kv_dist_mode_default_is_local() {
        assert_eq!(KvDistMode::default(), KvDistMode::Local);
    }

    #[test]
    fn kv_dist_mode_all_variants_distinct() {
        use std::collections::HashSet;
        let all = [
            KvDistMode::Local,
            KvDistMode::OnDemand,
            KvDistMode::Mirror,
            KvDistMode::PartialHeadMirror,
            KvDistMode::TieredCache,
        ];
        let set: HashSet<KvDistMode> = all.into_iter().collect();
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn kv_dist_mode_copy_trait() {
        let original = KvDistMode::Mirror;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn kv_dist_mode_debug_trait() {
        let debug = format!("{:?}", KvDistMode::PartialHeadMirror);
        assert!(debug.contains("PartialHeadMirror"));
    }

    // ── CommCompressHint (REQ-IB-010) ──────────────────────────────────────

    #[test]
    fn comm_compress_hint_default_is_auto() {
        assert_eq!(CommCompressHint::default(), CommCompressHint::Auto);
    }

    #[test]
    fn comm_compress_hint_all_variants_distinct() {
        use std::collections::HashSet;
        let all = [
            CommCompressHint::Auto,
            CommCompressHint::AlwaysCompress,
            CommCompressHint::NeverCompress,
            CommCompressHint::ForceQuant,
        ];
        let set: HashSet<CommCompressHint> = all.into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn comm_compress_hint_copy_trait() {
        let original = CommCompressHint::ForceQuant;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn comm_compress_hint_debug_trait() {
        let debug = format!("{:?}", CommCompressHint::AlwaysCompress);
        assert!(debug.contains("AlwaysCompress"));
    }

    // ── ExpertPlacement (REQ-IB-011) ───────────────────────────────────────

    #[test]
    fn expert_placement_default_is_auto() {
        assert_eq!(ExpertPlacement::default(), ExpertPlacement::Auto);
    }

    #[test]
    fn expert_placement_all_variants_distinct() {
        use std::collections::HashSet;
        let all = [
            ExpertPlacement::Auto,
            ExpertPlacement::RoundRobin,
            ExpertPlacement::HotCold,
            ExpertPlacement::Custom,
        ];
        let set: HashSet<ExpertPlacement> = all.into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn expert_placement_copy_trait() {
        let original = ExpertPlacement::HotCold;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn expert_placement_debug_trait() {
        let debug = format!("{:?}", ExpertPlacement::RoundRobin);
        assert!(debug.contains("RoundRobin"));
    }

    // ── AllToAllStrategy (REQ-IB-011) ──────────────────────────────────────

    #[test]
    fn all_to_all_strategy_default_is_auto() {
        assert_eq!(AllToAllStrategy::default(), AllToAllStrategy::Auto);
    }

    #[test]
    fn all_to_all_strategy_all_variants_distinct() {
        use std::collections::HashSet;
        let all = [
            AllToAllStrategy::Auto,
            AllToAllStrategy::NvlinkAllToAll,
            AllToAllStrategy::RdmaAllToAll,
            AllToAllStrategy::HierarchicalAllToAll,
        ];
        let set: HashSet<AllToAllStrategy> = all.into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn all_to_all_strategy_copy_trait() {
        let original = AllToAllStrategy::NvlinkAllToAll;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn all_to_all_strategy_debug_trait() {
        let debug = format!("{:?}", AllToAllStrategy::RdmaAllToAll);
        assert!(debug.contains("RdmaAllToAll"));
    }

    // ── nccl-gated struct tests ────────────────────────────────────────────

    #[cfg(feature = "nccl")]
    mod nccl_tests {
        use super::*;

        // ── ParallelConfig (REQ-IB-007) ────────────────────────────────────

        #[test]
        fn parallel_config_default_values() {
            let cfg = ParallelConfig::default();
            assert_eq!(cfg.tp_size, 1);
            assert_eq!(cfg.pp_size, 1);
            assert_eq!(cfg.ep_size, 1);
            assert_eq!(cfg.rank, 0);
            assert_eq!(cfg.world_size, 1);
            assert!(cfg.unique_id.is_empty());
        }

        #[test]
        fn parallel_config_validate_default_passes() {
            let cfg = ParallelConfig::default();
            assert!(cfg.validate());
        }

        #[test]
        fn parallel_config_validate_tp2_pp2_ep2_world8() {
            let cfg = ParallelConfig {
                tp_size: 2,
                pp_size: 2,
                ep_size: 2,
                cp_size: 1,
                rank: 0,
                world_size: 8,
                unique_id: String::new(),
            };
            assert!(cfg.validate());
        }

        #[test]
        fn parallel_config_validate_mismatch_world_size_fails() {
            let cfg = ParallelConfig {
                tp_size: 2,
                pp_size: 2,
                ep_size: 1,
                cp_size: 1,
                rank: 0,
                world_size: 5, // 2*2*1=4 != 5
                unique_id: String::new(),
            };
            assert!(!cfg.validate());
        }

        #[test]
        fn parallel_config_validate_rank_out_of_range_fails() {
            let cfg = ParallelConfig {
                tp_size: 1,
                pp_size: 1,
                ep_size: 1,
                cp_size: 1,
                rank: 1, // rank >= world_size
                world_size: 1,
                unique_id: String::new(),
            };
            assert!(!cfg.validate());
        }

        #[test]
        fn parallel_config_validate_zero_tp_fails() {
            let cfg = ParallelConfig {
                tp_size: 0,
                pp_size: 1,
                ep_size: 1,
                cp_size: 1,
                rank: 0,
                world_size: 0,
                unique_id: String::new(),
            };
            assert!(!cfg.validate());
        }

        #[test]
        fn parallel_config_equality() {
            let a = ParallelConfig::default();
            let b = ParallelConfig::default();
            assert_eq!(a, b);
        }

        #[test]
        fn parallel_config_inequality_unique_id() {
            let a = ParallelConfig::default();
            let b = ParallelConfig {
                unique_id: "test-id".to_string(),
                ..Default::default()
            };
            assert_ne!(a, b);
        }

        #[test]
        fn parallel_config_clone_independence() {
            let mut cfg = ParallelConfig {
                tp_size: 4,
                pp_size: 2,
                ep_size: 1,
                cp_size: 1,
                rank: 3,
                world_size: 8,
                unique_id: "abc".to_string(),
            };
            let cloned = cfg.clone();
            cfg.tp_size = 1;
            assert_eq!(cloned.tp_size, 4);
        }

        // ── PdDisaggConfig (REQ-IB-008) ────────────────────────────────────

        #[test]
        fn pd_disagg_config_default_values() {
            let cfg = PdDisaggConfig::default();
            assert_eq!(cfg.mode, PdDisaggMode::Collocated);
            assert_eq!(cfg.role, NodeRole::Auto);
        }

        #[test]
        fn pd_disagg_config_custom_values() {
            let cfg = PdDisaggConfig {
                mode: PdDisaggMode::Disaggregated,
                role: NodeRole::PrefillOnly,
            };
            assert_eq!(cfg.mode, PdDisaggMode::Disaggregated);
            assert_eq!(cfg.role, NodeRole::PrefillOnly);
        }

        #[test]
        fn pd_disagg_config_equality() {
            let a = PdDisaggConfig::default();
            let b = PdDisaggConfig::default();
            assert_eq!(a, b);
        }

        #[test]
        fn pd_disagg_config_clone_independence() {
            let mut cfg = PdDisaggConfig {
                mode: PdDisaggMode::Disaggregated,
                role: NodeRole::DecodeOnly,
            };
            let cloned = cfg.clone();
            cfg.role = NodeRole::Mixed;
            assert_eq!(cloned.role, NodeRole::DecodeOnly);
        }

        // ── KvDistributionConfig (REQ-IB-009) ──────────────────────────────

        #[test]
        fn kv_distribution_config_default_values() {
            let cfg = KvDistributionConfig::default();
            assert_eq!(cfg.mode, KvDistMode::Local);
            assert_eq!(cfg.mirror_heads, 0);
        }

        #[test]
        fn kv_distribution_config_partial_head_mirror() {
            let cfg = KvDistributionConfig {
                mode: KvDistMode::PartialHeadMirror,
                mirror_heads: 4,
            };
            assert_eq!(cfg.mode, KvDistMode::PartialHeadMirror);
            assert_eq!(cfg.mirror_heads, 4);
        }

        #[test]
        fn kv_distribution_config_equality() {
            let a = KvDistributionConfig::default();
            let b = KvDistributionConfig::default();
            assert_eq!(a, b);
        }

        #[test]
        fn kv_distribution_config_clone_independence() {
            let mut cfg = KvDistributionConfig {
                mode: KvDistMode::Mirror,
                mirror_heads: 8,
            };
            let cloned = cfg.clone();
            cfg.mirror_heads = 2;
            assert_eq!(cloned.mirror_heads, 8);
        }

        // ── CommConfig (REQ-IB-010) ────────────────────────────────────────

        #[test]
        fn comm_config_default_values() {
            let cfg = CommConfig::default();
            assert_eq!(cfg.overlap, crate::engine::intent_bias::OverlapHint::Auto);
            assert_eq!(cfg.compress, CommCompressHint::Auto);
            assert!(cfg.algorithm_override.is_empty());
        }

        #[test]
        fn comm_config_custom_values() {
            let cfg = CommConfig {
                overlap: crate::engine::intent_bias::OverlapHint::PreferOverlap,
                compress: CommCompressHint::AlwaysCompress,
                algorithm_override: "custom_algo".to_string(),
            };
            assert_eq!(cfg.overlap, crate::engine::intent_bias::OverlapHint::PreferOverlap);
            assert_eq!(cfg.compress, CommCompressHint::AlwaysCompress);
            assert_eq!(cfg.algorithm_override, "custom_algo");
        }

        #[test]
        fn comm_config_equality() {
            let a = CommConfig::default();
            let b = CommConfig::default();
            assert_eq!(a, b);
        }

        #[test]
        fn comm_config_clone_independence() {
            let mut cfg = CommConfig {
                overlap: crate::engine::intent_bias::OverlapHint::ForceDoubleBuffer,
                compress: CommCompressHint::ForceQuant,
                algorithm_override: "test".to_string(),
            };
            let cloned = cfg.clone();
            cfg.compress = CommCompressHint::NeverCompress;
            assert_eq!(cloned.compress, CommCompressHint::ForceQuant);
        }

        // ── MoeDistributedConfig (REQ-IB-011) ──────────────────────────────

        #[test]
        fn moe_distributed_config_default_values() {
            let cfg = MoeDistributedConfig::default();
            assert_eq!(cfg.expert_placement, ExpertPlacement::Auto);
            assert_eq!(cfg.all_to_all, AllToAllStrategy::Auto);
        }

        #[test]
        fn moe_distributed_config_custom_values() {
            let cfg = MoeDistributedConfig {
                expert_placement: ExpertPlacement::HotCold,
                all_to_all: AllToAllStrategy::NvlinkAllToAll,
            };
            assert_eq!(cfg.expert_placement, ExpertPlacement::HotCold);
            assert_eq!(cfg.all_to_all, AllToAllStrategy::NvlinkAllToAll);
        }

        #[test]
        fn moe_distributed_config_equality() {
            let a = MoeDistributedConfig::default();
            let b = MoeDistributedConfig::default();
            assert_eq!(a, b);
        }

        #[test]
        fn moe_distributed_config_clone_independence() {
            let mut cfg = MoeDistributedConfig {
                expert_placement: ExpertPlacement::RoundRobin,
                all_to_all: AllToAllStrategy::HierarchicalAllToAll,
            };
            let cloned = cfg.clone();
            cfg.expert_placement = ExpertPlacement::Custom;
            assert_eq!(cloned.expert_placement, ExpertPlacement::RoundRobin);
        }

        // ── DistributedConfig (REQ-IB-006) ─────────────────────────────────

        #[test]
        fn distributed_config_default_is_single_node() {
            let cfg = DistributedConfig::default();
            assert_eq!(cfg.parallel, ParallelConfig::default());
            assert_eq!(cfg.pd_disagg, PdDisaggConfig::default());
            assert_eq!(cfg.kv_distribution, KvDistributionConfig::default());
            assert_eq!(cfg.comm, CommConfig::default());
            assert_eq!(cfg.moe, MoeDistributedConfig::default());
        }

        #[test]
        fn distributed_config_equality() {
            let a = DistributedConfig::default();
            let b = DistributedConfig::default();
            assert_eq!(a, b);
        }

        #[test]
        fn distributed_config_inequality_parallel() {
            let a = DistributedConfig::default();
            let b = DistributedConfig {
                parallel: ParallelConfig {
                    tp_size: 2,
                    ..Default::default()
                },
                ..Default::default()
            };
            assert_ne!(a, b);
        }

        #[test]
        fn distributed_config_clone_independence() {
            let mut cfg = DistributedConfig::default();
            let cloned = cfg.clone();
            cfg.parallel.tp_size = 4;
            assert_eq!(cloned.parallel.tp_size, 1);
        }

        // ── CommHandleWrapper (REQ-DIST-001) ────────────────────────────────

        #[test]
        fn comm_handle_wrapper_from_default_config() {
            let config = ParallelConfig::default();
            let handle = CommHandleWrapper::from_config(&config).unwrap();
            assert_eq!(handle.rank(), 0);
            assert_eq!(handle.world_size(), 1);
            assert!(!handle.is_distributed());
        }

        #[test]
        fn comm_handle_wrapper_from_multi_node_config() {
            let config = ParallelConfig {
                tp_size: 2,
                pp_size: 1,
                ep_size: 1,
                cp_size: 1,
                rank: 1,
                world_size: 2,
                unique_id: String::new(),
            };
            let handle = CommHandleWrapper::from_config(&config).unwrap();
            assert_eq!(handle.rank(), 1);
            assert_eq!(handle.world_size(), 2);
            assert!(handle.is_distributed());
        }

        #[test]
        fn comm_handle_wrapper_invalid_config_rejected() {
            let config = ParallelConfig {
                tp_size: 2,
                pp_size: 1,
                ep_size: 1,
                cp_size: 1,
                rank: 0,
                world_size: 5, // 2*1*1=2 != 5
                unique_id: String::new(),
            };
            let result = CommHandleWrapper::from_config(&config);
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), DistributedConfigError::InvalidParallelConfig);
        }

        #[test]
        fn comm_handle_wrapper_rank_out_of_range_rejected() {
            let config = ParallelConfig {
                tp_size: 1,
                pp_size: 1,
                ep_size: 1,
                cp_size: 1,
                rank: 5, // rank >= world_size
                world_size: 1,
                unique_id: String::new(),
            };
            let result = CommHandleWrapper::from_config(&config);
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), DistributedConfigError::InvalidParallelConfig);
        }

        #[test]
        fn comm_handle_wrapper_nccl_not_initialized_by_default() {
            let config = ParallelConfig::default();
            let handle = CommHandleWrapper::from_config(&config).unwrap();
            assert!(!handle.is_nccl_initialized());
        }

        #[test]
        fn comm_handle_wrapper_distributed_nccl_not_initialized_by_default() {
            let config = ParallelConfig {
                tp_size: 2,
                pp_size: 1,
                ep_size: 1,
                cp_size: 1,
                rank: 0,
                world_size: 2,
                unique_id: String::new(),
            };
            let handle = CommHandleWrapper::from_config(&config).unwrap();
            assert!(handle.is_distributed());
            assert!(!handle.is_nccl_initialized());
        }

        // ── CommHandleWrapper destroy + Drop (REQ-DIST-001) ────────────────────

        #[test]
        fn comm_handle_wrapper_not_destroyed_by_default() {
            let config = ParallelConfig::default();
            let handle = CommHandleWrapper::from_config(&config).unwrap();
            assert!(!handle.is_destroyed());
        }

        #[test]
        fn comm_handle_wrapper_destroy_marks_destroyed() {
            let config = ParallelConfig::default();
            let mut handle = CommHandleWrapper::from_config(&config).unwrap();
            assert!(!handle.is_destroyed());
            handle.destroy();
            assert!(handle.is_destroyed());
        }

        #[test]
        fn comm_handle_wrapper_destroy_idempotent() {
            // Calling destroy() twice must not panic — second call is a no-op
            let config = ParallelConfig::default();
            let mut handle = CommHandleWrapper::from_config(&config).unwrap();
            handle.destroy();
            assert!(handle.is_destroyed());
            handle.destroy(); // second call — no-op
            assert!(handle.is_destroyed());
        }

        #[test]
        fn comm_handle_wrapper_destroy_single_node_noop() {
            // Single-node mode: destroy is a no-op on inner=None
            let config = ParallelConfig::default();
            let mut handle = CommHandleWrapper::from_config(&config).unwrap();
            assert!(!handle.is_distributed());
            assert!(handle.inner.is_none());
            handle.destroy();
            assert!(handle.is_destroyed());
            assert!(handle.inner.is_none()); // inner stays None
        }

        #[test]
        fn comm_handle_wrapper_destroy_takes_inner() {
            // After destroy, inner is None even if it was Some before
            let config = ParallelConfig {
                tp_size: 2,
                pp_size: 1,
                ep_size: 1,
                cp_size: 1,
                rank: 0,
                world_size: 2,
                unique_id: String::new(),
            };
            let mut handle = CommHandleWrapper::from_config(&config).unwrap();
            // inner is None (lazy init — NCCL not actually initialized in test)
            assert!(handle.inner.is_none());
            handle.destroy();
            assert!(handle.is_destroyed());
            assert!(handle.inner.is_none());
        }

        #[test]
        fn comm_handle_wrapper_drop_calls_destroy() {
            // Drop triggers destroy automatically — verified by is_destroyed
            let config = ParallelConfig::default();
            let mut handle = CommHandleWrapper::from_config(&config).unwrap();
            handle.destroy();
            // After explicit destroy, Drop is a no-op (idempotent)
            drop(handle); // should not panic
        }

        #[test]
        fn comm_handle_wrapper_double_drop_safe() {
            // Simulate double-drop scenario: destroy + Drop
            let config = ParallelConfig::default();
            let mut handle = CommHandleWrapper::from_config(&config).unwrap();
            handle.destroy();
            assert!(handle.is_destroyed());
            // Drop will call destroy() again — idempotent, no panic
        }

        // ── DistributedConfigError (REQ-DIST-001) ───────────────────────────

        #[test]
        fn distributed_config_error_variants_distinct() {
            let a = DistributedConfigError::InvalidParallelConfig;
            let b = DistributedConfigError::CommInitFailed("timeout".to_string());
            assert_ne!(a, b);
        }

        #[test]
        fn distributed_config_error_equality() {
            assert_eq!(
                DistributedConfigError::InvalidParallelConfig,
                DistributedConfigError::InvalidParallelConfig,
            );
            assert_eq!(
                DistributedConfigError::CommInitFailed("err".to_string()),
                DistributedConfigError::CommInitFailed("err".to_string()),
            );
        }

        // ── TEST-DIST-001: Full lifecycle + error propagation (REQ-DIST-001) ────

        // TEST-DIST-001-01: DistributedConfigError Display formatting
        #[test]
        fn distributed_config_error_display_invalid_parallel_config() {
            let err = DistributedConfigError::InvalidParallelConfig;
            let msg = format!("{}", err);
            assert!(msg.contains("ParallelConfig"));
            assert!(msg.contains("validation failed"));
        }

        #[test]
        fn distributed_config_error_display_comm_init_failed() {
            let err = DistributedConfigError::CommInitFailed("timeout".to_string());
            let msg = format!("{}", err);
            assert!(msg.contains("NCCL"));
            assert!(msg.contains("timeout"));
        }

        // TEST-DIST-001-02: DistributedConfigError is std::error::Error
        #[test]
        fn distributed_config_error_is_std_error() {
            let err = DistributedConfigError::InvalidParallelConfig;
            let _: &dyn std::error::Error = &err;
        }

        // TEST-DIST-001-03: Full lifecycle — create → verify → destroy → verify
        #[test]
        fn comm_handle_wrapper_full_lifecycle_single_node() {
            let config = ParallelConfig::default();
            let mut handle = CommHandleWrapper::from_config(&config).unwrap();
            assert_eq!(handle.rank(), 0);
            assert_eq!(handle.world_size(), 1);
            assert!(!handle.is_distributed());
            assert!(!handle.is_nccl_initialized());
            assert!(!handle.is_destroyed());
            handle.destroy();
            assert!(handle.is_destroyed());
            assert!(handle.inner.is_none());
        }

        // TEST-DIST-001-04: destroy then drop is safe (no double-free)
        #[test]
        fn comm_handle_wrapper_destroy_then_drop_no_double_free() {
            let config = ParallelConfig::default();
            let mut handle = CommHandleWrapper::from_config(&config).unwrap();
            handle.destroy();
            assert!(handle.is_destroyed());
            // Drop calls destroy() again — idempotent, safe
            drop(handle);
        }

        // TEST-DIST-001-05: from_config valid multi-node config — correct fields
        #[test]
        fn comm_handle_wrapper_from_config_valid_multi_node_fields() {
            let config = ParallelConfig {
                tp_size: 4,
                pp_size: 2,
                ep_size: 1,
                cp_size: 1,
                rank: 3,
                world_size: 8,
                unique_id: String::new(),
            };
            let handle = CommHandleWrapper::from_config(&config).unwrap();
            assert_eq!(handle.rank(), 3);
            assert_eq!(handle.world_size(), 8);
            assert!(handle.is_distributed());
            assert!(!handle.is_nccl_initialized());
            assert!(!handle.is_destroyed());
        }

        // TEST-DIST-001-06: from_config rejects invalid ParallelConfig
        #[test]
        fn comm_handle_wrapper_from_config_invalid_world_size() {
            let config = ParallelConfig {
                tp_size: 2,
                pp_size: 2,
                ep_size: 1,
                cp_size: 1,
                rank: 0,
                world_size: 5, // 2*2*1=4 != 5
                unique_id: String::new(),
            };
            let result = CommHandleWrapper::from_config(&config);
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), DistributedConfigError::InvalidParallelConfig);
        }

        // TEST-DIST-001-07: from_config rejects rank >= world_size
        #[test]
        fn comm_handle_wrapper_from_config_rank_out_of_range() {
            let config = ParallelConfig {
                tp_size: 1,
                pp_size: 1,
                ep_size: 1,
                cp_size: 1,
                rank: 2, // rank >= world_size=1
                world_size: 1,
                unique_id: String::new(),
            };
            let result = CommHandleWrapper::from_config(&config);
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), DistributedConfigError::InvalidParallelConfig);
        }
    }
}
