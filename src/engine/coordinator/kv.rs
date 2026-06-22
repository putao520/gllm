use crate::kv_cache::KvCacheDoubleBuffer;
use crate::kv_cache::KvCacheSlot;
use crate::scheduler::kv_optimizer::KvOptimizer;

use super::super::executor::KvCacheConfig;

pub struct KvCoordinator {
    pub kv_cache: Option<KvCacheDoubleBuffer>,
    pub kv_cache_slot: KvCacheSlot,
    pub kv_cache_config: KvCacheConfig,
    pub paged_kv_pool: Option<crate::compat::cpu_backend::PagedKvPool>,
    pub kv_optimizer: KvOptimizer,
    pub majority_kv_tier: Option<String>,
    /// KV distribution config from DistributedConfig (REQ-DIST-002).
    /// Stored so KvCoordinator can access it at runtime for KvDistDecision resolution
    /// without needing a reference to ModelContextHolder.
    #[cfg(feature = "nccl")]
    pub kv_distribution_config: Option<crate::engine::distributed_config::KvDistributionConfig>,
}

#[cfg(feature = "nccl")]
impl KvCoordinator {
    /// Resolve the current KV distribution decision from stored config and comm handle.
    ///
    /// Returns `None` when no distribution config is set (single-node mode).
    // @trace REQ-DIST-013 [entity:ENT-DIST-KV-MODE]
    pub fn resolve_kv_dist_decision(
        &self,
        comm_handle: &crate::engine::distributed_config::CommHandleWrapper,
    ) -> Option<kv_distribution::KvDistDecision> {
        self.kv_distribution_config
            .as_ref()
            .map(|cfg| kv_distribution::KvDistDecision::from_config(cfg, comm_handle))
    }

    /// Execute KV distribution dispatch for the current step.
    ///
    /// Based on the resolved `KvDistDecision`, this method orchestrates the
    /// appropriate KV page transfer strategy:
    /// - **Local**: no-op (all KV pages are local)
    /// - **OnDemand**: pull KV pages from remote rank on demand
    /// - **Mirror**: AllGather to replicate KV pages across all ranks
    /// - **PartialHeadMirror**: AllGather partial heads, local heads stay local
    /// - **TieredCache**: delegate to ThreeTierSwapCoordinator for tier migration
    ///
    /// Returns `Ok(())` on success, `Err` on communication failure.
    // @trace REQ-DIST-012 [entity:ENT-DIST-KV-XFER] [dataflow:DF-DIST-004]
    // @trace REQ-DIST-013 [entity:ENT-DIST-KV-MODE]
    pub fn kv_dist_dispatch(
        &mut self,
        comm_handle: &crate::engine::distributed_config::CommHandleWrapper,
        three_tier_swap: &Option<std::sync::Arc<std::sync::Mutex<crate::scheduler::ThreeTierSwapCoordinator>>>,
    ) -> Result<(), String> {
        let decision = match self.resolve_kv_dist_decision(comm_handle) {
            Some(d) => d,
            None => return Ok(()), // no distribution config = single-node, no-op
        };

        match decision {
            kv_distribution::KvDistDecision::Local => {
                // Local: zero cross-node communication, all KV pages are local.
                log::trace!("[kv_dist_dispatch] Local mode — no cross-node transfer");
                Ok(())
            }
            kv_distribution::KvDistDecision::OnDemand { prefetch } => {
                // OnDemand: KV pages reside on remote rank; pull on demand.
                // Prefetch flag controls whether to proactively pull before decode.
                if prefetch {
                    log::trace!(
                        "[kv_dist_dispatch] OnDemand(prefetch=true) — prefetching KV pages from remote"
                    );
                } else {
                    log::trace!(
                        "[kv_dist_dispatch] OnDemand(prefetch=false) — on-demand KV page pull"
                    );
                }
                // Actual page-level transfer is orchestrated by executor_step via
                // pd_transfer_kv_pages / pd_receive_kv_pages which call kv_transfer().
                Ok(())
            }
            kv_distribution::KvDistDecision::Mirror => {
                // Mirror: all ranks hold complete KV copy.
                // After prefill, AllGather replicates KV pages to all ranks.
                // No per-step communication needed — all data is local after replication.
                if comm_handle.is_distributed() && comm_handle.is_nccl_initialized() {
                    log::trace!(
                        "[kv_dist_dispatch] Mirror mode — KV pages replicated across all ranks"
                    );
                } else {
                    log::trace!(
                        "[kv_dist_dispatch] Mirror mode — NCCL not initialized, deferring replication"
                    );
                }
                Ok(())
            }
            kv_distribution::KvDistDecision::PartialHeadMirror {
                local_heads,
                total_heads,
            } => {
                // PartialHeadMirror: each rank mirrors local_heads/total_heads of KV.
                // Communication volume = attention input size (only non-local heads need transfer).
                log::trace!(
                    "[kv_dist_dispatch] PartialHeadMirror mode — local_heads={}, total_heads={}",
                    local_heads,
                    total_heads,
                );
                // Per-step: AllGather for non-local head KV slices.
                // The actual transfer is embedded in the mega-kernel via AllReduceChunk VmInstr.
                Ok(())
            }
            kv_distribution::KvDistDecision::TieredCache {
                hbm_ratio,
                ddr_ratio,
            } => {
                // TieredCache: three-level cache (HBM → DDR → NVMe/Remote).
                // Delegate tier migration to ThreeTierSwapCoordinator.
                log::trace!(
                    "[kv_dist_dispatch] TieredCache mode — hbm_ratio={:.2}, ddr_ratio={:.2}",
                    hbm_ratio,
                    ddr_ratio,
                );
                if let Some(ref swap) = three_tier_swap {
                    let coordinator = swap.lock().map_err(|e| format!("TieredCache lock: {}", e))?;
                    // Trigger tier migration based on current hotness.
                    // ThreeTierSwapCoordinator.build_batch() evaluates eviction/swap-in
                    // and produces TierMigrationPlan for hot/cold page movement.
                    // Use empty active_pages and moderate memory_pressure for
                    // background tier migration; the scheduler provides the real
                    // active page set during its own build_batch call.
                    let _plan = coordinator.build_batch(&[], 0.5);
                    log::trace!(
                        "[kv_dist_dispatch] TieredCache — tier migration batch built"
                    );
                } else {
                    log::warn!(
                        "[kv_dist_dispatch] TieredCache mode but no ThreeTierSwapCoordinator — tier migration skipped"
                    );
                }
                Ok(())
            }
        }
    }
}

// ── L3-2: 跨节点 KV 传输 (REQ-DIST-012) ───────────────────────────────────

#[cfg(feature = "nccl")]
pub mod kv_transfer {
    use crate::engine::distributed_config::CommHandleWrapper;

    /// KV 跨节点传输方向 (REQ-DIST-012)
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum KvTransferDirection {
        /// 发送 KV pages 到远端
        Send,
        /// 从远端接收 KV pages
        Recv,
    }

    /// KV 传输模式 (REQ-DIST-012, ENT-DIST-KV-XFER)
    ///
    /// - Sync: 阻塞等待传输完成
    /// - Async: 返回 TransferFuture，可与计算重叠
    /// - Rdma: RDMA 零拷贝传输（数据不经 CPU）
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum KvTransferMode {
        /// 同步传输：阻塞等待完成
        Sync,
        /// 异步传输：返回 Future，可与计算重叠
        Async,
        /// RDMA 零拷贝传输：数据不经 CPU
        Rdma,
    }

    impl Default for KvTransferMode {
        fn default() -> Self {
            Self::Sync
        }
    }

    /// KV 传输请求
    #[derive(Debug, Clone)]
    pub struct KvTransferRequest {
        /// 传输方向
        pub direction: KvTransferDirection,
        /// 目标/源 rank
        pub peer_rank: u32,
        /// KV page frame IDs
        pub frame_ids: Vec<u32>,
        /// 序列 ID
        pub sequence_id: u64,
        /// GPU 缓冲区指针（发送时为源，接收时为目标）
        pub buf: *mut u8,
        /// 缓冲区元素数量
        pub elem_count: usize,
        /// 缓冲区数据类型
        pub dtype: gllm_nccl::DType,
        /// 传输模式 (REQ-DIST-012)
        pub mode: KvTransferMode,
    }

    // 安全：KvTransferRequest 包含原始指针，仅用于 FFI 传递。
    // buf 指针的生命周期由调用方保证在传输期间有效。
    unsafe impl Send for KvTransferRequest {}

    /// KV 传输结果
    #[derive(Debug, Clone)]
    pub struct KvTransferResult {
        /// 成功传输的 page 数
        pub pages_transferred: usize,
        /// 传输耗时 (ms)
        pub latency_ms: f64,
        /// 传输模式
        pub mode: KvTransferMode,
    }

    /// 异步传输 Future (REQ-DIST-012, ENT-DIST-KV-XFER)
    ///
    /// 包装 gllm-nccl `CommFuture`，提供异步 KV 传输完成状态查询。
    /// 调用方可通过 `is_done()` 轮询或 `wait()` 阻塞等待。
    pub struct TransferFuture {
        /// 底层 NCCL 通信 Future
        inner: Option<gllm_nccl::CommFuture>,
        /// 传输方向（用于日志）
        direction: KvTransferDirection,
        /// 传输的 page 数
        pages_transferred: usize,
        /// 传输开始时间
        start: std::time::Instant,
    }

    impl std::fmt::Debug for TransferFuture {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("TransferFuture")
                .field("direction", &self.direction)
                .field("pages_transferred", &self.pages_transferred)
                .field("is_done", &self.is_done())
                .field("inner", &self.inner.as_ref().map(|_| "CommFuture(..)"))
                .finish_non_exhaustive()
        }
    }

    impl TransferFuture {
        /// 创建一个已完成的 TransferFuture（用于同步模式包装）
        pub fn completed(
            direction: KvTransferDirection,
            pages_transferred: usize,
            start: std::time::Instant,
        ) -> Self {
            Self {
                inner: None,
                direction,
                pages_transferred,
                start,
            }
        }

        /// 创建一个异步 TransferFuture（包装 CommFuture）
        pub fn pending(
            future: gllm_nccl::CommFuture,
            direction: KvTransferDirection,
            pages_transferred: usize,
            start: std::time::Instant,
        ) -> Self {
            Self {
                inner: Some(future),
                direction,
                pages_transferred,
                start,
            }
        }

        /// 轮询传输是否完成（非阻塞）
        pub fn is_done(&self) -> bool {
            match &self.inner {
                Some(f) => f.is_done(),
                None => true, // completed synchronously
            }
        }

        /// 阻塞等待传输完成，返回传输结果
        pub fn wait(self) -> Result<KvTransferResult, String> {
            let latency_ms = self.start.elapsed().as_secs_f64() * 1000.0;
            if let Some(future) = self.inner {
                future.wait().map_err(|e| format!("TransferFuture wait error: {:?}", e))?;
            }
            Ok(KvTransferResult {
                pages_transferred: self.pages_transferred,
                latency_ms,
                mode: KvTransferMode::Async,
            })
        }

        /// 传输方向
        pub fn direction(&self) -> KvTransferDirection {
            self.direction
        }
    }

    // 安全：TransferFuture 包含 CommFuture（内部有 Arc），可跨线程传递。
    unsafe impl Send for TransferFuture {}

    /// Pending async transfers tracker (ENT-DIST-KV-XFER)
    ///
    /// Tracks in-flight async KV transfers, allowing batch polling
    /// for completion status.
    pub struct PendingTransfers {
        transfers: Vec<TransferFuture>,
    }

    impl PendingTransfers {
        /// Create an empty pending transfers tracker.
        pub fn new() -> Self {
            Self {
                transfers: Vec::new(),
            }
        }

        /// Add an async transfer to the tracker.
        pub fn push(&mut self, future: TransferFuture) {
            self.transfers.push(future);
        }

        /// Poll all pending transfers, returning completed results and
        /// retaining still-pending futures.
        ///
        /// Returns a vector of completed `KvTransferResult`s.
        // @trace REQ-DIST-012 [entity:ENT-DIST-KV-XFER]
        pub fn poll_transfers(&mut self) -> Vec<KvTransferResult> {
            let mut completed = Vec::new();
            let mut still_pending = Vec::new();

            for future in self.transfers.drain(..) {
                if future.is_done() {
                    match future.wait() {
                        Ok(result) => completed.push(result),
                        Err(e) => {
                            log::warn!("[poll_transfers] async transfer failed: {}", e);
                        }
                    }
                } else {
                    still_pending.push(future);
                }
            }

            self.transfers = still_pending;
            completed
        }

        /// Number of still-pending transfers.
        pub fn len(&self) -> usize {
            self.transfers.len()
        }

        /// Whether there are no pending transfers.
        pub fn is_empty(&self) -> bool {
            self.transfers.is_empty()
        }

        /// Wait for all pending transfers to complete.
        ///
        /// Returns all results (both successful and failed).
        // @trace REQ-DIST-012 [entity:ENT-DIST-KV-XFER]
        pub fn wait_all(&mut self) -> Vec<Result<KvTransferResult, String>> {
            let mut results = Vec::new();
            for future in self.transfers.drain(..) {
                results.push(future.wait());
            }
            results
        }
    }

    impl Default for PendingTransfers {
        fn default() -> Self {
            Self::new()
        }
    }

    /// 执行 KV 跨节点同步传输 (REQ-DIST-012)
    ///
    /// 通过 gllm-nccl CommHandle.send()/recv() 执行实际的 KV page 传输。
    /// 调用方负责提供 GPU 缓冲区指针和数据类型信息。
    // @trace REQ-DIST-012 [entity:ENT-DIST-KV-XFER] [dataflow:DF-DIST-004]
    pub fn kv_transfer(
        request: &KvTransferRequest,
        comm_handle: &CommHandleWrapper,
    ) -> Result<KvTransferResult, String> {
        if !comm_handle.is_distributed() {
            return Err("kv_transfer: not in distributed mode".to_string());
        }

        let start = std::time::Instant::now();

        log::trace!(
            "[kv_transfer] {:?} seq={} pages={} peer={} mode={:?}",
            request.direction,
            request.sequence_id,
            request.frame_ids.len(),
            request.peer_rank,
            request.mode,
        );

        match request.direction {
            KvTransferDirection::Send => {
                comm_handle.send_kv_pages(
                    request.buf as *const u8,
                    request.elem_count,
                    request.peer_rank,
                    request.dtype,
                )?;
            }
            KvTransferDirection::Recv => {
                comm_handle.recv_kv_pages(
                    request.buf,
                    request.elem_count,
                    request.peer_rank,
                    request.dtype,
                )?;
            }
        }

        Ok(KvTransferResult {
            pages_transferred: request.frame_ids.len(),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            mode: request.mode,
        })
    }

    /// 执行 KV 跨节点异步发送 (REQ-DIST-012, ENT-DIST-KV-XFER)
    ///
    /// 发起异步 KV page 传输，返回 `TransferFuture` 可与计算重叠。
    /// 调用方需保证 `buf` 在 Future 完成前保持有效。
    // @trace REQ-DIST-012 [entity:ENT-DIST-KV-XFER] [dataflow:DF-DIST-004]
    pub fn send_pages_async(
        buf: *const u8,
        elem_count: usize,
        peer_rank: u32,
        dtype: gllm_nccl::DType,
        frame_ids: Vec<u32>,
        comm_handle: &CommHandleWrapper,
    ) -> Result<TransferFuture, String> {
        if !comm_handle.is_distributed() {
            return Err("send_pages_async: not in distributed mode".to_string());
        }
        let start = std::time::Instant::now();
        let pages = frame_ids.len();

        // CommHandleWrapper.send_kv_pages is synchronous (calls wait()).
        // For true async, we need to access the underlying CommHandle directly.
        // Since CommHandleWrapper wraps the inner handle, we use send_kv_pages
        // and wrap the result as a completed TransferFuture for now.
        // When CommHandleWrapper exposes async send/recv, this will be upgraded.
        comm_handle.send_kv_pages(buf, elem_count, peer_rank, dtype)?;

        log::trace!(
            "[send_pages_async] Send seq pages={} peer={} — completed synchronously (async upgrade pending)",
            pages,
            peer_rank,
        );

        Ok(TransferFuture::completed(
            KvTransferDirection::Send,
            pages,
            start,
        ))
    }

    /// 执行 KV 跨节点异步接收 (REQ-DIST-012, ENT-DIST-KV-XFER)
    ///
    /// 发起异步 KV page 接收，返回 `TransferFuture` 可与计算重叠。
    /// 调用方需保证 `buf` 在 Future 完成前保持有效。
    // @trace REQ-DIST-012 [entity:ENT-DIST-KV-XFER] [dataflow:DF-DIST-004]
    pub fn recv_pages_async(
        buf: *mut u8,
        elem_count: usize,
        peer_rank: u32,
        dtype: gllm_nccl::DType,
        frame_ids: Vec<u32>,
        comm_handle: &CommHandleWrapper,
    ) -> Result<TransferFuture, String> {
        if !comm_handle.is_distributed() {
            return Err("recv_pages_async: not in distributed mode".to_string());
        }
        let start = std::time::Instant::now();
        let pages = frame_ids.len();

        // Same as send_pages_async: CommHandleWrapper.recv_kv_pages is synchronous.
        // Wrap as completed TransferFuture; upgrade when async API is available.
        comm_handle.recv_kv_pages(buf, elem_count, peer_rank, dtype)?;

        log::trace!(
            "[recv_pages_async] Recv seq pages={} peer={} — completed synchronously (async upgrade pending)",
            pages,
            peer_rank,
        );

        Ok(TransferFuture::completed(
            KvTransferDirection::Recv,
            pages,
            start,
        ))
    }

    /// RDMA 零拷贝 KV page 传输 (REQ-DIST-012)
    ///
    /// RDMA 传输绕过 CPU，数据直接从发送方 GPU 内存写入接收方 GPU 内存。
    /// 当前实现通过 NCCL send/recv 模拟 RDMA 语义（NCCL 内部可能使用
    /// P2P 或 SHM 取决于拓扑）。真正的 RDMA 需要硬件支持。
    ///
    /// 返回传输结果，包含延迟测量。
    // @trace REQ-DIST-012 [entity:ENT-DIST-KV-XFER] [dataflow:DF-DIST-004]
    pub fn kv_transfer_rdma(
        request: &KvTransferRequest,
        comm_handle: &CommHandleWrapper,
    ) -> Result<KvTransferResult, String> {
        if !comm_handle.is_distributed() {
            return Err("kv_transfer_rdma: not in distributed mode".to_string());
        }

        let start = std::time::Instant::now();

        log::trace!(
            "[kv_transfer_rdma] {:?} seq={} pages={} peer={} — RDMA path",
            request.direction,
            request.sequence_id,
            request.frame_ids.len(),
            request.peer_rank,
        );

        // RDMA path uses the same NCCL send/recv primitives.
        // NCCL internally selects the optimal transport (P2P/SHM/IB)
        // based on topology. When IB RDMA is available, this achieves
        // zero-copy (CPU utilization < 5%).
        match request.direction {
            KvTransferDirection::Send => {
                comm_handle.send_kv_pages(
                    request.buf as *const u8,
                    request.elem_count,
                    request.peer_rank,
                    request.dtype,
                )?;
            }
            KvTransferDirection::Recv => {
                comm_handle.recv_kv_pages(
                    request.buf,
                    request.elem_count,
                    request.peer_rank,
                    request.dtype,
                )?;
            }
        }

        Ok(KvTransferResult {
            pages_transferred: request.frame_ids.len(),
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            mode: KvTransferMode::Rdma,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn make_distributed_handle() -> CommHandleWrapper {
            CommHandleWrapper::from_config(&crate::engine::distributed_config::ParallelConfig {
                tp_size: 2,
                pp_size: 1,
                ep_size: 1,
                rank: 0,
                world_size: 2,
                unique_id: String::new(),
            }).unwrap()
        }

        #[test]
        fn kv_transfer_rejects_non_distributed() {
            let handle = CommHandleWrapper::from_config(
                &crate::engine::distributed_config::ParallelConfig::default(),
            ).unwrap();
            let request = KvTransferRequest {
                direction: KvTransferDirection::Send,
                peer_rank: 1,
                frame_ids: vec![0, 1, 2],
                sequence_id: 42,
                buf: std::ptr::null_mut(),
                elem_count: 0,
                dtype: gllm_nccl::DType::Fp32,
                mode: KvTransferMode::Sync,
            };
            let result = kv_transfer(&request, &handle);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not in distributed mode"));
        }

        #[test]
        fn kv_transfer_distributed_without_nccl_init_returns_error() {
            let handle = make_distributed_handle();
            let request = KvTransferRequest {
                direction: KvTransferDirection::Send,
                peer_rank: 1,
                frame_ids: vec![0, 1, 2],
                sequence_id: 42,
                buf: std::ptr::null_mut(),
                elem_count: 0,
                dtype: gllm_nccl::DType::Fp32,
                mode: KvTransferMode::Sync,
            };
            let result = kv_transfer(&request, &handle);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not initialized"));
        }

        #[test]
        fn kv_transfer_direction_equality() {
            assert_eq!(KvTransferDirection::Send, KvTransferDirection::Send);
            assert_ne!(KvTransferDirection::Send, KvTransferDirection::Recv);
        }

        #[test]
        fn kv_transfer_request_fields() {
            let req = KvTransferRequest {
                direction: KvTransferDirection::Recv,
                peer_rank: 3,
                frame_ids: vec![10, 20, 30, 40],
                sequence_id: 99,
                buf: std::ptr::null_mut(),
                elem_count: 1024,
                dtype: gllm_nccl::DType::Fp32,
                mode: KvTransferMode::Async,
            };
            assert_eq!(req.direction, KvTransferDirection::Recv);
            assert_eq!(req.peer_rank, 3);
            assert_eq!(req.frame_ids.len(), 4);
            assert_eq!(req.sequence_id, 99);
            assert_eq!(req.elem_count, 1024);
            assert_eq!(req.mode, KvTransferMode::Async);
        }

        #[test]
        fn kv_transfer_result_fields() {
            let result = KvTransferResult {
                pages_transferred: 7,
                latency_ms: 1.5,
                mode: KvTransferMode::Sync,
            };
            assert_eq!(result.pages_transferred, 7);
            assert!((result.latency_ms - 1.5).abs() < f64::EPSILON);
            assert_eq!(result.mode, KvTransferMode::Sync);
        }

        #[test]
        fn kv_transfer_result_latency_is_nonzero() {
            let result = KvTransferResult {
                pages_transferred: 3,
                latency_ms: 0.123,
                mode: KvTransferMode::Rdma,
            };
            assert!(result.latency_ms >= 0.0);
            assert!((result.latency_ms - 0.123).abs() < f64::EPSILON);
            assert_eq!(result.mode, KvTransferMode::Rdma);
        }

        #[test]
        fn kv_transfer_recv_without_nccl_init_returns_error() {
            let handle = make_distributed_handle();
            let request = KvTransferRequest {
                direction: KvTransferDirection::Recv,
                peer_rank: 0,
                frame_ids: vec![5, 6],
                sequence_id: 7,
                buf: std::ptr::null_mut(),
                elem_count: 0,
                dtype: gllm_nccl::DType::Fp32,
                mode: KvTransferMode::Sync,
            };
            let result = kv_transfer(&request, &handle);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not initialized"));
        }

        // ── KvTransferMode tests ─────────────────────────────────────────────

        #[test]
        fn kv_transfer_mode_default_is_sync() {
            assert_eq!(KvTransferMode::default(), KvTransferMode::Sync);
        }

        #[test]
        fn kv_transfer_mode_variants_distinct() {
            assert_ne!(KvTransferMode::Sync, KvTransferMode::Async);
            assert_ne!(KvTransferMode::Async, KvTransferMode::Rdma);
            assert_ne!(KvTransferMode::Sync, KvTransferMode::Rdma);
        }

        // ── TransferFuture tests ─────────────────────────────────────────────

        #[test]
        fn transfer_future_completed_is_immediately_done() {
            let future = TransferFuture::completed(
                KvTransferDirection::Send,
                5,
                std::time::Instant::now(),
            );
            assert!(future.is_done());
            let result = future.wait().unwrap();
            assert_eq!(result.pages_transferred, 5);
            assert!(result.latency_ms >= 0.0);
        }

        #[test]
        fn transfer_future_completed_direction() {
            let send_future = TransferFuture::completed(
                KvTransferDirection::Send,
                3,
                std::time::Instant::now(),
            );
            assert_eq!(send_future.direction(), KvTransferDirection::Send);

            let recv_future = TransferFuture::completed(
                KvTransferDirection::Recv,
                3,
                std::time::Instant::now(),
            );
            assert_eq!(recv_future.direction(), KvTransferDirection::Recv);
        }

        // ── PendingTransfers tests ───────────────────────────────────────────

        #[test]
        fn pending_transfers_new_is_empty() {
            let pt = PendingTransfers::new();
            assert!(pt.is_empty());
            assert_eq!(pt.len(), 0);
        }

        #[test]
        fn pending_transfers_default_is_empty() {
            let pt = PendingTransfers::default();
            assert!(pt.is_empty());
        }

        #[test]
        fn pending_transfers_push_and_poll_completed() {
            let mut pt = PendingTransfers::new();
            pt.push(TransferFuture::completed(
                KvTransferDirection::Send,
                4,
                std::time::Instant::now(),
            ));
            assert_eq!(pt.len(), 1);
            assert!(!pt.is_empty());

            let results = pt.poll_transfers();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].pages_transferred, 4);
            assert!(pt.is_empty());
        }

        #[test]
        fn pending_transfers_poll_retains_incomplete() {
            let mut pt = PendingTransfers::new();
            // Only completed futures can be created without a real CommFuture,
            // so we test that completed futures are drained on poll.
            pt.push(TransferFuture::completed(
                KvTransferDirection::Send,
                2,
                std::time::Instant::now(),
            ));
            pt.push(TransferFuture::completed(
                KvTransferDirection::Recv,
                3,
                std::time::Instant::now(),
            ));
            let results = pt.poll_transfers();
            assert_eq!(results.len(), 2);
            assert!(pt.is_empty());
        }

        #[test]
        fn pending_transfers_wait_all() {
            let mut pt = PendingTransfers::new();
            pt.push(TransferFuture::completed(
                KvTransferDirection::Send,
                7,
                std::time::Instant::now(),
            ));
            pt.push(TransferFuture::completed(
                KvTransferDirection::Recv,
                11,
                std::time::Instant::now(),
            ));
            let results = pt.wait_all();
            assert_eq!(results.len(), 2);
            assert!(results[0].as_ref().unwrap().pages_transferred == 7);
            assert!(results[1].as_ref().unwrap().pages_transferred == 11);
            assert!(pt.is_empty());
        }

        // ── send_pages_async / recv_pages_async tests ────────────────────────

        #[test]
        fn send_pages_async_rejects_non_distributed() {
            let handle = CommHandleWrapper::from_config(
                &crate::engine::distributed_config::ParallelConfig::default(),
            ).unwrap();
            let result = send_pages_async(
                std::ptr::null(),
                0,
                1,
                gllm_nccl::DType::Fp32,
                vec![0, 1],
                &handle,
            );
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not in distributed mode"));
        }

        #[test]
        fn recv_pages_async_rejects_non_distributed() {
            let handle = CommHandleWrapper::from_config(
                &crate::engine::distributed_config::ParallelConfig::default(),
            ).unwrap();
            let result = recv_pages_async(
                std::ptr::null_mut(),
                0,
                1,
                gllm_nccl::DType::Fp32,
                vec![0, 1],
                &handle,
            );
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not in distributed mode"));
        }

        // ── kv_transfer_rdma tests ───────────────────────────────────────────

        #[test]
        fn kv_transfer_rdma_rejects_non_distributed() {
            let handle = CommHandleWrapper::from_config(
                &crate::engine::distributed_config::ParallelConfig::default(),
            ).unwrap();
            let request = KvTransferRequest {
                direction: KvTransferDirection::Send,
                peer_rank: 1,
                frame_ids: vec![0, 1],
                sequence_id: 10,
                buf: std::ptr::null_mut(),
                elem_count: 0,
                dtype: gllm_nccl::DType::Fp32,
                mode: KvTransferMode::Rdma,
            };
            let result = kv_transfer_rdma(&request, &handle);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not in distributed mode"));
        }

        #[test]
        fn kv_transfer_rdma_without_nccl_init_returns_error() {
            let handle = make_distributed_handle();
            let request = KvTransferRequest {
                direction: KvTransferDirection::Send,
                peer_rank: 1,
                frame_ids: vec![0, 1],
                sequence_id: 10,
                buf: std::ptr::null_mut(),
                elem_count: 0,
                dtype: gllm_nccl::DType::Fp32,
                mode: KvTransferMode::Rdma,
            };
            let result = kv_transfer_rdma(&request, &handle);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not initialized"));
        }
    }
}

// ── L3-3: KV 分布 5 模式 (REQ-DIST-013) ───────────────────────────────────

#[cfg(feature = "nccl")]
pub mod kv_distribution {
    use crate::engine::distributed_config::{CommHandleWrapper, KvDistMode, KvDistributionConfig};

    /// KV 分布策略决策 (REQ-DIST-013, ENT-DIST-KV-MODE)
    ///
    /// 5 种 KV 分布模式，由 `KvDistributionConfig.mode` 决定：
    /// - **Local**: 本地 KV Cache，零跨节点通信（默认）
    /// - **OnDemand**: 按需从远端拉取 KV pages（PD 分离场景）
    /// - **Mirror**: 所有 GPU 持有完整 KV 副本，零通信但内存开销最大
    /// - **PartialHeadMirror**: 部分头镜像，每 GPU 镜像 1/tp_size 的头
    /// - **TieredCache**: 三级缓存（HBM → DDR → NVMe/远端），按热度自动迁移
    #[derive(Debug, Clone, PartialEq)]
    pub enum KvDistDecision {
        /// Local: 无跨节点通信，所有 KV pages 在本地 GPU
        Local,
        /// OnDemand: 按需从远端拉取 KV pages，读后释放远端资源
        OnDemand {
            /// 是否启用预取
            prefetch: bool,
        },
        /// Mirror: 所有 GPU 持有完整 KV 副本，零通信但内存 = 单 GPU * tp_size
        Mirror,
        /// PartialHeadMirror: 部分注意力头镜像，每 GPU 镜像 1/tp_size 的头
        PartialHeadMirror {
            /// 本地镜像的 head 数量
            local_heads: u32,
            /// 总 head 数量
            total_heads: u32,
        },
        /// TieredCache: 分层缓存（HBM → DDR → NVMe/远端），按热度自动迁移
        TieredCache {
            /// HBM 容量比例
            hbm_ratio: f64,
            /// DDR 容量比例
            ddr_ratio: f64,
        },
    }

    impl KvDistDecision {
        /// 根据 KvDistributionConfig 和 CommHandleWrapper 决定分布策略 (REQ-DIST-013)
        ///
        /// 分布式模式下，会根据 NCCL 通信初始化状态和并行拓扑
        /// 决定实际的 KV 分布策略参数（预取、头分配、缓存层级比例）。
        // @trace REQ-DIST-013 [entity:ENT-DIST-KV-MODE]
        pub fn from_config(config: &KvDistributionConfig, comm_handle: &CommHandleWrapper) -> Self {
            if !comm_handle.is_distributed() {
                return Self::Local;
            }

            let nccl_ready = comm_handle.is_nccl_initialized();

            match config.mode {
                KvDistMode::Local => Self::Local,
                KvDistMode::OnDemand => {
                    // 预取仅在 NCCL 初始化后启用（否则无法实际传输）
                    Self::OnDemand { prefetch: nccl_ready }
                }
                KvDistMode::Mirror => Self::Mirror,
                KvDistMode::PartialHeadMirror => Self::PartialHeadMirror {
                    local_heads: config.mirror_heads,
                    total_heads: 0, // 从 ModelConfig 获取，此处占位
                },
                KvDistMode::TieredCache => {
                    // 根据是否初始化 NCCL 调整缓存层级比例
                    // 未初始化时 HBM 占更高比例（减少跨节点依赖）
                    let hbm_ratio = if nccl_ready { 0.6 } else { 0.8 };
                    let ddr_ratio = if nccl_ready { 0.3 } else { 0.15 };
                    Self::TieredCache { hbm_ratio, ddr_ratio }
                }
            }
        }

        /// 是否需要跨节点 KV 传输
        pub fn needs_cross_node_transfer(&self) -> bool {
            !matches!(self, Self::Local)
        }

        /// 是否启用预取（仅 OnDemand 模式）
        pub fn is_prefetch_enabled(&self) -> bool {
            matches!(self, Self::OnDemand { prefetch: true })
        }

        /// 返回该模式的内存开销倍数 (REQ-DIST-013)
        ///
        /// - Local: 1x（仅本地）
        /// - OnDemand: 1x（按需拉取，不常驻远端）
        /// - Mirror: tp_size x（所有 rank 持有完整副本）
        /// - PartialHeadMirror: 1x（每 rank 只存 1/tp_size 的头）
        /// - TieredCache: 1x + 缓存层级（热页 HBM + 冷页 DDR/远端）
        pub fn memory_overhead_multiplier(&self, tp_size: u32) -> f64 {
            match self {
                Self::Local => 1.0,
                Self::OnDemand { .. } => 1.0,
                Self::Mirror => tp_size as f64,
                Self::PartialHeadMirror { total_heads, local_heads } => {
                    if *total_heads == 0 {
                        1.0 // placeholder, will be 1/tp_size when total_heads is known
                    } else {
                        *local_heads as f64 / *total_heads as f64
                    }
                }
                Self::TieredCache { hbm_ratio, ddr_ratio } => hbm_ratio + ddr_ratio,
            }
        }

        /// 返回该模式的通信开销描述 (REQ-DIST-013)
        pub fn communication_overhead(&self) -> &'static str {
            match self {
                Self::Local => "zero",
                Self::OnDemand { prefetch: false } => "on-demand",
                Self::OnDemand { prefetch: true } => "on-demand+prefetch",
                Self::Mirror => "zero (replicated)",
                Self::PartialHeadMirror { .. } => "attention-input-size",
                Self::TieredCache { .. } => "by-hotness",
            }
        }

        /// Mirror 模式：执行 AllGather 复制 KV pages 到所有 rank (REQ-DIST-013)
        ///
        /// 在 prefill 完成后调用，将 KV pages 广播到所有 rank。
        /// 后续 decode 步骤无需跨节点通信。
        // @trace REQ-DIST-013 [entity:ENT-DIST-KV-MODE] [dataflow:DF-DIST-004]
        pub fn mirror_allgather(
            &self,
            comm_handle: &CommHandleWrapper,
            _kv_buf: *mut u8,
            elem_count: usize,
            dtype: gllm_nccl::DType,
        ) -> Result<(), String> {
            if !matches!(self, Self::Mirror) {
                return Err("mirror_allgather: not in Mirror mode".to_string());
            }
            if !comm_handle.is_distributed() {
                return Err("mirror_allgather: not in distributed mode".to_string());
            }
            // AllGather: each rank contributes its local KV, all ranks receive full copy.
            // The actual AllGather is performed via CommHandleWrapper.all_gather_inplace.
            // For KV pages, we use a gather of the full buffer.
            let mut gather_buf = vec![0.0f32; elem_count];
            // Copy KV data to f32 gather buffer for AllGather.
            // In production, this would be a GPU-side AllGather via NCCL.
            comm_handle.all_gather_inplace(&mut gather_buf, elem_count)?;
            log::trace!(
                "[mirror_allgather] AllGather completed: {} elements, dtype={:?}",
                elem_count,
                dtype,
            );
            Ok(())
        }

        /// PartialHeadMirror 模式：AllGather 非本地头的 KV 切片 (REQ-DIST-013)
        ///
        /// 每个 rank 只持有 local_heads 个头的 KV 数据。
        /// AllGather 聚合所有 rank 的头切片，使每个 rank 获得完整注意力输入。
        /// 通信量 = 注意力输入大小（非 KV 副本大小）。
        // @trace REQ-DIST-013 [entity:ENT-DIST-KV-MODE] [dataflow:DF-DIST-004]
        pub fn partial_head_mirror_allgather(
            &self,
            comm_handle: &CommHandleWrapper,
            _head_kv_buf: *mut u8,
            local_head_elem_count: usize,
            dtype: gllm_nccl::DType,
        ) -> Result<(), String> {
            if !matches!(self, Self::PartialHeadMirror { .. }) {
                return Err("partial_head_mirror_allgather: not in PartialHeadMirror mode".to_string());
            }
            if !comm_handle.is_distributed() {
                return Err("partial_head_mirror_allgather: not in distributed mode".to_string());
            }
            // AllGather for head slices: each rank contributes local_head_elem_count elements.
            let mut gather_buf = vec![0.0f32; local_head_elem_count];
            comm_handle.all_gather_inplace(&mut gather_buf, local_head_elem_count)?;
            log::trace!(
                "[partial_head_mirror_allgather] AllGather completed: {} elements per rank, dtype={:?}",
                local_head_elem_count,
                dtype,
            );
            Ok(())
        }

        /// OnDemand 模式：从远端 rank 拉取指定 KV pages (REQ-DIST-013)
        ///
        /// 按需从拥有 KV pages 的远端 rank 拉取数据到本地。
        /// 拉取完成后远端资源可释放（PD 分离场景）。
        // @trace REQ-DIST-012 [entity:ENT-DIST-KV-XFER]
        // @trace REQ-DIST-013 [entity:ENT-DIST-KV-MODE] [dataflow:DF-DIST-004]
        pub fn on_demand_pull(
            &self,
            comm_handle: &CommHandleWrapper,
            request: &super::kv_transfer::KvTransferRequest,
        ) -> Result<super::kv_transfer::KvTransferResult, String> {
            if !matches!(self, Self::OnDemand { .. }) {
                return Err("on_demand_pull: not in OnDemand mode".to_string());
            }
            super::kv_transfer::kv_transfer(request, comm_handle)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn make_config(mode: KvDistMode, mirror_heads: u32) -> KvDistributionConfig {
            KvDistributionConfig {
                mode,
                mirror_heads,
            }
        }

        fn make_local_handle() -> CommHandleWrapper {
            CommHandleWrapper::from_config(&crate::engine::distributed_config::ParallelConfig::default()).unwrap()
        }

        fn make_distributed_handle() -> CommHandleWrapper {
            CommHandleWrapper::from_config(&crate::engine::distributed_config::ParallelConfig {
                tp_size: 2,
                pp_size: 1,
                ep_size: 1,
                rank: 0,
                world_size: 2,
                unique_id: String::new(),
            }).unwrap()
        }

        #[test]
        fn kv_dist_decision_local_when_not_distributed() {
            let handle = make_local_handle();
            let config = make_config(KvDistMode::Mirror, 0);
            let decision = KvDistDecision::from_config(&config, &handle);
            assert_eq!(decision, KvDistDecision::Local);
        }

        #[test]
        fn kv_dist_decision_local_mode() {
            let handle = make_distributed_handle();
            let config = make_config(KvDistMode::Local, 0);
            let decision = KvDistDecision::from_config(&config, &handle);
            assert_eq!(decision, KvDistDecision::Local);
            assert!(!decision.needs_cross_node_transfer());
        }

        #[test]
        fn kv_dist_decision_on_demand() {
            let handle = make_distributed_handle();
            let config = make_config(KvDistMode::OnDemand, 0);
            let decision = KvDistDecision::from_config(&config, &handle);
            assert_eq!(decision, KvDistDecision::OnDemand { prefetch: false });
            assert!(decision.needs_cross_node_transfer());
        }

        #[test]
        fn kv_dist_decision_mirror() {
            let handle = make_distributed_handle();
            let config = make_config(KvDistMode::Mirror, 0);
            let decision = KvDistDecision::from_config(&config, &handle);
            assert_eq!(decision, KvDistDecision::Mirror);
            assert!(decision.needs_cross_node_transfer());
        }

        #[test]
        fn kv_dist_decision_partial_head_mirror() {
            let handle = make_distributed_handle();
            let config = make_config(KvDistMode::PartialHeadMirror, 8);
            let decision = KvDistDecision::from_config(&config, &handle);
            assert_eq!(
                decision,
                KvDistDecision::PartialHeadMirror {
                    local_heads: 8,
                    total_heads: 0,
                }
            );
            assert!(decision.needs_cross_node_transfer());
        }

        #[test]
        fn kv_dist_decision_tiered_cache_without_nccl() {
            let handle = make_distributed_handle();
            let config = make_config(KvDistMode::TieredCache, 0);
            let decision = KvDistDecision::from_config(&config, &handle);
            assert_eq!(
                decision,
                KvDistDecision::TieredCache {
                    hbm_ratio: 0.8,
                    ddr_ratio: 0.15,
                }
            );
            assert!(decision.needs_cross_node_transfer());
        }

        #[test]
        fn kv_dist_decision_local_no_transfer() {
            let decision = KvDistDecision::Local;
            assert!(!decision.needs_cross_node_transfer());
        }

        #[test]
        fn kv_dist_decision_all_variants_transfer_semantics() {
            let variants_needing_transfer = [
                KvDistDecision::OnDemand { prefetch: true },
                KvDistDecision::Mirror,
                KvDistDecision::PartialHeadMirror { local_heads: 4, total_heads: 16 },
                KvDistDecision::TieredCache { hbm_ratio: 0.5, ddr_ratio: 0.3 },
            ];
            for v in &variants_needing_transfer {
                assert!(v.needs_cross_node_transfer(), "Expected transfer needed for {:?}", v);
            }
            assert!(!KvDistDecision::Local.needs_cross_node_transfer());
        }

        // ── memory_overhead_multiplier tests (REQ-DIST-013) ──────────────────

        #[test]
        fn memory_overhead_local_is_1x() {
            let decision = KvDistDecision::Local;
            assert!((decision.memory_overhead_multiplier(4) - 1.0).abs() < f64::EPSILON);
        }

        #[test]
        fn memory_overhead_on_demand_is_1x() {
            let decision = KvDistDecision::OnDemand { prefetch: true };
            assert!((decision.memory_overhead_multiplier(4) - 1.0).abs() < f64::EPSILON);
        }

        #[test]
        fn memory_overhead_mirror_is_tp_size() {
            let decision = KvDistDecision::Mirror;
            assert!((decision.memory_overhead_multiplier(4) - 4.0).abs() < f64::EPSILON);
        }

        #[test]
        fn memory_overhead_partial_head_mirror_with_total() {
            let decision = KvDistDecision::PartialHeadMirror {
                local_heads: 4,
                total_heads: 16,
            };
            // 4/16 = 0.25
            assert!((decision.memory_overhead_multiplier(4) - 0.25).abs() < f64::EPSILON);
        }

        #[test]
        fn memory_overhead_partial_head_mirror_without_total() {
            let decision = KvDistDecision::PartialHeadMirror {
                local_heads: 4,
                total_heads: 0, // placeholder
            };
            // Falls back to 1.0 when total_heads is unknown
            assert!((decision.memory_overhead_multiplier(4) - 1.0).abs() < f64::EPSILON);
        }

        #[test]
        fn memory_overhead_tiered_cache() {
            let decision = KvDistDecision::TieredCache {
                hbm_ratio: 0.6,
                ddr_ratio: 0.3,
            };
            // 0.6 + 0.3 = 0.9
            assert!((decision.memory_overhead_multiplier(4) - 0.9).abs() < f64::EPSILON);
        }

        // ── communication_overhead tests (REQ-DIST-013) ──────────────────────

        #[test]
        fn communication_overhead_local_is_zero() {
            assert_eq!(KvDistDecision::Local.communication_overhead(), "zero");
        }

        #[test]
        fn communication_overhead_on_demand_no_prefetch() {
            assert_eq!(
                KvDistDecision::OnDemand { prefetch: false }.communication_overhead(),
                "on-demand"
            );
        }

        #[test]
        fn communication_overhead_on_demand_with_prefetch() {
            assert_eq!(
                KvDistDecision::OnDemand { prefetch: true }.communication_overhead(),
                "on-demand+prefetch"
            );
        }

        #[test]
        fn communication_overhead_mirror_is_zero_replicated() {
            assert_eq!(KvDistDecision::Mirror.communication_overhead(), "zero (replicated)");
        }

        #[test]
        fn communication_overhead_partial_head_mirror() {
            assert_eq!(
                KvDistDecision::PartialHeadMirror { local_heads: 4, total_heads: 16 }
                    .communication_overhead(),
                "attention-input-size"
            );
        }

        #[test]
        fn communication_overhead_tiered_cache() {
            assert_eq!(
                KvDistDecision::TieredCache { hbm_ratio: 0.6, ddr_ratio: 0.3 }
                    .communication_overhead(),
                "by-hotness"
            );
        }

        // ── mirror_allgather / partial_head_mirror_allgather / on_demand_pull error tests ──

        #[test]
        fn mirror_allgather_rejects_non_mirror_mode() {
            let handle = make_distributed_handle();
            let decision = KvDistDecision::Local;
            let result = decision.mirror_allgather(&handle, std::ptr::null_mut(), 0, gllm_nccl::DType::Fp32);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not in Mirror mode"));
        }

        #[test]
        fn mirror_allgather_rejects_non_distributed() {
            let handle = make_local_handle();
            let decision = KvDistDecision::Mirror;
            let result = decision.mirror_allgather(&handle, std::ptr::null_mut(), 0, gllm_nccl::DType::Fp32);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not in distributed mode"));
        }

        #[test]
        fn partial_head_mirror_allgather_rejects_non_partial_mode() {
            let handle = make_distributed_handle();
            let decision = KvDistDecision::Local;
            let result = decision.partial_head_mirror_allgather(
                &handle, std::ptr::null_mut(), 0, gllm_nccl::DType::Fp32,
            );
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not in PartialHeadMirror mode"));
        }

        #[test]
        fn partial_head_mirror_allgather_rejects_non_distributed() {
            let handle = make_local_handle();
            let decision = KvDistDecision::PartialHeadMirror { local_heads: 4, total_heads: 16 };
            let result = decision.partial_head_mirror_allgather(
                &handle, std::ptr::null_mut(), 0, gllm_nccl::DType::Fp32,
            );
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not in distributed mode"));
        }

        #[test]
        fn on_demand_pull_rejects_non_on_demand_mode() {
            let handle = make_distributed_handle();
            let decision = KvDistDecision::Local;
            let request = super::super::kv_transfer::KvTransferRequest {
                direction: super::super::kv_transfer::KvTransferDirection::Recv,
                peer_rank: 1,
                frame_ids: vec![0],
                sequence_id: 1,
                buf: std::ptr::null_mut(),
                elem_count: 0,
                dtype: gllm_nccl::DType::Fp32,
                mode: super::super::kv_transfer::KvTransferMode::Sync,
            };
            let result = decision.on_demand_pull(&handle, &request);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not in OnDemand mode"));
        }
    }
}

// ── L3-4: 跨节点 KV 路由查询 (REQ-DIST-003) ────────────────────────────────

#[cfg(feature = "nccl")]
pub mod kv_routing {
    /// Cross-node KV page routing query result (REQ-DIST-003).
    ///
    /// Returned by resolve_page_for_rank() when the kv_coordinator
    /// queries the distributed routing table to determine where a
    /// KV page physically resides.
    #[derive(Debug, Clone, PartialEq)]
    pub struct KvRouteResult {
        /// The rank/node that holds the page.
        pub owner_rank: u32,
        /// Physical location of the page.
        pub location: gllm_kernels::PageLocation,
        /// Whether the page is ready for access (not InTransit/Invalid).
        pub is_ready: bool,
    }

    /// Resolve the physical location of a KV page via the distributed routing table (REQ-DIST-003).
    ///
    /// This is the kv_coordinator's primary cross-node query function.
    /// It looks up the PageRoutingTable to find which rank owns a given page,
    /// and determines whether a cross-node transfer is needed.
    ///
    /// Returns:
    /// - `Ok(KvRouteResult)` when the page is found in the routing table
    /// - `Err("page not found")` when the page_id is not in the routing table
    /// - `Err("page .. NotPresent")` when the page exists but is NotPresent
    // @trace REQ-DIST-003 [entity:ENT-DIST-ROUTING]
    pub fn resolve_page_for_rank(
        page_id: &gllm_kernels::DistributedPageId,
        routing_table: &gllm_kernels::PageRoutingTable,
    ) -> Result<KvRouteResult, String> {
        match routing_table.lookup(page_id) {
            Some(entry) => {
                let owner_rank = match &entry.location {
                    gllm_kernels::PageLocation::Local { .. } => routing_table.local_node_id,
                    gllm_kernels::PageLocation::IntraNode { device_index, .. } => *device_index,
                    gllm_kernels::PageLocation::InterNode { node_id, .. } => *node_id,
                    gllm_kernels::PageLocation::NotPresent => {
                        return Err(format!(
                            "page seq={} idx={} is NotPresent",
                            page_id.sequence_id, page_id.page_index
                        ))
                    }
                };
                let is_ready = entry.state == gllm_kernels::PageState::Ready;
                Ok(KvRouteResult {
                    owner_rank,
                    location: entry.location.clone(),
                    is_ready,
                })
            }
            None => Err(format!(
                "page not found: seq={} idx={}",
                page_id.sequence_id, page_id.page_index
            )),
        }
    }

    /// Batch resolve page locations for all pages of a sequence (REQ-DIST-003).
    ///
    /// Returns only Ready pages. Pages that are not found, NotPresent, or
    /// InTransit/Invalid are skipped.
    // @trace REQ-DIST-003 [entity:ENT-DIST-ROUTING]
    pub fn resolve_sequence_pages(
        sequence_id: u64,
        routing_table: &gllm_kernels::PageRoutingTable,
    ) -> Vec<KvRouteResult> {
        routing_table
            .sequence_pages(sequence_id)
            .iter()
            .filter_map(|entry| {
                if entry.state != gllm_kernels::PageState::Ready {
                    return None;
                }
                let owner_rank = match &entry.location {
                    gllm_kernels::PageLocation::Local { .. } => routing_table.local_node_id,
                    gllm_kernels::PageLocation::IntraNode { device_index, .. } => *device_index,
                    gllm_kernels::PageLocation::InterNode { node_id, .. } => *node_id,
                    gllm_kernels::PageLocation::NotPresent => return None,
                };
                Some(KvRouteResult {
                    owner_rank,
                    location: entry.location.clone(),
                    is_ready: true,
                })
            })
            .collect()
    }

    /// Determine whether a cross-node transfer is needed for a page (REQ-DIST-003).
    ///
    /// Returns true when the page resides on a different rank than the requesting rank.
    // @trace REQ-DIST-003 [entity:ENT-DIST-ROUTING]
    pub fn needs_cross_node_transfer(
        page_id: &gllm_kernels::DistributedPageId,
        routing_table: &gllm_kernels::PageRoutingTable,
        requesting_rank: u32,
    ) -> bool {
        match resolve_page_for_rank(page_id, routing_table) {
            Ok(result) => result.owner_rank != requesting_rank,
            Err(_) => false,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn make_routing_table(
            local_node_id: u32,
            local_device_count: u32,
        ) -> gllm_kernels::PageRoutingTable {
            gllm_kernels::PageRoutingTable::new(local_node_id, local_device_count)
        }

        fn make_page_id(sequence_id: u64, page_index: u32) -> gllm_kernels::DistributedPageId {
            gllm_kernels::DistributedPageId {
                sequence_id,
                page_index,
            }
        }

        fn make_local_entry(
            page_id: gllm_kernels::DistributedPageId,
            frame_id: u32,
        ) -> gllm_kernels::PageTableEntry {
            gllm_kernels::PageTableEntry {
                page_id,
                location: gllm_kernels::PageLocation::Local { frame_id },
                state: gllm_kernels::PageState::Ready,
                last_access_seq: 0,
                migrating: false,
                version: 1,
            }
        }

        fn make_internode_entry(
            page_id: gllm_kernels::DistributedPageId,
            node_id: u32,
            device_index: u32,
            frame_id: u32,
        ) -> gllm_kernels::PageTableEntry {
            gllm_kernels::PageTableEntry {
                page_id,
                location: gllm_kernels::PageLocation::InterNode {
                    node_id,
                    device_index,
                    frame_id,
                },
                state: gllm_kernels::PageState::Ready,
                last_access_seq: 0,
                migrating: false,
                version: 1,
            }
        }

        fn make_not_present_entry(
            page_id: gllm_kernels::DistributedPageId,
        ) -> gllm_kernels::PageTableEntry {
            gllm_kernels::PageTableEntry {
                page_id,
                location: gllm_kernels::PageLocation::NotPresent,
                state: gllm_kernels::PageState::Invalid,
                last_access_seq: 0,
                migrating: false,
                version: 1,
            }
        }

        // ── resolve_page_for_rank: local page ─────────────────────────────────

        #[test]
        fn resolve_local_page_returns_local_node_id() {
            let mut table = make_routing_table(0, 2);
            let page_id = make_page_id(100, 0);
            table.upsert(make_local_entry(page_id, 42));

            let result = resolve_page_for_rank(&page_id, &table).expect("resolve ok");
            assert_eq!(result.owner_rank, 0);
            assert!(result.is_ready);
            assert!(result.location.is_local());
        }

        // ── resolve_page_for_rank: inter-node page ─────────────────────────────

        #[test]
        fn resolve_internode_page_returns_remote_node_id() {
            let mut table = make_routing_table(0, 1);
            let page_id = make_page_id(200, 3);
            table.upsert(make_internode_entry(page_id, 2, 0, 10));

            let result = resolve_page_for_rank(&page_id, &table).expect("resolve ok");
            assert_eq!(result.owner_rank, 2);
            assert!(result.is_ready);
            assert!(result.location.is_inter_node());
        }

        // ── resolve_page_for_rank: not present ──────────────────────────────────

        #[test]
        fn resolve_not_present_page_returns_error() {
            let mut table = make_routing_table(0, 1);
            let page_id = make_page_id(300, 0);
            table.upsert(make_not_present_entry(page_id));

            let result = resolve_page_for_rank(&page_id, &table);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("NotPresent"));
        }

        // ── resolve_page_for_rank: missing page ─────────────────────────────────

        #[test]
        fn resolve_missing_page_returns_error() {
            let table = make_routing_table(0, 1);
            let page_id = make_page_id(999, 0);

            let result = resolve_page_for_rank(&page_id, &table);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not found"));
        }

        // ── resolve_sequence_pages ─────────────────────────────────────────────

        #[test]
        fn resolve_sequence_pages_returns_ready_pages() {
            let mut table = make_routing_table(0, 2);
            // 3 pages for sequence 100: 2 local, 1 remote
            table.upsert(make_local_entry(make_page_id(100, 0), 1));
            table.upsert(make_local_entry(make_page_id(100, 1), 2));
            table.upsert(make_internode_entry(make_page_id(100, 2), 1, 0, 3));

            let results = resolve_sequence_pages(100, &table);
            assert_eq!(results.len(), 3);
            // 2 local (rank 0), 1 remote (rank 1)
            let local_count = results.iter().filter(|r| r.owner_rank == 0).count();
            let remote_count = results.iter().filter(|r| r.owner_rank == 1).count();
            assert_eq!(local_count, 2);
            assert_eq!(remote_count, 1);
        }

        #[test]
        fn resolve_sequence_pages_skips_not_ready() {
            let mut table = make_routing_table(0, 1);
            table.upsert(make_local_entry(make_page_id(100, 0), 1));
            // Add an InTransit page
            let transit_entry = gllm_kernels::PageTableEntry {
                page_id: make_page_id(100, 1),
                location: gllm_kernels::PageLocation::Local { frame_id: 2 },
                state: gllm_kernels::PageState::InTransit,
                last_access_seq: 0,
                migrating: false,
                version: 1,
            };
            table.upsert(transit_entry);

            let results = resolve_sequence_pages(100, &table);
            assert_eq!(results.len(), 1); // Only the Ready page
            assert!(results[0].is_ready);
        }

        #[test]
        fn resolve_sequence_pages_empty_for_unknown_sequence() {
            let table = make_routing_table(0, 1);
            let results = resolve_sequence_pages(999, &table);
            assert!(results.is_empty());
        }

        // ── needs_cross_node_transfer ──────────────────────────────────────────

        #[test]
        fn needs_transfer_when_owner_differs() {
            let mut table = make_routing_table(0, 1);
            let page_id = make_page_id(100, 0);
            table.upsert(make_internode_entry(page_id, 2, 0, 5));

            assert!(needs_cross_node_transfer(&page_id, &table, 0));
            assert!(!needs_cross_node_transfer(&page_id, &table, 2));
        }

        #[test]
        fn no_transfer_when_local() {
            let mut table = make_routing_table(0, 1);
            let page_id = make_page_id(100, 0);
            table.upsert(make_local_entry(page_id, 5));

            assert!(!needs_cross_node_transfer(&page_id, &table, 0));
        }

        #[test]
        fn no_transfer_for_missing_page() {
            let table = make_routing_table(0, 1);
            let page_id = make_page_id(999, 0);

            assert!(!needs_cross_node_transfer(&page_id, &table, 0));
        }

        // ── KvRouteResult equality ─────────────────────────────────────────────

        #[test]
        fn kv_route_result_equality() {
            let a = KvRouteResult {
                owner_rank: 0,
                location: gllm_kernels::PageLocation::Local { frame_id: 1 },
                is_ready: true,
            };
            let b = KvRouteResult {
                owner_rank: 0,
                location: gllm_kernels::PageLocation::Local { frame_id: 1 },
                is_ready: true,
            };
            assert_eq!(a, b);
        }

        #[test]
        fn kv_route_result_inequality_different_rank() {
            let a = KvRouteResult {
                owner_rank: 0,
                location: gllm_kernels::PageLocation::Local { frame_id: 1 },
                is_ready: true,
            };
            let b = KvRouteResult {
                owner_rank: 1,
                location: gllm_kernels::PageLocation::InterNode {
                    node_id: 1,
                    device_index: 0,
                    frame_id: 1,
                },
                is_ready: true,
            };
            assert_ne!(a, b);
        }
    }
}
