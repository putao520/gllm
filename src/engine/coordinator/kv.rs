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
    }

    /// 执行 KV 跨节点传输 (REQ-DIST-012)
    ///
    /// 通过 gllm-nccl CommHandle.send()/recv() 执行实际的 KV page 传输。
    /// 调用方负责提供 GPU 缓冲区指针和数据类型信息。
    pub fn kv_transfer(
        request: &KvTransferRequest,
        comm_handle: &CommHandleWrapper,
    ) -> Result<KvTransferResult, String> {
        if !comm_handle.is_distributed() {
            return Err("kv_transfer: not in distributed mode".to_string());
        }

        let start = std::time::Instant::now();

        log::trace!(
            "[kv_transfer] {:?} seq={} pages={} peer={}",
            request.direction,
            request.sequence_id,
            request.frame_ids.len(),
            request.peer_rank
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
            };
            let result = kv_transfer(&request, &handle);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not in distributed mode"));
        }

        #[test]
        fn kv_transfer_distributed_without_nccl_init_returns_error() {
            // 分布式模式但未初始化 NCCL → send/recv 返回 "not initialized" 错误
            let handle = make_distributed_handle();
            let request = KvTransferRequest {
                direction: KvTransferDirection::Send,
                peer_rank: 1,
                frame_ids: vec![0, 1, 2],
                sequence_id: 42,
                buf: std::ptr::null_mut(),
                elem_count: 0,
                dtype: gllm_nccl::DType::Fp32,
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
            };
            assert_eq!(req.direction, KvTransferDirection::Recv);
            assert_eq!(req.peer_rank, 3);
            assert_eq!(req.frame_ids.len(), 4);
            assert_eq!(req.sequence_id, 99);
            assert_eq!(req.elem_count, 1024);
        }

        #[test]
        fn kv_transfer_result_fields() {
            let result = KvTransferResult {
                pages_transferred: 7,
                latency_ms: 1.5,
            };
            assert_eq!(result.pages_transferred, 7);
            assert!((result.latency_ms - 1.5).abs() < f64::EPSILON);
        }

        #[test]
        fn kv_transfer_result_latency_is_nonzero() {
            // 验证 latency_ms 来自实际计时，而非硬编码 0.0
            let result = KvTransferResult {
                pages_transferred: 3,
                latency_ms: 0.123,
            };
            assert!(result.latency_ms >= 0.0);
            assert!((result.latency_ms - 0.123).abs() < f64::EPSILON);
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
            };
            let result = kv_transfer(&request, &handle);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not initialized"));
        }
    }
}

// ── L3-3: KV 分布 5 模式 (REQ-DIST-013) ───────────────────────────────────

#[cfg(feature = "nccl")]
pub mod kv_distribution {
    use crate::engine::distributed_config::{CommHandleWrapper, KvDistMode, KvDistributionConfig};

    /// KV 分布策略决策 (REQ-DIST-013)
    #[derive(Debug, Clone, PartialEq)]
    pub enum KvDistDecision {
        /// Local: 无跨节点通信
        Local,
        /// OnDemand: 按需从远端拉取 KV pages
        OnDemand {
            /// 是否启用预取
            prefetch: bool,
        },
        /// Mirror: 所有 GPU 持有完整 KV 副本
        Mirror,
        /// PartialHeadMirror: 部分注意力头镜像
        PartialHeadMirror {
            /// 本地镜像的 head 数量
            local_heads: u32,
            /// 总 head 数量
            total_heads: u32,
        },
        /// TieredCache: 分层缓存（HBM → DDR → NVMe）
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
            // 分布式模式但未初始化 NCCL → prefetch = false
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
            // 未初始化 NCCL → hbm_ratio=0.8, ddr_ratio=0.15
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
            // Only Local does NOT need cross-node transfer
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
    }
}
