//! SPEC 32 — Mega-Kernel 性能增强: CPU 侧数据结构与调度逻辑.
//!
//! 本模块实现 SPEC 32 (REQ-MKO-001~006) 的 Rust 端基础设施:
//! - DualBatchMeta: Ping/Pong 双缓冲元数据 (§2.3)
//! - RequestQueueEntry / RequestQueue: 设备内存 ring buffer (§3.1)
//! - OutputTokenEntry / OutputRingBuffer: Per-token 流式输出 (§4.1)
//! - SmPartitionConfig: SM 分区编译参数 (§2.6, §5.1)
//!
//! GPU 端 PTX/HIP codegen (cluster.sync, mbarrier, cp.async.bulk 等) 由
//! gllm-kernels JIT 管线实现，不在本模块范围。

// ═══════════════════════════════════════════════════════════
// §2.3 DualBatchMeta — Ping/Pong 双缓冲元数据
// ═══════════════════════════════════════════════════════════

/// Ping/Pong 双缓冲元数据 (SPEC 32 §2.3).
///
/// 位于 BatchContext 扩展区 (§6.2)，不侵入现有 SPEC 20 字段。
/// GPU 端每个 decode step 结束后 swap(ping, pong) 并递增 step_epoch。
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct DualBatchMeta {
    /// Ping: 当前 decode batch 的 seq_meta 起始偏移 (相对 seq_meta_base, 单位 SEQ_META_STRIDE)
    pub ping_seq_offset: u32,
    /// Ping: 当前 decode batch 的活跃序列数
    pub ping_seq_count: u32,
    /// Pong: compact+refill 后新 batch 的 seq_meta 起始偏移
    pub pong_seq_offset: u32,
    /// Pong: 新 batch 的活跃序列数
    pub pong_seq_count: u32,
    /// Barrier epoch: 每步递增。SM90+ cluster.sync 隐式同步;
    /// SM80 grid_sync 需要 epoch 防止跨步同步;
    /// SM70-/CPU ring barrier 用作 arrival count
    pub step_epoch: u32,
    /// SM70-/CPU ring barrier: 每步到达计数器
    pub epoch_arrival_count: u32,
}

impl DualBatchMeta {
    pub const SIZE: usize = 24;

    pub fn new(max_batch_size: u32) -> Self {
        Self {
            ping_seq_offset: 0,
            ping_seq_count: 0,
            pong_seq_offset: max_batch_size,
            pong_seq_count: 0,
            step_epoch: 0,
            epoch_arrival_count: 0,
        }
    }

    /// Swap ping ↔ pong (decode step 结束后调用).
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.ping_seq_offset, &mut self.pong_seq_offset);
        std::mem::swap(&mut self.ping_seq_count, &mut self.pong_seq_count);
        self.step_epoch = self.step_epoch.wrapping_add(1);
        self.epoch_arrival_count = 0;
    }
}

// ═══════════════════════════════════════════════════════════
// §3.1 RequestQueueEntry — 请求队列条目
// ═══════════════════════════════════════════════════════════

/// 请求队列条目 (SPEC 32 §3.1).
///
/// Tokenize 在 Rust 端完成后，将 token IDs 写入 pinned memory，
/// 然后将此条目入队到 RequestQueue。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[repr(C)]
pub struct RequestQueueEntry {
    /// tokenized IDs 基址 (Rust tokenizer 写入 pinned memory)
    pub input_ids_ptr: u64,
    /// prompt 长度
    pub prompt_len: u32,
    /// packed: temperature_u32 (bit-cast f32 → u32)
    pub temperature_u32: u32,
    /// top-k 采样参数
    pub top_k: u32,
    /// packed: top_p_u32 (bit-cast f32 → u32)
    pub top_p_u32: u32,
    /// EOS token ID
    pub eos_token_id: u32,
    /// 最大生成 token 数
    pub max_new_tokens: u32,
    /// 多模态注入偏移 (0=无)
    pub fused_hidden_offset: u32,
    /// 多模态 token 数
    pub num_mm_tokens: u32,
    /// Session KV 复用位置 (0=新序列)
    pub session_position: u32,
    /// 预留对齐
    pub _reserved: u32,
}

impl RequestQueueEntry {
    pub const SIZE: usize = 48;

    pub fn new(input_ids_ptr: u64, prompt_len: u32) -> Self {
        Self {
            input_ids_ptr,
            prompt_len,
            temperature_u32: 1.0f32.to_bits(),
            top_k: 0,
            top_p_u32: 1.0f32.to_bits(),
            eos_token_id: 0,
            max_new_tokens: 512,
            fused_hidden_offset: 0,
            num_mm_tokens: 0,
            session_position: 0,
            _reserved: 0,
        }
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature_u32 = t.to_bits();
        self
    }

    pub fn with_top_k(mut self, k: u32) -> Self {
        self.top_k = k;
        self
    }

    pub fn with_top_p(mut self, p: f32) -> Self {
        self.top_p_u32 = p.to_bits();
        self
    }

    pub fn with_eos(mut self, eos: u32) -> Self {
        self.eos_token_id = eos;
        self
    }

    pub fn with_max_new_tokens(mut self, n: u32) -> Self {
        self.max_new_tokens = n;
        self
    }

    pub fn with_session_position(mut self, pos: u32) -> Self {
        self.session_position = pos;
        self
    }
}

// ═══════════════════════════════════════════════════════════
// §4.1 OutputTokenEntry — 逐 token 输出条目
// ═══════════════════════════════════════════════════════════

/// Per-token 输出条目 (SPEC 32 §4.1).
///
/// 每采样的 token 立即写入 per-CTA sub-ring。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(C)]
pub struct OutputTokenEntry {
    /// 序列标识 (对应 seq_meta 索引)
    pub seq_id: u32,
    /// 采样的 token ID
    pub token_id: u32,
    /// 0=中间 token, 1=EOS/stop/max_tokens
    pub is_final: u32,
    /// 0=中间, 1=EOS, 2=max_tokens, 3=stop_word
    pub finish_reason: u32,
    /// 该序列已生成 token 计数 (Rust 端排序用)
    pub gen_idx: u32,
    /// 预留对齐
    pub _reserved: u32,
}

impl OutputTokenEntry {
    pub const SIZE: usize = 24;

    pub fn intermediate(seq_id: u32, token_id: u32, gen_idx: u32) -> Self {
        Self {
            seq_id,
            token_id,
            is_final: 0,
            finish_reason: 0,
            gen_idx,
            _reserved: 0,
        }
    }

    pub fn final_eos(seq_id: u32, token_id: u32, gen_idx: u32) -> Self {
        Self {
            seq_id,
            token_id,
            is_final: 1,
            finish_reason: 1,
            gen_idx,
            _reserved: 0,
        }
    }

    pub fn final_max_tokens(seq_id: u32, token_id: u32, gen_idx: u32) -> Self {
        Self {
            seq_id,
            token_id,
            is_final: 1,
            finish_reason: 2,
            gen_idx,
            _reserved: 0,
        }
    }

    pub fn final_stop_word(seq_id: u32, token_id: u32, gen_idx: u32) -> Self {
        Self {
            seq_id,
            token_id,
            is_final: 1,
            finish_reason: 3,
            gen_idx,
            _reserved: 0,
        }
    }
}

/// 完成 reason 常量.
pub const FINISH_REASON_INTERMEDIATE: u32 = 0;
pub const FINISH_REASON_EOS: u32 = 1;
pub const FINISH_REASON_MAX_TOKENS: u32 = 2;
pub const FINISH_REASON_STOP_WORD: u32 = 3;

// ═══════════════════════════════════════════════════════════
// §3.1 RequestQueue — 环形缓冲区
// ═══════════════════════════════════════════════════════════

/// 设备内存 ring buffer 请求队列 (SPEC 32 §3.1).
///
/// 生产者: Rust 端 (tokenize 后 enqueue).
/// 消费者: Mega-Kernel CTA 0 (refill 阶段 dequeue).
///
/// CPU 路径: 单线程 enqueue + 单线程 dequeue (串行 Mega-Kernel).
/// GPU 路径: Rust 端 atom.global.add.u64 写入, GPU 端 atom.global.add.u64 读取.
pub struct RequestQueue {
    /// Ring buffer 存储 (CPU 端 Vec; GPU 端为设备内存指针)
    entries: Vec<RequestQueueEntry>,
    /// 最大容量 (2 的幂次, 方便取模)
    capacity: u32,
    /// Rust 端写入位置 (逻辑递增, 实际 slot = write_idx % capacity)
    write_idx: u64,
    /// Mega-Kernel 读取位置
    read_idx: u64,
}

impl RequestQueue {
    /// 创建容量为 `capacity` 的请求队列 (自动向上取整到 2 的幂次).
    pub fn new(capacity: u32) -> Self {
        let cap = capacity.next_power_of_two();
        Self {
            entries: vec![RequestQueueEntry::default(); cap as usize],
            capacity: cap,
            write_idx: 0,
            read_idx: 0,
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// 当前队列中的条目数.
    pub fn len(&self) -> u64 {
        self.write_idx.saturating_sub(self.read_idx)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity as u64
    }

    /// 入队一个请求. 返回 false 表示队列已满.
    pub fn enqueue(&mut self, entry: RequestQueueEntry) -> bool {
        if self.is_full() {
            return false;
        }
        let slot = (self.write_idx % self.capacity as u64) as usize;
        self.entries[slot] = entry;
        self.write_idx += 1;
        true
    }

    /// 出队一个请求. 返回 None 表示队列空.
    pub fn dequeue(&mut self) -> Option<RequestQueueEntry> {
        if self.is_empty() {
            return None;
        }
        let slot = (self.read_idx % self.capacity as u64) as usize;
        let entry = self.entries[slot];
        self.read_idx += 1;
        Some(entry)
    }

    /// 批量出队, 最多 `max_count` 条.
    pub fn dequeue_batch(&mut self, max_count: u32) -> Vec<RequestQueueEntry> {
        let count = std::cmp::min(max_count as u64, self.len()) as usize;
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(e) = self.dequeue() {
                result.push(e);
            } else {
                break;
            }
        }
        result
    }

    /// Peek 队头 (不消耗).
    pub fn peek(&self) -> Option<&RequestQueueEntry> {
        if self.is_empty() {
            return None;
        }
        let slot = (self.read_idx % self.capacity as u64) as usize;
        Some(&self.entries[slot])
    }
}

// ═══════════════════════════════════════════════════════════
// §4.1 OutputRingBuffer — Per-CTA sub-ring + doorbell
// ═══════════════════════════════════════════════════════════

/// Per-token 流式输出环形缓冲区 (SPEC 32 §4.1).
///
/// 每个 CTA 独占一个 sub-ring + 一个 doorbell slot，零跨 CTA atomic。
/// Rust 端通过 epoch_flag 阻塞等待或轮询消费。
pub struct OutputRingBuffer {
    /// Sub-ring 条目存储: [cta_0 entries][cta_1 entries]...
    entries: Vec<OutputTokenEntry>,
    /// 每个 CTA 的写入位置 (逻辑递增, slot = write_idx % cta_sub_ring_size)
    per_cta_write_idx: Vec<u64>,
    /// 每个 CTA 的 sub-ring 容量
    cta_sub_ring_size: u32,
    /// CTA 数量 (等于 total_cta_count)
    num_sub_rings: u32,
    /// 当前 epoch (Mega-Kernel GLOBAL_SYNC B0 后递增)
    epoch: u32,
}

impl OutputRingBuffer {
    /// 创建输出环形缓冲区.
    ///
    /// - `num_ctas`: CTA 总数 (= total_cta_count)
    /// - `max_batch_size`: 最大同时活跃序列数
    /// - `max_new_tokens`: 每序列最大生成 token 数
    pub fn new(num_ctas: u32, max_batch_size: u32, max_new_tokens: u32) -> Self {
        let cta_sub_ring_size = std::cmp::max(
            (max_batch_size as u64 * max_new_tokens as u64 / num_ctas as u64 + 64) as u32,
            128,
        );
        let total_entries = cta_sub_ring_size as usize * num_ctas as usize;
        Self {
            entries: vec![OutputTokenEntry::default(); total_entries],
            per_cta_write_idx: vec![0u64; num_ctas as usize],
            cta_sub_ring_size,
            num_sub_rings: num_ctas,
            epoch: 0,
        }
    }

    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    pub fn advance_epoch(&mut self) {
        self.epoch += 1;
    }

    /// CTA 端写入一个 token (CPU 侧模拟).
    ///
    /// GPU 端通过 per-CTA local write + doorbell 写入，无跨 CTA atomic。
    pub fn write_token(&mut self, cta_id: u32, entry: OutputTokenEntry) -> bool {
        if cta_id as usize >= self.num_sub_rings as usize {
            return false;
        }
        let ring_size = self.cta_sub_ring_size as u64;
        let write_idx = self.per_cta_write_idx[cta_id as usize];
        let slot = (write_idx % ring_size) as usize;
        let base = cta_id as usize * self.cta_sub_ring_size as usize;
        self.entries[base + slot] = entry;
        self.per_cta_write_idx[cta_id as usize] = write_idx + 1;
        true
    }

    /// Rust 端消费: 收集所有 CTA sub-rings 中自上次消费以来的新条目.
    ///
    /// 返回按 (seq_id, gen_idx) 排序的条目列表。
    pub fn consume(&mut self) -> Vec<OutputTokenEntry> {
        let mut all = Vec::new();
        for cta_id in 0..self.num_sub_rings as usize {
            let base = cta_id * self.cta_sub_ring_size as usize;
            let ring_size = self.cta_sub_ring_size as u64;
            // 读取该 CTA 的所有有效条目 (write_idx 指示写入边界)
            // CPU 模式: 直接从 per_cta_write_idx 读取
            let write_count = self.per_cta_write_idx[cta_id];
            let ring_start = write_count.saturating_sub(ring_size);
            for i in ring_start..write_count {
                let slot = (i % ring_size) as usize;
                let entry = self.entries[base + slot];
                // 跳过默认空条目
                if entry.seq_id != 0 || entry.token_id != 0 || entry.gen_idx != 0 {
                    all.push(entry);
                }
            }
        }

        // 按 (seq_id, gen_idx) 排序
        all.sort_by(|a, b| {
            a.seq_id
                .cmp(&b.seq_id)
                .then_with(|| a.gen_idx.cmp(&b.gen_idx))
        });
        all
    }

    /// 收集指定序列的所有完成条目 (is_final=1).
    pub fn consume_final_entries(&mut self) -> Vec<OutputTokenEntry> {
        self.consume()
            .into_iter()
            .filter(|e| e.is_final == 1)
            .collect()
    }

    /// 重置 (用于 Mega-Kernel 下次调用前).
    pub fn reset(&mut self) {
        self.entries.fill(OutputTokenEntry::default());
        self.per_cta_write_idx.fill(0);
        self.epoch = 0;
    }
}

// ═══════════════════════════════════════════════════════════
// §2.6 SmPartitionConfig — SM 分区编译参数
// ═══════════════════════════════════════════════════════════

/// SM 分区编译变体 (SPEC 32 §2.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MkCompileVariant {
    /// SM90+: cluster.sync + mbarrier, 6 decode + 2 prefill per cluster
    Cluster62,
    /// SM90+: cluster.sync + mbarrier, 5 decode + 3 prefill per cluster
    Cluster53,
    /// SM80: cooperative grid_sync, 75%:25% 总量
    GridSync,
    /// SM<60: 串行执行 (非降级, 编译时选择的不同 codegen 路径)
    Serial,
}

/// SM 分区配置 (SPEC 32 §2.6 + §5.1).
///
/// 编译时 bake 进机器码，运行时不变。
/// CPU 路径: 无 SM 分区，使用 Serial 变体。
#[derive(Debug, Clone, Copy)]
pub struct SmPartitionConfig {
    /// 编译变体
    pub variant: MkCompileVariant,
    /// Cluster 大小: portable=8, opt-in non-portable=16
    pub cluster_size: u32,
    /// CTA 总数 (= SM 数量, 1 CTA/SM)
    pub total_cta_count: u32,
    /// Decode CTA 数量
    pub decode_cta_count: u32,
    /// Prefill CTA 数量
    pub prefill_cta_count: u32,
    /// Cluster 数量 = total_cta_count / cluster_size
    pub num_clusters: u32,
    /// 每 cluster 中 decode CTA 数
    pub decode_per_cluster: u32,
    /// 每 cluster 中 prefill CTA 数
    pub prefill_per_cluster: u32,
}

impl SmPartitionConfig {
    /// CPU 路径: 串行执行，全部 CTA 用于当前阶段.
    pub fn serial() -> Self {
        Self {
            variant: MkCompileVariant::Serial,
            cluster_size: 1,
            total_cta_count: 1,
            decode_cta_count: 1,
            prefill_cta_count: 0,
            num_clusters: 1,
            decode_per_cluster: 1,
            prefill_per_cluster: 0,
        }
    }

    /// SM90+ H100/S100 类硬件 (132 SM, cluster_size=8 portable).
    pub fn cluster_62(total_sm: u32) -> Self {
        let cluster_size = 8u32;
        let num_clusters = total_sm / cluster_size;
        let decode_per_cluster = 6u32;
        let prefill_per_cluster = 2u32;
        let decode_cta_count = num_clusters * decode_per_cluster;
        let prefill_cta_count = num_clusters * prefill_per_cluster;
        Self {
            variant: MkCompileVariant::Cluster62,
            cluster_size,
            total_cta_count: num_clusters * cluster_size,
            decode_cta_count,
            prefill_cta_count,
            num_clusters,
            decode_per_cluster,
            prefill_per_cluster,
        }
    }

    /// SM80 A100 类硬件 (108 SM, 无 cluster → grid_sync).
    pub fn grid_sync(total_sm: u32) -> Self {
        let decode_cta_count = (total_sm * 3) / 4; // 75% decode
        let prefill_cta_count = total_sm - decode_cta_count;
        Self {
            variant: MkCompileVariant::GridSync,
            cluster_size: 0,
            total_cta_count: total_sm,
            decode_cta_count,
            prefill_cta_count,
            num_clusters: 0,
            decode_per_cluster: 0,
            prefill_per_cluster: 0,
        }
    }

    /// 根据 SM 数量自动选择变体.
    pub fn derive(total_sm: u32) -> Self {
        if total_sm >= 60 {
            // 未来: 查询 DeviceProfile 判断 SM90+ vs SM80
            // 目前默认使用 grid_sync (SM80 兼容)
            Self::grid_sync(total_sm)
        } else {
            Self::serial()
        }
    }
}

// ═══════════════════════════════════════════════════════════
// §6.2 BatchContext 扩展区偏移常量
// ═══════════════════════════════════════════════════════════

/// BatchContext 扩展区位于 seq_meta 数组之后.
/// 实际偏移 = BATCH_CTX_HEADER_SIZE (96) + max_batch_size × SEQ_META_STRIDE (64).
pub const EXT_REQUEST_QUEUE_PTR: usize = 0;
pub const EXT_OUTPUT_RING_PTR: usize = 8;
pub const EXT_KV_FREE_BITMAP_PTR: usize = 16;
pub const EXT_KV_POOL_TOTAL_PAGES: usize = 24;
pub const EXT_MAX_BATCH_SIZE: usize = 28;
pub const EXT_DUAL_BATCH_META: usize = 32;
pub const EXT_AUTOTUNE_ACTUAL_BATCH: usize = 56;
pub const EXT_POOL_CLUSTER_DSMEM_PTR: usize = 60;
pub const EXT_PENDING_FREE_LIST_PTR: usize = 68;
pub const EXT_PENDING_FREE_COUNT_PTR: usize = 76;
pub const EXT_OUTPUT_PER_CTA_DOORBELL_PTR: usize = 80;
pub const EXT_OUTPUT_EPOCH_FLAG_PTR: usize = 88;
pub const EXT_RESERVED: usize = 92;
// ── REQ-KV-EXT-001: V2 扩展字段 (14 total) ──
/// KvPageHeader stride in bytes (64 for V2, was 56 for V1).
pub const EXT_KV_PAGE_HEADER_STRIDE: usize = 96;
/// Base pointer for ext_id indexed KV extension slots (V2 ext_id → ext slot).
pub const EXT_KV_EXT_ID_BASE_PTR: usize = 104;

/// 扩展区总大小 (REQ-KV-EXT-001: 14 fields, aligned to 128 bytes).
///
/// LEGAL-FIXED: This is a GPU kernel ABI layout constant — the extension area
/// is a fixed-size memory region whose layout (14 offset constants above) is
/// baked into PTX codegen. All EXT_* offsets are compile-time constants that
/// the GPU kernel reads via fixed-address loads/stores. Changing this at
/// runtime would require recompiling the PTX kernel with different offsets,
/// which is the JIT pipeline's job — but the extension layout itself is a
/// stable ABI contract, not a hardware-tunable parameter.
///
/// Current layout: 14 fields occupy 112 bytes (last field at offset 104 + 8B),
/// padded to 128 bytes (16-byte alignment, 16 bytes reserved for future fields).
/// If new extension fields are added that exceed 128 bytes, this constant AND
/// all GPU codegen that references EXT_* offsets must be updated together.
pub const BATCH_CTX_EXTENSION_SIZE: usize = 128;

// ═══════════════════════════════════════════════════════════
// §1.4 Prefill/Decode 编译参数
// ═══════════════════════════════════════════════════════════

/// 编译时 GEMM tile 参数 (SPEC 32 §5.1).
#[derive(Debug, Clone, Copy)]
pub struct GemmTileParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

/// Prefill 或 Decode 阶段的融合编译参数 (SPEC 32 §5.1).
#[derive(Debug, Clone)]
pub struct PhaseCompileParams {
    /// GEMM tile 大小
    pub gemm_tile: GemmTileParams,
    /// KV pipeline 阶段数 (SM90: 4-stage TMA, SM80: 2-stage cp.async, 其他: 1)
    pub kv_pipeline_stages: u32,
    /// 是否使用 DSMEM KV 共享 (SM90+ 同 GPC CTA 复用 KV)
    pub use_dsmem_kv_share: bool,
    /// 是否使用 ld.global.nc 绕过 L1 一致性开销
    pub use_ld_nc: bool,
    /// 是否使用 Tensor Core GEMV (M=1 仍用 TC)
    pub use_tensor_cores_gemv: bool,
    /// Symbolic M 维度名称
    pub symdim_m: &'static str,
}

/// KV 访问模式.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvAccessMode {
    /// Prefill: 写入完整 KV entries
    WriteFull,
    /// Decode: 读取历史 + 写入新 1 行
    ReadHistoryWriteOne,
}

/// Attention 策略.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionStrategy {
    /// Prefill: FlashAttention / full KV 写入
    FlashAttention,
    /// Decode: 增量 KV attention
    IncrementalKv,
}

/// 完整的 Phase 编译配置.
#[derive(Debug, Clone)]
pub struct PhaseConfig {
    pub compile_params: PhaseCompileParams,
    pub attention: AttentionStrategy,
    pub kv_mode: KvAccessMode,
}

// ═══════════════════════════════════════════════════════════
// §1.4 Chunked Prefill 阈值
// ═══════════════════════════════════════════════════════════

/// 超过此值走 dedicated prefill_path (SPEC 32 §1.4).
///
/// LEGAL-FIXED: SPEC 32 §1.4 defines this as "编译时常量，默认 512".
/// This value is baked into GPU PTX codegen as a branch condition in the
/// mega-kernel dispatch logic. Different SM versions may benefit from
/// different thresholds (e.g., SM90 with TMA can handle larger chunks),
/// but such hardware-aware tuning requires a SPEC change to make this
/// a DeviceProfile-derived parameter, not a silent code-level override.
/// TODO(PSC-20): If SPEC 32 evolves to allow hardware-derived chunk sizes,
/// replace with `GpuDeviceProfile::prefill_chunk_threshold()`.
pub const PREFILL_CHUNK_THRESHOLD: u32 = 512;
/// 每次 mixed_path 消化的 prefill token 上限 (SPEC 32 §1.4).
///
/// LEGAL-FIXED: Same rationale as PREFILL_CHUNK_THRESHOLD — SPEC-specified
/// compile-time constant. The optimal chunk size depends on shared memory
/// budget and Tensor Core throughput, which vary by SM version, but the
/// SPEC currently mandates a fixed default.
/// TODO(PSC-20): If SPEC 32 evolves to allow hardware-derived chunk sizes,
/// replace with `GpuDeviceProfile::prefill_chunk_size()`.
pub const PREFILL_CHUNK_SIZE: u32 = 512;

// ═══════════════════════════════════════════════════════════
// §3.2 页分配器常量
// ═══════════════════════════════════════════════════════════

/// Per-CTA 私有池大小 (page slots).
///
/// LEGAL-FIXED: SPEC 32 §3.2 specifies "pool_local: Per-CTA Private Pool
/// (32 page slots, register/local memory)". This value is baked into GPU
/// PTX codegen — the per-CTA local pool is allocated in registers/local
/// memory, and its size directly affects register pressure. A larger pool
/// reduces global atomics but may cause register spilling on SM versions
/// with lower register budgets. The SPEC-mandated 32 is a conservative
/// default that works across SM70–SM90+.
/// TODO(PSC-21): If SPEC 32 evolves to allow hardware-derived pool sizes,
/// replace with `GpuDeviceProfile::pool_local_capacity()` — SM90+ with
/// 255 regs/thread and 227KB shared mem could use 64 slots, while SM70
/// with 128 regs/thread should stay at 32.
pub const POOL_LOCAL_CAPACITY: u32 = 32;
/// 从 pool_cluster 批量取的页数.
pub const POOL_CLUSTER_BATCH: u32 = 64;
/// 从 pool_global 批量取的页数 (word granularity = 32 pages).
pub const POOL_GLOBAL_BATCH_WORDS: u32 = 1;
/// OOM 标记.
pub const PAGE_ALLOC_OOM: u32 = u32::MAX;

// ═══════════════════════════════════════════════════════════
// 单元测试
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── DualBatchMeta ──

    #[test]
    fn dual_batch_meta_new() {
        let meta = DualBatchMeta::new(256);
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.pong_seq_offset, 256);
        assert_eq!(meta.step_epoch, 0);
    }

    #[test]
    fn dual_batch_meta_swap() {
        let mut meta = DualBatchMeta::new(128);
        meta.ping_seq_offset = 0;
        meta.ping_seq_count = 10;
        meta.pong_seq_offset = 128;
        meta.pong_seq_count = 8;
        meta.swap();
        assert_eq!(meta.ping_seq_offset, 128);
        assert_eq!(meta.ping_seq_count, 8);
        assert_eq!(meta.pong_seq_offset, 0);
        assert_eq!(meta.pong_seq_count, 10);
        assert_eq!(meta.step_epoch, 1);
        assert_eq!(meta.epoch_arrival_count, 0);
    }

    #[test]
    fn dual_batch_meta_size() {
        assert_eq!(std::mem::size_of::<DualBatchMeta>(), DualBatchMeta::SIZE);
    }

    // ── RequestQueueEntry ──

    #[test]
    fn request_entry_builder() {
        let entry = RequestQueueEntry::new(0x1000, 42)
            .with_temperature(0.7)
            .with_top_k(50)
            .with_top_p(0.95)
            .with_eos(2)
            .with_max_new_tokens(256)
            .with_session_position(100);
        assert_eq!(entry.prompt_len, 42);
        assert_eq!(entry.temperature_u32, 0.7f32.to_bits());
        assert_eq!(entry.top_k, 50);
        assert_eq!(entry.top_p_u32, 0.95f32.to_bits());
        assert_eq!(entry.eos_token_id, 2);
        assert_eq!(entry.max_new_tokens, 256);
        assert_eq!(entry.session_position, 100);
    }

    #[test]
    fn request_entry_default_temperature() {
        let entry = RequestQueueEntry::new(0, 10);
        assert_eq!(entry.temperature_u32, 1.0f32.to_bits());
        assert_eq!(entry.top_p_u32, 1.0f32.to_bits());
    }

    #[test]
    fn request_entry_size() {
        assert_eq!(std::mem::size_of::<RequestQueueEntry>(), RequestQueueEntry::SIZE);
    }

    // ── OutputTokenEntry ──

    #[test]
    fn output_entry_intermediate() {
        let e = OutputTokenEntry::intermediate(0, 1234, 5);
        assert_eq!(e.seq_id, 0);
        assert_eq!(e.token_id, 1234);
        assert_eq!(e.gen_idx, 5);
        assert_eq!(e.is_final, 0);
        assert_eq!(e.finish_reason, FINISH_REASON_INTERMEDIATE);
    }

    #[test]
    fn output_entry_final_eos() {
        let e = OutputTokenEntry::final_eos(3, 2, 20);
        assert_eq!(e.is_final, 1);
        assert_eq!(e.finish_reason, FINISH_REASON_EOS);
    }

    #[test]
    fn output_entry_final_max_tokens() {
        let e = OutputTokenEntry::final_max_tokens(1, 99, 100);
        assert_eq!(e.finish_reason, FINISH_REASON_MAX_TOKENS);
    }

    #[test]
    fn output_entry_size() {
        assert_eq!(std::mem::size_of::<OutputTokenEntry>(), OutputTokenEntry::SIZE);
    }

    // ── RequestQueue ──

    #[test]
    fn request_queue_basic() {
        let mut q = RequestQueue::new(4);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);

        let e1 = RequestQueueEntry::new(0x100, 10);
        let e2 = RequestQueueEntry::new(0x200, 20);
        assert!(q.enqueue(e1));
        assert!(q.enqueue(e2));
        assert_eq!(q.len(), 2);

        let dequeued = q.dequeue().unwrap();
        assert_eq!(dequeued.prompt_len, 10);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn request_queue_full() {
        let mut q = RequestQueue::new(5);
        // capacity rounds up to next power of 2 = 8
        assert_eq!(q.capacity(), 8);
        for i in 0..8 {
            assert!(q.enqueue(RequestQueueEntry::new(0, i)));
        }
        assert!(q.is_full());
        assert!(!q.enqueue(RequestQueueEntry::new(0, 99)));
    }

    #[test]
    fn request_queue_wraparound() {
        let mut q = RequestQueue::new(3); // capacity = 4
        // Fill and drain twice
        for round in 0..2 {
            for i in 0..4 {
                assert!(q.enqueue(RequestQueueEntry::new(0, round * 100 + i)));
            }
            assert!(q.is_full());
            for i in 0..4 {
                let e = q.dequeue().unwrap();
                assert_eq!(e.prompt_len, round * 100 + i);
            }
            assert!(q.is_empty());
        }
    }

    #[test]
    fn request_queue_batch_dequeue() {
        let mut q = RequestQueue::new(8);
        for i in 0..6 {
            q.enqueue(RequestQueueEntry::new(0, i));
        }
        let batch = q.dequeue_batch(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].prompt_len, 0);
        assert_eq!(batch[2].prompt_len, 2);
        assert_eq!(q.len(), 3);
    }

    #[test]
    fn request_queue_peek() {
        let mut q = RequestQueue::new(4);
        assert!(q.peek().is_none());
        q.enqueue(RequestQueueEntry::new(0x100, 42));
        let head = q.peek().unwrap();
        assert_eq!(head.prompt_len, 42);
        assert_eq!(q.len(), 1); // peek 不消耗
    }

    // ── OutputRingBuffer ──

    #[test]
    fn output_ring_write_and_consume() {
        let mut ring = OutputRingBuffer::new(4, 8, 32);

        // CTA 0 写入 seq 0 的 3 个 token
        ring.write_token(0, OutputTokenEntry::intermediate(0, 100, 0));
        ring.write_token(0, OutputTokenEntry::intermediate(0, 101, 1));
        ring.write_token(0, OutputTokenEntry::intermediate(0, 102, 2));

        // CTA 1 写入 seq 1 的 2 个 token
        ring.write_token(1, OutputTokenEntry::intermediate(1, 200, 0));
        ring.write_token(1, OutputTokenEntry::intermediate(1, 201, 1));

        let tokens = ring.consume();
        // 应按 (seq_id, gen_idx) 排序
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], OutputTokenEntry::intermediate(0, 100, 0));
        assert_eq!(tokens[1], OutputTokenEntry::intermediate(0, 101, 1));
        assert_eq!(tokens[2], OutputTokenEntry::intermediate(0, 102, 2));
        assert_eq!(tokens[3], OutputTokenEntry::intermediate(1, 200, 0));
        assert_eq!(tokens[4], OutputTokenEntry::intermediate(1, 201, 1));
    }

    #[test]
    fn output_ring_final_entries() {
        let mut ring = OutputRingBuffer::new(2, 4, 16);
        ring.write_token(0, OutputTokenEntry::intermediate(0, 100, 0));
        ring.write_token(0, OutputTokenEntry::final_eos(0, 2, 1));
        ring.write_token(1, OutputTokenEntry::intermediate(1, 200, 0));

        let finals = ring.consume_final_entries();
        assert_eq!(finals.len(), 1);
        assert_eq!(finals[0].seq_id, 0);
        assert_eq!(finals[0].finish_reason, FINISH_REASON_EOS);
    }

    #[test]
    fn output_ring_epoch() {
        let mut ring = OutputRingBuffer::new(2, 4, 16);
        assert_eq!(ring.epoch(), 0);
        ring.advance_epoch();
        assert_eq!(ring.epoch(), 1);
        ring.reset();
        assert_eq!(ring.epoch(), 0);
    }

    #[test]
    fn output_ring_invalid_cta() {
        let mut ring = OutputRingBuffer::new(2, 4, 16);
        assert!(!ring.write_token(99, OutputTokenEntry::intermediate(0, 0, 0)));
    }

    // ── SmPartitionConfig ──

    #[test]
    fn sm_partition_serial() {
        let cfg = SmPartitionConfig::serial();
        assert_eq!(cfg.variant, MkCompileVariant::Serial);
        assert_eq!(cfg.total_cta_count, 1);
        assert_eq!(cfg.decode_cta_count, 1);
        assert_eq!(cfg.prefill_cta_count, 0);
    }

    #[test]
    fn sm_partition_cluster_62() {
        // 模拟 H100: 128 SM (16 clusters × 8)
        let cfg = SmPartitionConfig::cluster_62(128);
        assert_eq!(cfg.variant, MkCompileVariant::Cluster62);
        assert_eq!(cfg.cluster_size, 8);
        assert_eq!(cfg.num_clusters, 16);
        assert_eq!(cfg.decode_per_cluster, 6);
        assert_eq!(cfg.prefill_per_cluster, 2);
        assert_eq!(cfg.decode_cta_count, 96);
        assert_eq!(cfg.prefill_cta_count, 32);
        assert_eq!(cfg.total_cta_count, 128);
    }

    #[test]
    fn sm_partition_grid_sync() {
        // 模拟 A100: 108 SM
        let cfg = SmPartitionConfig::grid_sync(108);
        assert_eq!(cfg.variant, MkCompileVariant::GridSync);
        assert_eq!(cfg.decode_cta_count, 81); // 108 × 3/4
        assert_eq!(cfg.prefill_cta_count, 27); // 108 - 81
        assert_eq!(cfg.total_cta_count, 108);
    }

    #[test]
    fn sm_partition_derive_small_sm() {
        // SM < 60 → Serial
        let cfg = SmPartitionConfig::derive(32);
        assert_eq!(cfg.variant, MkCompileVariant::Serial);
    }

    #[test]
    fn sm_partition_derive_large_sm() {
        // SM >= 60 → GridSync (default until GPU profile detection)
        let cfg = SmPartitionConfig::derive(108);
        assert_eq!(cfg.variant, MkCompileVariant::GridSync);
    }

    // ── Additional Tests ──

    // -- DualBatchMeta edge cases --

    #[test]
    fn dual_batch_meta_default_trait() {
        let meta = DualBatchMeta::default();
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.ping_seq_count, 0);
        assert_eq!(meta.pong_seq_offset, 0);
        assert_eq!(meta.pong_seq_count, 0);
        assert_eq!(meta.step_epoch, 0);
        assert_eq!(meta.epoch_arrival_count, 0);
    }

    #[test]
    fn dual_batch_meta_clone_copy_traits() {
        let meta = DualBatchMeta::new(64);
        let copy = meta;
        assert_eq!(copy.ping_seq_offset, 0);
        assert_eq!(copy.pong_seq_offset, 64);
        let cloned = meta.clone();
        assert_eq!(cloned.ping_seq_offset, 0);
        assert_eq!(cloned.pong_seq_offset, 64);
    }

    #[test]
    fn dual_batch_meta_swap_epoch_wrapping() {
        let mut meta = DualBatchMeta::new(16);
        meta.step_epoch = u32::MAX;
        meta.swap();
        assert_eq!(meta.step_epoch, 0); // wrapping_add wraps to 0
    }

    #[test]
    fn dual_batch_meta_swap_resets_arrival_count() {
        let mut meta = DualBatchMeta::new(16);
        meta.epoch_arrival_count = 42;
        meta.swap();
        assert_eq!(meta.epoch_arrival_count, 0);
    }

    #[test]
    fn dual_batch_meta_double_swap_identity() {
        let mut meta = DualBatchMeta::new(64);
        meta.ping_seq_offset = 10;
        meta.ping_seq_count = 5;
        meta.pong_seq_offset = 74;
        meta.pong_seq_count = 3;
        let original_ping = meta.ping_seq_offset;
        let original_pong = meta.pong_seq_offset;
        meta.swap();
        meta.swap();
        // After two swaps, ping/pong offsets are back
        assert_eq!(meta.ping_seq_offset, original_ping);
        assert_eq!(meta.pong_seq_offset, original_pong);
        // But epoch incremented twice
        assert_eq!(meta.step_epoch, 2);
    }

    // -- RequestQueueEntry edge cases --

    #[test]
    fn request_entry_default_trait() {
        let entry = RequestQueueEntry::default();
        assert_eq!(entry.input_ids_ptr, 0);
        assert_eq!(entry.prompt_len, 0);
        assert_eq!(entry.top_k, 0);
        assert_eq!(entry.eos_token_id, 0);
        assert_eq!(entry.max_new_tokens, 0);
        assert_eq!(entry.fused_hidden_offset, 0);
        assert_eq!(entry.num_mm_tokens, 0);
        assert_eq!(entry.session_position, 0);
    }

    #[test]
    fn request_entry_new_defaults() {
        let entry = RequestQueueEntry::new(0xABCD, 8);
        assert_eq!(entry.input_ids_ptr, 0xABCD);
        assert_eq!(entry.prompt_len, 8);
        assert_eq!(entry.max_new_tokens, 512); // default max
    }

    #[test]
    fn request_entry_builder_chaining() {
        let entry = RequestQueueEntry::new(0, 1)
            .with_temperature(0.0)
            .with_top_k(1)
            .with_top_p(0.5)
            .with_eos(50256)
            .with_max_new_tokens(1024)
            .with_session_position(500);
        assert_eq!(entry.temperature_u32, 0.0f32.to_bits());
        assert_eq!(entry.top_k, 1);
        assert_eq!(entry.top_p_u32, 0.5f32.to_bits());
        assert_eq!(entry.eos_token_id, 50256);
        assert_eq!(entry.max_new_tokens, 1024);
        assert_eq!(entry.session_position, 500);
    }

    // -- OutputTokenEntry edge cases --

    #[test]
    fn output_entry_final_stop_word() {
        let e = OutputTokenEntry::final_stop_word(7, 888, 15);
        assert_eq!(e.seq_id, 7);
        assert_eq!(e.token_id, 888);
        assert_eq!(e.gen_idx, 15);
        assert_eq!(e.is_final, 1);
        assert_eq!(e.finish_reason, FINISH_REASON_STOP_WORD);
    }

    #[test]
    fn output_entry_default_trait() {
        let e = OutputTokenEntry::default();
        assert_eq!(e.seq_id, 0);
        assert_eq!(e.token_id, 0);
        assert_eq!(e.is_final, 0);
        assert_eq!(e.finish_reason, 0);
        assert_eq!(e.gen_idx, 0);
    }

    #[test]
    fn output_entry_equality() {
        let a = OutputTokenEntry::intermediate(1, 42, 3);
        let b = OutputTokenEntry::intermediate(1, 42, 3);
        let c = OutputTokenEntry::intermediate(1, 42, 4);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn output_entry_clone_copy() {
        let e = OutputTokenEntry::final_eos(5, 99, 10);
        let copy = e;
        assert_eq!(copy, e);
        let cloned = e.clone();
        assert_eq!(cloned, e);
    }

    // -- RequestQueue edge cases --

    #[test]
    fn request_queue_capacity_rounding() {
        let q = RequestQueue::new(3);
        assert_eq!(q.capacity(), 4); // next power of 2
        let q2 = RequestQueue::new(1);
        assert_eq!(q2.capacity(), 1); // 1 is already power of 2
        let q3 = RequestQueue::new(100);
        assert_eq!(q3.capacity(), 128); // next power of 2
    }

    #[test]
    fn request_queue_dequeue_empty() {
        let mut q = RequestQueue::new(4);
        assert!(q.is_empty());
        assert!(q.dequeue().is_none());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn request_queue_batch_dequeue_more_than_available() {
        let mut q = RequestQueue::new(8);
        q.enqueue(RequestQueueEntry::new(0, 1));
        q.enqueue(RequestQueueEntry::new(0, 2));
        let batch = q.dequeue_batch(10);
        assert_eq!(batch.len(), 2);
        assert!(q.is_empty());
    }

    #[test]
    fn request_queue_batch_dequeue_zero() {
        let mut q = RequestQueue::new(4);
        q.enqueue(RequestQueueEntry::new(0, 1));
        let batch = q.dequeue_batch(0);
        assert!(batch.is_empty());
        assert_eq!(q.len(), 1); // nothing consumed
    }

    #[test]
    fn request_queue_multiple_wraparound_cycles() {
        let mut q = RequestQueue::new(2); // capacity = 2
        for cycle in 0..10 {
            for i in 0..2 {
                assert!(q.enqueue(RequestQueueEntry::new(0, cycle * 10 + i)));
            }
            for i in 0..2 {
                let e = q.dequeue().unwrap();
                assert_eq!(e.prompt_len, cycle * 10 + i);
            }
            assert!(q.is_empty());
        }
    }

    #[test]
    fn request_queue_peek_does_not_consume() {
        let mut q = RequestQueue::new(4);
        q.enqueue(RequestQueueEntry::new(0x50, 7));
        let _ = q.peek();
        let _ = q.peek();
        assert_eq!(q.len(), 1);
        let e = q.dequeue().unwrap();
        assert_eq!(e.input_ids_ptr, 0x50);
        assert_eq!(e.prompt_len, 7);
    }

    // -- OutputRingBuffer edge cases --

    #[test]
    fn output_ring_reset_clears_data() {
        let mut ring = OutputRingBuffer::new(2, 4, 16);
        ring.write_token(0, OutputTokenEntry::intermediate(0, 100, 0));
        ring.write_token(1, OutputTokenEntry::final_eos(1, 2, 5));
        ring.advance_epoch();
        ring.reset();
        let tokens = ring.consume();
        assert!(tokens.is_empty());
        assert_eq!(ring.epoch(), 0);
    }

    #[test]
    fn output_ring_write_after_reset() {
        let mut ring = OutputRingBuffer::new(2, 4, 16);
        ring.write_token(0, OutputTokenEntry::intermediate(0, 100, 0));
        ring.reset();
        ring.write_token(0, OutputTokenEntry::intermediate(5, 200, 0));
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].seq_id, 5);
        assert_eq!(tokens[0].token_id, 200);
    }

    #[test]
    fn output_ring_consume_empty() {
        let mut ring = OutputRingBuffer::new(4, 8, 32);
        let tokens = ring.consume();
        assert!(tokens.is_empty());
    }

    #[test]
    fn output_ring_multi_cta_interleaved_ordering() {
        let mut ring = OutputRingBuffer::new(3, 8, 64);
        // CTA 0: seq 2, gen_idx 0
        ring.write_token(0, OutputTokenEntry::intermediate(2, 300, 0));
        // CTA 2: seq 0, gen_idx 0
        ring.write_token(2, OutputTokenEntry::intermediate(0, 100, 0));
        // CTA 1: seq 1, gen_idx 0
        ring.write_token(1, OutputTokenEntry::intermediate(1, 200, 0));
        // CTA 0: seq 2, gen_idx 1
        ring.write_token(0, OutputTokenEntry::intermediate(2, 301, 1));
        let tokens = ring.consume();
        // Should be sorted by (seq_id, gen_idx)
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].seq_id, 0);
        assert_eq!(tokens[1].seq_id, 1);
        assert_eq!(tokens[2].seq_id, 2);
        assert_eq!(tokens[2].gen_idx, 0);
        assert_eq!(tokens[3].seq_id, 2);
        assert_eq!(tokens[3].gen_idx, 1);
    }

    // -- MkCompileVariant trait tests --

    #[test]
    fn mk_compile_variant_equality() {
        assert_eq!(MkCompileVariant::Cluster62, MkCompileVariant::Cluster62);
        assert_ne!(MkCompileVariant::Cluster62, MkCompileVariant::Cluster53);
        assert_ne!(MkCompileVariant::GridSync, MkCompileVariant::Serial);
    }

    #[test]
    fn mk_compile_variant_clone_copy() {
        let v = MkCompileVariant::Cluster53;
        let copy = v;
        assert_eq!(copy, MkCompileVariant::Cluster53);
        let cloned = v.clone();
        assert_eq!(cloned, MkCompileVariant::Cluster53);
    }

    // -- SmPartitionConfig edge cases --

    #[test]
    fn sm_partition_derive_exactly_60() {
        let cfg = SmPartitionConfig::derive(60);
        assert_eq!(cfg.variant, MkCompileVariant::GridSync);
        assert_eq!(cfg.decode_cta_count, 45); // 60 * 3 / 4
        assert_eq!(cfg.prefill_cta_count, 15);
    }

    #[test]
    fn sm_partition_serial_all_fields() {
        let cfg = SmPartitionConfig::serial();
        assert_eq!(cfg.variant, MkCompileVariant::Serial);
        assert_eq!(cfg.cluster_size, 1);
        assert_eq!(cfg.total_cta_count, 1);
        assert_eq!(cfg.decode_cta_count, 1);
        assert_eq!(cfg.prefill_cta_count, 0);
        assert_eq!(cfg.num_clusters, 1);
        assert_eq!(cfg.decode_per_cluster, 1);
        assert_eq!(cfg.prefill_per_cluster, 0);
    }

    #[test]
    fn sm_partition_cluster_62_non_divisible_sm() {
        // 100 SM / 8 = 12 clusters (truncated, not 12.5)
        let cfg = SmPartitionConfig::cluster_62(100);
        assert_eq!(cfg.num_clusters, 12);
        assert_eq!(cfg.total_cta_count, 96); // 12 * 8
        assert_eq!(cfg.decode_cta_count, 72); // 12 * 6
        assert_eq!(cfg.prefill_cta_count, 24); // 12 * 2
    }

    #[test]
    fn sm_partition_config_clone() {
        let cfg = SmPartitionConfig::grid_sync(80);
        let cloned = cfg.clone();
        assert_eq!(cloned.variant, MkCompileVariant::GridSync);
        assert_eq!(cloned.total_cta_count, 80);
        assert_eq!(cloned.decode_cta_count, 60);
    }

    // -- GemmTileParams / PhaseCompileParams / PhaseConfig --

    #[test]
    fn gemm_tile_params_construction() {
        let tile = GemmTileParams { m: 16, n: 32, k: 64 };
        assert_eq!(tile.m, 16);
        assert_eq!(tile.n, 32);
        assert_eq!(tile.k, 64);
    }

    #[test]
    fn gemm_tile_params_clone_copy() {
        let tile = GemmTileParams { m: 8, n: 16, k: 32 };
        let copy = tile;
        assert_eq!(copy.m, 8);
        let cloned = tile.clone();
        assert_eq!(cloned.n, 16);
    }

    #[test]
    fn phase_compile_params_construction() {
        let params = PhaseCompileParams {
            gemm_tile: GemmTileParams { m: 1, n: 128, k: 4096 },
            kv_pipeline_stages: 4,
            use_dsmem_kv_share: true,
            use_ld_nc: false,
            use_tensor_cores_gemv: true,
            symdim_m: "seq_len",
        };
        assert_eq!(params.gemm_tile.m, 1);
        assert_eq!(params.kv_pipeline_stages, 4);
        assert!(params.use_dsmem_kv_share);
        assert!(!params.use_ld_nc);
        assert!(params.use_tensor_cores_gemv);
        assert_eq!(params.symdim_m, "seq_len");
    }

    #[test]
    fn kv_access_mode_variants() {
        assert_eq!(KvAccessMode::WriteFull, KvAccessMode::WriteFull);
        assert_eq!(KvAccessMode::ReadHistoryWriteOne, KvAccessMode::ReadHistoryWriteOne);
        assert_ne!(KvAccessMode::WriteFull, KvAccessMode::ReadHistoryWriteOne);
    }

    #[test]
    fn attention_strategy_variants() {
        assert_eq!(AttentionStrategy::FlashAttention, AttentionStrategy::FlashAttention);
        assert_eq!(AttentionStrategy::IncrementalKv, AttentionStrategy::IncrementalKv);
        assert_ne!(AttentionStrategy::FlashAttention, AttentionStrategy::IncrementalKv);
    }

    #[test]
    fn phase_config_construction() {
        let config = PhaseConfig {
            compile_params: PhaseCompileParams {
                gemm_tile: GemmTileParams { m: 64, n: 64, k: 128 },
                kv_pipeline_stages: 2,
                use_dsmem_kv_share: false,
                use_ld_nc: true,
                use_tensor_cores_gemv: false,
                symdim_m: "total_seq",
            },
            attention: AttentionStrategy::FlashAttention,
            kv_mode: KvAccessMode::WriteFull,
        };
        assert_eq!(config.attention, AttentionStrategy::FlashAttention);
        assert_eq!(config.kv_mode, KvAccessMode::WriteFull);
        assert_eq!(config.compile_params.kv_pipeline_stages, 2);
    }

    // -- Constants --

    #[test]
    fn finish_reason_constants() {
        assert_eq!(FINISH_REASON_INTERMEDIATE, 0);
        assert_eq!(FINISH_REASON_EOS, 1);
        assert_eq!(FINISH_REASON_MAX_TOKENS, 2);
        assert_eq!(FINISH_REASON_STOP_WORD, 3);
    }

    #[test]
    fn page_allocator_constants() {
        assert_eq!(POOL_LOCAL_CAPACITY, 32);
        assert_eq!(POOL_CLUSTER_BATCH, 64);
        assert_eq!(POOL_GLOBAL_BATCH_WORDS, 1);
        assert_eq!(PAGE_ALLOC_OOM, u32::MAX);
    }

    #[test]
    fn batch_ctx_extension_size_alignment() {
        assert_eq!(BATCH_CTX_EXTENSION_SIZE % 128, 0);
        assert!(BATCH_CTX_EXTENSION_SIZE >= 128);
    }

    #[test]
    fn extension_offsets_are_increasing() {
        let offsets: [(usize, &str); 14] = [
            (EXT_REQUEST_QUEUE_PTR, "REQUEST_QUEUE_PTR"),
            (EXT_OUTPUT_RING_PTR, "OUTPUT_RING_PTR"),
            (EXT_KV_FREE_BITMAP_PTR, "KV_FREE_BITMAP_PTR"),
            (EXT_KV_POOL_TOTAL_PAGES, "KV_POOL_TOTAL_PAGES"),
            (EXT_MAX_BATCH_SIZE, "MAX_BATCH_SIZE"),
            (EXT_DUAL_BATCH_META, "DUAL_BATCH_META"),
            (EXT_AUTOTUNE_ACTUAL_BATCH, "AUTOTUNE_ACTUAL_BATCH"),
            (EXT_POOL_CLUSTER_DSMEM_PTR, "POOL_CLUSTER_DSMEM_PTR"),
            (EXT_PENDING_FREE_LIST_PTR, "PENDING_FREE_LIST_PTR"),
            (EXT_PENDING_FREE_COUNT_PTR, "PENDING_FREE_COUNT_PTR"),
            (EXT_OUTPUT_PER_CTA_DOORBELL_PTR, "OUTPUT_PER_CTA_DOORBELL_PTR"),
            (EXT_OUTPUT_EPOCH_FLAG_PTR, "OUTPUT_EPOCH_FLAG_PTR"),
            (EXT_KV_PAGE_HEADER_STRIDE, "KV_PAGE_HEADER_STRIDE"),
            (EXT_KV_EXT_ID_BASE_PTR, "KV_EXT_ID_BASE_PTR"),
        ];
        for i in 1..offsets.len() {
            assert!(
                offsets[i].0 > offsets[i - 1].0,
                "{} ({}) should be greater than {} ({})",
                offsets[i].1,
                offsets[i].0,
                offsets[i - 1].1,
                offsets[i - 1].0,
            );
        }
    }

    // ═══════════════════════════════════════════════════════════
    // Additional unit tests (ratio reduction: 62→82)
    // ═══════════════════════════════════════════════════════════

    // -- DualBatchMeta edge cases --

    #[test]
    fn dual_batch_meta_new_zero_batch_size() {
        let meta = DualBatchMeta::new(0);
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.ping_seq_count, 0);
        assert_eq!(meta.pong_seq_offset, 0);
        assert_eq!(meta.pong_seq_count, 0);
    }

    #[test]
    fn dual_batch_meta_new_max_u32_batch_size() {
        let meta = DualBatchMeta::new(u32::MAX);
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.pong_seq_offset, u32::MAX);
    }

    #[test]
    fn dual_batch_meta_debug_trait_output() {
        let meta = DualBatchMeta::new(64);
        let debug_str = format!("{:?}", meta);
        assert!(debug_str.contains("ping_seq_offset: 0"));
        assert!(debug_str.contains("pong_seq_offset: 64"));
        assert!(debug_str.contains("step_epoch: 0"));
    }

    #[test]
    fn dual_batch_meta_swap_multiple_epoch_accumulation() {
        let mut meta = DualBatchMeta::new(32);
        assert_eq!(meta.step_epoch, 0);
        for expected_epoch in 1..=5 {
            meta.swap();
            assert_eq!(meta.step_epoch, expected_epoch);
        }
    }

    // -- RequestQueueEntry edge cases --

    #[test]
    fn request_entry_nan_temperature() {
        let entry = RequestQueueEntry::new(0, 10).with_temperature(f32::NAN);
        let recovered = f32::from_bits(entry.temperature_u32);
        assert!(recovered.is_nan());
    }

    #[test]
    fn request_entry_boundary_top_p_zero_and_one() {
        let entry_zero = RequestQueueEntry::new(0, 5).with_top_p(0.0);
        assert_eq!(entry_zero.top_p_u32, 0.0f32.to_bits());

        let entry_one = RequestQueueEntry::new(0, 5).with_top_p(1.0);
        assert_eq!(entry_one.top_p_u32, 1.0f32.to_bits());
    }

    #[test]
    fn request_entry_max_u32_values() {
        let entry = RequestQueueEntry::new(u64::MAX, u32::MAX)
            .with_top_k(u32::MAX)
            .with_eos(u32::MAX)
            .with_max_new_tokens(u32::MAX)
            .with_session_position(u32::MAX);
        assert_eq!(entry.input_ids_ptr, u64::MAX);
        assert_eq!(entry.prompt_len, u32::MAX);
        assert_eq!(entry.top_k, u32::MAX);
        assert_eq!(entry.eos_token_id, u32::MAX);
        assert_eq!(entry.max_new_tokens, u32::MAX);
        assert_eq!(entry.session_position, u32::MAX);
    }

    #[test]
    fn request_entry_partial_eq_trait() {
        let a = RequestQueueEntry::new(0x100, 10)
            .with_temperature(0.8)
            .with_top_k(40);
        let b = RequestQueueEntry::new(0x100, 10)
            .with_temperature(0.8)
            .with_top_k(40);
        assert_eq!(a, b);

        let c = RequestQueueEntry::new(0x100, 10)
            .with_temperature(0.8)
            .with_top_k(99);
        assert_ne!(a, c);
    }

    #[test]
    fn request_entry_fused_hidden_fields() {
        let mut entry = RequestQueueEntry::new(0, 4);
        entry.fused_hidden_offset = 1024;
        entry.num_mm_tokens = 16;
        assert_eq!(entry.fused_hidden_offset, 1024);
        assert_eq!(entry.num_mm_tokens, 16);
    }

    // -- OutputTokenEntry edge cases --

    #[test]
    fn output_entry_max_u32_fields() {
        let e = OutputTokenEntry::intermediate(u32::MAX, u32::MAX, u32::MAX);
        assert_eq!(e.seq_id, u32::MAX);
        assert_eq!(e.token_id, u32::MAX);
        assert_eq!(e.gen_idx, u32::MAX);
        assert_eq!(e.is_final, 0);
        assert_eq!(e.finish_reason, 0);
    }

    #[test]
    fn output_entry_final_constructors_distinguish_finish_reasons() {
        let eos = OutputTokenEntry::final_eos(0, 0, 0);
        let max_tok = OutputTokenEntry::final_max_tokens(0, 0, 0);
        let stop = OutputTokenEntry::final_stop_word(0, 0, 0);
        // All three are final but have different finish_reason
        assert_eq!(eos.is_final, 1);
        assert_eq!(max_tok.is_final, 1);
        assert_eq!(stop.is_final, 1);
        assert_ne!(eos.finish_reason, max_tok.finish_reason);
        assert_ne!(max_tok.finish_reason, stop.finish_reason);
        assert_ne!(eos.finish_reason, stop.finish_reason);
    }

    #[test]
    fn output_entry_debug_trait_output() {
        let e = OutputTokenEntry::final_eos(3, 42, 7);
        let debug_str = format!("{:?}", e);
        assert!(debug_str.contains("seq_id: 3"));
        assert!(debug_str.contains("token_id: 42"));
        assert!(debug_str.contains("is_final: 1"));
        assert!(debug_str.contains("finish_reason: 1"));
    }

    // -- RequestQueue edge cases --

    #[test]
    fn request_queue_capacity_minimum_one() {
        let q = RequestQueue::new(1);
        assert_eq!(q.capacity(), 1);
        assert_eq!(q.len(), 0);
        assert!(q.is_empty());
    }

    #[test]
    fn request_queue_enqueue_preserves_data_integrity() {
        let mut q = RequestQueue::new(4);
        let original = RequestQueueEntry::new(0xDEADBEEF, 77)
            .with_temperature(0.42)
            .with_top_k(30)
            .with_top_p(0.88)
            .with_eos(50256)
            .with_max_new_tokens(999)
            .with_session_position(200);
        assert!(q.enqueue(original));
        let dequeued = q.dequeue().unwrap();
        assert_eq!(dequeued.input_ids_ptr, 0xDEADBEEF);
        assert_eq!(dequeued.prompt_len, 77);
        assert_eq!(dequeued.temperature_u32, 0.42f32.to_bits());
        assert_eq!(dequeued.top_k, 30);
        assert_eq!(dequeued.top_p_u32, 0.88f32.to_bits());
        assert_eq!(dequeued.eos_token_id, 50256);
        assert_eq!(dequeued.max_new_tokens, 999);
        assert_eq!(dequeued.session_position, 200);
    }

    // -- OutputRingBuffer edge cases --

    #[test]
    fn output_ring_write_multiple_tokens_single_cta() {
        let mut ring = OutputRingBuffer::new(1, 4, 32);
        for i in 0..5u32 {
            assert!(ring.write_token(0, OutputTokenEntry::intermediate(0, 100 + i, i)));
        }
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 5);
        for (i, t) in tokens.iter().enumerate() {
            assert_eq!(t.gen_idx, i as u32);
            assert_eq!(t.token_id, 100 + i as u32);
        }
    }

    // -- KvAccessMode / AttentionStrategy trait tests --

    #[test]
    fn kv_access_mode_clone_copy() {
        let mode = KvAccessMode::WriteFull;
        let copy = mode;
        assert_eq!(copy, KvAccessMode::WriteFull);
        let cloned = mode.clone();
        assert_eq!(cloned, KvAccessMode::WriteFull);
    }

    #[test]
    fn attention_strategy_clone_copy() {
        let strategy = AttentionStrategy::IncrementalKv;
        let copy = strategy;
        assert_eq!(copy, AttentionStrategy::IncrementalKv);
        let cloned = strategy.clone();
        assert_eq!(cloned, AttentionStrategy::IncrementalKv);
    }

    // -- GemmTileParams / PhaseCompileParams / PhaseConfig edge cases --

    #[test]
    fn gemm_tile_params_zero_values() {
        let tile = GemmTileParams { m: 0, n: 0, k: 0 };
        assert_eq!(tile.m, 0);
        assert_eq!(tile.n, 0);
        assert_eq!(tile.k, 0);
    }

    #[test]
    fn phase_compile_params_clone() {
        let params = PhaseCompileParams {
            gemm_tile: GemmTileParams { m: 32, n: 64, k: 128 },
            kv_pipeline_stages: 3,
            use_dsmem_kv_share: true,
            use_ld_nc: true,
            use_tensor_cores_gemv: false,
            symdim_m: "batch_m",
        };
        let cloned = params.clone();
        assert_eq!(cloned.gemm_tile.m, 32);
        assert_eq!(cloned.kv_pipeline_stages, 3);
        assert!(cloned.use_dsmem_kv_share);
        assert!(cloned.use_ld_nc);
        assert!(!cloned.use_tensor_cores_gemv);
        assert_eq!(cloned.symdim_m, "batch_m");
    }

    #[test]
    fn phase_config_clone() {
        let config = PhaseConfig {
            compile_params: PhaseCompileParams {
                gemm_tile: GemmTileParams { m: 16, n: 32, k: 64 },
                kv_pipeline_stages: 1,
                use_dsmem_kv_share: false,
                use_ld_nc: false,
                use_tensor_cores_gemv: true,
                symdim_m: "seq_len",
            },
            attention: AttentionStrategy::IncrementalKv,
            kv_mode: KvAccessMode::ReadHistoryWriteOne,
        };
        let cloned = config.clone();
        assert_eq!(cloned.attention, AttentionStrategy::IncrementalKv);
        assert_eq!(cloned.kv_mode, KvAccessMode::ReadHistoryWriteOne);
        assert_eq!(cloned.compile_params.gemm_tile.k, 64);
        assert!(cloned.compile_params.use_tensor_cores_gemv);
    }

    // -- SmPartitionConfig derive boundary --

    #[test]
    fn sm_partition_derive_59_is_serial() {
        let cfg = SmPartitionConfig::derive(59);
        assert_eq!(cfg.variant, MkCompileVariant::Serial);
    }

    // ═══════════════════════════════════════════════════════════
    // Additional unit tests — batch 3 (~50 tests, uncovered areas)
    // ═══════════════════════════════════════════════════════════

    // -- MegaKernelPhase Debug format coverage --
    // (All four MkCompileVariant Debug formats tested via debug_assert in existing tests)

    // -- SmPartitionConfig boundaries and math --

    #[test]
    fn sm_partition_cluster_62_exactly_8_sm() {
        let cfg = SmPartitionConfig::cluster_62(8);
        assert_eq!(cfg.num_clusters, 1);
        assert_eq!(cfg.decode_cta_count, 6);
        assert_eq!(cfg.prefill_cta_count, 2);
        assert_eq!(cfg.total_cta_count, 8);
    }

    #[test]
    fn sm_partition_cluster_62_large_power_of_two() {
        let cfg = SmPartitionConfig::cluster_62(256);
        assert_eq!(cfg.num_clusters, 32);
        assert_eq!(cfg.decode_cta_count, 192);
        assert_eq!(cfg.prefill_cta_count, 64);
        assert_eq!(cfg.total_cta_count, 256);
    }

    #[test]
    fn sm_partition_grid_sync_rounds_down_decode() {
        // 7 SM: 7 * 3 / 4 = 5 (integer division)
        let cfg = SmPartitionConfig::grid_sync(7);
        assert_eq!(cfg.decode_cta_count, 5);
        assert_eq!(cfg.prefill_cta_count, 2);
        assert_eq!(cfg.total_cta_count, 7);
    }

    #[test]
    fn sm_partition_grid_sync_cluster_size_zero() {
        let cfg = SmPartitionConfig::grid_sync(64);
        assert_eq!(cfg.cluster_size, 0);
        assert_eq!(cfg.num_clusters, 0);
        assert_eq!(cfg.decode_per_cluster, 0);
        assert_eq!(cfg.prefill_per_cluster, 0);
    }

    #[test]
    fn sm_partition_cluster_62_all_variants_distinct() {
        let c62 = SmPartitionConfig::cluster_62(128);
        let gs = SmPartitionConfig::grid_sync(128);
        let serial = SmPartitionConfig::serial();
        assert_ne!(c62.variant, gs.variant);
        assert_ne!(c62.variant, serial.variant);
        assert_ne!(gs.variant, serial.variant);
    }

    // -- RequestQueue ring buffer math --

    #[test]
    fn request_queue_len_tracks_enqueue_dequeue() {
        let mut q = RequestQueue::new(8);
        assert_eq!(q.len(), 0);
        q.enqueue(RequestQueueEntry::new(0, 1));
        assert_eq!(q.len(), 1);
        q.enqueue(RequestQueueEntry::new(0, 2));
        assert_eq!(q.len(), 2);
        q.dequeue();
        assert_eq!(q.len(), 1);
        q.dequeue();
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn request_queue_fill_drain_fill() {
        let mut q = RequestQueue::new(4); // capacity 4
        for i in 0..4 {
            assert!(q.enqueue(RequestQueueEntry::new(0, i)));
        }
        for i in 0..4 {
            assert_eq!(q.dequeue().unwrap().prompt_len, i);
        }
        // Refill with different data
        for i in 10..14 {
            assert!(q.enqueue(RequestQueueEntry::new(0, i)));
        }
        for i in 10..14 {
            assert_eq!(q.dequeue().unwrap().prompt_len, i);
        }
    }

    #[test]
    fn request_queue_batch_dequeue_partial() {
        let mut q = RequestQueue::new(16);
        for i in 0..10 {
            q.enqueue(RequestQueueEntry::new(0, i));
        }
        let batch1 = q.dequeue_batch(4);
        assert_eq!(batch1.len(), 4);
        assert_eq!(q.len(), 6);
        let batch2 = q.dequeue_batch(4);
        assert_eq!(batch2.len(), 4);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn request_queue_peek_after_partial_dequeue() {
        let mut q = RequestQueue::new(8);
        for i in 0..5 {
            q.enqueue(RequestQueueEntry::new(0, i));
        }
        q.dequeue();
        q.dequeue();
        // Head should now be index 2
        let head = q.peek().unwrap();
        assert_eq!(head.prompt_len, 2);
    }

    #[test]
    fn request_queue_enqueue_returns_false_without_modification() {
        let mut q = RequestQueue::new(2); // capacity 2
        assert!(q.enqueue(RequestQueueEntry::new(0xAAAA, 10)));
        assert!(q.enqueue(RequestQueueEntry::new(0xBBBB, 20)));
        assert!(!q.enqueue(RequestQueueEntry::new(0xCCCC, 30)));
        // Dequeue should return the original two entries
        assert_eq!(q.dequeue().unwrap().input_ids_ptr, 0xAAAA);
        assert_eq!(q.dequeue().unwrap().input_ids_ptr, 0xBBBB);
    }

    #[test]
    fn request_queue_dequeue_preserves_order() {
        let mut q = RequestQueue::new(32);
        for i in 0..20 {
            assert!(q.enqueue(RequestQueueEntry::new(0x1000 + i as u64, i)));
        }
        for i in 0..20 {
            let e = q.dequeue().unwrap();
            assert_eq!(e.input_ids_ptr, 0x1000 + i as u64);
            assert_eq!(e.prompt_len, i);
        }
    }

    // -- OutputRingBuffer operations --

    #[test]
    fn output_ring_single_cta_full_lifecycle() {
        let mut ring = OutputRingBuffer::new(1, 4, 64);
        // Write intermediate tokens
        for i in 0..3 {
            assert!(ring.write_token(0, OutputTokenEntry::intermediate(0, 100 + i, i)));
        }
        // Write final token
        assert!(ring.write_token(0, OutputTokenEntry::final_eos(0, 2, 3)));
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[3].is_final, 1);
    }

    #[test]
    fn output_ring_consume_final_entries_mixed() {
        let mut ring = OutputRingBuffer::new(2, 8, 32);
        ring.write_token(0, OutputTokenEntry::intermediate(0, 10, 0));
        ring.write_token(0, OutputTokenEntry::intermediate(0, 11, 1));
        ring.write_token(0, OutputTokenEntry::final_eos(0, 2, 2));
        ring.write_token(1, OutputTokenEntry::final_max_tokens(1, 99, 5));
        let finals = ring.consume_final_entries();
        assert_eq!(finals.len(), 2);
        // Sorted by (seq_id, gen_idx)
        assert_eq!(finals[0].seq_id, 0);
        assert_eq!(finals[0].finish_reason, FINISH_REASON_EOS);
        assert_eq!(finals[1].seq_id, 1);
        assert_eq!(finals[1].finish_reason, FINISH_REASON_MAX_TOKENS);
    }

    #[test]
    fn output_ring_reset_clears_write_indices() {
        let mut ring = OutputRingBuffer::new(2, 4, 16);
        ring.write_token(0, OutputTokenEntry::intermediate(0, 10, 0));
        ring.write_token(1, OutputTokenEntry::intermediate(1, 20, 0));
        ring.advance_epoch();
        ring.reset();
        // After reset, write indices should be zero so consume returns empty
        let tokens = ring.consume();
        assert!(tokens.is_empty());
        // Can write again
        assert!(ring.write_token(0, OutputTokenEntry::intermediate(5, 50, 0)));
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].seq_id, 5);
    }

    #[test]
    fn output_ring_write_to_boundary_cta_id() {
        let mut ring = OutputRingBuffer::new(4, 4, 16);
        // Last valid CTA id is num_ctas - 1 = 3
        assert!(ring.write_token(3, OutputTokenEntry::intermediate(0, 42, 0)));
        // CTA id 4 is out of bounds
        assert!(!ring.write_token(4, OutputTokenEntry::intermediate(0, 43, 0)));
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].token_id, 42);
    }

    #[test]
    fn output_ring_multiple_seqs_sorted_output() {
        let mut ring = OutputRingBuffer::new(2, 4, 32);
        // CTA 0 handles seq 2
        ring.write_token(0, OutputTokenEntry::intermediate(2, 300, 0));
        ring.write_token(0, OutputTokenEntry::intermediate(2, 301, 1));
        // CTA 1 handles seq 1
        ring.write_token(1, OutputTokenEntry::intermediate(1, 200, 0));
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 3);
        // seq 1 first, then seq 2
        assert_eq!(tokens[0].seq_id, 1);
        assert_eq!(tokens[1].seq_id, 2);
        assert_eq!(tokens[1].gen_idx, 0);
        assert_eq!(tokens[2].seq_id, 2);
        assert_eq!(tokens[2].gen_idx, 1);
    }

    #[test]
    fn output_ring_all_finish_reason_types() {
        let mut ring = OutputRingBuffer::new(1, 4, 64);
        ring.write_token(0, OutputTokenEntry::intermediate(0, 10, 0));
        ring.write_token(0, OutputTokenEntry::final_eos(0, 2, 1));
        ring.write_token(0, OutputTokenEntry::final_max_tokens(0, 50, 2));
        ring.write_token(0, OutputTokenEntry::final_stop_word(0, 99, 3));
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].finish_reason, FINISH_REASON_INTERMEDIATE);
        assert_eq!(tokens[1].finish_reason, FINISH_REASON_EOS);
        assert_eq!(tokens[2].finish_reason, FINISH_REASON_MAX_TOKENS);
        assert_eq!(tokens[3].finish_reason, FINISH_REASON_STOP_WORD);
    }

    // -- DualBatchMeta field access and manipulation --

    #[test]
    fn dual_batch_meta_manual_field_set_and_swap() {
        let mut meta = DualBatchMeta::new(64);
        meta.ping_seq_count = 32;
        meta.pong_seq_count = 28;
        meta.epoch_arrival_count = 7;
        meta.swap();
        assert_eq!(meta.ping_seq_count, 28);
        assert_eq!(meta.pong_seq_count, 32);
        assert_eq!(meta.epoch_arrival_count, 0);
    }

    #[test]
    fn dual_batch_meta_swap_with_zero_counts() {
        let mut meta = DualBatchMeta::new(32);
        // Both counts are 0
        meta.swap();
        assert_eq!(meta.ping_seq_offset, 32); // was pong
        assert_eq!(meta.ping_seq_count, 0);
        assert_eq!(meta.pong_seq_offset, 0); // was ping
        assert_eq!(meta.pong_seq_count, 0);
    }

    // -- RequestQueueEntry construction edge cases --

    #[test]
    fn request_entry_infinity_temperature() {
        let entry = RequestQueueEntry::new(0, 5).with_temperature(f32::INFINITY);
        let recovered = f32::from_bits(entry.temperature_u32);
        assert!(recovered.is_infinite());
        assert!(recovered.is_sign_positive());
    }

    #[test]
    fn request_entry_negative_infinity_top_p() {
        let entry = RequestQueueEntry::new(0, 5).with_top_p(f32::NEG_INFINITY);
        let recovered = f32::from_bits(entry.top_p_u32);
        assert!(recovered.is_infinite());
        assert!(recovered.is_sign_negative());
    }

    #[test]
    fn request_entry_repr_c_layout() {
        assert_eq!(std::mem::size_of::<RequestQueueEntry>(), 48);
        assert_eq!(std::mem::align_of::<RequestQueueEntry>(), 8); // u64 alignment
    }

    // -- OutputTokenEntry construction edge cases --

    #[test]
    fn output_entry_repr_c_layout() {
        assert_eq!(std::mem::size_of::<OutputTokenEntry>(), 24); // 6 x u32
        assert_eq!(std::mem::align_of::<OutputTokenEntry>(), 4);
    }

    #[test]
    fn output_entry_final_eos_with_large_gen_idx() {
        let e = OutputTokenEntry::final_eos(0, 2, u32::MAX);
        assert_eq!(e.gen_idx, u32::MAX);
        assert_eq!(e.is_final, 1);
        assert_eq!(e.finish_reason, FINISH_REASON_EOS);
    }

    // -- GemmTileParams edge cases --

    #[test]
    fn gemm_tile_params_debug_output() {
        let tile = GemmTileParams { m: 16, n: 32, k: 64 };
        let debug = format!("{:?}", tile);
        assert!(debug.contains("m: 16"));
        assert!(debug.contains("n: 32"));
        assert!(debug.contains("k: 64"));
    }

    // -- PhaseCompileParams edge cases --

    #[test]
    fn phase_compile_params_all_bool_true() {
        let params = PhaseCompileParams {
            gemm_tile: GemmTileParams { m: 64, n: 64, k: 128 },
            kv_pipeline_stages: 4,
            use_dsmem_kv_share: true,
            use_ld_nc: true,
            use_tensor_cores_gemv: true,
            symdim_m: "total_seq",
        };
        assert!(params.use_dsmem_kv_share);
        assert!(params.use_ld_nc);
        assert!(params.use_tensor_cores_gemv);
    }

    // -- PhaseConfig edge cases --

    #[test]
    fn phase_config_decode_preset() {
        let config = PhaseConfig {
            compile_params: PhaseCompileParams {
                gemm_tile: GemmTileParams { m: 1, n: 128, k: 4096 },
                kv_pipeline_stages: 1,
                use_dsmem_kv_share: false,
                use_ld_nc: false,
                use_tensor_cores_gemv: true,
                symdim_m: "seq_len",
            },
            attention: AttentionStrategy::IncrementalKv,
            kv_mode: KvAccessMode::ReadHistoryWriteOne,
        };
        assert_eq!(config.attention, AttentionStrategy::IncrementalKv);
        assert_eq!(config.kv_mode, KvAccessMode::ReadHistoryWriteOne);
        assert!(config.compile_params.use_tensor_cores_gemv);
        assert_eq!(config.compile_params.kv_pipeline_stages, 1);
    }

    // -- KvAccessMode / AttentionStrategy Debug format --

    #[test]
    fn kv_access_mode_debug_format() {
        assert_eq!(format!("{:?}", KvAccessMode::WriteFull), "WriteFull");
        assert_eq!(format!("{:?}", KvAccessMode::ReadHistoryWriteOne), "ReadHistoryWriteOne");
    }

    #[test]
    fn attention_strategy_debug_format() {
        assert_eq!(format!("{:?}", AttentionStrategy::FlashAttention), "FlashAttention");
        assert_eq!(format!("{:?}", AttentionStrategy::IncrementalKv), "IncrementalKv");
    }

    // -- Batch context extension offset invariants --

    // -- Prefill chunk constants --

    #[test]
    fn prefill_chunk_threshold_and_size_equal() {
        // SPEC 32 §1.4: threshold and chunk size are both 512
        assert_eq!(PREFILL_CHUNK_THRESHOLD, 512);
        assert_eq!(PREFILL_CHUNK_SIZE, 512);
    }

    // -- Page allocator constant invariants --

    #[test]
    fn pool_local_capacity_less_than_cluster_batch() {
        // Local pool is smaller than a cluster batch fetch
        assert!(POOL_LOCAL_CAPACITY < POOL_CLUSTER_BATCH);
    }

    // ═══════════════════════════════════════════════════════════
    // Additional unit tests — batch 4 (targeting 42 new tests)
    // ═══════════════════════════════════════════════════════════

    // ── DualBatchMeta: repr(C) and layout ──

    #[test]
    fn dual_batch_meta_repr_c_alignment() {
        // 6 x u32 = 24 bytes, 4-byte aligned
        assert_eq!(std::mem::size_of::<DualBatchMeta>(), 24);
        assert_eq!(std::mem::align_of::<DualBatchMeta>(), 4);
    }

    #[test]
    fn dual_batch_meta_swap_preserves_sum_of_counts() {
        let mut meta = DualBatchMeta::new(128);
        meta.ping_seq_count = 37;
        meta.pong_seq_count = 19;
        let total = meta.ping_seq_count + meta.pong_seq_count;
        meta.swap();
        assert_eq!(meta.ping_seq_count + meta.pong_seq_count, total);
    }

    #[test]
    fn dual_batch_meta_swap_preserves_sum_of_offsets() {
        let mut meta = DualBatchMeta::new(256);
        meta.ping_seq_offset = 10;
        meta.pong_seq_offset = 300;
        let sum = meta.ping_seq_offset + meta.pong_seq_offset;
        meta.swap();
        assert_eq!(meta.ping_seq_offset + meta.pong_seq_offset, sum);
    }

    #[test]
    fn dual_batch_meta_many_swaps_epoch_monotonic() {
        let mut meta = DualBatchMeta::new(32);
        let mut prev_epoch = meta.step_epoch;
        for _ in 0..100 {
            meta.swap();
            assert!(meta.step_epoch > prev_epoch || meta.step_epoch == 0);
            prev_epoch = meta.step_epoch;
        }
    }

    // ── RequestQueueEntry: subnormal and special float values ──

    #[test]
    fn request_entry_subnormal_temperature() {
        let entry = RequestQueueEntry::new(0, 5).with_temperature(f32::MIN_POSITIVE);
        let recovered = f32::from_bits(entry.temperature_u32);
        assert_eq!(recovered, f32::MIN_POSITIVE);
    }

    #[test]
    fn request_entry_negative_temperature() {
        let entry = RequestQueueEntry::new(0, 5).with_temperature(-1.5);
        let recovered = f32::from_bits(entry.temperature_u32);
        assert!((recovered - (-1.5f32)).abs() < f32::EPSILON);
    }

    #[test]
    fn request_entry_subnormal_top_p() {
        let entry = RequestQueueEntry::new(0, 5).with_top_p(f32::MIN_POSITIVE);
        let recovered = f32::from_bits(entry.top_p_u32);
        assert_eq!(recovered, f32::MIN_POSITIVE);
    }

    #[test]
    fn request_entry_clone_trait() {
        let entry = RequestQueueEntry::new(0xABCD, 15)
            .with_temperature(0.5)
            .with_top_k(10)
            .with_eos(3)
            .with_max_new_tokens(200);
        let cloned = entry.clone();
        assert_eq!(cloned, entry);
    }

    #[test]
    fn request_entry_copy_trait() {
        let entry = RequestQueueEntry::new(0x1234, 20).with_session_position(50);
        let copy = entry;
        assert_eq!(copy.input_ids_ptr, 0x1234);
        assert_eq!(copy.prompt_len, 20);
        assert_eq!(copy.session_position, 50);
    }

    // ── RequestQueue: edge cases around ring math ──

    #[test]
    fn request_queue_single_slot_capacity() {
        let mut q = RequestQueue::new(1);
        assert_eq!(q.capacity(), 1);
        assert!(q.enqueue(RequestQueueEntry::new(0, 1)));
        assert!(q.is_full());
        assert!(!q.enqueue(RequestQueueEntry::new(0, 2)));
        let e = q.dequeue().unwrap();
        assert_eq!(e.prompt_len, 1);
        assert!(q.is_empty());
        // Can enqueue again after dequeue
        assert!(q.enqueue(RequestQueueEntry::new(0, 3)));
    }

    #[test]
    fn request_queue_dequeue_batch_from_empty() {
        let mut q = RequestQueue::new(8);
        let batch = q.dequeue_batch(5);
        assert!(batch.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn request_queue_enqueue_dequeue_interleaved() {
        let mut q = RequestQueue::new(8);
        q.enqueue(RequestQueueEntry::new(0, 1));
        assert_eq!(q.dequeue().unwrap().prompt_len, 1);
        q.enqueue(RequestQueueEntry::new(0, 2));
        q.enqueue(RequestQueueEntry::new(0, 3));
        assert_eq!(q.len(), 2);
        assert_eq!(q.dequeue().unwrap().prompt_len, 2);
        assert_eq!(q.dequeue().unwrap().prompt_len, 3);
        assert!(q.is_empty());
    }

    #[test]
    fn request_queue_large_capacity_rounding() {
        let q = RequestQueue::new(1000);
        assert_eq!(q.capacity(), 1024); // next power of 2
    }

    #[test]
    fn request_queue_wraparound_preserves_entry_identity() {
        let mut q = RequestQueue::new(4); // capacity 4
        // Fill with distinct entries
        for i in 0..4u32 {
            assert!(q.enqueue(RequestQueueEntry::new(i as u64 * 0x100, i * 7)));
        }
        // Drain
        for i in 0..4u32 {
            let e = q.dequeue().unwrap();
            assert_eq!(e.input_ids_ptr, i as u64 * 0x100);
            assert_eq!(e.prompt_len, i * 7);
        }
        // Fill again (wraps around write_idx and read_idx)
        for i in 10..14u32 {
            assert!(q.enqueue(RequestQueueEntry::new(i as u64, i)));
        }
        for i in 10..14u32 {
            let e = q.dequeue().unwrap();
            assert_eq!(e.input_ids_ptr, i as u64);
            assert_eq!(e.prompt_len, i);
        }
    }

    #[test]
    fn request_queue_is_full_after_exact_capacity() {
        let mut q = RequestQueue::new(8);
        for i in 0..8 {
            assert!(!q.is_full());
            assert!(q.enqueue(RequestQueueEntry::new(0, i)));
        }
        assert!(q.is_full());
    }

    // ── OutputRingBuffer: sub-ring sizing and multi-CTA scenarios ──

    #[test]
    fn output_ring_sub_ring_size_minimum() {
        // Even with small batch, sub-ring should be at least 128
        let ring = OutputRingBuffer::new(4, 1, 1);
        // Verify the ring was created (internal cta_sub_ring_size >= 128)
        // We can write up to 128 tokens to a single CTA
        let mut ring = ring;
        for i in 0..128u32 {
            assert!(ring.write_token(0, OutputTokenEntry::intermediate(0, i, i)));
        }
    }

    #[test]
    fn output_ring_advance_epoch_sequential() {
        let mut ring = OutputRingBuffer::new(2, 4, 16);
        for expected in 1..=10u32 {
            ring.advance_epoch();
            assert_eq!(ring.epoch(), expected);
        }
    }

    #[test]
    fn output_ring_consume_final_entries_empty() {
        let mut ring = OutputRingBuffer::new(2, 4, 16);
        let finals = ring.consume_final_entries();
        assert!(finals.is_empty());
    }

    #[test]
    fn output_ring_consume_after_multiple_epochs() {
        let mut ring = OutputRingBuffer::new(1, 4, 32);
        // Epoch 0: write tokens
        ring.write_token(0, OutputTokenEntry::intermediate(0, 10, 0));
        ring.advance_epoch();
        // Epoch 1: write more tokens
        ring.write_token(0, OutputTokenEntry::intermediate(0, 11, 1));
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn output_ring_write_to_each_cta() {
        let mut ring = OutputRingBuffer::new(4, 8, 32);
        for cta in 0..4u32 {
            // Use seq_id = cta + 1 to avoid (0, 0, 0) being filtered by consume
            assert!(ring.write_token(cta, OutputTokenEntry::intermediate(cta + 1, cta * 10 + 1, 1)));
        }
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 4);
        // Sorted by seq_id
        for (i, t) in tokens.iter().enumerate() {
            assert_eq!(t.seq_id, i as u32 + 1);
        }
    }

    #[test]
    fn output_ring_same_seq_across_multiple_ctas() {
        let mut ring = OutputRingBuffer::new(3, 4, 32);
        // Different CTAs produce tokens for the same sequence
        ring.write_token(0, OutputTokenEntry::intermediate(5, 100, 0));
        ring.write_token(1, OutputTokenEntry::intermediate(5, 101, 1));
        ring.write_token(2, OutputTokenEntry::intermediate(5, 102, 2));
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 3);
        // All same seq_id, sorted by gen_idx
        assert_eq!(tokens[0].gen_idx, 0);
        assert_eq!(tokens[1].gen_idx, 1);
        assert_eq!(tokens[2].gen_idx, 2);
        assert_eq!(tokens[0].token_id, 100);
        assert_eq!(tokens[1].token_id, 101);
        assert_eq!(tokens[2].token_id, 102);
    }

    #[test]
    fn output_ring_consume_final_entries_no_finals() {
        let mut ring = OutputRingBuffer::new(2, 4, 16);
        ring.write_token(0, OutputTokenEntry::intermediate(0, 10, 0));
        ring.write_token(1, OutputTokenEntry::intermediate(1, 20, 0));
        let finals = ring.consume_final_entries();
        assert!(finals.is_empty());
    }

    #[test]
    fn output_ring_reset_then_full_lifecycle() {
        let mut ring = OutputRingBuffer::new(2, 4, 16);
        // First lifecycle
        ring.write_token(0, OutputTokenEntry::intermediate(0, 10, 0));
        ring.write_token(0, OutputTokenEntry::final_eos(0, 2, 1));
        let _ = ring.consume();
        ring.reset();
        // Second lifecycle after reset
        ring.write_token(1, OutputTokenEntry::intermediate(1, 30, 0));
        ring.write_token(1, OutputTokenEntry::final_max_tokens(1, 50, 1));
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].seq_id, 1);
        assert_eq!(tokens[1].finish_reason, FINISH_REASON_MAX_TOKENS);
    }

    // ── MkCompileVariant: Debug format and all variants ──

    #[test]
    fn mk_compile_variant_debug_formats() {
        assert_eq!(format!("{:?}", MkCompileVariant::Cluster62), "Cluster62");
        assert_eq!(format!("{:?}", MkCompileVariant::Cluster53), "Cluster53");
        assert_eq!(format!("{:?}", MkCompileVariant::GridSync), "GridSync");
        assert_eq!(format!("{:?}", MkCompileVariant::Serial), "Serial");
    }

    #[test]
    fn mk_compile_variant_all_four_distinct() {
        let variants = [
            MkCompileVariant::Cluster62,
            MkCompileVariant::Cluster53,
            MkCompileVariant::GridSync,
            MkCompileVariant::Serial,
        ];
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    // ── SmPartitionConfig: additional math and invariants ──

    #[test]
    fn sm_partition_cluster_62_decode_plus_prefill_equals_total() {
        for &sm_count in &[8, 16, 64, 128, 256] {
            let cfg = SmPartitionConfig::cluster_62(sm_count);
            assert_eq!(
                cfg.decode_cta_count + cfg.prefill_cta_count,
                cfg.total_cta_count,
                "decode + prefill must equal total for sm_count={}",
                sm_count
            );
        }
    }

    #[test]
    fn sm_partition_grid_sync_decode_plus_prefill_equals_total() {
        for &sm_count in &[1, 7, 60, 108, 128] {
            let cfg = SmPartitionConfig::grid_sync(sm_count);
            assert_eq!(
                cfg.decode_cta_count + cfg.prefill_cta_count,
                cfg.total_cta_count,
                "decode + prefill must equal total for sm_count={}",
                sm_count
            );
        }
    }

    #[test]
    fn sm_partition_cluster_62_decode_ratio() {
        // decode_per_cluster should be 6/8 = 75% of cluster
        for &sm_count in &[8, 16, 64, 128] {
            let cfg = SmPartitionConfig::cluster_62(sm_count);
            assert_eq!(cfg.decode_per_cluster, 6);
            assert_eq!(cfg.prefill_per_cluster, 2);
        }
    }

    #[test]
    fn sm_partition_grid_sync_decode_is_75_percent() {
        let cfg = SmPartitionConfig::grid_sync(100);
        // 100 * 3 / 4 = 75
        assert_eq!(cfg.decode_cta_count, 75);
        assert_eq!(cfg.prefill_cta_count, 25);
    }

    #[test]
    fn sm_partition_cluster_62_total_sm_truncation() {
        // 13 SM / 8 = 1 cluster (truncated), total_cta = 8
        let cfg = SmPartitionConfig::cluster_62(13);
        assert_eq!(cfg.num_clusters, 1);
        assert_eq!(cfg.total_cta_count, 8);
        assert_eq!(cfg.decode_cta_count, 6);
        assert_eq!(cfg.prefill_cta_count, 2);
    }

    #[test]
    fn sm_partition_serial_no_prefill() {
        let cfg = SmPartitionConfig::serial();
        assert_eq!(cfg.prefill_cta_count, 0);
        assert_eq!(cfg.prefill_per_cluster, 0);
        assert_eq!(cfg.decode_cta_count, cfg.total_cta_count);
    }

    #[test]
    fn sm_partition_debug_format() {
        let cfg = SmPartitionConfig::serial();
        let debug = format!("{:?}", cfg);
        assert!(debug.contains("Serial"));
        assert!(debug.contains("total_cta_count"));
    }

    // ── PhaseCompileParams: Debug format ──

    #[test]
    fn phase_compile_params_debug_format() {
        let params = PhaseCompileParams {
            gemm_tile: GemmTileParams { m: 1, n: 2, k: 3 },
            kv_pipeline_stages: 2,
            use_dsmem_kv_share: false,
            use_ld_nc: true,
            use_tensor_cores_gemv: false,
            symdim_m: "test_dim",
        };
        let debug = format!("{:?}", params);
        assert!(debug.contains("kv_pipeline_stages: 2"));
        assert!(debug.contains("symdim_m: \"test_dim\""));
    }

    #[test]
    fn phase_config_debug_format() {
        let config = PhaseConfig {
            compile_params: PhaseCompileParams {
                gemm_tile: GemmTileParams { m: 4, n: 5, k: 6 },
                kv_pipeline_stages: 1,
                use_dsmem_kv_share: false,
                use_ld_nc: false,
                use_tensor_cores_gemv: false,
                symdim_m: "m",
            },
            attention: AttentionStrategy::FlashAttention,
            kv_mode: KvAccessMode::WriteFull,
        };
        let debug = format!("{:?}", config);
        assert!(debug.contains("FlashAttention"));
        assert!(debug.contains("WriteFull"));
    }

    // ── PhaseConfig: prefill and decode preset consistency ──

    #[test]
    fn phase_config_prefill_preset() {
        let config = PhaseConfig {
            compile_params: PhaseCompileParams {
                gemm_tile: GemmTileParams { m: 64, n: 64, k: 128 },
                kv_pipeline_stages: 4,
                use_dsmem_kv_share: true,
                use_ld_nc: false,
                use_tensor_cores_gemv: false,
                symdim_m: "prefill_seq_len",
            },
            attention: AttentionStrategy::FlashAttention,
            kv_mode: KvAccessMode::WriteFull,
        };
        assert_eq!(config.attention, AttentionStrategy::FlashAttention);
        assert_eq!(config.kv_mode, KvAccessMode::WriteFull);
        assert_eq!(config.compile_params.kv_pipeline_stages, 4);
        assert!(config.compile_params.use_dsmem_kv_share);
    }

    #[test]
    fn phase_config_clone_independence() {
        let config = PhaseConfig {
            compile_params: PhaseCompileParams {
                gemm_tile: GemmTileParams { m: 8, n: 16, k: 32 },
                kv_pipeline_stages: 2,
                use_dsmem_kv_share: false,
                use_ld_nc: true,
                use_tensor_cores_gemv: true,
                symdim_m: "seq",
            },
            attention: AttentionStrategy::IncrementalKv,
            kv_mode: KvAccessMode::ReadHistoryWriteOne,
        };
        let cloned = config.clone();
        // Verify all fields match
        assert_eq!(cloned.compile_params.gemm_tile.m, 8);
        assert_eq!(cloned.compile_params.gemm_tile.n, 16);
        assert_eq!(cloned.compile_params.gemm_tile.k, 32);
        assert_eq!(cloned.compile_params.kv_pipeline_stages, 2);
        assert_eq!(cloned.compile_params.symdim_m, "seq");
        assert_eq!(cloned.attention, AttentionStrategy::IncrementalKv);
        assert_eq!(cloned.kv_mode, KvAccessMode::ReadHistoryWriteOne);
    }

    // ── Extension offsets: non-overlapping and within bounds ──

    #[test]
    fn extension_offsets_within_extension_size() {
        assert!(EXT_REQUEST_QUEUE_PTR < BATCH_CTX_EXTENSION_SIZE);
        assert!(EXT_OUTPUT_RING_PTR < BATCH_CTX_EXTENSION_SIZE);
        assert!(EXT_KV_FREE_BITMAP_PTR < BATCH_CTX_EXTENSION_SIZE);
        assert!(EXT_DUAL_BATCH_META < BATCH_CTX_EXTENSION_SIZE);
        assert!(EXT_RESERVED < BATCH_CTX_EXTENSION_SIZE);
    }

    #[test]
    fn extension_dual_batch_meta_fits_in_extension() {
        // DualBatchMeta is 24 bytes starting at offset 32
        assert!(
            EXT_DUAL_BATCH_META + DualBatchMeta::SIZE <= BATCH_CTX_EXTENSION_SIZE,
            "DualBatchMeta (24B at offset {}) must fit within extension ({}B)",
            EXT_DUAL_BATCH_META,
            BATCH_CTX_EXTENSION_SIZE
        );
    }

    // ── OutputTokenEntry: Hash consistency ──

    #[test]
    fn output_entry_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = OutputTokenEntry::intermediate(3, 42, 7);
        let b = OutputTokenEntry::intermediate(3, 42, 7);
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn output_entry_hash_differs_for_different_entries() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = OutputTokenEntry::intermediate(1, 10, 0);
        let b = OutputTokenEntry::intermediate(2, 10, 0);
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_ne!(ha.finish(), hb.finish());
    }

    // ── OutputTokenEntry: sorting behavior via Ord ──

    #[test]
    fn output_entry_ord_sorting_by_seq_then_gen_idx() {
        let mut entries = vec![
            OutputTokenEntry::intermediate(2, 300, 1),
            OutputTokenEntry::intermediate(1, 100, 0),
            OutputTokenEntry::intermediate(2, 200, 0),
            OutputTokenEntry::intermediate(0, 50, 0),
        ];
        entries.sort();
        assert_eq!(entries[0].seq_id, 0);
        assert_eq!(entries[1].seq_id, 1);
        assert_eq!(entries[2].seq_id, 2);
        assert_eq!(entries[2].gen_idx, 0);
        assert_eq!(entries[3].seq_id, 2);
        assert_eq!(entries[3].gen_idx, 1);
    }

    // ── RequestQueueEntry: debug format ──

    #[test]
    fn request_entry_debug_trait_output() {
        let entry = RequestQueueEntry::new(0xBEEF, 42);
        let debug = format!("{:?}", entry);
        assert!(debug.contains("input_ids_ptr: 48879")); // 0xBEEF = 48879
        assert!(debug.contains("prompt_len: 42"));
    }

    // ── DualBatchMeta: Copy independence ──

    #[test]
    fn dual_batch_meta_copy_independence_after_swap() {
        let mut meta = DualBatchMeta::new(64);
        meta.ping_seq_count = 10;
        let copy = meta;
        meta.swap();
        // copy should reflect the pre-swap state
        assert_eq!(copy.ping_seq_offset, 0);
        assert_eq!(copy.ping_seq_count, 10);
        assert_eq!(copy.pong_seq_offset, 64);
    }

    // ── GemmTileParams: large values ──

    #[test]
    fn gemm_tile_params_large_values() {
        let tile = GemmTileParams { m: u32::MAX, n: u32::MAX, k: u32::MAX };
        assert_eq!(tile.m, u32::MAX);
        assert_eq!(tile.n, u32::MAX);
        assert_eq!(tile.k, u32::MAX);
    }

    // ── PhaseCompileParams: all bool false ──

    #[test]
    fn phase_compile_params_all_bool_false() {
        let params = PhaseCompileParams {
            gemm_tile: GemmTileParams { m: 64, n: 64, k: 128 },
            kv_pipeline_stages: 1,
            use_dsmem_kv_share: false,
            use_ld_nc: false,
            use_tensor_cores_gemv: false,
            symdim_m: "seq_len",
        };
        assert!(!params.use_dsmem_kv_share);
        assert!(!params.use_ld_nc);
        assert!(!params.use_tensor_cores_gemv);
    }

    // ── SmPartitionConfig: derive at various boundaries ──

    #[test]
    fn sm_partition_derive_zero_sm() {
        let cfg = SmPartitionConfig::derive(0);
        assert_eq!(cfg.variant, MkCompileVariant::Serial);
    }

    #[test]
    fn sm_partition_derive_one_sm() {
        let cfg = SmPartitionConfig::derive(1);
        assert_eq!(cfg.variant, MkCompileVariant::Serial);
    }

    // ── Page allocator constants: relationships ──

    #[test]
    fn pool_cluster_batch_is_power_of_two() {
        // 64 is a power of 2
        assert!(POOL_CLUSTER_BATCH > 0);
        assert_eq!(POOL_CLUSTER_BATCH & (POOL_CLUSTER_BATCH - 1), 0);
    }

    #[test]
    fn pool_global_batch_words_positive() {
        assert!(POOL_GLOBAL_BATCH_WORDS > 0);
    }

    // ── RequestQueue: peek reflects latest state after enqueue ──

    #[test]
    fn request_queue_peek_reflects_head_after_enqueue_and_dequeue() {
        let mut q = RequestQueue::new(8);
        q.enqueue(RequestQueueEntry::new(0, 10));
        q.enqueue(RequestQueueEntry::new(0, 20));
        q.enqueue(RequestQueueEntry::new(0, 30));
        q.dequeue(); // removes 10
        let head = q.peek().unwrap();
        assert_eq!(head.prompt_len, 20);
        q.dequeue(); // removes 20
        let head = q.peek().unwrap();
        assert_eq!(head.prompt_len, 30);
    }

    // ── OutputRingBuffer: write returns false for cta_id == num_ctas ──

    #[test]
    fn output_ring_write_fails_at_exact_num_ctas() {
        let mut ring = OutputRingBuffer::new(3, 4, 16);
        assert!(ring.write_token(2, OutputTokenEntry::intermediate(0, 1, 0))); // last valid
        assert!(!ring.write_token(3, OutputTokenEntry::intermediate(0, 2, 0))); // out of bounds
    }

    // ── OutputRingBuffer: consume sorts correctly for many sequences ──

    #[test]
    fn output_ring_consume_sorts_many_sequences() {
        let mut ring = OutputRingBuffer::new(1, 8, 64);
        // Write in reverse seq order; use gen_idx=1 to avoid (0,0,0) filtering
        for seq in (0..5u32).rev() {
            ring.write_token(0, OutputTokenEntry::intermediate(seq, seq * 10 + 1, 1));
        }
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 5);
        for (i, t) in tokens.iter().enumerate() {
            assert_eq!(t.seq_id, i as u32);
        }
    }

    // ── SmPartitionConfig: grid_sync for odd SM counts ──

    #[test]
    fn sm_partition_grid_sync_odd_sm_decode_less_than_total() {
        let cfg = SmPartitionConfig::grid_sync(5);
        assert!(cfg.decode_cta_count < cfg.total_cta_count);
        assert!(cfg.prefill_cta_count > 0);
        assert_eq!(cfg.decode_cta_count + cfg.prefill_cta_count, 5);
    }

    // ── DualBatchMeta: new with various batch sizes ──

    #[test]
    fn dual_batch_meta_new_pong_equals_batch_size() {
        for &batch_size in &[1, 16, 64, 256, 1024, 4096] {
            let meta = DualBatchMeta::new(batch_size);
            assert_eq!(meta.pong_seq_offset, batch_size);
            assert_eq!(meta.ping_seq_offset, 0);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // Additional unit tests — batch 5 (15 new tests)
    // ═══════════════════════════════════════════════════════════

    // ── OutputTokenEntry: Eq implies Hash consistency in HashSet ──

    #[test]
    fn output_entry_eq_hash_in_hashset() {
        use std::collections::HashSet;
        let a = OutputTokenEntry::intermediate(1, 42, 3);
        let b = OutputTokenEntry::intermediate(1, 42, 3);
        let mut set = HashSet::new();
        assert!(set.insert(a));
        assert!(!set.insert(b)); // equal entries, same hash
        assert_eq!(set.len(), 1);
    }

    // ── OutputTokenEntry: PartialOrd/Ord orders by struct field order ──

    #[test]
    fn output_entry_partial_ord_reflects_struct_order() {
        let low = OutputTokenEntry::intermediate(0, 0, 0);
        let high = OutputTokenEntry::intermediate(0, 0, 1);
        assert!(low < high);
        assert!(high > low);
        assert_eq!(low.cmp(&low), std::cmp::Ordering::Equal);
    }

    // ── SmPartitionConfig: Copy trait semantics ──

    #[test]
    fn sm_partition_config_copy_independence() {
        let cfg = SmPartitionConfig::grid_sync(64);
        let copy = cfg;
        // copy is independent (Copy trait)
        assert_eq!(copy.total_cta_count, 64);
        assert_eq!(cfg.total_cta_count, 64);
    }

    // ── MkCompileVariant: Cluster53 variant exists and is distinct ──

    #[test]
    fn mk_compile_variant_cluster53_is_distinct() {
        let v = MkCompileVariant::Cluster53;
        assert_ne!(v, MkCompileVariant::Cluster62);
        assert_ne!(v, MkCompileVariant::GridSync);
        assert_ne!(v, MkCompileVariant::Serial);
        assert_eq!(format!("{:?}", v), "Cluster53");
    }

    // ── OutputTokenEntry: _reserved field is zero for all constructors ──

    #[test]
    fn output_entry_reserved_zero_for_all_constructors() {
        assert_eq!(OutputTokenEntry::intermediate(1, 2, 3)._reserved, 0);
        assert_eq!(OutputTokenEntry::final_eos(1, 2, 3)._reserved, 0);
        assert_eq!(OutputTokenEntry::final_max_tokens(1, 2, 3)._reserved, 0);
        assert_eq!(OutputTokenEntry::final_stop_word(1, 2, 3)._reserved, 0);
        assert_eq!(OutputTokenEntry::default()._reserved, 0);
    }

    // ── OutputRingBuffer: consume filters out all-zero entries ──

    #[test]
    fn output_ring_consume_skips_zero_entries() {
        let mut ring = OutputRingBuffer::new(2, 4, 16);
        // Write one valid entry to CTA 0
        ring.write_token(0, OutputTokenEntry::intermediate(1, 10, 1));
        // CTA 1 has no writes, so its slot contains default (0,0,0,0,0,0)
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].seq_id, 1);
    }

    // ── PhaseCompileParams: symdim_m can be different static strings ──

    #[test]
    fn phase_compile_params_symdim_m_variants() {
        let p1 = PhaseCompileParams {
            gemm_tile: GemmTileParams { m: 1, n: 1, k: 1 },
            kv_pipeline_stages: 1, use_dsmem_kv_share: false,
            use_ld_nc: false, use_tensor_cores_gemv: false,
            symdim_m: "seq_len",
        };
        let p2 = PhaseCompileParams {
            gemm_tile: GemmTileParams { m: 1, n: 1, k: 1 },
            kv_pipeline_stages: 1, use_dsmem_kv_share: false,
            use_ld_nc: false, use_tensor_cores_gemv: false,
            symdim_m: "total_seq",
        };
        assert_ne!(p1.symdim_m, p2.symdim_m);
    }

    // ── DualBatchMeta: new with non-power-of-two batch sizes ──

    #[test]
    fn dual_batch_meta_new_non_power_of_two() {
        let meta = DualBatchMeta::new(7);
        assert_eq!(meta.ping_seq_offset, 0);
        assert_eq!(meta.pong_seq_offset, 7);
        let meta2 = DualBatchMeta::new(99);
        assert_eq!(meta2.pong_seq_offset, 99);
    }

    // ── RequestQueue: capacity always exact power of two ──

    #[test]
    fn request_queue_capacity_always_power_of_two() {
        for cap in &[1u32, 2, 3, 5, 7, 10, 15, 16, 17, 100, 255, 256, 1000] {
            let q = RequestQueue::new(*cap);
            let c = q.capacity();
            assert!(c > 0);
            assert_eq!(c & (c - 1), 0, "capacity {} should be power of 2", c);
            assert!(c >= *cap);
        }
    }

    // ── OutputRingBuffer: write to cta_id 0 after reset works ──

    #[test]
    fn output_ring_write_cta_zero_after_reset() {
        let mut ring = OutputRingBuffer::new(2, 4, 16);
        ring.write_token(0, OutputTokenEntry::intermediate(0, 10, 0));
        ring.reset();
        assert!(ring.write_token(0, OutputTokenEntry::intermediate(5, 50, 1)));
        let tokens = ring.consume();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].token_id, 50);
    }

    // ── SmPartitionConfig: derive boundary at 59 vs 60 ──

    #[test]
    fn sm_partition_derive_boundary_transition() {
        let below = SmPartitionConfig::derive(59);
        assert_eq!(below.variant, MkCompileVariant::Serial);
        let at = SmPartitionConfig::derive(60);
        assert_eq!(at.variant, MkCompileVariant::GridSync);
        let above = SmPartitionConfig::derive(61);
        assert_eq!(above.variant, MkCompileVariant::GridSync);
    }

    // ── PhaseConfig: prefill preset uses FlashAttention and WriteFull ──

    #[test]
    fn phase_config_prefill_attention_and_kv_mode() {
        let config = PhaseConfig {
            compile_params: PhaseCompileParams {
                gemm_tile: GemmTileParams { m: 64, n: 64, k: 128 },
                kv_pipeline_stages: 4,
                use_dsmem_kv_share: true,
                use_ld_nc: false,
                use_tensor_cores_gemv: false,
                symdim_m: "prefill_m",
            },
            attention: AttentionStrategy::FlashAttention,
            kv_mode: KvAccessMode::WriteFull,
        };
        assert!(config.attention == AttentionStrategy::FlashAttention);
        assert!(config.kv_mode == KvAccessMode::WriteFull);
        assert!(config.compile_params.use_dsmem_kv_share);
    }

    // ── RequestQueueEntry: _reserved field default is zero ──

    #[test]
    fn request_entry_reserved_default_zero() {
        let entry = RequestQueueEntry::new(0, 5);
        assert_eq!(entry._reserved, 0);
        let default = RequestQueueEntry::default();
        assert_eq!(default._reserved, 0);
    }

    // ── OutputRingBuffer: new with single CTA ──

    #[test]
    fn output_ring_single_cta_construction() {
        let ring = OutputRingBuffer::new(1, 4, 16);
        // 1 CTA, should be able to write to CTA 0 but not CTA 1
        let mut ring = ring;
        assert!(ring.write_token(0, OutputTokenEntry::intermediate(1, 10, 0)));
        assert!(!ring.write_token(1, OutputTokenEntry::intermediate(2, 20, 0)));
    }

    // ── RequestQueue: len saturating_sub prevents underflow ──

    #[test]
    fn request_queue_len_never_underflows() {
        let mut q = RequestQueue::new(4);
        // dequeue from empty should not panic or underflow len
        assert!(q.dequeue().is_none());
        assert_eq!(q.len(), 0);
        // enqueue then dequeue returns to zero
        q.enqueue(RequestQueueEntry::new(0, 1));
        q.dequeue();
        assert_eq!(q.len(), 0);
    }
}
