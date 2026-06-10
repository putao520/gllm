# 页级压缩与冷热换入换出 (PAGE-COMPRESSION-SWAPPING)

> **实现状态**: ✅ REQ-COMP-001~016 全部完成 (KvPageHeader 56B + CompressionCodec 5 变体 + StorageTier 三级 + PageMigrationActor 异步迁移 + EvictionWorker/SwapInWorker + Lz4/BitPackRle/ZstdDict CPU 压缩 + LZ4 JIT 解压 x86/AArch64/GPU 三后端 + BitPackRle JIT 解压三后端 + Mega-Kernel KV 页解压注入 + 权重页压缩 + NVMe Swap File + 评分驱动迁出 + 数值正确性测试)
>
> **SSOT**: 本文件定义 KV cache 与权重页的压缩属性、GPU 友好压缩算法、冷页换出 / 热页评分驱动迁出 / 压缩换入解压机制。与 §19 KV Cache 优化 (PrecisionTier) 和 §21 Weight Paging (WeightTierManager) 协同 — §19 管"在原物理页内的精度等级"，§21 管"权重页的 Tier (HBM/DRAM/NVMe)"，本文件管"页内字节流的可逆压缩 + 跨设备换入换出"。

<div data-cross-repo-xrefs>
<b>跨仓库依赖 (gllm-nccl)</b>:
本文件 REQ-COMP (页级压缩) 与 gllm-nccl REQ-COMP (通信压缩模板) 域名重叠但语义不同：
本文件 = KV/权重页压缩 (CPU+GPU JIT)；gllm-nccl = 通信管线压缩 (RDMA 传输优化)。
交叉依赖:
<a data-xref-id="REQ-DP-008" data-xref-type="req" href="../gllm-nccl/SPEC/08-DISTRIBUTED-PAGING.md#REQ-DP-008">REQ-DP-008</a>
(带量化压缩的 RDMA 页传输) 消费本文件 CompressionCodec 的 GPU 端 BitPackRle 解压 + gllm-nccl <a data-xref-id="REQ-COMP-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-COMP-001">REQ-COMP-001</a>~<a data-xref-id="REQ-COMP-004" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-COMP-004">REQ-COMP-004</a> (位打包/差分/RLE/稀疏编码模板) |
<a data-xref-id="REQ-QUANT-001" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-QUANT-001">REQ-QUANT-001</a>
<a data-xref-id="REQ-QUANT-002" data-xref-type="req" href="../gllm-nccl/SPEC/01-REQUIREMENTS.md#REQ-QUANT-002">REQ-QUANT-002</a>
(通信管线量化)
</div>
>
> 压缩是**在 PrecisionTier 之上的正交维度**。PrecisionTier 决定语义信息密度（FP16/FP8/KIVI4/...），压缩决定字节流可逆密度（已 quantize 的 bit stream 进一步用 GPU-friendly entropy coder 压缩）。
>
> 不与 §19 KIVI 量化 / §21 WeightTierManager 重复 — 这两者是上层调度，本文件是底层字节流。

## §1 设计原则

### §1.1 三层正交存储维度（扩展 §19 §1.1）

```
信息密度 (PrecisionTier, §19)
    ↓
比特流密度 (CompressionCodec, 本文件 §3)
    ↓
存储位置 (StorageTier, 本文件 §4)
```

| 维度 | 决定 | 谁写入 | 谁读取 |
|------|------|--------|--------|
| **PrecisionTier** | bits per element | Epilogue 量化 (§19 §3) | Mega-Kernel attention/QuantGemm |
| **CompressionCodec** | bytes per quantized block | Eviction Worker (CPU) | Swap-In Worker (GPU/CPU) |
| **StorageTier** | 物理位置 | HGAL 调度器 | 缺页处理器 |

### §1.2 压缩与解压的不对称性

- **压缩**：可以离线/异步执行（页变冷后才压缩），允许重计算
- **解压**：必须在 hot path（页被换入命中后立即可用），硬延迟 < 1ms
- **设计选择**：压缩端可用 "高比率 + 慢算法"（zstd），解压端必须 "GPU-native + 流式" — 见 §3.2 算法选型

### §1.3 与 PrecisionTier 的协同

| PrecisionTier | 是否再压缩 | 理由 |
|---------------|-----------|------|
| `FP16` (sink/全精度) | ✅ LZ4 | 浮点冗余高，LZ4 ~30% 压缩率 |
| `FP8` | ✅ LZ4 | 同 FP16 |
| `KIVI4` / `KIVI2` | ✅ Bit-pack + RLE | 量化已极限，但 nibble stream 中 0/sink 重复 |
| `Sparse` | ❌ | bitmap 已是稀疏编码，再压缩 ROI 低 |
| `Dictionary` | ❌ | dict_id 已是熵编码 |
| `Evicted` | ❌ | 直接释放 |

### §1.4 禁止规则

- ❌ 禁止 Rust 端解压 — 解压必须在 GPU kernel 内完成（或 SIMD JIT 解码）
- ❌ 禁止 CPU 解压后 memcpy 到 GPU — 必须 "压缩字节流 → cudaMemcpy → kernel 内解压"
- ❌ 禁止以"解压复杂"为由 fallback 到 PrecisionTier 升级（这是绕过压缩，不是压缩）
- ❌ 禁止压缩破坏页对齐（解压后必须保持原 PrecisionTier 的 page layout）

## §2 PageHeader 压缩区域语义

> **📌 物理布局 SSOT**: `KvPageHeader (56B)` 完整物理布局定义在 `03-DATA-STRUCTURE §8.0`。
> 本节仅描述压缩区域 (offset 44..52, 8B) 的**语义**。

压缩区域 (offset 44..52, 8B):
- `codec`: CompressionCodec 枚举 (见 §3.1)
- `compressed_size`: 压缩后字节数（0 = 未压缩）
- `checksum`: CRC16 校验（防腐）
- `storage_tier`: StorageTier 枚举 (见 §4.1)

WeightPage 同样扩展（详见 §6 与 §21 §3.1 的协同）。

## §3 CompressionCodec — GPU 友好压缩算法

### §3.1 Codec 枚举

```rust
#[repr(u8)]
pub enum CompressionCodec {
    /// 未压缩 — 默认（hot page）
    None = 0,
    /// LZ4-frame — 对 FP16/FP8 有效，GPU 解压成熟（nvCOMP / 自研 SIMD）
    /// 压缩端: CPU LZ4_compress_HC (~500 MB/s, 30-40% 比率)
    /// 解压端: GPU LZ4 stream decoder (kernel 内解压)
    Lz4 = 1,
    /// Bit-pack RLE — 对 KIVI4/KIVI2 nibble stream 有效
    /// 压缩端: 扫描连续 0/sink token → run-length encoding
    /// 解压端: SIMD/GPU 流式展开（无分支）
    BitPackRle = 2,
    /// nvCOMP ANS / Bitcomp — GPU 原生熵编码（仅 NVIDIA）
    /// 比率 50-60%, 解压 100+ GB/s on H100
    NvcompAns = 3,
    /// Zstd 字典模式 — 离线训练字典 (跨 page 共享)
    /// 仅 cold-tier (DRAM/NVMe)，不直接送回 GPU 解压
    ZstdDict = 4,
}
```

### §3.2 算法选型矩阵

| Codec | 压缩比 | 压缩速度 | 解压速度 | GPU 解压 | 适用场景 |
|-------|--------|----------|----------|----------|----------|
| `None` | 1.00 | — | — | — | hot page |
| `Lz4` | 0.30-0.40 | 500 MB/s (CPU) | 5 GB/s (CPU) / 80 GB/s (GPU) | ✅ JIT decoder | warm FP16/FP8 |
| `BitPackRle` | 0.20-0.30 | 1 GB/s (CPU SIMD) | 10 GB/s (SIMD) / 200 GB/s (GPU) | ✅ JIT decoder | KIVI4/KIVI2 nibble |
| `NvcompAns` | 0.50-0.60 | 200 MB/s (GPU encoder) | 100 GB/s (H100) | ✅ nvCOMP runtime | NVIDIA only, cold-tier |
| `ZstdDict` | 0.10-0.20 | 100 MB/s | 1 GB/s (CPU) | ❌ | NVMe → DRAM 阶段，必须先 CPU 解压再上 GPU |

> **量化格式的天然压缩**: 量化 KV page（如 Q4_0、KIVI4）本身已是极限紧凑表示（Q4_0 存储 32 元素仅需 18 字节，天然压缩率 ~55%）。对这些格式再应用 BitPackRle 或 Lz4 的 ROI 为负（解压 CPU 开销 > 传输带宽节省）。压缩选择策略应跳过已经高度压缩的量化格式，仅在 NVMe tier 考虑 ZstdDict 追加压缩。

### §3.3 GPU 解压实现路径

#### §3.3.1 LZ4 解压（自研 PTX/HIP/MSL）

```
LZ4 stream format: [token: u8][literal_len][match_offset: u16][match_len]
GPU kernel:
  - 1 thread / token (warp-level cooperation for match copy)
  - shared memory 4KB sliding window
  - 输出 stream 直接写入 page 物理位置
JIT 入口: VmInstr::Lz4Decode { src_compressed_ptr, dst_page_ptr, compressed_size }
```

#### §3.3.2 BitPackRle 解压（SIMD/GPU）

```
Format: [run_value: u4][run_len: u4] 一个 byte 一个 run, run_len=15 → escape
GPU kernel:
  - 1 thread / run（warp prefix-sum 计算 dst offset）
  - 全 lane 并行展开
JIT 入口: VmInstr::BitPackRleDecode { src, dst, format: PrecisionTier }
```

`PrecisionTier` 参数让 decoder 知道 nibble 的语义（KIVI4 → 4-bit-per-element + scale, KIVI2 → 2-bit + scale）。

#### §3.3.3 nvCOMP 集成

NVIDIA 平台优先使用 nvCOMP。编译期通过 `feature = "nvcomp"` 启用，runtime 检测 cuTensorMap + nvCOMP 库。无 nvCOMP → fallback 到自研 LZ4。

### §3.4 算法选择算法

```rust
fn select_codec(
    tier: PrecisionTier,
    hw: &DeviceProfile,
    page_kind: PageKind,
) -> CompressionCodec {
    match page_kind {
        PageKind::Weight => {
            // 权重页: 量化即压缩 (SPEC 23 QuantFormatDescriptor)，页级不再额外压缩
            // 量化权重页以量化格式存储 (AWQ4/GPTQ4/NVFP4 等)，GEMM 前由 JIT 解量化 prologue 处理
            CompressionCodec::None
        }
        PageKind::KvCache => {
            // KV 页: 动态压缩，根据精度等级选择最优 codec
            match tier {
                PrecisionTier::FP16 | PrecisionTier::FP8 => {
                    if hw.has_nvcomp() { CompressionCodec::NvcompAns }
                    else { CompressionCodec::Lz4 }
                }
                PrecisionTier::KIVI4 | PrecisionTier::KIVI2 => CompressionCodec::BitPackRle,
                PrecisionTier::Sparse | PrecisionTier::Dictionary => CompressionCodec::None,
                PrecisionTier::Evicted => CompressionCodec::None,
            }
        }
    }
}
```

cold-tier (DRAM → NVMe) 强制使用 ZstdDict，因为该 tier 不会回 GPU。

权重页不使用页级压缩（量化格式本身即是压缩，见 SPEC 23 QuantFormatDescriptor）。解量化由 GEMM 的 JIT prologue 完成（SPEC 24 QuantGather）。

## §4 StorageTier — 三级存储层级

### §4.1 Tier 定义

> **跨 SPEC Tier 映射**: `StorageTier` 与 06-RUNTIME `Tier`、21-WEIGHT-PAGING `WeightTier` 是同一物理层级的三套领域视图：
>
> | 本 SPEC `StorageTier` | 06-RUNTIME `Tier` | 21-WEIGHT-PAGING `WeightTier` | 物理位置 |
> |---|---|---|---|
> | `GpuHbm` | `L1` | `DeviceLocal` | GPU HBM（微秒级延迟） |
> | `CpuDram` | `L2` | `HostLocal` | CPU DRAM（~10ms PCIe 换入换出） |
> | `Nvme` | `L3` | `DiskMmap` | NVMe 磁盘（~100ms 文件 I/O） |
>
> `StorageTier` 用于 KV/权重页的压缩换入换出路由；`Tier` 用于 `GlobalMemoryManager` 全局容量管理；`WeightTier` 用于加载时权重放置决策。三方值 1:1 对应，实现时需 `From/TryFrom` 转换。

```rust
#[repr(u8)]
pub enum StorageTier {
    /// GPU HBM — hot, 微秒级延迟
    GpuHbm = 0,
    /// CPU DRAM — warm, 通过 PCIe 换入换出 ~10ms
    CpuDram = 1,
    /// NVMe — cold, 通过文件系统 ~100ms
    Nvme = 2,
}
```

### §4.2 Tier 迁移路径

```
       ┌─────────────────┐
       │  GpuHbm (hot)   │ ← 计算路径直接读取
       │  None / lz4-jit │
       └────┬───────▲────┘
   evict    │       │  swap-in
            ▼       │
       ┌─────────────────┐
       │  CpuDram (warm) │ ← 压缩存储, GPU 通过 cuMemcpyAsync 拉取后 kernel 解压
       │  lz4 / bit-pack │
       └────┬───────▲────┘
   evict    │       │  swap-in
            ▼       │
       ┌─────────────────┐
       │  Nvme (cold)    │ ← zstd-dict 高比率压缩，必须先 CPU 解压再上 DRAM
       │  zstd-dict      │
       └─────────────────┘
```

### §4.3 迁移触发条件

```
GpuHbm → CpuDram (evict 触发):
  - HBM 占用 > 90%
  - importance_score < 100 (§19) AND tier_age > 50 ticks
  - 选择 score 最低的 N 个 page 批量 evict

CpuDram → Nvme (evict 触发):
  - DRAM 用于 page 部分占用 > 80%
  - tier_age > 500 ticks (warm 也老化)

CpuDram → GpuHbm (swap-in 触发):
  - prefetch 命中（调度器预测下个 wave 需要）
  - decode 时 page fault（importance score 反弹）

Nvme → CpuDram (swap-in 触发):
  - 长上下文回滚（user re-engages session）
  - 仅触发批量 prefetch，不在 hot path
```

## §5 KV Page 压缩生命周期

### §5.1 状态机

```
[Computing] ──compute done──▶ [Hot, GpuHbm, None]
                                    │
                                    │ score 降低
                                    ▼
                           [Warm-Candidate, GpuHbm, None]
                                    │
                                    │ codec 选择 + GPU encode (异步)
                                    ▼
                           [Compressed, GpuHbm, Lz4/BitPackRle]
                                    │
                                    │ HBM 紧张 → evict to DRAM
                                    ▼
                           [Compressed, CpuDram, Lz4/BitPackRle]
                                    │
                                    │ score 反弹 / prefetch hit
                                    ▼
                           [Swap-In, GpuHbm, Lz4/BitPackRle]
                                    │
                                    │ 第一次 attention read 触发 JIT 解压
                                    ▼
                           [Hot, GpuHbm, None]
```

### §5.2 Eviction Worker（异步线程）

```rust
fn eviction_worker(scheduler: &Scheduler, hw: &DeviceProfile) {
    loop {
        let pressure = scheduler.gpu_hbm_pressure();
        if pressure < 0.85 { yield; continue; }

        // 1. 取 importance_score 最低的 N 个 page
        let evict_set = scheduler.lru_by_score(N);

        // 2. 异步压缩（compress kernel）
        for page in evict_set {
            let codec = select_codec(page.tier, hw);
            scheduler.enqueue_compress(page.id, codec);
        }

        // 3. 等待压缩完成 → memcpy 到 DRAM → 释放 HBM
        scheduler.wait_compress_complete();
        scheduler.batch_dma_to_dram(evict_set);
        scheduler.release_gpu_pages(evict_set);
    }
}
```

### §5.3 Swap-In Worker（hot-path 友好）

```rust
fn swap_in_worker(scheduler: &Scheduler, page_id: PageId) -> Result<()> {
    let header = scheduler.page_header(page_id);
    if header.storage_tier != StorageTier::CpuDram {
        return Err("not in DRAM");
    }

    // 1. 在 GPU 上申请 hbm 页（可能触发反向 evict）
    let gpu_slot = scheduler.allocate_gpu_page()?;

    // 2. cudaMemcpyAsync DRAM → HBM (压缩字节流)
    scheduler.dma_to_hbm(page_id, gpu_slot, header.compressed_size);

    // 3. 标记为 Swap-In，下一次 kernel 读取时 JIT 解压（lazy）
    scheduler.mark_swap_in(page_id, gpu_slot);
    Ok(())
}
```

### §5.4 JIT 解压注入 (Mega-Kernel epilogue)

Mega-Kernel attention kernel 在读取 page 第一行前查 `header.codec`：

```
Mega-Kernel 入口:
  for each page in batch:
    if page.codec != None:
      jit_emit_decode_kernel(page.codec, page.compressed_ptr, page.dst_buffer)
      barrier()
      page.codec = None  // 解压完成后改 header
    standard_attention_read(page)
```

JIT VmInstr 扩展（§3.3.1 / §3.3.2）：
- `Lz4Decode { src, dst, size }`
- `BitPackRleDecode { src, dst, tier }`
- `NvcompDecode { src, dst, manager_handle }` — 调用 nvCOMP runtime

## §6 Weight Page 压缩

### §6.1 与 §21 §2 WeightTierManager 协同

§21 已有 `WeightTier::{HBM, DRAM, NVMe}` 三层放置策略。本文件**仅扩展字节流压缩**，不改变 §21 的 placement 决策。

```rust
struct UnifiedVirtualPage {
    /// §21 已有字段
    payload: PagePayloadKind,
    tier: WeightTier,
    /// §22 新增字段
    codec: CompressionCodec,
    compressed_size: u32,
    decompressed_size: u32,
}
```

### §6.2 权重页压缩策略

| 权重类型 | tier=HBM | tier=DRAM | tier=NVMe |
|----------|----------|-----------|-----------|
| `DenseLayerWeight` (model size > 70% HBM) | None / BitPackRle (Q4_0) | Lz4 | ZstdDict |
| `ExpertWeight` (MoE) | None | BitPackRle (Q4) | ZstdDict |

热权重（layer 0/1, frequently activated experts）始终保持 None；冷权重（last-N layers, infrequent experts）默认 BitPackRle。

### §6.3 解压时机

权重解压发生在 ExpertWeightPrefetcher 命中时（§21 §2.2）：

```
Prefetcher 检测下个 wave 需要 expert E:
  if E.codec != None:
    enqueue_decompress(E)  // 与下个 wave 计算重叠
  schedule_to_hbm(E)
```

JIT QuantGemm 永远只看到解压后的 page，codec 字段在解压完成后清零。

## §7 数据流契约（与 §19/§21 协同）

### §7.1 Eviction 数据流

```
1. Scheduler.build_batch:
   - 收集所有页的 importance_score (来自 §19 §3.1 epilogue)
   - 找出 score 最低的 N 个 → eviction_candidates

2. Eviction Worker (异步):
   - codec = select_codec(tier, hw)
   - GPU compress kernel: page.data → page.compressed_buffer
   - header.codec, header.compressed_size, header.checksum 写入
   - DMA: compressed_buffer (GPU HBM) → DRAM
   - Scheduler.release_gpu_page(page_id)

3. 后续 batch 不会读到这些页 (因为 score 低), 直到:
```

### §7.2 Swap-In 数据流

```
1. Scheduler.build_batch:
   - sequence S 需要 page P
   - lookup: P.tier = CpuDram
   - enqueue_swap_in(P)

2. Swap-In Worker (与计算重叠):
   - allocate_gpu_page → gpu_slot
   - DMA: P.compressed_buffer (DRAM) → gpu_slot (HBM)
   - mark_swap_in(P, gpu_slot)

3. Mega-Kernel:
   - 读 P 第一行前检查 P.codec
   - codec != None → JIT emit decode kernel → 解压到 gpu_slot
   - 写回 P.codec = None
   - 继续 attention read
```

### §7.3 与 §19 PrecisionTier 升降级的关系

**正交关系**：
- §19 决定"该 page 用什么 PrecisionTier"（FP16/KIVI4/KIVI2）
- §22 决定"该 PrecisionTier 的字节流是否再压缩 + 在哪个 storage tier"

升降级与压缩独立触发：
- `score < 80` → §19 升级到 KIVI2（语义降级，hot path 重量化）
- `tier_age > 50` AND `score < 100` → §22 LZ4 压缩 + DRAM eviction（字节流压缩 + storage 迁移）
- 两者可以叠加：KIVI2 + Lz4 + DRAM 是最激进的冷页存储

## §7.5 物理 Tier 迁移路径（真实 DMA / NVMe I/O）

> **现状**：PageMigrationActor 异步线程已实现 (REQ-COMP-014)，消费 MigrationQueue 执行真实 DMA + NVMe I/O 搬运。EvictionWorker/SwapInWorker 已实现 (REQ-COMP-007/008)。NVMe Swap File 已实现 (REQ-COMP-015)。

### §7.5.1 PageMigrationActor

新增 `PageMigrationActor` 异步线程，消费 `MigrationQueue`：

```rust
pub enum MigrationCommand {
    /// GpuHbm → CpuDram (compress + DMA)
    EvictToDram { page_id: PageId, codec: CompressionCodec },
    /// CpuDram → GpuHbm (DMA + lazy decode marker)
    PromoteToHbm { page_id: PageId },
    /// CpuDram → Nvme (compress harder + file write)
    EvictToNvme { page_id: PageId, codec: CompressionCodec },
    /// Nvme → CpuDram (file read + decompress)
    PromoteToDram { page_id: PageId },
}

pub struct PageMigrationActor {
    cmd_rx: Receiver<MigrationCommand>,
    backend: Arc<dyn Backend>,         // GPU memcpy
    nvme_path: PathBuf,                // NVMe swap dir
    completion_tx: Sender<MigrationDone>,
}
```

### §7.5.2 物理搬运实现

**GpuHbm → CpuDram**：
```
1. backend.cuMemcpyDtoH(gpu_ptr, host_buffer, page_bytes)
2. 若 codec != None: 在 CPU 端进一步 lz4/zstd 压缩
3. memory_manager.migrate_page(L1, L2, src_id) 更新页表
4. 释放 GPU 物理页
```

**CpuDram → GpuHbm**：
```
1. memory_manager.allocate_page(L1) → gpu_dst_id（可能触发反向 evict）
2. backend.cuMemcpyHtoD(host_buffer, gpu_ptr, compressed_bytes)
3. memory_manager.migrate_page(L2, L1) 更新页表
4. mark_swap_in(page_id, gpu_slot)，下次 Mega-Kernel 读取时 JIT 解压（§5.4）
```

**CpuDram → Nvme**：
```
1. 在 CPU 端 zstd-dict 压缩（高比率）
2. write_all_at(nvme_file, offset, compressed_bytes)
   nvme 文件按页 ID 划分 slot（page_size 对齐）
3. memory_manager.migrate_page(L2, L3) 更新页表
4. 释放 CPU host_buffer
```

**Nvme → CpuDram**：
```
1. read_at(nvme_file, offset, compressed_bytes)
2. CPU zstd 解压（不上 GPU 直接解压）
3. memory_manager.allocate_page(L2)
4. memory_manager.migrate_page(L3, L2)
```

### §7.5.3 NVMe Swap File 格式

```
~/.gllm/swap/<session_id>.swap
布局: [SwapFileHeader(64B)][slot_0][slot_1]...[slot_N]
SwapFileHeader: magic + version + page_size + slot_count + checksum
每个 slot 固定大小 = max_compressed_page_size，按 page_id 索引

写入: pwrite() 在指定 offset
读取: pread() 在指定 offset
对齐: O_DIRECT + 4096 字节对齐（NVMe 优化）
```

### §7.5.4 触发协议

调度器层面触发：
```
PagedScheduler.build_batch:
  pressure_l1 = tier_manager.usage(L1).pressure();
  if pressure_l1 > 0.85:
    victims = memory_manager.select_victims(L1, count=evict_count)
    for v in victims:
      migration_queue.send(EvictToDram { page_id: v, codec: select_codec(v) })

  for req in active_requests:
    if req.needs_pages_in_l1():
      missing = req.missing_pages()
      for p in missing:
        match page_table.resolve(p).tier:
          L2 => migration_queue.send(PromoteToHbm { page_id: p })
          L3 => migration_queue.send(PromoteToDram { page_id: p })
                后续轮再 PromoteToHbm
```

### §7.5.5 与 Mega-Kernel 的同步

Mega-Kernel 启动前检查 `active_requests` 中所有 page 的 `storage_tier`：
- 全部 L1 → 直接执行
- 有 L2/L3 → 等待对应 `MigrationDone` 信号（带超时）→ 重检查
- 超时 → 该 request 此 wave 跳过，下个 wave 再试

## §8 REQ 清单
- 在 §19 §2.1 现有 48B 基础上加 8B：codec(1) + compressed_size(4) + checksum(2) + storage_tier(1)
- 验证：`std::mem::size_of::<KvPageHeader>() == 56`

### REQ-COMP-002: CompressionCodec 枚举与选择算法
- 实现 §3.1 的 4 个 codec 变体 (None/Lz4/BitPackRle/NvcompAns/ZstdDict)
- 实现 §3.4 的 `select_codec(tier, hw, page_kind) -> CompressionCodec`
- 权重页返回 None（量化即压缩，见 SPEC 23）
- 验证：单元测试覆盖每种 (PrecisionTier, hw_capability, PageKind) 组合

### REQ-COMP-003: GPU JIT LZ4 解压
- 实现 `VmInstr::Lz4Decode { src, dst, size }` 在 PTX/HIP/MSL 三方言的 lowering
- shared memory 4KB 滑窗 + warp 协作 match copy
- 验证：随机数据 round-trip（CPU LZ4 编码 → GPU 解码 → 字节级相等）

### REQ-COMP-004: SIMD/GPU JIT BitPackRle 解压
- 实现 `VmInstr::BitPackRleDecode { src, dst, tier }` 在 x86/ARM/GPU 三平台的 lowering
- KIVI4/KIVI2 nibble stream 流式展开
- 验证：随机 KIVI4 数据 round-trip

### REQ-COMP-005: nvCOMP 集成 (feature-gated)
- `cargo feature = "nvcomp"` 启用
- runtime 检测 nvCOMP 库存在，否则 fallback 到 LZ4
- 验证：H100 上 ANS encode/decode 性能 > 50 GB/s

### REQ-COMP-006: StorageTier 三级
- 实现 `StorageTier::{GpuHbm, CpuDram, Nvme}`
- 集成到 §21 `UnifiedVirtualPage.tier`（不破坏 §21 的 WeightTier 语义）
- 验证：`page_header.storage_tier` 与实际指针归属一致

### REQ-COMP-007: Eviction Worker
- 异步线程，HBM pressure > 0.85 触发
- 选 score 最低的 N 个 page，并行压缩 + DMA 到 DRAM
- 验证：HBM 占用从 95% → 70% 的 round-trip 测试

### REQ-COMP-008: Swap-In Worker
- 调度器调用 `prefetch_swap_in(page_id)`
- DMA 压缩字节流 → HBM, lazy decode
- 验证：长上下文回滚测试，命中率 > 95%

### REQ-COMP-009: Mega-Kernel JIT 解压注入
- attention kernel epilogue 检查 `header.codec`
- codec != None → JIT 调用对应解压 VmInstr → 写回 codec=None
- 验证：混合压缩/未压缩 page 的 batch 推理结果与全未压缩一致（数值完全相等）

### REQ-COMP-010: 权重页压缩 (与 §21 协同)
- 扩展 `UnifiedVirtualPage` 加 codec/compressed_size/decompressed_size
- ExpertWeightPrefetcher 命中后调用解压
- 验证：MoE 模型 cold experts 启用 BitPackRle 后内存节省 60%, 推理结果不变

### REQ-COMP-011: 评分驱动迁出
- 基于 §19 §3.1 importance_score
- 同时考虑 tier_age (新页不立即压缩) 和 layer 深度 (浅层不积极压缩)
- 验证：score 调控曲线下，hot pages 不被错误 evict

### REQ-COMP-012: Cold-Tier ZstdDict
- NVMe 层使用 ZstdDict
- 离线训练字典（每个模型一份），cold-load 时只 CPU 解压（不上 GPU 直接解压）
- 验证：512K context 的 KV cache 在 NVMe 占用 < 50 MB

### REQ-COMP-013: 数值正确性回归测试
- TEST-COMP-001: 全 None 路径 vs Lz4 路径输出 token 完全一致
- TEST-COMP-002: 全 None 路径 vs BitPackRle 路径输出 token 完全一致
- TEST-COMP-003: 混合 codec batch 输出 token 完全一致

### REQ-COMP-014: PageMigrationActor 物理搬运
- 实现 §7.5.1 异步 actor + 4 个 MigrationCommand 处理路径
- 真实调用 backend cuMemcpy / NVMe pwrite/pread（不止更新页表元数据）
- 验证：单元测试 GPU page 字节级 round-trip（HBM → DRAM → HBM 后内容相等）

### REQ-COMP-015: NVMe Swap File
- 实现 §7.5.3 文件格式：header + 固定大小 slot
- O_DIRECT + 4096 对齐 I/O
- 文件路径 `~/.gllm/swap/<session_id>.swap`
- 验证：长上下文测试（512K tokens）KV cache 真实落盘 NVMe

### REQ-COMP-016: 端到端 Tier 流转
- 调度器触发协议（§7.5.4）：pressure 触发 evict、缺页触发 promote
- Mega-Kernel 同步等待 swap-in 完成（§7.5.5）
- 验证：HBM 限制 1GB + 8GB context 测试，自动 evict 到 DRAM 并按需回流

## §9 实现顺序

```
Phase 1: PageHeader 扩展 (REQ-COMP-001) — 不影响现有逻辑
Phase 2: CompressionCodec 枚举 + select_codec (REQ-COMP-002)
Phase 3: PageMigrationActor 骨架 + GPU↔CPU 真实 DMA (REQ-COMP-014, REQ-COMP-007/008)
Phase 4: NVMe Swap File I/O (REQ-COMP-015, REQ-COMP-006)
Phase 5: 端到端 Tier 流转 + 调度器触发 (REQ-COMP-016)
Phase 6: BitPackRle JIT 解压 (REQ-COMP-004) — 简单先打通
Phase 7: Mega-Kernel JIT 解压注入 (REQ-COMP-009)
Phase 8: LZ4 JIT 解压 (REQ-COMP-003)
Phase 9: 权重页接入 (REQ-COMP-010)
Phase 10: nvCOMP 可选路径 (REQ-COMP-005)
Phase 11: ZstdDict cold-tier (REQ-COMP-012)
Phase 12: 数值正确性测试 (REQ-COMP-013)
```

## §10 与现有 SPEC 的关系

| 章节 | 关系 | 说明 |
|------|------|------|
| §19 KV Cache Optimization | 协同 | §19 管语义精度（PrecisionTier），§22 管字节流压缩。两者正交叠加。 |
| §21 Weight Paging | 协同 | §21 管 placement 决策（HBM/DRAM/NVMe），§22 管该 placement 内的字节流是否压缩。 |
| §16 Device Fusion | 弱耦合 | §16 融合规则不感知压缩；解压在 attention kernel epilogue，已经是 §16 融合后的产物。 |
| §01 JIT Pipeline §5.5 | 弱耦合 | DequantComputeVariant 处理"权重 dequant"，本文件的 codec decode 是"页字节流 decode"，两者是不同层次。 |
