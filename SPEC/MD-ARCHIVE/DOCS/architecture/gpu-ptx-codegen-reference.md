# GPU PTX Codegen 技术参考 (Shared Memory + Tensor Core + Sampling)

> **用途**: 本文档为 gllm GPU PTX codegen 提供 Shared Memory 管理、Tensor Core MMA 寄存器布局、
> TMA 描述符格式、Sampling JIT 实现等底层技术细节。
> 实现规范见 `SPEC/17-DEVICE-CODEGEN.md` 和 `SPEC/15-GPU-HOST-GLUE.md`。

## §1 PTX Shared Memory 管理

### §1.1 VmInstr → PTX 映射

| VmInstr | PTX 指令 | 说明 |
|---------|---------|------|
| `SharedMemAlloc { name, bytes }` | `.shared .align 16 .b8 name[bytes]` | 共享内存声明 |
| `SharedMemStore { name, dst_offset, src, ... }` | `st.shared.{dtype} [name + offset], %r` | Register → Shared |
| `SharedMemLoad { dst, name, src_offset, ... }` | `ld.shared.{dtype} %r, [name + offset]` | Shared → Register |
| `SharedMemAsyncStore { name, dst_offset, ... }` | SM80: `cp.async.ca.shared.global`; SM90+: `cp.async.bulk.tensor` | Global → Shared 异步 |
| `SharedMemAsyncWaitGroup { n }` | SM80: `cp.async.wait_group N`; SM90+: `mbarrier.wait.aligned` | 等待异步完成 |
| `BlockSync` | `bar.sync 0` | Block 同步 |
| `WarpSync` | `bar.sync {warp_level}` | Warp 同步 |

### §1.2 SM80 cp.async (Ampere)

每个线程独立或协作发起, Global → Shared 绕过寄存器:

```ptx
cp.async.ca.shared.global [%smem_ptr], [%gmem_ptr], 16;
// 异步拷贝 16 字节, .ca = cache-agnostic 缓存策略
cp.async.commit_group;    // 提交当前异步组
cp.async.wait_group 0;    // 等待所有组完成 (0 = 全部)
cp.async.wait_group 1;    // 等待到只剩 1 组未完成 (double buffering)
```

### §1.3 SM90+ TMA (Tensor Memory Accelerator)

单线程/单 Warp 控制硬件 TMA 单元, 多维 Tensor 块搬运:

```ptx
// 2D TMA 加载
cp.async.bulk.tensor.2d.shared.global
    [%smem_data], [%tma_desc_ptr], {%coord_x, %coord_y}, [%mbar_ptr];
// tma_desc_ptr: 预配置的 128 字节 CUtensorMap 描述符
// %coord_x, %coord_y: Tensor 坐标
// %mbar_ptr: 完成信号绑定的 mbarrier
```

TMA 优势: 单条指令搬运整个 Tile (如 128×128 矩阵块), 零线程参与。

### §1.4 Double Buffering (Ping-Pong) 流水线

Shared Memory 分配 = Stage 数 × Tile 大小:

```
.shared .align 16 .b8 smem_buffer[STAGES * TILE_SIZE];

Stage 0 (Ping): [MMA 计算消费] ← 依赖上一轮完成信号
Stage 1 (Pong): [TMA/cp.async 异步加载] ← 填入本轮搬运队列
```

JIT 编译器状态追踪:
- `stage_write_index`: 当前 AsyncStore 写入的共享内存偏移 (`index × tile_size`)
- `stage_read_index`: 当前 Tensor Core 消费的共享内存偏移

同步规则:
- AsyncStore 后: 插入 `wgmma.commit_group` / `cp.async.commit_group`
- 消费前: 根据 stage 差距插入 `cp.async.wait_group N` 或监控 mbarrier Phase bit
- 确保无读写冲突

## §2 Tensor Core MMA 寄存器布局

### §2.1 mma.sync.aligned.m16n8k16 Fragment Layout

32 个线程协作完成 16×8×16 矩阵乘。每个线程的寄存器碎片 (Fragment):

#### 矩阵 A (srcA, 形状 16×16)

- 每线程: 4 个 32-bit 寄存器 (8 个 16-bit 元素 / 16 个 8-bit 元素)
- 32 线程分为 4 个 Thread Group (每组 8 线程)
- Thread `i` 负责第 `⌊i/4⌋` 行和第 `⌊i/4⌋ + 8` 行
- 行内每线程持有 2 个相邻元素 (1 个 32-bit 寄存器打包)

#### 矩阵 B (srcB, 形状 16×8)

- 每线程: 2 个 32-bit 寄存器 (4 个 16-bit 元素)
- Thread `i` 负责第 `i mod 8` 列
- 列内负责 `⌊i/8⌋ × 2` 和 `⌊i/8⌋ × 2 + 1` 以及偏移 8 的共 4 个元素

#### FP16/BF16 vs INT8 打包

| 数据类型 | 每寄存器元素 | K 维度 | 打包格式 |
|---------|------------|-------|---------|
| FP16 | 2 个 16-bit | 16 | `[elem1 \| elem0]` 高低 16-bit |
| BF16 | 2 个 16-bit | 16 | 同 FP16 |
| INT8 | 4 个 8-bit | 32 (翻倍) | `[e3\|e2\|e1\|e0]` 紧密打包 |

### §2.2 Major (行列优先) 影响

改变矩阵 Major (如 A 从 Row → Col) 直接改变元素在寄存器组 `%r0-%r3` 内的索引顺序。
JIT codegen 必须插入 `prmt` (字节排列) 指令或改写加载源地址, 保证送入 MMA 前内存序正确。

### §2.3 WGMMA Matrix Descriptor (64-bit)

SM90 WGMMA 的共享内存操作数通过 64-bit 描述符传递:

```
Bit Field    Name                    Description
[0:13]       Start Address           共享内存基地址 (16B Paragraph 对齐)
[14:27]      Leading Dimension Offset 主维度步长 (Stride)
[28:41]      Stride Offset           第二维度步长 (32B/64B 块跳跃)
[42:44]      Base Offset             内部微调偏移
[46:48]      Swizzle Mode            交错排布模式 (消除 Bank Conflict)
                                     0x1: 32B xor, 0x2: 64B xor, 0x3: 128B xor
[62:63]      Data Type/Size          0=FP16, 1=BF16, ...
```

**Swizzle Mode 至关重要**: 定义共享内存的交错排布, 消除 Tensor Core 高并发读取时的 Bank Conflict。

### §2.4 CUtensorMap (Host 端, 128 字节)

Host 端通过 Driver API 配置, 传递给 GPU TMA:

```
CUtensorMap (128 bytes):
  - globalAddress:    全局内存基地址
  - globalDim:        各维度大小
  - globalStrides:    各维度步长
  - boxDim:           搬运块大小
  - elementStrides:   元素级步长
  - interleave:       交错模式
  - swizzle:          Bank conflict 消除模式
  - l2Promotion:      L2 缓存提升策略
```

## §3 Sampling JIT 实现

### §3.1 核心架构: 免排序 (Sorting-Free) 并行

现代 JIT 编译器 (FlashInfer / vLLM) **绝不使用标量循环** (Thread Divergence),
而是通过并行前缀和 + Warp Shuffle 实现。

### §3.2 融合流水线: Temperature → Top-K → Softmax → Top-P → Sample

```
[原始 Logits (vocab_size 维)]
    │
    ▼
1. Temperature ──→ 标量乘法 (mul.f32 logits, logits, inv_temp)
    │
    ▼
2. Top-K ───────→ Radix Select / Warp 内 Top-K 筛选
    │              (K ≤ 64 时寄存器内 Bit-mask 维护小顶堆)
    │              非 Top-K 元素概率置为 -INF → 后续 Exp 自然为 0
    ▼
3. Softmax ─────→ 并行 Exp → WarpReduceSum → 归一化为概率分布
    │
    ▼
4. Top-P ───────→ BlockScanIncl (并行前缀和) → 过滤 C_i > P 的 Token
    │              setp.ge.f32 predicate, C_i, top_p_threshold
    ▼
5. Sample ──────→ curand 随机数 → 拒绝采样 (Rejection Sampling)
                   或并行二分查找命中最终 Token ID
```

### §3.3 关键硬件原语

| 原语 | PTX 指令 | 用途 |
|------|---------|------|
| WarpReduceMax/Sum | `shfl.sync.bfly` | Softmax 分母、Max 数值稳定 |
| BlockScanIncl/Excl | Warp Shuffle + SMEM 级联 | Top-P 累积概率前缀和 |
| MatchAny / Ballot | `match.any.sync` / `ballot.sync` | 条件过滤, 追踪越过阈值的 Token |
| Predicate | `setp.ge.f32 p, val, threshold` | 单周期条件判断 |

### §3.4 免排序 Top-K 策略

当 K 较小 (≤64):
- 每线程用 Warp Shuffle (`shfl.sync`) 互换数据
- 维护寄存器内小顶堆 (Min-Heap)
- 剔除 Top-K 之外的 Token, 概率直接写 0
- Vocabulary 规模从数万骤降到 K 个

当 K 较大:
- 使用基数选择 (Radix Select): O(N) 时间找到第 K 大元素
- 单遍扫描过滤 ≥ threshold 的元素

### §3.5 Top-P 并行过滤

```
1. 对 Top-K 后的 K 个概率计算包含前缀和 (BlockScanIncl)
   C_i = Σ(j=0..i) P_j
2. 断言过滤: setp.lt.f32 keep, C_i, random_value
   (保留 C_i < random_value 的 Token)
3. 拒绝采样: 生成 [0, C_{K-1}) 随机数
   → 在存活 Token 中二分查找/线性扫描命中
```

### §3.6 gllm VmInstr 扩展需求

当前 VmInstr 只有 `Argmax` 和 `TemperatureScale`, 需要:

| 需要 | 建议方式 | 说明 |
|------|---------|------|
| Warp Shuffle | 新增 `WarpShuffle { op, src, dst, lane }` | `shfl.sync.bfly` 等 |
| WarpReduce | 已有 `WarpReduce { op, src, dst, width }` | ✅ 可直接用于 Softmax |
| BlockScan | 新增 `BlockScanIncl { dst, src, op, width }` | Top-P 前缀和 |
| Predicate Compare | 已有 `VecCmp` | ✅ 可用于阈值过滤 |
| Random Number | 新增 `RandUniform { dst, low, high }` | 映射 `curand` |

## §4 参考

- NVIDIA PTX ISA 8.x: `cp.async`, `cp.async.bulk.tensor`, `mma.sync`, `wgmma`
- NVIDIA CUTLASS: Double Buffering pipeline, TMA descriptor 配置
- FlashInfer: Sorting-Free Sampling JIT 实现
- vLLM: Parallel Top-P/Top-K fused kernel
- CUDA Driver API: `CUtensorMap`, `cuTensorMapEncodeIm2Col`
- SPEC/17-DEVICE-CODEGEN.md: GPU codegen 实现规范
- SPEC/15-GPU-HOST-GLUE.md: GPU Mega-Kernel Host 胶水
