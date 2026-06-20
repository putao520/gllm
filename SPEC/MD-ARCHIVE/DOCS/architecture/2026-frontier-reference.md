# 2026 前沿推理系统技术洞察参考

> **用途**: 本文档记录 2026 年工业界前沿推理系统（vLLM V1 / FlashInfer / LMCache / DeepGEMM）
> 的关键技术洞察，识别对 gllm 架构演进有价值的差异点与改进方向。
> gllm 的 Mega-Kernel 架构在多数维度已超越这些工业方案，
> 但本文记录的 3 项洞察是 gllm 当前 SPEC 尚未覆盖的盲区。

## §0 差异矩阵：gllm vs 2026 工业界

| 维度 | vLLM V1 / SGLang / FlashInfer | gllm | 评估 |
|------|------|------|------|
| 调度开销 | V1 Diff 增量 RPC（微秒级） | Mega-Kernel 单次 CALL（零 CPU 参与） | **gllm 远超** |
| 三级存储编排 | HMA (HBM → CXL → Network) | StorageTier (GpuHbm/CpuDram/Nvme) + DMA | **gllm 已覆盖** |
| KV Cache 管理 | PagedAttention + Radix Tree | PagedAttention + KvPrefixIndex trie | **gllm 已覆盖** |
| FP4 推理 | FP4/MXFP4 FlashAttention-3 | NVFP4 dual-path JIT codegen | **gllm 已覆盖** |
| Prefill/Decode 分离 | 编译时两套内核 | compile_model_graphs() 两套融合图 | **gllm 已覆盖** |
| 页级压缩 | HMA 透明压缩 | Lz4/BitPackRle + PageMigrationActor | **gllm 已覆盖** |
| 跨 SM 流水线 | Cluster-Specialized CGA | 单 SM Double Buffering | **gllm 未覆盖 — 洞察 1** |
| MLA 分段路由 | 短文本 Un-absorbed / 长文本 Absorbed | 统一 Absorbed 路径 | **gllm 未覆盖 — 洞察 2** |
| 跨节点 KV 共享 | LMCache + CacheBlend RDMA | 单机 KvPrefixIndex trie | **gllm 未覆盖 — 洞察 3** |

## §1 洞察 1: Cluster-Specialized CGA 流水线

### §1.1 背景: 从 Warp-Specialized 到 Cluster-Specialized

FlashAttention-3 (2025) 的 Warp-Specialized 模式:
- 单个 Thread Block 内划分 Producer Warp（TMA 搬运）和 Consumer Warp（WGMMA 计算）
- Producer 和 Consumer 共享同一 SM 的 Shared Memory 和发射总线

2026 FlashAttention-3 极限形态的演进:
- 利用 SM90+/SM100 的 **CGA (Cooperative Thread Array) / Thread Block Cluster** 机制
- 物理隔离: 指定某些物理 SM 专门做 TMA 搬运（Producer SM），
  另外几个物理 SM 专门做 WGMMA 计算（Consumer SM）
- **跨 SM 直写**: Producer SM 通过 TMA 从 HBM 读取数据后，
  通过片上网络（NoC）直接写入相邻 Consumer SM 的 Shared Memory
- Consumer SM 的发射总线完全不占用 → 计算利用率提升至 85%+

### §1.2 FP4/MXFP4 原生 Tile 调优

4-bit 精度对 Tile Size 的影响:
- 4-bit 占用显存极小，Blackwell 上极限 Tile Size 从 128×128 扩大到 **256×256**
- MXFP4 的 Block Quantization 共享比例因子在 TMA 搬运时硬件级自动完成 Scale 乘法
- WGMMA 在寄存器内无缝执行高密度 4-bit 矩阵乘，吞吐达 PFLOPs 级别

### §1.3 对 gllm 的意义

**当前 gllm GPU codegen**: 单 SM 内的 Double Buffering（Ping-Pong stage 追踪）
```
Stage 0 (Ping): [MMA 计算消费] ← 依赖上一轮完成信号
Stage 1 (Pong): [TMA/cp.async 异步加载] ← 填入本轮搬运队列
```

**演进方向**: gllm 的 GPU JIT codegen 可在 SM90+/SM100 Profile 下生成
Cluster-Specialized 内核:
- `DeviceProfile` 检测 SM 版本 → SM100 时选择 CGA 模式
- Producer Kernel: TMA 加载 + NoC 直写到 Consumer SMEM
- Consumer Kernel: 纯 WGMMA 计算，零加载开销
- 这需要在 GPU codegen 中新增 Thread Block Cluster 描述符和跨 SM SMEM 寻址

**SPEC 影响**:
- `SPEC/17-DEVICE-CODEGEN.md` — GPU VmInstr 可能需要 Cluster 级指令扩展
- `SPEC/16-DEVICE-FUSION.md` — SM100 Profile 融合规则需考虑 CGA 模式
- `SPEC/DOCS/architecture/gpu-ptx-codegen-reference.md` — 补充 CGA PTX 指令参考

## §2 洞察 2: MLA Prefill 短文本 Un-absorbed 路由

### §2.1 背景: Matrix Absorption 的 FLOPs 惩罚

DeepSeek MLA (Multi-head Latent Attention) 的核心思想:
- 将 KV 压缩到低秩隐空间: `KV_compressed = KV_original @ W_dkv`（down-projection）
- 注意力计算在压缩空间进行: `Attn = Softmax(Q @ W_uq @ W_dkv^T @ KV_compressed^T)`
- **Matrix Absorption**: 将 `W_uq @ W_dkv^T` 预合并为 `W_absorb`，
  避免运行时两次矩阵乘 → 减少显存带宽

**关键问题**: Absorbed 路径虽然节省带宽，但改变了计算顺序，**FLOPs 是 Un-absorbed 的 3.36 倍**。

### §2.2 2026 反转: 算力饱和时的分段路由

在 Blackwell/H100 等算力饱和平台上（特别是 FP4 进一步放大算力余量），
最优策略反转为:

| 阶段 | 路径 | 理由 |
|------|------|------|
| **短文本 Prefill** (≤ threshold) | **Un-absorbed dense MHA** | 算力充足，省 3.36× FLOPs > 省带宽；延迟暴降 |
| **长文本 Prefill** | Absorbed 矩阵吸收 | 长序列带宽瓶颈主导 |
| **Decode** | Absorbed 矩阵吸收 | decode 始终带宽受限 |

threshold 由 `seq_len × num_heads × head_dim` vs SM 算力预算动态决定。

### §2.3 对 gllm 的意义

**当前 gllm**: DeepSeek MLA 走统一的 Absorbed 路径
（`SPEC/11-MODELS.md` DeepSeek V3/R1 架构描述）

**演进方向**: 在 `StrategyArbiter`（`SPEC/12-STRATEGY-ARBITER.md`）中增加
MLA 分段路由决策:
- Prefill 短序列 → `compile_model_graphs()` 编译时选择 Un-absorbed 图
- Prefill 长序列 + Decode → 编译时选择 Absorbed 图
- 两套图在 `ModelJitCache` 中共存，运行时由 `seq_len` 阈值切换

**注意**: 这不违反 JIT 缓存协议 — 两套图在模型加载时都已编译，
运行时只是选择调用哪个已编译入口，类似 Prefill/Decode 双图模式。

**SPEC 影响**:
- `SPEC/11-MODELS.md` — DeepSeek MLA 架构描述需补充分段路由策略
- `SPEC/12-STRATEGY-ARBITER.md` — StrategyBias 需增加 MLA 路径维度
- `SPEC/01-JIT-PIPELINE.md` — 图构建器需支持 Un-absorbed MLA 图变体

## §3 洞察 3: LMCache 跨节点 CacheBlend

### §3.1 背景: 多节点 KV Cache 复用

工业场景（多 Agent / 大规模多轮对话）:
- 长文本前缀在 Node A 上计算过 KV Cache
- 请求路由到 Node B 时，需要重新 Prefill 同一前缀 → 巨大算力浪费

LMCache + CacheBlend 解决方案:
- Node A 的 KV Cache 物理页通过 RDMA/高速以太网传输到 Node B
- Node B 将收到的 KV 页"缝合（Blend）"到自己的 PagedAttention 物理页中
- Node B 只需 Prefill **差异部分**（新 prompt），省去 90%+ 重复 Prefill 算力

### §3.2 技术要点

**CacheBlend 缝合协议**:
- KV Cache 按 page 粒度传输（与 gllm 的 PagedAttention page 天然对齐）
- 每页附带: `[page_id, layer, head, position_range, checksum]`
- 接收端验证 checksum 后直接映射到物理页表（零拷贝）
- 跨节点的 page 引用计数管理（引用归零时回收）

**RDMA 传输优化**:
- KV Cache page 大小对齐 RDMA 注册内存区域（通常 4KB-64KB）
- 批量注册 + 零拷贝直传，避免 CPU 参与
- 传输延迟: InfiniBand ~1-2μs/page，RoCE ~5-10μs/page

### §3.3 vLLM V1 架构重构

作为对比参考，vLLM V1 的核心变化:

**V0 痛点**: 调度器和 Worker 挤在同一进程，臃肿难扩展
**V1 突破**: Worker 端分布式状态缓存 + CPU 调度器增量 Diff RPC

| 维度 | V0 | V1 |
|------|-----|-----|
| CPU 调度开销 | 2-3 ms/step | 微秒级 |
| 状态同步 | 全量复制 | 增量 Diff |
| Worker 扩展 | 单进程 | 分布式多 Worker |
| KV Cache 管理 | 本地 PagedAttention | + LMCache 跨节点 |

**gllm 的优势**: Mega-Kernel 架构天然消除了 V0 的 CPU 调度瓶颈。
但跨节点 KV 共享是 gllm 当前未覆盖的维度。

### §3.4 对 gllm 的意义

**当前 gllm**: 单机 KvPrefixIndex trie（prefix 去重 + page 共享）

**演进方向**: 
- `KvPrefixIndex` 扩展为分布式版本: 本地 trie + 远程节点 prefix hash 索引
- `PageMigrationActor` 增加 RDMA 传输模式（当前只有 GPU↔CPU DMA）
- `SPEC/21-WEIGHT-PAGING.md` 的权重分页机制可与 KV Cache 分页共享 RDMA 基础设施

**SPEC 影响**:
- 新增 `SPEC/24-DISTRIBUTED-KV-CACHE.md`（或类似）— 跨节点 KV Cache 共享设计
- `SPEC/03-DATA-STRUCTURE.md` — KvPrefixIndex 分布式扩展
- `SPEC/07-OBSERVABILITY.md` — 跨节点遥测（page 传输延迟、命中率）

## §4 参考

- vLLM V1 Core Engine: `vllm/v1/` — Diff-based incremental scheduler
- vLLM PR #31473: TOKENSPEED_MLA — MLA Un-absorbed short-prefill optimization
- vLLM PR #41228: HMA (Hybrid Memory Allocator) — Three-tier KV Cache management
- LMCache 2026 Q2 Roadmap: CacheBlend cross-node KV Cache sharing via RDMA
- FlashAttention-3: Cluster-Specialized CGA pipeline for SM90+/SM100
- DeepGEMM: JIT-compiled MLA low-rank recombination + MoE token filter fusion
- NVIDIA PTX ISA: Thread Block Cluster (CGA) — `cluster.arrive` / `cluster.wait`
- SPEC/17-DEVICE-CODEGEN.md: GPU VmInstr 当前覆盖
- SPEC/16-DEVICE-FUSION.md: 硬件特化融合规则
- SPEC/19-KV-CACHE-OPTIMIZATION.md: gllm 单机 KV Cache 优化
- SPEC/22-PAGE-COMPRESSION.md: gllm 页级压缩与三级换入换出
