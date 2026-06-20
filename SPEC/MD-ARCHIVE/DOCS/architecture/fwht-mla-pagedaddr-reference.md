# FWHT / MLA / PagedAttention 寻址技术参考

> **用途**: 本文档为 gllm JIT codegen 提供 TurboQuant FWHT 蝴蝶网络向量化、
> DeepSeek MLA 矩阵分解与吸收、PagedAttention GPU 端物理寻址的底层实现细节。
> 实现规范见 `SPEC/23-QUANT-CODEGEN-ALGO.md`、`SPEC/02-ARCHITECTURE.md §11`、`SPEC/11-MODELS.md`。

## §1 FWHT (Fast Walsh-Hadamard Transform) JIT 实现

### §1.1 算子特征

TurboQuant 引入 FWHT 作为量化激活值的"旋转"前处理，平滑离群值（Outliers）。
属于 **访存受限 + Log 级计算** 算子。3 个插入点（Softmax 后、SwiGLU 后、RoPE K 写入前）
均内联在 Mega-Kernel Epilogue 中，数据全寄存器驻留，零额外 Global Memory 读写。

**复杂度**: d=4096 时，12 级蝴蝶操作（`log₂(4096) = 12`），每级 add/sub 配对。
远低于 GEMM 的 O(d²)，可完全展开（Unroll）。

### §1.2 SIMD 向量化: 两阶段映射

12 级蝴蝶操作在 SIMD (AVX-512) 下分为两个阶段:

| 阶段 | 级别 | 蝴蝶跨度 (Stride) | 特征 |
|------|------|-------------------|------|
| **阶段 1: 寄存器内蝴蝶** | Stage 1-4 | 1, 2, 4, 8 | 单 zmm 内元素配对，需 vpermilps/vpermps 洗牌 |
| **阶段 2: 寄存器间蝴蝶** | Stage 5-12 | 16, 32, ..., 2048 | 寄存器对齐操作，纯 vaddps/vsubps，无需洗牌 |

### §1.3 寄存器分配策略 (AVX-512)

单个 zmm 容纳 16 个 FP32 元素。d=4096 需 256 个虚拟寄存器。
采用 **分块交错 (Blocked Interleaving)** 策略:
每次对 16 个 zmm 寄存器（处理 256 个元素）进行局部 FWHT，再进行跨块级联。

### §1.4 阶段 1: 寄存器内蝴蝶 (Stage 1-4)

针对单个 zmm0:

```
Stage 1 (Stride=1):
  vpermilps zmm1, zmm0, 0xB1    // 交换相邻元素 [1,0,3,2,...]
  vaddps    zmm2, zmm0, zmm1    // A + B
  vsubps    zmm3, zmm0, zmm1    // A - B
  vblendps  zmm0, zmm2, zmm3, 0xAA  // 掩码重拼

Stage 2 (Stride=2):
  vpermilps zmm1, zmm0, 0x4E    // 交换 [2,3,0,1,...]
  vaddps ... vsubps ... vblendps ...

Stage 3 & 4:
  vpermps + 静态编译洗牌控制字 (Permutation Indices)
  完成组内 4 和 8 跨度的变换
```

### §1.5 阶段 2: 寄存器间蝴蝶 (Stage 5-12)

当 Stride ≥ 16 时，蝴蝶操作演变为纯寄存器对齐操作，**无需洗牌**:

```
Stage 5 (Stride=16) — 寄存器对 (zmm_X, zmm_Y):
  vaddps  zmm_temp0, zmm0, zmm1   // zmm0 前半, zmm1 后半
  vsubps  zmm_temp1, zmm0, zmm1
  vmovaps zmm0, zmm_temp0         // 更新原寄存器
  vmovaps zmm1, zmm_temp1
```

**JIT 核心优化**: 对固定 d=4096，12 级完全 Unroll 展开。
所有洗牌控制字在编译期计算并硬编码入指令，消除运行时标量分支和循环计数开销。

### §1.6 gllm VmInstr 映射

FWHT 作为 Epilogue 融合算子，不引入新 VmInstr。通过现有指令组合:
- `VecBinOp { Add }` / `VecBinOp { Sub }` — 蝴蝶加/减
- `VecCast { Shuffle }` — vpermilps/vpermps 洗牌
- `ConditionalSelect` — vblendps 掩码选择

或可考虑新增 `HadamardStage { log2_stride }` 专用 VmInstr，
在 ISA Lowering 时根据 stride 大小选择寄存器内/寄存器间路径。

### §1.7 AVX2 降级策略

AVX2 每个 ymm 容纳 8 个 FP32。d=4096 需 512 个虚拟寄存器，溢出更严重。
分块更细: 每次处理 8 个 ymm（64 元素），局部 FWHT 后跨块级联。
洗牌指令: `vpermilps` / `vperm2f128`（跨 128-bit lane）。

## §2 DeepSeek MLA 矩阵分解与吸收

### §2.1 维度定义

| 符号 | 含义 | DeepSeek V3 值 |
|------|------|---------------|
| d | 每 Head 维度 | 128 / 192 |
| n_h | Attention 头数 | 128 |
| d_c | KV 压缩低秩维度 (Latent Dimension) | 512 |
| d_pe | 解耦 RoPE 专属维度 | 64 |

### §2.2 KV 压缩: 低秩分解数学

输入向量 X (形状 `[M, d_model]`):

**KV 压缩 (降维)**:
```
c_KV = X · W_DKV    (W_DKV ∈ R^{d_model × d_c})
```
PagedKV-Cache 中存储的是低秩向量 `c_KV`（512 维），不是全量 K/V。

**解耦 RoPE Key 提取**:
```
k_pe = X · W_KR     (W_KR ∈ R^{d_model × d_pe})
```

**每 Token 缓存物理量**: `c_KV (d_c=512)` + `k_pe (d_pe=64)` = 576 维
vs 标准 MHA: `K (n_h × d = 128 × 128 = 16384)` + `V (16384)` = 32768 维
**压缩比**: 32768 / 576 ≈ **56.9×**

### §2.3 Matrix Absorption (矩阵吸收) 推导

**传统解压还原**:
```
K = c_KV · W_UK      (W_UK ∈ R^{d_c × (n_h × d)})
K_m = c_KV · W_UK_m  (W_UK_m ∈ R^{d_c × d})  — 第 m 个 Head
```

**标准 Attention Score**:
```
Score_m = Q_m · K_m^T
        = Q_m · (c_KV · W_UK_m)^T
        = Q_m · W_UK_m^T · c_KV^T
```

**Absorbed Pathway — 利用括号结合律改变矩阵乘顺序**:

1. **吸收至 Query**:
```
Q_absorbed_m = Q_m · W_UK_m^T    (维度: [1,d] × [d,d_c] → [1,d_c])
```

2. **直接点积 (无 K 参与)**:
```
Score_m = Q_absorbed_m · c_KV^T   (维度: [1,d_c] × [d_c,1] → [1,1])
```

**JIT 意义**: FlashAttention 内核外层 KV 循环加载的是压缩的 `c_KV`。
硬件寄存器中**根本不需要还原**物理维度 `n_h × d` 的 K 矩阵，
避免寄存器空间浪费。`W_UK_m^T` 乘法在 Attention 启动前一次性完成。

### §2.4 Un-absorbed vs Absorbed 分段路由

| 阶段 | 路径 | 理由 |
|------|------|------|
| 短文本 Prefill (≤ threshold) | **Un-absorbed**: `K = c_KV · W_UK` 后走标准 MHA | 算力饱和，省 3.36× FLOPs > 省带宽 |
| 长文本 Prefill + Decode | **Absorbed**: `Q_absorbed = Q · W_UK^T` 后点积 c_KV | 带宽瓶颈主导，压缩加载 |

threshold 由 `seq_len × n_h × d` vs SM 算力预算动态决定。
详见 `SPEC/DOCS/architecture/2026-frontier-reference.md §2`。

### §2.5 gllm 实现指引

| 组件 | 当前 | 需要做的 |
|------|------|---------|
| `model_adapter.rs` | MLA 架构支持 | 增加 `mla_config: Option<MlaConfig>` (d_c, d_pe, W_UK/W_DKV 权重名) |
| `build_attention_graph` | 标准 MHA / GQA | MLA 路径: Absorbed 模式下 `Q_proj` 后追加 `Gemm(Q, W_UK^T)` → 点积 `c_KV` |
| `PagedScheduler` | KV cache per head | MLA: cache 存储 `c_KV` 而非 `K/V`，page 大小按 `d_c + d_pe` 计 |
| `SPEC/11-MODELS.md` | DeepSeek V3 描述 | 补充 MLA 数学公式 + 分段路由策略 |

## §3 PagedAttention 物理页表 GPU 端寻址

### §3.1 内存布局对比

| 方案 | 布局 | 缺陷 |
|------|------|------|
| vLLM 传统: `block_table[num_seqs][max_pages_per_seq]` | 2D Tensor | 2D 填充碎片；max_pages 按 batch 最长序列对齐，空间浪费；难支持 prefix sharing |
| **gllm / 2026 极致设计** | **扁平 1D 物理页索引池 + 多维步长描述符** | 零填充；原生支持 KvPrefixIndex trie 路由 |

### §3.2 核心参数定义

| 参数 | 含义 | 典型值 |
|------|------|--------|
| P_size | 单个物理页包含的 Token 数量 (Block Size) | 16 或 32 |
| D_head | 单个 Head 维度 | 128 / 192 |
| n_h | KV Head 数 | 8 (GQA) / 128 (MHA) |
| Stride_page | 一个完整物理页在 KV-Cache 的字节跨度 | `P_size × n_h × D_head × sizeof(FP16)` |

### §3.3 寻址两步走

**第一步: 求取物理页 ID**

```
Logical_Page_Index = floor(token_idx / P_size)
Physical_Page_ID   = page_table[ seq_mapping_ptr[seq_id] + Logical_Page_Index ]
```

`seq_mapping_ptr` 是一维偏移数组，每条序列记录其在扁平页表中的起始偏移。
**彻底消除 2D 填充碎片**。

**第二步: 计算硬件绝对地址**

```
Page_Offset  = (token_idx mod P_size) × D_head × sizeof(FP16)
Head_Offset  = head_id × P_size × D_head × sizeof(FP16)

Addr_final = KV_Cache_Base
           + (Physical_Page_ID × Stride_page)
           + Head_Offset
           + Page_Offset
```

### §3.4 JIT Codegen 映射

```
[Logical Token Index]
    │
    ▼ 1. 查阅一维紧凑页表
[Physical Page ID]
    │
    ▼ 2. 硬件 MAD (IMAD) 指令
[Final GPU Address] ◄── KV_Base + (PageID × Stride) + Head_Off + Page_Off
```

**关键**: JIT 编译器在 Codegen 时利用 GPU 的显式乘加指令 (MAD / IMAD)，
在单个 PTX 循环内完成物理指针的瞬时计算，
**完全规避传统 2D 数组的两次 Global Memory 指针间接寻址开销**。

### §3.5 VmInstr::PageTableAddr 发射策略

x86_64:
```
mov rax, [rbp + page_table_ptr]          ; 加载页表基址
mov ecx, [rbp + seq_mapping_ptr + seq_id*4]  ; 该序列页表偏移
mov edx, token_idx
shr edx, log2(P_size)                    ; Logical_Page_Index = token_idx / P_size
mov eax, [rax + rcx + rdx*4]             ; Physical_Page_ID = page_table[offset + LPI]
imul rax, Stride_page                    ; PageID × Stride
add rax, [rbp + kv_pool_base]            ; + KV_Base
; 后续加 Head_Offset + Page_Offset
```

GPU PTX:
```
ld.global.u32  %r_seq_off, [%seq_mapping + %seq_id * 4];
shr.u32        %r_lpi, %token_idx, LOG2_PAGE_SIZE;      // Logical Page Index
mad.wide.u32   %r_ppid, %r_lpi, 4, %r_seq_off;          // byte offset in page_table
ld.global.u32  %r_page_id, [%page_table + %r_ppid];     // Physical Page ID
mad.lo.u32     %r_addr, %r_page_id, STRIDE_PAGE, %kv_base; // Base + PageID*Stride
// 后续加 Head_Offset + Page_Offset via IMAD
```

### §3.6 gllm 实现指引

| 组件 | 当前 | 需要做的 |
|------|------|---------|
| `VmInstr::PageTableAddr` | 已有 | 确认实现遵循 1D 扁平寻址公式 |
| `BatchContext` (SPEC/20) | `page_table_flat_ptr` + `page_table_offset/len` per seq | 已是 1D 扁平设计 |
| `KvPrefixIndex` | 单机 trie + CoW 页映射 | GPU 端 page_table 物理页 ID 已包含 CoW 映射 |
| GPU codegen | PTX 路径 | 确认使用 IMAD 而非两次间接寻址 |

## §4 参考

- Walsh-Hadamard Transform: 快速蝴蝶算法，O(d log d)
- SpinQuant (ICLR 2025): 在线旋转平滑量化离群值
- RaBitQ (SIGMOD 2024): 无偏内积估计修正因子
- DeepSeek V3 Technical Report: MLA 低秩 KV 压缩 + 矩阵吸收
- vLLM V1 PR #31473: TOKENSPEED_MLA Un-absorbed 短 prefill 路由
- SPEC/02-ARCHITECTURE.md §11: TurboQuant 运行时优化白嫖全景
- SPEC/11-MODELS.md: DeepSeek V3/R1 架构描述
- SPEC/20-BATCH-CONCURRENT-INFERENCE.md: BatchContext flat memory 布局
- SPEC/DOCS/architecture/2026-frontier-reference.md §2: MLA 分段路由策略
