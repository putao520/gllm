# GGUF 文件格式规范 领域资料库

> 来源：ggml.ai GGUF spec（github.com/ggerganov/ggml/blob/master/docs/gguf.md，GGUF v3）+ llama.cpp ggml.h 实现 + 项目源码 `src/loader/gguf/`
> 建库触发：BCE-20260708-GGUF-QUANT 系列（loader 解析 GGUF 需格式规范地基；MoE 命名/dims 反转/row padding/dtype 枚举边界是反复踩坑点）
> 最后验证：2026-07-08

---

## 核心机制（与 loader 解析强相关）

### 文件总体结构（GGUF v3, little-endian）

```
gguf_file_t:
  gguf_header_t       header              // magic + version + counts + metadata_kv[]
  gguf_tensor_info_t  tensor_infos[N]     // N = header.tensor_count
  uint8_t             _padding[]          // pad 到 ALIGNMENT 倍数（header+tensor_infos 末尾）
  uint8_t             tensor_data[]       // 实际权重数据，每个 tensor 起始是 ALIGNMENT 倍数
```

### Header（`src/loader/gguf/reader.rs:63-86`）

| 字段 | 类型 | 值/说明 |
|------|------|---------|
| magic | u32 LE | `0x46554747`（字节序 `0x47 0x47 0x55 0x46` = "GGUF"）|
| version | u32 | `3`（v3 引入 big-endian；v2 把 count 从 u32 改 u64；v1 初始）|
| tensor_count | u64 | tensor 数量 |
| metadata_kv_count | u64 | 元数据 KV 对数 |
| metadata_kv[] | `{key: string, value_type: u32, value}` | key 是 ASCII hierarchical `lower_snake_case.dot.separated`，≤65535 字节 |

项目常量：`GGUF_MAGIC = 0x4655_4747`、`GGUF_SUPPORTED_VERSION = 3`（reader.rs:5-8）。版本≠3 直接 `UnsupportedVersion`。

### GgufValueType（13 种，u32 枚举 0-12）

`src/loader/gguf/types.rs:15-29`：UINT8=0 / INT8=1 / UINT16=2 / INT16=3 / UINT32=4 / INT32=5 / FLOAT32=6 / BOOL=7 / STRING=8 / ARRAY=9 / UINT64=10 / INT64=11 / FLOAT64=12。

- STRING：`len(u64) + bytes[len]`（UTF-8，**非 null-terminated**）
- ARRAY：`item_type(u32) + len(u64) + items[len]`（len 是元素数非字节数，可嵌套）
- BOOL：1 字节，0=false / 1=true，其他值非法

### TensorInfo（`src/loader/gguf/reader.rs:89-100, 159-165`）

```
gguf_tensor_info_t:
  string    name              // ≤64 字节
  u32       n_dimensions      // 当前≤4
  u64[n]    dimensions        // ★ GGUF 内核序：dimensions[0] = ne[0] = innermost（行宽）
  u32       type              // GgmlDType 枚举
  u64       offset            // ★ 相对 tensor_data，非文件起始；必须是 ALIGNMENT 倍数
```

绝对偏移 = `data_offset + rel_offset`（reader.rs:132），`data_offset` 是 header+tensor_infos+padding 后的 ALIGNMENT 对齐位置。

### ALIGNMENT 与 padding（`src/loader/gguf/reader.rs:102-117, 692-700`）

- `general.alignment` 元数据，**默认 32**（reader.rs:106 `unwrap_or(32)`），必须是 8 的倍数
- `align_offset(o) = o + (ALIGNMENT - (o % ALIGNMENT)) % ALIGNMENT`
- 两处 padding：① header+tensor_infos 末尾 pad 到 ALIGNMENT（定 data_offset）② tensor 之间空隙 pad 到 ALIGNMENT（tensor offset 必须是 ALIGNMENT 倍数）

### GgmlDType 枚举（★ 项目 36 variants，非官方 spec 的 30）

**官方 spec md（gguf.md）只列到 IQ1_M=29 + GGML_TYPE_COUNT**——文档滞后。实际 llama.cpp `ggml.h` 后续添加了 BF16/TQ1_0/TQ2_0/MXFP4 等，项目又自定义了 AWQ4/GPTQ4/SQUEEZE/NVFP4。完整表（`src/loader/gguf/types.rs:57-94`）：

| 值 | 变体 | block_size | block_bytes | 来源 |
|----|------|-----------|-------------|------|
| 0 | F32 | 1 | 4 | spec |
| 1 | F16 | 1 | 2 | spec |
| 2 | Q4_0 | 32 | 18 | spec |
| 3 | Q4_1 | 32 | 20 | spec |
| 4,5 | (removed Q4_2/Q4_3) | — | — | hole，TryFrom 返回 Err |
| 6 | Q5_0 | 32 | 22 | spec |
| 7 | Q5_1 | 32 | 24 | spec |
| 8 | Q8_0 | 32 | 34 | spec |
| 9 | Q8_1 | 32 | 36 | spec |
| 10-15 | Q2_K..Q8_K | 256 (QK_K) | 84/110/144/176/210/292 | spec |
| 16-23 | IQ2_XXS/IQ2_XS/IQ3_XXS/IQ1_S/IQ4_NL/IQ3_S/IQ2_S/IQ4_XS | 256 (IQ4_NL=32) | 各异 | spec |
| 24-28 | I8/I16/I32/I64/F64 | 1 | 1/2/4/8/8 | spec |
| 29 | IQ1_M | 256 | 56 | spec |
| 30 | BF16 | 1 | 2 | llama.cpp ext（spec md 未列）|
| 31-33 | hole | — | — | TryFrom Err |
| 34 | TQ1_0 | 256 | 54 | llama.cpp ext |
| 35 | TQ2_0 | 256 | 66 | llama.cpp ext |
| 36-38 | hole | — | — | TryFrom Err |
| 39 | MXFP4 | 32 | 17 | OCP MX FP4（spec md 未列）|
| 40-49 | hole | — | — | TryFrom Err |
| 50 | AWQ4 | 128 | 72 | ★ 项目自定义（非标准）|
| 51 | GPTQ4 | 128 | 72 | ★ 项目自定义（非标准）|
| 52 | SQUEEZE | 256 | 130 | ★ SqueezeLLM 3-bit（非标准）|
| 53 | NVFP4 | 64 | 36 | ★ NVIDIA FP4 E2M1+UE4M3（非标准）|

判定量：`GgmlDType::all().len() == 36`（types.rs 测试 `ggml_dtype_all_count_matches_variants`）。

### Row padding（量化 block 对齐，spec 未明说，来自 llama.cpp `ggml_row_size`）

量化张量的 **innermost 维（ne[0]）必须 pad 到 block_size 倍数**（`src/loader/gguf/types.rs:443-472` `tensor_nbytes`）：

```
ne0 = shape[0]  // GGUF 内核序 innermost
blocks_per_row = ceil(ne0 / block_size)
row_bytes = blocks_per_row * block_bytes
total = row_bytes * shape[1] * shape[2] * ...  // 所有 outer 维相乘
```

例：Q4_0 shape=[33] → ceil(33/32)=2 blocks × 18 = 36 字节（非 33/32×18）。F32/BF16 等 block_size=1 不 pad。

### Standardized tensor names（spec §Standardized tensor names）

- Base: `token_embd` / `pos_embd` / `output_norm` / `output` （+ `.weight`/`.bias`）
- Block: `blk.N.{attn_norm,attn_norm_2,attn_qkv,attn_q,attn_k,attn_v,attn_output,ffn_norm,ffn_up,ffn_gate,ffn_down}.weight/.bias`
- MoE: `blk.N.ffn_gate_inp`（routing）+ `blk.N.ffn_gate_exp` / `blk.N.ffn_down_exp` / `blk.N.ffn_up_exp`（per-expert，spec 推荐全称 `exp`）
- SSM (Mamba/RWKV): `blk.N.{ssm_in,ssm_conv1d,ssm_x,ssm_a,ssm_d,ssm_dt,ssm_out}`

### 关键元数据键

- **Required**: `general.architecture`（string，`[a-z0-9]+`，如 `llama`/`qwen2`/`gemma2`）、`general.quantization_version`（u32，量化模型必须）、`general.alignment`（u32，默认 32）
- LLM（`{arch}.` 前缀）：`context_length` / `embedding_length` / `block_count` / `feed_forward_length` / `expert_count` / `expert_used_count` / `attention.head_count` / `attention.head_count_kv` / `attention.layer_norm_rms_epsilon` / `rope.dimension_count` / `rope.freq_base` / `rope.scaling.{type,factor,original_context_length}`
- Tokenizer: `tokenizer.ggml.{model,tokens,scores,token_type,merges,added_tokens}` / `tokenizer.ggml.{bos,eos,unknown,separator,padding}_token_id` / `tokenizer.huggingface.json` / `tokenizer.chat_template`（Jinja）

---

## AI 易误判点（★ 核心价值，堵幻觉）

### 1. ❌ dims 直接当 HF shape 用 → ✅ GGUF 内核序，必须反转

- ❌ 误判：GGUF `dimensions=[8,4]` 当作 HF shape `[8,4]`（outermost=8）
- ✅ 正解：GGUF `dimensions[0]=ne[0]=innermost`（行宽）。reader.rs:593-598 明确注释并用 `.rev()` 反转给外部 `TensorMeta`。GGUF `[8,4]` → 外部 HF 序 `[4,8]`
- **证据**：reader.rs 测试 `iter_tensors_yields_reversed_shapes`（L2513）`assert_eq!(metas[0].shape, vec![4, 8]); // reversed from GGUF [8, 4]`；3D 测试（L6857）`[8,4,2] → [2,4,8]`
- **陷阱**：1D shape 不反转（reader.rs:5742 `tensor_provider_1d_shape_not_reversed`）

### 2. ❌ tensor 数据 `elem_bytes × ne0` 连续 → ✅ row padding 后每行非连续

- ❌ 误判：量化张量行宽 = `ne0 * elem_bytes`（按元素算）
- ✅ 正解：量化张量按 **block** 组织，行宽 = `ceil(ne0 / block_size) * block_bytes`。ne0 非 block_size 倍数时每行尾部有 padding 字节，dequant 必须按 block 跳进，不能按元素连续读
- **证据**：types.rs:451-459 注释 `GGUF pads the innermost dimension (ne[0]) to the block boundary, matching llama.cpp's ggml_row_size(type, ne[0])`
- **例**：Q4_0 shape=[33, 2]：行宽=ceil(33/32)×18=36（非 33/32×18≈18.6），总=36×2=72

### 3. ❌ MoE tensor 命名 `experts.{E}.gate_proj` → ✅ GGUF 用 `ffn_gate_ex{E}`（缩写）

- ❌ 误判：GGUF MoE 用 SafeTensors 风格 `blk.{L}.experts.{E}.gate_proj.weight`
- ✅ 正解：GGUF MoE 用 `blk.{L}.ffn_{gate,up,down}_ex{E}.weight`（**缩写 `ex` 非 `exp`**，尽管 spec 文档推荐 `exp`）。项目 name_map.rs:147-167 匹配 `ffn_gate_ex`/`ffn_up_ex`/`ffn_down_ex` 前缀解析 layer+expert+proj
- **双源适配**：name_map.rs Pass 1.5 同时处理 SafeTensors `experts.{E}.{proj}.weight` 和 GGUF `ffn_{proj}_ex{E}.weight`，归一化到 canonical `L{layer}.expert.{expert}.{gate_proj|up_proj|down_proj}`

### 4. ❌ GgmlDType 与 QuantType 1:1 → ✅ 部分无映射，MXFP4 仅 block_size=32

- ❌ 误判：每个 GgmlDType 对应一个 QuantType，反之亦然
- ✅ 正解（adapter.rs:94-191 双向映射）：
  - `QuantType::Bf16/F16/F32/Fp8E4M3/Fp8E5M2 → None`（native float / FP8，GGUF 不走 quant 路径，尽管 GGUF 有原生 F32/F16/BF16 dtype）
  - `QuantType::Mxfp4{block_size:32} → MXFP4`；`Mxfp4{其他 block_size} → None`（非标准 block 无 GgmlDType）
  - `GgmlDType::F32/F16/BF16/F64/I8/I16/I32/I64 → None`（非量化，ggml_dtype_to_quant_type）
  - AWQ4/GPTQ4/SQUEEZE/NVFP4（项目自定义 50-53）有对应 QuantType
- **关键**：QuantType 是 kernel dispatch 维度，GgmlDType 是文件存储维度，两者不重合

### 5. ❌ tensor offset 是文件绝对偏移 → ✅ 相对 tensor_data

- ❌ 误判：`TensorInfo.offset` 直接当文件偏移用
- ✅ 正解：spec 明确 `offset` 相对 `tensor_data`（为方便 writer）。绝对偏移 = `data_offset + offset`（reader.rs:132）。`data_offset` = `align_up(header_size + tensor_infos_size, ALIGNMENT)`

### 6. ❌ alignment 默认 8/16/64 → ✅ 默认 32

- ❌ 误判：按 CPU cache line（64）或自然对齐（8/16）猜默认
- ✅ 正解：spec 明确 `general.alignment` 缺省 **32**，必须是 8 的倍数。reader.rs:106 `unwrap_or(32)`。pad 计算 `(pos + 31) & !31`（reader.rs:800）

### 7. ❌ spec md 列的 dtype 就是全部 → ✅ spec md 滞后于 ggml.h 实现

- ❌ 误判：官方 spec md 列到 IQ1_M=29 + GGML_TYPE_COUNT，以为只有 30 个 dtype
- ✅ 正解：spec md 文档滞后。BF16(30)/TQ1_0(34)/TQ2_0(35)/MXFP4(39) 是 llama.cpp `ggml.h` 后续添加，spec md 未及时更新。项目还自定义 AWQ4(50)/GPTQ4(51)/SQUEEZE(52)/NVFP4(53) 用于 vendor 量化。共 36 variants
- **查阅姿势**：dtype 完整表以 llama.cpp `ggml.h` 源码为准，spec md 仅作架构参考

### 8. ❌ big-endian 自动检测 → ✅ v3 支持但无法识别，默认 little-endian

- ❌ 误判：GGUF v3 能从 magic 判断 endianness
- ✅ 正解：spec 原文 "at the time of writing, there is no way to determine if a model is big-endian"。默认 little-endian。项目 reader.rs 直接按 LE 读（`read_u32` LE），不支持 BE

### 9. ❌ `general.file_type` 决定解析 → ✅ 仅描述性，不影响 tensor 解析

- ❌ 误判：用 `general.file_type`（MOSTLY_F16=1 等）决定如何解析 tensor
- ✅ 正解：`file_type` 是"大部分 tensor 的类型"描述（leaderboard/分类用），可从 tensor types 推断。tensor 实际 dtype 由 `tensor_info.type` 决定，每个 tensor 独立

### 10. ❌ reranker 的 `output.weight` = lm_head → ✅ 是 classifier head

- ❌ 误判：GGUF 把 reranker `score.weight` 改名 `output.weight`，按 lm_head 处理走生成路径
- ✅ 正解：name_map.rs:226-245 ARCH-RERANKER-CLASSIFY 逻辑——reranker 的 `output` 是分类头 `[num_labels, hidden]`（通常 `[2, hidden]`），非 `[vocab, hidden]`。仅当 lm_head 是**独立 tensor**（非 tied embed）时 remap `lm_head → classifier` 走分类路径

---

## 解决问题时参考

### Loader 解析 GGUF 时

1. **dims 反转**：读 `tensor_info.dimensions` 后 `.rev()` 给外部 HF 序（reader.rs:598）；1D 不反转
2. **offset 换算**：绝对 = `data_offset + rel_offset`；`data_offset` 由 `general.alignment`（默认 32）对齐
3. **size 计算**：用 `tensor_nbytes(dtype, shape)`（types.rs:443），含 row padding，**禁止** `elem_bytes × prod(shape)`
4. **dtype 映射**：`ggml_dtype_to_quant_type`（adapter.rs:138）做 kernel dispatch；native float/FP8 返回 None
5. **命名归一**：name_map.rs 同时处理 GGUF `blk.{L}.ffn_{proj}_ex{E}` 和 SafeTensors `experts.{E}.{proj}` 两种 MoE 命名

### Dequant / JIT codegen 时

1. **row padding**：按 block 步进（`block_size` 元素一块），ne0 尾部 padding 字节跳过；stride 用 `blocks_per_row × block_bytes` 非 `ne0 × elem_bytes`
2. **block 边界**：Q4_0/Q4_1/Q5/Q8/IQ4_NL/MXFP4 block_size=32；K-Quant/IQ(部分)/TQ block_size=256(QK_K)；AWQ4/GPTQ4=128；NVFP4=64
3. **dtype 传播**：blob 保留原始 dtype 字节，JIT 按实际 dtype 特化解码（宪法 ARCH-JIT-DATA-YIELDS）

### Tensor 命名映射时

1. **GGUF → canonical**：`blk.{L}.{attn_q,attn_k,attn_v,...}.weight` → `L{L}.{q,k,v,...}_proj`；`token_embd` → `embed`；`output` → `lm_head`（reranker 独立时 → `classifier`）
2. **MoE 双源**：name_map.rs Pass 1.5 归一化两种命名到 `L{layer}.expert.{expert}.{gate_proj|up_proj|down_proj}`
3. **tied embed**：无独立 lm_head tensor 时，lm_head → embed 外部名（name_map.rs:206-211）

---

## 已知问题 / 边界

### 官方 spec 限制

- **dtype 文档滞后**：spec md 列到 IQ1_M=29，BF16/TQ1_0/TQ2_0/MXFP4 仅在 ggml.h 实现。查 dtype 以 ggml.h 源码为准
- **big-endian 不可识别**：v3 支持但无检测机制，默认 LE
- **n_dimensions ≤4**：spec "currently at most 4, but this may change"
- **tensor name ≤64 bytes**、**metadata key ≤65535 bytes**

### 项目扩展（非标准）

- **AWQ4=50/GPTQ4=51**：group_size=128，72 bytes/group（AWQ/GPTQ 4-bit 量化，非 GGUF 标准 dtype）
- **SQUEEZE=52**：SqueezeLLM 3-bit codebook，256 元素 block，130 bytes
- **NVFP4=53**：NVIDIA FP4 E2M1 + UE4M3 sub-block scales，64 元素 block，36 bytes
- **MXFP4=39**：OCP Microscaling FP4，32 元素 block，17 bytes（仅 block_size=32 映射 QuantType::Mxfp4）

### 元数据 jinja chat template

- `tokenizer.chat_template`（string，Jinja 模板）—— 大模型对话格式，项目 tokenizer 解析时需处理 Jinja 渲染（非 GGUF 格式本身一部分，是约定）

### 版本兼容

- 项目仅支持 GGUF v3（`GGUF_SUPPORTED_VERSION = 3`）。v1/v2 会被 `UnsupportedVersion` 拒绝。v2→v3 主要差是 count 从 u32→u64 + big-endian 支持
