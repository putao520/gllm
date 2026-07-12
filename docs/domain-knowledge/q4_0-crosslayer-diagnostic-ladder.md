# Q4_0 跨层 bug 诊断阶梯（consult 产出，非最终根因）

> 来源：architect consult 产出（Q4_0 跨层乱码诊断）+ 项目代码分析
> 建库触发：Q4_0 layer0 对 / layer27 全错（cosine=0.005）跨层断点定位方法论沉淀
> 最后验证：2026-07-12

> 目的：用「每步二分、不回头猜」的阶梯定位 layer0 对 / layer27 全错（cosine=0.005）的跨层断点。
> 打破 12 次硬磕的方法 = 从最廉价、最确定的检查开始，每步缩小一半嫌疑空间。

## 嫌疑（三者都是「每迭代增量」机制，layer0 全对未洗清任何一个）

- **A** `activation_alias` ping-pong swap（残差流跨层原地更新）
- **B** `weight_stride` 每层权重字节步幅（**唯一依赖权重量化 dtype**）
- **C** KV cache 跨层槽步进（prefill seq=5 下概率最低）

关键前提：iter0 的三个增量全部 ×0（offset 0 / 首次 swap 在 iter0 后 / KV 槽 ×0），
所以 **layer0-in-loop cosine=1.0 不排除 A/B/C 任何一个**，只把嫌疑收敛到跨层步进。

## 阶梯（严格按序，每步有 pass/fail 分支）

### 步骤 0 — 静态 weight_stride 核算（最廉价，无需跑，确定性）
手算一层 Q4_0 张量字节总和：q/k/v/o_proj + gate/up/down + norms，
每张量按**存储 dtype**（Q4_0: block=32 元素=18B = 2B fp16 scale + 16B packed 4-bit；norms 通常 F32/BF16）算。
对比实际 `_lc_weight_byte_cursor` / `weight_stride`（build_graph.inc.rs:2589 setup）。
- 不等 → **B 当场坐实**，去查是否某处用 compute-dtype(F32=4B/BF16=2B) 误算 Q4_0 stride。结束。
- 相等 → 进步骤 1。

### 【续轮更新】代码事实修正 — 方向 3 前提证伪
- `to_quant_precision()` 全库零匹配；报告里 lower_op.inc.rs:1499 的 KV 代码不存在（旧版/记忆串）。
- 真实 KV stride: abi_types.inc.rs:476 `num_kv_heads*head_dim*elem_bytes()`，
  `elem_bytes()`=`compute_dtype.size_bytes()`(abi_types:395)，size_bytes(adapter.rs:77)
  F32=>4/BF16=>2/U8=>1/PackedU8=>1，**无 18/32 截断，永不为 0**。
  → 「Q4_0 elem_bytes=0 → stride=0 → 层写同槽」物理不可能。**C 此形态证伪。**
- KV stride 与 KV buffer 分配同走一个全局 compute_dtype → 自洽，无 dual-dtype 错配。

### 【续轮更新】BF16 全 30 层 PASS（SmolLM2-135M safetensors）
activation_alias(A) / KV(C) 在 BF16 下正常。stride 链路三段(build_graph:342 /
pipeline:865 LoopBegin / lower LoopEnd)逻辑对 Q4_0 正确 → **B 经典"stride 误算"形态排除**。

### 步骤 0.5 — 【新增·最高优先】pack-stride vs loop-stride 对比（无需跑推理）
最可能根因：weight_helpers.rs:32 "GGUF row-padding for block quantization" — Q4_0 独有。
weight_physical_bytes 从逻辑 shape 推算，若打包器因 row-padding/SIMD 对齐实际写入字节数 ≠ 推算值，
则 loop-stride 与真实层间距漂移 δ：layer0 对(0×stride)、layer1+ 单调坍塌、Q4_0 特有(BF16 无块 padding)。
**这是 read-stride vs pack-stride 错配，符合全部证据。**
做法：dump 打包后 blob 相邻两层真实起始偏移之差，对比 _lc_weight_byte_cursor。
- 不等 → 根因坐实，δ = 每层漂移；查 weight_physical_bytes 漏算 padding/对齐。结束。
- 相等 → pack 清白，回退步骤 2。
命门验证：Qwen3-0.6B 维度多为 32 倍数(hidden=1024)，row-padding 可能为 0。
  须先核：有无张量内维非 32 倍数，或 norm/scale 存储 dtype 与推算假设不一致。若全 32 倍数则本假设不成立。

### 步骤 1 — F32/BF16 全 28 层控制跑（决定性二分）【已完成：BF16 PASS，见上】
跑非量化权重的全 28 层，看末层 cosine。
- **F32 跨层也崩** → bug 在激活路径 → 嫌疑翻转为 **A 第一**，Q4_0 是红鲱鱼。跳步骤 3。
- **F32 全对、只有 Q4_0 崩** → **B 几乎坐实**（A/C 不碰权重量化）。回步骤 0 找误算点。

### 步骤 2 — 逐层 cosine 曲线（diagnostic-layer-capture），看形状不只看 N
画完整 cosine-vs-layer 曲线：
- **奇偶交替 / 抖动** → **A**（ping-pong swap parity 签名）。
- **从 layer1 起单调坍塌** → **B 或 C**（读 base + k*错stride，误差随 k 增长）。
- **只某一特定层突然坍塌** → 那层具体权重/数据，另查。
预测 N=1；若 N>1 或非单调，回 architect 重开讨论。

### 步骤 3 — 首发散层 N 的残差输入三向对比（区分 A vs B/C）
dump「进入 layer N 前、compute 读到的残差输入 buffer」，同时对比：
(a) golden hs_N；(b) 已 capture 的 layer(N-1) 输出（那个 cosine=1.0 的）。
- **输入 ≠ capture 的 layer(N-1) 输出** → compute 与 capture 读的不是同一 buffer → **A 坐实**。
  （注意：capture 路径与 compute 路径若共享同一套 alias 逻辑，capture 可能被 A 污染却自洽，
   故必须同时对 golden 交叉验证，别省。）
- **输入 == layer(N-1) 输出（对）但 layer N 输出错** → layer N 读错权重/KV：
  - dump layer N 的 weight base 指针，算期望 `global_weight_base + N*weight_stride`，
    读该地址首个 Q4_0 block 的 fp16 scale，对比 GGUF 里 layer N q_proj 真实 scale → 不等则 **B**。
  - 对得上再查 KV 槽 base → **C**。

## 已排除（数值证明非根因，勿重查）
HeadRmsNorm groups_per_row(已修)、act_dt=F32(非违宪)、layer0 单算子(cosine=1.0)、
GQA dead code、QuantConcatSeq 方向、Assisted GEMV hi_act_off。
—— 注意这些验证的是「层内怎么读权重」，与「跨层读哪里」的步进 bug 正交。

## 模式根因（12 次同类错误的共性，非本实例根因）
缺少存储-dtype-aware 的「每张量字节布局」单一真相源；
weight_stride / KV stride / buffer 分配三处各自算字节，都可能落入
compute-dtype(F32) vs storage-dtype(Q4_0/BF16) 混淆陷阱（KV cache 已被咬过一次）。

建议（无论哪嫌疑胜出都做）：
1. 单一 `TensorByteLayout`（从 GGUF storage dtype → block size → bytes 派生），
   三处步进全从它取值。
2. graph build 期断言：`sum(per-layer tensor bytes) == weight_stride`
   且 `base + num_layers*weight_stride == total_weight_bytes`。
   此断言若早存在，B 会在构建期爆出，而非靠 28 层数值反查。
→ 建议新增 SPEC `REQ-JIT-LAYOUT-01`（详细定义交 spec_write）。

---

# 【最终实验设计】静态已穷尽，转数值诊断（executor 执行）

## 静态穷尽结论（勿再推公式）
- weight_stride 三段链路逻辑对；blob 用 graph offset(named_offsets)填、与 offset 自洽；
  Qwen3 全 32 倍数 → weight_physical_bytes == tensor_nbytes（无 padding 差、无洞无重叠）→ A(blob顺序)在 Qwen3 排除；
  KV stride dtype 从 K 张量派生逻辑对；weight_physical_bytes fallback `numel*4`(非32倍数)是真违宪 bug 但 Qwen3 不触发。
- 剩余活嫌疑：**A(activation_alias/swap)** 与 **Q4_0 特有 decode/累加 buffer 跨层复用**。
- 代码事实：compute_dtype 从 ModelGeometry 派生、用户可配 BF16（resolve.rs:27,89），非写死 F32。
  Q4_0 dequant scratch grep 返回空（削弱但不证否 decode-scratch 假设）。
- 真违宪 bug（独立修，勿等当前）：weight_physical_bytes 非32倍数 fallback `numel*4` 违反 GGUF `ceil(k/bs)*bb`；
  weight_physical_bytes(stride源) 与 tensor_nbytes(blob打包源) 双源不一致 = 模式根因具体体现。

## 🔴 Q3 循环论证漏洞（executor 必须堵，否则假性排除 A）
错误推理：「capture layer0 输出 == layer1 输入 → cos=1.0 证明 layer0 输出对 → layer1 输入对 → layer1 输出错=读错权重/kernel污染」。
为何循环：capture 在 swap **前**从 pong buffer 拷贝（测"layer0 写进输出buffer的内容"）；
layer1 的 **compute** 通过 swap 后 ptr 读输入。**若 swap 就是 bug，两者是不同 buffer** —— capture 显示 layer0 对，
而 layer1 compute 读到错位 buffer。故「capture 对→layer1 输入对」默认了 A 不是 bug，再用它排除 A = 循环。
**cos=1.0 只证明 layer0 把正确结果写进了它的输出 buffer，不证明 layer1 读到该 buffer。**
→ 严禁用 capture 推"layer1 输入对"。测 layer1 真实 compute 输入只能用「单层隔离 index=1 注入 golden hs_1」。

## 四步实验（严格按信息密度顺序）
1. **compute_dtype 双 run 确认**（最廉价，砍嫌疑数）：读 SmolLM2(BF16) 与 Qwen3(Q4_0) 两次 run 的 ModelGeometry.compute_dtype。
   - 两者都=F32 → swap 逻辑作用于相同 F32 buffer，BF16 全过 → **swap 没坏 → A 排除** → 唯一嫌疑=Q4_0 特有 decode/kernel。
   - BF16=native BF16(2B) 而 Q4_0=F32(4B) → swap 作用于不同尺寸 buffer → **A 仍存活**，BF16 过不能排除 Q4_0 的 A。
2. **逐层 capture cosine 曲线**（断点+形状指纹）：开 diagnostic-layer-capture，dump layer0..27 输出 vs golden。
   **仅用于**：(a) 断点确认（预期 layer0 对、layer1 起坍塌）；(b) 形状=指纹：**单调坍塌→weight-offset/decode 类；奇偶交替/抖动→swap parity(A)**。
   ❌ 不得用曲线推"输入对"。
3. **T2 量 δ**（运行时验证静态结论）：dump blob，比 layer1 q_proj 实际起始字节 vs `layer_blob_base_offset + 1*weight_stride + rel_off`。
   静态已推 Qwen3 下 δ 应=0；T2 验证运行时真成立（静态已被反转 3 次，兜底值得）。**δ≠0 是重大发现**。
4. **条件分支**：
   - **A 排除**（步1两者F32）→ 进 Q4_0 GEMM/GEMV 路径查「跨 layer 迭代复用且未重置的 buffer/累加器」（此时才用 debug_process 单步）。
   - **A 存活**（步1 dtype不同 或 步2 奇偶抖动）→ **单层隔离 index=1**：num_layers=1、权重偏移=`base+1*stride`、注入 golden hs_1。
     输出==golden hs_2 → layer1 权重+模板对，bug 只在多迭代 swap/plumbing → **A 坐实**（单层不 swap，干净剥离）；
     输出!=golden hs_2 → layer1 读错权重/decode，与 A 无关。

## Q2/T2 公式确认（已核对，正确）
- **Q2 golden 映射**：hs_0=embed输出(layer0输入)、hs_1=layer0输出、…、hs_28=layer27输出。
  capture 的 0-based layer N 输出（swap 前、含残差完整层输出）↔ golden hs_{N+1}。**正确**。
- **T2 偏移公式**：`layer_blob_base_offset + 1*weight_stride + rel_off(q_proj)`。**正确**。
  代码事实：layer_blob_base_offset=global_weight_bytes（embed/global 在层区之前）；weight_stride=单层模板 byte_cursor。
  **附带断言**：`base + 28*weight_stride ≤ blob.len()`（尾部有 final_norm/lm_head）且 `rel_off ∈ [0, weight_stride)`。
  若溢出或 rel_off≥stride → weight_stride 定义本身错。

## 三个未决问题（executor 先确认可行性）
1. **单层隔离 index=1 能否注入 golden hs_1 + 用 layer1 权重**？这是堵 Q3 漏洞、坐实/排除 A 的最强测试。
   若 harness 不支持，是本轮唯一需加 knob 处。先确认可行性。
2. **A 排除后 decode scratch grep 却返回空** → Q4_0 特有跨层差异究竟在哪？executor 须在 Q4_0 GEMM/GEMV 路径实际确认
   有无跨迭代复用未重置的 buffer/累加器；若也无，问题在更隐蔽处（如 Q4_0 权重指针某 per-layer 常量未随 li 更新）。
3. **BF16 与 Q4_0 是否共用同一 swap/ActivationSwap 代码路径**？若因 dtype 分叉成不同分支，则 BF16 过不能推 Q4_0，
   A 无法用 BF16 证据排除。executor 顺带确认 ActivationSwap 是否 dtype 无关的单一路径。

---

# 【裁决】step2 capture stride 算错 8192 倍 = 诊断工具 bug，非 GEMM 违宪

## 运行时现象（step2）
Qwen3-0.6B Q4_0 diagnostic-layer-capture：stride=167772160（预期 5×1024×4=20480，错 8192 倍），
scratchpad=6745.8MB，layer0 capture cosine=0.0293 *** DIVERGE ***。

## 1. capture stride bug 非 GEMM 违宪（三条独立证据）
real compute 的 GEMM 维度是数据派生、正确的：
- **build_graph.inc.rs:575-587**：q_n/q_k 从 weight_shapes(数据 TensorMeta)取，GEMM spec 用数据 dims（用户自证）。
- **layer0 非 capture cosine=1.0**：real GEMM 读错维度则 layer0 单独跑不可能对。
- **BF16 全 30 层过**：real GEMM 维度错则 BF16 也崩。
→ real GEMM 无病。capture stride 是 `diagnostic-layer-capture` feature 专属路径（layer_capture_stride 仅在
  gllm-kernels BufferLayout 编译期算，real kernel 不碰），错 8192 倍不影响 real 推理。
※ 代码事实：`op_gemm_dims` 全库零匹配（报告符号旧/串，勿查该函数），同 `to_quant_precision` 情形。

## 2. 8388608 = 2048×1024×4 = q_proj 权重 numel×4 字节，被当 hidden 维度返回
derive_capture_hidden 主路径失败(inputs[0] shape 非2维/last非Concrete) → fallback 扫 GEMM 反推 K →
返回的不是维度而是 q_proj 权重的 F32 字节数(2097152 elem×4)。非 trans_b K/N 互换(那给1024/2048)，
非任何 Qwen3 架构维度。**与 weight_physical_bytes 非32倍数 fallback `numel*4` 同病根：旁路 SSOT 二次推导 + numel×4 混淆（第二次发作）。**

## 3. layer0 capture cos=0.0293 是诊断工具污染假信号
stride 错 8192 倍 → 读位彻底错位 → 读到垃圾 → cos=0.0293 非 Q4_0 真 bug。
**修好 capture stride 前，逐层 cosine 曲线不可信，step2 数据作废需重跑。**

## 4. 修法：derive_capture_hidden 删 fallback，用 SSOT
删掉扫 GEMM 反推的 fallback，直接用 `ModelGeometry.hidden`(=1024)。
不必加临时 eprintln 区分 fallback 扫到哪个 GEMM —— 无论扫到哪，正确修法都是「别 fallback、用 SSOT hidden」，
诊断该分支无意义。诊断工具修复，风险低、可逆。

## 5. 宪法判断重定向：real GEMM 无病，违宪的是两处 fallback
用户「代码随数据变、不重复推导」原则对，但打偏了目标。真正违反的是：
(a) capture 的 derive_capture_hidden fallback 扫 GEMM 反推；
(b) weight_physical_bytes 非32倍数 fallback `numel*4`。
同一病根：**旁路 SSOT 去二次推导/猜测尺寸**。real GEMM 没这病，别去重构本来正确的路径。
→ REQ-JIT-LAYOUT-01 扩一条：**诊断/尺寸推导禁止扫 op 反推，必须取 ModelGeometry SSOT**（详细交 spec_write）。

## 6. 未决（修 fallback 后无后果，记一笔）
- derive_capture_hidden 主路径为何失败：inputs[0] shape 非2维（可能是 [batch,seq,hidden] 3维）或 hidden 是 SymDim 符号（非 Concrete）。
  说明 capture 工具对 activation shape 的假设与 real graph 不符。用 SSOT hidden 绕开即可，不阻塞。
- 修 capture 后验证：layer0 capture 应回到 cos≈1.0（确认 0.0293 纯工具污染）；若仍发散才需重审。

## 下一步（executor）
修 capture stride（derive_capture_hidden 用 SSOT）→ 重跑 step2 拿真逐层曲线 → 回四步实验 step2/3/4。

---

# 【裁决·换路径】capture 工具 ≥4 处缺陷，弃用于 Q4_0 狩猎，改单层隔离/早退

## 背景：修 derive_capture_hidden 后仍失败
hidden 修对(=1024)但 stride 仍=167772160、layer0 cos 仍=0.0293。
真根因链：per_layer_stride = max_seq_len(40960 KV上界,非实际seq5) × hidden × 4 = 167772160；
且 emit stride=335544320 = alloc×2（LayerCaptureInfo.per_layer_stride 某处翻倍）。

## ★决定性事实一：layer0 cos=0.0293 与 stride 无关 = 第 4 个独立缺陷
layer0 读位 = `base + 0×stride` = base，**无论 stride 取何值（167772160/335544320/40960上界）×0 恒为 0**。
故 emit/alloc 2倍差异、max_seq_len 上界膨胀全不影响 layer0 读位。layer0 仍 cos=0.0293
→ **capture 写入内容 或 读取长度/dtype 本身就错**，与 stride 无关的第 4 个缺陷。
**修 stride 100% 不会让 layer0 变对** → 单独否决"继续修 stride"。

## ★决定性事实二：所有逐层历史数据来自坏 capture，全部不可信
test_diag_qwen3_layer_bisect.rs:9-18 明写：layer27 cos=0.005、layer0-in-loop cos=1.0
**都用 diagnostic-layer-capture ring-buffer 测**。capture 已证 ≥4 处缺陷且 layer0 就污染。
→ 除 `GLLM_SINGLE_LAYER=1` 单层隔离 layer0 cos=1.0 外，**其余逐层结论（含"断点在哪"）全部不可信**。
问题升级：不是"仪器校准"，是"除单层隔离外历史逐层结论建立在坏仪器上"，须换可信路径重建。

## ★决定性事实三：单层隔离是唯一已验证可信路径（非冒险）
GLLM_SINGLE_LAYER=1 得 layer0 cos=1.0 不走 ring-buffer，是唯一产出过可信数值的仪器。
"单层隔离逐层跑"是回到已验证仪器，不是新冒险。

## 换路径：P1 → P2
- **P1（先做）real-loop 早退重建可信断点**：用已有 mid_layer_encode.rs 的 target_layer + ExitEarly，
  跑真实 28 层循环、layer N 早退、正常 decode 路径读 [seq,hidden] vs golden hs_{N+1}。
  用真实循环(真 stride 步进+真 swap)，不碰 ring-buffer。目的=可信重建首个发散层 N。
- **P2（后做，断点层）单层隔离 index=N 注入 golden hs_N**：即四步实验 step4 的 A-剥离，提前。
  输出==golden hs_{N+1} → layer N 权重+模板对，bug 在多迭代 plumbing/swap → **A 坐实**；
  输出!=golden hs_{N+1} → layer N 读错权重/decode，与 A 无关。
- 顺序理由：P1 复用 ExitEarly 廉价定断点(可能无需 knob)；P2 才需"注入上层输出+跑任意层"knob，只对断点层用。

## capture 修复降级为独立工作项（别现在追）
capture ring-buffer 按 max_seq 上界分配 stride 设计不算根本错（要容纳最长 seq），
但实现 ≥4 处独立缺陷、正压在数值狩猎关键路径上。
→ emit/alloc 2倍 stride 差异(335544320 vs 167772160) + layer0 写入/读取缺陷 归档为独立工作项，**别现在追**。
别在关键路径上修工具。

## ★P1 第一验证点：layer0 真实循环早退是否 cos=1.0
- P1 layer0 cos=1.0 → 断点在 layer0→1 之后，回原假设。
- P1 layer0 就发散 → 断点比想象更早，"layer0 在循环里对"是 capture 假象，
  问题在 layer0 的循环上下文（首次 swap/输入注入），故事重写。

## 三未决（executor 先确认）
1. P1 的 target_layer+ExitEarly 能否在 test harness 逐层设置并读输出？（若 mid_layer_encode callback 可 wire target_layer=0..27，P1 近零新代码）
2. P2 单层隔离注入 knob：跑任意 layer K + 注入 golden hs_K + 权重偏移 base+K×stride。仅断点层用。
3. capture emit/alloc 2倍 stride 差异归档为独立工作项，别追。

## C-2 铁律说明
工具调试已超 2 次阈值。换路径 = honoring 铁律（止损、别在死路硬磕），非违反。

---

# 【定案·根因锁定 embed dtype 错配】断点在 embed（layer 循环之前），三跨层嫌疑全排除

## 0. ★architect 搜索范围纠错（信任度必读）
`gllm-kernels` 是**同级 crate**（`/home/putao/code/rust/gllm-kernels/`，根 Cargo.toml:44 `path="../gllm-kernels"`）。
architect 前七轮所有 grep 只搜 `/home/putao/code/rust/gllm/`，**从未搜 kernel crate**。
→ 之前四次断言的"幽灵符号"（to_quant_precision / op_gemm_dims / QuantConcatSeq / lower_op.inc.rs:1499）
  **全是真实代码，executor 引用路径一直正确，是 architect 搜索范围错**。dtype 错配结论不受影响（884x 是数值物理事实）。

## 1. executor 发现（独立 raw_check，非 capture）
gllm embedding first5 量级 ~1–71；golden hs_0(HF BF16 embedding) ~0.01；cosine=-0.0115。
**embedding 本身就错 → 三跨层嫌疑(activation_alias/weight_stride/KV)全排除**：layer0 无跨层输入就发散，
断点在 embed（layer 循环之前）。这比七轮假设的跨层断点更靠前、更根本。

## 2. dtype 错配定案（884x 量级证明，非 nibble 顺序）
- 检查1：token_embd.weight GGUF type=2(Q4_0)，shape=[1024,151936] → embed 确是 Q4_0。
- 检查2：手算 embed block(token 785)：|d|=0.0066，SPLIT/INTER elem 量级都 ~0.05（与 golden ~0.01 同量级）；
  gllm ~1–71 比手算大 200–1000 倍。
- **INTERLEAVED/SPLIT 是块内 nibble 置换，保范数**，数学上解释不了 884x → 排除 nibble 顺序。
  884x = **把 Q4_0 raw 字节(scale+nibble)当浮点直读**（没 dequant），dtype 错配坐实。

## 3. R1/R2 二分（根因区域锁定，精确落点待 dump）
build_graph.inc.rs:205-207 **op 选择分支存在且正确**：
`weight_quant_types.get("embed")` 有值→`Op::QuantGather{quant_type}`(dequant)；None→`Op::Gather`(dense 按 dtype 直读)。
- **R1（最可能）**：运行时 get("embed")=None → 走 dense Gather → Q4_0 字节按 dense dtype 直读 → 884x。
  成因：token_embd.weight 的 canonical 名映射没落到 "embed" 键（executor_compile.rs:560 all_canonical_for 漏）。
- **R2**：QuantGather 正确 emit，但 kernel(quant_gather_emit.rs/quant_decode.rs)的 Q4_0 dequant scale/block 读错。

## 4. ★旁证支持 R1
layer0 单独跑 cos=1.0 → layer 权重(blk.N.*)的 QuantGemm dequant 对 + weight_quant_types 对 layer 填充对、kernel dequant 对。
**唯独 embed 错**，而 embed 外部名 token_embd.weight 与 layer 权重命名规则不同 → 极可能 **embed 专属 canonical 名映射漏** → embed 走 dense 分支。与"layer 全对、仅 embed 爆"完全自洽。

## 5. ★决定性 dump（派 root-fix 前唯一必做，定 R1/R2 + 定改哪 crate）
dump 编译后 graph 的 embed op 类型：
- **Op::Gather(dense) → R1 坐实** → 改**主 gllm crate**：修 weight_quant_types 对 embed 的 canonical 名映射
  (executor_compile.rs:560 / name_map)，让 get("embed") 命中 Q4_0。**不碰 kernel**。
- **Op::QuantGather → R2 坐实** → 改**gllm-kernels crate**：修 quant_gather_emit.rs/quant_decode.rs Q4_0 dequant。
R1/R2 修复点在**不同 crate、完全不同代码**，盲派有 50% 改错 crate = 第13次空转。此 dump 消除风险。

## 6. classic.rs INTERLEAVED vs SPLIT = 独立 issue，embed 之后修
classic.rs(interleaved) vs llama.cpp(split) 确有不一致，但**保范数**（症状 cos 低、量级对），
与 embed 的量级爆炸是两回事。**先修 embed dtype 错配（Q4_0 全崩直接根因），classic.rs 独立后修且须带数值证明单独验**。别混为一谈（否则又一次"改一堆没对准根因"）。

## 7. 架构落点：仍是"漏 SSOT"病根第 N 次发作
embed quant_type 没正确进 SSOT map（与 weight_physical_bytes fallback / derive_capture_hidden fallback 同病根）。
→ REQ-JIT-LAYOUT-01 扩条：**embed 与 layer 权重 quant_type 必须走同一 canonical 映射；
  build 期断言"GGUF 里量化的张量在 graph 里必走 Quant* op"**（量化张量走 dense op = 编译期报错）。详细交 spec_write。

## 下一步
主会话 read-only dump 编译后 graph embed op 类型 → 定 R1/R2 → 按 §5 分 crate 派 root-fix。
R1 修复须保证不误伤 BF16 embed（BF16 走 dense Gather 正确，直读即对）——只让**量化** embed 改道 QuantGather。

---

# 【定案·R2 坐实】QuantGather Q4_0 dequant 的 scale 指针错位（非共用解码数学）

## dump 结果
weight_quant_types.get("embed")=Some(Q4_0)、embed op=QuantGather → **R1 排除，R2 坐实**。
QuantGather 正确 emit，但 kernel Q4_0 dequant 输出 884x 量级垃圾。

## 关键代码事实：QuantGather 与 QuantGemm 共用同一 DecodeTraceBuilder
quant_gather_emit.rs:155 用 `DecodeTraceBuilder::new(desc,lanes).build()` 生成 decode trace，
auto_lower_trace_raw 展开、VecStore 存**解码后 F32**（:234-273）。此 builder 正是 QuantGemm 也用的
（quant_decode.rs:22-24 注释明写两路共用）。→ 三项排除：
- 排除"缺 dequant 直存 raw"：gather 存 decoded F32，非 raw。
- 排除"scale dtype f16 当 f32 读"：f16→f32 加载在共用解码器，layer0 GEMM cos=1.0 证明它对。
- 排除"块内 scale 偏移错"：块内 scale_offset 也共用，GEMM 同 desc 同样正确。
→ **bug 不在共用解码数学，在 QuantGather 特有的"喂给解码器的指针/步幅"。**

## 量级 884x 锁定统一根因：scale-load 指针指进 nibble 区
- "读对 scale 但读错 block"→ f16 scale 仍小(~0.006)→dequant ~0.05，给不出 1–71。排除"跨block 解读对"。
- 1–71 要求"scale"≈1.5–2.0(比真值大~230x)。**唯一产出 O(1) scale 的情形：把 packed nibble 2 字节当 f16 scale 读**
  （随机 nibble 解成 f16 = O(1)–O(100)，乘 (nibble−8)∈[−8,7] = O(1)–O(1000)，精确匹配 1–71）。
- **统一根因**：解码器 Input(0)(block_base，scale-load 用)未指向真实 f16 scale 字节，而指进 packed nibble 区。
- element 0 就 884x 错 → 非块间漂移，是初始 scale 读取/行基址 → 最可能 **行基址步幅错**：
  row_base = embed_ptr + token_id × row_stride，若 row_stride 误用输出步幅(hidden×4=4096)而非
  输入量化步幅((hidden/block_size)×block_bytes=32×18=576)→ 行基址落错内存 → 首字节非 scale。
  次可能：block_ptr 与 data_ptr 别名/被 data 前进污染。

## 决定性 instrumentation（root-fix 时先加，一次钉死成因）
打印 token 785 / block 0 时 block_ptr 实指的 2 字节，对比 GGUF 该 block 首 2 字节(真实 f16 d=-0.0066)：
- 不同 → scale 指针错位坐实，差值=偏移错误量 → 查 row_stride 计算(QuantOffsetDsl::derive_*) vs block_ptr 初始化。
- 相同 → 判断错，回到 scale 应用/store 环节（量级证据强烈指向前者）。

## classic.rs 无关（回答 R2 下是否一起修）
QuantGather 走 quant_decode.rs JIT trace(DecodeTraceBuilder + auto_lower_trace_raw)，**不走 classic.rs**
（classic.rs 是 CPU 标量运行时路径，gather 不调用）。→ **classic.rs INTERLEAVED/SPLIT 与本 bug 完全无关，
独立 issue，后修，带独立数值证明。这次别一起动。**

## 根治方案 + 派单
- 改 **gllm-kernels crate** 的 quant_gather_emit.rs 指针/行步幅（最可能 row_base 误用输出步幅 4096 vs 576）。
- **不改共用解码器**（对 GEMM 已验证正确，改它弄坏 layer 权重）。根治面小、只动 gather 特有指针/步幅 setup。
- 带 instrumentation 先钉死行基址 vs block_ptr，再动代码。
- 回归：修后 layer0 仍须 cos=1.0（确保没碰共用路径）；不误伤 BF16/其他量化格式。

## 架构落点（REQ-JIT-LAYOUT-01 再扩条）
量化 gather/gemm 输入行步幅必须派生自 block_bytes(存储)，禁止误用输出 compute_elem_bytes 步幅；
emit 期断言输入/输出步幅分离。这是"输入量化步幅 vs 输出 compute 步幅两套值取错"——漏 SSOT 病根的又一面。

---

# 【裁决·refute blob=F32】blob=Q4_0 raw 被 layer0=1.0 强制，row-stride 假设已证伪

## row-stride 误用假设证伪（architect 自纠）
quant_gather_emit.rs:100 row_stride_bytes=(hidden/block_size)*block_bytes=576（输入步幅，正确）；
out_row_bytes=hidden*4=4096（输出，line 94 严格分离）。row_base=token_id*576 正确。**architect 的"行基址误用 4096"假设证伪。**
dispatch：emit_quant_gather_inline 仅转发 emit_quant_gather_trace_driven(:59)，与 QuantGemm 共用 DecodeTraceBuilder+desc。
→ B(desc)/C(共用解码)/D(双路径)全由 layer0=1.0 排除。

## blob=F32 premise 被两条硬事实 refute
- **Fact1：DType 枚举无 Q4_0**（gllm-kernels types:885-899 只有 F32/F16/BF16/FP8/INT*）。Q4_0 只在 weight_quant_types，
  ModelGeometry.dtype = detect_weight_dtype()→floating_point_dtype() 返 F16/F32，**永不 Q4_0**。
  → `needs_dtype_conversion=(compute_dtype≠dtype)` 是 **F32-vs-F16/F32**，非"F32≠Q4_0=true"。用户前提是类型推理错。
- **Fact2：layer0=1.0 逻辑强制 blob=Q4_0**。build_graph 从 weight_quant_types(=Q4_0) 为 layer+embed **都**选 Quant*(Q4_0)。
  若 blob=F32 → QuantGemm(Q4_0) 读 F32-as-Q4_0 → layer 也该崩，但 layer0=1.0 → 矛盾。
  → blob **必是 Q4_0 raw**（zero-copy，needs_dtype_conversion=false）。"QuantGemm 期望 F32"子claim 错——它按 quant_type=Q4_0 解码。

## 决定性检查（零新代码）
executor_compile.rs:206/210 两条 log 已在现有 run 输出：
- "zero-copy weight path — N quantized tensors passed as-is" → blob=Q4_0 → **blob=F32 theory refuted**，bug 在 gather 指针/区域。
- "dtype conversion F16→F32: N converted" → 需重新解释 layer0 为何仍对（则 layer op 非 QuantGemm(Q4_0)，非对称选 op）。
grep 现有 run log 即定。architect 预测 "zero-copy"。第二确认：dump layer op 类型(QuantGemm vs Gemm)。

## 根因方向（不变，本轮 detour 排除 dtype 错配）
blob=Q4_0、双 op 解 Q4_0、layer 对 embed 错 → bug 是 **gather 特有指针/区域**：
A1(embed_ptr 值) 或 zero-copy embed 权重未落到 QuantGather 算的 `embed_ptr+token*576` 处。
dequantize-to-F32 违宪框架 moot（zero-copy 下没发生，别去重构 dequant 路径）。

## 未决（log 确认 zero-copy 后再追）
zero-copy 量化权重是 separate ext_ptrs(line 168-172 qt.data.as_ptr())，**不进 raw_floats blob**（pack 只读 raw_floats）。
→ 它们如何组装进 JIT layer-loop + embed gather 读的连续布局？若 embed zero-copy 指针没接到 QuantGather 的 embed_ptr 计算处 = A1 精确。这是 log 确认后下一个 trace 点。

---

# 【裁决·终结静态】"pack 只用 raw_floats→blob 零" 是误读，blob=Q4_0 已定，转运行时 dump

## 误读纠正（code fact）
pack_weights_from_graph 签名取**两个** map：weight_ptrs(:103) **和** raw_floats(:105)。用户只看到 raw_floats 参数。
标准权重路径是 **:417 `weight_ptrs.get(canonical_name)`**（非 raw_floats）；per-layer 也用 weight_ptrs(:360,:398)。
weight_ptrs **含 Q4_0 raw 指针**（executor_compile.rs:168-172 兜底 qt.data.as_ptr() 插入）。
→ layer + embed 的 Q4_0 raw 字节**都经 weight_ptrs 打进 blob**。blob=Q4_0 raw、embed 区已填。与 layer0=1.0 自洽。
→ **A2(embed 区零/未打包) 证伪；blob=F32 证伪；blob=零 证伪。**

## 已连续证伪四个静态假设 → 静态到头
row-stride 误用✗ / blob=F32✗ / raw_floats-only→blob零✗ / (更早)幽灵符号✗。
每个都被下一个 code fact 推翻。**教训：本 saga 反复"再来一轮静态"，每轮被事实打脸。根因是运行时值，只有 dump 能定。停止静态。**

## 图景已完全自洽，唯一落点 = QuantGather 运行时 A1
blob=Q4_0(双 op 均对)、解码器共用已验证、打包已验证 → bug 是 gather 特有**运行时值**：
embed_ptr 解析到错误 blob 偏移，或 embed canonical 名没进 weight_ptrs → 其区在 :420 计 missing → 留零。

## 两个廉价运行时检查（无需 JIT 插桩，立即做）
1. **grep 现有 run log :536** `pack_weights_from_graph: packed=X, missing=Y`。
   missing>0 → 有权重(可能 embed)未在 weight_ptrs 解析 → blob 区零 = bug（canonical 名/tied-embedding 解析）。同时确认 zero-copy。
2. **Rust 侧 post-pack dump** blob[embed_offset..+18] vs GGUF token_embd 首 18 字节：
   - 零 → embed 走 :420 missing；修 = weight_ptrs 对 embed 的 canonical 解析（主 crate）。
   - 有效 Q4_0 但推理仍垃圾 → embed_ptr 运行时值错（A1 地址组装）。
   - 匹配且 embed_ptr 对 → 才查 gather trace（但打包+解码+blob 均已验，概率低）。

## tied-embedding 强提示
token_embd.weight → 同时映射 "embed" + "lm_head"（name_map.rs:256）。canonical 解析/去重若在 tie 上出错，
embed 可能没拿到 weight_ptrs 条目 → missing → 零。这是检查 1 missing>0 时的头号嫌疑。

---

# 【裁决·A1 坐实并锐化】blob embed 正确，token0(offset0) 仍错 → 消除 stride 类，剩 A/B

## 运行时证据
[BCE-DIAG-PACK] packed=14 missing=21；embed offset=0 first18 与 GGUF token0 block0 **字节级一致** d=0.014771。
→ blob embed 区正确、blob=Q4_0 raw。A2 证伪。根因 = QuantGather 运行时读/解码错。

## ★突破在数据里：token0 在 offset0，与 stride 无关
gllm first5 是 **token0**（prompt "The"，hs_0[0][:5]）。token0 读 `embed_ptr + 0×row_stride` = **offset0，与 stride 无关**。
blob offset0 已验正确、scale d=0.0147 → 正确 dequant 最大量级 0.0147×8 = **0.118**。但 gllm token0 输出 **14.596**（884x）。
→ **消除 row_stride / embed_ptr-offset / 错 token 整类假设**（token0 不依赖 stride，读位数据已验对，输出仍 884x）。
剩两者：
- **(A) embed_ptr base ≠ weight_blob base** → token0 读错位置。
- **(B) embed_ptr base 对，但解码从块内错误子偏移读 f16 scale** → scale 取自 nibble 字节 → O(1) scale → 884x。

## code fact 缩窄 A/B
- embed_ptr = weight_blob_ptr(kernel **AbiArg(1)**, vm_state.rs:133) + embed 偏移(=0) → **应恰等于 weight_blob base**。
- QuantGather 用 `block_base == data_ptr`(quant_decode.rs:22,143) —— 这是 gather 相对 gemm **唯一独有**的指针 setup
  （QuantGemm 单独前进 data_ptr）。QuantGemm 共用同一解码器却对(layer0=1.0)。
  → 若是 (B)，错在 gather 的 block_base/data_ptr 初始化把 scale-load 喂了错指针（**kernel crate**，非共用解码数学）。倾向 (B)。

## 最廉价钉死（一次终结）
**dump embed_ptr 运行时值 vs weight_blob_ptr base**：kernel ABI 参数(AbiArg1)+编译期偏移，**Rust 侧 kernel 调用点可观测，无需机器码插桩**。
- embed_ptr == base → **(B)**：修 gather block_base/data_ptr init（gllm-kernels quant_gather_emit.rs 指针 setup）使 scale-load 读块首。
- embed_ptr != base → **(A)**：embed 权重输入偏移/缓冲 wiring 错。
免费旁证（近乎已证）：手算 blob[0..18](scale0.0147, nibble bytes[2..18]) = 正确 token0（≤0.118），已知 gllm=14.596≠它
→ 解码确未用 scale 0.0147。查哪种误读复现 ~14.6（scale 按 f32 读 4 字节 / scale 从 byte2 nibble 区读）即定确切 fix。

## missing=21 大概率独立
单模板设计：28 层共享 1 模板，多数 per-layer canonical 名合法不打包。packed=14 ≈ embed + 1 层模板(~11) + final_norm + lm_head。
**但须确认 lm_head(tied) 不在 21 missing 里**——若在，logits 独立于 embed bug 而坏。记一笔，暂不追。

## ★元教训（必须正视）
这是本 saga 第 5 个静态理论（row-stride / blob=F32 / blob=零 / 幽灵符号 / offset 变体），前 4 个都被下一个事实推翻。
**token0 观察本身是突破**——它是数据事实非代码假设，一行消除整个 stride/offset 类。
剩下 A/B 只是**一个运行时指针值**。dump 它，别再推理它。

---

# 【更正·token 785 非 0，stride 复活】token0 突破作废，改联合 dump

## 更正（architect 第 5 次分析失误）
prompt "The capital of France is" token ids=[785,6722,315,9625,374]。gllm first5 是**位置0=token 785**，非 token 0。
→ "token0 读 offset0、stride 无关" 推理**作废**。token 785 读 785×576=452160，**stride 有关**，wrong-stride 假设复活。

## 幸存的 token-无关证据：884x 量级仍约束机制
读**错但块对齐的合法 Q4_0 块**仍给 scale~0.006–0.05 → 量级~0.05，非 884x。故 884x 仍要求：
scale 取自 nibble 字节（mid-block/错子偏移）/ 读偏移落 mid-block(非 18 倍数，"scale"=packed data) / 读非 Q4_0 数据。
→ blob 单点 dump 不足以区分，必须加第二个 dump（QuantGather 实际读偏移）。

## A/B 排名：dump 前不猜
token=785 复活了 wrong-stride 路径（785×错stride 落 mid-block 也给 884x）。静态说 stride=576 但静态已错 5 次。
→ 不再"必是 B"。跑 dump，别猜。

## 联合 dump（两半都做，否则只分 pack-vs-read，不分 A-vs-B）
1. blob[452160..452178] vs GGUF token 785 — stride-576 位置的 pack 是否对？
2. QuantGather 位置0 实际读偏移 = embed_ptr_base + 785×runtime_row_stride；且 dump embed_ptr base vs weight_blob_ptr、row_stride 运行时值。

| blob[452160] | QuantGather 读偏移 | 结论 |
|---|---|---|
| ≠ GGUF785 | — | **pack bug**（token785 区写错；查 missing=21 / copy_size 截断）|
| == GGUF785 | ≠452160 | **stride/base bug (A类)**——runtime stride≠576 或 base 错；落 mid-block→884x |
| == GGUF785 | ==452160 | **decode 子偏移 bug (B)**——读对块、scale 从错字节→884x |

第 3 个 dump（读偏移/embed_ptr base/row_stride 值）才分 A/B；blob[452160] 单独把 A/B 缠一起。
最廉价：kernel entry 处 dump embed_ptr(AbiArg1+偏移) + ScalarToIndex 的 stride 立即数，验 stride==576 且 base==weight_blob。Rust 侧/build 期可观测，无需机器码插桩。

## 元教训强化
architect 本轮又错（token id 误设 0）。**规律：凡 architect 静态推理/假设即高概率错；凡运行时数据事实即推进。**
→ 后续一律"先 dump 数据事实，再解释"，architect 只做 dump 设计与真值表，不再出机制预判。

---

# 【定案·B 由 code fact 排除 → A，fix site 命名】

## dump 结果
- dump1：blob[452160]（token785 块）字节级 == GGUF token785，d=-0.006592。**pack 正确。**
- dump2：QuantGather build 期 row_stride_bytes=576（非 4096）。**stride 正确。**

## B 被 code fact 排除（非预测）
- Q4_0 desc：`scale_layout: BlockScalar{offset_bytes:0, dtype:F16}`（quant_format.rs:359）→ scale 从 block_base+0 读。
- scale-load 读 Input(0)=block_base（quant_decode.rs:185 emit_scale_load），**QuantGemm/QuantGather 共用**。
- QuantGemm 对(layer0=1.0) → scale-load-from-block_base+0 数学已证对。
- QuantGather block_ptr=row_ptr+blk_off、data_ptr=block_ptr build 已验（576✓、blk×18✓）。
→ **B(decode 子偏移) 不可能不同时弄坏 QuantGemm。** block_base 每项已验，唯 **embed_ptr 本身未验** → **是 A。**

## fix site：lower_op.inc.rs:812-814
```rust
let weight_ptr = op.inputs.get(1).copied()
    .and_then(|tid| resolver.materialize(prog, tid, abi))
    .unwrap_or(input_ptr);   // ← input_ptr = inputs[0] = indices(token-id) 指针!
```
embed_ptr = resolver.materialize(inputs[1])。两种产生 884x：
1. **materialize 返错指针**（非 weight_blob_base+0）——两源分裂：pack 用 named_offsets[embed]=0，materialize 用 resolver 自己的 tensor→ptr 图。
2. **`.unwrap_or(input_ptr)` fallback 触发**——materialize 返 None 时 embed_ptr 静默变成 **indices 指针**（token-id i32）。读 int 字节当 f16 scale → O(1) → 884x。**这个静默 fallback 是地雷：解析失败该报错，不该静默读 indices 当权重。**

## 最廉价 dump（区分两种，终结）
lower_op.inc.rs:812 或 kernel entry：dump 解析出的 weight_ptr 值 vs AbiArg(1)(weight_blob_ptr)+embed 偏移(0)。
- == weight_blob_base+0 → materialize 对，bug 在下游（每项已验，概率极低）。
- == input_ptr(indices) → **fallback 触发**，materialize(inputs[1])=None → fix=embed 权重张量为何不解析（主 crate 张量注册 / tied-embedding canonical）+ 删静默 fallback。
- == 其他 base → materialize 映射 embed 到错张量/偏移 → fix 在 resolver tensor→ptr 图。

## 独立必修（无论结果）
`.unwrap_or(input_ptr)` 本身是潜伏 bug：解析失败应 error，不应静默把 indices 当权重读。

## 本轮方法论验证
architect 守"只 code fact"承诺后收敛：desc + 共用解码器 + 已验项 by construction 把 B→A，fix site 定位到 3 行含危险静默 fallback 的块。

---

# 【裁决·静态穷尽且自相矛盾 → 转验证"测量本身"】

## dump3 + 后续 code fact：全部静态项已验证为对
- blob[452160] 对(dump1) / stride=576(dump2) / materialize 成功且 fallback 未触发(dump3) ✓
- 偏移源 = weight_layout 默认 0（embed 是 global，不在 wtids，keep Weight{offset:0} buffer_alloc:686）= 匹配 pack named_offsets[embed]=0 ✓
- abi.weight_ptr = blob base（layer0=1.0 证）✓
- embed 在**同一 mega-kernel、同一 blob、同一 weight_ptr**（abi_types:4 单一 JIT；executor_core:731 单 weight_blob.as_ptr）✓
→ **每个静态项都对，输出却 884x = 平地矛盾。再读代码无法解决**（计算的每个输入已证对）。继续静态 = 13 轮的反模式。

## ★被低估的关键事实：cosine 是 scale-invariant
executor 报 cosine(embedding, golden hs_0) = **−0.0115**（近正交）。**cosine 与量级无关**——纯量级错（scale 大 230×）cosine 仍 ≈ **+1.0**（方向同、仅缩放）。
正交 cosine = **方向乱**，非仅量级。→ **这不是"scale 读大了"，是读到根本不同/乱的值。** 884x + 正交 = 读的不是 token785 的干净 dequant。

## ★必须正视的前例：诊断 READ 本身当过 bug（capture cos=0.0293）
capture 那 4 轮，cos=0.0293 最终是**诊断读取工件**（stride 错），非真 compute bug。
"gllm embedding first5=[-3.898,14.596...]" **从未验证是 QuantGather 真实输出**——来自 DiagnosticScratchpad::embedding() 读 scratchpad **offset 0** + read_dtype_aware（pack_observe:731）。
若 QuantGather 写 embedding 到 ≠0 的 scratchpad 偏移，或读用错 dtype/layout → 诊断出垃圾而 QuantGather 是对的 = capture 同款工件。

## 两个剩余候选（对称于已犯过的错）
- **(X) QuantGather 真写垃圾**（真 compute bug），或
- **(Y) QuantGather 对，embedding 诊断读是工件**（write 偏移 vs embedding() 硬编码 offset-0 读 不一致）。

## 决定性检查（最廉价，无需 gather 内部 JIT 插桩）
QuantGather 执行后 dump **embedding 输出偏移的 RAW scratchpad 字节 + 手算 dequant**，且 dump QuantGather output_ptr 写偏移 vs embedding() 读偏移(0)：

| raw scratchpad 手算 | write-off vs read-off(0) | 结论 |
|---|---|---|
| = token785 正确 dequant | 不一致 | **(Y) 诊断读工件**——QuantGather 对，"884x" 从来不真（修读/停止信它）|
| = 垃圾 | — | **(X) 真 QuantGather bug**——才去插桩 gather 运行时 block_base |
| = 正确 | 一致 | 矛盾续 → 884x 来自另一读，查哪个读产生的 |

Rust 侧（raw scratchpad 切片 + 两个偏移值），非机器码插桩。

## ★元教训（13 轮最重要）
architect 反复把上报数字（884x / cos=0.0293 / token id 0）当真值来推理，其中三个本身错了（capture 读工件 / architect token-0 误设 / 可能这个）。
**下一次机制狩猎前，先验证测量本身。** cosine 的 scale-invariance 是早该 flag 的 tell：它说"非 scale bug"，而没人验过 884x 是否反映 QuantGather 真实输出。
→ 规则升级：**上报数值先问"这数怎么测的、测的是不是目标"，再据此推理。**

---

# 【裁决·第 3 个诊断读工件：embedding 槽被复用，884x 非 QuantGather 输出】

## 反 (Y) 但触发更深反构：读的是"正确 offset"但"错误 time"
raw_check 用 diagnostic_tensor_offset("embedding")=emb_off（正确 offset，非 embedding() 的 0）读 → (Y offset 工件) 排除。
但 test line 8：`diagnostic_prefill_scratchpad(&tokens)` 跑 **embedding→全 28 层→output**，**之后** line 24-30 才读 sp.data[emb_off]。
→ "正确 offset" ≠ "读时刻值正确"。

## code fact：embedding 槽在 layer0 后被 liveness 复用
buffer_alloc.rs:438-451 gather-output 排除（:450 `!gather_output_tids.contains`）**只把 embedding 移出 ping-pong，不延长其 lifetime**。
embedding 仍走标准 liveness（:154-160,:285-349）：last_use = 唯一消费者 = **layer0 输入**。layer0 消费后，linear-scan（:339）**释放 embedding 槽、复用给后续层 intermediate**。
→ raw_check 全 prefill 后读 emb_off = **读到后续层激活覆盖值**（量级 ~1-71，与 diag norm=60.7 一致；与 hs_0 无关 → cos≈0）。
→ **[-3.898,14.596,-71.331] cos=-0.0115 = 占据已释放 embedding 槽的层激活垃圾，非 QuantGather 输出。**

## 平地矛盾彻底消解
所有静态项对（blob/stride/materialize/offset）→ **QuantGather 极可能写的是正确 embedding** → 槽被复用 → 诊断读太晚 → 884x 垃圾。
**(X)"QuantGather 写垃圾"如述 likely 假。** 这是本 saga **第 3 个诊断读工件**（capture stride → 本次 liveness 复用），同族不同机制。

## 决定性检查（结束第14轮，无需 gather 内部插桩）
**在 layer0 运行前读 emb_off**（early-exit / target_layer=0 pre-node ExitEarly 于 layer0 前），非全 prefill 后：

| layer0 前读 emb_off | 结论 |
|---|---|
| = token785 正确 dequant(~0.05, cos≈1 vs hs_0) | **QuantGather 正确**。884x 是读后复用工件。真 Q4_0 bug 在别处或无 → 用可信 embedding 基线重开 |
| = 仍垃圾(~1-71) | QuantGather 真写垃圾 → 才插桩 gather 运行时 block_base |

## 改读取时刻，不是读取 offset
emb_off 是对的**位置**，全 prefill 后是错的**时刻**。别插桩 JIT，先在 layer0 前读。

## ★元教训（14 轮根本教训）
驱动整轮狩猎的"硬数据点"中至少 3 个是**测量工件**：capture cos=0.0293(stride) / embedding 884x(槽复用) / architect token-0 误设。
**调试全程把诊断读当真值。** 本可省 ~10 轮的纪律：**任何诊断读，先证明它在正确时刻+正确位置观测到目标，再用其值推理。**
prefill 后读 scratchpad 槽 ≠ "embedding"，而是"此刻占据该 offset 的东西"。
→ 若 layer0 前读回正确（我判大概率），QuantGather 无恙，自 ~round8 起在追鬼；真 bug（若 token 仍错）在下游，需自带可信基线。

---

# 【裁决·单层未反证工件，num_layers-invariance airtight 证污染】

## 单层测试有同款缺陷（不是"正确时刻"）
GLLM_SINGLE_LAYER=1 **仍跑 layer0**（num_layers=1，user 确认）；raw_check 在 prefill 完成**后**读 emb_off（layer0 已执行完）。
layer0 消费 embedding 后，**layer0 自己的 intermediate**（q/k/v/attn/ffn）可复用已释放的 embedding 槽。
移除 1-27 层 ≠ 移除 layer0 自身的槽复用。**同款读后复用缺陷。**

## ★airtight 论证：num_layers-invariance
embedding 在 schedule step0（任何层前）**计算一次**，其写入值**与 num_layers 完全无关**。
→ 任何**正确**读 embedding 必对 num_layers=1 和 28 给**相同值**。
但 user 读值**变了**：cos=−0.0115(28层) vs +0.0136(1层)。
→ **随 num_layers 变的值不可能是 embedding，只能是被复用槽的内容。变化本身=污染铁证**（无需猜机制）。
且两值都近 **0**（非近 +1.0）= 读复用/垃圾内存的签名，非稳定 QuantGather bug（后者给**稳定**错 cos）。
→ **(X) 未坐实，测量仍不可信。**

## 决定性测试（两向可判）：num_layers=0
代码支持 embedding-only/0 层（model_config tests:1916；semantic_gatekeeper zero-layer）。
num_layers=0 → **无任何层 op 运行 → embedding 槽不可能被复用** → 读 emb_off = 真 QuantGather 输出。

| num_layers=0 读 emb_off | 结论 |
|---|---|
| cos≈+1.0 vs hs_0(~0.05) | **QuantGather 正确**，全部发散是读后复用，自 ~round8 追鬼，真 bug 在下游 |
| 仍发散(cos≈0) | **(X) 真，QuantGather 写垃圾**——architect 判断被 falsify，才插桩 kernel |

## 更廉价·零运行的静态检查
emb_off 范围 [emb_off, emb_off+seq×hidden×4) 是否**重叠**任何 layer0 intermediate 的 offset 范围？
纯 named_offsets 算术（已有）。**重叠 → 污染静态坐实，无需跑。** 最廉价判别器。

## ★双失败模式规避（元）
- 不把上报数当真值（saga 核心错）：cos=0.0136 不等于"QuantGather 错"，等于"读值随 num_layers 变"=污染。
- 不固执（反向失败）：已给能 **falsify architect** 的测试——num_layers=0 若仍发散，architect 认错、转插桩 kernel。
决断：先跑静态重叠检查（免费）或 num_layers=0（决定性）。

---

# 【裁决·根因确认：embedding = ping-pong 残差 buffer，读后全被 swap 覆盖】

## code fact：embedding 就是 ActivationPing（残差流），非私有 scratchpad 槽
- context.inc.rs:434「layer1+ 永远读 embedding(=gather输出)，input_tid 强制 **ActivationPing**」
- buffer_alloc.rs:752「gather 输出(如 embedding)已由 activation_alias.in_tid 强制 **ActivationPing**」
- executor_core.inc.rs:299「VAM 把 embedding 映射为 ActivationPing」
→ embedding 是残差流输入 buffer，**每层 ActivationSwap 覆盖它**。其 "off=0" 是 **ping buffer 内**偏移，
  与 embed@0(权重 blob)、token_ids@0(ABI input arg) 是**三个不同 base**。dump 的 OVERLAP 标记跨地址空间=假警报，无碰撞。

## num_layers-invariance 论证由此机制精确坐实
embedding = 残差 buffer **初始**内容。N 层 swap 后 ping buffer 存 **layer-N 残差**，非 embedding。
prefill 后读 = 读最终残差 = 垃圾 vs hs_0，且值随 N 变（28层 cos−0.0115；1 swap 就不同）。
**随 num_layers 变的值不可能是 embedding = 被覆盖残差。精确坐实。**

## 近乎闭合：layer0=1.0 ⟹ embedding 当时是对的
若 layer0=1.0 真：embedding=layer0 输入(ping)。layer0 消费它 → 输出 hs_1 cos=1.0。
**垃圾输入的正确层变换不可能得正确 hs_1** → embedding 被 layer0 读时**是对的** → **QuantGather 对**。884x 是层覆盖后的读。
（诚实 caveat：layer0=1.0 provenance 亦可疑——来自 capture、GLLM_SINGLE_LAYER 未生效。逻辑成立但需可信复验。）

## dump 三问答案
1. embed@0/embedding@0/token_ids@0 **不碰撞**——权重blob/ActivationPing/ABI-input 三 base（buffer_alloc:24,33,35）。OVERLAP 假警报。
2. GLLM_SINGLE_LAYER=1 **未生效**(num_layers=28) → round-15 从非单层。两次 28 层 cos 不同(−0.0115 vs +0.0136)=**非确定性垃圾**=读未初始化/被覆盖内存。variance 本身再证污染。
3. [-3.898,14.596...] = ping buffer 里 28 次 swap 后的残差数据。非 token_ids(异 base)非 embedding。不值追。

## 决定性测试（含 falsifiability）
**无 isolated JIT QuantGather e2e 测试**（grep 空）——这正是缺失的可信 oracle。
**真 num_layers=0 跑**（embedding-only config 存在，model_config tests:1916）：0 层→0 ActivationSwap→ping 保留 embedding。

| num_layers=0 读 | 结论 |
|---|---|
| cos≈+1.0(~0.05) | **QuantGather 对**，全部发散=ping 覆盖工件，真 bug(argmax 15111 vs 12095)在下游，自 ~round8 追鬼 |
| 仍发散 | QuantGather 真错 → 建 isolated kernel 测试/插桩。**architect 被 falsify，认错** |

长期：**建缺失的 isolated QuantGather e2e 测试**（喂 token785 块，读 kernel **自己**输出 buffer，非共享 prefill scratchpad）=16 轮缺的永久 scratchpad-free oracle。

## ★16 轮根因（元）
**embedding 张量 = ping-pong 残差 buffer，无诊断考虑过这点。** prefill 后每次"embedding"读都读被覆盖残差。
叠加 capture-stride bug + env 未生效 = **测量装置三处独立损坏，却用它定位了 ~10 轮。**
代码注释显示此区分类(ActivationPing vs Intermediate)被反复改（BCE-20260629-005 加了又删，mod.rs:822）=真不稳定区。
**唯一从非 scratchpad 读的 ground truth——最终 argmax(15111 vs 12095)——是真 bug 存在的唯一可信证据。**
num_layers=0 确认 embedding 后，定位须从**输出反向**重启，每个读都须自证观测的是活的、未被覆盖的张量。

---

# 【裁决·第 4 个坏探针：encode_to_layer 返空/零，cos=0.0000 是路径工件】

## P1 结果 + code fact
P1 real-loop encode_to_layer：layer 0/1/2/3/14/27 **全 cos=0.0000**。
code fact：encode_at_layer 在 ExitEarly 时返回 **`Ok(vec![])` 空**（cpu_backend:704、mod.rs:176）；
MidLayerEncode pre_node ExitEarly 携带 self.captured，若 target 层 post_node 未捕获则 None。
→ 空/None → 测试见零/空向量 → cosine 返回 **精确 0.0000**。

## airtight tell：6 个不相干层全精确 0.0000
cosine 仅在一操作数**零范数**时返精确 0.0（防 0/0）。golden 非零 → gllm_last 必零。
6 个独立层的 compute 发散不可能全落精确 0.0 → **空/零返回的签名，非 layer 算错。**

## 不许反转成"bug 在 layer0"
layer0=1.0 确不可信（来自 capture / GLLM_SINGLE_LAYER 未生效）。但翻成"bug 在 layer0" = **同款错误反向**（信未验探针）。
cos=0.0000 极可能是**第 4 个坏探针**。**两向都无可信层测量。**

## ★结束猜测的招：拿已知good对象校准尺子
4 个探针全坏（capture stride → GLLM_SINGLE_LAYER no-op → ping-pong 覆盖读 → encode_to_layer 空）。
**停止在坏探针里找，拿已知对的东西测探针。**
**用 SmolLM2 BF16（round2 全 E2E 通过、已知端到端对）跑 encode_to_layer(0)：**

| SmolLM2 BF16 encode_to_layer(0) | 结论 |
|---|---|
| cos=0.0000/空 | **探针坏非模型坏**。encode_to_layer 无论对错都返空 → 弃其所有 cos → 先修探针(ExitEarly/capture) |
| cos≈1.0 | **探针可用** → Qwen3 Q4_0 layer0 真发散 → layer0/embedding 真 bug，16 轮被"1.0"误导 |

这是本可救全 saga 的一招：**信尺子前，先用它量已知长度。** SmolLM2 BF16 = 那个已知长度。

## dump 三答
1. layer0=1.0 不可信——对，但别反转"bug 在 layer0"。零可信层探针，先校准。
2. cos=0.0000 = 几乎确定空/零返回（cpu_backend:704 + 6层全精确零）。路径工件。
3. dump gllm_last first5——是；但更高价值 = SmolLM2 BF16 校准。两个都做。
4. 别下结论 bug 在 layer0。4 坏探针 = 从无可信中间测量（两向）。"被 1.0 误导"和"bug 在 layer0"是对称陷阱，都信了未验探针。

## ★元教训（第 17 轮）
**4 个诊断路径、4 种坏法、~16 轮用坏尺子定位。** 结束规则：**任何探针输出，未经已知good对象校准前不可采信。**
SmolLM2 BF16（E2E 通过）= 每个中间探针的校准标准。凡在 SmolLM2 上失败的探针，弃用不解读。
唯二非坏探针产出的事实：**argmax 错(15111 vs 12095)** = 真 bug 存在；**blob[452160]==GGUF** = 数据打包对。中间全是坏仪器测的。

---

# 【裁决·arch-controlled 差分：Qwen3 BF16-vs-Q4_0 E2E（round 18）】

## SmolLM2 校准确认探针坏（round-17 判断成立）
SmolLM2 BF16（E2E PASS 已知对）encode_to_layer(0) cos=0.1475 ≠ 1.0 → **探针坏非模型坏** → 弃 encode_to_layer 所有 cos，Qwen3 P1 cos=0.0000 不可信。

## ★连带塌方：layer0=1.0 塌 → "QuantGemm/共用解码器 correct" 也塌
16 轮"QuantGather 共用 QuantGemm 解码器、QuantGemm 对(layer0=1.0)→解码数学对"的证据 = 坏 layer0 探针。
→ **共用 DecodeTraceBuilder（QuantGather+QuantGemm 都用）不再 verified**。Q4_0 运行时 dequant 全面回到嫌疑。

## SmolLM2-vs-Qwen3 差分混淆两变量
SmolLM2-BF16(对) vs Qwen3-Q4_0(错) 差 **arch + quant 两者** → "Q4_0-specific" 未证。

## 决定性差分（现成、可信、控 arch）：Qwen3 BF16 vs Q4_0 E2E
test_e2e_generator.rs:554 存在 **Qwen3-0.6B BF16 E2E**。同 arch，仅量化不同，用可信 E2E 信号。

| Qwen3-BF16 E2E | Qwen3-Q4_0 E2E | 结论 |
|---|---|---|
| 对 | 错(已知) | **bug 100% Q4_0-quant 特有，arch 已控** → 缩到 Q4_0 dequant 运行时（共用 DecodeTraceBuilder / QuantGemm+QuantGather）|
| 错 | 错 | **bug Qwen3-arch 特有，非 quant** → SmolLM2 过仅表示其 arch 子集 work，另起 hunt |

无需修探针、无需新 oracle。控住一直搅局的 arch 混淆。**先跑这个。**

## 序列
1. **Qwen3-BF16 E2E**（test_e2e_generator:554）——arch 控、可信。证/否 "Q4_0-specific"。
2. 若 Q4_0-specific：跑 tests_quant.rs Q4_0 dequant 测试（:254 test_dequant_q4_0_known_values 自校准存在），
   **先确认它走 JIT DecodeTraceBuilder 路径而非仅 scalar cpu_kernels**。JIT 路径过→dequant 数学对→bug 在 wiring(喂 dequant 的 offset/ptr)；JIT 失败/仅 scalar→建 JIT isolated oracle 定位 dequant 执行 bug。
3. backlog：修 encode_to_layer（独立 bug，别阻塞）。

## ★元教训（18 轮）
现成有 **arch 控、无探针、只用 never-broken 信号类(E2E)** 的差分。
纪律：**优先(a)只用从未坏的信号类(E2E argmax/generation) (b)控混淆 的测量。** Qwen3-BF16-vs-Q4_0 正是。
一切探针定位在 SmolLM2 校准通过前不可采信（encode_to_layer 刚失败）。
架构层：**测量装置的系统性不可靠本身是最大教训**——4 探针坏、混淆变量未控、16 轮用坏尺子。REQ-JIT-LAYOUT-01 应加：诊断探针须有 SmolLM2-BF16 校准断言，未校准探针输出禁止用于定位。

---

# 【根因坐实·QuantGather PackedNibbles 解码 4+4 concat 破坏 SPLIT（x86 真实执行 oracle）】

## 可信 oracle（终于对的仪器）
test_q4_0_quant_gather_x86_oracle：构造已知 Q4_0(d=1.0, byte0 lo=10/hi=9)，emit→compile→mmap RWX→**真实执行**。
期望 SPLIT：out[0]=2.0, out[16]=1.0。**实测 out=[2.0,0,0,0, 1.0,0,0,0]** → out[16] 的 1.0 落到 out[4]。
scratchpad-free + 真执行 + 手算自校准，免疫 18 轮所有工件。

## 根因坐实（code + oracle 一致）
x86 lower_quant_concat_seq_x86:3906 **自述** `dst = [lo[0..3], hi[0..3]]`（vmovdqa lo + vinserti128 hi_low128）。
data_byte_advance=lanes/2=4（quant_offset_dsl:181）、4 sub_block → 输出 [lo0..3,hi0..3, lo4..7,hi4..7,...]=**4-lo-4-hi 块交织**，非 SPLIT [lo0..15,hi0..15]。
quant_decode.rs:657「result = interleave(lo,hi)」→ **DecodeTraceBuilder 有意 emit concat/interleave，对 SPLIT-layout Q4_0 是错的**。
→ 正是 ~round7 flag 的 interleave-vs-SPLIT 类，**magnitude-preserving 置换**（值 2.0/1.0 对、位置错）。当年 884x 让所有人（含 architect）误判"量级 bug 非置换"= ping-pong 探针工件误导。

## 修复方向：B（DecodeTraceBuilder 两阶段 SPLIT），非 A
- A（改 x86 QuantConcatSeq lower）：不足。8-lane 装不下 16 元素；sub_block 结构(advance=4/4次/输出偏移)围绕 4+4 concat 建。
- **B（改 DecodeTraceBuilder PackedNibbles 为两阶段）**：lo pass→elem[0..half]、hi pass→elem[half..block]，对齐 SPLIT + QuantGemm Assisted 已用结构。改 trace(去 concat)+输出偏移计算。
- C = B 的循环层表达，同效。

## 修复面
- **所有 PackedNibbles 格式**(Q4_0/Q4_1/Q5_0/Q5_1)都用 QuantConcatSeq(quant_decode:670；test:950 断言 Q4_1 用它)，均 GGUF-SPLIT、均同等坏，须一起修。
- **QuantGemm 不走此路**（用 QuantDequantFma 微内核 plan_lower.rs，x86_lower:3858 明确分流）→ **层权重解码不受本修复影响**，修复 scoped 到 gather/DecodeTraceBuilder。
- 锁死 bug 的测试须更新：quant_decode:950「Q4_1 should have QuantConcatSeq」。
- 查 NibbleWithHighBits(Q5_0/Q5_1/Q6_K，advance 同 lanes/2)是否同病。
- aarch64/GPU QuantConcatSeq lower：trace 停发后变死代码或需同步。

## ★充分性 caveat（必要非充分）
修 QuantGather 必要（坏 embedding 毒害 E2E）。但 **"QuantGemm 对" 现 UNVERIFIED**（唯一证据=坏 layer0 探针）。
QuantGemm 走不同路(QuantDequantFma)，可能有**自己的** SPLIT bug。
序列：1) 修 QuantGather(B) → 2) **给 QuantGemm 建同款 x86 oracle**（已知 Q4_0 权重块→GEMV→对手算 SPLIT）→ 3) 重跑 Qwen3-Q4_0 E2E。仍错=QuantGemm 另有 bug。

## ★元教训（saga 终）
**x86 真实执行隔离 oracle 是 round1 就该有的仪器。** 免疫 capture-stride/ping-pong 覆盖/encode_to_layer 空/GLLM_SINGLE_LAYER no-op 全部工件。
规则：**每条 quant 解码路径都须有真实执行已知值 oracle 测试。** QuantGather 已有；QuantGemm 须补齐才能信层。
早轮 architect 把 interleave/split 判为"独立、可延后、量级保持"——判断本身对（确是量级保持置换），但被 884x 探针工件误导以为不是根因。教训：**探针数值污染会让正确的机制判断被误弃。**

---

# 【新方向·oracle 只测 1 block，真实 32 block 未验（round 20）】

## 状态
QuantGather SPLIT 修复后：QuantGather oracle✅ + QuantGemm oracle✅ + BF16 E2E✅ + 全回归 7050✅，但 **Q4_0 E2E 仍乱码**（输出变了=修复生效但未修好）。
→ bug 不在 1-block 解码数学，在多 block / 接线 / dtype。

## code fact：两 oracle 都只测 1 block
Qwen3 hidden=1024, block_size=32 → **32 block/行**。oracle 用 hidden=32(QuantGather)/k=32(QuantGemm) = **恰 1 block**。
两阶段 SPLIT 修复是 **per-block**（quant_decode emit_unpack Lo/Hi 于每 block；block loop 迭代 row_blocks=hidden/block_size）。
oracle 的 out[16]=1.0 **验了 block 内 hi-half 偏移**，但 **blk_ctr 跨 block 乘子（out_offset=blk_ctr*block_size*elem+sub_off）从未测**。
→ 教科书式 **"N=1 对，N>1 错"**，1:1 对应"oracle 过、E2E 错"。E2E 输出变了=修复触达推理但多 block 仍错序。

## 嫌疑排序
1. **多 block 解码错序（QuantGather+QuantGemm）** TOP——唯一未测维度、Q4_0 特有、精确匹配症状。
2. **Q4_0 weight_stride/dtype 传播（层循环）** MED——Q4_0 特有、BF16 不能清；但 weight_physical_bytes Q4_0 数学 round3 验过、embed row_stride=576(dump2) 已确认，部分覆盖。
3. **QuantGemm caller dtype（G2b 第7孪生）** MED——1-block oracle 已过、须 scale-dependent 才成立。
4. **纯接线（embed→q_proj）** LOW——**BF16 E2E 用同一接线且过**，只有 Q4_0 解码不同。接线被 BF16 证对。

## 判别：扩 oracle 到多 block，别 dump 真实推理
纪律=**可信 oracle 在真实规模**。oracle 框架可信，唯一缺陷=只测 1 block。
- **A（推荐，分级）**：1) QuantGather oracle 扩 hidden=64/96(2-3 block)，各 block 不同 nibble，验跨 block SPLIT 序。失败→多block错序坐实(top)；过→embed 解码全清。
  2) QuantGemm oracle 扩 k=64+(2+ block)、n=5、真实 m，验多 block 累加。
  两者过→两解码路径全清→bug 在接线/dtype(嫌疑2/3)，再移过去。
- **不选 B/C**：B（dump 真实 embed）重入坏探针区(ping-pong 工件, round16)；C（静态读 weight_dtype）是误导 16 轮的代码推理。**扩 oracle 是唯一产出真相的 regime。** 仅当多 block oracle 过后才 B/C，且用 mini-oracle（embed+1 真实规模层隔离）。

## 四问答
Q1 最可能=多 block 解码错序。Q2 嫌疑4>2，且**先扩 oracle 测**（非推理区分）。Q3=选项A。Q4 embed→q_proj stride mismatch=LOW（BF16 同路过）。

## ★元教训（round 20）
结束 18 轮旱情的 oracle 有一缺陷：**N=1**。规则：**隔离 oracle 必须测真实路径用的 multiplicity（多 block、多层），非仅最小 case。1-block 过是必要非充分。**
这正是 saga 最初的"N=1 不触发 increment"教训（layer0-in-loop iter0×0）在 **block 层的复发**。扩到 N≥2 真相自现。

---

# 【新方向·decode 全清（4 oracle），bug 在 op-selection/pack（round 21）】

## 状态：4 oracle 全过（1-block + multi-block, QuantGather + QuantGemm）
decode 数学全清。bug 在接线/dtype。

## 两处 framing 纠正（code fact）
1. **weight_physical_bytes 用 weight_quant_types.get(canonical)（build_graph:345）= op-selection 同源（:114）**，非 tdt.as_quant。
   → stride 计算与 op 选择 by construction 一致：在 map 里→QuantGemm+Q4_0 stride；不在→dense Gemm+dense stride。无两源错配。
2. 非 quant fallback 是 **`numel * tdt().size_bytes()`（:366，dtype-aware）**，非 numel*4（那只在 quant 分支内非32倍数时）。
   Q4_0 q_proj(k=1024%32=0) → n*(k/32)*18 正确 —— **前提：该权重在 weight_quant_types 里。**

## 那个"前提"是全部关键，且 Q4_0-specific by construction
- 层权重在 weight_quant_types → QuantGemm+Q4_0 stride（对）。
- 层权重**缺失**（名不匹配/未填）→ **dense Gemm+dense stride** → Q4_0 raw 字节按 F32 读 → **垃圾**。
- **为何 Q4_0 特有**：BF16 走 dense 是对的（BF16 本就 dense）；Q4_0 走 dense 是灾难。→ 缺失破坏 Q4_0、放过 BF16 = **精确差分**。
- 直连 **round16 missing=21**（21 权重 pack 未解析，名/填充 gap，从未解释）。
- oracle 抓不到：它们证"QuantGemm 用对参数时 decode 对"，测不出真实图给某权重发了**错 op(dense)**。

## 嫌疑排序
1. **TOP：weight_quant_types 层权重填充 gap → 某 Q4_0 权重走 dense Gemm。** Q4_0 特有、连 missing=21、oracle 测不到、最廉价查。
2. **MED：层权重 blob 偏移真实规模。** blob[452160]==GGUF 只验了 **embed 一个点**，层权重(q_proj 等)blob 偏移从未验。
3. **LOW-MED：epilogue/累加 dtype。** oracle 已过 accum_dtype=F32，须 scale-dependent。

## 下一步（优先级，全 build-time 可信、非 scratchpad 探针）
**Step1（最廉价，测 TOP）**：真实 Q4_0 编译图里，对每个层权重 canonical(L0.q_proj/k/v/o/gate/up/down/norms) 打印
`{canonical, weight_quant_types.get()=Some(Q4_0)?/None, 实发 op=QuantGemm|Gemm}`。
- 任何 None 或 dense Gemm → **bug 坐实**（Q4_0 当 dense→垃圾）。likely canonical 名不匹配（executor_compile:556 从 GGUF 外部名经 name_map 键入 vs build_graph 用 L0.q_proj 查）。missing=21 在此解释。
- 全 Some+QuantGemm → op 选择清，进 Step2。

**Step2（测嫌疑2）**：dump blob[L0.q_proj_offset..+18] vs GGUF blk.0.attn_q.weight 首18字节。验层权重 pack（迄今只 embed 验过）。

**Step3（仅 1&2 清后）mini-oracle**：编译**微型真实结构图** QuantGather(hidden=1024)→QuantGemm(m=小,n=5,k=1024)，x86 执行 vs 手算。
测真实规模接线+dtype 传播隔离（可信 x86 执行，无全模型 scratchpad）。须在 op 选择+pack 清后建，否则在上游错 op 潜伏下测接线。

## 三问答
Q1 最有效=Step1（op-selection dump）。option C 直觉对但源错——查 weight_quant_types.get() 非 tdt.as_quant + 实发 op。option D(zero-copy vs dequant) 次要，Step1 dump 也会暴露。
Q2 mini-oracle=微型真实结构图 x86 执行 vs 手算，Step3。
Q3 pack 可能对层权重错——只 embed 验过；Step2 验。但 Step1 更廉价高概率，先做。

## ★元教训（round 21）
oracle 清了 **decode**，结构上**无法**清 **op-selection**（它们假设 op 对）。剩余 bug 在"真实图每权重发什么 op" vs "oracle 假设"的缝隙。
**round16 missing=21 是真未解释信号**，likely 就是线头。Step1 build-time 可信地拉它。

---

# 【根因坐实·QuantGemm A-load dtype=accum(BF16)≠act(F32)，G2b 第7孪生（round 22）】

## oracle 三连（x86 真实执行，可信）
F32accum+F32act→3.0✅ / **BF16accum+F32act→0.0❌（真实配置）** / BF16accum+BF16act→3.0✅。

## paradox 解开：两条 GEMM 路径 dtype 纪律不同
1. **act_dt=DType::F32 硬编码、非模型相关**（build_graph:94）。BF16 与 Q4_0 模型激活都存 F32。
2. **dense Gemm 取 3 个独立 dtype**（gemm_emit:110 a/b/c_dtype），A-load=a_dtype=F32、accum=accumulator_dtype()=F32 → 遵守 ARCH-DTYPE-MIXED-PRECISION → **BF16 E2E 过因走 dense Gemm**。
3. **QuantGemm 塌成单 dtype**=ctx.accum_dtype=compute_dtype=**BF16**（lower_op:195），A-load/B-mul/Fma 全用它 → **Q4_0 失败因 QuantGemm(Q4_0-only) 把 F32 激活按 BF16 读**。
→ 差分坐实：QuantGemm 仅 Q4_0 用；dense Gemm(BF16) 本就对。QuantGemm 从未拿到 a/b/c 分离。

## 修复方向：对但须延伸——accumulator 也错
- 用户"A-load→act_dt" 对，但留 accum=BF16 → A-load=F32+FMA=BF16 = **oracle 从未测的 mix**。
- 须落到 oracle 证过的 **全 F32**（case1=3.0✅）。
- **accum=BF16 本身错**：accumulator_dtype() 全库返 F32（p05/vision_audio:118/telemetry:20）。ctx.accum_dtype=compute_dtype=BF16 是命名/语义混淆（是 compute 非 accumulator）。

## 四问答
1. A-load→act_dt(F32) ✅ **且** accum/FMA→accumulator_dtype()=F32。B(解码Q4_0)已F32。净=QuantGemm CPU 路全 F32=oracle case1。别发 A-load=F32/accum=BF16 mix。
2. 修复面：quant_gemm.inc.rs A-load VecLoad **364,393**(Q4_0/Q4_1) + HighBitMerge **630/676**(Q5/Q6) + Q8_0 等所有 QuantGemm A-load；B-scale VecBinOp:373 + Fma:379 dtype 应 F32。参考 dense gemm_emit:110 a/b/c 分离。
3. ARCH-DTYPE铁律：A-load=激活dtype/B-load=权重dtype/FMA=accumulator(F32)/C-store=输出dtype。QuantGemm 塌成 accum(BF16) 违之。**这正是 dense Gemm 已遵守的铁律**（故 BF16 过）。修=同款 per-role 分离。
4. act_dt 传参：**需加**。emit_quant_gemm_inline 现仅 dtype。caller lower_op:195 有 op.inputs[0]→查其 graph tensor dtype(=act_dt=F32) 传入，镜像 dense 的 a_dtype。别在 emit 内推断（无 graph tensor）。QuantGemm 签名应对齐 dense a/b/c。

## ★saga-lesson caveat：先用 oracle 验修复后配置
修复产生特定 dtype 配置。**E2E 前先加"修复后精确配置(A-load=F32,accum=F32)"oracle 验=3.0**。case1 已证若修复落全 F32 即对。别信未执行配置（终结 18 轮旱情的纪律）。

## ★元教训：第 7 孪生 = 塌 dtype 违宪
G2b 是第 6，本 bug 第 7：dtype 消费者硬编码 accum_dtype 不跟随 per-role dtype。
根模式=A/B/C/accum 塌成单 dtype。应 **lint/断言**：GEMM-family emit 必取独立 a/b/c/accumulator dtype，且 A-load dtype == op.inputs[0] 存储 dtype。此断言本可在 emit 期抓住本 bug + G2b。入 REQ-DTYPE 新规。

---

# 【新方向·6 oracle 过 E2E 仍错，BF16 清共享→Q4_0 特有@真实规模（round 23）】

## 修正 + 两 lead refute
- round22 accum=BF16 判断基于错 dump（compute_dtype=BF16）；真实 emit dtype=F32（accum_dtype=act_dtype.accumulator_dtype()=恒F32，context:95）。修复 behavior-neutral，保留。
- **refute n-loop weight stride**：quant_row_stride=gguf_num_blocks*block_bytes=(1024/32)*18=**576 正确**（moe_emit:840-841）。18432 在 emit_gemm_float_from_plan（float 路，非 Q4_0）。
- **refute Gemv/m_bound**：m_bound=Const(1) if mega_decode_seq_len.is_some() else resolve_sym_dim(spec.m)，**QuantGemm(:181) 与 dense(:12) 同逻辑**。错则 BF16 也错→BF16 过→非此。Gemv dump 应是 decode 相（m=1 正确）非 prefill。

## ★arch-controlled 差分定位（关键 frame）
**BF16 E2E 过 → 清掉所有共享组件**：mega_kernel 集成/28层循环/activation ping-pong 接线/attention/KV/epilogue/QuantGather→q_proj 传递。全 work。
Q4_0 特有面仅：decode(✓6oracle)/op-selection(✓全QuantGemm)/dtype(✓F32)/in-op stride(✓576) —— **全验证，但只在单位规模(m=1,n=1,k=32/64)**。
→ **唯一未执行的 Q4_0 特有 = QuantGemm/QuantGather kernel 在真实规模(m=5,n=2048,k=1024)**。build-time 值对≠执行对（round20 block-multiplicity 同款）。cross-op 共享 BF16→已清→低嫌疑。

## 嫌疑排序
1. **TOP：Q4_0 kernel 真实规模执行(m=5,n=2048,k=1024)。** 唯一未执行 Q4_0 特有轴。单位 oracle 不 exercise 多-m 激活行迭代/多-n 输出写。
2. LOW-MED：cross-op 接线（BF16 过→低）。
3. LOW：attention/KV/epilogue/28层（全共享 BF16）。

## 下一步：分级真实规模 x86 oracle（round20 教训升级）
- **A(TOP)：QuantGemm oracle 真实规模** m=5,n=2048,k=1024，**每行(m)每列(n)不同已知值**使错迭代可见，x86 执行 vs 手算 [5×2048]。
  失败→in-op-at-scale，错位模式定位（跨n错=输出写/n累加；跨m错=a_row_stride 激活行；行内错=k/block）。过→QuantGemm 全清。
- **C(A过后)：两-op mini-oracle** QuantGather(1024)→q_proj QuantGemm(m=5,n=2048,k=1024) x86 vs 手算。测 Q4_0→Q4_0 handoff。失败→cross-op（BF16 过使其意外）；过→bug 在 q_proj 之外，用 BF16-清-共享 frame 重开。
- **不选 B**（真实推理 dump=ping-pong 探针=round16 工件）**不选 D**（接线已 BF16-清）。

## 三问答
Q1 最可能=#1 真实规模 in-op（BF16 清共享使 cross-op 更低非更高）。Q2 A 先，C 仅 A 过后；非 B 非 D。
Q3 "m=1→m=5/n=2048 会暴露吗"：诚实——stride 值已验对，**可能过**；但执行@规模真未测，历次"值对"读判被执行推翻。**执行它别推理它。** A 过=可信清（不同于代码读清）→升 C。分级 oracle 骗不了。

## ★元教训（round 23）
architect 已 ~5 个具体机制假设被后续事实 refute（accum-dtype/n-stride/Gemv...）。模式：**读 build-time 值→判"对"→值确对但没清 bug**。唯一 held 的清是 **x86 执行 oracle**。
纪律：**停止按代码读排序，按"什么从未被执行"排序，执行最大未测面（真实规模 in-op）。** BF16-过是 scope-cut：Q4_0-特有-@规模，执行它。

---

# 【新方向·层权重 blob pack 是唯一未验的 Q4_0 特有面（round 24）】

## 7 oracle 过（含真实规模 m=2,n=4,k=1024）+ emit 参数全匹配
- 之前真实规模 FAIL 是 oracle harness 布局错（col-major vs row-major），修正后过。**教训：oracle "过"=自洽于其手构假设，非=匹配真实推理。**
- 本轮验证：a_row_stride=k*elem、c_row_stride=n*elem（moe_emit:865-866）**real plan=oracle=dense gemm_emit:50 同源**。emit 参数无 gap。
- QuantGather seq 输出前进正确（:405 out_row+=hidden*4=4096/token）。
- refute 本轮所有 lead：strides/seq/handoff-params 全匹配。

## ★唯一从未验的 Q4_0 特有面：层权重 blob pack
blob[452160]==GGUF 只验 **embed（global@offset0，在 per-layer byte_cursor 累加之外）**。
**28 层权重(L0.q_proj/k/v/o/gate/up/down)在其累加 blob 偏移、字节是否匹配 GGUF —— 从未验。** stride 公式对(576/行)，但累加偏移+该处字节未测。
四合一最强嫌疑：
- **Q4_0 特有**：byte_cursor 累加 Q4_0 stride(576)；BF16 用不同 stride → BF16 过不清它。
- **oracle 结构上测不到**：oracle 手喂 weight ptr；真实推理 byte_cursor **计算**偏移。oracle 永久盲区。
- **embed 检查不覆盖**：embed 是 global@0，不在 per-layer 累加。
- **连 round16 missing=21**：层权重若 missing(区留零)或误偏移→QuantGemm 读对格式但错/零位置→垃圾。

## dump 真实推理：要可信的那种
两种"dump 真实推理"：runtime scratchpad 读=round16 ping-pong 工件（禁）；**build-time blob/emit 检查=可信**（blob 编译期静态）。层权重 pack 检查是可信的那种。转过去。

## 嫌疑排序 + 下一步
1. **TOP：验层权重 blob pack。** L0.q_proj：算其 blob 偏移(累加 byte_cursor)，dump blob[off..+18] vs GGUF blk.0.attn_q.weight 首块，**并查非零(missing)**。再验 L14/L27——偏移累加漂移则后层发散(round20 multiplicity 在**层**轴复发)。枚举 missing=21 具体是哪 21 个，若含层权重=bug。
2. MED：cross-op mini-oracle(B)。emit 参数已匹配故低。
3. LOW：其余(BF16 清或已验)。

## 三问答
Q1 最可能=层权重 blob pack（非 A/B/C 已验面的 oracle 扩展）。偏移是 oracle 唯一测不到的 Q4_0 特有物。
Q2 QuantGather→QuantGemm handoff 机制共享 BF16(ping-pong/[seq,hidden]F32/scale 都共享)→已清→低。**weight 侧偏移(byte_cursor)Q4_0 特有且未验=gap。**
Q3 hidden=1024 QuantGather 可能过(blk_ctr 已验、multi-block hidden=64 过)。**weight-offset 轴才是未测的，非 QuantGather block 轴。**

## ★元教训（round 24）
终于收窄的纪律：**"什么 Q4_0 特有 AND oracle 结构测不到 AND embed 检查不覆盖？"→唯一答案=层权重 blob 偏移/pack。**
oracle 手喂指针→**永不能验偏移计算**=oracle regime 永久盲区，故 7 过不闭合。可信 build-time blob dump 是对的仪器；missing=21 自 round16 起就是指向这里的未解释旗标。

---

# 【新方向·组合执行@真实寄存器压力 + 最小复现二分（round 25）】

## blob pack 也验对（round24 TOP 已查）
L0.q_proj offset=215149568、size=1179648(n=2048,k=1024,bs32,bb18)、raw Q4_0 bytes、canonical key=true；missing=21 全是 activation（非 weight）正常。
→ decode/op-select/dtype/blob-pack/emit-params 全验对，BF16 清共享，**E2E 仍乱码**。

## 候选机制（组合专属、Q4_0 专属、oracle 测不到）
finalize_quant scratch_ymm(0/1/2)=ymm13/14/15 硬编码；emit_helpers:317「分配器保留：**仅**高编号 ymm 不分配给**短活跃** VReg」。
QuantGemm accumulator **长活跃**（跨 k=1024 循环）。真实融合核高压下若 acc 落 ymm13-15，quant scratch 循环中途 clobber→错累加→垃圾。
- oracle 单 op 低压→acc 落低 ymm→scratch 不重叠→过。
- BF16 走 dense scratch 路→不用 quant scratch→过。
- **仅真实融合核(197 QuantGemm+活跃 acc)造成的压力逼出重叠。**
= "BF16 清组合"的洞：清 dense 组合，非 quant-scratch-under-pressure。（保留注释歧义，未定论，architect 已错 6 机制）

## ★决定性方法（机制无关）：最小复现 + 二分 argmax
24 轮从未做最基本调试：**缩小失败案例，用唯一从未坏的信号(最终 argmax)二分 onset。**
1. **0 层**（embed→final_norm→lm_head→argmax）真实组合/寄存器/scratchpad，仅少 op。argmax 垃圾/合理？
   垃圾→bug 在 **embed+head**（QuantGather+lm_head QuantGemm）真实组合→微型 2-op 真实图复现→直接调（register 候选在此可查）。
   合理→embed+head 组合对，加层。
2. **二分层数** 1/2/4/8/14/28（~5 跑）。首个垃圾 onset=断点深度。
3. BF16 同截断=合理输出对照。
闭合理由：**首个观测真实失败执行**（真实寄存器/scratchpad/fusion=oracle 无法复现）**用可信信号**（argmax=从未坏，不同于 capture/ping-pong/encode_to_layer）。机制无关，骗不了。

## 五问答
1. 剩余未验面=**真实寄存器压力下的组合执行**（无 oracle 复现）。scratch-reg clobber 是具体候选。
2. QuantGather hidden=1024 较低（blk_ctr 已验；0 层截断更决定性覆盖 embed@真实组合）。
3. 两-op mini-oracle 仍手构（自身低压分配）→可能不复现压力依赖 clobber。**截断真实图严格更优（真实压力）。**
4. embedding_scale=None(Qwen3) 已排除。
5. prefill m=5 由截断跑覆盖（就是真实 prefill）。

## ★元教训（round 25）：跳过 24 轮的招 = 最小复现 + 可信端信号
24 轮探中间值（全坏）+ 验隔离 op（全过），却没做标准调试招：**缩小真实失败案例、用输出信号二分 onset**。中间探针执念挤掉了它。argmax 全程可信。截断真实图→二分→定位深度（或 0 层的 embed+head）→对该最小真实复现套 register-scratch lens。

---

# 【定位·0 层乱码，retract register 假设，lm_head n=151936 未测（round 26）】

## 截断命中 + 两事实
- Q4_0 0 层（embed→final_norm→lm_head→argmax）乱码；BF16 0 层合理。bug 在 tiny 图。
- **tie_word_embeddings=True → lm_head=embed 同一 Q4_0 tensor**（name_map:206-209）。lm_head QuantGemm **n=vocab=151936**，oracle 9 只测 n=2048 → **74× 未测**。
- 0 层无 ActivationSwap 执行（layer_loop_config 只驱动层循环，0 层=0 swap）→ **embed 输出不被 ping-pong 覆盖→dump 可信**（round16 工件不适用）。

## ★retract round25 register-clobber 假设（0 层结果杀死它）
0 层=3 op=**低寄存器压力**，仍失败。round25 假设是**压力依赖** clobber（197 QuantGemm 造压力）→低压仍败反对它。
更紧：**oracle 9(单 QuantGemm,k=1024,acc+scratch_ymm) 过 → within-one-QuantGemm acc-vs-scratch clobber 被 refute**。残余只能 cross-op，3 op 下弱。**architect 提的假设被 0 层证据 demote，诚实纠正。**

## 领先假设（都具体、无 register 理论）
1. **lm_head QuantGemm @ n=151936**——唯一真未测规模(74×)。候选：large-n 偏移算术（weight_row_ptr 累进 151936×576=87.5MB，查 i32 截断）或仅大 n 可见的 per-row 迭代 bug。
2. **QuantGather 输出@真实组合**——oracle8 隔离过(hidden=1024)，组合可能不同；split 测定。

## 验证手段排序
1. **A(TOP)：0 层跑 dump embed(QuantGather) 输出 vs 手算。** 已确认可信(0 层无 swap)。机制无关 split：
   embed 错→QuantGather 组合坏→localize；embed 对→final_norm(dense/BF16 共享,不太可能)或 lm_head→再 dump lm_head 输入(=normed)，对→**lm_head QuantGemm @ n=151936 是 bug**。
2. **扩 oracle 9 到 n=151936**（非 2048）。失败→隔离复现 lm_head bug→可信可调。直测假设1。
3. C(ymm 保留检查)demote——仅 A/B 定位到 within-op register 才查(oracle9 过使其不太可能)。静态低优。
4. D(2-op mini-oracle)低——手构，不优于可信 0 层 split。

## 四问答
1. scratch clobber **retract**。0 层低压仍败 + oracle9 过 refute 压力/within-op 形式。非 lead。
2. 最高压 op moot（压力假设撤）。lm_head 工作量最大(n=151936)。
3. 排序：A→B→扩 oracle 到 n=151936→(C 静态低)→(D 低)。
4. 必然 register 压力？**否，撤**。最可能 lm_head n=151936 未测规模，或 QuantGather 组合细节。**split 测 A 机制无关地告诉哪个**，别先定机制。

## ★元教训（round 26）
0 层截断不仅定位——其**低寄存器压力 falsify 了 architect 自己的领先机制**，tie 事实曝出**具体 74× 未测规模(lm_head n=151936)**无 oracle 触及。
纪律：**让机制无关的 reduction(截断+可信信号)选靶，别预设机制**——architect 上轮预设 register-clobber 被数据 refute。做 split A，再扩 oracle 到真实 lm_head n。

## ★差分实测结果（2026-07-09，arch 控、never-broken E2E 信号类）

| 模型 | 量化 | 输出 | 结果 |
|------|------|------|------|
| Qwen3-0.6B | **BF16** | "Paris. The capital of Italy is Rome. The" | ✅ PASS（首 token "Paris" 正确）|
| Qwen3-0.6B | **Q4_0** | "estr%Bpapers药业 CapcomdanaaysrongEAR','','" | ❌ FAIL（完全乱码）|

**结论**：BF16 对 + Q4_0 错 = **bug 100% Q4_0-quant 特有**，arch 已控。
- 同架构（Qwen3-0.6B）、同 prompt（"The capital of France is"）、同生成参数（max_tokens=10, temperature=0.0）、同 loader
- 唯一变量 = 量化格式（BF16 vs Q4_0）
- BF16 路径（Gather + 标准精度 GEMM）端到端正确 → Qwen3 架构/图构建/executor/JIT 非 quant codegen 全部正确
- Q4_0 路径（QuantGather + QuantGemm）端到端乱码 → bug 必在 Q4_0 quant codegen 或其数据接线

**进入 priority 2**：缩到 Q4_0 dequant 运行时（共用 DecodeTraceBuilder / QuantGemm+QuantGather）。
关键认知重申："QuantGemm 正确（layer0=1.0）" 之前唯一证据来自坏 layer0 探针（encode_to_layer 返回空 → cos=0.0000 是空返回签名，非计算正确）。所以 **QuantGemm/共用解码器现在 UNVERIFIED**，重回嫌疑。

## ★解码顺序核查（2026-07-09，JIT DecodeTraceBuilder vs scalar classic.rs）

**Q4_0 nibble 布局权威**：llama.cpp = **SPLIT**（byte j: lo=elem[j], hi=elem[j+16]）。

| 代码路径 | 布局 | 来源 | 与标准一致 |
|---------|------|------|-----------|
| JIT DecodeTraceBuilder `QuantConcatSeq` | **SPLIT** | quant_decode.rs:670 `low_first=true` → `QuantConcatSeq{lo,hi}`; x86 lower (lower_instr_dispatch.inc.rs:3905-3918) `vinserti128` dst[0..127]=lo, dst[128..255]=hi → out[0..7]=lo, out[8..15]=hi | ✅ |
| JIT quant_gemm.inc.rs Assisted GEMV | **SPLIT** | :347 注释 `byte_i → lo=block_pos[i], hi=block_pos[16+i]`; :353 `half_block_elem=block_size/2*elem` 跳 hi 区 | ✅ |
| scalar classic.rs `dequant_q4_0` | **INTERLEAVED** | classic.rs:13-28 `out[2i]=lo, out[2i+1]=hi`; tests_quant.rs:257 注释 `qs[0]=0x12→lo=2,hi=1→out[0],out[1]` | ❌ 偏离标准 |

**关键判定**：
- **JIT 解码顺序正确**（SPLIT，与 llama.cpp 一致）→ 解码顺序不是 Q4_0 bug 源
- scalar classic.rs INTERLEAVED 偏离标准，但 NO-SCALAR 铁律禁止运行时调 scalar → 不影响运行时 E2E
- scalar `test_dequant_q4_0_known_values` 过 = 自洽验证 INTERLEAVED 语义（循环论证，非对真实 GGUF 正确），不能作为 dequant 数学对的证据
- **"QuantGemm/共用解码器 UNVERIFIED" 状态更新**：解码 trace 结构 + SPLIT 布局已核查正确；但 **JIT 执行数值仍未验证**（无隔离 oracle 跑编译后 JIT 对比 llama.cpp 参考值）

**结论**：bug 不在解码顺序。下一步 = JIT 隔离 oracle（跑编译后 Q4_0 dequant 对比真实 GGUF block 的 llama.cpp 参考值），或在接线（喂 dequant 的 offset/ptr）。

## ★★JIT 隔离 oracle 实测（2026-07-09，定位 bug 在 emit/lower）

**测试**：`test_q4_0_quant_gather_x86_oracle`（quant_gemv_tests.inc.rs，x86 真实执行 mmap+RWX）
- hidden=32, vocab=2, token 0, d=f16(1.0), qs[0]=0x9A(lo=10,hi=9), 其余 qs=0x88(zero)
- 手算 SPLIT 参考：elem[0]=2.0, elem[16]=1.0, 其余=0.0

**实测输出**（out[0..8]）：
```
out[0]=2.0, out[1]=0.0, out[2]=0.0, out[3]=0.0, out[4]=1.0, out[5]=0.0, out[6]=0.0, out[7]=0.0
out[8..32] = 全 0.0
```
- out[0]=2.0 ✅（byte0 lo nibble，SPLIT 正确位置）
- **out[4]=1.0 ❌**（byte0 hi nibble 应在 out[16]，实际落在 out[4]）
- out[16]=0.0 ❌（应为 1.0）

**根因定位**：`QuantConcatSeq` x86 lower（vinserti128, lower_instr_dispatch.inc.rs:3905-3918）产出 **8-lane YMM 内布局 [lo[0..3], hi[0..3]]**（低 128=lo 前4, 高 128=hi 前4），**不是 SPLIT 的 [lo[0..7]]**。

- vinserti128 语义：YMM 高 128 bit（=f32[4..7]）= hi 的低 128 bit（=hi[0..3]）
- → dst = [lo[0],lo[1],lo[2],lo[3], hi[0],hi[1],hi[2],hi[3]]
- 这是 **4-element 块内 interleave**，非 SPLIT 的 8+8

**块循环推断**（待确认）：block_size=32, lanes=8。每 sub_block 处理 8 元素。
- 第 1 次 sub_block: out[0..7] = [lo[0..3], hi[0..3]]（byte0..3 的 lo+hi）
- 第 2 次: out[8..15] = [lo[4..7], hi[4..7]]（byte4..7 的 lo+hi）
- → 输出顺序 = byte0lo,byte1lo,byte2lo,byte3lo, byte0hi,byte1hi,byte2hi,byte3hi, byte4lo..byte7lo, byte4hi..byte7hi
- 但 SPLIT 要 = byte0lo..byte7lo (elem[0..7]), byte0hi..byte7hi (elem[8..15]), 然后 byte8..15 同理

**结论**：bug 在 `QuantConcatSeq` x86 lower 或其与 QuantBlockLoad/块循环的配合。vinserti128 的 4+4 块内拼接产不出 SPLIT 的 8+8 布局。
- BF16 E2E 对：BF16 走 Gather（非 QuantGather）+ 标准 GEMM，不走 QuantConcatSeq → 不受影响 ✓
- Q4_0 E2E 错：走 QuantGather + QuantConcatSeq → 块内布局错 → embed dequant 全错 → 下游全乱 ✓

**下一步**：确认 QuantBlockLoad 加载宽度（8 字节？4 字节？）+ 块循环步进，设计正确的 SPLIT 布局 emit（可能需改 QuantConcatSeq lower 或块循环结构）。

## ★★architect round 19 裁决（2026-07-09，确认根因 + 修复方向）

**architect 确认根因坐实**：
- `lower_quant_concat_seq_x86`（3903-3914）注释自己承认 `dst = [lo[0..3], hi[0..3]]`
- 配合 `data_byte_advance=lanes/2=4` + 4 sub_block → 输出 [lo0..3,hi0..3, lo4..7,hi4..7,...] = **4-lo-4-hi 块交织**，非 SPLIT [lo0..15, hi0..15]
- oracle out[4]=1.0（应在 out[16]）精确吻合
- quant_decode.rs:657 注释「result = interleave(lo,hi)」→ 不是 QuantConcatSeq 失灵，是 **DecodeTraceBuilder 有意 emit concat/interleave 语义，对 SPLIT-layout Q4_0 是错的**
- 这是 ~round7 architect flag 过的 interleave-vs-SPLIT 类，当时被 884x 探针工件误导而误弃。**oracle 终于显出真签名：一个 magnitude-preserving 置换**（值对、位置错）

**修复方向 = Option B**（改 DecodeTraceBuilder PackedNibbles 两阶段 SPLIT）：
- ❌ Option A（只改 x86 lower）不足：8-lane YMM 装不下 16 元素，且整个 sub_block 结构（advance=4、4 次、输出偏移）都围绕 4+4 concat 建
- ✅ Option B：lo pass→elem[0..half]、hi pass→elem[half..block]，对齐 SPLIT，镜像 QuantGemm Assisted 已验证的两阶段。改 trace（去 concat）+ 输出偏移计算
- Option C 是 B 在循环层的表达，同效

**修复面**：
- 所有 PackedNibbles 格式（Q4_0/Q4_1/Q5_0/Q5_1）都用 QuantConcatSeq，都同病，**必须一起修**（禁 Q4_0 专属分支）
- QuantGemm 不走此路（用 QuantDequantFma 微内核 x86_lower:3858 分流，两阶段 Assisted）→ **层权重解码不受影响**
- 修复 scoped 在 gather/DecodeTraceBuilder
- 更新锁死 bug 的测试：quant_decode.rs:950「Q4_1 should have QuantConcatSeq」
- 查 NibbleWithHighBits（Q5_0/Q5_1/Q6_K，advance 同 lanes/2）是否同病
- aarch64/GPU 的 QuantConcatSeq lower：trace 停发后变死代码或需同步

**充分性 caveat（关键）**：
- 修 QuantGather（embedding）是**必要的**（坏 embedding 毒害 E2E），但**未必充分**
- "QuantGemm 对"现在 UNVERIFIED（唯一证据是坏 layer0 探针）
- QuantGemm 走不同路（QuantDequantFma），可能有它**自己的** SPLIT bug
- **别假设修完 QuantGather 就修好 E2E**

**修复序列**（architect 指定）：
1. 修 QuantGather（Option B 两阶段 SPLIT，覆盖所有 PackedNibbles）
2. **给 QuantGemm 建同款 x86 oracle**（喂已知 Q4_0 权重块，GEMV 输出对手算 SPLIT）—— 层从未得到的可信验证
3. 重跑 Qwen3-Q4_0 E2E。对→完成；仍错→QuantGemm 另有 bug，用其 oracle 抓

**元教训**：x86 真实执行隔离 oracle 是 round 1 就该有的仪器。scratchpad-free、真执行、手算自校准，免疫 18 轮所有工件（capture stride / ping-pong 覆盖 / encode_to_layer 空 / GLLM_SINGLE_LAYER no-op）。规则：**每条 quant 解码路径都需真实执行的已知值 oracle 测试**。最贵一课：探针数值污染会让正确的机制判断被丢掉（interleave/split 机制判断本对，被 884x 误导误弃）。

## ★★修复完成（2026-07-09，Option B 两阶段 SPLIT 落地）

**修改文件**（3 生产 + 测试更新）：
1. `gllm-kernels/src/compiler/codegen/vm/quant_decode.rs`:
   - 加 `NibblePhase` enum (Lo/Hi) + `DecodeTraceBuilder::nibble_phase` 字段 + `with_nibble_phase()` + `needs_two_phase_split()`
   - `emit_data_load` PackedNibbles: byte_count 从 `output_lanes/2` 改 `output_lanes`（每趟读 lanes 字节）
   - `emit_unpack` PackedNibbles: 按 phase 产纯 lo (`QuantAndMask(0x0F)`) 或纯 hi (`ShiftRight(4)+AndMask`)，**去掉 QuantConcatSeq**
2. `gllm-kernels/src/compiler/codegen/vm/quant_gather_emit.rs`:
   - trace 构建: PackedNibbles 建 lo_trace + hi_trace（两阶段）
   - sub_block 循环: 拆 Lo pass（store out[blk_off+0..half]）+ Hi pass（data_ptr 重置到 block 起点, store out[blk_off+half..block]）
   - `sub_blocks_per_phase = sub_blocks/2`（两阶段每趟减半）
3. `gllm-kernels/src/compiler/codegen/vm/quant_offset_dsl.rs`:
   - `derive_data_byte_advance` PackedNibbles: `lanes/2` 改 `lanes`（每趟前进 lanes 字节）
4. 测试更新: 7 个断言 QuantConcatSeq/QuantShiftRight 的测试改为两阶段语义（Q4_1×2, Squeeze, TQ2_0×2, output_lanes_variation, squeeze_unpack）

**验证**：
- `test_q4_0_quant_gather_x86_oracle` ✅ PASS（out[0]=2.0, out[16]=1.0, 其余=0.0）
- `test_q4_0_quant_gather_x86_oracle_pattern_diag` ✅ PASS
- 全量回归 `cargo test --lib`: **7050 passed, 0 failed**（Q5/Q8/K-Quant/IQ/Squeeze/TQ2_0 全不坏）
- QuantGemm 路径未触碰（QuantDequantFma），层权重解码不受影响

**修复面确认**：
- PackedNibbles 格式（Q4_0/Q4_1/Squeeze/TQ2_0）全修（禁 Q4_0 专属分支 ✓）
- NibbleWithHighBits（Q5_0/Q5_1/Q6_K）不同路径，未触碰，回归过
- aarch64/GPU 的 QuantConcatSeq lower: PackedNibbles 不再 emit 它，可能变少用但非死代码（其他格式可能用），编译通过

**下一步**（architect 序列）：
1. ✅ 修 QuantGather（完成）
2. ⏳ 给 QuantGemm 建同款 x86 oracle（层从未得到的可信验证，"QuantGemm 对"UNVERIFIED）
3. ⏳ 重跑 Qwen3 Q4_0 E2E

## ★★Q4_0 E2E 仍 FAIL（2026-07-09，architect 充分性 caveat 命中）

**修复后 Qwen3 Q4_0 E2E 输出**：`"hed']?></ unchangedownt明珠 dictsapus …\n\n lashesEAR"`

- 修复前：`"estr%Bpapers药业 CapcomdanaaysrongEAR"`
- 修复后：`"hed']?></ unchangedownt明珠 dictsapus …\n\n lashesEAR"`
- **输出改变了**（QuantGather 修复确实影响推理），但**仍乱码**

**architect round 19 充分性 caveat 命中**：
> "修 QuantGather 是必要的，但未必充分。QuantGemm 走不同路（QuantDequantFma），可能有它自己的 SPLIT bug。别假设修完 QuantGather 就修好 E2E。"

**结论**：
- QuantGather（embed）SPLIT bug 已修（oracle PASS）
- QuantGemm（层权重）走 `QuantDequantFma` 微内核（quant_gemm.inc.rs Assisted GEMV），**未验证**
- 注释说它用两阶段 SPLIT（half_block_elem 偏移），但注释≠代码正确
- embed 修好了（输入对），但 28 层 GEMM 若 dequant 错 → 输出仍乱

**进入 architect 序列第 2 步**：给 QuantGemm 建同款 x86 oracle（喂已知 Q4_0 权重块，GEMV 输出对手算 SPLIT）。这是层从未得到的可信验证。

## ★★QuantGemm oracle PASS（2026-07-09，层 dequant 验证正确）

**测试**：`test_q4_0_quant_gemm_x86_oracle`（quant_gemv_tests.inc.rs）
- m=1, n=1, k=32, weight d=1.0 qs[0]=0x9A(lo=10,hi=9), act[0]=1.0 act[16]=1.0
- SPLIT 手算: weight elem[0]=2.0, elem[16]=1.0 → output = 1.0×2.0 + 1.0×1.0 = 3.0

**实测**：`out = [3.0]` ✅ **PASS** — QuantGemm SPLIT 正确，层 dequant 对。

**关键结论**：
- QuantGather（embed）✅ 已修（SPLIT 对，oracle PASS）
- QuantGemm（层权重）✅ SPLIT 对（oracle PASS，从未坏过）
- **两个 Q4_0 dequant 路径都对**

但 Q4_0 E2E 仍乱码（输出从 "estr%Bpapers" 变 "hed']?></..."，QuantGather 修复影响了推理但未根治）。

**architect 充分性 caveat 更新**：
- 原担心"QuantGemm 可能有 SPLIT bug" → **oracle 证伪，QuantGemm 对**
- bug 不在 QuantGather/QuantGemm 的解码本身
- E2E 乱码根因在**别处**：可能是接线（QuantGather 输出喂给下游的方式）/ 混合精度 dtype 传播 / G2b（GEMM A-load dtype 硬编码）/ 其他

**下一步**：请 architect 分析。两个 dequant oracle 都过 + BF16 E2E 过 + Q4_0 E2E 错 → bug 在 quant 路径的**接线/集成**，非解码数学。需新方向定位。

## ★★4 oracle 全过 — bug 确定在接线/dtype 非 解码数学（architect round 20）

**4 个 x86 真实执行 oracle 全 PASS**：
1. QuantGather 1-block (hidden=32): out[0]=2.0, out[16]=1.0 ✅
2. QuantGather multi-block (hidden=64): out[0]=2.0, out[16]=1.0, out[32]=-2.0, out[48]=3.0 ✅
3. QuantGemm 1-block (k=32): out=3.0 ✅
4. QuantGemm multi-block (k=64): out=4.0 ✅

**architect round 20 top 嫌疑（多 block 错序）排除**：
- QuantGather 跨 block blk_ctr 乘子对（out[32]/out[48] 落正确位置）
- QuantGemm 多 block SPLIT 累加对（2 block 4 elem 求和 = 4.0）

**结论（architect Option A 第 3 步触发）**：
- 两条 Q4_0 dequant 路径（QuantGather embed + QuantGemm 层权重）在 1-block 和 multi-block 全部正确
- **bug 不在解码数学**，在**接线/dtype 传播**（architect 嫌疑 2/3）
- 嫌疑排序更新：
  - ~~嫌疑 1 多 block 错序~~ ❌ 排除
  - **嫌疑 2 Q4_0 weight_stride/dtype 传播** — 现为 TOP
  - **嫌疑 3 QuantGemm caller dtype（G2b 第 7 孪生）** — MED
  - ~~嫌疑 4 纯接线~~ LOW（BF16 同接线且过）

**下一步（architect 指引）**：
- oracle regime 已穷尽解码数学验证
- 移到嫌疑 2/3：需用 **mini-oracle**（embed + 1 个真实规模层隔离）或查 dtype 传播
- architect: "只有当多 block oracle 过之后，才用 B/C（dump 真实 embed / 静态读 weight_dtype），且要用 mini-oracle"
- G2b 知识库: GEMM A-load dtype 硬编码 ctx.accum_dtype（lower_op:1365）。Q4_0 场景激活 F32 + accum F32 → A-load 对。但 **weight_dtype 传播**（QuantGemm 的 weight block stride/decode dtype）需查

**元教训（architect round 20）**：oracle 必须测真实 multiplicity（多 block）。1-block 过是必要非充分。这是 saga 开头"N=1 不触发 increment"教训在 block 层级的复发。

## ★★★根因坐实：QuantGemm accum_dtype=BF16 时 FMA 输出 0（2026-07-09）

**BF16 accum oracle 测试**（test_q4_0_quant_gemm_x86_oracle_bf16_accum）：
- 同样的 Q4_0 weight + activation（F32 accum 时 output=3.0 ✅）
- 改 accum_dtype=BF16（真实推理配置）→ **output=0.0** ❌

**真实推理配置**（dump 确认）：
- `geometry.dtype=BF16, compute_dtype=BF16, needs_dtype_conversion=false`
- `accum_dtype = compute_dtype = BF16`（context.inc.rs:105）
- Q4_0 权重走 raw quant ptr（line 168-172，未 dequantize）
- 所有层权重走 QuantGemm（op-selection 全对，0 个 DENSE dump）

**根因链**：
1. Qwen3 Q4_0 推理: compute_dtype=BF16 → accum_dtype=BF16
2. QuantGemm emit 用 dtype=BF16（= accum_dtype）
3. Q4_0 decode 出 F32 值 → BF16 accum FMA 累加
4. **BF16 accum 路径下 FMA 输出 0**（oracle 实测 0.0 vs F32 accum 3.0）
5. 28 层 GEMM 全 0 → logits 全 0/乱 → E2E 乱码

**为何 BF16 E2E 过**：BF16 权重 + BF16 accum → 走 dense Gemm（非 QuantGemm），dense Gemm 的 BF16 路径对。Q4_0 走 QuantGemm，BF16 accum 路径坏。

**为何 oracle 之前漏**：4 个 oracle 全用 F32 accum_dtype（没考虑真实推理是 BF16 accum）。architect round 20 元教训"测真实 multiplicity"的又一个维度：**accum_dtype 也要测真实配置**。

**下一步**：定位 BF16 accum 下 QuantGemm FMA 为何输出 0。嫌疑：
- decode Q4_0 出 F32 后 VecNarrow 到 BF16 丢信息?（但 2.0/1.0 BF16 可表示，不丢）
- BF16 FMA 累加器初始化/类型问题?
- QuantGemm emit 在 dtype=BF16 时某条 VmInstr 路径错?

## ★★★最终根因坐实：QuantGemm A-load dtype 用 accum_dtype(BF16) 但 activation 存 F32（2026-07-09）

**oracle 三连验证**：
1. F32 accum + F32 act → output=3.0 ✅
2. BF16 accum + F32 act → **output=0.0 ❌**（错误配置）
3. BF16 accum + BF16 act → output=3.0 ✅

**真实推理配置**（code fact 确认）：
- `act_dt = DType::F32`（build_graph.inc.rs:94，激活存储 dtype = F32）
  - 注释明确："禁止 act_dt = config.compute_dtype。compute_dtype=BF16 语义是'权重 BF16'，非'激活 BF16 算'"
  - "act_dt = CPU 计算精度派生（当前 = F32）。CPU FMA 恒 F32"
- `accum_dtype = compute_dtype = BF16`（context.inc.rs:105）
- `geometry.dtype=BF16, compute_dtype=BF16`（dump 确认）

**违宪点**：`quant_gemm.inc.rs:364`
```rust
prog.emit(VmInstr::VecLoad { dst: a_val, base: input_ptr, offset: lo_act_off, width, dtype, predicate: None });
```
`dtype` = emit 参数 = accum_dtype = BF16。但 activation 内存是 F32（act_dt=F32）。
→ **BF16 加载 F32 内存**：每 4 字节当 2 个 BF16 读 → a_val 读错 → FMA 错 → output=0 → 28 层全 0 → E2E 乱码

**G2b 知识库关联**：
- G2b 知识库（g2b-root-cause-a_dtype-load-stride.md）已记录：GEMM A-load dtype 硬编码 ctx.accum_dtype（lower_op.inc.rs:1365），不跟随激活张量实际存储 dtype
- G2b 场景：act_dt=BF16 时激活按 2B 存按 4B 读（A-load=F32，act=BF16）
- **本 bug 是 G2b 的对偶**：act_dt=F32 时激活按 4B 存按 2B 读（A-load=BF16，act=F32）
- 同一违宪（A-load dtype 不跟随 activation 实际 dtype），两个方向都出错

**为何 BF16 E2E 过**：
- BF16 权重 → 走 dense Gemm（非 QuantGemm）
- dense Gemm 的 A-load 用 act_dt（F32）？或 dense Gemm 的 dtype 处理不同
- 需确认 dense Gemm A-load dtype（可能 dense Gemm 正确用 act_dt，QuantGemm 错用 accum_dtype）

**为何 4 个原 oracle 漏**：
- 全用 F32 accum + F32 act（accum_dtype=act_dt=F32，巧合一致）→ 没暴露 accum≠act 的错误配置
- architect round 20 元教训"测真实 multiplicity"再加一个维度：**accum_dtype vs act_dt 要测真实配置（BF16 accum + F32 act）**

**修复方向**（待 architect 确认）：
- QuantGemm A-load（a_val VecLoad）dtype 应该用 **act_dt（F32）** 而非 accum_dtype（BF16）
- 即 activation 加载用激活存储 dtype，FMA 累加用 accum_dtype
- 这是 ARCH-DTYPE-MIXED-PRECISION 铁律：A-load 用激活 dtype，B-load 用权重 dtype，FMA 用 accumulator dtype，C-store 用输出 dtype
- G2b 知识库已标注修复方向（A-load 从 op.inputs[0] tensor dtype 推断，非 ctx.accum_dtype）

## ★★★architect round 22 裁决：QuantGemm per-role dtype 分离（A-load=act_dt=F32, accum=F32）

**paradox 解开**（为何 BF16 过 Q4_0 挂）：
1. `act_dt=F32` 硬编码（build_graph:94），BF16/Q4_0 激活**都**存 F32
2. dense Gemm 已做 per-role dtype 分离（gemm_emit.rs:110 `a_dtype/b_dtype/c_dtype`）→ BF16 过
3. QuantGemm 把所有 dtype 塌成单一 `dtype`=accum_dtype=BF16 → Q4_0 挂（F32 激活按 BF16 读）

**修复方向**（architect 确认）：
- A-load（a_val VecLoad）→ **act_dt(F32)**（从 op.inputs[0] tensor dtype 推断）
- accum/FMA → **accumulator_dtype()=F32**（非 BF16！accum_dtype=BF16 是命名混淆，真 accumulator 全库返回 F32）
- B（Q4_0 解码后）已是 F32
- 净结果：QuantGemm CPU 路径**全 F32**（= oracle case 1, 3.0✅）
- **禁止** A-load=F32 + accum=BF16 mix（未测配置）

**修复面**：
- quant_gemm.inc.rs Assisted GEMV（Q4_0/Q4_1）: line 364, 393 A-load VecLoad dtype
- HighBitMerge（Q5/Q6）: line 630, 676 A-load
- Q8_0 等其他 QuantGemm A-load
- B-scale VecBinOp:373 + Fma:379 dtype → F32
- 参考 dense gemm_emit.rs:110 的 a/b/c 分离

**act_dt 传参**：
- emit_quant_gemm_inline 加 act_dt 参数（当前只有 dtype=accum_dtype）
- caller lower_op:195 从 op.inputs[0] graph tensor dtype 推断 act_dt 传入
- 镜像 dense Gemm 的 a_dtype

**修复后验证序列**（architect 强调）：
1. 先加"修复后精确配置(A-load=F32, accum=F32)" oracle 验 =3.0（已有 case 1 证明）
2. 再跑 Qwen3-Q4_0 E2E

**元教训**：同一违宪第 7 个孪生（G2b 第 6）。根模式 = 把 A/B/C/accum 塌成单一 dtype。应做成 lint/断言：GEMM-family emit 必须取独立 a/b/c/accumulator dtype，A-load dtype = op.inputs[0] 存储 dtype。进 REQ-DTYPE 新规。

## ★架构假设修正：真实 accum_dtype=F32（非 BF16），oracle 真实配置全过但 E2E 错（2026-07-09）

**dump 修正**（emit_quant_gemm_tiled 实际参数）：
```
quant=Q4_0 n=2048 k=1024 dtype=F32 dot_cap=SimdAssisted kernel=Assisted mode=Gemv
```
- **dtype(accum_dtype)=F32**（非 BF16！）
- kernel=Assisted（oracle 修的路径）✓
- dot_cap=SimdAssisted ✓

**accum_dtype 真实派生**（context.inc.rs:95-105）：
```rust
let compute_dtype = act_dtype.accumulator_dtype();  // BF16→F32, F32→F32
accum_dtype: compute_dtype  // 恒 F32!
```
- accum_dtype = act_dtype.accumulator_dtype() = **恒 F32**
- 之前 architect round 22 "accum=BF16" 判断基于 geometry.compute_dtype=BF16 dump, 但 emit 实际收 F32

**per-role dtype 修复影响**：
- 真实 accum=F32 时, act_dt=F32, acc_dtype=F32.accumulator_dtype()=F32 → 全 F32
- 原 code 也全 F32（dtype=F32）→ **修复在真实配置下零行为变化**
- BF16 accum oracle(output=0)暴露的 bug 在真实推理不触发（accum 恒 F32）

**新矛盾**：
- 6 oracle 全过（含 F32 accum 真实配置 + Assisted kernel + multi-block）
- 真实推理同配置（F32 accum + Assisted + Q4_0）
- 但 E2E 错
- 差异在**规模/接线/集成**：oracle 单 op(m=1,n=1,k=32/64), 真实 m=1024/2048, n=5, k=1024, 28层, fusion/epilogue

**修复保留**：per-role dtype 分离是正确架构改进(ARCH-DTYPE-MIXED-PRECISION), 即使真实 F32 下零行为变化。BF16 accum oracle 保留(验证 BF16 路径对, 防回归)。

**下一步**：bug 在规模/接线/集成, 非 emit 数学。需新方向。

## ★★★真实规模 oracle 定位：General 模式 m 迭代 bug（2026-07-09，architect round 23 命中）

**真实规模 oracle**（test_q4_0_quant_gemm_x86_oracle_realscale, m=2,n=4,k=1024）：
- output = [0,1,2,3, 0,0,0,0]（want [0,1,2,3, 1,2,3,4]）
- **m=0 行对，m=1 行全 0**
- 错位模式 = 跨 m 错（architect round 23 预测"a_row_stride 激活行迭代"）

**bug 定位**：quant_gemm.inc.rs:454-460 General 模式
```rust
GemmMode::General => {
    prog.emit_loop_try(plan.m_bound.clone(), 1, |prog, _i_ctr, i_cnt| {
        let weight_row_ptr = ...;
        prog.emit(VmInstr::GprBinOp { dst: weight_row_ptr, a: weight_ptr, b: GprOperand::VReg(zero_gpr), op: GprOp::Add });
        // ↑ weight_row_ptr = weight_ptr + 0, 每次 m 迭代都固定! 没随 m 前进
        do_m_block(prog, i_cnt, weight_row_ptr)
    });
}
```
- weight 布局 [m, n, blocks], m 维 stride = n × quant_row_stride
- General 模式 weight_row_ptr 应随 m 前进 `m × n × quant_row_stride`, 但代码固定 weight_ptr+0
- → 所有 m 迭代读同一 weight 行 (m=0)

**但 output m=1 全 0（非 m=0 的值）**, 说明还有 activation 行问题:
- i_cnt 用于 activation 行偏移 (a_row_stride × i_cnt)
- 若 a_row_stride 错或 i_cnt 没正确驱动 activation 行 → m=1 读错 activation → 0

**真实推理影响**:
- prefill m=seq_len=5 → 走 General → 触发此 bug → 首 token 就错 → E2E 乱码
- decode m=1 → 走 Gemv → 不触发
- Q4_0 E2E 生成第一个 token 就错 (prefill 阶段) = General 模式 bug

**为何 BF16 E2E 过**: BF16 走 dense Gemm (gemm_emit.rs), dense Gemm 的 General 模式 weight_row_ptr 正确前进. QuantGemm 的 General 路径有此 bug.

**下一步**: 修 General 模式 weight_row_ptr 随 m 前进 + 确认 activation 行迭代正确. 需查 dense Gemm General 模式作参考.

## 真实规模 oracle PASS（修正：oracle act 布局错，非 emit bug）

**修正**：之前 oracle 用 act [k,n] col-major 是错的。GEMM act 应 [m,k] row-major。
- a_row_stride = k × elem = 1024 × 4 = 4096（moe_emit:865）
- act[m_idx][k_idx] at act[m_idx*k + k_idx]
- 修正后: act[0*k+0]=1, act[1*k+0]=1

**真实规模 oracle PASS**（test_q4_0_quant_gemm_x86_oracle_realscale, m=2,n=4,k=1024）：
- output = [0,1,2,3, 0,1,2,3] ✅（weight elem[0]=ni, act=1 → output=ni）
- QuantGemm 真实规模全对

**7 oracle 全 PASS**（6 单位 + 1 真实规模）：
- QuantGather: 1-block, multi-block, pattern-diag ✅
- QuantGemm: 1-block, multi-block, BF16-accum, real-scale(m=2,n=4,k=1024) ✅

**结论**：QuantGather + QuantGemm 在所有规模/配置全对。emit 数学 + 接线全清。
但 Q4_0 E2E 仍错。

**下一步**（architect round 23 Option C）：两 op mini-oracle
- QuantGather(hidden=1024) → q_proj QuantGemm(m=?, n=1024, k=1024)
- x86 执行 vs 手算
- 测 Q4_0→Q4_0 handoff（QuantGather 输出喂给 QuantGemm 的 activation）
- 这是 oracle regime 最后未测面：跨 op 集成

**元教训**：oracle 构造时 act 布局必须匹配 emit 假设（[m,k] row-major, a_row_stride=k*elem）。错误的 oracle 布局会假阳性 FAIL。

## architect round 25：停止猜机制，最小复现 + argmax 二分（2026-07-10）

### 已完成进展（commit d32e18e5）
- QuantGather PackedNibbles SPLIT 修复（两阶段，7 oracle 验证）
- QuantGemm per-role dtype 分离
- 7055 回归过
- blob pack 验证对（offset/size/数据/ext_ptrs 全对）

### 但 E2E 仍乱码

### architect round 25 决定性方法
停止猜机制（已错过 6 个）。用**最小复现 + 可信信号(argmax)二分**：
1. 截断真实 Q4_0 图: 0层(embed→final_norm→lm_head→argmax). 真实寄存器/scratchpad/fusion.
   - argmax 垃圾 → bug 在 embed+head 真实组合 → 微型 2-op 真实图复现
   - argmax 合理 → 加层
2. 二分层数 1/2/4/8/14/28, 首个垃圾 onset = 断点深度
3. BF16 同截断 = 对照

### 候选机制（组合专属，未确认）
scratch 寄存器 clobber: emit_helpers.inc.rs:317 ymm13/14/15 保留给短活跃 VReg.
QuantGemm accumulator 长活跃(跨 k=1024 循环). 真实融合核高压下若 acc 落 ymm13-15,
quant scratch_ymm(0/1/2) 循环中途 clobber 它 → 错累加 → 垃圾.
- 隔离 oracle 近零压力 → acc 落低 ymm → 不重叠 → 过(漏!)
- BF16 走 dense scratch 路径, 不用 quant scratch → 过
- 只有真实融合核(197 QuantGemm + 长活跃 acc)压力逼出重叠

### 元教训
24 轮跳过了标准调试招: 缩小真实失败案例 + 输出信号二分 onset.
中间探针执念挤掉了它. argmax 全程可信.

### 下一步
实施截断测试: 改 geometry.num_layers (0/1/2/4/8/14/28), 跑 Q4_0 argmax 二分 onset.

## ★★★0 层截断定位：bug 在 embed+head 真实组合（architect round 25 命中）

**截断二分**（GLLM_TRUNCATE_LAYERS 环境变量，截断 layer_loop_config.num_layers）：
- Q4_0 **0 层**（embed→final_norm→lm_head→argmax）: output="eraçãoected肥ückenaptivemountgbaкупardotic" **乱码**
- Q4_0 1 层: "şa生態われる\n fluffy'],\r\nيح $? bulunduğugerald" 乱码
- BF16 0 层: "is is is is is is is is is is" **合理重复 token**（非乱码）

**结论**（architect round 25 预测命中）：
- 0 层垃圾 → bug 在 **embed + head 真实组合**（QuantGather + lm_head QuantGemm）
- BF16 0 层合理 → 共享组件（final_norm/argmax/层外接线）对
- Q4_0 0 层乱码 → Q4_0 特有的 embed(QuantGather) 或 lm_head(QuantGemm) 在真实规模错

**缩小到 2-op 真实图**：
- embed = QuantGather（真实: hidden=1024, vocab=151936, 32 block/token）
- lm_head = QuantGemm（真实: m=seq, n=vocab=151936, k=hidden=1024）
- oracle 只测 QuantGather hidden=32/64, QuantGemm k=32/64/1024(m=1,n=1/4)
- 真实规模 QuantGather(hidden=1024) + lm_head(n=151936) 未测

**下一步**：
- 建 QuantGather 真实规模 oracle（hidden=1024, 从真实 GGUF 读 embed block, 对比手算 SPLIT dequant）
- 或 dump 0 层截断下 QuantGather 输出（embed 后, final_norm 前）
- 候选: QuantGather 真实 hidden=1024 (32 block) 的 blk_ctr 跨 block 在真实规模暴露? 或 lm_head n=151936?

## ★1层截断 logits dump（architect round 26，Q4_0 vs BF16 量级对比）

**1层截断**（0层 SIGSEGV, num_layers=0 不合法）：
- Q4_0 1层: argmax=129008, logits|max|=**98.12**, vocab=151936
- BF16 1层: argmax=1172, logits|max|=**22.64**, vocab=151936
- Paris=12095 (两者 1层都不够, 但量级对比可信)

**关键**: Q4_0 logits|max| = 98.12 vs BF16 = 22.64 → Q4_0 **量级放大 ~4.3×**

**分析**:
- 4.3× 不是 nibble 当 float 读 (那会 100×+)
- 4.3× 可能: 某层 dequant 小错累积 / lm_head n=151936 偏移 / Q4_0 特有 dtype 传播
- 两者 argmax 都非 Paris (1层不够), 但 Q4_0 量级异常

**下一步**: 
- architect Step B: 扩 oracle 到 n=151936 (真实 lm_head vocab) 复现
- 或 dump Q4_0 1层的中间张量 (QuantGather 输出 / lm_head 输入) 定位量级放大点
- DiagnosticScratchpad.named_offsets 可读中间张量 (find_first_nan_tensor 模式)
