# GLLM 下一代大一统推理引擎实现蓝图 (Implementation SPEC)

**文档状态**: Final Draft (Review)  
**设计目标**: 实现百万级上下文（10M+ Context）的极限前向推理，强制实施“Zero-Overhead Freeloading”原则。

---

## 1. 硬件探测与 IR 抽象层 (MicroArch-to-IR Abstraction)

废弃散乱的指令集条件判断，所有在加载期的硬件探测结构必须严格坍缩为对底层 JIT 编译器的**强数学约束变量组 (Compiler Constraints)**，确保 JIT 逻辑与物理实体芯片解耦。

### 1.1 物理传感器指标
- **缓存与拓扑**: CPU `L1i`(指令缓存大小), GPU `L2_Cache`(分级驻留锚点), `CCX/NUMA` 边界, `RDMA_Latency` (跨节点通信掩蔽界限)。
- **x86 极化阵列**: 探测 `AMX`, `APX` (31 GPR通用寄存器扩展), `VNNI/AVX_VNNI_INT8`。
- **ARM 极化阵列**: 探测 `SVE2`, `SME/SME2` (`ZA_Array_Size`) 及 Streaming SVE 模式阈值。
- **总线协议**: GPU `TMA` (Hopper 异步内存张量加速)。

### 1.2 IR 抽象映射输出 (Target Execution Topology)
传感器数据转化为编译图（CompilerGraph）直接可用的环境变量：
- `max_gpr_count`: 控制寄存器溢写（Spill）阈值（普通 CPU=15, APX=31）。
- `optimal_tile_bits`: 决定 JIT 平铺展开的二维尺寸（启用 AMX/SME 阵列时极速扩大）。
- `native_int4_dot`: 标定是否可以通过 VNNI/SVE2 直接下发硬件解包。

---

## 2. TurboQuant 2.0 运行时数学精度优化与双轨显存池 (TurboQuant KV Architecture)

完全废弃传统的动态混合精度（Runtime Mixed-Precision FP16/INT8 分支决策）。QuantType 直接驱动 JIT 生成硬件原生内核。TurboQuant 是 gllm 在推理过程中执行的运行时数学优化——通过在线旋转、非对称 KV 量化、无偏修正，使推理精度逼近数学无损。详见 02-ARCHITECTURE.md §11。

> **学术依据**: SpinQuant (ICLR 2025), KurTail (2025), QuIP# (ICML 2024), RaBitQ (SIGMOD 2024), KIVI

### 2.1 在线旋转插入点（Online FWHT）
前向传播中 3 个非线性边界（Softmax、SwiGLU 门控乘法、RoPE），旋转变换无法穿越，必须在运行时内联执行 FWHT：
- Softmax(QK^T)·V 输出之后（Attention Epilogue 内联）
- SwiGLU(Gate)⊙Up 输出之后（FFN Epilogue 内联）
- RoPE(K) 存入 KV Cache 之前（KV Write 阶段内联）

### 2.2 双轨显存池 (The Dual-Track Memory Pool)
重构 `KvCacheConfig`，全面淘汰 `dtype_size`。由 `GlobalMemoryManager` 申请物理隔离的两轨架构：
- **主池 (Main Pool)**: 3-bit / 4-bit，组级缩放由 Epilogue 快递，KV 按 KIVI 非对称量化（Key per-channel, Value per-token）。
- **校验池 (QJL Pool)**: 并行的 1-bit XNOR 残差掩码阵列。
- **DTOD 与 RDMA 极速同步**: 多卡 KV Swap 及跨机数据移动仅传输原 FP16 内存量纲的 25%，突破总线墙。

---

## 3. Mega-Kernel 块级路由分发 (Spatial-Block Routing Mega-Kernel)

应对大吞吐并发下的请求形态分歧（Divergence），完全禁止外部宿主机（Host）分配多条 CUDA Stream/Thread Pool 来避免驱动层灾难。

### 3.1 块级内嵌路由 (In-Kernel Dispatch)
每一轮 Decode 或 Chunked Prefill，全系统 **仅 Launch 唯一一个 Mega-Kernel**。
- Mega-Kernel 内联编译了极端的计算支路（如：`Dense-Integer 核心路` / `Gate-First-Skip 稀疏路`）。
- **路由读表**：SM 核心内的 Thread Block（线程块 `blockIdx.x`）启动第一步，检索内存中的 `Request_State_Table`。
- **动态闭环**：如果不满足触发条件（比如当前请求的神经元死亡率超高），Thread Block 直接在内部通过 `jmp` 或控制掩码走向“稀疏路”。所有请求在同一次 Launch 周期内完成空间异步执行，消弭等待木桶效应并根除主机同步时滞。

### 3.2 张量动态挤压兜底 (Ragged Tensor Compaction)
对于 Batch 内极为零碎或不值得引发汇编层跳转的单点差异，JIT 利用 AVX512 的 `vcompress` 或 GPU `Warp Prefix Sum` 将活跃 Token **物理挤压聚拢（Compact）**，在没有 Padding 气泡的连续稠密矩阵中执行完计算后，再按原偏移原位散射（Scatter），维持计算单元 100% 满效率轰炸。

---

## 4. 就地记录的 Epilogue 白嫖网 (Telemetry-Cache In-Place Logging)

绝不为特征监控引入独立的无锁环形队列（Ring Buffer），彻底消灭由于监控带来的数据搬移开销。

### 4.1 尾段内联探测 (Epilogue Instrumentation)
在 `RmsNorm`、`Softmax` 和网络层级的底层 PTX/ASM 尾段，免费捎带计算 Token 的 `熵(Entropy)`、`概率质心(Centroid)`、`跨层能量差(Residual Delta)`。同时在 RMSNorm 规约循环尾段追加 `max` 指令，顺手为下一层输入计算 Group Scale（TurboQuant 2.0 白嫖路径），以及从 $\text{RMS} \cdot \sqrt{d}$ 免费获得 RaBitQ 修正因子所需的向量范数 $\|v\|$。

### 4.2 显存页就地存储 (Zero-Copy Paged Header Write)
这批特征数据随着最后一步写入 K/V Cache，利用常规内存写入操作（如 `STG`）**直接存储到该 Token 所在 KV Physical Page 的未使用页头（Header Padding）中**。

---

## 5. 全局共识的热修补进化 (Global Consensus Hot JMP Patching)

这层防线只为防范百万 Token 级别带来的“宏观长尾静滞期”模型执行状态蜕变（如代码补全时不变的超长 Context 所固化的注意力）。

### 5.1 卡尔曼迟滞监控 (Macro JIT Director Daemon)
后台沙盒常驻纯 Rust `JIT Director Daemon` 线程，定期无损挂载扫描全体 KV Page Headers。利用半衰期积分池（Decaying Reservoir）平滑指标变异（如注意力持续数百个 Batch 趋近静默态）。

### 5.2 静态键内联热替换 (Static Keys Trampolines)
一旦捕获到这种不可逆的全局共识，后台将立即启动原子核级别的指令变轨：
- Mega-Kernel 运行时默认使用**无条件直接跳转（`jmp`）** 与 **空白滑行区（`NOP Slide`）** 作为占位符，分支预测开销为 0。
- 当图拓扑确实需要坍缩或跳过时，后台在沙盒完成验证，计算新的指令绝对偏移量。
- 主系统调用原子写操作（Atomic Overwrite），将 `.text` 执行内存段上的 5-bytes 长 `jmp` 立刻热覆盖到新入口。
整个推理流水线在毫无波澜的一瞬间完成全局拓扑坍缩重构（Graph Collapsing/DCE）。

---

## 6. 旧世代优化理念的全面突变与升级 (Metamorphosis of Legacy Optimizations)

在采用了上文中“物理级隔离、指令级热修与 TurboQuant 降维”的架构后，原项目（P4/P5 规划）中那些极其优异的数学逃逸与跳过思路（如混合精度测算、残差绕开等），其**底层实现思路发生了根本性地突变**。
我们抛弃了过去所有在 CPU 或 Kernel 中写下 `if-else` 的幼稚做法，将它们转为真正的底层硬件法则：

### 6.1 动态混合精度检测 $\to$ 数学级静态湮灭 (Mathematical Annihilation)
- **旧思路**：在 `RmsNorm` 的尾端，拼命计算 `Amax`（最大极值），如果发现有 Outlier（离群点），下一层就滚回 FP16 慢速通道，发现没有，才敢降级到 FP8/INT8。这种反复横跳引发了严重的流水线不确定性。
- **新架构蜕变**: 详见 02-ARCHITECTURE.md §11 TurboQuant 2.0。QuantType 直接驱动 JIT 生成硬件原生内核，前向传播中 3 个非线性边界的在线 FWHT 旋转（$O(d\log d)$，内联 Mega-Kernel Epilogue）确保分布在穿越 Softmax/SwiGLU 后仍然保持量化友好。
- **执行定论**：所有的 `Amax` 运行时检测代码被**全盘删除**！QuantType 直接驱动 JIT 静态锁定执行路径，对量化权重发射 VNNI/SVE2 整数指令流。真正的零开销，是依靠经同行评审的数学理论（而非盲目随机旋转的统计假设）在物理层扼杀掉动态校验逻辑！

### 6.2 门控网络（Gate）失效截断 $\to$ 寄存器级动态挤压 (Register-Level Compaction)
- **旧思路**：算完 `Gate = SiLU(xW)` 后，如果发现大量行/列为零，则尝试在 C++/Rust 调度层进行切片，或者直接 `jmp` 绕开。
- **新架构蜕变**：在 Mega-Kernel 中，我们**不跳远，只挤压**。
- **执行定论**：直接调用向量外设的 **硬件谓词掩码（Predicated Execution）** 或张量挤压（AVX512 的 `vcompress`、GPU 的 `Prefix Sum`）。把失效死掉的神经元直接在共享内存（SMEM）里挤成一坨最小的密实体，用一个全负荷不停机的 `Up/Down GEMM` 把它们算平。不破坏任何控制流，仅靠向量算核的掩码位让它强行断电休息。

### 6.3 跨层残差旁路（Residual Bypass） $\to$ 内核路由自毁弃算 (In-Kernel Router Interrupt)
- **旧思路**：在每一层的 `Residual Add` 算一个方差 $\Delta \rho$。如果 $\Delta \rho < 0.001$，下一层的 `Attention` 走控制流分支短路。
- **架构校正（大并发禁区）**：绝对不能用宏观热修补（Hot Patching）来做！因为在 Continuous Batching 下，Request A 可以跳过，但 Request B 可能正处于高信息熵刚需期。如果物理修改汇编 `jmp`，会直接导致 Request B 产出乱码！
- **执行定论**：残差跳过归入 **3.1 节的 Mega-Kernel 块级内嵌路由（Block-Level Routing）**。
  - 在 Kernel 启动的第 0 纳秒，属于 Request A 的 Thread Block（线程块）读取到自己的历史 $\Delta \rho < 0.001$。该 Thread Block 直接触发 `Thread Exit` 或触发身份掩码，将输入原封不动抛给输出，原地休眠。
  - 属于 Request B 的 Thread Block 读取到 $\Delta \rho = 0.5$，正常展开矩阵乘法循环。
  - 这种物理上同源、逻辑上分块（Block Idx）的掩码路由，在不破坏全局机器码指令（不用 Hot Patch）的前提下，完美兼容了 Batch 内千奇百怪的跳过需求。

### 6.4 那么，内联热修补（Hot JMP Patching）到底用在哪？ $\to$ 全域冷专家/静滞流的截除
既然个体行为不能导致热修补，那热修补只用于**绝对的全局物理共识**：
- **场景 A（领域降维）**：如果当前引擎跑的是一个专门做翻译的业务大模型，MoE 网络中负责“代码生成”或“纯数学解题”的冷板凳专家（Expert）持续数百万个 Token 均有 0 命中率。JIT Director 就会启动热重载，直接将这些专家的访存分支和跳转入口用 `NOP/JMP` 物理抹平（DCE）。
- **场景 B（极致前缀复用）**：如果 128 个并发请求全部挂载在同一个长达 10K 的系统 System Prompt 前缀树上，这部分的前置处理图（Prefill 图）可以直接被热修补塌缩为一条共享直读存储指令。

### 6.5 注意力缓存预瞄 (Attention Delta) $\to$ 跨 PCIe 地址预载 (RDMA/PCIe Pipelined Prefetch)
- **旧思路**：根据上层的 Softmax 质心算位置，看看要不要算全量的 `Q*K^T`。
- **新架构蜕变**：将“算力的减法”转为“带宽的乘法”。
- **执行定论**：Softmax 质心的坐标数据通过页表直接馈入底层硬件预取系统（Prefetcher）。在 GPU 处理本层 Dense 网络的时候，总线通道上已经在并行为下一层利用 `cuMemPrefetchAsync` 甚至是在多机利用 `RDMA` 加载特定的 KV 块。这让大模型在极限吞吐下，彻底穿透了冯诺依曼架构的显存墙诅咒！

---

## 7. 终极考验：MoE 异构专家的极致落地 (Extreme MoE Implementation)

之前我们在 P4 阶段规划过“冷热专家异构调度（CPU/GPU/RDMA）”。在目前这套严酷的“大一统 JIT 架构”下，MoE 的异构计算不仅能实现，而且直接借助现有的 5 大支柱，将爆发出极其恐怖的性能（理论上限直接拉满）：

### 7.1 MoE 的核内分发 (无多内核启动延迟)
传统 MoE 引擎（如 vLLM/TGI）在算出路由后，会为每一个 Expert 调用独立的 Kernel，不仅导致严重的 Launch Overhead，而且各路请求互相排队。
- **基于第 3 支柱（Mega-Kernel 路由）**：我们的 MoE 层依然只启动 **1 个超级内核（Mega-Kernel）**！
- GPU 内部的 Thread Block 在拿到 `Softmax(Gate)` 后，利用内置字典读出自己该去哪个 Expert。然后在 Kernel 内部利用汇编 `jmp` 直接跃迁到对应专家的权重读取区。物理空间上并发解耦，零 Driver 启动开销。

### 7.2 TurboQuant + 预瞄 $\to$ 异步载入冷专家 (Zero-Stall Swapping)
如果该专家是个温/冷专家，它的权重在系统内存（CPU RAM）甚至远端 RDMA 内存中，该怎么办？
- **基于第 2 支柱（TurboQuant 极低精度）**：由于开启了 TurboQuant 2.0，专家的权重被暴压到了 4-bit 甚至 2-bit！PCIe 和 RDMA 的搬运延迟缩减了惊人的 75%～87.5%！
- **基于第 6.5 支柱（预取联动）**：当 Gate 层一算出路由表，立刻启动无阻塞的 `cuMemPrefetchAsync`。等到 Thread Block 真正走到那个 Expert 的汇编入口时，被极度压缩的 4-bit 权重已经躺在 GPU 的 L2 Cache 里了。所谓的“冷专家卡顿”被流水线完美掩盖（Pipelining）。

### 7.3 CPU/GPU 的真正并行 (The Free-Lunch Core Disaggregation)
如果真的需要异构运算（比如 GPU 算热专家，留给 CPU 顺手算温专家），完全不需要上层写多线程锁！
- **基于第 1 支柱（IR 抽象）**：CPU 的 NUMA 探针会暴露它的 `AMX` 或 `AVX512` 能力为 IR。编译器为这个温专家顺量生成了一段 CPU 特化的 JIT 代码。并且通过独立的并发流将少部分 Token 交给 CPU 算完后统一回写。

### 7.4 冷板凳专家的全域封杀与“复活陷阱” (Global DCE & Uncommon Traps)
这也是所有激进 JIT 编译器（如 V8 / Java HotSpot）必须解决的最核心难点：**如果物理截肢了一个专家，突然又来了一个请求需要它，怎么办？会不会产出乱码？**

- **绝对正确性底线（路由不可磨灭）**：在 MoE 中，门控（Gate Router）的计算开销极低（仅占总算力的不到 1%）。因此我们**绝不去热修补 Gate 概率计算本身**。
- **陷阱替换（Uncommon Trap）**：JIT Director 虽然抹除了冷门专家的上亿参数庞杂乘法网（`Up/Down GEMM`），但它修补的并不是无脑的 `NOP` 丢弃。它是把进入该专家的指令入口，覆写成了一条指向 **“去优化处理惩罚极 (Deopt Handler / Uncommon Trap)”** 的跳转指令。
- **冷触发与系统复活（OSR Bailout）**：
  1. 过去几百万次运行，由于无人需要冷专家，系统高速穿透了修补后的平滑防线。
  2. 突然，Request A 算出的 Gate 指向了被封杀的冷专家 7。
  3. 属于 Request A 的 Thread Block 顺理成章地一头撞进了 **“陷阱指令（Uncommon Trap）”**。
  4. 该 Thread Block 会立刻向显存写下一个 `DEOPT_REQUEST = 7` 的报错标志，并主动挂起自己（不输出错误数据）。同 Batch 的其他常规请求不受影响，继续极速算完。
  5. 待这一层结算时，引擎主循环发现了 `DEOPT_REQUEST`。此时 Host 触发一次不到 1 毫秒的微冷冻。JIT Director 瞬间将 `.text` 回写复原，把冷专家 7 的网格重连。并**异步唤回**其在主存的 4-bit 权重。
  6. 引擎为刚才挂起的 Request A 单独走一遍回炉重造（Re-evaluate）。

这种**“用万分之一的局部挂起代价，换走 99.99% 时间里的物理绝对零开销”**的设计，就是现代极速 JIT 引擎里的 **De-optimization（去优化回退）机制**！它构成了 Gllm 架构的最强韧防御底盘。

**结论**：在我们的这套“大一统 JIT + TurboQuant + Deopt 兜底”的底层生态下，MoE 的异构不再是一个需要单独硬写的孤岛功能，它本身就是这些底层组件相互摩擦后，必然产生的自然现象。

---

## 8. 残差连接的终极降维：开放式物理数据总线 (The Residual Bus)

你提到：“Layer 的残差连接对我们来说有更大的价值吗？” —— 这是一个直击 Transformer 第一性原理的深刻拷问。
在绝大多数推理框架眼里，$x_{out} = x_{in} + \text{Layer}(x_{in})$ 仅仅是为了梯度不消失而存在的数学公式。
但在我们的 **JIT Mega-Kernel 架构** 中，残差流（Residual Stream）被物理重构为一条**贯彻始终的数据总线（Data Bus）**。它不仅仅是向前传数据，它是一个**开放的插入端口（Injection Port）与召回端口（Recall Port）**！

### 8.1 超大知识的外挂注入点 (Late-Fusion RAG Injection)
过去为了让大模型读外部长文档（RAG），我们只能傻傻地把几万字的 Prompt 塞进第 0 层的 Embedding 里，让它苦哈哈地爬过所有的 40 层网络。
- **架构剥离**：既然残差是一条总线，外部的“超大知识储备”根本不需要从浅层视觉/语法层（Layers 0-10）开始算！
- **JIT 注入锚点（Injection Hooks）**：我们可以用一个极其便宜的小模型（如 BERT 或 2B 小模型）把外挂知识算出高维语义向量。在 JIT 调度时，系统直接拉开第 15 层的残差入口，利用一个极其极速的 `Vector Add` 指令，把这坨庞大的知识向量**生硬地“加”进该层正在计算的残差流（Residual Stream）里**！
- 我们的 Mega-Kernel 在生成时，会在指定的“知识融合层”预留一段外链读取汇编（`LDG.E` 外部图显存），使得外部长上下文可以物理级跳过前半段网络，直接在深层语义区完成**晚期融合（Late-Fusion）**。

### 8.2 任意层的数据召回与高维截断 (Early-Exit & PGSLE Speculative)
残差流里流淌的是模型在这个瞬间思考的“中间态投影”。
- **架构剥离**：为什么一定要等到最后一层（第 40 层）才接语言投影头（`lm_head / Vocab Projection`）？
- **JIT 探针召回（Intermediate Recall）**：在生成 Mega-Kernel 时，我们可以让 JIT 在底层的某些特定层（例如 Layer 20, 28, 35）的残差 Add 之后，**顺手附带一个超小的线性分类器（微型 lm_head）**。
- **动态深度截断 (Dynamic Depth)**：如果在第 20 层，这个微型分类器“召回”的中间数据算出的下一个 Token 概率已经逼近 99.9%（比如模型正在复颂极为熟悉的成语“大海(捞针)”）。该层 Thread Block 直接发射控制信标，**触发我们在第 6 节定下的内联热修补或块级路由跳过功能**，当场物理切断（Amputate）后续 20 层的计算！直接在半山腰把 Token 吐给用户！
- 这就是真正物理级的 **PGSLE 投机解码与早退机制（Early-Exit）**，让对于简单问题的运算量直接被物理腰斩！

### 8.3 纯解码的降维应用：通用多意图识别 (Universal Intent Recognition)
一旦放开了“必须走完全程”的思维枷锁，大模型本身就是一个无可匹敌的语义编码器（Encoder）。
- **新特性规划**：为了做多意图识别（NLU 分类），我们不需要让 LLM 吐出完整的回答。
- **物理截断执行 (Partial Execution)**：引擎提供 `Pure_Decode` API 模式。在 JIT 编译时，我们仅仅选取大模型最为核心的“理解区”——即前 15 或 20 层（剥离了后续负责具体语言学和逻辑生成的层叠）。
- **残差头提取**：用户的 Prompt 跑过这前段图后，JIT 直接把残差总线截留（Recall），送入一个我们训练好的轻量级多分类探针（Linear Probe）。这让引擎瞬间化身为一个不仅聪明绝顶，而且速度媲美小模型、成本近乎为 0 的**意图分类与特征提取神塔**！

### 8.4 零延迟的飞行巡航审查 (In-Flight Guardrail Layer)
市面上绝大多数“护栏（Guard）”模型（比如查毒、鉴黄、防止核心机密泄露），要么是在输入前拦截，要么是等模型回答完一段话后再去审查，又慢又容易被绕过。
- **新特性规划**：利用 Gllm 本土的 JIT 优势，直接在生成管线内部埋设 **安全审查层（Guard Review Layer）**！
- **嗅探探针的注入**：我们在模型深度的某一层（例如最后倒数几层），物理级强插入一个极小的安全审查头（Safety Head）。
- **瞬时拦截**：在自回归生成（Autoregressive）的过程中，每吐出一个新词，大模型不仅要预测下一个字，这个**寄生在残差总线上的 Guard 探针**还在不断“嗅探”当前高维语义流中是否存在“触发危险概念”的特征聚集。
- **物理熔断**：一旦 Guard 探针概率超标，Mega-Kernel 内的 Thread Block 直接抛出中断信号。模型在吐出危险词的**那一瞬间之前**，就被当场强行切断计算流！这是真正的**零延迟、无法越狱的安全物理护城河**！

这就是 Gllm 开放底层执行图并构建其为物理总线的可怕之处：诸如意图识别、RAG、安全护栏等原先必须架设微服务集群的复杂大生态，不仅不需要外设 API，而且被硬生生压成了底层 Kernel 里的几句附加指令，化为了引擎出厂自带的超光速内置兵器！

---

## 9. 知识注入的工程化封装：API 与结构设计 (The Injection SDK Engine)

正如你敏锐指出的，底层的汇编 Hack 再狂野，如果不能作为一套**优雅、结构化的 Library（库）被外部业务调用**，那它就是一团死代码。Gllm 的宏大目标是被集成为 ToB 企业级基础设施。因此，之前探讨的三大物理注入形态（侧载 KV、残差硬插、多路 LoRA），必须在工程代码结构上抽象化。

为了让开发者能够方便地“插管”，我们将深入重构引擎的对外接口与内部编译器拓扑：

### 9.1 数据源的多态抽象 (`KnowledgeDataSource` Trait)
开发者不需要理解底层什么是 `Ldg.E` 或虚拟页表，他们只需和纯粹的数据接口打交道。我们将设计一个极其顶层的抽象 Trait：
- `FrozenKvChunk (实现侧载)`：业务端直接传入一个 SSD 文件柄或网络地址（预存的 4-bit 财报）。
- `LateFusionVector (实现晚期插入)`：上游小模型（如 BERT）算好的密实特征向量列。
- `DynamicLoRA (实现领域特征挂载)`：带有特定领域特征缩放因子的极小权重片。

### 9.2 智能化的锚点层推断与知识注入 API (`Semantic Anchors` & `InjectionHook`)
正如你所指出的，强制开发者硬编码 `layer=15` 是极其愚蠢的，因为 Llama 7B 只有 32 层，而 70B 有 80 层。语义的深水区位置完全不同。
- **语义锚点推断 (Semantic Anchors)**：引擎内部会根据加载的模型拓扑，通过“熵分布曲线”自动标定锚点。定义 `LayerTarget::ShallowSyntax` (浅层词法)、`LayerTarget::MidSemantic` (中层语义)、`LayerTarget::DeepLogic` (深层逻辑)。
- **智能库接口**：
  ```rust
  // 引擎自动探测当前模型的中层语义区（例如模型 A 的第 16 层，模型 B 的第 40 层）进行残差加持
  engine.inject_knowledge(
      KnowledgeSource::from_vector_db(my_vector),
      LayerTarget::MidSemantic 
  );
  ```
- **编译时下放 (Lowering)**：当 JIT 开始编译时，一旦遇到这个特定的 `InjectionHook` 节点，它会自动展开成符合该层级的汇编代码（例如调用 `Vector Add` 的微指令），彻底对外部屏蔽硬件异构和模型深浅。

### 9.3 纯解码的降维应用 API (`Multi-Intent NLU` API)
为了支持第 8.3 节的“通用多意图识别”，我们开放专用的降维 API，直接砍掉生成尾部：
- **智能库接口**：
  ```rust
  // 直接以“语义提取器”模式运行，物理砍掉后续语言生成层的运算开销
  let intent_embedding = engine.encode_intent(
      prompt,
      LayerTarget::MidSemantic // 抽离残差流的截止深度
  );
  // intent_embedding 直接喂给业务层的小分类器，瞬间完成亿级并发判别
  ```

### 9.4 零延迟的飞行巡航审查 API (`In-Flight Guardrail` API)
对于第 8.4 节的“安全物理护城河（Zero-Latency Guard）”，API 的设计必须体现它的异步监听与物理熔断特性：
- **智能库接口**：
  ```rust
  // 寄生挂载一个微型审查头，并指定熔断策略
  engine.attach_guardrail(
      GuardProbe::from_safetensors("toxicity_head.safetensors"),
      LayerTarget::DeepLogic,                   // 挂接在深层即将爆词的位置
      SafetyPolicy::HaltAndVeto(threshold=0.95) // 超过 95% 危险概率，底层直接掐断 Mega-Kernel
  );
  ```
  这个 API 被调用后，底层的线程块（Thread Block）会自动携带这段串联的分类拦截汇编，绝不阻塞正常的并行吞吐。

### 9.5 零拷贝的页表管理器与分流器 (`KvSideloadManager` & `Multiplexer`)
- 针对 10 万字财报的 **侧载 KV (FrozenKvChunk)**，底层通过 `KvSideloadManager` 拦截当前 Request 的**逻辑页表（Logical Page Table）**，无任何 `memcpy` 地插入外部物理块 ID。
- 在千人千面的大吞吐流（Continuous Batching）中。Mega-Kernel 内部的 `ThreadIdx` 会先查阅 `Injection_Routing_Table`（注入路由表）。需要挂载意图识别和知识的请求跑专用特征算子，不需要的同轴共存，完美兼容批处理。

**工程结论**：通过语义锚点（Semantic Anchors）取代死板的层号计算，通过顶级 API 把知识挂接（Injection）、意图截断（Intent NLU）、安全熔断（Guardrail）全部降格为傻瓜式的库调用，而在 JIT 编译器的下层，这全是一条条精密无比的特化内核与热修补。Gllm 现在不仅是极致推理引擎，它成为了**构建未来 AI 原生应用（如零延迟 RAG、亿级审核并发集群）的最强物理底座**！

---

