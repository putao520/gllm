# GLLM 大一统推理引擎蓝图 — 索引

> **⚠️ 本文档已降级为索引文件。所有设计内容的 SSOT 为 [02-ARCHITECTURE.md](../../02-ARCHITECTURE.md)。**
> **降级原因**: 本文档 9 个章节与 02-ARCHITECTURE.md §9-§16 近乎 1:1 重复，违反 SSOT 原则。

---

## 章节索引

| 本文档原章节 | SSOT 位置 | 说明 |
|---|---|---|
| §1 硬件探测与 IR 抽象层 | [02-ARCHITECTURE.md §12.6](../../02-ARCHITECTURE.md) | MicroArch-to-IR 约束变量体系 |
| §2 TurboQuant 2.0 | [02-ARCHITECTURE.md §11](../../02-ARCHITECTURE.md) | 在线旋转、非对称 KV 量化、无偏修正、双轨显存池 |
| §3 Mega-Kernel 块级路由 | [02-ARCHITECTURE.md §9](../../02-ARCHITECTURE.md) | 单一 Launch、块级内嵌路由、挤压聚拢 |
| §4 Epilogue 白嫖网络 | [02-ARCHITECTURE.md §13](../../02-ARCHITECTURE.md) | 全链路 11 个白嫖点 + 硬件感知融合拓扑 |
| §5 全局共识热修补 | [02-ARCHITECTURE.md §9.2](../../02-ARCHITECTURE.md) | JIT Director Daemon + Trampoline 原子覆写 |
| §6 旧世代优化突变 | [02-ARCHITECTURE.md §14](../../02-ARCHITECTURE.md) | Amax 静态湮灭、Gate 挤压、残差内嵌路由 |
| §7 MoE 异构专家 | [02-ARCHITECTURE.md §15](../../02-ARCHITECTURE.md) | 核内分发、冷专家封杀、Uncommon Trap |
| §8 残差总线 | [02-ARCHITECTURE.md §16](../../02-ARCHITECTURE.md) | Late-Fusion RAG、Early-Exit、Guardrail |
| §9 知识注入 SDK | [02-ARCHITECTURE.md §16](../../02-ARCHITECTURE.md) + [04-API-DESIGN.md](../../04-API-DESIGN.md) | Semantic Anchors、InjectionHook API |

---

## 相关文档

- [02-ARCHITECTURE.md](../../02-ARCHITECTURE.md) — 架构设计 SSOT
- [04-API-DESIGN.md](../../04-API-DESIGN.md) — 公共 API 设计（含 Knowledge Injection SDK）
- [jit-cache-protocol.md](./jit-cache-protocol.md) — JIT 编译缓存协议
- [hgal-scheduler-algorithm.md](./hgal-scheduler-algorithm.md) — HGAL 调度算法
- [ai-development-guideline.md](./ai-development-guideline.md) — 极简化内核执行底线原则
