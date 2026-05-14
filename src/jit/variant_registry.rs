//! VariantRegistry — 编译时变体选择系统 (SPEC §18.4)
//!
//! 所有跨机制冲突通过编译时变体隔离消解，不在运行时引入条件分支。
//!
//! ## 核心职责
//! 1. 维护 VariantKey → CompiledVariant 的映射
//! 2. 编译时检查 L1i 预算（≤80%）
//! 3. build_batch() 阶段（Dispatch-Time）选择最优变体
//!
//! ## 约束
//! - 禁止 Mega-Kernel 内运行时 `if moe_enabled`
//! - 禁止 Mega-Kernel 内运行时 `if guardrail_active`
//! - 变体选择发生在批构建阶段，不在 Mega-Kernel 执行时

use std::collections::HashMap;


/// 代码段类型 — 控制指令在缓存层级中的驻留位置 (§9.6)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeSection {
    /// L1i 常驻 (≤80% L1i, 必须) — 热路径代码
    Hot,
    /// L2 常驻, 通过 NOP Trampoline 按需拉入 L1i — 温路径
    Warm,
    /// L3/DRAM, 长跳转 (Long JMP), 几乎不执行 — 冷路径
    Cold,
}

/// 机制标识 — 每个变体包含的优化机制清单
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MechanismId {
    /// 基础稠密前向 (RmsNorm → QKV → Attn → FFN → Residual)
    Dense,
    /// MoE 专家分发 (§9.4)
    MoeDispatch,
    /// 推测解码 Draft 阶段 (§17.1)
    SpecDraft,
    /// 推测解码 Verify 阶段 (§17.1)
    SpecVerify,
    /// Guardrail 安全探针 (§16.4)
    GuardrailProbe,
    /// Late-Fusion RAG 知识注入 (§16.1)
    RagInjection,
    /// TurboQuant FWHT 旋转 (§11.1)
    TurboQuantFwht,
    /// Gate-First Skip 死神经元跳过 (§13.1)
    GateFirstSkip,
    /// 残差旁路 Δρ 跳过 (§13.3)
    ResidualBypass,
    /// Epilogue 遥测 (§9.5)
    Telemetry,
    /// Early-Exit 微型 lm_head (§16.2)
    EarlyExit,
    /// Compact→Execute→Scatter 三段式 (§9.1)
    RaggedCompaction,
    /// KIVI 4/2 量化 KV 写回 Epilogue (§19 KV-OPT-004)
    KiviQuant,
}

/// 推测解码阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecPhase {
    /// Draft: 使用浅层变体生成候选 token
    Draft,
    /// Verify: 使用全量模型验证候选 token
    Verify,
}

/// 变体签名 — 决定需要哪些机制的代码 (§18.4 VariantKey)
///
/// 每个不同的 VariantKey 对应一份独立的 JIT 编译产物。
/// 变体选择发生在 Dispatch-Time (build_batch 阶段)，
/// 不在 Mega-Kernel 内部。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariantKey {
    /// 模型架构 (决定基础图结构)
    pub arch: String,
    /// 是否含 MoE 层 (决定专家分发代码)
    pub moe_enabled: bool,
    /// Guardrail 是否激活 (决定 probe 代码)
    pub guardrail_enabled: bool,
    /// 推测解码阶段 (None=标准, Some(Draft/Verify))
    pub spec_phase: Option<SpecPhase>,
    /// RAG 注入是否激活 (决定残差注入代码)
    pub rag_enabled: bool,
    /// 序列长度装筒 (Golden Size, §12.4)
    pub golden_size: usize,
    /// 量化精度标识 (决定 TurboQuant FWHT 代码), 存储 QuantType variant name
    pub quant_type: Option<String>,
    /// KV cache 精度等级 (决定 Attention KV load 微内核, None=FP16 基准)
    pub kv_tier: Option<String>,
}

impl std::fmt::Display for VariantKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}_{}{}_{}_{}_{}",
            self.arch,
            if self.moe_enabled { "moe_" } else { "" },
            match self.spec_phase {
                Some(SpecPhase::Draft) => "draft_",
                Some(SpecPhase::Verify) => "verify_",
                None => "",
            },
            if self.guardrail_enabled { "g" } else { "n" },
            if self.rag_enabled { "r" } else { "n" },
            self.golden_size,
        )?;
        if let Some(ref q) = self.quant_type {
            write!(f, "_{}", q)?;
        }
        if let Some(ref t) = self.kv_tier {
            write!(f, "_kv{}", t)?;
        }
        Ok(())
    }
}

/// 编译产物 — 包含指令足迹统计 (§18.4 CompiledVariant)
#[derive(Debug, Clone)]
pub struct CompiledVariant {
    /// JIT 编译的机器码 (可执行内存页)
    pub code: Vec<u8>,
    /// 指令足迹 (字节), 用于 L1i 预算检查
    pub instruction_footprint_bytes: usize,
    /// 该 Variant 涉及的机制列表 (用于审计)
    pub mechanisms: Vec<MechanismId>,
    /// 所属代码段 (.hot / .warm / .cold)
    pub section: CodeSection,
    /// 变体签名
    pub key: VariantKey,
}

/// L1i 预算超出错误
#[derive(Debug, Clone)]
pub struct L1iBudgetExceeded {
    pub footprint: usize,
    pub budget: usize,
    pub suggestion: String,
}

impl std::fmt::Display for L1iBudgetExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "L1i budget exceeded: {} bytes > {} bytes (80% of L1i). Suggestion: {}",
            self.footprint, self.budget, self.suggestion
        )
    }
}

impl std::error::Error for L1iBudgetExceeded {}

/// 变体注册表 — 所有机制冲突的消解中心 (§18.4 VariantRegistry)
///
/// Key = 场景签名, Value = 预编译的 JIT 产物。
/// 查找发生在 build_batch() 阶段 (Dispatch-Time), 不在 Mega-Kernel 内。
pub struct VariantRegistry {
    /// key = VariantKey → CompiledVariant
    entries: HashMap<VariantKey, CompiledVariant>,
    /// L1i 预算 (从 DeviceProfile 获取)
    l1i_budget_bytes: usize,
    /// 当前已用 L1i 最大值 (用于编译时检查)
    max_footprint_bytes: usize,
}

/// 默认 L1i 大小 (x86_64 主流服务器/桌面)
const DEFAULT_L1I_BYTES: usize = 32 * 1024; // 32 KB

/// L1i 预算比例 (SPEC §9.6: ≤80%)
const L1I_BUDGET_RATIO: f64 = 0.8;

impl VariantRegistry {
    /// 创建空注册表，使用默认 L1i 预算
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            l1i_budget_bytes: DEFAULT_L1I_BYTES,
            max_footprint_bytes: 0,
        }
    }

    /// 创建注册表，指定 L1i 大小
    pub fn with_l1i_budget(l1i_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            l1i_budget_bytes: l1i_bytes,
            max_footprint_bytes: 0,
        }
    }

    /// 返回 L1i 可用预算 (80%)
    pub fn available_budget(&self) -> usize {
        (self.l1i_budget_bytes as f64 * L1I_BUDGET_RATIO) as usize
    }

    /// 注册已编译变体
    ///
    /// 如果变体指令足迹超过 L1i 预算，返回 Err
    pub fn register(&mut self, variant: CompiledVariant) -> Result<(), L1iBudgetExceeded> {
        let budget = self.available_budget();
        if variant.instruction_footprint_bytes > budget {
            return Err(L1iBudgetExceeded {
                footprint: variant.instruction_footprint_bytes,
                budget,
                suggestion: "减少 batch 并发或禁用部分机制".to_string(),
            });
        }
        self.max_footprint_bytes = self.max_footprint_bytes.max(variant.instruction_footprint_bytes);
        self.entries.insert(variant.key.clone(), variant);
        Ok(())
    }

    /// 查找已编译变体
    pub fn get(&self, key: &VariantKey) -> Option<&CompiledVariant> {
        self.entries.get(key)
    }

    /// 查找或回退到最接近的变体 (非递归)
    ///
    /// 策略: 按以下优先级逐步放松约束:
    /// 1. 精确匹配 + golden_size 放松
    /// 2. 放松 spec_phase (None) + golden_size
    /// 3. 放松 guardrail_enabled (false) + spec_phase + golden_size
    /// 4. 放松 rag_enabled (false) + 全部放松
    pub fn find_closest(&self, key: &VariantKey) -> Option<&CompiledVariant> {
        // 原始约束
        if let Some(v) = self.try_relaxed(&key.arch, key.moe_enabled, key.guardrail_enabled,
            key.spec_phase, key.rag_enabled, &key.quant_type, &key.kv_tier, key.golden_size) {
            return Some(v);
        }
        // 放松 spec_phase=None
        if let Some(v) = self.try_relaxed(&key.arch, key.moe_enabled, key.guardrail_enabled,
            None, key.rag_enabled, &key.quant_type, &key.kv_tier, key.golden_size) {
            return Some(v);
        }
        // 放松 guardrail=false, spec_phase=original
        if let Some(v) = self.try_relaxed(&key.arch, key.moe_enabled, false,
            key.spec_phase, key.rag_enabled, &key.quant_type, &key.kv_tier, key.golden_size) {
            return Some(v);
        }
        // 放松 guardrail=false, spec_phase=None
        if let Some(v) = self.try_relaxed(&key.arch, key.moe_enabled, false,
            None, key.rag_enabled, &key.quant_type, &key.kv_tier, key.golden_size) {
            return Some(v);
        }
        // 放松 rag=false, guardrail=original, spec=original
        if let Some(v) = self.try_relaxed(&key.arch, key.moe_enabled, key.guardrail_enabled,
            key.spec_phase, false, &key.quant_type, &key.kv_tier, key.golden_size) {
            return Some(v);
        }
        // 全部放松
        if let Some(v) = self.try_relaxed(&key.arch, key.moe_enabled, false,
            None, false, &key.quant_type, &key.kv_tier, key.golden_size) {
            return Some(v);
        }
        None
    }

    /// 尝试精确匹配 + golden_size 放松
    fn try_relaxed(&self, arch: &str, moe: bool, guard: bool,
                   spec: Option<SpecPhase>, rag: bool, qt: &Option<String>,
                   kv_tier: &Option<String>, max_gs: usize) -> Option<&CompiledVariant> {
        let exact = VariantKey {
            arch: arch.to_string(), moe_enabled: moe, guardrail_enabled: guard,
            spec_phase: spec, rag_enabled: rag, golden_size: max_gs,
            quant_type: qt.clone(), kv_tier: kv_tier.clone(),
        };
        if let Some(v) = self.entries.get(&exact) {
            return Some(v);
        }
        // 放松 golden_size: 找 ≤ max_gs 的最大值
        let best = self.entries.keys()
            .filter(|k| {
                k.arch == arch && k.moe_enabled == moe
                    && k.guardrail_enabled == guard && k.spec_phase == spec
                    && k.rag_enabled == rag && k.quant_type == *qt
                    && k.kv_tier == *kv_tier
                    && k.golden_size <= max_gs
            })
            .map(|k| k.golden_size)
            .max();
        if let Some(best_gs) = best {
            let relaxed = VariantKey {
                arch: arch.to_string(), moe_enabled: moe, guardrail_enabled: guard,
                spec_phase: spec, rag_enabled: rag, golden_size: best_gs,
                quant_type: qt.clone(), kv_tier: kv_tier.clone(),
            };
            self.entries.get(&relaxed)
        } else {
            None
        }
    }

    /// 返回已注册变体数量
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 返回是否为空
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 返回当前最大指令足迹
    pub fn max_footprint(&self) -> usize {
        self.max_footprint_bytes
    }

    /// 返回所有已注册的变体键
    pub fn keys(&self) -> Vec<&VariantKey> {
        self.entries.keys().collect()
    }

    /// 返回所有已注册的变体
    pub fn variants(&self) -> Vec<&CompiledVariant> {
        self.entries.values().collect()
    }

    /// 根据批属性推导变体键
    ///
    /// 在 build_batch() 阶段调用，收集 batch 中所有请求属性后生成 key
    pub fn derive_key(
        arch: impl Into<String>,
        has_moe_layers: bool,
        guardrail_active: bool,
        spec_phase: Option<SpecPhase>,
        rag_active: bool,
        golden_size: usize,
        quant_type: Option<String>,
        kv_tier: Option<String>,
    ) -> VariantKey {
        VariantKey {
            arch: arch.into(),
            moe_enabled: has_moe_layers,
            guardrail_enabled: guardrail_active,
            spec_phase,
            rag_enabled: rag_active,
            golden_size,
            quant_type,
            kv_tier,
        }
    }

    /// 根据 KV tier 名称列表推导主 Variant (多数投票)
    ///
    /// 选择数量最多的 tier 作为主 Variant，少数 tier 的 page 预 dequant。
    pub fn majority_kv_tier(tiers: &[String]) -> Option<String> {
        if tiers.is_empty() {
            return None;
        }
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for t in tiers {
            *counts.entry(t.as_str()).or_insert(0) += 1;
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(k, _)| k.to_string())
    }
}

impl Default for VariantRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key(arch: &str, golden_size: usize) -> VariantKey {
        VariantKey {
            arch: arch.to_string(),
            moe_enabled: false,
            guardrail_enabled: false,
            spec_phase: None,
            rag_enabled: false,
            golden_size,
            quant_type: None,
            kv_tier: None,
        }
    }

    fn test_variant(key: VariantKey, footprint: usize) -> CompiledVariant {
        CompiledVariant {
            code: vec![0u8; footprint],
            instruction_footprint_bytes: footprint,
            mechanisms: vec![MechanismId::Dense],
            section: CodeSection::Hot,
            key,
        }
    }

    #[test]
    fn test_register_and_lookup() {
        let mut registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);
        let variant = test_variant(key.clone(), 1024);

        registry.register(variant).unwrap();
        assert_eq!(registry.len(), 1);

        let found = registry.get(&key).unwrap();
        assert_eq!(found.instruction_footprint_bytes, 1024);
    }

    #[test]
    fn test_l1i_budget_exceeded() {
        let mut registry = VariantRegistry::with_l1i_budget(1024); // tiny budget
        let key = test_key("qwen3", 64);
        let variant = test_variant(key, 2048); // exceeds 80% of 1024

        let result = registry.register(variant);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.footprint > err.budget);
    }

    #[test]
    fn test_find_closest_relaxes_golden_size() {
        let mut registry = VariantRegistry::new();

        // Register variants for golden_size = 32, 64
        let key32 = test_key("qwen3", 32);
        registry.register(test_variant(key32, 512)).unwrap();

        let key64 = test_key("qwen3", 64);
        registry.register(test_variant(key64, 1024)).unwrap();

        // Look for golden_size=128 — should fall back to 64
        let query = test_key("qwen3", 128);
        let found = registry.find_closest(&query).unwrap();
        assert_eq!(found.key.golden_size, 64);
    }

    #[test]
    fn test_find_closest_relaxes_guardrail() {
        let mut registry = VariantRegistry::new();

        // Register variant without guardrail
        let mut key_no_guard = test_key("qwen3", 64);
        key_no_guard.guardrail_enabled = false;
        registry.register(test_variant(key_no_guard, 1024)).unwrap();

        // Query with guardrail — should relax to non-guardrail variant
        let mut key_guard = test_key("qwen3", 64);
        key_guard.guardrail_enabled = true;
        let found = registry.find_closest(&key_guard).unwrap();
        assert!(!found.key.guardrail_enabled);
    }

    #[test]
    fn test_derive_key() {
        let key = VariantRegistry::derive_key(
            "qwen3",
            true,
            false,
            Some(SpecPhase::Draft),
            false,
            64,
            None,
            None,
        );
        assert_eq!(key.arch, "qwen3");
        assert!(key.moe_enabled);
        assert_eq!(key.spec_phase, Some(SpecPhase::Draft));
        assert_eq!(key.golden_size, 64);
    }

    #[test]
    fn test_display_key() {
        let key = VariantKey {
            arch: "qwen3moe".to_string(),
            moe_enabled: true,
            guardrail_enabled: false,
            spec_phase: Some(SpecPhase::Draft),
            rag_enabled: true,
            golden_size: 128,
            quant_type: None,
            kv_tier: None,
        };
        let s = format!("{}", key);
        assert!(s.contains("qwen3moe"));
        assert!(s.contains("moe_"));
        assert!(s.contains("draft_"));
    }

    #[test]
    fn test_max_footprint_tracking() {
        let mut registry = VariantRegistry::new();

        let k1 = test_key("qwen3", 32);
        registry.register(test_variant(k1, 512)).unwrap();
        assert_eq!(registry.max_footprint(), 512);

        let k2 = test_key("qwen3", 64);
        registry.register(test_variant(k2, 2048)).unwrap();
        assert_eq!(registry.max_footprint(), 2048);
    }

    #[test]
    fn test_available_budget() {
        let registry = VariantRegistry::with_l1i_budget(32768); // 32 KB
        let budget = registry.available_budget();
        assert_eq!(budget, (32768_f64 * 0.8) as usize); // 26214 bytes
    }
}
