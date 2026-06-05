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
    /// KIVI KV Cache 量化 (§19 KV-OPT-004)
    KiviQuant,
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
    /// Batch 维度装筒 (SPEC/20 REQ-BCI-009)
    /// None = 单序列模式 (当前行为)
    /// Some(golden_batch) = batch 模式，M 维度装筒值
    pub batch_golden_size: Option<usize>,
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
        if let Some(bs) = self.batch_golden_size {
            write!(f, "_b{}", bs)?;
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
    /// 1. 精确匹配 + golden_size 放松 + batch_golden_size 放松
    /// 2. 放松 spec_phase (None) + golden_size + batch_golden_size
    /// 3. 放松 guardrail_enabled (false) + spec_phase + golden_size + batch_golden_size
    /// 4. 放松 rag_enabled (false) + 全部放松
    pub fn find_closest(&self, key: &VariantKey) -> Option<&CompiledVariant> {
        // 原始约束
        if let Some(v) = self.try_relaxed(&key.arch, key.moe_enabled, key.guardrail_enabled,
            key.spec_phase, key.rag_enabled, &key.quant_type, &key.kv_tier, key.golden_size, key.batch_golden_size) {
            return Some(v);
        }
        // 放松 spec_phase=None
        if let Some(v) = self.try_relaxed(&key.arch, key.moe_enabled, key.guardrail_enabled,
            None, key.rag_enabled, &key.quant_type, &key.kv_tier, key.golden_size, key.batch_golden_size) {
            return Some(v);
        }
        // 放松 guardrail=false, spec_phase=original
        if let Some(v) = self.try_relaxed(&key.arch, key.moe_enabled, false,
            key.spec_phase, key.rag_enabled, &key.quant_type, &key.kv_tier, key.golden_size, key.batch_golden_size) {
            return Some(v);
        }
        // 放松 guardrail=false, spec_phase=None
        if let Some(v) = self.try_relaxed(&key.arch, key.moe_enabled, false,
            None, key.rag_enabled, &key.quant_type, &key.kv_tier, key.golden_size, key.batch_golden_size) {
            return Some(v);
        }
        // 放松 rag=false, guardrail=original, spec=original
        if let Some(v) = self.try_relaxed(&key.arch, key.moe_enabled, key.guardrail_enabled,
            key.spec_phase, false, &key.quant_type, &key.kv_tier, key.golden_size, key.batch_golden_size) {
            return Some(v);
        }
        // 全部放松
        if let Some(v) = self.try_relaxed(&key.arch, key.moe_enabled, false,
            None, false, &key.quant_type, &key.kv_tier, key.golden_size, key.batch_golden_size) {
            return Some(v);
        }
        None
    }

    /// 尝试精确匹配 + golden_size 放松 + batch_golden_size 放松
    fn try_relaxed(&self, arch: &str, moe: bool, guard: bool,
                   spec: Option<SpecPhase>, rag: bool, qt: &Option<String>,
                   kv_tier: &Option<String>, max_gs: usize, batch_gs: Option<usize>) -> Option<&CompiledVariant> {
        let exact = VariantKey {
            arch: arch.to_string(), moe_enabled: moe, guardrail_enabled: guard,
            spec_phase: spec, rag_enabled: rag, golden_size: max_gs,
            quant_type: qt.clone(), kv_tier: kv_tier.clone(),
            batch_golden_size: batch_gs,
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
                    && k.batch_golden_size == batch_gs
            })
            .map(|k| k.golden_size)
            .max();
        if let Some(best_gs) = best {
            let relaxed = VariantKey {
                arch: arch.to_string(), moe_enabled: moe, guardrail_enabled: guard,
                spec_phase: spec, rag_enabled: rag, golden_size: best_gs,
                quant_type: qt.clone(), kv_tier: kv_tier.clone(),
                batch_golden_size: batch_gs,
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
        batch_golden_size: Option<usize>,
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
            batch_golden_size,
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
            batch_golden_size: None,
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
            None,
        );
        assert_eq!(key.arch, "qwen3");
        assert!(key.moe_enabled);
        assert_eq!(key.spec_phase, Some(SpecPhase::Draft));
        assert_eq!(key.golden_size, 64);
        assert_eq!(key.batch_golden_size, None);
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
            batch_golden_size: None,
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

    // ---- Additional tests ----

    #[test]
    fn code_section_variants_distinct() {
        assert_ne!(CodeSection::Hot, CodeSection::Warm);
        assert_ne!(CodeSection::Warm, CodeSection::Cold);
        assert_ne!(CodeSection::Hot, CodeSection::Cold);
    }

    #[test]
    fn code_section_copy_clone_hash() {
        use std::collections::HashSet;
        let s = CodeSection::Hot;
        let s2 = s;
        assert_eq!(s, s2);
        let s3 = s.clone();
        assert_eq!(s3, CodeSection::Hot);
        let mut set = HashSet::new();
        set.insert(CodeSection::Hot);
        set.insert(CodeSection::Warm);
        set.insert(CodeSection::Cold);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn mechanism_id_variants_distinct() {
        let mechanisms = [
            MechanismId::Dense, MechanismId::MoeDispatch, MechanismId::SpecDraft,
            MechanismId::SpecVerify, MechanismId::GuardrailProbe, MechanismId::RagInjection,
            MechanismId::TurboQuantFwht, MechanismId::KiviQuant, MechanismId::GateFirstSkip,
            MechanismId::ResidualBypass, MechanismId::Telemetry, MechanismId::EarlyExit,
            MechanismId::RaggedCompaction,
        ];
        use std::collections::HashSet;
        let set: HashSet<_> = mechanisms.iter().collect();
        assert_eq!(set.len(), mechanisms.len(), "all MechanismId variants must be distinct");
    }

    #[test]
    fn spec_phase_variants() {
        assert_ne!(SpecPhase::Draft, SpecPhase::Verify);
        let s = SpecPhase::Draft;
        let s2 = s;
        assert_eq!(s, s2);
    }

    #[test]
    fn variant_key_equality_and_hash() {
        use std::collections::HashSet;
        let k1 = test_key("qwen3", 64);
        let k2 = test_key("qwen3", 64);
        assert_eq!(k1, k2);

        let mut k3 = test_key("qwen3", 64);
        k3.moe_enabled = true;
        assert_ne!(k1, k3);

        let mut set = HashSet::new();
        set.insert(k1.clone());
        set.insert(k3.clone());
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn variant_key_display_with_quant_and_kv() {
        let key = VariantKey {
            arch: "llama".into(),
            moe_enabled: false,
            guardrail_enabled: true,
            spec_phase: None,
            rag_enabled: false,
            golden_size: 32,
            quant_type: Some("awq4".into()),
            kv_tier: Some("fp16".into()),
            batch_golden_size: Some(4),
        };
        let s = format!("{}", key);
        assert!(s.contains("llama"));
        assert!(s.contains("g")); // guardrail
        assert!(s.contains("awq4"));
        assert!(s.contains("kvfp16"));
        assert!(s.contains("b4"));
    }

    #[test]
    fn variant_key_display_no_extras() {
        let key = test_key("test", 16);
        let s = format!("{}", key);
        assert!(s.contains("test"));
        assert!(s.contains("n")); // no guardrail, no rag
        assert!(s.contains("16"));
        assert!(!s.contains("_kv"));
        assert!(!s.contains("_b"));
    }

    #[test]
    fn compiled_variant_clone() {
        let key = test_key("qwen3", 64);
        let v = test_variant(key, 1024);
        let v2 = v.clone();
        assert_eq!(v2.instruction_footprint_bytes, 1024);
        assert_eq!(v2.mechanisms, vec![MechanismId::Dense]);
        assert_eq!(v2.section, CodeSection::Hot);
    }

    #[test]
    fn l1i_budget_exceeded_display() {
        let err = L1iBudgetExceeded {
            footprint: 50000,
            budget: 26214,
            suggestion: "disable telemetry".into(),
        };
        let s = err.to_string();
        assert!(s.contains("50000"));
        assert!(s.contains("26214"));
        assert!(s.contains("disable telemetry"));
    }

    #[test]
    fn l1i_budget_exceeded_is_error() {
        let err = L1iBudgetExceeded {
            footprint: 100,
            budget: 50,
            suggestion: "none".into(),
        };
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn registry_default() {
        let registry = VariantRegistry::default();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.max_footprint(), 0);
    }

    #[test]
    fn registry_new_vs_with_budget() {
        let r1 = VariantRegistry::new();
        let r2 = VariantRegistry::with_l1i_budget(65536);
        assert_ne!(r1.available_budget(), r2.available_budget());
    }

    #[test]
    fn registry_register_overwrite() {
        let mut registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);
        registry.register(test_variant(key.clone(), 512)).unwrap();
        registry.register(test_variant(key.clone(), 1024)).unwrap();
        assert_eq!(registry.len(), 1); // overwritten
        assert_eq!(registry.get(&key).unwrap().instruction_footprint_bytes, 1024);
    }

    #[test]
    fn registry_keys_and_variants() {
        let mut registry = VariantRegistry::new();
        let k1 = test_key("a", 32);
        let k2 = test_key("b", 64);
        registry.register(test_variant(k1, 100)).unwrap();
        registry.register(test_variant(k2, 200)).unwrap();
        assert_eq!(registry.keys().len(), 2);
        assert_eq!(registry.variants().len(), 2);
    }

    #[test]
    fn find_closest_empty_registry() {
        let registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);
        assert!(registry.find_closest(&key).is_none());
    }

    #[test]
    fn find_closest_exact_match() {
        let mut registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);
        registry.register(test_variant(key.clone(), 512)).unwrap();

        let query = test_key("qwen3", 64);
        let found = registry.find_closest(&query).unwrap();
        assert_eq!(found.key.golden_size, 64);
    }

    #[test]
    fn find_closest_relaxes_spec_phase() {
        let mut registry = VariantRegistry::new();
        let mut key_no_spec = test_key("qwen3", 64);
        key_no_spec.spec_phase = None;
        registry.register(test_variant(key_no_spec, 512)).unwrap();

        let mut query = test_key("qwen3", 64);
        query.spec_phase = Some(SpecPhase::Verify);
        let found = registry.find_closest(&query).unwrap();
        assert!(found.key.spec_phase.is_none());
    }

    #[test]
    fn find_closest_relaxes_rag() {
        let mut registry = VariantRegistry::new();
        let mut key_no_rag = test_key("qwen3", 64);
        key_no_rag.rag_enabled = false;
        registry.register(test_variant(key_no_rag, 512)).unwrap();

        let mut query = test_key("qwen3", 64);
        query.rag_enabled = true;
        let found = registry.find_closest(&query).unwrap();
        assert!(!found.key.rag_enabled);
    }

    #[test]
    fn majority_kv_tier_empty() {
        assert!(VariantRegistry::majority_kv_tier(&[]).is_none());
    }

    #[test]
    fn majority_kv_tier_single() {
        let tiers = vec!["fp16".to_string()];
        assert_eq!(VariantRegistry::majority_kv_tier(&tiers), Some("fp16".to_string()));
    }

    #[test]
    fn majority_kv_tier_majority_wins() {
        let tiers = vec![
            "fp16".to_string(), "fp16".to_string(), "fp16".to_string(),
            "int8".to_string(), "int8".to_string(),
        ];
        assert_eq!(VariantRegistry::majority_kv_tier(&tiers), Some("fp16".to_string()));
    }

    #[test]
    fn derive_key_full_fields() {
        let key = VariantRegistry::derive_key(
            "deepseek",
            true,
            true,
            Some(SpecPhase::Verify),
            true,
            128,
            Some("gptq4".into()),
            Some("int4".into()),
            Some(8),
        );
        assert_eq!(key.arch, "deepseek");
        assert!(key.moe_enabled);
        assert!(key.guardrail_enabled);
        assert_eq!(key.spec_phase, Some(SpecPhase::Verify));
        assert!(key.rag_enabled);
        assert_eq!(key.golden_size, 128);
        assert_eq!(key.quant_type, Some("gptq4".into()));
        assert_eq!(key.kv_tier, Some("int4".into()));
        assert_eq!(key.batch_golden_size, Some(8));
    }

    #[test]
    fn find_closest_with_batch_golden_size() {
        let mut registry = VariantRegistry::new();
        let mut key = test_key("qwen3", 64);
        key.batch_golden_size = Some(4);
        registry.register(test_variant(key, 512)).unwrap();

        let mut query = test_key("qwen3", 64);
        query.batch_golden_size = Some(4);
        let found = registry.find_closest(&query).unwrap();
        assert_eq!(found.key.batch_golden_size, Some(4));
    }

    // ---- New tests: trait implementations ----

    #[test]
    fn code_section_debug_format() {
        assert_eq!(format!("{:?}", CodeSection::Hot), "Hot");
        assert_eq!(format!("{:?}", CodeSection::Warm), "Warm");
        assert_eq!(format!("{:?}", CodeSection::Cold), "Cold");
    }

    #[test]
    fn mechanism_id_debug_format() {
        assert_eq!(format!("{:?}", MechanismId::Dense), "Dense");
        assert_eq!(format!("{:?}", MechanismId::MoeDispatch), "MoeDispatch");
        assert_eq!(format!("{:?}", MechanismId::SpecDraft), "SpecDraft");
        assert_eq!(format!("{:?}", MechanismId::SpecVerify), "SpecVerify");
        assert_eq!(format!("{:?}", MechanismId::GuardrailProbe), "GuardrailProbe");
        assert_eq!(format!("{:?}", MechanismId::RagInjection), "RagInjection");
        assert_eq!(format!("{:?}", MechanismId::TurboQuantFwht), "TurboQuantFwht");
        assert_eq!(format!("{:?}", MechanismId::KiviQuant), "KiviQuant");
        assert_eq!(format!("{:?}", MechanismId::GateFirstSkip), "GateFirstSkip");
        assert_eq!(format!("{:?}", MechanismId::ResidualBypass), "ResidualBypass");
        assert_eq!(format!("{:?}", MechanismId::Telemetry), "Telemetry");
        assert_eq!(format!("{:?}", MechanismId::EarlyExit), "EarlyExit");
        assert_eq!(format!("{:?}", MechanismId::RaggedCompaction), "RaggedCompaction");
    }

    #[test]
    fn mechanism_id_copy_clone() {
        let m = MechanismId::MoeDispatch;
        let m2 = m; // Copy
        assert_eq!(m, m2);
        let m3 = m.clone();
        assert_eq!(m3, MechanismId::MoeDispatch);
    }

    #[test]
    fn spec_phase_debug_clone_copy_hash() {
        assert_eq!(format!("{:?}", SpecPhase::Draft), "Draft");
        assert_eq!(format!("{:?}", SpecPhase::Verify), "Verify");

        let p = SpecPhase::Verify;
        let p2 = p; // Copy
        assert_eq!(p, p2);
        let p3 = p.clone();
        assert_eq!(p3, SpecPhase::Verify);

        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SpecPhase::Draft);
        set.insert(SpecPhase::Verify);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn variant_key_clone_independent() {
        let k1 = test_key("qwen3", 64);
        let k2 = k1.clone();
        assert_eq!(k1, k2);
        // Cloned key is independent (modifying the clone doesn't affect original is implied
        // since String fields are cloned)
        assert_eq!(k2.arch, "qwen3");
    }

    #[test]
    fn compiled_variant_debug_format() {
        let key = test_key("qwen3", 64);
        let v = test_variant(key, 1024);
        let debug = format!("{:?}", v);
        assert!(debug.contains("CompiledVariant"));
        assert!(debug.contains("instruction_footprint_bytes"));
    }

    #[test]
    fn l1i_budget_exceeded_debug_clone() {
        let err = L1iBudgetExceeded {
            footprint: 99999,
            budget: 1000,
            suggestion: "test suggestion".into(),
        };
        let debug = format!("{:?}", err);
        assert!(debug.contains("L1iBudgetExceeded"));

        let err2 = err.clone();
        assert_eq!(err2.footprint, 99999);
        assert_eq!(err2.budget, 1000);
        assert_eq!(err2.suggestion, "test suggestion");
    }

    // ---- New tests: Display edge cases ----

    #[test]
    fn variant_key_display_no_moe_no_spec() {
        let key = test_key("llama", 32);
        let s = format!("{}", key);
        // moe_enabled=false => no "moe_" prefix
        assert!(!s.contains("moe_"));
        // spec_phase=None => no "draft_" or "verify_"
        assert!(!s.contains("draft_"));
        assert!(!s.contains("verify_"));
    }

    #[test]
    fn variant_key_display_verify_phase() {
        let key = VariantKey {
            arch: "llama".into(),
            moe_enabled: false,
            guardrail_enabled: false,
            spec_phase: Some(SpecPhase::Verify),
            rag_enabled: false,
            golden_size: 64,
            quant_type: None,
            kv_tier: None,
            batch_golden_size: None,
        };
        let s = format!("{}", key);
        assert!(s.contains("verify_"));
    }

    #[test]
    fn variant_key_display_guardrail_and_rag_flags() {
        // Both enabled: format is "{arch}_{}{}_g_r_{golden_size}"
        let key_enabled = VariantKey {
            arch: "test".into(),
            moe_enabled: false,
            guardrail_enabled: true,
            spec_phase: None,
            rag_enabled: true,
            golden_size: 32,
            quant_type: None,
            kv_tier: None,
            batch_golden_size: None,
        };
        let s = format!("{}", key_enabled);
        // guardrail=g, rag=r, separated by _
        assert!(s.contains("g_r_"));

        // Both disabled: format has "n_n_" for guardrail and rag
        let key_disabled = VariantKey {
            arch: "test".into(),
            moe_enabled: false,
            guardrail_enabled: false,
            spec_phase: None,
            rag_enabled: false,
            golden_size: 32,
            quant_type: None,
            kv_tier: None,
            batch_golden_size: None,
        };
        let s = format!("{}", key_disabled);
        assert!(s.contains("n_n_"));
    }

    #[test]
    fn variant_key_display_kv_tier_only() {
        let key = VariantKey {
            arch: "model".into(),
            moe_enabled: false,
            guardrail_enabled: false,
            spec_phase: None,
            rag_enabled: false,
            golden_size: 16,
            quant_type: None,
            kv_tier: Some("int8".into()),
            batch_golden_size: None,
        };
        let s = format!("{}", key);
        assert!(s.contains("kvint8"));
        assert!(!s.contains("_b"));
    }

    #[test]
    fn variant_key_display_batch_only() {
        let key = VariantKey {
            arch: "model".into(),
            moe_enabled: false,
            guardrail_enabled: false,
            spec_phase: None,
            rag_enabled: false,
            golden_size: 16,
            quant_type: None,
            kv_tier: None,
            batch_golden_size: Some(8),
        };
        let s = format!("{}", key);
        assert!(s.contains("b8"));
        assert!(!s.contains("_kv"));
    }

    // ---- New tests: find_closest edge cases ----

    #[test]
    fn find_closest_arch_mismatch_returns_none() {
        let mut registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);
        registry.register(test_variant(key, 512)).unwrap();

        let query = test_key("llama", 64); // different arch
        assert!(registry.find_closest(&query).is_none());
    }

    #[test]
    fn find_closest_quant_type_mismatch_returns_none() {
        let mut registry = VariantRegistry::new();
        let mut key = test_key("qwen3", 64);
        key.quant_type = Some("awq4".into());
        registry.register(test_variant(key, 512)).unwrap();

        let mut query = test_key("qwen3", 64);
        query.quant_type = Some("gptq4".into());
        assert!(registry.find_closest(&query).is_none());
    }

    #[test]
    fn find_closest_quant_type_exact_match() {
        let mut registry = VariantRegistry::new();
        let mut key = test_key("qwen3", 64);
        key.quant_type = Some("awq4".into());
        registry.register(test_variant(key, 512)).unwrap();

        let mut query = test_key("qwen3", 64);
        query.quant_type = Some("awq4".into());
        let found = registry.find_closest(&query).unwrap();
        assert_eq!(found.key.quant_type.as_deref(), Some("awq4"));
    }

    #[test]
    fn find_closest_kv_tier_mismatch_returns_none() {
        let mut registry = VariantRegistry::new();
        let mut key = test_key("qwen3", 64);
        key.kv_tier = Some("fp16".into());
        registry.register(test_variant(key, 512)).unwrap();

        let mut query = test_key("qwen3", 64);
        query.kv_tier = Some("int4".into());
        assert!(registry.find_closest(&query).is_none());
    }

    #[test]
    fn find_closest_golden_size_picks_best_under() {
        let mut registry = VariantRegistry::new();
        let k16 = test_key("qwen3", 16);
        registry.register(test_variant(k16, 256)).unwrap();
        let k64 = test_key("qwen3", 64);
        registry.register(test_variant(k64, 512)).unwrap();
        let k128 = test_key("qwen3", 128);
        registry.register(test_variant(k128, 1024)).unwrap();

        // Query for 256 — should pick 128 (largest <= 256)
        let query = test_key("qwen3", 256);
        let found = registry.find_closest(&query).unwrap();
        assert_eq!(found.key.golden_size, 128);
    }

    #[test]
    fn find_closest_batch_golden_size_mismatch_returns_none() {
        let mut registry = VariantRegistry::new();
        let mut key = test_key("qwen3", 64);
        key.batch_golden_size = Some(4);
        registry.register(test_variant(key, 512)).unwrap();

        let mut query = test_key("qwen3", 64);
        query.batch_golden_size = Some(8); // different batch size
        assert!(registry.find_closest(&query).is_none());
    }

    #[test]
    fn find_closest_relaxes_spec_and_guardrail_together() {
        let mut registry = VariantRegistry::new();
        // Register a bare variant: no spec, no guardrail, no rag
        let mut bare_key = test_key("qwen3", 64);
        bare_key.spec_phase = None;
        bare_key.guardrail_enabled = false;
        bare_key.rag_enabled = false;
        registry.register(test_variant(bare_key, 512)).unwrap();

        // Query with spec_phase=Verify, guardrail=true, rag=true
        let mut query = test_key("qwen3", 64);
        query.spec_phase = Some(SpecPhase::Verify);
        query.guardrail_enabled = true;
        query.rag_enabled = true;
        let found = registry.find_closest(&query).unwrap();
        assert!(found.key.spec_phase.is_none());
        assert!(!found.key.guardrail_enabled);
        assert!(!found.key.rag_enabled);
    }

    // ---- New tests: majority_kv_tier edge cases ----

    #[test]
    fn majority_kv_tier_tie_returns_one_of_ties() {
        let tiers = vec![
            "fp16".to_string(),
            "int8".to_string(),
        ];
        let result = VariantRegistry::majority_kv_tier(&tiers);
        assert!(result.is_some());
        // Either "fp16" or "int8" is acceptable (tie)
        let val = result.unwrap();
        assert!(val == "fp16" || val == "int8");
    }

    #[test]
    fn majority_kv_tier_all_same() {
        let tiers = vec![
            "int4".to_string(),
            "int4".to_string(),
            "int4".to_string(),
        ];
        assert_eq!(VariantRegistry::majority_kv_tier(&tiers), Some("int4".to_string()));
    }

    // ---- New tests: budget edge cases ----

    #[test]
    fn register_exactly_at_budget_succeeds() {
        let budget = 4096_usize;
        let mut registry = VariantRegistry::with_l1i_budget(budget);
        let available = registry.available_budget(); // 80% of budget
        let key = test_key("qwen3", 64);
        let variant = test_variant(key, available); // exactly at budget
        assert!(registry.register(variant).is_ok());
    }

    #[test]
    fn register_one_byte_over_budget_fails() {
        let budget = 4096_usize;
        let mut registry = VariantRegistry::with_l1i_budget(budget);
        let available = registry.available_budget();
        let key = test_key("qwen3", 64);
        let variant = test_variant(key, available + 1);
        assert!(registry.register(variant).is_err());
    }

    // ---- New tests: CompiledVariant with varied fields ----

    #[test]
    fn compiled_variant_with_warm_section() {
        let key = test_key("qwen3", 64);
        let v = CompiledVariant {
            code: vec![0xCC; 256],
            instruction_footprint_bytes: 256,
            mechanisms: vec![MechanismId::Dense, MechanismId::Telemetry],
            section: CodeSection::Warm,
            key,
        };
        assert_eq!(v.section, CodeSection::Warm);
        assert_eq!(v.mechanisms.len(), 2);
        assert_eq!(v.code.len(), 256);
    }

    #[test]
    fn compiled_variant_with_cold_section() {
        let key = test_key("qwen3", 64);
        let v = CompiledVariant {
            code: vec![],
            instruction_footprint_bytes: 0,
            mechanisms: vec![MechanismId::EarlyExit],
            section: CodeSection::Cold,
            key,
        };
        assert_eq!(v.section, CodeSection::Cold);
        assert_eq!(v.instruction_footprint_bytes, 0);
    }

    #[test]
    fn compiled_variant_with_all_mechanisms() {
        let key = test_key("model", 32);
        let all_mechanisms = vec![
            MechanismId::Dense,
            MechanismId::MoeDispatch,
            MechanismId::SpecDraft,
            MechanismId::SpecVerify,
            MechanismId::GuardrailProbe,
            MechanismId::RagInjection,
            MechanismId::TurboQuantFwht,
            MechanismId::KiviQuant,
            MechanismId::GateFirstSkip,
            MechanismId::ResidualBypass,
            MechanismId::Telemetry,
            MechanismId::EarlyExit,
            MechanismId::RaggedCompaction,
        ];
        let v = CompiledVariant {
            code: vec![0x90; 4096],
            instruction_footprint_bytes: 4096,
            mechanisms: all_mechanisms.clone(),
            section: CodeSection::Hot,
            key,
        };
        assert_eq!(v.mechanisms.len(), 13);
        assert_eq!(v.mechanisms, all_mechanisms);
    }

    // ---- New tests: registry get returns correct variant ----

    #[test]
    fn registry_get_missing_key_returns_none() {
        let registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);
        assert!(registry.get(&key).is_none());
    }

    #[test]
    fn max_footprint_decreases_on_overwrite_smaller() {
        let mut registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);

        // Register with large footprint
        registry.register(test_variant(key.clone(), 2048)).unwrap();
        assert_eq!(registry.max_footprint(), 2048);

        // Overwrite with smaller footprint — max should still be 2048
        registry.register(test_variant(key.clone(), 512)).unwrap();
        assert_eq!(registry.max_footprint(), 2048);
    }

    // ---- Additional tests (round 2) ----

    #[test]
    fn registry_new_empty_state() {
        // Arrange & Act
        let registry = VariantRegistry::new();
        // Assert
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.max_footprint(), 0);
        assert!(registry.keys().is_empty());
        assert!(registry.variants().is_empty());
    }

    #[test]
    fn registry_with_l1i_budget_zero() {
        // Arrange
        let registry = VariantRegistry::with_l1i_budget(0);
        // Act
        let budget = registry.available_budget();
        // Assert: 80% of 0 is 0
        assert_eq!(budget, 0);
    }

    #[test]
    fn registry_with_l1i_budget_large_value() {
        // Arrange
        let registry = VariantRegistry::with_l1i_budget(1024 * 1024); // 1 MB
        // Act
        let budget = registry.available_budget();
        // Assert: 80% of 1MB = 838860 bytes
        assert_eq!(budget, (1024 * 1024_usize * 8) / 10);
    }

    #[test]
    fn registry_register_zero_footprint_succeeds() {
        // Arrange
        let mut registry = VariantRegistry::with_l1i_budget(0);
        let key = test_key("qwen3", 64);
        let variant = CompiledVariant {
            code: vec![],
            instruction_footprint_bytes: 0,
            mechanisms: vec![],
            section: CodeSection::Hot,
            key,
        };
        // Act
        let result = registry.register(variant);
        // Assert: zero footprint fits in zero budget
        assert!(result.is_ok());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn registry_get_on_empty_returns_none() {
        // Arrange
        let registry = VariantRegistry::new();
        let key = test_key("any", 1);
        // Act & Assert
        assert!(registry.get(&key).is_none());
    }

    #[test]
    fn code_section_all_variants_exhaustive() {
        // Arrange
        let variants = [CodeSection::Hot, CodeSection::Warm, CodeSection::Cold];
        // Act & Assert: exactly 3 distinct variants
        use std::collections::HashSet;
        let set: HashSet<_> = variants.iter().collect();
        assert_eq!(set.len(), 3);
        // Copy semantics
        for v in &variants {
            let copied = *v;
            assert_eq!(*v, copied);
        }
    }

    #[test]
    fn spec_phase_all_variants_copy_and_hash() {
        // Arrange
        use std::collections::HashSet;
        let mut set = HashSet::new();
        // Act
        set.insert(SpecPhase::Draft);
        set.insert(SpecPhase::Verify);
        // Assert: exactly 2 variants, both hashable and copy
        assert_eq!(set.len(), 2);
        let d = SpecPhase::Draft;
        let d_copy = d;
        assert_eq!(d, d_copy);
    }

    #[test]
    fn variant_key_different_golden_sizes_not_equal() {
        // Arrange
        let k1 = test_key("qwen3", 32);
        let k2 = test_key("qwen3", 64);
        // Act & Assert
        assert_ne!(k1, k2);
    }

    #[test]
    fn variant_key_moe_field_affects_equality() {
        // Arrange
        let mut k1 = test_key("qwen3", 64);
        let mut k2 = test_key("qwen3", 64);
        k1.moe_enabled = true;
        k2.moe_enabled = false;
        // Act & Assert
        assert_ne!(k1, k2);
    }

    #[test]
    fn variant_key_rag_field_affects_equality() {
        // Arrange
        let mut k1 = test_key("qwen3", 64);
        let mut k2 = test_key("qwen3", 64);
        k1.rag_enabled = true;
        k2.rag_enabled = false;
        // Act & Assert
        assert_ne!(k1, k2);
    }

    #[test]
    fn variant_key_batch_golden_size_affects_equality() {
        // Arrange
        let mut k1 = test_key("qwen3", 64);
        let mut k2 = test_key("qwen3", 64);
        k1.batch_golden_size = Some(4);
        k2.batch_golden_size = Some(8);
        // Act & Assert
        assert_ne!(k1, k2);
    }

    #[test]
    fn variant_key_none_vs_some_batch_golden_size_not_equal() {
        // Arrange
        let mut k1 = test_key("qwen3", 64);
        let mut k2 = test_key("qwen3", 64);
        k1.batch_golden_size = None;
        k2.batch_golden_size = Some(1);
        // Act & Assert
        assert_ne!(k1, k2);
    }

    #[test]
    fn compiled_variant_section_field_access() {
        // Arrange
        let key = test_key("qwen3", 64);
        let v = CompiledVariant {
            code: vec![0x00; 128],
            instruction_footprint_bytes: 128,
            mechanisms: vec![MechanismId::KiviQuant],
            section: CodeSection::Cold,
            key,
        };
        // Act & Assert: direct field access
        assert_eq!(v.section, CodeSection::Cold);
        assert_eq!(v.instruction_footprint_bytes, 128);
        assert_eq!(v.mechanisms.len(), 1);
        assert_eq!(v.code[0], 0x00);
        assert_eq!(v.code.len(), 128);
    }

    #[test]
    fn l1i_budget_exceeded_display_contains_all_fields() {
        // Arrange
        let err = L1iBudgetExceeded {
            footprint: 12345,
            budget: 6789,
            suggestion: "reduce mechanisms".into(),
        };
        // Act
        let display = format!("{}", err);
        // Assert
        assert!(display.contains("12345"), "footprint must appear");
        assert!(display.contains("6789"), "budget must appear");
        assert!(display.contains("reduce mechanisms"), "suggestion must appear");
        assert!(display.contains("L1i budget exceeded"));
    }

    #[test]
    fn registry_find_closest_moe_mismatch_returns_none() {
        // Arrange: register a non-MoE variant
        let mut registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);
        registry.register(test_variant(key, 512)).unwrap();

        // Act: query for MoE variant
        let mut query = test_key("qwen3", 64);
        query.moe_enabled = true;
        // Assert
        assert!(registry.find_closest(&query).is_none());
    }

    #[test]
    fn registry_max_footprint_updated_on_larger_variant() {
        // Arrange
        let mut registry = VariantRegistry::new();
        let k1 = test_key("a", 32);
        let k2 = test_key("b", 64);
        // Act
        registry.register(test_variant(k1, 100)).unwrap();
        assert_eq!(registry.max_footprint(), 100);
        registry.register(test_variant(k2, 300)).unwrap();
        // Assert: max updated to larger value
        assert_eq!(registry.max_footprint(), 300);
    }

    #[test]
    fn registry_default_equals_new() {
        // Arrange & Act
        let r1 = VariantRegistry::default();
        let r2 = VariantRegistry::new();
        // Assert: both have same initial state
        assert_eq!(r1.is_empty(), r2.is_empty());
        assert_eq!(r1.len(), r2.len());
        assert_eq!(r1.max_footprint(), r2.max_footprint());
        assert_eq!(r1.available_budget(), r2.available_budget());
    }

    #[test]
    fn registry_find_closest_prefers_exact_match_over_relaxed() {
        // Arrange: register both exact match and a smaller golden_size
        let mut registry = VariantRegistry::new();
        let k32 = test_key("qwen3", 32);
        registry.register(test_variant(k32, 256)).unwrap();
        let k64 = test_key("qwen3", 64);
        registry.register(test_variant(k64, 512)).unwrap();

        // Act: query for golden_size=64
        let query = test_key("qwen3", 64);
        let found = registry.find_closest(&query).unwrap();
        // Assert: exact match preferred
        assert_eq!(found.key.golden_size, 64);
        assert_eq!(found.instruction_footprint_bytes, 512);
    }

    #[test]
    fn derive_key_with_all_none_optional_fields() {
        // Arrange & Act
        let key = VariantRegistry::derive_key(
            "test_arch",
            false,
            false,
            None,
            false,
            32,
            None,
            None,
            None,
        );
        // Assert
        assert_eq!(key.arch, "test_arch");
        assert!(!key.moe_enabled);
        assert!(!key.guardrail_enabled);
        assert!(key.spec_phase.is_none());
        assert!(!key.rag_enabled);
        assert_eq!(key.golden_size, 32);
        assert!(key.quant_type.is_none());
        assert!(key.kv_tier.is_none());
        assert!(key.batch_golden_size.is_none());
    }

    // ---- Round 3: additional coverage (~45 tests) ----

    #[test]
    fn variant_key_display_moe_only() {
        let key = VariantKey {
            arch: "deepseek".into(),
            moe_enabled: true,
            guardrail_enabled: false,
            spec_phase: None,
            rag_enabled: false,
            golden_size: 64,
            quant_type: None,
            kv_tier: None,
            batch_golden_size: None,
        };
        let s = format!("{}", key);
        assert!(s.contains("moe_"));
        assert!(!s.contains("draft_"));
        assert!(!s.contains("verify_"));
    }

    #[test]
    fn variant_key_display_empty_arch() {
        let key = VariantKey {
            arch: String::new(),
            moe_enabled: false,
            guardrail_enabled: false,
            spec_phase: None,
            rag_enabled: false,
            golden_size: 1,
            quant_type: None,
            kv_tier: None,
            batch_golden_size: None,
        };
        let s = format!("{}", key);
        assert!(s.starts_with("_"));
        assert!(s.contains("1"));
    }

    #[test]
    fn variant_key_display_golden_size_zero() {
        let key = VariantKey {
            arch: "m".into(),
            moe_enabled: false,
            guardrail_enabled: false,
            spec_phase: None,
            rag_enabled: false,
            golden_size: 0,
            quant_type: None,
            kv_tier: None,
            batch_golden_size: None,
        };
        let s = format!("{}", key);
        assert!(s.contains("_0"));
    }

    #[test]
    fn variant_key_display_batch_golden_size_zero() {
        let key = VariantKey {
            arch: "m".into(),
            moe_enabled: false,
            guardrail_enabled: false,
            spec_phase: None,
            rag_enabled: false,
            golden_size: 64,
            quant_type: None,
            kv_tier: None,
            batch_golden_size: Some(0),
        };
        let s = format!("{}", key);
        assert!(s.contains("b0"));
    }

    #[test]
    fn variant_key_display_all_features_enabled() {
        let key = VariantKey {
            arch: "full".into(),
            moe_enabled: true,
            guardrail_enabled: true,
            spec_phase: Some(SpecPhase::Draft),
            rag_enabled: true,
            golden_size: 256,
            quant_type: Some("nvfp4".into()),
            kv_tier: Some("fp8".into()),
            batch_golden_size: Some(16),
        };
        let s = format!("{}", key);
        assert!(s.contains("moe_"));
        assert!(s.contains("draft_"));
        assert!(s.contains("g_r_"));
        assert!(s.contains("256"));
        assert!(s.contains("nvfp4"));
        assert!(s.contains("kvfp8"));
        assert!(s.contains("b16"));
    }

    #[test]
    fn variant_key_quant_type_none_vs_some_affects_equality() {
        let mut k1 = test_key("qwen3", 64);
        let mut k2 = test_key("qwen3", 64);
        k1.quant_type = None;
        k2.quant_type = Some("awq4".into());
        assert_ne!(k1, k2);
    }

    #[test]
    fn variant_key_kv_tier_none_vs_some_affects_equality() {
        let mut k1 = test_key("qwen3", 64);
        let mut k2 = test_key("qwen3", 64);
        k1.kv_tier = None;
        k2.kv_tier = Some("fp16".into());
        assert_ne!(k1, k2);
    }

    #[test]
    fn variant_key_different_quant_types_not_equal() {
        let mut k1 = test_key("qwen3", 64);
        let mut k2 = test_key("qwen3", 64);
        k1.quant_type = Some("awq4".into());
        k2.quant_type = Some("gptq4".into());
        assert_ne!(k1, k2);
    }

    #[test]
    fn variant_key_same_quant_type_equal() {
        let mut k1 = test_key("qwen3", 64);
        let mut k2 = test_key("qwen3", 64);
        k1.quant_type = Some("awq4".into());
        k2.quant_type = Some("awq4".into());
        assert_eq!(k1, k2);
    }

    #[test]
    fn variant_key_different_kv_tiers_not_equal() {
        let mut k1 = test_key("qwen3", 64);
        let mut k2 = test_key("qwen3", 64);
        k1.kv_tier = Some("fp16".into());
        k2.kv_tier = Some("int8".into());
        assert_ne!(k1, k2);
    }

    #[test]
    fn variant_key_same_kv_tier_equal() {
        let mut k1 = test_key("qwen3", 64);
        let mut k2 = test_key("qwen3", 64);
        k1.kv_tier = Some("fp16".into());
        k2.kv_tier = Some("fp16".into());
        assert_eq!(k1, k2);
    }

    #[test]
    fn variant_key_arch_case_sensitive() {
        let k1 = test_key("Qwen3", 64);
        let k2 = test_key("qwen3", 64);
        assert_ne!(k1, k2);
    }

    #[test]
    fn variant_key_spec_phase_draft_vs_verify_not_equal() {
        let mut k1 = test_key("qwen3", 64);
        let mut k2 = test_key("qwen3", 64);
        k1.spec_phase = Some(SpecPhase::Draft);
        k2.spec_phase = Some(SpecPhase::Verify);
        assert_ne!(k1, k2);
    }

    #[test]
    fn variant_key_spec_phase_some_vs_none_not_equal() {
        let mut k1 = test_key("qwen3", 64);
        let mut k2 = test_key("qwen3", 64);
        k1.spec_phase = Some(SpecPhase::Draft);
        k2.spec_phase = None;
        assert_ne!(k1, k2);
    }

    #[test]
    fn registry_register_multiple_different_archs() {
        let mut registry = VariantRegistry::new();
        let k1 = test_key("qwen3", 64);
        let k2 = test_key("llama", 64);
        let k3 = test_key("deepseek", 64);
        registry.register(test_variant(k1, 512)).unwrap();
        registry.register(test_variant(k2, 1024)).unwrap();
        registry.register(test_variant(k3, 2048)).unwrap();
        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn registry_get_by_full_key_match() {
        let mut registry = VariantRegistry::new();
        let mut key = test_key("qwen3", 64);
        key.moe_enabled = true;
        key.guardrail_enabled = true;
        key.quant_type = Some("awq4".into());
        registry.register(test_variant(key.clone(), 1024)).unwrap();

        let found = registry.get(&key).unwrap();
        assert!(found.key.moe_enabled);
        assert!(found.key.guardrail_enabled);
        assert_eq!(found.key.quant_type.as_deref(), Some("awq4"));
    }

    #[test]
    fn registry_get_wrong_spec_phase_returns_none() {
        let mut registry = VariantRegistry::new();
        let mut key = test_key("qwen3", 64);
        key.spec_phase = Some(SpecPhase::Draft);
        registry.register(test_variant(key, 512)).unwrap();

        let mut query = test_key("qwen3", 64);
        query.spec_phase = Some(SpecPhase::Verify);
        assert!(registry.get(&query).is_none());
    }

    #[test]
    fn find_closest_golden_size_zero_no_match() {
        let mut registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);
        registry.register(test_variant(key, 512)).unwrap();

        let query = test_key("qwen3", 0);
        assert!(registry.find_closest(&query).is_none());
    }

    #[test]
    fn find_closest_golden_size_one_match_when_registered() {
        let mut registry = VariantRegistry::new();
        let key = test_key("qwen3", 1);
        registry.register(test_variant(key, 64)).unwrap();

        let query = test_key("qwen3", 1);
        let found = registry.find_closest(&query).unwrap();
        assert_eq!(found.key.golden_size, 1);
    }

    #[test]
    fn find_closest_relaxes_guardrail_before_rag() {
        let mut registry = VariantRegistry::new();
        // Register: guardrail=false, rag=true
        let mut key = test_key("qwen3", 64);
        key.guardrail_enabled = false;
        key.rag_enabled = true;
        registry.register(test_variant(key, 512)).unwrap();

        // Query: guardrail=true, rag=true
        let mut query = test_key("qwen3", 64);
        query.guardrail_enabled = true;
        query.rag_enabled = true;
        let found = registry.find_closest(&query).unwrap();
        assert!(!found.key.guardrail_enabled);
        assert!(found.key.rag_enabled);
    }

    #[test]
    fn find_closest_relaxes_both_guardrail_and_spec() {
        let mut registry = VariantRegistry::new();
        let mut key = test_key("qwen3", 64);
        key.guardrail_enabled = false;
        key.spec_phase = None;
        registry.register(test_variant(key, 512)).unwrap();

        let mut query = test_key("qwen3", 64);
        query.guardrail_enabled = true;
        query.spec_phase = Some(SpecPhase::Draft);
        let found = registry.find_closest(&query).unwrap();
        assert!(!found.key.guardrail_enabled);
        assert!(found.key.spec_phase.is_none());
    }

    #[test]
    fn find_closest_uses_largest_golden_size_under_query() {
        let mut registry = VariantRegistry::new();
        let k10 = test_key("qwen3", 10);
        let k20 = test_key("qwen3", 20);
        let k30 = test_key("qwen3", 30);
        registry.register(test_variant(k10, 100)).unwrap();
        registry.register(test_variant(k20, 200)).unwrap();
        registry.register(test_variant(k30, 300)).unwrap();

        let query = test_key("qwen3", 25);
        let found = registry.find_closest(&query).unwrap();
        assert_eq!(found.key.golden_size, 20);
        assert_eq!(found.instruction_footprint_bytes, 200);
    }

    #[test]
    fn find_closest_does_not_match_higher_golden_size() {
        let mut registry = VariantRegistry::new();
        let k128 = test_key("qwen3", 128);
        registry.register(test_variant(k128, 1024)).unwrap();

        let query = test_key("qwen3", 64);
        // golden_size=128 > 64, so should NOT match
        assert!(registry.find_closest(&query).is_none());
    }

    #[test]
    fn find_closest_with_none_quant_matches_none_quant() {
        let mut registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);
        registry.register(test_variant(key, 512)).unwrap();

        let mut query = test_key("qwen3", 64);
        query.quant_type = None;
        let found = registry.find_closest(&query).unwrap();
        assert!(found.key.quant_type.is_none());
    }

    #[test]
    fn majority_kv_tier_single_element_vec() {
        let tiers = vec!["mxpf4".to_string()];
        assert_eq!(VariantRegistry::majority_kv_tier(&tiers), Some("mxpf4".to_string()));
    }

    #[test]
    fn majority_kv_tier_three_way_tie_returns_one() {
        let tiers = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ];
        let result = VariantRegistry::majority_kv_tier(&tiers);
        assert!(result.is_some());
        let val = result.unwrap();
        assert!(val == "a" || val == "b" || val == "c");
    }

    #[test]
    fn majority_kv_tier_landslide() {
        let tiers = vec![
            "x".to_string(), "x".to_string(), "x".to_string(),
            "x".to_string(), "x".to_string(),
            "y".to_string(),
        ];
        assert_eq!(VariantRegistry::majority_kv_tier(&tiers), Some("x".to_string()));
    }

    #[test]
    fn compiled_variant_with_empty_mechanisms() {
        let key = test_key("qwen3", 64);
        let v = CompiledVariant {
            code: vec![0x90],
            instruction_footprint_bytes: 1,
            mechanisms: vec![],
            section: CodeSection::Hot,
            key,
        };
        assert!(v.mechanisms.is_empty());
        assert_eq!(v.code.len(), 1);
    }

    #[test]
    fn compiled_variant_with_duplicate_mechanisms() {
        let key = test_key("qwen3", 64);
        let v = CompiledVariant {
            code: vec![],
            instruction_footprint_bytes: 0,
            mechanisms: vec![MechanismId::Dense, MechanismId::Dense],
            section: CodeSection::Warm,
            key,
        };
        assert_eq!(v.mechanisms.len(), 2);
        assert_eq!(v.mechanisms[0], v.mechanisms[1]);
    }

    #[test]
    fn compiled_variant_large_code_vec() {
        let key = test_key("qwen3", 64);
        let size = 1024 * 1024; // 1MB code
        let v = CompiledVariant {
            code: vec![0xCC; size],
            instruction_footprint_bytes: size,
            mechanisms: vec![MechanismId::Dense],
            section: CodeSection::Hot,
            key,
        };
        assert_eq!(v.code.len(), size);
        assert_eq!(v.instruction_footprint_bytes, size);
    }

    #[test]
    fn compiled_variant_key_matches_registered() {
        let mut registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);
        registry.register(test_variant(key.clone(), 512)).unwrap();

        let found = registry.get(&key).unwrap();
        assert_eq!(found.key.arch, "qwen3");
        assert_eq!(found.key.golden_size, 64);
        assert_eq!(found.key.moe_enabled, false);
    }

    #[test]
    fn registry_keys_returns_correct_archs() {
        let mut registry = VariantRegistry::new();
        registry.register(test_variant(test_key("a", 32), 100)).unwrap();
        registry.register(test_variant(test_key("b", 64), 200)).unwrap();

        let archs: Vec<&str> = registry.keys().iter().map(|k| k.arch.as_str()).collect();
        assert!(archs.contains(&"a"));
        assert!(archs.contains(&"b"));
    }

    #[test]
    fn registry_variants_returns_correct_footprints() {
        let mut registry = VariantRegistry::new();
        registry.register(test_variant(test_key("a", 32), 111)).unwrap();
        registry.register(test_variant(test_key("b", 64), 222)).unwrap();

        let footprints: Vec<usize> = registry.variants().iter()
            .map(|v| v.instruction_footprint_bytes)
            .collect();
        assert!(footprints.contains(&111));
        assert!(footprints.contains(&222));
    }

    #[test]
    fn available_budget_80_percent() {
        let registry = VariantRegistry::with_l1i_budget(10000);
        let budget = registry.available_budget();
        assert_eq!(budget, 8000);
    }

    #[test]
    fn available_budget_default_32k() {
        let registry = VariantRegistry::new();
        let budget = registry.available_budget();
        assert_eq!(budget, (32768_f64 * 0.8) as usize);
    }

    #[test]
    fn register_error_contains_correct_suggestion() {
        let mut registry = VariantRegistry::with_l1i_budget(100);
        let key = test_key("qwen3", 64);
        let variant = test_variant(key, 1000);
        let err = registry.register(variant).unwrap_err();
        assert!(!err.suggestion.is_empty());
    }

    #[test]
    fn register_error_footprint_greater_than_budget() {
        let mut registry = VariantRegistry::with_l1i_budget(256);
        let key = test_key("qwen3", 64);
        let variant = test_variant(key, 1000);
        let err = registry.register(variant).unwrap_err();
        assert!(err.footprint > err.budget);
        assert_eq!(err.footprint, 1000);
    }

    #[test]
    fn derive_key_accepts_owned_string() {
        let arch = String::from("qwen3");
        let key = VariantRegistry::derive_key(
            arch,
            false,
            false,
            None,
            false,
            32,
            None,
            None,
            None,
        );
        assert_eq!(key.arch, "qwen3");
    }

    #[test]
    fn derive_key_accepts_str_ref() {
        let key = VariantRegistry::derive_key(
            "qwen3",
            false,
            false,
            None,
            false,
            32,
            None,
            None,
            None,
        );
        assert_eq!(key.arch, "qwen3");
    }

    #[test]
    fn derive_key_guardrail_active_field() {
        let key = VariantRegistry::derive_key(
            "test", false, true, None, false, 32, None, None, None,
        );
        assert!(key.guardrail_enabled);
    }

    #[test]
    fn derive_key_rag_active_field() {
        let key = VariantRegistry::derive_key(
            "test", false, false, None, true, 32, None, None, None,
        );
        assert!(key.rag_enabled);
    }

    #[test]
    fn derive_key_moe_field() {
        let key = VariantRegistry::derive_key(
            "test", true, false, None, false, 32, None, None, None,
        );
        assert!(key.moe_enabled);
    }

    #[test]
    fn derive_key_spec_phase_verify() {
        let key = VariantRegistry::derive_key(
            "test", false, false, Some(SpecPhase::Verify), false, 32, None, None, None,
        );
        assert_eq!(key.spec_phase, Some(SpecPhase::Verify));
    }

    #[test]
    fn l1i_budget_exceeded_clone_independent() {
        let err = L1iBudgetExceeded {
            footprint: 100,
            budget: 50,
            suggestion: "hint".into(),
        };
        let err2 = err.clone();
        // Both are independent
        assert_eq!(err.footprint, err2.footprint);
        assert_eq!(err.budget, err2.budget);
        assert_eq!(err.suggestion, err2.suggestion);
    }

    #[test]
    fn registry_overwrite_preserves_len() {
        let mut registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);
        registry.register(test_variant(key.clone(), 100)).unwrap();
        assert_eq!(registry.len(), 1);
        registry.register(test_variant(key.clone(), 200)).unwrap();
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn registry_overwrite_updates_variant_footprint() {
        let mut registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);
        registry.register(test_variant(key.clone(), 100)).unwrap();
        registry.register(test_variant(key.clone(), 300)).unwrap();
        let found = registry.get(&key).unwrap();
        assert_eq!(found.instruction_footprint_bytes, 300);
    }

    #[test]
    fn registry_is_empty_after_new() {
        let registry = VariantRegistry::new();
        assert!(registry.is_empty());
    }

    #[test]
    fn registry_not_empty_after_register() {
        let mut registry = VariantRegistry::new();
        registry.register(test_variant(test_key("x", 16), 50)).unwrap();
        assert!(!registry.is_empty());
    }

    #[test]
    fn mechanism_id_all_variants_in_hashset() {
        use std::collections::HashSet;
        let all = [
            MechanismId::Dense, MechanismId::MoeDispatch, MechanismId::SpecDraft,
            MechanismId::SpecVerify, MechanismId::GuardrailProbe, MechanismId::RagInjection,
            MechanismId::TurboQuantFwht, MechanismId::KiviQuant, MechanismId::GateFirstSkip,
            MechanismId::ResidualBypass, MechanismId::Telemetry, MechanismId::EarlyExit,
            MechanismId::RaggedCompaction,
        ];
        let set: HashSet<MechanismId> = all.iter().copied().collect();
        assert_eq!(set.len(), 13);
    }

    #[test]
    fn code_section_copy_preserves_value() {
        let s = CodeSection::Warm;
        let s_copy = s;
        assert_eq!(s, s_copy);
    }

    #[test]
    fn registry_find_closest_moe_enabled_relaxation() {
        // find_closest does not relax moe_enabled — it's always strict
        let mut registry = VariantRegistry::new();
        let mut key = test_key("qwen3", 64);
        key.moe_enabled = false;
        registry.register(test_variant(key, 512)).unwrap();

        let mut query = test_key("qwen3", 64);
        query.moe_enabled = true;
        // moe_enabled mismatch cannot be relaxed by find_closest
        assert!(registry.find_closest(&query).is_none());
    }

    #[test]
    fn compiled_variant_section_hot() {
        let key = test_key("qwen3", 64);
        let v = CompiledVariant {
            code: vec![],
            instruction_footprint_bytes: 0,
            mechanisms: vec![],
            section: CodeSection::Hot,
            key,
        };
        assert_eq!(v.section, CodeSection::Hot);
    }

    #[test]
    fn find_closest_with_kv_tier_exact_match() {
        let mut registry = VariantRegistry::new();
        let mut key = test_key("qwen3", 64);
        key.kv_tier = Some("fp16".into());
        registry.register(test_variant(key, 512)).unwrap();

        let mut query = test_key("qwen3", 64);
        query.kv_tier = Some("fp16".into());
        let found = registry.find_closest(&query).unwrap();
        assert_eq!(found.key.kv_tier.as_deref(), Some("fp16"));
    }

    // ---- Round 4: remaining edge-case coverage ----

    #[test]
    fn l1i_budget_exceeded_error_source_is_none() {
        // Arrange: L1iBudgetExceeded has no underlying cause
        let err = L1iBudgetExceeded {
            footprint: 100,
            budget: 50,
            suggestion: "test".into(),
        };
        // Act
        let source = std::error::Error::source(&err);
        // Assert: no chained source
        assert!(source.is_none());
    }

    #[test]
    fn variant_key_display_empty_quant_type_string_vs_none() {
        // Arrange: quant_type = Some("") should still append the underscore+value
        let key_some_empty = VariantKey {
            arch: "m".into(),
            moe_enabled: false,
            guardrail_enabled: false,
            spec_phase: None,
            rag_enabled: false,
            golden_size: 16,
            quant_type: Some(String::new()),
            kv_tier: None,
            batch_golden_size: None,
        };
        // Act
        let s = format!("{}", key_some_empty);
        // Assert: Some("") produces a trailing "_\n" style segment with empty string
        assert!(s.contains("_") || s.ends_with("_"), "quant_type=Some('') must emit underscore separator");

        // Conversely, quant_type = None must NOT contain an extra quant segment
        let key_none = test_key("m", 16);
        let s_none = format!("{}", key_none);
        assert!(!s_none.contains("_kv"), "quant_type=None must not emit kv segment");
    }

    #[test]
    fn variant_key_display_empty_kv_tier_string() {
        // Arrange: kv_tier = Some("") should still produce the "kv" prefix
        let key = VariantKey {
            arch: "m".into(),
            moe_enabled: false,
            guardrail_enabled: false,
            spec_phase: None,
            rag_enabled: false,
            golden_size: 16,
            quant_type: None,
            kv_tier: Some(String::new()),
            batch_golden_size: None,
        };
        // Act
        let s = format!("{}", key);
        // Assert: "kv" prefix appears even with empty tier name
        assert!(s.contains("kv"), "kv_tier=Some('') must emit 'kv' prefix");
    }

    #[test]
    fn registry_overwrite_larger_variant_increases_max_footprint() {
        // Arrange
        let mut registry = VariantRegistry::new();
        let key = test_key("qwen3", 64);
        registry.register(test_variant(key.clone(), 100)).unwrap();
        assert_eq!(registry.max_footprint(), 100);

        // Act: overwrite with a larger variant
        registry.register(test_variant(key.clone(), 500)).unwrap();

        // Assert: max_footprint updated to the larger value
        assert_eq!(registry.max_footprint(), 500);
        assert_eq!(registry.get(&key).unwrap().instruction_footprint_bytes, 500);
    }

    #[test]
    fn find_closest_multiple_variants_picks_largest_golden_size_leq() {
        // Arrange: register golden_size 8, 16, 32, 64 for same arch
        let mut registry = VariantRegistry::new();
        for gs in [8, 16, 32, 64] {
            registry.register(test_variant(test_key("qwen3", gs), gs * 10)).unwrap();
        }

        // Act: query golden_size=48 → should pick 32 (largest <= 48)
        let query = test_key("qwen3", 48);
        let found = registry.find_closest(&query).unwrap();

        // Assert
        assert_eq!(found.key.golden_size, 32);
        assert_eq!(found.instruction_footprint_bytes, 320);
    }

    #[test]
    fn find_closest_with_batch_golden_size_none_vs_some_no_match() {
        // Arrange: register variant with batch_golden_size = Some(4)
        let mut registry = VariantRegistry::new();
        let mut key = test_key("qwen3", 64);
        key.batch_golden_size = Some(4);
        registry.register(test_variant(key, 512)).unwrap();

        // Act: query with batch_golden_size = None
        let query = test_key("qwen3", 64);
        let result = registry.find_closest(&query);

        // Assert: None != Some(4), cannot relax, returns None
        assert!(result.is_none());
    }

    #[test]
    fn find_closest_all_relaxation_stages_fail_returns_none() {
        // Arrange: register a variant that cannot match through any relaxation
        let mut registry = VariantRegistry::new();
        let mut key = test_key("qwen3", 64);
        key.moe_enabled = true; // find_closest never relaxes moe_enabled
        registry.register(test_variant(key, 512)).unwrap();

        // Act: query non-MoE — moe_enabled mismatch is not relaxable
        let mut query = test_key("qwen3", 64);
        query.moe_enabled = false;
        query.guardrail_enabled = true;
        query.rag_enabled = true;
        query.spec_phase = Some(SpecPhase::Verify);

        // Assert: all relaxation stages fail because moe_enabled is strict
        assert!(registry.find_closest(&query).is_none());
    }

    #[test]
    fn derive_key_golden_size_zero_boundary() {
        // Arrange & Act: golden_size = 0 is a valid boundary value
        let key = VariantRegistry::derive_key(
            "boundary_test", false, false, None, false, 0, None, None, None,
        );
        // Assert
        assert_eq!(key.golden_size, 0);
        assert_eq!(key.arch, "boundary_test");
    }

    #[test]
    fn registry_register_variants_with_same_arch_different_sections() {
        // Arrange: register 3 variants with same arch but different sections
        let mut registry = VariantRegistry::new();
        let k1 = test_key("qwen3", 32);
        let k2 = test_key("qwen3", 64);
        let k3 = test_key("qwen3", 128);

        registry.register(CompiledVariant {
            code: vec![0x90; 64],
            instruction_footprint_bytes: 64,
            mechanisms: vec![MechanismId::Dense],
            section: CodeSection::Hot,
            key: k1,
        }).unwrap();
        registry.register(CompiledVariant {
            code: vec![0x90; 128],
            instruction_footprint_bytes: 128,
            mechanisms: vec![MechanismId::Dense, MechanismId::Telemetry],
            section: CodeSection::Warm,
            key: k2,
        }).unwrap();
        registry.register(CompiledVariant {
            code: vec![],
            instruction_footprint_bytes: 0,
            mechanisms: vec![MechanismId::EarlyExit],
            section: CodeSection::Cold,
            key: k3,
        }).unwrap();

        // Act
        let variants = registry.variants();

        // Assert: all 3 registered with distinct sections
        assert_eq!(variants.len(), 3);
        let sections: Vec<CodeSection> = variants.iter().map(|v| v.section).collect();
        assert!(sections.contains(&CodeSection::Hot));
        assert!(sections.contains(&CodeSection::Warm));
        assert!(sections.contains(&CodeSection::Cold));
    }

    #[test]
    fn majority_kv_tier_large_input_with_clear_majority() {
        // Arrange: 100 entries, 60 "fp8", 30 "fp16", 10 "int4"
        let mut tiers: Vec<String> = (0..60).map(|_| "fp8".to_string()).collect();
        tiers.extend((0..30).map(|_| "fp16".to_string()));
        tiers.extend((0..10).map(|_| "int4".to_string()));

        // Act
        let result = VariantRegistry::majority_kv_tier(&tiers);

        // Assert
        assert_eq!(result, Some("fp8".to_string()));
    }
}
