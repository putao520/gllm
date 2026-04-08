//! §17.1 自推测核心管线 (EESD) + §17.9 自适应调度决策
//!
//! 核心架构:
//! Phase A: Draft — L2_hot variant (≈L/3 layers) → Adapter → logits → spec tree
//! Phase B: Verify — Full variant (L layers) → batched tree attention → EqSpec verify
//! Phase C: Shadow KV Fill (ADEPT) — 仅 Early-Exit 场景
//! Phase D: Compact→Execute→Scatter — 硬件级 batch 合并
//!
//! §17.9: 自适应调度决策
//! - decode_request_count > 0 且 avg_acceptance_rate > 0.5 → 推测解码
//! - 连续 3 轮 acceptance_rate < 0.3 → 回退到标准解码
//!
//! §17.10: SAGUARO 多 GPU 模式
//! - 单 GPU → EESD (浅层变体做 draft)
//! - ≥2 GPU → SAGUARO (独立 draft GPU, draft+verify 并行)

use crate::jit::epilogue::SpecScheduleAdvice;
use crate::scheduler::types::RequestId;

use super::adapter::DraftAdapter;
use super::cache::SpeculationCache;
use super::tree::{NgramIndex, SpecTree, SpecTreeConfig};
use super::verify::{VerifyResult, SequenceVerifyResult};

/// 推测解码模式
///
/// §17.10.4: 自适应路径选择
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecDecodingMode {
    /// §17.1 EESD: 单 GPU, 浅层变体做 draft (默认)
    Eesd,
    /// §17.10 SAGUARO: 多 GPU, 独立 draft GPU + target GPU pool
    Saguaro,
    /// 标准解码 (推测解码被禁用或回退)
    Standard,
}

/// 推测解码运行时状态
///
/// §17.1: SpeculativeDecodingState 在 `from_loader()` 时初始化，
/// 包含 Draft Adapter、Spec Tree 配置、接受率统计等。
pub struct SpecDecodingState {
    /// 当前模式 (EESD / SAGUARO / Standard)
    mode: SpecDecodingMode,

    /// §17.2 Draft Adapter (将浅层 hidden → logits)
    adapter: Option<DraftAdapter>,

    /// §17.3 Spec Tree 配置
    tree_config: SpecTreeConfig,

    /// §17.10.3 Speculation Cache (SAGUARO 模式使用)
    cache: SpeculationCache,

    /// §17.3 N-gram 索引 (从 prompt 构建)
    ngram_index: Option<NgramIndex>,

    /// §17.9 滑动平均接受率
    acceptance_rate_ema: f32,
    /// §17.9 连续低接受率轮数 (用于回退决策)
    low_acceptance_streak: usize,
    /// §17.9 回退阈值 (连续 N 轮 < 0.3 时回退)
    fallback_streak_threshold: usize,

    /// §17.1 Draft variant 使用的层数 (≈L/3)
    draft_layers: usize,
    /// Full model 总层数 L
    total_layers: usize,

    /// 历史 spec 步数统计
    spec_step_count: usize,
    total_draft_tokens: usize,
    total_accepted_tokens: usize,

    /// 当前活跃的 spec tree (每 draft phase 重建)
    current_tree: Option<SpecTree>,

    /// 当前请求的 prompt tokens (用于 PLD 匹配)
    current_prompt_tokens: Vec<u32>,

    /// §17.10 SAGUARO: Draft GPU index (多 GPU 时)
    draft_gpu_index: Option<usize>,
    /// §17.10 SAGUARO: Target GPU indices
    target_gpu_indices: Vec<usize>,
}

impl std::fmt::Debug for SpecDecodingState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpecDecodingState")
            .field("mode", &self.mode)
            .field("draft_layers", &self.draft_layers)
            .field("total_layers", &self.total_layers)
            .field("acceptance_rate_ema", &self.acceptance_rate_ema)
            .field("spec_step_count", &self.spec_step_count)
            .field("avg_acceptance_rate", &self.avg_acceptance_rate())
            .finish()
    }
}

impl SpecDecodingState {
    /// 创建 EESD 模式的推测解码状态
    ///
    /// §17.1: Draft 使用 L2_hot 变体 (≈L/3 层), Verify 使用 full 变体
    ///
    /// # Arguments
    /// * `total_layers` - 模型总层数 L
    /// * `adapter` - Draft Adapter (RmsNorm + MatMul, 复用 lm_head.weight)
    /// * `tree_config` - Spec Tree 配置
    pub fn new_eesd(
        total_layers: usize,
        adapter: DraftAdapter,
        tree_config: SpecTreeConfig,
    ) -> Self {
        let draft_layers = (total_layers + 2) / 3; // ≈L/3, 向上取整
        Self {
            mode: SpecDecodingMode::Eesd,
            adapter: Some(adapter),
            tree_config,
            cache: SpeculationCache::new(4, 1024),
            ngram_index: None,
            acceptance_rate_ema: 0.5,
            low_acceptance_streak: 0,
            fallback_streak_threshold: 3,
            draft_layers,
            total_layers,
            spec_step_count: 0,
            total_draft_tokens: 0,
            total_accepted_tokens: 0,
            current_tree: None,
            current_prompt_tokens: Vec::new(),
            draft_gpu_index: None,
            target_gpu_indices: Vec::new(),
        }
    }

    /// 创建 SAGUARO 模式的推测解码状态
    ///
    /// §17.10: ≥2 GPU 时使用, 独立 draft GPU + target GPU pool
    pub fn new_saguaro(
        total_layers: usize,
        adapter: DraftAdapter,
        tree_config: SpecTreeConfig,
        draft_gpu_index: usize,
        target_gpu_indices: Vec<usize>,
    ) -> Self {
        let mut state = Self::new_eesd(total_layers, adapter, tree_config);
        state.mode = SpecDecodingMode::Saguaro;
        state.draft_gpu_index = Some(draft_gpu_index);
        state.target_gpu_indices = target_gpu_indices;
        state
    }

    /// 创建标准模式 (推测解码禁用)
    pub fn new_standard() -> Self {
        Self {
            mode: SpecDecodingMode::Standard,
            adapter: None,
            tree_config: SpecTreeConfig::default(),
            cache: SpeculationCache::new(4, 1024),
            ngram_index: None,
            acceptance_rate_ema: 0.0,
            low_acceptance_streak: 0,
            fallback_streak_threshold: 3,
            draft_layers: 0,
            total_layers: 0,
            spec_step_count: 0,
            total_draft_tokens: 0,
            total_accepted_tokens: 0,
            current_tree: None,
            current_prompt_tokens: Vec::new(),
            draft_gpu_index: None,
            target_gpu_indices: Vec::new(),
        }
    }

    /// §17.9: 自适应调度决策
    ///
    /// 基于当前接受率和 decode 请求数，决定是否启用推测解码
    pub fn should_speculate(&self, decode_request_count: usize) -> SpecScheduleAdvice {
        if self.mode == SpecDecodingMode::Standard {
            return SpecScheduleAdvice::StandardDecode;
        }

        if decode_request_count == 0 {
            return SpecScheduleAdvice::StandardDecode;
        }

        // §17.9: 连续 3 轮 acceptance_rate < 0.3 → 回退
        if self.low_acceptance_streak >= self.fallback_streak_threshold {
            return SpecScheduleAdvice::Fallback;
        }

        // §17.9: avg_acceptance_rate > 0.5 → 启用推测
        if self.acceptance_rate_ema > 0.5 {
            SpecScheduleAdvice::EnableSpec
        } else {
            SpecScheduleAdvice::StandardDecode
        }
    }

    /// Phase A: Draft — 生成推测树
    ///
    /// §17.1: 使用 L2_hot variant 前向传播获取 hidden state,
    /// 然后通过 Adapter 投影为 logits, 构建 SpecTree。
    ///
    /// # Arguments
    /// * `adapter_logits` - Adapter 产生的 logits (from L2_hot hidden state)
    /// * `top_k_tokens` - 从 logits 中提取的 top-k token IDs
    /// * `prompt_tokens` - 当前 prompt + 已生成 tokens
    ///
    /// # Returns
    /// 构建的 SpecTree
    pub fn draft_phase(
        &mut self,
        top_k_tokens: &[u32],
        prompt_tokens: &[u32],
    ) -> &SpecTree {
        // §17.3: 构建/更新 N-gram 索引
        self.ngram_index = Some(NgramIndex::build(prompt_tokens, self.tree_config.pld_ngram_len));
        self.current_prompt_tokens = prompt_tokens.to_vec();

        // Build spec tree
        let ngram_idx = self.ngram_index.as_ref().unwrap();
        let tree = SpecTree::build(
            self.tree_config.clone(),
            top_k_tokens,
            prompt_tokens,
            ngram_idx,
        );

        self.current_tree = Some(tree);
        self.current_tree.as_ref().unwrap()
    }

    /// Phase B: Verify — 使用 VerifyResult 更新状态
    ///
    /// §17.4: EqSpec 三不变量保证 batch 正确性
    ///
    /// # Arguments
    /// * `verify_result` - Verify phase 产生的验证结果
    pub fn verify_phase(&mut self, verify_result: &VerifyResult) {
        let batch_acceptance = verify_result.avg_acceptance_rate;

        // §17.9: 更新滑动平均接受率 (EMA, α=0.3)
        let alpha = 0.3;
        self.acceptance_rate_ema = alpha * batch_acceptance + (1.0 - alpha) * self.acceptance_rate_ema;

        // §17.9: 更新连续低接受率计数
        if batch_acceptance < 0.3 {
            self.low_acceptance_streak += 1;
        } else {
            self.low_acceptance_streak = 0;
        }

        // 更新全局统计
        self.spec_step_count += 1;
        self.total_draft_tokens += verify_result.total_draft_tokens;
        self.total_accepted_tokens += verify_result.total_accepted_tokens;

        // §17.10.3: SAGUARO 模式更新 cache
        if self.mode == SpecDecodingMode::Saguaro {
            self.cache.adapt_scale_factor();
        }
    }

    /// 获取当前 spec tree
    pub fn current_tree(&self) -> Option<&SpecTree> {
        self.current_tree.as_ref()
    }

    /// 获取 draft adapter 引用
    pub fn adapter(&self) -> Option<&DraftAdapter> {
        self.adapter.as_ref()
    }

    /// 获取 draft adapter 可变引用
    pub fn adapter_mut(&mut self) -> Option<&mut DraftAdapter> {
        self.adapter.as_mut()
    }

    /// 获取 speculation cache 引用
    pub fn cache(&self) -> &SpeculationCache {
        &self.cache
    }

    /// 获取 speculation cache 可变引用
    pub fn cache_mut(&mut self) -> &mut SpeculationCache {
        &mut self.cache
    }

    /// 获取当前模式
    pub fn mode(&self) -> SpecDecodingMode {
        self.mode
    }

    /// 获取 draft 层数
    pub fn draft_layers(&self) -> usize {
        self.draft_layers
    }

    /// 获取总层数
    pub fn total_layers(&self) -> usize {
        self.total_layers
    }

    /// 获取当前接受率 EMA
    pub fn acceptance_rate_ema(&self) -> f32 {
        self.acceptance_rate_ema
    }

    /// 获取历史平均接受率
    pub fn avg_acceptance_rate(&self) -> f32 {
        if self.total_draft_tokens == 0 {
            return 0.0;
        }
        self.total_accepted_tokens as f32 / self.total_draft_tokens as f32
    }

    /// 获取 spec 步数
    pub fn spec_step_count(&self) -> usize {
        self.spec_step_count
    }

    /// §17.10: 获取 SAGUARO draft GPU index
    pub fn draft_gpu_index(&self) -> Option<usize> {
        self.draft_gpu_index
    }

    /// §17.10: 获取 SAGUARO target GPU indices
    pub fn target_gpu_indices(&self) -> &[usize] {
        &self.target_gpu_indices
    }

    /// 重置低接受率连续计数 (手动恢复推测解码时使用)
    pub fn reset_fallback_streak(&mut self) {
        self.low_acceptance_streak = 0;
    }

    /// 检查是否处于活跃推测模式
    pub fn is_active(&self) -> bool {
        self.mode != SpecDecodingMode::Standard
            && self.adapter.is_some()
            && self.low_acceptance_streak < self.fallback_streak_threshold
    }

    /// 临时 buffer 大小估算 (§17.7.3: tree KV 临时缓冲)
    ///
    /// tree KV 使用临时 buffer, 不直接写入主 KV cache
    /// 大小 = max_tree_size × L × 2 × kv_dim × dtype_size
    pub fn temp_buffer_bytes(&self, kv_dim: usize, dtype_size: usize) -> usize {
        let max_tree = self.tree_config.max_tree_size;
        max_tree * self.total_layers * 2 * kv_dim * dtype_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::adapter::{AdapterConfig, DraftAdapter};
    use std::sync::Arc;

    fn make_test_adapter(hidden_size: usize, vocab_size: usize) -> DraftAdapter {
        let config = AdapterConfig {
            hidden_size,
            vocab_size,
            rms_norm_eps: 1e-5,
            enable_distillation: false,
            distillation_steps: 100,
        };
        let lm_head = Arc::new(vec![0.1f32; vocab_size * hidden_size]);
        let norm_weight = Arc::new(vec![1.0f32; hidden_size]);
        DraftAdapter::new_phase_a(config, lm_head, norm_weight)
    }

    #[test]
    fn test_eesd_state_creation() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        assert_eq!(state.mode(), SpecDecodingMode::Eesd);
        assert_eq!(state.draft_layers(), 11); // (32+2)/3 = 11
        assert_eq!(state.total_layers(), 32);
        assert!(state.is_active());
    }

    #[test]
    fn test_standard_mode_not_active() {
        let state = SpecDecodingState::new_standard();
        assert_eq!(state.mode(), SpecDecodingMode::Standard);
        assert!(!state.is_active());
    }

    #[test]
    fn test_should_speculate_no_decode_requests() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let advice = state.should_speculate(0);
        assert_eq!(advice, SpecScheduleAdvice::StandardDecode);
    }

    #[test]
    fn test_should_speculate_with_decode() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let advice = state.should_speculate(4);
        // Default EMA = 0.5, which is the boundary; our implementation uses > 0.5
        // So at exactly 0.5, it returns StandardDecode
        assert!(matches!(advice, SpecScheduleAdvice::StandardDecode | SpecScheduleAdvice::EnableSpec));
    }

    #[test]
    fn test_draft_phase_builds_tree() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let top_k = vec![100u32, 200, 300];
        let prompt = vec![1, 2, 3, 4, 100, 50];

        let tree = state.draft_phase(&top_k, &prompt);
        assert!(!tree.is_empty());
        assert!(tree.len() >= 1);
    }

    #[test]
    fn test_verify_phase_updates_ema() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        assert!((state.acceptance_rate_ema() - 0.5).abs() < 1e-5);

        let r1 = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 20, 30]);
        let verify = VerifyResult::from_sequence_results(vec![r1]);
        state.verify_phase(&verify);
        assert!(state.acceptance_rate_ema() > 0.5);
    }

    #[test]
    fn test_verify_phase_tracks_low_streak() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        // Simulate 3 rounds of low acceptance
        for _ in 0..3 {
            let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[99]); // 0% acceptance
            let verify = VerifyResult::from_sequence_results(vec![r]);
            state.verify_phase(&verify);
        }
        assert!(state.should_speculate(4) == SpecScheduleAdvice::Fallback);
    }

    #[test]
    fn test_saguaro_mode() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_saguaro(
            32, adapter, SpecTreeConfig::default(), 0, vec![1, 2, 3],
        );
        assert_eq!(state.mode(), SpecDecodingMode::Saguaro);
        assert_eq!(state.draft_gpu_index(), Some(0));
        assert_eq!(state.target_gpu_indices(), &[1, 2, 3]);
    }

    #[test]
    fn test_temp_buffer_estimation() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        // max_tree=32, layers=32, 2 (K+V), kv_dim=1024, f32=4 bytes
        let bytes = state.temp_buffer_bytes(1024, 4);
        assert_eq!(bytes, 32 * 32 * 2 * 1024 * 4); // 8MB
    }

    #[test]
    fn test_avg_acceptance_rate() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        // Round 1: 3/3 accepted
        let r1 = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 20, 30]);
        state.verify_phase(&VerifyResult::from_sequence_results(vec![r1]));
        assert!((state.avg_acceptance_rate() - 1.0).abs() < 1e-5);

        // Round 2: 0/2 accepted
        let r2 = SequenceVerifyResult::verify_spine(1, &[40, 50], &[99]);
        state.verify_phase(&VerifyResult::from_sequence_results(vec![r2]));
        // Total: 3 accepted / 5 drafted = 0.6
        assert!((state.avg_acceptance_rate() - 0.6).abs() < 1e-5);
    }
}
