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
use super::adapter::{DraftAdapter, MtpEmaTracker};
use super::cache::SpeculationCache;
use super::eagle::{self, EagleConfig, EagleHead};
use super::mtp::{self, MtpConfig, MtpHead};
use super::tree::{NgramIndex, SpecTree, SpecTreeConfig};
use super::verify::VerifyResult;

/// 推测解码模式
///
/// §17.10.4: 自适应路径选择
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecDecodingMode {
    /// §17.1 EESD: 单 GPU, 浅层变体做 draft (默认)
    Eesd,
    /// §17.10 SAGUARO: 多 GPU, 独立 draft GPU + target GPU pool
    Saguaro,
    /// EAGLE: 训练好的轻量 draft head (1-2 层 transformer)
    Eagle,
    /// MTP: 模型内置多 token 预测 (DeepSeek V3, Qwen3)
    Mtp,
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
    /// §17.10 SAGUARO: 分布式配置 (REQ-DIST-017, ENT-DIST-SAGUARO)
    #[cfg(feature = "nccl")]
    saguaro_config: Option<super::saguaro::saguaro::SaguaroConfig>,
    /// §17.10 SAGUARO: 接受率累积跟踪器 (REQ-DIST-017)
    #[cfg(feature = "nccl")]
    saguaro_acceptance_tracker: super::saguaro::saguaro::SaguaroAcceptanceTracker,
    /// EAGLE draft head (§17, EAGLE 模式)
    eagle_head: Option<EagleHead>,
    /// EAGLE config
    eagle_config: Option<EagleConfig>,
    /// MTP head (§17, MTP 模式)
    mtp_head: Option<MtpHead>,
    /// MTP config
    mtp_config: Option<MtpConfig>,
    /// MTP draft logits from last forward (used by draft_phase in MTP mode)
    mtp_draft_logits: Vec<Vec<f32>>,

    /// §BCI9: EMA multi-token acceptance tracker
    /// Tracks per-position acceptance rates and dynamically adjusts
    /// the acceptance threshold based on draft model's historical accuracy.
    mtp_ema_tracker: MtpEmaTracker,
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
            .field("ema_threshold", &self.mtp_ema_tracker.threshold())
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
        let draft_layers = total_layers.div_ceil(3); // ≈L/3, 向上取整
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
            #[cfg(feature = "nccl")]
            saguaro_config: None,
            #[cfg(feature = "nccl")]
            saguaro_acceptance_tracker: super::saguaro::saguaro::SaguaroAcceptanceTracker::new(),
            eagle_head: None,
            eagle_config: None,
            mtp_head: None,
            mtp_config: None,
            mtp_draft_logits: Vec::new(),
            mtp_ema_tracker: MtpEmaTracker::new(4, 0.3),
        }
    }

    /// 创建 SAGUARO 模式的推测解码状态
    ///
    /// §17.10: ≥2 GPU 时使用, 独立 draft GPU + target GPU pool
    // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [lifecycle:init]
    pub fn new_saguaro(
        total_layers: usize,
        adapter: DraftAdapter,
        tree_config: SpecTreeConfig,
        draft_gpu_index: usize,
        target_gpu_indices: Vec<usize>,
    ) -> Self {
        let _verify_rank = target_gpu_indices.first().copied().unwrap_or(1) as u32;
        let mut state = Self::new_eesd(total_layers, adapter, tree_config);
        state.mode = SpecDecodingMode::Saguaro;
        state.draft_gpu_index = Some(draft_gpu_index);
        state.target_gpu_indices = target_gpu_indices;
        #[cfg(feature = "nccl")]
        {
            state.saguaro_config = Some(super::saguaro::saguaro::SaguaroConfig {
                draft_rank: draft_gpu_index as u32,
                verify_rank: _verify_rank,
                draft_length: 5,
            });
        }
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
            #[cfg(feature = "nccl")]
            saguaro_config: None,
            #[cfg(feature = "nccl")]
            saguaro_acceptance_tracker: super::saguaro::saguaro::SaguaroAcceptanceTracker::new(),
            eagle_head: None,
            eagle_config: None,
            mtp_head: None,
            mtp_config: None,
            mtp_draft_logits: Vec::new(),
            mtp_ema_tracker: MtpEmaTracker::new(1, 0.3),
        }
    }

    /// 创建 EAGLE 模式 (训练好的轻量 draft head)
    ///
    /// EAGLE 在主模型 hidden state 上添加 1-2 层轻量 transformer，
    /// 根据 draft logits 置信度动态构建推测树 (EAGLE-2)。
    pub fn new_eagle(
        total_layers: usize,
        adapter: DraftAdapter,
        tree_config: SpecTreeConfig,
        eagle_head: EagleHead,
        eagle_config: EagleConfig,
    ) -> Self {
        let mut state = Self::new_eesd(total_layers, adapter, tree_config);
        state.mode = SpecDecodingMode::Eagle;
        state.eagle_head = Some(eagle_head);
        state.eagle_config = Some(eagle_config);
        state
    }

    /// 创建 MTP 模式 (模型内置多 token 预测)
    ///
    /// MTP 模型 (DeepSeek V3, Qwen3) 一次前向输出 K 个 token logits，
    /// 相当于内置 draft model，无需额外权重。
    pub fn new_mtp(
        total_layers: usize,
        adapter: DraftAdapter,
        tree_config: SpecTreeConfig,
        mtp_head: MtpHead,
        mtp_config: MtpConfig,
    ) -> Self {
        let mut state = Self::new_eesd(total_layers, adapter, tree_config);
        state.mode = SpecDecodingMode::Mtp;
        state.mtp_head = Some(mtp_head);
        state.mtp_config = Some(mtp_config);
        state
    }

    /// §17.9: 自适应调度决策
    ///
    /// 基于当前接受率和 decode 请求数，决定是否启用推测解码
    // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [controlflow:CF-DIST-005]
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

        // §BCI9: Use EMA tracker's dynamic threshold instead of fixed 0.5.
        // The threshold adapts based on trend and variance of historical
        // acceptance rates. Positive trend → lower threshold (more aggressive),
        // high variance → higher threshold (more conservative).
        let threshold = self.mtp_ema_tracker.threshold();
        if self.acceptance_rate_ema > threshold {
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
        self.current_prompt_tokens = prompt_tokens.to_vec();

        match self.mode {
            SpecDecodingMode::Eagle => {
                // EAGLE: 使用 draft logits 置信度动态构建推测树
                // eagle_head 需要上一轮的 hidden_state 来产生 draft logits,
                // 这里先用 top_k_tokens 作为 draft candidates.
                // 真正的 forward 由 executor 调用 eagle_draft() 后
                // 通过 set_mtp_draft_logits() 传入 logits.
                let draft_tokens = if !self.mtp_draft_logits.is_empty() {
                    // 有上一轮 EAGLE draft logits → build_eagle_tree
                    let config = self.eagle_config.as_ref()
                        .expect("SpecDecodingMode::Eagle requires eagle_config — call set_eagle_config() first");
                    let flat_logits: Vec<f32> = self.mtp_draft_logits.iter().flat_map(|v| v.iter().copied()).collect();
                    let vocab_size = flat_logits.len().max(1);
                    eagle::build_eagle_tree(&flat_logits, vocab_size, config)
                } else {
                    top_k_tokens.to_vec()
                };
                let tree = SpecTree::build(
                    self.tree_config.clone(),
                    &draft_tokens,
                    prompt_tokens,
                    &NgramIndex::build(prompt_tokens, self.tree_config.pld_ngram_len),
                );
                self.current_tree = Some(tree);
            }
            SpecDecodingMode::Mtp => {
                // MTP: 从多 token 预测 logits 中提取 top-1 候选序列
                let draft_tokens = if !self.mtp_draft_logits.is_empty() {
                    mtp::mtp_candidates(&self.mtp_draft_logits)
                } else {
                    top_k_tokens.to_vec()
                };
                let tree = SpecTree::build(
                    self.tree_config.clone(),
                    &draft_tokens,
                    prompt_tokens,
                    &NgramIndex::build(prompt_tokens, self.tree_config.pld_ngram_len),
                );
                self.current_tree = Some(tree);
            }
            _ => {
                // EESD / SAGUARO: PLD n-gram 树
                self.ngram_index = Some(NgramIndex::build(prompt_tokens, self.tree_config.pld_ngram_len));
                // SAFETY: ngram_index set 2 lines above — mode invariants guarantee it exists
                let ngram_idx = self.ngram_index.as_ref()
                    .expect("ngram_index must exist after build");
                let tree = SpecTree::build(
                    self.tree_config.clone(),
                    top_k_tokens,
                    prompt_tokens,
                    ngram_idx,
                );
                self.current_tree = Some(tree);
            }
        }

        // SAFETY: current_tree set in all match arms above
        self.current_tree.as_ref()
            .expect("current_tree must exist after draft_phase")
    }

    /// Phase B: Verify — 使用 VerifyResult 更新状态
    ///
    /// §17.4: EqSpec 三不变量保证 batch 正确性
    ///
    /// # Arguments
    /// * `verify_result` - Verify phase 产生的验证结果
    // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO] [controlflow:CF-DIST-005]
    pub fn verify_phase(&mut self, verify_result: &VerifyResult) {
        let batch_acceptance = verify_result.avg_acceptance_rate;

        // §17.9: 更新滑动平均接受率 (EMA, α=0.3)
        let alpha = 0.3;
        self.acceptance_rate_ema = alpha * batch_acceptance + (1.0 - alpha) * self.acceptance_rate_ema;

        // §BCI9: Update multi-token EMA tracker with batch acceptance rate.
        // This dynamically adjusts the acceptance threshold based on
        // draft model's historical accuracy.
        self.mtp_ema_tracker.record_batch(batch_acceptance);

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

        // §17.10.3: SAGUARO 模式更新 cache + acceptance tracker (REQ-DIST-017)
        if self.mode == SpecDecodingMode::Saguaro {
            self.cache.adapt_scale_factor();
            #[cfg(feature = "nccl")]
            {
                self.saguaro_acceptance_tracker.record(
                    verify_result.total_draft_tokens as u32,
                    verify_result.total_accepted_tokens as u32,
                );
            }
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

    /// §17.10: 获取 SAGUARO draft GPU index (REQ-DIST-017)
    // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO]
    pub fn draft_gpu_index(&self) -> Option<usize> {
        self.draft_gpu_index
    }

    /// §17.10: 获取 SAGUARO target GPU indices (REQ-DIST-017)
    // @trace REQ-DIST-017 [entity:ENT-DIST-SAGUARO]
    pub fn target_gpu_indices(&self) -> &[usize] {
        &self.target_gpu_indices
    }

    /// §17.10: 获取 SAGUARO 配置 (REQ-DIST-017, ENT-DIST-SAGUARO)
    #[cfg(feature = "nccl")]
    pub fn saguaro_config(&self) -> Option<&super::saguaro::saguaro::SaguaroConfig> {
        self.saguaro_config.as_ref()
    }

    /// §17.10: 获取 SAGUARO 配置可变引用 (REQ-DIST-017)
    #[cfg(feature = "nccl")]
    pub fn saguaro_config_mut(&mut self) -> Option<&mut super::saguaro::saguaro::SaguaroConfig> {
        self.saguaro_config.as_mut()
    }

    /// §17.10: 获取 SAGUARO 接受率跟踪器 (REQ-DIST-017)
    #[cfg(feature = "nccl")]
    pub fn saguaro_acceptance_tracker(&self) -> &super::saguaro::saguaro::SaguaroAcceptanceTracker {
        &self.saguaro_acceptance_tracker
    }

    /// §17.10: 获取 SAGUARO 接受率跟踪器可变引用 (REQ-DIST-017)
    #[cfg(feature = "nccl")]
    pub fn saguaro_acceptance_tracker_mut(&mut self) -> &mut super::saguaro::saguaro::SaguaroAcceptanceTracker {
        &mut self.saguaro_acceptance_tracker
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

    /// 设置 EAGLE/MTP draft logits (由 executor forward 产生)
    ///
    /// executor 调用 eagle_draft() 或 mtp_draft() 后，将 logits 传入，
    /// draft_phase() 会从中提取 draft candidates。
    pub fn set_draft_logits(&mut self, logits: Vec<Vec<f32>>) {
        self.mtp_draft_logits = logits;
    }

    /// 获取 EAGLE head 引用
    pub fn eagle_head(&self) -> Option<&EagleHead> {
        self.eagle_head.as_ref()
    }

    /// 获取 EAGLE config 引用
    pub fn eagle_config(&self) -> Option<&EagleConfig> {
        self.eagle_config.as_ref()
    }

    /// 获取 MTP head 引用
    pub fn mtp_head(&self) -> Option<&MtpHead> {
        self.mtp_head.as_ref()
    }

    /// 获取 MTP config 引用
    pub fn mtp_config(&self) -> Option<&MtpConfig> {
        self.mtp_config.as_ref()
    }

    /// §BCI9: 获取 EMA multi-token acceptance tracker 引用
    ///
    /// Tracks per-position acceptance rates and provides a dynamic
    /// acceptance threshold based on historical draft model accuracy.
    pub fn mtp_ema_tracker(&self) -> &MtpEmaTracker {
        &self.mtp_ema_tracker
    }

    /// §BCI9: Multi-token acceptance decision.
    ///
    /// Returns the number of consecutive draft positions to accept,
    /// starting from position 0. Stops at the first position where
    /// `ema_accept(position)` returns false.
    pub fn multi_token_accept(&self, max_positions: usize) -> usize {
        self.mtp_ema_tracker.multi_token_accept(max_positions)
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
    use crate::speculative::verify::SequenceVerifyResult;
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

    // =========================================================================
    // SpecDecodingMode tests
    // =========================================================================

    #[test]
    fn test_spec_decoding_mode_equality() {
        assert_eq!(SpecDecodingMode::Eesd, SpecDecodingMode::Eesd);
        assert_ne!(SpecDecodingMode::Eesd, SpecDecodingMode::Saguaro);
        assert_ne!(SpecDecodingMode::Standard, SpecDecodingMode::Mtp);
        assert_ne!(SpecDecodingMode::Eagle, SpecDecodingMode::Mtp);
    }

    #[test]
    fn test_spec_decoding_mode_copy_bound() {
        let mode = SpecDecodingMode::Eesd;
        let copied = mode;
        assert_eq!(mode, copied);
    }

    // =========================================================================
    // SpecDecodingState constructors
    // =========================================================================

    #[test]
    fn test_new_eesd_draft_layers_div_ceil() {
        // total_layers=30 → (30+2)/3 = 10
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(30, adapter, SpecTreeConfig::default());
        assert_eq!(state.draft_layers(), 10);
        assert_eq!(state.total_layers(), 30);
    }

    #[test]
    fn test_new_eesd_draft_layers_div_ceil_rounds_up() {
        // total_layers=10 → (10+2)/3 = 4 (ceil)
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(10, adapter, SpecTreeConfig::default());
        assert_eq!(state.draft_layers(), 4);
    }

    #[test]
    fn test_new_eesd_draft_layers_single_layer() {
        // total_layers=1 → (1+2)/3 = 1
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(1, adapter, SpecTreeConfig::default());
        assert_eq!(state.draft_layers(), 1);
    }

    #[test]
    fn test_new_eesd_initial_state() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        assert_eq!(state.mode(), SpecDecodingMode::Eesd);
        assert_eq!(state.spec_step_count(), 0);
        assert!((state.acceptance_rate_ema() - 0.5).abs() < 1e-5);
        assert!(state.adapter().is_some());
        assert!(state.current_tree().is_none());
        assert!(state.draft_gpu_index().is_none());
        assert!(state.target_gpu_indices().is_empty());
        assert!(state.eagle_head().is_none());
        assert!(state.eagle_config().is_none());
        assert!(state.mtp_head().is_none());
        assert!(state.mtp_config().is_none());
    }

    #[test]
    fn test_new_standard_initial_state() {
        let state = SpecDecodingState::new_standard();
        assert_eq!(state.mode(), SpecDecodingMode::Standard);
        assert_eq!(state.draft_layers(), 0);
        assert_eq!(state.total_layers(), 0);
        assert_eq!(state.spec_step_count(), 0);
        assert!((state.acceptance_rate_ema() - 0.0).abs() < 1e-5);
        assert!(state.adapter().is_none());
        assert!(state.current_tree().is_none());
        assert!(state.draft_gpu_index().is_none());
        assert!(state.target_gpu_indices().is_empty());
        assert!(state.eagle_head().is_none());
        assert!(state.eagle_config().is_none());
        assert!(state.mtp_head().is_none());
        assert!(state.mtp_config().is_none());
    }

    #[test]
    fn test_new_saguaro_sets_gpu_indices() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_saguaro(
            32, adapter, SpecTreeConfig::default(), 2, vec![0, 1, 3],
        );
        assert_eq!(state.mode(), SpecDecodingMode::Saguaro);
        assert_eq!(state.draft_gpu_index(), Some(2));
        assert_eq!(state.target_gpu_indices(), &[0, 1, 3]);
        assert_eq!(state.draft_layers(), 11);
        assert!(state.adapter().is_some());
    }

    #[test]
    fn test_new_eagle_mode() {
        let adapter = make_test_adapter(64, 1000);
        let eagle_head = EagleHead {
            fc_weight: vec![0.1; 64 * 2],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let eagle_config = EagleConfig::default();
        let state = SpecDecodingState::new_eagle(
            32, adapter, SpecTreeConfig::default(), eagle_head, eagle_config,
        );
        assert_eq!(state.mode(), SpecDecodingMode::Eagle);
        assert!(state.eagle_head().is_some());
        assert!(state.eagle_config().is_some());
        assert!(state.mtp_head().is_none());
        assert!(state.mtp_config().is_none());
        assert!(state.adapter().is_some());
    }

    #[test]
    fn test_new_mtp_mode() {
        let adapter = make_test_adapter(64, 1000);
        let mtp_head = MtpHead {
            projections: vec![vec![0.1; 64 * 100]],
            config: MtpConfig::default(),
        };
        let mtp_config = MtpConfig::default();
        let state = SpecDecodingState::new_mtp(
            32, adapter, SpecTreeConfig::default(), mtp_head, mtp_config,
        );
        assert_eq!(state.mode(), SpecDecodingMode::Mtp);
        assert!(state.mtp_head().is_some());
        assert!(state.mtp_config().is_some());
        assert!(state.eagle_head().is_none());
        assert!(state.eagle_config().is_none());
        assert!(state.adapter().is_some());
    }

    // =========================================================================
    // is_active
    // =========================================================================

    #[test]
    fn test_is_active_eesd_with_adapter() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        assert!(state.is_active());
    }

    #[test]
    fn test_is_active_standard_mode() {
        let state = SpecDecodingState::new_standard();
        assert!(!state.is_active());
    }

    #[test]
    fn test_is_active_after_fallback_streak() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        assert!(state.is_active());

        // Simulate 3 rounds of low acceptance to trigger fallback
        for _ in 0..3 {
            let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[99]);
            let verify = VerifyResult::from_sequence_results(vec![r]);
            state.verify_phase(&verify);
        }
        assert!(!state.is_active());
    }

    // =========================================================================
    // reset_fallback_streak
    // =========================================================================

    #[test]
    fn test_reset_fallback_streak_restores_active() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        // Trigger fallback streak
        for _ in 0..3 {
            let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[99]);
            let verify = VerifyResult::from_sequence_results(vec![r]);
            state.verify_phase(&verify);
        }
        assert!(!state.is_active());

        // Reset should restore active state
        state.reset_fallback_streak();
        assert!(state.is_active());
    }

    // =========================================================================
    // should_speculate — scheduling decisions
    // =========================================================================

    #[test]
    fn test_should_speculate_standard_mode_always_standard() {
        let state = SpecDecodingState::new_standard();
        assert_eq!(state.should_speculate(10), SpecScheduleAdvice::StandardDecode);
        assert_eq!(state.should_speculate(0), SpecScheduleAdvice::StandardDecode);
    }

    #[test]
    fn test_should_speculate_zero_requests_returns_standard() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        assert_eq!(state.should_speculate(0), SpecScheduleAdvice::StandardDecode);
    }

    #[test]
    fn test_should_speculate_fallback_after_streak() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        // Drive acceptance EMA down to trigger threshold
        for _ in 0..3 {
            let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[99]);
            let verify = VerifyResult::from_sequence_results(vec![r]);
            state.verify_phase(&verify);
        }
        assert_eq!(state.should_speculate(4), SpecScheduleAdvice::Fallback);
    }

    // =========================================================================
    // verify_phase — EMA and statistics updates
    // =========================================================================

    #[test]
    fn test_verify_phase_ema_formula() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let initial_ema = state.acceptance_rate_ema(); // 0.5

        // 100% acceptance → EMA = 0.3 * 1.0 + 0.7 * 0.5 = 0.65
        let r = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 20, 30]);
        let verify = VerifyResult::from_sequence_results(vec![r]);
        state.verify_phase(&verify);

        let expected = 0.3 * 1.0 + 0.7 * initial_ema;
        assert!((state.acceptance_rate_ema() - expected).abs() < 1e-4);
    }

    #[test]
    fn test_verify_phase_zero_acceptance_updates_streak() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[99]);
        let verify = VerifyResult::from_sequence_results(vec![r]);
        state.verify_phase(&verify);
        // Low acceptance: batch_acceptance = 0.0 < 0.3, so streak increments
        assert_eq!(state.spec_step_count(), 1);
    }

    #[test]
    fn test_verify_phase_high_acceptance_resets_streak() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        // First create a low acceptance streak
        let r_low = SequenceVerifyResult::verify_spine(1, &[10, 20], &[99]);
        state.verify_phase(&VerifyResult::from_sequence_results(vec![r_low]));

        // Then high acceptance should reset streak
        let r_high = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 20, 30]);
        state.verify_phase(&VerifyResult::from_sequence_results(vec![r_high]));
        assert!(state.is_active()); // streak was reset
    }

    #[test]
    fn test_verify_phase_accumulates_stats() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        // Round 1: 3 drafted, 3 accepted
        let r1 = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 20, 30]);
        state.verify_phase(&VerifyResult::from_sequence_results(vec![r1]));
        assert_eq!(state.spec_step_count(), 1);

        // Round 2: 2 drafted, 1 accepted
        let r2 = SequenceVerifyResult::verify_spine(1, &[40, 50], &[40, 99]);
        state.verify_phase(&VerifyResult::from_sequence_results(vec![r2]));
        assert_eq!(state.spec_step_count(), 2);

        // Total: 4 accepted / 5 drafted = 0.8
        assert!((state.avg_acceptance_rate() - 0.8).abs() < 1e-5);
    }

    // =========================================================================
    // avg_acceptance_rate
    // =========================================================================

    #[test]
    fn test_avg_acceptance_rate_no_drafts() {
        let state = SpecDecodingState::new_standard();
        assert!((state.avg_acceptance_rate() - 0.0).abs() < 1e-5);
    }

    // =========================================================================
    // temp_buffer_bytes
    // =========================================================================

    #[test]
    fn test_temp_buffer_bytes_standard_mode() {
        let state = SpecDecodingState::new_standard();
        // total_layers=0 → buffer = 0
        assert_eq!(state.temp_buffer_bytes(1024, 4), 0);
    }

    #[test]
    fn test_temp_buffer_bytes_calculation() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(12, adapter, SpecTreeConfig::default());
        // max_tree_size=32 (default), total_layers=12, 2 (K+V), kv_dim=512, dtype=2 (f16)
        let bytes = state.temp_buffer_bytes(512, 2);
        assert_eq!(bytes, 32 * 12 * 2 * 512 * 2);
    }

    // =========================================================================
    // draft_phase
    // =========================================================================

    #[test]
    fn test_draft_phase_empty_top_k_creates_empty_tree() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let prompt = vec![1, 2, 3];
        let tree = state.draft_phase(&[], &prompt);
        assert!(tree.is_empty());
    }

    #[test]
    fn test_draft_phase_stores_current_tree() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let top_k = vec![100u32, 200];
        let prompt = vec![1, 2, 3];

        state.draft_phase(&top_k, &prompt);
        assert!(state.current_tree().is_some());
    }

    #[test]
    fn test_draft_phase_stores_prompt_tokens() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let top_k = vec![100u32];
        let prompt = vec![5, 10, 15, 20];

        state.draft_phase(&top_k, &prompt);
        // Verify the tree was built with the given prompt
        assert!(state.current_tree().is_some());
    }

    // =========================================================================
    // Debug trait impl
    // =========================================================================

    #[test]
    fn test_debug_format_eesd_state() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("SpecDecodingState"));
        assert!(debug_str.contains("mode"));
        assert!(debug_str.contains("Eesd"));
        assert!(debug_str.contains("draft_layers"));
        assert!(debug_str.contains("acceptance_rate_ema"));
        assert!(debug_str.contains("avg_acceptance_rate"));
    }

    #[test]
    fn test_debug_format_standard_state() {
        let state = SpecDecodingState::new_standard();
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("Standard"));
    }

    // =========================================================================
    // adapter accessors
    // =========================================================================

    #[test]
    fn test_adapter_accessors_eesd() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        assert!(state.adapter().is_some());
    }

    #[test]
    fn test_adapter_mut_accessors_eesd() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        assert!(state.adapter_mut().is_some());
    }

    #[test]
    fn test_adapter_accessors_standard() {
        let state = SpecDecodingState::new_standard();
        assert!(state.adapter().is_none());
        let mut state = state;
        assert!(state.adapter_mut().is_none());
    }

    // =========================================================================
    // cache accessors
    // =========================================================================

    #[test]
    fn test_cache_accessors() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let cache_ref = state.cache();
        let debug_str = format!("{:?}", cache_ref);
        assert!(debug_str.contains("SpeculationCache"));

        let cache_mut = state.cache_mut();
        let debug_str2 = format!("{:?}", cache_mut);
        assert!(debug_str2.contains("SpeculationCache"));
    }

    // =========================================================================
    // set_draft_logits
    // =========================================================================

    #[test]
    fn test_set_draft_logits_mtp_mode() {
        let adapter = make_test_adapter(64, 1000);
        let mtp_head = MtpHead {
            projections: vec![vec![0.1; 64 * 100]],
            config: MtpConfig::default(),
        };
        let mtp_config = MtpConfig::default();
        let mut state = SpecDecodingState::new_mtp(
            32, adapter, SpecTreeConfig::default(), mtp_head, mtp_config,
        );

        let logits = vec![vec![0.5, 0.3, 0.2], vec![0.1, 0.6, 0.3]];
        state.set_draft_logits(logits);
    }

    // =========================================================================
    // multi_token_accept
    // =========================================================================

    #[test]
    fn test_multi_token_accept_initial_state() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        // Initial position_rates all 0.5, threshold 0.5 → ema_accept is rate >= threshold
        // so all positions pass
        let accepted = state.multi_token_accept(4);
        assert_eq!(accepted, 4);
    }

    #[test]
    fn test_multi_token_accept_zero_max() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        assert_eq!(state.multi_token_accept(0), 0);
    }

    // =========================================================================
    // mtp_ema_tracker accessor
    // =========================================================================

    #[test]
    fn test_mtp_ema_tracker_accessible() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let tracker = state.mtp_ema_tracker();
        // Default tracker: 4 positions, nominal_alpha=0.3
        assert_eq!(tracker.steps(), 0);
        assert!((tracker.threshold() - 0.5).abs() < 1e-5);
    }

    // =========================================================================
    // spec_step_count
    // =========================================================================

    #[test]
    fn test_spec_step_count_increments() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        assert_eq!(state.spec_step_count(), 0);

        let r1 = SequenceVerifyResult::verify_spine(1, &[10], &[10]);
        state.verify_phase(&VerifyResult::from_sequence_results(vec![r1]));
        assert_eq!(state.spec_step_count(), 1);

        let r2 = SequenceVerifyResult::verify_spine(1, &[10], &[10]);
        state.verify_phase(&VerifyResult::from_sequence_results(vec![r2]));
        assert_eq!(state.spec_step_count(), 2);
    }

    // =========================================================================
    // EMA convergence
    // =========================================================================

    #[test]
    fn test_acceptance_rate_ema_convergence_with_high_rate() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        // Initial EMA = 0.5. Repeatedly feed 100% acceptance.
        // EMA should converge toward 1.0
        for _ in 0..20 {
            let r = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 20, 30]);
            let verify = VerifyResult::from_sequence_results(vec![r]);
            state.verify_phase(&verify);
        }
        assert!(state.acceptance_rate_ema() > 0.9);
    }

    // =========================================================================
    // Verify empty draft tokens
    // =========================================================================

    #[test]
    fn test_verify_phase_empty_draft() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        let r = SequenceVerifyResult::verify_spine(1, &[], &[]);
        let verify = VerifyResult::from_sequence_results(vec![r]);
        state.verify_phase(&verify);
        assert_eq!(state.spec_step_count(), 1);
        assert!((state.avg_acceptance_rate() - 0.0).abs() < 1e-5);
    }

    // =========================================================================
    // Additional tests — untested types, traits, and edge cases
    // =========================================================================

    // --- SpecDecodingMode: all pairwise inequality ---

    #[test]
    fn test_spec_decoding_mode_all_variants_distinct() {
        let variants = [
            SpecDecodingMode::Eesd,
            SpecDecodingMode::Saguaro,
            SpecDecodingMode::Eagle,
            SpecDecodingMode::Mtp,
            SpecDecodingMode::Standard,
        ];
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    // --- SpecDecodingMode: Debug output for each variant ---

    #[test]
    fn test_spec_decoding_mode_debug_output() {
        assert!(format!("{:?}", SpecDecodingMode::Eesd).contains("Eesd"));
        assert!(format!("{:?}", SpecDecodingMode::Saguaro).contains("Saguaro"));
        assert!(format!("{:?}", SpecDecodingMode::Eagle).contains("Eagle"));
        assert!(format!("{:?}", SpecDecodingMode::Mtp).contains("Mtp"));
        assert!(format!("{:?}", SpecDecodingMode::Standard).contains("Standard"));
    }

    // --- SpecDecodingMode: Clone produces equal value ---

    #[test]
    fn test_spec_decoding_mode_clone() {
        let mode = SpecDecodingMode::Mtp;
        let cloned = mode.clone();
        assert_eq!(mode, cloned);
    }

    // --- draft_layers: various total_layers edge cases ---

    #[test]
    fn test_draft_layers_two_layers() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(2, adapter, SpecTreeConfig::default());
        // (2+2)/3 = 1
        assert_eq!(state.draft_layers(), 1);
    }

    #[test]
    fn test_draft_layers_three_layers_exact() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(3, adapter, SpecTreeConfig::default());
        // (3+2)/3 = 1
        assert_eq!(state.draft_layers(), 1);
    }

    #[test]
    fn test_draft_layers_large_model() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(96, adapter, SpecTreeConfig::default());
        // (96+2)/3 = 32
        assert_eq!(state.draft_layers(), 32);
    }

    // --- SpecTreeConfig: default values ---

    #[test]
    fn test_spec_tree_config_defaults() {
        let cfg = SpecTreeConfig::default();
        assert_eq!(cfg.max_spine_depth, 5);
        assert_eq!(cfg.max_branches_per_node, 2);
        assert_eq!(cfg.pld_ngram_len, 3);
        assert_eq!(cfg.ngram_top_k, 2);
        assert_eq!(cfg.adapter_top_k, 3);
        assert_eq!(cfg.max_tree_size, 32);
    }

    // --- SpecTreeConfig: Clone produces equal config ---

    #[test]
    fn test_spec_tree_config_clone() {
        let cfg = SpecTreeConfig::default();
        let cloned = cfg.clone();
        assert_eq!(cloned.max_spine_depth, cfg.max_spine_depth);
        assert_eq!(cloned.max_tree_size, cfg.max_tree_size);
        assert_eq!(cloned.pld_ngram_len, cfg.pld_ngram_len);
    }

    // --- AdapterConfig: default values ---

    #[test]
    fn test_adapter_config_defaults() {
        let cfg = AdapterConfig::default();
        assert_eq!(cfg.hidden_size, 0);
        assert_eq!(cfg.vocab_size, 0);
        assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!(!cfg.enable_distillation);
        assert_eq!(cfg.distillation_steps, 100);
    }

    // --- DraftAdapter: Debug output ---

    #[test]
    fn test_draft_adapter_debug_output() {
        let adapter = make_test_adapter(128, 500);
        let debug_str = format!("{:?}", adapter);
        assert!(debug_str.contains("DraftAdapter"));
        assert!(debug_str.contains("hidden_size"));
        assert!(debug_str.contains("128"));
        assert!(debug_str.contains("vocab_size"));
        assert!(debug_str.contains("500"));
        assert!(debug_str.contains("has_residual_delta"));
    }

    // --- DraftAdapter: new_phase_b has residual_delta ---

    #[test]
    fn test_draft_adapter_phase_b_has_residual() {
        use crate::speculative::adapter::DraftAdapter;
        let config = AdapterConfig {
            hidden_size: 64,
            vocab_size: 100,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 50,
        };
        let lm_head = Arc::new(vec![0.1f32; 100 * 64]);
        let norm_weight = Arc::new(vec![1.0f32; 64]);
        let adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        let debug_str = format!("{:?}", adapter);
        assert!(debug_str.contains("has_residual_delta: true"));
    }

    // --- EagleConfig: default values ---

    #[test]
    fn test_eagle_config_defaults() {
        let cfg = EagleConfig::default();
        assert_eq!(cfg.num_draft_layers, 1);
        assert_eq!(cfg.hidden_size, 0);
        assert_eq!(cfg.num_draft_tokens, 5);
        assert!((cfg.confidence_threshold - 0.7).abs() < 1e-5);
    }

    // --- MtpConfig: default values ---

    #[test]
    fn test_mtp_config_defaults() {
        let cfg = MtpConfig::default();
        assert_eq!(cfg.depth, 2);
        assert_eq!(cfg.vocab_size, 0);
        assert_eq!(cfg.hidden_size, 0);
    }

    // --- SpeculationCache: empty state ---

    #[test]
    fn test_speculation_cache_empty_initially() {
        let cache = SpeculationCache::new(4, 100);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!((cache.hit_rate() - 0.0).abs() < 1e-5);
    }

    // --- SpeculationCache: insert and lookup ---

    #[test]
    fn test_speculation_cache_insert_lookup() {
        let mut cache = SpeculationCache::new(4, 100);
        let entry = super::super::cache::CacheEntry {
            prefix_hash: 123,
            position: 5,
            candidates: vec![10, 20],
            logits: vec![0.5, 0.3],
            accept_count: 1,
            total_count: 2,
        };
        cache.insert(entry);
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
        let found = cache.lookup(123, 5).unwrap();
        assert_eq!(found.candidates, vec![10, 20]);
    }

    // --- SpeculationCache: lookup miss ---

    #[test]
    fn test_speculation_cache_lookup_miss() {
        let cache = SpeculationCache::new(4, 100);
        assert!(cache.lookup(999, 0).is_none());
    }

    // --- SpeculationCache: refresh replaces all entries ---

    #[test]
    fn test_speculation_cache_refresh_clears_old() {
        let mut cache = SpeculationCache::new(4, 100);
        let old = super::super::cache::CacheEntry {
            prefix_hash: 1,
            position: 0,
            candidates: vec![100],
            logits: vec![1.0],
            accept_count: 0,
            total_count: 0,
        };
        cache.insert(old);
        assert_eq!(cache.len(), 1);

        let new = super::super::cache::CacheEntry {
            prefix_hash: 2,
            position: 0,
            candidates: vec![200],
            logits: vec![2.0],
            accept_count: 0,
            total_count: 0,
        };
        cache.refresh(vec![new]);
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(1, 0).is_none());
        assert!(cache.lookup(2, 0).is_some());
    }

    // --- FallbackStrategy: equality and copy ---

    #[test]
    fn test_fallback_strategy_variants() {
        use super::super::cache::FallbackStrategy;
        assert_eq!(FallbackStrategy::SlowDraft, FallbackStrategy::SlowDraft);
        assert_ne!(FallbackStrategy::SlowDraft, FallbackStrategy::FastNgram);
        let copied = FallbackStrategy::FastNgram;
        assert_eq!(FallbackStrategy::FastNgram, copied);
    }

    // --- FallbackStrategy: Debug output ---

    #[test]
    fn test_fallback_strategy_debug() {
        use super::super::cache::FallbackStrategy;
        assert!(format!("{:?}", FallbackStrategy::SlowDraft).contains("SlowDraft"));
        assert!(format!("{:?}", FallbackStrategy::FastNgram).contains("FastNgram"));
    }

    // --- VerifyResult: from empty sequence results ---

    #[test]
    fn test_verify_result_from_empty_results() {
        let verify = VerifyResult::from_sequence_results(vec![]);
        assert_eq!(verify.total_accepted_tokens, 0);
        assert_eq!(verify.total_draft_tokens, 0);
        assert!((verify.avg_acceptance_rate - 0.0).abs() < 1e-5);
        assert!(verify.all_invariants_passed);
    }

    // --- VerifyResult: result_for with matching and non-matching ids ---

    #[test]
    fn test_verify_result_result_for_found_and_missing() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[10, 20], &[10, 20]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[30], &[99]);
        let verify = VerifyResult::from_sequence_results(vec![r1, r2]);
        assert!(verify.result_for(1).is_some());
        assert!(verify.result_for(2).is_some());
        assert!(verify.result_for(999).is_none());
    }

    // --- VerifyResult: check topology invariant with mismatched lengths ---

    #[test]
    fn test_verify_result_topology_invariant_mismatched_length() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[10], &[10]);
        let verify = VerifyResult::from_sequence_results(vec![r1]);
        // Expected has 2 items, results has 1 → mismatch
        assert!(!verify.check_topology_invariant(&[1, 1]));
    }

    // --- VerifyResult: check topology invariant passes ---

    #[test]
    fn test_verify_result_topology_invariant_passes() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[10, 20], &[10, 20]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[30, 40], &[30, 99]);
        let verify = VerifyResult::from_sequence_results(vec![r1, r2]);
        assert!(verify.check_topology_invariant(&[2, 2]));
    }

    // --- VerifyResult: single verification invariant always true ---

    #[test]
    fn test_verify_result_single_verification_always_true() {
        let verify = VerifyResult::from_sequence_results(vec![]);
        assert!(verify.check_single_verification_invariant());
    }

    // --- VerifyResult: check all invariants ---

    #[test]
    fn test_verify_result_check_all_invariants() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[10], &[10]);
        let verify = VerifyResult::from_sequence_results(vec![r1]);
        let result = verify.check_all_invariants(&[1]);
        assert!(result.all_passed());
        assert!(result.i1_topology);
        assert!(result.i2_single_verification);
        assert!(result.i3_atomic_kv_commit);
    }

    // --- EqSpecInvariant: all variants are distinct ---

    #[test]
    fn test_eq_spec_invariant_variants_distinct() {
        use super::super::verify::EqSpecInvariant;
        assert_ne!(EqSpecInvariant::Topology, EqSpecInvariant::SingleVerification);
        assert_ne!(EqSpecInvariant::SingleVerification, EqSpecInvariant::AtomicKvCommit);
        assert_ne!(EqSpecInvariant::Topology, EqSpecInvariant::AtomicKvCommit);
    }

    // --- EqSpecCheckResult: all_passed true only when all three are true ---

    #[test]
    fn test_eq_spec_check_result_all_passed() {
        use super::super::verify::EqSpecCheckResult;
        let all_true = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: true,
            i3_atomic_kv_commit: true,
        };
        assert!(all_true.all_passed());

        let partial = EqSpecCheckResult {
            i1_topology: true,
            i2_single_verification: true,
            i3_atomic_kv_commit: false,
        };
        assert!(!partial.all_passed());
    }

    // --- SequenceVerifyResult: single token match ---

    #[test]
    fn test_sequence_verify_result_single_token_match() {
        let r = SequenceVerifyResult::verify_spine(1, &[42], &[42]);
        assert_eq!(r.accepted_count, 1);
        assert_eq!(r.accepted_tokens, vec![42]);
        assert_eq!(r.rejected_count, 0);
        assert!((r.acceptance_rate - 1.0).abs() < 1e-5);
        assert!(r.invariant_check_passed);
    }

    // --- SequenceVerifyResult: single token mismatch ---

    #[test]
    fn test_sequence_verify_result_single_token_mismatch() {
        let r = SequenceVerifyResult::verify_spine(1, &[42], &[99]);
        assert_eq!(r.accepted_count, 0);
        assert!(r.accepted_tokens.is_empty());
        assert_eq!(r.rejected_count, 1);
        assert!((r.acceptance_rate - 0.0).abs() < 1e-5);
    }

    // --- SequenceVerifyResult: partial match stops at first divergence ---

    #[test]
    fn test_sequence_verify_result_partial_match_stops_at_first_divergence() {
        let r = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 99, 30]);
        assert_eq!(r.accepted_count, 1);
        assert_eq!(r.accepted_tokens, vec![10]);
        assert_eq!(r.rejected_count, 2);
    }

    // --- SequenceVerifyResult: empty constructor ---

    #[test]
    fn test_sequence_verify_result_empty() {
        let r = SequenceVerifyResult::empty(7);
        assert_eq!(r.request_id, 7);
        assert_eq!(r.accepted_count, 0);
        assert!(r.accepted_tokens.is_empty());
        assert_eq!(r.rejected_count, 0);
        assert_eq!(r.draft_count, 0);
        assert!((r.acceptance_rate - 0.0).abs() < 1e-5);
        assert!(r.invariant_check_passed);
    }

    // --- DraftSource: Copy and PartialEq ---

    #[test]
    fn test_draft_source_copy_equality() {
        use super::super::tree::DraftSource;
        let a = DraftSource::PldSpine;
        let b = a;
        assert_eq!(a, b);

        let c = DraftSource::AdapterTopK { k: 3 };
        let d = c;
        assert_eq!(c, d);
        assert_ne!(a, c);

        let e = DraftSource::NgramBranch;
        assert_ne!(a, e);
        assert_ne!(c, e);
    }

    // --- MtpEmaTracker: reset restores initial state ---

    #[test]
    fn test_mtp_ema_tracker_reset() {
        let mut tracker = MtpEmaTracker::new(4, 0.3);

        // Drive some data through
        for _ in 0..5 {
            tracker.record_batch(1.0);
        }
        assert!(tracker.steps() > 0);
        assert!(tracker.global_ema_rate() > 0.5);

        // Reset
        tracker.reset();
        assert_eq!(tracker.steps(), 0);
        assert!((tracker.global_ema_rate() - 0.5).abs() < 1e-5);
        assert!((tracker.threshold() - 0.5).abs() < 1e-5);
    }

    // --- MtpEmaTracker: position out of bounds returns false ---

    #[test]
    fn test_mtp_ema_tracker_ema_accept_out_of_bounds() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let tracker = state.mtp_ema_tracker();
        // 4 positions tracked (EESD default), position 100 is out of bounds
        assert!(!tracker.ema_accept(100));
    }

    // --- MtpEmaTracker: ema_acceptance_rate out of bounds returns None ---

    #[test]
    fn test_mtp_ema_tracker_acceptance_rate_out_of_bounds() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let tracker = state.mtp_ema_tracker();
        assert!(tracker.ema_acceptance_rate(0).is_some());
        assert!(tracker.ema_acceptance_rate(999).is_none());
    }

    // --- SpecScheduleAdvice: all variants are distinct ---

    #[test]
    fn test_spec_schedule_advice_variants_distinct() {
        assert_ne!(SpecScheduleAdvice::EnableSpec, SpecScheduleAdvice::StandardDecode);
        assert_ne!(SpecScheduleAdvice::StandardDecode, SpecScheduleAdvice::Fallback);
        assert_ne!(SpecScheduleAdvice::EnableSpec, SpecScheduleAdvice::Fallback);
    }

    // --- SpecScheduleAdvice: Debug output ---

    #[test]
    fn test_spec_schedule_advice_debug() {
        assert!(format!("{:?}", SpecScheduleAdvice::EnableSpec).contains("EnableSpec"));
        assert!(format!("{:?}", SpecScheduleAdvice::StandardDecode).contains("StandardDecode"));
        assert!(format!("{:?}", SpecScheduleAdvice::Fallback).contains("Fallback"));
    }

    // --- should_speculate: after reset_fallback_streak, advice recovers ---

    #[test]
    fn test_should_speculate_recoverable_after_reset() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        // Drive into fallback
        for _ in 0..3 {
            let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[99]);
            state.verify_phase(&VerifyResult::from_sequence_results(vec![r]));
        }
        assert_eq!(state.should_speculate(4), SpecScheduleAdvice::Fallback);

        // Reset streak → should no longer return Fallback
        state.reset_fallback_streak();
        let advice = state.should_speculate(4);
        // Could be StandardDecode or EnableSpec depending on EMA, but NOT Fallback
        assert_ne!(advice, SpecScheduleAdvice::Fallback);
    }

    // --- temp_buffer_bytes with zero kv_dim ---

    #[test]
    fn test_temp_buffer_bytes_zero_kv_dim() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        assert_eq!(state.temp_buffer_bytes(0, 4), 0);
    }

    // --- temp_buffer_bytes with zero dtype_size ---

    #[test]
    fn test_temp_buffer_bytes_zero_dtype_size() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        assert_eq!(state.temp_buffer_bytes(1024, 0), 0);
    }

    // --- SAGUARO verify_phase updates cache ---

    #[test]
    fn test_saguaro_verify_phase_updates_cache() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_saguaro(
            32, adapter, SpecTreeConfig::default(), 0, vec![1],
        );

        let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[10, 20]);
        state.verify_phase(&VerifyResult::from_sequence_results(vec![r]));
        // The cache's adapt_scale_factor should have been called
        assert_eq!(state.spec_step_count(), 1);
    }

    // --- EMA convergence toward 0 with repeated zero acceptance ---

    #[test]
    fn test_acceptance_rate_ema_converges_toward_zero() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        // Initial EMA = 0.5. Repeatedly feed 0% acceptance.
        for _ in 0..20 {
            let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[99]);
            let verify = VerifyResult::from_sequence_results(vec![r]);
            state.verify_phase(&verify);
        }
        assert!(state.acceptance_rate_ema() < 0.1);
    }

    // --- Debug trait includes ema_threshold field ---

    #[test]
    fn test_debug_format_includes_ema_threshold() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("ema_threshold"));
    }

    // --- Multi-verify rounds accumulate correctly ---

    #[test]
    fn test_multiple_verify_rounds_accumulate_totals() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        // Round 1: 2 drafted, 2 accepted
        let r1 = SequenceVerifyResult::verify_spine(1, &[10, 20], &[10, 20]);
        state.verify_phase(&VerifyResult::from_sequence_results(vec![r1]));

        // Round 2: 3 drafted, 1 accepted
        let r2 = SequenceVerifyResult::verify_spine(1, &[30, 40, 50], &[30, 99, 99]);
        state.verify_phase(&VerifyResult::from_sequence_results(vec![r2]));

        // Round 3: 1 drafted, 0 accepted
        let r3 = SequenceVerifyResult::verify_spine(1, &[60], &[99]);
        state.verify_phase(&VerifyResult::from_sequence_results(vec![r3]));

        assert_eq!(state.spec_step_count(), 3);
        // Total: 3 accepted / 6 drafted = 0.5
        assert!((state.avg_acceptance_rate() - 0.5).abs() < 1e-5);
    }

    // =========================================================================
    // NEW TESTS — additional coverage
    // =========================================================================

    // --- draft_layers edge cases ---

    #[test]
    fn test_draft_layers_four_layers() {
        // total_layers=4 → (4+2)/3 = 2
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(4, adapter, SpecTreeConfig::default());
        assert_eq!(state.draft_layers(), 2);
    }

    #[test]
    fn test_draft_layers_seven_layers() {
        // total_layers=7 → (7+2)/3 = 3
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(7, adapter, SpecTreeConfig::default());
        assert_eq!(state.draft_layers(), 3);
    }

    #[test]
    fn test_draft_layers_five_layers_rounds_up() {
        // total_layers=5 → (5+2)/3 = 2 (ceil)
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(5, adapter, SpecTreeConfig::default());
        assert_eq!(state.draft_layers(), 2);
    }

    #[test]
    fn test_draft_layers_always_at_most_total() {
        // Even with very few layers, draft_layers <= total_layers
        for total in 1..=10 {
            let adapter = make_test_adapter(64, 1000);
            let state = SpecDecodingState::new_eesd(total, adapter, SpecTreeConfig::default());
            assert!(
                state.draft_layers() <= state.total_layers(),
                "draft_layers={} > total_layers={}",
                state.draft_layers(),
                state.total_layers()
            );
        }
    }

    // --- SpecDecodingState: EAGLE preserves EESD base state ---

    #[test]
    fn test_eagle_preserves_draft_layers() {
        let adapter = make_test_adapter(64, 1000);
        let eagle_head = EagleHead {
            fc_weight: vec![0.1; 64 * 2],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let eagle_config = EagleConfig::default();
        let state = SpecDecodingState::new_eagle(
            24, adapter, SpecTreeConfig::default(), eagle_head, eagle_config,
        );
        assert_eq!(state.draft_layers(), 8); // (24+2)/3 = 8
        assert_eq!(state.total_layers(), 24);
    }

    // --- SpecDecodingState: MTP preserves EESD base state ---

    #[test]
    fn test_mtp_preserves_draft_layers() {
        let adapter = make_test_adapter(64, 1000);
        let mtp_head = MtpHead {
            projections: vec![vec![0.1; 64 * 50]],
            config: MtpConfig::default(),
        };
        let mtp_config = MtpConfig::default();
        let state = SpecDecodingState::new_mtp(
            18, adapter, SpecTreeConfig::default(), mtp_head, mtp_config,
        );
        assert_eq!(state.draft_layers(), 6); // (18+2)/3 = 6
        assert_eq!(state.total_layers(), 18);
    }

    // --- SAGUARO preserves EESD base fields ---

    #[test]
    fn test_saguaro_preserves_eesd_base_fields() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_saguaro(
            30, adapter, SpecTreeConfig::default(), 1, vec![0, 2],
        );
        assert_eq!(state.draft_layers(), 10); // (30+2)/3 = 10
        assert_eq!(state.total_layers(), 30);
        assert!((state.acceptance_rate_ema() - 0.5).abs() < 1e-5);
        assert!(state.adapter().is_some());
        assert!(state.eagle_head().is_none());
        assert!(state.mtp_head().is_none());
    }

    // --- should_speculate: with 1 decode request ---

    #[test]
    fn test_should_speculate_single_decode_request() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let advice = state.should_speculate(1);
        // Single request should still produce advice
        assert!(matches!(
            advice,
            SpecScheduleAdvice::EnableSpec | SpecScheduleAdvice::StandardDecode
        ));
    }

    // --- should_speculate: EAGLE mode after high-acceptance verify ---

    #[test]
    fn test_should_speculate_eagle_mode_after_high_acceptance() {
        let adapter = make_test_adapter(64, 1000);
        let eagle_head = EagleHead {
            fc_weight: vec![0.1; 64 * 2],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let mut state = SpecDecodingState::new_eagle(
            32, adapter, SpecTreeConfig::default(), eagle_head, EagleConfig::default(),
        );
        // Feed high acceptance to raise EMA above threshold
        for _ in 0..5 {
            let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[10, 20]);
            state.verify_phase(&VerifyResult::from_sequence_results(vec![r]));
        }
        let advice = state.should_speculate(5);
        assert_eq!(advice, SpecScheduleAdvice::EnableSpec);
    }

    // --- should_speculate: MTP mode after high-acceptance verify ---

    #[test]
    fn test_should_speculate_mtp_mode_after_high_acceptance() {
        let adapter = make_test_adapter(64, 1000);
        let mtp_head = MtpHead {
            projections: vec![vec![0.1; 64 * 50]],
            config: MtpConfig::default(),
        };
        let mut state = SpecDecodingState::new_mtp(
            32, adapter, SpecTreeConfig::default(), mtp_head, MtpConfig::default(),
        );
        // Feed high acceptance to raise EMA above threshold
        for _ in 0..5 {
            let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[10, 20]);
            state.verify_phase(&VerifyResult::from_sequence_results(vec![r]));
        }
        let advice = state.should_speculate(5);
        assert_eq!(advice, SpecScheduleAdvice::EnableSpec);
    }

    // --- is_active: SAGUARO mode is active ---

    #[test]
    fn test_is_active_saguaro() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_saguaro(
            32, adapter, SpecTreeConfig::default(), 0, vec![1],
        );
        assert!(state.is_active());
    }

    // --- is_active: EAGLE mode is active ---

    #[test]
    fn test_is_active_eagle() {
        let adapter = make_test_adapter(64, 1000);
        let eagle_head = EagleHead {
            fc_weight: vec![0.1; 64 * 2],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let state = SpecDecodingState::new_eagle(
            32, adapter, SpecTreeConfig::default(), eagle_head, EagleConfig::default(),
        );
        assert!(state.is_active());
    }

    // --- is_active: MTP mode is active ---

    #[test]
    fn test_is_active_mtp() {
        let adapter = make_test_adapter(64, 1000);
        let mtp_head = MtpHead {
            projections: vec![vec![0.1; 64 * 50]],
            config: MtpConfig::default(),
        };
        let state = SpecDecodingState::new_mtp(
            32, adapter, SpecTreeConfig::default(), mtp_head, MtpConfig::default(),
        );
        assert!(state.is_active());
    }

    // --- reset_fallback_streak on fresh state is no-op ---

    #[test]
    fn test_reset_fallback_streak_noop_when_no_streak() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        assert!(state.is_active());
        state.reset_fallback_streak();
        assert!(state.is_active());
    }

    // --- set_draft_logits overwrites previous ---

    #[test]
    fn test_set_draft_logits_overwrites_previous() {
        let adapter = make_test_adapter(64, 1000);
        let mtp_head = MtpHead {
            projections: vec![vec![0.1; 64 * 50]],
            config: MtpConfig::default(),
        };
        let mut state = SpecDecodingState::new_mtp(
            32, adapter, SpecTreeConfig::default(), mtp_head, MtpConfig::default(),
        );
        state.set_draft_logits(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        // Set again — overwrite
        state.set_draft_logits(vec![vec![5.0]]);
        // Verify by checking draft_phase does not panic with the new logits
        let prompt = vec![1, 2, 3];
        let top_k = vec![100];
        let tree = state.draft_phase(&top_k, &prompt);
        assert!(!tree.is_empty());
    }

    // --- set_draft_logits with empty vec ---

    #[test]
    fn test_set_draft_logits_empty_clears() {
        let adapter = make_test_adapter(64, 1000);
        let mtp_head = MtpHead {
            projections: vec![vec![0.1; 64 * 50]],
            config: MtpConfig::default(),
        };
        let mut state = SpecDecodingState::new_mtp(
            32, adapter, SpecTreeConfig::default(), mtp_head, MtpConfig::default(),
        );
        state.set_draft_logits(vec![vec![1.0, 2.0]]);
        state.set_draft_logits(vec![]);
        let tree = state.draft_phase(&[100], &[1, 2, 3]);
        assert!(!tree.is_empty());
    }

    // --- temp_buffer_bytes: various kv_dim and dtype_size combinations ---

    #[test]
    fn test_temp_buffer_bytes_various_params() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(24, adapter, SpecTreeConfig::default());
        // max_tree=32, total_layers=24, K+V=2
        let b1 = state.temp_buffer_bytes(128, 2);
        assert_eq!(b1, 32 * 24 * 2 * 128 * 2);

        let b2 = state.temp_buffer_bytes(256, 4);
        assert_eq!(b2, 32 * 24 * 2 * 256 * 4);
    }

    // --- draft_phase with single token top_k ---

    #[test]
    fn test_draft_phase_single_token_top_k() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let tree = state.draft_phase(&[42], &[1, 2, 3]);
        assert!(tree.len() >= 1);
    }

    // --- draft_phase replaces previous tree ---

    #[test]
    fn test_draft_phase_replaces_previous_tree() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        let tree1 = state.draft_phase(&[10, 20, 30], &[1, 2, 3]);
        let len1 = tree1.len();

        let tree2 = state.draft_phase(&[40, 50], &[1, 2, 3, 4]);
        let len2 = tree2.len();

        // Both trees should be non-empty; they might differ in size
        assert!(len1 >= 1);
        assert!(len2 >= 1);
        assert!(state.current_tree().is_some());
    }

    // --- verify_phase: EMA with exact boundary at 0.3 acceptance ---

    #[test]
    fn test_verify_phase_exact_boundary_acceptance() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        // acceptance_rate exactly 0.3 (1 drafted, 1 accepted out of 1? No, we need 0.3 exactly)
        // 3 drafted, 1 accepted → 1/3 ≈ 0.333
        let r = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 99, 99]);
        let verify = VerifyResult::from_sequence_results(vec![r]);
        state.verify_phase(&verify);
        // acceptance_rate = 0.333 > 0.3, so streak should reset
        assert!(state.is_active());
    }

    // --- verify_phase: single sequence round trip ---

    #[test]
    fn test_verify_phase_single_sequence_round_trip() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        // Round 1: 1 drafted, 1 accepted → 100%
        let r = SequenceVerifyResult::verify_spine(1, &[42], &[42]);
        state.verify_phase(&VerifyResult::from_sequence_results(vec![r]));
        assert_eq!(state.spec_step_count(), 1);
        assert!((state.avg_acceptance_rate() - 1.0).abs() < 1e-5);
    }

    // --- avg_acceptance_rate stays in [0, 1] range ---

    #[test]
    fn test_avg_acceptance_rate_always_bounded() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        for _ in 0..20 {
            let r = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 20, 30]);
            state.verify_phase(&VerifyResult::from_sequence_results(vec![r]));
        }
        assert!(state.avg_acceptance_rate() >= 0.0);
        assert!(state.avg_acceptance_rate() <= 1.0);
    }

    // --- acceptance_rate_ema converges monotonically with constant input ---

    #[test]
    fn test_acceptance_rate_ema_monotonic_high() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());
        let mut prev = state.acceptance_rate_ema();

        for _ in 0..10 {
            let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[10, 20]);
            state.verify_phase(&VerifyResult::from_sequence_results(vec![r]));
            let current = state.acceptance_rate_ema();
            assert!(current >= prev, "EMA should increase with constant 100% input");
            prev = current;
        }
    }

    // --- SpecTreeConfig: PartialEq ---

    #[test]
    fn test_spec_tree_config_equality() {
        let a = SpecTreeConfig::default();
        let b = SpecTreeConfig::default();
        assert_eq!(a, b);
    }

    #[test]
    fn test_spec_tree_config_inequality() {
        let mut a = SpecTreeConfig::default();
        let b = SpecTreeConfig::default();
        a.max_tree_size = 64;
        assert_ne!(a, b);
    }

    // --- AdapterConfig: equality ---

    #[test]
    fn test_adapter_config_equality() {
        let a = AdapterConfig::default();
        let b = AdapterConfig::default();
        assert_eq!(a.hidden_size, b.hidden_size);
        assert_eq!(a.vocab_size, b.vocab_size);
    }

    // --- DraftAdapter: parameter_bytes for Phase A is zero ---

    #[test]
    fn test_draft_adapter_phase_a_zero_bytes() {
        let adapter = make_test_adapter(64, 1000);
        assert_eq!(adapter.parameter_bytes(), 0);
    }

    // --- DraftAdapter: parameter_bytes for Phase B is non-zero ---

    #[test]
    fn test_draft_adapter_phase_b_nonzero_bytes() {
        let config = AdapterConfig {
            hidden_size: 64,
            vocab_size: 100,
            rms_norm_eps: 1e-5,
            enable_distillation: true,
            distillation_steps: 50,
        };
        let lm_head = Arc::new(vec![0.1f32; 100 * 64]);
        let norm_weight = Arc::new(vec![1.0f32; 64]);
        let adapter = DraftAdapter::new_phase_b(config, lm_head, norm_weight);
        assert!(adapter.parameter_bytes() > 0);
        assert_eq!(adapter.parameter_bytes(), 64 * 100 * 4); // f32 = 4 bytes
    }

    // --- DraftAdapter: distillation progress for Phase A ---

    #[test]
    fn test_draft_adapter_phase_a_distillation_incomplete() {
        let adapter = make_test_adapter(64, 1000);
        assert!(!adapter.is_distillation_complete());
        let (step, total) = adapter.distillation_progress();
        assert_eq!(step, 0);
        assert_eq!(total, 100);
    }

    // --- MtpEmaTracker: new with zero positions uses max(1, 0) = 1 position ---

    #[test]
    fn test_mtp_ema_tracker_new_zero_positions() {
        let tracker = MtpEmaTracker::new(0, 0.3);
        // max_positions.max(1) → at least 1 position
        assert!(tracker.ema_acceptance_rate(0).is_some());
        assert!(tracker.ema_acceptance_rate(1).is_none());
    }

    // --- MtpEmaTracker: record_position updates global_rate ---

    #[test]
    fn test_mtp_ema_tracker_record_position_updates_global() {
        let mut tracker = MtpEmaTracker::new(4, 0.3);
        let initial = tracker.global_ema_rate();

        tracker.record_position(0, true);
        assert!(tracker.global_ema_rate() > initial);
    }

    // --- MtpEmaTracker: record_position out of bounds is no-op ---

    #[test]
    fn test_mtp_ema_tracker_record_position_out_of_bounds() {
        let mut tracker = MtpEmaTracker::new(2, 0.3);
        let before = tracker.global_ema_rate();
        tracker.record_position(100, true);
        assert!((tracker.global_ema_rate() - before).abs() < 1e-5);
    }

    // --- MtpEmaTracker: record_batch increases steps ---

    #[test]
    fn test_mtp_ema_tracker_record_batch_increments_steps() {
        let mut tracker = MtpEmaTracker::new(4, 0.3);
        assert_eq!(tracker.steps(), 0);
        tracker.record_batch(0.8);
        assert_eq!(tracker.steps(), 1);
        tracker.record_batch(0.5);
        assert_eq!(tracker.steps(), 2);
    }

    // --- MtpEmaTracker: record_batch with 1.0 raises global_rate ---

    #[test]
    fn test_mtp_ema_tracker_record_batch_high_rate() {
        let mut tracker = MtpEmaTracker::new(4, 0.3);
        for _ in 0..20 {
            tracker.record_batch(1.0);
        }
        assert!(tracker.global_ema_rate() > 0.9);
    }

    // --- MtpEmaTracker: record_batch with 0.0 lowers global_rate ---

    #[test]
    fn test_mtp_ema_tracker_record_batch_zero_rate() {
        let mut tracker = MtpEmaTracker::new(4, 0.3);
        for _ in 0..20 {
            tracker.record_batch(0.0);
        }
        assert!(tracker.global_ema_rate() < 0.1);
    }

    // --- MtpEmaTracker: multi_token_accept stops at first rejection ---

    #[test]
    fn test_mtp_ema_tracker_multi_token_stops_at_rejection() {
        let mut tracker = MtpEmaTracker::new(4, 0.3);
        // Feed high rate for position 0, then low rate
        tracker.record_position(0, true);
        tracker.record_position(1, false);
        // Position 1 has lower rate; multi_token_accept should stop there
        // (depends on threshold dynamics; at minimum it won't exceed 2)
        let accepted = tracker.multi_token_accept(4);
        assert!(accepted <= 2);
    }

    // --- MtpEmaTracker: ema_acceptance_rate valid range ---

    #[test]
    fn test_mtp_ema_tracker_acceptance_rate_valid_range() {
        let mut tracker = MtpEmaTracker::new(4, 0.3);
        for _ in 0..10 {
            tracker.record_batch(0.6);
        }
        for pos in 0..4 {
            if let Some(rate) = tracker.ema_acceptance_rate(pos) {
                assert!(rate >= 0.0 && rate <= 1.0, "rate {} out of range at pos {}", rate, pos);
            }
        }
    }

    // --- MtpEmaTracker: threshold stays in reasonable bounds ---

    #[test]
    fn test_mtp_ema_tracker_threshold_bounded() {
        let mut tracker = MtpEmaTracker::new(4, 0.3);
        for i in 0..50 {
            tracker.record_batch(if i % 2 == 0 { 1.0 } else { 0.0 });
        }
        let t = tracker.threshold();
        assert!(t >= 0.2 && t <= 0.8, "threshold {} out of bounds", t);
    }

    // --- MtpEmaTracker: Debug output ---

    #[test]
    fn test_mtp_ema_tracker_debug_format() {
        let tracker = MtpEmaTracker::new(4, 0.3);
        let debug = format!("{:?}", tracker);
        assert!(debug.contains("MtpEmaTracker"));
    }

    // --- SpeculationCache: eviction with total_count > 0 ---

    #[test]
    fn test_speculation_cache_eviction_with_total_count() {
        let mut cache = SpeculationCache::new(4, 2);
        let e1 = super::super::cache::CacheEntry {
            prefix_hash: 1, position: 0, candidates: vec![10],
            logits: vec![1.0], accept_count: 1, total_count: 5,
        };
        let e2 = super::super::cache::CacheEntry {
            prefix_hash: 2, position: 0, candidates: vec![20],
            logits: vec![2.0], accept_count: 3, total_count: 5,
        };
        cache.insert(e1);
        cache.insert(e2);
        assert_eq!(cache.len(), 2);
        // Insert a 3rd entry — should evict e1 (lower accept_count)
        let e3 = super::super::cache::CacheEntry {
            prefix_hash: 3, position: 0, candidates: vec![30],
            logits: vec![3.0], accept_count: 2, total_count: 5,
        };
        cache.insert(e3);
        assert!(cache.len() <= 3);
        // The entry with hash=1 should have been evicted (lowest accept_count)
        assert!(cache.lookup(1, 0).is_none());
    }

    // --- CacheEntry: fields accessible ---

    #[test]
    fn test_cache_entry_fields() {
        let entry = super::super::cache::CacheEntry {
            prefix_hash: 12345,
            position: 7,
            candidates: vec![10, 20, 30],
            logits: vec![0.5, 0.3, 0.2],
            accept_count: 5,
            total_count: 10,
        };
        assert_eq!(entry.prefix_hash, 12345);
        assert_eq!(entry.position, 7);
        assert_eq!(entry.candidates.len(), 3);
        assert_eq!(entry.logits.len(), 3);
        assert_eq!(entry.accept_count, 5);
        assert_eq!(entry.total_count, 10);
    }

    // --- DraftSource: Debug output for all variants ---

    #[test]
    fn test_draft_source_debug_all_variants() {
        use super::super::tree::DraftSource;
        let ds1 = DraftSource::PldSpine;
        let ds2 = DraftSource::AdapterTopK { k: 3 };
        let ds3 = DraftSource::NgramBranch;
        assert!(format!("{:?}", ds1).contains("PldSpine"));
        assert!(format!("{:?}", ds2).contains("AdapterTopK"));
        assert!(format!("{:?}", ds3).contains("NgramBranch"));
    }

    // --- DraftSource: Hash consistency ---

    #[test]
    fn test_draft_source_hash_consistency() {
        use super::super::tree::DraftSource;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(DraftSource::PldSpine);
        set.insert(DraftSource::NgramBranch);
        assert!(set.contains(&DraftSource::PldSpine));
        assert!(set.contains(&DraftSource::NgramBranch));
        assert!(!set.contains(&DraftSource::AdapterTopK { k: 1 }));
    }

    // --- SpecNode: default fields ---

    #[test]
    fn test_spec_node_construction() {
        let node = super::super::tree::SpecNode {
            node_id: 0,
            token_id: 42,
            parent_id: None,
            children: vec![1, 2],
            source: super::super::tree::DraftSource::PldSpine,
            estimated_acceptance: 0.7,
            position_offset: 0,
        };
        assert_eq!(node.node_id, 0);
        assert_eq!(node.token_id, 42);
        assert!(node.parent_id.is_none());
        assert_eq!(node.children, vec![1, 2]);
    }

    // --- EagleConfig: Clone produces equal config ---

    #[test]
    fn test_eagle_config_clone() {
        let cfg = EagleConfig::default();
        let cloned = cfg.clone();
        assert_eq!(cfg, cloned);
    }

    // --- EagleHead: Clone produces equal head ---

    #[test]
    fn test_eagle_head_clone() {
        let head = EagleHead {
            fc_weight: vec![0.5; 128],
            draft_layers: vec![],
            share_lm_head: true,
        };
        let cloned = head.clone();
        assert_eq!(head, cloned);
    }

    // --- EagleHead: Debug output ---

    #[test]
    fn test_eagle_head_debug() {
        let head = EagleHead {
            fc_weight: vec![0.1; 64],
            draft_layers: vec![],
            share_lm_head: false,
        };
        let debug = format!("{:?}", head);
        assert!(debug.contains("EagleHead"));
    }

    // --- DraftLayerWeights: Clone and equality ---

    #[test]
    fn test_draft_layer_weights_clone() {
        use super::super::eagle::DraftLayerWeights;
        let w = DraftLayerWeights {
            up_weight: vec![1.0, 2.0],
            down_weight: vec![3.0, 4.0],
        };
        let cloned = w.clone();
        assert_eq!(w, cloned);
    }

    // --- MtpConfig: Clone produces equal config ---

    #[test]
    fn test_mtp_config_clone() {
        let cfg = MtpConfig::default();
        let cloned = cfg.clone();
        assert_eq!(cfg, cloned);
    }

    // --- MtpConfig: Copy trait ---

    #[test]
    fn test_mtp_config_copy() {
        let cfg = MtpConfig::default();
        let copied = cfg;
        assert_eq!(cfg, copied);
    }

    // --- MtpConfig: Hash in HashSet ---

    #[test]
    fn test_mtp_config_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MtpConfig::default());
        assert!(set.contains(&MtpConfig::default()));
    }

    // --- MtpHead: Clone produces equal head ---

    #[test]
    fn test_mtp_head_clone() {
        let head = MtpHead {
            projections: vec![vec![0.1; 64]],
            config: MtpConfig::default(),
        };
        let cloned = head.clone();
        assert_eq!(head, cloned);
    }

    // --- MtpHead: Debug output ---

    #[test]
    fn test_mtp_head_debug() {
        let head = MtpHead {
            projections: vec![vec![0.1; 32]],
            config: MtpConfig::default(),
        };
        let debug = format!("{:?}", head);
        assert!(debug.contains("MtpHead"));
    }

    // --- SpecScheduleAdvice: Copy trait ---

    #[test]
    fn test_spec_schedule_advice_copy() {
        let a = SpecScheduleAdvice::EnableSpec;
        let b = a;
        assert_eq!(a, b);
    }

    // --- SpecScheduleAdvice: Clone trait ---

    #[test]
    fn test_spec_schedule_advice_clone() {
        let a = SpecScheduleAdvice::Fallback;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // --- VerifyResult: topology invariant with empty expected and non-empty results ---

    #[test]
    fn test_verify_result_topology_invariant_empty_expected_nonempty_results() {
        let r = SequenceVerifyResult::verify_spine(1, &[10], &[10]);
        let verify = VerifyResult::from_sequence_results(vec![r]);
        assert!(!verify.check_topology_invariant(&[]));
    }

    // --- VerifyResult: atomic_kv_commit with single passing result ---

    #[test]
    fn test_verify_result_atomic_kv_commit_single_pass() {
        let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[10, 20]);
        let verify = VerifyResult::from_sequence_results(vec![r]);
        assert!(verify.check_atomic_kv_commit_invariant());
    }

    // --- SequenceVerifyResult: verify_spine preserves draft_count ---

    #[test]
    fn test_sequence_verify_result_draft_count() {
        let r = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 99]);
        assert_eq!(r.draft_count, 3);
    }

    // --- SequenceVerifyResult: verify_spine all match long sequence ---

    #[test]
    fn test_sequence_verify_result_all_match() {
        let draft: Vec<u32> = (1..=10).collect();
        let target: Vec<u32> = (1..=10).collect();
        let r = SequenceVerifyResult::verify_spine(1, &draft, &target);
        assert_eq!(r.accepted_count, 10);
        assert_eq!(r.rejected_count, 0);
        assert!((r.acceptance_rate - 1.0).abs() < 1e-5);
    }

    // --- SequenceVerifyResult: Debug format ---

    #[test]
    fn test_sequence_verify_result_debug() {
        let r = SequenceVerifyResult::verify_spine(1, &[10], &[10]);
        let debug = format!("{:?}", r);
        assert!(debug.contains("SequenceVerifyResult"));
        assert!(debug.contains("request_id"));
        assert!(debug.contains("accepted_count"));
    }

    // --- NgramIndex: build with empty tokens ---

    #[test]
    fn test_ngram_index_build_empty() {
        let idx = NgramIndex::build(&[], 3);
        let cont = idx.get_continuations(42, 5);
        assert!(cont.is_empty());
    }

    // --- NgramIndex: build with short tokens (len <= n, no n-grams) ---

    #[test]
    fn test_ngram_index_build_short_tokens() {
        // tokens.len()=2, n=3 → 2 <= 3, so table is empty
        let idx = NgramIndex::build(&[1, 2], 3);
        let cont = idx.get_continuations(2, 5);
        assert!(cont.is_empty());
    }

    // --- NgramIndex: build with sufficient tokens and query continuations ---

    #[test]
    fn test_ngram_index_query_continuations() {
        // With n=2, tokens.len()=9 > 2, n-grams are built.
        // get_continuations hashes a 1-gram [token], which is a different key
        // than the 2-gram keys stored. So 1-gram lookups won't find 2-gram entries.
        // This tests that the index was built without panic and returns empty for
        // 1-gram queries when only 2-gram keys exist.
        let tokens = vec![1, 2, 3, 1, 2, 4, 1, 2, 5];
        let idx = NgramIndex::build(&tokens, 2);
        // 1-gram lookup won't match 2-gram keys
        let cont = idx.get_continuations(2, 3);
        assert!(cont.is_empty());
    }

    // --- NgramIndex: get_ngram_continuations with matching n-gram ---

    #[test]
    fn test_ngram_index_ngram_continuations() {
        // n=3, tokens.len()=6 > 3
        // n-grams: [10,20,30]→40, [20,30,10]→20, [30,10,20]→40
        let tokens = vec![10, 20, 30, 10, 20, 40];
        let idx = NgramIndex::build(&tokens, 3);
        let cont = idx.get_ngram_continuations(&[10, 20, 30], 5);
        assert!(cont.contains(&10));
    }

    // --- NgramIndex: get_ngram_continuations with no match ---

    #[test]
    fn test_ngram_index_ngram_continuations_no_match() {
        let tokens = vec![1, 2, 3, 4, 5, 6];
        let idx = NgramIndex::build(&tokens, 3);
        // [99, 98, 97] won't match any n-gram
        let cont = idx.get_ngram_continuations(&[99, 98, 97], 5);
        assert!(cont.is_empty());
    }

    // --- SpecTree: empty new ---

    #[test]
    fn test_spec_tree_new_empty() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
    }

    // --- SpecTree: all_token_ids on empty tree ---

    #[test]
    fn test_spec_tree_all_token_ids_empty() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.all_token_ids().is_empty());
    }

    // --- SpecTree: spine_ids on empty tree panics (no root node) ---
    // Skipping: spine_ids() requires at least one node; panics on empty tree by design.

    // --- SpecTree: node returns None for invalid id ---

    #[test]
    fn test_spec_tree_node_invalid_id() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.node(0).is_none());
        assert!(tree.node(999).is_none());
    }

    // --- SpecTree: nodes on empty tree ---

    #[test]
    fn test_spec_tree_nodes_empty() {
        let tree = SpecTree::new(SpecTreeConfig::default());
        assert!(tree.nodes().is_empty());
    }

    // --- SpecTree: branch_token_ids on empty tree panics (calls spine_ids) ---
    // Skipping: branch_token_ids() calls spine_ids() which panics on empty tree.

    // --- KvCommitInstruction: Clone ---

    #[test]
    fn test_kv_commit_instruction_clone() {
        use super::super::verify::KvCommitInstruction;
        let instr = KvCommitInstruction::Commit {
            request_id: 1,
            accepted_tokens: vec![10, 20],
            kv_pages_to_commit: vec![0, 1],
        };
        let cloned = instr.clone();
        match cloned {
            KvCommitInstruction::Commit { request_id, .. } => assert_eq!(request_id, 1),
            _ => panic!("expected Commit variant"),
        }
    }

    // --- generate_kv_commit_instructions for single all-accepted ---

    #[test]
    fn test_generate_kv_instructions_single_all_accepted() {
        use super::super::verify::{generate_kv_commit_instructions, KvCommitInstruction, SpeculativePages};
        let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[10, 20]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let instructions = generate_kv_commit_instructions(&batch, |_, accepted, rejected| {
            let draft = accepted + rejected;
            SpeculativePages {
                commit_pages: (0..draft as u64).collect(),
                rollback_pages: if accepted == 0 && rejected > 0 { (0..rejected as u64).collect() } else { Vec::new() },
            }
        });
        assert_eq!(instructions.len(), 1);
        assert!(matches!(instructions[0], KvCommitInstruction::Commit { .. }));
    }

    // --- generate_kv_commit_instructions for single all-rejected ---

    #[test]
    fn test_generate_kv_instructions_single_all_rejected() {
        use super::super::verify::{generate_kv_commit_instructions, KvCommitInstruction, SpeculativePages};
        let r = SequenceVerifyResult::verify_spine(1, &[10, 20], &[99, 99]);
        let batch = VerifyResult::from_sequence_results(vec![r]);
        let instructions = generate_kv_commit_instructions(&batch, |_, accepted, rejected| {
            let draft = accepted + rejected;
            SpeculativePages {
                commit_pages: (0..draft as u64).collect(),
                rollback_pages: if accepted == 0 && rejected > 0 { (0..rejected as u64).collect() } else { Vec::new() },
            }
        });
        assert_eq!(instructions.len(), 1);
        assert!(matches!(instructions[0], KvCommitInstruction::Rollback { .. }));
    }

    // --- FallbackStrategy: all variants distinct ---

    #[test]
    fn test_fallback_strategy_all_variants() {
        use super::super::cache::FallbackStrategy;
        let variants = [FallbackStrategy::SlowDraft, FallbackStrategy::FastNgram];
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    // --- FallbackStrategy: Hash and Eq in HashSet ---

    #[test]
    fn test_fallback_strategy_hashset() {
        use super::super::cache::FallbackStrategy;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(FallbackStrategy::SlowDraft);
        set.insert(FallbackStrategy::FastNgram);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&FallbackStrategy::SlowDraft));
        assert!(set.contains(&FallbackStrategy::FastNgram));
    }

    // --- EqSpecCheckResult: all false still has all_passed false ---

    #[test]
    fn test_eq_spec_check_result_all_false() {
        use super::super::verify::EqSpecCheckResult;
        let result = EqSpecCheckResult {
            i1_topology: false,
            i2_single_verification: false,
            i3_atomic_kv_commit: false,
        };
        assert!(!result.all_passed());
    }

    // --- Debug format for SpecDecodingState includes mode field ---

    #[test]
    fn test_debug_format_saguaro_state() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_saguaro(
            32, adapter, SpecTreeConfig::default(), 0, vec![1],
        );
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("Saguaro"));
    }

    // --- verify_phase: batch with multiple sequences ---

    #[test]
    fn test_verify_phase_multiple_sequences() {
        let adapter = make_test_adapter(64, 1000);
        let mut state = SpecDecodingState::new_eesd(32, adapter, SpecTreeConfig::default());

        let r1 = SequenceVerifyResult::verify_spine(1, &[10, 20], &[10, 20]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[30, 40], &[30, 99]);
        let verify = VerifyResult::from_sequence_results(vec![r1, r2]);
        state.verify_phase(&verify);

        assert_eq!(state.spec_step_count(), 1);
        // 3 accepted / 4 drafted = 0.75
        assert!((state.avg_acceptance_rate() - 0.75).abs() < 1e-5);
    }

    // --- temp_buffer_bytes: symmetry with kv_dim and dtype_size ---

    #[test]
    fn test_temp_buffer_bytes_symmetry() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(12, adapter, SpecTreeConfig::default());
        let a = state.temp_buffer_bytes(512, 4);
        let b = state.temp_buffer_bytes(4, 512);
        assert_eq!(a, b); // commutative multiplication
    }

    // =========================================================================
    // 15 NEW TESTS — additional edge case coverage
    // =========================================================================

    // --- mtp_candidates: empty input returns empty ---

    #[test]
    fn test_mtp_candidates_empty_input() {
        let candidates = mtp::mtp_candidates(&[]);
        assert!(candidates.is_empty());
    }

    // --- mtp_candidates: single logit vector picks argmax ---

    #[test]
    fn test_mtp_candidates_single_vector_argmax() {
        let logits = vec![vec![0.1, 0.9, 0.3]];
        let candidates = mtp::mtp_candidates(&logits);
        assert_eq!(candidates, vec![1]);
    }

    // --- mtp_candidates: multiple vectors each pick independent argmax ---

    #[test]
    fn test_mtp_candidates_multi_vector_independent() {
        let logits = vec![vec![0.5, 0.1], vec![0.1, 0.9]];
        let candidates = mtp::mtp_candidates(&logits);
        assert_eq!(candidates, vec![0, 1]);
    }

    // --- cache_aware_sample: returns same length as input when no cache hit ---

    #[test]
    fn test_cache_aware_sample_no_hit_preserves_length() {
        let cache = SpeculationCache::new(4, 100);
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let adjusted = cache.cache_aware_sample(&logits, 999);
        assert_eq!(adjusted.len(), logits.len());
    }

    // --- cache_aware_sample: modifies logits on cache hit ---

    #[test]
    fn test_cache_aware_sample_hit_modifies_logits() {
        let mut cache = SpeculationCache::new(4, 100);
        let entry = super::super::cache::CacheEntry {
            prefix_hash: 42,
            position: 0,
            candidates: vec![0],
            logits: vec![5.0],
            accept_count: 0,
            total_count: 0,
        };
        cache.insert(entry);
        let logits = vec![10.0, 5.0];
        let adjusted = cache.cache_aware_sample(&logits, 42);
        assert_eq!(adjusted.len(), 2);
        assert!(adjusted[0] != logits[0], "cached token should be scaled");
    }

    // --- set_batch_size + fallback_strategy interaction ---

    #[test]
    fn test_set_batch_size_switches_fallback_strategy() {
        let mut cache = SpeculationCache::new(4, 100);
        assert_eq!(cache.fallback_strategy(), super::super::cache::FallbackStrategy::SlowDraft);
        cache.set_batch_size(4);
        assert_eq!(cache.fallback_strategy(), super::super::cache::FallbackStrategy::FastNgram);
    }

    // --- VerifyResult: check_topology_invariant with matching lengths passes ---

    #[test]
    fn test_verify_result_topology_invariant_exact_match() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[10], &[10]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[20, 30], &[20, 99]);
        let verify = VerifyResult::from_sequence_results(vec![r1, r2]);
        assert!(verify.check_topology_invariant(&[1, 2]));
    }

    // --- VerifyResult: from_sequence_results with mixed acceptance rates ---

    #[test]
    fn test_verify_result_mixed_acceptance_rates() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[10, 20, 30], &[10, 20, 30]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[40, 50], &[99]);
        let verify = VerifyResult::from_sequence_results(vec![r1, r2]);
        // 3 accepted + 0 accepted = 3 / (3+2) = 0.6
        assert!((verify.avg_acceptance_rate - 0.6).abs() < 1e-5);
        assert_eq!(verify.total_accepted_tokens, 3);
        assert_eq!(verify.total_draft_tokens, 5);
    }

    // --- SequenceVerifyResult: verify_spine target longer than draft ---

    #[test]
    fn test_sequence_verify_result_target_longer_than_draft() {
        let r = SequenceVerifyResult::verify_spine(1, &[10], &[10, 20, 30]);
        assert_eq!(r.accepted_count, 1);
        assert_eq!(r.draft_count, 1);
        assert!((r.acceptance_rate - 1.0).abs() < 1e-5);
    }

    // --- SpeculationCache: hit_rate after insert with non-zero counts ---

    #[test]
    fn test_speculation_cache_hit_rate_with_counts() {
        let mut cache = SpeculationCache::new(4, 100);
        let entry = super::super::cache::CacheEntry {
            prefix_hash: 1,
            position: 0,
            candidates: vec![10],
            logits: vec![1.0],
            accept_count: 7,
            total_count: 10,
        };
        cache.insert(entry);
        assert!((cache.hit_rate() - 0.7).abs() < 1e-5);
    }

    // --- NgramIndex: get_continuations with n=1 and matching token ---

    #[test]
    fn test_ngram_index_unigram_continuations() {
        let tokens = vec![5, 10, 5, 20, 5, 30];
        let idx = NgramIndex::build(&tokens, 1);
        let cont = idx.get_continuations(5, 10);
        assert!(!cont.is_empty(), "should find continuations after token 5");
    }

    // --- AdapterConfig: Clone produces equal fields ---

    #[test]
    fn test_adapter_config_clone_equal() {
        let mut cfg = AdapterConfig::default();
        cfg.hidden_size = 256;
        cfg.vocab_size = 32000;
        let cloned = cfg.clone();
        assert_eq!(cfg.hidden_size, cloned.hidden_size);
        assert_eq!(cfg.vocab_size, cloned.vocab_size);
        assert_eq!(cfg.rms_norm_eps, cloned.rms_norm_eps);
        assert_eq!(cfg.enable_distillation, cloned.enable_distillation);
    }

    // --- VerifyResult: result_for returns correct per-sequence data ---

    #[test]
    fn test_verify_result_result_for_correct_data() {
        let r1 = SequenceVerifyResult::verify_spine(1, &[10], &[10]);
        let r2 = SequenceVerifyResult::verify_spine(2, &[20, 30], &[20, 99]);
        let verify = VerifyResult::from_sequence_results(vec![r1.clone(), r2.clone()]);
        let found1 = verify.result_for(1).unwrap();
        assert_eq!(found1.accepted_count, r1.accepted_count);
        let found2 = verify.result_for(2).unwrap();
        assert_eq!(found2.accepted_count, r2.accepted_count);
    }

    // --- SpecNode: Debug trait output ---

    #[test]
    fn test_spec_node_debug_output() {
        let node = super::super::tree::SpecNode {
            node_id: 0,
            token_id: 42,
            parent_id: None,
            children: vec![1, 2],
            source: super::super::tree::DraftSource::PldSpine,
            estimated_acceptance: 0.7,
            position_offset: 3,
        };
        let debug = format!("{:?}", node);
        assert!(debug.contains("token_id"));
        assert!(debug.contains("42"));
        assert!(debug.contains("PldSpine"));
    }

    // --- temp_buffer_bytes: large kv_dim does not overflow usize ---

    #[test]
    fn test_temp_buffer_bytes_large_kv_dim() {
        let adapter = make_test_adapter(64, 1000);
        let state = SpecDecodingState::new_eesd(1, adapter, SpecTreeConfig::default());
        // max_tree=32, total_layers=1, K+V=2, kv_dim=4096, dtype=4
        let bytes = state.temp_buffer_bytes(4096, 4);
        assert_eq!(bytes, 32 * 1 * 2 * 4096 * 4);
        assert!(bytes > 0);
    }
}
