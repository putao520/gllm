//! SemanticGatekeeperCallback — runtime state + mega-kernel bridge.
//!
//! 协议 SSOT: `SPEC/SEMANTIC-GATEKEEPER.md` §2.1 §5 §7;
//! 接入点: `SPEC/05-OPTIMIZATIONS.md §2.9`, 优先级 90.
//!
//! SG 已迁移到 Mega-Kernel JIT SgDetect/SgInject OpKind;
//! callback 检测/注入路径已删除, 由 JIT 共享内存驱动.

use std::sync::{Arc, RwLock};

#[cfg(test)]
use gllm_kernels::types::DType;
#[cfg(test)]
use half::{bf16, f16};

use crate::graph::layer_callback::LayerCallback;

use super::{
    active_state::ActiveState, level_keys::LevelKeysCache, ring_buffer::GatekeeperRingBuffer,
    AstSentinel, KnowledgeProvider, RetrieveContext, SemanticLevel, TokenizerLookup,
};

/// Priority for Semantic Gatekeeper in the CallbackChain
/// (`SPEC/05-OPTIMIZATIONS.md §8`).
pub const SEMANTIC_GATEKEEPER_PRIORITY: u32 = 90;

/// `KnowledgeProvider` 返回文本后由此 trait 编码为 `v_knowledge` 向量.
///
/// Phase D 的 `small_graph` + `LevelKeysCache::precompute` 会提供一个
/// 默认实现 (小 CompilerGraph 走 mega-kernel path 编码, ARCH-FULL-JIT 合规).
/// 本文件用 trait 而非具体类型以便单元测试注入 mock.
pub trait TextEncoder: Send + Sync {
    /// 将文本编码为 `hidden_size` 维向量.
    ///
    /// 约定: 使用主模型冻结 Token Embedding + mean-pool over seq,
    /// 确保输出与 hidden_state 处于同一语义空间 (SPEC §8.6).
    fn encode(&self, text: &str) -> Result<Vec<f32>, TextEncoderError>;
}

#[derive(Debug, thiserror::Error, Clone)]
pub enum TextEncoderError {
    #[error("tokenize failed: {0}")]
    Tokenize(String),
    #[error("graph execute failed: {0}")]
    Execute(String),
    #[error("encoder not initialized")]
    Uninitialized,
}

/// SG 运行时主体.
///
/// 注册到 `CallbackChain` 后由 mega-kernel node loop
/// 在检测层通过 JIT SgDetect/SgInject OpKind 触发.
#[allow(dead_code)]
pub struct SemanticGatekeeperCallback {
    pub(super) level_keys: Arc<LevelKeysCache>,
    pub(super) ring_buffer: Arc<GatekeeperRingBuffer>,
    pub(super) active_state: RwLock<ActiveState>,
    pub(super) provider: Arc<dyn KnowledgeProvider>,
    pub(super) ast_sentinel: Option<Arc<dyn AstSentinel>>,
    pub(super) text_encoder: Arc<dyn TextEncoder>,
    pub(super) tokenizer: Arc<dyn TokenizerLookup>,

    pub(super) gate_threshold: f32,
    pub(super) stability_threshold: f32,
    pub(super) alpha: f32,
    pub(super) hidden_size: usize,
}

impl SemanticGatekeeperCallback {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        level_keys: Arc<LevelKeysCache>,
        ring_buffer: Arc<GatekeeperRingBuffer>,
        provider: Arc<dyn KnowledgeProvider>,
        ast_sentinel: Option<Arc<dyn AstSentinel>>,
        text_encoder: Arc<dyn TextEncoder>,
        tokenizer: Arc<dyn TokenizerLookup>,
        gate_threshold: f32,
        stability_threshold: f32,
        alpha: f32,
        hidden_size: usize,
    ) -> Self {
        Self {
            level_keys,
            ring_buffer,
            active_state: RwLock::new(ActiveState::default()),
            provider,
            ast_sentinel,
            text_encoder,
            tokenizer,
            gate_threshold,
            stability_threshold,
            alpha,
            hidden_size,
        }
    }

    /// 清空 ActiveState (用户显式重置).
    pub fn reset_state(&self) {
        if let Ok(mut guard) = self.active_state.write() {
            guard.clear();
        }
    }

    /// 当前注册的检测层物理索引集合.
    pub fn detection_layers(&self) -> &[usize] {
        self.level_keys.detection_layers()
    }

    /// SG injection strength alpha.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Call KnowledgeProvider.retrieve and return confidence only.
    /// Used by mega-kernel callback to avoid String crossing NativeCall boundary.
    pub fn retrieve_confidence(&self, detect_hidden: &[f32]) -> Option<f32> {
        let ctx = RetrieveContext {
            generated_tokens: &[],
            ast: None,
            step: 0,
            request_id: 0,
        };
        self.provider.retrieve(detect_hidden, SemanticLevel::L1, &ctx)
            .map(|e| e.confidence)
    }

    /// Encode a single token via TextEncoder (bypasses tokenizer).
    /// Isolated to allow testing nested JIT call separately.
    pub fn encode_single(&self, token_id: u32) -> Option<Vec<f32>> {
        self.text_encoder.encode(&token_id.to_string()).ok()
    }

    /// Direct access to text_encoder for nested JIT debugging.
    pub fn text_encoder(&self) -> &Arc<dyn TextEncoder> {
        &self.text_encoder
    }

    /// Mega-kernel callback bridge: detect_hidden → KnowledgeProvider → TextEncoder
    /// → (knowledge_vector, confidence).
    ///
    /// Called from within the JIT mega-kernel via callback table slot 0
    /// (C ABI → `sg_knowledge_retrieve_callback` → this method).
    ///
    /// Returns `None` if the provider declines or encoding fails.
    pub fn retrieve_for_mega_kernel(
        &self,
        detect_hidden: &[f32],
    ) -> Option<(Vec<f32>, f32)> {
        let ctx = RetrieveContext {
            generated_tokens: &[],
            ast: None,
            step: 0,
            request_id: 0,
        };
        let entry = self.provider.retrieve(detect_hidden, SemanticLevel::L1, &ctx)?;
        let knowledge = self.text_encoder.encode(&entry.text).ok()?;
        // Effective confidence = entry.confidence × alpha.
        Some((knowledge, entry.confidence * self.alpha))
    }

    /// Same as retrieve_for_mega_kernel but uses detect_hidden directly as
    /// knowledge_vector (identity injection). Bypasses TextEncoder JIT graph
    /// to avoid nested JIT call issues.
    ///
    /// Returns `None` if the provider declines.
    pub fn retrieve_identity_for_mega_kernel(
        &self,
        detect_hidden: &[f32],
    ) -> Option<(Vec<f32>, f32)> {
        let ctx = RetrieveContext {
            generated_tokens: &[],
            ast: None,
            step: 0,
            request_id: 0,
        };
        let entry = self.provider.retrieve(detect_hidden, SemanticLevel::L1, &ctx)?;
        // Use detect_hidden as knowledge direction (identity injection).
        // The JIT SgInject will push hidden += confidence × alpha × detect_hidden,
        // which amplifies existing hidden direction by confidence × alpha.
        let knowledge = detect_hidden.to_vec();
        Some((knowledge, entry.confidence * self.alpha))
    }

    /// 当前检测层是否登记 (快速查找).
    #[allow(dead_code)]
    fn is_detection_layer(&self, layer_idx: usize) -> bool {
        self.detection_layers().contains(&layer_idx)
    }
}

// ============================================================================
// LayerCallback 实现
// ============================================================================

impl LayerCallback for SemanticGatekeeperCallback {
    fn priority(&self) -> u32 {
        SEMANTIC_GATEKEEPER_PRIORITY
    }

    fn target_layers(&self) -> Option<&[usize]> {
        Some(self.detection_layers())
    }

    fn name(&self) -> &str {
        "SemanticGatekeeper"
    }
}

// ============================================================================
// 私有状态更新载荷
// ============================================================================

#[allow(dead_code)]
struct ActiveStateUpdate {
    level: Option<SemanticLevel>,
    key_hash: Option<u64>,
    anchor_hidden: Option<Vec<f32>>,
    v_knowledge: Option<Vec<f32>>,
    ast_node_kind: Option<String>,
    last_step: u64,
}

// ============================================================================
// 数学辅助 (test-only)
// ============================================================================

#[cfg(test)]
/// 余弦相似度. 两向量需长度相等;对零向量返回 0.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

#[cfg(test)]
fn argmax3(scores: [f32; 3]) -> (usize, f32) {
    let mut best = 0usize;
    let mut val = scores[0];
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > val {
            val = s;
            best = i;
        }
    }
    (best, val)
}

#[cfg(test)]
#[allow(dead_code)]
fn hash_text(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

// ============================================================================
// dtype-aware bytes ↔ f32 转换
// ============================================================================

#[cfg(test)]
#[allow(dead_code)]
fn extract_last_token_hidden(
    hidden_bytes: &[u8],
    hidden_size: usize,
    dtype: DType,
) -> Result<Vec<f32>, ExtractError> {
    let elem_bytes = dtype.size_bytes();
    let row_bytes = hidden_size.checked_mul(elem_bytes).ok_or(ExtractError::Overflow)?;
    if row_bytes == 0 || hidden_bytes.len() < row_bytes {
        return Err(ExtractError::Truncated);
    }
    let last_start = hidden_bytes.len() - row_bytes;
    let row = &hidden_bytes[last_start..];
    decode_row(row, hidden_size, dtype)
}

#[cfg(test)]
fn decode_row(row: &[u8], hidden_size: usize, dtype: DType) -> Result<Vec<f32>, ExtractError> {
    let elem_bytes = dtype.size_bytes();
    if row.len() != hidden_size * elem_bytes {
        return Err(ExtractError::Truncated);
    }
    let mut out = Vec::with_capacity(hidden_size);
    match dtype {
        DType::F32 => {
            for i in 0..hidden_size {
                let off = i * 4;
                out.push(f32::from_le_bytes([
                    row[off],
                    row[off + 1],
                    row[off + 2],
                    row[off + 3],
                ]));
            }
        }
        DType::F16 => {
            for i in 0..hidden_size {
                let off = i * 2;
                out.push(f16::from_le_bytes([row[off], row[off + 1]]).to_f32());
            }
        }
        DType::BF16 => {
            for i in 0..hidden_size {
                let off = i * 2;
                out.push(bf16::from_le_bytes([row[off], row[off + 1]]).to_f32());
            }
        }
        _ => return Err(ExtractError::UnsupportedDtype(format!("{dtype:?}"))),
    }
    Ok(out)
}

#[cfg(test)]
fn encode_row(values: &[f32], dtype: DType) -> Vec<u8> {
    let elem_bytes = dtype.size_bytes();
    let mut out = Vec::with_capacity(values.len() * elem_bytes);
    match dtype {
        DType::F32 => {
            for &v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        DType::F16 => {
            for &v in values {
                out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
            }
        }
        DType::BF16 => {
            for &v in values {
                out.extend_from_slice(&bf16::from_f32(v).to_le_bytes());
            }
        }
        _ => {
            for &v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    out
}

#[cfg(test)]
/// 在 `hidden_bytes` 上只修改最后一个 token 行为 `h + alpha * delta`.
fn build_injected_hidden(
    hidden_bytes: &[u8],
    delta: &[f32],
    alpha: f32,
    hidden_size: usize,
    dtype: DType,
) -> Vec<u8> {
    let elem_bytes = dtype.size_bytes();
    let row_bytes = hidden_size * elem_bytes;
    let mut out = hidden_bytes.to_vec();
    if out.len() < row_bytes {
        return out;
    }
    let last_start = out.len() - row_bytes;
    let last_row = &out[last_start..];
    let mut last_f32 = match decode_row(last_row, hidden_size, dtype) {
        Ok(v) => v,
        Err(_) => return out,
    };
    for (i, d) in delta.iter().enumerate().take(hidden_size) {
        last_f32[i] += alpha * *d;
    }
    let encoded = encode_row(&last_f32, dtype);
    out[last_start..].copy_from_slice(&encoded);
    out
}

#[cfg(test)]
#[derive(Debug, thiserror::Error, Clone)]
enum ExtractError {
    #[error("hidden_bytes truncated or size mismatch")]
    Truncated,
    #[error("row_bytes overflow")]
    #[allow(dead_code)]
    Overflow,
    #[error("unsupported dtype {0}")]
    UnsupportedDtype(String),
}

// ============================================================================
// SemanticGatekeeperCallbackShim — Arc wrapper for LayerCallback
// ============================================================================

/// Thin wrapper that allows `Arc<Mutex<SemanticGatekeeperCallback>>` to be used as
/// `Box<dyn LayerCallback + Send>` in the per-forward `CallbackChain`.
///
/// Uses `std::sync::Mutex` to provide `&mut self` access through `Arc`.
pub struct SemanticGatekeeperCallbackShim {
    pub inner: Arc<std::sync::Mutex<SemanticGatekeeperCallback>>,
    pub hidden_size: usize,
}

impl LayerCallback for SemanticGatekeeperCallbackShim {
    fn priority(&self) -> u32 {
        SEMANTIC_GATEKEEPER_PRIORITY
    }

    fn target_layers(&self) -> Option<&[usize]> {
        match self.inner.lock() {
            Ok(cb) => Some(cb.detection_layers().to_vec().leak() as &[usize]),
            Err(_) => None,
        }
    }

    fn name(&self) -> &str {
        "SemanticGatekeeper"
    }
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use gllm_kernels::types::DType;

    #[test]
    fn cosine_identical_vectors_is_one() {
        let a = vec![1.0f32, 2.0, 3.0];
        let c = cosine(&a, &a);
        assert!((c - 1.0).abs() < 1e-5, "cosine(a,a) = {c}");
    }

    #[test]
    fn cosine_zero_vector_is_zero() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 2.0, 3.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }

    #[test]
    fn argmax3_picks_max() {
        assert_eq!(argmax3([0.1, 0.9, 0.5]).0, 1);
        assert_eq!(argmax3([0.9, 0.1, 0.5]).0, 0);
        assert_eq!(argmax3([0.1, 0.5, 0.9]).0, 2);
    }

    #[test]
    fn roundtrip_f32_decode_encode() {
        let values = vec![1.0f32, -2.5, 3.14, 0.0];
        let bytes = encode_row(&values, DType::F32);
        let decoded = decode_row(&bytes, values.len(), DType::F32).unwrap();
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn roundtrip_bf16_decode_encode() {
        let values = vec![1.0f32, -2.5, 3.125, 0.0];
        let bytes = encode_row(&values, DType::BF16);
        let decoded = decode_row(&bytes, values.len(), DType::BF16).unwrap();
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.05, "bf16 roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn build_injected_hidden_modifies_only_last_token() {
        let hidden: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let delta = vec![0.0f32, 100.0, 0.0];
        let alpha = 0.5;
        let out = build_injected_hidden(&hidden, &delta, alpha, 3, DType::F32);
        let decoded: Vec<f32> = out
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded[0], 1.0);
        assert_eq!(decoded[1], 2.0);
        assert_eq!(decoded[2], 3.0);
        assert_eq!(decoded[3], 4.0);
        assert!((decoded[4] - (5.0 + 0.5 * 100.0)).abs() < 1e-5);
        assert_eq!(decoded[5], 6.0);
    }

    // ---- Helper mocks ----

    struct MockProvider;
    impl super::super::KnowledgeProvider for MockProvider {
        fn retrieve(
            &self,
            _query: &[f32],
            _level: super::super::SemanticLevel,
            _ctx: &super::super::RetrieveContext<'_>,
        ) -> Option<super::super::KnowledgeEntry> {
            Some(super::super::KnowledgeEntry {
                text: "struct Foo".into(),
                confidence: 0.85,
            })
        }
    }

    struct DecliningProvider;
    impl super::super::KnowledgeProvider for DecliningProvider {
        fn retrieve(
            &self,
            _query: &[f32],
            _level: super::super::SemanticLevel,
            _ctx: &super::super::RetrieveContext<'_>,
        ) -> Option<super::super::KnowledgeEntry> {
            None
        }
    }

    struct MockEncoder;
    impl TextEncoder for MockEncoder {
        fn encode(&self, text: &str) -> Result<Vec<f32>, TextEncoderError> {
            Ok(vec![text.len() as f32; 4])
        }
    }

    struct FailingEncoder;
    impl TextEncoder for FailingEncoder {
        fn encode(&self, _text: &str) -> Result<Vec<f32>, TextEncoderError> {
            Err(TextEncoderError::Uninitialized)
        }
    }

    struct MockTokenizer;
    impl super::super::TokenizerLookup for MockTokenizer {
        fn decode(&self, tokens: &[u32]) -> String {
            tokens
                .iter()
                .map(|t| char::from_u32(*t).unwrap_or('?'))
                .collect()
        }
    }

    fn make_callback() -> SemanticGatekeeperCallback {
        use super::super::{GatekeeperRingBuffer, LevelKeysCache};
        SemanticGatekeeperCallback::new(
            Arc::new(LevelKeysCache::new(4)),
            Arc::new(GatekeeperRingBuffer::new(4, 4)),
            Arc::new(MockProvider),
            None,
            Arc::new(MockEncoder),
            Arc::new(MockTokenizer),
            0.7,
            0.9,
            0.5,
            4,
        )
    }

    fn make_callback_with_declining() -> SemanticGatekeeperCallback {
        use super::super::{GatekeeperRingBuffer, LevelKeysCache};
        SemanticGatekeeperCallback::new(
            Arc::new(LevelKeysCache::new(4)),
            Arc::new(GatekeeperRingBuffer::new(4, 4)),
            Arc::new(DecliningProvider),
            None,
            Arc::new(MockEncoder),
            Arc::new(MockTokenizer),
            0.7,
            0.9,
            0.5,
            4,
        )
    }

    fn make_callback_with_failing_encoder() -> SemanticGatekeeperCallback {
        use super::super::{GatekeeperRingBuffer, LevelKeysCache};
        SemanticGatekeeperCallback::new(
            Arc::new(LevelKeysCache::new(4)),
            Arc::new(GatekeeperRingBuffer::new(4, 4)),
            Arc::new(MockProvider),
            None,
            Arc::new(FailingEncoder),
            Arc::new(MockTokenizer),
            0.7,
            0.9,
            0.5,
            4,
        )
    }

    // ---- SemanticGatekeeperCallback tests ----

    #[test]
    fn callback_alpha_returns_configured_value() {
        let cb = make_callback();
        assert!((cb.alpha() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn callback_detection_layers_empty_initially() {
        let cb = make_callback();
        assert!(cb.detection_layers().is_empty());
    }

    #[test]
    fn callback_reset_state_clears_active() {
        let cb = make_callback();
        {
            let mut state = cb.active_state.write().unwrap();
            state.level = Some(super::super::SemanticLevel::L1);
            state.last_step = 42;
        }
        cb.reset_state();
        let state = cb.active_state.read().unwrap();
        assert!(state.level.is_none());
        assert_eq!(state.last_step, 0);
    }

    #[test]
    fn callback_retrieve_confidence_returns_value() {
        let cb = make_callback();
        let hidden = vec![0.1f32, 0.2, 0.3, 0.4];
        let conf = cb.retrieve_confidence(&hidden);
        assert!(conf.is_some());
        assert!((conf.unwrap() - 0.85).abs() < 1e-5);
    }

    #[test]
    fn callback_retrieve_confidence_declining_returns_none() {
        let cb = make_callback_with_declining();
        let hidden = vec![0.1f32, 0.2, 0.3, 0.4];
        assert!(cb.retrieve_confidence(&hidden).is_none());
    }

    #[test]
    fn callback_encode_single_returns_vector() {
        let cb = make_callback();
        let result = cb.encode_single(42);
        assert!(result.is_some());
        let vec = result.unwrap();
        assert_eq!(vec.len(), 4);
        // MockEncoder returns vec![text.len(); 4], "42".len() == 2
        assert!((vec[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn callback_encode_single_failing_returns_none() {
        let cb = make_callback_with_failing_encoder();
        assert!(cb.encode_single(42).is_none());
    }

    #[test]
    fn callback_retrieve_for_mega_kernel_returns_tuple() {
        let cb = make_callback();
        let hidden = vec![0.1f32, 0.2, 0.3, 0.4];
        let result = cb.retrieve_for_mega_kernel(&hidden);
        assert!(result.is_some());
        let (knowledge, effective_conf) = result.unwrap();
        assert_eq!(knowledge.len(), 4);
        // confidence * alpha = 0.85 * 0.5 = 0.425
        assert!((effective_conf - 0.425).abs() < 1e-5);
    }

    #[test]
    fn callback_retrieve_for_mega_kernel_declining_returns_none() {
        let cb = make_callback_with_declining();
        let hidden = vec![0.1f32, 0.2, 0.3, 0.4];
        assert!(cb.retrieve_for_mega_kernel(&hidden).is_none());
    }

    #[test]
    fn callback_retrieve_for_mega_kernel_failing_encoder_returns_none() {
        let cb = make_callback_with_failing_encoder();
        let hidden = vec![0.1f32, 0.2, 0.3, 0.4];
        assert!(cb.retrieve_for_mega_kernel(&hidden).is_none());
    }

    #[test]
    fn callback_retrieve_identity_returns_hidden_as_knowledge() {
        let cb = make_callback();
        let hidden = vec![0.1f32, 0.2, 0.3, 0.4];
        let result = cb.retrieve_identity_for_mega_kernel(&hidden);
        assert!(result.is_some());
        let (knowledge, effective_conf) = result.unwrap();
        assert_eq!(knowledge, hidden);
        // confidence * alpha = 0.85 * 0.5 = 0.425
        assert!((effective_conf - 0.425).abs() < 1e-5);
    }

    #[test]
    fn callback_retrieve_identity_declining_returns_none() {
        let cb = make_callback_with_declining();
        let hidden = vec![0.1f32, 0.2, 0.3, 0.4];
        assert!(cb.retrieve_identity_for_mega_kernel(&hidden).is_none());
    }

    #[test]
    fn callback_text_encoder_returns_arc_ref() {
        let cb = make_callback();
        let _encoder = cb.text_encoder();
    }

    // ---- LayerCallback impl tests ----

    #[test]
    fn layer_callback_priority_is_90() {
        let cb = make_callback();
        assert_eq!(LayerCallback::priority(&cb), 90);
    }

    #[test]
    fn layer_callback_name() {
        let cb = make_callback();
        assert_eq!(LayerCallback::name(&cb), "SemanticGatekeeper");
    }

    #[test]
    fn layer_callback_target_layers_empty() {
        let cb = make_callback();
        assert_eq!(LayerCallback::target_layers(&cb), Some(&[][..]));
    }

    // ---- SemanticGatekeeperCallbackShim tests ----

    #[test]
    fn shim_priority_matches_callback() {
        let cb = make_callback();
        let shim = SemanticGatekeeperCallbackShim {
            inner: Arc::new(std::sync::Mutex::new(cb)),
            hidden_size: 4,
        };
        assert_eq!(LayerCallback::priority(&shim), 90);
    }

    #[test]
    fn shim_name_matches_callback() {
        let cb = make_callback();
        let shim = SemanticGatekeeperCallbackShim {
            inner: Arc::new(std::sync::Mutex::new(cb)),
            hidden_size: 4,
        };
        assert_eq!(LayerCallback::name(&shim), "SemanticGatekeeper");
    }

    #[test]
    fn shim_target_layers_empty_initially() {
        let cb = make_callback();
        let shim = SemanticGatekeeperCallbackShim {
            inner: Arc::new(std::sync::Mutex::new(cb)),
            hidden_size: 4,
        };
        let layers = LayerCallback::target_layers(&shim);
        assert!(layers.is_some());
        assert!(layers.unwrap().is_empty());
    }

    #[test]
    fn shim_hidden_size_stored() {
        let cb = make_callback();
        let shim = SemanticGatekeeperCallbackShim {
            inner: Arc::new(std::sync::Mutex::new(cb)),
            hidden_size: 128,
        };
        assert_eq!(shim.hidden_size, 128);
    }

    // ---- Additional dtype/encode/decode tests ----

    #[test]
    fn roundtrip_f16_decode_encode() {
        let values = vec![1.0f32, -2.5, 3.125, 0.0];
        let bytes = encode_row(&values, DType::F16);
        let decoded = decode_row(&bytes, values.len(), DType::F16).unwrap();
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 0.1, "f16 roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn decode_row_truncated_returns_error() {
        let row = vec![0u8; 7]; // < 4 * 4 bytes for hidden_size=4
        let result = decode_row(&row, 4, DType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn extract_last_token_hidden_works() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = extract_last_token_hidden(&bytes, 3, DType::F32);
        assert!(result.is_ok());
        let last = result.unwrap();
        // 6 f32 values = 2 tokens × hidden_size=3, last token = [4.0, 5.0, 6.0]
        assert_eq!(last, vec![4.0f32, 5.0, 6.0]);
    }

    #[test]
    fn extract_last_token_hidden_truncated() {
        let bytes = vec![0u8; 8]; // less than hidden_size * 4
        let result = extract_last_token_hidden(&bytes, 4, DType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn cosine_orthogonal_vectors_is_zero() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert!((cosine(&a, &b)).abs() < 1e-5);
    }

    #[test]
    fn cosine_empty_returns_zero() {
        assert_eq!(cosine(&[], &[]), 0.0);
    }

    #[test]
    fn cosine_mismatched_len_returns_zero() {
        assert_eq!(cosine(&[1.0f32], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn argmax3_tie_first_wins() {
        let (idx, val) = argmax3([0.9, 0.9, 0.5]);
        assert_eq!(idx, 0);
        assert!((val - 0.9).abs() < 1e-6);
    }

    // ---- TextEncoderError tests ----

    #[test]
    fn text_encoder_error_display_variants() {
        let e = TextEncoderError::Tokenize("bad input".into());
        assert!(e.to_string().contains("bad input"));

        let e = TextEncoderError::Execute("segfault".into());
        assert!(e.to_string().contains("segfault"));

        let e = TextEncoderError::Uninitialized;
        assert!(e.to_string().contains("not initialized"));
    }

    #[test]
    fn text_encoder_error_clone() {
        let e = TextEncoderError::Tokenize("oops".into());
        let e2 = e.clone();
        assert_eq!(e.to_string(), e2.to_string());
    }

    #[test]
    fn build_injected_hidden_too_short_returns_unchanged() {
        let hidden = vec![0u8; 4]; // < row_bytes for hidden_size=3
        let out = build_injected_hidden(&hidden, &[1.0, 2.0, 3.0], 0.5, 3, DType::F32);
        assert_eq!(out, hidden);
    }

    #[test]
    fn build_injected_hidden_zero_alpha_no_change() {
        let hidden: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let delta = vec![100.0f32, 200.0, 300.0];
        let out = build_injected_hidden(&hidden, &delta, 0.0, 3, DType::F32);
        assert_eq!(out, hidden);
    }

    // ---- New tests ----

    #[test]
    fn text_encoder_error_debug_trait() {
        let e = TextEncoderError::Tokenize("debug msg".into());
        let debug_str = format!("{e:?}");
        assert!(debug_str.contains("Tokenize"));
    }

    #[test]
    fn extract_error_display_truncated() {
        let e = ExtractError::Truncated;
        assert!(e.to_string().contains("truncated"));
    }

    #[test]
    fn extract_error_display_overflow() {
        let e = ExtractError::Overflow;
        assert!(e.to_string().contains("overflow"));
    }

    #[test]
    fn extract_error_display_unsupported_dtype() {
        let e = ExtractError::UnsupportedDtype("U8".into());
        assert!(e.to_string().contains("U8"));
    }

    #[test]
    fn extract_error_clone_preserves_message() {
        let e = ExtractError::UnsupportedDtype("INT8".into());
        let e2 = e.clone();
        assert_eq!(e.to_string(), e2.to_string());
    }

    #[test]
    fn extract_error_debug_format() {
        let e = ExtractError::Overflow;
        let debug_str = format!("{e:?}");
        assert!(debug_str.contains("Overflow"));
    }

    #[test]
    fn decode_row_unsupported_dtype_returns_error() {
        // DType::U8 has elem_bytes=1, so hidden_size=4 => need exactly 4 bytes.
        let row = vec![0u8; 4];
        let result = decode_row(&row, 4, DType::U8);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported dtype"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn encode_row_fallback_dtype_uses_f32_bytes() {
        let values = vec![1.0f32, 2.0];
        let bytes = encode_row(&values, DType::U8);
        let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(bytes, expected);
    }

    #[test]
    fn extract_last_token_hidden_zero_hidden_size_returns_error() {
        let bytes = vec![0u8; 16];
        let result = extract_last_token_hidden(&bytes, 0, DType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn extract_last_token_hidden_exactly_one_row() {
        let values = vec![10.0f32, 20.0, 30.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = extract_last_token_hidden(&bytes, 3, DType::F32);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), values);
    }

    #[test]
    fn build_injected_hidden_bf16_roundtrip() {
        let values = vec![1.0f32, 2.0, 3.0];
        let bytes = encode_row(&values, DType::BF16);
        let delta = vec![0.5f32, 0.5, 0.5];
        let alpha = 1.0;
        let out = build_injected_hidden(&bytes, &delta, alpha, 3, DType::BF16);
        let decoded = decode_row(&out, 3, DType::BF16).unwrap();
        for (got, expected) in decoded.iter().zip([1.5f32, 2.5, 3.5].iter()) {
            assert!(
                (got - expected).abs() < 0.1,
                "bf16 injection: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn build_injected_hidden_f16_roundtrip() {
        let values = vec![1.0f32, 2.0, 3.0];
        let bytes = encode_row(&values, DType::F16);
        let delta = vec![0.5f32, 0.5, 0.5];
        let alpha = 1.0;
        let out = build_injected_hidden(&bytes, &delta, alpha, 3, DType::F16);
        let decoded = decode_row(&out, 3, DType::F16).unwrap();
        for (got, expected) in decoded.iter().zip([1.5f32, 2.5, 3.5].iter()) {
            assert!(
                (got - expected).abs() < 0.1,
                "f16 injection: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn hash_text_deterministic() {
        let h1 = hash_text("hello");
        let h2 = hash_text("hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn hash_text_different_inputs_differ() {
        let h1 = hash_text("foo");
        let h2 = hash_text("bar");
        assert_ne!(h1, h2);
    }

    #[test]
    fn cosine_negative_vector() {
        let a = vec![-1.0f32, -2.0, -3.0];
        let b = vec![1.0f32, 2.0, 3.0];
        let c = cosine(&a, &b);
        assert!((c + 1.0).abs() < 1e-5, "opposite vectors: cosine = {c}");
    }

    #[test]
    fn argmax3_all_zero() {
        let (idx, val) = argmax3([0.0, 0.0, 0.0]);
        assert_eq!(idx, 0);
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn semantic_gatekeeper_priority_constant_is_90() {
        assert_eq!(SEMANTIC_GATEKEEPER_PRIORITY, 90);
    }

    #[test]
    fn callback_new_stores_gate_threshold() {
        let cb = make_callback();
        assert!((cb.gate_threshold - 0.7).abs() < 1e-6);
    }

    #[test]
    fn callback_new_stores_stability_threshold() {
        let cb = make_callback();
        assert!((cb.stability_threshold - 0.9).abs() < 1e-6);
    }

    #[test]
    fn callback_new_stores_hidden_size() {
        let cb = make_callback();
        assert_eq!(cb.hidden_size, 4);
    }

    #[test]
    fn callback_is_detection_layer_returns_false_for_unregistered() {
        let cb = make_callback();
        assert!(!cb.is_detection_layer(0));
        assert!(!cb.is_detection_layer(99));
    }

    #[test]
    fn callback_retrieve_identity_effective_confidence_with_zero_alpha() {
        use super::super::{GatekeeperRingBuffer, LevelKeysCache};
        let cb = SemanticGatekeeperCallback::new(
            Arc::new(LevelKeysCache::new(4)),
            Arc::new(GatekeeperRingBuffer::new(4, 4)),
            Arc::new(MockProvider),
            None,
            Arc::new(MockEncoder),
            Arc::new(MockTokenizer),
            0.7,
            0.9,
            0.0, // alpha = 0
            4,
        );
        let hidden = vec![0.1f32, 0.2, 0.3, 0.4];
        let result = cb.retrieve_identity_for_mega_kernel(&hidden);
        assert!(result.is_some());
        let (_, effective_conf) = result.unwrap();
        assert!((effective_conf - 0.0).abs() < 1e-6);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Additional tests — 42 new tests for comprehensive callback coverage
    // ════════════════════════════════════════════════════════════════════════

    // ── cosine: additional edge cases ──

    #[test]
    fn cosine_single_element_vectors() {
        let a = vec![3.0f32];
        let b = vec![4.0f32];
        let c = cosine(&a, &b);
        assert!(
            (c - 1.0).abs() < 1e-5,
            "same direction single-element: cosine = {c}"
        );
    }

    #[test]
    fn cosine_single_element_opposite() {
        let a = vec![1.0f32];
        let b = vec![-1.0f32];
        let c = cosine(&a, &b);
        assert!(
            (c + 1.0).abs() < 1e-5,
            "opposite single-element: cosine = {c}"
        );
    }

    #[test]
    fn cosine_both_zero_vectors() {
        let a = vec![0.0f32, 0.0];
        let b = vec![0.0f32, 0.0];
        assert_eq!(cosine(&a, &b), 0.0);
    }

    #[test]
    fn cosine_symmetry() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let ab = cosine(&a, &b);
        let ba = cosine(&b, &a);
        assert!(
            (ab - ba).abs() < 1e-6,
            "cosine should be symmetric: {ab} vs {ba}"
        );
    }

    #[test]
    fn cosine_large_values_no_overflow() {
        let a = vec![1e30f32, 1e30];
        let b = vec![1e30f32, 1e30];
        let c = cosine(&a, &b);
        assert!(
            (c - 1.0).abs() < 1e-3,
            "large same-direction vectors: cosine = {c}"
        );
    }

    #[test]
    fn cosine_45_degree_angle() {
        // cos(45deg) ~ 0.7071
        let a = vec![1.0f32, 0.0];
        let b = vec![1.0f32, 1.0];
        let c = cosine(&a, &b);
        assert!(
            (c - 0.7071).abs() < 0.01,
            "45 degree angle: cosine = {c}"
        );
    }

    // ── argmax3: additional edge cases ──

    #[test]
    fn argmax3_negative_values() {
        let (idx, val) = argmax3([-5.0, -1.0, -3.0]);
        assert_eq!(idx, 1);
        assert!((val - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn argmax3_single_dominant() {
        let (idx, val) = argmax3([0.001, 0.002, 100.0]);
        assert_eq!(idx, 2);
        assert!((val - 100.0).abs() < 1e-6);
    }

    #[test]
    fn argmax3_last_wins_tie() {
        // 0.9 at index 1 and 0.9 at index 2: first 0.9 wins (index 1)
        let (idx, _) = argmax3([0.5, 0.9, 0.9]);
        assert_eq!(idx, 1, "tie: first occurrence should win");
    }

    // ── decode_row / encode_row: additional edge cases ──

    #[test]
    fn decode_row_f32_zero_values() {
        let bytes = vec![0u8; 8]; // 2 f32 zeros
        let decoded = decode_row(&bytes, 2, DType::F32).unwrap();
        assert_eq!(decoded, vec![0.0f32, 0.0]);
    }

    #[test]
    fn decode_row_f32_negative_values() {
        let values = vec![-1.0f32, -100.5];
        let bytes = encode_row(&values, DType::F32);
        let decoded = decode_row(&bytes, 2, DType::F32).unwrap();
        assert!((decoded[0] - (-1.0)).abs() < 1e-6);
        assert!((decoded[1] - (-100.5)).abs() < 1e-6);
    }

    #[test]
    fn decode_row_single_element_f32() {
        let values = vec![42.5f32];
        let bytes = encode_row(&values, DType::F32);
        let decoded = decode_row(&bytes, 1, DType::F32).unwrap();
        assert!((decoded[0] - 42.5).abs() < 1e-6);
    }

    #[test]
    fn encode_row_f32_produces_correct_byte_count() {
        let values = vec![1.0f32, 2.0, 3.0];
        let bytes = encode_row(&values, DType::F32);
        assert_eq!(bytes.len(), 12); // 3 * 4 bytes
    }

    #[test]
    fn encode_row_f16_produces_correct_byte_count() {
        let values = vec![1.0f32, 2.0, 3.0];
        let bytes = encode_row(&values, DType::F16);
        assert_eq!(bytes.len(), 6); // 3 * 2 bytes
    }

    #[test]
    fn encode_row_bf16_produces_correct_byte_count() {
        let values = vec![1.0f32, 2.0, 3.0];
        let bytes = encode_row(&values, DType::BF16);
        assert_eq!(bytes.len(), 6); // 3 * 2 bytes
    }

    #[test]
    fn decode_row_exact_byte_match_required() {
        // Provide 1 extra byte — should fail
        let mut bytes = vec![0u8; 9]; // 9 bytes != 2 * 4 = 8
        let result = decode_row(&bytes, 2, DType::F32);
        assert!(result.is_err());
        // Now provide exactly correct bytes
        bytes.pop();
        let result = decode_row(&bytes, 2, DType::F32);
        assert!(result.is_ok());
    }

    // ── extract_last_token_hidden: additional edge cases ──

    #[test]
    fn extract_last_token_hidden_multiple_rows_extracts_last() {
        // 3 tokens * hidden_size=2
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = extract_last_token_hidden(&bytes, 2, DType::F32).unwrap();
        assert_eq!(result, vec![5.0f32, 6.0]);
    }

    #[test]
    fn extract_last_token_hidden_f16_roundtrip() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0]; // 2 tokens * hidden_size=2
        let bytes = encode_row(&values, DType::F16);
        let result = extract_last_token_hidden(&bytes, 2, DType::F16);
        assert!(result.is_ok());
        let last = result.unwrap();
        assert!((last[0] - 3.0).abs() < 0.1, "f16 last token: got {}", last[0]);
        assert!((last[1] - 4.0).abs() < 0.1, "f16 last token: got {}", last[1]);
    }

    #[test]
    fn extract_last_token_hidden_bf16_roundtrip() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0]; // 2 tokens * hidden_size=2
        let bytes = encode_row(&values, DType::BF16);
        let result = extract_last_token_hidden(&bytes, 2, DType::BF16);
        assert!(result.is_ok());
        let last = result.unwrap();
        assert!((last[0] - 3.0).abs() < 0.1, "bf16 last token: got {}", last[0]);
        assert!((last[1] - 4.0).abs() < 0.1, "bf16 last token: got {}", last[1]);
    }

    // ── build_injected_hidden: additional edge cases ──

    #[test]
    fn build_injected_hidden_negative_alpha() {
        let values = vec![1.0f32, 2.0, 3.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let delta = vec![1.0f32, 1.0, 1.0];
        let alpha = -0.5;
        let out = build_injected_hidden(&bytes, &delta, alpha, 3, DType::F32);
        let decoded: Vec<f32> = out
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert!((decoded[0] - (1.0 + (-0.5) * 1.0)).abs() < 1e-5);
        assert!((decoded[1] - (2.0 + (-0.5) * 1.0)).abs() < 1e-5);
        assert!((decoded[2] - (3.0 + (-0.5) * 1.0)).abs() < 1e-5);
    }

    #[test]
    fn build_injected_hidden_large_alpha() {
        let values = vec![1.0f32, 2.0, 3.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let delta = vec![0.1f32, 0.2, 0.3];
        let alpha = 10.0;
        let out = build_injected_hidden(&bytes, &delta, alpha, 3, DType::F32);
        let decoded: Vec<f32> = out
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert!((decoded[0] - (1.0 + 10.0 * 0.1)).abs() < 1e-4);
        assert!((decoded[1] - (2.0 + 10.0 * 0.2)).abs() < 1e-4);
        assert!((decoded[2] - (3.0 + 10.0 * 0.3)).abs() < 1e-4);
    }

    #[test]
    fn build_injected_hidden_preserves_first_tokens() {
        // 3 tokens * hidden_size=2 = 6 f32 values
        let values = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let delta = vec![100.0f32, 200.0];
        let alpha = 1.0;
        let out = build_injected_hidden(&bytes, &delta, alpha, 2, DType::F32);
        let decoded: Vec<f32> = out
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // First 2 tokens (4 values) unchanged
        assert_eq!(decoded[0], 10.0);
        assert_eq!(decoded[1], 20.0);
        assert_eq!(decoded[2], 30.0);
        assert_eq!(decoded[3], 40.0);
        // Last token modified
        assert!((decoded[4] - (50.0 + 100.0)).abs() < 1e-4);
        assert!((decoded[5] - (60.0 + 200.0)).abs() < 1e-4);
    }

    #[test]
    fn build_injected_hidden_delta_shorter_than_hidden_size() {
        let values = vec![1.0f32, 2.0, 3.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // delta has 2 elements but hidden_size=3: take() limits to hidden_size
        let delta = vec![10.0f32, 20.0];
        let alpha = 1.0;
        let out = build_injected_hidden(&bytes, &delta, alpha, 3, DType::F32);
        let decoded: Vec<f32> = out
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // Only first 2 elements of delta applied (take(3) but delta is shorter)
        assert!((decoded[0] - (1.0 + 10.0)).abs() < 1e-5);
        assert!((decoded[1] - (2.0 + 20.0)).abs() < 1e-5);
        assert!((decoded[2] - 3.0).abs() < 1e-5); // unchanged
    }

    // ── SemanticGatekeeperCallback: additional state & retrieval tests ──

    #[test]
    fn callback_retrieve_confidence_returns_provider_confidence() {
        // MockProvider returns confidence 0.85
        let cb = make_callback();
        let hidden = vec![0.0f32; 4];
        let conf = cb.retrieve_confidence(&hidden).unwrap();
        assert!(
            (conf - 0.85).abs() < 1e-5,
            "expected 0.85, got {conf}"
        );
    }

    #[test]
    fn callback_retrieve_for_mega_kernel_knowledge_matches_encoder() {
        let cb = make_callback();
        let hidden = vec![0.5f32; 4];
        let (knowledge, _) = cb.retrieve_for_mega_kernel(&hidden).unwrap();
        // MockEncoder returns vec![text.len(); 4], "struct Foo".len() == 10
        assert_eq!(knowledge.len(), 4);
        assert!((knowledge[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn callback_retrieve_identity_preserves_hidden_length() {
        let cb = make_callback();
        let hidden = vec![0.1f32, 0.2, 0.3, 0.4];
        let (knowledge, _) = cb.retrieve_identity_for_mega_kernel(&hidden).unwrap();
        assert_eq!(knowledge.len(), hidden.len());
    }

    #[test]
    fn callback_encode_single_uses_text_encoder() {
        let cb = make_callback();
        // token_id=42 → "42" → MockEncoder returns vec![2; 4] (len("42")=2)
        let vec = cb.encode_single(42).unwrap();
        assert_eq!(vec.len(), 4);
        assert!((vec[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn callback_encode_single_token_id_zero() {
        let cb = make_callback();
        // "0".len() == 1
        let vec = cb.encode_single(0).unwrap();
        assert!((vec[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn callback_encode_single_large_token_id() {
        let cb = make_callback();
        let large_id = 99999u32;
        let vec = cb.encode_single(large_id).unwrap();
        // "99999".len() == 5
        assert!((vec[0] - 5.0).abs() < 1e-5);
    }

    // ── Callback with custom alpha and gate_threshold ──

    #[test]
    fn callback_custom_alpha_affects_effective_confidence() {
        use super::super::{GatekeeperRingBuffer, LevelKeysCache};
        let cb = SemanticGatekeeperCallback::new(
            Arc::new(LevelKeysCache::new(4)),
            Arc::new(GatekeeperRingBuffer::new(4, 4)),
            Arc::new(MockProvider), // confidence = 0.85
            None,
            Arc::new(MockEncoder),
            Arc::new(MockTokenizer),
            0.7,
            0.9,
            0.2, // alpha = 0.2
            4,
        );
        let (_, effective_conf) = cb.retrieve_for_mega_kernel(&[0.0; 4]).unwrap();
        assert!(
            (effective_conf - 0.85 * 0.2).abs() < 1e-5,
            "expected {}, got {}",
            0.85 * 0.2,
            effective_conf
        );
    }

    #[test]
    fn callback_alpha_one_passthrough() {
        use super::super::{GatekeeperRingBuffer, LevelKeysCache};
        let cb = SemanticGatekeeperCallback::new(
            Arc::new(LevelKeysCache::new(4)),
            Arc::new(GatekeeperRingBuffer::new(4, 4)),
            Arc::new(MockProvider),
            None,
            Arc::new(MockEncoder),
            Arc::new(MockTokenizer),
            0.7,
            0.9,
            1.0, // alpha = 1.0
            4,
        );
        let (_, effective_conf) = cb.retrieve_for_mega_kernel(&[0.0; 4]).unwrap();
        assert!(
            (effective_conf - 0.85).abs() < 1e-5,
            "alpha=1.0 should pass through confidence: {effective_conf}"
        );
    }

    // ── SemanticGatekeeperCallback: reset_state edge cases ──

    #[test]
    fn callback_reset_state_double_reset_is_idempotent() {
        let cb = make_callback();
        {
            let mut state = cb.active_state.write().unwrap();
            state.last_step = 100;
        }
        cb.reset_state();
        cb.reset_state(); // second reset
        let state = cb.active_state.read().unwrap();
        assert!(state.level.is_none());
        assert_eq!(state.last_step, 0);
    }

    #[test]
    fn callback_reset_state_clears_anchor_hidden() {
        let cb = make_callback();
        {
            let mut state = cb.active_state.write().unwrap();
            state.anchor_hidden = Some(vec![1.0, 2.0, 3.0, 4.0]);
        }
        cb.reset_state();
        let state = cb.active_state.read().unwrap();
        assert!(state.anchor_hidden.is_none());
    }

    #[test]
    fn callback_reset_state_clears_v_knowledge() {
        let cb = make_callback();
        {
            let mut state = cb.active_state.write().unwrap();
            state.v_knowledge = Some(vec![5.0, 6.0, 7.0, 8.0]);
        }
        cb.reset_state();
        let state = cb.active_state.read().unwrap();
        assert!(state.v_knowledge.is_none());
    }

    // ── SemanticGatekeeperCallbackShim: additional tests ──

    #[test]
    fn shim_target_layers_returns_sorted_layers() {
        use super::super::LevelKeysCache;
        let mut lk = LevelKeysCache::new(4);
        let keys: [Vec<f32>; 3] = [
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, 2.0, 3.0, 4.0],
        ];
        lk.insert(10, keys.clone()).unwrap();
        lk.insert(5, keys.clone()).unwrap();
        lk.insert(15, keys).unwrap();
        let cb = SemanticGatekeeperCallback::new(
            Arc::new(lk),
            Arc::new(super::super::GatekeeperRingBuffer::new(4, 4)),
            Arc::new(MockProvider),
            None,
            Arc::new(MockEncoder),
            Arc::new(MockTokenizer),
            0.7,
            0.9,
            0.5,
            4,
        );
        let shim = SemanticGatekeeperCallbackShim {
            inner: Arc::new(std::sync::Mutex::new(cb)),
            hidden_size: 4,
        };
        let layers = LayerCallback::target_layers(&shim).unwrap();
        assert_eq!(layers, &[5, 10, 15]);
    }

    #[test]
    fn shim_wraps_callback_methods() {
        let cb = make_callback();
        let shim = SemanticGatekeeperCallbackShim {
            inner: Arc::new(std::sync::Mutex::new(cb)),
            hidden_size: 4,
        };
        // Verify shim delegates to inner callback
        let inner = shim.inner.lock().unwrap();
        assert!((inner.alpha() - 0.5).abs() < 1e-6);
        assert_eq!(inner.hidden_size, 4);
    }

    // ── TextEncoderError: additional trait coverage ──

    #[test]
    fn text_encoder_error_all_variants_clone() {
        let errors = vec![
            TextEncoderError::Tokenize("t".into()),
            TextEncoderError::Execute("e".into()),
            TextEncoderError::Uninitialized,
        ];
        for e in &errors {
            let cloned = e.clone();
            assert_eq!(e.to_string(), cloned.to_string());
        }
    }

    #[test]
    fn text_encoder_error_tokenize_debug() {
        let e = TextEncoderError::Tokenize("bad".into());
        let debug = format!("{e:?}");
        assert!(debug.contains("Tokenize"), "Debug: {debug}");
    }

    #[test]
    fn text_encoder_error_execute_debug() {
        let e = TextEncoderError::Execute("fail".into());
        let debug = format!("{e:?}");
        assert!(debug.contains("Execute"), "Debug: {debug}");
    }

    // ── ExtractError: additional trait coverage ──

    #[test]
    fn extract_error_truncated_debug() {
        let e = ExtractError::Truncated;
        let debug = format!("{e:?}");
        assert!(debug.contains("Truncated"), "Debug: {debug}");
    }

    #[test]
    fn extract_error_unsupported_dtype_debug() {
        let e = ExtractError::UnsupportedDtype("I64".into());
        let debug = format!("{e:?}");
        assert!(debug.contains("UnsupportedDtype"), "Debug: {debug}");
    }

    #[test]
    fn extract_error_all_variants_clone() {
        let errors = vec![
            ExtractError::Truncated,
            ExtractError::Overflow,
            ExtractError::UnsupportedDtype("U8".into()),
        ];
        for e in &errors {
            let cloned = e.clone();
            assert_eq!(e.to_string(), cloned.to_string());
        }
    }

    // ── ActiveState interaction through callback ──

    #[test]
    fn callback_active_state_default_fields() {
        let cb = make_callback();
        let state = cb.active_state.read().unwrap();
        assert!(state.level.is_none());
        assert!(state.key_hash.is_none());
        assert!(state.anchor_hidden.is_none());
        assert!(state.v_knowledge.is_none());
        assert!(state.ast_node_kind.is_none());
        assert_eq!(state.last_step, 0);
        assert!(state.last_request.is_none());
    }

    #[test]
    fn callback_active_state_set_and_read() {
        let cb = make_callback();
        {
            let mut state = cb.active_state.write().unwrap();
            state.level = Some(super::super::SemanticLevel::L2);
            state.key_hash = Some(12345);
            state.last_step = 99;
        }
        let state = cb.active_state.read().unwrap();
        assert_eq!(state.level, Some(super::super::SemanticLevel::L2));
        assert_eq!(state.key_hash, Some(12345));
        assert_eq!(state.last_step, 99);
    }

    // ── hash_text: additional coverage ──

    #[test]
    fn hash_text_empty_string() {
        let h = hash_text("");
        // Deterministic: same empty input produces same hash
        assert_eq!(h, hash_text(""));
    }

    #[test]
    fn hash_text_unicode_input() {
        let h1 = hash_text("你好世界");
        let h2 = hash_text("你好世界");
        let h3 = hash_text("你好世办"); // one char different
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn hash_text_long_input() {
        let long = "a".repeat(10000);
        let h1 = hash_text(&long);
        let h2 = hash_text(&long);
        assert_eq!(h1, h2);
    }

    // ── encode_row / decode_row: roundtrip with all supported dtypes ──

    #[test]
    fn roundtrip_f32_preserves_special_values() {
        let values = vec![0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY];
        let bytes = encode_row(&values, DType::F32);
        let decoded = decode_row(&bytes, 4, DType::F32).unwrap();
        assert_eq!(decoded[0], 0.0f32);
        assert_eq!(decoded[1], -0.0f32);
        assert!(decoded[2].is_infinite() && decoded[2].is_sign_positive());
        assert!(decoded[3].is_infinite() && decoded[3].is_sign_negative());
    }

    #[test]
    fn encode_decode_roundtrip_preserves_vector_length() {
        let values = vec![1.0f32; 128];
        let bytes = encode_row(&values, DType::F32);
        let decoded = decode_row(&bytes, 128, DType::F32).unwrap();
        assert_eq!(decoded.len(), 128);
        for v in &decoded {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    // ── decode_row: boundary size ──

    #[test]
    fn decode_row_f32_exactly_one_element() {
        let bytes = 1.5f32.to_le_bytes().to_vec();
        let decoded = decode_row(&bytes, 1, DType::F32).unwrap();
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn decode_row_f16_exactly_one_element() {
        let bytes = half::f16::from_f32(2.5).to_le_bytes().to_vec();
        let decoded = decode_row(&bytes, 1, DType::F16).unwrap();
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - 2.5).abs() < 0.1);
    }

    // ── extract_last_token_hidden: unsupported dtype ──

    #[test]
    fn extract_last_token_hidden_unsupported_dtype_returns_error() {
        let bytes = vec![0u8; 4]; // DType::U8, 4 elements
        let result = extract_last_token_hidden(&bytes, 4, DType::U8);
        assert!(result.is_err());
    }

    // ── build_injected_hidden: exactly row_bytes input ──

    #[test]
    fn build_injected_hidden_exactly_one_row_modifies_all() {
        let values = vec![1.0f32, 2.0, 3.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let delta = vec![0.1f32, 0.2, 0.3];
        let alpha = 2.0;
        let out = build_injected_hidden(&bytes, &delta, alpha, 3, DType::F32);
        let decoded: Vec<f32> = out
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert!((decoded[0] - (1.0 + 2.0 * 0.1)).abs() < 1e-4);
        assert!((decoded[1] - (2.0 + 2.0 * 0.2)).abs() < 1e-4);
        assert!((decoded[2] - (3.0 + 2.0 * 0.3)).abs() < 1e-4);
    }

    // ════════════════════════════════════════════════════════════════════════
    // 13 additional tests — edge cases & uncovered callback paths
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn callback_retrieve_confidence_empty_hidden_slice() {
        // Arrange: provider receives a zero-length hidden slice
        let cb = make_callback();
        let hidden: Vec<f32> = vec![];

        // Act: MockProvider does not inspect query length, always returns Some
        let conf = cb.retrieve_confidence(&hidden);

        // Assert: confidence comes back from MockProvider
        assert!(conf.is_some());
        assert!((conf.unwrap() - 0.85).abs() < 1e-5);
    }

    #[test]
    fn callback_retrieve_for_mega_kernel_with_empty_hidden() {
        // Arrange
        let cb = make_callback();
        let hidden: Vec<f32> = vec![];

        // Act
        let result = cb.retrieve_for_mega_kernel(&hidden);

        // Assert: MockProvider + MockEncoder both succeed regardless of input length
        assert!(result.is_some());
        let (knowledge, effective_conf) = result.unwrap();
        assert_eq!(knowledge.len(), 4);
        assert!((effective_conf - 0.425).abs() < 1e-5);
    }

    #[test]
    fn callback_retrieve_identity_returns_cloned_hidden_not_reference() {
        // Arrange
        let cb = make_callback();
        let hidden = vec![0.1f32, 0.2, 0.3, 0.4];

        // Act
        let result = cb.retrieve_identity_for_mega_kernel(&hidden);
        let (knowledge, _) = result.unwrap();

        // Assert: knowledge is a separate allocation, not the same pointer
        assert_eq!(knowledge, hidden);
        assert_ne!(knowledge.as_ptr(), hidden.as_ptr());
    }

    #[test]
    fn callback_encode_single_token_id_one_char() {
        // Arrange: token_id 0..9 produce single-character strings
        let cb = make_callback();

        // Act: token_id=7 → "7".len() == 1
        let vec = cb.encode_single(7).unwrap();

        // Assert: MockEncoder returns vec![text.len(); 4]
        assert_eq!(vec.len(), 4);
        assert!((vec[0] - 1.0).abs() < 1e-5, "expected 1.0, got {}", vec[0]);
    }

    #[test]
    fn shim_target_layers_poisoned_mutex_returns_none() {
        // Arrange: create a shim and poison its mutex
        let cb = make_callback();
        let shim = SemanticGatekeeperCallbackShim {
            inner: Arc::new(std::sync::Mutex::new(cb)),
            hidden_size: 4,
        };

        // Poison the mutex by panicking while holding the lock
        let inner_clone = Arc::clone(&shim.inner);
        let handle = std::thread::spawn(move || {
            let _lock = inner_clone.lock().unwrap();
            panic!("intentional panic to poison mutex");
        });
        // Wait for thread to finish (it will panic)
        let _ = handle.join();

        // Act: target_layers should return None because mutex is poisoned
        let layers = LayerCallback::target_layers(&shim);

        // Assert
        assert!(layers.is_none(), "poisoned mutex should yield None");
    }

    #[test]
    fn text_encoder_trait_object_dispatch() {
        // Arrange: verify trait object dispatch works correctly
        let encoder: Arc<dyn TextEncoder> = Arc::new(MockEncoder);

        // Act
        let result = encoder.encode("hello");

        // Assert: MockEncoder returns vec![text.len(); 4]
        assert!(result.is_ok());
        let vec = result.unwrap();
        assert_eq!(vec.len(), 4);
        assert!((vec[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn text_encoder_error_all_variants_produce_nonempty_display() {
        // Arrange: ensure no Display variant returns empty string
        let variants = vec![
            TextEncoderError::Tokenize("msg".into()),
            TextEncoderError::Execute("msg".into()),
            TextEncoderError::Uninitialized,
        ];

        // Act & Assert
        for e in &variants {
            assert!(!e.to_string().is_empty(), "Display should not be empty for {:?}", e);
        }
    }

    #[test]
    fn callback_retrieve_identity_vs_normal_path_different_knowledge() {
        // Arrange
        let cb = make_callback();
        let hidden = vec![0.5f32, 0.6, 0.7, 0.8];

        // Act: get knowledge from both paths
        let (normal_knowledge, _) = cb.retrieve_for_mega_kernel(&hidden).unwrap();
        let (identity_knowledge, _) = cb.retrieve_identity_for_mega_kernel(&hidden).unwrap();

        // Assert: identity path returns the hidden itself; normal path returns encoder output
        assert_eq!(identity_knowledge, hidden);
        // MockEncoder encodes "struct Foo" (len 10) → vec![10.0; 4]
        assert!((normal_knowledge[0] - 10.0).abs() < 1e-5);
        assert_ne!(normal_knowledge, identity_knowledge);
    }

    #[test]
    fn callback_with_zero_hidden_size_still_creates() {
        // Arrange: edge case where hidden_size = 0
        use super::super::{GatekeeperRingBuffer, LevelKeysCache};
        let cb = SemanticGatekeeperCallback::new(
            Arc::new(LevelKeysCache::new(0)),
            Arc::new(GatekeeperRingBuffer::new(0, 0)),
            Arc::new(MockProvider),
            None,
            Arc::new(MockEncoder),
            Arc::new(MockTokenizer),
            0.5,
            0.9,
            0.3,
            0, // hidden_size = 0
        );

        // Assert: callback constructed without panic
        assert_eq!(cb.hidden_size, 0);
        assert!((cb.alpha() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn callback_gate_and_stability_thresholds_boundary_one() {
        // Arrange: both thresholds at exactly 1.0
        use super::super::{GatekeeperRingBuffer, LevelKeysCache};
        let cb = SemanticGatekeeperCallback::new(
            Arc::new(LevelKeysCache::new(4)),
            Arc::new(GatekeeperRingBuffer::new(4, 4)),
            Arc::new(MockProvider),
            None,
            Arc::new(MockEncoder),
            Arc::new(MockTokenizer),
            1.0, // gate_threshold = 1.0
            1.0, // stability_threshold = 1.0
            0.5,
            4,
        );

        // Assert: stored values are exactly 1.0
        assert!((cb.gate_threshold - 1.0).abs() < 1e-6);
        assert!((cb.stability_threshold - 1.0).abs() < 1e-6);
    }

    #[test]
    fn callback_ast_sentinel_none_by_default() {
        // Arrange
        let cb = make_callback();

        // Assert: ast_sentinel field is None (constructed with None)
        assert!(cb.ast_sentinel.is_none());
    }

    #[test]
    fn callback_multiple_retrieve_confidence_calls_are_independent() {
        // Arrange
        let cb = make_callback();
        let hidden_a = vec![1.0f32, 0.0, 0.0, 0.0];
        let hidden_b = vec![0.0f32, 1.0, 0.0, 0.0];

        // Act: two sequential retrieve_confidence calls
        let conf_a = cb.retrieve_confidence(&hidden_a);
        let conf_b = cb.retrieve_confidence(&hidden_b);

        // Assert: MockProvider returns same confidence regardless of input
        assert!(conf_a.is_some());
        assert!(conf_b.is_some());
        assert!((conf_a.unwrap() - conf_b.unwrap()).abs() < 1e-6);
    }

    #[test]
    fn extract_last_token_hidden_single_token_f32() {
        // Arrange: exactly one token with hidden_size=4
        let values = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Act
        let result = extract_last_token_hidden(&bytes, 4, DType::F32);

        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), values);
    }

    // ════════════════════════════════════════════════════════════════════════
    // 10 additional tests — uncovered paths & edge cases
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn text_encoder_error_source_returns_none() {
        // Arrange: TextEncoderError has no #[source] attribute on any variant,
        // so std::error::Error::source() should return None for all variants.

        // Act & Assert
        let variants: Vec<TextEncoderError> = vec![
            TextEncoderError::Tokenize("msg".into()),
            TextEncoderError::Execute("msg".into()),
            TextEncoderError::Uninitialized,
        ];
        for e in &variants {
            assert!(
                std::error::Error::source(e).is_none(),
                "TextEncoderError::source() must be None for {:?}",
                e
            );
        }
    }

    #[test]
    fn text_encoder_error_partial_eq_same_variant_same_message() {
        // Arrange: TextEncoderError derives Clone but not PartialEq;
        // verify equality via Display string comparison as a proxy.

        // Act
        let a = TextEncoderError::Tokenize("abc".into());
        let b = TextEncoderError::Tokenize("abc".into());
        let c = TextEncoderError::Tokenize("xyz".into());

        // Assert: same variant + same message → identical Display
        assert_eq!(a.to_string(), b.to_string());
        assert_ne!(a.to_string(), c.to_string());
    }

    #[test]
    fn failing_encoder_propagates_uninitialized_error_text() {
        // Arrange
        let encoder = FailingEncoder;

        // Act
        let err = encoder.encode("test").unwrap_err();

        // Assert
        let msg = err.to_string();
        assert!(
            msg.contains("not initialized"),
            "FailingEncoder should produce Uninitialized error, got: {msg}"
        );
    }

    #[test]
    fn retrieve_context_default_fields_are_valid() {
        // Arrange: construct a RetrieveContext and verify all fields
        let tokens: [u32; 0] = [];
        let ast_ctx: Option<super::super::AstContext<'_>> = None;

        // Act
        let ctx = super::super::RetrieveContext {
            generated_tokens: &tokens,
            ast: ast_ctx,
            step: 42,
            request_id: 7,
        };

        // Assert
        assert!(ctx.generated_tokens.is_empty());
        assert!(ctx.ast.is_none());
        assert_eq!(ctx.step, 42);
        assert_eq!(ctx.request_id, 7);
    }

    #[test]
    fn knowledge_entry_fields_roundtrip() {
        // Arrange: construct a KnowledgeEntry and verify field access
        let entry = super::super::KnowledgeEntry {
            text: "fn main() {}".to_string(),
            confidence: 0.92,
        };

        // Act — access fields
        let text = &entry.text;
        let conf = entry.confidence;

        // Assert
        assert_eq!(text, "fn main() {}");
        assert!((conf - 0.92).abs() < 1e-6);
    }

    #[test]
    fn knowledge_entry_clone_produces_equal_display() {
        // Arrange
        let entry = super::super::KnowledgeEntry {
            text: "impl Foo".into(),
            confidence: 0.5,
        };

        // Act
        let cloned = entry.clone();

        // Assert
        assert_eq!(entry.text, cloned.text);
        assert!((entry.confidence - cloned.confidence).abs() < f32::EPSILON);
    }

    #[test]
    fn callback_new_with_negative_thresholds_stores_values() {
        // Arrange: negative thresholds are technically allowed (caller's responsibility)
        use super::super::{GatekeeperRingBuffer, LevelKeysCache};
        let cb = SemanticGatekeeperCallback::new(
            Arc::new(LevelKeysCache::new(4)),
            Arc::new(GatekeeperRingBuffer::new(4, 4)),
            Arc::new(MockProvider),
            None,
            Arc::new(MockEncoder),
            Arc::new(MockTokenizer),
            -0.5,  // gate_threshold
            -1.0,  // stability_threshold
            0.3,
            4,
        );

        // Assert: values stored exactly as given (no clamping)
        assert!((cb.gate_threshold - (-0.5)).abs() < 1e-6);
        assert!((cb.stability_threshold - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn encode_row_empty_values_produces_zero_bytes() {
        // Arrange
        let values: Vec<f32> = vec![];

        // Act
        let bytes = encode_row(&values, DType::F32);

        // Assert: no values → no output bytes
        assert!(bytes.is_empty());
    }

    #[test]
    fn decode_row_zero_hidden_size_returns_empty_vec() {
        // Arrange: hidden_size=0 means row needs 0 bytes
        let row: Vec<u8> = vec![];

        // Act
        let result = decode_row(&row, 0, DType::F32);

        // Assert: succeeds with empty output
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn extract_last_token_hidden_overflow_hidden_size_returns_error() {
        // Arrange: hidden_size so large that row_bytes overflows usize
        let bytes = vec![0u8; 16];

        // Act: usize::MAX / 4 + 1 will overflow when multiplied by elem_bytes=4
        let huge_hidden_size = usize::MAX / 4 + 1;
        let result = extract_last_token_hidden(&bytes, huge_hidden_size, DType::F32);

        // Assert: overflow detected and reported as error
        assert!(result.is_err());
    }
}
