//! SemanticGatekeeperCallback — LayerCallback 实现.
//!
//! 协议 SSOT: `SPEC/SEMANTIC-GATEKEEPER.md` §2.1 §5 §7;
//! 接入点: `SPEC/05-OPTIMIZATIONS.md §2.9`, 优先级 90.
//!
//! 单步推理流程(每次 pre_node 触发):
//!  1. 检查是否在检测层 (target_layers 限定),非检测层直接 Continue.
//!  2. 稳定性追踪 (§5): anchor 相似度 > threshold 且 AST 未变 → 复用 v_knowledge.
//!  3. 从 Q-tap ring buffer 读取当前层 Q[-1] 向量 (§4).
//!  4. 层级路由: cosine(Q, K_Lx) argmax; < gate_threshold → Continue.
//!  5. 通过 KnowledgeProvider 检索知识条目 (返回 None → Continue).
//!  6. 知识文本经主模型 embed 层编码为 v_knowledge (Phase D: small_graph pending).
//!  7. 残差相加 h_new_last = h_last + alpha * confidence * v_knowledge,
//!     通过现有 `CallbackAction::InjectHidden { data }` 返回.
//!  8. 更新 ActiveState.

use std::sync::{Arc, RwLock};

use gllm_kernels::types::DType;
use half::{bf16, f16};

use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
use crate::scheduler::types::RequestId;

use super::{
    active_state::ActiveState, level_keys::LevelKeysCache, ring_buffer::GatekeeperRingBuffer,
    AstContext, AstSentinel, KnowledgeProvider, RetrieveContext, SemanticLevel, TokenizerLookup,
};

/// Priority for Semantic Gatekeeper in the CallbackChain
/// (`SPEC/05-OPTIMIZATIONS.md §8`).
pub const SEMANTIC_GATEKEEPER_PRIORITY: u32 = 90;

/// `KnowledgeProvider` 返回文本后由此 trait 编码为 `v_knowledge` 向量.
///
/// Phase D 的 `small_graph` + `LevelKeysCache::precompute` 会提供一个
/// 默认实现 (小 CompilerGraph 走 FusedGraphExecutor 编码, ARCH-FULL-JIT 合规).
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
/// 注册到 `CallbackChain` 后由 `FusedGraphExecutor::run_with_kv_cache_with_callbacks`
/// 在每个检测层 `pre_node` 触发.
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

    /// 当前检测层是否登记 (快速查找).
    fn is_detection_layer(&self, layer_idx: usize) -> bool {
        self.detection_layers().contains(&layer_idx)
    }
}

// ============================================================================
// LayerCallback 实现
// ============================================================================

impl LayerCallback for SemanticGatekeeperCallback {
    fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
        // 1. 仅在检测层触发.
        if !self.is_detection_layer(ctx.layer_idx) {
            return CallbackAction::Continue;
        }

        // 2. Level Keys cache 未填充该层时返回 Continue (未完成预计算).
        let keys = match self.level_keys.get(ctx.layer_idx) {
            Some(k) => k,
            None => return CallbackAction::Continue,
        };

        let dtype = ctx.model_config.geometry.dtype;

        // 3. 从 hidden_state 切出最后一个 token 向量.
        let last_hidden = match extract_last_token_hidden(ctx.hidden_state, self.hidden_size, dtype)
        {
            Ok(v) => v,
            // NO_SILENT_FALLBACK: 解码失败直接跳过而非注入垃圾.
            Err(_) => return CallbackAction::Continue,
        };

        // 4. 请求边界刷新.
        {
            let mut state = match self.active_state.write() {
                Ok(s) => s,
                Err(_) => return CallbackAction::Continue,
            };
            let req_id: RequestId = ctx.request_id;
            if state.needs_request_boundary_refresh(req_id) {
                state.clear();
            }
            state.last_request = Some(req_id);
        }

        // 5. AST 上下文 (用于节点变更判定).
        let ast_ctx: Option<AstContext<'_>> = self
            .ast_sentinel
            .as_ref()
            .and_then(|s| s.current_context(&[], self.tokenizer.as_ref()));

        // 6. 稳定性追踪 — anchor 高相似度 && AST 未变 → 复用.
        let step = (ctx.position + ctx.seq_len) as u64;
        let reuse = {
            let state = match self.active_state.read() {
                Ok(s) => s,
                Err(_) => return CallbackAction::Continue,
            };
            match &state.anchor_hidden {
                Some(anchor) if anchor.len() == last_hidden.len() => {
                    let sim = cosine(&last_hidden, anchor);
                    let ast_unchanged = state.ast_node_kind.as_deref()
                        == ast_ctx.as_ref().map(|c| c.node_kind);
                    if sim > self.stability_threshold && ast_unchanged {
                        state.v_knowledge.clone()
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };

        let (v_knowledge, updated_state) = if let Some(v) = reuse {
            (v, None)
        } else {
            // 7. Q-tap 读取.
            let q = match self.ring_buffer.read_latest(step) {
                Ok(q) => q,
                Err(_) => return CallbackAction::Continue,
            };

            // 8. 层级路由.
            let scores = [
                cosine(&q, &keys[0]),
                cosine(&q, &keys[1]),
                cosine(&q, &keys[2]),
            ];
            let (best_idx, best_score) = argmax3(scores);
            if best_score < self.gate_threshold {
                return CallbackAction::Continue;
            }
            let level = match SemanticLevel::from_idx(best_idx) {
                Some(l) => l,
                None => return CallbackAction::Continue,
            };

            // 9. 知识检索.
            let retrieve_ctx = RetrieveContext {
                generated_tokens: &[],
                ast: ast_ctx,
                step,
                request_id: ctx.request_id as RequestId,
            };
            let entry = match self.provider.retrieve(&q, level, &retrieve_ctx) {
                Some(e) => e,
                None => return CallbackAction::Continue,
            };
            if !entry.confidence.is_finite() || entry.confidence <= 0.0 {
                return CallbackAction::Continue;
            }

            // 10. 文本编码为 v_knowledge.
            let mut v_raw = match self.text_encoder.encode(&entry.text) {
                Ok(v) => v,
                Err(_) => return CallbackAction::Continue,
            };
            if v_raw.len() != self.hidden_size {
                return CallbackAction::Continue;
            }

            // 把 confidence 纳入向量模值, 方便后续 ReuseCache 复用时已编码 α' 分量.
            for x in v_raw.iter_mut() {
                *x *= entry.confidence;
            }

            // 11. 构造 ActiveState 更新载荷.
            let ast_kind = ast_ctx.as_ref().map(|c| c.node_kind.to_string());
            let key_hash = hash_text(&entry.text);
            let updated = ActiveStateUpdate {
                level: Some(level),
                key_hash: Some(key_hash),
                anchor_hidden: Some(last_hidden.clone()),
                v_knowledge: Some(v_raw.clone()),
                ast_node_kind: ast_kind,
                last_step: step,
            };
            (v_raw, Some(updated))
        };

        // 12. 写回 ActiveState (在 RwLock write guard 下).
        if let Some(update) = updated_state {
            if let Ok(mut state) = self.active_state.write() {
                state.level = update.level;
                state.key_hash = update.key_hash;
                state.anchor_hidden = update.anchor_hidden;
                state.v_knowledge = update.v_knowledge;
                state.ast_node_kind = update.ast_node_kind;
                state.last_step = update.last_step;
            }
        }

        // 13. 残差相加 + 编码回 bytes (保留前面所有 token 不变, 仅修改最后一个).
        let new_bytes = build_injected_hidden(
            ctx.hidden_state,
            &v_knowledge,
            self.alpha,
            self.hidden_size,
            dtype,
        );
        CallbackAction::InjectHidden { data: new_bytes }
    }

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

struct ActiveStateUpdate {
    level: Option<SemanticLevel>,
    key_hash: Option<u64>,
    anchor_hidden: Option<Vec<f32>>,
    v_knowledge: Option<Vec<f32>>,
    ast_node_kind: Option<String>,
    last_step: u64,
}

// ============================================================================
// 数学辅助
// ============================================================================

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

fn hash_text(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

// ============================================================================
// dtype-aware bytes ↔ f32 转换
// ============================================================================

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

#[derive(Debug, thiserror::Error, Clone)]
enum ExtractError {
    #[error("hidden_bytes truncated or size mismatch")]
    Truncated,
    #[error("row_bytes overflow")]
    Overflow,
    #[error("unsupported dtype {0}")]
    UnsupportedDtype(String),
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
}
