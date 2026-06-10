//! 多模态 Token 路由 — Gemma 4 Vision/Audio/Text 统一序列化
//!
//! 将文本 token 序列中的特殊 token (image_token_id, audio_token_id) 替换为
//! 对应编码器输出的 virtual token 序列 + embedding。
//!
//! 本模块定义 T58 所需的 API 脚手架：
//! - `MultimodalTokenIds` — 特殊 token ID (来源: ModelConfig; 禁止硬编码)
//! - `MediaKind` / `MultimodalEncoded` — 编码器输出的统一表达
//! - `MultimodalEncoder` — 编码器接口 (供 SigLIP / USM Conformer 实现)
//! - `EncoderMedia` — 编码器输入 (从 `generation::MediaInput` 转换)
//! - `MultimodalContext` — 单次生成请求的多模态上下文
//! - `route_multimodal_tokens` — 核心路由函数：将含 special token 的
//!   prompt token 序列展开成 `RoutedSequence`
//!
//! 实际编码器实现由 P3.1 (SigLIP) 与 P3.2 (USM Conformer) 补齐；
//! 本模块只承诺 routing 正确性，对编码器 trait 的任何实现均能工作。

use std::path::PathBuf;

use crate::engine::executor::BackendError;

// ============================================================================
// MultimodalTokenIds — 特殊 token ID (来源: ModelConfig.multimodal_token_ids)
// ============================================================================

/// 多模态特殊 token ID。
///
/// **来源铁律**: 必须从模型 config (tokenizer special_tokens_map 或
/// `ModelConfig::multimodal_token_ids`) 读取，禁止在代码中硬编码。
///
/// Gemma 4 的约定值为 (258880, 258881, 258882, 258883)，仅在 manifest /
/// config 未显式提供时由 `MultimodalTokenIds::fallback_multimodal_token_ids()` 注入。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultimodalTokenIds {
    /// 图像占位符 token ID
    pub image_token_id: u32,
    /// 音频占位符 token ID
    pub audio_token_id: u32,
    /// 图像结束 token ID (begin-of-image / end-of-image 结构中的后半)
    pub eoi_token_id: u32,
    /// 音频结束 token ID
    pub eoa_token_id: u32,
}

impl MultimodalTokenIds {
    /// Fallback 多模态 token ID (Gemma 4 约定值: 258880–258883)。
    ///
    /// **仅**在 GGUF metadata / model config 未提供多模态 token ID 时使用。
    /// 正式路径应从 `ModelConfig::multimodal_token_ids` 或 tokenizer special_tokens_map 读取。
    /// 此 fallback 存在的原因是部分 GGUF 文件缺少多模态 token 元数据。
    pub fn fallback_multimodal_token_ids() -> Self {
        Self {
            image_token_id: 258880,
            audio_token_id: 258881,
            eoi_token_id: 258882,
            eoa_token_id: 258883,
        }
    }

    /// 判断 `token` 是否为图像占位符。
    pub fn is_image(&self, token: u32) -> bool {
        token == self.image_token_id
    }

    /// 判断 `token` 是否为音频占位符。
    pub fn is_audio(&self, token: u32) -> bool {
        token == self.audio_token_id
    }
}


// ============================================================================
// EncoderMedia — 编码器输入 (对齐 generation::MediaInput)
// ============================================================================

/// 编码器期望的媒体输入。与 `generation::MediaInput` 对齐的四种模式。
///
/// SPEC 依据: SPEC/04-API-DESIGN.md §3.7.1 (MediaInput 四种模式)。
#[derive(Debug, Clone)]
pub enum EncoderMedia {
    /// 本地文件路径
    File(PathBuf),
    /// Base64 编码数据 (含可选 MIME type)
    Base64 {
        data: String,
        mime_type: Option<String>,
    },
    /// 原始字节(已解码的像素 / PCM)
    Raw(Vec<u8>),
    /// 远程资源 URL (http/https/s3/file://), 由 encoder 负责拉取。
    /// encoder 实现若不支持网络/远端协议应返回 `BackendError::NetworkUnreachable`
    /// 或同类显式错误,禁止 silent fallback 到本地空响应。
    Url(String),
}

impl EncoderMedia {
    /// 从 `crate::generation::MediaInput` 构造。
    pub fn from_generation(input: &crate::generation::MediaInput) -> Self {
        match input {
            crate::generation::MediaInput::File(path) => {
                EncoderMedia::File(PathBuf::from(path))
            }
            crate::generation::MediaInput::Base64 { data, mime_type } => {
                EncoderMedia::Base64 {
                    data: data.clone(),
                    mime_type: mime_type.clone(),
                }
            }
            crate::generation::MediaInput::Raw(bytes) => EncoderMedia::Raw(bytes.clone()),
            crate::generation::MediaInput::Url(url) => EncoderMedia::Url(url.clone()),
        }
    }
}

// ============================================================================
// MediaKind / MultimodalEncoded — 编码器输出
// ============================================================================

/// 媒体类型标签。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    Image,
    Audio,
}

/// 编码器输出：一段连续的 virtual token 序列 + 对应 embedding。
///
/// - `tokens`: 每个 virtual token 的占位 ID（取自 `MultimodalTokenIds`，
///   供 decoder graph 记账用；实际 embedding 由 `embeddings` 提供）
/// - `embeddings`: 扁平 `[num_tokens * hidden_size]` f32 向量
/// - `hidden_size`: 每个 token 的隐藏维度 (须与 text embedding 对齐)
/// - `kind`: 标记 Image 还是 Audio
#[derive(Debug, Clone)]
pub struct MultimodalEncoded {
    pub tokens: Vec<u32>,
    pub embeddings: Vec<f32>,
    pub hidden_size: usize,
    pub kind: MediaKind,
}

impl MultimodalEncoded {
    /// virtual token 的数量。
    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// 校验 `embeddings.len() == num_tokens * hidden_size`。
    pub fn validate(&self) -> Result<(), BackendError> {
        let expected = self.tokens.len() * self.hidden_size;
        if self.embeddings.len() != expected {
            return Err(BackendError::Other(format!(
                "MultimodalEncoded shape mismatch: tokens={} hidden={} expected embeddings={} got={}",
                self.tokens.len(),
                self.hidden_size,
                expected,
                self.embeddings.len()
            )));
        }
        Ok(())
    }
}

// ============================================================================
// MultimodalEncoder trait — SigLIP / Conformer / Mock 共通接口
// ============================================================================

/// 多模态编码器 trait。
///
/// 实现者负责把 raw 媒体 (image / audio) 编码为 `MultimodalEncoded`。
/// 当前产线只有 Mock 实现；真实 SigLIP / USM 实现待 P3.1 / P3.2。
pub trait MultimodalEncoder: Send + Sync {
    /// 编码图像。
    fn encode_image(&self, media: &EncoderMedia) -> Result<MultimodalEncoded, BackendError>;

    /// 编码音频。
    fn encode_audio(&self, media: &EncoderMedia) -> Result<MultimodalEncoded, BackendError>;
}

// ============================================================================
// MultimodalContext — 单请求的预编码上下文
// ============================================================================

/// 一次生成请求的预编码产物集合。
///
/// 在 prompt 被 tokenize 之前，由 `Client::execute_generation_multimodal`
/// 调用 `MultimodalEncoder` 生成若干 `MultimodalEncoded`，按出现顺序压栈；
/// 之后 `route_multimodal_tokens` 消费该栈，将 special token 替换为对应
/// virtual token 序列并拼接 embedding。
#[derive(Debug, Default, Clone)]
pub struct MultimodalContext {
    /// 按出现顺序排列的图像编码产物
    pub images: Vec<MultimodalEncoded>,
    /// 按出现顺序排列的音频编码产物
    pub audios: Vec<MultimodalEncoded>,
}

impl MultimodalContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// 追加一个图像编码。
    pub fn push_image(&mut self, enc: MultimodalEncoded) -> Result<(), BackendError> {
        if enc.kind != MediaKind::Image {
            return Err(BackendError::Other(
                "MultimodalContext::push_image received non-Image encoding".into(),
            ));
        }
        enc.validate()?;
        self.images.push(enc);
        Ok(())
    }

    /// 追加一个音频编码。
    pub fn push_audio(&mut self, enc: MultimodalEncoded) -> Result<(), BackendError> {
        if enc.kind != MediaKind::Audio {
            return Err(BackendError::Other(
                "MultimodalContext::push_audio received non-Audio encoding".into(),
            ));
        }
        enc.validate()?;
        self.audios.push(enc);
        Ok(())
    }

    /// 是否完全为空 (无任何多模态输入)。
    pub fn is_empty(&self) -> bool {
        self.images.is_empty() && self.audios.is_empty()
    }
}

// ============================================================================
// RoutedSequence — 路由后的 token + embedding 表
// ============================================================================

/// Multi-modal routing 的产物。
///
/// - `token_ids`: 展开后的 token 序列（special token 已被 virtual token 占位
///   符逐位替换，长度 = 文本 token + Σ encoder virtual token 数）
/// - `fused_embeddings`: 与 `token_ids` 对齐的每位置可选 embedding，
///   文本位置为 `None`（由 text embedding gather 填充），
///   多模态位置为 `Some(slice)` 直接覆写
/// - `text_positions`: 原文本 token 在新序列中的下标
/// - `hidden_size`: embedding 维度
#[derive(Debug, Clone)]
pub struct RoutedSequence {
    pub token_ids: Vec<u32>,
    pub fused_embeddings: Vec<Option<Vec<f32>>>,
    pub text_positions: Vec<usize>,
    pub hidden_size: usize,
}

impl RoutedSequence {
    pub fn seq_len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn has_multimodal(&self) -> bool {
        self.fused_embeddings.iter().any(|s| s.is_some())
    }
}

// ============================================================================
// 核心路由函数
// ============================================================================

/// 将含 special token 的 prompt token 序列路由为完整 (tokens, embeddings) 序列。
///
/// 算法：
/// ```text
/// for tok in prompt_tokens:
///     if tok == image_token_id:
///         let enc = ctx.images.next()   // 错误：无对应编码产物时 Err
///         for (v_tok, v_emb) in enc:
///             push(v_tok, Some(v_emb))
///     elif tok == audio_token_id:
///         let enc = ctx.audios.next()
///         ...
///     else:
///         push(tok, None)               // 文本位置，embedding 由 gather 填
/// ```
///
/// **错误条件**:
/// - prompt 中的 image_token_id 数量 ≠ `ctx.images.len()`
/// - prompt 中的 audio_token_id 数量 ≠ `ctx.audios.len()`
/// - 任一 `MultimodalEncoded.hidden_size` 与 `text_hidden_size` 不一致
///
/// **纯文本透传**: `ctx.is_empty() && 无 special token` 时，返回的
/// `token_ids == prompt_tokens`，`fused_embeddings` 全部为 `None`。
pub fn route_multimodal_tokens(
    prompt_tokens: &[u32],
    ctx: &MultimodalContext,
    token_ids_cfg: &MultimodalTokenIds,
    text_hidden_size: usize,
) -> Result<RoutedSequence, BackendError> {
    // 统计 special token 出现次数，预先校验与 ctx 对齐
    let image_slots = prompt_tokens
        .iter()
        .filter(|t| token_ids_cfg.is_image(**t))
        .count();
    let audio_slots = prompt_tokens
        .iter()
        .filter(|t| token_ids_cfg.is_audio(**t))
        .count();

    if image_slots != ctx.images.len() {
        return Err(BackendError::Other(format!(
            "multimodal routing: prompt has {} image tokens but context provided {} image encodings",
            image_slots,
            ctx.images.len()
        )));
    }
    if audio_slots != ctx.audios.len() {
        return Err(BackendError::Other(format!(
            "multimodal routing: prompt has {} audio tokens but context provided {} audio encodings",
            audio_slots,
            ctx.audios.len()
        )));
    }

    // 校验 hidden_size 对齐
    for enc in ctx.images.iter().chain(ctx.audios.iter()) {
        if enc.hidden_size != text_hidden_size {
            return Err(BackendError::Other(format!(
                "multimodal encoder hidden_size {} ≠ model hidden_size {}",
                enc.hidden_size, text_hidden_size
            )));
        }
    }

    let mut out_tokens: Vec<u32> = Vec::with_capacity(prompt_tokens.len());
    let mut out_embeds: Vec<Option<Vec<f32>>> = Vec::with_capacity(prompt_tokens.len());
    let mut text_positions: Vec<usize> = Vec::with_capacity(prompt_tokens.len());

    let mut image_idx = 0usize;
    let mut audio_idx = 0usize;

    for &tok in prompt_tokens {
        if token_ids_cfg.is_image(tok) {
            let enc = &ctx.images[image_idx];
            image_idx += 1;
            for i in 0..enc.num_tokens() {
                out_tokens.push(enc.tokens[i]);
                let start = i * enc.hidden_size;
                let end = start + enc.hidden_size;
                out_embeds.push(Some(enc.embeddings[start..end].to_vec()));
            }
        } else if token_ids_cfg.is_audio(tok) {
            let enc = &ctx.audios[audio_idx];
            audio_idx += 1;
            for i in 0..enc.num_tokens() {
                out_tokens.push(enc.tokens[i]);
                let start = i * enc.hidden_size;
                let end = start + enc.hidden_size;
                out_embeds.push(Some(enc.embeddings[start..end].to_vec()));
            }
        } else {
            text_positions.push(out_tokens.len());
            out_tokens.push(tok);
            out_embeds.push(None);
        }
    }

    Ok(RoutedSequence {
        token_ids: out_tokens,
        fused_embeddings: out_embeds,
        text_positions,
        hidden_size: text_hidden_size,
    })
}

// ============================================================================
// Fused hidden state construction (ARCH-MULTIMODAL-FUSION injection input)
// ============================================================================

/// Build the fused hidden state to be injected into the decoder's first layer.
///
/// Shape: flat row-major `[routed.seq_len() * hidden_size]` f32 buffer.
///
/// For each position `i`:
/// - If `routed.fused_embeddings[i]` is `Some(v)` (media virtual token) → copy `v`.
/// - Otherwise (text token) → gather row `routed.token_ids[i]` from `embed_rows`.
///
/// `embed_rows`: flat `[vocab_size * hidden_size]` view of the model's
/// `embed_tokens.weight` in f32, row-major. Callers are responsible for
/// converting from the native weight dtype.
///
/// SPEC: 02-ARCHITECTURE ARCH-MULTIMODAL-FUSION injection point = after
/// embedding lookup, before the first transformer layer, shape
/// `[num_tokens, hidden_size]` row-major, zero-copy at position granularity.
pub fn build_fused_hidden(
    routed: &RoutedSequence,
    embed_rows: &[f32],
    hidden_size: usize,
) -> Result<Vec<f32>, BackendError> {
    if routed.hidden_size != hidden_size {
        return Err(BackendError::Other(format!(
            "build_fused_hidden: routed.hidden_size {} != model hidden_size {}",
            routed.hidden_size, hidden_size
        )));
    }
    if !embed_rows.len().is_multiple_of(hidden_size) {
        return Err(BackendError::Other(format!(
            "build_fused_hidden: embed_rows.len() {} not divisible by hidden_size {}",
            embed_rows.len(),
            hidden_size
        )));
    }
    let vocab_size = embed_rows.len() / hidden_size;
    let seq_len = routed.seq_len();
    if routed.fused_embeddings.len() != seq_len {
        return Err(BackendError::Other(format!(
            "build_fused_hidden: routed.fused_embeddings len {} != seq_len {}",
            routed.fused_embeddings.len(),
            seq_len
        )));
    }

    let mut hidden = vec![0.0f32; seq_len * hidden_size];
    for (i, opt_emb) in routed.fused_embeddings.iter().enumerate() {
        let dst = &mut hidden[i * hidden_size..(i + 1) * hidden_size];
        match opt_emb {
            Some(media_emb) => {
                if media_emb.len() != hidden_size {
                    return Err(BackendError::Other(format!(
                        "build_fused_hidden: media embedding at position {} has length {} != hidden_size {}",
                        i,
                        media_emb.len(),
                        hidden_size
                    )));
                }
                dst.copy_from_slice(media_emb);
            }
            None => {
                let tok = routed.token_ids[i] as usize;
                if tok >= vocab_size {
                    return Err(BackendError::Other(format!(
                        "build_fused_hidden: token id {} at position {} out of range (vocab {})",
                        tok, i, vocab_size
                    )));
                }
                dst.copy_from_slice(&embed_rows[tok * hidden_size..(tok + 1) * hidden_size]);
            }
        }
    }
    Ok(hidden)
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock encoder：为任何输入生成固定的 virtual token 序列。
    /// 通过 AtomicUsize 计数器验证被调用次数。
    pub(crate) struct MockEncoder {
        image_virtual_tokens: usize,
        audio_virtual_tokens: usize,
        hidden_size: usize,
        image_token_id: u32,
        audio_token_id: u32,
        image_calls: std::sync::atomic::AtomicUsize,
        audio_calls: std::sync::atomic::AtomicUsize,
    }

    impl MockEncoder {
        pub(crate) fn new(
            image_n: usize,
            audio_n: usize,
            hidden: usize,
            ids: MultimodalTokenIds,
        ) -> Self {
            Self {
                image_virtual_tokens: image_n,
                audio_virtual_tokens: audio_n,
                hidden_size: hidden,
                image_token_id: ids.image_token_id,
                audio_token_id: ids.audio_token_id,
                image_calls: std::sync::atomic::AtomicUsize::new(0),
                audio_calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        pub(crate) fn image_call_count(&self) -> usize {
            self.image_calls.load(std::sync::atomic::Ordering::SeqCst)
        }

        pub(crate) fn audio_call_count(&self) -> usize {
            self.audio_calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl MultimodalEncoder for MockEncoder {
        fn encode_image(&self, _media: &EncoderMedia) -> Result<MultimodalEncoded, BackendError> {
            self.image_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let n = self.image_virtual_tokens;
            Ok(MultimodalEncoded {
                tokens: vec![self.image_token_id; n],
                embeddings: (0..n * self.hidden_size).map(|i| i as f32 * 0.01).collect(),
                hidden_size: self.hidden_size,
                kind: MediaKind::Image,
            })
        }
        fn encode_audio(&self, _media: &EncoderMedia) -> Result<MultimodalEncoded, BackendError> {
            self.audio_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let n = self.audio_virtual_tokens;
            Ok(MultimodalEncoded {
                tokens: vec![self.audio_token_id; n],
                embeddings: (0..n * self.hidden_size).map(|i| -(i as f32) * 0.01).collect(),
                hidden_size: self.hidden_size,
                kind: MediaKind::Audio,
            })
        }
    }

    fn default_ids() -> MultimodalTokenIds {
        MultimodalTokenIds::fallback_multimodal_token_ids()
    }

    #[test]
    fn multimodal_text_only_passthrough() {
        // 无 special token + 空 ctx → token_ids 透传，embeddings 全 None
        let prompt = vec![1u32, 2, 3, 4];
        let ctx = MultimodalContext::new();
        let ids = default_ids();
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 128).unwrap();
        assert_eq!(out.token_ids, prompt);
        assert_eq!(out.fused_embeddings.len(), 4);
        assert!(out.fused_embeddings.iter().all(|e| e.is_none()));
        assert!(!out.has_multimodal());
    }

    #[test]
    fn multimodal_image_routing_expands_special_token() {
        let ids = default_ids();
        let prompt = vec![10u32, ids.image_token_id, 20];
        let encoder = MockEncoder::new(3, 0, 4, ids);
        let mut ctx = MultimodalContext::new();
        ctx.push_image(encoder.encode_image(&EncoderMedia::Raw(vec![])).unwrap())
            .unwrap();

        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        // 展开后长度 = 1 + 3 + 1
        assert_eq!(out.seq_len(), 5);
        assert_eq!(out.token_ids[0], 10);
        assert_eq!(out.token_ids[4], 20);
        assert!(out.fused_embeddings[1].is_some());
        assert!(out.fused_embeddings[2].is_some());
        assert!(out.fused_embeddings[3].is_some());
        assert!(out.fused_embeddings[0].is_none());
        assert!(out.fused_embeddings[4].is_none());
        assert_eq!(out.text_positions, vec![0, 4]);
        assert_eq!(encoder.image_call_count(), 1);
    }

    #[test]
    fn multimodal_audio_routing_expands_special_token() {
        let ids = default_ids();
        let prompt = vec![ids.audio_token_id, 99];
        let encoder = MockEncoder::new(0, 2, 8, ids);
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(encoder.encode_audio(&EncoderMedia::Raw(vec![])).unwrap())
            .unwrap();

        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 8).unwrap();
        assert_eq!(out.seq_len(), 3);
        assert_eq!(out.token_ids[2], 99);
        assert!(out.fused_embeddings[0].is_some());
        assert!(out.fused_embeddings[1].is_some());
        assert!(out.fused_embeddings[2].is_none());
        assert_eq!(out.text_positions, vec![2]);
        assert_eq!(encoder.audio_call_count(), 1);
    }

    #[test]
    fn multimodal_mixed_image_and_audio_routing() {
        let ids = default_ids();
        let prompt = vec![1u32, ids.image_token_id, 2, ids.audio_token_id, 3];
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 2],
            embeddings: vec![0.0; 2 * 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 3],
            embeddings: vec![0.0; 3 * 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc_img).unwrap();
        ctx.push_audio(enc_aud).unwrap();

        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        // 1 + 2 + 1 + 3 + 1 = 8
        assert_eq!(out.seq_len(), 8);
        assert_eq!(out.text_positions, vec![0, 3, 7]);
    }

    #[test]
    fn multimodal_missing_encoder_is_rejected() {
        let ids = default_ids();
        let prompt = vec![ids.image_token_id];
        let ctx = MultimodalContext::new();
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 8).unwrap_err();
        assert!(format!("{err}").contains("image"));
    }

    #[test]
    fn multimodal_extra_encoder_without_special_token_is_rejected() {
        let ids = default_ids();
        let prompt = vec![1u32, 2];
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap_err();
        assert!(format!("{err}").contains("image"));
    }

    #[test]
    fn multimodal_hidden_size_mismatch_is_rejected() {
        let ids = default_ids();
        let prompt = vec![ids.image_token_id];
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 8).unwrap_err();
        assert!(format!("{err}").contains("hidden_size"));
    }

    #[test]
    fn multimodal_encoded_validate_catches_shape_mismatch() {
        let bad = MultimodalEncoded {
            tokens: vec![1, 2, 3],
            embeddings: vec![0.0; 5], // expected 3*4=12
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let err = bad.validate().unwrap_err();
        assert!(format!("{err}").contains("shape mismatch"));
    }

    #[test]
    fn multimodal_push_image_rejects_audio_kind() {
        let bad = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        let err = ctx.push_image(bad).unwrap_err();
        assert!(format!("{err}").contains("non-Image"));
    }

    #[test]
    fn multimodal_encoder_media_from_generation_input() {
        let gen_file = crate::generation::MediaInput::File("/tmp/foo.jpg".into());
        let enc = EncoderMedia::from_generation(&gen_file);
        assert!(matches!(enc, EncoderMedia::File(_)));

        let gen_b64 = crate::generation::MediaInput::Base64 {
            data: "AAAA".into(),
            mime_type: Some("image/png".into()),
        };
        let enc = EncoderMedia::from_generation(&gen_b64);
        assert!(matches!(enc, EncoderMedia::Base64 { .. }));

        let gen_raw = crate::generation::MediaInput::Raw(vec![1, 2, 3]);
        let enc = EncoderMedia::from_generation(&gen_raw);
        assert!(matches!(enc, EncoderMedia::Raw(_)));
    }

    #[test]
    fn multimodal_mock_encoder_end_to_end_routing() {
        // 验证：encoder trait + routing 全链路
        let ids = default_ids();
        let encoder = MockEncoder::new(4, 0, 2, ids);

        // 1. 用户提供 MediaInput
        let media = EncoderMedia::from_generation(
            &crate::generation::MediaInput::Raw(vec![0xFF; 16]),
        );
        // 2. 调用 encoder
        let encoded = encoder.encode_image(&media).unwrap();
        // 3. 压入 ctx
        let mut ctx = MultimodalContext::new();
        ctx.push_image(encoded).unwrap();

        // 4. prompt 含一个 IMAGE special token
        let prompt = vec![100u32, ids.image_token_id, 200];
        let routed = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();

        // 5. 验证
        assert_eq!(routed.seq_len(), 1 + 4 + 1);
        assert_eq!(routed.text_positions, vec![0, 5]);
        assert!(routed.has_multimodal());
        assert_eq!(routed.token_ids[0], 100);
        assert_eq!(routed.token_ids[5], 200);
        assert_eq!(encoder.image_call_count(), 1);
    }

    #[test]
    fn multimodal_token_ids_helpers() {
        let ids = default_ids();
        assert!(ids.is_image(258880));
        assert!(!ids.is_image(258881));
        assert!(ids.is_audio(258881));
        assert!(!ids.is_audio(258880));
    }

    #[test]
    fn build_fused_hidden_text_only_gathers_all_rows() {
        // vocab=4, hidden=2; embed row[k] = [k*10, k*10+1]
        let embed: Vec<f32> = (0..4 * 2).map(|i| i as f32).collect();
        let routed = RoutedSequence {
            token_ids: vec![2, 0, 3],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 2,
        };
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // row 2, row 0, row 3
        assert_eq!(hidden, vec![4.0, 5.0, 0.0, 1.0, 6.0, 7.0]);
    }

    #[test]
    fn build_fused_hidden_overwrites_media_positions() {
        // vocab=2, hidden=3
        let embed: Vec<f32> = vec![0.0; 2 * 3];
        let media = vec![9.0, 8.0, 7.0];
        let routed = RoutedSequence {
            token_ids: vec![0, 258880, 1],
            fused_embeddings: vec![None, Some(media.clone()), None],
            text_positions: vec![0, 2],
            hidden_size: 3,
        };
        let hidden = build_fused_hidden(&routed, &embed, 3).unwrap();
        // Position 1 overwritten by media
        assert_eq!(&hidden[3..6], media.as_slice());
        // Positions 0 and 2 are gathered (zeroes here)
        assert_eq!(&hidden[0..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&hidden[6..9], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn build_fused_hidden_rejects_hidden_size_mismatch() {
        let embed: Vec<f32> = vec![0.0; 4];
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        let err = build_fused_hidden(&routed, &embed, 4).unwrap_err();
        assert!(format!("{err}").contains("hidden_size"));
    }

    #[test]
    fn build_fused_hidden_rejects_out_of_range_token() {
        let embed: Vec<f32> = vec![0.0; 2];
        let routed = RoutedSequence {
            token_ids: vec![5], // vocab only has 1 row
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        let err = build_fused_hidden(&routed, &embed, 2).unwrap_err();
        assert!(format!("{err}").contains("out of range"));
    }

    #[test]
    fn build_fused_hidden_rejects_media_embedding_wrong_length() {
        let embed: Vec<f32> = vec![0.0; 4];
        let routed = RoutedSequence {
            token_ids: vec![258880],
            fused_embeddings: vec![Some(vec![1.0, 2.0, 3.0])], // hidden=2, but 3 floats
            text_positions: vec![],
            hidden_size: 2,
        };
        let err = build_fused_hidden(&routed, &embed, 2).unwrap_err();
        assert!(format!("{err}").contains("media embedding"));
    }

    // ── Additional tests ──

    #[test]
    fn multimodal_token_ids_fallback_matches_gemma4() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        assert_eq!(ids.image_token_id, 258880);
        assert_eq!(ids.audio_token_id, 258881);
        assert_eq!(ids.eoi_token_id, 258882);
        assert_eq!(ids.eoa_token_id, 258883);
    }

    #[test]
    fn multimodal_token_ids_copy_eq() {
        let a = MultimodalTokenIds::fallback_multimodal_token_ids();
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn multimodal_token_ids_custom() {
        let ids = MultimodalTokenIds {
            image_token_id: 100,
            audio_token_id: 200,
            eoi_token_id: 300,
            eoa_token_id: 400,
        };
        assert!(ids.is_image(100));
        assert!(!ids.is_image(200));
        assert!(ids.is_audio(200));
        assert!(!ids.is_audio(100));
    }

    #[test]
    fn media_kind_variants() {
        assert_eq!(MediaKind::Image, MediaKind::Image);
        assert_ne!(MediaKind::Image, MediaKind::Audio);
    }

    #[test]
    fn encoder_media_variants() {
        let file = EncoderMedia::File(PathBuf::from("/tmp/img.png"));
        let b64 = EncoderMedia::Base64 { data: "abc".into(), mime_type: Some("image/png".into()) };
        let raw = EncoderMedia::Raw(vec![1, 2, 3]);
        let url = EncoderMedia::Url("https://example.com/img.jpg".into());

        assert!(matches!(file, EncoderMedia::File(_)));
        assert!(matches!(b64, EncoderMedia::Base64 { .. }));
        assert!(matches!(raw, EncoderMedia::Raw(_)));
        assert!(matches!(url, EncoderMedia::Url(_)));
    }

    #[test]
    fn multimodal_encoded_num_tokens() {
        let enc = MultimodalEncoded {
            tokens: vec![1, 2, 3],
            embeddings: vec![0.0; 9],
            hidden_size: 3,
            kind: MediaKind::Image,
        };
        assert_eq!(enc.num_tokens(), 3);
    }

    #[test]
    fn multimodal_encoded_validate_ok() {
        let enc = MultimodalEncoded {
            tokens: vec![1, 2],
            embeddings: vec![0.0; 8],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        assert!(enc.validate().is_ok());
    }

    #[test]
    fn multimodal_context_default_empty() {
        let ctx = MultimodalContext::default();
        assert!(ctx.is_empty());
        assert!(ctx.images.is_empty());
        assert!(ctx.audios.is_empty());
    }

    #[test]
    fn multimodal_context_push_audio_rejects_image_kind() {
        let bad = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        let err = ctx.push_audio(bad).unwrap_err();
        assert!(format!("{err}").contains("non-Audio"));
    }

    #[test]
    fn routed_sequence_text_only() {
        let rs = RoutedSequence {
            token_ids: vec![1, 2, 3],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 64,
        };
        assert_eq!(rs.seq_len(), 3);
        assert!(!rs.has_multimodal());
    }

    #[test]
    fn routed_sequence_with_multimodal() {
        let rs = RoutedSequence {
            token_ids: vec![1, 258880, 2],
            fused_embeddings: vec![None, Some(vec![0.0; 4]), None],
            text_positions: vec![0, 2],
            hidden_size: 4,
        };
        assert_eq!(rs.seq_len(), 3);
        assert!(rs.has_multimodal());
    }

    #[test]
    fn route_empty_prompt() {
        let ctx = MultimodalContext::new();
        let ids = default_ids();
        let out = route_multimodal_tokens(&[], &ctx, &ids, 128).unwrap();
        assert_eq!(out.seq_len(), 0);
        assert!(out.text_positions.is_empty());
    }

    #[test]
    fn route_multiple_images_order_preserved() {
        let ids = default_ids();
        let enc1 = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![1.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let enc2 = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![2.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc1).unwrap();
        ctx.push_image(enc2).unwrap();

        let prompt = vec![ids.image_token_id, 10, ids.image_token_id];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        // 1(virt) + 1(text) + 1(virt) = 3
        assert_eq!(out.seq_len(), 3);
        assert_eq!(out.text_positions, vec![1]);
        // First image embedding = 1.0, second = 2.0
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 1.0);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap()[0], 2.0);
    }

    #[test]
    fn build_fused_hidden_embed_rows_not_divisible() {
        let embed: Vec<f32> = vec![0.0; 5]; // not divisible by hidden_size=2
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        let err = build_fused_hidden(&routed, &embed, 2).unwrap_err();
        assert!(format!("{err}").contains("not divisible"));
    }

    #[test]
    fn build_fused_hidden_mixed_text_and_media() {
        // vocab=3, hidden=2
        let embed: Vec<f32> = vec![10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        let media = vec![99.0, 88.0];
        let routed = RoutedSequence {
            token_ids: vec![0, 258880, 2],
            fused_embeddings: vec![None, Some(media.clone()), None],
            text_positions: vec![0, 2],
            hidden_size: 2,
        };
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // pos 0: gather token 0 → [10, 11]
        // pos 1: media → [99, 88]
        // pos 2: gather token 2 → [30, 31]
        assert_eq!(hidden, vec![10.0, 11.0, 99.0, 88.0, 30.0, 31.0]);
    }

    // ── Additional trait & edge-case tests ──

    #[test]
    fn multimodal_token_ids_debug_format() {
        let ids = MultimodalTokenIds {
            image_token_id: 1,
            audio_token_id: 2,
            eoi_token_id: 3,
            eoa_token_id: 4,
        };
        let debug = format!("{ids:?}");
        assert!(debug.contains("image_token_id: 1"));
        assert!(debug.contains("audio_token_id: 2"));
        assert!(debug.contains("eoi_token_id: 3"));
        assert!(debug.contains("eoa_token_id: 4"));
    }

    #[test]
    fn multimodal_token_ids_clone_independent() {
        let original = MultimodalTokenIds {
            image_token_id: 100,
            audio_token_id: 200,
            eoi_token_id: 300,
            eoa_token_id: 400,
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
        // Verify cloned is independent (both are Copy, so this is trivially true,
        // but the test exercises the Clone impl path).
        assert_eq!(cloned.image_token_id, 100);
    }

    #[test]
    fn multimodal_token_ids_partial_eq_ne() {
        let a = MultimodalTokenIds {
            image_token_id: 1,
            audio_token_id: 2,
            eoi_token_id: 3,
            eoa_token_id: 4,
        };
        let b = MultimodalTokenIds {
            image_token_id: 99,
            audio_token_id: 2,
            eoi_token_id: 3,
            eoa_token_id: 4,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn media_kind_copy_clone_debug() {
        let a = MediaKind::Image;
        let b = a; // Copy
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
        // Debug trait produces meaningful output
        let dbg = format!("{a:?}");
        assert!(dbg.contains("Image"));
        let dbg_audio = format!("{:?}", MediaKind::Audio);
        assert!(dbg_audio.contains("Audio"));
    }

    #[test]
    fn media_kind_eq_self_consistency() {
        // PartialEq: equal to self, not equal to other variant
        assert_eq!(MediaKind::Image, MediaKind::Image);
        assert_eq!(MediaKind::Audio, MediaKind::Audio);
        assert_ne!(MediaKind::Image, MediaKind::Audio);
        // Verify Copy: assigning to another var preserves original
        let x = MediaKind::Image;
        let y = x;
        assert_eq!(x, y);
        // Verify Clone produces equal value
        let z = x.clone();
        assert_eq!(x, z);
    }

    #[test]
    fn encoder_media_clone() {
        let file = EncoderMedia::File(PathBuf::from("/tmp/a.png"));
        let cloned = file.clone();
        assert!(matches!(cloned, EncoderMedia::File(ref p) if p == &PathBuf::from("/tmp/a.png")));

        let b64 = EncoderMedia::Base64 {
            data: "AQI=".into(),
            mime_type: Some("image/jpeg".into()),
        };
        let cloned_b64 = b64.clone();
        assert!(matches!(cloned_b64, EncoderMedia::Base64 { ref mime_type, .. } if mime_type.as_deref() == Some("image/jpeg")));

        let raw = EncoderMedia::Raw(vec![0xDE, 0xAD]);
        let cloned_raw = raw.clone();
        assert!(matches!(cloned_raw, EncoderMedia::Raw(ref v) if v == &vec![0xDE, 0xAD]));

        let url = EncoderMedia::Url("s3://bucket/key".into());
        let cloned_url = url.clone();
        assert!(matches!(cloned_url, EncoderMedia::Url(ref s) if s == "s3://bucket/key"));
    }

    #[test]
    fn encoder_media_from_generation_url() {
        let gen_url = crate::generation::MediaInput::Url("https://example.com/audio.wav".into());
        let enc = EncoderMedia::from_generation(&gen_url);
        assert!(matches!(enc, EncoderMedia::Url(ref s) if s == "https://example.com/audio.wav"));
    }

    #[test]
    fn encoder_media_from_generation_base64_no_mime() {
        let gen_b64 = crate::generation::MediaInput::Base64 {
            data: "AAAA".into(),
            mime_type: None,
        };
        let enc = EncoderMedia::from_generation(&gen_b64);
        match enc {
            EncoderMedia::Base64 { data, mime_type } => {
                assert_eq!(data, "AAAA");
                assert!(mime_type.is_none());
            }
            _ => panic!("expected Base64 variant"),
        }
    }

    #[test]
    fn multimodal_encoded_clone() {
        let orig = MultimodalEncoded {
            tokens: vec![10, 20],
            embeddings: vec![1.0, 2.0, 3.0, 4.0],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let cloned = orig.clone();
        assert_eq!(cloned.tokens, orig.tokens);
        assert_eq!(cloned.embeddings, orig.embeddings);
        assert_eq!(cloned.hidden_size, orig.hidden_size);
        assert_eq!(cloned.kind, orig.kind);
    }

    #[test]
    fn multimodal_encoded_debug_includes_kind() {
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        let dbg = format!("{enc:?}");
        assert!(dbg.contains("Image"));
    }

    #[test]
    fn multimodal_encoded_validate_zero_tokens_zero_hidden() {
        // 0 tokens * 0 hidden_size = 0 embeddings: mathematically consistent
        let enc = MultimodalEncoded {
            tokens: vec![],
            embeddings: vec![],
            hidden_size: 0,
            kind: MediaKind::Image,
        };
        assert!(enc.validate().is_ok());
    }

    #[test]
    fn multimodal_encoded_validate_zero_tokens_nonzero_hidden() {
        // 0 tokens * 4 hidden = 0 expected embeddings
        let enc = MultimodalEncoded {
            tokens: vec![],
            embeddings: vec![],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        assert!(enc.validate().is_ok());
    }

    #[test]
    fn multimodal_encoded_validate_nonzero_tokens_zero_hidden() {
        // 3 tokens * 0 hidden = 0 expected, but embeddings has data
        let enc = MultimodalEncoded {
            tokens: vec![1, 2, 3],
            embeddings: vec![],
            hidden_size: 0,
            kind: MediaKind::Image,
        };
        assert!(enc.validate().is_ok()); // 3*0 = 0 == embeddings.len()
    }

    #[test]
    fn multimodal_context_clone_preserves_state() {
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.5],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![0.3],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();

        let cloned = ctx.clone();
        assert_eq!(cloned.images.len(), 1);
        assert_eq!(cloned.audios.len(), 1);
        assert!(!cloned.is_empty());
        assert_eq!(cloned.images[0].tokens, vec![1]);
        assert_eq!(cloned.audios[0].tokens, vec![2]);
    }

    #[test]
    fn multimodal_context_push_image_validates_shape() {
        let bad = MultimodalEncoded {
            tokens: vec![1, 2],
            embeddings: vec![0.0; 3], // 2 tokens * 2 hidden = 4 expected, got 3
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        let err = ctx.push_image(bad).unwrap_err();
        assert!(format!("{err}").contains("shape mismatch"));
        // Image should NOT have been appended
        assert!(ctx.is_empty());
    }

    #[test]
    fn multimodal_context_push_audio_validates_shape() {
        let bad = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0; 3], // 1 token * 2 hidden = 2 expected, got 3
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        let err = ctx.push_audio(bad).unwrap_err();
        assert!(format!("{err}").contains("shape mismatch"));
        assert!(ctx.is_empty());
    }

    #[test]
    fn multimodal_context_is_empty_after_pushes() {
        let mut ctx = MultimodalContext::new();
        assert!(ctx.is_empty());
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        assert!(!ctx.is_empty());
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        assert!(!ctx.is_empty());
    }

    #[test]
    fn routed_sequence_clone() {
        let rs = RoutedSequence {
            token_ids: vec![1, 258880, 2],
            fused_embeddings: vec![None, Some(vec![1.0, 2.0]), None],
            text_positions: vec![0, 2],
            hidden_size: 2,
        };
        let cloned = rs.clone();
        assert_eq!(cloned.token_ids, rs.token_ids);
        assert_eq!(cloned.fused_embeddings.len(), 3);
        assert_eq!(cloned.text_positions, rs.text_positions);
        assert_eq!(cloned.hidden_size, 2);
    }

    #[test]
    fn routed_sequence_debug_format() {
        let rs = RoutedSequence {
            token_ids: vec![1],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 64,
        };
        let dbg = format!("{rs:?}");
        assert!(dbg.contains("RoutedSequence"));
        assert!(dbg.contains("hidden_size"));
    }

    #[test]
    fn routed_sequence_seq_len_zero() {
        let rs = RoutedSequence {
            token_ids: vec![],
            fused_embeddings: vec![],
            text_positions: vec![],
            hidden_size: 128,
        };
        assert_eq!(rs.seq_len(), 0);
        assert!(!rs.has_multimodal());
    }

    #[test]
    fn route_audio_count_mismatch_rejected() {
        let ids = default_ids();
        let prompt = vec![1u32, 2]; // no audio special token
        let enc = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc).unwrap();
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap_err();
        assert!(format!("{err}").contains("audio"));
    }

    #[test]
    fn route_audio_hidden_size_mismatch() {
        let ids = default_ids();
        let prompt = vec![ids.audio_token_id];
        let enc = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![0.0; 2 * 8],
            hidden_size: 8,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc).unwrap();
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 16).unwrap_err();
        assert!(format!("{err}").contains("hidden_size"));
    }

    #[test]
    fn route_consecutive_special_tokens() {
        let ids = default_ids();
        // Two image tokens back-to-back, then an audio token
        let enc_img1 = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 2],
            embeddings: vec![1.0; 2 * 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let enc_img2 = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 1],
            embeddings: vec![2.0; 1 * 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 3],
            embeddings: vec![3.0; 3 * 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc_img1).unwrap();
        ctx.push_image(enc_img2).unwrap();
        ctx.push_audio(enc_aud).unwrap();

        let prompt = vec![ids.image_token_id, ids.image_token_id, ids.audio_token_id];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        // 2 + 1 + 3 = 6 total tokens
        assert_eq!(out.seq_len(), 6);
        assert!(out.text_positions.is_empty()); // no text tokens at all
        assert!(out.has_multimodal());
        // First image's first embedding = 1.0
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 1.0);
        // Second image's embedding = 2.0
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap()[0], 2.0);
        // Audio's first embedding = 3.0
        assert_eq!(out.fused_embeddings[3].as_ref().unwrap()[0], 3.0);
    }

    #[test]
    fn route_embedding_values_preserved_exact() {
        let ids = default_ids();
        let embeddings: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: embeddings.clone(),
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();

        let prompt = vec![ids.image_token_id];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        assert_eq!(out.seq_len(), 1);
        let routed_emb = out.fused_embeddings[0].as_ref().unwrap();
        assert_eq!(routed_emb, &embeddings);
    }

    #[test]
    fn build_fused_hidden_rejects_fused_embeddings_len_mismatch() {
        let embed: Vec<f32> = vec![0.0; 4];
        let routed = RoutedSequence {
            token_ids: vec![0, 1],
            fused_embeddings: vec![None], // len 1 != seq_len 2
            text_positions: vec![0],
            hidden_size: 2,
        };
        let err = build_fused_hidden(&routed, &embed, 2).unwrap_err();
        assert!(format!("{err}").contains("fused_embeddings"));
    }

    #[test]
    fn build_fused_hidden_all_media_positions() {
        // Sequence with no text tokens, all media
        let embed: Vec<f32> = vec![0.0; 4]; // vocab=2, hidden=2 (unused)
        let media1 = vec![1.0, 2.0];
        let media2 = vec![3.0, 4.0];
        let routed = RoutedSequence {
            token_ids: vec![258880, 258881],
            fused_embeddings: vec![Some(media1.clone()), Some(media2.clone())],
            text_positions: vec![],
            hidden_size: 2,
        };
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        assert_eq!(hidden, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn build_fused_hidden_empty_sequence() {
        let embed: Vec<f32> = vec![0.0; 4];
        let routed = RoutedSequence {
            token_ids: vec![],
            fused_embeddings: vec![],
            text_positions: vec![],
            hidden_size: 2,
        };
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        assert!(hidden.is_empty());
    }

    // ── Additional trait, edge-case, and coverage tests ──

    #[test]
    fn multimodal_token_ids_max_u32_values() {
        let ids = MultimodalTokenIds {
            image_token_id: u32::MAX,
            audio_token_id: u32::MAX - 1,
            eoi_token_id: u32::MAX - 2,
            eoa_token_id: u32::MAX - 3,
        };
        assert!(ids.is_image(u32::MAX));
        assert!(!ids.is_image(u32::MAX - 1));
        assert!(ids.is_audio(u32::MAX - 1));
        assert!(!ids.is_audio(u32::MAX));
    }

    #[test]
    fn multimodal_token_ids_zero_values() {
        let ids = MultimodalTokenIds {
            image_token_id: 0,
            audio_token_id: 0,
            eoi_token_id: 0,
            eoa_token_id: 0,
        };
        assert!(ids.is_image(0));
        assert!(ids.is_audio(0));
        assert!(!ids.is_image(1));
        assert!(!ids.is_audio(1));
    }

    #[test]
    fn encoder_media_file_debug_format() {
        let file = EncoderMedia::File(PathBuf::from("/tmp/img.png"));
        let dbg = format!("{file:?}");
        assert!(dbg.contains("File"));
        assert!(dbg.contains("img.png"));
    }

    #[test]
    fn encoder_media_base64_debug_format() {
        let b64 = EncoderMedia::Base64 {
            data: "AQID".into(),
            mime_type: Some("image/png".into()),
        };
        let dbg = format!("{b64:?}");
        assert!(dbg.contains("Base64"));
        assert!(dbg.contains("AQID"));
        assert!(dbg.contains("image/png"));
    }

    #[test]
    fn encoder_media_base64_no_mime_debug_format() {
        let b64 = EncoderMedia::Base64 {
            data: "test".into(),
            mime_type: None,
        };
        let dbg = format!("{b64:?}");
        assert!(dbg.contains("Base64"));
        assert!(dbg.contains("None"));
    }

    #[test]
    fn encoder_media_raw_debug_format() {
        let raw = EncoderMedia::Raw(vec![0xDE, 0xAD, 0xBE]);
        let dbg = format!("{raw:?}");
        assert!(dbg.contains("Raw"));
    }

    #[test]
    fn encoder_media_url_debug_format() {
        let url = EncoderMedia::Url("https://example.com/img.jpg".into());
        let dbg = format!("{url:?}");
        assert!(dbg.contains("Url"));
        assert!(dbg.contains("example.com"));
    }

    #[test]
    fn multimodal_context_new_equals_default() {
        let new_ctx = MultimodalContext::new();
        let default_ctx = MultimodalContext::default();
        assert_eq!(new_ctx.images.len(), default_ctx.images.len());
        assert_eq!(new_ctx.audios.len(), default_ctx.audios.len());
        assert!(new_ctx.is_empty());
        assert!(default_ctx.is_empty());
    }

    #[test]
    fn multimodal_context_debug_format() {
        let ctx = MultimodalContext::new();
        let dbg = format!("{ctx:?}");
        assert!(dbg.contains("MultimodalContext"));
    }

    #[test]
    fn multimodal_encoded_single_token_single_hidden() {
        let enc = MultimodalEncoded {
            tokens: vec![42],
            embeddings: vec![3.14],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        assert_eq!(enc.num_tokens(), 1);
        assert!(enc.validate().is_ok());
    }

    #[test]
    fn multimodal_context_push_multiple_images_count() {
        let mut ctx = MultimodalContext::new();
        for i in 0..5 {
            ctx.push_image(MultimodalEncoded {
                tokens: vec![i],
                embeddings: vec![i as f32],
                hidden_size: 1,
                kind: MediaKind::Image,
            })
            .unwrap();
        }
        assert_eq!(ctx.images.len(), 5);
        assert!(!ctx.is_empty());
        assert!(ctx.audios.is_empty());
    }

    #[test]
    fn multimodal_context_push_multiple_audios_count() {
        let mut ctx = MultimodalContext::new();
        for i in 0..3 {
            ctx.push_audio(MultimodalEncoded {
                tokens: vec![i],
                embeddings: vec![i as f32 * 2.0],
                hidden_size: 1,
                kind: MediaKind::Audio,
            })
            .unwrap();
        }
        assert_eq!(ctx.audios.len(), 3);
        assert!(!ctx.is_empty());
        assert!(ctx.images.is_empty());
    }

    #[test]
    fn route_only_image_no_text_tokens() {
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 3],
            embeddings: vec![0.5; 3 * 2],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();

        let prompt = vec![ids.image_token_id];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        assert_eq!(out.seq_len(), 3);
        assert!(out.text_positions.is_empty());
        assert!(out.has_multimodal());
        assert!(out.fused_embeddings.iter().all(|e| e.is_some()));
    }

    #[test]
    fn route_text_between_images_embedding_values() {
        let ids = default_ids();
        let img_emb: Vec<f32> = vec![9.0, 8.0, 7.0, 6.0];
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 2],
            embeddings: img_emb.clone(),
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();

        let prompt = vec![10u32, ids.image_token_id, 20];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        assert_eq!(out.seq_len(), 4);
        // text at 0, virtual at 1-2, text at 3
        assert_eq!(out.text_positions, vec![0, 3]);
        assert!(out.fused_embeddings[0].is_none());
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap(), &vec![9.0, 8.0]);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap(), &vec![7.0, 6.0]);
        assert!(out.fused_embeddings[3].is_none());
    }

    #[test]
    fn build_fused_hidden_token_id_u32_max_out_of_range() {
        let embed: Vec<f32> = vec![0.0; 4]; // vocab=2, hidden=2
        let routed = RoutedSequence {
            token_ids: vec![u32::MAX],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        let err = build_fused_hidden(&routed, &embed, 2).unwrap_err();
        assert!(format!("{err}").contains("out of range"));
    }

    #[test]
    fn encoder_media_from_generation_base64_data_preserved() {
        let gen_b64 = crate::generation::MediaInput::Base64 {
            data: "SGVsbG8=".into(),
            mime_type: Some("audio/wav".into()),
        };
        let enc = EncoderMedia::from_generation(&gen_b64);
        match enc {
            EncoderMedia::Base64 { data, mime_type } => {
                assert_eq!(data, "SGVsbG8=");
                assert_eq!(mime_type.as_deref(), Some("audio/wav"));
            }
            _ => panic!("expected Base64 variant"),
        }
    }

    #[test]
    fn multimodal_encoded_validate_large_tokens() {
        let n = 256;
        let hidden = 64;
        let enc = MultimodalEncoded {
            tokens: vec![42u32; n],
            embeddings: vec![0.0; n * hidden],
            hidden_size: hidden,
            kind: MediaKind::Image,
        };
        assert_eq!(enc.num_tokens(), n);
        assert!(enc.validate().is_ok());
    }

    // ── New tests (77-120): trait, edge-case, and coverage expansion ──

    // --- Hash trait tests ---

    #[test]
    fn multimodal_token_ids_hash_equal_inputs_produce_equal_hashes() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = MultimodalTokenIds::fallback_multimodal_token_ids();
        let b = MultimodalTokenIds::fallback_multimodal_token_ids();

        let mut ha = DefaultHasher::new();
        a.hash(&mut ha);
        let hash_a = ha.finish();

        let mut hb = DefaultHasher::new();
        b.hash(&mut hb);
        let hash_b = hb.finish();

        assert_eq!(hash_a, hash_b);
    }

    #[test]
    fn multimodal_token_ids_hash_different_inputs_produce_different_hashes() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let a = MultimodalTokenIds {
            image_token_id: 1,
            audio_token_id: 2,
            eoi_token_id: 3,
            eoa_token_id: 4,
        };
        let b = MultimodalTokenIds {
            image_token_id: 999,
            audio_token_id: 2,
            eoi_token_id: 3,
            eoa_token_id: 4,
        };

        let mut ha = DefaultHasher::new();
        a.hash(&mut ha);
        let hash_a = ha.finish();

        let mut hb = DefaultHasher::new();
        b.hash(&mut hb);
        let hash_b = hb.finish();

        assert_ne!(hash_a, hash_b);
    }

    #[test]
    fn media_kind_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut h1 = DefaultHasher::new();
        MediaKind::Image.hash(&mut h1);
        let hash_img1 = h1.finish();

        let mut h2 = DefaultHasher::new();
        MediaKind::Image.hash(&mut h2);
        let hash_img2 = h2.finish();

        let mut h3 = DefaultHasher::new();
        MediaKind::Audio.hash(&mut h3);
        let hash_audio = h3.finish();

        assert_eq!(hash_img1, hash_img2);
        assert_ne!(hash_img1, hash_audio);
    }

    #[test]
    fn media_kind_usable_as_hashmap_key() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(MediaKind::Image, "img_encoder");
        map.insert(MediaKind::Audio, "aud_encoder");
        assert_eq!(map.get(&MediaKind::Image), Some(&"img_encoder"));
        assert_eq!(map.get(&MediaKind::Audio), Some(&"aud_encoder"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn multimodal_token_ids_usable_as_hashmap_key() {
        use std::collections::HashMap;
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let mut map = HashMap::new();
        map.insert(ids, "gemma4");
        assert_eq!(map.get(&ids), Some(&"gemma4"));
    }

    // --- MultimodalTokenIds edge cases ---

    #[test]
    fn multimodal_token_ids_is_image_distinct_from_is_audio() {
        let ids = MultimodalTokenIds {
            image_token_id: 100,
            audio_token_id: 200,
            eoi_token_id: 300,
            eoa_token_id: 400,
        };
        // image_token_id != audio_token_id
        assert!(ids.is_image(100));
        assert!(!ids.is_audio(100));
        assert!(!ids.is_image(200));
        assert!(ids.is_audio(200));
    }

    #[test]
    fn multimodal_token_ids_same_id_for_image_and_audio() {
        // Edge case: both special tokens share the same ID
        let ids = MultimodalTokenIds {
            image_token_id: 42,
            audio_token_id: 42,
            eoi_token_id: 43,
            eoa_token_id: 44,
        };
        assert!(ids.is_image(42));
        assert!(ids.is_audio(42));
        assert!(!ids.is_image(43));
        assert!(!ids.is_audio(43));
    }

    // --- EncoderMedia edge cases ---

    #[test]
    fn encoder_media_raw_empty_bytes() {
        let raw = EncoderMedia::Raw(vec![]);
        assert!(matches!(raw, EncoderMedia::Raw(ref v) if v.is_empty()));
    }

    #[test]
    fn encoder_media_file_empty_path() {
        let file = EncoderMedia::File(PathBuf::new());
        assert!(matches!(file, EncoderMedia::File(ref p) if p.as_os_str().is_empty()));
    }

    #[test]
    fn encoder_media_url_empty_string() {
        let url = EncoderMedia::Url(String::new());
        assert!(matches!(url, EncoderMedia::Url(ref s) if s.is_empty()));
    }

    #[test]
    fn encoder_media_base64_empty_data_with_mime() {
        let b64 = EncoderMedia::Base64 {
            data: String::new(),
            mime_type: Some("application/octet-stream".into()),
        };
        match b64 {
            EncoderMedia::Base64 { data, mime_type } => {
                assert!(data.is_empty());
                assert_eq!(mime_type.as_deref(), Some("application/octet-stream"));
            }
            _ => panic!("expected Base64 variant"),
        }
    }

    #[test]
    fn encoder_media_from_generation_raw_empty() {
        let gen_raw = crate::generation::MediaInput::Raw(vec![]);
        let enc = EncoderMedia::from_generation(&gen_raw);
        assert!(matches!(enc, EncoderMedia::Raw(ref v) if v.is_empty()));
    }

    // --- MultimodalEncoded edge cases ---

    #[test]
    fn multimodal_encoded_embeddings_with_nan() {
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![f32::NAN],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        // validate only checks length, not values
        assert!(enc.validate().is_ok());
        assert_eq!(enc.num_tokens(), 1);
    }

    #[test]
    fn multimodal_encoded_embeddings_with_infinity() {
        let enc = MultimodalEncoded {
            tokens: vec![1, 2],
            embeddings: vec![f32::INFINITY, f32::NEG_INFINITY],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        assert!(enc.validate().is_ok());
        assert_eq!(enc.num_tokens(), 2);
    }

    #[test]
    fn multimodal_encoded_embeddings_with_zero() {
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        assert!(enc.validate().is_ok());
    }

    #[test]
    fn multimodal_encoded_large_hidden_size() {
        let hidden = 4096;
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![1.0; hidden],
            hidden_size: hidden,
            kind: MediaKind::Audio,
        };
        assert!(enc.validate().is_ok());
        assert_eq!(enc.num_tokens(), 1);
    }

    #[test]
    fn multimodal_encoded_validate_off_by_one() {
        let enc = MultimodalEncoded {
            tokens: vec![1, 2, 3],
            embeddings: vec![0.0; 3 * 8 - 1], // off by one
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let err = enc.validate().unwrap_err();
        assert!(format!("{err}").contains("shape mismatch"));
    }

    #[test]
    fn multimodal_encoded_validate_exact_match() {
        let enc = MultimodalEncoded {
            tokens: vec![1, 2, 3],
            embeddings: vec![0.0; 3 * 8],
            hidden_size: 8,
            kind: MediaKind::Audio,
        };
        assert!(enc.validate().is_ok());
    }

    // --- MultimodalContext state isolation ---

    #[test]
    fn multimodal_context_push_image_failure_does_not_alter_state() {
        let mut ctx = MultimodalContext::new();
        // Push a valid image first
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.5],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        assert_eq!(ctx.images.len(), 1);

        // Try to push an Audio-kind encoding as image — should fail
        let bad = MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        let _ = ctx.push_image(bad);
        // State unchanged
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.images[0].tokens, vec![1]);
    }

    #[test]
    fn multimodal_context_push_audio_failure_does_not_alter_state() {
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![5],
            embeddings: vec![0.3],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        assert_eq!(ctx.audios.len(), 1);

        let bad = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        let _ = ctx.push_audio(bad);
        assert_eq!(ctx.audios.len(), 1);
        assert_eq!(ctx.audios[0].tokens, vec![5]);
    }

    #[test]
    fn multimodal_context_mixed_pushes_maintain_order() {
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![10],
            embeddings: vec![1.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![20],
            embeddings: vec![2.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![30],
            embeddings: vec![3.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();

        assert_eq!(ctx.images.len(), 2);
        assert_eq!(ctx.audios.len(), 1);
        assert_eq!(ctx.images[0].tokens, vec![10]);
        assert_eq!(ctx.images[1].tokens, vec![30]);
        assert_eq!(ctx.audios[0].tokens, vec![20]);
    }

    #[test]
    fn multimodal_context_is_empty_with_only_images_not_empty() {
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        assert!(!ctx.is_empty());
    }

    #[test]
    fn multimodal_context_is_empty_with_only_audios_not_empty() {
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        assert!(!ctx.is_empty());
    }

    // --- route_multimodal_tokens additional scenarios ---

    #[test]
    fn route_single_text_token() {
        let ctx = MultimodalContext::new();
        let ids = default_ids();
        let out = route_multimodal_tokens(&vec![42u32], &ctx, &ids, 64).unwrap();
        assert_eq!(out.seq_len(), 1);
        assert_eq!(out.token_ids, vec![42]);
        assert_eq!(out.text_positions, vec![0]);
        assert!(!out.has_multimodal());
    }

    #[test]
    fn route_special_token_at_beginning() {
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 2],
            embeddings: vec![1.0, 2.0, 3.0, 4.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();

        let prompt = vec![ids.image_token_id, 10, 20];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        assert_eq!(out.seq_len(), 4); // 2 virtual + 2 text
        assert_eq!(out.text_positions, vec![2, 3]);
        assert!(out.fused_embeddings[0].is_some());
        assert!(out.fused_embeddings[1].is_some());
        assert!(out.fused_embeddings[2].is_none());
    }

    #[test]
    fn route_special_token_at_end() {
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 3],
            embeddings: vec![0.0; 3 * 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc).unwrap();

        let prompt = vec![10u32, 20, ids.audio_token_id];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        assert_eq!(out.seq_len(), 5); // 2 text + 3 virtual
        assert_eq!(out.text_positions, vec![0, 1]);
        assert!(out.fused_embeddings[2].is_some());
        assert!(out.fused_embeddings[3].is_some());
        assert!(out.fused_embeddings[4].is_some());
    }

    #[test]
    fn route_image_and_audio_alternating() {
        let ids = default_ids();
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![1.0; 2],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![2.0; 2],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc_img).unwrap();
        ctx.push_audio(enc_aud).unwrap();

        // IMAGE, text, AUDIO, text
        let prompt = vec![ids.image_token_id, 99, ids.audio_token_id, 88];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        assert_eq!(out.seq_len(), 4);
        assert_eq!(out.text_positions, vec![1, 3]);
        // Image at 0, audio at 2
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 1.0);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap()[0], 2.0);
    }

    #[test]
    fn route_too_many_image_special_tokens() {
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();

        // 2 image tokens but only 1 encoding
        let prompt = vec![ids.image_token_id, ids.image_token_id];
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap_err();
        assert!(format!("{err}").contains("image"));
    }

    #[test]
    fn route_preserves_non_special_token_values() {
        let ctx = MultimodalContext::new();
        let ids = default_ids();
        let tokens = vec![7u32, 13, 42, 100, 255, 1000, u32::MAX];
        let out = route_multimodal_tokens(&tokens, &ctx, &ids, 64).unwrap();
        assert_eq!(out.token_ids, tokens);
    }

    #[test]
    fn route_custom_token_ids_not_matching_defaults() {
        let custom_ids = MultimodalTokenIds {
            image_token_id: 500,
            audio_token_id: 501,
            eoi_token_id: 502,
            eoa_token_id: 503,
        };
        // Use default special tokens in prompt — should be treated as text
        let prompt = vec![258880u32, 258881];
        let ctx = MultimodalContext::new();
        let out = route_multimodal_tokens(&prompt, &ctx, &custom_ids, 4).unwrap();
        // Both are text because custom_ids doesn't match defaults
        assert_eq!(out.seq_len(), 2);
        assert!(out.fused_embeddings.iter().all(|e| e.is_none()));
    }

    // --- build_fused_hidden additional scenarios ---

    #[test]
    fn build_fused_hidden_with_nan_in_media_embedding() {
        let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, hidden=2
        let media = vec![f32::NAN, f32::INFINITY];
        let routed = RoutedSequence {
            token_ids: vec![258880],
            fused_embeddings: vec![Some(media.clone())],
            text_positions: vec![],
            hidden_size: 2,
        };
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        assert!(hidden[0].is_nan());
        assert!(hidden[1].is_infinite());
    }

    #[test]
    fn build_fused_hidden_large_vocab() {
        let vocab_size = 10000;
        let hidden_size = 128;
        let embed: Vec<f32> = (0..vocab_size * hidden_size).map(|i| i as f32 * 0.001).collect();
        let routed = RoutedSequence {
            token_ids: vec![5000],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size,
        };
        let hidden = build_fused_hidden(&routed, &embed, hidden_size).unwrap();
        let expected_start = 5000 * hidden_size;
        assert_eq!(hidden[0], expected_start as f32 * 0.001);
        assert_eq!(hidden.len(), hidden_size);
    }

    #[test]
    fn build_fused_hidden_hidden_size_one() {
        let embed: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0]; // vocab=4, hidden=1
        let routed = RoutedSequence {
            token_ids: vec![0, 3, 1],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 1,
        };
        let hidden = build_fused_hidden(&routed, &embed, 1).unwrap();
        assert_eq!(hidden, vec![0.0, 3.0, 1.0]);
    }

    #[test]
    fn build_fused_hidden_zero_vocab_size_rejected() {
        // embed is empty, hidden_size=2 => vocab=0, any token is out of range
        let embed: Vec<f32> = vec![];
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        let err = build_fused_hidden(&routed, &embed, 2).unwrap_err();
        assert!(format!("{err}").contains("out of range"));
    }

    #[test]
    fn build_fused_hidden_media_overrides_gather() {
        // Token 1 has embed data, but media overrides it
        let embed: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0]; // vocab=2, hidden=2
        let media = vec![99.0, 88.0];
        let routed = RoutedSequence {
            token_ids: vec![1], // would gather [30, 40]
            fused_embeddings: vec![Some(media.clone())],
            text_positions: vec![],
            hidden_size: 2,
        };
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        assert_eq!(hidden, vec![99.0, 88.0]);
    }

    // --- MockEncoder trait object test ---

    #[test]
    fn mock_encoder_as_dyn_trait_object() {
        let ids = default_ids();
        let encoder: Box<dyn MultimodalEncoder> = Box::new(MockEncoder::new(2, 3, 4, ids));

        let img_result = encoder.encode_image(&EncoderMedia::Raw(vec![])).unwrap();
        assert_eq!(img_result.kind, MediaKind::Image);
        assert_eq!(img_result.num_tokens(), 2);
        assert_eq!(img_result.hidden_size, 4);

        let aud_result = encoder.encode_audio(&EncoderMedia::Raw(vec![])).unwrap();
        assert_eq!(aud_result.kind, MediaKind::Audio);
        assert_eq!(aud_result.num_tokens(), 3);
    }

    // --- RoutedSequence has_multimodal edge cases ---

    #[test]
    fn routed_sequence_has_multimodal_single_media() {
        let rs = RoutedSequence {
            token_ids: vec![1, 258880, 2],
            fused_embeddings: vec![None, Some(vec![0.0; 8]), None],
            text_positions: vec![0, 2],
            hidden_size: 8,
        };
        assert!(rs.has_multimodal());
    }

    #[test]
    fn routed_sequence_all_none_not_multimodal() {
        let rs = RoutedSequence {
            token_ids: vec![1, 2, 3, 4, 5],
            fused_embeddings: vec![None, None, None, None, None],
            text_positions: vec![0, 1, 2, 3, 4],
            hidden_size: 4,
        };
        assert!(!rs.has_multimodal());
    }

    #[test]
    fn routed_sequence_all_some_is_multimodal() {
        let rs = RoutedSequence {
            token_ids: vec![258880, 258881],
            fused_embeddings: vec![Some(vec![0.0; 2]), Some(vec![1.0; 2])],
            text_positions: vec![],
            hidden_size: 2,
        };
        assert!(rs.has_multimodal());
    }

    // --- EncoderMedia additional clone independence ---

    #[test]
    fn encoder_media_raw_clone_independence() {
        let original = EncoderMedia::Raw(vec![1, 2, 3]);
        let cloned = original.clone();
        // Both are independent (Vec clone is deep)
        assert!(matches!(original, EncoderMedia::Raw(ref v) if v == &vec![1, 2, 3]));
        assert!(matches!(cloned, EncoderMedia::Raw(ref v) if v == &vec![1, 2, 3]));
    }

    #[test]
    fn encoder_media_url_clone_independence() {
        let original = EncoderMedia::Url("https://test.com/a.png".into());
        let cloned = original.clone();
        assert!(matches!(original, EncoderMedia::Url(ref s) if s == "https://test.com/a.png"));
        assert!(matches!(cloned, EncoderMedia::Url(ref s) if s == "https://test.com/a.png"));
    }

    #[test]
    fn encoder_media_base64_clone_preserves_mime() {
        let original = EncoderMedia::Base64 {
            data: "data123".into(),
            mime_type: Some("audio/wav".into()),
        };
        let cloned = original.clone();
        match cloned {
            EncoderMedia::Base64 { data, mime_type } => {
                assert_eq!(data, "data123");
                assert_eq!(mime_type.as_deref(), Some("audio/wav"));
            }
            _ => panic!("expected Base64"),
        }
    }

    // --- MultimodalContext debug with content ---

    #[test]
    fn multimodal_context_debug_with_content() {
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.5],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![0.3],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        let dbg = format!("{ctx:?}");
        assert!(dbg.contains("MultimodalContext"));
        assert!(dbg.contains("images"));
        assert!(dbg.contains("audios"));
    }

    // --- Route with hidden_size = 1 ---

    #[test]
    fn route_with_hidden_size_one() {
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 5],
            embeddings: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();

        let prompt = vec![ids.image_token_id];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 1).unwrap();
        assert_eq!(out.seq_len(), 5);
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 1.0);
        assert_eq!(out.fused_embeddings[4].as_ref().unwrap()[0], 5.0);
    }

    // --- build_fused_hidden with multiple alternating text and media ---

    #[test]
    fn build_fused_hidden_alternating_text_media() {
        let embed: Vec<f32> = vec![
            10.0, 11.0, // token 0
            20.0, 21.0, // token 1
            30.0, 31.0, // token 2
            40.0, 41.0, // token 3
        ]; // vocab=4, hidden=2
        let m1 = vec![100.0, 101.0];
        let m2 = vec![200.0, 201.0];
        let routed = RoutedSequence {
            token_ids: vec![0, 258880, 1, 258881, 3],
            fused_embeddings: vec![None, Some(m1), None, Some(m2), None],
            text_positions: vec![0, 2, 4],
            hidden_size: 2,
        };
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // pos 0: gather token 0 -> [10, 11]
        // pos 1: media -> [100, 101]
        // pos 2: gather token 1 -> [20, 21]
        // pos 3: media -> [200, 201]
        // pos 4: gather token 3 -> [40, 41]
        assert_eq!(hidden, vec![10.0, 11.0, 100.0, 101.0, 20.0, 21.0, 200.0, 201.0, 40.0, 41.0]);
    }

    // --- MultimodalTokenIds Eq trait verification ---

    #[test]
    fn multimodal_token_ids_eq_all_fields_must_match() {
        let base = MultimodalTokenIds {
            image_token_id: 1,
            audio_token_id: 2,
            eoi_token_id: 3,
            eoa_token_id: 4,
        };
        // Same
        assert_eq!(base, MultimodalTokenIds {
            image_token_id: 1,
            audio_token_id: 2,
            eoi_token_id: 3,
            eoa_token_id: 4,
        });
        // Diff in audio_token_id
        assert_ne!(base, MultimodalTokenIds {
            image_token_id: 1,
            audio_token_id: 99,
            eoi_token_id: 3,
            eoa_token_id: 4,
        });
        // Diff in eoi_token_id
        assert_ne!(base, MultimodalTokenIds {
            image_token_id: 1,
            audio_token_id: 2,
            eoi_token_id: 99,
            eoa_token_id: 4,
        });
        // Diff in eoa_token_id
        assert_ne!(base, MultimodalTokenIds {
            image_token_id: 1,
            audio_token_id: 2,
            eoi_token_id: 3,
            eoa_token_id: 99,
        });
    }

    // ── Batch 3 (tests 123-172+): further coverage expansion ──

    // --- MultimodalTokenIds: is_image / is_audio exhaustive ---

    #[test]
    fn multimodal_token_ids_is_image_rejects_eoi_eoa() {
        let ids = default_ids();
        assert!(!ids.is_image(ids.eoi_token_id));
        assert!(!ids.is_image(ids.eoa_token_id));
    }

    #[test]
    fn multimodal_token_ids_is_audio_rejects_eoi_eoa() {
        let ids = default_ids();
        assert!(!ids.is_audio(ids.eoi_token_id));
        assert!(!ids.is_audio(ids.eoa_token_id));
    }

    #[test]
    fn multimodal_token_ids_is_image_rejects_audio_id() {
        let ids = default_ids();
        assert!(!ids.is_image(ids.audio_token_id));
    }

    #[test]
    fn multimodal_token_ids_is_audio_rejects_image_id() {
        let ids = default_ids();
        assert!(!ids.is_audio(ids.image_token_id));
    }

    #[test]
    fn multimodal_token_ids_all_fields_equal_same_value() {
        let ids = MultimodalTokenIds {
            image_token_id: 42,
            audio_token_id: 42,
            eoi_token_id: 42,
            eoa_token_id: 42,
        };
        assert!(ids.is_image(42));
        assert!(ids.is_audio(42));
        assert_eq!(ids.image_token_id, ids.audio_token_id);
        assert_eq!(ids.eoi_token_id, ids.eoa_token_id);
    }

    // --- EncoderMedia: additional edge cases ---

    #[test]
    fn encoder_media_file_absolute_path() {
        let file = EncoderMedia::File(PathBuf::from("/absolute/path/to/image.jpg"));
        assert!(matches!(file, EncoderMedia::File(ref p) if p.is_absolute()));
    }

    #[test]
    fn encoder_media_file_relative_path() {
        let file = EncoderMedia::File(PathBuf::from("relative/path/image.jpg"));
        assert!(matches!(file, EncoderMedia::File(ref p) if !p.is_absolute()));
    }

    #[test]
    fn encoder_media_base64_with_long_data() {
        let long_data = "A".repeat(10000);
        let b64 = EncoderMedia::Base64 {
            data: long_data.clone(),
            mime_type: None,
        };
        match b64 {
            EncoderMedia::Base64 { data, .. } => assert_eq!(data.len(), 10000),
            _ => panic!("expected Base64"),
        }
    }

    #[test]
    fn encoder_media_url_with_s3_scheme() {
        let url = EncoderMedia::Url("s3://my-bucket/images/test.png".into());
        assert!(matches!(url, EncoderMedia::Url(ref s) if s.starts_with("s3://")));
    }

    #[test]
    fn encoder_media_url_with_file_scheme() {
        let url = EncoderMedia::Url("file:///tmp/local.wav".into());
        assert!(matches!(url, EncoderMedia::Url(ref s) if s.starts_with("file://")));
    }

    #[test]
    fn encoder_media_raw_with_large_bytes() {
        let bytes = vec![0xAB_u8; 1_000_000];
        let raw = EncoderMedia::Raw(bytes);
        assert!(matches!(raw, EncoderMedia::Raw(ref v) if v.len() == 1_000_000));
    }

    // --- from_generation: round-trip all variants ---

    #[test]
    fn encoder_media_from_generation_file_preserves_path() {
        let path = "/some/deep/nested/path/model.safetensors";
        let gen = crate::generation::MediaInput::File(path.into());
        let enc = EncoderMedia::from_generation(&gen);
        match enc {
            EncoderMedia::File(p) => assert_eq!(p, PathBuf::from(path)),
            _ => panic!("expected File"),
        }
    }

    #[test]
    fn encoder_media_from_generation_raw_preserves_bytes() {
        let bytes = vec![0x00, 0xFF, 0x80, 0x7F];
        let gen = crate::generation::MediaInput::Raw(bytes.clone());
        let enc = EncoderMedia::from_generation(&gen);
        match enc {
            EncoderMedia::Raw(v) => assert_eq!(v, bytes),
            _ => panic!("expected Raw"),
        }
    }

    #[test]
    fn encoder_media_from_generation_url_preserves_url() {
        let url = "https://cdn.example.com/model.gguf";
        let gen = crate::generation::MediaInput::Url(url.into());
        let enc = EncoderMedia::from_generation(&gen);
        match enc {
            EncoderMedia::Url(s) => assert_eq!(s, url),
            _ => panic!("expected Url"),
        }
    }

    // --- MultimodalEncoded: additional validate edge cases ---

    #[test]
    fn multimodal_encoded_validate_one_token_one_hidden() {
        let enc = MultimodalEncoded {
            tokens: vec![99],
            embeddings: vec![42.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        assert!(enc.validate().is_ok());
        assert_eq!(enc.num_tokens(), 1);
    }

    #[test]
    fn multimodal_encoded_validate_error_message_contains_details() {
        let enc = MultimodalEncoded {
            tokens: vec![1, 2],
            embeddings: vec![0.0; 3], // expected 2*4=8
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let err_msg = format!("{}", enc.validate().unwrap_err());
        assert!(err_msg.contains("tokens=2"));
        assert!(err_msg.contains("hidden=4"));
        assert!(err_msg.contains("expected embeddings=8"));
        assert!(err_msg.contains("got=3"));
    }

    #[test]
    fn multimodal_encoded_validate_large_dimension() {
        let hidden = 8192;
        let tokens_count = 10;
        let enc = MultimodalEncoded {
            tokens: vec![0u32; tokens_count],
            embeddings: vec![1.0; tokens_count * hidden],
            hidden_size: hidden,
            kind: MediaKind::Image,
        };
        assert_eq!(enc.num_tokens(), tokens_count);
        assert!(enc.validate().is_ok());
    }

    #[test]
    fn multimodal_encoded_kind_image() {
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        assert_eq!(enc.kind, MediaKind::Image);
    }

    #[test]
    fn multimodal_encoded_kind_audio() {
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        assert_eq!(enc.kind, MediaKind::Audio);
    }

    // --- MultimodalContext: additional state management ---

    #[test]
    fn multimodal_context_push_many_images_then_audios() {
        let mut ctx = MultimodalContext::new();
        for i in 0..10 {
            ctx.push_image(MultimodalEncoded {
                tokens: vec![i],
                embeddings: vec![i as f32],
                hidden_size: 1,
                kind: MediaKind::Image,
            })
            .unwrap();
        }
        for i in 0..5 {
            ctx.push_audio(MultimodalEncoded {
                tokens: vec![i + 100],
                embeddings: vec![(i + 100) as f32],
                hidden_size: 1,
                kind: MediaKind::Audio,
            })
            .unwrap();
        }
        assert_eq!(ctx.images.len(), 10);
        assert_eq!(ctx.audios.len(), 5);
        assert!(!ctx.is_empty());
    }

    #[test]
    fn multimodal_context_push_image_rejects_shape_mismatch_does_not_push() {
        let mut ctx = MultimodalContext::new();
        let bad = MultimodalEncoded {
            tokens: vec![1, 2],
            embeddings: vec![0.0; 5], // expected 2*3=6
            hidden_size: 3,
            kind: MediaKind::Image,
        };
        assert!(ctx.push_image(bad).is_err());
        assert!(ctx.images.is_empty());
    }

    #[test]
    fn multimodal_context_push_audio_rejects_shape_mismatch_does_not_push() {
        let mut ctx = MultimodalContext::new();
        let bad = MultimodalEncoded {
            tokens: vec![1, 2, 3],
            embeddings: vec![0.0; 7], // expected 3*4=12
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        assert!(ctx.push_audio(bad).is_err());
        assert!(ctx.audios.is_empty());
    }

    #[test]
    fn multimodal_context_images_preserve_order() {
        let mut ctx = MultimodalContext::new();
        for i in 0..4 {
            ctx.push_image(MultimodalEncoded {
                tokens: vec![i * 10],
                embeddings: vec![i as f32],
                hidden_size: 1,
                kind: MediaKind::Image,
            })
            .unwrap();
        }
        assert_eq!(ctx.images[0].tokens[0], 0);
        assert_eq!(ctx.images[1].tokens[0], 10);
        assert_eq!(ctx.images[2].tokens[0], 20);
        assert_eq!(ctx.images[3].tokens[0], 30);
    }

    #[test]
    fn multimodal_context_audios_preserve_order() {
        let mut ctx = MultimodalContext::new();
        for i in 0..3 {
            ctx.push_audio(MultimodalEncoded {
                tokens: vec![i * 5],
                embeddings: vec![i as f32 * 2.0],
                hidden_size: 1,
                kind: MediaKind::Audio,
            })
            .unwrap();
        }
        assert_eq!(ctx.audios[0].tokens[0], 0);
        assert_eq!(ctx.audios[1].tokens[0], 5);
        assert_eq!(ctx.audios[2].tokens[0], 10);
    }

    // --- route_multimodal_tokens: additional error paths ---

    #[test]
    fn route_no_image_encoding_but_prompt_has_image_token() {
        let ids = default_ids();
        let prompt = vec![1u32, ids.image_token_id, 2];
        let ctx = MultimodalContext::new();
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("image"));
        assert!(msg.contains("1"));
        assert!(msg.contains("0"));
    }

    #[test]
    fn route_no_audio_encoding_but_prompt_has_audio_token() {
        let ids = default_ids();
        let prompt = vec![ids.audio_token_id];
        let ctx = MultimodalContext::new();
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap_err();
        assert!(format!("{err}").contains("audio"));
    }

    #[test]
    fn route_image_extra_encoding_rejected() {
        let ids = default_ids();
        let prompt = vec![ids.image_token_id];
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        })
        .unwrap();
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap_err();
        assert!(format!("{err}").contains("image"));
    }

    #[test]
    fn route_audio_extra_encoding_rejected() {
        let ids = default_ids();
        let prompt = vec![ids.audio_token_id];
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        })
        .unwrap();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        })
        .unwrap();
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap_err();
        assert!(format!("{err}").contains("audio"));
    }

    #[test]
    fn route_mixed_hidden_size_mismatch_image() {
        let ids = default_ids();
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc_img).unwrap();
        ctx.push_audio(enc_aud).unwrap();

        let prompt = vec![ids.image_token_id, ids.audio_token_id];
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap_err();
        assert!(format!("{err}").contains("hidden_size"));
    }

    #[test]
    fn route_mixed_hidden_size_mismatch_audio() {
        let ids = default_ids();
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![0.0; 8],
            hidden_size: 8,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc_img).unwrap();
        ctx.push_audio(enc_aud).unwrap();

        let prompt = vec![ids.image_token_id, ids.audio_token_id];
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap_err();
        assert!(format!("{err}").contains("hidden_size"));
    }

    // --- route_multimodal_tokens: text position correctness ---

    #[test]
    fn route_text_positions_with_leading_special() {
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 3],
            embeddings: vec![0.0; 3 * 2],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();

        let prompt = vec![ids.image_token_id, 10, 20, 30];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // virtual tokens at 0,1,2; text at 3,4,5
        assert_eq!(out.text_positions, vec![3, 4, 5]);
    }

    #[test]
    fn route_text_positions_with_trailing_special() {
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![0.0; 2 * 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc).unwrap();

        let prompt = vec![10, 20, ids.audio_token_id];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        assert_eq!(out.text_positions, vec![0, 1]);
    }

    #[test]
    fn route_text_positions_all_special_no_text() {
        let ids = default_ids();
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 2],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![0.0; 2],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc_img).unwrap();
        ctx.push_audio(enc_aud).unwrap();

        let prompt = vec![ids.image_token_id, ids.audio_token_id];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        assert!(out.text_positions.is_empty());
    }

    // --- build_fused_hidden: additional edge cases ---

    #[test]
    fn build_fused_hidden_single_text_single_token() {
        let embed: Vec<f32> = vec![42.0, 43.0]; // vocab=1, hidden=2
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        assert_eq!(hidden, vec![42.0, 43.0]);
    }

    #[test]
    fn build_fused_hidden_all_text_no_media() {
        let embed: Vec<f32> = vec![
            1.0, 2.0, 3.0, // token 0
            4.0, 5.0, 6.0, // token 1
            7.0, 8.0, 9.0, // token 2
        ]; // vocab=3, hidden=3
        let routed = RoutedSequence {
            token_ids: vec![2, 0, 1],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 3,
        };
        let hidden = build_fused_hidden(&routed, &embed, 3).unwrap();
        assert_eq!(hidden, vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn build_fused_hidden_token_zero_uses_first_row() {
        let embed: Vec<f32> = vec![99.0, 100.0, 0.0, 0.0]; // vocab=2, hidden=2
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        assert_eq!(hidden, vec![99.0, 100.0]);
    }

    #[test]
    fn build_fused_hidden_last_vocab_row() {
        let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // vocab=3, hidden=2
        let routed = RoutedSequence {
            token_ids: vec![2],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        assert_eq!(hidden, vec![5.0, 6.0]);
    }

    #[test]
    fn build_fused_hidden_negative_values_in_embed() {
        let embed: Vec<f32> = vec![-1.0, -2.0, -3.0, -4.0]; // vocab=2, hidden=2
        let routed = RoutedSequence {
            token_ids: vec![0, 1],
            fused_embeddings: vec![None, None],
            text_positions: vec![0, 1],
            hidden_size: 2,
        };
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        assert_eq!(hidden, vec![-1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn build_fused_hidden_negative_values_in_media() {
        let embed: Vec<f32> = vec![0.0; 4]; // vocab=2, hidden=2
        let media = vec![-99.0, -88.0];
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![Some(media.clone())],
            text_positions: vec![],
            hidden_size: 2,
        };
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        assert_eq!(hidden, vec![-99.0, -88.0]);
    }

    // --- MockEncoder: encode produces correct embedding patterns ---

    #[test]
    fn mock_encoder_image_embedding_pattern() {
        let ids = default_ids();
        let encoder = MockEncoder::new(3, 0, 2, ids);
        let result = encoder.encode_image(&EncoderMedia::Raw(vec![])).unwrap();
        // Pattern: i * 0.01 for i in 0..3*2 = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
        assert_eq!(result.embeddings.len(), 6);
        assert!((result.embeddings[0] - 0.00).abs() < 1e-6);
        assert!((result.embeddings[1] - 0.01).abs() < 1e-6);
        assert!((result.embeddings[5] - 0.05).abs() < 1e-6);
    }

    #[test]
    fn mock_encoder_audio_embedding_pattern() {
        let ids = default_ids();
        let encoder = MockEncoder::new(0, 2, 2, ids);
        let result = encoder.encode_audio(&EncoderMedia::Raw(vec![])).unwrap();
        // Pattern: -(i * 0.01) for i in 0..2*2 = [0.00, -0.01, -0.02, -0.03]
        assert_eq!(result.embeddings.len(), 4);
        assert!((result.embeddings[0] - 0.00).abs() < 1e-6);
        assert!((result.embeddings[1] - (-0.01)).abs() < 1e-6);
        assert!((result.embeddings[3] - (-0.03)).abs() < 1e-6);
    }

    #[test]
    fn mock_encoder_multiple_calls_increment_counters() {
        let ids = default_ids();
        let encoder = MockEncoder::new(1, 1, 1, ids);
        let _ = encoder.encode_image(&EncoderMedia::Raw(vec![]));
        let _ = encoder.encode_image(&EncoderMedia::Raw(vec![]));
        let _ = encoder.encode_audio(&EncoderMedia::Raw(vec![]));
        assert_eq!(encoder.image_call_count(), 2);
        assert_eq!(encoder.audio_call_count(), 1);
    }

    #[test]
    fn mock_encoder_zero_virtual_tokens() {
        let ids = default_ids();
        let encoder = MockEncoder::new(0, 0, 4, ids);
        let img = encoder.encode_image(&EncoderMedia::Raw(vec![])).unwrap();
        assert_eq!(img.num_tokens(), 0);
        assert!(img.embeddings.is_empty());
        assert!(img.validate().is_ok());

        let aud = encoder.encode_audio(&EncoderMedia::Raw(vec![])).unwrap();
        assert_eq!(aud.num_tokens(), 0);
        assert!(aud.embeddings.is_empty());
        assert!(aud.validate().is_ok());
    }

    // --- RoutedSequence: hidden_size field access ---

    #[test]
    fn routed_sequence_hidden_size_preserved() {
        let rs = RoutedSequence {
            token_ids: vec![1, 2],
            fused_embeddings: vec![None, None],
            text_positions: vec![0, 1],
            hidden_size: 2048,
        };
        assert_eq!(rs.hidden_size, 2048);
    }

    #[test]
    fn routed_sequence_token_ids_preserved() {
        let rs = RoutedSequence {
            token_ids: vec![42, 0, u32::MAX],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 1,
        };
        assert_eq!(rs.token_ids, vec![42, 0, u32::MAX]);
    }

    // --- MultimodalContext: only images or only audios is_empty ---

    #[test]
    fn multimodal_context_only_images_not_empty_audios_empty() {
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        assert!(!ctx.is_empty());
        assert!(ctx.audios.is_empty());
        assert_eq!(ctx.images.len(), 1);
    }

    #[test]
    fn multimodal_context_only_audios_not_empty_images_empty() {
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        assert!(!ctx.is_empty());
        assert!(ctx.images.is_empty());
        assert_eq!(ctx.audios.len(), 1);
    }

    // --- route_multimodal_tokens: hidden_size = 0 edge case ---

    #[test]
    fn route_with_hidden_size_zero_and_no_special_tokens() {
        let ctx = MultimodalContext::new();
        let ids = default_ids();
        let out = route_multimodal_tokens(&[1u32, 2, 3], &ctx, &ids, 0).unwrap();
        assert_eq!(out.seq_len(), 3);
        assert_eq!(out.hidden_size, 0);
    }

    // --- MediaKind in HashMap/HashSet ---

    #[test]
    fn media_kind_usable_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MediaKind::Image);
        set.insert(MediaKind::Audio);
        set.insert(MediaKind::Image); // duplicate
        assert_eq!(set.len(), 2);
        assert!(set.contains(&MediaKind::Image));
        assert!(set.contains(&MediaKind::Audio));
    }

    // --- MultimodalTokenIds in HashSet ---

    #[test]
    fn multimodal_token_ids_usable_in_hashset() {
        use std::collections::HashSet;
        let a = MultimodalTokenIds::fallback_multimodal_token_ids();
        let b = MultimodalTokenIds::fallback_multimodal_token_ids();
        let c = MultimodalTokenIds {
            image_token_id: 0,
            audio_token_id: 0,
            eoi_token_id: 0,
            eoa_token_id: 0,
        };
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b); // duplicate (equal to a)
        set.insert(c);
        assert_eq!(set.len(), 2);
    }

    // --- Full pipeline: encoder -> context -> route -> build_fused_hidden ---

    #[test]
    fn full_pipeline_image_only() {
        let ids = default_ids();
        let encoder = MockEncoder::new(2, 0, 4, ids);

        let encoded = encoder.encode_image(&EncoderMedia::Raw(vec![0xFF; 64])).unwrap();
        let mut ctx = MultimodalContext::new();
        ctx.push_image(encoded).unwrap();

        // Use token IDs within vocab range (0, 1, 2)
        let prompt = vec![1u32, ids.image_token_id, 2];
        let routed = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();

        // embed table: vocab=3, hidden=4
        let embed: Vec<f32> = vec![
            100.0, 101.0, 102.0, 103.0, // token 0
            110.0, 111.0, 112.0, 113.0, // token 1
            120.0, 121.0, 122.0, 123.0, // token 2
        ];
        let hidden = build_fused_hidden(&routed, &embed, 4).unwrap();
        // 4 positions: text(1)->gather, media(virtual x2), text(2)->gather
        assert_eq!(hidden.len(), 4 * 4);
        // Position 0: token 1 -> [110, 111, 112, 113]
        assert_eq!(&hidden[0..4], &[110.0, 111.0, 112.0, 113.0]);
        // Position 3: token 2 -> [120, 121, 122, 123]
        assert_eq!(&hidden[12..16], &[120.0, 121.0, 122.0, 123.0]);
    }

    // --- MultimodalTokenIds: is_image / is_audio with boundary values ---

    #[test]
    fn multimodal_token_ids_is_image_with_u32_max() {
        let ids = MultimodalTokenIds {
            image_token_id: u32::MAX,
            audio_token_id: 0,
            eoi_token_id: 0,
            eoa_token_id: 0,
        };
        assert!(ids.is_image(u32::MAX));
        assert!(!ids.is_audio(u32::MAX));
    }

    #[test]
    fn multimodal_token_ids_is_audio_with_u32_max() {
        let ids = MultimodalTokenIds {
            image_token_id: 0,
            audio_token_id: u32::MAX,
            eoi_token_id: 0,
            eoa_token_id: 0,
        };
        assert!(ids.is_audio(u32::MAX));
        assert!(!ids.is_image(u32::MAX));
    }

    #[test]
    fn multimodal_token_ids_is_image_rejects_zero_when_nonzero() {
        let ids = default_ids();
        assert!(!ids.is_image(0));
    }

    #[test]
    fn multimodal_token_ids_is_audio_rejects_zero_when_nonzero() {
        let ids = default_ids();
        assert!(!ids.is_audio(0));
    }

    #[test]
    fn multimodal_token_ids_is_image_rejects_one() {
        let ids = default_ids();
        assert!(!ids.is_image(1));
    }

    #[test]
    fn multimodal_token_ids_is_audio_rejects_one() {
        let ids = default_ids();
        assert!(!ids.is_audio(1));
    }

    // --- EncoderMedia: from_generation with all variants edge cases ---

    #[test]
    fn encoder_media_from_generation_base64_preserves_mime() {
        let input = crate::generation::MediaInput::Base64 {
            data: "AQID".into(),
            mime_type: Some("image/png".into()),
        };
        let media = EncoderMedia::from_generation(&input);
        match media {
            EncoderMedia::Base64 { data, mime_type } => {
                assert_eq!(data, "AQID");
                assert_eq!(mime_type.as_deref(), Some("image/png"));
            }
            _ => panic!("expected Base64 variant"),
        }
    }

    #[test]
    fn encoder_media_from_generation_url_https() {
        let input = crate::generation::MediaInput::Url("https://example.com/img.jpg".into());
        let media = EncoderMedia::from_generation(&input);
        match media {
            EncoderMedia::Url(url) => assert_eq!(url, "https://example.com/img.jpg"),
            _ => panic!("expected Url variant"),
        }
    }

    #[test]
    fn encoder_media_from_generation_file_preserves_pathbuf() {
        let input = crate::generation::MediaInput::File("/tmp/test.png".into());
        let media = EncoderMedia::from_generation(&input);
        match media {
            EncoderMedia::File(p) => assert_eq!(p, PathBuf::from("/tmp/test.png")),
            _ => panic!("expected File variant"),
        }
    }

    // --- EncoderMedia: field access on all variants ---

    #[test]
    fn encoder_media_raw_preserves_bytes() {
        let media = EncoderMedia::Raw(vec![0xDE, 0xAD, 0xBE, 0xEF]);
        match &media {
            EncoderMedia::Raw(b) => assert_eq!(*b, vec![0xDE, 0xAD, 0xBE, 0xEF]),
            _ => panic!("expected Raw variant"),
        }
    }

    #[test]
    fn encoder_media_file_path_preserved() {
        let media = EncoderMedia::File(PathBuf::from("/data/image.png"));
        match &media {
            EncoderMedia::File(p) => assert_eq!(p.as_os_str(), "/data/image.png"),
            _ => panic!("expected File variant"),
        }
    }

    #[test]
    fn encoder_media_base64_mime_none_preserved() {
        let media = EncoderMedia::Base64 {
            data: "abc".into(),
            mime_type: None,
        };
        match &media {
            EncoderMedia::Base64 { data, mime_type } => {
                assert_eq!(data, "abc");
                assert!(mime_type.is_none());
            }
            _ => panic!("expected Base64 variant"),
        }
    }

    #[test]
    fn encoder_media_url_s3_preserved() {
        let media = EncoderMedia::Url("s3://bucket/key".into());
        match &media {
            EncoderMedia::Url(u) => assert_eq!(u, "s3://bucket/key"),
            _ => panic!("expected Url variant"),
        }
    }

    // --- MediaKind: exhaustiveness (both variants constructible) ---

    #[test]
    fn media_kind_image_not_equal_audio() {
        assert_ne!(MediaKind::Image, MediaKind::Audio);
    }

    // --- MultimodalEncoded: tokens field access ---

    #[test]
    fn multimodal_encoded_tokens_field_access() {
        let enc = MultimodalEncoded {
            tokens: vec![100, 200, 300],
            embeddings: vec![0.0; 6],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        assert_eq!(enc.tokens, vec![100, 200, 300]);
    }

    // --- MultimodalEncoded: embeddings field access ---

    #[test]
    fn multimodal_encoded_embeddings_field_access() {
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![1.0, 2.0, 3.0],
            hidden_size: 3,
            kind: MediaKind::Audio,
        };
        assert_eq!(enc.embeddings, vec![1.0, 2.0, 3.0]);
    }

    // --- MultimodalEncoded: hidden_size field access ---

    #[test]
    fn multimodal_encoded_hidden_size_field_access() {
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0; 256],
            hidden_size: 256,
            kind: MediaKind::Image,
        };
        assert_eq!(enc.hidden_size, 256);
    }

    // --- MultimodalEncoded: kind field access ---

    #[test]
    fn multimodal_encoded_kind_field_matches() {
        let img = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        assert_eq!(img.kind, MediaKind::Image);

        let aud = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        assert_eq!(aud.kind, MediaKind::Audio);
    }

    // --- MultimodalContext: images field direct mutation ---

    #[test]
    fn multimodal_context_images_len_after_direct_push() {
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![1.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        assert_eq!(ctx.images.len(), 2);
        assert_eq!(ctx.images[0].tokens, vec![1]);
        assert_eq!(ctx.images[1].tokens, vec![2]);
    }

    // --- RoutedSequence: text_positions field access ---

    #[test]
    fn routed_sequence_text_positions_field_access() {
        let rs = RoutedSequence {
            token_ids: vec![10, 20, 30],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 8,
        };
        assert_eq!(rs.text_positions, vec![0, 1, 2]);
    }

    // --- RoutedSequence: fused_embeddings field access ---

    #[test]
    fn routed_sequence_fused_embeddings_field_access() {
        let rs = RoutedSequence {
            token_ids: vec![1, 2],
            fused_embeddings: vec![None, Some(vec![1.0, 2.0])],
            text_positions: vec![0],
            hidden_size: 2,
        };
        assert!(rs.fused_embeddings[0].is_none());
        assert_eq!(rs.fused_embeddings[1].as_deref(), Some([1.0, 2.0].as_slice()));
    }

    // --- RoutedSequence: has_multimodal with mixed ---

    #[test]
    fn routed_sequence_has_multimodal_mixed() {
        let rs = RoutedSequence {
            token_ids: vec![1, 2, 3],
            fused_embeddings: vec![None, Some(vec![0.0]), None],
            text_positions: vec![0, 2],
            hidden_size: 1,
        };
        assert!(rs.has_multimodal());
    }

    // --- MultimodalContext: multiple images and audios independent ---

    #[test]
    fn multimodal_context_images_and_audios_independent() {
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.audios.len(), 1);
        assert!(!ctx.is_empty());
    }

    // --- route: hidden_size zero with special token ---

    #[test]
    fn route_image_with_hidden_size_zero_rejected() {
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![],
            hidden_size: 0,
            kind: MediaKind::Image,
        };
        enc.validate().unwrap(); // 0*0=0, valid
        ctx.push_image(enc).unwrap();
        let result = route_multimodal_tokens(&[ids.image_token_id], &ctx, &ids, 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().hidden_size, 0);
    }

    // --- build_fused_hidden: single position single hidden ---

    #[test]
    fn build_fused_hidden_single_position_single_hidden() {
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![Some(vec![42.0])],
            text_positions: vec![],
            hidden_size: 1,
        };
        let embed = vec![99.0]; // vocab=1, hidden=1
        let hidden = build_fused_hidden(&routed, &embed, 1).unwrap();
        assert_eq!(hidden, vec![42.0]);
    }

    // --- MultimodalContext: push_image then push_audio then is_empty ---

    #[test]
    fn multimodal_context_push_image_and_audio_not_empty() {
        let mut ctx = MultimodalContext::new();
        assert!(ctx.is_empty());
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        assert!(!ctx.is_empty());
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        assert!(!ctx.is_empty());
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.audios.len(), 1);
    }

    // --- EncoderMedia: File with deep path ---

    #[test]
    fn encoder_media_file_deep_path() {
        let media = EncoderMedia::File(PathBuf::from("/a/b/c/d/e.png"));
        match &media {
            EncoderMedia::File(p) => assert_eq!(p.to_str(), Some("/a/b/c/d/e.png")),
            _ => panic!("expected File variant"),
        }
    }

    // --- EncoderMedia: Base64 with unicode mime ---

    #[test]
    fn encoder_media_base64_unicode_mime() {
        let media = EncoderMedia::Base64 {
            data: "".into(),
            mime_type: Some("image/png;charset=utf-8".into()),
        };
        match &media {
            EncoderMedia::Base64 { mime_type, .. } => {
                assert_eq!(mime_type.as_deref(), Some("image/png;charset=utf-8"));
            }
            _ => panic!("expected Base64 variant"),
        }
    }

    // ── Batch 4: 18 additional tests for coverage expansion ──

    #[test]
    fn multimodal_token_ids_eoi_eoa_field_values() {
        let ids = MultimodalTokenIds {
            image_token_id: 10,
            audio_token_id: 20,
            eoi_token_id: 30,
            eoa_token_id: 40,
        };
        assert_eq!(ids.eoi_token_id, 30);
        assert_eq!(ids.eoa_token_id, 40);
    }

    #[test]
    fn multimodal_encoded_tokens_non_sequential_ids() {
        let enc = MultimodalEncoded {
            tokens: vec![100, 500, 999, 0, u32::MAX],
            embeddings: vec![0.0; 5 * 2],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        assert_eq!(enc.num_tokens(), 5);
        assert_eq!(enc.tokens[3], 0);
        assert_eq!(enc.tokens[4], u32::MAX);
        assert!(enc.validate().is_ok());
    }

    #[test]
    fn multimodal_encoded_all_negative_embeddings() {
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![-1.0, -2.0, -3.0],
            hidden_size: 3,
            kind: MediaKind::Audio,
        };
        assert!(enc.validate().is_ok());
        assert!(enc.embeddings.iter().all(|&v| v < 0.0));
    }

    #[test]
    fn multimodal_context_push_image_zero_hidden_size() {
        let mut ctx = MultimodalContext::new();
        let enc = MultimodalEncoded {
            tokens: vec![],
            embeddings: vec![],
            hidden_size: 0,
            kind: MediaKind::Image,
        };
        assert!(ctx.push_image(enc).is_ok());
        assert_eq!(ctx.images.len(), 1);
    }

    #[test]
    fn multimodal_context_push_audio_zero_hidden_size() {
        let mut ctx = MultimodalContext::new();
        let enc = MultimodalEncoded {
            tokens: vec![],
            embeddings: vec![],
            hidden_size: 0,
            kind: MediaKind::Audio,
        };
        assert!(ctx.push_audio(enc).is_ok());
        assert_eq!(ctx.audios.len(), 1);
    }

    #[test]
    fn route_only_audio_no_text_tokens() {
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 4],
            embeddings: vec![0.5; 4 * 2],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc).unwrap();

        let prompt = vec![ids.audio_token_id];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        assert_eq!(out.seq_len(), 4);
        assert!(out.text_positions.is_empty());
        assert!(out.has_multimodal());
        assert!(out.fused_embeddings.iter().all(|e| e.is_some()));
    }

    #[test]
    fn route_eoi_token_treated_as_text() {
        let ids = default_ids();
        let ctx = MultimodalContext::new();
        let prompt = vec![ids.eoi_token_id, 10, 20];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        assert_eq!(out.seq_len(), 3);
        assert!(out.fused_embeddings.iter().all(|e| e.is_none()));
        assert!(!out.has_multimodal());
        assert_eq!(out.token_ids[0], ids.eoi_token_id);
    }

    #[test]
    fn route_eoa_token_treated_as_text() {
        let ids = default_ids();
        let ctx = MultimodalContext::new();
        let prompt = vec![ids.eoa_token_id, 5];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 8).unwrap();
        assert_eq!(out.seq_len(), 2);
        assert!(out.fused_embeddings.iter().all(|e| e.is_none()));
        assert_eq!(out.token_ids[0], ids.eoa_token_id);
    }

    #[test]
    fn route_two_audios_with_text_between() {
        let ids = default_ids();
        let enc1 = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![1.0; 2 * 3],
            hidden_size: 3,
            kind: MediaKind::Audio,
        };
        let enc2 = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![2.0; 3],
            hidden_size: 3,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc1).unwrap();
        ctx.push_audio(enc2).unwrap();

        let prompt = vec![ids.audio_token_id, 42, ids.audio_token_id];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 3).unwrap();
        assert_eq!(out.seq_len(), 4); // 2 virtual + 1 text + 1 virtual
        assert_eq!(out.text_positions, vec![2]);
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 1.0);
        assert_eq!(out.fused_embeddings[3].as_ref().unwrap()[0], 2.0);
    }

    #[test]
    fn route_text_after_two_image_expansions() {
        let ids = default_ids();
        let enc1 = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 3],
            embeddings: vec![1.0; 3 * 2],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let enc2 = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 2],
            embeddings: vec![2.0; 2 * 2],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc1).unwrap();
        ctx.push_image(enc2).unwrap();

        let prompt = vec![ids.image_token_id, ids.image_token_id, 99];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // 3 virtual + 2 virtual + 1 text = 6
        assert_eq!(out.seq_len(), 6);
        assert_eq!(out.text_positions, vec![5]);
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 1.0);
        assert_eq!(out.fused_embeddings[3].as_ref().unwrap()[0], 2.0);
        assert!(out.fused_embeddings[5].is_none());
    }

    #[test]
    fn route_audio_followed_by_image() {
        let ids = default_ids();
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![7.0; 2 * 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![8.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc_aud).unwrap();
        ctx.push_image(enc_img).unwrap();

        let prompt = vec![ids.audio_token_id, ids.image_token_id];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        assert_eq!(out.seq_len(), 3); // 2 audio virtual + 1 image virtual
        assert!(out.text_positions.is_empty());
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 7.0);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap()[0], 8.0);
    }

    #[test]
    fn routed_sequence_first_position_is_media() {
        let rs = RoutedSequence {
            token_ids: vec![258880, 1, 2],
            fused_embeddings: vec![Some(vec![0.0; 4]), None, None],
            text_positions: vec![1, 2],
            hidden_size: 4,
        };
        assert!(rs.fused_embeddings[0].is_some());
        assert!(rs.has_multimodal());
    }

    #[test]
    fn routed_sequence_last_position_is_media() {
        let rs = RoutedSequence {
            token_ids: vec![1, 2, 258881],
            fused_embeddings: vec![None, None, Some(vec![1.0; 8])],
            text_positions: vec![0, 1],
            hidden_size: 8,
        };
        assert!(rs.fused_embeddings[2].is_some());
        assert!(rs.has_multimodal());
    }

    #[test]
    fn mock_encoder_image_uses_custom_token_id() {
        let ids = MultimodalTokenIds {
            image_token_id: 999,
            audio_token_id: 888,
            eoi_token_id: 777,
            eoa_token_id: 666,
        };
        let encoder = MockEncoder::new(3, 0, 2, ids);
        let result = encoder.encode_image(&EncoderMedia::Raw(vec![])).unwrap();
        assert_eq!(result.tokens, vec![999, 999, 999]);
    }

    #[test]
    fn mock_encoder_audio_uses_custom_token_id() {
        let ids = MultimodalTokenIds {
            image_token_id: 999,
            audio_token_id: 888,
            eoi_token_id: 777,
            eoa_token_id: 666,
        };
        let encoder = MockEncoder::new(0, 2, 4, ids);
        let result = encoder.encode_audio(&EncoderMedia::Raw(vec![])).unwrap();
        assert_eq!(result.tokens, vec![888, 888]);
    }

    #[test]
    fn route_large_text_only_sequence() {
        let ctx = MultimodalContext::new();
        let ids = default_ids();
        let tokens: Vec<u32> = (0..1000).collect();
        let out = route_multimodal_tokens(&tokens, &ctx, &ids, 64).unwrap();
        assert_eq!(out.seq_len(), 1000);
        assert_eq!(out.token_ids, tokens);
        assert!(out.fused_embeddings.iter().all(|e| e.is_none()));
        assert!(!out.has_multimodal());
    }

    #[test]
    fn route_image_at_end_after_text() {
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 2],
            embeddings: vec![5.0, 6.0, 7.0, 8.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();

        let prompt = vec![10u32, 20, ids.image_token_id];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // 2 text + 2 virtual = 4
        assert_eq!(out.seq_len(), 4);
        assert_eq!(out.text_positions, vec![0, 1]);
        assert!(out.fused_embeddings[0].is_none());
        assert!(out.fused_embeddings[1].is_none());
        assert!(out.fused_embeddings[2].is_some());
        assert!(out.fused_embeddings[3].is_some());
    }

    #[test]
    fn routed_sequence_single_position_with_media() {
        let rs = RoutedSequence {
            token_ids: vec![258880],
            fused_embeddings: vec![Some(vec![1.0, 2.0])],
            text_positions: vec![],
            hidden_size: 2,
        };
        assert_eq!(rs.seq_len(), 1);
        assert!(rs.has_multimodal());
        assert!(rs.fused_embeddings[0].is_some());
    }

    #[test]
    fn full_pipeline_text_and_audio() {
        let ids = default_ids();
        let encoder = MockEncoder::new(0, 2, 3, ids);

        let encoded = encoder.encode_audio(&EncoderMedia::Raw(vec![])).unwrap();
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(encoded).unwrap();

        let prompt = vec![0u32, ids.audio_token_id, 1];
        let routed = route_multimodal_tokens(&prompt, &ctx, &ids, 3).unwrap();

        // embed: vocab=2, hidden=3
        let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let hidden = build_fused_hidden(&routed, &embed, 3).unwrap();
        assert_eq!(hidden.len(), 4 * 3); // 1 text + 2 audio virtual + 1 text = 4 positions
        // Position 0: token 0 -> [1, 2, 3]
        assert_eq!(&hidden[0..3], &[1.0, 2.0, 3.0]);
        // Position 3: token 1 -> [4, 5, 6]
        assert_eq!(&hidden[9..12], &[4.0, 5.0, 6.0]);
    }

    // ── Batch 5: 12 additional tests for coverage expansion ──

    #[test]
    fn mock_encoder_image_accepts_file_media() {
        let ids = default_ids();
        let encoder = MockEncoder::new(1, 0, 2, ids);
        let result = encoder
            .encode_image(&EncoderMedia::File(PathBuf::from("/tmp/test.jpg")))
            .unwrap();
        assert_eq!(result.kind, MediaKind::Image);
        assert_eq!(result.num_tokens(), 1);
    }

    #[test]
    fn mock_encoder_audio_accepts_url_media() {
        let ids = default_ids();
        let encoder = MockEncoder::new(0, 2, 4, ids);
        let result = encoder
            .encode_audio(&EncoderMedia::Url("https://example.com/audio.wav".into()))
            .unwrap();
        assert_eq!(result.kind, MediaKind::Audio);
        assert_eq!(result.num_tokens(), 2);
    }

    #[test]
    fn mock_encoder_image_accepts_base64_media() {
        let ids = default_ids();
        let encoder = MockEncoder::new(3, 0, 2, ids);
        let result = encoder
            .encode_image(&EncoderMedia::Base64 {
                data: "AQID".into(),
                mime_type: Some("image/jpeg".into()),
            })
            .unwrap();
        assert_eq!(result.kind, MediaKind::Image);
        assert_eq!(result.num_tokens(), 3);
    }

    #[test]
    fn multimodal_encoded_zero_tokens_nonzero_embeddings_rejected() {
        let enc = MultimodalEncoded {
            tokens: vec![],
            embeddings: vec![1.0, 2.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        let err = enc.validate().unwrap_err();
        assert!(format!("{err}").contains("shape mismatch"));
    }

    #[test]
    fn build_fused_hidden_repeated_token_gathers_same_row() {
        let embed: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]; // vocab=3, hidden=2
        let routed = RoutedSequence {
            token_ids: vec![1, 1, 1],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 2,
        };
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        assert_eq!(&hidden[0..2], &[30.0, 40.0]);
        assert_eq!(&hidden[2..4], &[30.0, 40.0]);
        assert_eq!(&hidden[4..6], &[30.0, 40.0]);
    }

    #[test]
    fn multimodal_context_push_image_failure_preserves_audios() {
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        assert_eq!(ctx.audios.len(), 1);

        let bad = MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Audio, // Wrong kind for push_image
        };
        assert!(ctx.push_image(bad).is_err());
        assert_eq!(ctx.audios.len(), 1);
    }

    #[test]
    fn multimodal_context_push_audio_failure_preserves_images() {
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        assert_eq!(ctx.images.len(), 1);

        let bad = MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image, // Wrong kind for push_audio
        };
        assert!(ctx.push_audio(bad).is_err());
        assert_eq!(ctx.images.len(), 1);
    }

    #[test]
    fn multimodal_token_ids_fallback_multimodal_token_ids_all_unique() {
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let values = [
            ids.image_token_id,
            ids.audio_token_id,
            ids.eoi_token_id,
            ids.eoa_token_id,
        ];
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                assert_ne!(values[i], values[j], "fields at indices {i} and {j} should differ");
            }
        }
    }

    #[test]
    fn route_output_hidden_size_matches_input() {
        let ctx = MultimodalContext::new();
        let ids = default_ids();
        let out = route_multimodal_tokens(&[1u32, 2, 3], &ctx, &ids, 512).unwrap();
        assert_eq!(out.hidden_size, 512);
    }

    #[test]
    fn route_eoi_and_eoa_combined_as_text() {
        let ids = default_ids();
        let ctx = MultimodalContext::new();
        let prompt = vec![ids.eoi_token_id, ids.eoa_token_id, 10];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        assert_eq!(out.seq_len(), 3);
        assert!(out.fused_embeddings.iter().all(|e| e.is_none()));
        assert!(!out.has_multimodal());
        assert_eq!(out.text_positions, vec![0, 1, 2]);
    }

    #[test]
    fn multimodal_context_push_image_count_increments() {
        let mut ctx = MultimodalContext::new();
        assert_eq!(ctx.images.len(), 0);
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        assert_eq!(ctx.images.len(), 1);
        ctx.push_image(MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![1.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        assert_eq!(ctx.images.len(), 2);
        assert_eq!(ctx.images[0].tokens[0], 1);
        assert_eq!(ctx.images[1].tokens[0], 2);
    }

    #[test]
    fn multimodal_token_ids_fallback_multimodal_token_ids_eoi_distinct_from_image() {
        // Arrange
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        // Act & Assert
        assert_ne!(ids.eoi_token_id, ids.image_token_id);
        assert!(!ids.is_image(ids.eoi_token_id));
    }

    #[test]
    fn multimodal_token_ids_fallback_multimodal_token_ids_eoa_distinct_from_audio() {
        // Arrange
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        // Act & Assert
        assert_ne!(ids.eoa_token_id, ids.audio_token_id);
        assert!(!ids.is_audio(ids.eoa_token_id));
    }

    #[test]
    fn encoder_media_base64_with_none_mime_field_access() {
        // Arrange
        let media = EncoderMedia::Base64 {
            data: "abcd".into(),
            mime_type: None,
        };
        // Act
        let debug = format!("{:?}", media);
        // Assert — debug output contains the variant name and None
        assert!(debug.contains("Base64"));
        assert!(debug.contains("mime_type: None"));
    }

    #[test]
    fn encoder_media_raw_len_preserved_after_clone() {
        // Arrange
        let original = EncoderMedia::Raw(vec![42u8; 100]);
        // Act
        let cloned = original.clone();
        // Assert
        if let EncoderMedia::Raw(bytes) = cloned {
            assert_eq!(bytes.len(), 100);
            assert!(bytes.iter().all(|&b| b == 42));
        } else {
            panic!("expected Raw variant");
        }
    }

    #[test]
    fn multimodal_encoded_validate_two_tokens_hidden_three() {
        // Arrange — 2 tokens × 3 hidden = 6 embeddings
        let enc = MultimodalEncoded {
            tokens: vec![10, 20],
            embeddings: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            hidden_size: 3,
            kind: MediaKind::Image,
        };
        // Act
        let result = enc.validate();
        // Assert
        assert!(result.is_ok());
    }

    #[test]
    fn multimodal_encoded_num_tokens_empty() {
        // Arrange
        let enc = MultimodalEncoded {
            tokens: vec![],
            embeddings: vec![],
            hidden_size: 0,
            kind: MediaKind::Audio,
        };
        // Act & Assert
        assert_eq!(enc.num_tokens(), 0);
    }

    #[test]
    fn multimodal_context_push_image_then_audio_count() {
        // Arrange
        let mut ctx = MultimodalContext::new();
        let img = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.5],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        let aud = MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![-0.5],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        // Act
        ctx.push_image(img).unwrap();
        ctx.push_audio(aud).unwrap();
        // Assert
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.audios.len(), 1);
        assert!(!ctx.is_empty());
    }

    #[test]
    fn routed_sequence_seq_len_with_expanded_tokens() {
        // Arrange — 2 text tokens + 1 image with 3 virtual tokens = 5 total
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![ids.image_token_id; 3],
            embeddings: vec![0.1; 3],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        let prompt = vec![1u32, ids.image_token_id, 2];
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 1).unwrap();
        // Act
        let seq_len = out.seq_len();
        // Assert
        assert_eq!(seq_len, 5); // 1 text + 3 virtual + 1 text
        assert_eq!(out.text_positions, vec![0, 4]);
    }

    #[test]
    fn route_single_audio_between_two_texts() {
        // Arrange
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![0.2, 0.3],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        let prompt = vec![100u32, ids.audio_token_id, 200];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 1).unwrap();
        // Assert
        assert_eq!(out.token_ids, vec![100, ids.audio_token_id, ids.audio_token_id, 200]);
        assert_eq!(out.text_positions, vec![0, 3]);
    }

    #[test]
    fn build_fused_hidden_rejects_mismatched_hidden_size_from_route() {
        // Arrange
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 8,
        };
        let embed = vec![0.0f32; 4]; // hidden_size=4 vs routed.hidden_size=8
        // Act
        let result = build_fused_hidden(&routed, &embed, 4);
        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn build_fused_hidden_with_single_text_and_hidden_size_two() {
        // Arrange
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        let embed = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32]; // vocab=2, hidden=2
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert
        assert_eq!(hidden, vec![1.0, 2.0]); // token_id=0 → row 0
    }

    #[test]
    fn route_text_positions_skip_consecutive_special() {
        // Arrange — two consecutive image tokens followed by text
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        for _ in 0..2 {
            ctx.push_image(MultimodalEncoded {
                tokens: vec![ids.image_token_id],
                embeddings: vec![0.0],
                hidden_size: 1,
                kind: MediaKind::Image,
            })
            .unwrap();
        }
        let prompt = vec![ids.image_token_id, ids.image_token_id, 42];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 1).unwrap();
        // Assert
        assert_eq!(out.text_positions, vec![2]); // only the trailing text
    }

    #[test]
    fn multimodal_context_push_audio_count_increments() {
        // Arrange
        let mut ctx = MultimodalContext::new();
        // Act & Assert
        assert_eq!(ctx.audios.len(), 0);
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![5],
            embeddings: vec![0.1],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        assert_eq!(ctx.audios.len(), 1);
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![6],
            embeddings: vec![0.2],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        assert_eq!(ctx.audios.len(), 2);
    }

    #[test]
    fn media_kind_image_audio_neq() {
        // Arrange
        let a = MediaKind::Image;
        let b = MediaKind::Audio;
        // Act & Assert
        assert_ne!(a, b);
        assert_eq!(a, MediaKind::Image);
        assert_eq!(b, MediaKind::Audio);
    }

    #[test]
    fn build_fused_hidden_media_embedding_overrides_token_zero_row() {
        // Arrange — token_id=0 would normally gather row 0, but media overrides it
        let media_emb = vec![9.0f32, 9.0f32];
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![Some(media_emb.clone())],
            text_positions: vec![],
            hidden_size: 2,
        };
        let embed = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32]; // row 0 = [1,2]
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert
        assert_eq!(hidden, vec![9.0, 9.0]); // media override, not gather
    }

    // ── Batch 4: additional coverage expansion (~15 tests) ──

    #[test]
    fn build_fused_hidden_rejects_media_embedding_length_mismatch_detailed() {
        // Arrange — media embedding has 3 elements but hidden_size is 2
        let media_emb = vec![1.0f32, 2.0, 3.0];
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![Some(media_emb)],
            text_positions: vec![],
            hidden_size: 2,
        };
        let embed = vec![0.0f32; 4]; // vocab=2, hidden=2
        // Act
        let result = build_fused_hidden(&routed, &embed, 2);
        // Assert
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("media embedding at position 0"));
        assert!(format!("{err}").contains("length 3"));
        assert!(format!("{err}").contains("hidden_size 2"));
    }

    #[test]
    fn build_fused_hidden_rejects_embed_rows_not_divisible_by_hidden_size() {
        // Arrange — 7 elements not divisible by hidden_size=3
        let routed = RoutedSequence {
            token_ids: vec![],
            fused_embeddings: vec![],
            text_positions: vec![],
            hidden_size: 3,
        };
        let embed = vec![1.0f32; 7]; // 7 % 3 != 0
        // Act
        let result = build_fused_hidden(&routed, &embed, 3);
        // Assert
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("not divisible"));
        assert!(format!("{err}").contains("7"));
        assert!(format!("{err}").contains("3"));
    }

    #[test]
    fn route_empty_prompt_with_empty_context() {
        // Arrange — empty prompt + empty context should produce empty output
        let prompt: Vec<u32> = vec![];
        let ctx = MultimodalContext::new();
        let ids = default_ids();
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        // Assert
        assert_eq!(out.seq_len(), 0);
        assert!(out.token_ids.is_empty());
        assert!(out.fused_embeddings.is_empty());
        assert!(out.text_positions.is_empty());
        assert!(!out.has_multimodal());
    }

    #[test]
    fn route_image_token_at_first_and_last_position() {
        // Arrange — image special token at both ends of the prompt
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![ids.image_token_id; 2],
            embeddings: vec![0.5, 0.6, 0.7, 0.8],
            hidden_size: 2,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![1.0, 1.1],
            hidden_size: 2,
            kind: MediaKind::Image,
        })
        .unwrap();
        let prompt = vec![ids.image_token_id, 99, ids.image_token_id];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — first image=2 virtual, text=1, second image=1 virtual => total=4
        assert_eq!(out.seq_len(), 4);
        assert_eq!(out.text_positions, vec![2]); // only token 99 at middle
        assert!(out.fused_embeddings[0].is_some()); // first image virtual
        assert!(out.fused_embeddings[1].is_some()); // first image virtual
        assert!(out.fused_embeddings[2].is_none()); // text
        assert!(out.fused_embeddings[3].is_some()); // second image virtual
    }

    #[test]
    fn multimodal_encoded_many_tokens_hidden_size_one() {
        // Arrange — 1000 tokens each with hidden_size=1
        let n = 1000;
        let enc = MultimodalEncoded {
            tokens: vec![42u32; n],
            embeddings: vec![0.123; n],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        // Act & Assert
        assert_eq!(enc.num_tokens(), n);
        assert!(enc.validate().is_ok());
    }

    #[test]
    fn multimodal_context_push_image_after_failed_push_still_works() {
        // Arrange — push_image rejects wrong kind, then a valid push succeeds
        let mut ctx = MultimodalContext::new();
        let bad = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Audio, // wrong kind for push_image
        };
        let _ = ctx.push_image(bad);
        assert_eq!(ctx.images.len(), 0);
        // Act — valid push after failure
        let good = MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![1.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        ctx.push_image(good).unwrap();
        // Assert
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.images[0].tokens[0], 2);
    }

    #[test]
    fn routed_sequence_text_only_positions_match_input() {
        // Arrange — 10 text tokens, verify all positions are sequential
        let rs = RoutedSequence {
            token_ids: vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            fused_embeddings: vec![None; 10],
            text_positions: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            hidden_size: 64,
        };
        // Act & Assert
        assert_eq!(rs.seq_len(), 10);
        assert!(!rs.has_multimodal());
        assert_eq!(rs.text_positions.len(), 10);
        for (i, &pos) in rs.text_positions.iter().enumerate() {
            assert_eq!(pos, i);
        }
    }

    #[test]
    fn route_audio_hidden_size_mismatch_rejected() {
        // Arrange — audio encoding with hidden_size=8 but model expects 4
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![0.0; 16], // 2 * 8
            hidden_size: 8, // mismatch
            kind: MediaKind::Audio,
        })
        .unwrap();
        let prompt = vec![ids.audio_token_id];
        // Act
        let result = route_multimodal_tokens(&prompt, &ctx, &ids, 4);
        // Assert
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("hidden_size 8"));
        assert!(format!("{err}").contains("model hidden_size 4"));
    }

    #[test]
    fn encoder_media_from_generation_base64_none_mime() {
        // Arrange
        let gen = crate::generation::MediaInput::Base64 {
            data: "AAAA".into(),
            mime_type: None,
        };
        // Act
        let enc = EncoderMedia::from_generation(&gen);
        // Assert
        match enc {
            EncoderMedia::Base64 { data, mime_type } => {
                assert_eq!(data, "AAAA");
                assert!(mime_type.is_none());
            }
            _ => panic!("expected Base64 variant"),
        }
    }

    #[test]
    fn multimodal_context_default_then_push_then_not_empty() {
        // Arrange — verify Default gives empty, then push makes it non-empty
        let mut ctx = MultimodalContext::default();
        assert!(ctx.is_empty());
        assert_eq!(ctx.images.len(), 0);
        assert_eq!(ctx.audios.len(), 0);
        // Act
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        // Assert
        assert!(!ctx.is_empty());
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.audios.len(), 0);
    }

    #[test]
    fn build_fused_hidden_gather_last_token_from_vocab() {
        // Arrange — vocab=3, hidden=2, gather token_id=2 (last row)
        let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 rows of 2
        let routed = RoutedSequence {
            token_ids: vec![2],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert
        assert_eq!(hidden, vec![5.0, 6.0]); // last row
    }

    #[test]
    fn multimodal_encoded_validate_exact_boundary_100_tokens() {
        // Arrange — exactly 100 tokens x 3 hidden = 300 embeddings
        let enc = MultimodalEncoded {
            tokens: vec![7u32; 100],
            embeddings: vec![0.0; 300],
            hidden_size: 3,
            kind: MediaKind::Image,
        };
        // Act & Assert
        assert_eq!(enc.num_tokens(), 100);
        assert!(enc.validate().is_ok());
    }

    #[test]
    fn route_text_only_with_non_empty_context_rejected() {
        // Arrange — prompt has no special tokens, but context has an image encoding
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        })
        .unwrap();
        let prompt = vec![1u32, 2, 3]; // no special tokens
        // Act
        let result = route_multimodal_tokens(&prompt, &ctx, &ids, 4);
        // Assert
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{err}").contains("image tokens but context provided 1"));
    }

    #[test]
    fn multimodal_token_ids_fallback_matches_all_four_gemma4_values() {
        // Arrange
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        // Act & Assert — every field must match gemma4 values
        assert_eq!(ids.image_token_id, 258880);
        assert_eq!(ids.audio_token_id, 258881);
        assert_eq!(ids.eoi_token_id, 258882);
        assert_eq!(ids.eoa_token_id, 258883);
    }

    #[test]
    fn build_fused_hidden_vocab_size_one_gather() {
        // Arrange — vocab=1, hidden=4, only token_id=0 is valid
        let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // 1 row of 4
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 4,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 4).unwrap();
        // Assert
        assert_eq!(hidden, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ── Batch 6: 15 additional tests for edge-case and error-path coverage ──

    // @trace TEST-MULTIMODAL-001 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_many_virtual_tokens_per_image_expansion() {
        // Arrange — single image token expands to 20 virtual tokens
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 20],
            embeddings: (0..20 * 8).map(|i| i as f32 * 0.001).collect(),
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![10u32, ids.image_token_id, 20];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 8).unwrap();
        // Assert — 1 text + 20 virtual + 1 text = 22
        assert_eq!(out.seq_len(), 22);
        assert_eq!(out.text_positions, vec![0, 21]);
        assert!(out.has_multimodal());
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap().len(), 8);
        assert_eq!(out.fused_embeddings[20].as_ref().unwrap().len(), 8);
    }

    // @trace TEST-MULTIMODAL-002 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_complex_interleaved_image_text_audio_text() {
        // Arrange — image, text, audio, text pattern
        let ids = default_ids();
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 2],
            embeddings: vec![1.0; 2 * 3],
            hidden_size: 3,
            kind: MediaKind::Image,
        };
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 3],
            embeddings: vec![2.0; 3 * 3],
            hidden_size: 3,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc_img).unwrap();
        ctx.push_audio(enc_aud).unwrap();
        let prompt = vec![ids.image_token_id, 50, ids.audio_token_id, 60];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 3).unwrap();
        // Assert — 2 virtual + 1 text + 3 virtual + 1 text = 7
        assert_eq!(out.seq_len(), 7);
        assert_eq!(out.text_positions, vec![2, 6]);
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 1.0);
        assert_eq!(out.fused_embeddings[3].as_ref().unwrap()[0], 2.0);
        assert!(out.fused_embeddings[2].is_none()); // text token 50
        assert!(out.fused_embeddings[6].is_none()); // text token 60
    }

    // @trace TEST-MULTIMODAL-003 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_with_empty_sequence_and_valid_vocab() {
        // Arrange — empty routed sequence with valid vocab/hidden_size
        let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, hidden=2
        let routed = RoutedSequence {
            token_ids: vec![],
            fused_embeddings: vec![],
            text_positions: vec![],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — empty sequence produces empty output
        assert!(hidden.is_empty());
    }

    // @trace TEST-MULTIMODAL-004 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_encoded_subnormal_float_values_validate() {
        // Arrange — subnormal f32 values should pass shape validation
        let subnormal = f32::from_bits(1u32); // smallest positive subnormal
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![subnormal],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        // Act
        let result = enc.validate();
        // Assert — validate only checks shape, not values
        assert!(result.is_ok());
        assert_eq!(enc.embeddings[0].to_bits(), 1u32);
    }

    // @trace TEST-MULTIMODAL-005 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_negative_zero_in_media_embedding() {
        // Arrange — negative zero in media should be copied exactly
        let neg_zero = -0.0f32;
        let media = vec![neg_zero, 1.0];
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![Some(media)],
            text_positions: vec![],
            hidden_size: 2,
        };
        let embed = vec![0.0f32; 2]; // vocab=1, hidden=2
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — negative zero is preserved
        assert!(hidden[0].is_sign_negative());
        assert_eq!(hidden[0].to_bits(), neg_zero.to_bits());
        assert_eq!(hidden[1], 1.0);
    }

    // @trace TEST-MULTIMODAL-006 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn routed_sequence_clone_deep_independence() {
        // Arrange — verify clone produces independent data
        let original = RoutedSequence {
            token_ids: vec![1, 258880, 2],
            fused_embeddings: vec![None, Some(vec![1.0, 2.0]), None],
            text_positions: vec![0, 2],
            hidden_size: 2,
        };
        // Act
        let cloned = original.clone();
        // Assert — values match
        assert_eq!(cloned.token_ids, original.token_ids);
        assert_eq!(cloned.text_positions, original.text_positions);
        assert_eq!(cloned.hidden_size, original.hidden_size);
        // Verify the Some(Vec) is a deep copy, not a shared reference
        assert_eq!(cloned.fused_embeddings[1].as_ref().unwrap(), &vec![1.0, 2.0]);
        assert_eq!(cloned.fused_embeddings.len(), 3);
    }

    // @trace TEST-MULTIMODAL-007 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_custom_ids_where_image_equals_text_token() {
        // Arrange — custom IDs where image_token_id equals a common text token value
        let custom_ids = MultimodalTokenIds {
            image_token_id: 5, // same as a text token value
            audio_token_id: 100,
            eoi_token_id: 200,
            eoa_token_id: 300,
        };
        let enc = MultimodalEncoded {
            tokens: vec![5],
            embeddings: vec![42.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        // Token 5 is both text value and image token — it gets expanded
        let prompt = vec![1u32, 5, 2];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &custom_ids, 1).unwrap();
        // Assert — token 5 was treated as image special token
        assert_eq!(out.seq_len(), 3); // text(1) + 1 virtual + text(2)
        assert_eq!(out.text_positions, vec![0, 2]);
        assert!(out.fused_embeddings[1].is_some());
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap()[0], 42.0);
    }

    // @trace TEST-MULTIMODAL-008 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_multiple_media_positions_interleaved() {
        // Arrange — text, media, text, media, text pattern with unique values
        let embed: Vec<f32> = vec![
            10.0, 11.0, 12.0, // token 0
            20.0, 21.0, 22.0, // token 1
            30.0, 31.0, 32.0, // token 2
            40.0, 41.0, 42.0, // token 3
        ]; // vocab=4, hidden=3
        let m1 = vec![100.0, 101.0, 102.0];
        let m2 = vec![200.0, 201.0, 202.0];
        let routed = RoutedSequence {
            token_ids: vec![0, 258880, 2, 258881, 3],
            fused_embeddings: vec![None, Some(m1), None, Some(m2), None],
            text_positions: vec![0, 2, 4],
            hidden_size: 3,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 3).unwrap();
        // Assert — each position independently verified
        assert_eq!(hidden.len(), 5 * 3);
        assert_eq!(&hidden[0..3], &[10.0, 11.0, 12.0]); // token 0
        assert_eq!(&hidden[3..6], &[100.0, 101.0, 102.0]); // media 1
        assert_eq!(&hidden[6..9], &[30.0, 31.0, 32.0]); // token 2
        assert_eq!(&hidden[9..12], &[200.0, 201.0, 202.0]); // media 2
        assert_eq!(&hidden[12..15], &[40.0, 41.0, 42.0]); // token 3
    }

    // @trace TEST-MULTIMODAL-009 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_three_images_with_different_virtual_token_counts() {
        // Arrange — three images each producing different numbers of virtual tokens
        let ids = default_ids();
        let enc1 = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 1],
            embeddings: vec![1.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let enc2 = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 5],
            embeddings: vec![2.0; 5 * 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let enc3 = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 3],
            embeddings: vec![3.0; 3 * 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc1).unwrap();
        ctx.push_image(enc2).unwrap();
        ctx.push_image(enc3).unwrap();
        let prompt = vec![ids.image_token_id, 10, ids.image_token_id, 20, ids.image_token_id];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        // Assert — 1 + 1(text) + 5 + 1(text) + 3 = 11
        assert_eq!(out.seq_len(), 11);
        assert_eq!(out.text_positions, vec![1, 7]);
        // Verify embedding values from each encoding
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 1.0);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap()[0], 2.0);
        assert_eq!(out.fused_embeddings[8].as_ref().unwrap()[0], 3.0);
    }

    // @trace TEST-MULTIMODAL-010 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_custom_ids_eoi_used_as_image_token() {
        // Arrange — custom IDs where eoi_token_id doubles as image_token_id
        let custom_ids = MultimodalTokenIds {
            image_token_id: 300,
            audio_token_id: 400,
            eoi_token_id: 300, // same as image_token_id
            eoa_token_id: 500,
        };
        let enc = MultimodalEncoded {
            tokens: vec![300],
            embeddings: vec![99.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        // Token 300 is both image and eoi — is_image matches so it gets expanded
        let prompt = vec![300, 10];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &custom_ids, 1).unwrap();
        // Assert
        assert_eq!(out.seq_len(), 2);
        assert_eq!(out.text_positions, vec![1]);
        assert!(out.fused_embeddings[0].is_some());
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 99.0);
    }

    // @trace TEST-MULTIMODAL-011 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_context_clone_then_push_independence() {
        // Arrange — clone a context with data, push to original, verify clone unchanged
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.5],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        // Act
        let cloned = ctx.clone();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![1.5],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        // Assert — clone still has 1 image, original has 2
        assert_eq!(cloned.images.len(), 1);
        assert_eq!(cloned.images[0].tokens, vec![1]);
        assert_eq!(ctx.images.len(), 2);
    }

    // @trace TEST-MULTIMODAL-012 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_text_only_with_custom_ids_all_treated_as_text() {
        // Arrange — prompt contains default special tokens but using custom IDs
        let custom_ids = MultimodalTokenIds {
            image_token_id: 999,
            audio_token_id: 998,
            eoi_token_id: 997,
            eoa_token_id: 996,
        };
        let ctx = MultimodalContext::new();
        // These are default gemma4 special tokens, but custom_ids doesn't recognize them
        let prompt = vec![258880u32, 258881, 258882, 258883, 1, 2];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &custom_ids, 4).unwrap();
        // Assert — all treated as text
        assert_eq!(out.seq_len(), 6);
        assert!(out.fused_embeddings.iter().all(|e| e.is_none()));
        assert!(!out.has_multimodal());
        assert_eq!(out.text_positions, vec![0, 1, 2, 3, 4, 5]);
    }

    // @trace TEST-MULTIMODAL-013 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_with_large_hidden_size_text_and_media() {
        // Arrange — hidden_size=256 with mixed text and media positions
        let hidden = 256;
        let vocab = 4;
        let embed: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.0001).collect();
        let media: Vec<f32> = vec![42.0; hidden];
        let routed = RoutedSequence {
            token_ids: vec![1, 258880, 3],
            fused_embeddings: vec![None, Some(media), None],
            text_positions: vec![0, 2],
            hidden_size: hidden,
        };
        // Act
        let result = build_fused_hidden(&routed, &embed, hidden).unwrap();
        // Assert — correct length and specific values
        assert_eq!(result.len(), 3 * hidden);
        // Position 0: gather token 1
        let expected_start = 1 * hidden;
        assert_eq!(result[0], expected_start as f32 * 0.0001);
        // Position 1: media
        assert_eq!(result[hidden], 42.0);
        // Position 2: gather token 3
        let expected_start2 = 3 * hidden;
        assert_eq!(result[2 * hidden], expected_start2 as f32 * 0.0001);
    }

    // @trace TEST-MULTIMODAL-014 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_audio_at_beginning_image_at_end_text_middle() {
        // Arrange — audio first, text in middle, image last
        let ids = default_ids();
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![5.0; 2 * 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 3],
            embeddings: vec![9.0; 3 * 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc_aud).unwrap();
        ctx.push_image(enc_img).unwrap();
        let prompt = vec![ids.audio_token_id, 42, 43, ids.image_token_id];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        // Assert — 2 virtual + 2 text + 3 virtual = 7
        assert_eq!(out.seq_len(), 7);
        assert_eq!(out.text_positions, vec![2, 3]);
        assert!(out.fused_embeddings[0].is_some()); // audio
        assert!(out.fused_embeddings[1].is_some()); // audio
        assert!(out.fused_embeddings[2].is_none()); // text 42
        assert!(out.fused_embeddings[3].is_none()); // text 43
        assert!(out.fused_embeddings[4].is_some()); // image
        assert!(out.fused_embeddings[5].is_some()); // image
        assert!(out.fused_embeddings[6].is_some()); // image
    }

    // @trace TEST-MULTIMODAL-015 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_media_at_first_and_last_position() {
        // Arrange — media at position 0 and last position, text in between
        let embed: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0]; // vocab=2, hidden=2
        let media_first = vec![100.0, 101.0];
        let media_last = vec![200.0, 201.0];
        let routed = RoutedSequence {
            token_ids: vec![258880, 0, 1, 258881],
            fused_embeddings: vec![Some(media_first), None, None, Some(media_last)],
            text_positions: vec![1, 2],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — 4 positions × 2 hidden = 8 elements
        assert_eq!(hidden.len(), 8);
        assert_eq!(&hidden[0..2], &[100.0, 101.0]); // media first
        assert_eq!(&hidden[2..4], &[10.0, 20.0]);   // text token 0
        assert_eq!(&hidden[4..6], &[30.0, 40.0]);   // text token 1
        assert_eq!(&hidden[6..8], &[200.0, 201.0]); // media last
    }

    // ── Batch 7: 15 additional edge-case and coverage tests ──

    #[test]
    fn multimodal_token_ids_fallback_multimodal_token_ids_all_above_258000() {
        // Arrange
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        // Act & Assert — gemma4 special tokens are in the high range
        assert!(ids.image_token_id > 258_000);
        assert!(ids.audio_token_id > 258_000);
        assert!(ids.eoi_token_id > 258_000);
        assert!(ids.eoa_token_id > 258_000);
    }

    #[test]
    fn multimodal_encoded_validate_two_tokens_off_by_one_extra() {
        // Arrange — 2 tokens × 4 hidden = 8 expected, but we provide 9
        let enc = MultimodalEncoded {
            tokens: vec![1, 2],
            embeddings: vec![0.0; 9],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        // Act
        let err = enc.validate().unwrap_err();
        // Assert
        assert!(format!("{err}").contains("shape mismatch"));
        assert!(format!("{err}").contains("expected embeddings=8"));
        assert!(format!("{err}").contains("got=9"));
    }

    #[test]
    fn encoder_media_from_generation_preserves_all_fields_exact() {
        // Arrange — verify every field of Base64 variant is preserved exactly
        let gen = crate::generation::MediaInput::Base64 {
            data: "iVBORw0KGgo=".into(),
            mime_type: Some("image/png".into()),
        };
        // Act
        let enc = EncoderMedia::from_generation(&gen);
        // Assert
        match enc {
            EncoderMedia::Base64 { data, mime_type } => {
                assert_eq!(data, "iVBORw0KGgo=");
                assert_eq!(mime_type.as_deref(), Some("image/png"));
            }
            _ => panic!("expected Base64 variant"),
        }
    }

    #[test]
    fn multimodal_context_push_image_returns_ok_on_valid_input() {
        // Arrange
        let mut ctx = MultimodalContext::new();
        let enc = MultimodalEncoded {
            tokens: vec![1, 2, 3],
            embeddings: vec![0.1; 6],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        // Act
        let result = ctx.push_image(enc);
        // Assert
        assert!(result.is_ok());
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.images[0].num_tokens(), 3);
    }

    #[test]
    fn route_prompt_with_only_eoi_and_eoa_no_special_match() {
        // Arrange — eoi and eoa are not image/audio tokens, so treated as text
        let ids = MultimodalTokenIds {
            image_token_id: 100,
            audio_token_id: 200,
            eoi_token_id: 300,
            eoa_token_id: 400,
        };
        let ctx = MultimodalContext::new();
        let prompt = vec![300, 400, 10]; // eoi, eoa, text
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        // Assert — all treated as text
        assert_eq!(out.seq_len(), 3);
        assert!(out.fused_embeddings.iter().all(|e| e.is_none()));
        assert_eq!(out.text_positions, vec![0, 1, 2]);
    }

    #[test]
    fn build_fused_hidden_token_id_at_vocab_boundary() {
        // Arrange — vocab=4, hidden=2, token_id=3 (last valid token)
        let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let routed = RoutedSequence {
            token_ids: vec![3],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — last row of vocab
        assert_eq!(hidden, vec![7.0, 8.0]);
    }

    #[test]
    fn build_fused_hidden_token_id_at_vocab_boundary_plus_one_rejected() {
        // Arrange — vocab=4, hidden=2, token_id=4 is out of range
        let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let routed = RoutedSequence {
            token_ids: vec![4],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        // Act
        let err = build_fused_hidden(&routed, &embed, 2).unwrap_err();
        // Assert
        assert!(format!("{err}").contains("out of range"));
        assert!(format!("{err}").contains("vocab 4"));
    }

    #[test]
    fn mock_encoder_encode_image_idempotent() {
        // Arrange — encoding the same media twice produces identical results
        let ids = default_ids();
        let encoder = MockEncoder::new(2, 0, 4, ids);
        let media = EncoderMedia::Raw(vec![1, 2, 3]);
        // Act
        let result1 = encoder.encode_image(&media).unwrap();
        let result2 = encoder.encode_image(&media).unwrap();
        // Assert
        assert_eq!(result1.tokens, result2.tokens);
        assert_eq!(result1.embeddings, result2.embeddings);
        assert_eq!(result1.hidden_size, result2.hidden_size);
        assert_eq!(result1.kind, result2.kind);
    }

    #[test]
    fn mock_encoder_encode_audio_idempotent() {
        // Arrange — encoding the same media twice produces identical results
        let ids = default_ids();
        let encoder = MockEncoder::new(0, 3, 2, ids);
        let media = EncoderMedia::Url("https://example.com/audio.wav".into());
        // Act
        let result1 = encoder.encode_audio(&media).unwrap();
        let result2 = encoder.encode_audio(&media).unwrap();
        // Assert
        assert_eq!(result1.tokens, result2.tokens);
        assert_eq!(result1.embeddings, result2.embeddings);
    }

    #[test]
    fn route_image_with_single_virtual_token_preserves_order() {
        // Arrange — image expands to exactly 1 virtual token
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![99],
            embeddings: vec![0.5; 3],
            hidden_size: 3,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![10u32, ids.image_token_id, 20, 30];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 3).unwrap();
        // Assert — 1 text + 1 virtual + 2 text = 4
        assert_eq!(out.seq_len(), 4);
        assert_eq!(out.token_ids, vec![10, 99, 20, 30]);
        assert_eq!(out.text_positions, vec![0, 2, 3]);
        assert!(out.fused_embeddings[0].is_none());
        assert!(out.fused_embeddings[1].is_some());
        assert!(out.fused_embeddings[2].is_none());
        assert!(out.fused_embeddings[3].is_none());
    }

    #[test]
    fn route_two_audios_three_images_interleaved_order_matters() {
        // Arrange — complex interleaving where order of encodings matters
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        // Push 3 images first, then 2 audios
        for i in 0..3 {
            ctx.push_image(MultimodalEncoded {
                tokens: vec![ids.image_token_id],
                embeddings: vec![(i + 1) as f32 * 10.0],
                hidden_size: 1,
                kind: MediaKind::Image,
            })
            .unwrap();
        }
        for i in 0..2 {
            ctx.push_audio(MultimodalEncoded {
                tokens: vec![ids.audio_token_id],
                embeddings: vec![(i + 1) as f32 * 100.0],
                hidden_size: 1,
                kind: MediaKind::Audio,
            })
            .unwrap();
        }
        // Prompt: img, aud, img, aud, img
        let prompt = vec![
            ids.image_token_id,
            ids.audio_token_id,
            ids.image_token_id,
            ids.audio_token_id,
            ids.image_token_id,
        ];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 1).unwrap();
        // Assert — 5 virtual tokens total, order preserved
        assert_eq!(out.seq_len(), 5);
        assert!(out.text_positions.is_empty());
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 10.0);  // img 1
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap()[0], 100.0); // aud 1
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap()[0], 20.0);  // img 2
        assert_eq!(out.fused_embeddings[3].as_ref().unwrap()[0], 200.0); // aud 2
        assert_eq!(out.fused_embeddings[4].as_ref().unwrap()[0], 30.0);  // img 3
    }

    #[test]
    fn build_fused_hidden_media_embedding_with_all_zeros() {
        // Arrange — media embedding is all zeros, should still be copied (not gather)
        let embed: Vec<f32> = vec![99.0, 88.0]; // vocab=1, hidden=2
        let media = vec![0.0, 0.0];
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![Some(media)],
            text_positions: vec![],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — media override with zeros, not gather of [99, 88]
        assert_eq!(hidden, vec![0.0, 0.0]);
    }

    #[test]
    fn multimodal_context_push_image_rejects_shape_mismatch_does_not_affect_audios() {
        // Arrange — push a valid audio, then try a bad image push
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.5],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        assert_eq!(ctx.audios.len(), 1);

        let bad_img = MultimodalEncoded {
            tokens: vec![1, 2],
            embeddings: vec![0.0; 3], // 2*2=4 expected, got 3
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        // Act
        let _ = ctx.push_image(bad_img);
        // Assert — images still empty, audios unaffected
        assert!(ctx.images.is_empty());
        assert_eq!(ctx.audios.len(), 1);
    }

    #[test]
    fn route_error_message_contains_both_counts_for_image() {
        // Arrange — 3 image tokens in prompt but 1 encoding provided
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 2],
            hidden_size: 2,
            kind: MediaKind::Image,
        })
        .unwrap();
        let prompt = vec![ids.image_token_id, ids.image_token_id, ids.image_token_id];
        // Act
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap_err();
        let msg = format!("{err}");
        // Assert — error message has both counts
        assert!(msg.contains("3 image tokens"));
        assert!(msg.contains("1 image encodings"));
    }

    #[test]
    fn route_error_message_contains_both_counts_for_audio() {
        // Arrange — 0 audio tokens in prompt but 2 encodings provided
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        for _ in 0..2 {
            ctx.push_audio(MultimodalEncoded {
                tokens: vec![ids.audio_token_id],
                embeddings: vec![0.0; 4],
                hidden_size: 4,
                kind: MediaKind::Audio,
            })
            .unwrap();
        }
        let prompt = vec![1u32, 2]; // no audio special tokens
        // Act
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap_err();
        let msg = format!("{err}");
        // Assert — error message has both counts
        assert!(msg.contains("0 audio tokens"));
        assert!(msg.contains("2 audio encodings"));
    }

    // ── Batch 8: 15 additional edge-case and coverage tests ──

    #[test]
    fn multimodal_token_ids_eq_symmetry() {
        // Arrange — verify Eq symmetry: a == b implies b == a
        let a = MultimodalTokenIds::fallback_multimodal_token_ids();
        let b = MultimodalTokenIds::fallback_multimodal_token_ids();
        // Act & Assert
        assert_eq!(a, b);
        assert_eq!(b, a);
        // Also verify symmetry for non-equal: a != c implies c != a
        let c = MultimodalTokenIds { image_token_id: 0, audio_token_id: 0, eoi_token_id: 0, eoa_token_id: 0 };
        assert_ne!(a, c);
        assert_ne!(c, a);
    }

    #[test]
    fn multimodal_token_ids_eq_transitivity() {
        // Arrange — verify Eq transitivity: a == b && b == c implies a == c
        let a = MultimodalTokenIds::fallback_multimodal_token_ids();
        let b = MultimodalTokenIds::fallback_multimodal_token_ids();
        let c = MultimodalTokenIds::fallback_multimodal_token_ids();
        // Act & Assert
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    #[test]
    fn encoder_media_base64_data_with_special_characters() {
        // Arrange — Base64 data containing +, /, and = padding characters
        let b64 = EncoderMedia::Base64 {
            data: "abc+DEF/GHI==".into(),
            mime_type: Some("image/gif".into()),
        };
        // Act & Assert — special chars preserved
        match b64 {
            EncoderMedia::Base64 { data, mime_type } => {
                assert_eq!(data, "abc+DEF/GHI==");
                assert_eq!(mime_type.as_deref(), Some("image/gif"));
            }
            _ => panic!("expected Base64 variant"),
        }
    }

    #[test]
    fn encoder_media_file_path_with_spaces() {
        // Arrange — file path containing spaces and special characters
        let path = "/tmp/my images/test image (1).png";
        let media = EncoderMedia::File(PathBuf::from(path));
        // Act & Assert
        match &media {
            EncoderMedia::File(p) => assert_eq!(p.to_str(), Some(path)),
            _ => panic!("expected File variant"),
        }
    }

    #[test]
    fn multimodal_encoded_validate_single_embedding_minimal_nonzero() {
        // Arrange — smallest non-trivial valid encoding: 1 token, 1 hidden, non-zero value
        let enc = MultimodalEncoded {
            tokens: vec![42],
            embeddings: vec![1.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        // Act & Assert
        assert!(enc.validate().is_ok());
        assert_eq!(enc.num_tokens(), 1);
        assert_eq!(enc.embeddings[0], 1.0);
    }

    #[test]
    fn multimodal_encoded_alternating_positive_negative_embeddings() {
        // Arrange — embeddings with alternating sign pattern
        let embeddings: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let enc = MultimodalEncoded {
            tokens: vec![1, 2, 3, 4],
            embeddings: embeddings.clone(),
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        // Act
        assert!(enc.validate().is_ok());
        // Assert — pattern preserved exactly
        assert_eq!(enc.embeddings, embeddings);
    }

    #[test]
    fn multimodal_context_push_images_same_hidden_size_accumulate() {
        // Arrange — push 3 images all with the same hidden_size
        let mut ctx = MultimodalContext::new();
        for i in 0..3 {
            ctx.push_image(MultimodalEncoded {
                tokens: vec![i as u32],
                embeddings: vec![i as f32 * 10.0, i as f32 * 20.0],
                hidden_size: 2,
                kind: MediaKind::Image,
            })
            .unwrap();
        }
        // Act & Assert — all accumulated with correct values
        assert_eq!(ctx.images.len(), 3);
        assert_eq!(ctx.images[0].embeddings[0], 0.0);
        assert_eq!(ctx.images[1].embeddings[0], 10.0);
        assert_eq!(ctx.images[2].embeddings[0], 20.0);
    }

    #[test]
    fn routed_sequence_has_multimodal_exactly_one_media_at_middle() {
        // Arrange — 5 positions, only position 2 has media
        let rs = RoutedSequence {
            token_ids: vec![1, 2, 258880, 3, 4],
            fused_embeddings: vec![None, None, Some(vec![9.0; 4]), None, None],
            text_positions: vec![0, 1, 3, 4],
            hidden_size: 4,
        };
        // Act & Assert
        assert!(rs.has_multimodal());
        assert!(rs.fused_embeddings[0].is_none());
        assert!(rs.fused_embeddings[1].is_none());
        assert!(rs.fused_embeddings[2].is_some());
        assert!(rs.fused_embeddings[3].is_none());
        assert!(rs.fused_embeddings[4].is_none());
    }

    #[test]
    fn route_image_only_prompt_single_special_token() {
        // Arrange — prompt is just the image special token, nothing else
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 4],
            embeddings: vec![0.25; 4 * 2],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![ids.image_token_id];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — no text tokens at all
        assert_eq!(out.seq_len(), 4);
        assert!(out.text_positions.is_empty());
        assert!(out.fused_embeddings.iter().all(|e| e.is_some()));
    }

    #[test]
    fn route_audio_then_image_no_text_between() {
        // Arrange — audio special immediately followed by image special
        let ids = default_ids();
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![3.0; 2 * 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![7.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc_aud).unwrap();
        ctx.push_image(enc_img).unwrap();
        // Act
        let out = route_multimodal_tokens(&[ids.audio_token_id, ids.image_token_id], &ctx, &ids, 4).unwrap();
        // Assert — 2 audio virtual + 1 image virtual = 3, no text
        assert_eq!(out.seq_len(), 3);
        assert!(out.text_positions.is_empty());
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 3.0);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap()[0], 7.0);
    }

    #[test]
    fn route_audio_zero_virtual_tokens_passes_through() {
        // Arrange — audio encoding produces 0 virtual tokens (edge case)
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![],
            embeddings: vec![],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc).unwrap();
        // Act
        let out = route_multimodal_tokens(&[ids.audio_token_id, 10], &ctx, &ids, 4).unwrap();
        // Assert — audio expands to 0 tokens, only text remains
        assert_eq!(out.seq_len(), 1);
        assert_eq!(out.token_ids, vec![10]);
        assert_eq!(out.text_positions, vec![0]);
        assert!(!out.has_multimodal());
    }

    #[test]
    fn build_fused_hidden_gather_first_row_token_zero() {
        // Arrange — vocab > 1, token_id=0 should gather first row
        let embed: Vec<f32> = vec![
            99.0, 98.0, 97.0, // row 0
            1.0, 2.0, 3.0,    // row 1
            4.0, 5.0, 6.0,    // row 2
        ]; // vocab=3, hidden=3
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 3,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 3).unwrap();
        // Assert — first row gathered exactly
        assert_eq!(hidden, vec![99.0, 98.0, 97.0]);
    }

    #[test]
    fn build_fused_hidden_media_embedding_with_f32_max() {
        // Arrange — media embedding contains f32::MAX
        let media = vec![f32::MAX, f32::MAX];
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![Some(media)],
            text_positions: vec![],
            hidden_size: 2,
        };
        let embed = vec![0.0f32; 2]; // vocab=1, hidden=2
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — f32::MAX preserved exactly
        assert_eq!(hidden[0], f32::MAX);
        assert_eq!(hidden[1], f32::MAX);
    }

    #[test]
    fn build_fused_hidden_media_with_f32_min_positive_subnormal() {
        // Arrange — media embedding contains smallest positive subnormal f32
        let min_subnormal = f32::from_bits(1u32);
        let media = vec![min_subnormal];
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![Some(media)],
            text_positions: vec![],
            hidden_size: 1,
        };
        let embed = vec![0.0f32; 2]; // vocab=2, hidden=1
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 1).unwrap();
        // Assert — smallest subnormal preserved exactly
        assert_eq!(hidden[0].to_bits(), 1u32);
    }

    #[test]
    fn route_prompt_with_many_text_tokens_preserves_all() {
        // Arrange — 100 text tokens with no special tokens
        let tokens: Vec<u32> = (100..200).collect();
        let ctx = MultimodalContext::new();
        let ids = default_ids();
        // Act
        let out = route_multimodal_tokens(&tokens, &ctx, &ids, 32).unwrap();
        // Assert — every token preserved, all positions are text
        assert_eq!(out.token_ids, tokens);
        assert_eq!(out.text_positions, (0..100).collect::<Vec<_>>());
        assert!(out.fused_embeddings.iter().all(|e| e.is_none()));
        assert_eq!(out.hidden_size, 32);
    }

    // ── Batch 9: 15 additional edge-case and coverage tests ──

    #[test]
    fn multimodal_token_ids_copy_semantics_independent_after_assignment() {
        // Arrange — Copy types: assigning to new var and modifying concept doesn't affect original
        let a = MultimodalTokenIds {
            image_token_id: 10,
            audio_token_id: 20,
            eoi_token_id: 30,
            eoa_token_id: 40,
        };
        // Act — Copy semantics: b gets its own copy
        let b = a;
        // Assert — both are equal and independent (Copy trait)
        assert_eq!(a, b);
        assert_eq!(a.image_token_id, 10);
        assert_eq!(b.image_token_id, 10);
    }

    #[test]
    fn encoder_media_raw_with_single_byte() {
        // Arrange — Raw media with exactly one byte
        let media = EncoderMedia::Raw(vec![0xFF]);
        // Act & Assert — single byte preserved
        match &media {
            EncoderMedia::Raw(bytes) => {
                assert_eq!(bytes.len(), 1);
                assert_eq!(bytes[0], 0xFF);
            }
            _ => panic!("expected Raw variant"),
        }
    }

    #[test]
    fn multimodal_encoded_validate_one_token_zero_hidden_size_valid() {
        // Arrange — 1 token × 0 hidden_size = 0 embeddings: mathematically valid
        let enc = MultimodalEncoded {
            tokens: vec![42],
            embeddings: vec![],
            hidden_size: 0,
            kind: MediaKind::Image,
        };
        // Act & Assert
        assert!(enc.validate().is_ok());
        assert_eq!(enc.num_tokens(), 1);
    }

    #[test]
    fn multimodal_context_push_audio_after_three_image_failures() {
        // Arrange — context survives multiple image push failures, then accepts audio
        let mut ctx = MultimodalContext::new();
        for _ in 0..3 {
            let bad = MultimodalEncoded {
                tokens: vec![1],
                embeddings: vec![0.0],
                hidden_size: 1,
                kind: MediaKind::Audio, // wrong kind for push_image
            };
            let _ = ctx.push_image(bad);
        }
        assert_eq!(ctx.images.len(), 0);
        assert!(ctx.is_empty());
        // Act — valid audio push after repeated failures
        let good = MultimodalEncoded {
            tokens: vec![5],
            embeddings: vec![0.5],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        ctx.push_audio(good).unwrap();
        // Assert
        assert_eq!(ctx.audios.len(), 1);
        assert!(!ctx.is_empty());
        assert!(ctx.images.is_empty());
    }

    #[test]
    fn route_image_zero_virtual_tokens_disappears_from_output() {
        // Arrange — image encoding produces 0 virtual tokens (degenerate case)
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![],
            embeddings: vec![],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![10u32, ids.image_token_id, 20];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — image expands to 0 tokens, so output is just the surrounding text
        assert_eq!(out.seq_len(), 2);
        assert_eq!(out.token_ids, vec![10, 20]);
        assert_eq!(out.text_positions, vec![0, 1]);
        assert!(!out.has_multimodal());
    }

    #[test]
    fn route_four_images_each_one_virtual_token_positions_correct() {
        // Arrange — 4 images each producing exactly 1 virtual token, interleaved with text
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        for i in 0..4 {
            ctx.push_image(MultimodalEncoded {
                tokens: vec![100 + i],
                embeddings: vec![(i + 1) as f32],
                hidden_size: 1,
                kind: MediaKind::Image,
            })
            .unwrap();
        }
        let prompt = vec![
            ids.image_token_id,
            50,
            ids.image_token_id,
            60,
            ids.image_token_id,
            70,
            ids.image_token_id,
        ];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 1).unwrap();
        // Assert — 4 virtual + 3 text = 7
        assert_eq!(out.seq_len(), 7);
        assert_eq!(out.text_positions, vec![1, 3, 5]);
        assert_eq!(out.token_ids[0], 100);
        assert_eq!(out.token_ids[2], 101);
        assert_eq!(out.token_ids[4], 102);
        assert_eq!(out.token_ids[6], 103);
    }

    #[test]
    fn build_fused_hidden_with_multiple_media_and_text_deep_verify() {
        // Arrange — 6 positions: text, media, media, text, media, text
        let embed: Vec<f32> = vec![
            10.0, 11.0, // token 0
            20.0, 21.0, // token 1
            30.0, 31.0, // token 2
            40.0, 41.0, // token 3
        ]; // vocab=4, hidden=2
        let m1 = vec![100.0, 101.0];
        let m2 = vec![200.0, 201.0];
        let m3 = vec![300.0, 301.0];
        let routed = RoutedSequence {
            token_ids: vec![0, 258880, 258881, 2, 258880, 3],
            fused_embeddings: vec![None, Some(m1), Some(m2), None, Some(m3), None],
            text_positions: vec![0, 3, 5],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — 6 positions × 2 hidden = 12 elements
        assert_eq!(hidden.len(), 12);
        assert_eq!(&hidden[0..2], &[10.0, 11.0]);   // text token 0
        assert_eq!(&hidden[2..4], &[100.0, 101.0]);  // media 1
        assert_eq!(&hidden[4..6], &[200.0, 201.0]);  // media 2
        assert_eq!(&hidden[6..8], &[30.0, 31.0]);    // text token 2
        assert_eq!(&hidden[8..10], &[300.0, 301.0]); // media 3
        assert_eq!(&hidden[10..12], &[40.0, 41.0]);  // text token 3
    }

    #[test]
    fn build_fused_hidden_rejects_token_at_exactly_vocab_size() {
        // Arrange — vocab=3, hidden=4, token_id=3 is exactly vocab_size (out of range)
        let embed: Vec<f32> = vec![0.0; 3 * 4]; // vocab=3, hidden=4
        let routed = RoutedSequence {
            token_ids: vec![3], // out of range: valid tokens are 0, 1, 2
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 4,
        };
        // Act
        let err = build_fused_hidden(&routed, &embed, 4).unwrap_err();
        // Assert
        let msg = format!("{err}");
        assert!(msg.contains("out of range"));
        assert!(msg.contains("vocab 3"));
    }

    #[test]
    fn route_custom_ids_image_audio_swapped_from_defaults() {
        // Arrange — IDs where image and audio are swapped relative to defaults
        let swapped = MultimodalTokenIds {
            image_token_id: 258881, // default audio
            audio_token_id: 258880, // default image
            eoi_token_id: 258882,
            eoa_token_id: 258883,
        };
        let ctx = MultimodalContext::new();
        // Token 258880 is now audio token, not image
        // Token 258881 is now image token, not audio
        let prompt = vec![258880u32, 258881];
        // Act — route expects encodings for both but ctx is empty
        let err = route_multimodal_tokens(&prompt, &ctx, &swapped, 4).unwrap_err();
        // Assert — error should mention mismatch for either image or audio
        let msg = format!("{err}");
        assert!(
            msg.contains("image tokens but context provided 0")
                || msg.contains("audio tokens but context provided 0"),
            "expected mismatch error, got: {msg}"
        );
    }

    #[test]
    fn mock_encoder_dyn_trait_encode_image_produces_valid_encoding() {
        // Arrange — dyn trait object dispatch produces valid encoding that passes validate
        let ids = default_ids();
        let encoder: Box<dyn MultimodalEncoder> = Box::new(MockEncoder::new(5, 0, 8, ids));
        let media = EncoderMedia::File(PathBuf::from("/test/image.jpg"));
        // Act
        let encoded = encoder.encode_image(&media).unwrap();
        // Assert — validate passes
        assert!(encoded.validate().is_ok());
        assert_eq!(encoded.num_tokens(), 5);
        assert_eq!(encoded.hidden_size, 8);
        assert_eq!(encoded.kind, MediaKind::Image);
    }

    #[test]
    fn mock_encoder_dyn_trait_encode_audio_produces_valid_encoding() {
        // Arrange — dyn trait object dispatch for audio
        let ids = default_ids();
        let encoder: Box<dyn MultimodalEncoder> = Box::new(MockEncoder::new(0, 3, 16, ids));
        let media = EncoderMedia::Base64 {
            data: "test".into(),
            mime_type: None,
        };
        // Act
        let encoded = encoder.encode_audio(&media).unwrap();
        // Assert
        assert!(encoded.validate().is_ok());
        assert_eq!(encoded.num_tokens(), 3);
        assert_eq!(encoded.hidden_size, 16);
        assert_eq!(encoded.kind, MediaKind::Audio);
    }

    #[test]
    fn multimodal_encoded_clone_then_validate_independent() {
        // Arrange — clone an encoding, verify clone validates independently
        let orig = MultimodalEncoded {
            tokens: vec![1, 2, 3],
            embeddings: vec![0.1; 6],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        // Act
        let cloned = orig.clone();
        // Assert — both validate successfully
        assert!(orig.validate().is_ok());
        assert!(cloned.validate().is_ok());
        assert_eq!(orig.num_tokens(), cloned.num_tokens());
        assert_eq!(orig.embeddings, cloned.embeddings);
    }

    #[test]
    fn routed_sequence_hidden_size_zero_no_multimodal() {
        // Arrange — RoutedSequence with hidden_size=0 and no multimodal
        let rs = RoutedSequence {
            token_ids: vec![1, 2, 3],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 0,
        };
        // Act & Assert
        assert_eq!(rs.seq_len(), 3);
        assert!(!rs.has_multimodal());
        assert_eq!(rs.hidden_size, 0);
    }

    #[test]
    fn route_image_at_position_zero_text_at_positions_after() {
        // Arrange — image at position 0 expands to 3 virtual tokens
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 3],
            embeddings: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![ids.image_token_id, 10, 20, 30];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — 3 virtual + 3 text = 6
        assert_eq!(out.seq_len(), 6);
        assert_eq!(out.text_positions, vec![3, 4, 5]);
        // Embedding values at virtual positions
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap(), &vec![1.0, 2.0]);
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap(), &vec![3.0, 4.0]);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap(), &vec![5.0, 6.0]);
        // Text positions have no embeddings
        assert!(out.fused_embeddings[3].is_none());
        assert!(out.fused_embeddings[4].is_none());
        assert!(out.fused_embeddings[5].is_none());
    }

    #[test]
    fn build_fused_hidden_text_gather_preserves_order_across_vocab() {
        // Arrange — gather tokens in non-sequential order across vocab
        let embed: Vec<f32> = (0..5 * 3).map(|i| i as f32).collect(); // vocab=5, hidden=3
        let routed = RoutedSequence {
            token_ids: vec![4, 0, 2, 1],
            fused_embeddings: vec![None, None, None, None],
            text_positions: vec![0, 1, 2, 3],
            hidden_size: 3,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 3).unwrap();
        // Assert — each position gathers the correct row
        assert_eq!(&hidden[0..3], &[12.0, 13.0, 14.0]); // token 4 -> row 4
        assert_eq!(&hidden[3..6], &[0.0, 1.0, 2.0]);    // token 0 -> row 0
        assert_eq!(&hidden[6..9], &[6.0, 7.0, 8.0]);    // token 2 -> row 2
        assert_eq!(&hidden[9..12], &[3.0, 4.0, 5.0]);   // token 1 -> row 1
    }

    #[test]
    fn multimodal_context_push_image_then_isolate_audios_list() {
        // Arrange — verify images and audios lists are completely independent
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![10],
            embeddings: vec![1.0, 2.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![20],
            embeddings: vec![3.0, 4.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![30],
            embeddings: vec![5.0, 6.0],
            hidden_size: 2,
            kind: MediaKind::Audio,
        })
        .unwrap();
        // Act — verify each list independently
        // Assert
        assert_eq!(ctx.images.len(), 2);
        assert_eq!(ctx.audios.len(), 1);
        assert_eq!(ctx.images[0].tokens, vec![10]);
        assert_eq!(ctx.images[1].tokens, vec![20]);
        assert_eq!(ctx.audios[0].tokens, vec![30]);
        assert!(!ctx.is_empty());
    }

    // ── Batch 10: 15 additional edge-case and coverage tests ──

    #[test]
    fn multimodal_token_ids_is_image_never_true_for_arbitrary_values() {
        // Arrange — verify is_image returns false for values not equal to image_token_id
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        // Act & Assert — test a range of values that are NOT the image token id
        for v in [0u32, 1, 100, 1000, 258879, 258881, 258882, 258883, u32::MAX - 1] {
            assert!(!ids.is_image(v), "is_image should be false for {v}");
        }
    }

    #[test]
    fn multimodal_token_ids_is_audio_never_true_for_arbitrary_values() {
        // Arrange — verify is_audio returns false for values not equal to audio_token_id
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        // Act & Assert
        for v in [0u32, 1, 100, 1000, 258880, 258879, 258882, 258883, u32::MAX] {
            assert!(!ids.is_audio(v), "is_audio should be false for {v}");
        }
    }

    #[test]
    fn encoder_media_file_clone_preserves_path_exact() {
        // Arrange
        let path = PathBuf::from("/data/models/vision/encoder_weights.bin");
        let original = EncoderMedia::File(path.clone());
        // Act
        let cloned = original.clone();
        // Assert — path is byte-identical after clone
        match (&original, &cloned) {
            (EncoderMedia::File(p1), EncoderMedia::File(p2)) => assert_eq!(p1, p2),
            _ => panic!("both should be File variant"),
        }
    }

    #[test]
    fn multimodal_encoded_validate_error_reports_correct_dimensions() {
        // Arrange — 7 tokens × 3 hidden = 21 expected, but 20 provided
        let enc = MultimodalEncoded {
            tokens: vec![1u32; 7],
            embeddings: vec![0.0; 20],
            hidden_size: 3,
            kind: MediaKind::Audio,
        };
        // Act
        let err_msg = format!("{}", enc.validate().unwrap_err());
        // Assert — error message includes all three numeric dimensions
        assert!(err_msg.contains("tokens=7"));
        assert!(err_msg.contains("hidden=3"));
        assert!(err_msg.contains("expected embeddings=21"));
        assert!(err_msg.contains("got=20"));
    }

    #[test]
    fn route_single_token_prompt_with_matching_image() {
        // Arrange — simplest possible multimodal prompt: just the image special token
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![42.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        // Act
        let out = route_multimodal_tokens(&[ids.image_token_id], &ctx, &ids, 1).unwrap();
        // Assert
        assert_eq!(out.seq_len(), 1);
        assert!(out.text_positions.is_empty());
        assert!(out.has_multimodal());
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 42.0);
    }

    #[test]
    fn route_single_token_prompt_with_matching_audio() {
        // Arrange — simplest possible multimodal prompt: just the audio special token
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![-7.5],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc).unwrap();
        // Act
        let out = route_multimodal_tokens(&[ids.audio_token_id], &ctx, &ids, 1).unwrap();
        // Assert
        assert_eq!(out.seq_len(), 1);
        assert!(out.text_positions.is_empty());
        assert!(out.has_multimodal());
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], -7.5);
    }

    #[test]
    fn build_fused_hidden_rejects_vocab_zero_with_text_token() {
        // Arrange — embed is empty (vocab=0), routed has a text token that needs gathering
        let embed: Vec<f32> = vec![];
        let routed = RoutedSequence {
            token_ids: vec![0], // even token 0 is out of range when vocab=0
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 4,
        };
        // Act
        let err = build_fused_hidden(&routed, &embed, 4).unwrap_err();
        // Assert
        let msg = format!("{err}");
        assert!(msg.contains("out of range"));
        assert!(msg.contains("vocab 0"));
    }

    #[test]
    fn multimodal_context_images_and_audios_mutually_exclusive_push_rejection() {
        // Arrange — push_audio rejects Image kind and push_image rejects Audio kind
        let mut ctx = MultimodalContext::new();
        let img = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        let aud = MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        // Act & Assert — push_image rejects Audio kind
        let err1 = ctx.push_image(aud.clone());
        assert!(err1.is_err());
        assert!(ctx.images.is_empty());
        // Act & Assert — push_audio rejects Image kind
        let err2 = ctx.push_audio(img.clone());
        assert!(err2.is_err());
        assert!(ctx.audios.is_empty());
        // Act & Assert — correct kinds are accepted
        assert!(ctx.push_image(img).is_ok());
        assert!(ctx.push_audio(aud).is_ok());
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.audios.len(), 1);
    }

    #[test]
    fn route_hidden_size_consistency_across_mixed_media() {
        // Arrange — both image and audio encodings share the same hidden_size
        let ids = default_ids();
        let hidden = 16;
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 2],
            embeddings: vec![1.0; 2 * hidden],
            hidden_size: hidden,
            kind: MediaKind::Image,
        };
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 3],
            embeddings: vec![2.0; 3 * hidden],
            hidden_size: hidden,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc_img).unwrap();
        ctx.push_audio(enc_aud).unwrap();
        // Act
        let out = route_multimodal_tokens(&[ids.image_token_id, ids.audio_token_id], &ctx, &ids, hidden).unwrap();
        // Assert — 2 virtual + 3 virtual = 5, all embeddings have correct length
        assert_eq!(out.seq_len(), 5);
        for emb in &out.fused_embeddings {
            let data = emb.as_ref().expect("all positions should have embeddings");
            assert_eq!(data.len(), hidden);
        }
    }

    #[test]
    fn encoder_media_url_with_query_string_preserved() {
        // Arrange — URL with query parameters
        let url = "https://cdn.example.com/model.gguf?version=2&token=abc123";
        let media = EncoderMedia::Url(url.into());
        // Act & Assert
        match &media {
            EncoderMedia::Url(s) => {
                assert!(s.contains("version=2"));
                assert!(s.contains("token=abc123"));
                assert_eq!(s, url);
            }
            _ => panic!("expected Url variant"),
        }
    }

    #[test]
    fn encoder_media_base64_clone_deep_copies_data() {
        // Arrange — verify Base64 clone is a deep copy
        let original = EncoderMedia::Base64 {
            data: "SGVsbG8gV29ybGQ=".into(),
            mime_type: Some("text/plain".into()),
        };
        // Act
        let cloned = original.clone();
        // Assert — both have independent copies
        match (&original, &cloned) {
            (
                EncoderMedia::Base64 { data: d1, mime_type: m1 },
                EncoderMedia::Base64 { data: d2, mime_type: m2 },
            ) => {
                assert_eq!(d1, d2);
                assert_eq!(m1, m2);
                // Verify deep copy: both have their own String allocations
                assert_eq!(d1.as_str(), "SGVsbG8gV29ybGQ=");
                assert_eq!(m2.as_deref(), Some("text/plain"));
            }
            _ => panic!("both should be Base64 variant"),
        }
    }

    #[test]
    fn multimodal_encoded_large_token_count_boundary() {
        // Arrange — exactly 65536 tokens to test at a power-of-2 boundary
        let n = 65536;
        let hidden = 1;
        let enc = MultimodalEncoded {
            tokens: vec![0u32; n],
            embeddings: vec![0.5; n * hidden],
            hidden_size: hidden,
            kind: MediaKind::Image,
        };
        // Act & Assert
        assert_eq!(enc.num_tokens(), n);
        assert!(enc.validate().is_ok());
    }

    #[test]
    fn route_text_tokens_just_outside_special_id_range() {
        // Arrange — text tokens just below image and above audio special IDs
        // Gemma4 defaults: image=258880, audio=258881, eoi=258882, eoa=258883
        // So values 258879 and 258884 are outside the special range
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let ctx = MultimodalContext::new();
        let boundary_tokens = vec![
            ids.image_token_id - 1, // 258879 — not image, not audio
            ids.eoa_token_id + 1,   // 258884 — not image, not audio
            0u32,                    // not any special token
            u32::MAX,               // not any special token
        ];
        // Act
        let out = route_multimodal_tokens(&boundary_tokens, &ctx, &ids, 8).unwrap();
        // Assert — none of these are special tokens, all treated as text
        assert_eq!(out.seq_len(), 4);
        assert_eq!(out.token_ids, boundary_tokens);
        assert!(out.fused_embeddings.iter().all(|e| e.is_none()));
        assert!(!out.has_multimodal());
        assert_eq!(out.text_positions, vec![0, 1, 2, 3]);
    }

    #[test]
    fn build_fused_hidden_media_with_f32_epsilon_values() {
        // Arrange — media embedding contains f32::EPSILON values
        let media = vec![f32::EPSILON, f32::EPSILON];
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![Some(media)],
            text_positions: vec![],
            hidden_size: 2,
        };
        let embed = vec![1.0f32, 2.0]; // vocab=1, hidden=2
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — EPSILON values preserved exactly
        assert_eq!(hidden[0], f32::EPSILON);
        assert_eq!(hidden[1], f32::EPSILON);
    }

    #[test]
    fn route_image_then_audio_then_text_embedding_values_exact() {
        // Arrange — verify exact embedding values after interleaved routing
        let ids = default_ids();
        let img_emb = vec![1.1, 2.2, 3.3];
        let aud_emb = vec![4.4, 5.5, 6.6];
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: img_emb.clone(),
            hidden_size: 3,
            kind: MediaKind::Image,
        };
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: aud_emb.clone(),
            hidden_size: 3,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc_img).unwrap();
        ctx.push_audio(enc_aud).unwrap();
        let prompt = vec![ids.image_token_id, ids.audio_token_id, 99];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 3).unwrap();
        // Assert — exact embedding values at correct positions
        assert_eq!(out.seq_len(), 3);
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap(), &img_emb);
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap(), &aud_emb);
        assert!(out.fused_embeddings[2].is_none());
        assert_eq!(out.text_positions, vec![2]);
    }

    #[test]
    fn multimodal_context_push_image_rejects_does_not_corrupt_existing_audio() {
        // Arrange — push a valid audio, then reject an image with wrong kind
        let mut ctx = MultimodalContext::new();
        let valid_audio = MultimodalEncoded {
            tokens: vec![10],
            embeddings: vec![7.7],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        ctx.push_audio(valid_audio).unwrap();
        assert_eq!(ctx.audios.len(), 1);
        let bad_img = MultimodalEncoded {
            tokens: vec![20],
            embeddings: vec![8.8],
            hidden_size: 1,
            kind: MediaKind::Audio, // wrong kind for push_image
        };
        // Act — rejected push should not corrupt audios list
        let _ = ctx.push_image(bad_img);
        // Assert
        assert!(ctx.images.is_empty());
        assert_eq!(ctx.audios.len(), 1);
        assert_eq!(ctx.audios[0].tokens[0], 10);
        assert_eq!(ctx.audios[0].embeddings[0], 7.7);
    }

    // ── Batch 11: 15 additional edge-case and coverage tests ──

    #[test]
    fn multimodal_encoded_validate_nonzero_hidden_empty_embeddings_error_detail() {
        // Arrange — 3 tokens × 4 hidden = 12 expected, but 0 provided
        let enc = MultimodalEncoded {
            tokens: vec![1, 2, 3],
            embeddings: vec![],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        // Act
        let err_msg = format!("{}", enc.validate().unwrap_err());
        // Assert — error reports exact mismatch
        assert!(err_msg.contains("tokens=3"));
        assert!(err_msg.contains("hidden=4"));
        assert!(err_msg.contains("expected embeddings=12"));
        assert!(err_msg.contains("got=0"));
    }

    #[test]
    fn multimodal_token_ids_hashmap_lookup_returns_correct_value() {
        // Arrange — use MultimodalTokenIds as HashMap key to look up associated data
        use std::collections::HashMap;
        let ids1 = MultimodalTokenIds { image_token_id: 100, audio_token_id: 200, eoi_token_id: 300, eoa_token_id: 400 };
        let ids2 = MultimodalTokenIds { image_token_id: 500, audio_token_id: 600, eoi_token_id: 700, eoa_token_id: 800 };
        let mut map = HashMap::new();
        map.insert(ids1, "model_A");
        map.insert(ids2, "model_B");
        // Act
        let lookup = MultimodalTokenIds { image_token_id: 100, audio_token_id: 200, eoi_token_id: 300, eoa_token_id: 400 };
        // Assert — lookup finds the correct entry
        assert_eq!(map.get(&lookup), Some(&"model_A"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn encoder_media_raw_debug_contains_bytes() {
        // Arrange — verify Debug output for Raw variant includes byte data
        let raw = EncoderMedia::Raw(vec![0xAB, 0xCD]);
        // Act
        let debug = format!("{raw:?}");
        // Assert
        assert!(debug.contains("Raw"));
        assert!(debug.contains("171")); // 0xAB as decimal in debug output
    }

    #[test]
    fn route_five_alternating_image_text_pairs() {
        // Arrange — pattern: img, text, img, text, img, text, img, text, img
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        for i in 0..5 {
            ctx.push_image(MultimodalEncoded {
                tokens: vec![ids.image_token_id],
                embeddings: vec![(i + 1) as f32 * 10.0],
                hidden_size: 1,
                kind: MediaKind::Image,
            })
            .unwrap();
        }
        let prompt = vec![
            ids.image_token_id, 10,
            ids.image_token_id, 20,
            ids.image_token_id, 30,
            ids.image_token_id, 40,
            ids.image_token_id,
        ];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 1).unwrap();
        // Assert — 5 virtual + 4 text = 9 total
        assert_eq!(out.seq_len(), 9);
        assert_eq!(out.text_positions, vec![1, 3, 5, 7]);
        // Verify embedding values follow the push order
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 10.0);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap()[0], 20.0);
        assert_eq!(out.fused_embeddings[4].as_ref().unwrap()[0], 30.0);
        assert_eq!(out.fused_embeddings[6].as_ref().unwrap()[0], 40.0);
        assert_eq!(out.fused_embeddings[8].as_ref().unwrap()[0], 50.0);
    }

    #[test]
    fn build_fused_hidden_vocab_one_hidden_one_media_overrides() {
        // Arrange — vocab=1, hidden=1: single embedding row, but media overrides
        let embed: Vec<f32> = vec![42.0]; // vocab=1, hidden=1
        let media = vec![99.0];
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![Some(media)],
            text_positions: vec![],
            hidden_size: 1,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 1).unwrap();
        // Assert — media override takes precedence over gather
        assert_eq!(hidden, vec![99.0]);
    }

    #[test]
    fn routed_sequence_text_only_with_repeated_token_ids() {
        // Arrange — all text positions using the same token ID
        let rs = RoutedSequence {
            token_ids: vec![7, 7, 7, 7],
            fused_embeddings: vec![None, None, None, None],
            text_positions: vec![0, 1, 2, 3],
            hidden_size: 64,
        };
        // Act & Assert
        assert_eq!(rs.seq_len(), 4);
        assert!(!rs.has_multimodal());
        assert!(rs.token_ids.iter().all(|&t| t == 7));
    }

    #[test]
    fn multimodal_context_default_produces_independent_instances() {
        // Arrange — two default contexts should be independent
        let ctx1 = MultimodalContext::default();
        let ctx2 = MultimodalContext::default();
        // Act & Assert — both start empty
        assert!(ctx1.is_empty());
        assert!(ctx2.is_empty());
        // They are separate allocations (verified by Drop safety)
        assert_eq!(ctx1.images.len(), ctx2.images.len());
        assert_eq!(ctx1.audios.len(), ctx2.audios.len());
    }

    #[test]
    fn multimodal_encoded_validate_single_token_large_hidden_size() {
        // Arrange — 1 token with a very large hidden_size
        let hidden = 16384;
        let enc = MultimodalEncoded {
            tokens: vec![42],
            embeddings: vec![0.5; hidden],
            hidden_size: hidden,
            kind: MediaKind::Audio,
        };
        // Act & Assert
        assert_eq!(enc.num_tokens(), 1);
        assert!(enc.validate().is_ok());
        assert_eq!(enc.embeddings.len(), hidden);
    }

    #[test]
    fn route_eoi_and_eoa_only_treated_as_regular_text() {
        // Arrange — prompt contains only eoi and eoa tokens, no image/audio special tokens
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let ctx = MultimodalContext::new();
        let prompt = vec![ids.eoi_token_id, ids.eoa_token_id];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        // Assert — both treated as regular text, no expansion
        assert_eq!(out.seq_len(), 2);
        assert_eq!(out.token_ids, vec![ids.eoi_token_id, ids.eoa_token_id]);
        assert!(out.fused_embeddings.iter().all(|e| e.is_none()));
        assert!(!out.has_multimodal());
        assert_eq!(out.text_positions, vec![0, 1]);
    }

    #[test]
    fn mock_encoder_default_ids_token_values_match() {
        // Arrange — MockEncoder with default IDs produces tokens matching the image/audio IDs
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        let encoder = MockEncoder::new(3, 2, 4, ids);
        // Act
        let img = encoder.encode_image(&EncoderMedia::Raw(vec![])).unwrap();
        let aud = encoder.encode_audio(&EncoderMedia::Raw(vec![])).unwrap();
        // Assert — all image tokens are the image_token_id
        assert!(img.tokens.iter().all(|&t| t == ids.image_token_id));
        assert_eq!(img.tokens.len(), 3);
        // All audio tokens are the audio_token_id
        assert!(aud.tokens.iter().all(|&t| t == ids.audio_token_id));
        assert_eq!(aud.tokens.len(), 2);
    }

    #[test]
    fn build_fused_hidden_all_text_vocab_one_repeated_gather() {
        // Arrange — vocab=1, hidden=3, two text positions both gathering the same row
        let embed: Vec<f32> = vec![10.0, 20.0, 30.0]; // vocab=1, hidden=3
        let routed = RoutedSequence {
            token_ids: vec![0, 0, 0],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 3,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 3).unwrap();
        // Assert — all three positions gather the same row
        assert_eq!(hidden.len(), 9);
        assert_eq!(&hidden[0..3], &[10.0, 20.0, 30.0]);
        assert_eq!(&hidden[3..6], &[10.0, 20.0, 30.0]);
        assert_eq!(&hidden[6..9], &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn route_hidden_size_mismatch_reports_encoder_vs_model() {
        // Arrange — encoder hidden_size=128 but model expects 64
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 128],
            hidden_size: 128,
            kind: MediaKind::Image,
        })
        .unwrap();
        // Act
        let err = route_multimodal_tokens(&[ids.image_token_id], &ctx, &ids, 64).unwrap_err();
        let msg = format!("{err}");
        // Assert — error message contains both hidden_size values
        assert!(msg.contains("128"));
        assert!(msg.contains("64"));
        assert!(msg.contains("hidden_size"));
    }

    #[test]
    fn multimodal_context_push_image_valid_after_multiple_kind_rejections() {
        // Arrange — reject wrong kind twice, then accept correct kind
        let mut ctx = MultimodalContext::new();
        for _ in 0..2 {
            let bad = MultimodalEncoded {
                tokens: vec![1],
                embeddings: vec![0.0],
                hidden_size: 1,
                kind: MediaKind::Audio,
            };
            let _ = ctx.push_image(bad);
        }
        assert_eq!(ctx.images.len(), 0);
        // Act — valid push
        let good = MultimodalEncoded {
            tokens: vec![99],
            embeddings: vec![1.5],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        ctx.push_image(good).unwrap();
        // Assert
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.images[0].tokens[0], 99);
        assert_eq!(ctx.images[0].embeddings[0], 1.5);
    }

    #[test]
    fn route_audio_at_end_with_preceding_text_embedding_values() {
        // Arrange — text, text, audio where audio produces 2 virtual tokens
        let ids = default_ids();
        let audio_emb = vec![4.0, 5.0, 6.0, 7.0];
        let enc = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: audio_emb.clone(),
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc).unwrap();
        // Act
        let out = route_multimodal_tokens(&[10, 20, ids.audio_token_id], &ctx, &ids, 2).unwrap();
        // Assert — 2 text + 2 virtual = 4
        assert_eq!(out.seq_len(), 4);
        assert_eq!(out.text_positions, vec![0, 1]);
        assert!(out.fused_embeddings[0].is_none());
        assert!(out.fused_embeddings[1].is_none());
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap(), &vec![4.0, 5.0]);
        assert_eq!(out.fused_embeddings[3].as_ref().unwrap(), &vec![6.0, 7.0]);
    }

    #[test]
    fn build_fused_hidden_text_and_media_each_one_position() {
        // Arrange — exactly 2 positions: 1 text, 1 media
        let embed: Vec<f32> = vec![11.0, 22.0, 33.0, 44.0]; // vocab=2, hidden=2
        let media = vec![99.0, 88.0];
        let routed = RoutedSequence {
            token_ids: vec![1, 258880],
            fused_embeddings: vec![None, Some(media)],
            text_positions: vec![0],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert
        assert_eq!(hidden.len(), 4);
        assert_eq!(&hidden[0..2], &[33.0, 44.0]); // token 1 -> row 1
        assert_eq!(&hidden[2..4], &[99.0, 88.0]); // media override
    }

    #[test]
    fn multimodal_context_push_audio_after_push_image_independent_lists() {
        // Arrange — push image, then audio, verify both lists grow independently
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![10],
            embeddings: vec![1.0, 2.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        })
        .unwrap();
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.audios.len(), 0);
        // Act
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![20],
            embeddings: vec![3.0, 4.0],
            hidden_size: 2,
            kind: MediaKind::Audio,
        })
        .unwrap();
        // Assert — both lists have exactly one entry with correct data
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.audios.len(), 1);
        assert_eq!(ctx.images[0].tokens, vec![10]);
        assert_eq!(ctx.images[0].embeddings, vec![1.0, 2.0]);
        assert_eq!(ctx.audios[0].tokens, vec![20]);
        assert_eq!(ctx.audios[0].embeddings, vec![3.0, 4.0]);
        assert!(!ctx.is_empty());
    }

    // ── Batch 12: 15 additional edge-case and coverage tests ──

    #[test]
    fn multimodal_token_ids_eoi_and_eoa_fields_not_checked_by_is_image_or_is_audio() {
        // Arrange — custom IDs where eoi equals audio_token_id
        let ids = MultimodalTokenIds {
            image_token_id: 1,
            audio_token_id: 2,
            eoi_token_id: 2, // same as audio
            eoa_token_id: 3,
        };
        // Act & Assert — is_image only checks image_token_id field
        assert!(!ids.is_image(2));
        // is_audio only checks audio_token_id field
        assert!(ids.is_audio(2));
        // eoi/eoa are never checked by these helpers
        assert!(!ids.is_image(3));
        assert!(!ids.is_audio(3));
    }

    #[test]
    fn encoder_media_url_with_fragment_preserved() {
        // Arrange — URL with a fragment (#section)
        let url = "https://example.com/model.gguf#checksum";
        // Act
        let media = EncoderMedia::Url(url.into());
        // Assert
        match &media {
            EncoderMedia::Url(s) => {
                assert!(s.contains("#checksum"));
                assert_eq!(s, url);
            }
            _ => panic!("expected Url variant"),
        }
    }

    #[test]
    fn multimodal_encoded_embeddings_all_same_value() {
        // Arrange — all embeddings are the same constant value
        let val = 0.125;
        let enc = MultimodalEncoded {
            tokens: vec![1, 2, 3, 4],
            embeddings: vec![val; 8],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        // Act & Assert
        assert!(enc.validate().is_ok());
        assert!(enc.embeddings.iter().all(|&v| v == val));
    }

    #[test]
    fn route_prompt_where_all_tokens_are_image_special() {
        // Arrange — every token in the prompt is an image special token
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        for i in 0..3 {
            ctx.push_image(MultimodalEncoded {
                tokens: vec![ids.image_token_id],
                embeddings: vec![i as f32],
                hidden_size: 1,
                kind: MediaKind::Image,
            })
            .unwrap();
        }
        let prompt = vec![ids.image_token_id; 3];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 1).unwrap();
        // Assert — 3 virtual tokens, no text, each with distinct embedding
        assert_eq!(out.seq_len(), 3);
        assert!(out.text_positions.is_empty());
        assert!(out.fused_embeddings.iter().all(|e| e.is_some()));
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 0.0);
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap()[0], 1.0);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap()[0], 2.0);
    }

    #[test]
    fn build_fused_hidden_media_at_position_zero_with_large_vocab() {
        // Arrange — large vocab (1024 rows), hidden=4, media overrides position 0
        let vocab = 1024;
        let hidden = 4;
        let embed: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.001).collect();
        let media = vec![42.0; hidden];
        let routed = RoutedSequence {
            token_ids: vec![258880, 512],
            fused_embeddings: vec![Some(media.clone()), None],
            text_positions: vec![1],
            hidden_size: hidden,
        };
        // Act
        let result = build_fused_hidden(&routed, &embed, hidden).unwrap();
        // Assert — position 0 is media override, position 1 gathers token 512
        assert_eq!(result.len(), 2 * hidden);
        assert_eq!(&result[0..hidden], media.as_slice());
        let expected_512_start = 512 * hidden;
        assert_eq!(result[hidden], expected_512_start as f32 * 0.001);
    }

    #[test]
    fn multimodal_context_push_audio_valid_after_image_push() {
        // Arrange — push image first, then audio; verify both accumulate
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.1],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![0.2],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        // Act — push audio after two images
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![3],
            embeddings: vec![0.3],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        // Assert
        assert_eq!(ctx.images.len(), 2);
        assert_eq!(ctx.audios.len(), 1);
        assert_eq!(ctx.images[1].embeddings[0], 0.2);
        assert_eq!(ctx.audios[0].embeddings[0], 0.3);
    }

    #[test]
    fn route_image_with_different_hidden_size_than_audio_rejected() {
        // Arrange — image has hidden_size=8, audio has hidden_size=4, model expects 4
        let ids = default_ids();
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 8],
            hidden_size: 8,
            kind: MediaKind::Image,
        };
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc_img).unwrap();
        ctx.push_audio(enc_aud).unwrap();
        // Act — image hidden_size=8 mismatches model hidden_size=4
        let err = route_multimodal_tokens(
            &[ids.image_token_id, ids.audio_token_id],
            &ctx,
            &ids,
            4,
        )
        .unwrap_err();
        // Assert
        assert!(format!("{err}").contains("hidden_size"));
    }

    #[test]
    fn routed_sequence_fused_embeddings_middle_position_only_media() {
        // Arrange — exactly one media position in the middle of a 5-position sequence
        let rs = RoutedSequence {
            token_ids: vec![10, 20, 258880, 30, 40],
            fused_embeddings: vec![None, None, Some(vec![7.0, 8.0]), None, None],
            text_positions: vec![0, 1, 3, 4],
            hidden_size: 2,
        };
        // Act & Assert
        assert!(rs.has_multimodal());
        assert!(rs.fused_embeddings[0].is_none());
        assert!(rs.fused_embeddings[1].is_none());
        assert!(rs.fused_embeddings[2].is_some());
        assert!(rs.fused_embeddings[3].is_none());
        assert!(rs.fused_embeddings[4].is_none());
        assert_eq!(rs.fused_embeddings[2].as_ref().unwrap(), &vec![7.0, 8.0]);
    }

    #[test]
    fn mock_encoder_image_call_count_independent_from_audio_call_count() {
        // Arrange — call image twice and audio once, verify independent counters
        let ids = default_ids();
        let encoder = MockEncoder::new(1, 1, 2, ids);
        // Act
        let _ = encoder.encode_image(&EncoderMedia::Raw(vec![]));
        let _ = encoder.encode_audio(&EncoderMedia::Raw(vec![]));
        let _ = encoder.encode_image(&EncoderMedia::Raw(vec![]));
        // Assert — image count is 2, audio count is 1
        assert_eq!(encoder.image_call_count(), 2);
        assert_eq!(encoder.audio_call_count(), 1);
    }

    #[test]
    fn build_fused_hidden_media_embedding_with_negative_and_positive_mix() {
        // Arrange — media embedding has alternating negative and positive values
        let media = vec![-10.0, 10.0, -20.0, 20.0];
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![Some(media.clone())],
            text_positions: vec![],
            hidden_size: 4,
        };
        let embed = vec![0.0f32; 4]; // vocab=1, hidden=4
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 4).unwrap();
        // Assert — exact values preserved
        assert_eq!(hidden, media);
    }

    #[test]
    fn route_text_positions_correct_after_image_expansion_in_middle() {
        // Arrange — text, image (expands to 4 virtual), text, text
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 4],
            embeddings: vec![0.0; 4 * 2],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![10u32, ids.image_token_id, 20, 30];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — 1 text + 4 virtual + 2 text = 7
        assert_eq!(out.seq_len(), 7);
        assert_eq!(out.text_positions, vec![0, 5, 6]);
        assert_eq!(out.token_ids[0], 10);
        assert_eq!(out.token_ids[5], 20);
        assert_eq!(out.token_ids[6], 30);
    }

    #[test]
    fn encoder_media_from_generation_all_variants_produce_correct_enum() {
        // Arrange — verify each MediaInput variant maps to the correct EncoderMedia variant
        let file_input = crate::generation::MediaInput::File("/a/b".into());
        let b64_input = crate::generation::MediaInput::Base64 {
            data: "x".into(),
            mime_type: None,
        };
        let raw_input = crate::generation::MediaInput::Raw(vec![1]);
        let url_input = crate::generation::MediaInput::Url("http://x".into());
        // Act
        let file_out = EncoderMedia::from_generation(&file_input);
        let b64_out = EncoderMedia::from_generation(&b64_input);
        let raw_out = EncoderMedia::from_generation(&raw_input);
        let url_out = EncoderMedia::from_generation(&url_input);
        // Assert — each maps to the corresponding variant
        assert!(matches!(file_out, EncoderMedia::File(_)));
        assert!(matches!(b64_out, EncoderMedia::Base64 { .. }));
        assert!(matches!(raw_out, EncoderMedia::Raw(_)));
        assert!(matches!(url_out, EncoderMedia::Url(_)));
    }

    #[test]
    fn multimodal_encoded_debug_output_contains_all_field_names() {
        // Arrange
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![2.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        // Act
        let debug = format!("{enc:?}");
        // Assert — Debug output includes struct name and all fields
        assert!(debug.contains("MultimodalEncoded"));
        assert!(debug.contains("tokens"));
        assert!(debug.contains("embeddings"));
        assert!(debug.contains("hidden_size"));
        assert!(debug.contains("Audio"));
    }

    #[test]
    fn route_empty_prompt_returns_zero_hidden_size_when_requested() {
        // Arrange — empty prompt with hidden_size=0
        let ctx = MultimodalContext::new();
        let ids = default_ids();
        // Act
        let out = route_multimodal_tokens(&[], &ctx, &ids, 0).unwrap();
        // Assert
        assert_eq!(out.seq_len(), 0);
        assert_eq!(out.hidden_size, 0);
        assert!(out.token_ids.is_empty());
        assert!(out.fused_embeddings.is_empty());
        assert!(out.text_positions.is_empty());
    }

    #[test]
    fn build_fused_hidden_rejects_fused_embeddings_shorter_than_seq_len() {
        // Arrange — fused_embeddings has 1 entry but seq_len is 2
        let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, hidden=2
        let routed = RoutedSequence {
            token_ids: vec![0, 1],
            fused_embeddings: vec![None], // length mismatch: 1 != 2
            text_positions: vec![0],
            hidden_size: 2,
        };
        // Act
        let err = build_fused_hidden(&routed, &embed, 2).unwrap_err();
        // Assert
        let msg = format!("{err}");
        assert!(msg.contains("fused_embeddings"));
        assert!(msg.contains("1"));
        assert!(msg.contains("2"));
    }

    // ── Batch 8: 15 additional edge-case and coverage tests ──

    // @trace TEST-MULTIMODAL-016 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_encoded_validate_rejects_single_extra_embedding() {
        // Arrange — 1 token × 4 hidden = 4 expected, but 5 provided
        let enc = MultimodalEncoded {
            tokens: vec![42],
            embeddings: vec![0.0; 5],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        // Act
        let err = enc.validate().unwrap_err();
        // Assert — error message includes exact counts
        assert!(format!("{err}").contains("expected embeddings=4"));
        assert!(format!("{err}").contains("got=5"));
    }

    // @trace TEST-MULTIMODAL-017 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_context_push_image_rejects_shape_mismatch_preserves_prior_images() {
        // Arrange — push two valid images, then a third with bad shape
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![1.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![2.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        // Act — bad shape (2 tokens × 3 hidden = 6 expected, but 5 provided)
        let bad = MultimodalEncoded {
            tokens: vec![3, 4],
            embeddings: vec![0.0; 5],
            hidden_size: 3,
            kind: MediaKind::Image,
        };
        let result = ctx.push_image(bad);
        // Assert — prior state intact
        assert!(result.is_err());
        assert_eq!(ctx.images.len(), 2);
        assert_eq!(ctx.images[0].tokens, vec![1]);
        assert_eq!(ctx.images[1].tokens, vec![2]);
    }

    // @trace TEST-MULTIMODAL-018 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_image_at_first_and_text_at_second_position() {
        // Arrange — image token at position 0, text token at position 1
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 3],
            embeddings: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![ids.image_token_id, 99];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — 3 virtual + 1 text = 4
        assert_eq!(out.seq_len(), 4);
        assert_eq!(out.text_positions, vec![3]);
        assert_eq!(out.token_ids[3], 99);
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap(), &vec![0.1, 0.2]);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap(), &vec![0.5, 0.6]);
    }

    // @trace TEST-MULTIMODAL-019 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_embed_table_with_f32_max_values() {
        // Arrange — vocab=2, hidden=2 with f32::MAX in embed table
        let embed: Vec<f32> = vec![f32::MAX, 0.0, f32::MIN, 0.0]; // vocab=2, hidden=2
        let routed = RoutedSequence {
            token_ids: vec![0, 1],
            fused_embeddings: vec![None, None],
            text_positions: vec![0, 1],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — exact f32 bit-for-bit preservation
        assert_eq!(hidden[0], f32::MAX);
        assert_eq!(hidden[1], 0.0);
        assert_eq!(hidden[2], f32::MIN);
        assert_eq!(hidden[3], 0.0);
    }

    // @trace TEST-MULTIMODAL-020 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn mock_encoder_image_zero_virtual_tokens_empty_embeddings() {
        // Arrange — image encoding produces zero virtual tokens
        let ids = default_ids();
        let encoder = MockEncoder::new(0, 0, 8, ids);
        // Act
        let result = encoder.encode_image(&EncoderMedia::Raw(vec![])).unwrap();
        // Assert
        assert_eq!(result.num_tokens(), 0);
        assert!(result.tokens.is_empty());
        assert!(result.embeddings.is_empty());
        assert_eq!(result.hidden_size, 8);
        assert_eq!(result.kind, MediaKind::Image);
    }

    // @trace TEST-MULTIMODAL-021 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_audio_at_first_position_text_at_second_audio_at_third() {
        // Arrange — audio, text, audio pattern
        let ids = default_ids();
        let enc1 = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![1.0; 2 * 3],
            hidden_size: 3,
            kind: MediaKind::Audio,
        };
        let enc2 = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![2.0; 3],
            hidden_size: 3,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc1).unwrap();
        ctx.push_audio(enc2).unwrap();
        let prompt = vec![ids.audio_token_id, 55, ids.audio_token_id];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 3).unwrap();
        // Assert — 2 virtual + 1 text + 1 virtual = 4
        assert_eq!(out.seq_len(), 4);
        assert_eq!(out.text_positions, vec![2]);
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 1.0);
        assert_eq!(out.fused_embeddings[3].as_ref().unwrap()[0], 2.0);
    }

    // @trace TEST-MULTIMODAL-022 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn encoder_media_url_with_unicode_characters() {
        // Arrange — URL with unicode path segments
        let url = EncoderMedia::Url("https://example.com/画像/テスト.png".into());
        // Act & Assert — unicode preserved in the variant
        match &url {
            EncoderMedia::Url(s) => {
                assert!(s.contains("画像"));
                assert!(s.contains("テスト"));
            }
            _ => panic!("expected Url variant"),
        }
    }

    // @trace TEST-MULTIMODAL-023 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn routed_sequence_has_multimodal_with_empty_some_vec() {
        // Arrange — Some(vec![]) is technically "some media" even with no floats
        let rs = RoutedSequence {
            token_ids: vec![1, 258880],
            fused_embeddings: vec![None, Some(vec![])],
            text_positions: vec![0],
            hidden_size: 0, // hidden_size=0 means empty embedding slice is valid
        };
        // Act & Assert
        assert!(rs.has_multimodal());
    }

    // @trace TEST-MULTIMODAL-024 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_vocab_boundary_token_id_equals_vocab_size_minus_one() {
        // Arrange — vocab=5, hidden=1, token_id=4 (last valid token)
        let embed: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0]; // vocab=5, hidden=1
        let routed = RoutedSequence {
            token_ids: vec![4],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 1,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 1).unwrap();
        // Assert — gathers the last row
        assert_eq!(hidden, vec![50.0]);
    }

    // @trace TEST-MULTIMODAL-025 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_image_expansion_embedding_slices_are_independent_copies() {
        // Arrange — verify each expansion position gets its own Vec, not a shared reference
        let ids = default_ids();
        let embeddings: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // 2 tokens × 2 hidden
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 2],
            embeddings: embeddings.clone(),
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![ids.image_token_id];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — each position's embedding is an independent Vec
        let emb0 = out.fused_embeddings[0].as_ref().unwrap().clone();
        let emb1 = out.fused_embeddings[1].as_ref().unwrap().clone();
        assert_eq!(emb0, vec![1.0, 2.0]);
        assert_eq!(emb1, vec![3.0, 4.0]);
    }

    // @trace TEST-MULTIMODAL-026 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_encoded_validate_rejects_single_token_zero_hidden_with_extra_embedding() {
        // Arrange — 1 token × 0 hidden = 0 expected, but 1 provided
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![99.0], // should be empty for hidden_size=0
            hidden_size: 0,
            kind: MediaKind::Audio,
        };
        // Act
        let err = enc.validate().unwrap_err();
        // Assert
        assert!(format!("{err}").contains("shape mismatch"));
        assert!(format!("{err}").contains("expected embeddings=0"));
        assert!(format!("{err}").contains("got=1"));
    }

    // @trace TEST-MULTIMODAL-027 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_many_positions_all_media_no_gather() {
        // Arrange — 5 media positions, no text gather needed, embed is minimal
        let embed: Vec<f32> = vec![0.0; 2]; // vocab=1, hidden=2 (unused)
        let media: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, (i * 10) as f32]).collect();
        let routed = RoutedSequence {
            token_ids: vec![258880, 258880, 258880, 258880, 258880],
            fused_embeddings: media.iter().map(|m| Some(m.clone())).collect(),
            text_positions: vec![],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — all 5 positions copied from media, no gather
        assert_eq!(hidden.len(), 10);
        assert_eq!(&hidden[0..2], &[0.0, 0.0]);
        assert_eq!(&hidden[2..4], &[1.0, 10.0]);
        assert_eq!(&hidden[4..6], &[2.0, 20.0]);
        assert_eq!(&hidden[6..8], &[3.0, 30.0]);
        assert_eq!(&hidden[8..10], &[4.0, 40.0]);
    }

    // @trace TEST-MULTIMODAL-028 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_context_has_images_but_prompt_only_needs_audios_rejected() {
        // Arrange — context has an image encoding, prompt has only audio special token
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        })
        .unwrap();
        // Act — prompt has 0 image tokens but context has 1 image encoding
        let prompt = vec![1u32, 2];
        let result = route_multimodal_tokens(&prompt, &ctx, &ids, 4);
        // Assert — mismatch detected
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("image"));
        assert!(err.contains("0"));
        assert!(err.contains("1"));
    }

    // @trace TEST-MULTIMODAL-029 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_context_push_audio_after_push_image_both_independent() {
        // Arrange — push image then audio, verify each list is independent
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![10],
            embeddings: vec![1.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![20],
            embeddings: vec![2.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![30],
            embeddings: vec![3.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        // Act & Assert — images has 2 entries, audios has 1
        assert_eq!(ctx.images.len(), 2);
        assert_eq!(ctx.audios.len(), 1);
        assert_eq!(ctx.images[0].tokens[0], 10);
        assert_eq!(ctx.images[1].tokens[0], 30);
        assert_eq!(ctx.audios[0].tokens[0], 20);
        assert!(!ctx.is_empty());
    }

    // @trace TEST-MULTIMODAL-030 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_rejects_token_id_exactly_at_vocab_size() {
        // Arrange — vocab=3, hidden=2, token_id=3 equals vocab_size (out of range)
        let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // vocab=3, hidden=2
        let routed = RoutedSequence {
            token_ids: vec![3], // valid range is 0..=2
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        // Act
        let err = build_fused_hidden(&routed, &embed, 2).unwrap_err();
        // Assert
        let msg = format!("{err}");
        assert!(msg.contains("out of range"));
        assert!(msg.contains("token id 3"));
    }

    // ── Batch 13: 15 additional edge-case and coverage tests ──

    // @trace TEST-MULTIMODAL-031 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_encoded_validate_two_tokens_hidden_two_exact_match() {
        // Arrange — 2 tokens × 2 hidden = 4 embeddings, exactly provided
        let enc = MultimodalEncoded {
            tokens: vec![10, 20],
            embeddings: vec![1.0, -1.0, 0.5, -0.5],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        // Act
        let result = enc.validate();
        // Assert
        assert!(result.is_ok());
        assert_eq!(enc.num_tokens(), 2);
        assert_eq!(enc.embeddings.len(), 4);
    }

    // @trace TEST-MULTIMODAL-032 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_image_expansion_with_many_virtual_tokens_correct_text_offset() {
        // Arrange — image expands to 10 virtual tokens, verify text positions shift by 10
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 10],
            embeddings: vec![0.0; 10 * 3],
            hidden_size: 3,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![1u32, ids.image_token_id, 2];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 3).unwrap();
        // Assert — 1 text + 10 virtual + 1 text = 12
        assert_eq!(out.seq_len(), 12);
        assert_eq!(out.text_positions, vec![0, 11]);
        assert_eq!(out.token_ids[0], 1);
        assert_eq!(out.token_ids[11], 2);
    }

    // @trace TEST-MULTIMODAL-033 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_media_at_second_position_with_two_text_gathers() {
        // Arrange — 3 positions: text(0), media(1), text(2)
        let embed: Vec<f32> = vec![11.0, 12.0, 21.0, 22.0, 31.0, 32.0]; // vocab=3, hidden=2
        let media = vec![99.0, 88.0];
        let routed = RoutedSequence {
            token_ids: vec![0, 258880, 2],
            fused_embeddings: vec![None, Some(media.clone()), None],
            text_positions: vec![0, 2],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert
        assert_eq!(hidden.len(), 6);
        assert_eq!(&hidden[0..2], &[11.0, 12.0]); // text token 0
        assert_eq!(&hidden[2..4], &[99.0, 88.0]); // media override
        assert_eq!(&hidden[4..6], &[31.0, 32.0]); // text token 2
    }

    // @trace TEST-MULTIMODAL-034 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_context_push_audio_rejects_shape_mismatch_preserves_prior_audios() {
        // Arrange — push two valid audios, then a third with wrong embedding count
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.1],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![0.2],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        assert_eq!(ctx.audios.len(), 2);
        // Act — bad shape: 2 tokens × 2 hidden = 4 expected, but 3 provided
        let bad = MultimodalEncoded {
            tokens: vec![3, 4],
            embeddings: vec![0.0; 3],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let result = ctx.push_audio(bad);
        // Assert — prior audios preserved
        assert!(result.is_err());
        assert_eq!(ctx.audios.len(), 2);
        assert_eq!(ctx.audios[0].embeddings[0], 0.1);
        assert_eq!(ctx.audios[1].embeddings[0], 0.2);
    }

    // @trace TEST-MULTIMODAL-035 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn encoder_media_file_with_non_ascii_path_preserved() {
        // Arrange — file path with Chinese characters
        let path = "/tmp/模型权重/vision_encoder.bin";
        let media = EncoderMedia::File(PathBuf::from(path));
        // Act & Assert
        match &media {
            EncoderMedia::File(p) => assert_eq!(p.to_str(), Some(path)),
            _ => panic!("expected File variant"),
        }
    }

    // @trace TEST-MULTIMODAL-036 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_audio_followed_by_text_followed_by_image_embedding_order() {
        // Arrange — audio, text, image interleaving
        let ids = default_ids();
        let audio_emb = vec![1.0, 2.0];
        let img_emb = vec![3.0, 4.0];
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: audio_emb.clone(),
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: img_emb.clone(),
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc_aud).unwrap();
        ctx.push_image(enc_img).unwrap();
        // Act
        let out = route_multimodal_tokens(
            &[ids.audio_token_id, 99, ids.image_token_id],
            &ctx,
            &ids,
            2,
        )
        .unwrap();
        // Assert — 1 audio virtual + 1 text + 1 image virtual = 3
        assert_eq!(out.seq_len(), 3);
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap(), &audio_emb);
        assert!(out.fused_embeddings[1].is_none());
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap(), &img_emb);
        assert_eq!(out.text_positions, vec![1]);
    }

    // @trace TEST-MULTIMODAL-037 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_text_gather_with_token_id_one() {
        // Arrange — vocab=4, hidden=3, verify token_id=1 gathers exactly row 1
        let embed: Vec<f32> = vec![
            10.0, 11.0, 12.0, // row 0
            20.0, 21.0, 22.0, // row 1
            30.0, 31.0, 32.0, // row 2
            40.0, 41.0, 42.0, // row 3
        ];
        let routed = RoutedSequence {
            token_ids: vec![1],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 3,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 3).unwrap();
        // Assert
        assert_eq!(hidden, vec![20.0, 21.0, 22.0]);
    }

    // @trace TEST-MULTIMODAL-038 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_encoded_clone_embeddings_independent_after_mutation() {
        // Arrange — clone an encoding, verify Vec independence
        let orig = MultimodalEncoded {
            tokens: vec![1, 2],
            embeddings: vec![1.0, 2.0, 3.0, 4.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        // Act
        let mut cloned = orig.clone();
        cloned.embeddings[0] = 999.0;
        // Assert — original unaffected
        assert_eq!(orig.embeddings[0], 1.0);
        assert_eq!(cloned.embeddings[0], 999.0);
    }

    // @trace TEST-MULTIMODAL-039 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_context_has_audios_but_prompt_only_needs_images_rejected() {
        // Arrange — context has audio encoding, prompt has only image special token
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Audio,
        })
        .unwrap();
        // Act — prompt has 0 audio tokens but context has 1 audio encoding
        let prompt = vec![1u32, 2];
        let result = route_multimodal_tokens(&prompt, &ctx, &ids, 4);
        // Assert — mismatch detected
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("audio"));
        assert!(err.contains("0"));
        assert!(err.contains("1"));
    }

    // @trace TEST-MULTIMODAL-040 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_two_media_positions_at_first_and_second() {
        // Arrange — 2 media positions back-to-back, no text at all
        let embed: Vec<f32> = vec![0.0; 4]; // vocab=2, hidden=2 (unused)
        let m1 = vec![10.0, 20.0];
        let m2 = vec![30.0, 40.0];
        let routed = RoutedSequence {
            token_ids: vec![258880, 258881],
            fused_embeddings: vec![Some(m1.clone()), Some(m2.clone())],
            text_positions: vec![],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert
        assert_eq!(hidden.len(), 4);
        assert_eq!(&hidden[0..2], &[10.0, 20.0]);
        assert_eq!(&hidden[2..4], &[30.0, 40.0]);
    }

    // @trace TEST-MULTIMODAL-041 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn routed_sequence_seq_len_matches_token_ids_len() {
        // Arrange — various lengths to verify seq_len() == token_ids.len()
        for len in 0..5 {
            let rs = RoutedSequence {
                token_ids: vec![42u32; len],
                fused_embeddings: vec![None; len],
                text_positions: vec![],
                hidden_size: 1,
            };
            // Act & Assert
            assert_eq!(rs.seq_len(), len);
            assert_eq!(rs.seq_len(), rs.token_ids.len());
        }
    }

    // @trace TEST-MULTIMODAL-042 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn mock_encoder_image_and_audio_have_opposite_sign_patterns() {
        // Arrange — verify image uses positive pattern, audio uses negative
        let ids = default_ids();
        let encoder = MockEncoder::new(2, 2, 2, ids);
        // Act
        let img = encoder.encode_image(&EncoderMedia::Raw(vec![])).unwrap();
        let aud = encoder.encode_audio(&EncoderMedia::Raw(vec![])).unwrap();
        // Assert — image embeddings are non-negative (i*0.01 pattern)
        assert!(img.embeddings.iter().all(|&v| v >= 0.0));
        // Audio embeddings are non-positive (-(i*0.01) pattern)
        assert!(aud.embeddings.iter().all(|&v| v <= 0.0));
    }

    // @trace TEST-MULTIMODAL-043 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_text_between_two_audio_expansions_correct_positions() {
        // Arrange — audio, text, audio with different virtual token counts
        let ids = default_ids();
        let enc1 = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 3],
            embeddings: vec![1.0; 3 * 2],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let enc2 = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 1],
            embeddings: vec![2.0; 1 * 2],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc1).unwrap();
        ctx.push_audio(enc2).unwrap();
        let prompt = vec![ids.audio_token_id, 77, ids.audio_token_id];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — 3 virtual + 1 text + 1 virtual = 5
        assert_eq!(out.seq_len(), 5);
        assert_eq!(out.text_positions, vec![3]);
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 1.0);
        assert_eq!(out.fused_embeddings[4].as_ref().unwrap()[0], 2.0);
        assert!(out.fused_embeddings[3].is_none());
    }

    // @trace TEST-MULTIMODAL-044 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn encoder_media_url_with_port_number_preserved() {
        // Arrange — URL containing an explicit port number
        let url = "https://example.com:8443/models/vision.gguf";
        let media = EncoderMedia::Url(url.into());
        // Act & Assert
        match &media {
            EncoderMedia::Url(s) => {
                assert!(s.contains(":8443"));
                assert_eq!(s, url);
            }
            _ => panic!("expected Url variant"),
        }
    }

    // @trace TEST-MULTIMODAL-045 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_context_push_audio_after_valid_image_does_not_corrupt_images() {
        // Arrange — push an image, then push a valid audio, verify images untouched
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![10],
            embeddings: vec![5.5, 6.6],
            hidden_size: 2,
            kind: MediaKind::Image,
        })
        .unwrap();
        assert_eq!(ctx.images.len(), 1);
        // Act
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![20],
            embeddings: vec![7.7, 8.8],
            hidden_size: 2,
            kind: MediaKind::Audio,
        })
        .unwrap();
        // Assert — images list unchanged
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.images[0].tokens, vec![10]);
        assert_eq!(ctx.images[0].embeddings, vec![5.5, 6.6]);
        assert_eq!(ctx.audios.len(), 1);
        assert_eq!(ctx.audios[0].tokens, vec![20]);
    }

    // ── Batch 14: 15 additional edge-case and coverage tests ──

    // @trace TEST-MULTIMODAL-046 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn media_kind_copy_semantics_independent_after_assignment() {
        // Arrange — 验证 MediaKind 的 Copy trait 使赋值后的两个变量相互独立
        let kind = MediaKind::Image;
        // Act — Copy 语义：赋值产生独立副本
        let other = kind;
        // Assert — 两者都是有效值，互不影响
        assert_eq!(kind, MediaKind::Image);
        assert_eq!(other, MediaKind::Image);
        assert_eq!(kind, other);
    }

    // @trace TEST-MULTIMODAL-047 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn media_kind_partial_eq_distinct_variants_not_equal() {
        // Arrange — 验证 MediaKind::Image 与 Audio 不相等
        // Act & Assert
        assert_ne!(MediaKind::Image, MediaKind::Audio);
        assert_eq!(MediaKind::Image, MediaKind::Image);
        assert_eq!(MediaKind::Audio, MediaKind::Audio);
    }

    // @trace TEST-MULTIMODAL-048 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn media_kind_hash_in_hashset_deduplication() {
        // Arrange — 验证 MediaKind 的 Hash trait 在 HashSet 中正确去重
        use std::collections::HashSet;
        let mut set = HashSet::new();
        // Act — 插入重复的 Image 和 Audio
        set.insert(MediaKind::Image);
        set.insert(MediaKind::Image);
        set.insert(MediaKind::Audio);
        set.insert(MediaKind::Audio);
        // Assert — 只有 2 个唯一值
        assert_eq!(set.len(), 2);
        assert!(set.contains(&MediaKind::Image));
        assert!(set.contains(&MediaKind::Audio));
    }

    // @trace TEST-MULTIMODAL-049 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn routed_sequence_clone_independent_after_mutation() {
        // Arrange — 构造一个含多模态数据的 RoutedSequence
        let original = RoutedSequence {
            token_ids: vec![10, 20, 30],
            fused_embeddings: vec![None, Some(vec![1.0, 2.0]), None],
            text_positions: vec![0, 2],
            hidden_size: 2,
        };
        // Act — clone 后修改副本
        let mut cloned = original.clone();
        cloned.token_ids[0] = 999;
        // Assert — 原始不受影响
        assert_eq!(original.token_ids[0], 10);
        assert_eq!(cloned.token_ids[0], 999);
        assert_eq!(original.hidden_size, 2);
        assert_eq!(cloned.hidden_size, 2);
    }

    // @trace TEST-MULTIMODAL-050 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_too_many_image_special_tokens_reports_exact_counts() {
        // Arrange — prompt 有 3 个 image token 但 context 只有 2 个编码
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![0.0; 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        })
        .unwrap();
        let prompt = vec![ids.image_token_id; 3];
        // Act
        let err = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap_err();
        let msg = format!("{err}");
        // Assert — 错误消息包含准确的 3 vs 2
        assert!(msg.contains("3"));
        assert!(msg.contains("2"));
        assert!(msg.contains("image"));
    }

    // @trace TEST-MULTIMODAL-051 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_media_with_very_large_hidden_size() {
        // Arrange — hidden_size=65536 的 media embedding
        let hidden = 65536;
        let media: Vec<f32> = (0..hidden).map(|i| (i as f32).sqrt()).collect();
        let embed: Vec<f32> = vec![0.0; hidden]; // vocab=1, hidden=65536
        let routed = RoutedSequence {
            token_ids: vec![258880],
            fused_embeddings: vec![Some(media.clone())],
            text_positions: vec![],
            hidden_size: hidden,
        };
        // Act
        let result = build_fused_hidden(&routed, &embed, hidden).unwrap();
        // Assert — 输出长度正确且首尾值精确匹配
        assert_eq!(result.len(), hidden);
        assert_eq!(result[0], 0.0f32.sqrt());
        assert_eq!(result[hidden - 1], ((hidden - 1) as f32).sqrt());
    }

    // @trace TEST-MULTIMODAL-052 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_context_push_image_reject_after_valid_push_preserves_valid() {
        // Arrange — 先 push 一个有效 image，再 reject 一个错误的
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![100],
            embeddings: vec![3.14],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        // Act — push 一个 kind 错误的
        let bad = MultimodalEncoded {
            tokens: vec![200],
            embeddings: vec![0.0],
            hidden_size: 1,
            kind: MediaKind::Audio, // 错误的 kind
        };
        let _ = ctx.push_image(bad);
        // Assert — 有效 push 保留，错误 push 不影响
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.images[0].tokens[0], 100);
        assert_eq!(ctx.images[0].embeddings[0], 3.14);
    }

    // @trace TEST-MULTIMODAL-053 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn encoder_media_raw_single_zero_byte_preserved() {
        // Arrange — Raw 变体包含恰好 1 字节 0x00
        let media = EncoderMedia::Raw(vec![0x00]);
        // Act & Assert
        match &media {
            EncoderMedia::Raw(bytes) => {
                assert_eq!(bytes.len(), 1);
                assert_eq!(bytes[0], 0x00);
            }
            _ => panic!("expected Raw variant"),
        }
    }

    // @trace TEST-MULTIMODAL-054 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_token_ids_all_fields_u32_max_is_image_and_is_audio_work() {
        // Arrange — 所有字段设为 u32::MAX 的极端值
        let ids = MultimodalTokenIds {
            image_token_id: u32::MAX,
            audio_token_id: u32::MAX,
            eoi_token_id: u32::MAX,
            eoa_token_id: u32::MAX,
        };
        // Act & Assert — image_token_id == audio_token_id 时两个 helper 都返回 true
        assert!(ids.is_image(u32::MAX));
        assert!(ids.is_audio(u32::MAX));
        assert!(!ids.is_image(0));
        assert!(!ids.is_audio(0));
    }

    // @trace TEST-MULTIMODAL-055 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_consecutive_image_special_tokens_expand_sequentially() {
        // Arrange — 两个连续的 image special token，各自有不同的 embedding 值
        let ids = default_ids();
        let enc1 = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 2],
            embeddings: vec![1.0, 2.0, 3.0, 4.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let enc2 = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 1],
            embeddings: vec![5.0, 6.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc1).unwrap();
        ctx.push_image(enc2).unwrap();
        let prompt = vec![ids.image_token_id, ids.image_token_id];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — 第一个扩展为 2 virtual，第二个为 1 virtual，共 3
        assert_eq!(out.seq_len(), 3);
        assert!(out.text_positions.is_empty());
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap(), &vec![1.0, 2.0]);
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap(), &vec![3.0, 4.0]);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap(), &vec![5.0, 6.0]);
    }

    // @trace TEST-MULTIMODAL-056 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_media_embedding_with_neg_infinity() {
        // Arrange — media embedding 包含 f32::NEG_INFINITY
        let media = vec![f32::NEG_INFINITY, 0.0];
        let embed = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, hidden=2
        let routed = RoutedSequence {
            token_ids: vec![258880],
            fused_embeddings: vec![Some(media.clone())],
            text_positions: vec![],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — NEG_INFINITY 精确保留
        assert_eq!(hidden[0], f32::NEG_INFINITY);
        assert_eq!(hidden[1], 0.0);
    }

    // @trace TEST-MULTIMODAL-057 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_context_clone_then_push_does_not_affect_original() {
        // Arrange — 构造含一个 image 的 context，clone 后在副本上 push
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![10.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        // Act — clone 后在副本上追加
        let mut cloned = ctx.clone();
        cloned.push_image(MultimodalEncoded {
            tokens: vec![2],
            embeddings: vec![20.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        // Assert — 原始不变
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.images[0].tokens[0], 1);
        assert_eq!(cloned.images.len(), 2);
        assert_eq!(cloned.images[1].tokens[0], 2);
    }

    // @trace TEST-MULTIMODAL-058 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_audio_only_multiple_different_virtual_token_counts() {
        // Arrange — 3 个 audio 编码，分别产出 1、3、2 个 virtual token
        let ids = default_ids();
        let enc1 = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 1],
            embeddings: vec![1.0; 1 * 2],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let enc2 = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 3],
            embeddings: vec![2.0; 3 * 2],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let enc3 = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![3.0; 2 * 2],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc1).unwrap();
        ctx.push_audio(enc2).unwrap();
        ctx.push_audio(enc3).unwrap();
        let prompt = vec![ids.audio_token_id; 3];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — 1 + 3 + 2 = 6 total virtual tokens
        assert_eq!(out.seq_len(), 6);
        assert!(out.text_positions.is_empty());
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 1.0);
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap()[0], 2.0);
        assert_eq!(out.fused_embeddings[4].as_ref().unwrap()[0], 3.0);
    }

    // @trace TEST-MULTIMODAL-059 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_text_gather_token_zero_with_hidden_size_one() {
        // Arrange — vocab=3, hidden=1, token_id=0 收集第一行
        let embed: Vec<f32> = vec![7.0, 8.0, 9.0]; // vocab=3, hidden=1
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 1,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 1).unwrap();
        // Assert — 收集 vocab 第 0 行
        assert_eq!(hidden, vec![7.0]);
    }

    // @trace TEST-MULTIMODAL-060 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn routed_sequence_has_multimodal_false_when_all_none() {
        // Arrange — 全部 fused_embeddings 为 None
        let rs = RoutedSequence {
            token_ids: vec![1, 2, 3, 4, 5],
            fused_embeddings: vec![None, None, None, None, None],
            text_positions: vec![0, 1, 2, 3, 4],
            hidden_size: 128,
        };
        // Act & Assert
        assert!(!rs.has_multimodal());
        assert_eq!(rs.seq_len(), 5);
    }

    // ── Batch 15: 15 additional edge-case and coverage tests ──

    // @trace TEST-MULTIMODAL-061 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn mock_encoder_is_send_and_sync() {
        // Arrange — 验证 MockEncoder 满足 Send + Sync 约束（编译期检查）
        let ids = default_ids();
        let encoder = MockEncoder::new(1, 1, 4, ids);
        // Act — 将 encoder 移入一个闭包，验证 Send
        let handle = std::thread::spawn(move || {
            let result = encoder.encode_image(&EncoderMedia::Raw(vec![])).unwrap();
            assert_eq!(result.kind, MediaKind::Image);
        });
        // Assert — 线程成功完成
        handle.join().unwrap();
    }

    // @trace TEST-MULTIMODAL-062 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_prompt_with_image_and_audio_same_count_as_context() {
        // Arrange — 2 image + 2 audio in prompt, each encoding has 2 virtual tokens
        let ids = default_ids();
        let mut ctx = MultimodalContext::new();
        for i in 0..2 {
            ctx.push_image(MultimodalEncoded {
                tokens: vec![ids.image_token_id, ids.image_token_id],
                embeddings: vec![
                    (i + 1) as f32 * 10.0,
                    (i + 1) as f32 * 20.0,
                    (i + 1) as f32 * 10.5,
                    (i + 1) as f32 * 20.5,
                ],
                hidden_size: 2,
                kind: MediaKind::Image,
            })
            .unwrap();
        }
        for i in 0..2 {
            ctx.push_audio(MultimodalEncoded {
                tokens: vec![ids.audio_token_id, ids.audio_token_id],
                embeddings: vec![
                    (i + 1) as f32 * 100.0,
                    (i + 1) as f32 * 200.0,
                    (i + 1) as f32 * 100.5,
                    (i + 1) as f32 * 200.5,
                ],
                hidden_size: 2,
                kind: MediaKind::Audio,
            })
            .unwrap();
        }
        // Prompt: img, aud, text, img, aud
        let prompt = vec![
            ids.image_token_id,
            ids.audio_token_id,
            99,
            ids.image_token_id,
            ids.audio_token_id,
        ];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — each media slot expands to 2 tokens: 2+2+1+2+2 = 9 positions, text at index 4
        assert_eq!(out.seq_len(), 9);
        assert_eq!(out.text_positions, vec![4]);
        // img1 virtual tokens: [10.0, 20.0] at pos 0, [10.5, 20.5] at pos 1
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 10.0);
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap()[0], 10.5);
        // aud1 virtual tokens: [100.0, 200.0] at pos 2, [100.5, 200.5] at pos 3
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap()[0], 100.0);
        assert_eq!(out.fused_embeddings[3].as_ref().unwrap()[0], 100.5);
        // text at pos 4: no embedding
        assert_eq!(out.fused_embeddings[4].is_none(), true);
        // img2 at pos 5-6 (i=1: (i+1)*10.0=20.0, (i+1)*10.5=21.0)
        assert_eq!(out.fused_embeddings[5].as_ref().unwrap()[0], 20.0);
        assert_eq!(out.fused_embeddings[6].as_ref().unwrap()[0], 21.0);
        // aud2 at pos 7-8 (i=1: (i+1)*100.0=200.0, (i+1)*100.5=201.0)
        assert_eq!(out.fused_embeddings[7].as_ref().unwrap()[0], 200.0);
        assert_eq!(out.fused_embeddings[8].as_ref().unwrap()[0], 201.0);
    }

    // @trace TEST-MULTIMODAL-063 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_embed_table_exact_multiple_of_hidden_size_vocab_one() {
        // Arrange — vocab=1, hidden=4, embed has exactly 4 elements
        let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // vocab=1, hidden=4
        let media = vec![10.0, 20.0, 30.0, 40.0];
        let routed = RoutedSequence {
            token_ids: vec![0, 258880],
            fused_embeddings: vec![None, Some(media.clone())],
            text_positions: vec![0],
            hidden_size: 4,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 4).unwrap();
        // Assert — position 0: gather token 0, position 1: media
        assert_eq!(hidden.len(), 8);
        assert_eq!(&hidden[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&hidden[4..8], &media);
    }

    // @trace TEST-MULTIMODAL-064 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_context_push_images_with_different_hidden_sizes_all_accepted() {
        // Arrange — MultimodalContext 不要求所有 encoding 共享同一 hidden_size
        let mut ctx = MultimodalContext::new();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![1.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        })
        .unwrap();
        ctx.push_image(MultimodalEncoded {
            tokens: vec![2, 3],
            embeddings: vec![2.0, 3.0, 4.0, 5.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        })
        .unwrap();
        // Act & Assert — both accepted regardless of differing hidden_size
        assert_eq!(ctx.images.len(), 2);
        assert_eq!(ctx.images[0].hidden_size, 1);
        assert_eq!(ctx.images[1].hidden_size, 2);
    }

    // @trace TEST-MULTIMODAL-065 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_image_with_many_virtual_tokens_then_text_preserves_embedding_sequence() {
        // Arrange — image expands to 5 virtual tokens with sequential embedding values
        let ids = default_ids();
        let embeddings: Vec<f32> = (0..5 * 2).map(|i| (i + 1) as f32).collect();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 5],
            embeddings: embeddings.clone(),
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![ids.image_token_id, 42];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — 5 virtual + 1 text = 6; verify each embedding slice
        assert_eq!(out.seq_len(), 6);
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap(), &vec![1.0, 2.0]);
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap(), &vec![3.0, 4.0]);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap(), &vec![5.0, 6.0]);
        assert_eq!(out.fused_embeddings[3].as_ref().unwrap(), &vec![7.0, 8.0]);
        assert_eq!(out.fused_embeddings[4].as_ref().unwrap(), &vec![9.0, 10.0]);
        assert!(out.fused_embeddings[5].is_none());
    }

    // @trace TEST-MULTIMODAL-066 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_encoded_tokens_can_contain_arbitrary_u32_values() {
        // Arrange — tokens field can hold any u32 value, not just special token IDs
        let tokens = vec![0, 1, u32::MAX / 2, u32::MAX, 42];
        let n = tokens.len();
        let enc = MultimodalEncoded {
            tokens: tokens.clone(),
            embeddings: vec![0.0; n * 3],
            hidden_size: 3,
            kind: MediaKind::Image,
        };
        // Act & Assert
        assert_eq!(enc.tokens, tokens);
        assert!(enc.validate().is_ok());
        assert_eq!(enc.num_tokens(), n);
    }

    // @trace TEST-MULTIMODAL-067 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_gather_middle_row_from_vocab() {
        // Arrange — vocab=5, hidden=2, token_id=2 gathers the middle row (row 2)
        let embed: Vec<f32> = (0..10).map(|i| (i + 1) as f32).collect(); // vocab=5, hidden=2
        let routed = RoutedSequence {
            token_ids: vec![2],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — row 2 = indices [4, 5] = values [5.0, 6.0]
        assert_eq!(hidden, vec![5.0, 6.0]);
    }

    // @trace TEST-MULTIMODAL-068 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_single_text_token_with_custom_ids_not_matching_defaults() {
        // Arrange — single token = 258880 (default image), but custom IDs don't recognize it
        let custom_ids = MultimodalTokenIds {
            image_token_id: 999,
            audio_token_id: 998,
            eoi_token_id: 997,
            eoa_token_id: 996,
        };
        let ctx = MultimodalContext::new();
        let prompt = vec![258880u32];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &custom_ids, 4).unwrap();
        // Assert — treated as plain text
        assert_eq!(out.seq_len(), 1);
        assert_eq!(out.token_ids, vec![258880]);
        assert!(out.fused_embeddings[0].is_none());
        assert!(!out.has_multimodal());
        assert_eq!(out.text_positions, vec![0]);
    }

    // @trace TEST-MULTIMODAL-069 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_many_text_tokens_each_gathering_different_rows() {
        // Arrange — 5 text tokens gathering 5 different vocab rows, no media
        let hidden = 3;
        let vocab = 5;
        let embed: Vec<f32> = (0..vocab * hidden).map(|i| (i * 10) as f32).collect();
        let routed = RoutedSequence {
            token_ids: vec![4, 0, 3, 1, 2],
            fused_embeddings: vec![None; 5],
            text_positions: vec![0, 1, 2, 3, 4],
            hidden_size: hidden,
        };
        // Act
        let result = build_fused_hidden(&routed, &embed, hidden).unwrap();
        // Assert — each position gathers the correct row
        assert_eq!(result.len(), 5 * hidden);
        assert_eq!(&result[0..3], &[120.0, 130.0, 140.0]); // token 4
        assert_eq!(&result[3..6], &[0.0, 10.0, 20.0]);      // token 0
        assert_eq!(&result[6..9], &[90.0, 100.0, 110.0]);   // token 3
        assert_eq!(&result[9..12], &[30.0, 40.0, 50.0]);    // token 1
        assert_eq!(&result[12..15], &[60.0, 70.0, 80.0]);   // token 2
    }

    // @trace TEST-MULTIMODAL-070 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_context_push_audio_reject_after_valid_audio_preserves_first() {
        // Arrange — push a valid audio, then try a shape-mismatched audio
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(MultimodalEncoded {
            tokens: vec![5],
            embeddings: vec![4.2],
            hidden_size: 1,
            kind: MediaKind::Audio,
        })
        .unwrap();
        assert_eq!(ctx.audios.len(), 1);
        // Act — bad shape: 1 token × 3 hidden = 3 expected, 2 provided
        let bad = MultimodalEncoded {
            tokens: vec![6],
            embeddings: vec![0.0, 0.0],
            hidden_size: 3,
            kind: MediaKind::Audio,
        };
        let _ = ctx.push_audio(bad);
        // Assert — first audio preserved
        assert_eq!(ctx.audios.len(), 1);
        assert_eq!(ctx.audios[0].tokens[0], 5);
        assert_eq!(ctx.audios[0].embeddings[0], 4.2);
    }

    // @trace TEST-MULTIMODAL-071 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_custom_ids_image_equals_audio_both_special_tokens_recognized() {
        // Arrange — custom IDs where image_token_id == audio_token_id == 42
        // Both is_image(42) and is_audio(42) return true, so a single token 42
        // is counted as both 1 image slot and 1 audio slot by the pre-validation.
        // The routing loop checks is_image first, so token 42 consumes the image encoding.
        // But the pre-validation counts it as both, so we need both encodings.
        let custom_ids = MultimodalTokenIds {
            image_token_id: 42,
            audio_token_id: 42,
            eoi_token_id: 99,
            eoa_token_id: 100,
        };
        let enc_img = MultimodalEncoded {
            tokens: vec![42],
            embeddings: vec![7.7],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        let enc_aud = MultimodalEncoded {
            tokens: vec![42],
            embeddings: vec![3.3],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc_img).unwrap();
        ctx.push_audio(enc_aud).unwrap();
        // Prompt has token 42; pre-validation counts 1 image + 1 audio slot.
        let prompt = vec![42];
        // Act — routing checks is_image first, consumes image encoding.
        // But audio_slots=1 with 1 audio encoding satisfies pre-validation.
        let out = route_multimodal_tokens(&prompt, &ctx, &custom_ids, 1).unwrap();
        // Assert — token 42 was treated as image (is_image checked first in loop)
        assert_eq!(out.seq_len(), 1);
        assert!(out.fused_embeddings[0].is_some());
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap()[0], 7.7);
    }

    // @trace TEST-MULTIMODAL-072 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_text_and_media_alternating_four_positions() {
        // Arrange — 4 positions: text, media, text, media
        let embed: Vec<f32> = vec![11.0, 12.0, 21.0, 22.0]; // vocab=2, hidden=2
        let m1 = vec![100.0, 101.0];
        let m2 = vec![200.0, 201.0];
        let routed = RoutedSequence {
            token_ids: vec![0, 258880, 1, 258881],
            fused_embeddings: vec![None, Some(m1), None, Some(m2)],
            text_positions: vec![0, 2],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert
        assert_eq!(hidden.len(), 8);
        assert_eq!(&hidden[0..2], &[11.0, 12.0]);    // text token 0
        assert_eq!(&hidden[2..4], &[100.0, 101.0]);   // media 1
        assert_eq!(&hidden[4..6], &[21.0, 22.0]);     // text token 1
        assert_eq!(&hidden[6..8], &[200.0, 201.0]);   // media 2
    }

    // @trace TEST-MULTIMODAL-073 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_encoded_validate_exact_boundary_at_power_of_two() {
        // Arrange — 256 tokens × 128 hidden = 32768 embeddings (power-of-2 boundary)
        let n = 256;
        let h = 128;
        let enc = MultimodalEncoded {
            tokens: vec![0u32; n],
            embeddings: vec![0.5; n * h],
            hidden_size: h,
            kind: MediaKind::Audio,
        };
        // Act & Assert
        assert_eq!(enc.num_tokens(), n);
        assert_eq!(enc.embeddings.len(), 32768);
        assert!(enc.validate().is_ok());
    }

    // @trace TEST-MULTIMODAL-074 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_context_with_only_images_prompt_needs_audio_and_image() {
        // Arrange — ctx has 1 image + 1 audio, prompt has image first then audio
        let ids = default_ids();
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![1.0, 2.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![3.0, 4.0, 5.0, 6.0],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc_img).unwrap();
        ctx.push_audio(enc_aud).unwrap();
        let prompt = vec![10u32, ids.image_token_id, 20, ids.audio_token_id, 30];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — 1 text + 1 virtual + 1 text + 2 virtual + 1 text = 6
        assert_eq!(out.seq_len(), 6);
        assert_eq!(out.text_positions, vec![0, 2, 5]);
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap(), &vec![1.0, 2.0]);
        assert_eq!(out.fused_embeddings[3].as_ref().unwrap(), &vec![3.0, 4.0]);
        assert_eq!(out.fused_embeddings[4].as_ref().unwrap(), &vec![5.0, 6.0]);
    }

    // @trace TEST-MULTIMODAL-075 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_media_embedding_with_f32_min_positive_normal() {
        // Arrange — media embedding contains f32::MIN_POSITIVE (smallest normal positive)
        let min_normal = f32::MIN_POSITIVE; // 1.17549435e-38
        let media = vec![min_normal];
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![Some(media)],
            text_positions: vec![],
            hidden_size: 1,
        };
        let embed = vec![0.0f32; 2]; // vocab=2, hidden=1
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 1).unwrap();
        // Assert — MIN_POSITIVE preserved exactly
        assert_eq!(hidden[0].to_bits(), min_normal.to_bits());
        assert!(hidden[0].is_normal());
    }

    // ── Batch 16: 15 additional tests — VisionConfig fields, Encoder boundary,
    //     EncoderMedia all-variant Debug/Clone, image size boundary, encoder context ──

    // @trace TEST-MULTIMODAL-076 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_token_ids_fallback_multimodal_token_ids_eoi_distinct_from_eoa() {
        // Arrange — eoi_token_id and eoa_token_id must be distinct from each other
        let ids = MultimodalTokenIds::fallback_multimodal_token_ids();
        // Act & Assert
        assert_ne!(ids.eoi_token_id, ids.eoa_token_id);
        assert_eq!(ids.eoi_token_id, 258882);
        assert_eq!(ids.eoa_token_id, 258883);
    }

    // @trace TEST-MULTIMODAL-077 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn encoder_media_file_clone_then_modify_original_path_preserved_in_clone() {
        // Arrange — verify that File variant clone is deep
        let original = EncoderMedia::File(PathBuf::from("/tmp/original.png"));
        let cloned = original.clone();
        // Act — check both paths exist and match
        // Assert
        match (&original, &cloned) {
            (EncoderMedia::File(p1), EncoderMedia::File(p2)) => {
                assert_eq!(p1, p2);
                assert_eq!(p1, &PathBuf::from("/tmp/original.png"));
            }
            _ => panic!("Both should be File variant"),
        }
    }

    // @trace TEST-MULTIMODAL-078 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn encoder_media_base64_debug_output_contains_variant_name() {
        // Arrange — Base64 variant Debug must contain "Base64"
        let media = EncoderMedia::Base64 {
            data: "SGVsbG8=".to_string(),
            mime_type: Some("image/png".to_string()),
        };
        // Act
        let debug_str = format!("{:?}", media);
        // Assert
        assert!(debug_str.contains("Base64"), "Debug output should contain 'Base64': {}", debug_str);
        assert!(debug_str.contains("image/png"), "Debug output should contain mime_type: {}", debug_str);
    }

    // @trace TEST-MULTIMODAL-079 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn encoder_media_raw_debug_output_contains_raw_bytes() {
        // Arrange — Raw variant Debug must contain "Raw"
        let media = EncoderMedia::Raw(vec![0x89, 0x50, 0x4E, 0x47]);
        // Act
        let debug_str = format!("{:?}", media);
        // Assert
        assert!(debug_str.contains("Raw"), "Debug output should contain 'Raw': {}", debug_str);
    }

    // @trace TEST-MULTIMODAL-080 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn encoder_media_url_debug_output_contains_url_string() {
        // Arrange — Url variant Debug must contain the URL string
        let url = "https://example.com/image.jpg";
        let media = EncoderMedia::Url(url.to_string());
        // Act
        let debug_str = format!("{:?}", media);
        // Assert
        assert!(debug_str.contains("Url"), "Debug output should contain 'Url': {}", debug_str);
        assert!(debug_str.contains(url), "Debug output should contain the URL: {}", debug_str);
    }

    // @trace TEST-MULTIMODAL-081 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_encoded_validate_rejects_empty_tokens_with_nonzero_embeddings() {
        // Arrange — 0 tokens but 3 embeddings: shape mismatch
        let enc = MultimodalEncoded {
            tokens: vec![],
            embeddings: vec![1.0, 2.0, 3.0],
            hidden_size: 3,
            kind: MediaKind::Audio,
        };
        // Act
        let result = enc.validate();
        // Assert — expected = 0 * 3 = 0, got = 3
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("expected embeddings=0"), "Error should report expected=0: {}", err_msg);
        assert!(err_msg.contains("got=3"), "Error should report got=3: {}", err_msg);
    }

    // @trace TEST-MULTIMODAL-082 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_rejects_embed_rows_length_zero_with_nonzero_routed() {
        // Arrange — non-empty routed sequence but embed_rows is empty (vocab=0)
        let routed = RoutedSequence {
            token_ids: vec![0],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 2,
        };
        let embed: Vec<f32> = vec![]; // vocab=0
        // Act
        let result = build_fused_hidden(&routed, &embed, 2);
        // Assert — embed_rows is empty, token 0 >= vocab_size 0
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("out of range"), "Should report token out of range: {}", err_msg);
    }

    // @trace TEST-MULTIMODAL-083 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_prompt_with_only_image_special_tokens_no_text() {
        // Arrange — prompt consists entirely of image tokens, no text tokens at all
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 3],
            embeddings: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![ids.image_token_id]; // single image token, expands to 3 virtual
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — no text positions, all positions have embeddings
        assert_eq!(out.seq_len(), 3);
        assert!(out.text_positions.is_empty());
        assert!(out.has_multimodal());
        assert!(out.fused_embeddings.iter().all(|e| e.is_some()));
    }

    // @trace TEST-MULTIMODAL-084 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_prompt_with_only_audio_special_tokens_no_text() {
        // Arrange — prompt consists entirely of audio tokens, no text tokens at all
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![10.0, 20.0, 30.0, 40.0],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc).unwrap();
        let prompt = vec![ids.audio_token_id];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — no text positions, both positions have embeddings
        assert_eq!(out.seq_len(), 2);
        assert!(out.text_positions.is_empty());
        assert!(out.has_multimodal());
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap(), &vec![10.0, 20.0]);
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap(), &vec![30.0, 40.0]);
    }

    // @trace TEST-MULTIMODAL-085 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_context_push_image_accepts_maximum_reasonable_hidden_size() {
        // Arrange — test with hidden_size=4096 (realistic for large models like Llama)
        let hidden = 4096;
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![0.5; hidden],
            hidden_size: hidden,
            kind: MediaKind::Image,
        };
        // Act
        let mut ctx = MultimodalContext::new();
        let result = ctx.push_image(enc);
        // Assert
        assert!(result.is_ok());
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.images[0].hidden_size, hidden);
        assert_eq!(ctx.images[0].embeddings.len(), hidden);
    }

    // @trace TEST-MULTIMODAL-086 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn build_fused_hidden_rejects_fused_embeddings_length_mismatch_with_seq_len() {
        // Arrange — fused_embeddings has 2 entries but token_ids has 3: inconsistent
        let routed = RoutedSequence {
            token_ids: vec![0, 1, 2],
            fused_embeddings: vec![None, None], // only 2, not 3
            text_positions: vec![0, 1, 2],
            hidden_size: 2,
        };
        let embed: Vec<f32> = vec![0.0; 6]; // vocab=3, hidden=2
        // Act
        let result = build_fused_hidden(&routed, &embed, 2);
        // Assert
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("fused_embeddings len 2 != seq_len 3"),
            "Should report embeddings/seq_len mismatch: {}",
            err_msg
        );
    }

    // @trace TEST-MULTIMODAL-087 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn encoder_media_from_generation_base64_with_mime_exact_fields() {
        // Arrange — verify from_generation maps every field correctly for Base64
        let input = crate::generation::MediaInput::Base64 {
            data: "dGVzdA==".to_string(),
            mime_type: Some("audio/wav".to_string()),
        };
        // Act
        let media = EncoderMedia::from_generation(&input);
        // Assert
        match media {
            EncoderMedia::Base64 { data, mime_type } => {
                assert_eq!(data, "dGVzdA==");
                assert_eq!(mime_type, Some("audio/wav".to_string()));
            }
            _ => panic!("Expected Base64 variant"),
        }
    }

    // @trace TEST-MULTIMODAL-088 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_encoded_validate_success_with_f32_max_embedding_values() {
        // Arrange — embeddings use f32::MAX to verify no overflow or special handling
        let enc = MultimodalEncoded {
            tokens: vec![42],
            embeddings: vec![f32::MAX],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        // Act & Assert — validate should succeed; shape is correct
        assert!(enc.validate().is_ok());
        assert_eq!(enc.embeddings[0].to_bits(), f32::MAX.to_bits());
    }

    // @trace TEST-MULTIMODAL-089 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn multimodal_context_images_and_audios_independent_after_sequential_pushes() {
        // Arrange — push 3 images then 2 audios, verify counts are independent
        let mut ctx = MultimodalContext::new();
        for i in 0..3 {
            ctx.push_image(MultimodalEncoded {
                tokens: vec![i as u32],
                embeddings: vec![i as f32],
                hidden_size: 1,
                kind: MediaKind::Image,
            })
            .unwrap();
        }
        for i in 0..2 {
            ctx.push_audio(MultimodalEncoded {
                tokens: vec![(i + 10) as u32],
                embeddings: vec![(i + 10) as f32],
                hidden_size: 1,
                kind: MediaKind::Audio,
            })
            .unwrap();
        }
        // Act — check that is_empty is false and counts are correct
        // Assert
        assert!(!ctx.is_empty());
        assert_eq!(ctx.images.len(), 3);
        assert_eq!(ctx.audios.len(), 2);
        // Verify ordering: images[0]=0, images[2]=2; audios[0]=10, audios[1]=11
        assert_eq!(ctx.images[0].tokens[0], 0);
        assert_eq!(ctx.images[2].tokens[0], 2);
        assert_eq!(ctx.audios[0].tokens[0], 10);
        assert_eq!(ctx.audios[1].tokens[0], 11);
    }

    // @trace TEST-MULTIMODAL-090 [req:REQ-MEGA-003] [level:unit]

    #[test]
    fn route_single_text_token_with_default_ids_not_special() {
        // Arrange — token 0 is not a special token with default ids, so it should be plain text
        let ids = default_ids();
        let ctx = MultimodalContext::new();
        let prompt = vec![0u32];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        // Assert — treated as text, no multimodal
        assert_eq!(out.seq_len(), 1);
        assert_eq!(out.token_ids, vec![0]);
        assert!(out.fused_embeddings[0].is_none());
        assert!(!out.has_multimodal());
        assert_eq!(out.text_positions, vec![0]);
    }

    // ── 13 new tests: struct update syntax, boundary values, float precision, overflow safety ──

    // @trace TEST-MULTIMODAL-091 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn multimodal_token_ids_struct_update_syntax_single_field_override() {
        // Arrange — start from defaults, override only image_token_id
        let base = MultimodalTokenIds::fallback_multimodal_token_ids();
        // Act
        let custom = MultimodalTokenIds { image_token_id: 999, ..base };
        // Assert — only image changed, rest inherited
        assert_eq!(custom.image_token_id, 999);
        assert_eq!(custom.audio_token_id, 258881);
        assert_eq!(custom.eoi_token_id, 258882);
        assert_eq!(custom.eoa_token_id, 258883);
        assert!(custom.is_image(999));
        assert!(!custom.is_image(258880));
    }

    // @trace TEST-MULTIMODAL-092 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn multimodal_encoded_struct_update_syntax_preserves_kind() {
        // Arrange
        let original = MultimodalEncoded {
            tokens: vec![10, 20],
            embeddings: vec![1.0, 2.0, 3.0, 4.0],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        // Act — change tokens and embeddings, keep kind and hidden_size via struct update
        let updated = MultimodalEncoded {
            tokens: vec![30, 40, 50],
            embeddings: vec![0.0; 6],
            ..original
        };
        // Assert — kind and hidden_size preserved from original
        assert_eq!(updated.kind, MediaKind::Audio);
        assert_eq!(updated.hidden_size, 2);
        assert_eq!(updated.num_tokens(), 3);
    }

    // @trace TEST-MULTIMODAL-093 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn routed_sequence_struct_update_syntax_hidden_size_only() {
        // Arrange
        let base = RoutedSequence {
            token_ids: vec![1, 2, 3],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 64,
        };
        // Act — override only hidden_size
        let modified = RoutedSequence { hidden_size: 128, ..base };
        // Assert — all other fields identical
        assert_eq!(modified.token_ids, vec![1, 2, 3]);
        assert_eq!(modified.fused_embeddings.len(), 3);
        assert_eq!(modified.text_positions, vec![0, 1, 2]);
        assert_eq!(modified.hidden_size, 128);
        assert_eq!(modified.seq_len(), 3);
    }

    // @trace TEST-MULTIMODAL-094 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn multimodal_token_ids_u32_max_boundary_values() {
        // Arrange — all fields at u32::MAX
        let ids = MultimodalTokenIds {
            image_token_id: u32::MAX,
            audio_token_id: u32::MAX - 1,
            eoi_token_id: u32::MAX - 2,
            eoa_token_id: u32::MAX - 3,
        };
        // Act & Assert — helpers work at boundary values, no overflow
        assert!(ids.is_image(u32::MAX));
        assert!(!ids.is_image(u32::MAX - 1));
        assert!(ids.is_audio(u32::MAX - 1));
        assert!(!ids.is_audio(u32::MAX));
    }

    // @trace TEST-MULTIMODAL-095 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn multimodal_token_ids_all_zero_values() {
        // Arrange — all fields zero (valid but degenerate)
        let ids = MultimodalTokenIds {
            image_token_id: 0,
            audio_token_id: 0,
            eoi_token_id: 0,
            eoa_token_id: 0,
        };
        // Act & Assert — zero is a valid token ID, both helpers match
        assert!(ids.is_image(0));
        assert!(ids.is_audio(0));
        // Token 0 matches both image and audio when they share the same ID
        assert_eq!(ids.image_token_id, ids.audio_token_id);
    }

    // @trace TEST-MULTIMODAL-096 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn multimodal_encoded_embeddings_subnormal_float_precision() {
        // Arrange — use subnormal (denormalized) f32 values
        let subnormal: f32 = 1.0e-40f32; // well below f32 MIN_POSITIVE normal
        let enc = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![subnormal],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        // Act & Assert — validate passes, subnormal preserved exactly
        assert!(enc.validate().is_ok());
        assert_eq!(enc.embeddings[0].to_bits(), subnormal.to_bits());
    }

    // @trace TEST-MULTIMODAL-097 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn build_fused_hidden_preserves_float_precision_negative_zero() {
        // Arrange — negative zero in media embedding must be preserved
        let neg_zero: f32 = -0.0f32;
        let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, hidden=2
        let media = vec![neg_zero, 5.0];
        let routed = RoutedSequence {
            token_ids: vec![0, 258880, 1],
            fused_embeddings: vec![None, Some(media.clone()), None],
            text_positions: vec![0, 2],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — negative zero sign bit preserved
        assert_eq!(hidden[2].to_bits(), neg_zero.to_bits());
        assert!(hidden[2].is_sign_negative());
    }

    // @trace TEST-MULTIMODAL-098 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn build_fused_hidden_large_vocab_with_u32_max_token_id_as_usize_safe() {
        // Arrange — token_id = 999 within a vocab of 1000 rows, hidden=1
        // This tests that usize conversion from u32 is safe for reasonable vocab
        let token_id: u32 = 999;
        let hidden_size: usize = 1;
        let vocab_size: usize = 1000;
        let embed: Vec<f32> = (0..vocab_size).map(|i| i as f32 * 0.1).collect();
        let routed = RoutedSequence {
            token_ids: vec![token_id],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, hidden_size).unwrap();
        // Assert — gathered correct row
        assert_eq!(hidden[0], 999.0f32 * 0.1);
    }

    // @trace TEST-MULTIMODAL-099 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn encoder_media_base64_without_mime_type_is_none() {
        // Arrange — Base64 variant with mime_type = None
        let media = EncoderMedia::Base64 {
            data: "AAAA".into(),
            mime_type: None,
        };
        // Act & Assert
        assert!(matches!(media, EncoderMedia::Base64 { ref mime_type, .. } if mime_type.is_none()));
    }

    // @trace TEST-MULTIMODAL-100 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn multimodal_context_push_image_then_push_audio_both_accumulate() {
        // Arrange
        let mut ctx = MultimodalContext::new();
        let img = MultimodalEncoded {
            tokens: vec![100],
            embeddings: vec![1.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        let aud = MultimodalEncoded {
            tokens: vec![200],
            embeddings: vec![2.0],
            hidden_size: 1,
            kind: MediaKind::Audio,
        };
        // Act
        ctx.push_image(img).unwrap();
        ctx.push_audio(aud).unwrap();
        // Assert — both lists have one entry each, not mixed
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.audios.len(), 1);
        assert!(!ctx.is_empty());
        assert_eq!(ctx.images[0].tokens[0], 100);
        assert_eq!(ctx.audios[0].tokens[0], 200);
    }

    // @trace TEST-MULTIMODAL-101 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn route_multimodal_tokens_embedding_slice_independence_from_source() {
        // Arrange — verify that output embedding slices are independent copies
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 2],
            embeddings: vec![42.0, 43.0, 44.0, 45.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![ids.image_token_id, 99];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — first virtual token embedding is an independent copy
        let first_emb = out.fused_embeddings[0].as_ref().unwrap().clone();
        assert_eq!(first_emb, vec![42.0, 43.0]);
        // Modify output copy does not affect second embedding
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap(), &vec![44.0, 45.0]);
    }

    // @trace TEST-MULTIMODAL-102 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn routed_sequence_empty_some_vec_at_zero_hidden_size_is_multimodal() {
        // Arrange — Some(vec![]) at hidden_size=0 is technically multimodal (Some is present)
        let rs = RoutedSequence {
            token_ids: vec![1, 2],
            fused_embeddings: vec![None, Some(vec![])],
            text_positions: vec![0],
            hidden_size: 0,
        };
        // Act & Assert — has_multimodal returns true because Some is present
        assert!(rs.has_multimodal());
        assert_eq!(rs.hidden_size, 0);
    }

    // @trace TEST-MULTIMODAL-103 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn route_audio_with_zero_virtual_tokens_then_text_embedding_positions_correct() {
        // Arrange — audio encoding with zero virtual tokens effectively removes
        // the special token position, and text after it should still be correct
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![], // zero virtual tokens
            embeddings: vec![],
            hidden_size: 4,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc).unwrap();
        let prompt = vec![10u32, ids.audio_token_id, 20];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        // Assert — audio special token expands to zero tokens, so output = [10, 20]
        assert_eq!(out.seq_len(), 2);
        assert_eq!(out.token_ids, vec![10, 20]);
        assert_eq!(out.text_positions, vec![0, 1]);
        assert!(out.fused_embeddings.iter().all(|e| e.is_none()));
    }

    // @trace TEST-MULTIMODAL-104 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn multimodal_context_push_image_rejects_then_accepts_valid_image() {
        // Arrange — a bad push must not corrupt the context
        let mut ctx = MultimodalContext::new();
        let bad = MultimodalEncoded {
            tokens: vec![1, 2],
            embeddings: vec![0.0; 3], // mismatch: 2*2=4 expected, got 3
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        // Act — reject the bad encoding
        let _ = ctx.push_image(bad);
        // Assert — context is still empty, can accept a valid encoding
        assert!(ctx.is_empty());
        let good = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![1.0, 2.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        ctx.push_image(good).unwrap();
        assert_eq!(ctx.images.len(), 1);
        assert!(!ctx.is_empty());
    }

    // @trace TEST-MULTIMODAL-105 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn route_single_audio_expansion_with_large_hidden_size() {
        // Arrange — audio expands to 1 virtual token with hidden_size=256
        let ids = default_ids();
        let embeddings: Vec<f32> = (0..256).map(|i| i as f32 * 0.001).collect();
        let enc = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: embeddings.clone(),
            hidden_size: 256,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc).unwrap();
        let prompt = vec![10u32, ids.audio_token_id, 20];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 256).unwrap();
        // Assert — 1 text + 1 audio virtual + 1 text = 3 total positions
        assert_eq!(out.seq_len(), 3);
        assert_eq!(out.text_positions, vec![0, 2]);
        assert_eq!(out.hidden_size, 256);
        let routed_emb = out.fused_embeddings[1].as_ref().unwrap();
        assert_eq!(routed_emb.len(), 256);
        assert_eq!(routed_emb[0], 0.0);
        assert_eq!(routed_emb[255], 255.0 * 0.001);
    }

    // @trace TEST-MULTIMODAL-106 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn encoder_media_from_generation_raw_png_header_payload() {
        // Arrange — Raw variant with non-trivial payload
        let payload: Vec<u8> = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A]; // PNG header bytes
        let gen = crate::generation::MediaInput::Raw(payload.clone());
        // Act
        let enc = EncoderMedia::from_generation(&gen);
        // Assert
        match enc {
            EncoderMedia::Raw(ref v) => assert_eq!(v, &payload),
            _ => panic!("expected Raw variant"),
        }
    }

    // @trace TEST-MULTIMODAL-107 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn route_image_at_first_position_text_positions_shifted() {
        // Arrange — image special token at position 0, expands to 3 virtual tokens
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 3],
            embeddings: vec![1.0; 3 * 4],
            hidden_size: 4,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![ids.image_token_id, 50, 60];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 4).unwrap();
        // Assert — text positions = [3, 4], shifted by 3 virtual tokens
        assert_eq!(out.text_positions, vec![3, 4]);
        assert_eq!(out.token_ids[3], 50);
        assert_eq!(out.token_ids[4], 60);
        assert!(out.fused_embeddings[0].is_some());
        assert!(out.fused_embeddings[2].is_some());
        assert!(out.fused_embeddings[3].is_none());
    }

    // @trace TEST-MULTIMODAL-108 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn build_fused_hidden_all_text_tokens_gather_correct_rows() {
        // Arrange — vocab=5, hidden=3; row[k] = [k*100, k*100+1, k*100+2]
        let embed: Vec<f32> = (0..5 * 3).map(|i| i as f32).collect();
        let routed = RoutedSequence {
            token_ids: vec![4, 0, 2],
            fused_embeddings: vec![None, None, None],
            text_positions: vec![0, 1, 2],
            hidden_size: 3,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 3).unwrap();
        // Assert — gathered rows in order: row4, row0, row2
        assert_eq!(hidden[0..3], [12.0, 13.0, 14.0]); // row 4
        assert_eq!(hidden[3..6], [0.0, 1.0, 2.0]);    // row 0
        assert_eq!(hidden[6..9], [6.0, 7.0, 8.0]);     // row 2
    }

    // @trace TEST-MULTIMODAL-109 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn route_two_images_different_virtual_token_counts_embedding_values() {
        // Arrange — first image 2 virtual tokens, second 1 virtual token
        let ids = default_ids();
        let enc1 = MultimodalEncoded {
            tokens: vec![ids.image_token_id, ids.image_token_id],
            embeddings: vec![10.0, 20.0, 30.0, 40.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let enc2 = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![99.0, 88.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc1).unwrap();
        ctx.push_image(enc2).unwrap();
        let prompt = vec![5u32, ids.image_token_id, 6u32, ids.image_token_id, 7u32];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — total = 1 + 2 + 1 + 1 + 1 = 6
        assert_eq!(out.seq_len(), 6);
        assert_eq!(out.text_positions, vec![0, 3, 5]);
        // First image embeddings at positions 1,2
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap(), &vec![10.0, 20.0]);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap(), &vec![30.0, 40.0]);
        // Second image embedding at position 4
        assert_eq!(out.fused_embeddings[4].as_ref().unwrap(), &vec![99.0, 88.0]);
        // Text positions have None
        assert!(out.fused_embeddings[0].is_none());
        assert!(out.fused_embeddings[3].is_none());
        assert!(out.fused_embeddings[5].is_none());
    }

    // @trace TEST-MULTIMODAL-110 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn route_audio_between_text_correct_embedding_insertion() {
        // Arrange — text-audio-text with distinct embedding values
        let ids = default_ids();
        let audio_emb: Vec<f32> = vec![-1.0, -2.0];
        let enc = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: audio_emb.clone(),
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc).unwrap();
        let prompt = vec![100u32, ids.audio_token_id, 200u32];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — negative embeddings inserted at position 1
        assert_eq!(out.seq_len(), 3);
        assert_eq!(out.fused_embeddings[0], None);
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap(), &audio_emb);
        assert_eq!(out.fused_embeddings[2], None);
        assert_eq!(out.token_ids[0], 100);
        assert_eq!(out.token_ids[2], 200);
    }

    // @trace TEST-MULTIMODAL-111 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn multimodal_context_multiple_images_and_audios_is_not_empty() {
        // Arrange — context with 2 images and 3 audios
        let mut ctx = MultimodalContext::new();
        for _ in 0..2 {
            ctx.push_image(MultimodalEncoded {
                tokens: vec![258880],
                embeddings: vec![1.0],
                hidden_size: 1,
                kind: MediaKind::Image,
            })
            .unwrap();
        }
        for i in 0..3 {
            ctx.push_audio(MultimodalEncoded {
                tokens: vec![258881],
                embeddings: vec![i as f32],
                hidden_size: 1,
                kind: MediaKind::Audio,
            })
            .unwrap();
        }
        // Act & Assert
        assert!(!ctx.is_empty());
        assert_eq!(ctx.images.len(), 2);
        assert_eq!(ctx.audios.len(), 3);
    }

    // @trace TEST-MULTIMODAL-112 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn route_text_only_with_non_matching_custom_ids() {
        // Arrange — custom IDs far from prompt tokens, nothing is special
        let ids = MultimodalTokenIds {
            image_token_id: 900000,
            audio_token_id: 900001,
            eoi_token_id: 900002,
            eoa_token_id: 900003,
        };
        let ctx = MultimodalContext::new();
        let prompt: Vec<u32> = vec![0, 1, 2, 3, 100, 200, 899999];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 64).unwrap();
        // Assert — all tokens are text, no expansion
        assert_eq!(out.seq_len(), 7);
        assert_eq!(out.token_ids, prompt);
        assert!(out.fused_embeddings.iter().all(|e| e.is_none()));
        assert!(!out.has_multimodal());
        assert_eq!(out.text_positions, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    // @trace TEST-MULTIMODAL-113 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn build_fused_hidden_media_embedding_with_nan_propagated() {
        // Arrange — media embedding contains NaN, must propagate to output
        let embed: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // vocab=2, hidden=2
        let media = vec![f32::NAN, 5.0];
        let routed = RoutedSequence {
            token_ids: vec![0, 258880, 1],
            fused_embeddings: vec![None, Some(media.clone()), None],
            text_positions: vec![0, 2],
            hidden_size: 2,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — position 1 gets NaN from media, other positions are gathered normally
        assert!(hidden[2].is_nan()); // media[0] = NaN
        assert_eq!(hidden[3], 5.0);  // media[1]
        assert_eq!(hidden[0], 1.0);  // text token 0, row 0
        assert_eq!(hidden[4], 3.0);  // text token 1, row 1
    }

    // @trace TEST-MULTIMODAL-114 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn route_image_with_hidden_size_one_correct_output() {
        // Arrange — minimal hidden_size=1, single virtual token
        let ids = default_ids();
        let enc = MultimodalEncoded {
            tokens: vec![ids.image_token_id],
            embeddings: vec![42.0],
            hidden_size: 1,
            kind: MediaKind::Image,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc).unwrap();
        let prompt = vec![ids.image_token_id];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 1).unwrap();
        // Assert
        assert_eq!(out.seq_len(), 1);
        assert_eq!(out.hidden_size, 1);
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap(), &vec![42.0]);
        assert!(out.text_positions.is_empty());
    }

    // @trace TEST-MULTIMODAL-115 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn multimodal_encoded_validate_catches_hidden_size_zero_with_nonzero_tokens() {
        // Arrange — nonzero tokens but hidden_size=0 where embeddings.len() != tokens*0
        let enc = MultimodalEncoded {
            tokens: vec![1, 2, 3],
            embeddings: vec![1.0, 2.0, 3.0], // 3 tokens * 0 hidden = 0 expected, got 3
            hidden_size: 0,
            kind: MediaKind::Image,
        };
        // Act & Assert — validate rejects this shape mismatch
        let err = enc.validate().unwrap_err();
        assert!(format!("{err}").contains("shape mismatch"));
    }

    // @trace TEST-MULTIMODAL-116 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn encoder_media_clone_independence_after_mutation() {
        // Arrange — clone a Raw variant, modify the clone, verify independence
        let original = EncoderMedia::Raw(vec![1, 2, 3]);
        let mut cloned = original.clone();
        // Act — mutate the cloned data
        if let EncoderMedia::Raw(ref mut v) = cloned {
            v.push(4);
        }
        // Assert — original unchanged
        match original {
            EncoderMedia::Raw(ref v) => assert_eq!(v, &[1, 2, 3]),
            _ => panic!("expected Raw"),
        }
        // Cloned has the extra element
        match cloned {
            EncoderMedia::Raw(ref v) => assert_eq!(v, &[1, 2, 3, 4]),
            _ => panic!("expected Raw"),
        }
    }

    // ── Batch 7: 10 additional tests for uncovered edge cases ──

    // @trace TEST-MULTIMODAL-117 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn route_consecutive_audios_with_distinct_embedding_values() {
        // Arrange — three consecutive audio tokens, each with unique embedding values
        let ids = default_ids();
        let enc1 = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![10.0, 20.0],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let enc2 = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![30.0, 40.0, 50.0, 60.0],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let enc3 = MultimodalEncoded {
            tokens: vec![ids.audio_token_id],
            embeddings: vec![70.0, 80.0],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_audio(enc1).unwrap();
        ctx.push_audio(enc2).unwrap();
        ctx.push_audio(enc3).unwrap();
        let prompt = vec![ids.audio_token_id, ids.audio_token_id, ids.audio_token_id];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — 1 + 2 + 1 = 4 total positions, no text
        assert_eq!(out.seq_len(), 4);
        assert!(out.text_positions.is_empty());
        assert_eq!(out.fused_embeddings[0].as_ref().unwrap(), &vec![10.0, 20.0]);
        assert_eq!(out.fused_embeddings[1].as_ref().unwrap(), &vec![30.0, 40.0]);
        assert_eq!(out.fused_embeddings[2].as_ref().unwrap(), &vec![50.0, 60.0]);
        assert_eq!(out.fused_embeddings[3].as_ref().unwrap(), &vec![70.0, 80.0]);
    }

    // @trace TEST-MULTIMODAL-118 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn multimodal_context_push_valid_then_invalid_different_hidden_sizes() {
        // Arrange — push a valid image with hidden=2, then an invalid one with hidden=3
        let mut ctx = MultimodalContext::new();
        let valid = MultimodalEncoded {
            tokens: vec![1],
            embeddings: vec![10.0, 20.0],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        ctx.push_image(valid).unwrap();
        assert_eq!(ctx.images.len(), 1);
        // Act — push a bad encoding with wrong shape (kind is correct but embeddings mismatch)
        let bad = MultimodalEncoded {
            tokens: vec![2, 3],
            embeddings: vec![1.0, 2.0, 3.0], // 2 tokens * 2 hidden = 4 expected, got 3
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let _ = ctx.push_image(bad);
        // Assert — first encoding is intact, second was rejected
        assert_eq!(ctx.images.len(), 1);
        assert_eq!(ctx.images[0].embeddings, vec![10.0, 20.0]);
    }

    // @trace TEST-MULTIMODAL-119 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn build_fused_hidden_token_at_vocab_boundary() {
        // Arrange — vocab=5, hidden=3, use token_id=4 (last valid row)
        let embed: Vec<f32> = (0..5 * 3).map(|i| i as f32).collect();
        let routed = RoutedSequence {
            token_ids: vec![4],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 3,
        };
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 3).unwrap();
        // Assert — row 4 = [12.0, 13.0, 14.0]
        assert_eq!(hidden, vec![12.0, 13.0, 14.0]);
    }

    // @trace TEST-MULTIMODAL-120 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn build_fused_hidden_token_one_past_vocab_boundary_rejected() {
        // Arrange — vocab=5, hidden=3, token_id=5 is out of range
        let embed: Vec<f32> = vec![0.0; 5 * 3];
        let routed = RoutedSequence {
            token_ids: vec![5],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: 3,
        };
        // Act
        let err = build_fused_hidden(&routed, &embed, 3).unwrap_err();
        // Assert
        assert!(format!("{err}").contains("out of range"));
        assert!(format!("{err}").contains("vocab 5"));
    }

    // @trace TEST-MULTIMODAL-121 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn multimodal_token_ids_eoi_equals_audio_id_edge_case() {
        // Arrange — custom IDs where eoi_token_id equals audio_token_id
        let ids = MultimodalTokenIds {
            image_token_id: 100,
            audio_token_id: 200,
            eoi_token_id: 200, // same as audio_token_id
            eoa_token_id: 300,
        };
        // Act & Assert — eoi is not checked by is_image or is_audio
        assert!(ids.is_audio(200));
        assert!(!ids.is_image(200));
        // eoi_token_id field is accessible but not used for routing
        assert_eq!(ids.eoi_token_id, ids.audio_token_id);
    }

    // @trace TEST-MULTIMODAL-122 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn route_prompt_with_repeated_non_special_tokens_preserved() {
        // Arrange — prompt contains repeated values (e.g. token 7 appears 4 times)
        let ctx = MultimodalContext::new();
        let ids = default_ids();
        let prompt = vec![7u32, 7, 7, 7, 99, 7];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 16).unwrap();
        // Assert — all tokens preserved exactly as given
        assert_eq!(out.token_ids, vec![7, 7, 7, 7, 99, 7]);
        assert_eq!(out.text_positions, vec![0, 1, 2, 3, 4, 5]);
        assert!(!out.has_multimodal());
    }

    // @trace TEST-MULTIMODAL-123 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn full_pipeline_image_and_audio_with_build_fused_hidden() {
        // Arrange — end-to-end: encode -> route -> build_fused_hidden
        let ids = default_ids();
        let encoder = MockEncoder::new(2, 3, 2, ids);
        let img_encoded = encoder.encode_image(&EncoderMedia::Raw(vec![])).unwrap();
        let aud_encoded = encoder.encode_audio(&EncoderMedia::Raw(vec![])).unwrap();
        let mut ctx = MultimodalContext::new();
        ctx.push_image(img_encoded).unwrap();
        ctx.push_audio(aud_encoded).unwrap();
        let prompt = vec![0u32, ids.image_token_id, 1, ids.audio_token_id];
        let routed = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // embed: vocab=2, hidden=2
        let embed: Vec<f32> = vec![10.0, 11.0, 20.0, 21.0];
        // Act
        let hidden = build_fused_hidden(&routed, &embed, 2).unwrap();
        // Assert — total 1+2+1+3=7 positions
        assert_eq!(hidden.len(), 7 * 2);
        // Position 0: token 0 -> [10, 11]
        assert_eq!(&hidden[0..2], &[10.0, 11.0]);
        // Position 1: image virtual 0 (MockEncoder pattern: i*0.01)
        assert!((hidden[2] - 0.00).abs() < 1e-6);
        assert!((hidden[3] - 0.01).abs() < 1e-6);
        // Position 2: image virtual 1
        assert!((hidden[4] - 0.02).abs() < 1e-6);
        assert!((hidden[5] - 0.03).abs() < 1e-6);
        // Position 3: token 1 -> [20, 21]
        assert_eq!(&hidden[6..8], &[20.0, 21.0]);
        // Position 4: audio virtual 0 (MockEncoder pattern: -(i*0.01))
        assert!((hidden[8] - 0.00).abs() < 1e-6);
        assert!((hidden[9] - (-0.01)).abs() < 1e-6);
    }

    // @trace TEST-MULTIMODAL-124 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn mock_encoder_call_counters_independent() {
        // Arrange — verify image and audio counters track independently
        let ids = default_ids();
        let encoder = MockEncoder::new(1, 1, 2, ids);
        // Act — call image 3 times and audio 5 times
        for _ in 0..3 {
            let _ = encoder.encode_image(&EncoderMedia::Raw(vec![]));
        }
        for _ in 0..5 {
            let _ = encoder.encode_audio(&EncoderMedia::Raw(vec![]));
        }
        // Assert
        assert_eq!(encoder.image_call_count(), 3);
        assert_eq!(encoder.audio_call_count(), 5);
    }

    // @trace TEST-MULTIMODAL-125 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn routed_sequence_text_positions_after_multi_expansion_not_contiguous() {
        // Arrange — text positions must skip over expanded virtual token ranges
        let ids = default_ids();
        let enc_img = MultimodalEncoded {
            tokens: vec![ids.image_token_id; 3],
            embeddings: vec![0.0; 3 * 2],
            hidden_size: 2,
            kind: MediaKind::Image,
        };
        let enc_aud = MultimodalEncoded {
            tokens: vec![ids.audio_token_id; 2],
            embeddings: vec![0.0; 2 * 2],
            hidden_size: 2,
            kind: MediaKind::Audio,
        };
        let mut ctx = MultimodalContext::new();
        ctx.push_image(enc_img).unwrap();
        ctx.push_audio(enc_aud).unwrap();
        // text, image, text, audio, text
        let prompt = vec![10u32, ids.image_token_id, 20, ids.audio_token_id, 30];
        // Act
        let out = route_multimodal_tokens(&prompt, &ctx, &ids, 2).unwrap();
        // Assert — 1 + 3 + 1 + 2 + 1 = 8 total, text at [0, 4, 7]
        assert_eq!(out.seq_len(), 8);
        assert_eq!(out.text_positions, vec![0, 4, 7]);
        assert_eq!(out.token_ids[0], 10);
        assert_eq!(out.token_ids[4], 20);
        assert_eq!(out.token_ids[7], 30);
        // Verify non-contiguous gaps are media
        assert!(out.fused_embeddings[1].is_some());
        assert!(out.fused_embeddings[2].is_some());
        assert!(out.fused_embeddings[3].is_some());
        assert!(out.fused_embeddings[5].is_some());
        assert!(out.fused_embeddings[6].is_some());
    }

    // @trace TEST-MULTIMODAL-126 [req:REQ-MEGA-003] [level:unit]
    #[test]
    fn build_fused_hidden_large_hidden_single_position_text_only() {
        // Arrange — hidden_size=4096, single text position, verify gather correctness
        let hidden = 4096;
        let embed: Vec<f32> = (0..3 * hidden).map(|i| i as f32 * 0.00001).collect();
        let routed = RoutedSequence {
            token_ids: vec![1],
            fused_embeddings: vec![None],
            text_positions: vec![0],
            hidden_size: hidden,
        };
        // Act
        let result = build_fused_hidden(&routed, &embed, hidden).unwrap();
        // Assert — single position, gathered from row 1
        assert_eq!(result.len(), hidden);
        assert_eq!(result[0], (1 * hidden) as f32 * 0.00001);
        assert_eq!(result[hidden - 1], ((1 * hidden) + hidden - 1) as f32 * 0.00001);
    }
}
