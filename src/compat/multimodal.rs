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
/// config 未显式提供时由 `MultimodalTokenIds::gemma4_defaults()` 注入。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Gemma 4 默认值 (from `gemma4-implementation-plan.md §P3.3`)。
    ///
    /// 仅作为 `ModelConfig` 未显式声明多模态能力时的回退，由 loader 根据
    /// `vision_config` 是否存在决定是否注入。
    pub fn gemma4_defaults() -> Self {
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

// 兼容历史类型别名 (loader / model_config 曾期望 Default impl)。
impl Default for MultimodalTokenIds {
    fn default() -> Self {
        Self::gemma4_defaults()
    }
}

// ============================================================================
// EncoderMedia — 编码器输入 (对齐 generation::MediaInput)
// ============================================================================

/// 编码器期望的媒体输入 (路径或原始字节)。
///
/// `generation::MediaInput` 的内部表示：文件路径保留为 `PathBuf`，
/// Base64 延后解码，原始字节直接透传。
#[derive(Debug, Clone)]
pub enum EncoderMedia {
    /// 本地文件路径
    File(PathBuf),
    /// Base64 编码数据 (含可选 MIME type)
    Base64 {
        data: String,
        mime_type: Option<String>,
    },
    /// 原始字节（已解码的像素 / PCM）
    Raw(Vec<u8>),
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
        }
    }
}

// ============================================================================
// MediaKind / MultimodalEncoded — 编码器输出
// ============================================================================

/// 媒体类型标签。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        MultimodalTokenIds::gemma4_defaults()
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
}
