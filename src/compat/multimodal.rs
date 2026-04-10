//! 多模态 Token 路由 — Gemma 4 Vision/Audio/Text 统一序列化
//!
//! 将文本 token 序列中的特殊 token (image_token_id, audio_token_id) 替换为
//! 对应编码器输出的 token 序列。
//!
//! P3 阶段骨架，完整实现待后续填充。

use crate::engine::executor::BackendError;

/// 多模态特殊 token ID (从 config.json 读取)
#[derive(Debug, Clone)]
pub struct MultimodalTokenIds {
    /// 图像占位符 token ID (Gemma 4: 258880)
    pub image_token_id: u32,
    /// 音频占位符 token ID (Gemma 4: 258881)
    pub audio_token_id: u32,
    /// 图像结束 token ID (Gemma 4: 258882)
    pub eoi_token_id: u32,
    /// 音频结束 token ID (Gemma 4: 258883)
    pub eoa_token_id: u32,
}

impl Default for MultimodalTokenIds {
    fn default() -> Self {
        Self {
            image_token_id: 258880,
            audio_token_id: 258881,
            eoi_token_id: 258882,
            eoa_token_id: 258883,
        }
    }
}

/// 多模态输入 — 文本 + 可选的图像/音频数据
#[derive(Debug, Clone)]
pub struct MultimodalInput {
    /// 文本 token 序列 (含特殊占位符 token)
    pub token_ids: Vec<u32>,
    /// 图像像素数据 [channels, height, width] (None = 纯文本)
    pub image_pixels: Option<Vec<f32>>,
    /// 音频波形数据 [num_samples] (None = 无音频)
    pub audio_samples: Option<Vec<f32>>,
}

/// 多模态嵌入 — 文本/图像/音频 token 的统一嵌入序列
#[derive(Debug)]
pub struct MultimodalEmbeddings {
    /// 融合后的嵌入序列 [total_seq_len, hidden_size]
    pub embeddings: Vec<f32>,
    /// 融合后的总序列长度
    pub seq_len: usize,
    /// 原始文本 token 在融合序列中的位置映射
    pub text_positions: Vec<usize>,
}

/// 将多模态输入融合为统一嵌入序列。
///
/// Pipeline:
/// 1. 文本 token → embedding lookup
/// 2. 检测 image_token_id → 调用 vision_encode → 替换为视觉 token 序列
/// 3. 检测 audio_token_id → 调用 audio_encode → 替换为音频 token 序列
/// 4. 拼接为连续嵌入序列
///
/// 暂未实现，等待 P3.1 (Vision) 和 P3.2 (Audio) 完成后填充。
pub fn fuse_multimodal_embeddings(
    _input: &MultimodalInput,
    _text_embeddings: &[f32],
    _hidden_size: usize,
    _token_ids: &MultimodalTokenIds,
) -> Result<MultimodalEmbeddings, BackendError> {
    Err(BackendError::Other(
        "multimodal fusion not yet implemented (P3.3)".into(),
    ))
}
