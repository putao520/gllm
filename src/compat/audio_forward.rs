//! Audio Encoder — USM-style Conformer (Gemma 4 多模态)
//!
//! 将音频波形编码为音频 token 序列，插入到文本序列中。
//! P3 阶段骨架，完整实现待后续填充。

use crate::engine::executor::BackendError;

/// USM Conformer Audio Encoder 配置。
///
/// 从 config.json 的 `audio_config` 子对象解析。
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// 采样率 (Hz)，通常 16000
    pub sample_rate: usize,
    /// Conformer 隐藏维度
    pub hidden_size: usize,
    /// Conformer 层数
    pub num_layers: usize,
    /// 注意力头数
    pub num_heads: usize,
    /// 卷积核大小
    pub conv_kernel_size: usize,
    /// FFN 中间维度
    pub intermediate_size: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            hidden_size: 512,
            num_layers: 12,
            num_heads: 8,
            conv_kernel_size: 31,
            intermediate_size: 2048,
        }
    }
}

/// 将音频波形编码为音频 token 序列。
///
/// 输入: raw_audio [num_samples] (单声道 f32 PCM)
/// 输出: audio_tokens [num_frames, hidden_size]
///
/// Pipeline:
/// 1. Mel spectrogram 提取
/// 2. Conformer encoder (DepthwiseConv1D + Self-Attention 交替)
/// 3. 下采样到目标帧率
pub fn audio_encode(
    _raw_audio: &[f32],
    _config: &AudioConfig,
) -> Result<Vec<f32>, BackendError> {
    Err(BackendError::Other(
        "audio encoder not yet implemented (P3.2)".into(),
    ))
}
