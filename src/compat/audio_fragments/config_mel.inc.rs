// ============================================================================
// AudioConfig — USM Conformer geometry
// ============================================================================

/// USM Conformer Audio Encoder 配置。
///
/// 从 config.json 的 `audio_config` 子对象解析。
/// 默认值对齐 Gemma 4 音频塔 (USM-v2 / Conformer-512) 的官方参数。
#[derive(Debug, Clone, PartialEq)]
pub struct AudioConfig {
    /// 采样率 (Hz),通常 16000。
    pub sample_rate: usize,
    /// Conformer 隐藏维度。
    pub hidden_size: usize,
    /// Conformer 层数。
    pub num_layers: usize,
    /// 自注意力头数。
    pub num_heads: usize,
    /// Conv module 的 depthwise 卷积核大小 (奇数,非因果 SAME padding)。
    pub conv_kernel_size: usize,
    /// FFN 中间维度 (典型 = 4 × hidden_size)。
    pub intermediate_size: usize,
    /// Mel filterbank 数量 (= Conformer 首层输入维度的 channel 数)。
    pub num_mel_bins: usize,
    /// 短时 FFT 长度 (必须是 2 的幂,例如 512)。
    pub fft_size: usize,
    /// 分帧跳跃长度 (sample 数,10ms@16kHz = 160)。
    pub hop_length: usize,
    /// 分析窗长度 (sample 数,25ms@16kHz = 400)。
    pub win_length: usize,
    /// LayerNorm epsilon。
    pub layer_norm_eps: f32,
    /// Mel spec → Conformer 输入投影后的时间下采样倍率 (通常 4,
    /// 把 10ms 帧率压到 40ms)。当前实现直接传递 mel 帧,无额外下采样
    /// (下采样由 subsample_conv 权重学习,尚未纳入骨架;
    /// 设为 1 表示透传,>=2 时跨帧 stride)。
    pub stride: usize,
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
            num_mel_bins: 80,
            fft_size: 512,
            hop_length: 160,
            win_length: 400,
            layer_norm_eps: 1e-5,
            stride: 1,
        }
    }
}

impl AudioConfig {
    /// Derived: head_dim = hidden_size / num_heads。调用前必须保证 num_heads > 0
    /// 且 hidden_size % num_heads == 0 (由 validate 保证)。
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// 校验配置一致性。
    pub fn validate(&self) -> Result<(), BackendError> {
        if self.num_heads == 0 || self.hidden_size == 0 {
            return Err(BackendError::Other(
                "AudioConfig: hidden_size 与 num_heads 必须为正".into(),
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(BackendError::Other(format!(
                "AudioConfig: hidden_size {} 不是 num_heads {} 的整数倍",
                self.hidden_size, self.num_heads
            )));
        }
        if self.conv_kernel_size == 0 || self.conv_kernel_size.is_multiple_of(2) {
            return Err(BackendError::Other(format!(
                "AudioConfig: conv_kernel_size 必须为奇数 (non-causal SAME padding 约束), got {}",
                self.conv_kernel_size
            )));
        }
        if !self.fft_size.is_power_of_two() {
            return Err(BackendError::Other(format!(
                "AudioConfig: fft_size 必须是 2 的幂, got {}",
                self.fft_size
            )));
        }
        if self.win_length == 0 || self.win_length > self.fft_size {
            return Err(BackendError::Other(format!(
                "AudioConfig: win_length ({}) 必须 ∈ (0, fft_size={}]",
                self.win_length, self.fft_size
            )));
        }
        if self.hop_length == 0 {
            return Err(BackendError::Other(
                "AudioConfig: hop_length 必须为正".into(),
            ));
        }
        if self.num_mel_bins == 0 || self.num_mel_bins > self.fft_size / 2 + 1 {
            return Err(BackendError::Other(format!(
                "AudioConfig: num_mel_bins ({}) 必须 ∈ (0, fft_size/2 + 1 = {}]",
                self.num_mel_bins,
                self.fft_size / 2 + 1
            )));
        }
        if self.stride == 0 {
            return Err(BackendError::Other("AudioConfig: stride 必须为正".into()));
        }
        Ok(())
    }
}

// ============================================================================
// AudioTensorLookup trait — 权重查询 (caller 注入)
// ============================================================================

/// 音频编码器权重查询接口,仿 `VisionTensorLookup`。
///
/// Gemma 4 约定所有 audio tower 权重都挂在 `audio_tower.encoder.layers.{i}.*`
/// 下,caller 负责把原始 safetensors / GGUF tensor 暴露成 `&[f32]` (已按
/// 存储顺序反量化)。
pub trait AudioTensorLookup {
    /// 返回 tensor 对应的 f32 扁平数据;缺失返回 None。
    fn get_audio_tensor(&self, name: &str) -> Option<&[f32]>;

    /// 返回 tensor 形状;缺失返回 None。
    fn audio_tensor_shape(&self, name: &str) -> Option<&[usize]>;
}

// ============================================================================
// Mel Spectrogram 预处理 — Pure Rust Cooley-Tukey FFT + mel filterbank
// ============================================================================

/// In-place Cooley-Tukey radix-2 FFT。
///
/// `real` / `imag` 必须同长度,且长度为 2 的幂。
fn fft_radix2(real: &mut [f32], imag: &mut [f32]) {
    let n = real.len();
    debug_assert_eq!(imag.len(), n);
    debug_assert!(n.is_power_of_two());
    if n <= 1 {
        return;
    }
    // Bit reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }
    // Cooley-Tukey
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / (len as f32);
        let w_re = angle.cos();
        let w_im = angle.sin();
        let mut i = 0usize;
        while i < n {
            let mut wr = 1.0f32;
            let mut wi = 0.0f32;
            for k in 0..half {
                let a_re = real[i + k];
                let a_im = imag[i + k];
                let b_re = real[i + k + half] * wr - imag[i + k + half] * wi;
                let b_im = real[i + k + half] * wi + imag[i + k + half] * wr;
                real[i + k] = a_re + b_re;
                imag[i + k] = a_im + b_im;
                real[i + k + half] = a_re - b_re;
                imag[i + k + half] = a_im - b_im;
                // Rotate twiddle: (wr, wi) *= (w_re, w_im)
                let new_wr = wr * w_re - wi * w_im;
                let new_wi = wr * w_im + wi * w_re;
                wr = new_wr;
                wi = new_wi;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Hz → mel 转换 (HTK 公式)。
#[inline]
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// mel → Hz。
#[inline]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// 构造 `[num_mel_bins, n_fft_bins]` 的三角 mel filterbank。
///
/// `n_fft_bins = fft_size / 2 + 1`。bank[m,k] 权重 row-major 展平返回。
fn build_mel_filterbank(
    num_mel_bins: usize,
    fft_size: usize,
    sample_rate: usize,
) -> Vec<f32> {
    let n_bins = fft_size / 2 + 1;
    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(sample_rate as f32 / 2.0);
    // num_mel_bins + 2 个 mel 点 (含左右端点)
    let mut mel_points = Vec::with_capacity(num_mel_bins + 2);
    for i in 0..num_mel_bins + 2 {
        let m = mel_low + (mel_high - mel_low) * (i as f32) / ((num_mel_bins + 1) as f32);
        mel_points.push(mel_to_hz(m));
    }
    // 转成 FFT bin 索引 (浮点)
    let bin_points: Vec<f32> = mel_points
        .iter()
        .map(|&hz| hz * (fft_size as f32) / (sample_rate as f32))
        .collect();

    let mut bank = vec![0.0f32; num_mel_bins * n_bins];
    for m in 0..num_mel_bins {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];
        for k in 0..n_bins {
            let x = k as f32;
            let w = if x >= left && x <= center && center > left {
                (x - left) / (center - left)
            } else if x > center && x <= right && right > center {
                (right - x) / (right - center)
            } else {
                0.0
            };
            bank[m * n_bins + k] = w.max(0.0);
        }
    }
    bank
}

/// Hann 分析窗 (长度 = `win_length`, 零填充到 `fft_size`)。
fn hann_window(win_length: usize, fft_size: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; fft_size];
    for n in 0..win_length {
        let v = 0.5 * (1.0 - (2.0 * PI * (n as f32) / ((win_length - 1).max(1) as f32)).cos());
        w[n] = v;
    }
    w
}

/// 提取 log-mel spectrogram。
///
/// 输入: raw PCM f32 `[num_samples]`。
/// 输出: `(frames_flat, num_frames)` — `frames_flat` 是 row-major
/// `[num_frames, num_mel_bins]` (每帧 num_mel_bins 维)。
pub fn mel_spectrogram(
    raw_audio: &[f32],
    config: &AudioConfig,
) -> Result<(Vec<f32>, usize), BackendError> {
    config.validate()?;
    if raw_audio.is_empty() {
        return Err(BackendError::Other(
            "mel_spectrogram: 空音频输入".into(),
        ));
    }
    let fft_size = config.fft_size;
    let win_length = config.win_length;
    let hop = config.hop_length;
    let num_mels = config.num_mel_bins;
    let n_bins = fft_size / 2 + 1;

    // 最小长度 = win_length; 音频短于 win_length 时零填充到 win_length
    let padded_len = raw_audio.len().max(win_length);
    let num_frames = if padded_len >= win_length {
        1 + (padded_len - win_length) / hop
    } else {
        1
    };

    let window = hann_window(win_length, fft_size);
    let filterbank = build_mel_filterbank(num_mels, fft_size, config.sample_rate);

    let mut out = vec![0.0f32; num_frames * num_mels];
    let mut real_buf = vec![0.0f32; fft_size];
    let mut imag_buf = vec![0.0f32; fft_size];
    let mut power = vec![0.0f32; n_bins];

    for frame in 0..num_frames {
        let start = frame * hop;
        // 加窗
        for n in 0..fft_size {
            let idx = start + n;
            let sample = if idx < raw_audio.len() { raw_audio[idx] } else { 0.0 };
            real_buf[n] = sample * window[n];
            imag_buf[n] = 0.0;
        }
        fft_radix2(&mut real_buf, &mut imag_buf);
        // 功率谱: |X[k]|^2
        for k in 0..n_bins {
            power[k] = real_buf[k] * real_buf[k] + imag_buf[k] * imag_buf[k];
        }
        // mel 投影 + log
        for m in 0..num_mels {
            let mut energy = 0.0f32;
            for k in 0..n_bins {
                energy += filterbank[m * n_bins + k] * power[k];
            }
            // log(max(energy, eps)) — 避免 log(0)
            out[frame * num_mels + m] = (energy.max(1e-10)).ln();
        }
    }
    Ok((out, num_frames))
}

// ============================================================================
