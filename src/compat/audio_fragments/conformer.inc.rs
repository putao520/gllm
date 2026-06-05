// UsmConformerEncoder — MultimodalEncoder impl
// ============================================================================

/// USM-style Conformer 音频编码器,实现 `MultimodalEncoder::encode_audio`。
///
/// 由 loader 在检测到 `ModelConfig::audio_config` 时构造,注入到 `Client` 的
/// multimodal encoder 槽。`encode_image` 暂不支持 (返回错误);真实图像编码
/// 由 `SigLipVisionEncoder` (T44 worktree) 提供。两个编码器会 compose
/// 到同一个 MultiEncoder 委派结构中(见 `client.rs`)。
pub struct UsmConformerEncoder {
    config: AudioConfig,
    weights: std::sync::Arc<dyn AudioTensorLookup + Send + Sync>,
    /// audio token 占位 ID (来自 `MultimodalTokenIds.audio_token_id`)。
    audio_token_id: u32,
}

impl std::fmt::Debug for UsmConformerEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UsmConformerEncoder")
            .field("config", &self.config)
            .field("audio_token_id", &self.audio_token_id)
            .finish()
    }
}

impl UsmConformerEncoder {
    /// 构造一个 USM Conformer encoder。
    pub fn new(
        config: AudioConfig,
        weights: std::sync::Arc<dyn AudioTensorLookup + Send + Sync>,
        audio_token_id: u32,
    ) -> Result<Self, BackendError> {
        config.validate()?;
        Ok(Self {
            config,
            weights,
            audio_token_id,
        })
    }

    /// 把 `EncoderMedia` 解析为原始 PCM 样本 (f32, 单声道, sample_rate Hz)。
    ///
    /// 当前只接受 `EncoderMedia::Raw` — 其中 bytes 被 reinterpret 成
    /// `&[f32]` (little-endian)。其他模式(File / Base64 / Url)留给后续
    /// task 扩展(需要接入 WAV/FLAC 解码或 HTTP 下载,非 JIT 路径)。
    fn decode_media_to_pcm(&self, media: &EncoderMedia) -> Result<Vec<f32>, BackendError> {
        match media {
            EncoderMedia::Raw(bytes) => {
                if bytes.len() % 4 != 0 {
                    return Err(BackendError::Other(format!(
                        "UsmConformerEncoder: Raw PCM 字节数 {} 不是 4 的倍数 (f32 对齐)",
                        bytes.len()
                    )));
                }
                let n = bytes.len() / 4;
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    out.push(f32::from_le_bytes([
                        bytes[i * 4],
                        bytes[i * 4 + 1],
                        bytes[i * 4 + 2],
                        bytes[i * 4 + 3],
                    ]));
                }
                Ok(out)
            }
            EncoderMedia::File(path) => Err(BackendError::Other(format!(
                "UsmConformerEncoder: EncoderMedia::File 解码未接入 (path={}); \
                 caller 应预先解码 WAV/FLAC 到 f32 PCM 并用 EncoderMedia::Raw 传入",
                path.display()
            ))),
            EncoderMedia::Base64 { mime_type, .. } => Err(BackendError::Other(format!(
                "UsmConformerEncoder: EncoderMedia::Base64 解码未接入 (mime_type={:?}); \
                 caller 应预先解码并用 EncoderMedia::Raw 传入",
                mime_type
            ))),
            EncoderMedia::Url(url) => Err(BackendError::Other(format!(
                "UsmConformerEncoder: EncoderMedia::Url 解码未接入 (url={}); \
                 caller 应预先下载并解码到 f32 PCM",
                url
            ))),
        }
    }
}

impl MultimodalEncoder for UsmConformerEncoder {
    fn encode_image(&self, _media: &EncoderMedia) -> Result<MultimodalEncoded, BackendError> {
        Err(BackendError::Other(
            "UsmConformerEncoder::encode_image: 音频编码器不处理图像; \
             caller 应注册独立的 vision encoder (SigLIP)"
                .into(),
        ))
    }

    fn encode_audio(&self, media: &EncoderMedia) -> Result<MultimodalEncoded, BackendError> {
        let pcm = self.decode_media_to_pcm(media)?;
        let embeddings = audio_encode(&pcm, &self.config, &*self.weights)?;
        let hidden = self.config.hidden_size;
        if embeddings.len() % hidden != 0 {
            return Err(BackendError::Other(format!(
                "UsmConformerEncoder: audio_encode 输出 {} 不是 hidden_size {} 的整数倍",
                embeddings.len(),
                hidden
            )));
        }
        let num_tokens = embeddings.len() / hidden;
        let tokens = vec![self.audio_token_id; num_tokens];
        let encoded = MultimodalEncoded {
            tokens,
            embeddings,
            hidden_size: hidden,
            kind: MediaKind::Audio,
        };
        encoded.validate()?;
        Ok(encoded)
    }
}

// ============================================================================
// try_build_usm_from_tensors — 集成入口 (与 try_build_siglip_from_tensors 对偶)
// ============================================================================

/// 按 Conformer block + 顶层 mel_projection / encoder_final_norm 需求列出
/// caller 必须提供的权重张量名称。用于 UsmConformerEncoder 构造前的
/// weight presence 预检。
fn usm_conformer_required_tensors(config: &AudioConfig) -> Vec<String> {
    let mut names = Vec::with_capacity(2 + config.num_layers * 23 + 2);
    names.push("audio_tower.feature_projection.weight".into());
    for i in 0..config.num_layers {
        let base = format!("audio_tower.encoder.layers.{i}");
        for suffix in [
            "norm_ff1.weight",
            "norm_ff1.bias",
            "ff1_module.linear_in.weight",
            "ff1_module.linear_out.weight",
            "norm_self_attn.weight",
            "norm_self_attn.bias",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "conv_module.norm.weight",
            "conv_module.norm.bias",
            "conv_module.pointwise_conv1.weight",
            "conv_module.depthwise_conv.weight",
            "conv_module.bn.weight",
            "conv_module.bn.bias",
            "conv_module.pointwise_conv2.weight",
            "norm_ff2.weight",
            "norm_ff2.bias",
            "ff2_module.linear_in.weight",
            "ff2_module.linear_out.weight",
            "norm_final.weight",
            "norm_final.bias",
        ] {
            names.push(format!("{base}.{suffix}"));
        }
    }
    names.push("audio_tower.encoder.final_norm.weight".into());
    names.push("audio_tower.encoder.final_norm.bias".into());
    names
}

/// 从 caller 提供的 tensor lookup closure 构造 `UsmConformerEncoder`。
///
/// - `Ok(Some(encoder))`: 全部权重具备,编码器可用。
/// - `Ok(None)`: 至少一个 USM Conformer 权重缺失 (意味着模型只是声明了
///   `audio_config` 但未打包音频塔权重),caller 可降级为无音频能力。
/// - `Err`: 尺寸/形状错误等硬故障。
pub fn try_build_usm_from_tensors<F>(
    config: &AudioConfig,
    token_ids: crate::compat::multimodal::MultimodalTokenIds,
    mut fetch: F,
) -> Result<Option<UsmConformerEncoder>, BackendError>
where
    F: FnMut(&str) -> Option<(Vec<f32>, Vec<usize>)>,
{
    config.validate()?;
    let required = usm_conformer_required_tensors(config);
    let mut weights = InMemoryAudioWeights::new();
    for name in &required {
        let Some((data, shape)) = fetch(name) else {
            log::debug!("USM Conformer encoder: missing tensor '{name}', skipping auto-build");
            return Ok(None);
        };
        let total: usize = shape.iter().product();
        if data.len() != total {
            return Err(BackendError::Other(format!(
                "try_build_usm_from_tensors: tensor '{name}' data len {} != shape product {:?}",
                data.len(),
                shape
            )));
        }
        weights.insert(name.clone(), data, shape);
    }
    let encoder = UsmConformerEncoder::new(
        config.clone(),
        std::sync::Arc::new(weights),
        token_ids.audio_token_id,
    )?;
    Ok(Some(encoder))
}

// ============================================================================
// In-memory AudioTensorLookup — 简单实现,用于集成路径 & 测试
// ============================================================================

/// 由 (名字 → f32 扁平向量 + shape) 字典支撑的 `AudioTensorLookup` 实现。
///
/// loader 在构造 `UsmConformerEncoder` 时用此结构持有已解量化的权重快照。
#[derive(Debug, Default)]
pub struct InMemoryAudioWeights {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl InMemoryAudioWeights {
    pub fn new() -> Self {
        Self { tensors: HashMap::new() }
    }

    pub fn insert(&mut self, name: impl Into<String>, data: Vec<f32>, shape: Vec<usize>) {
        let total: usize = shape.iter().product();
        debug_assert_eq!(
            data.len(),
            total,
            "InMemoryAudioWeights::insert: data.len()={} != shape prod={} for '{:?}'",
            data.len(),
            total,
            shape
        );
        self.tensors.insert(name.into(), (data, shape));
    }
}

impl AudioTensorLookup for InMemoryAudioWeights {
    fn get_audio_tensor(&self, name: &str) -> Option<&[f32]> {
        self.tensors.get(name).map(|(d, _)| d.as_slice())
    }

    fn audio_tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|(_, s)| s.as_slice())
    }
}

// ============================================================================
