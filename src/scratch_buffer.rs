use crate::model_config::ModelConfig;

#[derive(Clone, Copy, Debug)]
pub struct ScratchConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
}

impl ScratchConfig {
    pub fn from_model_config(config: &ModelConfig) -> Self {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size.unwrap_or(hidden_size * 4);
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads.unwrap_or(num_attention_heads);
        let head_dim = config.head_dim.unwrap_or(hidden_size / num_attention_heads);
        Self {
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
        }
    }
}

pub struct FfnWorkspace<'a> {
    pub gate: &'a mut [f32],
    pub up: &'a mut [f32],
    pub output: &'a mut [f32],
}

pub struct AttnWorkspace<'a> {
    pub q: &'a mut [f32],
    pub k: &'a mut [f32],
    pub v: &'a mut [f32],
    pub out: &'a mut [f32],
}

pub struct ScratchBuffer {
    config: ScratchConfig,
    ffn_gate: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_output: Vec<f32>,
    attn_q: Vec<f32>,
    attn_k: Vec<f32>,
    attn_v: Vec<f32>,
    attn_out: Vec<f32>,
}

impl ScratchBuffer {
    pub fn new(config: &ModelConfig, max_batch: usize, max_seq: usize) -> Self {
        Self::from_config(ScratchConfig::from_model_config(config), max_batch, max_seq)
    }

    pub fn from_config(config: ScratchConfig, max_batch: usize, max_seq: usize) -> Self {
        let ffn_len = max_batch * max_seq * config.intermediate_size;
        let ffn_output_len = max_batch * max_seq * config.hidden_size;
        let attn_q_len = max_batch * max_seq * config.num_attention_heads * config.head_dim;
        let attn_kv_len = max_batch * max_seq * config.num_key_value_heads * config.head_dim;

        Self {
            config,
            ffn_gate: vec![0.0f32; ffn_len],
            ffn_up: vec![0.0f32; ffn_len],
            ffn_output: vec![0.0f32; ffn_output_len],
            attn_q: vec![0.0f32; attn_q_len],
            attn_k: vec![0.0f32; attn_kv_len],
            attn_v: vec![0.0f32; attn_kv_len],
            attn_out: vec![0.0f32; attn_q_len],
        }
    }

    pub fn ffn_workspace(&mut self, batch: usize, seq: usize) -> FfnWorkspace<'_> {
        let ffn_len = batch * seq * self.config.intermediate_size;
        let out_len = batch * seq * self.config.hidden_size;
        let gate = ensure_len(&mut self.ffn_gate, ffn_len);
        let up = ensure_len(&mut self.ffn_up, ffn_len);
        let output = ensure_len(&mut self.ffn_output, out_len);
        FfnWorkspace { gate, up, output }
    }

    pub fn attn_workspace(&mut self, batch: usize, seq: usize) -> AttnWorkspace<'_> {
        let q_len = batch * seq * self.config.num_attention_heads * self.config.head_dim;
        let kv_len = batch * seq * self.config.num_key_value_heads * self.config.head_dim;
        let q = ensure_len(&mut self.attn_q, q_len);
        let k = ensure_len(&mut self.attn_k, kv_len);
        let v = ensure_len(&mut self.attn_v, kv_len);
        let out = ensure_len(&mut self.attn_out, q_len);
        AttnWorkspace { q, k, v, out }
    }
}

fn ensure_len(buffer: &mut Vec<f32>, len: usize) -> &mut [f32] {
    if buffer.len() < len {
        buffer.resize(len, 0.0);
    }
    &mut buffer[..len]
}
