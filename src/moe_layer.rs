use crate::tensor::Tensor3;
use crate::types::{Error, Result};
use crate::weight_loader::LinearWeights;
use gllm_kernels::{linear_forward, moe_route, MoERoutingConfig, MoERoutingResult};
use gllm_kernels::silu_inplace;

#[derive(Clone)]
pub struct ExpertFFN {
    pub(crate) gate_proj: LinearWeights,
    pub(crate) up_proj: LinearWeights,
    pub(crate) down_proj: LinearWeights,
    hidden_size: usize,
}

impl ExpertFFN {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: LinearWeights::zeros(intermediate_size, hidden_size),
            up_proj: LinearWeights::zeros(intermediate_size, hidden_size),
            down_proj: LinearWeights::zeros(hidden_size, intermediate_size),
            hidden_size,
        }
    }

    pub fn forward_flat(&self, input: &[f32], rows: usize) -> Result<Vec<f32>> {
        if input.len() != rows * self.hidden_size {
            return Err(Error::InferenceError(
                "ExpertFFN input length mismatch".into(),
            ));
        }
        let mut gate = vec![0.0f32; rows * self.gate_proj.weight.rows];
        linear_forward(
            input,
            self.gate_proj.weight.as_slice(),
            self.gate_proj.bias.as_ref().map(|b| b.as_slice()),
            &mut gate,
            rows,
            self.gate_proj.weight.cols,
            self.gate_proj.weight.rows,
        );

        let mut up = vec![0.0f32; rows * self.up_proj.weight.rows];
        linear_forward(
            input,
            self.up_proj.weight.as_slice(),
            self.up_proj.bias.as_ref().map(|b| b.as_slice()),
            &mut up,
            rows,
            self.up_proj.weight.cols,
            self.up_proj.weight.rows,
        );

        if gate.len() != up.len() {
            return Err(Error::InferenceError(
                "ExpertFFN gate/up size mismatch".into(),
            ));
        }

        silu_inplace(&mut gate);
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g *= u;
        }

        let mut down = vec![0.0f32; rows * self.down_proj.weight.rows];
        linear_forward(
            &gate,
            self.down_proj.weight.as_slice(),
            self.down_proj.bias.as_ref().map(|b| b.as_slice()),
            &mut down,
            rows,
            self.down_proj.weight.cols,
            self.down_proj.weight.rows,
        );
        Ok(down)
    }
}

#[derive(Clone)]
pub struct MoERouter {
    gate_weights: Vec<f32>,
    num_experts: usize,
    num_experts_per_tok: usize,
    hidden_size: usize,
}

impl MoERouter {
    pub fn new(hidden_size: usize, num_experts: usize, num_experts_per_tok: usize) -> Self {
        Self {
            gate_weights: vec![0.0f32; hidden_size * num_experts],
            num_experts,
            num_experts_per_tok,
            hidden_size,
        }
    }

    pub fn set_gate(&mut self, gate: LinearWeights) -> Result<()> {
        if gate.weight.rows != self.num_experts || gate.weight.cols != self.hidden_size {
            return Err(Error::InferenceError(
                "MoE gate weight shape mismatch".into(),
            ));
        }
        let mut transposed = vec![0.0f32; self.hidden_size * self.num_experts];
        for out in 0..gate.weight.rows {
            for inp in 0..gate.weight.cols {
                let src = out * gate.weight.cols + inp;
                let dst = inp * self.num_experts + out;
                transposed[dst] = gate.weight.data[src];
            }
        }
        self.gate_weights = transposed;
        Ok(())
    }

    pub fn route(&self, hidden: &[f32], batch: usize, seq_len: usize) -> MoERoutingResult {
        let config = MoERoutingConfig {
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            hidden_size: self.hidden_size,
        };
        moe_route(hidden, &self.gate_weights, batch, seq_len, &config)
    }
}

#[derive(Clone)]
pub struct MoELayer {
    pub(crate) router: MoERouter,
    pub(crate) experts: Vec<ExpertFFN>,
    pub(crate) shared_expert: Option<ExpertFFN>,
    hidden_size: usize,
}

impl MoELayer {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        n_shared_experts: usize,
    ) -> Self {
        let router = MoERouter::new(hidden_size, num_experts, num_experts_per_tok);
        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            experts.push(ExpertFFN::new(hidden_size, intermediate_size));
        }
        let shared_expert = (n_shared_experts > 0)
            .then(|| ExpertFFN::new(hidden_size, intermediate_size));

        Self {
            router,
            experts,
            shared_expert,
            hidden_size,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor3) -> Result<Tensor3> {
        let (batch, seq_len, hidden) = hidden_states.shape();
        if hidden != self.hidden_size {
            return Err(Error::InferenceError(
                "MoE hidden size mismatch".into(),
            ));
        }
        let routing = self.router.route(&hidden_states.data, batch, seq_len);
        let mut output = vec![0.0f32; hidden_states.data.len()];
        self.apply_experts(&hidden_states.data, &routing, &mut output)?;

        if let Some(shared) = &self.shared_expert {
            let shared_out = shared.forward_flat(&hidden_states.data, batch * seq_len)?;
            for (dst, src) in output.iter_mut().zip(shared_out.iter()) {
                *dst += src;
            }
        }

        Tensor3::new(output, batch, seq_len, hidden)
    }

    fn apply_experts(
        &self,
        hidden: &[f32],
        routing: &MoERoutingResult,
        output: &mut [f32],
    ) -> Result<()> {
        let top_k = routing.top_k;
        let token_count = routing.num_tokens;
        if token_count == 0 || top_k == 0 {
            return Ok(());
        }

        for token in 0..token_count {
            let token_start = token * self.hidden_size;
            let token_input = &hidden[token_start..token_start + self.hidden_size];
            for k in 0..top_k {
                let idx = token * top_k + k;
                let expert_idx = routing
                    .expert_indices
                    .get(idx)
                    .copied()
                    .unwrap_or(0) as usize;
                if expert_idx >= self.experts.len() {
                    continue;
                }
                let weight = routing.expert_weights.get(idx).copied().unwrap_or(0.0);
                let expert_out = self.experts[expert_idx].forward_flat(token_input, 1)?;
                for i in 0..self.hidden_size {
                    output[token_start + i] += expert_out[i] * weight;
                }
            }
        }
        Ok(())
    }
}
