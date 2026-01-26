//! Mixture of Experts (MoE) Layer using gllm-kernels.

use crate::types::Result;
use crate::weight_loader::LinearWeights;
use gllm_kernels::backend::{Backend, TensorSlice};
use gllm_kernels::{moe_route, silu_inplace, MoERoutingConfig};
use std::sync::Arc;

/// Expert FFN weights.
#[derive(Clone)]
pub struct ExpertWeights {
    pub gate_proj: LinearWeights,
    pub up_proj: LinearWeights,
    pub down_proj: LinearWeights,
}

impl ExpertWeights {
    pub fn zeros(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: LinearWeights::zeros(intermediate_size, hidden_size),
            up_proj: LinearWeights::zeros(intermediate_size, hidden_size),
            down_proj: LinearWeights::zeros(hidden_size, intermediate_size),
        }
    }
}

/// MoE Layer with sparse routing.
pub struct MoELayer {
    pub gate_weights: Vec<f32>,  // [num_experts, hidden_size]
    pub experts: Vec<ExpertWeights>,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    backend: Arc<dyn Backend>,
}

impl MoELayer {
    pub fn new(
        num_experts: usize,
        num_experts_per_tok: usize,
        hidden_size: usize,
        intermediate_size: usize,
        backend: Arc<dyn Backend>,
    ) -> Self {
        let experts = (0..num_experts)
            .map(|_| ExpertWeights::zeros(hidden_size, intermediate_size))
            .collect();

        Self {
            gate_weights: vec![0.0; num_experts * hidden_size],
            experts,
            num_experts,
            num_experts_per_tok,
            hidden_size,
            intermediate_size,
            backend,
        }
    }

    /// Forward pass through MoE layer.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        let batch_seq = input.len() / self.hidden_size;

        // Route tokens to experts
        let routing_config = MoERoutingConfig {
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            hidden_size: self.hidden_size,
        };

        let routing_result = self
            .backend
            .moe_route(
                TensorSlice::F32(input),
                &self.gate_weights,
                1,
                batch_seq,
                &routing_config,
            )
            .unwrap_or_else(|_| {
                // Fallback: use CPU routing
                moe_route(input, &self.gate_weights, 1, batch_seq, &routing_config)
            });

        // Process each token through its assigned experts
        let mut output = vec![0.0f32; batch_seq * self.hidden_size];

        for token_idx in 0..batch_seq {
            let token_input = &input[token_idx * self.hidden_size..(token_idx + 1) * self.hidden_size];
            let mut token_output = vec![0.0f32; self.hidden_size];

            for k in 0..self.num_experts_per_tok {
                let expert_idx = routing_result.expert_indices[token_idx * self.num_experts_per_tok + k] as usize;
                let weight = routing_result.expert_weights[token_idx * self.num_experts_per_tok + k];

                if expert_idx < self.num_experts {
                    let expert_output = self.expert_forward(expert_idx, token_input)?;
                    for (o, e) in token_output.iter_mut().zip(expert_output.iter()) {
                        *o += weight * e;
                    }
                }
            }

            output[token_idx * self.hidden_size..(token_idx + 1) * self.hidden_size]
                .copy_from_slice(&token_output);
        }

        Ok(output)
    }

    fn expert_forward(&self, expert_idx: usize, input: &[f32]) -> Result<Vec<f32>> {
        let expert = &self.experts[expert_idx];

        // Gate projection
        let mut gate = vec![0.0f32; self.intermediate_size];
        expert
            .gate_proj
            .forward(input, &mut gate, 1, self.backend.as_ref())?;

        // Up projection
        let mut up = vec![0.0f32; self.intermediate_size];
        expert
            .up_proj
            .forward(input, &mut up, 1, self.backend.as_ref())?;

        // SiLU activation on gate
        silu_inplace(&mut gate);

        // Element-wise multiply
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g *= u;
        }

        // Down projection
        let mut output = vec![0.0f32; self.hidden_size];
        expert
            .down_proj
            .forward(&gate, &mut output, 1, self.backend.as_ref())?;

        Ok(output)
    }

    /// Access expert weights for loading.
    pub fn expert_mut(&mut self, idx: usize) -> Option<&mut ExpertWeights> {
        self.experts.get_mut(idx)
    }

    /// Access gate weights for loading.
    pub fn gate_weights_mut(&mut self) -> &mut [f32] {
        &mut self.gate_weights
    }
}
