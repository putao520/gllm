//! MoE (Mixture of Experts) Layer implementation.
//!
//! Supports both CPU and GPU execution paths:
//! - CPU: Uses standard linear_forward operations
//! - GPU: Uses fused moe_forward_gpu kernel (single kernel launch for all experts)
//!
//! The fused GPU path requires packed expert weights for optimal performance.

use crate::tensor::Tensor3;
use crate::types::{Error, Result};
use crate::weight_loader::LinearWeights;
use gllm_kernels::gpu_types::{GpuTensor, TensorDtype};
use gllm_kernels::{
    linear_forward, moe_route, silu_inplace, BackendType, KernelDispatcher, LinearParams,
    MoEForwardConfig, MoERoutingConfig, MoERoutingGpuConfig, MoERoutingResult,
};

/// Expert FFN block (gate-up-down projections).
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

    /// CPU forward pass for a single token.
    #[inline]
    pub fn forward_single_cpu(&self, input: &[f32], output: &mut [f32], scratch: &mut [f32]) {
        let intermediate = self.gate_proj.weight.rows;
        let (gate, up) = scratch.split_at_mut(intermediate);

        // Gate projection
        linear_forward(
            input,
            self.gate_proj.weight.as_slice(),
            self.gate_proj.bias.as_ref().map(|b| b.as_slice()),
            gate,
            1,
            self.hidden_size,
            intermediate,
        );

        // Up projection
        linear_forward(
            input,
            self.up_proj.weight.as_slice(),
            self.up_proj.bias.as_ref().map(|b| b.as_slice()),
            up,
            1,
            self.hidden_size,
            intermediate,
        );

        // SiLU(gate) * up
        silu_inplace(gate);
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g *= u;
        }

        // Down projection
        linear_forward(
            gate,
            self.down_proj.weight.as_slice(),
            self.down_proj.bias.as_ref().map(|b| b.as_slice()),
            output,
            1,
            intermediate,
            self.hidden_size,
        );
    }

    /// CPU forward pass with weighted accumulation to residual.
    #[inline]
    pub fn forward_single_cpu_weighted(
        &self,
        input: &[f32],
        residual: &mut [f32],
        scratch: &mut [f32],
        weight: f32,
    ) {
        let intermediate = self.gate_proj.weight.rows;
        let (gate, rest) = scratch.split_at_mut(intermediate);
        let (up, output) = rest.split_at_mut(intermediate);

        // Gate projection
        linear_forward(
            input,
            self.gate_proj.weight.as_slice(),
            self.gate_proj.bias.as_ref().map(|b| b.as_slice()),
            gate,
            1,
            self.hidden_size,
            intermediate,
        );

        // Up projection
        linear_forward(
            input,
            self.up_proj.weight.as_slice(),
            self.up_proj.bias.as_ref().map(|b| b.as_slice()),
            up,
            1,
            self.hidden_size,
            intermediate,
        );

        // SiLU(gate) * up
        silu_inplace(gate);
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g *= u;
        }

        // Down projection to temp output
        linear_forward(
            gate,
            self.down_proj.weight.as_slice(),
            self.down_proj.bias.as_ref().map(|b| b.as_slice()),
            &mut output[..self.hidden_size],
            1,
            intermediate,
            self.hidden_size,
        );

        // Weighted accumulation
        for (r, o) in residual.iter_mut().zip(output[..self.hidden_size].iter()) {
            *r += o * weight;
        }
    }

    pub fn intermediate_size(&self) -> usize {
        self.gate_proj.weight.rows
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Check if GPU weights are available.
    pub fn has_gpu_weights(&self) -> bool {
        self.gate_proj.weight.gpu_tensor.is_some()
            && self.up_proj.weight.gpu_tensor.is_some()
            && self.down_proj.weight.gpu_tensor.is_some()
    }

    /// Get GPU weight references for kernel dispatch.
    pub fn gpu_weights(&self) -> Option<(&GpuTensor, &GpuTensor, &GpuTensor)> {
        match (
            &self.gate_proj.weight.gpu_tensor,
            &self.up_proj.weight.gpu_tensor,
            &self.down_proj.weight.gpu_tensor,
        ) {
            (Some(g), Some(u), Some(d)) => Some((g, u, d)),
            _ => None,
        }
    }
}

/// MoE router for selecting top-k experts per token.
#[derive(Clone)]
pub struct MoERouter {
    gate_weights: Vec<f32>,
    gpu_gate_weights: Option<GpuTensor>,
    num_experts: usize,
    num_experts_per_tok: usize,
    hidden_size: usize,
}

impl MoERouter {
    pub fn new(hidden_size: usize, num_experts: usize, num_experts_per_tok: usize) -> Self {
        Self {
            gate_weights: vec![0.0f32; hidden_size * num_experts],
            gpu_gate_weights: None,
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
        // Transpose for efficient routing: [hidden, experts]
        let mut transposed = vec![0.0f32; self.hidden_size * self.num_experts];
        for out in 0..gate.weight.rows {
            for inp in 0..gate.weight.cols {
                let src = out * gate.weight.cols + inp;
                let dst = inp * self.num_experts + out;
                transposed[dst] = gate.weight.data[src];
            }
        }
        self.gate_weights = transposed;
        self.gpu_gate_weights = None;
        let dispatcher = KernelDispatcher::new();
        if dispatcher.backend() != BackendType::Cpu {
            if let Err(err) = self.init_gpu_gate_weights(dispatcher.backend_dispatched()) {
                log::warn!("Failed to upload MoE gate weights to GPU: {err}");
            }
        }
        Ok(())
    }

    /// Compute routing for given hidden states.
    #[inline]
    pub fn route(&self, hidden: &[f32], batch: usize, seq_len: usize) -> MoERoutingResult {
        let config = MoERoutingConfig {
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            hidden_size: self.hidden_size,
        };
        moe_route(hidden, &self.gate_weights, batch, seq_len, &config)
    }

    pub fn init_gpu_gate_weights(
        &mut self,
        backend: &gllm_kernels::DispatchedBackend,
    ) -> Result<()> {
        if backend.backend_type() == BackendType::Cpu {
            self.gpu_gate_weights = None;
            return Ok(());
        }

        let bytes = f32_slice_to_bytes(&self.gate_weights);
        let gate_gpu = backend
            .allocate_weights(&bytes, vec![self.hidden_size, self.num_experts], TensorDtype::F32)
            .map_err(Error::InferenceError)?;
        self.gpu_gate_weights = Some(gate_gpu);
        Ok(())
    }

    pub fn route_gpu(
        &self,
        hidden_states: &GpuTensor,
        expert_indices: &mut GpuTensor,
        expert_weights: &mut GpuTensor,
        backend: &gllm_kernels::DispatchedBackend,
    ) -> Result<()> {
        let gate_gpu = self.gpu_gate_weights.as_ref().ok_or_else(|| {
            Error::InferenceError("GPU gate weights not initialized".into())
        })?;
        if gate_gpu.backend != hidden_states.backend {
            return Err(Error::InferenceError(
                "MoE gate weights backend mismatch".into(),
            ));
        }

        let config = MoERoutingGpuConfig {
            num_tokens: 1,
            hidden_size: self.hidden_size,
            num_experts: self.num_experts,
            top_k: self.num_experts_per_tok,
        };
        backend
            .moe_route_gpu(hidden_states, gate_gpu, expert_indices, expert_weights, config)
            .map_err(Error::InferenceError)
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn num_experts_per_tok(&self) -> usize {
        self.num_experts_per_tok
    }
}

/// MoE Layer combining router and experts.
#[derive(Clone)]
pub struct MoELayer {
    pub(crate) router: MoERouter,
    pub(crate) experts: Vec<ExpertFFN>,
    pub(crate) shared_expert: Option<ExpertFFN>,
    hidden_size: usize,
    intermediate_size: usize,
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
        let shared_expert =
            (n_shared_experts > 0).then(|| ExpertFFN::new(hidden_size, intermediate_size));

        Self {
            router,
            experts,
            shared_expert,
            hidden_size,
            intermediate_size,
        }
    }

    /// CPU forward pass using Tensor3 (legacy interface).
    pub fn forward(&self, hidden_states: &Tensor3) -> Result<Tensor3> {
        let (batch, seq_len, hidden) = hidden_states.shape();
        if hidden != self.hidden_size {
            return Err(Error::InferenceError("MoE hidden size mismatch".into()));
        }

        let mut output = vec![0.0f32; hidden_states.data.len()];
        let mut scratch = vec![0.0f32; self.intermediate_size * 2 + self.hidden_size];

        let routing = self.router.route(&hidden_states.data, batch, seq_len);
        self.apply_experts_cpu(&hidden_states.data, &routing, &mut output, &mut scratch)?;

        if let Some(shared) = &self.shared_expert {
            for token in 0..(batch * seq_len) {
                let start = token * self.hidden_size;
                let input = &hidden_states.data[start..start + self.hidden_size];
                let out = &mut output[start..start + self.hidden_size];
                let mut temp = vec![0.0f32; self.hidden_size];
                shared.forward_single_cpu(input, &mut temp, &mut scratch);
                for (o, t) in out.iter_mut().zip(temp.iter()) {
                    *o += t;
                }
            }
        }

        Tensor3::new(output, batch, seq_len, hidden)
    }

    /// CPU inplace forward: input is normed FFN input, output accumulated to residual.
    pub fn forward_inplace_cpu(
        &self,
        input: &[f32],
        residual: &mut [f32],
        scratch: &mut MoEScratchCpu,
    ) -> Result<()> {
        // Route tokens to experts
        let routing = self.router.route(input, 1, 1);

        // Apply routed experts with weighted accumulation
        let top_k = routing.top_k;
        for k in 0..top_k {
            let expert_idx = routing.expert_indices.get(k).copied().unwrap_or(0) as usize;
            if expert_idx >= self.experts.len() {
                continue;
            }
            let weight = routing.expert_weights.get(k).copied().unwrap_or(0.0);
            self.experts[expert_idx].forward_single_cpu_weighted(
                input,
                residual,
                &mut scratch.expert_scratch,
                weight,
            );
        }

        // Shared expert (no routing weight, direct add)
        if let Some(shared) = &self.shared_expert {
            shared.forward_single_cpu_weighted(input, residual, &mut scratch.expert_scratch, 1.0);
        }

        Ok(())
    }

    /// GPU inplace forward: operates on GPU tensors, routing computed from readback.
    /// This method requires the backend to support MoE GPU operations.
    pub fn forward_inplace_gpu(
        &self,
        normed_input: &GpuTensor,
        residual: &mut GpuTensor,
        scratch: &mut MoEScratchGpu,
        backend: &gllm_kernels::DispatchedBackend,
    ) -> Result<()> {
        let routing = self.route_gpu_or_cpu(normed_input, scratch, backend)?;

        // Zero the MoE output accumulator
        backend
            .tensor_zero_gpu(&mut scratch.moe_output)
            .map_err(Error::InferenceError)?;

        // For each selected expert, run FFN on GPU and accumulate
        let top_k = routing.top_k;
        for k in 0..top_k {
            let expert_idx = routing.expert_indices.get(k).copied().unwrap_or(0) as usize;
            if expert_idx >= self.experts.len() {
                continue;
            }
            let weight = routing.expert_weights.get(k).copied().unwrap_or(0.0);

            let expert = &self.experts[expert_idx];
            let (gate_gpu, up_gpu, down_gpu) = expert.gpu_weights().ok_or_else(|| {
                Error::InferenceError(format!("Expert {} missing GPU weights", expert_idx))
            })?;

            let gate_up_params = LinearParams {
                in_features: self.hidden_size as u32,
                out_features: self.intermediate_size as u32,
                has_bias: 0,
                padding: 0,
            };
            let down_params = LinearParams {
                in_features: self.intermediate_size as u32,
                out_features: self.hidden_size as u32,
                has_bias: 0,
                padding: 0,
            };

            // Run expert FFN: normed_input -> expert_output
            backend
                .ffn_forward_gpu(
                    normed_input,
                    gate_gpu,
                    up_gpu,
                    down_gpu,
                    &mut scratch.intermediate,
                    &mut scratch.expert_output,
                    gate_up_params,
                    down_params,
                )
                .map_err(Error::InferenceError)?;

            // Scale and accumulate: moe_output += expert_output * weight
            backend
                .tensor_scale_add_gpu(&scratch.expert_output, &mut scratch.moe_output, 0, weight)
                .map_err(Error::InferenceError)?;
        }

        // Handle shared expert
        if let Some(shared) = &self.shared_expert {
            let (gate_gpu, up_gpu, down_gpu) = shared.gpu_weights().ok_or_else(|| {
                Error::InferenceError("Shared expert missing GPU weights".into())
            })?;

            let gate_up_params = LinearParams {
                in_features: self.hidden_size as u32,
                out_features: self.intermediate_size as u32,
                has_bias: 0,
                padding: 0,
            };
            let down_params = LinearParams {
                in_features: self.intermediate_size as u32,
                out_features: self.hidden_size as u32,
                has_bias: 0,
                padding: 0,
            };

            backend
                .ffn_forward_gpu(
                    normed_input,
                    gate_gpu,
                    up_gpu,
                    down_gpu,
                    &mut scratch.intermediate,
                    &mut scratch.expert_output,
                    gate_up_params,
                    down_params,
                )
                .map_err(Error::InferenceError)?;

            // Add shared expert output (weight = 1.0)
            backend
                .tensor_add_gpu(&mut scratch.moe_output, &scratch.expert_output)
                .map_err(Error::InferenceError)?;
        }

        // Add MoE output to residual
        backend
            .tensor_add_gpu(residual, &scratch.moe_output)
            .map_err(Error::InferenceError)?;

        Ok(())
    }

    /// Fused GPU forward: single kernel launch for all experts (optimal performance).
    ///
    /// This method requires pre-packed expert weights for maximum efficiency.
    /// Use this when weights are already packed via `PackedExpertWeights::pack()`.
    pub fn forward_inplace_gpu_fused(
        &self,
        normed_input: &GpuTensor,
        residual: &mut GpuTensor,
        packed_weights: &PackedExpertWeights,
        scratch: &mut MoEScratchGpu,
        backend: &gllm_kernels::DispatchedBackend,
    ) -> Result<()> {
        let top_k = self.router.num_experts_per_tok();
        scratch.ensure_routing_buffers(top_k, normed_input.backend)?;
        self.router
            .route_gpu(
                normed_input,
                &mut scratch.expert_indices_gpu,
                &mut scratch.expert_weights_gpu,
                backend,
            )?;

        // Create MoE config
        let config = MoEForwardConfig::new(
            packed_weights.hidden_size,
            packed_weights.intermediate_size,
            1, // num_tokens (single token for now)
            top_k,
            packed_weights.num_experts,
        );

        // Fused MoE forward: single kernel launch for all selected experts
        backend
            .moe_forward_gpu_pure(
                normed_input,
                &scratch.expert_indices_gpu,
                &scratch.expert_weights_gpu,
                &packed_weights.all_gate,
                &packed_weights.all_up,
                &packed_weights.all_down,
                &mut scratch.moe_output,
                config,
            )
            .map_err(Error::InferenceError)?;

        // Handle shared expert (if any) - still uses separate kernel
        if let Some(shared) = &self.shared_expert {
            let (gate_gpu, up_gpu, down_gpu) = shared.gpu_weights().ok_or_else(|| {
                Error::InferenceError("Shared expert missing GPU weights".into())
            })?;

            let gate_up_params = LinearParams {
                in_features: self.hidden_size as u32,
                out_features: self.intermediate_size as u32,
                has_bias: 0,
                padding: 0,
            };
            let down_params = LinearParams {
                in_features: self.intermediate_size as u32,
                out_features: self.hidden_size as u32,
                has_bias: 0,
                padding: 0,
            };

            backend
                .ffn_forward_gpu(
                    normed_input,
                    gate_gpu,
                    up_gpu,
                    down_gpu,
                    &mut scratch.intermediate,
                    &mut scratch.expert_output,
                    gate_up_params,
                    down_params,
                )
                .map_err(Error::InferenceError)?;

            // Add shared expert output (weight = 1.0)
            backend
                .tensor_add_gpu(&mut scratch.moe_output, &scratch.expert_output)
                .map_err(Error::InferenceError)?;
        }

        // Add MoE output to residual
        backend
            .tensor_add_gpu(residual, &scratch.moe_output)
            .map_err(Error::InferenceError)?;

        Ok(())
    }

    /// Pack expert weights for fused MoE execution.
    pub fn pack_weights(
        &self,
        backend: BackendType,
        dispatcher: &gllm_kernels::DispatchedBackend,
    ) -> Result<PackedExpertWeights> {
        PackedExpertWeights::pack(&self.experts, backend, dispatcher)
    }

    fn route_gpu_or_cpu(
        &self,
        normed_input: &GpuTensor,
        scratch: &mut MoEScratchGpu,
        backend: &gllm_kernels::DispatchedBackend,
    ) -> Result<MoERoutingResult> {
        if self.router.gpu_gate_weights.is_some() {
            match self.route_on_gpu(normed_input, scratch, backend) {
                Ok(routing) => return Ok(routing),
                Err(err) => {
                    log::warn!("GPU MoE routing failed, falling back to CPU: {err}");
                }
            }
        }
        self.route_on_cpu(normed_input, scratch, backend)
    }

    fn route_on_gpu(
        &self,
        normed_input: &GpuTensor,
        scratch: &mut MoEScratchGpu,
        backend: &gllm_kernels::DispatchedBackend,
    ) -> Result<MoERoutingResult> {
        let top_k = self.router.num_experts_per_tok();
        scratch.ensure_routing_buffers(top_k, normed_input.backend)?;
        self.router.route_gpu(
            normed_input,
            &mut scratch.expert_indices_gpu,
            &mut scratch.expert_weights_gpu,
            backend,
        )?;

        let mut expert_indices = vec![0u32; top_k];
        backend
            .readback_u32(&scratch.expert_indices_gpu, &mut expert_indices)
            .map_err(Error::InferenceError)?;

        let mut expert_weights = vec![0.0f32; top_k];
        backend
            .readback(&scratch.expert_weights_gpu, &mut expert_weights)
            .map_err(Error::InferenceError)?;

        Ok(MoERoutingResult {
            expert_indices,
            expert_weights,
            num_tokens: 1,
            top_k,
        })
    }

    fn route_on_cpu(
        &self,
        normed_input: &GpuTensor,
        scratch: &mut MoEScratchGpu,
        backend: &gllm_kernels::DispatchedBackend,
    ) -> Result<MoERoutingResult> {
        let total_elements = normed_input.shape.iter().product::<usize>();
        scratch.routing_buffer.resize(total_elements, 0.0);
        backend
            .linear_forward_host_io_readback(normed_input, &mut scratch.routing_buffer)
            .map_err(Error::InferenceError)?;
        Ok(self.router.route(&scratch.routing_buffer, 1, 1))
    }

    fn apply_experts_cpu(
        &self,
        hidden: &[f32],
        routing: &MoERoutingResult,
        output: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<()> {
        let top_k = routing.top_k;
        let token_count = routing.num_tokens;
        if token_count == 0 || top_k == 0 {
            return Ok(());
        }

        for token in 0..token_count {
            let token_start = token * self.hidden_size;
            let token_input = &hidden[token_start..token_start + self.hidden_size];
            let token_output = &mut output[token_start..token_start + self.hidden_size];

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
                self.experts[expert_idx].forward_single_cpu_weighted(
                    token_input,
                    token_output,
                    scratch,
                    weight,
                );
            }
        }
        Ok(())
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Check if all experts have GPU weights.
    pub fn has_gpu_weights(&self) -> bool {
        self.experts.iter().all(|e| e.has_gpu_weights())
            && self
                .shared_expert
                .as_ref()
                .map(|e| e.has_gpu_weights())
                .unwrap_or(true)
    }
}

/// CPU scratch buffers for MoE computation.
pub struct MoEScratchCpu {
    pub expert_scratch: Vec<f32>,
}

impl MoEScratchCpu {
    pub fn new(intermediate_size: usize, hidden_size: usize) -> Self {
        // Need space for: gate, up, output
        Self {
            expert_scratch: vec![0.0f32; intermediate_size * 2 + hidden_size],
        }
    }
}

/// GPU scratch buffers for MoE computation.
pub struct MoEScratchGpu {
    pub routing_buffer: Vec<f32>,
    pub expert_indices_gpu: GpuTensor,
    pub expert_weights_gpu: GpuTensor,
    pub intermediate: GpuTensor,
    pub expert_output: GpuTensor,
    pub moe_output: GpuTensor,
}

impl MoEScratchGpu {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        backend: BackendType,
    ) -> Result<Self> {
        Self::new_with_routing(hidden_size, intermediate_size, 1, backend)
    }

    pub fn new_with_routing(
        hidden_size: usize,
        intermediate_size: usize,
        num_experts_per_tok: usize,
        backend: BackendType,
    ) -> Result<Self> {
        let intermediate = GpuTensor::new_temp(vec![1, intermediate_size], TensorDtype::F32, backend)
            .map_err(Error::InferenceError)?;
        let expert_output = GpuTensor::new_temp(vec![1, hidden_size], TensorDtype::F32, backend)
            .map_err(Error::InferenceError)?;
        let moe_output = GpuTensor::new_temp(vec![1, hidden_size], TensorDtype::F32, backend)
            .map_err(Error::InferenceError)?;
        let expert_indices_gpu = GpuTensor::new_temp(
            vec![1, num_experts_per_tok],
            TensorDtype::U32,
            backend,
        )
        .map_err(Error::InferenceError)?;
        let expert_weights_gpu = GpuTensor::new_temp(
            vec![1, num_experts_per_tok],
            TensorDtype::F32,
            backend,
        )
        .map_err(Error::InferenceError)?;

        Ok(Self {
            routing_buffer: Vec::new(),
            expert_indices_gpu,
            expert_weights_gpu,
            intermediate,
            expert_output,
            moe_output,
        })
    }

    pub fn ensure_routing_buffers(
        &mut self,
        num_experts_per_tok: usize,
        backend: BackendType,
    ) -> Result<()> {
        if num_experts_per_tok == 0 {
            return Err(Error::InferenceError(
                "MoE routing top_k must be > 0".into(),
            ));
        }

        if self.expert_indices_gpu.len() < num_experts_per_tok
            || self.expert_indices_gpu.backend != backend
        {
            let new_tensor = GpuTensor::new_temp(
                vec![1, num_experts_per_tok],
                TensorDtype::U32,
                backend,
            )
            .map_err(Error::InferenceError)?;
            let old_tensor = std::mem::replace(&mut self.expert_indices_gpu, new_tensor);
            old_tensor.release();
        }
        if self.expert_weights_gpu.len() < num_experts_per_tok
            || self.expert_weights_gpu.backend != backend
        {
            let new_tensor = GpuTensor::new_temp(
                vec![1, num_experts_per_tok],
                TensorDtype::F32,
                backend,
            )
            .map_err(Error::InferenceError)?;
            let old_tensor = std::mem::replace(&mut self.expert_weights_gpu, new_tensor);
            old_tensor.release();
        }

        Ok(())
    }

    pub fn release(self) {
        self.expert_indices_gpu.release();
        self.expert_weights_gpu.release();
        self.intermediate.release();
        self.expert_output.release();
        self.moe_output.release();
    }
}

/// Packed expert weights for fused MoE GPU execution.
///
/// This structure holds all expert weights packed into contiguous GPU tensors,
/// enabling single-kernel-launch MoE computation.
pub struct PackedExpertWeights {
    /// All experts' gate weights [num_experts, intermediate, hidden]
    pub all_gate: GpuTensor,
    /// All experts' up weights [num_experts, intermediate, hidden]
    pub all_up: GpuTensor,
    /// All experts' down weights [num_experts, hidden, intermediate]
    pub all_down: GpuTensor,
    /// Number of experts
    pub num_experts: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Intermediate size
    pub intermediate_size: usize,
}

impl PackedExpertWeights {
    /// Pack individual expert weights into contiguous GPU tensors.
    pub fn pack(
        experts: &[ExpertFFN],
        backend: BackendType,
        dispatcher: &gllm_kernels::DispatchedBackend,
    ) -> Result<Self> {
        if experts.is_empty() {
            return Err(Error::InferenceError("No experts to pack".into()));
        }

        let num_experts = experts.len();
        let hidden_size = experts[0].hidden_size();
        let intermediate_size = experts[0].intermediate_size();

        // Pack gate weights: [num_experts, intermediate, hidden]
        let gate_size = num_experts * intermediate_size * hidden_size;
        let mut gate_data = vec![0.0f32; gate_size];
        for (e, expert) in experts.iter().enumerate() {
            let offset = e * intermediate_size * hidden_size;
            let src = expert.gate_proj.weight.as_slice();
            gate_data[offset..offset + src.len()].copy_from_slice(src);
        }

        // Pack up weights: [num_experts, intermediate, hidden]
        let up_size = num_experts * intermediate_size * hidden_size;
        let mut up_data = vec![0.0f32; up_size];
        for (e, expert) in experts.iter().enumerate() {
            let offset = e * intermediate_size * hidden_size;
            let src = expert.up_proj.weight.as_slice();
            up_data[offset..offset + src.len()].copy_from_slice(src);
        }

        // Pack down weights: [num_experts, hidden, intermediate]
        let down_size = num_experts * hidden_size * intermediate_size;
        let mut down_data = vec![0.0f32; down_size];
        for (e, expert) in experts.iter().enumerate() {
            let offset = e * hidden_size * intermediate_size;
            let src = expert.down_proj.weight.as_slice();
            down_data[offset..offset + src.len()].copy_from_slice(src);
        }

        // Upload to GPU
        let all_gate = Self::upload_weights(&gate_data, vec![num_experts, intermediate_size, hidden_size], backend, dispatcher)?;
        let all_up = Self::upload_weights(&up_data, vec![num_experts, intermediate_size, hidden_size], backend, dispatcher)?;
        let all_down = Self::upload_weights(&down_data, vec![num_experts, hidden_size, intermediate_size], backend, dispatcher)?;

        Ok(Self {
            all_gate,
            all_up,
            all_down,
            num_experts,
            hidden_size,
            intermediate_size,
        })
    }

    fn upload_weights(
        data: &[f32],
        shape: Vec<usize>,
        _backend: BackendType,
        dispatcher: &gllm_kernels::DispatchedBackend,
    ) -> Result<GpuTensor> {
        let bytes = f32_slice_to_bytes(data);
        dispatcher.allocate_weights(&bytes, shape, TensorDtype::F32)
            .map_err(Error::InferenceError)
    }

    pub fn release(self) {
        self.all_gate.release();
        self.all_up.release();
        self.all_down.release();
    }
}

fn f32_slice_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|&f| f.to_le_bytes()).collect()
}
