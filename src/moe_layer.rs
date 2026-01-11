use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor};

#[derive(Clone)]
pub struct ExpertFFN<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> ExpertFFN<B> {
    pub fn new(device: &B::Device, hidden_size: usize, intermediate_size: usize) -> Self {
        let gate_proj = LinearConfig::new(hidden_size, intermediate_size).init(device);
        let up_proj = LinearConfig::new(hidden_size, intermediate_size).init(device);
        let down_proj = LinearConfig::new(intermediate_size, hidden_size).init(device);

        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

#[derive(Clone)]
pub struct MoERouter<B: Backend> {
    gate: Linear<B>,
    num_experts: usize,
    num_experts_per_tok: usize,
}

impl<B: Backend> MoERouter<B> {
    pub fn new(
        device: &B::Device,
        hidden_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
    ) -> Self {
        let gate = LinearConfig::new(hidden_size, num_experts).init(device);
        Self {
            gate,
            num_experts,
            num_experts_per_tok,
        }
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> (Tensor<B, 3, Int>, Tensor<B, 3>) {
        let logits = self.gate.forward(hidden_states);
        let (values, indices) = logits.topk_with_indices(self.num_experts_per_tok, 2);
        let weights = softmax(values, 2);
        (indices, weights)
    }
}

#[derive(Clone)]
pub struct MoELayer<B: Backend> {
    router: MoERouter<B>,
    experts: Vec<ExpertFFN<B>>,
    shared_expert: Option<ExpertFFN<B>>,
}

impl<B: Backend> MoELayer<B> {
    pub fn new(
        device: &B::Device,
        hidden_size: usize,
        intermediate_size: usize,
        num_experts: usize,
        num_experts_per_tok: usize,
        n_shared_experts: usize,
    ) -> Self {
        let router = MoERouter::new(device, hidden_size, num_experts, num_experts_per_tok);
        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            experts.push(ExpertFFN::new(device, hidden_size, intermediate_size));
        }
        let shared_expert = (n_shared_experts > 0)
            .then(|| ExpertFFN::new(device, hidden_size, intermediate_size));

        Self {
            router,
            experts,
            shared_expert,
        }
    }

    /// MoE forward with on-device routing/grouping and batched expert updates.
    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = hidden_states.device();
        let [batch_size, seq_len, hidden_size] = hidden_states.dims();
        let tokens = batch_size.saturating_mul(seq_len);
        let top_k = self.router.num_experts_per_tok;
        let assignments = tokens.saturating_mul(top_k);

        if assignments == 0 {
            return hidden_states;
        }

        let (expert_indices, expert_weights) = self.router.forward(hidden_states.clone());
        let expert_indices = expert_indices.reshape([assignments]);
        let expert_weights = expert_weights.reshape([assignments]);

        let token_indices = Tensor::<B, 1, Int>::arange(0..tokens as i64, &device)
            .unsqueeze_dim::<2>(1)
            .repeat(&[1, top_k])
            .reshape([assignments]);

        let (sorted_experts, sort_indices) = expert_indices.sort_with_indices(0);
        let sorted_tokens = token_indices.select(0, sort_indices.clone());
        let sorted_weights = expert_weights.select(0, sort_indices);

        let ones = Tensor::<B, 1, Int>::ones([assignments], &device);
        let counts = Tensor::<B, 1, Int>::zeros([self.router.num_experts], &device)
            .scatter(0, sorted_experts, ones);
        let offsets = counts.cumsum(0);
        let offsets_data = offsets
            .into_data()
            .into_vec::<B::IntElem>()
            .expect("MoE expert offsets should match backend int element type");

        let hidden_states_flat = hidden_states.clone().reshape([tokens, hidden_size]);
        let mut output = Tensor::<B, 2>::zeros([tokens, hidden_size], &device);

        let mut start = 0usize;
        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let end = offsets_data
                .get(expert_idx)
                .map(|v| v.elem::<i64>() as usize)
                .unwrap_or(start);
            if start != end {
                let token_slice = sorted_tokens.clone().slice([start..end]);
                let weight_slice = sorted_weights.clone().slice([start..end]);
                let selected = hidden_states_flat.clone().select(0, token_slice.clone());
                let selected = selected.reshape([1, end - start, hidden_size]);
                let expert_output = expert.forward(selected).reshape([end - start, hidden_size]);
                let weighted_output = expert_output * weight_slice.reshape([end - start, 1]);

                output = output.select_assign(0, token_slice, weighted_output);
            }
            start = end;
        }

        let mut output = output.reshape([batch_size, seq_len, hidden_size]);
        if let Some(shared_expert) = &self.shared_expert {
            output = output + shared_expert.forward(hidden_states);
        }

        output
    }
}
