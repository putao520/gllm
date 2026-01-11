use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};

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

    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = hidden_states.device();
        let [batch_size, seq_len, hidden_size] = hidden_states.dims();
        let tokens = batch_size.saturating_mul(seq_len);
        let top_k = self.router.num_experts_per_tok;

        let (expert_indices, expert_weights) = self.router.forward(hidden_states.clone());

        let indices_data = expert_indices
            .into_data()
            .into_vec::<B::IntElem>()
            .expect("MoE expert indices should match backend int element type");
        let weights_data = expert_weights
            .into_data()
            .into_vec::<B::FloatElem>()
            .expect("MoE expert weights should match backend float element type");

        let mut per_expert_indices: Vec<Vec<B::IntElem>> =
            vec![Vec::new(); self.router.num_experts];
        let mut per_expert_weights: Vec<Vec<B::FloatElem>> =
            vec![Vec::new(); self.router.num_experts];

        for token_idx in 0..tokens {
            let base = token_idx * top_k;
            for k in 0..top_k {
                let expert_idx = indices_data[base + k].elem::<i64>() as usize;
                if expert_idx < self.router.num_experts {
                    per_expert_indices[expert_idx]
                        .push(B::IntElem::from_elem(token_idx as i64));
                    per_expert_weights[expert_idx].push(weights_data[base + k]);
                }
            }
        }

        let hidden_states_flat = hidden_states.clone().reshape([tokens, hidden_size]);
        let mut output = Tensor::<B, 2>::zeros([tokens, hidden_size], &device);

        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let token_indices = &per_expert_indices[expert_idx];
            if token_indices.is_empty() {
                continue;
            }

            let num_tokens = token_indices.len();
            let indices_tensor = Tensor::<B, 1, Int>::from_data(
                TensorData::new(token_indices.clone(), [num_tokens]),
                &device,
            );

            let selected = hidden_states_flat.clone().select(0, indices_tensor.clone());
            let selected = selected.reshape([1, num_tokens, hidden_size]);
            let expert_output = expert.forward(selected).reshape([num_tokens, hidden_size]);

            let weights_tensor = Tensor::<B, 2>::from_data(
                TensorData::new(per_expert_weights[expert_idx].clone(), [num_tokens, 1]),
                &device,
            );
            let weighted_output = expert_output * weights_tensor;

            output = output.select_assign(0, indices_tensor, weighted_output);
        }

        let mut output = output.reshape([batch_size, seq_len, hidden_size]);
        if let Some(shared_expert) = &self.shared_expert {
            output = output + shared_expert.forward(hidden_states);
        }

        output
    }
}
