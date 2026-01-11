use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::cmp::Ordering;

pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

pub fn sample_next_token<B: Backend>(
    logits: Tensor<B, 2>,
    config: &SamplingConfig,
    device: &B::Device,
) -> Vec<i64> {
    let _ = device;
    let [batch_size, vocab_size] = logits.dims();
    let data = match logits.into_data().into_vec::<f32>() {
        Ok(data) => data,
        Err(_) => return vec![0; batch_size],
    };

    let mut rng = thread_rng();
    let mut outputs = Vec::with_capacity(batch_size);

    for batch in 0..batch_size {
        let start = batch * vocab_size;
        let end = start + vocab_size;
        let row = &data[start..end];

        if config.temperature <= 0.0 {
            let mut best = 0usize;
            let mut best_logit = f32::NEG_INFINITY;
            for (idx, &value) in row.iter().enumerate() {
                if value > best_logit {
                    best_logit = value;
                    best = idx;
                }
            }
            outputs.push(best as i64);
            continue;
        }

        let mut candidates: Vec<(usize, f32)> = row
            .iter()
            .enumerate()
            .map(|(idx, &value)| (idx, value / config.temperature))
            .collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        if config.top_k > 0 && candidates.len() > config.top_k {
            candidates.truncate(config.top_k);
        }

        let max_logit = candidates
            .iter()
            .map(|(_, logit)| *logit)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut weights: Vec<f32> = candidates
            .iter()
            .map(|(_, logit)| (logit - max_logit).exp())
            .collect();

        if config.top_p > 0.0 && config.top_p < 1.0 {
            let mut filtered = Vec::new();
            let mut filtered_weights = Vec::new();
            let mut cumulative = 0.0;
            let sum: f32 = weights.iter().sum::<f32>().max(1e-6);

            for ((token, _), weight) in candidates.iter().zip(weights.iter()) {
                let prob = weight / sum;
                filtered.push(*token);
                filtered_weights.push(*weight);
                cumulative += prob;
                if cumulative >= config.top_p {
                    break;
                }
            }

            candidates = filtered.into_iter().map(|id| (id, 0.0)).collect();
            weights = filtered_weights;
        }

        let sample = WeightedIndex::new(&weights)
            .ok()
            .map(|dist| candidates[dist.sample(&mut rng)].0)
            .unwrap_or_else(|| candidates.first().map(|(id, _)| *id).unwrap_or(0));

        outputs.push(sample as i64);
    }

    outputs
}
