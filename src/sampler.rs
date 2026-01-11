use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

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
    let [batch_size, vocab_size] = logits.dims();
    if config.temperature <= 0.0 {
        let indices = match logits.argmax(1).into_data().into_vec::<B::IntElem>() {
            Ok(data) => data,
            Err(_) => return vec![0; batch_size],
        };
        return indices.into_iter().map(|v| v.elem::<i64>()).collect();
    }

    let logits = if (config.temperature - 1.0).abs() > f32::EPSILON {
        logits / config.temperature
    } else {
        logits
    };

    let apply_top_p = config.top_p > 0.0 && config.top_p < 1.0;
    let k = if config.top_k > 0 {
        config.top_k.min(vocab_size)
    } else {
        vocab_size
    };

    let (values, indices) = if config.top_k > 0 || apply_top_p {
        logits.topk_with_indices(k, 1)
    } else {
        let indices = Tensor::<B, 1, Int>::arange(0..vocab_size as i64, device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[batch_size, 1]);
        (logits, indices)
    };
    let probs = softmax(values, 1);

    let prob_data = match probs.into_data().into_vec::<f32>() {
        Ok(data) => data,
        Err(_) => return vec![0; batch_size],
    };
    let index_data = match indices.into_data().into_vec::<B::IntElem>() {
        Ok(data) => data,
        Err(_) => return vec![0; batch_size],
    };

    let mut rng = thread_rng();
    let mut outputs = Vec::with_capacity(batch_size);
    let stride = if config.top_k > 0 || apply_top_p {
        k
    } else {
        vocab_size
    };

    for batch in 0..batch_size {
        let start = batch * stride;
        let end = start + stride;
        let row_probs = &prob_data[start..end];
        let row_indices = &index_data[start..end];
        let mut candidates: Vec<(i64, f32)> = row_indices
            .iter()
            .zip(row_probs.iter())
            .map(|(idx, &prob)| (idx.elem::<i64>(), prob))
            .collect();

        if apply_top_p {
            let mut filtered = Vec::new();
            let mut filtered_weights = Vec::new();
            let mut cumulative = 0.0;

            for (token, weight) in candidates.iter() {
                filtered.push(*token);
                filtered_weights.push(*weight);
                cumulative += *weight;
                if cumulative >= config.top_p {
                    break;
                }
            }

            candidates = filtered.into_iter().map(|id| (id, 0.0)).collect();
            let weights = filtered_weights;

            let sample = WeightedIndex::new(&weights)
                .ok()
                .map(|dist| candidates[dist.sample(&mut rng)].0)
                .unwrap_or_else(|| candidates.first().map(|(id, _)| *id).unwrap_or(0));
            outputs.push(sample);
            continue;
        }

        let weights: Vec<f32> = candidates.iter().map(|(_, prob)| *prob).collect();
        let sample = WeightedIndex::new(&weights)
            .ok()
            .map(|dist| candidates[dist.sample(&mut rng)].0)
            .unwrap_or_else(|| candidates.first().map(|(id, _)| *id).unwrap_or(0));

        outputs.push(sample);
    }

    outputs
}
