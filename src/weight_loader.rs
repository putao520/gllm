//! Weight loader facade (re-exports loader).

pub use crate::loader::{
    ChecksumPolicy, Loader, LoaderConfig, LoaderError, ParallelPolicy, TensorInfo, UploadedTensor,
    WeightsHandle,
};

// ---------------------------------------------------------------------------
// SharedKvRef (§P1.1): helpers so loaders can tolerate missing K/V weights
// on shared-consumer layers (Gemma 4 E2B / E4B).
// ---------------------------------------------------------------------------

/// Returns `true` when layer `layer_i` is a KV-sharing *consumer* and the
/// checkpoint therefore does NOT contain its `self_attn.k_proj.weight` /
/// `self_attn.v_proj.weight`.
///
/// Non-shared layers still require the K/V projections and the loader must
/// error if they are absent.
#[inline]
pub fn layer_allows_missing_kv_weights(
    layer_i: usize,
    num_hidden_layers: usize,
    num_kv_shared_layers: usize,
) -> bool {
    if num_kv_shared_layers == 0 || layer_i >= num_hidden_layers {
        return false;
    }
    layer_i + num_kv_shared_layers >= num_hidden_layers
}

/// True when `role` describes a K/V projection weight (the weights that shared
/// consumer layers are allowed to omit).
#[inline]
pub fn is_kv_projection_role(role: crate::manifest::TensorRole) -> bool {
    matches!(
        role,
        crate::manifest::TensorRole::AttentionKey | crate::manifest::TensorRole::AttentionValue
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::TensorRole;

    #[test]
    fn consumer_layers_allow_missing_kv_weights_gemma4_e2b() {
        // Gemma 4 E2B: 32 layers, last 20 share KV.
        let layers = 32;
        let shared = 20;
        for l in 0..12 {
            assert!(!layer_allows_missing_kv_weights(l, layers, shared));
        }
        for l in 12..32 {
            assert!(layer_allows_missing_kv_weights(l, layers, shared));
        }
    }

    #[test]
    fn consumer_layers_allow_missing_kv_weights_gemma4_e4b() {
        // Gemma 4 E4B: 34 layers, last 18 share KV.
        let layers = 34;
        let shared = 18;
        for l in 0..16 {
            assert!(!layer_allows_missing_kv_weights(l, layers, shared));
        }
        for l in 16..34 {
            assert!(layer_allows_missing_kv_weights(l, layers, shared));
        }
    }

    #[test]
    fn no_shared_layers_disables_relaxation() {
        for l in 0..32 {
            assert!(!layer_allows_missing_kv_weights(l, 32, 0));
        }
    }

    #[test]
    fn kv_projection_role_classification() {
        assert!(is_kv_projection_role(TensorRole::AttentionKey));
        assert!(is_kv_projection_role(TensorRole::AttentionValue));
        assert!(!is_kv_projection_role(TensorRole::AttentionQuery));
        assert!(!is_kv_projection_role(TensorRole::Embedding));
    }
}
