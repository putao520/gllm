//! SigLIP Vision Encoder infrastructure for multimodal models (Gemma 4).
//!
//! Provides the type skeleton and forward pass entry point for encoding
//! raw image pixels into a visual token sequence consumable by the
//! language model backbone.  Full implementation is deferred to P3.

use crate::engine::executor::BackendError;

/// Trait for looking up named vision encoder weight tensors.
pub trait VisionTensorLookup {
    /// Returns the raw f32 data for the given weight name, or `None` if absent.
    fn get_vision_tensor(&self, name: &str) -> Option<&[f32]>;

    /// Returns the shape of the given weight tensor, or `None` if absent.
    fn vision_tensor_shape(&self, name: &str) -> Option<&[usize]>;
}

/// SigLIP Vision Encoder configuration.
///
/// Parsed from the `"vision_config"` sub-object in `config.json`.
/// All fields mirror the HuggingFace SigLIP / ViT config schema.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Input image resolution (pixels per side, e.g. 224 or 384).
    pub image_size: usize,
    /// Patch size (pixels per side, e.g. 14 or 16).
    pub patch_size: usize,
    /// Hidden dimension of the vision transformer.
    pub hidden_size: usize,
    /// Number of transformer encoder layers.
    pub num_layers: usize,
    /// Number of self-attention heads.
    pub num_heads: usize,
    /// Feed-forward intermediate dimension.
    pub intermediate_size: usize,
}

impl VisionConfig {
    /// Number of spatial patches = (image_size / patch_size)^2.
    pub fn num_patches(&self) -> usize {
        let grid = self.image_size / self.patch_size;
        grid * grid
    }
}

/// Encode raw image pixel data into a visual token sequence.
///
/// # Arguments
///
/// * `pixels`  - Flattened pixel data in `[channels, height, width]` layout
///               (channel-first, normalised to `[0, 1]` or model-specific range).
/// * `config`  - Vision encoder geometry.
/// * `weights` - Provider for vision encoder weight tensors.
///
/// # Returns
///
/// A flat `Vec<f32>` of shape `[num_patches, hidden_size]` representing the
/// visual token embeddings, or an error.
///
/// # Current status
///
/// Returns `Err` unconditionally — full implementation is planned for P3.
pub fn vision_encode(
    _pixels: &[f32],
    _config: &VisionConfig,
    _weights: &dyn VisionTensorLookup,
) -> Result<Vec<f32>, BackendError> {
    Err(BackendError::Other(
        "vision encoder not yet implemented (P3)".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EmptyLookup;

    impl VisionTensorLookup for EmptyLookup {
        fn get_vision_tensor(&self, _name: &str) -> Option<&[f32]> {
            None
        }
        fn vision_tensor_shape(&self, _name: &str) -> Option<&[usize]> {
            None
        }
    }

    #[test]
    fn vision_encode_returns_unimplemented() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        let result = vision_encode(&[], &config, &EmptyLookup);
        assert!(result.is_err());
    }

    #[test]
    fn num_patches_calculation() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
        };
        assert_eq!(config.num_patches(), 256); // (224/14)^2 = 16^2 = 256
    }
}
