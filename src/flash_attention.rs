use gllm_kernels::{FlashAttentionConfig, KernelDispatcher, KernelFloat};

pub use gllm_kernels::FlashAttentionConfig as KernelFlashAttentionConfig;

/// Thin wrapper around gllm-kernels flash attention.
pub fn flash_attention_forward<T: KernelFloat>(
    q: &[T],
    k: &[T],
    v: &[T],
    output: &mut [T],
    config: FlashAttentionConfig,
) {
    let dispatcher = KernelDispatcher::new();
    dispatcher.flash_attention(q, k, v, output, config);
}
