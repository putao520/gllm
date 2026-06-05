use std::sync::Arc;
use gllm::compat::cpu_backend::CpuBackend;
use gllm::compat::backend_trait::Backend;
use gllm::engine::executor::KvCacheConfig;
use gllm::model_config::ModelGeometry;
use gllm_kernels::types::DType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Dual-Track Memory Pool Demo ===\n");

    // Create CPU backend
    let backend = CpuBackend::<f32>::new();

    // Allocate KV cache with FP16 dtype for reduced memory footprint
    let geometry = Arc::new(ModelGeometry {
        hidden_size: 512,
        num_layers: 12,
        vocab_size: 32000,
        intermediate_size: 2048,
        num_heads: 8,
        num_kv_heads: 8,
        head_dim: 64,
        max_seq_len: 2048,
        rope_theta: 10000.0,
        rope_scale: 1.0,
        rope_interleaved: false,
        global_rope_theta: 0.0,
        rope_partial_ratio: 1.0,
        attention_pattern: Vec::new(),
        sliding_window: 0,
        num_kv_shared_layers: 0,
        global_head_dim: 0,
        hidden_size_per_layer_input: 0,
        dtype: DType::F32,
        norm_eps: 1e-5,
        num_experts: 0,
        moe_top_k: 0,
        expert_intermediate_size: 0,
                position_offset: None,
                rope_scaling: None,
                final_logit_softcapping: None,
                hidden_act: None,
                compute_dtype: DType::F32,
                mla_d_c: 0,
                mla_d_rope: 0,
                mla_unabsorbed_threshold: 0,
    });

    let config = KvCacheConfig {
        geometry,
        page_size: 16,
        kv_dtype: DType::F16,
        swap_config: None,
    };

    println!("Allocating KV cache:");
    println!("  Layers: {}", config.num_layers());
    println!("  Heads: {}", config.num_heads());
    println!("  Head dim: {}", config.head_dim());
    println!("  Max seq len: {}", config.max_seq_len());
    println!("  Page size: {}", config.page_size);
    println!("  KV DType: {:?} ({} bytes/element)\n", config.kv_dtype, config.dtype_size());

    let handle = backend.alloc_kv_cache(&config)?;
    println!("KV cache allocated (handle: {})\n", handle.0);

    println!("Note: Dual-track pool was a deprecated design concept.");
    println!("      gllm now uses single-buffer KvCacheState with pipeline");
    println!("      pre-scheduling for efficient KV cache memory management.");
    println!("      See executor.rs pipeline pre-scheduling for details.");

    Ok(())
}
