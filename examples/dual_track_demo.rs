use gllm::compat::cpu_backend::CpuBackend;
use gllm::compat::backend_trait::Backend;
use gllm::engine::executor::KvCacheConfig;
use gllm_kernels::types::DType;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Dual-Track Memory Pool Demo ===\n");

    // Create CPU backend
    let backend = CpuBackend::<f32>::new();

    // Allocate KV cache with FP16 dtype for reduced memory footprint
    let config = KvCacheConfig {
        num_layers: 12,
        num_heads: 8,
        head_dim: 64,
        max_seq_len: 2048,
        page_size: 16,
        kv_dtype: DType::F16,
        swap_config: None,
        quant: None,
    };

    println!("Allocating KV cache:");
    println!("  Layers: {}", config.num_layers);
    println!("  Heads: {}", config.num_heads);
    println!("  Head dim: {}", config.head_dim);
    println!("  Max seq len: {}", config.max_seq_len);
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
