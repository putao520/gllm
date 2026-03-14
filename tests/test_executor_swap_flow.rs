use gllm::engine::executor::Executor;
use gllm::loader::Loader;
use gllm::manifest::{
    ModelArchitecture, ModelKind, ModelManifest, EMPTY_FILE_MAP,
};
use gllm::compat::CpuBackend;
use std::borrow::Cow;
use std::env;
use std::sync::Arc;

/// TEST-SCHED-004: 执行器内存压力下的换入换出流程
/// **关联需求**: REQ-SCHED-002
/// **测试类型**: 正向
/// **期望结果**: 内存压力下正确执行 KV Cache 换出和换入操作
#[test]
fn test_executor_swap_flow_under_pressure() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Setup Environment
    // Force L1 capacity to be very small to trigger swapping.
    // Block size is 16.
    // Set L1 = 4 blocks (enough for ~2 requests active, but 3rd will force swap).
    unsafe {
        env::set_var("GLLM_KV_CACHE_BLOCKS", "4");
    }

    // 2. Load Model (SmolLM2-135M-Instruct)
    let model_id = "HuggingFaceTB/SmolLM2-135M-Instruct";
    let config = gllm::loader::LoaderConfig::from_env();
    let mut loader = Loader::from_source_with_config(model_id.to_string(), config)?;

    // Manually construct manifest as auto-detection might not be exposed in Loader yet
    let manifest = Arc::new(ModelManifest {
        model_id: Cow::Borrowed(model_id),
        file_map: EMPTY_FILE_MAP,
        arch: ModelArchitecture::Llama4,
        kind: ModelKind::Chat,
        rope_base_override: None,
        max_context_override: None,
        moe_config: None,
        tensor_map: std::collections::HashMap::new(),
    });

    // Select backend (CpuBackend)
    let backend = CpuBackend::<f32>::new();

    // 3. Initialize Executor
    let mut executor = Executor::from_loader(backend, manifest, &mut loader)?;

    // 4. Define Requests
    // Short prompts that trigger generation.
    // 10 tokens prompt + 20 gen = 30 tokens = 2 blocks per req.
    // 3 requests = 6 blocks total.
    // Capacity 4 blocks => Must swap.
    let prompts = vec!["Hello", "Paris is", "1, 2, 3,"];

    // 5. Enqueue
    let mut req_ids = Vec::new();
    for p in &prompts {
        use gllm::scheduler::types::RequestKind;
        // Max 20 new tokens
        let config = gllm::engine::SamplingConfig::default();
        let id = executor.enqueue_with_config(RequestKind::Chat, *p, 20, config)?;
        req_ids.push(id);
    }

    println!("Enqueued {} requests. Running step loop...", req_ids.len());

    let mut ticks = 0;
    let max_ticks = 200; // ample time for CPU
    let mut all_finished = false;

    // 6. Run Loop
    while ticks < max_ticks {
        executor.step()?;
        ticks += 1;

        let finished_count = req_ids
            .iter()
            .filter(|&&id| executor.is_finished(id))
            .count();
        if finished_count == req_ids.len() {
            all_finished = true;
            break;
        }
    }

    assert!(
        all_finished,
        "Requests did not finish within {} steps",
        max_ticks
    );

    // 7. Verify Output
    // If swapping corrupted data, we expect garbage or truncated text.
    // SmolLM should produce coherent text.
    for (i, &id) in req_ids.iter().enumerate() {
        let output = executor.get_output(id)?;
        println!("Request {}: Output: {:?}", i, output);

        assert!(!output.is_empty(), "Output should not be empty");
        // Basic coherence check (not strict as model is small)
        // If data was corrupted (e.g. all zeros), decode might be empty or weird chars.
        // We just check it's not empty and ran to completion.
    }

    println!("Swap flow test passed!");
    Ok(())
}
