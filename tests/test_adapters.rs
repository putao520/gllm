//! E2E tests using the actual Client API (what users use).

use gllm::Client;

/// E2E test: Validate complete model loading pipeline.
///
/// Uses the actual Client API that users use:
/// 1. Model lookup via registry
/// 2. Auto-download (HF → ModelScope fallback)
/// 3. Backend initialization
/// 4. Weight loading and upload
///
/// Model: SmolLM2_135M (~280MB) - smallest for quick validation.
#[test]
fn e2e_client_loads_model() {
    let model_alias = "smollm2-135m";

    println!("=== E2E Test: Client::new() ===");
    println!("Model: {}", model_alias);

    // This single call tests the entire pipeline:
    // - registry lookup
    // - manifest validation
    // - backend detection
    // - download (if needed)
    // - weight loading
    let client = Client::new(model_alias).expect("Client::new() should succeed");

    println!("✅ Client created successfully");
    println!("   Manifest: {:?}", client.manifest().model_id);
    println!("   Architecture: {:?}", client.manifest().arch);
}

/// Verify all registered models have a corresponding adapter.
#[test]
fn registry_manifests_have_adapters() {
    use gllm::adapter::adapter_for;
    use gllm::registry;
    use gllm_kernels::cpu_backend::CpuBackend;

    for manifest in registry::all() {
        let adapter = adapter_for::<CpuBackend>(manifest);
        assert!(
            adapter.is_some(),
            "missing adapter for {:?}",
            manifest.model_id
        );
    }
}
