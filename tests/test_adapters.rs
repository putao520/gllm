mod common;

use common::TestModelFiles;
use gllm::adapter::adapter_for;
use gllm::registry;
use gllm_kernels::cpu_backend::CpuBackend;

#[test]
fn adapter_registry_covers_all_manifests() {
    for manifest in registry::all() {
        let adapter = adapter_for::<CpuBackend>(manifest);
        assert!(
            adapter.is_some(),
            "missing adapter for {:?}",
            manifest.model_id
        );
    }
}

/// Test that adapters can load weights for models that are already downloaded.
///
/// This test only checks models that are already present in the cache directory.
/// Missing models are skipped with a warning rather than failing the test.
#[test]
fn adapters_load_weights_with_local_files() {
    let files = TestModelFiles::new().expect("test model files");
    let backend = CpuBackend::new();

    let mut tested = 0;
    let mut passed = 0;

    for manifest in registry::all() {
        let alias = manifest.aliases.first().copied().expect("manifest alias");
        println!("Testing adapter for: {} (hf_repo: {})", alias, manifest.hf_repo);

        // Check if the model directory exists in cache
        let cache_path = files.base_dir().join(manifest.hf_repo.replace('/', "--"));
        let model_exists = cache_path.exists();

        if !model_exists {
            println!("⚠️  {} skipped (not downloaded)", alias);
            continue;
        }

        tested += 1;
        let adapter = adapter_for::<CpuBackend>(manifest).expect("adapter");
        let mut loader = files.loader(alias).expect("loader");

        match adapter.load_weights(&mut loader, &backend) {
            Ok(weights) => {
                assert!(
                    !weights.handle.tensors.is_empty(),
                    "adapter {} returned empty weights",
                    alias
                );
                println!("✓ {} loaded successfully", alias);
                passed += 1;
            }
            Err(e) => {
                // Models that exist but fail to load are test failures
                panic!("adapter {} failed to load: {}", alias, e);
            }
        }
    }

    println!("\nTested {} models, {} passed", tested, passed);
    assert!(tested > 0, "No models found in cache - please download at least one model first");
}
