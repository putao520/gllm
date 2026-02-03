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

#[test]
fn adapters_load_weights_with_local_files() {
    let files = TestModelFiles::new().expect("test model files");
    let backend = CpuBackend::new();

    for manifest in registry::all() {
        let alias = manifest.aliases.first().copied().expect("manifest alias");
        let adapter = adapter_for::<CpuBackend>(manifest).expect("adapter");
        let mut loader = files.loader(alias).expect("loader");
        let weights = adapter
            .load_weights(&mut loader, &backend)
            .expect("weights");
        assert!(
            !weights.handle.tensors.is_empty(),
            "adapter {} returned empty weights",
            alias
        );
    }
}
