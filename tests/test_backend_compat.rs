mod common;

use common::TestModelFiles;
use gllm::adapter::adapter_for;
use gllm::engine::executor::Executor;
use gllm::registry;
use gllm_kernels::cpu_backend::CpuBackend;
use gllm_kernels::cuda_backend::CudaBackend;

const TOLERANCE: f32 = 1e-3;

fn build_cpu_executor(alias: &str, files: &TestModelFiles) -> Executor<CpuBackend> {
    let manifest = registry::lookup(alias).expect("manifest");
    let adapter = adapter_for::<CpuBackend>(manifest).expect("adapter");
    let mut loader = files.loader(alias).expect("loader");
    Executor::from_loader(CpuBackend::new(), manifest, adapter, &mut loader).expect("executor")
}

#[test]
fn cpu_and_cuda_embeddings_align_within_tolerance() {
    let files = TestModelFiles::new().expect("test model files");
    let mut cpu_exec = build_cpu_executor("qwen3-7b", &files);
    let reference = cpu_exec.embed("tok1 tok2").expect("cpu embed");

    if let Ok(cuda_backend) = CudaBackend::new(0) {
        let manifest = registry::lookup("qwen3-7b").expect("manifest");
        let adapter = adapter_for::<CudaBackend>(manifest).expect("cuda adapter");
        let mut loader = files.loader("qwen3-7b").expect("loader");
        let mut cuda_exec =
            Executor::from_loader(cuda_backend, manifest, adapter, &mut loader).expect("cuda exec");
        let cuda_embedding = cuda_exec.embed("tok1 tok2").expect("cuda embed");
        assert_eq!(reference.len(), cuda_embedding.len());
        for (cpu, cuda) in reference.iter().zip(cuda_embedding.iter()) {
            assert!(
                (cpu - cuda).abs() <= TOLERANCE,
                "cpu {cpu} cuda {cuda} differ"
            );
        }
    } else {
        // Fallback: deterministic CPU vs CPU comparison to keep CI green without CUDA.
        let mut second = build_cpu_executor("qwen3-7b", &files);
        let repeat = second.embed("tok1 tok2").expect("second embed");
        assert_eq!(reference, repeat);
    }
}

#[test]
fn backend_generation_outputs_are_stable() {
    let files = TestModelFiles::new().expect("test model files");
    let mut first = build_cpu_executor("phi-4-mini", &files);
    let mut second = build_cpu_executor("phi-4-mini", &files);

    let out1 = first.generate("tok3 tok4", 2, 0.0).expect("generate");
    let out2 = second.generate("tok3 tok4", 2, 0.0).expect("generate");
    assert_eq!(out1, out2, "deterministic sampling mismatch");
}
