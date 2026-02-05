//! E2E: GGUF loader + Q4_0 quantization + basic inference.

use std::path::PathBuf;
use std::sync::Arc;

use gllm::adapter::{AdapterResult, AdapterWeights, ModelAdapter};
use gllm::engine::executor::Executor;
use gllm::loader::{
    config as loader_config, CacheLayout, HfHubClient, Loader, LoaderConfig, LoaderError,
    ParallelLoader, TensorInfo, UploadedTensor, WeightsHandle,
};
use gllm::manifest::{FileMap, ModelArchitecture, ModelKind, ModelManifest, EMPTY_FILE_MAP};
use gllm_kernels::backend_trait::Backend;
use gllm_kernels::cpu_backend::CpuBackend;
use gllm_kernels::QuantizedType;

const BASE_REPO: &str = "HuggingFaceTB/SmolLM-135M-Instruct";
const GGUF_REPO: &str = "mav23/SmolLM-135M-Instruct-GGUF";
const GGUF_FILE: &str = "smollm-135m-instruct.Q4_0.gguf";

const GGUF_FILE_MAP: FileMap = &[("model.gguf", GGUF_FILE)];

struct GgufRemapAdapter;

static GGUF_ADAPTER: GgufRemapAdapter = GgufRemapAdapter;

impl<B: Backend> ModelAdapter<B> for GgufRemapAdapter {
    fn supports(&self, manifest: &ModelManifest) -> bool {
        matches!(manifest.arch, ModelArchitecture::Llama4)
    }

    fn load_weights(&self, loader: &mut Loader, backend: &B) -> AdapterResult<AdapterWeights<B>> {
        let handle = loader.upload_weights(backend)?;
        let handle = remap_gguf_handle(handle)?;
        Ok(AdapterWeights::new(handle))
    }
}

#[test]
fn gguf_loader_smollm_q4_0_e2e() {
    let (config_path, tokenizer_path, manifest) =
        download_base_files().expect("download base config/tokenizer");
    let gguf_path = download_gguf_file().expect("download gguf weights");

    let mut loader = Loader::from_local_files_with_manifest(
        BASE_REPO,
        vec![gguf_path],
        vec![config_path, tokenizer_path],
        Some(&manifest),
    )
    .expect("gguf loader");

    let backend = CpuBackend::new();
    let mut executor =
        Executor::from_loader(backend, Arc::new(manifest), &GGUF_ADAPTER, &mut loader)
            .expect("executor init");

    let q4_tensors: Vec<_> = executor
        .weights()
        .handle
        .meta
        .iter()
        .filter(|(_, info)| info.quantized == Some(QuantizedType::Q4_0))
        .collect();
    assert!(!q4_tensors.is_empty(), "expected at least one Q4_0 tensor");

    let q4_name = q4_tensors[0].0.as_str();
    match executor.weights().handle.get(q4_name) {
        Some(UploadedTensor::F32(values)) => {
            let sum: f32 = values.iter().map(|v| v.abs()).sum();
            assert!(sum > 0.01, "dequantized tensor appears to be all zeros");
        }
        _ => panic!("missing dequantized tensor for {q4_name}"),
    }

    let output = executor
        .generate("The capital of", 8, 0.0)
        .expect("generate output");
    assert!(!output.trim().is_empty(), "generation output empty");
}

fn download_base_files() -> Result<(PathBuf, PathBuf, ModelManifest), String> {
    let config = LoaderConfig::default();
    let files = loader_config::download_config_files(BASE_REPO, &config, EMPTY_FILE_MAP)
        .map_err(|e| format!("download config files for {BASE_REPO} failed: {e}"))?;
    let tokenizer_path = files
        .tokenizer_path
        .ok_or_else(|| "tokenizer.json missing in base repo".to_string())?;
    let config_value =
        loader_config::load_config_value(&files.config_path).map_err(|e| e.to_string())?;
    let manifest = loader_config::manifest_from_config(BASE_REPO, &config_value, ModelKind::Chat)
        .map_err(|e| e.to_string())?;
    Ok((files.config_path, tokenizer_path, manifest))
}

fn download_gguf_file() -> Result<PathBuf, String> {
    let cache = CacheLayout::new(None).map_err(|e| e.to_string())?;
    let hf = HfHubClient::new(cache.hf_cache_dir()).map_err(|e| e.to_string())?;
    let files = hf
        .download_model_files(GGUF_REPO, GGUF_FILE_MAP, ParallelLoader::new(false))
        .map_err(|e| format!("download gguf file for {GGUF_REPO} failed: {e}"))?;
    files
        .weights
        .into_iter()
        .next()
        .ok_or_else(|| "gguf weights missing".to_string())
}

fn remap_gguf_handle<B: Backend>(
    handle: WeightsHandle<B>,
) -> Result<WeightsHandle<B>, LoaderError> {
    let WeightsHandle { tensors, meta } = handle;
    let mut out = WeightsHandle::default();

    for (name, tensor) in tensors {
        let mapped = map_gguf_name(&name).unwrap_or_else(|| name.clone());
        if out.tensors.contains_key(&mapped) {
            return Err(LoaderError::DuplicateTensor(mapped));
        }
        if let Some(info) = meta.get(&name) {
            let shape = normalize_gguf_shape(&info.shape);
            out.meta.insert(
                mapped.clone(),
                TensorInfo {
                    shape,
                    dtype: info.dtype,
                    quantized: info.quantized,
                },
            );
        }
        out.tensors.insert(mapped, tensor);
    }

    Ok(out)
}

// GGUF stores matrix shapes as [cols, rows]; backend expects [rows, cols].
fn normalize_gguf_shape(shape: &[usize]) -> Vec<usize> {
    if shape.len() == 2 {
        vec![shape[1], shape[0]]
    } else {
        shape.to_vec()
    }
}

fn map_gguf_name(name: &str) -> Option<String> {
    match name {
        "token_embd.weight" => {
            return Some("model.embed_tokens.weight".to_string());
        }
        "output_norm.weight" => {
            return Some("model.norm.weight".to_string());
        }
        "output.weight" => {
            return Some("lm_head.weight".to_string());
        }
        _ => {}
    }

    let rest = name.strip_prefix("blk.")?;
    let mut parts = rest.splitn(2, '.');
    let layer = parts.next()?.parse::<usize>().ok()?;
    let suffix = parts.next()?;

    let mapped = match suffix {
        "attn_norm.weight" => format!("model.layers.{layer}.input_layernorm.weight"),
        "attn_q.weight" => format!("model.layers.{layer}.self_attn.q_proj.weight"),
        "attn_k.weight" => format!("model.layers.{layer}.self_attn.k_proj.weight"),
        "attn_v.weight" => format!("model.layers.{layer}.self_attn.v_proj.weight"),
        "attn_output.weight" => format!("model.layers.{layer}.self_attn.o_proj.weight"),
        "ffn_norm.weight" => format!("model.layers.{layer}.post_attention_layernorm.weight"),
        "ffn_gate.weight" => format!("model.layers.{layer}.mlp.gate_proj.weight"),
        "ffn_up.weight" => format!("model.layers.{layer}.mlp.up_proj.weight"),
        "ffn_down.weight" => format!("model.layers.{layer}.mlp.down_proj.weight"),
        _ => return None,
    };

    Some(mapped)
}
