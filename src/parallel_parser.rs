use crate::quantized::{NativeQLinear, QuantizedWeight};
use crate::types::{Error, Result};
use crate::weight_loader::{
    convert_to_f32_cow, LayerNormWeights, LinearWeights, LoadedTensor, LoadedTensorView,
    MultiHeadAttentionWeights, RawTensor, RawTensorView, WeightLoader,
};
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug, Clone, Copy)]
pub enum ProgressStage {
    CheckingCache,
    Downloading,
    Parsing,
    LoadingWeights,
    Complete,
}

#[derive(Debug, Clone, Copy)]
pub struct LoadProgress {
    pub stage: ProgressStage,
    pub current_shard: usize,
    pub total_shards: usize,
    pub bytes_loaded: u64,
}

pub struct LoadConfig {
    /// Maximum number of parsing threads.
    pub max_parse_threads: usize,
    /// Whether to use mmap for shard reads.
    pub use_mmap: bool,
    /// Optional progress callback.
    pub progress_callback: Option<Box<dyn Fn(LoadProgress) + Send + Sync>>,
}

impl Default for LoadConfig {
    fn default() -> Self {
        let max_parse_threads = std::thread::available_parallelism()
            .map(|v| v.get())
            .unwrap_or(1);
        Self {
            max_parse_threads,
            use_mmap: true,
            progress_callback: None,
        }
    }
}

enum ShardBytes {
    Mmap(Arc<Mmap>),
    Heap(Arc<[u8]>),
}

impl ShardBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Mmap(mmap) => &mmap[..],
            Self::Heap(bytes) => bytes,
        }
    }

    fn len(&self) -> u64 {
        self.as_slice().len() as u64
    }
}

pub struct ShardData {
    pub path: PathBuf,
    #[allow(dead_code)]
    bytes: ShardBytes,
    tensors: SafeTensors<'static>,
}

pub struct ParsedShards {
    shards: Vec<ShardData>,
    shard_name_map: HashMap<String, usize>,
}

impl ParsedShards {
    fn new(shards: Vec<ShardData>) -> Result<Self> {
        let mut shard_name_map = HashMap::new();
        for (idx, shard) in shards.iter().enumerate() {
            let name = shard
                .path
                .file_name()
                .ok_or_else(|| {
                    Error::LoadError(format!(
                        "Shard path is missing filename: {}",
                        shard.path.display()
                    ))
                })?
                .to_string_lossy()
                .into_owned();
            shard_name_map.insert(name, idx);
        }
        Ok(Self {
            shards,
            shard_name_map,
        })
    }

    fn shard_for_tensor(&self, tensor_name: &str, index: &HashMap<String, usize>) -> Result<&ShardData> {
        let shard_idx = index.get(tensor_name).ok_or_else(|| {
            Error::LoadError(format!("Tensor '{}' not found in shard index", tensor_name))
        })?;
        self.shards
            .get(*shard_idx)
            .ok_or_else(|| Error::LoadError("Shard index out of range".into()))
    }

    fn shard_index(&self, shard_name: &str) -> Option<usize> {
        self.shard_name_map.get(shard_name).copied()
    }
}

static PARSED_SHARD_CACHE: OnceLock<Mutex<HashMap<PathBuf, ParsedShards>>> = OnceLock::new();

fn shard_cache() -> &'static Mutex<HashMap<PathBuf, ParsedShards>> {
    PARSED_SHARD_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) fn cache_parsed_shards(model_dir: PathBuf, parsed: ParsedShards) {
    if let Ok(mut cache) = shard_cache().lock() {
        cache.insert(model_dir, parsed);
    }
}

pub(crate) fn take_cached_shards(model_dir: &Path) -> Option<ParsedShards> {
    shard_cache().lock().ok()?.remove(model_dir)
}

pub(crate) fn is_shard_index(path: &Path) -> bool {
    path.file_name()
        .map(|name| name == "model.safetensors.index.json")
        .unwrap_or(false)
}

pub fn parse_shards(shard_files: Vec<PathBuf>, config: &LoadConfig) -> Result<ParsedShards> {
    if shard_files.is_empty() {
        return Err(Error::LoadError("No shard files provided".into()));
    }

    let total_shards = shard_files.len();
    let bytes_loaded = AtomicU64::new(0);
    let completed_shards = AtomicUsize::new(0);
    let callback = config.progress_callback.as_deref();
    let max_threads = config.max_parse_threads.max(1);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build()
        .map_err(|err| Error::LoadError(format!("Failed to build rayon pool: {err}")))?;

    let shards = pool.install(|| {
        shard_files
            .par_iter()
            .map(|path| {
                let bytes = load_shard_bytes(path, config.use_mmap)?;
                let tensors = parse_tensors(&bytes).map_err(|err| {
                    Error::LoadError(format!(
                        "Failed to parse SafeTensors shard {}: {err}",
                        path.display()
                    ))
                })?;

                let shard_bytes = bytes.len();
                let total_bytes = bytes_loaded.fetch_add(shard_bytes, Ordering::SeqCst) + shard_bytes;
                let current_shard = completed_shards.fetch_add(1, Ordering::SeqCst) + 1;
                if let Some(callback) = callback {
                    callback(LoadProgress {
                        stage: ProgressStage::Parsing,
                        current_shard,
                        total_shards,
                        bytes_loaded: total_bytes,
                    });
                }

                Ok(ShardData {
                    path: path.clone(),
                    bytes,
                    tensors,
                })
            })
            .collect::<Result<Vec<_>>>()
    })?;

    ParsedShards::new(shards)
}

fn load_shard_bytes(path: &Path, use_mmap: bool) -> Result<ShardBytes> {
    if use_mmap {
        if let Ok(file) = File::open(path) {
            if let Ok(mmap) = unsafe { Mmap::map(&file) } {
                return Ok(ShardBytes::Mmap(Arc::new(mmap)));
            }
        }
    }

    let bytes = fs::read(path).map_err(|err| {
        Error::LoadError(format!("Failed to read shard {}: {err}", path.display()))
    })?;
    Ok(ShardBytes::Heap(Arc::from(bytes)))
}

fn parse_tensors(bytes: &ShardBytes) -> Result<SafeTensors<'static>> {
    let slice = bytes.as_slice();
    // Safety: the bytes are owned by ShardData, ensuring the slice lives as long as the SafeTensors.
    let static_slice: &'static [u8] = unsafe { std::mem::transmute(slice) };
    SafeTensors::deserialize(static_slice)
        .map_err(|err| Error::LoadError(format!("SafeTensors parse failed: {err}")))
}

pub trait TensorLoader {
    fn has_tensor(&self, name: &str) -> bool;
    fn load_tensor(&self, name: &str) -> Result<LoadedTensor>;
    fn load_raw_tensor(&self, name: &str) -> Result<RawTensor>;
    fn load_tensor_view(&self, name: &str) -> Result<LoadedTensorView<'_>> {
        let LoadedTensor { data, shape } = self.load_tensor(name)?;
        Ok(LoadedTensorView {
            data: Cow::Owned(data),
            shape,
        })
    }
    fn load_raw_tensor_view(&self, name: &str) -> Result<RawTensorView<'_>> {
        let RawTensor { data, shape, dtype } = self.load_raw_tensor(name)?;
        Ok(RawTensorView {
            data: Cow::Owned(data),
            shape,
            dtype,
        })
    }
    fn is_awq_model(&self) -> bool;
}

impl<'a> TensorLoader for WeightLoader<'a> {
    fn has_tensor(&self, name: &str) -> bool {
        self.has_tensor(name)
    }

    fn load_tensor(&self, name: &str) -> Result<LoadedTensor> {
        self.load_tensor(name)
    }

    fn load_raw_tensor(&self, name: &str) -> Result<RawTensor> {
        self.load_raw_tensor(name)
    }

    fn load_tensor_view(&self, name: &str) -> Result<LoadedTensorView<'_>> {
        WeightLoader::load_tensor_view(self, name)
    }

    fn load_raw_tensor_view(&self, name: &str) -> Result<RawTensorView<'_>> {
        WeightLoader::load_raw_tensor_view(self, name)
    }

    fn is_awq_model(&self) -> bool {
        self.is_awq_model()
    }
}

pub struct ShardedTensorLoader<'a> {
    shards: &'a ParsedShards,
    tensor_to_shard: HashMap<String, usize>,
}

impl<'a> ShardedTensorLoader<'a> {
    pub fn new(
        parsed: &'a ParsedShards,
        index: &crate::weight_loader::shards::ShardIndex,
    ) -> Result<Self> {
        let mut tensor_to_shard = HashMap::new();
        for (tensor_name, shard_file) in &index.tensor_to_shard {
            let idx = parsed.shard_index(shard_file).ok_or_else(|| {
                Error::LoadError(format!(
                    "Shard '{}' referenced by '{}' is missing",
                    shard_file, tensor_name
                ))
            })?;
            tensor_to_shard.insert(tensor_name.clone(), idx);
        }
        Ok(Self {
            shards: parsed,
            tensor_to_shard,
        })
    }

    fn shard_tensors(&self, tensor_name: &str) -> Result<&SafeTensors<'static>> {
        let shard = self.shards.shard_for_tensor(tensor_name, &self.tensor_to_shard)?;
        Ok(&shard.tensors)
    }
}

impl<'a> TensorLoader for ShardedTensorLoader<'a> {
    fn has_tensor(&self, name: &str) -> bool {
        self.tensor_to_shard.contains_key(name)
    }

    fn load_tensor(&self, name: &str) -> Result<LoadedTensor> {
        self.load_tensor_view(name).map(LoadedTensorView::into_owned)
    }

    fn load_raw_tensor(&self, name: &str) -> Result<RawTensor> {
        self.load_raw_tensor_view(name).map(RawTensorView::into_owned)
    }

    fn load_tensor_view(&self, name: &str) -> Result<LoadedTensorView<'_>> {
        let tensors = self.shard_tensors(name)?;
        let tensor_view = tensors.tensor(name).map_err(|err| {
            Error::LoadError(format!("Failed to load tensor '{}': {err}", name))
        })?;
        let shape = tensor_view.shape().to_vec();
        let dtype = tensor_view.dtype();
        let raw_data = tensor_view.data();
        let data = convert_to_f32_cow(raw_data, dtype)?;
        Ok(LoadedTensorView { data, shape })
    }

    fn load_raw_tensor_view(&self, name: &str) -> Result<RawTensorView<'_>> {
        let tensors = self.shard_tensors(name)?;
        let tensor_view = tensors.tensor(name).map_err(|err| {
            Error::LoadError(format!("Failed to load tensor '{}': {err}", name))
        })?;
        let shape = tensor_view.shape().to_vec();
        let dtype = tensor_view.dtype();
        let data = Cow::Borrowed(tensor_view.data());
        Ok(RawTensorView { data, shape, dtype })
    }

    fn is_awq_model(&self) -> bool {
        self.has_tensor("model.layers.0.self_attn.q_proj.qweight")
    }
}

pub(crate) fn load_linear<L: TensorLoader>(
    loader: &L,
    weight_name: &str,
    bias_name: Option<&str>,
) -> Result<LinearWeights> {
    if loader.has_tensor(weight_name) {
        let weight_tensor = loader.load_tensor(weight_name)?;
        let weight = weight_tensor.into_weight_matrix()?;

        let bias = if let Some(bias_name) = bias_name {
            if loader.has_tensor(bias_name) {
                let bias_tensor = loader.load_tensor(bias_name)?;
                Some(bias_tensor.into_weight_vector()?)
            } else {
                None
            }
        } else {
            None
        };

        return Ok(LinearWeights::from_dense(weight, bias));
    }

    let prefix = weight_name
        .strip_suffix(".weight")
        .ok_or_else(|| Error::LoadError("Quantized weight name missing .weight suffix".into()))?;

    if loader.is_awq_model() {
        let awq = crate::awq::AwqQuantizedWeight::from_safetensors(loader, prefix)?;
        let bias = if let Some(bias_name) = bias_name {
            if loader.has_tensor(bias_name) {
                let bias_tensor = loader.load_tensor(bias_name)?;
                Some(bias_tensor.into_weight_vector()?.data)
            } else {
                None
            }
        } else {
            None
        };
        let native = NativeQLinear::new(
            QuantizedWeight::Awq {
                weight: awq.weight,
                shape: awq.shape,
            },
            bias,
        )?;
        return Ok(LinearWeights::from_quantized(native));
    }

    Err(Error::LoadError(format!(
        "Linear weight '{}' not found",
        weight_name
    )))
}

pub(crate) fn load_embedding<L: TensorLoader>(loader: &L, weight_name: &str) -> Result<gllm_kernels::WeightMatrix> {
    let weight_tensor = loader.load_tensor(weight_name)?;
    weight_tensor.into_weight_matrix()
}

pub(crate) fn load_layer_norm<L: TensorLoader>(
    loader: &L,
    weight_name: &str,
    bias_name: Option<&str>,
    d_model: usize,
    epsilon: f64,
) -> Result<LayerNormWeights> {
    let gamma_tensor = loader.load_tensor(weight_name)?;
    let gamma = gamma_tensor.into_weight_vector()?;
    if gamma.len() != d_model {
        return Err(Error::LoadError(format!(
            "LayerNorm gamma length mismatch: expected {}, got {}",
            d_model,
            gamma.len()
        )));
    }

    let beta = if let Some(bias_name) = bias_name {
        if loader.has_tensor(bias_name) {
            let beta_tensor = loader.load_tensor(bias_name)?;
            let beta_vec = beta_tensor.into_weight_vector()?;
            if beta_vec.len() != d_model {
                return Err(Error::LoadError(format!(
                    "LayerNorm beta length mismatch: expected {}, got {}",
                    d_model,
                    beta_vec.len()
                )));
            }
            Some(beta_vec)
        } else {
            None
        }
    } else {
        None
    };

    Ok(LayerNormWeights {
        gamma,
        beta,
        eps: epsilon as f32,
    })
}

pub(crate) fn load_mha<L: TensorLoader>(
    loader: &L,
    query_weight: &str,
    query_bias: Option<&str>,
    key_weight: &str,
    key_bias: Option<&str>,
    value_weight: &str,
    value_bias: Option<&str>,
    output_weight: &str,
    output_bias: Option<&str>,
    d_model: usize,
    n_heads: usize,
    _dropout: f64,
) -> Result<MultiHeadAttentionWeights> {
    Ok(MultiHeadAttentionWeights {
        query: load_linear(loader, query_weight, query_bias)?,
        key: load_linear(loader, key_weight, key_bias)?,
        value: load_linear(loader, value_weight, value_bias)?,
        output: load_linear(loader, output_weight, output_bias)?,
        d_model,
        n_heads,
    })
}
