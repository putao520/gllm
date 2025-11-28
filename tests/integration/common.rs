#[cfg(feature = "async")]
use gllm::AsyncClient;
use gllm::{Client, ClientConfig, Device, Error, Result};
use safetensors::Dtype;
use safetensors::tensor::{TensorView, serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

pub(crate) const EMBEDDING_DIM: usize = 128;

pub(crate) fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("env lock")
}

pub(crate) fn init_test_env() {
    unsafe {
        std::env::set_var("GLLM_TEST_MODE", "1");
        std::env::set_var("GLLM_SKIP_DOWNLOAD", "1");
        std::env::set_var("HF_HUB_OFFLINE", "1");
    }
}

pub(crate) fn preferred_device() -> Device {
    if cfg!(feature = "cpu") {
        Device::Cpu
    } else {
        Device::Auto
    }
}

pub(crate) fn write_dummy_weights(path: &Path) {
    let weights: Vec<u8> = vec![0u8; 64];
    let shape = vec![4usize, 4usize];
    let tensor = TensorView::new(Dtype::F32, shape, &weights).expect("tensor view");
    let data = serialize([("dense.weight", tensor)].into_iter(), &None).expect("serialize");
    fs::write(path, data).expect("write weights");
}

pub(crate) struct TempContext {
    pub config: ClientConfig,
    pub repo_dir: PathBuf,
    pub _temp_dir: tempfile::TempDir,
}

pub(crate) fn prepare_context(model: &str, device: Device) -> Result<(Client, TempContext)> {
    init_test_env();

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let mut config = ClientConfig::default();
    config.models_dir = temp_dir.path().to_path_buf();
    config.device = device;

    let client = Client::with_config(model, config.clone())?;
    let repo_dir = discover_repo_dir(&config.models_dir)?;

    Ok((
        client,
        TempContext {
            config,
            repo_dir,
            _temp_dir: temp_dir,
        },
    ))
}

pub(crate) fn prepare_context_with_weights(
    model: &str,
    device: Device,
) -> Result<(Client, TempContext)> {
    let (_client, ctx) = prepare_context(model, device)?;
    let weights = ctx.repo_dir.join("model.safetensors");
    write_dummy_weights(&weights);

    let refreshed = Client::with_config(model, ctx.config.clone())?;
    Ok((refreshed, ctx))
}

#[cfg(feature = "async")]
pub(crate) async fn prepare_async_context_with_weights(
    model: &str,
    device: Device,
) -> Result<(AsyncClient, TempContext)> {
    let (_client, ctx) = prepare_context_with_weights(model, device)?;
    let client = AsyncClient::with_config(model, ctx.config.clone()).await?;
    Ok((client, ctx))
}

fn discover_repo_dir(base: &Path) -> Result<PathBuf> {
    for entry in fs::read_dir(base)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            return Ok(entry.path());
        }
    }

    Err(Error::LoadError("Model directory was not created".into()))
}

pub(crate) fn is_backend_unavailable(err: &Error) -> bool {
    matches!(err, Error::InvalidConfig(_))
}
