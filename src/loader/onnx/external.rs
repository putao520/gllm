use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::MmapOptions;
use prost::bytes::Bytes;

use super::{LoaderError, Result};

#[derive(Debug)]
pub(super) struct ExternalDataResolver {
    base_dir: PathBuf,
    cache: HashMap<PathBuf, Bytes>,
}

impl ExternalDataResolver {
    pub(super) fn new(model_path: &Path) -> Self {
        let base_dir = model_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Self {
            base_dir,
            cache: HashMap::new(),
        }
    }

    pub(super) fn resolve(
        &mut self,
        location: &str,
        offset: usize,
        length: usize,
    ) -> Result<Bytes> {
        let path = self.base_dir.join(location);
        let bytes = self.mmap_file(&path)?;
        let end = offset.checked_add(length).ok_or_else(|| {
            LoaderError::Onnx(format!("external data offset overflow for {location}"))
        })?;
        if end > bytes.len() {
            return Err(LoaderError::Onnx(format!(
                "external data slice out of bounds for {location}"
            )));
        }
        Ok(bytes.slice(offset..end))
    }

    fn mmap_file(&mut self, path: &Path) -> Result<Bytes> {
        if let Some(cached) = self.cache.get(path) {
            return Ok(cached.clone());
        }
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let bytes = Bytes::from_owner(mmap);
        self.cache.insert(path.to_path_buf(), bytes.clone());
        Ok(bytes)
    }
}
