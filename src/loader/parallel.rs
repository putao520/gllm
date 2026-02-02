//! Parallel loading helpers (rayon).

use rayon::prelude::*;
use std::path::PathBuf;

use super::Result;

#[derive(Debug, Clone, Copy)]
pub struct ParallelLoader {
    enabled: bool,
}

impl ParallelLoader {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn map_paths<T, F>(&self, paths: &[PathBuf], f: F) -> Result<Vec<T>>
    where
        T: Send,
        F: Fn(&PathBuf) -> Result<T> + Sync,
    {
        let results: Vec<Result<T>> = if self.enabled {
            paths.par_iter().map(|path| f(path)).collect()
        } else {
            paths.iter().map(|path| f(path)).collect()
        };

        let mut out = Vec::with_capacity(results.len());
        for result in results {
            out.push(result?);
        }
        Ok(out)
    }
}

pub fn enforce_parallel(is_moe: bool, requested: bool) -> bool {
    if is_moe {
        true
    } else {
        requested
    }
}

pub fn decide_parallel(is_moe: bool, requested: bool, shards: usize) -> bool {
    if is_moe {
        true
    } else {
        requested && shards > 1
    }
}
