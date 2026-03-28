//! Macroscopic Unified Graph Execution Cache Protocol
//! 
//! Replaces legacy operator-level cache lock mechanism.
//! 
//! Implements REQ-JIT-CACHE-003~007: persistent disk hashing
//! for the unified FusedGraph execution pipeline.

use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{SystemTime, Duration};
use crate::graph::types::FusedGraph;

pub struct ArtifactCache {
    cache_dir: PathBuf,
}

impl ArtifactCache {
    /// Create a new Artifact Cache, defaulting to ~/.gllm/jit_cache/
    pub fn new(cache_dir: Option<PathBuf>) -> Self {
        let dir = cache_dir.unwrap_or_else(|| {
            let mut p = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            p.push(".gllm");
            p.push("jit_cache");
            p
        });
        
        let mut ac = Self { cache_dir: dir };
        
        #[cfg(not(debug_assertions))]
        let _ = ac.cleanup_ttl_cache(Duration::from_secs(7 * 24 * 3600)); // 7 days TTL

        ac
    }

    /// Calculate MAC fingerprintf for the macro-graph + hardware signature
    pub fn get_blueprint_hash(&self, model_id: &str, graph: &FusedGraph, hardware_fingerprint: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(model_id.as_bytes());
        hasher.update(b"|");
        
        // Use Debug print of the graph structure as structural unique key
        // This inherently binds to all node connections and parameters mapped in FusedOps.
        let graph_content = format!("{:?}", graph);
        hasher.update(graph_content.as_bytes());
        hasher.update(b"|");
        hasher.update(hardware_fingerprint.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Load pure binary artifacts using the hashed blueprint key.
    pub fn load_blob(&self, hash: &str) -> Option<Vec<u8>> {
        if cfg!(debug_assertions) {
            return None; // Disable cache hits in debug mode to force recompile
        }
        
        let mut path = self.cache_dir.clone();
        path.push(format!("{}.bin", hash));
        fs::read(&path).ok()
    }

    /// Atomically save pure binary artifacts to disk using the hashed blueprint key.
    pub fn save_blob(&self, hash: &str, blob: &[u8]) -> std::io::Result<()> {
        if cfg!(debug_assertions) {
            return Ok(()); // Disable cache writes in debug mode
        }
        if let Err(e) = fs::create_dir_all(&self.cache_dir) {
            eprintln!("[ArtifactCache] Failed to create cache dir: {}", e);
            return Err(e);
        }
        
        let mut path = self.cache_dir.clone();
        path.push(format!("{}.bin", hash));
        
        let mut temp_path = path.clone();
        temp_path.set_extension("tmp");
        
        {
            let mut file = fs::File::create(&temp_path)?;
            file.write_all(blob)?;
        }
        fs::rename(&temp_path, &path)?;
        Ok(())
    }

    fn cleanup_ttl_cache(&mut self, ttl: Duration) -> std::io::Result<()> {
        if !self.cache_dir.exists() {
            return Ok(());
        }
        
        let now = SystemTime::now();
        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("bin") {
                if let Ok(metadata) = entry.metadata() {
                    if let Ok(mtime) = metadata.modified() {
                        if let Ok(duration) = now.duration_since(mtime) {
                            if duration > ttl {
                                let _ = fs::remove_file(&path);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
