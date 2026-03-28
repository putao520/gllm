//! JIT Compilation Cache Protocol (REQ-JIT-CACHE-001~007).
//!
//! **Iron Law**: Compilation only occurs at model load time.
//! The inference hot path (decode step layer loop) MUST NOT contain any
//! `InferenceCompiler::new()`, `compile_graph()`, or `build_*_graph()` calls.
//!
//! # Three-Level Cache Hierarchy
//!
//! | Level | Lifetime             | Hit Scenario                    | Latency       |
//! |:-----:|:---------------------|:--------------------------------|:-------------:|
//! | L1    | Model instance       | Same model, multiple requests   | 0 (ptr deref) |
//! | L2    | Process global (LRU) | Model hot-swap then reload      | ~1μs          |
//! | L3    | Disk persistent      | Process restart, same model     | ~1ms          |
//!
//! # Cache Key = ModelArchKey + GraphType
//!
//! Dynamic dimensions (`seq_len`, `total_seq`) are NOT part of the key — they bind
//! at launch time via `SymDim::Symbolic` + `ShapeBinding`.
//!
//! # Cache Granularity = Full-Layer Fused Graphs
//!
//! Each model compiles a small number of `GraphType` variants (≤5).
//! NO per-operator graph types. NO per-seq_len recompilation.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

use crate::compat::DType;

// ---------------------------------------------------------------------------
// Cache Keys (REQ-JIT-CACHE-002, REQ-JIT-CACHE-007)
// ---------------------------------------------------------------------------

/// Model architecture signature — cache key for L2/L3.
///
/// Dynamic dimensions (seq_len, total_seq) are NOT included.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelArchKey {
    pub model_id: String,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub inter_size: usize,
    pub num_layers: usize,
    pub dtype: DType,
    pub compute_dtype: DType,
    /// Backend kind (Cuda/Rocm/Metal/Cpu) — prevents cross-backend cache poisoning.
    pub backend: crate::backend::BackendType,
    /// ISA version — sm_version (CUDA), gfx_arch (ROCm), gpu_family (Metal),
    /// or cpu_fingerprint() (CPU). Prevents cross-ISA cache poisoning.
    pub isa_version: u32,
}

/// Full-layer fused graph type — the ONLY granularity of cache entries.
///
/// NO per-operator variants (QRope, Norm2, CachedGqa are GONE).
/// Each variant = one complete layer-level fused computation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GraphType {
    /// BERT encoder layer (fused)
    BertLayer,
    /// BERT mean pooling
    BertMeanPool,
}

/// Cache key = architecture + graph type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JitCacheKey {
    pub arch: ModelArchKey,
    pub graph: GraphType,
}

// ---------------------------------------------------------------------------
// Level 1: Model-instance cache (REQ-JIT-CACHE-001)
// ---------------------------------------------------------------------------

/// Pre-compiled full-layer fused graphs for one model instance.
///
/// Compiled at model load time. Zero compilation in the hot path.
/// Dynamic dimensions handled via ShapeBinding at launch time.
// Model-local cache entirely removed in favor of whole-model GraphExecutor
// ---------------------------------------------------------------------------
// Level 2: Process-global LRU cache (REQ-JIT-CACHE-002)
// ---------------------------------------------------------------------------

struct LruCache {
    map: HashMap<JitCacheKey, Arc<gllm_kernels::compiler::CompiledLayer>>,
    order: VecDeque<JitCacheKey>,
    capacity: usize,
}

impl LruCache {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    fn get(&mut self, key: &JitCacheKey) -> Option<Arc<gllm_kernels::compiler::CompiledLayer>> {
        if self.map.contains_key(key) {
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                self.order.remove(pos);
                self.order.push_back(key.clone());
            }
            self.map.get(key).cloned()
        } else {
            None
        }
    }

    fn insert(&mut self, key: JitCacheKey, value: Arc<gllm_kernels::compiler::CompiledLayer>) {
        if self.map.contains_key(&key) {
            if let Some(pos) = self.order.iter().position(|k| k == &key) {
                self.order.remove(pos);
            }
            self.order.push_back(key.clone());
            self.map.insert(key, value);
            return;
        }
        if self.map.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.map.len()
    }
}

// ---------------------------------------------------------------------------
// Level 3: Disk cache (REQ-JIT-CACHE-003)
// ---------------------------------------------------------------------------

const DISK_MAGIC: &[u8; 8] = b"GLLMJITC";
const GLLM_VERSION: &str = env!("CARGO_PKG_VERSION");

fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000003b4c5563);
    }
    hash
}

pub fn cpu_fingerprint() -> u64 {
    let mut bits: u64 = 0;
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2")       { bits |= 1 << 0; }
        if std::is_x86_feature_detected!("avx512f")    { bits |= 1 << 1; }
        if std::is_x86_feature_detected!("avx512bf16") { bits |= 1 << 2; }
        if std::is_x86_feature_detected!("fma")        { bits |= 1 << 4; }
    }
    #[cfg(target_arch = "aarch64")]
    {
        bits |= 1 << 8;
    }
    bits
}

fn cache_root() -> Option<PathBuf> {
    let base = std::env::var("GLLM_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_next_home().unwrap_or_else(|| PathBuf::from("/tmp"))
                .join(".gllm")
                .join("models")
        });
    let root = base.parent()?.join("jit_cache");
    Some(root)
}

fn dirs_next_home() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

fn version_string() -> String {
    format!("{}:{}", GLLM_VERSION, cpu_fingerprint())
}

fn check_or_init_version(root: &std::path::Path) -> bool {
    let ver_path = root.join("version");
    let expected = version_string();
    if let Ok(existing) = std::fs::read_to_string(&ver_path) {
        if existing.trim() == expected.trim() {
            return true;
        }
        let _ = std::fs::remove_dir_all(root).map_err(|e| log::debug!("JIT cache version mismatch, root removal failed: {}", e));
    }
    let _ = std::fs::create_dir_all(root).map_err(|e| log::debug!("JIT cache root creation failed: {}", e));
    let _ = std::fs::write(&ver_path, &expected).map_err(|e| log::debug!("JIT cache version write failed: {}", e));
    true
}

fn disk_path(root: &std::path::Path, key: &JitCacheKey) -> PathBuf {
    let arch_bytes = format!("{:?}", key.arch);
    let graph_bytes = format!("{:?}", key.graph);
    let arch_hash = fnv1a_hash(arch_bytes.as_bytes());
    let graph_hash = fnv1a_hash(graph_bytes.as_bytes());
    root.join(format!("{:016x}", arch_hash))
        .join(format!("{:016x}.bin", graph_hash))
}

fn disk_write(path: &std::path::Path, layer: &gllm_kernels::compiler::CompiledLayer) {
    let do_write = || -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let ver = version_string();
        let ver_bytes = ver.as_bytes();
        let code = layer.code_bytes();
        let mut buf = Vec::with_capacity(8 + 4 + ver_bytes.len() + 8 + 8 + 8 + code.len());
        buf.extend_from_slice(DISK_MAGIC);
        buf.extend_from_slice(&(ver_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(ver_bytes);
        buf.extend_from_slice(&(layer.scratchpad_bytes as u64).to_le_bytes());
        buf.extend_from_slice(&layer.config_hash.to_le_bytes());
        buf.extend_from_slice(&(code.len() as u64).to_le_bytes());
        buf.extend_from_slice(code);
        std::fs::write(path, &buf)
    };
    let _ = do_write().map_err(|e| log::debug!("JIT cache disk write failed for {:?}: {}", path, e));
}

fn disk_read(path: &std::path::Path) -> Option<gllm_kernels::compiler::CompiledLayer> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 8 { return None; }
    if &data[..8] != DISK_MAGIC { return None; }
    let mut off = 8usize;
    if off + 4 > data.len() { return None; }
    let ver_len = u32::from_le_bytes(data[off..off+4].try_into().ok()?) as usize;
    off += 4;
    if off + ver_len > data.len() { return None; }
    let stored_ver = std::str::from_utf8(&data[off..off+ver_len]).ok()?;
    if stored_ver != version_string() { return None; }
    off += ver_len;
    if off + 24 > data.len() { return None; }
    let scratchpad = u64::from_le_bytes(data[off..off+8].try_into().ok()?) as usize;
    off += 8;
    let config_hash = u64::from_le_bytes(data[off..off+8].try_into().ok()?);
    off += 8;
    let code_len = u64::from_le_bytes(data[off..off+8].try_into().ok()?) as usize;
    off += 8;
    if off + code_len > data.len() { return None; }
    let code_bytes = &data[off..off+code_len];
    gllm_kernels::compiler::CompiledLayer::from_code(code_bytes, scratchpad, config_hash).ok()
}

// ---------------------------------------------------------------------------
// L3 Disk Cache TTL Cleanup
// ---------------------------------------------------------------------------

/// Remove stale L3 disk cache entries older than `max_age_secs` seconds.
///
/// Scans `root` for `.bin` files, deleting those with mtime exceeding the TTL.
/// Empty parent directories are cleaned up after deletion.
/// All I/O errors are silently ignored — cache cleanup must never block startup.
fn cleanup_stale_entries(root: &std::path::Path, max_age_secs: u64) {
    let now = std::time::SystemTime::now();
    let max_age = std::time::Duration::from_secs(max_age_secs);

    let walk = || -> std::io::Result<()> {
        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                // Arch-hash subdirectory: scan for .bin files inside
                if let Ok(sub_entries) = std::fs::read_dir(&path) {
                    for sub in sub_entries.flatten() {
                        let bin_path = sub.path();
                        if bin_path.extension().map_or(false, |e| e == "bin") {
                            if let Ok(meta) = std::fs::metadata(&bin_path) {
                                if let Ok(modified) = meta.modified() {
                                    if let Ok(age) = now.duration_since(modified) {
                                        if age > max_age {
                                            let _ = std::fs::remove_file(&bin_path).map_err(|e| log::debug!("JIT cache stale entry deletion failed for {:?}: {}", bin_path, e));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Remove arch-hash dir if now empty
                let _ = std::fs::remove_dir(&path).map_err(|e| {
                    if e.kind() != std::io::ErrorKind::DirectoryNotEmpty {
                        log::debug!("JIT cache clean dir removal failed for {:?}: {}", path, e);
                    }
                });
            }
        }
        Ok(())
    };
    let _ = walk().map_err(|e| log::debug!("JIT cache stale cleanup scan failed: {}", e));
}

// ---------------------------------------------------------------------------
// Global cache singleton (REQ-JIT-CACHE-002)
// ---------------------------------------------------------------------------

pub struct GlobalJitCache {
    lru: RwLock<LruCache>,
    disk_root: Option<PathBuf>,
}

impl GlobalJitCache {
    fn new() -> Self {
        // Debug builds (cargo test / cargo build): disable L3 disk cache entirely.
        // This ensures JIT codegen changes take effect immediately without
        // stale disk-cached binaries masking modifications.
        #[cfg(debug_assertions)]
        let disk_root: Option<PathBuf> = None;

        #[cfg(not(debug_assertions))]
        let disk_root = cache_root().and_then(|root| {
            check_or_init_version(&root);
            // Purge entries older than 7 days on startup
            cleanup_stale_entries(&root, 7 * 24 * 3600);
            Some(root)
        });

        Self {
            lru: RwLock::new(LruCache::new(512)),
            disk_root,
        }
    }

    #[cfg(test)]
    fn with_disk_root(root: Option<PathBuf>) -> Self {
        Self {
            lru: RwLock::new(LruCache::new(512)),
            disk_root: root,
        }
    }

    /// Look up: L2 memory → L3 disk.
    pub fn get(&self, key: &JitCacheKey) -> Option<Arc<gllm_kernels::compiler::CompiledLayer>> {
        {
            let mut lru = self.lru.write().ok()?;
            if let Some(layer) = lru.get(key) {
                return Some(layer);
            }
        }
        if let Some(root) = &self.disk_root {
            let path = disk_path(root, key);
            if let Some(layer) = disk_read(&path) {
                let arc = Arc::new(layer);
                if let Ok(mut lru) = self.lru.write() {
                    lru.insert(key.clone(), arc.clone());
                }
                return Some(arc);
            }
        }
        None
    }

    /// Insert into L2 + L3.
    pub fn insert(
        &self,
        key: JitCacheKey,
        layer: gllm_kernels::compiler::CompiledLayer,
    ) -> Arc<gllm_kernels::compiler::CompiledLayer> {
        let arc = Arc::new(layer);
        if let Some(root) = &self.disk_root {
            let path = disk_path(root, &key);
            disk_write(&path, &arc);
        }
        if let Ok(mut lru) = self.lru.write() {
            lru.insert(key, arc.clone());
        }
        arc
    }

    /// Get or compile. Compile errors propagate; cache errors are silent.
    pub fn get_or_compile(
        &self,
        key: JitCacheKey,
        compile_fn: impl FnOnce() -> Result<gllm_kernels::compiler::CompiledLayer, String>,
    ) -> Result<Arc<gllm_kernels::compiler::CompiledLayer>, String> {
        if let Some(cached) = self.get(&key) {
            return Ok(cached);
        }
        let layer = compile_fn()?;
        Ok(self.insert(key, layer))
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.lru.read().map(|l| l.len()).unwrap_or(0)
    }
}

static GLOBAL_JIT_CACHE: OnceLock<GlobalJitCache> = OnceLock::new();

pub fn global_jit_cache() -> &'static GlobalJitCache {
    GLOBAL_JIT_CACHE.get_or_init(GlobalJitCache::new)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key(name: &str, graph: GraphType) -> JitCacheKey {
        JitCacheKey {
            arch: ModelArchKey {
                model_id: name.into(),
                hidden_size: 256,
                num_heads: 4,
                num_kv_heads: 4,
                head_dim: 64,
                inter_size: 512,
                num_layers: 4,
                dtype: DType::F32,
                compute_dtype: DType::F32,
                backend: crate::backend::BackendType::Cpu,
                isa_version: crate::compat::jit_cache::cpu_fingerprint() as u32,
            },
            graph,
        }
    }

    #[test]
    fn test_global_cache_dedup() {
        let cache = GlobalJitCache::new();
        let key = make_key("test_dedup", GraphType::FusedAttentionLayer);
        cache.insert(key.clone(), compile_trivial_layer());
        cache.insert(key.clone(), compile_trivial_layer());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_eviction() {
        let mut lru = LruCache::new(3);
        for i in 0..4usize {
            let key = make_key(&format!("arch{i}"), GraphType::FusedFfnLayer);
            lru.insert(key, Arc::new(compile_trivial_layer()));
        }
        assert_eq!(lru.len(), 3);
    }

    #[test]
    fn test_disk_failure_silent() {
        let cache = GlobalJitCache {
            lru: RwLock::new(LruCache::new(512)),
            disk_root: Some(PathBuf::from("/proc/nonexistent_gllm_test")),
        };
        let key = make_key("test_disk_fail", GraphType::FusedAttentionLayer);
        let result = cache.get_or_compile(key, || Ok(compile_trivial_layer()));
        assert!(result.is_ok());
    }

    #[test]
    fn test_version_mismatch_invalidates() {
        let dir = std::env::temp_dir().join("gllm_jit_test_ver2");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("version"), "0.0.0:0").unwrap();
        check_or_init_version(&dir);
        let ver = std::fs::read_to_string(dir.join("version")).unwrap();
        assert_eq!(ver.trim(), version_string().trim());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_disk_roundtrip() {
        let dir = std::env::temp_dir().join("gllm_jit_test_rt2");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        check_or_init_version(&dir);

        let key = make_key("test_rt", GraphType::FusedFfnLayer);
        let layer = compile_trivial_layer();
        let path = disk_path(&dir, &key);
        disk_write(&path, &layer);
        let loaded = disk_read(&path);
        assert!(loaded.is_some(), "disk round-trip should succeed");
        let loaded = loaded.unwrap();
        assert_eq!(loaded.scratchpad_bytes, layer.scratchpad_bytes);
        assert_eq!(loaded.config_hash, layer.config_hash);
        assert_eq!(loaded.code_bytes().len(), layer.code_bytes().len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_compiles_once() {
        let cache = GlobalJitCache::with_disk_root(None);
        let key = make_key("test_once", GraphType::FusedAttentionLayer);
        let mut compile_count = 0usize;
        let _r1 = cache.get_or_compile(key.clone(), || {
            compile_count += 1;
            Ok(compile_trivial_layer())
        }).expect("first compile");
        let _r2 = cache.get_or_compile(key.clone(), || {
            compile_count += 1;
            Ok(compile_trivial_layer())
        }).expect("second lookup");
        assert_eq!(compile_count, 1, "compile_fn must be called exactly once");
    }

    fn compile_trivial_layer() -> gllm_kernels::compiler::CompiledLayer {
        use gllm_kernels::compiler::{CompilerGraph, InferenceCompiler, OpKind};
        use gllm_kernels::types::DType;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let b = g.add_tensor_concrete("b", &[4, 4], DType::F32);
        g.inputs = vec![a, b];
        let c = g.add_tensor_concrete("c", &[1, 4], DType::F32);
        g.add_op(OpKind::Gemm { m: 1.into(), n: 4, k: 4, dtype: DType::F32 }, vec![a, b], vec![c], "g");
        g.outputs = vec![c];
        let mut compiler = InferenceCompiler::new();
        compiler.compile_graph(&g).expect("trivial compile")
    }

    #[test]
    fn test_cleanup_stale_entries() {
        let dir = std::env::temp_dir().join("gllm_jit_test_cleanup");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Create a fake arch-hash subdirectory with a .bin file
        let arch_dir = dir.join("deadbeef01234567");
        std::fs::create_dir_all(&arch_dir).unwrap();
        let bin_file = arch_dir.join("cafebabe.bin");
        std::fs::write(&bin_file, b"fake_jit_code").unwrap();

        // Set mtime to 8 days ago using filetime crate equivalent
        // We can't easily set mtime in pure std, so test with max_age_secs=0
        // which should delete ANY file regardless of age.
        assert!(bin_file.exists());
        cleanup_stale_entries(&dir, 0); // 0 seconds = delete everything
        assert!(!bin_file.exists(), "stale .bin file should be deleted");
        // arch_dir should also be removed since it's now empty
        assert!(!arch_dir.exists(), "empty arch dir should be removed");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_debug_mode_disables_disk() {
        // In debug builds (cargo test), GlobalJitCache::new() sets disk_root = None
        let cache = GlobalJitCache::new();
        #[cfg(debug_assertions)]
        assert!(cache.disk_root.is_none(), "debug mode must disable L3 disk cache");
        #[cfg(not(debug_assertions))]
        {
            // In release mode, disk_root should be Some (if HOME is set)
            if std::env::var("HOME").is_ok() {
                assert!(cache.disk_root.is_some(), "release mode should enable L3 disk cache");
            }
        }
    }
}
