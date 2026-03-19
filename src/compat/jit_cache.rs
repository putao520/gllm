//! Three-level JIT compilation cache (REQ-JIT-CACHE-001/002/003).
//!
//! Level 1: Model-instance cache — `ModelJitCache` held by the caller, lives with the model.
//! Level 2: Process-global LRU cache — `GLOBAL_JIT_CACHE` singleton, capacity 512.
//! Level 3: Disk persistence — `~/.gllm/jit_cache/` binary files with magic + version header.
//!
//! Lookup order: L2 global → L3 disk → JIT compile → write L2 + L3.
//! L1 is managed by the caller (decoder_forward receives &mut ModelJitCache).

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};

use crate::compat::DType;

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

/// Uniquely identifies a model architecture configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelArchKey {
    pub arch_name: String,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub dtype: DType,
}

/// Identifies a specific compiled graph within an architecture.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GraphType {
    QRope,
    Norm2,
    CachedGqa { total_seq: usize },
    MoePreAttn,
    MoeOGemm,
    MoeNorm2,
    KvProjection,
    Gpt2LnQkv,
    Gpt2OProj,
    Gpt2LnMlp,
    Gpt2FinalLnLmHead { vocab_size: usize },
    Gpt2CachedGqa { total_seq: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JitCacheKey {
    pub arch: ModelArchKey,
    pub graph: GraphType,
}

// ---------------------------------------------------------------------------
// Level 1: Model-instance cache
// ---------------------------------------------------------------------------

/// Holds pre-compiled JIT graphs for one model instance.
/// Lives as long as the model is loaded; all inference calls share these graphs.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub struct ModelJitCache {
    /// Standard decoder: RmsNorm + Q + RoPE
    pub q_rope: Option<Arc<gllm_kernels::compiler::CompiledLayer>>,
    /// Standard decoder: pre-FFN RmsNorm
    pub norm2: Option<Arc<gllm_kernels::compiler::CompiledLayer>>,
    /// Standard decoder: KV projection
    pub kv_proj: Option<Arc<gllm_kernels::compiler::CompiledLayer>>,
    /// Standard decoder: CachedGQA per total_seq
    pub gqa_cache: HashMap<usize, Arc<gllm_kernels::compiler::CompiledLayer>>,
    /// MoE decoder: pre-attention graph
    pub moe_pre_attn: Option<Arc<gllm_kernels::compiler::CompiledLayer>>,
    /// MoE decoder: O projection GEMM
    pub moe_o_gemm: Option<Arc<gllm_kernels::compiler::CompiledLayer>>,
    /// MoE decoder: pre-FFN RmsNorm
    pub moe_norm2: Option<Arc<gllm_kernels::compiler::CompiledLayer>>,
    /// MoE decoder: CachedGQA per total_seq
    pub moe_gqa_cache: HashMap<usize, Arc<gllm_kernels::compiler::CompiledLayer>>,
    /// GPT-2: LayerNorm1 + fused QKV
    pub gpt2_ln_qkv: Option<Arc<gllm_kernels::compiler::CompiledLayer>>,
    /// GPT-2: O projection
    pub gpt2_o_proj: Option<Arc<gllm_kernels::compiler::CompiledLayer>>,
    /// GPT-2: LayerNorm2 + MLP
    pub gpt2_ln_mlp: Option<Arc<gllm_kernels::compiler::CompiledLayer>>,
    /// GPT-2: final LayerNorm + lm_head
    pub gpt2_final_ln_lm_head: Option<Arc<gllm_kernels::compiler::CompiledLayer>>,
    /// GPT-2: CachedGQA per total_seq
    pub gpt2_gqa_cache: HashMap<usize, Arc<gllm_kernels::compiler::CompiledLayer>>,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl ModelJitCache {
    pub fn new() -> Self {
        Self {
            q_rope: None,
            norm2: None,
            kv_proj: None,
            gqa_cache: HashMap::new(),
            moe_pre_attn: None,
            moe_o_gemm: None,
            moe_norm2: None,
            moe_gqa_cache: HashMap::new(),
            gpt2_ln_qkv: None,
            gpt2_o_proj: None,
            gpt2_ln_mlp: None,
            gpt2_final_ln_lm_head: None,
            gpt2_gqa_cache: HashMap::new(),
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl std::fmt::Debug for ModelJitCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelJitCache")
            .field("q_rope", &self.q_rope.is_some())
            .field("norm2", &self.norm2.is_some())
            .field("kv_proj", &self.kv_proj.is_some())
            .field("gqa_cache_len", &self.gqa_cache.len())
            .field("moe_pre_attn", &self.moe_pre_attn.is_some())
            .field("moe_o_gemm", &self.moe_o_gemm.is_some())
            .field("moe_norm2", &self.moe_norm2.is_some())
            .field("moe_gqa_cache_len", &self.moe_gqa_cache.len())
            .field("gpt2_ln_qkv", &self.gpt2_ln_qkv.is_some())
            .field("gpt2_o_proj", &self.gpt2_o_proj.is_some())
            .field("gpt2_ln_mlp", &self.gpt2_ln_mlp.is_some())
            .field("gpt2_final_ln_lm_head", &self.gpt2_final_ln_lm_head.is_some())
            .field("gpt2_gqa_cache_len", &self.gpt2_gqa_cache.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Level 2: Process-global LRU cache
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
            // Move to back (most recently used)
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
            // Update existing — move to back
            if let Some(pos) = self.order.iter().position(|k| k == &key) {
                self.order.remove(pos);
            }
            self.order.push_back(key.clone());
            self.map.insert(key, value);
            return;
        }
        // Evict LRU if at capacity
        if self.map.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            }
        }
        self.order.push_back(key.clone());
        self.map.insert(key, value);
    }

    fn len(&self) -> usize {
        self.map.len()
    }
}

// ---------------------------------------------------------------------------
// Level 3: Disk cache
// ---------------------------------------------------------------------------

const DISK_MAGIC: &[u8; 8] = b"GLLMJITC";
const GLLM_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Simple FNV-1a hash for key → directory name (no external deps).
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000003b4c5563);
    }
    hash
}

fn cpu_fingerprint() -> u64 {
    // Encode detected CPU features as a bitmask for cache invalidation.
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
        bits |= 1 << 8; // aarch64 baseline
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
    // jit_cache lives next to models/
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
        // Version mismatch — wipe cache
        let _ = std::fs::remove_dir_all(root);
    }
    let _ = std::fs::create_dir_all(root);
    let _ = std::fs::write(&ver_path, &expected);
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

/// Write a CompiledLayer to disk.
/// Format: [magic 8B][version_len 4B][version bytes][scratchpad 8B][config_hash 8B][code_len 8B][code bytes]
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
    let _ = do_write();
}

/// Read a CompiledLayer from disk. Returns None on any error or version mismatch.
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
// Global cache singleton
// ---------------------------------------------------------------------------

pub struct GlobalJitCache {
    lru: RwLock<LruCache>,
    disk_root: Option<PathBuf>,
}

impl GlobalJitCache {
    fn new() -> Self {
        let disk_root = cache_root().and_then(|root| {
            check_or_init_version(&root);
            Some(root)
        });
        Self {
            lru: RwLock::new(LruCache::new(512)),
            disk_root,
        }
    }

    /// Look up a compiled layer: L2 memory → L3 disk.
    pub fn get(&self, key: &JitCacheKey) -> Option<Arc<gllm_kernels::compiler::CompiledLayer>> {
        // L2: memory
        {
            let mut lru = self.lru.write().ok()?;
            if let Some(layer) = lru.get(key) {
                return Some(layer);
            }
        }
        // L3: disk
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

    /// Insert a compiled layer into L2 and asynchronously write to L3.
    pub fn insert(
        &self,
        key: JitCacheKey,
        layer: gllm_kernels::compiler::CompiledLayer,
    ) -> Arc<gllm_kernels::compiler::CompiledLayer> {
        let arc = Arc::new(layer);
        // Write to disk (synchronous but errors are silenced)
        if let Some(root) = &self.disk_root {
            let path = disk_path(root, &key);
            disk_write(&path, &arc);
        }
        // Insert into L2
        if let Ok(mut lru) = self.lru.write() {
            lru.insert(key, arc.clone());
        }
        arc
    }

    /// Get from cache or compile. Errors from compile_fn propagate; disk/memory errors are silent.
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

    /// Number of entries currently in the L2 memory cache.
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
                arch_name: name.into(),
                hidden_size: 256,
                num_heads: 4,
                num_kv_heads: 4,
                head_dim: 64,
                dtype: DType::F32,
            },
            graph,
        }
    }

    /// TEST-JIT-CACHE-002: same key inserted twice → cache len stays 1
    #[test]
    fn test_global_cache_dedup() {
        let cache = GlobalJitCache::new();
        let key = make_key("test_dedup", GraphType::Norm2);
        // Compile a trivial graph to get a real CompiledLayer
        let layer1 = compile_trivial_layer();
        let layer2 = compile_trivial_layer();
        cache.insert(key.clone(), layer1);
        cache.insert(key.clone(), layer2);
        assert_eq!(cache.len(), 1);
    }

    /// TEST-JIT-CACHE-003: LRU eviction at capacity
    #[test]
    fn test_lru_eviction() {
        let mut lru = LruCache::new(3);
        for i in 0..4usize {
            let key = make_key(&format!("arch{i}"), GraphType::Norm2);
            let layer = compile_trivial_layer();
            lru.insert(key, Arc::new(layer));
        }
        assert_eq!(lru.len(), 3);
    }

    /// TEST-JIT-CACHE-007: disk I/O failure → silent degradation, no error
    #[test]
    fn test_disk_failure_silent() {
        // Point disk root at an unwritable path; get_or_compile must still succeed
        let cache = GlobalJitCache {
            lru: RwLock::new(LruCache::new(512)),
            disk_root: Some(PathBuf::from("/proc/nonexistent_gllm_test")),
        };
        let key = make_key("test_disk_fail", GraphType::QRope);
        let result = cache.get_or_compile(key, || Ok(compile_trivial_layer()));
        assert!(result.is_ok());
    }

    /// TEST-JIT-CACHE-006: version mismatch invalidates disk cache
    #[test]
    fn test_version_mismatch_invalidates() {
        let dir = std::env::temp_dir().join("gllm_jit_test_ver");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Write wrong version
        std::fs::write(dir.join("version"), "0.0.0:0").unwrap();
        // check_or_init_version should wipe and reinit
        check_or_init_version(&dir);
        let ver = std::fs::read_to_string(dir.join("version")).unwrap();
        assert_eq!(ver.trim(), version_string().trim());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// TEST-JIT-CACHE-004/005: disk round-trip preserves code bytes
    #[test]
    fn test_disk_roundtrip() {
        let dir = std::env::temp_dir().join("gllm_jit_test_rt");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        check_or_init_version(&dir);

        let key = make_key("test_rt", GraphType::Norm2);
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

    fn compile_trivial_layer() -> gllm_kernels::compiler::CompiledLayer {
        use gllm_kernels::compiler::{CompilerGraph, InferenceCompiler, OpKind};
        use gllm_kernels::types::DType;
        let mut g = CompilerGraph::new();
        let a = g.add_tensor_concrete("a", &[1, 4], DType::F32);
        let b = g.add_tensor_concrete("b", &[4, 4], DType::F32);
        g.inputs = vec![a, b];
        let c = g.add_tensor_concrete("c", &[1, 4], DType::F32);
        g.add_op(OpKind::Gemm { m: 1, n: 4, k: 4, dtype: DType::F32 }, vec![a, b], vec![c], "g");
        g.outputs = vec![c];
        let mut compiler = InferenceCompiler::new();
        compiler.compile_graph(&g).expect("trivial compile")
    }
}
