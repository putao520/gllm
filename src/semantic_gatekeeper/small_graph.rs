//! Level Keys 预计算 + 知识文本编码使用的小 CompilerGraph (SPEC §3.3).
//!
//! 两个小图复用主模型的 `InferenceCompiler` + 完整 JIT 管线
//! (ARCH-FULL-JIT + ARCH-CPU-GPU-UNIFIED 合规):
//!
//! - **EmbedLookupOnlyGraph**: `Gather(embed_weight)`
//!   用于 (1) Level Keys 预计算的首段,(2) 运行时知识文本编码.
//!
//! - **KProjOnlyGraph@layer_L**: `RmsNorm(input_layernorm @L) → Gemm(k_proj @L)`
//!   仅用于 Level Keys 预计算的末段. 拆分为两个 ops 以利用现有 codegen 路径.
//!
//! 所有维度严格使用 `SymDim::Symbolic { name: "seq_len", ... }` 穿透执行期,
//! 描述文本 token 数以 ABI `seq_len` 参数运行时绑定 (SPEC §3.4, SymDim 穿透
//! 铁律).

use std::sync::Arc;

use gllm_kernels::compiler::graph::SYMDIM_MAX_SEQ_LEN;
use gllm_kernels::compiler::{CompiledLayer, CompilerGraph, InferenceCompiler, OpKind, SymDim};
use gllm_kernels::types::DType;

use super::SemanticGatekeeperError;

// ============================================================================
// 内部工具: SymDim 构造与 f32 bytes 编码
// ============================================================================

fn sym_seq() -> SymDim {
    SymDim::Symbolic {
        name: "seq_len".to_string(),
        max_value: Some(SYMDIM_MAX_SEQ_LEN),
    }
}

/// 编码 u32 token id 序列为 JIT Gather 期待的索引字节流.
///
/// `lower_gather` (gllm-kernels) 的 ScalarLoad 使用 `vmovss → vmovd` 将
/// 4 字节原封不动搬到 GPR 作为整数值使用 (IntMulStride 做整数乘法).
/// 因此 host 侧必须写入原始 u32 位模式, 不能转为 f32 浮点值.
fn encode_indices_as_u32_bytes(tokens: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * std::mem::size_of::<u32>());
    for &t in tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    out
}

fn align_up(n: usize, align: usize) -> usize {
    if align <= 1 {
        n
    } else {
        (n + align - 1) / align * align
    }
}

/// 按 64 字节对齐准备 scratchpad. 运行时 JIT 可能依赖 SIMD 向量寄存器对齐访问.
fn alloc_scratchpad(required: usize) -> Vec<u8> {
    // Over-allocate + align 保障返回 slice 满足 AVX-512 对齐要求.
    let aligned = align_up(required.max(64), 64);
    vec![0u8; aligned]
}

// ============================================================================
// EmbedLookupOnlyGraph
// ============================================================================

/// `Gather` 算子的全 JIT 小图.
///
/// 编译一次后 `encode_tokens` 可以任意次执行 (seq_len 经 ABI `seq_len` 参数
/// 运行时绑定, 不触发重编译).
pub struct EmbedLookupOnlyGraph {
    pub(super) hidden_size: usize,
    pub(super) vocab_size: usize,
    pub(super) dtype: DType,
    /// JIT 编译产物 (mmap executable + scratchpad 元数据).
    compiled: CompiledLayer,
    /// Embed table 权重字节 blob (按主模型 Token Embedding 冻结传入).
    ///
    /// Shape: `[vocab_size, hidden_size]` row-major, `dtype` 元素.
    /// 由 `Arc<[u8]>` 持有以在多次 `encode_tokens` 调用间共享指针稳定性.
    embed_weight_blob: Arc<[u8]>,
}

impl std::fmt::Debug for EmbedLookupOnlyGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbedLookupOnlyGraph")
            .field("hidden_size", &self.hidden_size)
            .field("vocab_size", &self.vocab_size)
            .field("dtype", &self.dtype)
            .field("scratchpad_bytes", &self.compiled.scratchpad_bytes)
            .field("weight_blob_bytes", &self.embed_weight_blob.len())
            .finish()
    }
}

impl EmbedLookupOnlyGraph {
    /// 构造 `CompilerGraph` 并走完整 JIT 管线.
    ///
    /// - `embed_weight`: `[vocab_size, hidden_size]` row-major 字节流, 元素类型 `dtype`.
    pub fn build_and_compile(
        hidden_size: usize,
        vocab_size: usize,
        dtype: DType,
        embed_weight: &[u8],
    ) -> Result<Self, SemanticGatekeeperError> {
        if hidden_size == 0 || vocab_size == 0 {
            return Err(SemanticGatekeeperError::SmallGraph(format!(
                "invalid dims: hidden_size={hidden_size} vocab_size={vocab_size}"
            )));
        }
        let expected_bytes = vocab_size
            .checked_mul(hidden_size)
            .and_then(|n| n.checked_mul(dtype.size_bytes()))
            .ok_or_else(|| {
                SemanticGatekeeperError::SmallGraph(
                    "embed_weight size overflow".to_string(),
                )
            })?;
        if embed_weight.len() != expected_bytes {
            return Err(SemanticGatekeeperError::SmallGraph(format!(
                "embed_weight byte length mismatch: got {} expected {}",
                embed_weight.len(),
                expected_bytes
            )));
        }

        let mut g = CompilerGraph::new();

        let indices = g.add_tensor("sg_indices", vec![sym_seq()], DType::F32);
        let table = g.add_tensor_concrete("sg_embed_table", &[vocab_size, hidden_size], dtype);
        let output = g.add_tensor(
            "sg_embed_out",
            vec![sym_seq(), SymDim::Concrete(hidden_size)],
            dtype,
        );

        g.inputs = vec![indices, table];
        g.outputs = vec![output];

        g.add_op(
            OpKind::Gather {
                table_rows: vocab_size,
                embed_dim: hidden_size,
                index_dim: sym_seq(),
                indices_kind: Default::default(),
            },
            vec![indices, table],
            vec![output],
            "sg_embed_gather",
        );

        let mut compiler = InferenceCompiler::new();
        let compiled = compiler
            .compile_graph(&g)
            .map_err(|e| SemanticGatekeeperError::SmallGraph(format!("Gather compile failed: {e}")))?;
        compiler.print_resource_report();

        Ok(Self {
            hidden_size,
            vocab_size,
            dtype,
            compiled,
            embed_weight_blob: Arc::<[u8]>::from(embed_weight.to_vec().into_boxed_slice()),
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 对 token id 序列执行 embedding lookup, 返回 `seq_len × hidden_size` 字节.
    ///
    /// 调用流程:
    ///  1. 将 `tokens` (u32) 按 f32 编码为 `indices_bytes` (JIT 约定).
    ///  2. 分配 output buffer (`seq_len × hidden_size × dtype.size_bytes()`).
    ///  3. 分配 scratchpad (`compiled.scratchpad_bytes`).
    ///  4. `CompiledLayer::execute` with 10 参数 ABI.
    pub fn encode_tokens(&self, tokens: &[u32]) -> Result<Vec<u8>, SemanticGatekeeperError> {
        if tokens.is_empty() {
            return Err(SemanticGatekeeperError::SmallGraph(
                "encode_tokens: empty tokens".to_string(),
            ));
        }
        for (i, &t) in tokens.iter().enumerate() {
            if (t as usize) >= self.vocab_size {
                return Err(SemanticGatekeeperError::SmallGraph(format!(
                    "token[{i}]={t} >= vocab_size={}",
                    self.vocab_size
                )));
            }
        }
        if tokens.len() > SYMDIM_MAX_SEQ_LEN {
            return Err(SemanticGatekeeperError::SmallGraph(format!(
                "tokens.len()={} exceeds compile-time max {}",
                tokens.len(),
                SYMDIM_MAX_SEQ_LEN
            )));
        }

        let seq_len = tokens.len();
        let indices_bytes = encode_indices_as_u32_bytes(tokens);
        let out_bytes = seq_len * self.hidden_size * self.dtype.size_bytes();
        let mut output = vec![0u8; out_bytes];
        let mut scratchpad = alloc_scratchpad(self.compiled.scratchpad_bytes);

        // SAFETY: 所有指针均来自 host Vec, 生命周期与本函数调用期一致;
        // execute 是同步调用, 不会在返回后持有指针. batch_size=1, seq_len=tokens.len()
        // 经由 ABI arg6 传入,JIT 的 SymDim::Symbolic("seq_len") 通过 SymDimSlotMap
        // 解析到同一槽位.
        unsafe {
            self.compiled.execute(
                indices_bytes.as_ptr(),
                self.embed_weight_blob.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                1,
                seq_len,
                output.as_mut_ptr(),
                scratchpad.as_mut_ptr(),
            );
        }
        Ok(output)
    }
}

// ============================================================================
// EmbedTextEncoder — TextEncoder implementation using EmbedLookupOnlyGraph
// ============================================================================

/// Encodes text to `hidden_size`-dim f32 vector via tokenize → embed → mean-pool.
///
/// Uses the main model's frozen embedding table to ensure `v_knowledge`
/// lives in the same semantic space as hidden states (SPEC §8.6).
pub struct EmbedTextEncoder {
    graph: EmbedLookupOnlyGraph,
    tokenizer: Box<dyn super::TokenizerEncoder>,
    hidden_size: usize,
    dtype: DType,
}

impl EmbedTextEncoder {
    pub fn new(
        graph: EmbedLookupOnlyGraph,
        tokenizer: Box<dyn super::TokenizerEncoder>,
        hidden_size: usize,
        dtype: DType,
    ) -> Self {
        Self { graph, tokenizer, hidden_size, dtype }
    }
}

impl super::callback::TextEncoder for EmbedTextEncoder {
    fn encode(&self, text: &str) -> Result<Vec<f32>, super::callback::TextEncoderError> {
        let tokens = self.tokenizer.encode(text)
            .map_err(|e| super::callback::TextEncoderError::Tokenize(format!("{e}")))?;
        if tokens.is_empty() {
            return Err(super::callback::TextEncoderError::Tokenize("empty tokens".into()));
        }
        let bytes = self.graph.encode_tokens(&tokens)
            .map_err(|e| super::callback::TextEncoderError::Execute(format!("{e}")))?;
        // Mean-pool: bytes layout is [seq_len, hidden_size] × dtype_size.
        let seq_len = tokens.len();
        let elem_size = self.dtype.size_bytes();
        let hs = self.hidden_size;
        let mut result = vec![0.0f32; hs];
        for s in 0..seq_len {
            for h in 0..hs {
                let byte_off = (s * hs + h) * elem_size;
                let val: f32 = match self.dtype {
                    DType::F32 => {
                        let mut buf = [0u8; 4];
                        buf.copy_from_slice(&bytes[byte_off..byte_off + 4]);
                        f32::from_le_bytes(buf)
                    }
                    DType::F16 => {
                        let mut buf = [0u8; 2];
                        buf.copy_from_slice(&bytes[byte_off..byte_off + 2]);
                        half::f16::from_le_bytes(buf).to_f32()
                    }
                    DType::BF16 => {
                        let mut buf = [0u8; 2];
                        buf.copy_from_slice(&bytes[byte_off..byte_off + 2]);
                        half::bf16::from_le_bytes(buf).to_f32()
                    }
                    _ => return Err(super::callback::TextEncoderError::Execute(
                        format!("unsupported dtype {:?}", self.dtype),
                    )),
                };
                result[h] += val;
            }
        }
        let inv = 1.0 / seq_len as f32;
        for x in result.iter_mut() {
            *x *= inv;
        }
        Ok(result)
    }
}

// ============================================================================
// KProjOnlyGraph
// ============================================================================

/// `RmsNorm → Gemm` 两阶段小图, 对应某检测层的 `k_proj(RmsNorm(hidden))`.
///
/// 内部拆为两个独立的单 op JIT 图 (RmsNorm + Gemm), 分别编译执行,
/// 避免 multi-op 图的 scratchpad intermediate tensor ABI 映射问题.
pub struct KProjOnlyGraph {
    pub(super) layer_idx: usize,
    pub(super) hidden_size: usize,
    pub(super) kv_dim: usize,
    pub(super) rms_eps: f32,
    pub(super) dtype: DType,
    norm_compiled: CompiledLayer,
    gemm_compiled: CompiledLayer,
    norm_weight: Arc<[u8]>,
    gemm_weight: Arc<[u8]>,
}

impl std::fmt::Debug for KProjOnlyGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KProjOnlyGraph")
            .field("layer_idx", &self.layer_idx)
            .field("hidden_size", &self.hidden_size)
            .field("kv_dim", &self.kv_dim)
            .field("rms_eps", &self.rms_eps)
            .field("dtype", &self.dtype)
            .field("norm_scratchpad", &self.norm_compiled.scratchpad_bytes)
            .field("gemm_scratchpad", &self.gemm_compiled.scratchpad_bytes)
            .finish()
    }
}

impl KProjOnlyGraph {
    /// 构造两个独立的单 op CompilerGraph 并走完整 JIT 管线.
    pub fn build_and_compile(
        layer_idx: usize,
        hidden_size: usize,
        kv_dim: usize,
        rms_eps: f32,
        dtype: DType,
        input_layernorm_weight: &[u8],
        k_proj_weight: &[u8],
    ) -> Result<Self, SemanticGatekeeperError> {
        if hidden_size == 0 || kv_dim == 0 {
            return Err(SemanticGatekeeperError::SmallGraph(format!(
                "invalid dims: hidden={hidden_size} kv_dim={kv_dim}"
            )));
        }
        if !rms_eps.is_finite() || rms_eps <= 0.0 {
            return Err(SemanticGatekeeperError::SmallGraph(format!(
                "invalid rms_eps={rms_eps}"
            )));
        }

        let elem = dtype.size_bytes();
        let ln_expected = hidden_size * elem;
        let kp_expected = hidden_size
            .checked_mul(kv_dim)
            .and_then(|n| n.checked_mul(elem))
            .ok_or_else(|| {
                SemanticGatekeeperError::SmallGraph("k_proj size overflow".to_string())
            })?;

        if input_layernorm_weight.len() != ln_expected {
            return Err(SemanticGatekeeperError::SmallGraph(format!(
                "input_layernorm_weight byte length mismatch: got {} expected {}",
                input_layernorm_weight.len(),
                ln_expected
            )));
        }
        if k_proj_weight.len() != kp_expected {
            return Err(SemanticGatekeeperError::SmallGraph(format!(
                "k_proj_weight byte length mismatch: got {} expected {}",
                k_proj_weight.len(),
                kp_expected
            )));
        }

        // ── Stage 1: RmsNorm(input, norm_w) → norm_out ──
        let mut gn = CompilerGraph::new();
        let norm_in = gn.add_tensor(
            "sg_norm_input",
            vec![sym_seq(), SymDim::Concrete(hidden_size)],
            dtype,
        );
        let norm_w = gn.add_tensor_concrete("sg_norm_w", &[hidden_size], dtype);
        let norm_out = gn.add_tensor(
            "sg_norm_out",
            vec![sym_seq(), SymDim::Concrete(hidden_size)],
            dtype,
        );
        gn.inputs = vec![norm_in, norm_w];
        gn.outputs = vec![norm_out];
        gn.add_op(
            OpKind::RmsNorm { eps: rms_eps },
            vec![norm_in, norm_w],
            vec![norm_out],
            "sg_rmsnorm",
        );

        let mut compiler = InferenceCompiler::new();
        let norm_compiled = compiler.compile_graph(&gn).map_err(|e| {
            SemanticGatekeeperError::SmallGraph(format!(
                "RmsNorm compile failed for layer {layer_idx}: {e}"
            ))
        })?;
        compiler.print_resource_report();

        // ── Stage 2: Gemm(norm_out, k_w) → k_out ──
        let mut gg = CompilerGraph::new();
        let gemm_in = gg.add_tensor(
            "sg_gemm_input",
            vec![sym_seq(), SymDim::Concrete(hidden_size)],
            dtype,
        );
        let k_w = gg.add_tensor_concrete("sg_gemm_k_w", &[hidden_size, kv_dim], dtype);
        let k_out = gg.add_tensor(
            "sg_gemm_out",
            vec![sym_seq(), SymDim::Concrete(kv_dim)],
            dtype,
        );
        gg.inputs = vec![gemm_in, k_w];
        gg.outputs = vec![k_out];
        gg.add_op(
            OpKind::Gemm {
                m: sym_seq(),
                n: kv_dim,
                k: hidden_size,
                dtype,
            },
            vec![gemm_in, k_w],
            vec![k_out],
            "sg_gemm",
        );

        let gemm_compiled = compiler.compile_graph(&gg).map_err(|e| {
            SemanticGatekeeperError::SmallGraph(format!(
                "Gemm compile failed for layer {layer_idx}: {e}"
            ))
        })?;

        Ok(Self {
            layer_idx,
            hidden_size,
            kv_dim,
            rms_eps,
            dtype,
            norm_compiled,
            gemm_compiled,
            norm_weight: Arc::<[u8]>::from(input_layernorm_weight.to_vec().into_boxed_slice()),
            gemm_weight: Arc::<[u8]>::from(k_proj_weight.to_vec().into_boxed_slice()),
        })
    }

    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }
    pub fn rms_eps(&self) -> f32 {
        self.rms_eps
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 对 embedding 张量执行 `RmsNorm → Gemm`, 返回 `seq_len × kv_dim` 字节.
    ///
    /// `embed_bytes` 必须是 `[seq_len, hidden_size]` row-major, `dtype` 元素布局,
    /// 通常由 `EmbedLookupOnlyGraph::encode_tokens` 产出.
    pub fn run_on_embed(&self, embed_bytes: &[u8]) -> Result<Vec<u8>, SemanticGatekeeperError> {
        let elem = self.dtype.size_bytes();
        if elem == 0 || self.hidden_size == 0 {
            return Err(SemanticGatekeeperError::SmallGraph(
                "run_on_embed: zero element or hidden_size".to_string(),
            ));
        }
        let row_bytes = self.hidden_size * elem;
        if embed_bytes.len() % row_bytes != 0 {
            return Err(SemanticGatekeeperError::SmallGraph(format!(
                "embed_bytes ({}) not divisible by row_bytes ({})",
                embed_bytes.len(),
                row_bytes
            )));
        }
        let seq_len = embed_bytes.len() / row_bytes;
        if seq_len == 0 {
            return Err(SemanticGatekeeperError::SmallGraph(
                "run_on_embed: seq_len=0".to_string(),
            ));
        }
        if seq_len > SYMDIM_MAX_SEQ_LEN {
            return Err(SemanticGatekeeperError::SmallGraph(format!(
                "seq_len={} exceeds compile-time max {}",
                seq_len, SYMDIM_MAX_SEQ_LEN
            )));
        }

        // Stage 1: RmsNorm
        let norm_out_bytes = seq_len * self.hidden_size * elem;
        let mut norm_output = vec![0u8; norm_out_bytes];
        let mut norm_scratch = alloc_scratchpad(self.norm_compiled.scratchpad_bytes);
        unsafe {
            self.norm_compiled.execute(
                embed_bytes.as_ptr(),
                self.norm_weight.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                1,
                seq_len,
                norm_output.as_mut_ptr(),
                norm_scratch.as_mut_ptr(),
            );
        }

        // Stage 2: Gemm
        let out_bytes = seq_len * self.kv_dim * elem;
        let mut output = vec![0u8; out_bytes];
        let mut gemm_scratch = alloc_scratchpad(self.gemm_compiled.scratchpad_bytes);
        unsafe {
            self.gemm_compiled.execute(
                norm_output.as_ptr(),
                self.gemm_weight.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                1,
                seq_len,
                output.as_mut_ptr(),
                gemm_scratch.as_mut_ptr(),
            );
        }

        Ok(output)
    }
}
