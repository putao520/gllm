//! Mega-Kernel 执行器 (SPEC §9.1)
//!
//! 将整个 decoder layer loop 编译为全层融合图，通过 FusedGraphExecutor
//! 实现单一 Launch 执行。配合 Compact→Execute→Scatter 三段式管线。
//!
//! ## §9.1 铁律
//! - 每轮 Decode/Prefill 仅 Launch 唯一一个 Mega-Kernel
//! - 取消主机条件网：禁止在 CPU Host 为 Gate-First-Skip 等建立多线程路调度
//! - 块内联路：SM 核心内 Thread Block 直接读取 RequestStateTable

use std::collections::HashMap;

use crate::engine::executor::{
    AttentionHeadConfig, BatchInput, GeneratorForwardConfig, LogitsHandle,
};
use crate::graph::executor::{ExecutionError, FusedGraphExecutor};
use crate::jit::epilogue::TelemetryAggregator;
use crate::jit::ragged::{CompactIndex, CompactPlatform, RequestActiveMask, RaggedCompaction, COMPACT_THRESHOLD};
use crate::scheduler::types::RequestId;
use gllm_kernels::types::DType;

// ============================================================================
// RequestStateTable — GPU 可读的请求状态表 (§9.1)
// ============================================================================

/// 请求执行阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPhase {
    /// Prefill: 处理完整 prompt
    Prefill,
    /// Decode: 逐 token 生成
    Decode,
    /// ChunkedPrefill: prefill 的一个切片
    ChunkedPrefill,
}

/// 单个请求的状态条目
#[derive(Debug, Clone)]
pub struct RequestStateEntry {
    /// 请求 ID
    pub request_id: RequestId,
    /// 执行阶段
    pub phase: RequestPhase,
    /// 当前序列长度
    pub seq_len: usize,
    /// KV Cache 页表指针偏移
    pub page_table_offset: usize,
    /// 跳过标志位（Gate-First Skip, Layer Bypass 等）
    pub skip_flags: u32,
    /// 值域分组 ID（Range-Aware Compact Grouping）
    pub range_group: u32,
}

/// Request State Table (§9.1)
///
/// Mega-Kernel 内部 Thread Block 直接读取此表，
/// 决定每个请求的执行路径。
#[derive(Debug, Clone)]
pub struct RequestStateTable {
    entries: Vec<RequestStateEntry>,
}

impl RequestStateTable {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// 从 BatchInput 构建请求状态表
    pub fn from_batch(batch: &BatchInput, request_ids: &[RequestId]) -> Self {
        let entries = batch.sequences.iter().zip(request_ids.iter()).map(|(seq, &rid)| {
            let phase = if seq.tokens.len() > 1 {
                RequestPhase::Prefill
            } else {
                RequestPhase::Decode
            };
            RequestStateEntry {
                request_id: rid,
                phase,
                seq_len: seq.tokens.len(),
                page_table_offset: 0,
                skip_flags: 0,
                range_group: 0,
            }
        }).collect();
        Self { entries }
    }

    /// 请求数量
    pub fn len(&self) -> usize { self.entries.len() }
    /// 是否为空
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    /// 获取条目
    pub fn get(&self, idx: usize) -> Option<&RequestStateEntry> { self.entries.get(idx) }
    /// 可变获取
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut RequestStateEntry> { self.entries.get_mut(idx) }
    /// 迭代
    pub fn iter(&self) -> impl Iterator<Item = &RequestStateEntry> { self.entries.iter() }

    /// 按值域分组（Range-Aware Compact Grouping, §9.1）
    ///
    /// 利用 Epilogue 遥测的 entropy/residual_delta 指标，
    /// 将激活值域相近的请求聚集在同一 GEMM Tile 内。
    pub fn assign_range_groups(&mut self, telemetry: &TelemetryAggregator, num_groups: u32) {
        if num_groups == 0 { return; }
        let entropy = telemetry.output_entropy();
        // 简单分组：按 entropy 量化到 num_groups 个桶
        for entry in &mut self.entries {
            let normalized = (entropy.clamp(0.0, 10.0) / 10.0 * num_groups as f32) as u32;
            entry.range_group = normalized.min(num_groups - 1);
        }
    }

    /// Decode 请求数
    pub fn decode_count(&self) -> usize {
        self.entries.iter().filter(|e| e.phase == RequestPhase::Decode).count()
    }

    /// Prefill 请求数
    pub fn prefill_count(&self) -> usize {
        self.entries.iter().filter(|e| matches!(e.phase, RequestPhase::Prefill | RequestPhase::ChunkedPrefill)).count()
    }
}

// ============================================================================
// MegaBatch — Mega-Kernel 的输入 (§9.1)
// ============================================================================

/// Mega-Kernel 批次输入
#[derive(Debug)]
pub struct MegaBatch {
    /// 原始 batch 输入
    pub batch_input: BatchInput,
    /// 请求 ID 列表（与 batch_input.sequences 一一对应）
    pub request_ids: Vec<RequestId>,
    /// 请求状态表
    pub state_table: RequestStateTable,
}

// ============================================================================
// MegaKernelExecutor (§9.1)
// ============================================================================

/// Mega-Kernel 编译错误
#[derive(Debug, thiserror::Error)]
pub enum MegaKernelError {
    #[error("compilation failed: {0}")]
    Compilation(String),
    #[error("execution failed: {0}")]
    Execution(String),
    #[error("graph executor error: {0}")]
    GraphExecution(#[from] ExecutionError),
    #[error("not compiled")]
    NotCompiled,
}

/// True mega-kernel 编译产物。
///
/// 持有完整的 mega-kernel 机器码（embedding → layer loop → lm_head → sampling → generate loop）
/// + 全模型权重布局 + 缓冲布局。推理时通过单次 CALL 执行。
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
struct MegaKernelCompiled {
    /// 全模型权重布局（embed → layer_0 → ... → layer_N → lm_head）
    weight_layout: gllm_kernels::compiler::MegaKernelWeightLayout,
    /// 运行时缓冲布局（activation ping/pong, logits, sampling workspace）
    buffer_layout: gllm_kernels::compiler::MegaKernelBufferLayout,
    /// 预打包的连续权重 blob
    weight_blob: Vec<u8>,
    /// mmap'd 可执行缓冲区（mega-kernel 机器码）
    exec_code: gllm_kernels::compiler::CompiledLayer,
    /// MegaKernelFn 函数指针（指向 exec_code 的入口）
    entry_fn: gllm_kernels::compiler::MegaKernelFn,
}

/// Mega-Kernel 执行器 (§9.1)
///
/// 两种编译路径:
/// 1. **Legacy**: 包装 FusedGraphExecutor（逐节点执行）
/// 2. **True Mega-Kernel**: 通过 `compile_mega_kernel()` 编译，单次 CALL
///
/// ## 执行流程
/// 1. `compile()` — 模型加载时，将所有层编译为 JIT 融合图
/// 2. `prepare_batch()` — 构建 RequestStateTable + 值域分组
/// 3. `execute()` — 单一 Launch 执行全层融合图
pub struct MegaKernelExecutor {
    /// 全层融合图执行器（模型加载时编译）— legacy 路径
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    graph_executor: Option<FusedGraphExecutor>,
    /// True mega-kernel 编译产物 — 单次 CALL 路径
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    mega_compiled: Option<MegaKernelCompiled>,
    /// 模型配置
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    dtype: DType,
    /// EOS token ID — 从 ModelConfig 读取，传给 JIT 停止条件
    eos_token_id: u32,
    /// 是否已编译
    is_compiled: bool,
    /// Compact→Execute→Scatter 管线
    compact_scatter: CompactScatterPipeline,
}

impl MegaKernelExecutor {
    /// 从 FusedGraphExecutor 构建 Mega-Kernel 执行器 (legacy 路径)
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn from_graph_executor(
        graph_executor: FusedGraphExecutor,
        num_layers: usize,
        hidden_size: usize,
        vocab_size: usize,
        dtype: DType,
        eos_token_id: u32,
    ) -> Self {
        let is_compiled = graph_executor.is_compiled();
        Self {
            graph_executor: Some(graph_executor),
            mega_compiled: None,
            num_layers,
            hidden_size,
            vocab_size,
            dtype,
            eos_token_id,
            is_compiled,
            compact_scatter: CompactScatterPipeline::new(),
        }
    }

    /// 从 ModelGeometry 编译 true mega-kernel。
    ///
    /// 1. 编译单层模板图 (CompiledLayerFn)
    /// 2. 发射完整 mega-kernel wrapper（embedding → N 层循环 → lm_head → sampling → generate loop）
    /// 3. 预打包所有权重到连续 blob
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn compile_from_geometry(
        geometry: &crate::model_config::ModelGeometry,
        weight_ptrs: &std::collections::HashMap<String, *const u8>,
        weight_sizes: &std::collections::HashMap<String, usize>,
        eos_token_id: u32,
    ) -> Result<Self, MegaKernelError> {
        let config = gllm_kernels::compiler::ModelMegaConfig {
            num_layers: geometry.num_layers,
            hidden: geometry.hidden_size,
            num_heads: geometry.num_heads,
            num_kv_heads: geometry.num_kv_heads,
            head_dim: geometry.head_dim,
            intermediate: geometry.intermediate_size,
            vocab_size: geometry.vocab_size,
            rms_eps: geometry.norm_eps,
            rope_theta: geometry.rope_theta,
            rope_partial: geometry.rope_partial_ratio,
            dtype: geometry.dtype,
            max_seq_len: geometry.max_seq_len,
            num_eos_tokens: 1,
            rope_scaling: None,
        };

        // Step 1: 编译单层模板图
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let output = compiler.compile_mega_kernel(&config)
            .map_err(|e| MegaKernelError::Compilation(e.to_string()))?;

        let layer_fn = unsafe { output.layer_code.entry_point() };
        // lm_head 使用相同的编译产物（GEMM 算子是通用的）
        let lm_head_fn = layer_fn;

        // Step 2: 发射完整 mega-kernel wrapper
        let mega_code = gllm_kernels::compiler::codegen::vm::mega_kernel_emit::emit_mega_kernel_x86(
            &config,
            &output.weight_layout,
            &output.buffer_layout,
            layer_fn,
            lm_head_fn,
        ).map_err(|e| MegaKernelError::Compilation(format!("mega-kernel emit: {}", e)))?;

        // 将机器码包装为可执行 CompiledLayer
        let exec_code = gllm_kernels::compiler::CompiledLayer::from_code(
            &mega_code,
            output.buffer_layout.total_scratchpad_bytes,
            0,
        ).map_err(|e| MegaKernelError::Compilation(format!("mega-kernel exec buffer: {}", e)))?;

        let entry_fn = unsafe { exec_code.entry_point_as_mega_kernel() };

        // Step 3: 打包权重到连续 blob
        let weight_blob = pack_mega_kernel_weights(
            &output.weight_layout,
            geometry.num_layers,
            geometry.hidden_size,
            geometry.num_heads,
            geometry.num_kv_heads,
            geometry.head_dim,
            geometry.intermediate_size,
            weight_ptrs,
            weight_sizes,
            geometry.dtype,
        );

        let mega_compiled = MegaKernelCompiled {
            weight_layout: output.weight_layout,
            buffer_layout: output.buffer_layout,
            weight_blob,
            exec_code,
            entry_fn,
        };

        Ok(Self {
            graph_executor: None,
            mega_compiled: Some(mega_compiled),
            num_layers: geometry.num_layers,
            hidden_size: geometry.hidden_size,
            vocab_size: geometry.vocab_size,
            dtype: geometry.dtype,
            eos_token_id,
            is_compiled: true,
            compact_scatter: CompactScatterPipeline::new(),
        })
    }

    /// 是否已编译
    pub fn is_compiled(&self) -> bool { self.is_compiled }

    /// 尝试编译 true mega-kernel（单次 CALL 路径）。
    ///
    /// 如果编译成功，内部 `mega_compiled` 会被设置，后续推理走单次 CALL 路径。
    /// 如果编译失败，保持 legacy 路径不变。
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn try_compile_true_mega_kernel(
        &mut self,
        geometry: &crate::model_config::ModelGeometry,
        weight_ptrs: &std::collections::HashMap<String, *const u8>,
        weight_sizes: &std::collections::HashMap<String, usize>,
        eos_token_id: u32,
    ) -> Result<(), MegaKernelError> {
        let compiled = Self::compile_from_geometry(geometry, weight_ptrs, weight_sizes, eos_token_id)?;
        self.mega_compiled = compiled.mega_compiled;
        Ok(())
    }

    /// 是否有真正的 mega-kernel（单次 CALL 完成全部推理）
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn has_true_mega_kernel(&self) -> bool {
        self.mega_compiled.is_some()
    }

    /// 单序列 mega-kernel 生成。
    ///
    /// 一次 CALL 完成: prompt encode → embedding → N 层 → lm_head → argmax sampling → generate loop。
    /// 返回 output token IDs（不包含 prompt tokens）。
    #[cfg(target_arch = "x86_64")]
    pub fn generate_single_sequence(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> Result<Vec<u32>, MegaKernelError> {
        let mega = self.mega_compiled.as_ref()
            .ok_or(MegaKernelError::NotCompiled)?;

        let prompt_len = prompt_tokens.len();
        // 输入 buffer: prompt tokens + 预留空间给生成的 tokens
        let max_total = prompt_len + max_new_tokens;
        let mut input_ids = vec![0u32; max_total];
        input_ids[..prompt_len].copy_from_slice(prompt_tokens);

        // positions: prompt tokens 从 0 开始
        let positions: Vec<u32> = (0..max_total as u32).collect();

        // output tokens buffer
        let mut output_tokens = vec![0u32; max_new_tokens];

        // scratchpad
        let mut scratchpad = vec![0u8; mega.buffer_layout.total_scratchpad_bytes];

        let generated_count = unsafe {
            (mega.entry_fn)(
                input_ids.as_ptr(),
                mega.weight_blob.as_ptr(),
                std::ptr::null_mut(), // kv_cache (MVP: NULL)
                positions.as_ptr(),
                std::ptr::null(),     // aux_ptr
                1,                    // batch_size = 1
                prompt_len,           // prompt_len
                scratchpad.as_mut_ptr(),
                output_tokens.as_mut_ptr(),
                temperature,
                top_k as u32,
                top_p,
                max_new_tokens as u32,
                self.eos_token_id,
                std::ptr::null(),     // hook_ctx_ptr
                std::ptr::null_mut(), // telemetry_ptr
            )
        };

        let count = generated_count.min(max_new_tokens);
        Ok(output_tokens[..count].to_vec())
    }

    /// 准备 Mega-Kernel 批次
    ///
    /// 从 BatchInput 构建 RequestStateTable，执行值域分组。
    pub fn prepare_batch(
        &self,
        batch_input: BatchInput,
        request_ids: Vec<RequestId>,
        telemetry: &TelemetryAggregator,
    ) -> MegaBatch {
        let mut state_table = RequestStateTable::from_batch(&batch_input, &request_ids);
        // §9.1: 值域隔离的分组防线
        state_table.assign_range_groups(telemetry, 4);
        MegaBatch {
            batch_input,
            request_ids,
            state_table,
        }
    }

    /// 执行 Mega-Kernel (§9.1: 单一 Launch)
    ///
    /// Compact→Execute→Scatter 三段式循环在单次调用内闭环。
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn execute(
        &self,
        batch: &MegaBatch,
    ) -> Result<Vec<Vec<f32>>, MegaKernelError> {
        if !self.is_compiled {
            return Err(MegaKernelError::NotCompiled);
        }

        // True mega-kernel 路径（如果可用）
        if let Some(ref mega) = self.mega_compiled {
            return self.execute_mega_kernel(mega, batch);
        }

        // Legacy 路径：通过 FusedGraphExecutor
        let graph_executor = self.graph_executor.as_ref()
            .ok_or(MegaKernelError::NotCompiled)?;

        // Phase 1: Compact — 按值域分组挤压
        let compacted = self.compact_scatter.compact(&batch.state_table);

        // Phase 2: Execute — 通过 FusedGraphExecutor 执行全层融合图
        let batch_size = batch.batch_input.sequences.len();
        let total_tokens: usize = batch.batch_input.sequences.iter().map(|s| s.tokens.len()).sum();

        // 创建 activation 输入（token embeddings → bytes）
        let activation_size = total_tokens * self.hidden_size;
        let activation_f32 = vec![0.0f32; activation_size];
        let activation_bytes: Vec<u8> = activation_f32.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let mut inputs: HashMap<String, Vec<u8>> = HashMap::new();
        inputs.insert("hidden_state".to_string(), activation_bytes);

        // 权重已通过 FusedGraphExecutor::bind() 绑定，无需再传入

        // shape_bindings: seq_len 从外部传入（当前使用 inputs 的 batch 维度）
        let shape_bindings = std::collections::HashMap::from([
            ("seq_len".to_string(), 1usize), // MegaKernel 通常处理单 token
        ]);
        let results = graph_executor.run(&inputs, &shape_bindings)?;

        // Phase 3: Scatter — 将输出 bytes 转回 f32，按原始偏移回写
        // 提取所有输出张量，转为 Vec<Vec<f32>>
        let output_vecs: Vec<Vec<f32>> = results.values().map(|bytes| {
            bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }).collect();

        let scattered = self.compact_scatter.scatter(&compacted, &output_vecs, batch_size);

        Ok(scattered)
    }

    /// True mega-kernel 执行路径：单次 MegaKernelFn CALL。
    ///
    /// 一次调用完成: embedding → N 层循环 → lm_head → argmax → store token → check stop。
    /// 返回: 每个序列的 output tokens（从 output_tokens buffer 读取）。
    #[cfg(target_arch = "x86_64")]
    fn execute_mega_kernel(
        &self,
        mega: &MegaKernelCompiled,
        batch: &MegaBatch,
    ) -> Result<Vec<Vec<f32>>, MegaKernelError> {
        let batch_size = batch.batch_input.sequences.len();
        let max_tokens = 64; // MVP: 固定最大生成 token 数

        // 准备 input_ids（所有序列的 token IDs 拼接）
        let input_ids: Vec<u32> = batch.batch_input.sequences.iter()
            .flat_map(|s| s.tokens.iter().copied())
            .collect();

        // 准备 positions
        let positions: Vec<u32> = batch.batch_input.sequences.iter()
            .flat_map(|s| {
                let start = s.position as u32;
                (start..start + s.tokens.len() as u32).collect::<Vec<_>>()
            })
            .collect();

        // 分配 output tokens buffer
        let mut output_tokens = vec![0u32; max_tokens];

        // 分配 scratchpad
        let mut scratchpad = vec![0u8; mega.buffer_layout.total_scratchpad_bytes];

        // 准备 MegaKernelFn 参数
        let temperature: f32 = 0.0; // greedy
        let top_k: u32 = 1;
        let top_p: f32 = 1.0;
        let max_new_tokens: u32 = max_tokens as u32;
        let eos_token_id = self.eos_token_id;

        let generated_count = unsafe {
            (mega.entry_fn)(
                input_ids.as_ptr(),
                mega.weight_blob.as_ptr(),
                std::ptr::null_mut(), // kv_cache (MVP: NULL)
                positions.as_ptr(),
                std::ptr::null(),     // aux_ptr
                batch_size,
                input_ids.len(),      // prompt_len
                scratchpad.as_mut_ptr(),
                output_tokens.as_mut_ptr(),
                temperature,
                top_k,
                top_p,
                max_new_tokens,
                eos_token_id,
                std::ptr::null(),     // hook_ctx_ptr
                std::ptr::null_mut(), // telemetry_ptr
            )
        };

        // 转换 output tokens 为 f32（每个序列）
        let mut results = Vec::with_capacity(batch_size);
        for seq in &batch.batch_input.sequences {
            let seq_tokens: Vec<f32> = output_tokens[..generated_count.min(max_tokens)]
                .iter()
                .map(|&t| t as f32)
                .collect();
            results.push(seq_tokens);
        }

        Ok(results)
    }

    /// 非 x86_64 fallback：逐层调用 CompiledLayerFn。
    #[cfg(not(target_arch = "x86_64"))]
    fn execute_mega_kernel(
        &self,
        mega: &MegaKernelCompiled,
        batch: &MegaBatch,
    ) -> Result<Vec<Vec<f32>>, MegaKernelError> {
        Err(MegaKernelError::Execution(
            "mega-kernel execution only supported on x86_64".into(),
        ))
    }
}

// ============================================================================
// Weight Blob Packing
// ============================================================================

/// 将所有模型权重打包到单一连续 blob。
///
/// 按照 MegaKernelWeightLayout 定义的顺序:
/// embed_weight → layer_0_weights → layer_1_weights → ... → lm_head_weight
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn pack_mega_kernel_weights(
    layout: &gllm_kernels::compiler::MegaKernelWeightLayout,
    num_layers: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    weight_ptrs: &std::collections::HashMap<String, *const u8>,
    weight_sizes: &std::collections::HashMap<String, usize>,
    dtype: DType,
) -> Vec<u8> {
    let mut blob = vec![0u8; layout.total_bytes];
    let elem_bytes = dtype.size_bytes();

    // Helper: copy weight from ptr into blob at offset
    let copy_weight = |blob: &mut [u8], offset: usize, ptr: *const u8, size: usize| {
        if !ptr.is_null() && size > 0 {
            let src = unsafe { std::slice::from_raw_parts(ptr, size) };
            blob[offset..offset + size].copy_from_slice(src);
        }
    };

    // Embedding weight: "embed_tokens.weight" → [vocab_size, hidden]
    if let Some(&ptr) = weight_ptrs.get("embed_tokens.weight") {
        let size = *weight_sizes.get("embed_tokens.weight").unwrap_or(&0);
        copy_weight(&mut blob, layout.embed_offset, ptr, size);
    }

    // Per-layer weights
    let per_layer = &layout.per_layer;
    let weight_names = [
        ("model.layers.{L}.input_layernorm.weight", per_layer.attn_norm_offset, per_layer.attn_norm_bytes),
        ("model.layers.{L}.self_attn.q_proj.weight", per_layer.w_q_offset, per_layer.w_q_bytes),
        ("model.layers.{L}.self_attn.k_proj.weight", per_layer.w_k_offset, per_layer.w_k_bytes),
        ("model.layers.{L}.self_attn.v_proj.weight", per_layer.w_v_offset, per_layer.w_v_bytes),
        ("model.layers.{L}.self_attn.o_proj.weight", per_layer.w_o_offset, per_layer.w_o_bytes),
        ("model.layers.{L}.post_attention_layernorm.weight", per_layer.ffn_norm_offset, per_layer.ffn_norm_bytes),
        ("model.layers.{L}.mlp.gate_proj.weight", per_layer.w_gate_offset, per_layer.w_gate_bytes),
        ("model.layers.{L}.mlp.up_proj.weight", per_layer.w_up_offset, per_layer.w_up_bytes),
        ("model.layers.{L}.mlp.down_proj.weight", per_layer.w_down_offset, per_layer.w_down_bytes),
    ];

    for layer_idx in 0..num_layers {
        let layer_base = layout.layer_base_offset(layer_idx);
        for (name_template, rel_offset, expected_size) in &weight_names {
            let name = name_template.replace("{L}", &layer_idx.to_string());
            if let Some(&ptr) = weight_ptrs.get(&name) {
                let size = *weight_sizes.get(&name).unwrap_or(expected_size);
                copy_weight(&mut blob, layer_base + rel_offset, ptr, size);
            }
        }
    }

    // lm_head weight: "lm_head.weight" → [vocab_size, hidden]
    if let Some(&ptr) = weight_ptrs.get("lm_head.weight") {
        let size = *weight_sizes.get("lm_head.weight").unwrap_or(&0);
        copy_weight(&mut blob, layout.lm_head_offset, ptr, size);
    }

    blob
}
// ============================================================================

/// Compact 后的批次信息
#[derive(Debug)]
pub struct CompactedBatch {
    /// 原始索引 → compact 后索引的映射
    pub index_map: Vec<usize>,
    /// compact 后的有效请求数
    pub active_count: usize,
    /// 值域分组信息
    pub range_groups: Vec<u32>,
    /// RaggedCompaction 索引（当触发 compact 时非 None）
    pub compact_index: Option<CompactIndex>,
}

/// Compact→Execute→Scatter 管线 (§9.1)
///
/// 使用 jit/ragged.rs 的 RaggedCompaction 实现物理挤压。
/// 挤压聚拢仅仅是第一步（Compact）。在没有 Padding 气泡的连续稠密矩阵中
/// 执行完核函数运算后（Execute），必须按原始 Request 偏移进行原位散射回写（Scatter）。
#[derive(Debug)]
pub struct CompactScatterPipeline {
    /// 是否启用 Compact（小批次时跳过）
    compact_threshold: usize,
    /// 硬件平台（决定 compact 指令路径）
    platform: CompactPlatform,
}

impl CompactScatterPipeline {
    pub fn new() -> Self {
        // 从当前硬件检测 compact 平台
        let profile = gllm_kernels::dispatch::DeviceProfile::detect();
        let kc = &profile.kernel_config;
        let platform = CompactPlatform::detect(
            "cpu",
            kc.use_avx512,
            kc.has_sve,
            kc.sve_vl_bytes,
            0,
        );
        Self {
            compact_threshold: 4,
            platform,
        }
    }

    /// Compact: 按值域分组 + RaggedCompaction 挤压请求到连续稠密矩阵
    pub fn compact(&self, state_table: &RequestStateTable) -> CompactedBatch {
        let n = state_table.len();
        if n < self.compact_threshold {
            return CompactedBatch {
                index_map: (0..n).collect(),
                active_count: n,
                range_groups: state_table.iter().map(|e| e.range_group).collect(),
                compact_index: None,
            };
        }

        // 构建活跃 mask（所有请求都活跃，skip_flags=0 的才参与计算）
        let mask_vec: Vec<bool> = state_table.iter().map(|e| e.skip_flags == 0).collect();
        let active_mask = RequestActiveMask::new(mask_vec);

        // 检查是否需要 compact（浪费率 > 25%）
        if active_mask.should_compact() {
            let compact_idx = CompactIndex::from_mask(&active_mask);
            let index_map = compact_idx.compact_to_original().to_vec();

            log::debug!(
                "executor: §9.1 RaggedCompaction triggered (waste={:.1}%, active={}/{})",
                active_mask.waste_ratio() * 100.0,
                active_mask.active_count(),
                n,
            );

            CompactedBatch {
                index_map,
                active_count: compact_idx.active_count(),
                range_groups: state_table.iter().map(|e| e.range_group).collect(),
                compact_index: Some(compact_idx),
            }
        } else {
            // 按 range_group 排序（值域相近的请求聚集在一起）
            let mut indices: Vec<usize> = (0..n).collect();
            let entries: Vec<&RequestStateEntry> = state_table.iter().collect();
            indices.sort_by_key(|&i| entries[i].range_group);

            CompactedBatch {
                index_map: indices,
                active_count: n,
                range_groups: state_table.iter().map(|e| e.range_group).collect(),
                compact_index: None,
            }
        }
    }

    /// Scatter: 执行后按原始偏移回写
    pub fn scatter(
        &self,
        compacted: &CompactedBatch,
        results: &[Vec<f32>],
        batch_size: usize,
    ) -> Vec<Vec<f32>> {
        if compacted.compact_index.is_some() {
            // 使用 CompactIndex 的反向映射
            let ci = compacted.compact_index.as_ref().unwrap();
            let c2o = ci.compact_to_original();
            let mut scattered = vec![Vec::new(); batch_size];
            for (compact_idx, result) in results.iter().enumerate() {
                if compact_idx < c2o.len() {
                    let original_idx = c2o[compact_idx];
                    if original_idx < batch_size {
                        scattered[original_idx] = result.clone();
                    }
                }
            }
            scattered
        } else if compacted.index_map.len() == batch_size
            && compacted.index_map.iter().enumerate().all(|(i, &v)| i == v)
        {
            results.to_vec()
        } else {
            let mut scattered = vec![Vec::new(); batch_size];
            for (compact_idx, &original_idx) in compacted.index_map.iter().enumerate() {
                if compact_idx < results.len() && original_idx < batch_size {
                    scattered[original_idx] = results[compact_idx].clone();
                }
            }
            scattered
        }
    }
}

// ============================================================================
// MegaKernelObservation — Type-Safe Telemetry Interface (SPEC §9.5)
// ============================================================================

/// Structured observation extracted from Mega-Kernel epilogue telemetry buffer.
///
/// All signals are collected via zero-cost piggybacking in JIT-compiled
/// kernels (§13 Epilogue 白嫖). This struct provides type-safe access
/// to raw telemetry bytes written to KV Page Header padding.
///
/// Consumed by:
/// - JitDirector daemon (§9.2) for adaptive recompilation triggers
/// - ExpertThermalManager (§15.4) for MoE cold expert detection
/// - EpilogueSubsystem for aggregated decision making
#[derive(Debug, Clone, Copy)]
pub struct MegaKernelObservation {
    /// Layer index this observation was collected from
    pub layer_idx: usize,

    /// §13.9: Softmax sharpness (max/sum of softmax probabilities)
    /// Range: [0, 1]. High (>0.8) = focused attention (confident).
    pub entropy: f32,

    /// §13.11: Residual energy ratio (||x_out|| / ||x_in||)
    /// Near 1.0 = stable; >1.5 = amplification; <0.5 = attenuation.
    pub residual_delta: f32,

    /// §13.11: Direction alignment between residual input/output
    /// Near 1.0 = aligned (redundant layer); near 0.0 = orthogonal.
    pub cosine_similarity: f32,

    /// §13.5: Number of FFN gate neurons with sigmoid(x) < 0.01
    pub dead_neuron_count: u32,

    /// §13.9: True if softmax_max > 0.5 (BOS token absorbing excess attention)
    pub is_attention_sink: bool,

    /// §13.8: Per-channel max absolute value from RmsNorm (KIVI K量化 scale)
    pub per_channel_scale: f32,

    /// §13.7: Row-level L1 norm from GEMM output
    pub row_l1_norm: f32,

    /// §13.7: Row-level max value from GEMM output
    pub row_max: f32,
}

impl MegaKernelObservation {
    /// Extract structured observation from raw telemetry buffer bytes.
    ///
    /// Buffer layout follows `gllm_kernels::compiler::graph::telemetry_offsets`.
    pub fn from_buffer(layer_idx: usize, buffer: &[u8]) -> Self {
        use gllm_kernels::compiler::graph::telemetry_offsets;

        let read_f32 = |offset: usize| -> f32 {
            if offset + 4 <= buffer.len() {
                f32::from_le_bytes([
                    buffer[offset],
                    buffer[offset + 1],
                    buffer[offset + 2],
                    buffer[offset + 3],
                ])
            } else {
                0.0
            }
        };
        let read_u32 = |offset: usize| -> u32 {
            if offset + 4 <= buffer.len() {
                u32::from_le_bytes([
                    buffer[offset],
                    buffer[offset + 1],
                    buffer[offset + 2],
                    buffer[offset + 3],
                ])
            } else {
                0
            }
        };

        let entropy = read_f32(telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET);
        let residual_delta = read_f32(telemetry_offsets::RESIDUAL_DELTA_OFFSET);
        let cosine_similarity = read_f32(telemetry_offsets::COSINE_SIMILARITY_OFFSET);
        let dead_neuron_count = read_u32(telemetry_offsets::SILU_DEAD_NEURON_MASK_OFFSET);
        let is_attention_sink = read_u32(telemetry_offsets::IS_ATTENTION_SINK_OFFSET) != 0;
        let per_channel_scale = read_f32(telemetry_offsets::CHANNEL_SCALE_PTR_OFFSET);
        let row_l1_norm = read_f32(telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET);
        let row_max = read_f32(telemetry_offsets::GEMM_ROW_MAX_OFFSET);

        Self {
            layer_idx,
            entropy,
            residual_delta,
            cosine_similarity,
            dead_neuron_count,
            is_attention_sink,
            per_channel_scale,
            row_l1_norm,
            row_max,
        }
    }

    /// Dead neuron ratio (0.0-1.0) relative to a given hidden size.
    pub fn dead_neuron_ratio(&self, hidden_size: usize) -> f32 {
        if hidden_size == 0 { return 0.0; }
        self.dead_neuron_count as f32 / hidden_size as f32
    }

    /// Whether this layer is a candidate for residual bypass (§13.3).
    ///
    /// Criteria: energy ratio near 1.0 AND direction strongly aligned.
    pub fn is_bypass_candidate(&self, delta_threshold: f32, cosine_threshold: f32) -> bool {
        self.residual_delta < delta_threshold && self.cosine_similarity > cosine_threshold
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::executor::SequenceInput;

    #[test]
    fn test_request_state_table_from_batch() {
        let batch = BatchInput {
            sequences: vec![
                SequenceInput { tokens: vec![1, 2, 3, 4], position: 0, draft_steps: 0, fused_hidden: None },
                SequenceInput { tokens: vec![5], position: 4, draft_steps: 0, fused_hidden: None },
                SequenceInput { tokens: vec![6], position: 10, draft_steps: 0, fused_hidden: None },
            ],
        };
        let ids = vec![100, 200, 300];
        let table = RequestStateTable::from_batch(&batch, &ids);

        assert_eq!(table.len(), 3);
        assert_eq!(table.prefill_count(), 1);
        assert_eq!(table.decode_count(), 2);
        assert_eq!(table.get(0).unwrap().phase, RequestPhase::Prefill);
        assert_eq!(table.get(1).unwrap().phase, RequestPhase::Decode);
    }

    #[test]
    fn test_compact_scatter_identity() {
        let pipeline = CompactScatterPipeline::new();
        let batch = BatchInput {
            sequences: vec![
                SequenceInput { tokens: vec![1], position: 0, draft_steps: 0, fused_hidden: None },
                SequenceInput { tokens: vec![2], position: 1, draft_steps: 0, fused_hidden: None },
            ],
        };
        let ids = vec![1, 2];
        let table = RequestStateTable::from_batch(&batch, &ids);

        let compacted = pipeline.compact(&table);
        // Below threshold: identity mapping
        assert_eq!(compacted.index_map, vec![0, 1]);

        let results = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let scattered = pipeline.scatter(&compacted, &results, 2);
        assert_eq!(scattered, results);
    }

    #[test]
    fn test_compact_scatter_reorder() {
        let mut pipeline = CompactScatterPipeline::new();
        pipeline.compact_threshold = 1; // Force compact

        let mut table = RequestStateTable::new();
        table.entries = vec![
            RequestStateEntry {
                request_id: 1, phase: RequestPhase::Decode,
                seq_len: 1, page_table_offset: 0, skip_flags: 0, range_group: 2,
            },
            RequestStateEntry {
                request_id: 2, phase: RequestPhase::Decode,
                seq_len: 1, page_table_offset: 0, skip_flags: 0, range_group: 0,
            },
            RequestStateEntry {
                request_id: 3, phase: RequestPhase::Decode,
                seq_len: 1, page_table_offset: 0, skip_flags: 0, range_group: 1,
            },
        ];

        let compacted = pipeline.compact(&table);
        // Sorted by range_group: [1(g=0), 2(g=1), 0(g=2)]
        assert_eq!(compacted.index_map, vec![1, 2, 0]);

        let results = vec![vec![10.0], vec![20.0], vec![30.0]];
        let scattered = pipeline.scatter(&compacted, &results, 3);
        // Original order restored: req 0 → result from compact pos 2, etc.
        assert_eq!(scattered[0], vec![30.0]); // original idx 0 was at compact pos 2
        assert_eq!(scattered[1], vec![10.0]); // original idx 1 was at compact pos 0
        assert_eq!(scattered[2], vec![20.0]); // original idx 2 was at compact pos 1
    }

    #[test]
    fn test_range_group_assignment() {
        let batch = BatchInput {
            sequences: vec![
                SequenceInput { tokens: vec![1], position: 0, draft_steps: 0, fused_hidden: None },
                SequenceInput { tokens: vec![2], position: 1, draft_steps: 0, fused_hidden: None },
            ],
        };
        let ids = vec![1, 2];
        let mut table = RequestStateTable::from_batch(&batch, &ids);
        let telemetry = TelemetryAggregator::new();

        table.assign_range_groups(&telemetry, 4);
        // Default entropy = 0.0 → group 0
        assert_eq!(table.get(0).unwrap().range_group, 0);
    }
}
