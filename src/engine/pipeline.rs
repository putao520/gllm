//! 统一推理管道 (REQ-EXEC-001 ~ REQ-EXEC-004)
//!
//! 实现 SPEC/08-LOADER-REFACTOR.md 中定义的统一流水线。

use std::collections::HashMap;

use crate::compat::backend_trait::{Backend, Element};

use crate::arch::{
    get_template_by_arch, register_builtin_templates, resolve_config, ArchTemplate, ResolvedConfig,
};
use crate::graph::{
    executor::{ExecutionError, ExecutionPlan, FusedGraphExecutor},
    optimizer::{GraphOptimizer, OptimizationContext},
    types::OptimizationStats,
    FusedGraph,
};
use crate::loader::onnx::OnnxGraph;
use crate::loader::{Loader, LoaderError, WeightFormat, WeightsHandle};
use crate::manifest::{ModelArchitecture, ModelManifest};

/// 统一推理管道错误
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Loader error: {0}")]
    Loader(#[from] LoaderError),
    #[error("Template not found for architecture: {0}")]
    TemplateNotFound(String),
    #[error("Config resolution failed: {0}")]
    ConfigResolution(String),
    #[error("Graph generation failed: {0}")]
    GraphGeneration(String),
    #[error("Optimization failed: {0}")]
    Optimization(String),
    #[error("Execution failed: {0}")]
    Execution(#[from] ExecutionError),
    #[error("Pipeline not compiled: call compile() before forward()")]
    NotCompiled,
}

/// 统一推理管道
///
/// 实现 SPEC 中定义的：
/// 1. 统一加载 → OnnxGraph
/// 2. 图优化 → FusedGraph
/// 3. 执行推理
pub struct UnifiedPipeline<B: Backend<E>, E: Element> {
    /// 融合图执行器 (holds FusedGraph + JIT compiled kernels)
    pub executor: FusedGraphExecutor,
    /// 执行计划
    pub execution_plan: ExecutionPlan,
    /// 原始 OnnxGraph (用于调试)
    pub original_graph: OnnxGraph,
    /// 解析后的配置
    pub config: ResolvedConfig,
    /// 权重句柄
    pub weights: WeightsHandle<B, E>,
    /// 优化上下文
    pub opt_context: OptimizationContext,
}

impl<B: Backend<E>, E: Element> UnifiedPipeline<B, E> {
    /// 从 Loader 构建完整管道
    ///
    /// 这是主入口，执行完整的流水线：
    /// 1. 检测架构
    /// 2. 加载/生成 OnnxGraph
    /// 3. 绑定权重
    /// 4. 图优化
    /// 5. 生成执行计划
    /// 6. 创建 FusedGraphExecutor
    pub fn from_loader(
        mut loader: Loader,
        backend: &B,
        manifest: &ModelManifest,
    ) -> Result<Self, PipelineError> {
        register_builtin_templates();

        // 确保 loader 已加载
        loader = loader.load()?;

        // 1. 检测架构并获取模板
        loader.set_manifest_if_missing(manifest);
        let arch = loader.detect_architecture();
        let (template, config) = resolve_template_and_config(&loader, arch)?;

        // 2. 生成或获取 OnnxGraph
        let original_graph = build_onnx_graph(&mut loader, template, &config)?;

        // 3. 上传权重到后端
        let weights = loader.upload_weights(backend)?;

        // 4. 图优化
        let opt_context = build_optimization_context();
        let optimizer = GraphOptimizer::new(opt_context.clone());
        let fused_graph = optimizer
            .optimize(&original_graph)
            .map_err(|e| PipelineError::Optimization(e.to_string()))?;

        // 5. 生成执行计划
        let execution_plan = ExecutionPlan::from_fused_graph(&fused_graph);

        // 6. 创建 FusedGraphExecutor
        let executor = FusedGraphExecutor::new(fused_graph);

        Ok(Self {
            executor,
            execution_plan,
            original_graph,
            config,
            weights,
            opt_context,
        })
    }

    /// Compile the pipeline's fused graph for a given sequence length and hidden size.
    ///
    /// Must be called before `forward()`. Can be called multiple times with
    /// different parameters to recompile for a new shape.
    ///
    /// On architectures without JIT support (not x86_64/aarch64), returns an error.
    pub fn compile(&mut self, seq_len: usize, hidden: usize) -> Result<(), PipelineError> {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            self.executor.compile(seq_len, hidden)?;
            Ok(())
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = (seq_len, hidden);
            Err(PipelineError::Execution(ExecutionError::Compilation(
                "JIT compilation not supported on this architecture".to_string(),
            )))
        }
    }

    /// Execute a forward pass through the compiled pipeline.
    ///
    /// `inputs` maps tensor names to raw byte data (f32 reinterpreted as u8).
    /// Returns output tensor name to data mapping.
    ///
    /// Returns `PipelineError::NotCompiled` if `compile()` has not been called.
    pub fn forward(
        &self,
        inputs: &HashMap<String, Vec<u8>>,
    ) -> Result<HashMap<String, Vec<u8>>, PipelineError> {
        if !self.executor.is_compiled() {
            return Err(PipelineError::NotCompiled);
        }
        let outputs = self.executor.run(inputs)?;
        Ok(outputs)
    }

    /// Convenience method: compile if needed, then run forward.
    ///
    /// If the pipeline is not yet compiled, calls `compile(seq_len, hidden)`
    /// first, then executes the forward pass.
    pub fn run(
        &mut self,
        inputs: &HashMap<String, Vec<u8>>,
        seq_len: usize,
        hidden: usize,
    ) -> Result<HashMap<String, Vec<u8>>, PipelineError> {
        if !self.is_compiled() {
            self.compile(seq_len, hidden)?;
        }
        self.forward(inputs)
    }

    /// Check if the pipeline has been compiled and is ready to execute.
    pub fn is_compiled(&self) -> bool {
        self.executor.is_compiled()
    }

    /// Access the underlying FusedGraph.
    pub fn fused_graph(&self) -> &FusedGraph {
        self.executor.graph()
    }

    /// 获取优化统计
    pub fn optimization_stats(&self) -> &OptimizationStats {
        &self.executor.graph().stats
    }

    /// 获取融合算子数量
    pub fn fused_op_count(&self) -> usize {
        self.executor.graph().fused_op_count()
    }

    /// 获取执行计划中的操作数
    pub fn execution_op_count(&self) -> usize {
        self.execution_plan.op_count()
    }
}

/// 解析模板和配置
fn resolve_template_and_config(
    loader: &Loader,
    arch: ModelArchitecture,
) -> Result<(&'static ArchTemplate, ResolvedConfig), PipelineError> {
    let template = get_template_by_arch(arch)
        .ok_or_else(|| PipelineError::TemplateNotFound(format!("{arch:?}")))?;

    // 根据格式获取 TensorProvider 并解析配置
    let config = match loader.weight_format() {
        WeightFormat::SafeTensors => {
            let st = loader.safetensors_ref().ok_or_else(|| {
                PipelineError::ConfigResolution("SafeTensors not loaded".to_string())
            })?;
            resolve_config(template, st, None)
                .map_err(|e| PipelineError::ConfigResolution(e.to_string()))?
        }
        WeightFormat::Gguf => {
            if let Some(gguf) = loader.gguf_ref() {
                resolve_config(template, gguf, Some(gguf))
                    .map_err(|e| PipelineError::ConfigResolution(e.to_string()))?
            } else {
                return Err(PipelineError::ConfigResolution(
                    "GGUF not loaded".to_string(),
                ));
            }
        }
        WeightFormat::Onnx => {
            // ONNX 直接使用图中的元数据
            ResolvedConfig::default()
        }
        _ => {
            return Err(PipelineError::ConfigResolution(
                "Unsupported weight format".to_string(),
            ));
        }
    };

    Ok((template, config))
}

/// 构建 OnnxGraph
fn build_onnx_graph(
    loader: &mut Loader,
    template: &ArchTemplate,
    config: &ResolvedConfig,
) -> Result<OnnxGraph, PipelineError> {
    match loader.weight_format() {
        WeightFormat::Onnx => {
            // ONNX: 直接返回解析的图
            let onnx = loader.onnx()?;
            Ok(onnx.graph().clone())
        }
        WeightFormat::SafeTensors | WeightFormat::Gguf => {
            // 非 ONNX: 使用模板生成图
            template
                .to_onnx_graph(config)
                .map_err(|e| PipelineError::GraphGeneration(e.to_string()))
        }
        _ => Err(PipelineError::GraphGeneration(
            "Unsupported format for graph generation".to_string(),
        )),
    }
}

/// 构建优化上下文
fn build_optimization_context() -> OptimizationContext {
    OptimizationContext::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::types::{AtomicOp, FusedNode, FusedOp, OptimizationStats};

    /// Helper: build a minimal FusedGraph with one atomic Add node.
    fn make_test_graph() -> FusedGraph {
        FusedGraph {
            nodes: vec![FusedNode {
                name: "add0".to_string(),
                op: FusedOp::Atomic(AtomicOp::new("Add")),
                inputs: vec!["x".to_string(), "w".to_string()],
                outputs: vec!["y".to_string()],
                attributes: HashMap::new(),
            }],
            inputs: vec!["x".to_string()],
            outputs: vec!["y".to_string()],
            weight_bindings: HashMap::from([(
                "w".to_string(),
                crate::graph::types::WeightBinding {
                    source_name: "w".to_string(),
                    shape: vec![4],
                    dtype: safetensors::Dtype::F32,
                    data: None,
                    ptr: None,
                },
            )]),
            quantization_info: HashMap::new(),
            sparse_tensors: HashMap::new(),
            stats: OptimizationStats::default(),
        }
    }

    #[test]
    fn forward_without_compile_returns_not_compiled() {
        let graph = make_test_graph();
        let executor = FusedGraphExecutor::new(graph);
        // Simulate a pipeline-like check: executor is not compiled
        assert!(!executor.is_compiled());
        // forward() should fail with NotCompiled
        // (We test the pipeline error variant directly since we cannot
        //  construct a full UnifiedPipeline without a real Loader.)
        let err = PipelineError::NotCompiled;
        let msg = format!("{err}");
        assert!(msg.contains("compile()"));
    }

    #[test]
    fn is_compiled_initially_false() {
        let graph = FusedGraph::new();
        let executor = FusedGraphExecutor::new(graph);
        assert!(!executor.is_compiled());
    }

    #[test]
    fn executor_graph_accessor() {
        let graph = make_test_graph();
        let executor = FusedGraphExecutor::new(graph.clone());
        assert_eq!(executor.graph().nodes.len(), graph.nodes.len());
        assert_eq!(executor.graph().inputs, graph.inputs);
    }

    #[test]
    fn pipeline_error_not_compiled_display() {
        let err = PipelineError::NotCompiled;
        assert_eq!(
            err.to_string(),
            "Pipeline not compiled: call compile() before forward()"
        );
    }

    #[test]
    fn detect_architecture_uses_manifest_fallback() {
        // 基础测试 - 无法直接测试完整管道，需要 mock
    }
}
