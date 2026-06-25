//! ONNX loader with graph parsing and fused-first pattern matching.

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::MmapOptions;
use prost::bytes::Bytes;
use prost::Message;
use safetensors::Dtype;

use super::{LoaderError, Result, TensorSlice};

#[allow(clippy::doc_overindented_list_items)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

mod attributes;
mod external;
pub mod graph_convert;
mod model;
mod pack;
mod tensor;
mod types;

#[cfg(test)]
mod tests;

pub use attributes::{OnnxAttribute, OnnxAttributeValue};
pub use model::{
    OnnxFunction, OnnxGraph, OnnxModel, OnnxNode, OnnxQuantizationAnnotation, OnnxValueInfo,
};
pub use tensor::{OnnxSparseFormat, OnnxSparseTensor, OnnxTensor};
pub use types::{OnnxDim, OnnxMapType, OnnxTensorShape, OnnxTensorType, OnnxType};

use external::ExternalDataResolver;

#[derive(Debug)]
pub struct OnnxLoader {
    path: PathBuf,
    model: OnnxModel,
    alias_map: HashMap<String, String>,     // semantic_name → onnx_name
    reverse_alias: HashMap<String, String>,  // onnx_name → semantic_name
    /// ARCH-WEIGHT-CANONICAL-LAYOUT: per-weight transpose hint from ONNX node
    /// attributes. Key = onnx tensor name (initializer name), value = whether
    /// the shape is HF [out, in] and needs transpose to canonical [K, N].
    ///   - Gemm with transB=1 → true  (HF layout)
    ///   - Gemm with transB=0 → false (already canonical)
    ///   - MatMul with 2D constant weight → true (ONNX stores [K,N], needs transpose to [N,K])
    /// Indexed by onnx_name; queried via tensor_info / alias resolution.
    layout_hints: HashMap<String, bool>,
}


/// Build alias mapping from ONNX graph nodes.
///
/// Scans graph nodes for anonymous initializer names (starting with "onnx::")
/// and derives semantic names from the node's name path.
///
/// Rules:
/// - MatMul: input[1] → ".weight"
/// - Gemm: input[1] → ".weight"
/// - Gather: input[0] → ".weight"
/// - Mul: whichever input is an onnx:: initializer → ".weight"
fn build_alias_map(graph: &OnnxGraph) -> (
    HashMap<String, String>,
    HashMap<String, String>,
    HashMap<String, bool>,
) {
    let mut alias_map: HashMap<String, String> = HashMap::new(); // semantic → onnx
    let mut reverse_alias: HashMap<String, String> = HashMap::new(); // onnx → semantic
    let mut layout_hints: HashMap<String, bool> = HashMap::new(); // onnx_name → needs_transpose

    // Pass 1: Build aliases from graph node analysis (anonymous + named initializers).
    for node in &graph.nodes {
        let (candidates, weight_transpose): (Vec<(usize, &str)>, Option<bool>) = match node.op_type.as_str() {
            "MatMul" => (vec![(1, ".weight")], None),
            "Gemm" => {
                let trans_b = node.attributes.get("transB")
                    .and_then(|a| match &a.value {
                        OnnxAttributeValue::Int(v) => Some(*v != 0),
                        _ => None,
                    })
                    .unwrap_or(false);
                (vec![(1, ".weight")], Some(trans_b))
            }
            "Gather" => (vec![(0, ".weight")], None),
            "Mul" => {
                let mut v = Vec::new();
                for (i, input) in node.inputs.iter().enumerate() {
                    if input.starts_with("onnx::") && graph.initializers.contains_key(input) {
                        v.push((i, ".weight"));
                    }
                }
                (v, None)
            }
            _ => {
                // Fallback for unknown op_types: treat all initializer inputs as
                // candidate weights and derive semantic names via node-name path
                // matching (same strategy as SafeTensors suffix patterns).
                let mut v = Vec::new();
                for (i, input) in node.inputs.iter().enumerate() {
                    if graph.initializers.contains_key(input) {
                        v.push((i, ".weight"));
                    }
                }
                (v, None)
            }
        };

        for (input_idx, _suffix) in candidates {
            let Some(onnx_name) = node.inputs.get(input_idx) else {
                continue;
            };

            if !graph.initializers.contains_key(onnx_name) {
                continue;
            }

            // Record layout hint for Gemm weights regardless of naming
            if let Some(trans) = weight_transpose {
                layout_hints.entry(onnx_name.clone()).or_insert(trans);
            }

            // MatMul: ONNX semantics store weight B as [K, N], which is the
            // transposed form of the canonical [N, K] layout. When the second
            // input (B matrix) is a 2D constant initializer, mark it as
            // needing transpose — same semantics as Gemm with transB=1.
            if node.op_type == "MatMul" && input_idx == 1 {
                if let Some(tensor) = graph.initializers.get(onnx_name) {
                    if tensor.shape.len() == 2 {
                        layout_hints.entry(onnx_name.clone()).or_insert(true);
                    }
                }
            }

            if onnx_name.starts_with("onnx::") {
                // Anonymous: derive semantic name from node.name path
                let semantic = derive_semantic_name(&node.name, &node.op_type, ".weight");
                if !semantic.is_empty()
                    && !graph.initializers.contains_key(&semantic)
                    && !alias_map.contains_key(&semantic)
                {
                    alias_map.insert(semantic.clone(), onnx_name.clone());
                    reverse_alias.insert(onnx_name.clone(), semantic);
                }
            } else {
                // Named initializer: try to build a canonical alias.
                // ONNX names may differ from canonical HF names in two ways:
                //   1. ".MatMul.weight" / ".Gemm.weight" suffix → ".weight"
                //   2. "attn" → "self_attn" (ONNX exporter shortens module names)
                if let Some(canonical) = onnx_name_to_canonical(onnx_name) {
                    if canonical != *onnx_name
                        && !graph.initializers.contains_key(&canonical)
                        && !alias_map.contains_key(&canonical)
                    {
                        alias_map.insert(canonical.clone(), onnx_name.clone());
                        reverse_alias.insert(onnx_name.clone(), canonical);
                    }
                }
            }
        }
    }

    // Pass 2: For any remaining initializers without canonical aliases,
    // apply blanket ONNX→canonical normalization (covers norm weights etc.).
    for onnx_name in graph.initializers.keys() {
        if reverse_alias.contains_key(onnx_name.as_str()) {
            continue;
        }
        if let Some(canonical) = onnx_name_to_canonical(onnx_name) {
            if canonical != *onnx_name
                && !graph.initializers.contains_key(&canonical)
                && !alias_map.contains_key(&canonical)
            {
                alias_map.insert(canonical.clone(), onnx_name.clone());
                reverse_alias.insert(onnx_name.clone(), canonical);
            }
        }
    }

    (alias_map, reverse_alias, layout_hints)
}

/// Normalize an ONNX initializer name to canonical HuggingFace form.
///
/// Handles common ONNX export naming differences:
/// 1. Op-type infix: `model.layers.0.attn.q_proj.MatMul.weight` → `model.layers.0.attn.q_proj.weight`
/// 2. Module abbreviation: `attn` → `self_attn`
/// 3. Final norm placement: `model.layers.{N}.final_norm_layernorm.weight` → `model.norm.weight`
fn onnx_name_to_canonical(onnx_name: &str) -> Option<String> {
    let ops = [".MatMul.", ".Gemm.", ".Add.", ".Mul.", ".Div.", ".Sub.", ".Reshape."];
    let mut canonical = onnx_name.to_string();
    let mut changed = false;

    // Final norm: ONNX exporters place it as model.layers.{N}.final_norm_layernorm.weight
    // instead of the canonical model.norm.weight.
    if canonical.contains(".final_norm_") && canonical.ends_with(".weight") {
        canonical = "model.norm.weight".to_string();
        return Some(canonical);
    }

    // Strip op-type infix (e.g. ".MatMul." → ".")
    for op in ops {
        if let Some(idx) = canonical.find(op) {
            // Verify the op infix is followed by "weight" or "bias"
            let after_op = &canonical[idx + op.len()..];
            if after_op.starts_with("weight") || after_op.starts_with("bias") {
                canonical = format!("{}{}", &canonical[..idx], &canonical[idx + op.len() - 1..]);
                // Remove the double-dot: "attn.q_proj..weight" → "attn.q_proj.weight"
                canonical = canonical.replace("..", ".");
                changed = true;
                break;
            }
        }
    }

    // Module name normalization: "attn" → "self_attn" (within "model.layers.N." prefix)
    if canonical.contains(".layers.") {
        for old_new in [(".attn.", ".self_attn.")] {
            if canonical.contains(old_new.0) {
                canonical = canonical.replace(old_new.0, old_new.1);
                changed = true;
            }
        }
    }

    if changed { Some(canonical) } else { None }
}

/// Derive a semantic tensor name from an ONNX node name.
///
/// Example: "/encoder/layer.0/attention/self/query/MatMul" with op_type "MatMul" and suffix ".weight"
/// → "encoder.layer.0.attention.self.query.weight"
fn derive_semantic_name(node_name: &str, op_type: &str, suffix: &str) -> String {
    // Replace "/" with "."
    let mut name = node_name.replace('/', ".");

    // Remove leading dots
    while name.starts_with('.') {
        name = name[1..].to_string();
    }

    // Remove trailing ".{op_type}" if present
    let op_suffix = format!(".{op_type}");
    if name.ends_with(&op_suffix) {
        name.truncate(name.len() - op_suffix.len());
    }

    // Append the tensor suffix (e.g. ".weight")
    name.push_str(suffix);
    name
}

impl OnnxLoader {
    pub fn from_path(path: &Path) -> Result<Self> {
        let model_proto = decode_model(path)?;
        let mut resolver = ExternalDataResolver::new(path);
        let model = OnnxModel::from_proto(model_proto, &mut resolver)?;
        log::debug!("[onnx-loader] initializers: {}, nodes: {}, inputs: {}",
            model.graph.initializers.len(), model.graph.nodes.len(), model.graph.inputs.len());
        let (alias_map, reverse_alias, layout_hints) = build_alias_map(&model.graph);
        Ok(Self {
            path: path.to_path_buf(),
            model,
            alias_map,
            reverse_alias,
            layout_hints,
        })
    }

    /// Resolve a tensor name: direct lookup first, then alias fallback.
    fn resolve<'a>(&'a self, name: &'a str) -> Option<&'a str> {
        if self.model.graph.initializers.contains_key(name) {
            Some(name)
        } else {
            self.alias_map.get(name).map(|s| s.as_str())
        }
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .model
            .graph
            .initializers
            .keys()
            .map(|k| {
                self.reverse_alias
                    .get(k)
                    .cloned()
                    .unwrap_or_else(|| k.clone()) // LEGAL: 无别名时使用原始名称
            })
            .collect();
        names.sort();
        names
    }

    pub fn tensor(&self, name: &str) -> Result<TensorSlice<'_>> {
        let resolved = self.resolve(name).unwrap_or(name); // LEGAL: 解析失败时使用原始名称
        let tensor = self
            .model
            .graph
            .initializers
            .get(resolved)
            .ok_or_else(|| LoaderError::MissingTensor(name.to_string()))?;
        Ok(tensor.slice())
    }

    pub fn tensor_dtype(&self, name: &str) -> Result<Dtype> {
        let resolved = self.resolve(name).unwrap_or(name); // LEGAL: 解析失败时使用原始名称
        let tensor = self
            .model
            .graph
            .initializers
            .get(resolved)
            .ok_or_else(|| LoaderError::MissingTensor(name.to_string()))?;
        Ok(tensor.dtype)
    }

    pub fn precision_by_tensor(&self) -> Vec<(String, Dtype)> {
        let mut out = self
            .model
            .graph
            .initializers
            .iter()
            .map(|(name, tensor)| {
                let display_name = self
                    .reverse_alias
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| name.clone()); // LEGAL: 无别名时使用原始名称
                (display_name, tensor.dtype)
            })
            .collect::<Vec<_>>();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    pub fn unique_precisions(&self) -> Vec<Dtype> {
        let mut out = Vec::new();
        for (_, dtype) in self.precision_by_tensor() {
            if !out.contains(&dtype) {
                out.push(dtype);
            }
        }
        out.sort_by_key(dtype_rank);
        out
    }

    pub fn graph(&self) -> &OnnxGraph {
        &self.model.graph
    }

    pub fn model(&self) -> &OnnxModel {
        &self.model
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl super::TensorProvider for OnnxLoader {
    fn tensor_info(&self, name: &str) -> Option<super::TensorMeta> {
        let resolved = self.resolve(name)?;
        let tensor = self.model.graph.initializers.get(resolved)?;
        Some(super::TensorMeta {
            name: name.to_string(),
            shape: tensor.shape.clone(),
            dtype: tensor.dtype,
        })
    }

    fn iter_tensors(&self) -> impl Iterator<Item = super::TensorMeta> {
        self.model
            .graph
            .initializers
            .iter()
            .map(|(name, tensor)| {
                let display_name = self
                    .reverse_alias
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| name.clone()); // LEGAL: 无别名时使用原始名称
                super::TensorMeta {
                    name: display_name,
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype,
                }
            })
    }

    fn load_tensor_data(&self, name: &str) -> super::Result<Cow<'_, [u8]>> {
        let tensor = self.tensor(name)?;
        Ok(Cow::Borrowed(tensor.data))
    }

    fn weight_layout_hint(&self, name: &str) -> Option<bool> {
        // 先按 semantic name 查 alias, 得到 onnx initializer name, 再查 hint。
        let onnx_name = self.resolve(name)?;
        self.layout_hints.get(onnx_name).copied()
    }
}

/// Collects external tensor data locations declared by an ONNX model.
///
/// Returned paths are model-relative (as stored in ONNX `external_data.location`)
/// and sorted/deduplicated.
pub fn external_data_locations(path: &Path) -> Result<Vec<String>> {
    let model = decode_model(path)?;
    let mut out = BTreeSet::new();
    if let Some(graph) = model.graph.as_ref() {
        collect_graph_external_locations(graph, &mut out);
    }
    Ok(out.into_iter().collect())
}

fn dtype_rank(dtype: &Dtype) -> u8 {
    match dtype {
        Dtype::F64 => 0,
        Dtype::F32 => 1,
        Dtype::BF16 => 2,
        Dtype::F16 => 3,
        Dtype::F8_E5M2 => 4,
        Dtype::F8_E4M3 => 5,
        Dtype::I64 => 6,
        Dtype::U64 => 7,
        Dtype::I32 => 8,
        Dtype::U32 => 9,
        Dtype::I16 => 10,
        Dtype::U16 => 11,
        Dtype::I8 => 12,
        Dtype::U8 => 13,
        Dtype::BOOL => 14,
        _ => u8::MAX,
    }
}

fn decode_model(path: &Path) -> Result<proto::ModelProto> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let bytes = Bytes::from_owner(mmap);
    proto::ModelProto::decode(bytes)
        .map_err(|err| LoaderError::Onnx(format!("onnx decode failed: {err}")))
}

fn collect_graph_external_locations(graph: &proto::GraphProto, out: &mut BTreeSet<String>) {
    for tensor in &graph.initializer {
        collect_tensor_external_location(tensor, out);
    }
    for sparse in &graph.sparse_initializer {
        if let Some(values) = sparse.values.as_ref() {
            collect_tensor_external_location(values, out);
        }
        if let Some(indices) = sparse.indices.as_ref() {
            collect_tensor_external_location(indices, out);
        }
    }
    for node in &graph.node {
        for attr in &node.attribute {
            collect_attribute_external_locations(attr, out);
        }
    }
}

fn collect_attribute_external_locations(attr: &proto::AttributeProto, out: &mut BTreeSet<String>) {
    if let Some(tensor) = attr.t.as_ref() {
        collect_tensor_external_location(tensor, out);
    }
    for tensor in &attr.tensors {
        collect_tensor_external_location(tensor, out);
    }
    if let Some(sparse) = attr.sparse_tensor.as_ref() {
        if let Some(values) = sparse.values.as_ref() {
            collect_tensor_external_location(values, out);
        }
        if let Some(indices) = sparse.indices.as_ref() {
            collect_tensor_external_location(indices, out);
        }
    }
    for sparse in &attr.sparse_tensors {
        if let Some(values) = sparse.values.as_ref() {
            collect_tensor_external_location(values, out);
        }
        if let Some(indices) = sparse.indices.as_ref() {
            collect_tensor_external_location(indices, out);
        }
    }
    if let Some(graph) = attr.g.as_ref() {
        collect_graph_external_locations(graph, out);
    }
    for graph in &attr.graphs {
        collect_graph_external_locations(graph, out);
    }
}

fn collect_tensor_external_location(tensor: &proto::TensorProto, out: &mut BTreeSet<String>) {
    let is_external = tensor
        .data_location
        .is_some_and(|value| value == proto::tensor_proto::DataLocation::External as i32);
    if !is_external {
        return;
    }
    for entry in &tensor.external_data {
        if entry.key.as_deref() != Some("location") {
            continue;
        }
        if let Some(location) = entry.value.as_ref() {
            if !location.is_empty() {
                out.insert(location.clone());
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::loader::onnx::attributes::{OnnxAttribute, OnnxAttributeValue};
    use crate::loader::onnx::tensor::OnnxTensor;
    use prost::bytes::Bytes;
    use safetensors::Dtype;

    // Helper: create an OnnxTensor with F32 data for initializer maps.
    fn make_f32_tensor(name: &str, shape: Vec<usize>) -> OnnxTensor {
        let count: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        OnnxTensor::new(
            name.to_string(),
            Dtype::F32,
            shape,
            Bytes::from(vec![0u8; count * 4]),
        )
    }

    // Helper: build an OnnxGraph from nodes and initializers.
    fn build_graph(
        nodes: Vec<OnnxNode>,
        initializers: Vec<OnnxTensor>,
    ) -> OnnxGraph {
        let init_map: HashMap<String, OnnxTensor> = initializers
            .into_iter()
            .map(|t| (t.name.clone(), t))
            .collect();
        OnnxGraph {
            name: "test_graph".to_string(),
            doc_string: String::new(),
            nodes,
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: init_map,
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        }
    }

    // ── derive_semantic_name ────────────────────────────────────────────

    #[test]
    fn derive_semantic_name_replaces_slashes_with_dots() {
        let result = derive_semantic_name(
            "/encoder/layer.0/attention/self/query/MatMul",
            "MatMul",
            ".weight",
        );
        assert_eq!(result, "encoder.layer.0.attention.self.query.weight");
    }

    #[test]
    fn derive_semantic_name_strips_leading_dots() {
        let result = derive_semantic_name("///layer/MatMul", "MatMul", ".weight");
        assert_eq!(result, "layer.weight");
    }

    #[test]
    fn derive_semantic_name_removes_trailing_op_type() {
        let result = derive_semantic_name("/model/dense/Gemm", "Gemm", ".weight");
        assert_eq!(result, "model.dense.weight");
    }

    #[test]
    fn derive_semantic_name_preserves_path_without_trailing_op() {
        let result = derive_semantic_name("/model/dense/Add", "MatMul", ".weight");
        assert_eq!(result, "model.dense.Add.weight");
    }

    #[test]
    fn derive_semantic_name_empty_path_produces_suffix_only() {
        let result = derive_semantic_name("", "MatMul", ".weight");
        assert_eq!(result, ".weight");
    }

    #[test]
    fn derive_semantic_name_single_segment() {
        let result = derive_semantic_name("/embeddings/Gather", "Gather", ".weight");
        assert_eq!(result, "embeddings.weight");
    }

    // ── onnx_name_to_canonical ──────────────────────────────────────────

    #[test]
    fn onnx_name_to_canonical_strips_matmul_infix() {
        // Input has both MatMul infix and "attn" → both are normalized
        let result = onnx_name_to_canonical("model.layers.0.attn.q_proj.MatMul.weight");
        assert_eq!(result, Some("model.layers.0.self_attn.q_proj.weight".to_string()));
    }

    #[test]
    fn onnx_name_to_canonical_strips_gemm_infix() {
        // Input has both Gemm infix and "attn" → both are normalized
        let result = onnx_name_to_canonical("model.layers.0.attn.q_proj.Gemm.weight");
        assert_eq!(result, Some("model.layers.0.self_attn.q_proj.weight".to_string()));
    }

    #[test]
    fn onnx_name_to_canonical_strips_matmul_infix_without_attn() {
        // No "attn" in path → only MatMul infix is stripped
        let result = onnx_name_to_canonical("model.layers.0.mlp.fc1.MatMul.weight");
        assert_eq!(result, Some("model.layers.0.mlp.fc1.weight".to_string()));
    }

    #[test]
    fn onnx_name_to_canonical_strips_reshape_infix_for_bias() {
        let result = onnx_name_to_canonical("model.layers.0.mlp.fc1.Reshape.bias");
        assert_eq!(result, Some("model.layers.0.mlp.fc1.bias".to_string()));
    }

    #[test]
    fn onnx_name_to_canonical_normalizes_attn_to_self_attn() {
        let result = onnx_name_to_canonical("model.layers.0.attn.q_proj.MatMul.weight");
        assert!(result.is_some());
        let canonical = result.unwrap();
        assert!(canonical.contains("self_attn"));
        assert!(!canonical.contains(".attn."));
    }

    #[test]
    fn onnx_name_to_canonical_final_norm_replacement() {
        let result = onnx_name_to_canonical("model.layers.23.final_norm_layernorm.weight");
        assert_eq!(result, Some("model.norm.weight".to_string()));
    }

    #[test]
    fn onnx_name_to_canonical_no_change_returns_none() {
        let result = onnx_name_to_canonical("model.norm.weight");
        assert!(result.is_none());
    }

    #[test]
    fn onnx_name_to_canonical_preserves_names_without_layers() {
        let result = onnx_name_to_canonical("embeddings.attn.weight");
        assert!(result.is_none());
    }

    #[test]
    fn onnx_name_to_canonical_strips_add_infix_for_weight() {
        let result = onnx_name_to_canonical("model.layers.0.mlp.fc1.Add.weight");
        assert_eq!(result, Some("model.layers.0.mlp.fc1.weight".to_string()));
    }

    // ── dtype_rank ──────────────────────────────────────────────────────

    #[test]
    fn dtype_rank_ordering_f64_before_f32() {
        assert!(dtype_rank(&Dtype::F64) < dtype_rank(&Dtype::F32));
    }

    #[test]
    fn dtype_rank_ordering_f32_before_bf16() {
        assert!(dtype_rank(&Dtype::F32) < dtype_rank(&Dtype::BF16));
    }

    #[test]
    fn dtype_rank_ordering_bf16_before_f16() {
        assert!(dtype_rank(&Dtype::BF16) < dtype_rank(&Dtype::F16));
    }

    #[test]
    fn dtype_rank_ordering_f16_before_f8_e5m2() {
        assert!(dtype_rank(&Dtype::F16) < dtype_rank(&Dtype::F8_E5M2));
    }

    #[test]
    fn dtype_rank_ordering_integers_after_floats() {
        assert!(dtype_rank(&Dtype::I64) > dtype_rank(&Dtype::F8_E4M3));
    }

    #[test]
    fn dtype_rank_ordering_i64_before_u64() {
        assert!(dtype_rank(&Dtype::I64) < dtype_rank(&Dtype::U64));
    }

    #[test]
    fn dtype_rank_ordering_i32_before_u32() {
        assert!(dtype_rank(&Dtype::I32) < dtype_rank(&Dtype::U32));
    }

    #[test]
    fn dtype_rank_ordering_u8_last_standard() {
        assert!(dtype_rank(&Dtype::U8) > dtype_rank(&Dtype::I8));
    }

    #[test]
    fn dtype_rank_bool_after_u8() {
        assert!(dtype_rank(&Dtype::BOOL) > dtype_rank(&Dtype::U8));
    }

    // ── build_alias_map: MatMul ─────────────────────────────────────────

    #[test]
    fn build_alias_map_matmul_anonymous_weight() {
        let weight = make_f32_tensor("onnx::MatMul_977", vec![3, 3]);
        let node = OnnxNode {
            name: "/encoder/layer.0/attention/self/query/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "onnx::MatMul_977".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (alias_map, reverse_alias, _layout_hints) = build_alias_map(&graph);

        let semantic = "encoder.layer.0.attention.self.query.weight";
        assert_eq!(alias_map.get(semantic).unwrap(), "onnx::MatMul_977");
        assert_eq!(reverse_alias.get("onnx::MatMul_977").unwrap(), semantic);
    }

    #[test]
    fn build_alias_map_matmul_2d_weight_sets_layout_hint_true() {
        // ONNX MatMul stores weight B as [K, N], which needs transpose to [N, K]
        let weight = make_f32_tensor("onnx::MatMul_1", vec![2, 2]);
        let node = OnnxNode {
            name: "/layer/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string(), "onnx::MatMul_1".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        assert_eq!(layout_hints.get("onnx::MatMul_1"), Some(&true));
    }

    #[test]
    fn build_alias_map_matmul_1d_weight_no_layout_hint() {
        // 1D weight (e.g. bias mistakenly placed at input[1]) should not get a layout hint
        let weight = make_f32_tensor("onnx::MatMul_1d", vec![16]);
        let node = OnnxNode {
            name: "/layer/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string(), "onnx::MatMul_1d".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        assert!(layout_hints.get("onnx::MatMul_1d").is_none());
    }

    #[test]
    fn build_alias_map_matmul_3d_weight_no_layout_hint() {
        // 3D weight (e.g. batched MatMul) should not get a layout hint
        let weight = make_f32_tensor("onnx::MatMul_3d", vec![2, 3, 4]);
        let node = OnnxNode {
            name: "/layer/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string(), "onnx::MatMul_3d".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        assert!(layout_hints.get("onnx::MatMul_3d").is_none());
    }

    #[test]
    fn build_alias_map_matmul_non_initializer_second_input_no_hint() {
        // MatMul with runtime (non-initializer) second input gets no layout hint
        let node = OnnxNode {
            name: "/layer/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string(), "runtime_tensor".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        assert!(layout_hints.is_empty());
    }

    // ── build_alias_map: Gemm ───────────────────────────────────────────

    #[test]
    fn build_alias_map_gemm_transb_true_sets_layout_hint_true() {
        let weight = make_f32_tensor("onnx::Gemm_42", vec![4, 4]);
        let node = OnnxNode {
            name: "/dense/Gemm".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "onnx::Gemm_42".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::from([(
                "transB".to_string(),
                OnnxAttribute {
                    name: "transB".to_string(),
                    value: OnnxAttributeValue::Int(1),
                    doc_string: String::new(),
                    ref_attr_name: None,
                    attr_type: None,
                },
            )]),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        assert_eq!(layout_hints.get("onnx::Gemm_42"), Some(&true));
    }

    #[test]
    fn build_alias_map_gemm_transb_false_sets_layout_hint_false() {
        let weight = make_f32_tensor("onnx::Gemm_10", vec![4, 4]);
        let node = OnnxNode {
            name: "/dense/Gemm".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "onnx::Gemm_10".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::from([(
                "transB".to_string(),
                OnnxAttribute {
                    name: "transB".to_string(),
                    value: OnnxAttributeValue::Int(0),
                    doc_string: String::new(),
                    ref_attr_name: None,
                    attr_type: None,
                },
            )]),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        assert_eq!(layout_hints.get("onnx::Gemm_10"), Some(&false));
    }

    #[test]
    fn build_alias_map_gemm_no_transb_attribute_defaults_to_false() {
        let weight = make_f32_tensor("onnx::Gemm_99", vec![2, 2]);
        let node = OnnxNode {
            name: "/dense/Gemm".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "onnx::Gemm_99".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        assert_eq!(layout_hints.get("onnx::Gemm_99"), Some(&false));
    }

    // ── build_alias_map: Gather ─────────────────────────────────────────

    #[test]
    fn build_alias_map_gather_anonymous_weight() {
        let weight = make_f32_tensor("onnx::Gather_10", vec![100, 8]);
        let node = OnnxNode {
            name: "/embeddings/word_embeddings/Gather".to_string(),
            op_type: "Gather".to_string(),
            domain: String::new(),
            inputs: vec!["onnx::Gather_10".to_string(), "input_ids".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (alias_map, _, _) = build_alias_map(&graph);

        assert_eq!(
            alias_map.get("embeddings.word_embeddings.weight").unwrap(),
            "onnx::Gather_10",
        );
    }

    // ── build_alias_map: Mul ────────────────────────────────────────────

    #[test]
    fn build_alias_map_mul_anonymous_initializer() {
        let weight = make_f32_tensor("onnx::Mul_5", vec![4]);
        let node = OnnxNode {
            name: "/model/layer_norm/Mul".to_string(),
            op_type: "Mul".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string(), "onnx::Mul_5".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (alias_map, _, _) = build_alias_map(&graph);

        assert_eq!(
            alias_map.get("model.layer_norm.weight").unwrap(),
            "onnx::Mul_5",
        );
    }

    // ── build_alias_map: named initializer canonicalization ─────────────

    #[test]
    fn build_alias_map_named_initializer_with_matmul_suffix() {
        let weight = make_f32_tensor(
            "model.layers.0.attn.q_proj.MatMul.weight",
            vec![2, 2],
        );
        let node = OnnxNode {
            name: "/model/l0/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec![
                "x".to_string(),
                "model.layers.0.attn.q_proj.MatMul.weight".to_string(),
            ],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (alias_map, reverse_alias, _) = build_alias_map(&graph);

        let canonical = "model.layers.0.self_attn.q_proj.weight";
        assert_eq!(
            alias_map.get(canonical).unwrap(),
            "model.layers.0.attn.q_proj.MatMul.weight",
        );
        assert_eq!(
            reverse_alias
                .get("model.layers.0.attn.q_proj.MatMul.weight")
                .unwrap(),
            canonical,
        );
    }

    // ── build_alias_map: pass 2 blanket normalization ───────────────────

    #[test]
    fn build_alias_map_pass2_normalizes_remaining_initializers() {
        let weight = make_f32_tensor(
            "model.layers.0.attn.k_proj.MatMul.weight",
            vec![2, 2],
        );
        let graph = build_graph(vec![], vec![weight]);

        let (alias_map, reverse_alias, _) = build_alias_map(&graph);

        let canonical = "model.layers.0.self_attn.k_proj.weight";
        assert_eq!(
            alias_map.get(canonical).unwrap(),
            "model.layers.0.attn.k_proj.MatMul.weight",
        );
        assert_eq!(
            reverse_alias
                .get("model.layers.0.attn.k_proj.MatMul.weight")
                .unwrap(),
            canonical,
        );
    }

    // ── build_alias_map: no overwriting existing initializers ───────────

    #[test]
    fn build_alias_map_does_not_overwrite_existing_initializer_name() {
        let real = make_f32_tensor("encoder.layer.0.weight", vec![2, 2]);
        let anon = make_f32_tensor("onnx::MatMul_99", vec![2, 2]);
        let node = OnnxNode {
            name: "/encoder/layer.0/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "onnx::MatMul_99".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![real, anon]);

        let (alias_map, _, _) = build_alias_map(&graph);

        assert!(alias_map.get("encoder.layer.0.weight").is_none());
    }

    // ── build_alias_map: skip non-initializer inputs ────────────────────

    #[test]
    fn build_alias_map_skips_input_not_in_initializers() {
        let node = OnnxNode {
            name: "/layer/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "not_an_initializer".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![]);

        let (alias_map, reverse_alias, layout_hints) = build_alias_map(&graph);

        assert!(alias_map.is_empty());
        assert!(reverse_alias.is_empty());
        assert!(layout_hints.is_empty());
    }

    // ── build_alias_map: unknown op_type falls back to name-based derivation ─

    #[test]
    fn build_alias_map_unknown_op_type_falls_back_to_name_derivation() {
        // Unknown op_type "Conv" — fallback derives alias from node name path
        let weight = make_f32_tensor("onnx::Conv_1", vec![3, 3]);
        let node = OnnxNode {
            name: "/layer/Conv".to_string(),
            op_type: "Conv".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "onnx::Conv_1".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (alias_map, reverse_alias, _) = build_alias_map(&graph);

        // Fallback: derive_semantic_name("/layer/Conv", "Conv", ".weight")
        // → "layer.Conv" → strip ".Conv" → "layer" → + ".weight" → "layer.weight"
        assert_eq!(alias_map.get("layer.weight"), Some(&"onnx::Conv_1".to_string()));
        assert_eq!(reverse_alias.get("onnx::Conv_1"), Some(&"layer.weight".to_string()));
    }

    #[test]
    fn build_alias_map_unknown_op_type_no_initializer_inputs_no_alias() {
        // Unknown op_type with no initializer inputs → no alias derived
        let node = OnnxNode {
            name: "/layer/CustomOp".to_string(),
            op_type: "CustomOp".to_string(),
            domain: String::new(),
            inputs: vec!["dynamic_input".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![]);

        let (alias_map, reverse_alias, _) = build_alias_map(&graph);

        assert!(alias_map.is_empty());
        assert!(reverse_alias.is_empty());
    }

    #[test]
    fn build_alias_map_custom_op_anonymous_weight_derives_alias() {
        // Custom op with anonymous onnx:: initializer and semantic node name path.
        // The fallback should derive alias from the node name, enabling
        // match_tensor_role to recognize the semantic role later.
        let weight = make_f32_tensor("onnx::CustomOp_1", vec![64, 64]);
        let node = OnnxNode {
            name: "/encoder/q_proj/CustomOp".to_string(),
            op_type: "CustomOp".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "onnx::CustomOp_1".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (alias_map, reverse_alias, _) = build_alias_map(&graph);

        // Fallback: derive_semantic_name("/encoder/q_proj/CustomOp", "CustomOp", ".weight")
        // → "encoder.q_proj.CustomOp" → strip ".CustomOp" → "encoder.q_proj"
        // → + ".weight" → "encoder.q_proj.weight"
        assert!(!alias_map.is_empty(), "alias_map must not be empty for custom op with semantic node name");
        assert_eq!(
            alias_map.get("encoder.q_proj.weight"),
            Some(&"onnx::CustomOp_1".to_string())
        );
        assert_eq!(
            reverse_alias.get("onnx::CustomOp_1"),
            Some(&"encoder.q_proj.weight".to_string())
        );
    }

    // ── collect_tensor_external_location ─────────────────────────────────

    #[test]
    fn collect_tensor_external_location_extracts_location() {
        let tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("weights/part1.bin".to_string()),
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_tensor_external_location(&tensor, &mut out);

        assert!(out.contains("weights/part1.bin"));
    }

    #[test]
    fn collect_tensor_external_location_skips_non_external() {
        let tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::Default as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("weights/part1.bin".to_string()),
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_tensor_external_location(&tensor, &mut out);

        assert!(out.is_empty());
    }

    #[test]
    fn collect_tensor_external_location_skips_no_data_location() {
        let tensor = proto::TensorProto {
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("weights/part1.bin".to_string()),
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_tensor_external_location(&tensor, &mut out);

        assert!(out.is_empty());
    }

    #[test]
    fn collect_tensor_external_location_skips_empty_location() {
        let tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some(String::new()),
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_tensor_external_location(&tensor, &mut out);

        assert!(out.is_empty());
    }

    #[test]
    fn collect_tensor_external_location_skips_non_location_key() {
        let tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("offset".to_string()),
                value: Some("1024".to_string()),
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_tensor_external_location(&tensor, &mut out);

        assert!(out.is_empty());
    }

    #[test]
    fn collect_tensor_external_location_deduplicates() {
        let tensor1 = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("shared.bin".to_string()),
            }],
            ..Default::default()
        };
        let tensor2 = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("shared.bin".to_string()),
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_tensor_external_location(&tensor1, &mut out);
        collect_tensor_external_location(&tensor2, &mut out);

        assert_eq!(out.len(), 1);
    }

    // ── collect_attribute_external_locations ─────────────────────────────

    #[test]
    fn collect_attribute_external_locations_from_tensor() {
        let attr = proto::AttributeProto {
            t: Some(proto::TensorProto {
                data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                external_data: vec![proto::StringStringEntryProto {
                    key: Some("location".to_string()),
                    value: Some("attr_weights.bin".to_string()),
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_attribute_external_locations(&attr, &mut out);

        assert!(out.contains("attr_weights.bin"));
    }

    #[test]
    fn collect_attribute_external_locations_from_tensors_list() {
        let attr = proto::AttributeProto {
            tensors: vec![proto::TensorProto {
                data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                external_data: vec![proto::StringStringEntryProto {
                    key: Some("location".to_string()),
                    value: Some("tensors_weights.bin".to_string()),
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_attribute_external_locations(&attr, &mut out);

        assert!(out.contains("tensors_weights.bin"));
    }

    #[test]
    fn collect_attribute_external_locations_from_subgraph() {
        let attr = proto::AttributeProto {
            g: Some(proto::GraphProto {
                initializer: vec![proto::TensorProto {
                    data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                    external_data: vec![proto::StringStringEntryProto {
                        key: Some("location".to_string()),
                        value: Some("subgraph_data.bin".to_string()),
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_attribute_external_locations(&attr, &mut out);

        assert!(out.contains("subgraph_data.bin"));
    }

    #[test]
    fn collect_attribute_external_locations_from_graphs_list() {
        let attr = proto::AttributeProto {
            graphs: vec![proto::GraphProto {
                initializer: vec![proto::TensorProto {
                    data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                    external_data: vec![proto::StringStringEntryProto {
                        key: Some("location".to_string()),
                        value: Some("graph_list_data.bin".to_string()),
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_attribute_external_locations(&attr, &mut out);

        assert!(out.contains("graph_list_data.bin"));
    }

    // ── collect_graph_external_locations ─────────────────────────────────

    #[test]
    fn collect_graph_external_locations_from_node_attributes() {
        let graph = proto::GraphProto {
            node: vec![proto::NodeProto {
                attribute: vec![proto::AttributeProto {
                    t: Some(proto::TensorProto {
                        data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                        external_data: vec![proto::StringStringEntryProto {
                            key: Some("location".to_string()),
                            value: Some("node_attr_data.bin".to_string()),
                        }],
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_graph_external_locations(&graph, &mut out);

        assert!(out.contains("node_attr_data.bin"));
    }

    #[test]
    fn collect_graph_external_locations_from_sparse_initializer() {
        let graph = proto::GraphProto {
            sparse_initializer: vec![proto::SparseTensorProto {
                values: Some(proto::TensorProto {
                    data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                    external_data: vec![proto::StringStringEntryProto {
                        key: Some("location".to_string()),
                        value: Some("sparse_values.bin".to_string()),
                    }],
                    ..Default::default()
                }),
                indices: Some(proto::TensorProto {
                    data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                    external_data: vec![proto::StringStringEntryProto {
                        key: Some("location".to_string()),
                        value: Some("sparse_indices.bin".to_string()),
                    }],
                    ..Default::default()
                }),
                dims: vec![10, 10],
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_graph_external_locations(&graph, &mut out);

        assert!(out.contains("sparse_values.bin"));
        assert!(out.contains("sparse_indices.bin"));
        assert_eq!(out.len(), 2);
    }

    // ── collect_attribute_external_locations: sparse attribute entries ──

    #[test]
    fn collect_attribute_external_locations_from_sparse_tensor() {
        let attr = proto::AttributeProto {
            sparse_tensor: Some(proto::SparseTensorProto {
                values: Some(proto::TensorProto {
                    data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                    external_data: vec![proto::StringStringEntryProto {
                        key: Some("location".to_string()),
                        value: Some("sparse_attr_values.bin".to_string()),
                    }],
                    ..Default::default()
                }),
                indices: None,
                dims: vec![],
            }),
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_attribute_external_locations(&attr, &mut out);

        assert!(out.contains("sparse_attr_values.bin"));
    }

    #[test]
    fn collect_attribute_external_locations_from_sparse_tensors_list() {
        let attr = proto::AttributeProto {
            sparse_tensors: vec![proto::SparseTensorProto {
                indices: Some(proto::TensorProto {
                    data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                    external_data: vec![proto::StringStringEntryProto {
                        key: Some("location".to_string()),
                        value: Some("sparse_list_indices.bin".to_string()),
                    }],
                    ..Default::default()
                }),
                values: None,
                dims: vec![],
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_attribute_external_locations(&attr, &mut out);

        assert!(out.contains("sparse_list_indices.bin"));
    }

    // ── Additional: dtype_rank comprehensive ────────────────────────────

    #[test]
    fn dtype_rank_f8_e5m2_before_f8_e4m3() {
        assert!(dtype_rank(&Dtype::F8_E5M2) < dtype_rank(&Dtype::F8_E4M3));
    }

    #[test]
    fn dtype_rank_i16_before_u16() {
        assert!(dtype_rank(&Dtype::I16) < dtype_rank(&Dtype::U16));
    }

    #[test]
    fn dtype_rank_i8_before_u8() {
        assert!(dtype_rank(&Dtype::I8) < dtype_rank(&Dtype::U8));
    }

    #[test]
    fn dtype_rank_complete_ordering_chain() {
        let chain = [
            Dtype::F64, Dtype::F32, Dtype::BF16, Dtype::F16,
            Dtype::F8_E5M2, Dtype::F8_E4M3,
            Dtype::I64, Dtype::U64, Dtype::I32, Dtype::U32,
            Dtype::I16, Dtype::U16, Dtype::I8, Dtype::U8, Dtype::BOOL,
        ];
        for window in chain.windows(2) {
            assert!(
                dtype_rank(&window[0]) < dtype_rank(&window[1]),
                "{:?} should rank before {:?}",
                window[0],
                window[1],
            );
        }
    }

    // ── Additional: onnx_name_to_canonical edge cases ───────────────────

    #[test]
    fn onnx_name_to_canonical_strips_mul_infix_for_weight() {
        let result = onnx_name_to_canonical("model.layers.0.mlp.fc1.Mul.weight");
        assert_eq!(result, Some("model.layers.0.mlp.fc1.weight".to_string()));
    }

    #[test]
    fn onnx_name_to_canonical_strips_div_infix_for_weight() {
        let result = onnx_name_to_canonical("model.layers.0.norm.Div.weight");
        assert_eq!(result, Some("model.layers.0.norm.weight".to_string()));
    }

    #[test]
    fn onnx_name_to_canonical_strips_sub_infix_for_bias() {
        let result = onnx_name_to_canonical("model.layers.0.norm.Sub.bias");
        assert_eq!(result, Some("model.layers.0.norm.bias".to_string()));
    }

    #[test]
    fn onnx_name_to_canonical_infix_only_strips_before_weight_or_bias() {
        // Op infix before a non-weight/bias suffix should not be stripped
        let result = onnx_name_to_canonical("model.layers.0.MatMul.other");
        assert!(result.is_none());
    }

    #[test]
    fn onnx_name_to_canonical_empty_string() {
        let result = onnx_name_to_canonical("");
        assert!(result.is_none());
    }

    #[test]
    fn onnx_name_to_canonical_final_norm_only_for_weight_suffix() {
        // final_norm with non-weight suffix should not trigger the final_norm shortcut
        let result = onnx_name_to_canonical("model.layers.23.final_norm_layernorm.scale");
        assert!(result.is_none() || !result.unwrap().contains("model.norm"));
    }

    #[test]
    fn onnx_name_to_canonical_attn_normalization_requires_layers_prefix() {
        // "attn" outside ".layers." context should not be normalized to "self_attn"
        let result = onnx_name_to_canonical("encoder.attn.MatMul.weight");
        // The MatMul infix is stripped, but attn stays since no ".layers."
        // After stripping .MatMul. the name becomes "encoder.attn.weight"
        // But attn normalization requires ".layers." prefix, so no further change
        assert!(result.is_none() || !result.unwrap().contains("self_attn"));
    }

    // ── Additional: derive_semantic_name edge cases ─────────────────────

    #[test]
    fn derive_semantic_name_custom_suffix() {
        let result = derive_semantic_name("/model/norm/Add", "Add", ".bias");
        assert_eq!(result, "model.norm.bias");
    }

    #[test]
    fn derive_semantic_name_op_type_not_at_end() {
        // If the op_type string appears mid-path, only the trailing occurrence is removed
        let result = derive_semantic_name("/MatMul/layer/Gemm", "Gemm", ".weight");
        assert_eq!(result, "MatMul.layer.weight");
    }

    #[test]
    fn derive_semantic_name_all_slashes() {
        let result = derive_semantic_name("///", "MatMul", ".weight");
        assert_eq!(result, ".weight");
    }

    // ── Additional: build_alias_map edge cases ──────────────────────────

    #[test]
    fn build_alias_map_empty_graph() {
        let graph = build_graph(vec![], vec![]);

        let (alias_map, reverse_alias, layout_hints) = build_alias_map(&graph);

        assert!(alias_map.is_empty());
        assert!(reverse_alias.is_empty());
        assert!(layout_hints.is_empty());
    }

    #[test]
    fn build_alias_map_layout_hint_first_write_wins() {
        // Two Gemm nodes referencing the same weight; first transB=1 wins via or_insert
        let weight = make_f32_tensor("shared_weight", vec![4, 4]);
        let node1 = OnnxNode {
            name: "/dense1/Gemm".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["x1".to_string(), "shared_weight".to_string()],
            outputs: vec!["y1".to_string()],
            attributes: HashMap::from([(
                "transB".to_string(),
                OnnxAttribute {
                    name: "transB".to_string(),
                    value: OnnxAttributeValue::Int(1),
                    doc_string: String::new(),
                    ref_attr_name: None,
                    attr_type: None,
                },
            )]),
        };
        let node2 = OnnxNode {
            name: "/dense2/Gemm".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["x2".to_string(), "shared_weight".to_string()],
            outputs: vec!["y2".to_string()],
            attributes: HashMap::from([(
                "transB".to_string(),
                OnnxAttribute {
                    name: "transB".to_string(),
                    value: OnnxAttributeValue::Int(0),
                    doc_string: String::new(),
                    ref_attr_name: None,
                    attr_type: None,
                },
            )]),
        };
        let graph = build_graph(vec![node1, node2], vec![weight]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        // or_insert keeps the first value
        assert_eq!(layout_hints.get("shared_weight"), Some(&true));
    }

    #[test]
    fn build_alias_map_gather_no_layout_hint() {
        // Gather does not produce layout hints (weight_transpose is None)
        let weight = make_f32_tensor("onnx::Gather_5", vec![10, 8]);
        let node = OnnxNode {
            name: "/embed/Gather".to_string(),
            op_type: "Gather".to_string(),
            domain: String::new(),
            inputs: vec!["onnx::Gather_5".to_string(), "ids".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        assert!(layout_hints.is_empty());
    }

    // ── Additional: collect_tensor_external_location edge cases ─────────

    #[test]
    fn collect_tensor_external_location_multiple_locations() {
        let tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![
                proto::StringStringEntryProto {
                    key: Some("location".to_string()),
                    value: Some("part1.bin".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("location".to_string()),
                    value: Some("part2.bin".to_string()),
                },
            ],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_tensor_external_location(&tensor, &mut out);

        assert_eq!(out.len(), 2);
        assert!(out.contains("part1.bin"));
        assert!(out.contains("part2.bin"));
    }

    #[test]
    fn collect_tensor_external_location_skips_none_value() {
        let tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: None,
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_tensor_external_location(&tensor, &mut out);

        assert!(out.is_empty());
    }

    // ── Additional: collect_graph_external_locations edge cases ─────────

    #[test]
    fn collect_graph_external_locations_from_initializer() {
        let graph = proto::GraphProto {
            initializer: vec![proto::TensorProto {
                data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                external_data: vec![proto::StringStringEntryProto {
                    key: Some("location".to_string()),
                    value: Some("init_weights.bin".to_string()),
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_graph_external_locations(&graph, &mut out);

        assert!(out.contains("init_weights.bin"));
    }

    #[test]
    fn collect_graph_external_locations_empty_graph() {
        let graph = proto::GraphProto::default();
        let mut out = BTreeSet::new();

        collect_graph_external_locations(&graph, &mut out);

        assert!(out.is_empty());
    }

    #[test]
    fn collect_graph_external_locations_returns_sorted() {
        let graph = proto::GraphProto {
            initializer: vec![
                proto::TensorProto {
                    data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                    external_data: vec![proto::StringStringEntryProto {
                        key: Some("location".to_string()),
                        value: Some("z_last.bin".to_string()),
                    }],
                    ..Default::default()
                },
                proto::TensorProto {
                    data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                    external_data: vec![proto::StringStringEntryProto {
                        key: Some("location".to_string()),
                        value: Some("a_first.bin".to_string()),
                    }],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_graph_external_locations(&graph, &mut out);

        // BTreeSet iteration is sorted
        let locations: Vec<&str> = out.iter().map(|s| s.as_str()).collect();
        assert_eq!(locations, vec!["a_first.bin", "z_last.bin"]);
    }

    // ── Additional: collect_attribute_external_locations edge cases ──────

    #[test]
    fn collect_attribute_external_locations_empty_attribute() {
        let attr = proto::AttributeProto::default();
        let mut out = BTreeSet::new();

        collect_attribute_external_locations(&attr, &mut out);

        assert!(out.is_empty());
    }

    // ══════════════════════════════════════════════════════════════════════
    // NEW TESTS (40 additional)
    // ══════════════════════════════════════════════════════════════════════

    // ── LoaderError variant Display messages ────────────────────────────────

    #[test]
    fn loader_error_io_display() {
        let err = LoaderError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "file missing"));
        let msg = format!("{err}");
        assert!(msg.contains("IO error"));
        assert!(msg.contains("file missing"));
    }

    #[test]
    fn loader_error_missing_tensor_display() {
        let err = LoaderError::MissingTensor("embedding.weight".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Missing tensor"));
        assert!(msg.contains("embedding.weight"));
    }

    #[test]
    fn loader_error_unsupported_dtype_display() {
        let err = LoaderError::UnsupportedDtype(Dtype::F64);
        let msg = format!("{err}");
        assert!(msg.contains("Unsupported dtype"));
        assert!(msg.contains("F64"));
    }

    #[test]
    fn loader_error_network_display() {
        let err = LoaderError::Network("connection refused".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Network error"));
        assert!(msg.contains("connection refused"));
    }

    #[test]
    fn loader_error_cache_display() {
        let err = LoaderError::Cache("corrupted index".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Cache error"));
        assert!(msg.contains("corrupted index"));
    }

    #[test]
    fn loader_error_missing_weights_display() {
        let err = LoaderError::MissingWeights;
        let msg = format!("{err}");
        assert!(msg.contains("Missing weights file"));
    }

    #[test]
    fn loader_error_gguf_display() {
        let err = LoaderError::Gguf("bad header".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("GGUF error"));
        assert!(msg.contains("bad header"));
    }

    #[test]
    fn loader_error_gllm_display() {
        let err = LoaderError::Gllm("invalid format".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("GLLM error"));
    }

    #[test]
    fn loader_error_invalid_quant_display() {
        let err = LoaderError::InvalidQuantization("mismatched scale".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Invalid quantization metadata"));
        assert!(msg.contains("mismatched scale"));
    }

    #[test]
    fn loader_error_arch_detection_display() {
        let err = LoaderError::ArchDetection("unknown arch".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Architecture detection failed"));
    }

    #[test]
    fn loader_error_backend_display() {
        let err = LoaderError::Backend("cuda not found".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Backend error"));
    }

    // ── build_alias_map: named initializer with attn normalization ──────

    #[test]
    fn build_alias_map_named_initializer_attn_to_self_attn_pass2() {
        let weight = make_f32_tensor("model.layers.5.attn.k_proj.bias", vec![8]);
        let graph = build_graph(vec![], vec![weight]);

        let (alias_map, reverse_alias, _) = build_alias_map(&graph);

        // "attn" in "model.layers.N" should normalize to "self_attn"
        let canonical = "model.layers.5.self_attn.k_proj.bias";
        assert_eq!(
            alias_map.get(canonical).unwrap(),
            "model.layers.5.attn.k_proj.bias",
        );
        assert_eq!(
            reverse_alias.get("model.layers.5.attn.k_proj.bias").unwrap(),
            canonical,
        );
    }

    // ── build_alias_map: Mul with no onnx:: initializer ─────────────────

    #[test]
    fn build_alias_map_mul_no_anonymous_initializer_skipped() {
        let node = OnnxNode {
            name: "/layer/Mul".to_string(),
            op_type: "Mul".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string(), "y".to_string()],
            outputs: vec!["z".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![]);

        let (alias_map, _, _) = build_alias_map(&graph);

        assert!(alias_map.is_empty());
    }

    // ── build_alias_map: Gemm transB as non-int attribute ignored ──────

    #[test]
    fn build_alias_map_gemm_transb_float_attribute_defaults_to_false() {
        let weight = make_f32_tensor("onnx::Gemm_55", vec![2, 2]);
        let node = OnnxNode {
            name: "/dense/Gemm".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "onnx::Gemm_55".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::from([(
                "transB".to_string(),
                OnnxAttribute {
                    name: "transB".to_string(),
                    value: OnnxAttributeValue::Float(1.0),
                    doc_string: String::new(),
                    ref_attr_name: None,
                    attr_type: None,
                },
            )]),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        // Float transB is not a valid Int, defaults to false
        assert_eq!(layout_hints.get("onnx::Gemm_55"), Some(&false));
    }

    // ── derive_semantic_name: op_type appears multiple times ────────────

    #[test]
    fn derive_semantic_name_op_type_repeated_in_path() {
        let result = derive_semantic_name("/a/MatMul/b/MatMul", "MatMul", ".weight");
        // Only the trailing .MatMul is removed
        assert_eq!(result, "a.MatMul.b.weight");
    }

    // ── derive_semantic_name: empty suffix ──────────────────────────────

    #[test]
    fn derive_semantic_name_empty_suffix() {
        let result = derive_semantic_name("/encoder/layer", "MatMul", "");
        assert_eq!(result, "encoder.layer");
    }

    // ── onnx_name_to_canonical: bias with MatMul infix ──────────────────

    #[test]
    fn onnx_name_to_canonical_strips_matmul_infix_for_bias() {
        let result = onnx_name_to_canonical("model.layers.0.attn.q_proj.MatMul.bias");
        assert_eq!(result, Some("model.layers.0.self_attn.q_proj.bias".to_string()));
    }

    // ── onnx_name_to_canonical: multiple op infixes, only last valid stripped ─

    #[test]
    fn onnx_name_to_canonical_strips_last_valid_op_infix() {
        let result = onnx_name_to_canonical("model.layers.0.MatMul.Gemm.weight");
        // ".MatMul." is first but after_op is "Gemm.weight" (not starting with "weight"/"bias"),
        // so it's skipped. ".Gemm." matches with after_op="weight", so it gets stripped.
        assert_eq!(result, Some("model.layers.0.MatMul.weight".to_string()));
    }

    // ── onnx_name_to_canonical: name outside layers without ops ─────────

    #[test]
    fn onnx_name_to_canonical_outside_layers_no_op() {
        let result = onnx_name_to_canonical("model.embed_tokens.weight");
        assert!(result.is_none());
    }

    // ── dtype_rank: unknown dtype gets max ──────────────────────────────

    #[test]
    fn dtype_rank_unknown_type_is_max() {
        // Use a variant that's not in the match — all standard ones are covered,
        // but future types would fall through to the _ => u8::MAX arm.
        // Test the known max is BOOL = 14
        assert_eq!(dtype_rank(&Dtype::BOOL), 14);
        // Any known type ranks less than u8::MAX
        assert!(dtype_rank(&Dtype::BOOL) < u8::MAX);
    }

    // ── collect_graph_external_locations: deeply nested subgraph ────────

    #[test]
    fn collect_graph_external_locations_nested_subgraph_in_node_attr() {
        let inner_tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("deep_nested.bin".to_string()),
            }],
            ..Default::default()
        };
        let inner_graph = proto::GraphProto {
            initializer: vec![inner_tensor],
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            g: Some(inner_graph),
            ..Default::default()
        };
        let node = proto::NodeProto {
            attribute: vec![attr],
            ..Default::default()
        };
        let graph = proto::GraphProto {
            node: vec![node],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_graph_external_locations(&graph, &mut out);

        assert!(out.contains("deep_nested.bin"));
    }

    // ── collect_tensor_external_location: multiple location keys ────────

    #[test]
    fn collect_tensor_external_location_first_location_key_wins_but_all_collected() {
        let tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![
                proto::StringStringEntryProto {
                    key: Some("offset".to_string()),
                    value: Some("1024".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("location".to_string()),
                    value: Some("data_part.bin".to_string()),
                },
                proto::StringStringEntryProto {
                    key: Some("length".to_string()),
                    value: Some("4096".to_string()),
                },
            ],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_tensor_external_location(&tensor, &mut out);

        assert_eq!(out.len(), 1);
        assert!(out.contains("data_part.bin"));
    }

    // ── build_alias_map: node with missing input index ──────────────────

    #[test]
    fn build_alias_map_node_with_empty_inputs() {
        let node = OnnxNode {
            name: "/layer/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![]);

        let (alias_map, reverse_alias, layout_hints) = build_alias_map(&graph);

        assert!(alias_map.is_empty());
        assert!(reverse_alias.is_empty());
        assert!(layout_hints.is_empty());
    }

    // ── build_alias_map: alias prevents collision with existing alias ───

    #[test]
    fn build_alias_map_does_not_create_duplicate_aliases() {
        let w1 = make_f32_tensor("onnx::MatMul_1", vec![2, 2]);
        let w2 = make_f32_tensor("onnx::MatMul_2", vec![2, 2]);
        // Both nodes derive to the same semantic name "layer.weight"
        let node1 = OnnxNode {
            name: "/layer/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["x1".to_string(), "onnx::MatMul_1".to_string()],
            outputs: vec!["y1".to_string()],
            attributes: HashMap::new(),
        };
        let node2 = OnnxNode {
            name: "/layer/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["x2".to_string(), "onnx::MatMul_2".to_string()],
            outputs: vec!["y2".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node1, node2], vec![w1, w2]);

        let (alias_map, _, _) = build_alias_map(&graph);

        // Only the first MatMul should get the alias (alias_map already contains key)
        assert_eq!(
            alias_map.get("layer.weight").unwrap(),
            "onnx::MatMul_1",
        );
    }

    // ── onnx_name_to_canonical: attn normalization requires layers context

    #[test]
    fn onnx_name_to_canonical_attn_without_layers_no_normalization() {
        let result = onnx_name_to_canonical("encoder.attn.weight");
        // No ".layers." prefix, no op infix, no final_norm → no change
        assert!(result.is_none());
    }

    // ── derive_semantic_name: unicode in path ───────────────────────────

    #[test]
    fn derive_semantic_name_unicode_node_name() {
        let result = derive_semantic_name("/编码器/层_0/注意力/MatMul", "MatMul", ".weight");
        assert_eq!(result, "编码器.层_0.注意力.weight");
    }

    // ── onnx_name_to_canonical: Add infix before weight ─────────────────

    #[test]
    fn onnx_name_to_canonical_add_infix_bias() {
        let result = onnx_name_to_canonical("model.layers.0.norm.Add.bias");
        assert_eq!(result, Some("model.layers.0.norm.bias".to_string()));
    }

    // ── onnx_name_to_canonical: Div infix before weight ─────────────────

    #[test]
    fn onnx_name_to_canonical_div_infix_weight() {
        let result = onnx_name_to_canonical("model.layers.0.norm.Div.weight");
        assert_eq!(result, Some("model.layers.0.norm.weight".to_string()));
    }

    // ── build_alias_map: Gather named initializer no layout hint ────────

    #[test]
    fn build_alias_map_gather_named_initializer_no_hint() {
        let weight = make_f32_tensor("embed.weight", vec![100, 64]);
        let node = OnnxNode {
            name: "/embed/Gather".to_string(),
            op_type: "Gather".to_string(),
            domain: String::new(),
            inputs: vec!["embed.weight".to_string(), "ids".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        // Gather has weight_transpose = None, so no layout hint
        assert!(layout_hints.is_empty());
    }

    // ── collect_attribute_external_locations: sparse_tensors in list ────

    #[test]
    fn collect_attribute_external_locations_sparse_tensors_both_values_and_indices() {
        let values_tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("sparse_vals_list.bin".to_string()),
            }],
            ..Default::default()
        };
        let indices_tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("sparse_idx_list.bin".to_string()),
            }],
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            sparse_tensors: vec![proto::SparseTensorProto {
                values: Some(values_tensor),
                indices: Some(indices_tensor),
                dims: vec![],
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_attribute_external_locations(&attr, &mut out);

        assert!(out.contains("sparse_vals_list.bin"));
        assert!(out.contains("sparse_idx_list.bin"));
        assert_eq!(out.len(), 2);
    }

    // ── collect_graph_external_locations: node with empty attributes ────

    #[test]
    fn collect_graph_external_locations_node_with_no_attributes() {
        let node = proto::NodeProto {
            attribute: vec![],
            ..Default::default()
        };
        let graph = proto::GraphProto {
            node: vec![node],
            initializer: vec![],
            sparse_initializer: vec![],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_graph_external_locations(&graph, &mut out);

        assert!(out.is_empty());
    }

    // ── derive_semantic_name: very long path ────────────────────────────

    #[test]
    fn derive_semantic_name_long_path() {
        let path = "/a/b/c/d/e/f/g/h/i/j/MatMul";
        let result = derive_semantic_name(path, "MatMul", ".weight");
        assert_eq!(result, "a.b.c.d.e.f.g.h.i.j.weight");
    }

    // ── onnx_name_to_canonical: final_norm with bias suffix no match ────

    #[test]
    fn onnx_name_to_canonical_final_norm_bias_not_replaced() {
        let result = onnx_name_to_canonical("model.layers.23.final_norm_layernorm.bias");
        // final_norm replacement only applies to .weight suffix
        assert!(result.is_none() || !result.as_ref().unwrap().contains("model.norm"));
    }

    // ── build_alias_map: MatMul with non-onnx named initializer ─────────

    #[test]
    fn build_alias_map_matmul_named_2d_weight_gets_layout_hint_true() {
        let weight = make_f32_tensor("encoder.weight", vec![4, 4]);
        let node = OnnxNode {
            name: "/encoder/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "encoder.weight".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (alias_map, _, layout_hints) = build_alias_map(&graph);

        // Named initializer does not start with "onnx::", so no alias created
        assert!(alias_map.is_empty());
        // 2D MatMul weight still gets layout hint = true (needs transpose)
        assert_eq!(layout_hints.get("encoder.weight"), Some(&true));
    }

    // ── dtype_rank: all floats rank before all ints ─────────────────────

    #[test]
    fn dtype_rank_all_floats_before_all_ints() {
        let float_types = [Dtype::F64, Dtype::F32, Dtype::BF16, Dtype::F16, Dtype::F8_E5M2, Dtype::F8_E4M3];
        let int_types = [Dtype::I64, Dtype::U64, Dtype::I32, Dtype::U32, Dtype::I16, Dtype::U16, Dtype::I8, Dtype::U8];
        for ft in &float_types {
            for it in &int_types {
                assert!(
                    dtype_rank(ft) < dtype_rank(it),
                    "{ft:?} should rank before {it:?}",
                );
            }
        }
    }

    // ── collect_tensor_external_location: external with no entries ──────

    #[test]
    fn collect_tensor_external_location_no_external_data_entries() {
        let tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_tensor_external_location(&tensor, &mut out);

        assert!(out.is_empty());
    }

    // ── build_alias_map: Gemm with second input not in initializers ─────

    #[test]
    fn build_alias_map_gemm_second_input_not_initializer_no_alias() {
        let node = OnnxNode {
            name: "/dense/Gemm".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "runtime_weight".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![]);

        let (alias_map, reverse_alias, layout_hints) = build_alias_map(&graph);

        // "runtime_weight" is not in initializers, so no alias or hint
        assert!(alias_map.is_empty());
        assert!(reverse_alias.is_empty());
        assert!(layout_hints.is_empty());
    }

    // ── derive_semantic_name: single character segments ─────────────────

    #[test]
    fn derive_semantic_name_single_char_segments() {
        let result = derive_semantic_name("/a/b/c/MatMul", "MatMul", ".weight");
        assert_eq!(result, "a.b.c.weight");
    }

    // ── onnx_name_to_canonical: Sub infix before weight ─────────────────

    #[test]
    fn onnx_name_to_canonical_sub_infix_weight() {
        let result = onnx_name_to_canonical("model.layers.0.norm.Sub.weight");
        assert_eq!(result, Some("model.layers.0.norm.weight".to_string()));
    }

    // ── collect_attribute_external_locations: graphs list with multiple ─

    #[test]
    fn collect_attribute_external_locations_graphs_list_multiple() {
        let g1 = proto::GraphProto {
            initializer: vec![proto::TensorProto {
                data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                external_data: vec![proto::StringStringEntryProto {
                    key: Some("location".to_string()),
                    value: Some("graph1_data.bin".to_string()),
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let g2 = proto::GraphProto {
            initializer: vec![proto::TensorProto {
                data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                external_data: vec![proto::StringStringEntryProto {
                    key: Some("location".to_string()),
                    value: Some("graph2_data.bin".to_string()),
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            graphs: vec![g1, g2],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_attribute_external_locations(&attr, &mut out);

        assert!(out.contains("graph1_data.bin"));
        assert!(out.contains("graph2_data.bin"));
        assert_eq!(out.len(), 2);
    }

    // ── build_alias_map: Gemm transB = 2 (non-zero, truthy) ─────────────

    #[test]
    fn build_alias_map_gemm_transb_nonzero_is_true() {
        let weight = make_f32_tensor("onnx::Gemm_77", vec![4, 4]);
        let node = OnnxNode {
            name: "/fc/Gemm".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string(), "onnx::Gemm_77".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::from([(
                "transB".to_string(),
                OnnxAttribute {
                    name: "transB".to_string(),
                    value: OnnxAttributeValue::Int(2),
                    doc_string: String::new(),
                    ref_attr_name: None,
                    attr_type: None,
                },
            )]),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        // transB=2 is != 0, so layout hint is true
        assert_eq!(layout_hints.get("onnx::Gemm_77"), Some(&true));
    }

    // ── onnx_name_to_canonical: no layers prefix with Gemm infix ────────

    #[test]
    fn onnx_name_to_canonical_gemm_infix_without_layers() {
        let result = onnx_name_to_canonical("embed.Gemm.weight");
        // Strips ".Gemm." → "embed.weight" (changed=true, no attn normalization)
        assert_eq!(result, Some("embed.weight".to_string()));
    }

    // ── build_alias_map: Mul with both inputs anonymous ─────────────────

    #[test]
    fn build_alias_map_mul_both_inputs_anonymous() {
        let w1 = make_f32_tensor("onnx::Mul_1", vec![4]);
        let w2 = make_f32_tensor("onnx::Mul_2", vec![4]);
        let node = OnnxNode {
            name: "/norm/scale/Mul".to_string(),
            op_type: "Mul".to_string(),
            domain: String::new(),
            inputs: vec!["onnx::Mul_1".to_string(), "onnx::Mul_2".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![w1, w2]);

        let (alias_map, _, _) = build_alias_map(&graph);

        // Both anonymous initializers start with "onnx::", both get .weight suffix
        // First one creates alias, second tries same key but alias_map already has it
        assert!(alias_map.contains_key("norm.scale.weight"));
    }

    // ── derive_semantic_name: path with trailing slash ───────────────────

    #[test]
    fn derive_semantic_name_trailing_slash_before_op() {
        let result = derive_semantic_name("/encoder/MatMul", "MatMul", ".weight");
        assert_eq!(result, "encoder.weight");
    }

    // ── onnx_name_to_canonical: Mul infix for bias ──────────────────────

    #[test]
    fn onnx_name_to_canonical_mul_infix_bias() {
        let result = onnx_name_to_canonical("model.layers.0.attn.Mul.bias");
        assert_eq!(result, Some("model.layers.0.self_attn.bias".to_string()));
    }

    // ── collect_graph_external_locations: sparse initializer without values ──

    #[test]
    fn collect_graph_external_locations_sparse_initializer_no_values() {
        let sparse = proto::SparseTensorProto {
            values: None,
            indices: Some(proto::TensorProto {
                data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                external_data: vec![proto::StringStringEntryProto {
                    key: Some("location".to_string()),
                    value: Some("only_indices.bin".to_string()),
                }],
                ..Default::default()
            }),
            dims: vec![5, 5],
        };
        let graph = proto::GraphProto {
            sparse_initializer: vec![sparse],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_graph_external_locations(&graph, &mut out);

        assert!(out.contains("only_indices.bin"));
        assert_eq!(out.len(), 1);
    }

    // ══════════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS (50 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── LoaderError Display: remaining uncovered variants ────────────────

    #[test]
    fn loader_error_authentication_display() {
        let err = LoaderError::AuthenticationError {
            hint: "set HF_TOKEN".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("Authentication error"));
        assert!(msg.contains("set HF_TOKEN"));
    }

    #[test]
    fn loader_error_hfhub_display() {
        let err = LoaderError::HfHub("rate limited".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("HfHub error"));
        assert!(msg.contains("rate limited"));
    }

    #[test]
    fn loader_error_pytorch_display() {
        let err = LoaderError::Pytorch("unsupported pickle".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("PyTorch error"));
        assert!(msg.contains("unsupported pickle"));
    }

    #[test]
    fn loader_error_unsupported_weight_extension_display() {
        let err = LoaderError::UnsupportedWeightExtension(".h5".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Unsupported weight extension"));
        assert!(msg.contains(".h5"));
    }

    #[test]
    fn loader_error_format_not_found_display() {
        use super::super::WeightFormat;
        let err = LoaderError::FormatNotFound(WeightFormat::Onnx);
        let msg = format!("{err}");
        assert!(msg.contains("Format not found"));
    }

    #[test]
    fn loader_error_multiple_weight_formats_display() {
        use super::super::WeightFormat;
        let err = LoaderError::MultipleWeightFormats(vec![
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
        ]);
        let msg = format!("{err}");
        assert!(msg.contains("Multiple weight formats"));
    }

    #[test]
    fn loader_error_duplicate_tensor_display() {
        let err = LoaderError::DuplicateTensor("layer.weight".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("Duplicate tensor"));
        assert!(msg.contains("layer.weight"));
    }

    // ── WeightFormat enum coverage ───────────────────────────────────────

    #[test]
    fn weight_format_all_variants() {
        use super::super::WeightFormat;
        let formats = [
            WeightFormat::SafeTensors,
            WeightFormat::Gguf,
            WeightFormat::Onnx,
            WeightFormat::PyTorch,
            WeightFormat::Gllm,
        ];
        // Verify Debug produces meaningful output for each variant
        for fmt in &formats {
            let debug = format!("{fmt:?}");
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn weight_format_equality_and_distinction() {
        use super::super::WeightFormat;
        assert_eq!(WeightFormat::Onnx, WeightFormat::Onnx);
        assert_ne!(WeightFormat::Onnx, WeightFormat::Gguf);
        assert_ne!(WeightFormat::SafeTensors, WeightFormat::PyTorch);
        assert_ne!(WeightFormat::Gllm, WeightFormat::Onnx);
    }

    #[test]
    fn weight_format_copy_trait() {
        use super::super::WeightFormat;
        let a = WeightFormat::Gguf;
        let b = a; // Copy, not move
        let _ = a; // still usable
        assert_eq!(a, b);
    }

    #[test]
    fn weight_format_hash_set_dedup() {
        use super::super::WeightFormat;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(WeightFormat::Onnx));
        assert!(!set.insert(WeightFormat::Onnx));
        assert!(set.insert(WeightFormat::Gguf));
        assert_eq!(set.len(), 2);
    }

    // ── onnx_name_to_canonical: deep layer nesting ──────────────────────

    #[test]
    fn onnx_name_to_canonical_deep_layer_nesting() {
        let result = onnx_name_to_canonical(
            "model.layers.99.attn.k_proj.MatMul.weight",
        );
        assert_eq!(
            result,
            Some("model.layers.99.self_attn.k_proj.weight".to_string())
        );
    }

    #[test]
    fn onnx_name_to_canonical_multiple_attn_occurrences() {
        // "attn" appears twice; both should be normalized
        let result = onnx_name_to_canonical(
            "model.layers.0.attn.attn_norm.MatMul.weight",
        );
        assert!(result.is_some());
        let canonical = result.unwrap();
        assert!(!canonical.contains(".attn."));
        assert!(canonical.contains("self_attn"));
    }

    #[test]
    fn onnx_name_to_canonical_very_long_name() {
        let long_segment = "a".repeat(200);
        let name = format!("model.layers.0.{long_segment}.MatMul.weight");
        let result = onnx_name_to_canonical(&name);
        assert!(result.is_some());
        let canonical = result.unwrap();
        assert!(canonical.contains(&long_segment));
        assert!(!canonical.contains(".MatMul."));
    }

    #[test]
    fn onnx_name_to_canonical_op_infix_in_middle_of_segment() {
        // ".MatMul." inside a segment name (not as an op infix separator)
        let result = onnx_name_to_canonical("model.layers.0.MatMul_proj.weight");
        // "MatMul_proj" does not match the op infix pattern ".MatMul.weight/bias"
        assert!(result.is_none());
    }

    #[test]
    fn onnx_name_to_canonical_final_norm_various_layers() {
        // Different layer numbers should all map to model.norm.weight
        for layer_num in [0, 1, 11, 23, 99] {
            let name = format!("model.layers.{layer_num}.final_norm_layernorm.weight");
            let result = onnx_name_to_canonical(&name);
            assert_eq!(
                result,
                Some("model.norm.weight".to_string()),
                "layer {layer_num} should normalize to model.norm.weight"
            );
        }
    }

    // ── derive_semantic_name: additional edge cases ─────────────────────

    #[test]
    fn derive_semantic_name_only_op_type_segment() {
        // When path == op_type, it is still stripped and replaced with dots
        let result = derive_semantic_name("MatMul", "MatMul", ".weight");
        assert_eq!(result, "MatMul.weight");
    }

    #[test]
    fn derive_semantic_name_numeric_segments() {
        let result = derive_semantic_name("/layer/0/1/2/MatMul", "MatMul", ".weight");
        assert_eq!(result, "layer.0.1.2.weight");
    }

    #[test]
    fn derive_semantic_name_mixed_separators() {
        // Only "/" is replaced; other chars preserved
        let result = derive_semantic_name("/encoder-layer_0/MatMul", "MatMul", ".weight");
        assert_eq!(result, "encoder-layer_0.weight");
    }

    #[test]
    fn derive_semantic_name_dot_in_path() {
        let result = derive_semantic_name("/model.layer.0/MatMul", "MatMul", ".weight");
        assert_eq!(result, "model.layer.0.weight");
    }

    // ── build_alias_map: Gemm transB as string attribute ────────────────

    #[test]
    fn build_alias_map_gemm_transb_string_attribute_defaults_to_false() {
        let weight = make_f32_tensor("onnx::Gemm_88", vec![2, 2]);
        let node = OnnxNode {
            name: "/dense/Gemm".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "onnx::Gemm_88".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::from([(
                "transB".to_string(),
                OnnxAttribute {
                    name: "transB".to_string(),
                    value: OnnxAttributeValue::String("true".to_string()),
                    doc_string: String::new(),
                    ref_attr_name: None,
                    attr_type: None,
                },
            )]),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        // String transB is not a valid Int, defaults to false
        assert_eq!(layout_hints.get("onnx::Gemm_88"), Some(&false));
    }

    // ── build_alias_map: graph with initializers only (pass 2) ──────────

    #[test]
    fn build_alias_map_pass2_only_named_initializers() {
        let w1 = make_f32_tensor(
            "model.layers.0.attn.q_proj.MatMul.weight",
            vec![2, 2],
        );
        let w2 = make_f32_tensor(
            "model.layers.0.attn.v_proj.MatMul.weight",
            vec![2, 2],
        );
        let graph = build_graph(vec![], vec![w1, w2]);

        let (alias_map, reverse_alias, _) = build_alias_map(&graph);

        // Both should get canonical names via pass 2
        assert!(alias_map.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(alias_map.contains_key("model.layers.0.self_attn.v_proj.weight"));
        assert_eq!(reverse_alias.len(), 2);
    }

    // ── build_alias_map: Mul with one onnx:: and one named initializer ──

    #[test]
    fn build_alias_map_mul_mixed_inputs() {
        let anon = make_f32_tensor("onnx::Mul_1", vec![4]);
        let named = make_f32_tensor("scale.weight", vec![4]);
        let node = OnnxNode {
            name: "/norm/scale/Mul".to_string(),
            op_type: "Mul".to_string(),
            domain: String::new(),
            inputs: vec!["onnx::Mul_1".to_string(), "scale.weight".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![anon, named]);

        let (alias_map, _, _) = build_alias_map(&graph);

        // Only the onnx:: initializer gets an alias
        assert_eq!(
            alias_map.get("norm.scale.weight").unwrap(),
            "onnx::Mul_1",
        );
    }

    // ── build_alias_map: same onnx name used by MatMul and Gather ───────

    #[test]
    fn build_alias_map_same_initializer_two_nodes_different_ops() {
        let weight = make_f32_tensor("onnx::Shared_1", vec![4, 4]);
        let node1 = OnnxNode {
            name: "/embed/Gather".to_string(),
            op_type: "Gather".to_string(),
            domain: String::new(),
            inputs: vec!["onnx::Shared_1".to_string(), "ids".to_string()],
            outputs: vec!["emb".to_string()],
            attributes: HashMap::new(),
        };
        let node2 = OnnxNode {
            name: "/fc/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["emb".to_string(), "onnx::Shared_1".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node1, node2], vec![weight]);

        let (alias_map, _, layout_hints) = build_alias_map(&graph);

        // First node (Gather) creates the alias; second (MatMul) finds alias_map already has key
        assert!(alias_map.contains_key("embed.weight"));
        // MatMul still records layout hint (true for 2D) even if alias already taken
        assert_eq!(layout_hints.get("onnx::Shared_1"), Some(&true));
    }

    // ── dtype_rank: signed vs unsigned pairs ────────────────────────────

    #[test]
    fn dtype_rank_u32_before_i16() {
        assert!(dtype_rank(&Dtype::U32) < dtype_rank(&Dtype::I16));
    }

    #[test]
    fn dtype_rank_u16_before_i8() {
        assert!(dtype_rank(&Dtype::U16) < dtype_rank(&Dtype::I8));
    }

    #[test]
    fn dtype_rank_i64_u64_i32_form_ascending_triplet() {
        assert!(dtype_rank(&Dtype::I64) < dtype_rank(&Dtype::U64));
        assert!(dtype_rank(&Dtype::U64) < dtype_rank(&Dtype::I32));
    }

    // ── collect_tensor_external_location: sparse external with None key ──

    #[test]
    fn collect_tensor_external_location_none_key_entry_skipped() {
        let tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: None,
                value: Some("orphan.bin".to_string()),
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_tensor_external_location(&tensor, &mut out);

        assert!(out.is_empty());
    }

    // ── collect_attribute_external_locations: nested graph with sparse ───

    #[test]
    fn collect_attribute_external_locations_nested_graph_with_sparse() {
        let sparse_values = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("nested_sparse.bin".to_string()),
            }],
            ..Default::default()
        };
        let inner_graph = proto::GraphProto {
            sparse_initializer: vec![proto::SparseTensorProto {
                values: Some(sparse_values),
                indices: None,
                dims: vec![],
            }],
            ..Default::default()
        };
        let attr = proto::AttributeProto {
            g: Some(inner_graph),
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_attribute_external_locations(&attr, &mut out);

        assert!(out.contains("nested_sparse.bin"));
    }

    // ── collect_graph_external_locations: multiple sparse initializers ───

    #[test]
    fn collect_graph_external_locations_multiple_sparse_initializers() {
        let make_sparse = |location: &str| proto::SparseTensorProto {
            values: Some(proto::TensorProto {
                data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                external_data: vec![proto::StringStringEntryProto {
                    key: Some("location".to_string()),
                    value: Some(location.to_string()),
                }],
                ..Default::default()
            }),
            indices: None,
            dims: vec![],
        };
        let graph = proto::GraphProto {
            sparse_initializer: vec![
                make_sparse("sparse_a.bin"),
                make_sparse("sparse_b.bin"),
            ],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_graph_external_locations(&graph, &mut out);

        assert_eq!(out.len(), 2);
        assert!(out.contains("sparse_a.bin"));
        assert!(out.contains("sparse_b.bin"));
    }

    // ── derive_semantic_name: op_type not present in path at all ────────

    #[test]
    fn derive_semantic_name_op_type_not_in_path() {
        let result = derive_semantic_name("/model/dense", "Gemm", ".weight");
        // The trailing ".Gemm" is not found, so the full path minus slashes is kept + suffix
        assert_eq!(result, "model.dense.weight");
    }

    // ── onnx_name_to_canonical: attn normalization with bias ────────────

    #[test]
    fn onnx_name_to_canonical_attn_bias_normalization() {
        let result = onnx_name_to_canonical("model.layers.3.attn.out_proj.Add.bias");
        assert_eq!(
            result,
            Some("model.layers.3.self_attn.out_proj.bias".to_string())
        );
    }

    // ── onnx_name_to_canonical: no change when name matches canonical ───

    #[test]
    fn onnx_name_to_canonical_already_canonical_returns_none() {
        let result = onnx_name_to_canonical("model.layers.0.self_attn.q_proj.weight");
        assert!(result.is_none());
    }

    // ── build_alias_map: node with single input (no second input) ───────

    #[test]
    fn build_alias_map_gemm_single_input_no_alias() {
        let node = OnnxNode {
            name: "/dense/Gemm".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![]);

        let (alias_map, reverse_alias, layout_hints) = build_alias_map(&graph);

        assert!(alias_map.is_empty());
        assert!(reverse_alias.is_empty());
        assert!(layout_hints.is_empty());
    }

    // ── build_alias_map: multiple MatMul nodes different semantic names ─

    #[test]
    fn build_alias_map_multiple_matmul_distinct_semantic_names() {
        let w1 = make_f32_tensor("onnx::MatMul_1", vec![2, 2]);
        let w2 = make_f32_tensor("onnx::MatMul_2", vec![2, 2]);
        let w3 = make_f32_tensor("onnx::MatMul_3", vec![2, 2]);
        let nodes = vec![
            OnnxNode {
                name: "/encoder/q_proj/MatMul".to_string(),
                op_type: "MatMul".to_string(),
                domain: String::new(),
                inputs: vec!["x".to_string(), "onnx::MatMul_1".to_string()],
                outputs: vec!["q".to_string()],
                attributes: HashMap::new(),
            },
            OnnxNode {
                name: "/encoder/k_proj/MatMul".to_string(),
                op_type: "MatMul".to_string(),
                domain: String::new(),
                inputs: vec!["x".to_string(), "onnx::MatMul_2".to_string()],
                outputs: vec!["k".to_string()],
                attributes: HashMap::new(),
            },
            OnnxNode {
                name: "/encoder/v_proj/MatMul".to_string(),
                op_type: "MatMul".to_string(),
                domain: String::new(),
                inputs: vec!["x".to_string(), "onnx::MatMul_3".to_string()],
                outputs: vec!["v".to_string()],
                attributes: HashMap::new(),
            },
        ];
        let graph = build_graph(nodes, vec![w1, w2, w3]);

        let (alias_map, _, layout_hints) = build_alias_map(&graph);

        assert_eq!(alias_map.get("encoder.q_proj.weight").unwrap(), "onnx::MatMul_1");
        assert_eq!(alias_map.get("encoder.k_proj.weight").unwrap(), "onnx::MatMul_2");
        assert_eq!(alias_map.get("encoder.v_proj.weight").unwrap(), "onnx::MatMul_3");
        assert_eq!(layout_hints.len(), 3);
        for hint in layout_hints.values() {
            assert_eq!(*hint, true);
        }
    }

    // ── collect_graph_external_locations: node attr with multiple tensors ─

    #[test]
    fn collect_attribute_external_locations_tensors_list_multiple() {
        let attr = proto::AttributeProto {
            tensors: vec![
                proto::TensorProto {
                    data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                    external_data: vec![proto::StringStringEntryProto {
                        key: Some("location".to_string()),
                        value: Some("attr_tensor_a.bin".to_string()),
                    }],
                    ..Default::default()
                },
                proto::TensorProto {
                    data_location: Some(proto::tensor_proto::DataLocation::External as i32),
                    external_data: vec![proto::StringStringEntryProto {
                        key: Some("location".to_string()),
                        value: Some("attr_tensor_b.bin".to_string()),
                    }],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_attribute_external_locations(&attr, &mut out);

        assert_eq!(out.len(), 2);
        assert!(out.contains("attr_tensor_a.bin"));
        assert!(out.contains("attr_tensor_b.bin"));
    }

    // ── TensorMeta construction and field access ────────────────────────

    #[test]
    fn tensor_meta_fields() {
        let meta = super::super::TensorMeta {
            name: "weight".to_string(),
            shape: vec![4, 4],
            dtype: Dtype::F32,
        };
        assert_eq!(meta.name, "weight");
        assert_eq!(meta.shape, vec![4, 4]);
        assert_eq!(meta.dtype, Dtype::F32);
    }

    #[test]
    fn tensor_meta_clone() {
        let meta = super::super::TensorMeta {
            name: "bias".to_string(),
            shape: vec![8],
            dtype: Dtype::BF16,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.name, "bias");
        assert_eq!(cloned.shape, vec![8]);
        assert_eq!(cloned.dtype, Dtype::BF16);
    }

    // ── onnx_name_to_canonical: Div and Sub infix with attn normalization

    #[test]
    fn onnx_name_to_canonical_div_with_attn() {
        let result = onnx_name_to_canonical("model.layers.0.attn.norm.Div.weight");
        assert_eq!(
            result,
            Some("model.layers.0.self_attn.norm.weight".to_string())
        );
    }

    #[test]
    fn onnx_name_to_canonical_sub_with_attn() {
        let result = onnx_name_to_canonical("model.layers.0.attn.norm.Sub.bias");
        assert_eq!(
            result,
            Some("model.layers.0.self_attn.norm.bias".to_string())
        );
    }

    // ── build_alias_map: Gemm transB negative treated as truthy ─────────

    #[test]
    fn build_alias_map_gemm_transb_negative_is_true() {
        let weight = make_f32_tensor("onnx::Gemm_33", vec![4, 4]);
        let node = OnnxNode {
            name: "/fc/Gemm".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string(), "onnx::Gemm_33".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::from([(
                "transB".to_string(),
                OnnxAttribute {
                    name: "transB".to_string(),
                    value: OnnxAttributeValue::Int(-1),
                    doc_string: String::new(),
                    ref_attr_name: None,
                    attr_type: None,
                },
            )]),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (_, _, layout_hints) = build_alias_map(&graph);

        // transB=-1 is != 0, so layout hint is true
        assert_eq!(layout_hints.get("onnx::Gemm_33"), Some(&true));
    }

    // ── collect_tensor_external_location: external with value key none ──

    #[test]
    fn collect_tensor_external_location_key_location_value_some_empty() {
        let tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some(String::new()),
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_tensor_external_location(&tensor, &mut out);

        // Empty location value is skipped
        assert!(out.is_empty());
    }

    // ── derive_semantic_name: suffix with multiple dots ─────────────────

    #[test]
    fn derive_semantic_name_suffix_with_multiple_dots() {
        let result = derive_semantic_name("/model/norm/MatMul", "MatMul", ".layer_norm.weight");
        assert_eq!(result, "model.norm.layer_norm.weight");
    }

    // ── onnx_name_to_canonical: Add infix with attn ─────────────────────

    #[test]
    fn onnx_name_to_canonical_add_with_attn_normalization() {
        let result = onnx_name_to_canonical("model.layers.5.attn.out_proj.Add.weight");
        assert_eq!(
            result,
            Some("model.layers.5.self_attn.out_proj.weight".to_string())
        );
    }

    // ── build_alias_map: Mul with only non-onnx named inputs ────────────

    #[test]
    fn build_alias_map_mul_named_inputs_no_alias() {
        let weight = make_f32_tensor("gamma.weight", vec![4]);
        let bias = make_f32_tensor("beta.weight", vec![4]);
        let node = OnnxNode {
            name: "/norm/Mul".to_string(),
            op_type: "Mul".to_string(),
            domain: String::new(),
            inputs: vec!["gamma.weight".to_string(), "beta.weight".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight, bias]);

        let (alias_map, _, _) = build_alias_map(&graph);

        // Neither input starts with "onnx::", so no alias created
        assert!(alias_map.is_empty());
    }

    // ── collect_attribute_external_locations: deeply nested attribute ───

    #[test]
    fn collect_attribute_external_locations_deeply_nested_graphs() {
        let innermost_tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("deep_nested_attr.bin".to_string()),
            }],
            ..Default::default()
        };
        let inner_graph = proto::GraphProto {
            initializer: vec![innermost_tensor],
            ..Default::default()
        };
        let outer_attr = proto::AttributeProto {
            g: Some(inner_graph),
            ..Default::default()
        };
        let outer_graph = proto::GraphProto {
            node: vec![proto::NodeProto {
                attribute: vec![outer_attr],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_graph_external_locations(&outer_graph, &mut out);

        assert!(out.contains("deep_nested_attr.bin"));
    }

    // ── dtype_rank: F64 is the lowest (highest precision) ───────────────

    #[test]
    fn dtype_rank_f64_is_lowest() {
        let f64_rank = dtype_rank(&Dtype::F64);
        assert_eq!(f64_rank, 0);
    }

    // ── build_alias_map: Gather with input[0] not in initializers ───────

    #[test]
    fn build_alias_map_gather_input_not_in_initializers() {
        let node = OnnxNode {
            name: "/embed/Gather".to_string(),
            op_type: "Gather".to_string(),
            domain: String::new(),
            inputs: vec!["runtime_table".to_string(), "ids".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![]);

        let (alias_map, reverse_alias, layout_hints) = build_alias_map(&graph);

        assert!(alias_map.is_empty());
        assert!(reverse_alias.is_empty());
        assert!(layout_hints.is_empty());
    }

    // ── derive_semantic_name: path with consecutive slashes ─────────────

    #[test]
    fn derive_semantic_name_consecutive_slashes() {
        // Double slashes produce double dots after replacement
        let result = derive_semantic_name("/encoder//layer/MatMul", "MatMul", ".weight");
        assert_eq!(result, "encoder..layer.weight");
    }

    // ── onnx_name_to_canonical: Reshape infix for weight ────────────────

    #[test]
    fn onnx_name_to_canonical_reshape_infix_weight() {
        let result = onnx_name_to_canonical("model.layers.0.mlp.fc1.Reshape.weight");
        assert_eq!(
            result,
            Some("model.layers.0.mlp.fc1.weight".to_string())
        );
    }

    // ── LoaderError Debug trait coverage ────────────────────────────────

    #[test]
    fn loader_error_debug_format_variants() {
        let variants: Vec<LoaderError> = vec![
            LoaderError::MissingWeights,
            LoaderError::DuplicateTensor("dup".to_string()),
            LoaderError::UnsupportedDtype(Dtype::F64),
            LoaderError::AuthenticationError { hint: "token".to_string() },
        ];
        for err in &variants {
            let debug = format!("{err:?}");
            assert!(!debug.is_empty());
        }
    }

    // ── OnnxNode construction ───────────────────────────────────────────

    #[test]
    fn onnx_node_fields() {
        let node = OnnxNode {
            name: "/layer0/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "weight".to_string()],
            outputs: vec!["output".to_string()],
            attributes: HashMap::new(),
        };
        assert_eq!(node.name, "/layer0/MatMul");
        assert_eq!(node.op_type, "MatMul");
        assert_eq!(node.inputs.len(), 2);
        assert_eq!(node.outputs.len(), 1);
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn onnx_node_clone() {
        let node = OnnxNode {
            name: "n".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        let cloned = node.clone();
        assert_eq!(cloned.op_type, "Add");
    }

    // ── OnnxValueInfo construction ──────────────────────────────────────

    #[test]
    fn onnx_value_info_fields() {
        let vi = OnnxValueInfo {
            name: "input_ids".to_string(),
            value_type: None,
            doc_string: "the input".to_string(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(vi.name, "input_ids");
        assert!(vi.value_type.is_none());
        assert_eq!(vi.doc_string, "the input");
    }

    #[test]
    fn onnx_value_info_with_type() {
        let vi = OnnxValueInfo {
            name: "hidden".to_string(),
            value_type: Some(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape {
                    dims: vec![OnnxDim::Param("batch".to_string()), OnnxDim::Known(768)],
                },
            })),
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert!(vi.value_type.is_some());
        if let Some(OnnxType::Tensor(tt)) = vi.value_type {
            assert_eq!(tt.shape.dims.len(), 2);
        } else {
            panic!("expected Tensor type");
        }
    }

    #[test]
    fn onnx_value_info_clone() {
        let vi = OnnxValueInfo {
            name: "v".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let cloned = vi.clone();
        assert_eq!(cloned.name, "v");
    }

    // ── OnnxQuantizationAnnotation construction ─────────────────────────

    #[test]
    fn onnx_quantization_annotation_fields() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "weight_0".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.005),
            zero_point: Some(128),
            axis: Some(0),
        };
        assert_eq!(qa.tensor_name, "weight_0");
        assert_eq!(qa.scale, Some(0.005));
        assert_eq!(qa.zero_point, Some(128));
        assert_eq!(qa.axis, Some(0));
    }

    #[test]
    fn onnx_quantization_annotation_minimal() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "w".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        assert!(qa.scale.is_none());
        assert!(qa.zero_point.is_none());
        assert!(qa.axis.is_none());
    }

    #[test]
    fn onnx_quantization_annotation_clone() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "t".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: None,
            zero_point: None,
            axis: None,
        };
        let cloned = qa.clone();
        assert_eq!(cloned.tensor_name, "t");
    }

    // ── OnnxFunction construction ───────────────────────────────────────

    #[test]
    fn onnx_function_fields() {
        let func = OnnxFunction {
            name: "CustomOp".to_string(),
            domain: "custom.domain".to_string(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: "custom op".to_string(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.name, "CustomOp");
        assert_eq!(func.domain, "custom.domain");
        assert_eq!(func.inputs, vec!["X"]);
        assert_eq!(func.outputs, vec!["Y"]);
    }

    // ── OnnxGraph construction ──────────────────────────────────────────

    #[test]
    fn onnx_graph_empty_construction() {
        let graph = OnnxGraph {
            name: "empty".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.name, "empty");
        assert!(graph.nodes.is_empty());
        assert!(graph.initializers.is_empty());
    }

    // ── OnnxNode debug ────────────────────────────────────────────────

    #[test]
    fn onnx_node_debug() {
        let node = OnnxNode {
            name: "/encoder/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["x".to_string(), "w".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        };
        let debug = format!("{node:?}");
        assert!(debug.contains("MatMul"));
        assert!(debug.contains("/encoder/MatMul"));
    }

    // ── OnnxNode with attributes ──────────────────────────────────────

    #[test]
    fn onnx_node_with_attributes() {
        let mut attrs = HashMap::new();
        attrs.insert("transB".to_string(), OnnxAttribute {
            name: "transB".to_string(),
            value: OnnxAttributeValue::Int(1),
            doc_string: String::new(),
            ref_attr_name: None,
            attr_type: None,
        });
        let node = OnnxNode {
            name: "dense".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: attrs,
        };
        assert_eq!(node.attributes.len(), 1);
        assert!(node.attributes.contains_key("transB"));
    }

    // ── OnnxValueInfo debug ───────────────────────────────────────────

    #[test]
    fn onnx_value_info_debug() {
        let vi = OnnxValueInfo {
            name: "input_ids".to_string(),
            value_type: None,
            doc_string: "input tensor".to_string(),
            metadata_props: HashMap::new(),
        };
        let debug = format!("{vi:?}");
        assert!(debug.contains("input_ids"));
    }

    // ── OnnxValueInfo with metadata_props ─────────────────────────────

    #[test]
    fn onnx_value_info_with_metadata_props() {
        let mut props = HashMap::new();
        props.insert("source".to_string(), "tokenizer".to_string());
        let vi = OnnxValueInfo {
            name: "tokens".to_string(),
            value_type: None,
            doc_string: String::new(),
            metadata_props: props,
        };
        assert_eq!(vi.metadata_props.get("source").unwrap(), "tokenizer");
    }

    // ── OnnxGraph with nodes ──────────────────────────────────────────

    #[test]
    fn onnx_graph_with_nodes_and_inputs() {
        let graph = OnnxGraph {
            name: "bert_encoder".to_string(),
            doc_string: "BERT encoder graph".to_string(),
            nodes: vec![
                OnnxNode {
                    name: "embed".to_string(),
                    op_type: "Gather".to_string(),
                    domain: String::new(),
                    inputs: vec!["ids".to_string()],
                    outputs: vec!["emb".to_string()],
                    attributes: HashMap::new(),
                },
            ],
            inputs: vec![OnnxValueInfo {
                name: "ids".to_string(),
                value_type: None,
                doc_string: String::new(),
                metadata_props: HashMap::new(),
            }],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.nodes[0].op_type, "Gather");
    }

    #[test]
    fn onnx_graph_debug() {
        let graph = OnnxGraph {
            name: "debug_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let debug = format!("{graph:?}");
        assert!(debug.contains("debug_graph"));
    }

    // ── OnnxQuantizationAnnotation debug ──────────────────────────────

    #[test]
    fn onnx_quantization_annotation_debug() {
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "weight_q".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.01),
            zero_point: None,
            axis: Some(0),
        };
        let debug = format!("{qa:?}");
        assert!(debug.contains("weight_q"));
    }

    // ── OnnxQuantizationAnnotation with quant params ──────────────────

    #[test]
    fn onnx_quantization_annotation_with_params() {
        let mut params = HashMap::new();
        params.insert("scale".to_string(), "weight_scale".to_string());
        params.insert("zp".to_string(), "weight_zp".to_string());
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "q_weight".to_string(),
            quant_param_tensor_names: params,
            scale: None,
            zero_point: Some(128),
            axis: None,
        };
        assert_eq!(qa.quant_param_tensor_names.len(), 2);
        assert_eq!(qa.quant_param_tensor_names.get("scale").unwrap(), "weight_scale");
        assert_eq!(qa.zero_point, Some(128));
    }

    // ── OnnxFunction clone ────────────────────────────────────────────

    #[test]
    fn onnx_function_clone() {
        let func = OnnxFunction {
            name: "CustomRelu".to_string(),
            domain: "custom.ops".to_string(),
            overload: String::new(),
            inputs: vec!["X".to_string()],
            outputs: vec!["Y".to_string()],
            attributes: vec!["threshold".to_string()],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: "custom relu".to_string(),
            metadata_props: HashMap::new(),
        };
        let cloned = func.clone();
        assert_eq!(cloned.name, "CustomRelu");
        assert_eq!(cloned.domain, "custom.ops");
        assert_eq!(cloned.inputs, vec!["X"]);
        assert_eq!(cloned.attributes, vec!["threshold"]);
    }

    // ── OnnxFunction debug ────────────────────────────────────────────

    #[test]
    fn onnx_function_debug() {
        let func = OnnxFunction {
            name: "If".to_string(),
            domain: "ai.onnx".to_string(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let debug = format!("{func:?}");
        assert!(debug.contains("If"));
    }

    // ── OnnxFunction with metadata_props ──────────────────────────────

    #[test]
    fn onnx_function_with_metadata_props() {
        let mut props = HashMap::new();
        props.insert("author".to_string(), "gllm".to_string());
        let func = OnnxFunction {
            name: "Scan".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: props,
        };
        assert_eq!(func.metadata_props.get("author").unwrap(), "gllm");
    }

    // ══════════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS (15 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxModelMetadata construction and field access ──────────────────

    #[test]
    fn onnx_model_metadata_fields() {
        use crate::loader::onnx::model::OnnxModelMetadata;
        use crate::loader::onnx::model::OnnxOperatorSet;
        let md = OnnxModelMetadata {
            ir_version: 8,
            producer_name: "gllm-test".to_string(),
            producer_version: "1.0".to_string(),
            domain: "ai.onnx".to_string(),
            model_version: 42,
            doc_string: "test model".to_string(),
            opset_import: vec![OnnxOperatorSet {
                domain: "ai.onnx".to_string(),
                version: 17,
            }],
            metadata_props: HashMap::new(),
        };
        assert_eq!(md.ir_version, 8);
        assert_eq!(md.producer_name, "gllm-test");
        assert_eq!(md.producer_version, "1.0");
        assert_eq!(md.domain, "ai.onnx");
        assert_eq!(md.model_version, 42);
        assert_eq!(md.doc_string, "test model");
        assert_eq!(md.opset_import.len(), 1);
        assert_eq!(md.opset_import[0].version, 17);
    }

    #[test]
    fn onnx_model_metadata_clone() {
        use crate::loader::onnx::model::OnnxModelMetadata;
        let md = OnnxModelMetadata {
            ir_version: 6,
            producer_name: String::new(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::from([("key".to_string(), "val".to_string())]),
        };
        let cloned = md.clone();
        assert_eq!(cloned.ir_version, 6);
        assert_eq!(cloned.metadata_props.get("key").unwrap(), "val");
    }

    #[test]
    fn onnx_model_metadata_debug() {
        use crate::loader::onnx::model::OnnxModelMetadata;
        let md = OnnxModelMetadata {
            ir_version: 7,
            producer_name: "test".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 0,
            doc_string: String::new(),
            opset_import: vec![],
            metadata_props: HashMap::new(),
        };
        let debug = format!("{md:?}");
        assert!(debug.contains("ir_version"));
        assert!(debug.contains("test"));
    }

    // ── OnnxOperatorSet construction ─────────────────────────────────────

    #[test]
    fn onnx_operator_set_fields() {
        use crate::loader::onnx::model::OnnxOperatorSet;
        let os = OnnxOperatorSet {
            domain: "ai.onnx.ml".to_string(),
            version: 3,
        };
        assert_eq!(os.domain, "ai.onnx.ml");
        assert_eq!(os.version, 3);
    }

    // ── OnnxModel construction and clone ─────────────────────────────────

    #[test]
    fn onnx_model_construction() {
        let model = OnnxModel {
            metadata: crate::loader::onnx::model::OnnxModelMetadata {
                ir_version: 8,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "main".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        assert_eq!(model.graph.name, "main");
        assert!(model.functions.is_empty());
    }

    #[test]
    fn onnx_model_clone() {
        let model = OnnxModel {
            metadata: crate::loader::onnx::model::OnnxModelMetadata {
                ir_version: 9,
                producer_name: "cloner".to_string(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "g".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        let cloned = model.clone();
        assert_eq!(cloned.metadata.ir_version, 9);
        assert_eq!(cloned.metadata.producer_name, "cloner");
        assert_eq!(cloned.graph.name, "g");
    }

    // ── OnnxSparseFormat variants and traits ─────────────────────────────

    #[test]
    fn onnx_sparse_format_variants_equality() {
        assert_eq!(OnnxSparseFormat::Coo, OnnxSparseFormat::Coo);
        assert_eq!(OnnxSparseFormat::Csr, OnnxSparseFormat::Csr);
        assert_eq!(OnnxSparseFormat::Csc, OnnxSparseFormat::Csc);
        assert_ne!(OnnxSparseFormat::Coo, OnnxSparseFormat::Csr);
        assert_ne!(OnnxSparseFormat::Csr, OnnxSparseFormat::Csc);
        assert_ne!(OnnxSparseFormat::Csc, OnnxSparseFormat::Coo);
    }

    #[test]
    fn onnx_sparse_format_copy_trait() {
        let a = OnnxSparseFormat::Coo;
        let b = a;
        let _ = a;
        assert_eq!(a, b);
    }

    #[test]
    fn onnx_sparse_format_hash_dedup() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(OnnxSparseFormat::Coo));
        assert!(!set.insert(OnnxSparseFormat::Coo));
        assert!(set.insert(OnnxSparseFormat::Csr));
        assert!(set.insert(OnnxSparseFormat::Csc));
        assert_eq!(set.len(), 3);
    }

    // ── OnnxTensor: new_string constructor ───────────────────────────────

    #[test]
    fn onnx_tensor_new_string_sets_is_string_flag() {
        let t = OnnxTensor::new_string("text_data".to_string(), vec![3], Bytes::from(vec![0u8; 10]));
        assert!(t.is_string);
        assert_eq!(t.dtype, Dtype::U8); // string placeholder
        assert_eq!(t.shape, vec![3]);
        assert_eq!(t.name, "text_data");
    }

    // ── OnnxSparseTensor construction ────────────────────────────────────

    #[test]
    fn onnx_sparse_tensor_construction() {
        let values = OnnxTensor::new("vals".to_string(), Dtype::F32, vec![3], Bytes::from(vec![0u8; 12]));
        let indices = OnnxTensor::new("idxs".to_string(), Dtype::I64, vec![3], Bytes::from(vec![0u8; 24]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![10, 10],
            format: OnnxSparseFormat::Coo,
        };
        assert_eq!(sparse.values.name, "vals");
        assert_eq!(sparse.indices.name, "idxs");
        assert_eq!(sparse.dims, vec![10, 10]);
        assert_eq!(sparse.format, OnnxSparseFormat::Coo);
    }

    #[test]
    fn onnx_sparse_tensor_clone() {
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![5],
            format: OnnxSparseFormat::Csr,
        };
        let cloned = sparse.clone();
        assert_eq!(cloned.format, OnnxSparseFormat::Csr);
        assert_eq!(cloned.dims, vec![5]);
    }

    // ── OnnxAttribute construction and field access ──────────────────────

    #[test]
    fn onnx_attribute_ref_variant() {
        let attr = OnnxAttribute {
            name: "body".to_string(),
            value: OnnxAttributeValue::Ref("then_branch".to_string()),
            doc_string: "subgraph ref".to_string(),
            ref_attr_name: Some("then_branch".to_string()),
            attr_type: None,
        };
        assert!(matches!(attr.value, OnnxAttributeValue::Ref(r) if r == "then_branch"));
        assert_eq!(attr.ref_attr_name.as_deref(), Some("then_branch"));
        assert_eq!(attr.doc_string, "subgraph ref");
    }

    // ── OnnxAttributeValue: additional variant matching ──────────────────

    #[test]
    fn onnx_attribute_value_floats_variant() {
        let val = OnnxAttributeValue::Floats(vec![1.0, 2.0, 3.0]);
        assert!(matches!(val, OnnxAttributeValue::Floats(ref v) if v.len() == 3));
    }

    #[test]
    fn onnx_attribute_value_strings_variant() {
        let val = OnnxAttributeValue::Strings(vec!["a".to_string(), "b".to_string()]);
        assert!(matches!(val, OnnxAttributeValue::Strings(ref v) if v[0] == "a"));
    }

    // ── OnnxGraph with metadata_props and clone ──────────────────────────

    #[test]
    fn onnx_graph_clone_preserves_all_fields() {
        let mut props = HashMap::new();
        props.insert("source".to_string(), "test".to_string());
        let graph = OnnxGraph {
            name: "clone_test".to_string(),
            doc_string: "doc".to_string(),
            nodes: vec![OnnxNode {
                name: "n1".to_string(),
                op_type: "Relu".to_string(),
                domain: String::new(),
                inputs: vec![],
                outputs: vec![],
                attributes: HashMap::new(),
            }],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: props,
        };
        let cloned = graph.clone();
        assert_eq!(cloned.name, "clone_test");
        assert_eq!(cloned.nodes.len(), 1);
        assert_eq!(cloned.nodes[0].op_type, "Relu");
        assert_eq!(cloned.metadata_props.get("source").unwrap(), "test");
    }

    // ══════════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS (15 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── LoaderError: Onnx variant display ────────────────────────────────

    #[test]
    fn loader_error_onnx_display() {
        let err = LoaderError::Onnx("invalid graph structure".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("ONNX error"));
        assert!(msg.contains("invalid graph structure"));
    }

    // ── LoaderError: SafeTensors variant from conversion ─────────────────

    #[test]
    fn loader_error_safetensors_display() {
        let err = LoaderError::SafeTensors(safetensors::SafeTensorError::InvalidHeader);
        let msg = format!("{err}");
        assert!(msg.contains("SafeTensors error"));
    }

    // ── LoaderError: Json variant display ────────────────────────────────

    #[test]
    fn loader_error_json_display() {
        let err = LoaderError::Json(serde_json::from_str::<serde_json::Value>("{bad json").unwrap_err());
        let msg = format!("{err}");
        assert!(msg.contains("JSON error"));
    }

    // ── OnnxTensor: scalar_f32 returns None for multi-element tensor ─────

    #[test]
    fn onnx_tensor_scalar_f32_multi_element_returns_none() {
        let data = vec![0u8; 8]; // 2 f32 elements
        let tensor = OnnxTensor::new("multi".to_string(), Dtype::F32, vec![2], Bytes::from(data));
        assert!(tensor.scalar_f32().is_none());
    }

    // ── OnnxTensor: scalar_f32 returns None for empty shape ──────────────

    #[test]
    fn onnx_tensor_scalar_f32_empty_data_returns_none() {
        let tensor = OnnxTensor::new("empty".to_string(), Dtype::F32, vec![0], Bytes::new());
        assert!(tensor.scalar_f32().is_none());
    }

    // ── OnnxTensor: scalar_i64 returns None for multi-element tensor ─────

    #[test]
    fn onnx_tensor_scalar_i64_multi_element_returns_none() {
        let data = vec![0u8; 16]; // 2 i64 elements
        let tensor = OnnxTensor::new("multi_i64".to_string(), Dtype::I64, vec![2], Bytes::from(data));
        assert!(tensor.scalar_i64().is_none());
    }

    // ── OnnxTensor: raw_data returns the underlying bytes ────────────────

    #[test]
    fn onnx_tensor_raw_data_returns_bytes() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let tensor = OnnxTensor::new("raw".to_string(), Dtype::F32, vec![2], Bytes::from(data.clone()));
        assert_eq!(tensor.raw_data(), data.as_slice());
    }

    // ── OnnxTensor: new with BF16 dtype preserves shape ──────────────────

    #[test]
    fn onnx_tensor_bf16_preserves_shape() {
        let data = vec![0u8; 4]; // 2 bf16 elements
        let tensor = OnnxTensor::new("bf16_w".to_string(), Dtype::BF16, vec![2], Bytes::from(data));
        assert_eq!(tensor.shape, vec![2]);
        assert_eq!(tensor.dtype, Dtype::BF16);
        assert!(!tensor.is_string);
    }

    // ── OnnxAttributeValue: Ints variant matching ────────────────────────

    #[test]
    fn onnx_attribute_value_ints_variant() {
        let val = OnnxAttributeValue::Ints(vec![1, 2, 3, 4]);
        assert!(matches!(val, OnnxAttributeValue::Ints(ref v) if v.len() == 4 && v[2] == 3));
    }

    // ── OnnxAttributeValue: Tensor variant matching ──────────────────────

    #[test]
    fn onnx_attribute_value_tensor_variant() {
        let tensor = OnnxTensor::new("attr_tensor".to_string(), Dtype::F32, vec![3], Bytes::from(vec![0u8; 12]));
        let val = OnnxAttributeValue::Tensor(tensor);
        assert!(matches!(val, OnnxAttributeValue::Tensor(ref t) if t.name == "attr_tensor"));
    }

    // ── OnnxAttributeValue: clone preserves data ─────────────────────────

    #[test]
    fn onnx_attribute_value_clone_preserves_data() {
        let val = OnnxAttributeValue::Int(42);
        let cloned = val.clone();
        assert!(matches!(cloned, OnnxAttributeValue::Int(42)));

        let float_val = OnnxAttributeValue::Float(3.14);
        let float_cloned = float_val.clone();
        assert!(matches!(float_cloned, OnnxAttributeValue::Float(f) if (f - 3.14).abs() < f32::EPSILON));
    }

    // ── OnnxAttribute: debug output includes field names ─────────────────

    #[test]
    fn onnx_attribute_debug_output() {
        let attr = OnnxAttribute {
            name: "alpha".to_string(),
            value: OnnxAttributeValue::Float(0.1),
            doc_string: "learning rate".to_string(),
            ref_attr_name: None,
            attr_type: None,
        };
        let debug = format!("{attr:?}");
        assert!(debug.contains("alpha"));
        assert!(debug.contains("Float"));
    }

    // ── OnnxOperatorSet: equality and distinction ────────────────────────

    #[test]
    fn onnx_operator_set_equality() {
        use crate::loader::onnx::model::OnnxOperatorSet;
        let a = OnnxOperatorSet {
            domain: "ai.onnx".to_string(),
            version: 17,
        };
        let b = OnnxOperatorSet {
            domain: "ai.onnx".to_string(),
            version: 17,
        };
        let c = OnnxOperatorSet {
            domain: "ai.onnx".to_string(),
            version: 18,
        };
        // OnnxOperatorSet derives Clone but not PartialEq — test clone
        let cloned = a.clone();
        assert_eq!(cloned.domain, a.domain);
        assert_eq!(cloned.version, a.version);
        assert_ne!(c.version, a.version);
    }

    // ── OnnxType: nested optional of sequence round-trip construction ────

    #[test]
    fn onnx_type_nested_optional_sequence_construction() {
        let inner_tensor = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
        });
        let seq = OnnxType::Sequence(Box::new(inner_tensor));
        let opt = OnnxType::Optional(Box::new(seq));
        match &opt {
            OnnxType::Optional(inner) => {
                assert!(matches!(**inner, OnnxType::Sequence(_)));
            }
            _ => panic!("expected Optional"),
        }
    }

    // ── build_alias_map: Gemm with first input missing (index 1 absent) ──

    #[test]
    fn build_alias_map_gemm_input_index_1_missing_from_inputs() {
        let node = OnnxNode {
            name: "/fc/Gemm".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["x_only".to_string()], // index 1 does not exist
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![]);

        let (alias_map, reverse_alias, layout_hints) = build_alias_map(&graph);

        assert!(alias_map.is_empty());
        assert!(reverse_alias.is_empty());
        assert!(layout_hints.is_empty());
    }

    // ══════════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS (15 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxAttributeValue: Graph variant with subgraph ───────────────────

    #[test]
    fn onnx_attribute_value_graph_variant() {
        let subgraph = OnnxGraph {
            name: "then_branch".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let val = OnnxAttributeValue::Graph(Box::new(subgraph));
        assert!(matches!(val, OnnxAttributeValue::Graph(g) if g.name == "then_branch"));
    }

    // ── OnnxAttributeValue: SparseTensor variant ──────────────────────────

    #[test]
    fn onnx_attribute_value_sparse_tensor_variant() {
        let values = OnnxTensor::new("v".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let indices = OnnxTensor::new("i".to_string(), Dtype::I64, vec![1], Bytes::from(vec![0u8; 8]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![3, 3],
            format: OnnxSparseFormat::Coo,
        };
        let val = OnnxAttributeValue::SparseTensor(sparse);
        assert!(matches!(val, OnnxAttributeValue::SparseTensor(ref s) if s.dims == vec![3, 3]));
    }

    // ── OnnxAttributeValue: Type variant ──────────────────────────────────

    #[test]
    fn onnx_attribute_value_type_variant() {
        let ty = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10)] },
        });
        let val = OnnxAttributeValue::Type(ty);
        assert!(matches!(val, OnnxAttributeValue::Type(OnnxType::Tensor(_))));
    }

    // ── OnnxAttributeValue: Tensors variant with multiple elements ────────

    #[test]
    fn onnx_attribute_value_tensors_variant() {
        let t1 = OnnxTensor::new("a".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let t2 = OnnxTensor::new("b".to_string(), Dtype::F32, vec![1], Bytes::from(vec![0u8; 4]));
        let val = OnnxAttributeValue::Tensors(vec![t1, t2]);
        assert!(matches!(val, OnnxAttributeValue::Tensors(ref v) if v.len() == 2));
    }

    // ── OnnxQuantizationAnnotation with all fields populated ──────────────

    #[test]
    fn onnx_quantization_annotation_all_fields() {
        let mut params = HashMap::new();
        params.insert("scale".to_string(), "s_0".to_string());
        let qa = OnnxQuantizationAnnotation {
            tensor_name: "q_linear".to_string(),
            quant_param_tensor_names: params,
            scale: Some(0.015),
            zero_point: Some(128),
            axis: Some(0),
        };
        assert_eq!(qa.tensor_name, "q_linear");
        assert_eq!(qa.quant_param_tensor_names.len(), 1);
        assert!((qa.scale.unwrap() - 0.015).abs() < f32::EPSILON);
        assert_eq!(qa.zero_point, Some(128));
        assert_eq!(qa.axis, Some(0));
    }

    // ── OnnxModel debug output includes metadata ──────────────────────────

    #[test]
    fn onnx_model_debug_includes_metadata_and_graph() {
        let model = OnnxModel {
            metadata: crate::loader::onnx::model::OnnxModelMetadata {
                ir_version: 8,
                producer_name: "test_producer".to_string(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "debug_graph".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![],
        };
        let debug = format!("{model:?}");
        assert!(debug.contains("test_producer"));
        assert!(debug.contains("debug_graph"));
    }

    // ── derive_semantic_name: op_type not preceded by slash ───────────────

    #[test]
    fn derive_semantic_name_op_type_without_leading_slash() {
        // "layer0Gemm" does not end with ".Gemm", so full path minus slashes + suffix
        let result = derive_semantic_name("layer0Gemm", "Gemm", ".weight");
        assert_eq!(result, "layer0Gemm.weight");
    }

    // ── onnx_name_to_canonical: final_norm with non-weight, non-bias suffix

    #[test]
    fn onnx_name_to_canonical_final_norm_non_weight_bias_no_shortcut() {
        let result = onnx_name_to_canonical("model.layers.0.final_norm_layernorm.running_mean");
        // final_norm shortcut only applies when suffix is ".weight"
        assert!(result.is_none() || !result.unwrap().contains("model.norm"));
    }

    // ── TensorSlice construction and field access ─────────────────────────

    #[test]
    fn tensor_slice_fields_match_tensor() {
        let tensor = make_f32_tensor("slice_test", vec![3, 4]);
        let slice = tensor.slice();
        assert_eq!(slice.shape, vec![3, 4]);
        assert_eq!(slice.dtype, Dtype::F32);
        assert_eq!(slice.data.len(), 3 * 4 * 4);
    }

    // ── LoaderError implements std::error::Error ──────────────────────────

    #[test]
    fn loader_error_implements_std_error() {
        let err = LoaderError::Onnx("test error".to_string());
        let _: &dyn std::error::Error = &err;
        assert!(std::error::Error::description(&err).len() > 0);
    }

    // ── OnnxAttributeValue: Floats with empty vec ────────────────────────

    #[test]
    fn onnx_attribute_value_floats_empty() {
        let val = OnnxAttributeValue::Floats(vec![]);
        assert!(matches!(val, OnnxAttributeValue::Floats(ref v) if v.is_empty()));
    }

    // ── OnnxAttributeValue: Ints with empty vec ───────────────────────────

    #[test]
    fn onnx_attribute_value_ints_empty() {
        let val = OnnxAttributeValue::Ints(vec![]);
        assert!(matches!(val, OnnxAttributeValue::Ints(ref v) if v.is_empty()));
    }

    // ── OnnxNode domain field with custom value ───────────────────────────

    #[test]
    fn onnx_node_custom_domain() {
        let node = OnnxNode {
            name: "custom_op".to_string(),
            op_type: "MyRelu".to_string(),
            domain: "custom.domain".to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        };
        assert_eq!(node.domain, "custom.domain");
        assert_eq!(node.op_type, "MyRelu");
    }

    // ── build_alias_map: Gather at input[0] followed by MatMul at input[1]

    #[test]
    fn build_alias_map_gather_then_matmul_same_weight_first_alias_wins() {
        let weight = make_f32_tensor("onnx::Shared", vec![8, 8]);
        let gather = OnnxNode {
            name: "/embed/Gather".to_string(),
            op_type: "Gather".to_string(),
            domain: String::new(),
            inputs: vec!["onnx::Shared".to_string(), "ids".to_string()],
            outputs: vec!["emb".to_string()],
            attributes: HashMap::new(),
        };
        let matmul = OnnxNode {
            name: "/proj/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec!["emb".to_string(), "onnx::Shared".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![gather, matmul], vec![weight]);

        let (alias_map, _, layout_hints) = build_alias_map(&graph);

        // First node (Gather) creates the alias for "embed.weight"
        assert!(alias_map.contains_key("embed.weight"));
        // MatMul still records layout hint (true for 2D) even if alias already exists
        assert_eq!(layout_hints.get("onnx::Shared"), Some(&true));
    }

    // ── LoaderError: source chain for Io variant ──────────────────────────

    #[test]
    fn loader_error_io_source_chain() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let err = LoaderError::Io(io_err);
        let source = std::error::Error::source(&err);
        assert!(source.is_some());
    }

    // ── OnnxDim boundary: Known with i64::MIN ─────────────────────────────

    #[test]
    fn onnx_dim_known_i64_min_boundary() {
        let dim = OnnxDim::Known(i64::MIN);
        assert!(matches!(dim, OnnxDim::Known(v) if v == i64::MIN));
    }

    // ══════════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS (15 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxSparseFormat Debug trait output ─────────────────────────────────

    #[test]
    fn onnx_sparse_format_debug_output() {
        let debug_coo = format!("{:?}", OnnxSparseFormat::Coo);
        let debug_csr = format!("{:?}", OnnxSparseFormat::Csr);
        let debug_csc = format!("{:?}", OnnxSparseFormat::Csc);
        assert!(debug_coo.contains("Coo"));
        assert!(debug_csr.contains("Csr"));
        assert!(debug_csc.contains("Csc"));
        // All three debug strings must be distinct
        assert_ne!(debug_coo, debug_csr);
        assert_ne!(debug_csr, debug_csc);
    }

    // ── TensorSlice::new constructor ────────────────────────────────────────

    #[test]
    fn tensor_slice_new_constructor() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let slice = TensorSlice::new(Dtype::F32, vec![2], &data);
        assert_eq!(slice.dtype, Dtype::F32);
        assert_eq!(slice.shape, vec![2]);
        assert_eq!(slice.data, data.as_slice());
    }

    // ── OnnxModel with functions field populated ────────────────────────────

    #[test]
    fn onnx_model_with_functions() {
        let func = OnnxFunction {
            name: "If".to_string(),
            domain: "ai.onnx".to_string(),
            overload: String::new(),
            inputs: vec!["cond".to_string()],
            outputs: vec!["result".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: "conditional".to_string(),
            metadata_props: HashMap::new(),
        };
        let model = OnnxModel {
            metadata: crate::loader::onnx::model::OnnxModelMetadata {
                ir_version: 8,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "main".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![func],
        };
        assert_eq!(model.functions.len(), 1);
        assert_eq!(model.functions[0].name, "If");
        assert_eq!(model.functions[0].inputs, vec!["cond"]);
    }

    // ── OnnxGraph with sparse_initializers populated ────────────────────────

    #[test]
    fn onnx_graph_with_sparse_initializers() {
        let values = OnnxTensor::new("vals".to_string(), Dtype::F32, vec![3], Bytes::from(vec![0u8; 12]));
        let indices = OnnxTensor::new("idxs".to_string(), Dtype::I64, vec![3], Bytes::from(vec![0u8; 24]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![10, 10],
            format: OnnxSparseFormat::Csr,
        };
        let graph = OnnxGraph {
            name: "sparse_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![sparse],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.sparse_initializers.len(), 1);
        assert_eq!(graph.sparse_initializers[0].format, OnnxSparseFormat::Csr);
        assert_eq!(graph.sparse_initializers[0].dims, vec![10, 10]);
    }

    // ── OnnxAttributeValue: SparseTensors variant ───────────────────────────

    #[test]
    fn onnx_attribute_value_sparse_tensors_variant() {
        let values = OnnxTensor::new("sv".to_string(), Dtype::F32, vec![2], Bytes::from(vec![0u8; 8]));
        let indices = OnnxTensor::new("si".to_string(), Dtype::I64, vec![2], Bytes::from(vec![0u8; 16]));
        let s1 = OnnxSparseTensor {
            values,
            indices,
            dims: vec![4, 4],
            format: OnnxSparseFormat::Coo,
        };
        let val = OnnxAttributeValue::SparseTensors(vec![s1]);
        assert!(matches!(val, OnnxAttributeValue::SparseTensors(ref v) if v.len() == 1));
    }

    // ── OnnxAttributeValue: Graphs variant ──────────────────────────────────

    #[test]
    fn onnx_attribute_value_graphs_variant() {
        let g = OnnxGraph {
            name: "sub".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        let val = OnnxAttributeValue::Graphs(vec![g]);
        assert!(matches!(val, OnnxAttributeValue::Graphs(ref v) if v[0].name == "sub"));
    }

    // ── OnnxAttributeValue: Types variant ───────────────────────────────────

    #[test]
    fn onnx_attribute_value_types_variant() {
        let ty = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(5)] },
        });
        let val = OnnxAttributeValue::Types(vec![ty]);
        assert!(matches!(val, OnnxAttributeValue::Types(ref v) if v.len() == 1));
    }

    // ── OnnxTensor: scalar_f32 with BF16 scalar value ───────────────────────

    #[test]
    fn onnx_tensor_scalar_f32_from_bf16_scalar() {
        // BF16 bits for 1.0: 0x3F80 (upper 16 bits of IEEE 754 1.0 = 0x3F800000)
        let bf16_bits: u16 = 0x3F80;
        let data = bf16_bits.to_le_bytes().to_vec();
        let tensor = OnnxTensor::new("bf16_scalar".to_string(), Dtype::BF16, vec![], Bytes::from(data));
        let val = tensor.scalar_f32();
        assert!(val.is_some());
        let f = val.unwrap();
        assert!((f - 1.0f32).abs() < 0.01);
    }

    // ── OnnxTensor: scalar_i64 from U8 scalar ───────────────────────────────

    #[test]
    fn onnx_tensor_scalar_i64_from_u8_scalar() {
        let data = vec![42u8];
        let tensor = OnnxTensor::new("u8_scalar".to_string(), Dtype::U8, vec![], Bytes::from(data));
        assert_eq!(tensor.scalar_i64(), Some(42));
    }

    // ── OnnxGraph with quantization_annotation populated ────────────────────

    #[test]
    fn onnx_graph_with_quantization_annotations() {
        let qa1 = OnnxQuantizationAnnotation {
            tensor_name: "weight_q".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.01),
            zero_point: Some(128),
            axis: Some(0),
        };
        let qa2 = OnnxQuantizationAnnotation {
            tensor_name: "bias_q".to_string(),
            quant_param_tensor_names: HashMap::new(),
            scale: Some(0.001),
            zero_point: Some(0),
            axis: None,
        };
        let graph = OnnxGraph {
            name: "quant_graph".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![qa1, qa2],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.quantization_annotation.len(), 2);
        assert_eq!(graph.quantization_annotation[0].tensor_name, "weight_q");
        assert_eq!(graph.quantization_annotation[1].scale, Some(0.001));
    }

    // ── build_alias_map: pass 2 skips when canonical equals onnx name ───────

    #[test]
    fn build_alias_map_pass2_skips_when_canonical_equals_onnx_name() {
        // "model.embed_tokens.weight" is already canonical: no op infix, no attn, no final_norm
        let weight = make_f32_tensor("model.embed_tokens.weight", vec![100, 64]);
        let graph = build_graph(vec![], vec![weight]);

        let (alias_map, reverse_alias, _) = build_alias_map(&graph);

        // No alias should be created because canonical == onnx name
        assert!(alias_map.is_empty());
        assert!(reverse_alias.is_empty());
    }

    // ── build_alias_map: pass 2 handles multiple distinct canonicalizations ─

    #[test]
    fn build_alias_map_pass2_multiple_distinct_initializers() {
        let w1 = make_f32_tensor("model.layers.0.attn.q_proj.MatMul.weight", vec![2, 2]);
        let w2 = make_f32_tensor("model.layers.1.attn.k_proj.MatMul.weight", vec![2, 2]);
        let w3 = make_f32_tensor("model.layers.0.mlp.fc1.Gemm.bias", vec![4]);
        let graph = build_graph(vec![], vec![w1, w2, w3]);

        let (alias_map, reverse_alias, _) = build_alias_map(&graph);

        // All three should get canonical names via pass 2
        assert!(alias_map.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(alias_map.contains_key("model.layers.1.self_attn.k_proj.weight"));
        assert!(alias_map.contains_key("model.layers.0.mlp.fc1.bias"));
        assert_eq!(reverse_alias.len(), 3);
    }

    // ── OnnxTensor: scalar_f32 returns Some for scalar shape ────────────────

    #[test]
    fn onnx_tensor_scalar_f32_from_f32_scalar_shape() {
        // Scalar tensor with vec![] shape and 4 bytes (one F32)
        let data: Vec<u8> = 3.14f32.to_le_bytes().to_vec();
        let tensor = OnnxTensor::new("f32_scalar".to_string(), Dtype::F32, vec![], Bytes::from(data));
        let val = tensor.scalar_f32();
        assert!(val.is_some());
        assert!((val.unwrap() - 3.14f32).abs() < 0.001);
    }

    // ── derive_semantic_name: op_type is a substring of a path segment ──────

    #[test]
    fn derive_semantic_name_op_type_as_substring_of_segment() {
        // "MatMul" appears as a substring in "PreMatMul" but not as trailing "/MatMul"
        let result = derive_semantic_name("/PreMatMul/Gemm", "Gemm", ".weight");
        // Trailing ".Gemm" is stripped
        assert_eq!(result, "PreMatMul.weight");
    }

    // ── onnx_name_to_canonical: Gemm infix for bias without attn ────────────

    #[test]
    fn onnx_name_to_canonical_gemm_infix_bias_no_layers() {
        let result = onnx_name_to_canonical("output.Gemm.bias");
        // Strips ".Gemm." before "bias" -> "output.bias", changed=true, no attn normalization
        assert_eq!(result, Some("output.bias".to_string()));
    }

    // ══════════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS (15 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxTensor: scalar_f32 from I32 scalar ───────────────────────────
    // @trace TEST-ONNX-268 [level:unit]

    #[test]
    fn onnx_tensor_scalar_f32_from_i32_scalar() {
        let val: i32 = -7;
        let data = val.to_le_bytes().to_vec();
        let tensor = OnnxTensor::new("i32_scalar".to_string(), Dtype::I32, vec![], Bytes::from(data));
        let result = tensor.scalar_f32();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), -7.0f32);
    }

    // ── OnnxTensor: scalar_f32 from U16 scalar ───────────────────────────
    // @trace TEST-ONNX-269 [level:unit]

    #[test]
    fn onnx_tensor_scalar_f32_from_u16_scalar() {
        let val: u16 = 1000;
        let data = val.to_le_bytes().to_vec();
        let tensor = OnnxTensor::new("u16_scalar".to_string(), Dtype::U16, vec![], Bytes::from(data));
        let result = tensor.scalar_f32();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 1000.0f32);
    }

    // ── OnnxTensor: scalar_f32 returns None for unsupported dtype ────────
    // @trace TEST-ONNX-270 [level:unit]

    #[test]
    fn onnx_tensor_scalar_f32_unsupported_dtype_returns_none() {
        let data = vec![0u8; 1];
        let tensor = OnnxTensor::new("bool_scalar".to_string(), Dtype::BOOL, vec![], Bytes::from(data));
        assert!(tensor.scalar_f32().is_none());
    }

    // ── OnnxTensor: scalar_i64 from I32 scalar ───────────────────────────
    // @trace TEST-ONNX-271 [level:unit]

    #[test]
    fn onnx_tensor_scalar_i64_from_i32_scalar() {
        let val: i32 = 256;
        let data = val.to_le_bytes().to_vec();
        let tensor = OnnxTensor::new("i32_scalar".to_string(), Dtype::I32, vec![], Bytes::from(data));
        assert_eq!(tensor.scalar_i64(), Some(256i64));
    }

    // ── OnnxTensor: scalar_i64 from F32 scalar truncates to integer ──────
    // @trace TEST-ONNX-272 [level:unit]

    #[test]
    fn onnx_tensor_scalar_i64_from_f32_scalar_truncates() {
        let val: f32 = 42.75;
        let data = val.to_le_bytes().to_vec();
        let tensor = OnnxTensor::new("f32_scalar".to_string(), Dtype::F32, vec![], Bytes::from(data));
        let result = tensor.scalar_i64();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 42i64);
    }

    // ── OnnxTensor: scalar_i64 returns None for empty data ───────────────
    // @trace TEST-ONNX-273 [level:unit]

    #[test]
    fn onnx_tensor_scalar_i64_empty_shape_returns_none() {
        let tensor = OnnxTensor::new("empty_i64".to_string(), Dtype::I64, vec![0], Bytes::new());
        assert!(tensor.scalar_i64().is_none());
    }

    // ── OnnxDim: Known, Param, Unknown variant equality and distinction ──
    // @trace TEST-ONNX-274 [level:unit]

    #[test]
    fn onnx_dim_variants_equality_and_distinction() {
        assert_eq!(OnnxDim::Known(5), OnnxDim::Known(5));
        assert_ne!(OnnxDim::Known(5), OnnxDim::Known(6));
        assert_ne!(OnnxDim::Known(0), OnnxDim::Unknown);
        assert_ne!(OnnxDim::Param("batch".to_string()), OnnxDim::Unknown);
        assert_eq!(OnnxDim::Unknown, OnnxDim::Unknown);
        assert_eq!(
            OnnxDim::Param("seq".to_string()),
            OnnxDim::Param("seq".to_string())
        );
    }

    // ── OnnxDim: hash consistency for Known values ────────────────────────
    // @trace TEST-ONNX-275 [level:unit]

    #[test]
    fn onnx_dim_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(OnnxDim::Known(768)));
        assert!(!set.insert(OnnxDim::Known(768)));
        assert!(set.insert(OnnxDim::Unknown));
        assert!(set.insert(OnnxDim::Param("heads".to_string())));
        assert_eq!(set.len(), 3);
    }

    // ── OnnxType: Map variant construction with key/value types ───────────
    // @trace TEST-ONNX-276 [level:unit]

    #[test]
    fn onnx_type_map_variant_construction() {
        let map_type = OnnxType::Map(OnnxMapType {
            key_type: proto::tensor_proto::DataType::String,
            value_type: Box::new(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Int64,
                shape: OnnxTensorShape { dims: vec![] },
            })),
        });
        assert!(matches!(map_type, OnnxType::Map(m) if m.key_type == proto::tensor_proto::DataType::String));
    }

    // ── OnnxType: SparseTensor variant construction ───────────────────────
    // @trace TEST-ONNX-277 [level:unit]

    #[test]
    fn onnx_type_sparse_tensor_variant() {
        let sp = OnnxType::SparseTensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(10), OnnxDim::Known(20)] },
        });
        assert!(matches!(sp, OnnxType::SparseTensor(t) if t.shape.dims.len() == 2));
    }

    // ── OnnxType: equality round-trip for nested Optional(Sequence(Tensor))
    // @trace TEST-ONNX-278 [level:unit]

    #[test]
    fn onnx_type_nested_equality_round_trip() {
        let tensor_type = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(5)] },
        });
        let seq = OnnxType::Sequence(Box::new(tensor_type.clone()));
        let opt = OnnxType::Optional(Box::new(seq.clone()));
        // Reconstruct identical types and verify equality
        let tensor_type2 = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Float,
            shape: OnnxTensorShape { dims: vec![OnnxDim::Known(5)] },
        });
        assert_eq!(tensor_type, tensor_type2);
        assert_eq!(opt, OnnxType::Optional(Box::new(OnnxType::Sequence(Box::new(tensor_type2)))));
    }

    // ── build_alias_map: pass 2 does not overwrite existing pass 1 alias ─
    // @trace TEST-ONNX-279 [level:unit]

    #[test]
    fn build_alias_map_pass2_does_not_overwrite_pass1_alias() {
        // Pass 1: node creates alias via named initializer canonicalization
        let weight = make_f32_tensor(
            "model.layers.0.attn.q_proj.MatMul.weight",
            vec![2, 2],
        );
        let node = OnnxNode {
            name: "/model/layers.0/attn/q_proj/MatMul".to_string(),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            inputs: vec![
                "x".to_string(),
                "model.layers.0.attn.q_proj.MatMul.weight".to_string(),
            ],
            outputs: vec!["out".to_string()],
            attributes: HashMap::new(),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (alias_map, reverse_alias, _) = build_alias_map(&graph);

        // The canonical name should point to the original ONNX name
        let canonical = "model.layers.0.self_attn.q_proj.weight";
        assert_eq!(
            alias_map.get(canonical).unwrap(),
            "model.layers.0.attn.q_proj.MatMul.weight",
        );
        // reverse_alias should have exactly one entry (pass 2 finds it already mapped)
        assert_eq!(reverse_alias.len(), 1);
    }

    // ── derive_semantic_name: path exactly equals op_type string ──────────
    // @trace TEST-ONNX-280 [level:unit]

    #[test]
    fn derive_semantic_name_path_equals_op_type() {
        // When path is exactly the op_type with no slashes, the trailing match removes it
        let result = derive_semantic_name("Gemm", "Gemm", ".weight");
        // After removing "Gemm" (trailing op_type) we get empty string + ".weight"
        // But there are no leading dots to strip, so we get the suffix
        assert_eq!(result, "Gemm.weight");
    }

    // ── OnnxTensor: scalar_f32 from U32 scalar ────────────────────────────
    // @trace TEST-ONNX-281 [level:unit]

    #[test]
    fn onnx_tensor_scalar_f32_from_u32_scalar() {
        let val: u32 = 100;
        let data = val.to_le_bytes().to_vec();
        let tensor = OnnxTensor::new("u32_scalar".to_string(), Dtype::U32, vec![], Bytes::from(data));
        let result = tensor.scalar_f32();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 100.0f32);
    }

    // ── OnnxGraph with outputs populated ──────────────────────────────────
    // @trace TEST-ONNX-282 [level:unit]

    #[test]
    fn onnx_graph_with_outputs() {
        let output = OnnxValueInfo {
            name: "logits".to_string(),
            value_type: Some(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape {
                    dims: vec![OnnxDim::Param("batch".to_string()), OnnxDim::Known(30522)],
                },
            })),
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let graph = OnnxGraph {
            name: "lm_head".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![output],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.outputs.len(), 1);
        assert_eq!(graph.outputs[0].name, "logits");
        assert!(graph.outputs[0].value_type.is_some());
    }

    // ══════════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS (15 new) — OnnxLoader, TensorProvider, shape via slice
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxTensor: scalar shape slice has correct byte count ───────────
    // @trace TEST-ONNX-283 [level:unit]

    #[test]
    fn onnx_tensor_scalar_shape_slice_byte_count() {
        let tensor = OnnxTensor::new("scalar".to_string(), Dtype::F32, vec![], Bytes::from(vec![0u8; 4]));
        let slice = tensor.slice();
        assert!(slice.shape.is_empty());
        assert_eq!(slice.data.len(), 4); // one f32
        assert_eq!(slice.dtype, Dtype::F32);
    }

    // ── OnnxTensor: 3D shape slice has correct byte count ───────────────
    // @trace TEST-ONNX-284 [level:unit]

    #[test]
    fn onnx_tensor_3d_shape_slice_byte_count() {
        let tensor = OnnxTensor::new("3d".to_string(), Dtype::F32, vec![2, 3, 4], Bytes::from(vec![0u8; 96]));
        let slice = tensor.slice();
        assert_eq!(slice.shape, vec![2, 3, 4]);
        assert_eq!(slice.data.len(), 2 * 3 * 4 * 4);
    }

    // ── OnnxLoader::names returns sorted canonical names ────────────────
    // @trace TEST-ONNX-285 [level:unit]

    #[test]
    fn onnx_loader_names_returns_sorted_display_names() {
        let w1 = make_f32_tensor("model.layers.0.attn.q_proj.MatMul.weight", vec![2, 2]);
        let w2 = make_f32_tensor("embed.weight", vec![10, 8]);
        let graph = build_graph(vec![], vec![w1, w2]);

        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::from([
                ("model.layers.0.self_attn.q_proj.weight".to_string(), "model.layers.0.attn.q_proj.MatMul.weight".to_string()),
            ]),
            reverse_alias: HashMap::from([
                ("model.layers.0.attn.q_proj.MatMul.weight".to_string(), "model.layers.0.self_attn.q_proj.weight".to_string()),
            ]),
            layout_hints: HashMap::new(),
        };

        let names = loader.names();

        // embed.weight has no alias so it stays as-is; the other gets its canonical name
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "embed.weight");
        assert_eq!(names[1], "model.layers.0.self_attn.q_proj.weight");
    }

    // ── OnnxLoader::tensor resolves canonical name to data ──────────────
    // @trace TEST-ONNX-286 [level:unit]

    #[test]
    fn onnx_loader_tensor_resolves_alias() {
        let w = make_f32_tensor("onnx::MatMul_1", vec![2, 3]);
        let graph = build_graph(vec![], vec![w]);

        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::from([
                ("encoder.weight".to_string(), "onnx::MatMul_1".to_string()),
            ]),
            reverse_alias: HashMap::from([
                ("onnx::MatMul_1".to_string(), "encoder.weight".to_string()),
            ]),
            layout_hints: HashMap::new(),
        };

        // Resolve via alias
        let slice = loader.tensor("encoder.weight").unwrap();
        assert_eq!(slice.shape, vec![2, 3]);
        assert_eq!(slice.dtype, Dtype::F32);
        assert_eq!(slice.data.len(), 2 * 3 * 4);
    }

    // ── OnnxLoader::tensor returns error for missing tensor ─────────────
    // @trace TEST-ONNX-287 [level:unit]

    #[test]
    fn onnx_loader_tensor_missing_returns_error() {
        let graph = build_graph(vec![], vec![]);

        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::new(),
            reverse_alias: HashMap::new(),
            layout_hints: HashMap::new(),
        };

        let result = loader.tensor("nonexistent.weight");
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("nonexistent.weight"));
    }

    // ── OnnxLoader::tensor_dtype returns correct dtype ──────────────────
    // @trace TEST-ONNX-288 [level:unit]

    #[test]
    fn onnx_loader_tensor_dtype_returns_dtype() {
        let bf16_data = vec![0u8; 4];
        let tensor = OnnxTensor::new("bf16_w".to_string(), Dtype::BF16, vec![2], Bytes::from(bf16_data));
        let graph = build_graph(vec![], vec![tensor]);

        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::new(),
            reverse_alias: HashMap::new(),
            layout_hints: HashMap::new(),
        };

        assert_eq!(loader.tensor_dtype("bf16_w").unwrap(), Dtype::BF16);
    }

    // ── OnnxLoader::unique_precisions deduplicates and sorts ────────────
    // @trace TEST-ONNX-289 [level:unit]

    #[test]
    fn onnx_loader_unique_precisions_deduplicates_and_sorts() {
        let t1 = make_f32_tensor("w1", vec![2]);
        let t2 = make_f32_tensor("w2", vec![2]);
        let bf16_data = vec![0u8; 4];
        let t3 = OnnxTensor::new("bf16_w".to_string(), Dtype::BF16, vec![2], Bytes::from(bf16_data));
        let graph = build_graph(vec![], vec![t1, t2, t3]);

        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::new(),
            reverse_alias: HashMap::new(),
            layout_hints: HashMap::new(),
        };

        let precisions = loader.unique_precisions();

        // Two distinct dtypes (F32, BF16), deduplicated and sorted by dtype_rank (lower = higher precision first)
        assert_eq!(precisions.len(), 2);
        assert_eq!(precisions[0], Dtype::F32); // rank=1
        assert_eq!(precisions[1], Dtype::BF16); // rank=2
    }

    // ── OnnxLoader::precision_by_tensor returns sorted list ─────────────
    // @trace TEST-ONNX-290 [level:unit]

    #[test]
    fn onnx_loader_precision_by_tensor_sorted_by_name() {
        let t_bf16 = OnnxTensor::new("a_bf16".to_string(), Dtype::BF16, vec![1], Bytes::from(vec![0u8; 2]));
        let t_f32 = make_f32_tensor("b_f32", vec![1]);
        let graph = build_graph(vec![], vec![t_bf16, t_f32]);

        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::new(),
            reverse_alias: HashMap::new(),
            layout_hints: HashMap::new(),
        };

        let precisions = loader.precision_by_tensor();

        assert_eq!(precisions.len(), 2);
        assert_eq!(precisions[0].0, "a_bf16");
        assert_eq!(precisions[0].1, Dtype::BF16);
        assert_eq!(precisions[1].0, "b_f32");
        assert_eq!(precisions[1].1, Dtype::F32);
    }

    // ── OnnxLoader graph/model/path accessors ───────────────────────────
    // @trace TEST-ONNX-291 [level:unit]

    #[test]
    fn onnx_loader_accessors_return_correct_references() {
        let graph = build_graph(vec![], vec![]);
        let model = OnnxModel {
            metadata: crate::loader::onnx::model::OnnxModelMetadata {
                ir_version: 8,
                producer_name: "accessor_test".to_string(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph,
            functions: vec![],
        };

        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/accessor_test.onnx"),
            model,
            alias_map: HashMap::new(),
            reverse_alias: HashMap::new(),
            layout_hints: HashMap::new(),
        };

        assert_eq!(loader.path(), Path::new("/tmp/accessor_test.onnx"));
        assert_eq!(loader.graph().name, "test_graph");
        assert_eq!(loader.model().metadata.producer_name, "accessor_test");
    }

    // ── TensorProvider::tensor_info via fully-qualified call ────────────
    // @trace TEST-ONNX-292 [level:unit]

    #[test]
    fn tensor_provider_tensor_info_with_alias_resolution() {
        use super::super::TensorProvider;
        let w = make_f32_tensor("onnx::MatMul_42", vec![4, 8]);
        let graph = build_graph(vec![], vec![w]);

        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::from([
                ("encoder.weight".to_string(), "onnx::MatMul_42".to_string()),
            ]),
            reverse_alias: HashMap::from([
                ("onnx::MatMul_42".to_string(), "encoder.weight".to_string()),
            ]),
            layout_hints: HashMap::new(),
        };

        let meta = TensorProvider::tensor_info(&loader, "encoder.weight");
        assert!(meta.is_some());
        let m = meta.unwrap();
        assert_eq!(m.name, "encoder.weight");
        assert_eq!(m.shape, vec![4, 8]);
        assert_eq!(m.dtype, Dtype::F32);
    }

    // ── TensorProvider::tensor_info returns None for missing ────────────
    // @trace TEST-ONNX-293 [level:unit]

    #[test]
    fn tensor_provider_tensor_info_missing_returns_none() {
        use super::super::TensorProvider;
        let graph = build_graph(vec![], vec![]);

        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::new(),
            reverse_alias: HashMap::new(),
            layout_hints: HashMap::new(),
        };

        assert!(TensorProvider::tensor_info(&loader, "missing.weight").is_none());
    }

    // ── TensorProvider::iter_tensors returns all tensors via trait ──────
    // @trace TEST-ONNX-294 [level:unit]

    #[test]
    fn tensor_provider_iter_tensors_returns_all_with_display_names() {
        use super::super::TensorProvider;
        let t1 = make_f32_tensor("onnx::MatMul_1", vec![3, 3]);
        let t2 = make_f32_tensor("bias", vec![3]);
        let graph = build_graph(vec![], vec![t1, t2]);

        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::from([
                ("fc.weight".to_string(), "onnx::MatMul_1".to_string()),
            ]),
            reverse_alias: HashMap::from([
                ("onnx::MatMul_1".to_string(), "fc.weight".to_string()),
            ]),
            layout_hints: HashMap::new(),
        };

        let mut metas: Vec<_> = loader.iter_tensors().collect();
        metas.sort_by(|a, b| a.name.cmp(&b.name));

        assert_eq!(metas.len(), 2);
        assert_eq!(metas[0].name, "bias");
        assert_eq!(metas[0].shape, vec![3]);
        assert_eq!(metas[1].name, "fc.weight");
        assert_eq!(metas[1].shape, vec![3, 3]);
    }

    // ── TensorProvider::load_tensor_data returns correct bytes ──────────
    // @trace TEST-ONNX-295 [level:unit]

    #[test]
    fn tensor_provider_load_tensor_data_returns_correct_bytes() {
        use super::super::TensorProvider;
        let tensor = OnnxTensor::new("direct_w".to_string(), Dtype::F32, vec![2], Bytes::from(vec![1u8, 2, 3, 4, 5, 6, 7, 8]));
        let graph = build_graph(vec![], vec![tensor]);

        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::new(),
            reverse_alias: HashMap::new(),
            layout_hints: HashMap::new(),
        };

        let data = loader.load_tensor_data("direct_w").unwrap();
        assert_eq!(&*data, &[1u8, 2, 3, 4, 5, 6, 7, 8]);
    }

    // ── TensorProvider::weight_layout_hint resolves via alias ───────────
    // @trace TEST-ONNX-296 [level:unit]

    #[test]
    fn tensor_provider_weight_layout_hint_resolves_alias() {
        use super::super::TensorProvider;
        let w = make_f32_tensor("onnx::Gemm_7", vec![4, 4]);
        let graph = build_graph(vec![], vec![w]);

        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::from([
                ("fc.weight".to_string(), "onnx::Gemm_7".to_string()),
            ]),
            reverse_alias: HashMap::from([
                ("onnx::Gemm_7".to_string(), "fc.weight".to_string()),
            ]),
            layout_hints: HashMap::from([
                ("onnx::Gemm_7".to_string(), true),
            ]),
        };

        // Alias name resolves to onnx name, then looks up layout hint
        assert_eq!(loader.weight_layout_hint("fc.weight"), Some(true));
        // Direct onnx name also works
        assert_eq!(loader.weight_layout_hint("onnx::Gemm_7"), Some(true));
        // Unknown name returns None
        assert_eq!(loader.weight_layout_hint("missing"), None);
    }

    // ── OnnxLoader::resolve prefers direct initializer over alias ───────
    // @trace TEST-ONNX-297 [level:unit]

    #[test]
    fn onnx_loader_resolve_prefers_direct_initializer() {
        let w = make_f32_tensor("direct_name", vec![2, 2]);
        let graph = build_graph(vec![], vec![w]);

        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::new(),
            reverse_alias: HashMap::new(),
            layout_hints: HashMap::new(),
        };

        // Direct initializer name resolves to itself
        assert_eq!(loader.resolve("direct_name"), Some("direct_name"));
        // Unknown name returns None
        assert_eq!(loader.resolve("nonexistent"), None);
    }

    // ══════════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS (15 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── decode_model with non-existent file returns IO error ──────────────
    // @trace TEST-ONNX-298 [level:unit]

    #[test]
    fn decode_model_nonexistent_file_returns_io_error() {
        let result = decode_model(Path::new("/nonexistent/path/model.onnx"));
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("IO error"));
    }

    // ── decode_model with invalid protobuf bytes returns Onnx error ──────
    // @trace TEST-ONNX-299 [level:unit]

    #[test]
    fn decode_model_invalid_bytes_returns_onnx_error() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("invalid.onnx");
        std::fs::write(&file_path, b"not valid protobuf data").unwrap();
        let result = decode_model(&file_path);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("onnx decode failed"));
    }

    // ── external_data_locations with model missing graph returns empty ────
    // @trace TEST-ONNX-300 [level:unit]

    #[test]
    fn external_data_locations_no_graph_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("no_graph.onnx");
        let model = proto::ModelProto {
            graph: None,
            ..Default::default()
        };
        let mut buf = Vec::new();
        prost::Message::encode(&model, &mut buf).unwrap();
        std::fs::write(&file_path, &buf).unwrap();
        let result = external_data_locations(&file_path).unwrap();
        assert!(result.is_empty());
    }

    // ── external_data_locations with empty graph returns empty ────────────
    // @trace TEST-ONNX-301 [level:unit]

    #[test]
    fn external_data_locations_empty_graph_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("empty_graph.onnx");
        let graph = proto::GraphProto::default();
        let model = proto::ModelProto {
            graph: Some(graph),
            ..Default::default()
        };
        let mut buf = Vec::new();
        prost::Message::encode(&model, &mut buf).unwrap();
        std::fs::write(&file_path, &buf).unwrap();
        let result = external_data_locations(&file_path).unwrap();
        assert!(result.is_empty());
    }

    // ── OnnxLoader::from_path with corrupted file returns error ───────────
    // @trace TEST-ONNX-302 [level:unit]

    #[test]
    fn onnx_loader_from_path_corrupted_file_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("bad.onnx");
        std::fs::write(&file_path, b"corrupt data that is not protobuf").unwrap();
        let result = OnnxLoader::from_path(&file_path);
        assert!(result.is_err());
    }

    // ── OnnxLoader with zero initializers returns empty names ─────────────
    // @trace TEST-ONNX-303 [level:unit]

    #[test]
    fn onnx_loader_zero_initializers_returns_empty_names() {
        let graph = build_graph(vec![], vec![]);
        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/zero_init.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::new(),
            reverse_alias: HashMap::new(),
            layout_hints: HashMap::new(),
        };

        assert!(loader.names().is_empty());
        assert!(loader.unique_precisions().is_empty());
        assert!(loader.precision_by_tensor().is_empty());
    }

    // ── OnnxGraph with value_info populated preserves entries ─────────────
    // @trace TEST-ONNX-304 [level:unit]

    #[test]
    fn onnx_graph_with_value_info_preserves_entries() {
        let vi1 = OnnxValueInfo {
            name: "intermediate.0".to_string(),
            value_type: Some(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Float,
                shape: OnnxTensorShape {
                    dims: vec![OnnxDim::Param("seq".to_string()), OnnxDim::Known(768)],
                },
            })),
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let vi2 = OnnxValueInfo {
            name: "intermediate.1".to_string(),
            value_type: Some(OnnxType::Tensor(OnnxTensorType {
                elem_type: proto::tensor_proto::DataType::Int64,
                shape: OnnxTensorShape {
                    dims: vec![OnnxDim::Known(10)],
                },
            })),
            doc_string: "index".to_string(),
            metadata_props: HashMap::new(),
        };
        let graph = OnnxGraph {
            name: "value_info_test".to_string(),
            doc_string: String::new(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![vi1, vi2],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.value_info.len(), 2);
        assert_eq!(graph.value_info[0].name, "intermediate.0");
        assert_eq!(graph.value_info[1].name, "intermediate.1");
        assert_eq!(graph.value_info[1].doc_string, "index");
    }

    // ── OnnxDim::Param with empty string is distinct from Unknown ─────────
    // @trace TEST-ONNX-305 [level:unit]

    #[test]
    fn onnx_dim_param_empty_string_distinct_from_unknown() {
        let param_empty = OnnxDim::Param(String::new());
        assert_ne!(param_empty, OnnxDim::Unknown);
        assert_ne!(param_empty, OnnxDim::Known(0));
        // Two empty Param should be equal
        assert_eq!(OnnxDim::Param(String::new()), OnnxDim::Param(String::new()));
    }

    // ── OnnxType clone round-trip for all variant shapes ──────────────────
    // @trace TEST-ONNX-306 [level:unit]

    #[test]
    fn onnx_type_clone_roundtrip_preserves_shape() {
        let original = OnnxType::Tensor(OnnxTensorType {
            elem_type: proto::tensor_proto::DataType::Bfloat16,
            shape: OnnxTensorShape {
                dims: vec![OnnxDim::Param("batch".to_string()), OnnxDim::Known(4096)],
            },
        });
        let cloned = original.clone();
        assert_eq!(original, cloned);
        if let OnnxType::Tensor(tt) = &cloned {
            assert_eq!(tt.elem_type, proto::tensor_proto::DataType::Bfloat16);
            assert_eq!(tt.shape.dims.len(), 2);
        } else {
            panic!("expected Tensor variant");
        }
    }

    // ── OnnxSparseTensor with Csc format preserves all fields ─────────────
    // @trace TEST-ONNX-307 [level:unit]

    #[test]
    fn onnx_sparse_tensor_csc_format_preserves_fields() {
        let values = OnnxTensor::new("csc_vals".to_string(), Dtype::F32, vec![5], Bytes::from(vec![0u8; 20]));
        let indices = OnnxTensor::new("csc_idx".to_string(), Dtype::I64, vec![5], Bytes::from(vec![0u8; 40]));
        let sparse = OnnxSparseTensor {
            values,
            indices,
            dims: vec![8, 8],
            format: OnnxSparseFormat::Csc,
        };
        assert_eq!(sparse.format, OnnxSparseFormat::Csc);
        assert_eq!(sparse.dims, vec![8, 8]);
        assert_eq!(sparse.values.name, "csc_vals");
        assert_eq!(sparse.indices.name, "csc_idx");
        assert_eq!(sparse.values.dtype, Dtype::F32);
        assert_eq!(sparse.indices.dtype, Dtype::I64);
    }

    // ── OnnxModel with multiple functions preserves order and content ─────
    // @trace TEST-ONNX-308 [level:unit]

    #[test]
    fn onnx_model_multiple_functions_preserves_order() {
        let func1 = OnnxFunction {
            name: "If".to_string(),
            domain: "ai.onnx".to_string(),
            overload: String::new(),
            inputs: vec!["cond".to_string()],
            outputs: vec!["result".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let func2 = OnnxFunction {
            name: "Loop".to_string(),
            domain: "ai.onnx".to_string(),
            overload: String::new(),
            inputs: vec!["max_trip_count".to_string(), "condition".to_string()],
            outputs: vec!["output".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        let model = OnnxModel {
            metadata: crate::loader::onnx::model::OnnxModelMetadata {
                ir_version: 8,
                producer_name: String::new(),
                producer_version: String::new(),
                domain: String::new(),
                model_version: 0,
                doc_string: String::new(),
                opset_import: vec![],
                metadata_props: HashMap::new(),
            },
            graph: OnnxGraph {
                name: "main".to_string(),
                doc_string: String::new(),
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                value_info: vec![],
                initializers: HashMap::new(),
                sparse_initializers: vec![],
                quantization_annotation: vec![],
                metadata_props: HashMap::new(),
            },
            functions: vec![func1, func2],
        };
        assert_eq!(model.functions.len(), 2);
        assert_eq!(model.functions[0].name, "If");
        assert_eq!(model.functions[1].name, "Loop");
        assert_eq!(model.functions[1].inputs.len(), 2);
    }

    // ── OnnxAttributeValue: clone round-trip for Int variant ──────────────
    // @trace TEST-ONNX-309 [level:unit]

    #[test]
    fn onnx_attribute_value_int_clone_roundtrip() {
        let val = OnnxAttributeValue::Int(42);
        let cloned = val.clone();
        assert!(matches!(cloned, OnnxAttributeValue::Int(42)));
    }

    // ── build_alias_map: pass 2 with canonical name colliding with initializer name
    // @trace TEST-ONNX-310 [level:unit]

    #[test]
    fn build_alias_map_pass2_skips_when_canonical_matches_existing_initializer() {
        // A real initializer named "model.layers.0.self_attn.q_proj.weight" already exists
        let real = make_f32_tensor("model.layers.0.self_attn.q_proj.weight", vec![2, 2]);
        // A second initializer whose canonical would collide
        let onnx_named = make_f32_tensor(
            "model.layers.0.attn.q_proj.MatMul.weight",
            vec![2, 2],
        );
        let graph = build_graph(vec![], vec![real, onnx_named]);

        let (alias_map, reverse_alias, _) = build_alias_map(&graph);

        // The canonical "model.layers.0.self_attn.q_proj.weight" already exists as an
        // initializer, so no alias should be created for the onnx-named one
        assert!(alias_map.get("model.layers.0.self_attn.q_proj.weight").is_none()
            || alias_map.get("model.layers.0.self_attn.q_proj.weight").unwrap()
                != &"model.layers.0.attn.q_proj.MatMul.weight".to_string()
                || reverse_alias.is_empty());
    }

    // ── OnnxAttribute with all optional fields populated ──────────────────
    // @trace TEST-ONNX-311 [level:unit]

    #[test]
    fn onnx_attribute_all_fields_populated() {
        let attr = OnnxAttribute {
            name: "padding".to_string(),
            value: OnnxAttributeValue::Ints(vec![1, 2, 3]),
            doc_string: "padding sizes".to_string(),
            ref_attr_name: Some("pad_ref".to_string()),
            attr_type: Some(proto::attribute_proto::AttributeType::Ints),
        };
        assert_eq!(attr.name, "padding");
        assert!(matches!(attr.value, OnnxAttributeValue::Ints(ref v) if v == &vec![1, 2, 3]));
        assert_eq!(attr.doc_string, "padding sizes");
        assert_eq!(attr.ref_attr_name.as_deref(), Some("pad_ref"));
        assert!(matches!(attr.attr_type, Some(proto::attribute_proto::AttributeType::Ints)));
    }

    // ── collect_graph_external_locations with mixed sources returns all ───
    // @trace TEST-ONNX-312 [level:unit]

    #[test]
    fn collect_graph_external_locations_mixed_sources_returns_all() {
        let init_tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("init.bin".to_string()),
            }],
            ..Default::default()
        };
        let attr_tensor = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("attr.bin".to_string()),
            }],
            ..Default::default()
        };
        let sparse_vals = proto::TensorProto {
            data_location: Some(proto::tensor_proto::DataLocation::External as i32),
            external_data: vec![proto::StringStringEntryProto {
                key: Some("location".to_string()),
                value: Some("sparse.bin".to_string()),
            }],
            ..Default::default()
        };
        let graph = proto::GraphProto {
            initializer: vec![init_tensor],
            sparse_initializer: vec![proto::SparseTensorProto {
                values: Some(sparse_vals),
                indices: None,
                dims: vec![],
            }],
            node: vec![proto::NodeProto {
                attribute: vec![proto::AttributeProto {
                    t: Some(attr_tensor),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut out = BTreeSet::new();

        collect_graph_external_locations(&graph, &mut out);

        assert_eq!(out.len(), 3);
        assert!(out.contains("init.bin"));
        assert!(out.contains("attr.bin"));
        assert!(out.contains("sparse.bin"));
    }

    // ── TensorProvider::load_tensor_data returns error for missing tensor ─
    // @trace TEST-ONNX-313 [level:unit]

    #[test]
    fn tensor_provider_load_tensor_data_missing_returns_error() {
        use super::super::TensorProvider;
        let graph = build_graph(vec![], vec![]);
        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::new(),
            reverse_alias: HashMap::new(),
            layout_hints: HashMap::new(),
        };

        let result = loader.load_tensor_data("nonexistent_tensor");
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("nonexistent_tensor"));
    }

    // ── OnnxTensor: raw_data on empty tensor returns empty slice ──────────
    // @trace TEST-ONNX-314 [level:unit]

    #[test]
    fn onnx_tensor_raw_data_empty_tensor() {
        let tensor = OnnxTensor::new("empty_t".to_string(), Dtype::F32, vec![0], Bytes::new());
        assert!(tensor.raw_data().is_empty());
        assert_eq!(tensor.shape, vec![0]);
    }

    // ── OnnxGraph clone round-trip preserves initializers count ───────────
    // @trace TEST-ONNX-315 [level:unit]

    #[test]
    fn onnx_graph_clone_roundtrip_preserves_initializers() {
        let w1 = make_f32_tensor("weight_a", vec![3, 3]);
        let w2 = make_f32_tensor("weight_b", vec![5, 5]);
        let w3 = OnnxTensor::new("bf16_bias".to_string(), Dtype::BF16, vec![5], Bytes::from(vec![0u8; 10]));
        let graph = build_graph(
            vec![OnnxNode {
                name: "fc".to_string(),
                op_type: "Gemm".to_string(),
                domain: String::new(),
                inputs: vec!["x".to_string(), "weight_a".to_string()],
                outputs: vec!["y".to_string()],
                attributes: HashMap::new(),
            }],
            vec![w1, w2, w3],
        );
        let cloned = graph.clone();
        assert_eq!(cloned.initializers.len(), 3);
        assert!(cloned.initializers.contains_key("weight_a"));
        assert!(cloned.initializers.contains_key("weight_b"));
        assert!(cloned.initializers.contains_key("bf16_bias"));
        assert_eq!(cloned.nodes.len(), 1);
        assert_eq!(cloned.nodes[0].op_type, "Gemm");
    }

    // ══════════════════════════════════════════════════════════════════════
    // ADDITIONAL TESTS (10 new)
    // ══════════════════════════════════════════════════════════════════════

    // ── OnnxLoader::tensor_dtype returns error for missing tensor name ────
    // @trace TEST-ONNX-316 [level:unit]

    #[test]
    fn onnx_loader_tensor_dtype_missing_returns_error() {
        let graph = build_graph(vec![], vec![]);
        let loader = OnnxLoader {
            path: PathBuf::from("/tmp/test.onnx"),
            model: OnnxModel {
                metadata: crate::loader::onnx::model::OnnxModelMetadata {
                    ir_version: 8,
                    producer_name: String::new(),
                    producer_version: String::new(),
                    domain: String::new(),
                    model_version: 0,
                    doc_string: String::new(),
                    opset_import: vec![],
                    metadata_props: HashMap::new(),
                },
                graph,
                functions: vec![],
            },
            alias_map: HashMap::new(),
            reverse_alias: HashMap::new(),
            layout_hints: HashMap::new(),
        };

        let result = loader.tensor_dtype("ghost_tensor");
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("ghost_tensor"));
    }

    // ── OnnxAttributeValue::String variant preserves value ────────────────
    // @trace TEST-ONNX-318 [level:unit]

    #[test]
    fn onnx_attribute_value_string_variant_preserves_value() {
        let val = OnnxAttributeValue::String("NCHW".to_string());
        assert!(matches!(val, OnnxAttributeValue::String(ref s) if s == "NCHW"));

        let cloned = val.clone();
        assert!(matches!(cloned, OnnxAttributeValue::String(ref s) if s == "NCHW"));
    }

    // ── OnnxFunction with overload field populated ────────────────────────
    // @trace TEST-ONNX-319 [level:unit]

    #[test]
    fn onnx_function_with_overload_field() {
        let func = OnnxFunction {
            name: "MyOp".to_string(),
            domain: "custom".to_string(),
            overload: "v2_overload".to_string(),
            inputs: vec!["A".to_string()],
            outputs: vec!["B".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.overload, "v2_overload");
        assert_eq!(func.name, "MyOp");
        assert_eq!(func.domain, "custom");
    }

    // ── OnnxFunction with opset_import populated ──────────────────────────
    // @trace TEST-ONNX-320 [level:unit]

    #[test]
    fn onnx_function_with_opset_import() {
        use crate::loader::onnx::model::OnnxOperatorSet;
        let func = OnnxFunction {
            name: "Scan".to_string(),
            domain: "ai.onnx".to_string(),
            overload: String::new(),
            inputs: vec![],
            outputs: vec![],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![],
            opset_import: vec![
                OnnxOperatorSet { domain: "ai.onnx".to_string(), version: 17 },
                OnnxOperatorSet { domain: "custom".to_string(), version: 1 },
            ],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.opset_import.len(), 2);
        assert_eq!(func.opset_import[0].version, 17);
        assert_eq!(func.opset_import[1].domain, "custom");
    }

    // ── OnnxFunction with non-empty nodes field ───────────────────────────
    // @trace TEST-ONNX-321 [level:unit]

    #[test]
    fn onnx_function_with_nodes() {
        let inner_node = OnnxNode {
            name: "add_0".to_string(),
            op_type: "Add".to_string(),
            domain: String::new(),
            inputs: vec!["a".to_string(), "b".to_string()],
            outputs: vec!["c".to_string()],
            attributes: HashMap::new(),
        };
        let func = OnnxFunction {
            name: "Reshape".to_string(),
            domain: String::new(),
            overload: String::new(),
            inputs: vec!["data".to_string()],
            outputs: vec!["reshaped".to_string()],
            attributes: vec![],
            attribute_protos: HashMap::new(),
            nodes: vec![inner_node],
            opset_import: vec![],
            value_info: vec![],
            doc_string: String::new(),
            metadata_props: HashMap::new(),
        };
        assert_eq!(func.nodes.len(), 1);
        assert_eq!(func.nodes[0].op_type, "Add");
        assert_eq!(func.nodes[0].inputs, vec!["a", "b"]);
    }

    // ── OnnxModelMetadata with populated metadata_props map ───────────────
    // @trace TEST-ONNX-322 [level:unit]

    #[test]
    fn onnx_model_metadata_with_populated_metadata_props() {
        use crate::loader::onnx::model::OnnxModelMetadata;
        let mut props = HashMap::new();
        props.insert("license".to_string(), "Apache-2.0".to_string());
        props.insert("model_name".to_string(), "bert-base".to_string());
        let md = OnnxModelMetadata {
            ir_version: 8,
            producer_name: "onnx-test".to_string(),
            producer_version: "2.0".to_string(),
            domain: "ai.onnx".to_string(),
            model_version: 1,
            doc_string: "test model metadata".to_string(),
            opset_import: vec![],
            metadata_props: props,
        };
        assert_eq!(md.metadata_props.len(), 2);
        assert_eq!(md.metadata_props.get("license").unwrap(), "Apache-2.0");
        assert_eq!(md.metadata_props.get("model_name").unwrap(), "bert-base");
        assert_eq!(md.model_version, 1);
        assert_eq!(md.doc_string, "test model metadata");
    }

    // ── OnnxTensor::new_string with empty bytes and zero shape ────────────
    // @trace TEST-ONNX-323 [level:unit]

    #[test]
    fn onnx_tensor_new_string_empty_bytes_zero_shape() {
        let t = OnnxTensor::new_string("empty_str".to_string(), vec![0], Bytes::new());
        assert!(t.is_string);
        assert_eq!(t.dtype, Dtype::U8);
        assert_eq!(t.shape, vec![0]);
        assert!(t.raw_data().is_empty());
        assert_eq!(t.name, "empty_str");
    }

    // ── build_alias_map: Gemm named initializer with transB gets hint ─────
    // @trace TEST-ONNX-324 [level:unit]

    #[test]
    fn build_alias_map_gemm_named_initializer_transb_gets_hint() {
        let weight = make_f32_tensor("encoder.weight", vec![4, 4]);
        let node = OnnxNode {
            name: "/encoder/Gemm".to_string(),
            op_type: "Gemm".to_string(),
            domain: String::new(),
            inputs: vec!["input".to_string(), "encoder.weight".to_string()],
            outputs: vec!["out".to_string()],
            attributes: HashMap::from([(
                "transB".to_string(),
                OnnxAttribute {
                    name: "transB".to_string(),
                    value: OnnxAttributeValue::Int(1),
                    doc_string: String::new(),
                    ref_attr_name: None,
                    attr_type: None,
                },
            )]),
        };
        let graph = build_graph(vec![node], vec![weight]);

        let (alias_map, _, layout_hints) = build_alias_map(&graph);

        // Named initializer does not start with "onnx::" and has no op infix, so no alias
        assert!(alias_map.is_empty());
        // But layout hint is recorded: transB=1 means true
        assert_eq!(layout_hints.get("encoder.weight"), Some(&true));
    }

    // ── OnnxAttributeValue::Float variant preserves value ──────────────────
    // @trace TEST-ONNX-325 [level:unit]

    #[test]
    fn onnx_attribute_value_float_variant_preserves_value() {
        let val = OnnxAttributeValue::Float(0.0078125);
        assert!(matches!(val, OnnxAttributeValue::Float(f) if (f - 0.0078125).abs() < f32::EPSILON));

        let cloned = val.clone();
        assert!(matches!(cloned, OnnxAttributeValue::Float(f) if (f - 0.0078125).abs() < f32::EPSILON));
    }

    // ── OnnxGraph with doc_string preserves field ─────────────────────────
    // @trace TEST-ONNX-326 [level:unit]

    #[test]
    fn onnx_graph_doc_string_preserves_field() {
        let graph = OnnxGraph {
            name: "bert_encoder".to_string(),
            doc_string: "BERT encoder graph with 12 layers".to_string(),
            nodes: vec![],
            inputs: vec![],
            outputs: vec![],
            value_info: vec![],
            initializers: HashMap::new(),
            sparse_initializers: vec![],
            quantization_annotation: vec![],
            metadata_props: HashMap::new(),
        };
        assert_eq!(graph.doc_string, "BERT encoder graph with 12 layers");
        assert_eq!(graph.name, "bert_encoder");
    }
}
