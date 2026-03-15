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

use external::ExternalDataResolver;

#[derive(Debug)]
pub struct OnnxLoader {
    path: PathBuf,
    model: OnnxModel,
    alias_map: HashMap<String, String>,     // semantic_name → onnx_name
    reverse_alias: HashMap<String, String>,  // onnx_name → semantic_name
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
fn build_alias_map(graph: &OnnxGraph) -> (HashMap<String, String>, HashMap<String, String>) {
    let mut alias_map = HashMap::new(); // semantic → onnx
    let mut reverse_alias = HashMap::new(); // onnx → semantic

    for node in &graph.nodes {
        // Determine which input index holds the weight and what suffix to use
        let candidates: Vec<(usize, &str)> = match node.op_type.as_str() {
            "MatMul" => vec![(1, ".weight")],
            "Gemm" => vec![(1, ".weight")],
            "Gather" => vec![(0, ".weight")],
            "Mul" => {
                // For Mul, find whichever input is an onnx:: initializer
                let mut v = Vec::new();
                for (i, input) in node.inputs.iter().enumerate() {
                    if input.starts_with("onnx::") && graph.initializers.contains_key(input) {
                        v.push((i, ".weight"));
                    }
                }
                v
            }
            _ => continue,
        };

        for (input_idx, suffix) in candidates {
            let Some(onnx_name) = node.inputs.get(input_idx) else {
                continue;
            };

            // Only alias anonymous onnx:: names
            if !onnx_name.starts_with("onnx::") {
                continue;
            }

            // Must be an actual initializer
            if !graph.initializers.contains_key(onnx_name) {
                continue;
            }

            // Derive semantic name from node.name:
            //   "/encoder/layer.0/attention/self/query/MatMul"
            //   → "encoder.layer.0.attention.self.query" + ".weight"
            let semantic = derive_semantic_name(&node.name, &node.op_type, suffix);
            if semantic.is_empty() {
                continue;
            }

            // Don't overwrite if this semantic name already exists as a real initializer
            if graph.initializers.contains_key(&semantic) {
                continue;
            }

            // Don't overwrite existing aliases (first match wins)
            if alias_map.contains_key(&semantic) {
                continue;
            }

            alias_map.insert(semantic.clone(), onnx_name.clone());
            reverse_alias.insert(onnx_name.clone(), semantic);
        }
    }

    (alias_map, reverse_alias)
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
        let (alias_map, reverse_alias) = build_alias_map(&model.graph);
        Ok(Self {
            path: path.to_path_buf(),
            model,
            alias_map,
            reverse_alias,
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
                    .unwrap_or_else(|| k.clone())
            })
            .collect();
        names.sort();
        names
    }

    pub fn tensor(&self, name: &str) -> Result<TensorSlice<'_>> {
        let resolved = self.resolve(name).unwrap_or(name);
        let tensor = self
            .model
            .graph
            .initializers
            .get(resolved)
            .ok_or_else(|| LoaderError::MissingTensor(name.to_string()))?;
        Ok(tensor.slice())
    }

    pub fn tensor_dtype(&self, name: &str) -> Result<Dtype> {
        let resolved = self.resolve(name).unwrap_or(name);
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
                    .unwrap_or_else(|| name.clone());
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
                    .unwrap_or_else(|| name.clone());
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
