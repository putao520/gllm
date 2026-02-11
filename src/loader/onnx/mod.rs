//! ONNX loader with graph parsing and fused-first pattern matching.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::MmapOptions;
use prost::bytes::Bytes;
use prost::Message;
use safetensors::Dtype;

use super::{LoaderError, Result, TensorSlice};

pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

mod attributes;
mod external;
mod matcher;
mod model;
mod pack;
mod tensor;
mod types;

#[cfg(test)]
mod tests;

pub use attributes::{OnnxAttribute, OnnxAttributeValue};
pub use matcher::{FusedGraph, FusedKernel, FusedOp};
pub use model::{OnnxGraph, OnnxModel, OnnxNode, OnnxValueInfo};
pub use tensor::{OnnxSparseTensor, OnnxTensor};

use external::ExternalDataResolver;

#[derive(Debug)]
pub struct OnnxLoader {
    path: PathBuf,
    model: OnnxModel,
    fused: FusedGraph,
}

impl OnnxLoader {
    pub fn from_path(path: &Path) -> Result<Self> {
        let model_proto = decode_model(path)?;
        let mut resolver = ExternalDataResolver::new(path);
        let model = OnnxModel::from_proto(model_proto, &mut resolver)?;
        let fused = matcher::build_fused_graph(&model.graph)?;
        Ok(Self {
            path: path.to_path_buf(),
            model,
            fused,
        })
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.model.graph.initializers.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn tensor(&self, name: &str) -> Result<TensorSlice<'_>> {
        let tensor = self
            .model
            .graph
            .initializers
            .get(name)
            .ok_or_else(|| LoaderError::MissingTensor(name.to_string()))?;
        Ok(tensor.slice())
    }

    pub fn tensor_dtype(&self, name: &str) -> Result<Dtype> {
        let tensor = self
            .model
            .graph
            .initializers
            .get(name)
            .ok_or_else(|| LoaderError::MissingTensor(name.to_string()))?;
        Ok(tensor.dtype)
    }

    pub fn precision_by_tensor(&self) -> Vec<(String, Dtype)> {
        let mut out = self
            .model
            .graph
            .initializers
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.dtype))
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

    pub fn fused_graph(&self) -> &FusedGraph {
        &self.fused
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl super::TensorProvider for OnnxLoader {
    fn tensor_info(&self, name: &str) -> Option<super::TensorMeta> {
        let tensor = self.model.graph.initializers.get(name)?;
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
            .map(|(name, tensor)| super::TensorMeta {
                name: name.clone(),
                shape: tensor.shape.clone(),
                dtype: tensor.dtype,
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
