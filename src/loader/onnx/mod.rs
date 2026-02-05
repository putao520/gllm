//! ONNX loader with graph parsing and fused-first pattern matching.

use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::MmapOptions;
use prost::bytes::Bytes;
use prost::Message;

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

pub use matcher::{FusedGraph, FusedKernel, FusedOp};
pub use model::{OnnxGraph, OnnxModel, OnnxNode};
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

fn decode_model(path: &Path) -> Result<proto::ModelProto> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let bytes = Bytes::from_owner(mmap);
    proto::ModelProto::decode(bytes)
        .map_err(|err| LoaderError::Onnx(format!("onnx decode failed: {err}")))
}
