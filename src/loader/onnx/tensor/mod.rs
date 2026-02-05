use prost::bytes::Bytes;
use safetensors::Dtype;

use super::external::ExternalDataResolver;
use super::{pack, proto, LoaderError, Result, TensorSlice};

mod parse;

#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    data: Bytes,
}

#[derive(Debug, Clone)]
pub struct OnnxSparseTensor {
    pub values: OnnxTensor,
    pub indices: OnnxTensor,
    pub dims: Vec<usize>,
}

impl OnnxTensor {
    pub(super) fn from_initializer(
        proto: proto::TensorProto,
        resolver: &mut ExternalDataResolver,
    ) -> Result<Self> {
        Self::from_proto(proto, resolver, NamePolicy::Require)
    }

    pub(super) fn from_attribute(
        proto: proto::TensorProto,
        resolver: &mut ExternalDataResolver,
        fallback: &str,
    ) -> Result<Self> {
        Self::from_proto(proto, resolver, NamePolicy::Fallback(fallback))
    }

    pub(super) fn slice(&self) -> TensorSlice<'_> {
        TensorSlice {
            dtype: self.dtype,
            shape: self.shape.clone(),
            data: self.data.as_ref(),
        }
    }

    pub(super) fn scalar_f32(&self) -> Option<f32> {
        if self.element_count() != 1 {
            return None;
        }
        let bytes = self.data.as_ref();
        match self.dtype {
            Dtype::F32 => parse::slice_to_f32(bytes.get(0..4)?),
            Dtype::F16 => parse::slice_to_f16(bytes.get(0..2)?)
                .map(|bits| half::f16::from_bits(bits).to_f32()),
            Dtype::BF16 => parse::slice_to_f16(bytes.get(0..2)?)
                .map(|bits| half::bf16::from_bits(bits).to_f32()),
            Dtype::I32 => parse::slice_to_i32(bytes.get(0..4)?).map(|value| value as f32),
            Dtype::I64 => parse::slice_to_i64(bytes.get(0..8)?).map(|value| value as f32),
            Dtype::U8 => bytes.get(0).map(|b| *b as f32),
            Dtype::U16 => parse::slice_to_u16(bytes.get(0..2)?).map(|value| value as f32),
            Dtype::U32 => parse::slice_to_u32(bytes.get(0..4)?).map(|value| value as f32),
            Dtype::U64 => parse::slice_to_u64(bytes.get(0..8)?).map(|value| value as f32),
            _ => None,
        }
    }

    fn from_proto(
        proto: proto::TensorProto,
        resolver: &mut ExternalDataResolver,
        name_policy: NamePolicy<'_>,
    ) -> Result<Self> {
        let proto::TensorProto {
            dims,
            data_type,
            segment,
            float_data,
            int32_data,
            string_data,
            int64_data,
            name,
            raw_data,
            double_data,
            uint64_data,
            data_location,
            external_data,
            ..
        } = proto;

        let name = resolve_name(name.unwrap_or_default(), name_policy)?;
        let data_type = parse::parse_data_type(
            data_type
                .ok_or_else(|| LoaderError::Onnx(format!("tensor {name} missing data_type")))?,
            &name,
        )?;
        let dtype = parse::map_dtype(data_type, &name)?;
        let shape = parse::parse_dims(&dims, &name)?;
        let element_count = parse::element_count(&shape, &name)?;

        if segment.is_some() {
            return Err(LoaderError::Onnx(format!(
                "segmented tensor not supported: {name}"
            )));
        }

        let data_location = data_location.unwrap_or_default();
        let data = if data_location == proto::tensor_proto::DataLocation::External as i32 {
            parse::load_external_data(resolver, &external_data, dtype, element_count, &name)?
        } else {
            pack::build_tensor_bytes(
                data_type,
                dtype,
                element_count,
                raw_data.unwrap_or_default(),
                float_data,
                int32_data,
                int64_data,
                double_data,
                uint64_data,
                string_data,
                &name,
            )?
        };

        Ok(Self {
            name,
            dtype,
            shape,
            data,
        })
    }

    fn element_count(&self) -> usize {
        if self.shape.is_empty() {
            return 1;
        }
        self.shape.iter().product()
    }
}

impl OnnxSparseTensor {
    pub(super) fn from_proto(
        proto: proto::SparseTensorProto,
        resolver: &mut ExternalDataResolver,
    ) -> Result<Self> {
        let proto::SparseTensorProto {
            values,
            indices,
            dims,
        } = proto;
        let values = values
            .ok_or_else(|| LoaderError::Onnx("sparse tensor missing values tensor".to_string()))?;
        let indices = indices
            .ok_or_else(|| LoaderError::Onnx("sparse tensor missing indices tensor".to_string()))?;
        let values = OnnxTensor::from_initializer(values, resolver)?;
        let indices = OnnxTensor::from_attribute(indices, resolver, "sparse_indices")?;
        let dims = parse::parse_dims(&dims, "sparse_tensor")?;
        Ok(Self {
            values,
            indices,
            dims,
        })
    }
}

enum NamePolicy<'a> {
    Require,
    Fallback(&'a str),
}

fn resolve_name(name: String, policy: NamePolicy<'_>) -> Result<String> {
    if !name.is_empty() {
        return Ok(name);
    }
    match policy {
        NamePolicy::Require => Err(LoaderError::Onnx(
            "initializer tensor missing name".to_string(),
        )),
        NamePolicy::Fallback(fallback) => Ok(fallback.to_string()),
    }
}
