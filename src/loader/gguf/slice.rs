use super::GgmlDType;

#[derive(Debug, Clone)]
pub struct TensorSlice<'a> {
    dtype: GgmlDType,
    shape: Vec<u64>,
    data: &'a [u8],
}

impl<'a> TensorSlice<'a> {
    pub(crate) fn new(dtype: GgmlDType, shape: Vec<u64>, data: &'a [u8]) -> Self {
        Self { dtype, shape, data }
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.data
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}
