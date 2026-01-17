use crate::types::{Error, Result};

#[derive(Clone, Debug)]
pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(Error::InferenceError(
                "Matrix data length does not match shape".into(),
            ));
        }
        Ok(Self { data, rows, cols })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn row(&self, idx: usize) -> &[f32] {
        let start = idx * self.cols;
        &self.data[start..start + self.cols]
    }

    pub fn row_mut(&mut self, idx: usize) -> &mut [f32] {
        let start = idx * self.cols;
        &mut self.data[start..start + self.cols]
    }

    pub fn into_tensor3(self, dim0: usize, dim1: usize, dim2: usize) -> Result<Tensor3> {
        if dim0 * dim1 * dim2 != self.data.len() {
            return Err(Error::InferenceError(
                "Tensor3 shape does not match matrix data length".into(),
            ));
        }
        Ok(Tensor3 {
            data: self.data,
            dim0,
            dim1,
            dim2,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Tensor3 {
    pub data: Vec<f32>,
    pub dim0: usize,
    pub dim1: usize,
    pub dim2: usize,
}

impl Tensor3 {
    pub fn new(data: Vec<f32>, dim0: usize, dim1: usize, dim2: usize) -> Result<Self> {
        if data.len() != dim0 * dim1 * dim2 {
            return Err(Error::InferenceError(
                "Tensor3 data length does not match shape".into(),
            ));
        }
        Ok(Self {
            data,
            dim0,
            dim1,
            dim2,
        })
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.dim0, self.dim1, self.dim2)
    }

    pub fn zeros(dim0: usize, dim1: usize, dim2: usize) -> Self {
        Self {
            data: vec![0.0; dim0 * dim1 * dim2],
            dim0,
            dim1,
            dim2,
        }
    }

    pub fn slice(&self, d0: usize, d1: usize) -> &[f32] {
        let start = (d0 * self.dim1 + d1) * self.dim2;
        &self.data[start..start + self.dim2]
    }

    pub fn slice_mut(&mut self, d0: usize, d1: usize) -> &mut [f32] {
        let start = (d0 * self.dim1 + d1) * self.dim2;
        &mut self.data[start..start + self.dim2]
    }

    pub fn into_matrix(self) -> Matrix {
        Matrix {
            data: self.data,
            rows: self.dim0 * self.dim1,
            cols: self.dim2,
        }
    }
}
