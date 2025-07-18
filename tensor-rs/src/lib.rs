//! Library crate for tensor_rs
//!

use small_vec::{SmallVec, small_vec};
use std::sync::Arc;

mod constructive;
mod display;
mod error;
mod iterator;
mod math;
mod matmul;
mod misc;
mod slicing;
mod view;

pub use crate::error::TensorError;

pub type Result<T> = std::result::Result<T, error::TensorError>;

#[derive(Clone)]
pub struct Tensor<T, const N: usize = 4> {
    data: Arc<Vec<T>>,
    shape: SmallVec<usize, N>,
    strides: SmallVec<usize, N>,
    offset: usize,
}
