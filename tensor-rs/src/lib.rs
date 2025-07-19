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

/// Represents a multi-dimensional tensor with generic type T and a fixed number of dimensions N.
/// Allow reshaping, slicing, and mathematical operations on tensors.
///
/// The tensor is stored as a contiguous block of memory, with shape and strides defined for each dimension. Slicing and
/// reshaping operations are not allocating new memory, but rather creating views into the existing data.
///
/// # Performance
/// This implementation is not optimized for performance, but provides a clear and flexible API for tensor operations.
///
/// One of the biggest performance bottlenecks is random access to elements: accessing an element at a specific index
/// requires calculating the offset based on the shape and strides.  This makes it slower than accessing elements in
/// a contiguous array which makes it less suitable for high-performance applications (e.g. matrix multiplications in ML).
#[derive(Clone)]
pub struct Tensor<T, const N: usize = 4> {
    data: Arc<Vec<T>>,
    shape: SmallVec<usize, N>,
    strides: SmallVec<usize, N>,
    offset: usize,
}
