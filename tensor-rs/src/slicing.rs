use super::*;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

/// Represents different types of slice indices for tensor slicing operations
#[derive(Debug, Clone, PartialEq)]
pub enum SliceIndex {
    /// Single index that removes a dimension
    Single(isize),
    /// Range with optional start, end, and step
    Range {
        start: Option<isize>,
        end: Option<isize>,
        step: isize,
    },
}

impl SliceIndex {
    /// Create a new range slice with default step of 1
    pub fn range(start: Option<isize>, end: Option<isize>) -> Self {
        Self::Range {
            start,
            end,
            step: 1,
        }
    }

    /// Create a new range slice with custom step
    pub fn range_with_step(start: Option<isize>, end: Option<isize>, step: isize) -> Self {
        if step <= 0 {
            panic!("Step must be positive: {step}");
        }
        Self::Range { start, end, step }
    }

    /// Resolve slice to concrete indices given dimension size
    fn resolve(&self, dim_size: usize) -> Result<SliceResult> {
        let dim_size_i = dim_size as isize;

        match self {
            SliceIndex::Single(idx) => {
                let resolved = normalize_index(*idx, dim_size_i)?;
                Ok(SliceResult::Single(resolved))
            }
            SliceIndex::Range { start, end, step } => {
                let start = start
                    .map(|s| normalize_index(s, dim_size_i))
                    .transpose()?
                    .unwrap_or(0);
                let end = end
                    .map(|e| normalize_index(e, dim_size_i))
                    .transpose()?
                    .unwrap_or(dim_size);

                let start = start.min(dim_size);
                let end = end.min(dim_size);
                let size = if start < end {
                    ((end - start) as isize + step - 1) / step
                } else {
                    0
                };

                Ok(SliceResult::Range {
                    start,
                    size: size as usize,
                    step: *step,
                })
            }
        }
    }
}

/// Result of resolving a slice index
#[derive(Debug, Clone, PartialEq)]
enum SliceResult {
    Single(usize),
    Range {
        start: usize,
        size: usize,
        step: isize,
    },
}

/// Normalize a potentially negative index to a positive one
fn normalize_index(idx: isize, dim_size: isize) -> Result<usize> {
    let resolved = if idx < 0 { dim_size + idx } else { idx };

    if resolved < 0 || resolved >= dim_size {
        return Err(TensorError::IndexOutOfBounds(format!(
            "Index {idx} out of bounds for dimension of size {dim_size}"
        )));
    }

    Ok(resolved as usize)
}

// Consolidated macro for implementing From trait for all range types and numeric types
macro_rules! impl_slice_index_from {
    ($($numeric_type:ty),+) => {
        $(
            // Single index (numeric types)
            impl From<$numeric_type> for SliceIndex {
                fn from(idx: $numeric_type) -> Self {
                    SliceIndex::Single(idx as isize)
                }
            }

            // Range<T>
            impl From<Range<$numeric_type>> for SliceIndex {
                fn from(range: Range<$numeric_type>) -> Self {
                    SliceIndex::range(Some(range.start as isize), Some(range.end as isize))
                }
            }

            // RangeFrom<T>
            impl From<RangeFrom<$numeric_type>> for SliceIndex {
                fn from(range: RangeFrom<$numeric_type>) -> Self {
                    SliceIndex::range(Some(range.start as isize), None)
                }
            }

            // RangeTo<T>
            impl From<RangeTo<$numeric_type>> for SliceIndex {
                fn from(range: RangeTo<$numeric_type>) -> Self {
                    SliceIndex::range(None, Some(range.end as isize))
                }
            }

            // RangeToInclusive<T>
            impl From<RangeToInclusive<$numeric_type>> for SliceIndex {
                fn from(range: RangeToInclusive<$numeric_type>) -> Self {
                    SliceIndex::range(None, Some(range.end as isize + 1))
                }
            }

            // RangeInclusive<T>
            impl From<RangeInclusive<$numeric_type>> for SliceIndex {
                fn from(range: RangeInclusive<$numeric_type>) -> Self {
                    SliceIndex::range(
                        Some(*range.start() as isize),
                        Some(*range.end() as isize + 1)
                    )
                }
            }
        )+
    };
}

// Single macro invocation for all numeric types and range combinations
impl_slice_index_from!(usize, isize, i32, i64, u32, u64);

// RangeFull doesn't need to be parameterized by numeric type
impl From<RangeFull> for SliceIndex {
    fn from(_: RangeFull) -> Self {
        SliceIndex::range(None, None)
    }
}

/// Macro for creating slice indices
#[macro_export]
macro_rules! s {
    () => {
        &[] as &[SliceIndex]
    };
    ($($slice:expr),* $(,)?) => {
        &[$($crate::slicing::SliceIndex::from($slice)),*]
    };
}

/// Macro for creating stepped slices
#[macro_export]
macro_rules! step {
    ($range:expr, $step:expr) => {{
        let base = SliceIndex::from($range);
        match base {
            SliceIndex::Range { start, end, .. } => SliceIndex::range_with_step(start, end, $step),
            _ => panic!("step! can only be used with range expressions"),
        }
    }};
}

impl<T> Tensor<T> {
    pub fn slice(&self, indices: &[SliceIndex]) -> Result<Tensor<T>> {
        if indices.len() > self.shape.len() {
            return Err(TensorError::ShapeMismatch(format!(
                "Too many slice dimensions: {} > {}",
                indices.len(),
                self.shape.len()
            )));
        }

        let mut new_shape = SmallVec::new();
        let mut new_strides = SmallVec::new();
        let mut new_offset = self.offset;

        // Process explicit slice indices
        for (i, slice_idx) in indices.iter().enumerate() {
            let dim_size = self.shape[i];
            let stride = self.strides[i];

            match slice_idx.resolve(dim_size)? {
                SliceResult::Single(start) => {
                    new_offset += start * stride;
                    // Dimension is removed, don't add to new shape/strides
                }
                SliceResult::Range { start, size, step } => {
                    new_shape.push(size);
                    new_strides.push(stride * step as usize);
                    new_offset += start * stride;
                }
            }
        }

        // Add remaining (unsliced) dimensions
        for i in indices.len()..self.shape.len() {
            new_shape.push(self.shape[i]);
            new_strides.push(self.strides[i]);
        }

        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tensor() -> Result<Tensor<i32>> {
        Tensor::arange(24)?.view(&[4, 6])
    }

    fn create_3d_test_tensor() -> Result<Tensor<i32>> {
        Tensor::arange(24)?.view(&[2, 3, 4])
    }

    #[test]
    fn test_basic_slicing() -> Result<()> {
        let tensor = create_test_tensor()?;

        // Full slice
        let full_slice = tensor.slice(s![.., ..])?;
        assert_eq!(full_slice, tensor);

        // Row slice
        let row_slice = tensor.slice(s![1, ..])?;
        let expected = Tensor::from(vec![6, 7, 8, 9, 10, 11]);
        assert_eq!(row_slice, expected);

        // Column slice
        let col_slice = tensor.slice(s![.., 2])?;
        let expected = Tensor::from(vec![2, 8, 14, 20]);
        assert_eq!(col_slice, expected);

        // Range slice
        let range_slice = tensor.slice(s![1..3, 2..5])?;
        let expected = Tensor::new(vec![8, 9, 10, 14, 15, 16], &[2, 3])?;
        assert_eq!(range_slice, expected);

        Ok(())
    }

    #[test]
    fn test_negative_indices() -> Result<()> {
        let tensor = create_test_tensor()?;

        // Negative single index
        let last_row = tensor.slice(s![-1, ..])?;
        let expected = Tensor::from(vec![18, 19, 20, 21, 22, 23]);
        assert_eq!(last_row, expected);

        // Negative range
        let last_two_rows = tensor.slice(s![-2.., ..])?;
        let expected = Tensor::try_from(vec![[12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]])?;
        assert_eq!(last_two_rows, expected);

        // Negative end
        let without_last_col = tensor.slice(s![.., ..-1])?;
        let expected = Tensor::try_from(vec![
            [0, 1, 2, 3, 4],
            [6, 7, 8, 9, 10],
            [12, 13, 14, 15, 16],
            [18, 19, 20, 21, 22],
        ])?;
        assert_eq!(without_last_col, expected);

        Ok(())
    }

    #[test]
    fn test_stepped_slicing() -> Result<()> {
        let tensor = create_test_tensor()?;

        // Step by 2 on first dimension
        let stepped = tensor.slice(&[step![.., 2]])?;
        let expected = Tensor::try_from(vec![[0, 1, 2, 3, 4, 5], [12, 13, 14, 15, 16, 17]])?;
        assert_eq!(stepped, expected);

        // Step with range
        let stepped = tensor.slice(&[step![0..3, 2]])?;
        let expected = Tensor::try_from(vec![[0, 1, 2, 3, 4, 5], [12, 13, 14, 15, 16, 17]])?;
        assert_eq!(stepped, expected);

        Ok(())
    }

    #[test]
    fn test_mixed_slicing() -> Result<()> {
        let tensor = create_3d_test_tensor()?;

        // Mix of single index and ranges
        let mixed = tensor.slice(s![0, 0..2, ..])?;
        let expected = Tensor::try_from(vec![[0, 1, 2, 3], [4, 5, 6, 7]])?;
        assert_eq!(mixed, expected);

        // Mix with negative indices
        let mixed = tensor.slice(s![-1, .., 1..3])?;
        let expected = Tensor::try_from(vec![[13, 14], [17, 18], [21, 22]])?;
        assert_eq!(mixed, expected);

        Ok(())
    }

    #[test]
    fn test_inclusive_ranges() -> Result<()> {
        let tensor = create_test_tensor()?;

        let inclusive = tensor.slice(s![1..=2, 2..=4])?;
        let expected = Tensor::try_from(vec![[8, 9, 10], [14, 15, 16]])?;
        assert_eq!(inclusive, expected);

        let inclusive_to = tensor.slice(s![..=1, ..=2])?;
        let expected = Tensor::try_from(vec![[0, 1, 2], [6, 7, 8]])?;
        assert_eq!(inclusive_to, expected);

        Ok(())
    }

    #[test]
    fn test_edge_cases() -> Result<()> {
        let tensor = create_test_tensor()?;

        // Empty slice
        let empty = tensor.slice(s![1..1, ..])?;
        assert_eq!(empty.shape(), &[0, 6]);

        // Single element
        let single = tensor.slice(s![1, 2])?;
        assert_eq!(single.shape(), &[]);
        assert_eq!(single.elements().count(), 1);
        assert_eq!(single.data[single.offset], 8);

        Ok(())
    }

    #[test]
    fn test_error_conditions() -> Result<()> {
        let tensor = create_test_tensor()?;

        // Index out of bounds
        assert!(tensor.slice(s![10, ..]).is_err());
        assert!(tensor.slice(s![-10, ..]).is_err());

        // Too many dimensions
        assert!(tensor.slice(s![.., .., ..]).is_err());

        Ok(())
    }

    #[test]
    fn test_slice_index_equality() {
        assert_eq!(SliceIndex::Single(1), SliceIndex::Single(1));
        assert_eq!(
            SliceIndex::range(Some(0), Some(10)),
            SliceIndex::range(Some(0), Some(10))
        );
        assert_ne!(SliceIndex::Single(1), SliceIndex::Single(2));
    }

    #[test]
    #[should_panic(expected = "Step must be positive")]
    fn test_invalid_step() {
        SliceIndex::range_with_step(Some(0), Some(10), 0);
    }

    #[test]
    #[should_panic(expected = "step! can only be used with range expressions")]
    fn test_invalid_step_macro() {
        let _ = step![5, 2]; // Should panic because 5 is not a range
    }

    #[test]
    fn test_s_macro_basic() -> Result<()> {
        let tensor = Tensor::arange(20)?.view(&[4, 5])?;

        // Full slice
        let slice1 = tensor.slice(s![.., ..])?;
        assert_eq!(slice1.shape(), &[4, 5]);

        // Mixed slicing
        let slice2 = tensor.slice(s![0..2, ..])?;
        assert_eq!(slice2.shape(), &[2, 5]);

        let slice3 = tensor.slice(s![.., 1..4])?;
        assert_eq!(slice3.shape(), &[4, 3]);

        Ok(())
    }

    #[test]
    fn test_s_macro_negative_indices() -> Result<()> {
        let tensor = Tensor::arange(20)?.view(&[4, 5])?;

        // Negative single index
        let slice1 = tensor.slice(s![-1, ..])?;
        assert_eq!(slice1.shape(), &[5]);

        // Negative range
        let slice2 = tensor.slice(s![-2..-1, ..])?;
        assert_eq!(slice2.shape(), &[1, 5]);

        // Negative end
        let slice3 = tensor.slice(s![.., ..-1])?;
        assert_eq!(slice3.shape(), &[4, 4]);

        Ok(())
    }

    #[test]
    fn test_s_macro_step_positive() -> Result<()> {
        let tensor = Tensor::arange(24)?.view(&[6, 4])?;

        // no end
        let slice = tensor.slice(&[step![0.., 2]])?;
        assert_eq!(
            Tensor::new(vec![0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19], &[3, 4])?,
            slice
        );

        // with end (non-inclusive)
        let slice = tensor.slice(&[step![0..4, 2]])?;
        assert_eq!(Tensor::new(vec![0, 1, 2, 3, 8, 9, 10, 11], &[2, 4])?, slice);

        // with end (inclusive)
        let slice = tensor.slice(&[step![0..=4, 2]])?;
        assert_eq!(
            Tensor::new(vec![0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19], &[3, 4])?,
            slice
        );

        Ok(())
    }

    #[test]
    fn test_s_macro_step_negative_ranges() -> Result<()> {
        let tensor = Tensor::arange(24)?.view(&[6, 4])?;

        // negative start
        let slice = tensor.slice(&[step![-4.., 2]])?;
        assert_eq!(
            Tensor::try_from(vec![[8, 9, 10, 11], [16, 17, 18, 19]])?,
            slice
        );

        // negative end
        let slice = tensor.slice(&[step![..-2, 2]])?;
        assert_eq!(Tensor::try_from(vec![[0, 1, 2, 3], [8, 9, 10, 11]])?, slice);

        // negative range (exclusive)
        let slice = tensor.slice(&[step![-6..-2, 3]])?;
        assert_eq!(
            Tensor::try_from(vec![[0, 1, 2, 3], [12, 13, 14, 15]])?,
            slice
        );

        // negative range (inclusive)
        let slice = tensor.slice(&[step![-6..=-2, 2]])?;
        assert_eq!(
            Tensor::try_from(vec![[0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19]])?,
            slice
        );

        Ok(())
    }

    #[test]
    fn test_more_than_one_step_range() -> Result<()> {
        let tensor = Tensor::arange(24)?.view(&[6, 4])?;

        let two_steps = tensor.slice(&[step![-6..=-2, 2], step!(.., 2)])?;
        let expected = Tensor::try_from(vec![[0, 2], [8, 10], [16, 18]])?;
        assert_eq!(two_steps, expected);

        let one_step_one_norm = tensor.slice(&[step![-6..=-2, 2], SliceIndex::Single(2)])?;
        let expected = Tensor::from(vec![2, 10, 18]);
        assert_eq!(one_step_one_norm, expected);

        let mixed_inside_s = tensor.slice(s![.., step!(.., 2)])?;
        let expected =
            Tensor::try_from(vec![[0, 2], [4, 6], [8, 10], [12, 14], [16, 18], [20, 22]])?;
        assert_eq!(mixed_inside_s, expected);

        Ok(())
    }
}
