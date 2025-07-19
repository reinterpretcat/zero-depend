use crate::slicing::SliceIndex;

use super::*;
use par_iter::*;

use std::marker::PhantomData;

/// An iterator over the elements of a tensor in a row-major order.
pub struct TensorIter<'a, T, const N: usize> {
    tensor: &'a Tensor<T, N>,
    current_index: usize,
}

impl<'a, T: Clone, const N: usize> IntoIterator for &'a Tensor<T, N> {
    type Item = Tensor<T, N>;
    type IntoIter = TensorIter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        TensorIter {
            tensor: self,
            current_index: 0,
        }
    }
}

impl<'a, T: Clone, const N: usize> Iterator for TensorIter<'a, T, N> {
    type Item = Tensor<T, N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.tensor.shape.is_empty() || self.current_index >= self.tensor.shape[0] {
            return None;
        }
        let sub_tensor = self.tensor.slice(&[self.current_index.into()]).ok();
        self.current_index += 1;
        sub_tensor
    }
}

impl<T, const N: usize> Tensor<T, N> {
    /// Returns an iterator over the elements of the tensor.
    /// This iterator traverses the tensor's elements in a row-major order.
    pub fn elements(&self) -> impl Iterator<Item = &T> {
        let is_done = self.shape.iter().product::<usize>() == 0 && !self.shape.is_empty();
        let start_index = if self.shape.is_empty() {
            vec![]
        } else {
            vec![0; self.shape.len()]
        };
        ElementsIter {
            tensor: self,
            current_index: start_index,
            is_done,
        }
    }
}

/// An iterator over all elements of a tensor.
struct ElementsIter<'a, T, const N: usize> {
    tensor: &'a Tensor<T, N>,
    current_index: Vec<usize>,
    is_done: bool,
}

impl<'a, T, const N: usize> Iterator for ElementsIter<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_done {
            return None;
        }

        let item = self.tensor.get(&self.current_index).ok()?;

        if self.tensor.shape.is_empty() {
            self.is_done = true;
            return Some(item);
        }

        let mut dim = self.tensor.shape.len() - 1;
        loop {
            self.current_index[dim] += 1;
            if self.current_index[dim] < self.tensor.shape[dim] {
                break;
            }
            self.current_index[dim] = 0;
            if dim == 0 {
                self.is_done = true;
                break;
            }
            dim -= 1;
        }
        Some(item)
    }
}

// Parallel iterator implementation for Tensor

// Extension trait for tensor parallel iteration
pub trait ParallelTensor<T, const N: usize> {
    fn par_rows(&self) -> ParIter<TensorRowProducer<T, N>>;
    fn par_batch_rows(&self) -> ParIter<BatchRowProducer<T, N>>;
}

impl<T: Send + Sync, const N: usize> ParallelTensor<T, N> for Tensor<T, N> {
    fn par_rows(&self) -> ParIter<TensorRowProducer<T, N>> {
        ParIter::new(TensorRowProducer::new(self))
    }
    fn par_batch_rows(&self) -> ParIter<BatchRowProducer<T, N>> {
        ParIter::new(BatchRowProducer::new(self))
    }
}

/// Producer for tensor rows that returns Tensor objects for each row
#[doc(hidden)]
pub struct TensorRowProducer<'a, T, const N: usize> {
    tensor: &'a Tensor<T, N>,
    num_rows: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T, const N: usize> TensorRowProducer<'a, T, N> {
    pub fn new(tensor: &'a Tensor<T, N>) -> Self {
        let num_rows = if tensor.shape.is_empty() {
            0
        } else {
            tensor.shape[0]
        };
        Self {
            tensor,
            num_rows,
            _phantom: PhantomData,
        }
    }
}

unsafe impl<'a, T: Send + Sync, const N: usize> Send for TensorRowProducer<'a, T, N> {}
unsafe impl<'a, T: Send + Sync, const N: usize> Sync for TensorRowProducer<'a, T, N> {}

impl<'a, T: Send + Sync, const N: usize> ParallelProducer for TensorRowProducer<'a, T, N> {
    type Item = (usize, Tensor<T, N>); // (row_index, row_tensor)

    fn len(&self) -> usize {
        self.num_rows
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        if index >= self.num_rows {
            return None;
        }

        // Create a tensor slice for this row: tensor[index, :]
        let row_tensor = self.tensor.slice(s![index]).ok()?;
        Some((index, row_tensor))
    }
}

#[doc(hidden)]
pub struct BatchRowProducer<'a, T, const N: usize> {
    tensor: &'a Tensor<T, N>,
    rows_per_batch: usize,
    total_pairs: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T, const N: usize> BatchRowProducer<'a, T, N> {
    pub fn new(tensor: &'a Tensor<T, N>) -> Self {
        let batch_size = if tensor.shape.len() >= 1 {
            tensor.shape[0]
        } else {
            0
        };
        let rows_per_batch = if tensor.shape.len() >= 2 {
            tensor.shape[1]
        } else {
            0
        };
        let total_pairs = batch_size * rows_per_batch;

        Self {
            tensor,
            rows_per_batch,
            total_pairs,
            _phantom: PhantomData,
        }
    }
}

unsafe impl<'a, T: Send + Sync, const N: usize> Send for BatchRowProducer<'a, T, N> {}
unsafe impl<'a, T: Send + Sync, const N: usize> Sync for BatchRowProducer<'a, T, N> {}

impl<'a, T: Send + Sync, const N: usize> ParallelProducer for BatchRowProducer<'a, T, N> {
    type Item = (usize, usize, Tensor<T, N>); // (batch_index, row_index, row_tensor)

    fn len(&self) -> usize {
        self.total_pairs
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        if index >= self.total_pairs {
            return None;
        }

        let batch_idx = index / self.rows_per_batch;
        let row_idx = index % self.rows_per_batch;

        // Create tensor slice: tensor[batch_idx, row_idx, :]
        let row_tensor = self
            .tensor
            .slice(&[
                SliceIndex::Single(batch_idx as isize),
                SliceIndex::Single(row_idx as isize),
            ])
            .ok()?;

        Some((batch_idx, row_idx, row_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_iterator() -> Result<()> {
        let tensor = Tensor::<i32>::arange(16)?.view(&[4, 2, 2])?;

        let mut iter = tensor.into_iter();
        assert_eq!(
            iter.next().unwrap(),
            Tensor::new(vec![0, 1, 2, 3], &[2, 2])?
        );
        assert_eq!(
            iter.next().unwrap(),
            Tensor::new(vec![4, 5, 6, 7], &[2, 2])?
        );
        assert_eq!(
            iter.next().unwrap(),
            Tensor::new(vec![8, 9, 10, 11], &[2, 2])?
        );
        assert_eq!(
            iter.next().unwrap(),
            Tensor::new(vec![12, 13, 14, 15], &[2, 2])?
        );
        assert!(iter.next().is_none());

        Ok(())
    }

    #[test]
    fn test_parallel_tensor_rows() -> Result<()> {
        let tensor = Tensor::<i32>::arange(12)?.view(&[3, 4])?;

        let rows: Vec<(usize, Tensor<i32>)> = tensor.par_rows().collect();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], (0, Tensor::from(vec![0, 1, 2, 3])));
        assert_eq!(rows[1], (1, Tensor::from(vec![4, 5, 6, 7])));
        assert_eq!(rows[2], (2, Tensor::from(vec![8, 9, 10, 11])));

        Ok(())
    }

    #[test]
    fn test_parallel_batch_rows() -> Result<()> {
        // 2 batches, 2 rows each, 3 cols
        let tensor = Tensor::<i32>::arange(12)?.view(&[2, 2, 3])?; 

        let batch_rows: Vec<(usize, usize, Tensor<i32>)> = tensor.par_batch_rows().collect();
        
        assert_eq!(batch_rows.len(), 4); // 2 batches × 2 rows
        assert_eq!(batch_rows[0], (0, 0, Tensor::<i32>::from(vec![0, 1, 2])));
        assert_eq!(batch_rows[1], (0, 1, Tensor::<i32>::from(vec![3, 4, 5])));
        assert_eq!(batch_rows[2], (1, 0, Tensor::<i32>::from(vec![6, 7, 8])));
        assert_eq!(batch_rows[3], (1, 1, Tensor::<i32>::from(vec![9, 10, 11])));

        Ok(())
    }
}
