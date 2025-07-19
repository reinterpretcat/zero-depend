use crate::iterator::ParallelTensor;
use par_iter::*;

use super::*;
use std::ops::{Add, AddAssign, Mul};

impl<T, const N: usize> Tensor<T, N>
where
    T: Default + Clone + AddAssign<T> + Mul<Output = T>,
{
    /// Performs matrix multiplication between two tensors on single thread.
    /// Supports 2D and 3D tensors, where the last dimension of the first tensor
    /// must match the second-to-last dimension of the second tensor.
    /// Returns a new tensor containing the result of the multiplication.
    pub fn matmul(&self, other: &Tensor<T, N>) -> Result<Tensor<T, N>> {
        match (self.shape.len(), other.shape.len()) {
            (2, 2) => self.matmul_2d(other),
            (3, 2) => self.matmul_3d_2d(other),
            _ => Err(TensorError::UnsupportedOperation(format!(
                "Unsupported dimensions for matmul: {:?} x {:?}",
                self.shape, other.shape
            ))),
        }
    }

    fn matmul_2d(&self, other: &Tensor<T, N>) -> Result<Tensor<T, N>> {
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k != k2 {
            return Err(TensorError::ShapeMismatch(format!(
                "Matrix dimensions incompatible for multiplication: {k} != {k2}"
            )));
        }

        let mut result_data = vec![T::default(); m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for l in 0..k {
                    let a_val = self.get(&[i, l])?.clone();
                    let b_val = other.get(&[l, j])?.clone();
                    sum += a_val * b_val;
                }
                result_data[i * n + j] = sum;
            }
        }

        Ok(Tensor {
            data: Arc::new(result_data),
            shape: small_vec![m, n],
            strides: small_vec![n, 1],
            offset: 0,
        })
    }

    fn matmul_3d_2d(&self, other: &Tensor<T, N>) -> Result<Tensor<T, N>> {
        let (batch, m, k) = (self.shape[0], self.shape[1], self.shape[2]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k != k2 {
            return Err(TensorError::ShapeMismatch(format!(
                "Matrix dimensions incompatible for multiplication: {k} != {k2}"
            )));
        }

        let mut result_data = vec![T::default(); batch * m * n];

        for b in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::default();
                    for l in 0..k {
                        let a_val = self.get(&[b, i, l])?.clone();
                        let b_val = other.get(&[l, j])?.clone();
                        sum += a_val * b_val;
                    }
                    result_data[b * m * n + i * n + j] = sum;
                }
            }
        }

        Ok(Tensor {
            data: Arc::new(result_data),
            shape: small_vec![batch, m, n],
            strides: small_vec![m * n, n, 1],
            offset: 0,
        })
    }
}

impl<T, const N: usize> Tensor<T, N>
where
    T: Clone + Send + Sync + Mul<Output = T> + Add<Output = T> + AddAssign + Default,
{
    /// Performs matrix multiplication between two tensors on multiple threads.
    /// Supports 2D and 3D tensors, where the last dimension of the first tensor
    /// must match the second-to-last dimension of the second tensor.
    /// Returns a new tensor containing the result of the multiplication.
    pub fn matmul_par(&self, other: &Tensor<T, N>) -> Result<Tensor<T, N>> {
        match (self.shape.len(), other.shape.len()) {
            (2, 2) => self.matmul_2d_par(other),
            (3, 2) => self.matmul_3d_2d_par(other),
            _ => Err(TensorError::UnsupportedOperation(format!(
                "Unsupported dimensions for matmul: {:?} x {:?}",
                self.shape, other.shape
            ))),
        }
    }

    fn matmul_2d_par(&self, other: &Tensor<T, N>) -> Result<Tensor<T, N>> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TensorError::ShapeMismatch(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }

        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k != k2 {
            return Err(TensorError::ShapeMismatch(format!(
                "Matrix dimensions don't match: {}x{} @ {}x{}",
                m, k, k2, n
            )));
        }

        let mut result_data = vec![T::default(); m * n];

        self.par_rows()
            .cartesian_product(other.transpose(0, 1)?.par_rows())
            .zip(result_data.par_iter_mut())
            .try_for_each(|(((_, row_tensor), (_, col_tensor)), v)| {
                *v = row_tensor.dot(&col_tensor)?;
                Ok(())
            })?;

        Ok(Tensor {
            data: Arc::new(result_data),
            shape: small_vec![m, n],
            strides: small_vec![n, 1],
            offset: 0,
        })
    }

    fn matmul_3d_2d_par(&self, other: &Tensor<T, N>) -> Result<Tensor<T, N>> {
        if self.shape.len() != 3 || other.shape.len() != 2 {
            return Err(TensorError::ShapeMismatch(
                "Matrix multiplication requires 3D and 2D tensors".to_string(),
            ));
        }

        let (batch, m, k) = (self.shape[0], self.shape[1], self.shape[2]);
        let (k2, n) = (other.shape[0], other.shape[1]);

        if k != k2 {
            return Err(TensorError::ShapeMismatch(format!(
                "Matrix dimensions don't match: {}x{} @ {}x{}",
                batch * m,
                k,
                k2,
                n
            )));
        }

        let mut result_data = vec![T::default(); batch * m * n];

        self.par_batch_rows()
            .cartesian_product(other.transpose(0, 1)?.par_rows())
            .zip(result_data.par_iter_mut())
            .try_for_each(|(((_, _, row_tensor), (_, col_tensor)), data)| {
                *data = row_tensor.dot(&col_tensor)?;
                Ok(())
            })?;

        Ok(Tensor {
            data: Arc::new(result_data),
            shape: small_vec![batch, m, n],
            strides: small_vec![m * n, n, 1],
            offset: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul() -> Result<()> {
        let a = Tensor::<i32>::arange(12)?.view(&[3, 4])?;
        let b = Tensor::<i32>::arange(12)?.view(&[4, 3])?;
        let result = a.matmul(&b)?;
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(*result.get(&[0, 0])?, 42);
        assert_eq!(*result.get(&[0, 1])?, 48);
        assert_eq!(*result.get(&[0, 2])?, 54);

        Ok(())
    }

    #[test]
    fn test_matmul_2d() -> Result<()> {
        let a = Tensor::<i32>::new((1..=12).collect(), &[3, 4])?;
        let b = Tensor::<i32>::new((1..=12).collect(), &[4, 3])?;
        let expected =
            Tensor::<i32>::try_from(vec![[70, 80, 90], [158, 184, 210], [246, 288, 330]])?;

        assert_eq!(a.matmul_2d(&b)?, expected, "sync matmul failed");
        assert_eq!(a.matmul_2d_par(&b)?, expected, "parallel matmul failed");

        Ok(())
    }

    #[test]
    fn test_matmul_3d_2d() -> Result<()> {
        let a = Tensor::<i32>::arange(24)?.view(&[4, 3, 2])?;
        let b = Tensor::<i32>::arange(6)?.view(&[2, 3])?;
        let expected = Tensor::new(
            vec![
                3, 4, 5, //
                9, 14, 19, //
                15, 24, 33, //
                //
                21, 34, 47, //
                27, 44, 61, //
                33, 54, 75, //
                //
                39, 64, 89, //
                45, 74, 103, //
                51, 84, 117, //
                //
                57, 94, 131, //
                63, 104, 145, //
                69, 114, 159, //
            ],
            &[4, 3, 3],
        )?;

        assert_eq!(a.matmul_3d_2d(&b)?, expected, "sync matmul failed");
        assert_eq!(a.matmul_3d_2d_par(&b)?, expected, "parallel matmul failed");

        Ok(())
    }

    #[test]
    fn test_matmul_3d_2d_large() -> Result<()> {
        let a = Tensor::<i32>::arange(15360)?.view(&[10, 64, 24])?;
        let b = Tensor::<i32>::arange(384)?.view(&[24, 16])?;

        let sync_result = a.matmul_3d_2d(&b)?;
        let par_result = a.matmul_3d_2d_par(&b)?;

        assert_eq!(sync_result.shape(), &[10, 64, 16]);
        assert_eq!(par_result.shape(), &[10, 64, 16]);

        assert_eq!(
            sync_result, par_result,
            "Parallel and sync matmul results differ"
        );

        println!("{par_result}");

        Ok(())
    }
}
