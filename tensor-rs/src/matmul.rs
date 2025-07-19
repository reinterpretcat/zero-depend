use super::*;
use std::ops::{AddAssign, Mul};

impl<T, const N: usize> Tensor<T, N>
where
    T: Default + Clone + AddAssign<T> + Mul<Output = T>,
{
    /// Performs matrix multiplication between two tensors.
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
        let a = Tensor::<i32>::arange(6)?.view(&[2, 3])?;
        let b = Tensor::<i32>::arange(6)?.view(&[3, 2])?;
        let result = a.matmul(&b)?;
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(*result.get(&[0, 0])?, 10);
        assert_eq!(*result.get(&[0, 1])?, 13);
        assert_eq!(*result.get(&[1, 0])?, 28);
        assert_eq!(*result.get(&[1, 1])?, 40);
        Ok(())
    }

    #[test]
    fn test_matmul_3d_2d() -> Result<()> {
        let a = Tensor::<i32>::arange(24)?.view(&[4, 3, 2])?;
        let b = Tensor::<i32>::arange(6)?.view(&[2, 3])?;
        let result = a.matmul(&b)?;
        assert_eq!(result.shape(), &[4, 3, 3]);
        assert_eq!(*result.get(&[0, 0, 0])?, 3);
        assert_eq!(*result.get(&[0, 0, 1])?, 4);
        assert_eq!(*result.get(&[0, 0, 2])?, 5);

        assert_eq!(*result.get(&[0, 1, 1])?, 14);
        assert_eq!(*result.get(&[1, 2, 1])?, 54);
        assert_eq!(*result.get(&[2, 2, 0])?, 51);
        assert_eq!(*result.get(&[3, 2, 1])?, 114);
        Ok(())
    }
}
