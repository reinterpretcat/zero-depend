use super::*;
use std::ops::{AddAssign, Mul};

impl<T, const N: usize> Tensor<T, N>
where
    T: Clone + Default + AddAssign<T> + Mul<Output = T>,
{
    /// Computes the dot product of two 1D tensors.
    /// Returns an error if the tensors are not 1D or have different sizes.
    pub fn dot(&self, other: &Tensor<T, N>) -> Result<T> {
        if self.shape.len() != 1 || other.shape.len() != 1 {
            return Err(TensorError::UnsupportedOperation(format!(
                "Dot product requires 1D tensors, but got shapes {:?} and {:?}",
                self.shape, other.shape
            )));
        }

        if self.shape[0] != other.shape[0] {
            return Err(TensorError::ShapeMismatch(format!(
                "Tensors must have the same size for dot product: {} != {}",
                self.shape[0], other.shape[0]
            )));
        }

        let mut result = T::default();
        for i in 0..self.shape[0] {
            result += self.get(&[i])?.clone() * other.get(&[i])?.clone();
        }

        Ok(result)
    }

    /// Transpose the tensor by swapping two dimensions.
    /// Returns a new tensor with the specified dimensions swapped.
    /// Returns an error if the dimensions are out of bounds.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor<T, N>> {
        if dim0 >= self.shape.len() || dim1 >= self.shape.len() {
            return Err(TensorError::ShapeMismatch(format!(
                "Invalid dimensions for transpose: {} and {}",
                dim0, dim1
            )));
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() -> Result<()> {
        let a = Tensor::<i32>::arange(4)?;
        let b = Tensor::<i32>::from(vec![0, 3, 6, 9]);
        let result = a.dot(&b)?;
        assert_eq!(result, 42);

        Ok(())
    }

    #[test]
    fn test_transpose() -> Result<()> {
        let tensor = Tensor::<i32>::arange(25)?.view(&[5, 5])?;
        let transposed = tensor.transpose(0, 1)?;
        assert_eq!(transposed.shape(), &[5, 5]);
        assert_eq!(*transposed.get(&[0, 0])?, 0);
        assert_eq!(*transposed.get(&[0, 1])?, 5);
        assert_eq!(*transposed.get(&[1, 0])?, 1);
        Ok(())
    }
}
