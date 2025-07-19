use super::*;

impl<T, const N: usize> Tensor<T, N> {
    // View operation - no data copying
    pub fn view(&self, new_shape: &[usize]) -> Result<Tensor<T, N>> {
        let old_size: usize = self.shape.iter().product();
        let new_size: usize = new_shape.iter().product();

        if old_size != new_size {
            return Err(TensorError::ShapeMismatch(format!(
                "Cannot reshape tensor of size {old_size} to size {new_size}"
            )));
        }

        let new_strides = Self::compute_strides(&new_shape);
        let new_shape = new_shape.iter().copied().collect::<SmallVec<_, N>>();

        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    // Get element at multi-dimensional index
    fn get_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.shape.len() {
            return Err(TensorError::ShapeMismatch(format!(
                "Index dimension mismatch: {} != {}",
                indices.len(),
                self.shape.len()
            )));
        }

        let mut linear_index = self.offset;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(TensorError::IndexOutOfBounds(format!(
                    "Index {idx} out of bounds for dimension {i} with size {}",
                    self.shape[i]
                )));
            }
            linear_index += idx * self.strides[i];
        }
        Ok(linear_index)
    }

    // Get element at multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> Result<&T> {
        let linear_index = self.get_index(indices)?;
        Ok(&self.data[linear_index])
    }

    // Get shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_view() -> Result<()> {
        let tensor = Tensor::<i32>::arange(25)?;
        let view = tensor.view(&[5, 5])?;
        assert_eq!(view.shape(), &[5, 5]);
        assert_eq!(*view.get(&[0, 0])?, 0);
        assert_eq!(*view.get(&[1, 0])?, 5);
        assert_eq!(*view.get(&[4, 4])?, 24);

        Ok(())
    }
}
