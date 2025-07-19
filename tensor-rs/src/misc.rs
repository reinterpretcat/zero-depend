use super::*;
use std::fmt;

impl<T: PartialEq + Clone + fmt::Debug, const N: usize> PartialEq for Tensor<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.elements().eq(other.elements())
    }
}

impl<T: Eq + Clone + fmt::Debug, const N: usize> Eq for Tensor<T, N> {}

impl<T: fmt::Debug, const N: usize> fmt::Debug for Tensor<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("elements", &self.elements().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_compare_tensors() -> Result<()> {
        use super::*;

        let tensor1 = Tensor::<i32>::arange(5)?;
        let tensor2 = Tensor::<i32>::arange(5)?;
        let tensor3 = Tensor::<i32>::arange(6)?;

        assert_eq!(tensor1, tensor2);
        assert_ne!(tensor1, tensor3);

        Ok(())
    }

    #[test]
    fn compare_viewed_tensors() -> Result<()> {
        let tensor = Tensor::<i32>::arange(8)?;

        let view1 = tensor.view(&[2, 4])?;
        let view2 = tensor.view(&[2, 2, 2])?;

        assert_ne!(view1, view2);
        assert_eq!(
            Tensor::new((0..8).collect(), &[2, 4])?,
            tensor.view(&[2, 4])?
        );
        assert_eq!(
            Tensor::new((0..8).collect(), &[2, 2, 2])?,
            tensor.view(&[2, 2, 2])?
        );

        Ok(())
    }

    #[test]
    fn compare_sliced_tensors() -> Result<()> {
        let tensor = Tensor::<i32>::arange(8)?;

        // 2D
        let slice = tensor.view(&[2, 4])?.slice(s![0])?;
        assert_eq!(Tensor::new((0..4).collect(), &[4])?, slice);
        assert_ne!(Tensor::new((1..5).collect(), &[4])?, slice);
        assert_ne!(Tensor::new((0..4).collect(), &[2, 2])?, slice);

        // 3D
        let view = tensor.view(&[2, 2, 2])?;
        assert_eq!(Tensor::new(vec![4, 5, 6, 7], &[2, 2])?, view.slice(s![1])?);
        assert_eq!(Tensor::new(vec![4, 5], &[2])?, view.slice(s![1, 0])?);

        Ok(())
    }
}
