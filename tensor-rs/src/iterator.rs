use super::*;

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
}
