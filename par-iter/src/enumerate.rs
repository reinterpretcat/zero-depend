use super::*;

#[doc(hidden)]
pub struct Enumerate<P> {
    inner: P,
}

impl<P> Enumerate<P> {
    pub fn new(inner: P) -> Self {
        Self { inner }
    }
}

impl<P: ParallelProducer> ParallelProducer for Enumerate<P> {
    type Item = (usize, P::Item);

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        self.inner.get_item(index).map(|item| (index, item))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_level_enumerate() {
        let mut data = vec![0; 20];

        data.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = i;
        });

        // Check that each element has its index value
        for (i, &val) in data.iter().enumerate() {
            assert_eq!(val, i);
        }
    }
}
