use super::*;

#[doc(hidden)]
pub struct Skip<P> {
    inner: P,
    skip_count: usize,
}

impl<P> Skip<P> {
    pub fn new(inner: P, skip_count: usize) -> Self {
        Self { inner, skip_count }
    }
}

impl<P: ParallelProducer> ParallelProducer for Skip<P> {
    type Item = P::Item;

    fn len(&self) -> usize {
        self.inner.len().saturating_sub(self.skip_count)
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        self.inner.get_item(index + self.skip_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skip() {
        let data = (0..1000).collect::<Vec<_>>();

        let results: Vec<i32> = data.into_par_iter().skip(10).collect();
        assert_eq!(results, (10..1000).collect::<Vec<_>>());
    }

    #[test]
    fn test_skip_take() {
        let data = (0..1000).collect::<Vec<_>>();

        let results: Vec<i32> = data.into_par_iter().skip(100).take(200).collect();
        assert_eq!(results, (100..300).collect::<Vec<_>>());
    }
}
