use super::*;

#[doc(hidden)]
pub struct Take<P> {
    inner: P,
    take_count: usize,
}

impl<P: ParallelProducer> Take<P> {
    pub fn new(inner: P, take_count: usize) -> Self {
        let take_count = take_count.min(inner.len());
        Take { inner, take_count }
    }
}

impl<P: ParallelProducer> ParallelProducer for Take<P> {
    type Item = P::Item;

    fn len(&self) -> usize {
        self.take_count.min(self.inner.len())
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        if index < self.take_count {
            self.inner.get_item(index)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_take() {
        let data = (0..1000).collect::<Vec<_>>();

        let results: Vec<i32> = data.into_par_iter().take(10).collect();
        assert_eq!(results, (0..10).collect::<Vec<_>>());
    }
}
