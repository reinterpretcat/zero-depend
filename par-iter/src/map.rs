use super::*;

#[doc(hidden)]
pub struct Map<P, F> {
    inner: P,
    map_fn: F,
}

impl<P, F> Map<P, F> {
    pub fn new(inner: P, map_fn: F) -> Self {
        Self { inner, map_fn }
    }
}

impl<P, F, U> ParallelProducer for Map<P, F>
where
    P: ParallelProducer,
    F: Fn(P::Item) -> U + Send + Sync,
{
    type Item = U;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        self.inner.get_item(index).map(&self.map_fn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_level_map() {
        let data = Vec::from_iter(1..=100);
        let expected = data.iter().map(|&x| x * 2).collect::<Vec<_>>();

        let results: Vec<i32> = data.par_iter().map(|&x| x * 2).collect();

        assert_eq!(results, expected);
    }
}
