use super::*;

pub struct Cloned<P> {
    inner: P,
}

impl<P> Cloned<P> {
    pub fn new(inner: P) -> Self {
        Self { inner }
    }
}

impl<'a, P, T> ParallelProducer for Cloned<P>
where
    T: 'a + Clone + Send + Sync,
    P: ParallelProducer<Item = &'a T>,
{
    type Item = T;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        self.inner.get_item(index).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

        #[test]
    fn test_combinators_with_cloned() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let result: Vec<i32> = data.par_iter().cloned().map(|x| x * 2).take(3).collect();

        assert_eq!(result, vec![2, 4, 6]);
    }
}
