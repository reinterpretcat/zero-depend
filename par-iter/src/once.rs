use super::*;

/// Creates a parallel iterator that yields a single item.
pub fn once<T: Send + Sync>(item: &T) -> ParIter<Once<'_, T>> {
    ParIter::new(Once::new(item))
}

#[doc(hidden)]
pub struct Once<'a, T> {
    item: &'a T,
}

impl<'a, T> Once<'a, T> {
    pub fn new(item: &'a T) -> Self {
        Self { item }
    }
}

impl<'a, T: Send + Sync> ParallelProducer for Once<'a, T> {
    type Item = &'a T;

    fn len(&self) -> usize {
        1
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        if index == 0 { Some(&self.item) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_once() {
        let item = 42;
        let result: Vec<_> = (1..10).into_par_iter().zip(once(&item)).collect();
        assert_eq!(result, vec![(1, &item)]);
    }
}
