use super::*;

/// A parallel producer that generates the cartesian product of two producers.
/// Each item is a tuple (A, B) where A comes from the first producer and B from the second.
/// The total number of items is len_a * len_b.
#[doc(hidden)]
pub struct CartesianProduct<P, Q> {
    left: P,
    right: Q,
}

impl<P, Q> CartesianProduct<P, Q> {
    pub fn new(left: P, right: Q) -> Self {
        Self { left, right }
    }
}

impl<P, Q> ParallelProducer for CartesianProduct<P, Q>
where
    P: ParallelProducer,
    Q: ParallelProducer,
    P::Item: Clone + Send + Sync,
    Q::Item: Clone + Send + Sync,
{
    type Item = (P::Item, Q::Item);

    fn len(&self) -> usize {
        self.left.len().saturating_mul(self.right.len())
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        let left_len = self.left.len();
        let right_len = self.right.len();

        if left_len == 0 || right_len == 0 || index >= self.len() {
            return None;
        }

        let left_idx = index / right_len;
        let right_idx = index % right_len;

        let left_item = self.left.get_item(left_idx)?;
        let right_item = self.right.get_item(right_idx)?;

        Some((left_item, right_item))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartesian_product_basic() {
        let vec1 = vec![1, 2];
        let vec2 = vec!['a', 'b', 'c'];

        let mut result: Vec<_> = vec1.par_iter().cartesian_product(vec2.par_iter()).collect();

        // Sort for deterministic comparison since parallel execution might reorder
        result.sort();

        let expected = vec![
            (&1, &'a'),
            (&1, &'b'),
            (&1, &'c'),
            (&2, &'a'),
            (&2, &'b'),
            (&2, &'c'),
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_cartesian_product_with_ranges() {
        let result: Vec<_> = (0..3)
            .into_par_iter()
            .cartesian_product((10..12).into_par_iter())
            .collect();

        assert_eq!(result.len(), 6); // 3 * 2 = 6
        assert!(result.contains(&(0, 10)));
        assert!(result.contains(&(0, 11)));
        assert!(result.contains(&(2, 11)));
    }

    #[test]
    fn test_cartesian_product_empty() {
        let vec1: Vec<i32> = vec![];
        let vec2 = vec![1, 2, 3];

        let result: Vec<_> = vec1.par_iter().cartesian_product(vec2.par_iter()).collect();

        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_cartesian_product_single_elements() {
        let vec1 = vec![42];
        let vec2 = vec!["hello"];

        let result: Vec<_> = vec1.par_iter().cartesian_product(vec2.par_iter()).collect();

        assert_eq!(result, vec![(&42, &"hello")]);
    }

    #[test]
    fn test_cartesian_product_with_map() {
        let result: Vec<_> = (1..3)
            .into_par_iter()
            .cartesian_product((10..12).into_par_iter())
            .map(|(a, b)| a * b)
            .collect();

        assert_eq!(result.len(), 4);
        assert!(result.contains(&10)); // 1 * 10
        assert!(result.contains(&11)); // 1 * 11  
        assert!(result.contains(&20)); // 2 * 10
        assert!(result.contains(&22)); // 2 * 11
    }
}
