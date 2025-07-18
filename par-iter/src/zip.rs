use super::*;

#[doc(hidden)]
pub struct Zip<A, B> {
    a: A,
    b: B,
}

impl<A, B> Zip<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: ParallelProducer, B: ParallelProducer> ParallelProducer for Zip<A, B> {
    type Item = (A::Item, B::Item);

    fn len(&self) -> usize {
        self.a.len().min(self.b.len())
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        Some((self.a.get_item(index)?, self.b.get_item(index)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par_chunks_zip_one() {
        let mut data1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut data2 = vec![10, 20, 30, 40, 50, 60, 70, 80];

        data1
            .par_chunks_mut(2)
            .zip(data2.par_chunks_mut(2))
            .for_each(|(chunk1, chunk2)| {
                for (a, b) in chunk1.iter_mut().zip(chunk2.iter_mut()) {
                    *a += 100;
                    *b += 200;
                }
            });

        assert_eq!(data1, vec![101, 102, 103, 104, 105, 106, 107, 108]);
        assert_eq!(data2, vec![210, 220, 230, 240, 250, 260, 270, 280]);
    }

    #[test]
    fn test_zip_different_iterator_types() {
        let mut data1 = vec![1, 2, 3, 4];
        let data2 = vec![10, 20, 30, 40];
        let range = 0..4;

        data1
            .par_iter_mut()
            .zip(data2.par_iter())
            .zip(range.into_par_iter())
            .for_each(|((a, b), i)| {
                *a += b + i;
            });

        assert_eq!(data1, vec![11, 23, 35, 47]);
    }
}
