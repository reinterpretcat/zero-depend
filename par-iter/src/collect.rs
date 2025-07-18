use super::*;

pub(super) struct Collect<P> {
    producer: P,
    config: ExecutionConfig,
}

impl<P> Collect<P>
where
    P: ParallelProducer,
{
    pub fn new(producer: P, config: ExecutionConfig) -> Self {
        Self { producer, config }
    }

    pub fn collect<T, B>(self) -> B
    where
        P::Item: Into<T>,
        B: FromIterator<T>,
        T: Send,
    {
        let total_items = self.producer.len();
        if total_items == 0 {
            return std::iter::empty().collect();
        }

        let num_threads = self.config.get_num_threads();
        let chunk_size = self.config.get_chunk_size(total_items);

        let counter = Arc::new(AtomicUsize::new(0));
        let producer_ref = &self.producer;
        let mut temp_results = Vec::new();

        thread::scope(|s| {
            let handles: Vec<_> = (0..num_threads)
                .map(|_| {
                    let counter = Arc::clone(&counter);
                    s.spawn(move || {
                        let mut local_results = Vec::new();
                        loop {
                            let start = counter.fetch_add(chunk_size, Ordering::Relaxed);
                            if start >= total_items {
                                break;
                            }

                            let end = (start + chunk_size).min(total_items);
                            for i in start..end {
                                if let Some(item) = producer_ref.get_item(i) {
                                    local_results.push((i, item.into()));
                                }
                            }
                        }
                        local_results
                    })
                })
                .collect();

            for handle in handles {
                if let Ok(results) = handle.join() {
                    temp_results.extend(results);
                }
            }
        });

        // Sort by index to maintain order
        temp_results.sort_unstable_by_key(|(i, _)| *i);
        temp_results.into_iter().map(|(_, item)| item).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    #[test]
    fn test_collect() {
        let data = (0..1000).collect::<Vec<_>>();
        let results: Vec<i32> = data.par_iter().cloned().collect();
        assert_eq!(results, data);

        let results: HashSet<i32> = data.into_par_iter().collect();
        assert_eq!(results.len(), 1000);
    }

    #[test]
    fn test_collect_empty() {
        let data: Vec<i32> = Vec::new();
        let results: Vec<i32> = data.par_iter().cloned().collect();
        assert!(results.is_empty());
    }

    #[test]
    fn test_collect_chained() {
        let data = (0..100).collect::<Vec<_>>();

        let results: Vec<(usize, i32)> = data
            .par_iter()
            .map(|&x| x * 2)
            .enumerate()
            .take(50)
            .collect();

        assert_eq!(
            results,
            (0..50_usize)
                .map(|x| (x, (x * 2) as i32))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_range_iterator() {
        let results: Vec<usize> = (0..100).into_par_iter().take(50).collect();

        assert_eq!(results, (0..50).collect::<Vec<_>>());
    }
}
