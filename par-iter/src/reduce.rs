use super::*;

pub(super) struct Reduce<P> {
    producer: P,
    config: ExecutionConfig,
}

impl<P: ParallelProducer> Reduce<P> {
    pub fn new(producer: P, config: ExecutionConfig) -> Self {
        Self { producer, config }
    }

    /// Reduces the parallel iterator to a single value.
    pub fn reduce<F>(self, identity: P::Item, op: F) -> P::Item
    where
        P::Item: Send + Clone,
        F: Fn(P::Item, P::Item) -> P::Item + Send + Sync,
    {
        let total_items = self.producer.len();
        if total_items == 0 {
            return identity;
        }

        let num_threads = self.config.get_num_threads();
        let chunk_size = self.config.get_chunk_size(total_items);

        let counter = Arc::new(AtomicUsize::new(0));
        let producer_ref = &self.producer;
        let op_ref = &op;
        let mut partial_results = Vec::new();

        thread::scope(|s| {
            let handles: Vec<_> = (0..num_threads)
                .map(|_| {
                    let counter = Arc::clone(&counter);
                    let identity = identity.clone();
                    s.spawn(move || {
                        let mut local_acc = identity.clone();
                        let mut has_items = false;

                        loop {
                            let start = counter.fetch_add(chunk_size, Ordering::Relaxed);
                            if start >= total_items {
                                break;
                            }

                            let end = (start + chunk_size).min(total_items);
                            for i in start..end {
                                if let Some(item) = producer_ref.get_item(i) {
                                    local_acc = if has_items {
                                        op_ref(local_acc, item)
                                    } else {
                                        has_items = true;
                                        item
                                    };
                                }
                            }
                        }

                        if has_items { Some(local_acc) } else { None }
                    })
                })
                .collect();

            for handle in handles {
                if let Ok(Some(result)) = handle.join() {
                    partial_results.push(result);
                }
            }
        });

        partial_results.into_iter().fold(identity, op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce() {
        let data = (0..1000).collect::<Vec<_>>();
        let sum = data.par_iter().map(|&x| x).reduce(0, |acc, x| acc + x);
        assert_eq!(sum, 499500);
    }

    #[test]
    fn test_map_reduce() {
        let data = (0..1000).collect::<Vec<_>>();
        let sum: i32 = data.par_iter().map(|&x| x * 2).reduce(0, |acc, x| acc + x);
        assert_eq!(sum, 999000);
    }
}
