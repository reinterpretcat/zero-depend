use super::*;

pub(super) struct Find<P> {
    producer: P,
    config: ExecutionConfig,
}

impl<P> Find<P>
where
    P: ParallelProducer,
    P::Item: Send + Sync,
{
    pub fn new(producer: P, config: ExecutionConfig) -> Self {
        Self { producer, config }
    }

    pub fn find<F>(self, predicate: F) -> Option<P::Item>
    where
        F: Fn(&P::Item) -> bool + Send + Sync,
    {
        let total_items = self.producer.len();
        if total_items == 0 {
            return None;
        }

        let num_threads = self.config.get_num_threads();
        let chunk_size = self.config.get_chunk_size(total_items);

        let counter = Arc::new(AtomicUsize::new(0));
        let found = Arc::new(AtomicUsize::new(usize::MAX));
        let producer_ref = &self.producer;
        let predicate_ref = &predicate;

        thread::scope(|s| {
            let handles: Vec<_> = (0..num_threads)
                .map(|_| {
                    let counter = Arc::clone(&counter);
                    let found = Arc::clone(&found);
                    s.spawn(move || {
                        loop {
                            let start = counter.fetch_add(chunk_size, Ordering::Relaxed);
                            if start >= total_items || found.load(Ordering::Relaxed) != usize::MAX {
                                break;
                            }

                            let end = (start + chunk_size).min(total_items);
                            for i in start..end {
                                if found.load(Ordering::Relaxed) != usize::MAX {
                                    break;
                                }
                                if let Some(item) = producer_ref.get_item(i) {
                                    if predicate_ref(&item) {
                                        found.store(i, Ordering::Relaxed);
                                        return Some(item);
                                    }
                                }
                            }
                        }
                        None
                    })
                })
                .collect();

            for handle in handles {
                if let Ok(Some(result)) = handle.join() {
                    return Some(result);
                }
            }

            None
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find() {
        let data = (0..1000).collect::<Vec<_>>();
        let found = data.par_iter().find(|&&x| x == 333);
        assert_eq!(found, Some(&333));
    }
}
