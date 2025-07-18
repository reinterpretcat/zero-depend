use super::*;

#[doc(hidden)]
pub(super) struct ForEach<P> {
    producer: P,
    config: ExecutionConfig,
}

impl<P> ForEach<P>
where
    P: ParallelProducer,
{
    pub fn new(producer: P, config: ExecutionConfig) -> Self {
        Self { producer, config }
    }

    /// Executes the parallel iteration.
    pub fn for_each<F>(self, f: F)
    where
        F: Fn(P::Item) + Send + Sync,
    {
        let total_items = self.producer.len();
        if total_items == 0 {
            return;
        }

        let num_threads = self.config.get_num_threads();
        let chunk_size = self.config.get_chunk_size(total_items);

        let counter = Arc::new(AtomicUsize::new(0));
        let f_ref = &f;
        let producer_ref = &self.producer;

        thread::scope(|s| {
            for _ in 0..num_threads {
                let counter = Arc::clone(&counter);
                s.spawn(move || {
                    let mut local_buffer = Vec::new();
                    loop {
                        let start = counter.fetch_add(chunk_size, Ordering::Relaxed);
                        if start >= total_items {
                            break;
                        }

                        let end = (start + chunk_size).min(total_items);

                        // Collect items into local buffer to minimize producer contention
                        local_buffer.clear();
                        for i in start..end {
                            if let Some(item) = producer_ref.get_item(i) {
                                local_buffer.push(item);
                            }
                        }

                        // Process items from local buffer
                        for item in local_buffer.drain(..) {
                            f_ref(item);
                        }
                    }
                });
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_element_for_each() {
        let mut data = vec![0; 20];
        data.par_iter_mut().for_each(|val| {
            *val = 42;
        });
        assert_eq!(data, vec![42; 20]);
    }

    #[test]
    fn test_for_each_with_index() {
        let mut data = vec![0; 20];
        data.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = i as i32 * 2;
        });
        for (i, &val) in data.iter().enumerate() {
            assert_eq!(val, (i as i32) * 2);
        }
    }

    #[test]
    fn test_simple_iterator_odd() {
        let mut data = vec![0; 313];

        data.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = i * 2;
        });

        for i in 0..data.len() {
            assert_eq!(data[i], i * 2);
        }
    }

    #[test]
    fn test_enumerate_take_for_each() {
        let mut data = vec![0; 100];

        data.par_iter_mut()
            .enumerate()
            .take(50)
            .for_each(|(i, val)| {
                *val = i * 2;
            });

        for i in 0..50 {
            assert_eq!(data[i], i * 2);
        }
        for i in 50..100 {
            assert_eq!(data[i], 0);
        }
    }
}
