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
        // Use try_for_each with a closure that always returns Ok
        let _ = self.try_for_each::<_, ()>(|item| {
            f(item);
            Ok(())
        });
    }

    /// Executes the parallel iteration, returning early on error.
    pub fn try_for_each<F, E>(self, f: F) -> Result<(), E>
    where
        F: Fn(P::Item) -> Result<(), E> + Send + Sync,
        E: Send + Sync + 'static,
    {
        let total_items = self.producer.len();
        if total_items == 0 {
            return Ok(());
        }

        let num_threads = self.config.get_num_threads();
        let chunk_size = self.config.get_chunk_size(total_items);

        let counter = Arc::new(AtomicUsize::new(0));
        let producer_ref = &self.producer;
        let f_ref = &f;
        let error = Arc::new(std::sync::Mutex::new(None));

        thread::scope(|s| {
            for _ in 0..num_threads {
                let counter = Arc::clone(&counter);
                let error = Arc::clone(&error);
                s.spawn(move || {
                    let mut local_buffer = Vec::new();
                    loop {
                        if error.lock().unwrap().is_some() {
                            break;
                        }
                        let start = counter.fetch_add(chunk_size, Ordering::Relaxed);
                        if start >= total_items {
                            break;
                        }
                        let end = (start + chunk_size).min(total_items);
                        local_buffer.clear();
                        for i in start..end {
                            if let Some(item) = producer_ref.get_item(i) {
                                local_buffer.push(item);
                            }
                        }
                        for item in local_buffer.drain(..) {
                            if let Err(e) = f_ref(item) {
                                *error.lock().unwrap() = Some(e);
                                return;
                            }
                        }
                    }
                });
            }
        });
        let error = match Arc::try_unwrap(error) {
            Ok(mutex) => match mutex.into_inner() {
                Ok(opt) => opt,
                Err(_) => None,
            },
            Err(_) => None,
        };
        match error {
            Some(e) => Err(e),
            None => Ok(()),
        }
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
