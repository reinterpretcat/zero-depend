use std::marker::PhantomData;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

// --- Core Iterator Type ---

pub struct ParIter<P> {
    producer: P,
    config: ExecutionConfig,
}

impl<P> ParIter<P> {
    pub fn new(producer: P) -> Self {
        Self {
            producer,
            config: ExecutionConfig::default(),
        }
    }

    /// Sets the number of threads to use for the parallel computation.
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.config.num_threads = Some(num_threads.max(1));
        self
    }

    /// Sets the chunk size for distributing work to threads.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.config.chunk_size = Some(chunk_size.max(1));
        self
    }
}

impl<P: ParallelProducer> ParIter<P> {
    pub fn take(self, n: usize) -> ParIter<Take<P>> {
        ParIter {
            producer: Take {
                inner: self.producer,
                take_count: n,
            },
            config: self.config,
        }
    }

    pub fn enumerate(self) -> ParIter<Enumerate<P>> {
        ParIter {
            producer: Enumerate {
                inner: self.producer,
            },
            config: self.config,
        }
    }

    pub fn map<F, U>(self, f: F) -> ParIter<Map<P, F>>
    where
        F: Fn(P::Item) -> U + Send + Sync,
    {
        ParIter {
            producer: Map {
                inner: self.producer,
                map_fn: f,
            },
            config: self.config,
        }
    }

    pub fn zip<Q: ParallelProducer>(self, other: ParIter<Q>) -> ParIter<Zip<P, Q>> {
        ParIter::new(Zip::new(self.producer, other.producer))
    }

    pub fn skip(self, n: usize) -> ParIter<Skip<P>> {
        ParIter {
            producer: Skip {
                inner: self.producer,
                skip_count: n,
            },
            config: self.config,
        }
    }

    pub fn step_by(self, step: usize) -> ParIter<StepBy<P>> {
        ParIter {
            producer: StepBy {
                inner: self.producer,
                step,
            },
            config: self.config,
        }
    }

    pub fn cloned<'a, T>(self) -> ParIter<Cloned<P>>
    where
        T: 'a + Clone + Send + Sync,
        P: ParallelProducer<Item = &'a T>,
    {
        ParIter {
            producer: Cloned {
                inner: self.producer,
            },
            config: self.config,
        }
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

    /// Collects all items into a Vec, preserving the original order.
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

                        if has_items {
                            Some(local_acc)
                        } else {
                            None
                        }
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

    /// Finds the first item that matches the predicate.
    pub fn find<F>(self, predicate: F) -> Option<P::Item>
    where
        F: Fn(&P::Item) -> bool + Send + Sync,
        P::Item: Send + Sync,
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

// --- Configuration ---

#[derive(Debug, Default, Clone)]
struct ExecutionConfig {
    num_threads: Option<usize>,
    chunk_size: Option<usize>,
}

impl ExecutionConfig {
    fn get_num_threads(&self) -> usize {
        self.num_threads
            .unwrap_or_else(|| thread::available_parallelism().map_or(4, |n| n.get()))
    }

    fn get_chunk_size(&self, total_items: usize) -> usize {
        let num_threads = self.get_num_threads();
        self.chunk_size
            .unwrap_or_else(|| ((total_items + num_threads - 1) / num_threads).max(1))
    }
}

// --- Core Trait ---

pub trait ParallelProducer: Send + Sync {
    type Item;

    /// Returns the total number of items to be processed.
    fn len(&self) -> usize;

    /// Gets a single item by its index.
    fn get_item(&self, index: usize) -> Option<Self::Item>;
}

// --- Conversion Traits ---

#[doc(hidden)]
pub trait IntoParallelIterator {
    type Item;
    type Producer: ParallelProducer<Item = Self::Item>;

    fn into_par_iter(self) -> ParIter<Self::Producer>;
}

#[doc(hidden)]
pub trait ParallelSlice<T: Send + Sync> {
    fn par_iter(&self) -> ParIter<SliceIter<T>>;
    fn par_iter_mut(&mut self) -> ParIter<SliceIterMut<T>>;
    fn par_chunks(&self, chunk_size: usize) -> ParIter<Chunks<T>>;
    fn par_chunks_mut(&mut self, chunk_size: usize) -> ParIter<ChunksMut<T>>;
}

impl<T: Send + Sync> ParallelSlice<T> for [T] {
    fn par_iter(&self) -> ParIter<SliceIter<T>> {
        ParIter::new(SliceIter::new(self))
    }

    fn par_iter_mut(&mut self) -> ParIter<SliceIterMut<T>> {
        ParIter::new(SliceIterMut::new(self))
    }

    fn par_chunks(&self, chunk_size: usize) -> ParIter<Chunks<T>> {
        ParIter::new(Chunks::new(self, chunk_size))
    }

    fn par_chunks_mut(&mut self, chunk_size: usize) -> ParIter<ChunksMut<T>> {
        ParIter::new(ChunksMut::new(self, chunk_size))
    }
}

impl<T: Send + Sync> ParallelSlice<T> for Vec<T> {
    fn par_iter(&self) -> ParIter<SliceIter<T>> {
        self.as_slice().par_iter()
    }

    fn par_iter_mut(&mut self) -> ParIter<SliceIterMut<T>> {
        self.as_mut_slice().par_iter_mut()
    }

    fn par_chunks(&self, chunk_size: usize) -> ParIter<Chunks<T>> {
        self.as_slice().par_chunks(chunk_size)
    }

    fn par_chunks_mut(&mut self, chunk_size: usize) -> ParIter<ChunksMut<T>> {
        self.as_mut_slice().par_chunks_mut(chunk_size)
    }
}

impl IntoParallelIterator for Range<usize> {
    type Item = usize;
    type Producer = RangeProducer;

    fn into_par_iter(self) -> ParIter<Self::Producer> {
        ParIter::new(RangeProducer::new(self))
    }
}

impl<T: Send + Sync> IntoParallelIterator for Vec<T> {
    type Item = T;
    type Producer = VecProducer<T>;

    fn into_par_iter(self) -> ParIter<Self::Producer> {
        ParIter::new(VecProducer::new(self))
    }
}

// --- Producer Implementations ---

// Producer for owned Vec<T>
#[doc(hidden)]
pub struct VecProducer<T> {
    data: Vec<T>,
}

impl<T> VecProducer<T> {
    fn new(data: Vec<T>) -> Self {
        Self { data }
    }
}

impl<T: Send + Sync> ParallelProducer for VecProducer<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        // This is safe because we're moving out of the Vec and each index is accessed at most once
        if index < self.data.len() {
            unsafe {
                let ptr = self.data.as_ptr().add(index);
                Some(std::ptr::read(ptr))
            }
        } else {
            None
        }
    }
}

// Producer for immutable slices
#[doc(hidden)]
pub struct SliceIter<'a, T: 'a> {
    ptr: *const T,
    len: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T: Send + Sync> SliceIter<'a, T> {
    fn new(slice: &'a [T]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
            _phantom: PhantomData,
        }
    }
}

unsafe impl<'a, T: Send + Sync> Send for SliceIter<'a, T> {}
unsafe impl<'a, T: Send + Sync> Sync for SliceIter<'a, T> {}

impl<'a, T: Send + Sync> ParallelProducer for SliceIter<'a, T> {
    type Item = &'a T;

    fn len(&self) -> usize {
        self.len
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        if index < self.len {
            Some(unsafe { &*self.ptr.add(index) })
        } else {
            None
        }
    }
}

// Producer for mutable slices
#[doc(hidden)]
pub struct SliceIterMut<'a, T: 'a> {
    ptr: *mut T,
    len: usize,
    _phantom: PhantomData<&'a mut T>,
}

impl<'a, T: Send + Sync> SliceIterMut<'a, T> {
    fn new(slice: &'a mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _phantom: PhantomData,
        }
    }
}

unsafe impl<'a, T: Send + Sync> Send for SliceIterMut<'a, T> {}
unsafe impl<'a, T: Send + Sync> Sync for SliceIterMut<'a, T> {}

impl<'a, T: Send + Sync> ParallelProducer for SliceIterMut<'a, T> {
    type Item = &'a mut T;

    fn len(&self) -> usize {
        self.len
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        if index < self.len {
            Some(unsafe { &mut *self.ptr.add(index) })
        } else {
            None
        }
    }
}

// Producer for ranges
#[doc(hidden)]
pub struct RangeProducer {
    range: Range<usize>,
}

impl RangeProducer {
    fn new(range: Range<usize>) -> Self {
        Self { range }
    }
}

impl ParallelProducer for RangeProducer {
    type Item = usize;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        let val = self.range.start + index;
        if val < self.range.end {
            Some(val)
        } else {
            None
        }
    }
}

// Producer for immutable chunks
#[doc(hidden)]
pub struct Chunks<'a, T: 'a> {
    ptr: *const T,
    len: usize,
    chunk_size: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T: Send + Sync> Chunks<'a, T> {
    fn new(slice: &'a [T], chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk size must be positive");
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
            chunk_size,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Send + Sync> ParallelProducer for Chunks<'a, T> {
    type Item = &'a [T];

    fn len(&self) -> usize {
        (self.len + self.chunk_size - 1) / self.chunk_size
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        let start = index * self.chunk_size;
        if start >= self.len {
            return None;
        }
        let end = (start + self.chunk_size).min(self.len);
        let actual_size = end - start;
        Some(unsafe { std::slice::from_raw_parts(self.ptr.add(start), actual_size) })
    }
}

unsafe impl<'a, T: Send + Sync> Send for Chunks<'a, T> {}
unsafe impl<'a, T: Send + Sync> Sync for Chunks<'a, T> {}

// Producer for mutable chunks
#[doc(hidden)]
pub struct ChunksMut<'a, T: 'a> {
    ptr: *mut T,
    len: usize,
    chunk_size: usize,
    _phantom: PhantomData<&'a mut T>,
}

impl<'a, T: Send + Sync> ChunksMut<'a, T> {
    fn new(slice: &'a mut [T], chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk size must be positive");
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            chunk_size,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Send + Sync> ParallelProducer for ChunksMut<'a, T> {
    type Item = &'a mut [T];

    fn len(&self) -> usize {
        (self.len + self.chunk_size - 1) / self.chunk_size
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        let start = index * self.chunk_size;
        if start >= self.len {
            return None;
        }
        let end = (start + self.chunk_size).min(self.len);
        let actual_size = end - start;
        Some(unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), actual_size) })
    }
}

unsafe impl<'a, T: Send + Sync> Send for ChunksMut<'a, T> {}
unsafe impl<'a, T: Send + Sync> Sync for ChunksMut<'a, T> {}

// --- Combinator Implementations ---

#[doc(hidden)]
pub struct Take<P> {
    inner: P,
    take_count: usize,
}

impl<P: ParallelProducer> ParallelProducer for Take<P> {
    type Item = P::Item;

    fn len(&self) -> usize {
        self.take_count.min(self.inner.len())
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        if index < self.take_count {
            self.inner.get_item(index)
        } else {
            None
        }
    }
}

#[doc(hidden)]
pub struct Skip<P> {
    inner: P,
    skip_count: usize,
}

impl<P: ParallelProducer> ParallelProducer for Skip<P> {
    type Item = P::Item;

    fn len(&self) -> usize {
        self.inner.len().saturating_sub(self.skip_count)
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        self.inner.get_item(index + self.skip_count)
    }
}

#[doc(hidden)]
pub struct StepBy<P> {
    inner: P,
    step: usize,
}

impl<P: ParallelProducer> ParallelProducer for StepBy<P> {
    type Item = P::Item;

    fn len(&self) -> usize {
        if self.step == 0 {
            0
        } else {
            (self.inner.len() + self.step - 1) / self.step
        }
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        self.inner.get_item(index * self.step)
    }
}

#[doc(hidden)]
pub struct Enumerate<P> {
    inner: P,
}

impl<P: ParallelProducer> ParallelProducer for Enumerate<P> {
    type Item = (usize, P::Item);

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get_item(&self, index: usize) -> Option<Self::Item> {
        self.inner.get_item(index).map(|item| (index, item))
    }
}

#[doc(hidden)]
pub struct Map<P, F> {
    inner: P,
    map_fn: F,
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

#[doc(hidden)]
pub struct Zip<A, B> {
    a: A,
    b: B,
}

impl<A, B> Zip<A, B> {
    fn new(a: A, b: B) -> Self {
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

pub struct Cloned<P> {
    inner: P,
}

impl<'a, P, T> ParallelProducer for Cloned<P>
where
    T: 'a + Clone + Send + Sync,
    P: ParallelProducer<Item = &'a T>,
{
    // This is the key change: the item type is the owned type T, not the reference.
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
    use std::collections::{HashMap, HashSet};
    use std::sync::Mutex;

    #[test]
    fn test_simple_iterator() {
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
    fn test_element_level_enumerate() {
        let mut data = vec![0; 20];

        data.par_iter_mut().enumerate().for_each(|(i, val)| {
            *val = i;
        });

        // Check that each element has its index value
        for (i, &val) in data.iter().enumerate() {
            assert_eq!(val, i);
        }
    }

    #[test]
    fn test_element_level_map() {
        let data = Vec::from_iter(1..=100);
        let expected = data.iter().map(|&x| x * 2).collect::<Vec<_>>();

        let results: Vec<i32> = data.par_iter().map(|&x| x * 2).collect();

        assert_eq!(results, expected);
    }

    #[test]
    fn test_range_iterator() {
        let results: Vec<usize> = (0..100).into_par_iter().take(50).collect();

        assert_eq!(results, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn test_chained_operations() {
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

    #[test]
    fn test_par_chunks_zip_two() {
        let mut data1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut data2 = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let n_heads = 4;

        data1
            .par_chunks_mut(2)
            .zip(data2.par_chunks_mut(2))
            .zip((0..n_heads).into_par_iter())
            .for_each(|((chunk1, chunk2), head_idx)| {
                for (a, b) in chunk1.iter_mut().zip(chunk2.iter_mut()) {
                    *a += head_idx;
                    *b += head_idx * 10;
                }
            });

        // Expected results based on the logic:
        // Chunk 0 (head_idx 0): data1[0,1] += 0, data2[0,1] += 0 -> data1=[1,2], data2=[10,20]
        // Chunk 1 (head_idx 1): data1[2,3] += 1, data2[2,3] += 10 -> data1=[4,5], data2=[40,50]
        // Chunk 2 (head_idx 2): data1[4,5] += 2, data2[4,5] += 20 -> data1=[7,8], data2=[70,80]
        // Chunk 3 (head_idx 3): data1[6,7] += 3, data2[6,7] += 30 -> data1=[10,11], data2=[100,110]
        assert_eq!(data1, vec![1, 2, 4, 5, 7, 8, 10, 11]);
        assert_eq!(data2, vec![10, 20, 40, 50, 70, 80, 100, 110]);
    }

    #[test]
    fn test_attention_pattern_with_chained_zip() {
        let seq_len = 4;
        let head_dim = 8;
        let n_heads = 4;

        let mut att = vec![0.0f32; n_heads * seq_len];
        let mut xb = vec![0.0f32; n_heads * head_dim];
        let mut another_vec = vec![0.0f32; n_heads * seq_len];

        att.par_chunks_mut(seq_len)
            .zip(xb.par_chunks_mut(head_dim))
            .zip(another_vec.par_chunks_mut(seq_len))
            .zip((0..n_heads).into_par_iter())
            .for_each(move |(((att_slice, xb_slice), another_slice), head_idx)| {
                assert_eq!(att_slice.len(), seq_len);
                assert_eq!(xb_slice.len(), head_dim);
                assert_eq!(another_slice.len(), seq_len);

                for val in att_slice.iter_mut() {
                    *val = head_idx as f32;
                }
                for val in xb_slice.iter_mut() {
                    *val = (head_idx * 10) as f32;
                }
                for val in another_slice.iter_mut() {
                    *val = (head_idx * 100) as f32;
                }
            });

        for head in 0..n_heads {
            for i in 0..seq_len {
                assert_eq!(att[head * seq_len + i], head as f32);
                assert_eq!(another_vec[head * seq_len + i], (head * 100) as f32);
            }
            for i in 0..head_dim {
                assert_eq!(xb[head * head_dim + i], (head * 10) as f32);
            }
        }
    }

    #[test]
    fn test_find() {
        let data = (0..1000).collect::<Vec<_>>();
        let found = data.par_iter().find(|&&x| x == 333);
        assert_eq!(found, Some(&333));
    }

    #[test]
    fn test_step_by() {
        let data = (0..1000).collect::<Vec<_>>();
        let expected = (0..1000).step_by(10).collect::<Vec<_>>();

        let results: Vec<i32> = data.par_iter().step_by(10).cloned().collect();
        assert_eq!(results, expected);

        let results: Vec<i32> = data.into_par_iter().step_by(10).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_collect() {
        let data = (0..1000).collect::<Vec<_>>();
        let results: Vec<i32> = data.par_iter().cloned().collect();
        assert_eq!(results, data);

        let results: HashSet<i32> = data.into_par_iter().collect();
        assert_eq!(results.len(), 1000);
    }

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

    #[test]
    fn test_skip() {
        let data = (0..1000).collect::<Vec<_>>();

        let results: Vec<i32> = data.into_par_iter().skip(10).collect();
        assert_eq!(results, (10..1000).collect::<Vec<_>>());
    }

    #[test]
    fn test_take() {
        let data = (0..1000).collect::<Vec<_>>();

        let results: Vec<i32> = data.into_par_iter().take(10).collect();
        assert_eq!(results, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_skip_take() {
        let data = (0..1000).collect::<Vec<_>>();

        let results: Vec<i32> = data.into_par_iter().skip(100).take(200).collect();
        assert_eq!(results, (100..300).collect::<Vec<_>>());
    }

    #[test]
    fn test_combinators() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let result: Vec<i32> = data.par_iter().cloned().map(|x| x * 2).take(3).collect();

        assert_eq!(result, vec![2, 4, 6]);
    }

    #[test]
    fn test_work_distribution_stable() {
        // We only run this test if we have more than one core available.
        let num_cores = thread::available_parallelism().unwrap().get();
        if num_cores <= 1 {
            println!("Skipping work distribution test on single-core machine.");
            return;
        }

        // A map to store work count per thread. Key: ThreadId, Value: count.
        let work_counts = Arc::new(Mutex::new(HashMap::new()));

        let mut data = vec![0; 100_000]; // A large enough workload to ensure distribution.

        data.par_iter_mut().for_each(|_| {
            let id = thread::current().id();
            let mut counts = work_counts.lock().unwrap();
            *counts.entry(id).or_insert(0) += 1;
        });

        // After execution, lock the map and check the results.
        let final_counts = work_counts.lock().unwrap();

        // The number of entries in the map is the number of unique threads that did work.
        let threads_that_did_work = final_counts.len();

        println!(
            "Work was distributed among {} threads.",
            threads_that_did_work
        );
        for (id, count) in final_counts.iter() {
            println!("- Thread {:?} processed {} items.", id, count);
        }

        // For a large workload on a multi-core machine, we expect more than one thread to participate.
        assert!(
            threads_that_did_work > 1,
            "Work was not distributed to more than one thread."
        );

        // Also, check that the total work done matches the number of items.
        let total_work_done: usize = final_counts.values().sum();
        assert_eq!(
            total_work_done,
            data.len(),
            "The total number of processed items is incorrect."
        );
    }

    #[test]
    fn test_threads_config() {
        let data = (0..1000).collect::<Vec<_>>();

        let work_counts = Arc::new(Mutex::new(HashSet::new()));
        let sum = Arc::new(AtomicUsize::new(0));
        let sum_clone = sum.clone();

        data.par_iter().with_threads(2).for_each(|&x| {
            let id = thread::current().id();
            let mut counts = work_counts.lock().unwrap();
            counts.insert(id);

            sum_clone.fetch_add(x, Ordering::Relaxed);
        });

        let threads_that_did_work = work_counts.lock().unwrap().len();

        assert_eq!(
            sum.load(Ordering::Relaxed),
            499500,
            "Sum of items is incorrect."
        );
        assert_eq!(
            threads_that_did_work, 2,
            "Expected work to be distributed across 2 threads."
        );
    }
}
