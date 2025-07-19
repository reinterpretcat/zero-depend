use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

mod take;
pub use self::take::Take;

mod cartesian;
pub use self::cartesian::CartesianProduct;
mod chunks;
pub use self::chunks::{Chunks, ChunksMut};
mod enumerate;
pub use self::enumerate::Enumerate;
mod map;
pub use self::map::Map;
mod zip;
pub use self::zip::Zip;
mod skip;
pub use self::skip::Skip;
mod step_by;
pub use self::step_by::StepBy;
mod cloned;
pub use self::cloned::Cloned;

mod collect;
use self::collect::Collect;
mod find;
use self::find::Find;
mod for_each;
use self::for_each::ForEach;
mod reduce;
use self::reduce::Reduce;

/// A parallel iterator that allows for efficient parallel processing of items.
/// It provides methods to configure the number of threads and chunk size for work distribution.
/// The iterator can be created from various data structures like slices, vectors, and ranges.
/// It supports operations like `for_each`, `collect`, `reduce`, and `find`.
/// This iterator is designed to work with types that implement the `ParallelProducer` trait,
/// which defines how to produce items in parallel.
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
    /// Creates a cartesian product with another parallel iterator.
    /// Each item in the result is a tuple (A, B) where A comes from self and B from other.
    ///
    /// # Example
    /// ```
    /// use par-iter::*;
    ///
    /// let vec1 = vec![1, 2, 3];
    /// let vec2 = vec!['a', 'b'];
    ///
    /// let result: Vec<_> = vec1.par_iter()
    ///     .cartesian_product(vec2.par_iter())
    ///     .collect();
    ///
    /// // Result contains: [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b'), (3, 'a'), (3, 'b')]
    /// ```
    pub fn cartesian_product<Q: ParallelProducer>(
        self,
        other: ParIter<Q>,
    ) -> ParIter<CartesianProduct<P, Q>>
    where
        P::Item: Clone + Send + Sync,
        Q::Item: Clone + Send + Sync,
    {
        ParIter {
            producer: CartesianProduct::new(self.producer, other.producer),
            config: self.config,
        }
    }

    pub fn take(self, take_count: usize) -> ParIter<Take<P>> {
        ParIter {
            producer: Take::new(self.producer, take_count),
            config: self.config,
        }
    }

    pub fn enumerate(self) -> ParIter<Enumerate<P>> {
        ParIter {
            producer: Enumerate::new(self.producer),
            config: self.config,
        }
    }

    pub fn map<F, U>(self, f: F) -> ParIter<Map<P, F>>
    where
        F: Fn(P::Item) -> U + Send + Sync,
    {
        ParIter {
            producer: Map::new(self.producer, f),
            config: self.config,
        }
    }

    pub fn zip<Q: ParallelProducer>(self, other: ParIter<Q>) -> ParIter<Zip<P, Q>> {
        ParIter::new(Zip::new(self.producer, other.producer))
    }

    pub fn skip(self, skip_count: usize) -> ParIter<Skip<P>> {
        ParIter {
            producer: Skip::new(self.producer, skip_count),
            config: self.config,
        }
    }

    pub fn step_by(self, step: usize) -> ParIter<StepBy<P>> {
        ParIter {
            producer: StepBy::new(self.producer, step),
            config: self.config,
        }
    }

    pub fn cloned<'a, T>(self) -> ParIter<Cloned<P>>
    where
        T: 'a + Clone + Send + Sync,
        P: ParallelProducer<Item = &'a T>,
    {
        ParIter {
            producer: Cloned::new(self.producer),
            config: self.config,
        }
    }

    /// Executes the parallel iteration.
    pub fn for_each<F>(self, f: F)
    where
        F: Fn(P::Item) + Send + Sync,
    {
        ForEach::new(self.producer, self.config).for_each(f);
    }

    pub fn try_for_each<F, E>(self, f: F) -> Result<(), E>
    where
        F: Fn(P::Item) -> Result<(), E> + Send + Sync,
        E: Send + Sync + 'static,
    {
        ForEach::new(self.producer, self.config).try_for_each(f)
    }

    /// Collects all items into a Vec, preserving the original order.
    pub fn collect<B>(self) -> B
    where
        B: FromIterator<P::Item>,
        P::Item: Send,
    {
        Collect::new(self.producer, self.config).collect()
    }

    /// Reduces the parallel iterator to a single value.
    pub fn reduce<F>(self, identity: P::Item, op: F) -> P::Item
    where
        P::Item: Send + Clone,
        F: Fn(P::Item, P::Item) -> P::Item + Send + Sync,
    {
        Reduce::new(self.producer, self.config).reduce(identity, op)
    }

    /// Finds the first item that matches the predicate.
    pub fn find<F>(self, predicate: F) -> Option<P::Item>
    where
        F: Fn(&P::Item) -> bool + Send + Sync,
        P::Item: Send + Sync,
    {
        Find::new(self.producer, self.config).find(predicate)
    }
}

/// Configuration for parallel execution.
/// Allows setting the number of threads and chunk size for work distribution.
/// If not set, defaults to the number of available threads and a chunk size based on the total number of items.
/// This configuration can be adjusted to optimize performance based on the workload and system capabilities.
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

/// Trait for parallel producers that can be used with `ParIter`.
pub trait ParallelProducer: Send + Sync {
    type Item;

    /// Returns the total number of items to be processed.
    fn len(&self) -> usize;

    /// Gets a single item by its index.
    fn get_item(&self, index: usize) -> Option<Self::Item>;
}

/// Trait for types that can be converted into a parallel iterator.
#[doc(hidden)]
pub trait IntoParallelIterator {
    type Item;
    type Producer: ParallelProducer<Item = Self::Item>;

    fn into_par_iter(self) -> ParIter<Self::Producer>;
}

/// Trait for slices that can be iterated in parallel.
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

/// Producer for owned Vec<T>
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

/// Producer for immutable slices
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

/// Producer for mutable slices
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

/// Producer for ranges
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::sync::Mutex;

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
