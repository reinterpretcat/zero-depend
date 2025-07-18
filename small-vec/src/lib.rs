use std::{mem::MaybeUninit, ops::Deref};

/// A vector that can store up to N elements on the stack,
/// and falls back to heap allocation for larger sizes.
pub struct SmallVec<T, const N: usize> {
    data: SmallVecData<T, N>,
    len: usize,
}

enum SmallVecData<T, const N: usize> {
    Stack([MaybeUninit<T>; N]),
    Heap(Vec<T>),
}

impl<T, const N: usize> SmallVec<T, N> {
    /// Creates a new empty SmallVec
    pub fn new() -> Self {
        Self {
            data: SmallVecData::Stack([const { MaybeUninit::uninit() }; N]),
            len: 0,
        }
    }

    /// Returns the number of elements in the SmallVec
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the SmallVec is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the capacity of the SmallVec
    pub fn capacity(&self) -> usize {
        match &self.data {
            SmallVecData::Stack(_) => N,
            SmallVecData::Heap(vec) => vec.capacity(),
        }
    }

    /// Adds an element to the end of the SmallVec
    pub fn push(&mut self, value: T) {
        match &mut self.data {
            SmallVecData::Stack(arr) => {
                if self.len < N {
                    // Safe: we're writing to an uninitialized MaybeUninit
                    arr[self.len].write(value);
                    self.len += 1;
                } else {
                    // Convert to heap allocation
                    self.spill_to_heap();
                    if let SmallVecData::Heap(vec) = &mut self.data {
                        vec.push(value);
                        self.len += 1;
                    }
                }
            }
            SmallVecData::Heap(vec) => {
                vec.push(value);
                self.len += 1;
            }
        }
    }

    /// Removes and returns the last element, or None if empty
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        match &mut self.data {
            SmallVecData::Stack(arr) => {
                self.len -= 1;
                // SAFETY: we know this element is initialized because len > 0
                // and we only increment len when we write to elements
                Some(unsafe { arr[self.len].assume_init_read() })
            }
            SmallVecData::Heap(vec) => {
                self.len -= 1;
                vec.pop()
            }
        }
    }

    /// Returns a reference to the element at the given index
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }

        match &self.data {
            SmallVecData::Stack(arr) => {
                // SAFETY: we've checked that index < len, and we only increment len
                // when we write to elements
                Some(unsafe { arr[index].assume_init_ref() })
            }
            SmallVecData::Heap(vec) => vec.get(index),
        }
    }

    /// Returns a mutable reference to the element at the given index
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }

        match &mut self.data {
            SmallVecData::Stack(arr) => {
                // SAFETY: we've checked that index < len, and we only increment len
                // when we write to elements
                Some(unsafe { arr[index].assume_init_mut() })
            }
            SmallVecData::Heap(vec) => vec.get_mut(index),
        }
    }

    /// Clears the SmallVec, removing all elements
    pub fn clear(&mut self) {
        match &mut self.data {
            SmallVecData::Stack(arr) => {
                // SAFETY: we only drop elements that are initialized (0..len)
                for i in 0..self.len {
                    unsafe { arr[i].assume_init_drop() };
                }
            }
            SmallVecData::Heap(vec) => {
                vec.clear();
            }
        }
        self.len = 0;
    }

    /// Converts stack storage to heap storage
    fn spill_to_heap(&mut self) {
        if let SmallVecData::Stack(arr) = &mut self.data {
            let mut vec = Vec::with_capacity(N * 2);

            // Move all elements from stack to heap
            for i in 0..self.len {
                // SAFETY: we only read from initialized elements (0..len)
                let value = unsafe { arr[i].assume_init_read() };
                vec.push(value);
            }

            self.data = SmallVecData::Heap(vec);
        }
    }

    /// Returns an iterator over the elements
    pub fn iter(&self) -> SmallVecIter<T, N> {
        SmallVecIter {
            small_vec: self,
            index: 0,
        }
    }

    /// Creates a SmallVec from an iterator
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut sv = Self::new();
        for item in iter {
            sv.push(item);
        }
        sv
    }

    /// Reserves capacity for at least `additional` more elements
    pub fn reserve(&mut self, additional: usize) {
        let required_capacity = self.len + additional;

        if required_capacity <= self.capacity() {
            return;
        }

        // If we need more capacity than stack can provide, spill to heap
        if required_capacity > N {
            self.spill_to_heap();
            if let SmallVecData::Heap(vec) = &mut self.data {
                vec.reserve(additional);
            }
        }
    }

    /// Inserts an element at the given index
    pub fn insert(&mut self, index: usize, element: T) -> bool {
        if index > self.len {
            return false;
        }

        if self.len == self.capacity() {
            self.spill_to_heap();
        }

        match &mut self.data {
            SmallVecData::Stack(arr) => {
                // Shift elements to the right
                for i in (index..self.len).rev() {
                    // Safe: we're moving from initialized to uninitialized slots
                    let value = unsafe { arr[i].assume_init_read() };
                    arr[i + 1].write(value);
                }
                arr[index].write(element);
                self.len += 1;
            }
            SmallVecData::Heap(vec) => {
                vec.insert(index, element);
                self.len += 1;
            }
        }

        return true;
    }

    /// Removes and returns the element at the given index
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }

        match &mut self.data {
            SmallVecData::Stack(arr) => {
                // Safe: we've checked that index < len
                let element = unsafe { arr[index].assume_init_read() };

                // Shift elements to the left
                for i in index..self.len - 1 {
                    // Safe: moving from initialized to initialized slots
                    let value = unsafe { arr[i + 1].assume_init_read() };
                    arr[i].write(value);
                }

                self.len -= 1;
                Some(element)
            }
            SmallVecData::Heap(vec) => {
                self.len -= 1;
                Some(vec.remove(index))
            }
        }
    }

    pub fn swap(&mut self, a: usize, b: usize) -> bool {
        if a >= self.len || b >= self.len {
            return false;
        }

        match &mut self.data {
            SmallVecData::Stack(arr) => {
                // Safe: we've checked that a and b are within bounds
                let temp = unsafe { arr[a].assume_init_read() };
                arr[a].write(unsafe { arr[b].assume_init_read() });
                arr[b].write(temp);
            }
            SmallVecData::Heap(vec) => {
                vec.swap(a, b);
            }
        }

        true
    }
}

impl<T, const N: usize> std::ops::Index<usize> for SmallVec<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("Index out of bounds")
    }
}

impl<T, const N: usize> std::ops::IndexMut<usize> for SmallVec<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("Index out of bounds")
    }
}

impl<T, const N: usize> Default for SmallVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Drop for SmallVec<T, N> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T: Clone, const N: usize> Clone for SmallVec<T, N> {
    fn clone(&self) -> Self {
        let mut new_vec = Self::new();
        for item in self.iter() {
            new_vec.push(item.clone());
        }
        new_vec
    }
}

impl<T: PartialEq, const N: usize> PartialEq for SmallVec<T, N> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }

        for i in 0..self.len {
            if self.get(i) != other.get(i) {
                return false;
            }
        }

        true
    }
}

impl<T: std::fmt::Debug, const N: usize> std::fmt::Debug for SmallVec<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

// Iterator implementation
pub struct SmallVecIter<'a, T, const N: usize> {
    small_vec: &'a SmallVec<T, N>,
    index: usize,
}

impl<'a, T, const N: usize> Iterator for SmallVecIter<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.small_vec.len() {
            let result = self.small_vec.get(self.index);
            self.index += 1;
            result
        } else {
            None
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a SmallVec<T, N> {
    type Item = &'a T;
    type IntoIter = SmallVecIter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T, const N: usize> FromIterator<T> for SmallVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from_iter(iter)
    }
}

// Extend trait implementation
impl<T, const N: usize> Extend<T> for SmallVec<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T, const N: usize> Deref for SmallVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match &self.data {
            SmallVecData::Stack(arr) => {
                // SAFETY: only initialized elements (0..len)
                unsafe { std::slice::from_raw_parts(arr.as_ptr() as *const T, self.len) }
            }
            SmallVecData::Heap(vec) => vec.deref(),
        }
    }
}

#[macro_export]
macro_rules! small_vec {
    ($($elem:expr),* $(,)?) => {
        {
            let mut sv = $crate::SmallVec::new();
            $(sv.push($elem);)*
            sv
        }
    };
    ($elem:expr; $n:expr) => {
        {
            let mut sv = $crate::SmallVec::new();
            for _ in 0..$n {
                sv.push($elem.clone());
            }
            sv
        }
    };
}

// Example usage and tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_vec_basic() {
        let mut sv: SmallVec<i32, 4> = SmallVec::new();
        assert_eq!(sv.len(), 0);
        assert!(sv.is_empty());

        sv.push(1);
        sv.push(2);
        sv.push(3);
        assert_eq!(sv.len(), 3);
        assert_eq!(sv[0], 1);
        assert_eq!(sv[1], 2);
        assert_eq!(sv[2], 3);
    }

    #[test]
    fn test_spill_to_heap() {
        let mut sv: SmallVec<i32, 2> = SmallVec::new();
        sv.push(1);
        sv.push(2);

        // Should still be on stack
        assert_eq!(sv.capacity(), 2);
        assert!(matches!(sv.data, SmallVecData::Stack(_)));

        sv.push(3); // This should spill to heap
        assert!(matches!(sv.data, SmallVecData::Heap(_)));
        assert!(sv.capacity() > 2);
        assert_eq!(sv.len(), 3);
        assert_eq!(sv[0], 1);
        assert_eq!(sv[1], 2);
        assert_eq!(sv[2], 3);
    }

    #[test]
    fn test_pop() {
        let mut sv: SmallVec<i32, 4> = SmallVec::new();
        sv.push(1);
        sv.push(2);

        assert_eq!(sv.pop(), Some(2));
        assert_eq!(sv.pop(), Some(1));
        assert_eq!(sv.pop(), None);
        assert!(sv.is_empty());
    }

    #[test]
    fn test_iterator() {
        let mut sv: SmallVec<i32, 4> = SmallVec::new();
        sv.push(1);
        sv.push(2);
        sv.push(3);

        let collected: Vec<&i32> = sv.iter().collect();
        assert_eq!(collected, vec![&1, &2, &3]);
    }

    #[test]
    fn test_insert_remove() {
        let mut sv: SmallVec<i32, 4> = SmallVec::new();
        sv.push(1);
        sv.push(3);

        sv.insert(1, 2);
        assert_eq!(sv.len(), 3);
        assert_eq!(sv[0], 1);
        assert_eq!(sv[1], 2);
        assert_eq!(sv[2], 3);

        let removed = sv.remove(1).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(sv.len(), 2);
        assert_eq!(sv[0], 1);
        assert_eq!(sv[1], 3);
    }

    #[test]
    fn test_from_iterator() {
        let sv: SmallVec<i32, 4> = (1..=3).collect();
        assert_eq!(sv.len(), 3);
        assert_eq!(sv[0], 1);
        assert_eq!(sv[1], 2);
        assert_eq!(sv[2], 3);
    }

    #[test]
    fn test_clone() {
        let mut sv1: SmallVec<i32, 4> = SmallVec::new();
        sv1.push(1);
        sv1.push(2);

        let sv2 = sv1.clone();
        assert_eq!(sv1, sv2);
    }

    #[test]
    fn test_macro() {
        let sv: SmallVec<i32, 4> = small_vec![1, 2, 3];
        assert_eq!(sv.len(), 3);
        assert_eq!(sv[0], 1);
        assert_eq!(sv[1], 2);
        assert_eq!(sv[2], 3);

        let sv2: SmallVec<i32, 4> = small_vec![0; 5];
        assert_eq!(sv2.len(), 5);
        for i in 0..5 {
            assert_eq!(sv2[i], 0);
        }
    }

    #[test]
    fn test_swap() {
        let mut sv: SmallVec<i32, 4> = small_vec![1, 2, 3];
        assert!(sv.swap(0, 1));
        assert_eq!(sv[0], 2);
        assert_eq!(sv[1], 1);
        assert!(sv.swap(1, 2));
        assert_eq!(sv[1], 3);
        assert_eq!(sv[2], 1);
        assert!(!sv.swap(1, 4));
    }

    #[test]
    fn test_deref() {
        let mut sv: SmallVec<i32, 4> = SmallVec::new();
        sv.push(1);
        sv.push(2);
        sv.push(3);

        let slice: &[i32] = &sv;
        assert_eq!(slice.len(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);
    }
}
