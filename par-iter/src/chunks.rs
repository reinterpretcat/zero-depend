use super::*;

// Producer for immutable chunks
#[doc(hidden)]
pub struct Chunks<'a, T: 'a> {
    ptr: *const T,
    len: usize,
    chunk_size: usize,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T: Send + Sync> Chunks<'a, T> {
    pub fn new(slice: &'a [T], chunk_size: usize) -> Self {
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
    pub fn new(slice: &'a mut [T], chunk_size: usize) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
