# Zero Dependency Rust Workspace

This repository demonstrates a **zero dependency** philosophy: all crates are implemented using only the Rust standard library, with no external dependencies. The primary goal is educational—showcasing how to build robust, efficient, and idiomatic Rust libraries from scratch, without relying on third-party crates. While the code aims for high quality, it is primarily a learning project and may not always reach full production-grade standards.

Some of the code in this workspace was generated or assisted by large language models (LLMs), exploring how AI can help design and implement foundational Rust libraries from scratch.

Cross-references between crates are allowed where appropriate (for example, `small-vec` is used as a building block in `tensor-rs`).

This approach helps:

- Deepen understanding of Rust's core features and standard library
- Encourage learning by re-implementing common abstractions
- Improve portability and auditability (no supply chain risk)
- Minimize compile times and binary size
- Provide a clear reference for how things work "under the hood"

## Crates in this Workspace

* **tensor-rs**: A minimal, PyTorch-like tensor library. Demonstrates multi-dimensional array operations, views, and basic math.
* **par-iter**: A parallel iterator library inspired by rayon. Implements parallel iteration and combinators (map, zip, enumerate, etc.) using only threads and atomics from the standard library.
* **small-vec**: A small vector implementation. Provides a vector-like container optimized for small sizes, storing elements on stack when possible, and falling back to heap allocation for larger sizes.

Each crate is a standalone Rust library, ready for further development or as a learning resource.


## Examples

### tensor-rs

#### Basic Usage
```rust
use tensor_rs::Tensor;

// 1D tensor
let t = Tensor::<i32>::arange(5)?;
assert_eq!(format!("{t}"), "[0, 1, 2, 3, 4]");

// 2D tensor
let t2 = Tensor::<i32>::arange(6)?.view(&[2, 3])?;
assert_eq!(format!("{t2}"), "[[0, 1, 2],\n [3, 4, 5]]");
```

#### Matrix Operations
```rust
use tensor_rs::Tensor;

// Matrix multiplication
let a = Tensor::<i32>::arange(6)?.view(&[2, 3])?;
let b = Tensor::<i32>::arange(6)?.view(&[3, 2])?;
println!("{}", a.matmul(&b)?);
//[[10, 13],
// [28, 40]]

// 2D x 2D batch matmul
let a = Tensor::<i32>::arange(12)?.view(&[2, 3, 2])?;
let b = Tensor::<i32>::arange(6)?.view(&[2, 3])?;
println!("{}", a.matmul(&b)?);
// [[[ 3,  4,  5],
//   [ 9, 14, 19],
//   [15, 24, 33]],

//  [[21, 34, 47],
//   [27, 44, 61],
//   [33, 54, 75]]]

```

#### Slicing and Views
```rust

// Create a 4x6 test tensor for slicing examples
let tensor = Tensor::<i32>::arange(24)?.view(&[4, 6])?;

// 1. Full slice (returns the whole tensor)
let full_slice = tensor.slice(s![.., ..])?;
assert_eq!(full_slice, tensor);

// 2. Row slice (second row)
let row_slice = tensor.slice(s![1, ..])?;
assert_eq!(row_slice, Tensor::from(vec![6, 7, 8, 9, 10, 11]));

// 3. Column slice (third column)
let col_slice = tensor.slice(s![.., 2])?;
assert_eq!(col_slice, Tensor::from(vec![2, 8, 14, 20]));

// 4. Range slice (rows 1..3, cols 2..5)
let range_slice = tensor.slice(s![1..3, 2..5])?;
assert_eq!(range_slice, Tensor::new(vec![8, 9, 10, 14, 15, 16], &[2, 3])?);

// 5. Negative single index (last row)
let last_row = tensor.slice(s![-1, ..])?;
assert_eq!(last_row, Tensor::from(vec![18, 19, 20, 21, 22, 23]));

// 6. Negative range (last two rows)
let last_two_rows = tensor.slice(s![-2.., ..])?;
assert_eq!(last_two_rows, Tensor::try_from(vec![[12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23]])?);

// 7. Negative end (all but last column)
let without_last_col = tensor.slice(s![.., ..-1])?;
assert_eq!(without_last_col.shape(), &[4, 5]);

// 8. Step by 2 on first dimension
let stepped = tensor.slice(&[step![.., 2]])?;
assert_eq!(stepped.shape(), &[2, 6]);

// 9. Step with range
let stepped_range = tensor.slice(&[step![0..3, 2]])?;
assert_eq!(stepped_range.shape(), &[2, 6]);

// 10. Inclusive range
let inclusive = tensor.slice(s![1..=2, 2..=4])?;
assert_eq!(inclusive.shape(), &[2, 3]);

// 11. Empty slice
let empty = tensor.slice(s![1..1, ..])?;
assert_eq!(empty.shape(), &[0, 6]);

// 12. Single element
let single = tensor.slice(s![1, 2])?;
assert_eq!(single.shape(), &[]);
assert_eq!(single.elements().count(), 1);

// 13. Macro-based slicing with negative indices
let slice1 = tensor.slice(s![-1, ..])?;
assert_eq!(slice1.shape(), &[6]);

let slice2 = tensor.slice(s![-2..-1, ..])?;
assert_eq!(slice2.shape(), &[1, 6]);

let slice3 = tensor.slice(s![.., ..-1])?;
assert_eq!(slice3.shape(), &[4, 5]);

// 14. Stepped slicing with macro
let tensor2 = Tensor::<i32>::arange(24)?.view(&[6, 4])?;
let slice = tensor2.slice(&[step![0.., 2]])?;
assert_eq!(slice.shape(), &[3, 4]);

// 15. Multiple step ranges
let two_steps = tensor2.slice(&[step![-6..=-2, 2], step!(.., 2)])?;
assert_eq!(two_steps.shape(), &[3, 2]);

// 16. Mixed slicing: single index and range
let tensor3 = Tensor::<i32>::arange(24)?.view(&[2, 3, 4])?;
let mixed = tensor3.slice(s![0, 0..2, ..])?;
assert_eq!(mixed.shape(), &[2, 4]);

// More..

```

---

### par-iter

#### Basic Usage
```rust
use par_iter::ParIter;

// Parallel map and collect
let data = (1..=10).collect::<Vec<_>>();
let doubled: Vec<_> = data.par_iter().map(|&x| x * 2).collect();
assert_eq!(doubled, vec![2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);

// Parallel for_each with enumerate
let mut data = vec![0; 20];
data
    .par_iter_mut()
    .enumerate()
    .for_each(|(i, val)| {
        *val = i as i32 * 2;
    });
assert_eq!(data, (0..20).map(|i| i * 2).collect::<Vec<_>>());
```

#### Advanced Combinators
```rust
use par_iter::ParIter;

// Parallel zip
let a = vec![1, 2, 3, 4];
let b = vec![10, 20, 30, 40];
let mut result = vec![0; 4];
a.par_iter()
    .zip(b.par_iter())
    .zip(result.par_iter_mut())
    .for_each(|((x, y), r)| {
        *r = x + y;
    });
assert_eq!(result, vec![11, 22, 33, 44]);

// Parallel find
let data = (0..1000).collect::<Vec<_>>();
let found = data.par_iter().find(|&&x| x == 333);
assert_eq!(found, Some(&333));

// Parallel reduce
let data = (0..1000).collect::<Vec<_>>();
let sum = data.par_iter().map(|&x| x).reduce(0, |acc, x| acc + x);
assert_eq!(sum, 499500);

// Parallel chunked processing
let mut data = vec![0; 16];
let chunk_size = 4;
data
    .par_chunks_mut(chunk_size)
    .enumerate()
    .for_each(|(chunk_idx, chunk)| {
        for (i, val) in chunk.iter_mut().enumerate() {
            *val = (chunk_idx * chunk_size + i) as i32;
        }
    });
assert_eq!(data, (0..16).collect::<Vec<_>>());
```

#### Matrix multiplication with Tensors

```rust
let a = Tensor::<i32>::new((1..=12).collect(), &[3, 4])?;
let b = Tensor::<i32>::new((1..=12).collect(), &[4, 3])?;

let mut result_data = vec![0; 3 * 3];

a.par_rows()
    .cartesian_product(b.transpose(0, 1)?.par_rows())
    .zip(result_data.par_iter_mut())
    .try_for_each(|(((_, row_tensor), (_, col_tensor)), v)| {
        *v = row_tensor.dot(&col_tensor)?;
        Ok(())
    })?;

assert_eq!(result_data, vec![70, 80, 90, 158, 184, 210, 246, 288, 330])

```

---

### small-vec

#### Basic Usage
```rust
use small_vec::{SmallVec, small_vec};

// Stack-allocated for small sizes
let mut v: SmallVec<i32, 4> = small_vec![1, 2, 3];
v.push(4);
v.push(5); // May move to heap if capacity exceeded
assert_eq!(&v[..], &[1, 2, 3, 4, 5]);
```

#### Capacity and Heap Promotion
```rust
use small_vec::{SmallVec, small_vec};

let mut v: SmallVec<i32, 2> = small_vec![];
v.push(1);
v.push(2);
v.push(3); // Moves to heap
assert_eq!(v, small_vec![1, 2, 3]);
```

