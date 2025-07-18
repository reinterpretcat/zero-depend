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

- **tensor-rs**: A minimal, PyTorch-like tensor library. Demonstrates multi-dimensional array operations, views, and basic math, all built from first principles.
- **par-iter**: A parallel iterator library inspired by Rayon. Implements parallel iteration and combinators (map, zip, enumerate, etc.) using only threads and atomics from the standard library.
- **small-vec**: A small vector implementation. Provides a vector-like container optimized for small sizes, storing elements on stack when possible, and falling back to heap allocation for larger sizes.

Each crate is a standalone Rust library, ready for further development or as a learning resource.

