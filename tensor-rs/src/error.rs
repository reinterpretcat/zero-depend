use std::fmt;

#[derive(Clone, Debug)]
pub enum TensorError {
    CastError(String),
    IndexOutOfBounds(String),
    ShapeMismatch(String),
    UnsupportedOperation(String),
}

impl std::error::Error for TensorError {}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::CastError(msg) => write!(f, "Cast Error: {}", msg),
            TensorError::IndexOutOfBounds(msg) => write!(f, "Index Out of Bounds: {}", msg),
            TensorError::ShapeMismatch(msg) => write!(f, "Shape Mismatch: {}", msg),
            TensorError::UnsupportedOperation(msg) => write!(f, "Unsupported Operation: {}", msg),
        }
    }
}
