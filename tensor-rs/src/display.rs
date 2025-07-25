use crate::Tensor;
use std::fmt;

// Display implementation
impl<T: fmt::Display, const N: usize> fmt::Display for Tensor<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Calculate the maximum width needed for any element
        let max_width = self
            .data
            .iter()
            .map(|value| format!("{}", value).len())
            .max()
            .unwrap_or(1);

        self.fmt_recursive(f, &mut vec![0; self.shape.len()], 0, max_width)
    }
}

impl<T: fmt::Display, const N: usize> Tensor<T, N> {
    fn fmt_recursive(
        &self,
        f: &mut fmt::Formatter<'_>,
        indices: &mut Vec<usize>,
        depth: usize,
        max_width: usize,
    ) -> fmt::Result {
        if depth == self.shape.len() {
            let value = self.get(indices).map_err(|_| fmt::Error)?;
            if self.shape.len() == 1 {
                // 1D: no padding
                write!(f, "{}", value)
            } else {
                // 2D+: right-align with calculated width
                write!(f, "{:>width$}", value, width = max_width)
            }
        } else {
            let current_dim_size = self.shape[depth];
            let is_last_dim = depth == self.shape.len() - 1;

            write!(f, "[")?;

            for i in 0..current_dim_size {
                indices[depth] = i;

                if i > 0 {
                    if is_last_dim {
                        write!(f, ", ")?;
                    } else {
                        write!(f, ",\n")?;

                        // Add extra newlines based on depth for higher dimensions
                        if self.shape.len() >= 3 {
                            match depth {
                                0 => write!(f, "\n")?, // Extra newline between top-level elements
                                1 if self.shape.len() >= 4 => write!(f, "\n")?, // Extra newline between second-level elements in 4D+
                                _ => {}
                            }
                        }

                        // Simple indentation: just depth + 1 spaces
                        let indent = depth + 1;
                        for _ in 0..indent {
                            write!(f, " ")?;
                        }
                    }
                }

                self.fmt_recursive(f, indices, depth + 1, max_width)?;
            }

            write!(f, "]")?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Result;

    #[test]
    fn test_tensor_display() -> Result<()> {
        let tensor = Tensor::<i32>::arange(5)?;
        assert_eq!(format!("{tensor}"), "[0, 1, 2, 3, 4]");
        Ok(())
    }

    #[test]
    fn test_tensor_2d_display() -> Result<()> {
        let tensor = Tensor::<i32>::arange(6)?.view(&[2, 3])?;
        let expected = r#"
[[0, 1, 2],
 [3, 4, 5]]"#;
        assert_eq!(format!("{tensor}"), expected.trim());
        Ok(())
    }

    #[test]
    fn test_tensor_3d_display() -> Result<()> {
        let tensor = Tensor::<i32>::arange(4 * 3 * 2)?.view(&[4, 3, 2])?;
        let expected = r#"
[[[ 0,  1],
  [ 2,  3],
  [ 4,  5]],

 [[ 6,  7],
  [ 8,  9],
  [10, 11]],

 [[12, 13],
  [14, 15],
  [16, 17]],

 [[18, 19],
  [20, 21],
  [22, 23]]]"#;
        assert_eq!(format!("{tensor}"), expected.trim());
        Ok(())
    }

    #[test]
    fn test_tensor_3d_display_3digits() -> Result<()> {
        let tensor = Tensor::<i32>::arange(4 * 3 * 10)?.view(&[4, 3, 10])?;
        let expected = r#"
[[[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9],
  [ 10,  11,  12,  13,  14,  15,  16,  17,  18,  19],
  [ 20,  21,  22,  23,  24,  25,  26,  27,  28,  29]],

 [[ 30,  31,  32,  33,  34,  35,  36,  37,  38,  39],
  [ 40,  41,  42,  43,  44,  45,  46,  47,  48,  49],
  [ 50,  51,  52,  53,  54,  55,  56,  57,  58,  59]],

 [[ 60,  61,  62,  63,  64,  65,  66,  67,  68,  69],
  [ 70,  71,  72,  73,  74,  75,  76,  77,  78,  79],
  [ 80,  81,  82,  83,  84,  85,  86,  87,  88,  89]],

 [[ 90,  91,  92,  93,  94,  95,  96,  97,  98,  99],
  [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
  [110, 111, 112, 113, 114, 115, 116, 117, 118, 119]]]"#;
        assert_eq!(format!("{tensor}"), expected.trim());
        Ok(())
    }

    #[test]
    fn test_tensor_4d_display() -> Result<()> {
        let tensor = Tensor::<i32>::arange(2 * 3 * 4 * 2)?.view(&[2, 3, 4, 2])?;
        let expected = r#"
[[[[ 0,  1],
   [ 2,  3],
   [ 4,  5],
   [ 6,  7]],

  [[ 8,  9],
   [10, 11],
   [12, 13],
   [14, 15]],

  [[16, 17],
   [18, 19],
   [20, 21],
   [22, 23]]],

 [[[24, 25],
   [26, 27],
   [28, 29],
   [30, 31]],

  [[32, 33],
   [34, 35],
   [36, 37],
   [38, 39]],

  [[40, 41],
   [42, 43],
   [44, 45],
   [46, 47]]]]"#;
        assert_eq!(format!("{tensor}"), expected.trim());
        Ok(())
    }
}
