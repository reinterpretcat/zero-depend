use super::*;

#[doc(hidden)]
pub struct StepBy<P> {
    inner: P,
    step: usize,
}

impl<P> StepBy<P> {
    pub fn new(inner: P, step: usize) -> Self {
        Self { inner, step }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_by() {
        let data = (0..1000).collect::<Vec<_>>();
        let expected = (0..1000).step_by(10).collect::<Vec<_>>();

        let results: Vec<i32> = data.par_iter().step_by(10).cloned().collect();
        assert_eq!(results, expected);

        let results: Vec<i32> = data.into_par_iter().step_by(10).collect();
        assert_eq!(results, expected);
    }
}
