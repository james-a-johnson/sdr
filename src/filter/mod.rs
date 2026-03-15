pub mod iir;

use crate::complex::Complex;

pub trait Filter<T: Copy> {
    fn filter(&mut self, data: &[Complex<T>]) -> Vec<Complex<T>>;
}

pub struct Decimate<const N: usize>;

impl<const N: usize, T: Copy> Filter<T> for Decimate<N> {
    fn filter(&mut self, data: &[Complex<T>]) -> Vec<Complex<T>> {
        let mut filtered = Vec::with_capacity(data.len() / N + 1);
        filtered.extend(data.iter().step_by(N));
        filtered
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn decimation_filter() {
        let mut filter: Decimate<3> = Decimate;
        let samples = vec![
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
            (8, 8),
        ];
        let samples: Vec<Complex<i32>> = samples.into_iter().map(|e| e.into()).collect();
        let filtered = filter.filter(&samples);
        assert_eq!(filtered[0], (0, 0));
        assert_eq!(filtered[1], (3, 3));
        assert_eq!(filtered[2], (6, 6));
        assert_eq!(filtered.len(), 3);
    }
}
