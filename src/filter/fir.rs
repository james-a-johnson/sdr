//! FIR (Finite Impulse Response) filter with circular delay-line state.

use std::collections::VecDeque;

use super::Filter;
use crate::complex::Complex;

/// Stateful FIR filter for complex IQ streams.
///
/// Coefficients are real-valued (`f32`); I and Q channels are filtered
/// independently with the same coefficients. State persists across [`Filter::filter`]
/// calls, so splitting a stream into blocks and filtering each block in sequence
/// produces the same result as filtering the entire stream at once.
///
/// # Example
///
/// ```
/// use sdr::filter::fir::Fir;
/// use sdr::filter::Filter;
/// use sdr::complex::Complex;
///
/// // Identity filter (single tap = 1.0)
/// let mut fir = Fir::new(vec![1.0_f32]);
/// let samples = vec![Complex::new(1.0_f32, 2.0), Complex::new(3.0, 4.0)];
/// let out = fir.filter(&samples);
/// assert_eq!(out[0], samples[0]);
/// assert_eq!(out[1], samples[1]);
/// ```
pub struct Fir {
    coeffs: Vec<f32>,
    state: VecDeque<Complex<f32>>,
}

impl Fir {
    /// Create a new FIR filter with the given coefficients.
    ///
    /// The delay line is initialized to zero. The filter length (number of
    /// taps) equals `coeffs.len()`.
    pub fn new(coeffs: Vec<f32>) -> Self {
        let len = coeffs.len();
        let mut state = VecDeque::with_capacity(len);
        for _ in 0..len {
            state.push_back(Complex::new(0.0, 0.0));
        }
        Self { coeffs, state }
    }

    /// Return the filter coefficients.
    pub fn coeffs(&self) -> &[f32] {
        &self.coeffs
    }

    /// Zero the delay-line state.
    ///
    /// Call before reusing the filter on a new, unrelated stream to prevent
    /// the previous signal's tail from bleeding into the new one.
    pub fn reset(&mut self) {
        for s in self.state.iter_mut() {
            *s = Complex::new(0.0, 0.0);
        }
    }
}

impl Filter<f32> for Fir {
    /// Filter a block of complex samples, updating internal state.
    fn filter(&mut self, input: &[Complex<f32>]) -> Vec<Complex<f32>> {
        let mut out = Vec::with_capacity(input.len());
        for &x in input {
            self.state.pop_front();
            self.state.push_back(x);
            let mut sum_i = 0.0_f32;
            let mut sum_q = 0.0_f32;
            // state is ordered oldest..newest; coeffs[0] is applied to newest
            for (coeff, s) in self.coeffs.iter().zip(self.state.iter().rev()) {
                sum_i += coeff * s.i;
                sum_q += coeff * s.q;
            }
            out.push(Complex::new(sum_i, sum_q));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        let mut fir = Fir::new(vec![1.0]);
        let samples: Vec<Complex<f32>> = (0..8).map(|i| Complex::new(i as f32, -(i as f32))).collect();
        let out = fir.filter(&samples);
        assert_eq!(out.len(), samples.len());
        for (a, b) in out.iter().zip(samples.iter()) {
            assert!((a.i - b.i).abs() < 1e-6);
            assert!((a.q - b.q).abs() < 1e-6);
        }
    }

    #[test]
    fn mean_filter() {
        let mut fir = Fir::new(vec![0.5, 0.5]);
        // Input: 0, 2, 4, 6, ...
        let samples: Vec<Complex<f32>> = (0..6).map(|i| Complex::new(i as f32 * 2.0, 0.0)).collect();
        let out = fir.filter(&samples);
        // First output: avg(0, 0) = 0 (state starts at zero)
        assert!((out[0].i - 0.0).abs() < 1e-6);
        // Second output: avg(0, 2) = 1
        assert!((out[1].i - 1.0).abs() < 1e-6);
        // Third output: avg(2, 4) = 3
        assert!((out[2].i - 3.0).abs() < 1e-6);
    }

    #[test]
    fn streaming_consistency() {
        let coeffs = vec![0.25, 0.5, 0.25];
        let samples: Vec<Complex<f32>> = (0..16).map(|i| Complex::new((i as f32 * 0.3).sin(), (i as f32 * 0.2).cos())).collect();

        let mut fir_full = Fir::new(coeffs.clone());
        let out_full = fir_full.filter(&samples);

        let mut fir_split = Fir::new(coeffs.clone());
        let out_a = fir_split.filter(&samples[..8]);
        let out_b = fir_split.filter(&samples[8..]);
        let out_split: Vec<_> = out_a.into_iter().chain(out_b).collect();

        for (a, b) in out_full.iter().zip(out_split.iter()) {
            assert!((a.i - b.i).abs() < 1e-6, "I mismatch: {} vs {}", a.i, b.i);
            assert!((a.q - b.q).abs() < 1e-6, "Q mismatch: {} vs {}", a.q, b.q);
        }
    }

    #[test]
    fn reset_clears_state() {
        let coeffs = vec![0.5, 0.5];
        let samples: Vec<Complex<f32>> = (0..8).map(|i| Complex::new(i as f32, 0.0)).collect();

        let mut fir = Fir::new(coeffs.clone());
        let _ = fir.filter(&samples); // warm up state
        fir.reset();
        let out_after_reset = fir.filter(&samples);

        let mut fir_fresh = Fir::new(coeffs);
        let out_fresh = fir_fresh.filter(&samples);

        for (a, b) in out_after_reset.iter().zip(out_fresh.iter()) {
            assert!((a.i - b.i).abs() < 1e-6);
            assert!((a.q - b.q).abs() < 1e-6);
        }
    }
}
