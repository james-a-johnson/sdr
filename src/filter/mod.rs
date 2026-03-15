//! Streaming filters for IQ sample processing.
//!
//! All filters implement the [`Filter`] trait, which processes a slice of
//! [`Complex<T>`](crate::complex::Complex) samples and returns a new `Vec`.
//! State is stored inside the filter struct and persists across calls, making
//! every filter safe to use on a continuous stream split into arbitrary blocks.
//!
//! # Available filters
//!
//! | Type | Description |
//! |---|---|
//! | [`Decimate<N>`] | Keep every N-th sample (integer decimation, no anti-aliasing) |
//! | [`iir::Iir`] | Biquad IIR filter with lowpass / highpass / bandpass presets |
//! | [`freq_shift::FreqShift`] | Frequency shift by multiplying with a rotating phasor |
//! | [`fir::Fir`] | FIR filter with arbitrary real-valued coefficients |
//! | [`resample::RationalResampler`] | Polyphase rational resampler for arbitrary L:M rate conversion |

pub mod fir;
pub mod freq_shift;
pub mod iir;
pub mod resample;

use crate::complex::Complex;

/// Common interface for all streaming IQ filters.
///
/// Implementations must be stateful: calling `filter` twice on two halves of
/// a stream must produce the same result as calling it once on the whole
/// stream.
pub trait Filter<T: Copy> {
    /// Process `data` and return the filtered samples.
    ///
    /// The returned `Vec` may have a different length than `data` (e.g.
    /// [`Decimate`] shortens the output).
    fn filter(&mut self, data: &[Complex<T>]) -> Vec<Complex<T>>;
}

/// Integer decimation filter that keeps every `N`-th sample.
///
/// No anti-aliasing is applied; pair with an [`iir::Iir`] lowpass filter
/// before decimating if aliasing matters.
///
/// # Example
///
/// ```
/// use sdr::filter::{Decimate, Filter};
/// use sdr::complex::Complex;
///
/// let samples: Vec<Complex<i32>> = (0..9).map(|i| Complex::new(i, 0)).collect();
/// let mut dec: Decimate<3> = Decimate;
/// let out = dec.filter(&samples);
/// assert_eq!(out.len(), 3); // samples 0, 3, 6
/// ```
pub struct Decimate<const N: usize>;

impl<const N: usize, T: Copy> Filter<T> for Decimate<N> {
    /// Return every `N`-th input sample starting from index 0.
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
