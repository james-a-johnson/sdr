//! Digital encoding pipelines: composable TX and RX chains for byte-oriented protocols.
//!
//! [`DigitalTxPipeline<E>`] and [`DigitalRxPipeline<D>`] wrap any [`Encode`] or
//! [`Decode`] implementation with an ordered chain of IQ filters (resamplers,
//! frequency shifts, or custom [`Filter<f32>`](crate::filter::Filter) implementations).
//! Builder methods append filters to the chain; `transmit`/`receive` execute the chain.
//!
//! The key difference from the analog pipeline is that encode/decode boundaries are
//! bytes, not `f32` audio, and decoding is fallible (`Option<Vec<u8>>`).
//!
//! # Example
//!
//! ```
//! use sdr::pipeline::digital::{DigitalTxPipeline, DigitalRxPipeline, Encode, Decode};
//! use sdr::complex::Complex;
//!
//! struct PassThrough;
//!
//! impl Encode for PassThrough {
//!     fn encode(&mut self, data: &[u8]) -> Vec<Complex<f32>> {
//!         data.iter().map(|&b| Complex::new(b as f32, 0.0)).collect()
//!     }
//! }
//!
//! impl Decode for PassThrough {
//!     fn decode(&mut self, iq: &[Complex<f32>]) -> Option<Vec<u8>> {
//!         Some(iq.iter().map(|c| c.i.round() as u8).collect())
//!     }
//! }
//!
//! let mut tx = DigitalTxPipeline::new(PassThrough);
//! let iq = tx.transmit(b"hi");
//! let mut rx = DigitalRxPipeline::new(PassThrough);
//! assert_eq!(rx.receive(&iq), Some(b"hi".to_vec()));
//! ```

use crate::complex::Complex;
use crate::filter::{freq_shift::FreqShift, iir::Iir, resample::RationalResampler, Filter};

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Full digital TX encoder: bytes → IQ samples.
///
/// Implementors handle the complete chain from raw bytes to IQ samples:
/// protocol framing, FEC encoding, symbol mapping, and modulation.
/// IQ filters (resampling, frequency shift) are applied by the enclosing
/// [`DigitalTxPipeline`].
pub trait Encode {
    fn encode(&mut self, data: &[u8]) -> Vec<Complex<f32>>;
}

/// Full digital RX decoder: IQ samples → bytes (fallible).
///
/// Implementors handle the complete chain from IQ samples to recovered bytes:
/// demodulation, frame sync, FEC decoding, and integrity checking.
/// Returns `None` if decoding fails (e.g. LDPC non-convergence, CRC mismatch).
pub trait Decode {
    fn decode(&mut self, iq: &[Complex<f32>]) -> Option<Vec<u8>>;
}

/// Digital transmit pipeline: encoder followed by zero or more IQ filters.
///
/// Filters are applied to the encoder output in the order they were added.
pub struct DigitalTxPipeline<E> {
    encoder: E,
    iq_filters: Vec<Box<dyn Filter<f32>>>,
}

/// Digital receive pipeline: zero or more IQ filters followed by a decoder.
///
/// Filters are applied to the incoming IQ in the order they were added,
/// before the decoder sees the data.
pub struct DigitalRxPipeline<D> {
    iq_filters: Vec<Box<dyn Filter<f32>>>,
    decoder: D,
}

impl<E: Encode> DigitalTxPipeline<E> {
    /// Create a pipeline wrapping `encoder` with no IQ filters.
    pub fn new(encoder: E) -> Self {
        Self { encoder, iq_filters: Vec::new() }
    }

    /// Append a [`RationalResampler`] with the given interpolation and
    /// decimation factors.
    pub fn with_resampler(mut self, interp: usize, decim: usize) -> Self {
        self.iq_filters.push(Box::new(RationalResampler::new(interp, decim)));
        self
    }

    /// Append a [`FreqShift`] that shifts IQ by `shift_hz`.
    pub fn with_freq_shift(mut self, sample_rate: f32, shift_hz: f32) -> Self {
        self.iq_filters.push(Box::new(FreqShift::new(sample_rate, shift_hz)));
        self
    }

    /// Append any [`Filter<f32>`](crate::filter::Filter) implementation.
    pub fn with_filter(mut self, f: impl Filter<f32> + 'static) -> Self {
        self.iq_filters.push(Box::new(f));
        self
    }

    /// Append a Butterworth biquad lowpass filter.
    ///
    /// * `sample_rate` — sample rate of the IQ stream in Hz
    /// * `cutoff_hz`   — −3 dB cutoff frequency in Hz
    /// * `q`           — quality factor; `std::f64::consts::FRAC_1_SQRT_2` (~0.707)
    ///   gives a maximally-flat Butterworth response
    pub fn with_lowpass(self, sample_rate: f64, cutoff_hz: f64, q: f64) -> Self {
        self.with_filter(Iir::lowpass(sample_rate, cutoff_hz, q))
    }

    /// Append a resampler that converts from `in_rate` to `out_rate` Hz,
    /// reducing the ratio by their GCD first.
    pub fn with_rates(self, in_rate: u32, out_rate: u32) -> Self {
        let g = gcd(in_rate, out_rate);
        self.with_resampler((out_rate / g) as usize, (in_rate / g) as usize)
    }

    /// Encode `data` then run the IQ filter chain, returning the final IQ.
    pub fn transmit(&mut self, data: &[u8]) -> Vec<Complex<f32>> {
        let iq = self.encoder.encode(data);
        self.iq_filters.iter_mut().fold(iq, |acc, f| f.filter(&acc))
    }
}

impl<D: Decode> DigitalRxPipeline<D> {
    /// Create a pipeline wrapping `decoder` with no IQ filters.
    pub fn new(decoder: D) -> Self {
        Self { iq_filters: Vec::new(), decoder }
    }

    /// Append a [`RationalResampler`] with the given interpolation and
    /// decimation factors.
    pub fn with_resampler(mut self, interp: usize, decim: usize) -> Self {
        self.iq_filters.push(Box::new(RationalResampler::new(interp, decim)));
        self
    }

    /// Append a [`FreqShift`] that shifts IQ by `shift_hz`.
    pub fn with_freq_shift(mut self, sample_rate: f32, shift_hz: f32) -> Self {
        self.iq_filters.push(Box::new(FreqShift::new(sample_rate, shift_hz)));
        self
    }

    /// Append any [`Filter<f32>`](crate::filter::Filter) implementation.
    pub fn with_filter(mut self, f: impl Filter<f32> + 'static) -> Self {
        self.iq_filters.push(Box::new(f));
        self
    }

    /// Append a Butterworth biquad lowpass filter.
    ///
    /// * `sample_rate` — sample rate of the IQ stream in Hz
    /// * `cutoff_hz`   — −3 dB cutoff frequency in Hz
    /// * `q`           — quality factor; `std::f64::consts::FRAC_1_SQRT_2` (~0.707)
    ///   gives a maximally-flat Butterworth response
    pub fn with_lowpass(self, sample_rate: f64, cutoff_hz: f64, q: f64) -> Self {
        self.with_filter(Iir::lowpass(sample_rate, cutoff_hz, q))
    }

    /// Append a resampler that converts from `in_rate` to `out_rate` Hz,
    /// reducing the ratio by their GCD first.
    pub fn with_rates(self, in_rate: u32, out_rate: u32) -> Self {
        let g = gcd(in_rate, out_rate);
        self.with_resampler((out_rate / g) as usize, (in_rate / g) as usize)
    }

    /// Run the IQ filter chain then decode, returning `None` on decode failure.
    pub fn receive(&mut self, iq: &[Complex<f32>]) -> Option<Vec<u8>> {
        let iq = self.iq_filters.iter_mut().fold(iq.to_vec(), |acc, f| f.filter(&acc));
        self.decoder.decode(&iq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct IdentityCodec;

    impl Encode for IdentityCodec {
        fn encode(&mut self, data: &[u8]) -> Vec<Complex<f32>> {
            data.iter().map(|&b| Complex::new(b as f32, 0.0)).collect()
        }
    }

    impl Decode for IdentityCodec {
        fn decode(&mut self, iq: &[Complex<f32>]) -> Option<Vec<u8>> {
            Some(iq.iter().map(|c| c.i.round() as u8).collect())
        }
    }

    #[test]
    fn digital_pipeline_roundtrip() {
        let data = b"hello";
        let mut tx = DigitalTxPipeline::new(IdentityCodec);
        let iq = tx.transmit(data);
        let mut rx = DigitalRxPipeline::new(IdentityCodec);
        assert_eq!(rx.receive(&iq), Some(data.to_vec()));
    }

    #[test]
    fn digital_decode_returns_none_on_failure() {
        struct AlwaysFail;
        impl Decode for AlwaysFail {
            fn decode(&mut self, _iq: &[Complex<f32>]) -> Option<Vec<u8>> {
                None
            }
        }
        let iq = vec![Complex::new(1.0_f32, 0.0)];
        let mut rx = DigitalRxPipeline::new(AlwaysFail);
        assert_eq!(rx.receive(&iq), None);
    }

    #[test]
    fn gcd_correct() {
        assert_eq!(gcd(250_000, 48_000), 2_000);
    }
}
