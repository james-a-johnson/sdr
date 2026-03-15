//! Modulation pipelines: composable TX and RX chains.
//!
//! [`TxPipeline<M>`] and [`RxPipeline<D>`] wrap any [`Modulate`] or
//! [`Demodulate`] implementation with an ordered chain of IQ filters
//! (resamplers, frequency shifts, or custom [`Filter<f32>`](crate::filter::Filter)
//! implementations). Builder methods append filters to the chain; the
//! [`Transmitter`] and [`Receiver`] trait implementations execute the chain at
//! call time.
//!
//! # Example
//!
//! ```
//! use sdr::pipeline::am::{AmTx, AmRx};
//! use sdr::modulation::am::{AmModulator, AmDemodulator};
//! use sdr::pipeline::{Transmitter, Receiver};
//!
//! let mut tx = AmTx::new(AmModulator::new(0.8))
//!     .with_rates(48_000, 250_000);
//! let audio: Vec<f32> = (0..480).map(|i| (i as f32 * 0.1).sin()).collect();
//! let iq = tx.transmit(&audio);
//! assert!((iq.len() as f32 - 480.0 * 250_000.0 / 48_000.0).abs() <= 2.0);
//! ```

pub mod am;
pub mod fm;
pub mod fsk8;

use crate::complex::Complex;
use crate::filter::{freq_shift::FreqShift, iir::Iir, resample::RationalResampler, Filter};

/// Encode baseband `f32` audio into IQ samples.
pub trait Modulate {
    fn modulate(&mut self, audio: &[f32]) -> Vec<Complex<f32>>;
}

/// Decode IQ samples into baseband `f32` audio.
pub trait Demodulate {
    fn demodulate(&mut self, iq: &[Complex<f32>]) -> Vec<f32>;
}

/// High-level transmit interface: audio in, IQ out.
pub trait Transmitter {
    fn transmit(&mut self, audio: &[f32]) -> Vec<Complex<f32>>;
}

/// High-level receive interface: IQ in, audio out.
pub trait Receiver {
    fn receive(&mut self, iq: &[Complex<f32>]) -> Vec<f32>;
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Transmit pipeline: modulator followed by zero or more IQ filters.
///
/// Filters are applied to the modulator output in the order they were added.
pub struct TxPipeline<M> {
    modulator: M,
    iq_filters: Vec<Box<dyn Filter<f32>>>,
}

/// Receive pipeline: zero or more IQ filters followed by a demodulator.
///
/// Filters are applied to the incoming IQ in the order they were added,
/// before the demodulator sees the data.
pub struct RxPipeline<D> {
    iq_filters: Vec<Box<dyn Filter<f32>>>,
    demodulator: D,
}

impl<M: Modulate> TxPipeline<M> {
    /// Create a pipeline wrapping `modulator` with no IQ filters.
    pub fn new(modulator: M) -> Self {
        Self {
            modulator,
            iq_filters: Vec::new(),
        }
    }

    /// Append a [`RationalResampler`] with the given interpolation and
    /// decimation factors.
    pub fn with_resampler(mut self, interp: usize, decim: usize) -> Self {
        self.iq_filters
            .push(Box::new(RationalResampler::new(interp, decim)));
        self
    }

    /// Append a [`FreqShift`] that shifts IQ by `shift_hz`.
    pub fn with_freq_shift(mut self, sample_rate: f32, shift_hz: f32) -> Self {
        self.iq_filters
            .push(Box::new(FreqShift::new(sample_rate, shift_hz)));
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
    ///                   gives a maximally-flat Butterworth response
    pub fn with_lowpass(self, sample_rate: f64, cutoff_hz: f64, q: f64) -> Self {
        self.with_filter(Iir::lowpass(sample_rate, cutoff_hz, q))
    }

    /// Append a resampler that converts from `in_rate` to `out_rate` Hz,
    /// reducing the ratio by their GCD first.
    pub fn with_rates(self, in_rate: u32, out_rate: u32) -> Self {
        let g = gcd(in_rate, out_rate);
        self.with_resampler((out_rate / g) as usize, (in_rate / g) as usize)
    }
}

impl<D: Demodulate> RxPipeline<D> {
    /// Create a pipeline wrapping `demodulator` with no IQ filters.
    pub fn new(demodulator: D) -> Self {
        Self {
            iq_filters: Vec::new(),
            demodulator,
        }
    }

    /// Append a [`RationalResampler`] with the given interpolation and
    /// decimation factors.
    pub fn with_resampler(mut self, interp: usize, decim: usize) -> Self {
        self.iq_filters
            .push(Box::new(RationalResampler::new(interp, decim)));
        self
    }

    /// Append a [`FreqShift`] that shifts IQ by `shift_hz`.
    pub fn with_freq_shift(mut self, sample_rate: f32, shift_hz: f32) -> Self {
        self.iq_filters
            .push(Box::new(FreqShift::new(sample_rate, shift_hz)));
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
    ///                   gives a maximally-flat Butterworth response
    pub fn with_lowpass(self, sample_rate: f64, cutoff_hz: f64, q: f64) -> Self {
        self.with_filter(Iir::lowpass(sample_rate, cutoff_hz, q))
    }

    /// Append a resampler that converts from `in_rate` to `out_rate` Hz,
    /// reducing the ratio by their GCD first.
    pub fn with_rates(self, in_rate: u32, out_rate: u32) -> Self {
        let g = gcd(in_rate, out_rate);
        self.with_resampler((out_rate / g) as usize, (in_rate / g) as usize)
    }
}

impl<M: Modulate> Transmitter for TxPipeline<M> {
    fn transmit(&mut self, audio: &[f32]) -> Vec<Complex<f32>> {
        let iq = self.modulator.modulate(audio);
        self.iq_filters.iter_mut().fold(iq, |acc, f| f.filter(&acc))
    }
}

impl<D: Demodulate> Receiver for RxPipeline<D> {
    fn receive(&mut self, iq: &[Complex<f32>]) -> Vec<f32> {
        let iq = self
            .iq_filters
            .iter_mut()
            .fold(iq.to_vec(), |acc, f| f.filter(&acc));
        self.demodulator.demodulate(&iq)
    }
}

#[cfg(test)]
mod tests {
    use super::gcd;

    #[test]
    fn gcd_correct() {
        let g = gcd(250_000, 48_000);
        assert_eq!(g, 2_000);
        assert_eq!(250_000 / g, 125);
        assert_eq!(48_000 / g, 24);
    }
}
