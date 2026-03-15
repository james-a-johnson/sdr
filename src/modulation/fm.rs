//! FM (frequency modulation) modulator and demodulator.
//!
//! Both types operate on `f32` samples. The modulator encodes a real-valued
//! baseband signal into IQ samples; the demodulator inverts the process using
//! a conjugate-multiply discriminator.

use crate::complex::Complex;
use std::f32::consts::PI;

/// FM modulator: encodes a real-valued baseband signal as an IQ waveform.
///
/// Uses a phase accumulator to integrate the instantaneous frequency deviation:
///
/// ```text
/// φ[n] = φ[n−1] + 2π·deviation·x[n] / sample_rate   (mod 2π)
/// y[n] = cos(φ[n]) + j·sin(φ[n])
/// ```
///
/// where `x[n]` is the normalized baseband sample (expected range `[−1, 1]`).
/// The phase is wrapped via `rem_euclid(2π)` every sample to prevent `f32`
/// precision loss over long streams.
///
/// # Example
///
/// ```
/// use sdr::modulation::fm::FmModulator;
/// use std::f32::consts::PI;
///
/// let mut mod_ = FmModulator::new(48_000.0, 5_000.0);
/// let baseband: Vec<f32> = (0..1024)
///     .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48_000.0).sin())
///     .collect();
/// let iq = mod_.modulate(&baseband);
/// assert_eq!(iq.len(), 1024);
/// ```
pub struct FmModulator {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Maximum frequency deviation in Hz (corresponds to a baseband amplitude of 1).
    pub deviation: f32,
    phase: f32,
    phase_increment: f32,
}

impl FmModulator {
    /// Create a new FM modulator.
    ///
    /// * `sample_rate` — sample rate in Hz
    /// * `deviation`   — maximum frequency deviation in Hz; a baseband sample
    ///   of `±1.0` produces a carrier offset of `±deviation` Hz
    pub fn new(sample_rate: f32, deviation: f32) -> Self {
        let phase_increment = 2.0 * PI * deviation / sample_rate;
        Self {
            sample_rate,
            deviation,
            phase: 0.0,
            phase_increment,
        }
    }

    /// Return the current phase accumulator value in radians, in `[0, 2π)`.
    pub fn phase(&self) -> f32 {
        self.phase
    }

    /// Set the phase accumulator to `phase` radians.
    ///
    /// The value is wrapped into `[0, 2π)` via `rem_euclid`. Use this to
    /// align the modulator's carrier phase with an external reference or to
    /// resume modulation from a known starting phase.
    pub fn set_phase(&mut self, phase: f32) {
        self.phase = phase.rem_euclid(2.0 * PI);
    }

    /// Modulate a slice of baseband samples into IQ output.
    ///
    /// Returns one complex sample per input sample. The phase accumulator is
    /// updated in place so successive calls to `modulate` produce a
    /// phase-continuous output stream.
    pub fn modulate(&mut self, baseband: &[f32]) -> Vec<Complex<f32>> {
        baseband
            .iter()
            .map(|&sample| {
                self.phase = (self.phase + self.phase_increment * sample).rem_euclid(2.0 * PI);
                Complex::from_polar(1.0, self.phase)
            })
            .collect()
    }
}

/// FM demodulator: recovers a real-valued baseband signal from IQ samples.
///
/// Uses a conjugate-multiply (differential) discriminator:
///
/// ```text
/// z[n]    = iq[n] · conj(iq[n−1])
/// out[n]  = atan2(z.q, z.i) · scale
/// scale   = sample_rate / (2π·deviation)
/// ```
///
/// `atan2` extracts the instantaneous phase difference between consecutive
/// samples, which is proportional to the instantaneous frequency. Multiplying
/// by `scale` maps the result back to the original baseband amplitude range.
///
/// # Example
///
/// ```
/// use sdr::modulation::fm::{FmModulator, FmDemodulator};
/// use std::f32::consts::PI;
///
/// let sample_rate = 48_000.0_f32;
/// let deviation = 5_000.0_f32;
/// let baseband: Vec<f32> = (0..1024)
///     .map(|i| (2.0 * PI * 1000.0 * i as f32 / sample_rate).sin())
///     .collect();
///
/// let iq = FmModulator::new(sample_rate, deviation).modulate(&baseband);
/// let recovered = FmDemodulator::new(sample_rate, deviation).demodulate(&iq);
/// // After a short transient the recovered signal matches the original
/// assert!((baseband[128] - recovered[128]).abs() < 0.05);
/// ```
pub struct FmDemodulator {
    /// Sample rate in Hz.
    pub sample_rate: f32,
    /// Maximum frequency deviation in Hz, matching the modulator's setting.
    pub deviation: f32,
    /// Previous IQ sample, used to compute the conjugate product.
    prev: Complex<f32>,
    /// Precomputed scale factor: `sample_rate / (2π·deviation)`.
    scale: f32,
}

impl FmDemodulator {
    /// Create a new FM demodulator.
    ///
    /// * `sample_rate` — sample rate in Hz; must match the modulator
    /// * `deviation`   — maximum frequency deviation in Hz; must match the
    ///   modulator
    pub fn new(sample_rate: f32, deviation: f32) -> Self {
        let scale = sample_rate / (2.0 * PI * deviation);
        Self {
            sample_rate,
            deviation,
            prev: Complex::new(1.0, 0.0),
            scale,
        }
    }

    /// Set the reference phase used by the discriminator.
    ///
    /// The demodulator tracks phase via a `prev` sample rather than a scalar
    /// angle. `set_phase` constructs the equivalent unit-magnitude phasor
    /// `(cos(phase), sin(phase))` and stores it as the new reference, so the
    /// next output sample is computed relative to that angle.
    ///
    /// The value is wrapped into `[0, 2π)` via `rem_euclid`.
    pub fn set_phase(&mut self, phase: f32) {
        let p = phase.rem_euclid(2.0 * PI);
        self.prev = Complex::from_polar(1.0, p);
    }

    /// Demodulate a slice of IQ samples into a real baseband signal.
    ///
    /// Returns one `f32` sample per input IQ sample. The `prev` register is
    /// updated in place so successive calls produce a continuous output stream.
    pub fn demodulate(&mut self, iq: &[Complex<f32>]) -> Vec<f32> {
        iq.iter()
            .map(|&sample| {
                let z = sample * self.prev.conjugate();
                let freq = z.arg() * self.scale;
                self.prev = sample;
                freq
            })
            .collect()
    }
}

impl crate::pipeline::Modulate for FmModulator {
    fn modulate(&mut self, audio: &[f32]) -> Vec<crate::complex::Complex<f32>> {
        self.modulate(audio)
    }
}

impl crate::pipeline::Demodulate for FmDemodulator {
    fn demodulate(&mut self, iq: &[crate::complex::Complex<f32>]) -> Vec<f32> {
        self.demodulate(iq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulate_demodulate_roundtrip() {
        let sample_rate = 48000.0_f32;
        let deviation = 5000.0_f32;
        let freq = 1000.0_f32;
        let n = 1024;

        let baseband: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect();

        let mut modulator = FmModulator::new(sample_rate, deviation);
        let iq = modulator.modulate(&baseband);

        let mut demodulator = FmDemodulator::new(sample_rate, deviation);
        let recovered = demodulator.demodulate(&iq);

        // Skip transient at start, compare steady-state portion
        let start = 64;
        for (orig, rec) in baseband[start..].iter().zip(recovered[start..].iter()) {
            assert!(
                (orig - rec).abs() < 0.05,
                "Roundtrip mismatch: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn phase_wrapping_no_nan() {
        let mut modulator = FmModulator::new(48000.0, 5000.0);
        let baseband: Vec<f32> = (0..1_000_000)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48000.0).sin())
            .collect();
        let iq = modulator.modulate(&baseband);
        for (i, c) in iq.iter().enumerate() {
            assert!(!c.i.is_nan(), "NaN at sample {}", i);
            assert!(!c.q.is_nan(), "NaN at sample {}", i);
        }
    }
}
