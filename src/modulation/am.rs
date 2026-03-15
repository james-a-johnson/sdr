//! AM (amplitude modulation) modulator and demodulator.
//!
//! Both types operate on `f32` samples. The modulator encodes a real-valued
//! baseband signal into IQ samples using DSB-LC (double-sideband large-carrier)
//! at baseband; the demodulator inverts the process using envelope detection.

use crate::complex::Complex;

/// AM modulator: encodes a real-valued baseband signal as an IQ waveform.
///
/// Uses DSB-LC modulation at baseband (carrier at DC):
///
/// ```text
/// y[n].i = 1.0 + depth * x[n]
/// y[n].q = 0.0
/// ```
///
/// where `x[n]` is the normalized baseband sample and `depth` is the
/// modulation depth in `(0.0, 1.0]`. Pair with [`FreqShift`] to place the
/// carrier at a non-zero frequency.
///
/// [`FreqShift`]: crate::filter::FreqShift
///
/// # Example
///
/// ```
/// use sdr::modulation::am::AmModulator;
/// use std::f32::consts::PI;
///
/// let mut mod_ = AmModulator::new(1.0);
/// let baseband: Vec<f32> = (0..1024)
///     .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48_000.0).sin())
///     .collect();
/// let iq = mod_.modulate(&baseband);
/// assert_eq!(iq.len(), 1024);
/// ```
pub struct AmModulator {
    /// Modulation depth in `(0.0, 1.0]`; `1.0` corresponds to 100% modulation.
    modulation_depth: f32,
}

impl AmModulator {
    /// Create a new AM modulator.
    ///
    /// * `modulation_depth` — modulation depth in `(0.0, 1.0]`; a value of
    ///   `1.0` means 100% modulation (the carrier amplitude reaches zero at the
    ///   negative peaks of a full-scale input signal)
    pub fn new(modulation_depth: f32) -> Self {
        Self { modulation_depth }
    }

    /// Return the modulation depth.
    pub fn modulation_depth(&self) -> f32 {
        self.modulation_depth
    }

    /// Modulate a slice of baseband samples into IQ output.
    ///
    /// Returns one complex sample per input sample. The Q component is always
    /// `0.0`; the I component encodes the carrier plus the scaled baseband
    /// signal: `1.0 + depth * x[n]`.
    pub fn modulate(&mut self, baseband: &[f32]) -> Vec<Complex<f32>> {
        baseband
            .iter()
            .map(|&x| Complex::new(1.0 + self.modulation_depth * x, 0.0))
            .collect()
    }
}

/// AM demodulator: recovers a real-valued baseband signal from IQ samples.
///
/// Uses envelope detection (magnitude extraction followed by DC removal):
///
/// ```text
/// envelope[n] = |iq[n]|
/// out[n]      = (envelope[n] − 1.0) / depth
/// ```
///
/// Assumes unit carrier amplitude and no over-modulation
/// (`depth * x[n] > −1` for all `n`).
///
/// # Example
///
/// ```
/// use sdr::modulation::am::{AmModulator, AmDemodulator};
/// use std::f32::consts::PI;
///
/// let depth = 1.0_f32;
/// let baseband: Vec<f32> = (0..1024)
///     .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48_000.0).sin())
///     .collect();
///
/// let iq = AmModulator::new(depth).modulate(&baseband);
/// let recovered = AmDemodulator::new(depth).demodulate(&iq);
/// for (orig, rec) in baseband.iter().zip(recovered.iter()) {
///     assert!((orig - rec).abs() < 1e-5);
/// }
/// ```
pub struct AmDemodulator {
    /// Modulation depth in `(0.0, 1.0]`, matching the modulator's setting.
    modulation_depth: f32,
}

impl AmDemodulator {
    /// Create a new AM demodulator.
    ///
    /// * `modulation_depth` — modulation depth in `(0.0, 1.0]`; must match
    ///   the modulator's setting
    pub fn new(modulation_depth: f32) -> Self {
        Self { modulation_depth }
    }

    /// Return the modulation depth.
    pub fn modulation_depth(&self) -> f32 {
        self.modulation_depth
    }

    /// Demodulate a slice of IQ samples into a real baseband signal.
    ///
    /// Returns one `f32` sample per input IQ sample using envelope detection:
    /// `(|iq[n]| − 1.0) / depth`.
    pub fn demodulate(&mut self, iq: &[Complex<f32>]) -> Vec<f32> {
        iq.iter()
            .map(|c| (c.abs() - 1.0) / self.modulation_depth)
            .collect()
    }
}

impl crate::pipeline::Modulate for AmModulator {
    fn modulate(&mut self, audio: &[f32]) -> Vec<crate::complex::Complex<f32>> {
        self.modulate(audio)
    }
}

impl crate::pipeline::Demodulate for AmDemodulator {
    fn demodulate(&mut self, iq: &[crate::complex::Complex<f32>]) -> Vec<f32> {
        self.demodulate(iq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn roundtrip() {
        let depth = 0.8_f32;
        let n = 1024;
        let baseband: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48_000.0).sin())
            .collect();

        let iq = AmModulator::new(depth).modulate(&baseband);
        let recovered = AmDemodulator::new(depth).demodulate(&iq);

        for (orig, rec) in baseband.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 1e-5,
                "Roundtrip mismatch: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn dc_input() {
        let depth = 0.5_f32;
        let value = 0.3_f32;
        let baseband = vec![value; 64];

        let iq = AmModulator::new(depth).modulate(&baseband);
        let expected_envelope = 1.0 + depth * value;
        for c in &iq {
            assert!((c.abs() - expected_envelope).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_depth_edge() {
        let baseband: Vec<f32> = (0..64)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48_000.0).sin())
            .collect();

        let iq = AmModulator::new(0.0).modulate(&baseband);
        for c in &iq {
            assert!((c.i - 1.0).abs() < 1e-6, "Expected i == 1.0, got {}", c.i);
            assert!(c.q.abs() < 1e-6, "Expected q == 0.0, got {}", c.q);
        }
    }

    #[test]
    fn full_modulation_sine() {
        let depth = 1.0_f32;
        let n = 256;
        let baseband: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48_000.0).sin())
            .collect();

        let iq = AmModulator::new(depth).modulate(&baseband);
        for (i, (&x, c)) in baseband.iter().zip(iq.iter()).enumerate() {
            let expected_i = 1.0 + x;
            assert!(
                (c.i - expected_i).abs() < 1e-6,
                "Sample {}: expected i = {}, got {}",
                i,
                expected_i,
                c.i
            );
        }
    }

    #[test]
    fn q_always_zero() {
        let baseband: Vec<f32> = (0..256)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48_000.0).sin())
            .collect();

        let iq = AmModulator::new(0.9).modulate(&baseband);
        for (i, c) in iq.iter().enumerate() {
            assert!(
                c.q.abs() < 1e-6,
                "Sample {}: expected q == 0.0, got {}",
                i,
                c.q
            );
        }
    }
}
