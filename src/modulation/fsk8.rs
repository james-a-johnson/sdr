//! 8-FSK (8-tone Frequency Shift Keying) modulator and demodulator.
//!
//! Both types implement CPFSK (continuous-phase FSK): phase is accumulated
//! across calls so chunk boundaries produce no phase discontinuity.
//!
//! Symbol values are integers `0..=7`. Symbol `k` is transmitted at
//! `k * tone_spacing_hz` above the baseband centre (DC = symbol 0).
//! Pair with [`FreqShift`] to place the signal at any carrier frequency.
//!
//! **FT8 defaults**: `sample_rate = 12_000.0`, `symbol_rate = 6.25`,
//! `tone_spacing_hz = 6.25`.
//!
//! [`FreqShift`]: crate::filter::FreqShift

use crate::complex::Complex;
use std::f32::consts::TAU;

/// 8-FSK modulator: encodes integer symbols `0..=7` as a CPFSK IQ waveform.
///
/// Each symbol `k` is transmitted at frequency `k * tone_spacing_hz`. The
/// phase accumulator persists across calls, guaranteeing phase continuity even
/// when the input is delivered in chunks.
///
/// # Example
///
/// ```
/// use sdr::modulation::fsk8::Fsk8Modulator;
///
/// let mut tx = Fsk8Modulator::new(12_000.0, 6.25, 6.25);
/// let iq = tx.modulate(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
/// assert_eq!(iq.len(), 8 * tx.samples_per_symbol());
/// ```
pub struct Fsk8Modulator {
    sample_rate: f32,
    tone_spacing_hz: f32,
    samples_per_symbol: usize,
    phase: f32,
}

impl Fsk8Modulator {
    /// Create a new 8-FSK modulator.
    ///
    /// * `sample_rate`      — sample rate in Hz
    /// * `symbol_rate`      — symbol rate in symbols/s
    /// * `tone_spacing_hz`  — frequency spacing between adjacent tones in Hz
    ///
    /// # Panics
    ///
    /// Panics if `symbol_rate >= sample_rate` (zero samples per symbol) or if
    /// the highest tone (`7 * tone_spacing_hz`) meets or exceeds the Nyquist
    /// frequency (`sample_rate / 2`).
    pub fn new(sample_rate: f32, symbol_rate: f32, tone_spacing_hz: f32) -> Self {
        let samples_per_symbol = (sample_rate / symbol_rate).round() as usize;
        assert!(
            samples_per_symbol > 0,
            "symbol_rate ({symbol_rate}) must be less than sample_rate ({sample_rate})"
        );
        assert!(
            7.0 * tone_spacing_hz < sample_rate / 2.0,
            "highest tone ({} Hz) must be below Nyquist ({} Hz)",
            7.0 * tone_spacing_hz,
            sample_rate / 2.0,
        );
        let _ = symbol_rate; // consumed only for samples_per_symbol calculation
        Self {
            sample_rate,
            tone_spacing_hz,
            samples_per_symbol,
            phase: 0.0,
        }
    }

    /// Return the current phase accumulator value in radians, in `[0, 2π)`.
    pub fn phase(&self) -> f32 {
        self.phase
    }

    /// Set the phase accumulator.
    ///
    /// The value is wrapped into `[0, 2π)` via `rem_euclid`.
    pub fn set_phase(&mut self, phase: f32) {
        self.phase = phase.rem_euclid(TAU);
    }

    /// Number of IQ samples produced per symbol.
    pub fn samples_per_symbol(&self) -> usize {
        self.samples_per_symbol
    }

    /// Modulate a slice of symbols into IQ output.
    ///
    /// Each element should be a symbol value in `[0.0, 7.0]`; non-integer
    /// values are rounded and clamped. Returns
    /// `symbols.len() * samples_per_symbol` complex samples.
    pub fn modulate(&mut self, symbols: &[f32]) -> Vec<Complex<f32>> {
        let mut out = Vec::with_capacity(symbols.len() * self.samples_per_symbol);
        for &sym in symbols {
            let k = sym.round().clamp(0.0, 7.0) as usize;
            let freq_hz = k as f32 * self.tone_spacing_hz;
            let phase_inc = TAU * freq_hz / self.sample_rate;
            for _ in 0..self.samples_per_symbol {
                self.phase = (self.phase + phase_inc).rem_euclid(TAU);
                out.push(Complex::from_polar(1.0, self.phase));
            }
        }
        out
    }
}

/// 8-FSK demodulator: recovers integer symbols from a CPFSK IQ stream.
///
/// Uses an FM discriminator (conjugate-multiply) followed by a symbol-rate
/// integrate-and-dump filter. State is preserved across calls so the
/// demodulator can process streaming IQ without sample-boundary alignment.
///
/// Output length is `floor(total_accumulated_samples / samples_per_symbol)`.
/// Any partial window at the end of a call is carried over to the next call.
///
/// # Example
///
/// ```
/// use sdr::modulation::fsk8::{Fsk8Modulator, Fsk8Demodulator};
///
/// let symbols: Vec<f32> = (0..8).map(|k| k as f32).collect();
/// let iq = Fsk8Modulator::new(12_000.0, 6.25, 6.25).modulate(&symbols);
/// let recovered = Fsk8Demodulator::new(12_000.0, 6.25, 6.25).demodulate(&iq);
/// assert_eq!(recovered, symbols);
/// ```
pub struct Fsk8Demodulator {
    sample_rate: f32,
    tone_spacing_hz: f32,
    samples_per_symbol: usize,
    prev: Complex<f32>,
    sample_count: usize,
    freq_accum: f32,
}

impl Fsk8Demodulator {
    /// Create a new 8-FSK demodulator.
    ///
    /// Parameters must match the corresponding [`Fsk8Modulator`].
    ///
    /// # Panics
    ///
    /// Same conditions as [`Fsk8Modulator::new`].
    pub fn new(sample_rate: f32, symbol_rate: f32, tone_spacing_hz: f32) -> Self {
        let samples_per_symbol = (sample_rate / symbol_rate).round() as usize;
        assert!(
            samples_per_symbol > 0,
            "symbol_rate ({symbol_rate}) must be less than sample_rate ({sample_rate})"
        );
        assert!(
            7.0 * tone_spacing_hz < sample_rate / 2.0,
            "highest tone ({} Hz) must be below Nyquist ({} Hz)",
            7.0 * tone_spacing_hz,
            sample_rate / 2.0,
        );
        let _ = symbol_rate; // consumed only for samples_per_symbol calculation
        Self {
            sample_rate,
            tone_spacing_hz,
            samples_per_symbol,
            prev: Complex::new(1.0, 0.0),
            sample_count: 0,
            freq_accum: 0.0,
        }
    }

    /// Reset the demodulator state (previous sample, accumulator, counter).
    pub fn reset(&mut self) {
        self.prev = Complex::new(1.0, 0.0);
        self.sample_count = 0;
        self.freq_accum = 0.0;
    }

    /// Demodulate a slice of IQ samples into recovered symbols.
    ///
    /// Returns one `f32` symbol (rounded and clamped to `[0.0, 7.0]`) per
    /// complete symbol window. Partial windows are held in state and completed
    /// on the next call.
    pub fn demodulate(&mut self, iq: &[Complex<f32>]) -> Vec<f32> {
        let mut out = Vec::new();
        for &cur in iq {
            let z = cur * self.prev.conjugate();
            let inst_phase_diff = z.arg();
            self.prev = cur;
            self.freq_accum += inst_phase_diff;
            self.sample_count += 1;

            if self.sample_count == self.samples_per_symbol {
                let avg_freq_hz =
                    (self.freq_accum / self.samples_per_symbol as f32) * self.sample_rate / TAU;
                let symbol = (avg_freq_hz / self.tone_spacing_hz)
                    .round()
                    .clamp(0.0, 7.0);
                out.push(symbol);
                self.freq_accum = 0.0;
                self.sample_count = 0;
            }
        }
        out
    }
}

impl crate::pipeline::Modulate for Fsk8Modulator {
    fn modulate(&mut self, symbols: &[f32]) -> Vec<crate::complex::Complex<f32>> {
        self.modulate(symbols)
    }
}

impl crate::pipeline::Demodulate for Fsk8Demodulator {
    fn demodulate(&mut self, iq: &[crate::complex::Complex<f32>]) -> Vec<f32> {
        self.demodulate(iq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: f32 = 1_000.0;
    const SYM_RATE: f32 = 10.0;
    const SPACING: f32 = 10.0;

    fn modulator() -> Fsk8Modulator {
        Fsk8Modulator::new(SR, SYM_RATE, SPACING)
    }

    fn demodulator() -> Fsk8Demodulator {
        Fsk8Demodulator::new(SR, SYM_RATE, SPACING)
    }

    #[test]
    fn output_length() {
        let mut tx = modulator();
        let sps = tx.samples_per_symbol();
        let n = 13;
        let symbols: Vec<f32> = (0..n).map(|i| (i % 8) as f32).collect();
        let iq = tx.modulate(&symbols);
        assert_eq!(iq.len(), n * sps);
    }

    #[test]
    fn unit_amplitude() {
        let mut tx = modulator();
        let symbols: Vec<f32> = (0..8).map(|k| k as f32).collect();
        let iq = tx.modulate(&symbols);
        for (i, c) in iq.iter().enumerate() {
            assert!(
                (c.abs() - 1.0).abs() < 1e-5,
                "sample {i}: amplitude = {}",
                c.abs()
            );
        }
    }

    #[test]
    fn phase_continuity() {
        let mut tx_split = modulator();
        let iq_a = tx_split.modulate(&[3.0]);
        let iq_b = tx_split.modulate(&[5.0]);
        let split: Vec<_> = iq_a.into_iter().chain(iq_b).collect();

        let mut tx_single = modulator();
        let single = tx_single.modulate(&[3.0, 5.0]);

        assert_eq!(split.len(), single.len());
        for (i, (a, b)) in split.iter().zip(single.iter()).enumerate() {
            assert!(
                (a.i - b.i).abs() < 1e-5 && (a.q - b.q).abs() < 1e-5,
                "sample {i}: split = ({}, {}), single = ({}, {})",
                a.i,
                a.q,
                b.i,
                b.q
            );
        }
    }

    #[test]
    fn tone_frequency_correctness() {
        // sample_rate=1000, tone_spacing=10 → symbol k at k*10 Hz
        // expected phase_inc per sample = 2π * k * 10 / 1000
        let mut tx = modulator();
        let sps = tx.samples_per_symbol();

        for k in 0u8..8 {
            let mut single_tx = modulator();
            let iq = single_tx.modulate(&[k as f32]);
            // average phase increment over the symbol window
            let mut prev = Complex::new(1.0_f32, 0.0);
            let mut phase_sum = 0.0_f32;
            for &c in &iq {
                let z = c * prev.conjugate();
                phase_sum += z.arg();
                prev = c;
            }
            let avg_inc = phase_sum / sps as f32;
            let expected_inc = TAU * k as f32 * SPACING / SR;
            assert!(
                (avg_inc - expected_inc).abs() < 1e-4,
                "symbol {k}: avg_inc={avg_inc:.6}, expected={expected_inc:.6}"
            );
            drop(tx); // suppress unused warning
            tx = modulator();
        }
    }

    #[test]
    fn symbol_roundtrip() {
        let mut tx = modulator();
        let mut rx = demodulator();
        let symbols: Vec<f32> = (0..8).map(|k| k as f32).collect();
        let iq = tx.modulate(&symbols);
        let recovered = rx.demodulate(&iq);
        assert_eq!(recovered, symbols, "roundtrip mismatch");
    }

    #[test]
    fn streaming_demodulator() {
        let symbols: Vec<f32> = vec![0.0, 3.0, 7.0, 1.0, 5.0];
        let mut tx = modulator();
        let iq = tx.modulate(&symbols);
        let sps = tx.samples_per_symbol();

        // All at once
        let mut rx_all = demodulator();
        let all_at_once = rx_all.demodulate(&iq);

        // One symbol at a time
        let mut rx_sym = demodulator();
        let mut one_by_one: Vec<f32> = Vec::new();
        for chunk in iq.chunks(sps) {
            one_by_one.extend(rx_sym.demodulate(chunk));
        }

        assert_eq!(all_at_once, one_by_one, "all-at-once vs one-symbol-at-a-time");

        // Mid-symbol split (split within symbol boundary)
        let split_at = sps / 2;
        let mut rx_mid = demodulator();
        let mut mid_split: Vec<f32> = Vec::new();
        mid_split.extend(rx_mid.demodulate(&iq[..split_at]));
        mid_split.extend(rx_mid.demodulate(&iq[split_at..]));

        assert_eq!(all_at_once, mid_split, "all-at-once vs mid-symbol split");
    }

    #[test]
    fn phase_wrapping_no_nan() {
        let mut tx = Fsk8Modulator::new(12_000.0, 6.25, 6.25);
        let symbols = vec![7.0_f32; 100_000];
        let iq = tx.modulate(&symbols);
        for (i, c) in iq.iter().enumerate() {
            assert!(!c.i.is_nan(), "NaN i at sample {i}");
            assert!(!c.q.is_nan(), "NaN q at sample {i}");
            assert!(!c.i.is_infinite(), "Inf i at sample {i}");
            assert!(!c.q.is_infinite(), "Inf q at sample {i}");
        }
    }

    #[test]
    fn demodulator_clamp() {
        // Synthesise a tone above 7 * tone_spacing (e.g. symbol 9) and verify
        // the demodulator clamps the output to 7.0.
        let sps = (SR / SYM_RATE).round() as usize;
        let freq_hz = 9.0 * SPACING; // above the valid range
        let phase_inc = TAU * freq_hz / SR;
        let mut phase = 0.0_f32;
        let iq: Vec<Complex<f32>> = (0..sps)
            .map(|_| {
                phase = (phase + phase_inc).rem_euclid(TAU);
                Complex::from_polar(1.0, phase)
            })
            .collect();

        let mut rx = demodulator();
        let out = rx.demodulate(&iq);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 7.0, "expected clamp to 7.0, got {}", out[0]);
    }
}
