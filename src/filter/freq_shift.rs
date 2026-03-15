use super::Filter;
use crate::complex::Complex;
use std::f32::consts::PI;

/// Frequency-shift filter (mixer / heterodyne).
///
/// Multiplies each input sample by the rotating phasor `e^(j·2π·shift·n/fs)`,
/// translating the entire spectrum by `shift_hz`. A positive shift moves
/// energy upward in frequency; negative shifts it down.
///
/// The phase accumulator persists across `filter()` calls so the filter
/// is safe to use on a continuous stream of blocks.
pub struct FreqShift {
    phase: f32,
    phase_step: f32,
}

impl FreqShift {
    /// Create a new frequency-shift filter.
    ///
    /// * `sample_rate` — sample rate of the IQ stream in Hz
    /// * `shift_hz`    — frequency offset to apply; positive = shift up
    pub fn new(sample_rate: f32, shift_hz: f32) -> Self {
        Self {
            phase: 0.0,
            phase_step: 2.0 * PI * shift_hz / sample_rate,
        }
    }

    /// Zero the phase accumulator (useful when reusing the filter on a new stream).
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Current phase accumulator value in radians.
    pub fn phase(&self) -> f32 {
        self.phase
    }
}

impl Filter<f32> for FreqShift {
    fn filter(&mut self, data: &[Complex<f32>]) -> Vec<Complex<f32>> {
        data.iter()
            .map(|&s| {
                let mixer = Complex::new(self.phase.cos(), self.phase.sin());
                self.phase = (self.phase + self.phase_step).rem_euclid(2.0 * PI);
                s * mixer
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// A pure tone at `freq_hz` produces a complex exponential; after shifting
    /// by `-freq_hz` it should collapse to a DC component (0 Hz).
    #[test]
    fn shift_tone_to_dc() {
        let sample_rate = 48_000.0_f32;
        let tone_hz = 5_000.0_f32;
        let n = 1024_usize;

        // Generate a complex tone at tone_hz
        let samples: Vec<Complex<f32>> = (0..n)
            .map(|i| {
                let phase = 2.0 * PI * tone_hz * i as f32 / sample_rate;
                Complex::new(phase.cos(), phase.sin())
            })
            .collect();

        // Shift down by tone_hz — should produce a DC (0 Hz) signal
        let mut shift = FreqShift::new(sample_rate, -tone_hz);
        let shifted = shift.filter(&samples);

        // All samples should be approximately (1, 0)
        for (i, c) in shifted.iter().enumerate() {
            assert!(
                (c.i - 1.0).abs() < 2e-4,
                "sample {i}: I={} expected ~1.0",
                c.i
            );
            assert!(c.q.abs() < 2e-4, "sample {i}: Q={} expected ~0.0", c.q);
        }
    }

    /// Shifting up by `f` then down by `f` should recover the original signal.
    #[test]
    fn shift_roundtrip() {
        let sample_rate = 48_000.0_f32;
        let n = 512_usize;

        let samples: Vec<Complex<f32>> = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate;
                Complex::new((2.0 * PI * 1000.0 * t).sin(), (2.0 * PI * 3000.0 * t).cos())
            })
            .collect();

        let mut up = FreqShift::new(sample_rate, 8_000.0);
        let mut down = FreqShift::new(sample_rate, -8_000.0);

        let shifted = up.filter(&samples);
        let recovered = down.filter(&shifted);

        for (i, (orig, rec)) in samples.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig.i - rec.i).abs() < 1e-4,
                "sample {i}: I orig={} rec={}",
                orig.i,
                rec.i
            );
            assert!(
                (orig.q - rec.q).abs() < 1e-4,
                "sample {i}: Q orig={} rec={}",
                orig.q,
                rec.q
            );
        }
    }

    /// Phase accumulator must be continuous across block boundaries.
    #[test]
    fn state_persists_across_calls() {
        let sample_rate = 48_000.0_f32;
        let n = 256_usize;
        let samples: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new((i as f32 * 0.1).sin(), 0.0))
            .collect();

        let mut filter_a = FreqShift::new(sample_rate, 2_000.0);
        let mut filter_b = FreqShift::new(sample_rate, 2_000.0);

        let full = filter_a.filter(&samples);
        let first_half = filter_b.filter(&samples[..n / 2]);
        let second_half = filter_b.filter(&samples[n / 2..]);

        let split: Vec<_> = first_half
            .iter()
            .chain(second_half.iter())
            .cloned()
            .collect();

        for (i, (a, b)) in full.iter().zip(split.iter()).enumerate() {
            assert!(
                (a.i - b.i).abs() < 1e-6,
                "sample {i}: I full={} split={}",
                a.i,
                b.i
            );
            assert!(
                (a.q - b.q).abs() < 1e-6,
                "sample {i}: Q full={} split={}",
                a.q,
                b.q
            );
        }
    }
}
