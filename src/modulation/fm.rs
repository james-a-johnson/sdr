use crate::complex::Complex;
use std::f32::consts::PI;

pub struct FmModulator {
    pub sample_rate: f32,
    pub deviation: f32,
    phase: f32,
    phase_increment: f32,
}

impl FmModulator {
    pub fn new(sample_rate: f32, deviation: f32) -> Self {
        let phase_increment = 2.0 * PI * deviation / sample_rate;
        Self {
            sample_rate,
            deviation,
            phase: 0.0,
            phase_increment,
        }
    }

    pub fn modulate(&mut self, baseband: &[f32]) -> Vec<Complex<f32>> {
        baseband
            .iter()
            .map(|&sample| {
                self.phase = (self.phase + self.phase_increment * sample).rem_euclid(2.0 * PI);
                Complex::new(self.phase.cos(), self.phase.sin())
            })
            .collect()
    }
}

pub struct FmDemodulator {
    pub sample_rate: f32,
    pub deviation: f32,
    prev: Complex<f32>,
    scale: f32,
}

impl FmDemodulator {
    pub fn new(sample_rate: f32, deviation: f32) -> Self {
        let scale = sample_rate / (2.0 * PI * deviation);
        Self {
            sample_rate,
            deviation,
            prev: Complex::new(1.0, 0.0),
            scale,
        }
    }

    pub fn demodulate(&mut self, iq: &[Complex<f32>]) -> Vec<f32> {
        iq.iter()
            .map(|&sample| {
                let z = sample * self.prev.conjugate();
                let freq = z.q.atan2(z.i) * self.scale;
                self.prev = sample;
                freq
            })
            .collect()
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
