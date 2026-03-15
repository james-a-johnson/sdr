use super::Filter;
use crate::complex::Complex;
use num_traits::Float;

fn f<T: Float>(x: f64) -> T {
    T::from(x).unwrap()
}

pub struct Iir<T> {
    b0: T,
    b1: T,
    b2: T,
    a1: T,
    a2: T,
    w1_i: T,
    w2_i: T,
    w1_q: T,
    w2_q: T,
}

impl<T: Float + Copy> Iir<T> {
    pub fn new(b0: T, b1: T, b2: T, a1: T, a2: T) -> Self {
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            w1_i: f(0.0),
            w2_i: f(0.0),
            w1_q: f(0.0),
            w2_q: f(0.0),
        }
    }

    pub fn lowpass(sample_rate: f64, cutoff: f64, q: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * cutoff / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;
        let b0 = (1.0 - cos_w0) / 2.0 / a0;
        let b1 = (1.0 - cos_w0) / a0;
        let b2 = (1.0 - cos_w0) / 2.0 / a0;
        let a1 = -2.0 * cos_w0 / a0;
        let a2 = (1.0 - alpha) / a0;
        Self::new(f(b0), f(b1), f(b2), f(a1), f(a2))
    }

    pub fn highpass(sample_rate: f64, cutoff: f64, q: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * cutoff / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;
        let b0 = (1.0 + cos_w0) / 2.0 / a0;
        let b1 = -(1.0 + cos_w0) / a0;
        let b2 = (1.0 + cos_w0) / 2.0 / a0;
        let a1 = -2.0 * cos_w0 / a0;
        let a2 = (1.0 - alpha) / a0;
        Self::new(f(b0), f(b1), f(b2), f(a1), f(a2))
    }

    pub fn bandpass(sample_rate: f64, cutoff: f64, q: f64) -> Self {
        let w0 = 2.0 * std::f64::consts::PI * cutoff / sample_rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;
        let b0 = alpha / a0;
        let b1 = 0.0;
        let b2 = -alpha / a0;
        let a1 = -2.0 * cos_w0 / a0;
        let a2 = (1.0 - alpha) / a0;
        Self::new(f(b0), f(b1), f(b2), f(a1), f(a2))
    }

    pub fn reset(&mut self) {
        self.w1_i = f(0.0);
        self.w2_i = f(0.0);
        self.w1_q = f(0.0);
        self.w2_q = f(0.0);
    }
}

#[inline(always)]
fn process_sample<T: Float + Copy>(
    b0: T,
    b1: T,
    b2: T,
    a1: T,
    a2: T,
    x: T,
    w1: &mut T,
    w2: &mut T,
) -> T {
    let y = b0 * x + *w1;
    *w1 = b1 * x - a1 * y + *w2;
    *w2 = b2 * x - a2 * y;
    y
}

impl<T: Float + Copy> Filter<T> for Iir<T> {
    fn filter(&mut self, data: &[Complex<T>]) -> Vec<Complex<T>> {
        let (b0, b1, b2, a1, a2) = (self.b0, self.b1, self.b2, self.a1, self.a2);
        data.iter()
            .map(|c| {
                let i = process_sample(b0, b1, b2, a1, a2, c.i, &mut self.w1_i, &mut self.w2_i);
                let q = process_sample(b0, b1, b2, a1, a2, c.q, &mut self.w1_q, &mut self.w2_q);
                Complex::new(i, q)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_persists_across_calls() {
        let mut filter = Iir::<f32>::lowpass(48000.0, 1000.0, 0.707);
        let samples: Vec<Complex<f32>> = (0..64)
            .map(|i| Complex::new((i as f32 * 0.1).sin(), 0.0))
            .collect();
        let out_full = filter.filter(&samples);
        filter.reset();
        let out_a = filter.filter(&samples[..32]);
        let out_b = filter.filter(&samples[32..]);
        let out_split: Vec<_> = out_a.iter().chain(out_b.iter()).cloned().collect();
        for (a, b) in out_full.iter().zip(out_split.iter()) {
            assert!((a.i - b.i).abs() < 1e-6, "I mismatch: {} vs {}", a.i, b.i);
            assert!((a.q - b.q).abs() < 1e-6, "Q mismatch: {} vs {}", a.q, b.q);
        }
    }

    #[test]
    fn iq_channels_independent() {
        let mut filter = Iir::<f32>::lowpass(48000.0, 1000.0, 0.707);
        let samples: Vec<Complex<f32>> = (0..32)
            .map(|i| Complex::new(1.0, (i as f32 * 0.5).sin()))
            .collect();
        let out = filter.filter(&samples);
        // I channel should differ from Q channel since they have different inputs
        let i_vals: Vec<f32> = out.iter().map(|c| c.i).collect();
        let q_vals: Vec<f32> = out.iter().map(|c| c.q).collect();
        assert!(
            i_vals != q_vals,
            "I and Q channels should produce different outputs for different inputs"
        );
    }

    #[test]
    fn dc_lowpass_convergence() {
        let mut filter = Iir::<f32>::lowpass(48000.0, 4000.0, 0.707);
        // Feed DC signal
        let samples: Vec<Complex<f32>> = vec![Complex::new(1.0, 0.0); 512];
        let out = filter.filter(&samples);
        // Last sample should be close to 1.0 (DC passes through lowpass)
        let last = out.last().unwrap();
        assert!(
            (last.i - 1.0).abs() < 0.01,
            "DC should pass lowpass, got {}",
            last.i
        );
    }
}
