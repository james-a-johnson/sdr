use crate::complex::Complex;
use std::f32::consts::PI;

fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

fn bit_reverse_permute(buf: &mut Vec<Complex<f32>>) {
    let n = buf.len();
    let bits = n.ilog2() as usize;
    for i in 0..n {
        let j = reverse_bits(i, bits);
        if i < j {
            buf.swap(i, j);
        }
    }
}

fn reverse_bits(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

fn fft_inplace(buf: &mut Vec<Complex<f32>>, inverse: bool) {
    let n = buf.len();
    assert!(is_power_of_two(n), "FFT length must be a power of 2");

    bit_reverse_permute(buf);

    let sign = if inverse { 1.0_f32 } else { -1.0_f32 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let w_step = Complex::new(
            (sign * 2.0 * PI / len as f32).cos(),
            (sign * 2.0 * PI / len as f32).sin(),
        );
        for i in (0..n).step_by(len) {
            let mut w = Complex::new(1.0_f32, 0.0_f32);
            for j in 0..half {
                let u = buf[i + j];
                let v = buf[i + j + half] * w;
                buf[i + j] = u + v;
                buf[i + j + half] = u - v;
                w = w * w_step;
            }
        }
        len *= 2;
    }
}

pub fn fft(samples: &[Complex<f32>]) -> Vec<Complex<f32>> {
    let mut buf = samples.to_vec();
    fft_inplace(&mut buf, false);
    buf
}

pub fn ifft(spectrum: &[Complex<f32>]) -> Vec<Complex<f32>> {
    let n = spectrum.len();
    let mut buf = spectrum.to_vec();
    fft_inplace(&mut buf, true);
    let scale = 1.0 / n as f32;
    for c in &mut buf {
        c.i *= scale;
        c.q *= scale;
    }
    buf
}

pub fn magnitude_db(spectrum: &[Complex<f32>]) -> Vec<f32> {
    spectrum
        .iter()
        .map(|c| {
            let power = c.i * c.i + c.q * c.q;
            10.0 * power.max(1e-12).log10()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impulse_flat_spectrum() {
        let n = 64;
        let mut samples = vec![Complex::new(0.0_f32, 0.0_f32); n];
        samples[0] = Complex::new(1.0, 0.0);
        let spectrum = fft(&samples);
        for c in &spectrum {
            let mag = (c.i * c.i + c.q * c.q).sqrt();
            assert!((mag - 1.0).abs() < 1e-5, "Impulse should give flat spectrum, got {}", mag);
        }
    }

    #[test]
    fn ifft_roundtrip() {
        let n = 64;
        let samples: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new((i as f32 * 0.1).sin(), (i as f32 * 0.1).cos()))
            .collect();
        let reconstructed = ifft(&fft(&samples));
        for (a, b) in samples.iter().zip(reconstructed.iter()) {
            assert!((a.i - b.i).abs() < 1e-4, "I mismatch: {} vs {}", a.i, b.i);
            assert!((a.q - b.q).abs() < 1e-4, "Q mismatch: {} vs {}", a.q, b.q);
        }
    }

    #[test]
    fn magnitude_db_no_nan() {
        let spectrum: Vec<Complex<f32>> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
        ];
        let db = magnitude_db(&spectrum);
        for &v in &db {
            assert!(!v.is_nan(), "magnitude_db should not produce NaN");
            assert!(v.is_finite() || v == f32::NEG_INFINITY, "unexpected value {}", v);
            assert!(v >= -120.0, "floor should be >= -120 dBFS, got {}", v);
        }
    }
}
