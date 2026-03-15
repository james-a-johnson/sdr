//! Polyphase rational resampler for arbitrary L:M sample-rate conversion.
//!
//! Converts a signal sampled at `fs_in` to `fs_out` where
//! `fs_out / fs_in = L / M` (after GCD reduction). The resampler builds a
//! windowed-sinc lowpass prototype filter and decomposes it into `L` polyphase
//! branches, which avoids computing the full upsampled signal explicitly.
//!
//! # Example: 250 kHz IQ → 48 kHz
//!
//! ```
//! use sdr::filter::resample::RationalResampler;
//! use sdr::filter::Filter;
//! use sdr::complex::Complex;
//!
//! // 250_000 * 24 / 125 = 48_000
//! let mut rs = RationalResampler::new(24, 125);
//! let iq: Vec<Complex<f32>> = (0..1000).map(|_| Complex::new(1.0_f32, 0.0)).collect();
//! let out = rs.process(&iq);
//! // output length ≈ 1000 * 24 / 125 = 192
//! assert!((out.len() as i64 - 192).abs() <= 1);
//! ```

use std::collections::VecDeque;

use super::Filter;
use crate::complex::Complex;

/// Generate a windowed-sinc lowpass FIR with `num_taps` coefficients.
///
/// * `cutoff_norm` — normalized cutoff (1.0 = input Nyquist = fs/2)
/// * `gain`        — linear DC gain (set to `L` to compensate upsampling)
fn sinc_lowpass(num_taps: usize, cutoff_norm: f32, gain: f32) -> Vec<f32> {
    let n = num_taps;
    let center = (n - 1) as f32 / 2.0;
    (0..n)
        .map(|i| {
            let x = i as f32 - center;
            // Sinc
            let s = if x.abs() < 1e-7 {
                1.0
            } else {
                let px = std::f32::consts::PI * x * cutoff_norm * 2.0;
                px.sin() / px
            };
            // Blackman window
            let w = 0.42
                - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos()
                + 0.08 * (4.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos();
            s * w * gain
        })
        .collect()
}

/// Polyphase rational resampler.
///
/// Converts IQ data sampled at `fs_in` to `fs_out` where
/// `fs_out / fs_in = interp / decim`. Anti-aliasing is built in via a
/// windowed-sinc lowpass prototype filter.
///
/// State persists across [`process`](RationalResampler::process) calls, so
/// streaming use (processing in chunks) produces the same result as a single
/// bulk call.
pub struct RationalResampler {
    interp: usize,
    decim: usize,
    branches: Vec<Vec<f32>>,
    state: VecDeque<Complex<f32>>,
    phase: usize,
}

impl RationalResampler {
    /// Create a resampler with 16 taps per polyphase branch.
    pub fn new(interp: usize, decim: usize) -> Self {
        Self::with_taps_per_phase(interp, decim, 16)
    }

    /// Create a resampler with a custom number of taps per polyphase branch.
    ///
    /// More taps improve stopband attenuation at the cost of higher CPU usage
    /// and a longer startup transient equal to `taps_per_phase` input samples.
    pub fn with_taps_per_phase(interp: usize, decim: usize, taps_per_phase: usize) -> Self {
        assert!(interp > 0, "interp must be > 0");
        assert!(decim > 0, "decim must be > 0");
        assert!(taps_per_phase > 0, "taps_per_phase must be > 0");

        let l = interp;
        let m = decim;
        let k = taps_per_phase;
        let num_taps = l * k;

        // Cutoff: protect the narrower side; normalized so 0.5 = input Nyquist.
        // Capped at 0.45 to avoid the degenerate all-zero filter that arises when
        // L == M (fc would hit 0.5, where sinc has zeros at every integer tap for
        // even-length windows with no exact center sample).
        let fc = (0.5_f32 / l.max(m) as f32).min(0.45);
        let mut h = sinc_lowpass(num_taps, fc, l as f32);

        // Normalize prototype so sum(h) = L, guaranteeing DC gain = 1 per output
        // sample.  The windowed-sinc approximation has small gain error for short
        // filters or fc near Nyquist; this corrects it without changing the
        // frequency-response shape.
        let h_sum: f32 = h.iter().sum();
        if h_sum.abs() > 1e-9 {
            let scale = l as f32 / h_sum;
            for tap in h.iter_mut() {
                *tap *= scale;
            }
        }

        // Polyphase decomposition: branches[p][k] = h[p + k*L]
        let branches: Vec<Vec<f32>> = (0..l)
            .map(|p| (0..k).map(|ki| h[p + ki * l]).collect())
            .collect();

        let mut state = VecDeque::with_capacity(k);
        for _ in 0..k {
            state.push_back(Complex::new(0.0, 0.0));
        }

        Self {
            interp: l,
            decim: m,
            branches,
            state,
            phase: 0,
        }
    }

    /// The upsample factor L.
    pub fn interp(&self) -> usize {
        self.interp
    }

    /// The downsample factor M.
    pub fn decim(&self) -> usize {
        self.decim
    }

    /// Resample a block of complex samples.
    ///
    /// The returned length is approximately `input.len() * interp / decim`
    /// (±1 due to phase alignment at block boundaries).
    pub fn process(&mut self, input: &[Complex<f32>]) -> Vec<Complex<f32>> {
        let l = self.interp;
        let m = self.decim;
        let k = self.branches[0].len();
        let capacity = input.len() * l / m + 2;
        let mut out = Vec::with_capacity(capacity);

        for &x in input {
            // Advance delay line
            self.state.pop_front();
            self.state.push_back(x);

            // Emit output samples while phase < L
            while self.phase < l {
                let branch = &self.branches[self.phase];
                let mut sum_i = 0.0_f32;
                let mut sum_q = 0.0_f32;
                // state is oldest..newest; branch[0] aligns with newest sample
                for (coeff, s) in branch.iter().zip(self.state.iter().rev().take(k)) {
                    sum_i += coeff * s.i;
                    sum_q += coeff * s.q;
                }
                out.push(Complex::new(sum_i, sum_q));
                self.phase += m;
            }
            self.phase -= l;
        }
        out
    }
}

impl Filter<f32> for RationalResampler {
    /// Delegates to [`process`](RationalResampler::process).
    fn filter(&mut self, input: &[Complex<f32>]) -> Vec<Complex<f32>> {
        self.process(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_length() {
        // For various L/M ratios the output length must be within 1 of expected.
        let cases = [(24, 125, 1000), (1, 4, 1000), (3, 7, 500), (2, 3, 300)];
        for (l, m, n) in cases {
            let mut rs = RationalResampler::new(l, m);
            let input: Vec<Complex<f32>> = vec![Complex::new(1.0, 0.0); n];
            let out = rs.process(&input);
            let expected = n * l / m;
            assert!(
                (out.len() as i64 - expected as i64).abs() <= 1,
                "L={l} M={m} n={n}: expected ~{expected}, got {}",
                out.len()
            );
        }
    }

    #[test]
    fn integer_decimate_equivalent() {
        use crate::filter::{Decimate, Filter};
        let n = 400;
        let input: Vec<Complex<f32>> = (0..n).map(|i| Complex::new(i as f32, 0.0)).collect();

        let mut rs = RationalResampler::new(1, 4);
        let rs_out = rs.process(&input);

        let mut dec: Decimate<4> = Decimate;
        let dec_out = dec.filter(&input);

        // Lengths should match
        assert_eq!(
            rs_out.len(),
            dec_out.len(),
            "RationalResampler(1,4) and Decimate<4> lengths differ"
        );
    }

    #[test]
    fn streaming_consistency() {
        let input: Vec<Complex<f32>> = (0..200)
            .map(|i| Complex::new((i as f32 * 0.1).sin(), (i as f32 * 0.07).cos()))
            .collect();

        let mut rs_bulk = RationalResampler::new(3, 7);
        let out_bulk = rs_bulk.process(&input);

        let mut rs_stream = RationalResampler::new(3, 7);
        let chunk = 50;
        let mut out_stream = Vec::new();
        for c in input.chunks(chunk) {
            out_stream.extend(rs_stream.process(c));
        }

        assert_eq!(out_bulk.len(), out_stream.len(), "length mismatch");
        for (i, (a, b)) in out_bulk.iter().zip(out_stream.iter()).enumerate() {
            assert!(
                (a.i - b.i).abs() < 1e-5,
                "I mismatch at {i}: {} vs {}",
                a.i,
                b.i
            );
            assert!(
                (a.q - b.q).abs() < 1e-5,
                "Q mismatch at {i}: {} vs {}",
                a.q,
                b.q
            );
        }
    }

    #[test]
    fn dc_preserved() {
        let dc = Complex::new(1.0_f32, 0.5);
        // Use 1:1 ratio (no rate change) with enough input to flush the transient
        let mut rs = RationalResampler::with_taps_per_phase(1, 1, 8);
        let warmup: Vec<Complex<f32>> = vec![dc; 32];
        let out = rs.process(&warmup);
        // After settling (skip first K samples), DC should be preserved
        let k = 8_usize; // taps_per_phase
        for s in out.iter().skip(k) {
            assert!(
                (s.i - dc.i).abs() < 0.05,
                "DC I not preserved: got {}",
                s.i
            );
            assert!(
                (s.q - dc.q).abs() < 0.05,
                "DC Q not preserved: got {}",
                s.q
            );
        }
    }

    #[test]
    fn tone_frequency_preserved() {
        // Generate a tone at f Hz in a signal sampled at fs_in,
        // resample to fs_out, verify the tone is still at f Hz.
        let fs_in = 250_000.0_f32;
        let fs_out = 48_000.0_f32;
        let l = 24_usize;
        let m = 125_usize;
        let f_tone = 1_000.0_f32; // 1 kHz tone

        // Generate enough samples to overcome filter transient
        let n_in = 2048_usize;
        let input: Vec<Complex<f32>> = (0..n_in)
            .map(|i| {
                let t = i as f32 / fs_in;
                let angle = 2.0 * std::f32::consts::PI * f_tone * t;
                Complex::new(angle.cos(), angle.sin())
            })
            .collect();

        let mut rs = RationalResampler::new(l, m);
        let out = rs.process(&input);

        // Skip initial transient (taps_per_phase = 16 input samples → ~3 output samples)
        let skip = 5;
        let out_trim = &out[skip..];
        let n_out = out_trim.len();

        // Find the peak frequency via DFT at f_tone
        // Expected bin: f_tone / fs_out * n_out
        let expected_phase_step = 2.0 * std::f32::consts::PI * f_tone / fs_out;
        let mut sum_re = 0.0_f32;
        let mut sum_im = 0.0_f32;
        for (i, s) in out_trim.iter().enumerate() {
            let angle = expected_phase_step * i as f32;
            sum_re += s.i * angle.cos() + s.q * angle.sin();
            sum_im += s.q * angle.cos() - s.i * angle.sin();
        }
        let power_at_f = (sum_re * sum_re + sum_im * sum_im).sqrt() / n_out as f32;

        // Also check power at a nearby wrong frequency (2 kHz)
        let wrong_step = 2.0 * std::f32::consts::PI * 2000.0_f32 / fs_out;
        let mut wr = 0.0_f32;
        let mut wi = 0.0_f32;
        for (i, s) in out_trim.iter().enumerate() {
            let angle = wrong_step * i as f32;
            wr += s.i * angle.cos() + s.q * angle.sin();
            wi += s.q * angle.cos() - s.i * angle.sin();
        }
        let power_wrong = (wr * wr + wi * wi).sqrt() / n_out as f32;

        assert!(
            power_at_f > power_wrong * 5.0,
            "Tone at {f_tone} Hz not dominant after resampling: power_at_f={power_at_f}, power_wrong={power_wrong}"
        );
    }
}
