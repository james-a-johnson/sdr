//! Biquad IIR filter using the Direct Form II Transposed structure.

use super::Filter;
use crate::complex::Complex;
use num_traits::Float;

/// Cast an `f64` literal to any [`Float`] type at construction time.
fn f<T: Float>(x: f64) -> T {
    T::from(x).unwrap()
}

/// Stateful biquad IIR filter (Direct Form II Transposed).
///
/// Processes I and Q channels independently with shared coefficients, so a
/// single filter instance handles a full complex IQ stream without cross-talk
/// between channels.
///
/// ## Difference equation
///
/// Given input `x[n]` and output `y[n]`, the filter computes:
///
/// ```text
/// y[n]  = b0·x[n] + w1
/// w1    = b1·x[n] − a1·y[n] + w2
/// w2    = b2·x[n] − a2·y[n]
/// ```
///
/// where `(w1, w2)` are the delay-line state registers. The denominator
/// polynomial is `1 + a1·z⁻¹ + a2·z⁻²` (i.e. `a0` is normalized to 1).
///
/// ## Streaming safety
///
/// State persists across [`Filter::filter`] calls. Splitting a stream into
/// blocks and filtering each block in sequence produces the same result as
/// filtering the entire stream at once.
///
/// ## Constructors
///
/// Use [`Iir::new`] for raw coefficients, or one of the Audio EQ Cookbook
/// presets: [`Iir::lowpass`], [`Iir::highpass`], [`Iir::bandpass`].
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
    /// Create a biquad filter from raw normalized coefficients.
    ///
    /// Coefficients follow the convention where the denominator leading term
    /// is 1 (i.e. `a0 = 1`). Pass feedback coefficients already divided by
    /// `a0` if your source uses the un-normalized form.
    ///
    /// * `b0, b1, b2` — feedforward (numerator) coefficients
    /// * `a1, a2`     — feedback (denominator) coefficients, sign convention:
    ///   the denominator is `1 + a1·z⁻¹ + a2·z⁻²`
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

    /// Design a second-order Butterworth lowpass filter using the Audio EQ
    /// Cookbook formulas.
    ///
    /// * `sample_rate` — sample rate in Hz
    /// * `cutoff`      — −3 dB cutoff frequency in Hz
    /// * `q`           — quality factor (0.707 ≈ maximally flat / Butterworth)
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

    /// Design a second-order Butterworth highpass filter using the Audio EQ
    /// Cookbook formulas.
    ///
    /// * `sample_rate` — sample rate in Hz
    /// * `cutoff`      — −3 dB cutoff frequency in Hz
    /// * `q`           — quality factor (0.707 ≈ maximally flat / Butterworth)
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

    /// Design a second-order bandpass filter using the Audio EQ Cookbook
    /// formulas (constant 0 dB peak gain).
    ///
    /// * `sample_rate` — sample rate in Hz
    /// * `cutoff`      — center frequency in Hz
    /// * `q`           — quality factor; higher Q = narrower passband
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

    /// Zero all internal state registers.
    ///
    /// Call this before reusing the filter on a new, unrelated stream to
    /// prevent the previous signal's tail from bleeding into the new one.
    pub fn reset(&mut self) {
        self.w1_i = f(0.0);
        self.w2_i = f(0.0);
        self.w1_q = f(0.0);
        self.w2_q = f(0.0);
    }
}

/// Direct Form II Transposed single-sample update.
///
/// Inlined free function to avoid the borrow-checker conflict that arises when
/// `&self` (for coefficients) and `&mut self.wN` (for state) are held
/// simultaneously inside a method.
#[allow(clippy::too_many_arguments)]
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
    /// Filter a block of complex samples, updating internal state.
    ///
    /// The I and Q channels are processed independently: I-channel state
    /// (`w1_i`, `w2_i`) is never mixed with Q-channel state (`w1_q`, `w2_q`).
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
