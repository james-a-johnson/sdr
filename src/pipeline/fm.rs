//! FM pipeline type aliases.
//!
//! [`FmTx`] and [`FmRx`] are [`TxPipeline`] / [`RxPipeline`] specialised for
//! FM modulation. They are thin type aliases — all builder methods and trait
//! implementations come from the generic pipeline types.

use crate::modulation::fm::{FmDemodulator, FmModulator};
use super::{RxPipeline, TxPipeline};

/// FM transmit pipeline: [`FmModulator`] plus an optional IQ filter chain.
pub type FmTx = TxPipeline<FmModulator>;

/// FM receive pipeline: an optional IQ filter chain plus [`FmDemodulator`].
pub type FmRx = RxPipeline<FmDemodulator>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{Receiver, Transmitter};
    use std::f32::consts::PI;

    #[test]
    fn fm_roundtrip() {
        let sample_rate = 48_000.0_f32;
        let deviation = 5_000.0_f32;
        let n = 1024;
        let audio: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / sample_rate).sin())
            .collect();

        let mut tx = FmTx::new(FmModulator::new(sample_rate, deviation));
        let mut rx = FmRx::new(FmDemodulator::new(sample_rate, deviation));

        let iq = tx.transmit(&audio);
        let recovered = rx.receive(&iq);

        let start = 64;
        for (orig, rec) in audio[start..].iter().zip(recovered[start..].iter()) {
            assert!(
                (orig - rec).abs() < 0.05,
                "mismatch: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn fm_with_rates_length() {
        let n = 480_usize;
        let audio = vec![0.0_f32; n];
        let mut tx = FmTx::new(FmModulator::new(48_000.0, 5_000.0)).with_rates(48_000, 250_000);
        let iq = tx.transmit(&audio);
        let expected = n as f32 * 250_000.0 / 48_000.0;
        assert!(
            (iq.len() as f32 - expected).abs() <= 2.0,
            "expected ~{}, got {}",
            expected,
            iq.len()
        );
    }
}
