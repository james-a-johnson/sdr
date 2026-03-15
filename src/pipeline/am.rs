//! AM pipeline type aliases.
//!
//! [`AmTx`] and [`AmRx`] are [`TxPipeline`] / [`RxPipeline`] specialised for
//! AM modulation. They are thin type aliases — all builder methods and trait
//! implementations come from the generic pipeline types.

use crate::modulation::am::{AmDemodulator, AmModulator};
use super::{RxPipeline, TxPipeline};

/// AM transmit pipeline: [`AmModulator`] plus an optional IQ filter chain.
pub type AmTx = TxPipeline<AmModulator>;

/// AM receive pipeline: an optional IQ filter chain plus [`AmDemodulator`].
pub type AmRx = RxPipeline<AmDemodulator>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;
    use crate::filter::Filter;
    use crate::pipeline::{Receiver, Transmitter};
    use std::cell::RefCell;
    use std::f32::consts::PI;
    use std::rc::Rc;

    #[test]
    fn am_roundtrip() {
        let n = 1024;
        let audio: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f32 / 48_000.0).sin())
            .collect();

        let mut tx = AmTx::new(AmModulator::new(1.0));
        let mut rx = AmRx::new(AmDemodulator::new(1.0));

        let iq = tx.transmit(&audio);
        let recovered = rx.receive(&iq);

        for (orig, rec) in audio.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 1e-5,
                "mismatch: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn am_tx_with_resampler_length() {
        let n = 100;
        let audio = vec![0.5_f32; n];
        let mut tx = AmTx::new(AmModulator::new(1.0)).with_resampler(5, 1);
        let iq = tx.transmit(&audio);
        assert!(
            (iq.len() as i64 - (n * 5) as i64).abs() <= 1,
            "expected ~{}, got {}",
            n * 5,
            iq.len()
        );
    }

    #[test]
    fn am_rx_with_resampler_length() {
        let n = 400;
        let iq: Vec<Complex<f32>> = vec![Complex::new(1.0, 0.0); n];
        let mut rx = AmRx::new(AmDemodulator::new(1.0)).with_resampler(1, 4);
        let out = rx.receive(&iq);
        assert!(
            (out.len() as i64 - (n / 4) as i64).abs() <= 1,
            "expected ~{}, got {}",
            n / 4,
            out.len()
        );
    }

    #[test]
    fn am_rx_with_lowpass_builds() {
        let n = 256;
        let iq: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new((i as f32 * 0.01).sin(), (i as f32 * 0.01).cos()))
            .collect();
        let mut rx = AmRx::new(AmDemodulator::new(0.8))
            .with_lowpass(48_000.0, 4_000.0, 0.707);
        let out = rx.receive(&iq);
        assert_eq!(out.len(), n);
    }

    #[test]
    fn am_filter_chain_order() {
        struct RecordingFilter {
            log: Rc<RefCell<Vec<usize>>>,
            id: usize,
        }

        impl Filter<f32> for RecordingFilter {
            fn filter(&mut self, data: &[Complex<f32>]) -> Vec<Complex<f32>> {
                self.log.borrow_mut().push(self.id);
                data.to_vec()
            }
        }

        let log: Rc<RefCell<Vec<usize>>> = Rc::new(RefCell::new(Vec::new()));

        let mut tx = AmTx::new(AmModulator::new(1.0))
            .with_filter(RecordingFilter {
                log: Rc::clone(&log),
                id: 1,
            })
            .with_filter(RecordingFilter {
                log: Rc::clone(&log),
                id: 2,
            });

        tx.transmit(&[0.5_f32; 10]);

        assert_eq!(*log.borrow(), vec![1, 2]);
    }
}
