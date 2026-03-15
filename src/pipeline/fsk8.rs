//! 8-FSK pipeline type aliases.
//!
//! [`Fsk8Tx`] and [`Fsk8Rx`] are [`TxPipeline`] / [`RxPipeline`] specialised
//! for 8-FSK modulation. They are thin type aliases — all builder methods and
//! trait implementations come from the generic pipeline types.

use crate::modulation::fsk8::{Fsk8Demodulator, Fsk8Modulator};
use super::{RxPipeline, TxPipeline};

/// 8-FSK transmit pipeline: [`Fsk8Modulator`] plus an optional IQ filter chain.
pub type Fsk8Tx = TxPipeline<Fsk8Modulator>;

/// 8-FSK receive pipeline: an optional IQ filter chain plus [`Fsk8Demodulator`].
pub type Fsk8Rx = RxPipeline<Fsk8Demodulator>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{Receiver, Transmitter};

    #[test]
    fn fsk8_pipeline_roundtrip() {
        let sample_rate = 1_000.0_f32;
        let symbol_rate = 10.0_f32;
        let tone_spacing = 10.0_f32;

        let symbols: Vec<f32> = (0..8).map(|k| k as f32).collect();

        let mut tx = Fsk8Tx::new(Fsk8Modulator::new(sample_rate, symbol_rate, tone_spacing));
        let mut rx = Fsk8Rx::new(Fsk8Demodulator::new(sample_rate, symbol_rate, tone_spacing));

        let iq = tx.transmit(&symbols);
        let recovered = rx.receive(&iq);

        assert_eq!(recovered, symbols, "pipeline roundtrip mismatch");
    }
}
