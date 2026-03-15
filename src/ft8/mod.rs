//! FT8 digital mode encoder and decoder.
//!
//! FT8 is a weak-signal digital mode designed for amateur radio. Key parameters:
//!
//! - 77-bit message payload (callsigns, grid, power)
//! - CRC-14 integrity check → 91 bits total
//! - LDPC(174, 91) forward error correction → 174 code bits
//! - 174 bits interleaved and mapped to 58 eight-FSK symbols (3 bits/symbol)
//! - 21 Costas synchronization symbols inserted at frame positions 0–6, 36–42, 72–78
//! - 79 symbols total at 6.25 baud with 6.25 Hz tone spacing
//! - Transmitted at 12 000 Hz sample rate → 1920 samples/symbol → 151 680 samples/frame
//!
//! # Type aliases
//!
//! [`codec::Ft8Tx`] and [`codec::Ft8Rx`] are [`DigitalTxPipeline`] /
//! [`DigitalRxPipeline`] specialised for FT8. All builder methods and the
//! `transmit` / `receive` interface come from the generic pipeline types.
//!
//! [`DigitalTxPipeline`]: crate::pipeline::digital::DigitalTxPipeline
//! [`DigitalRxPipeline`]: crate::pipeline::digital::DigitalRxPipeline

pub mod codec;
pub mod frame;
