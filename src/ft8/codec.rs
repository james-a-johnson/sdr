//! FT8 encoder and decoder implementing [`Encode`] and [`Decode`].
//!
//! # Encoding chain
//!
//! ```text
//! bytes (≤10) → 77 payload bits
//!             → CRC-14 appended → 91 bits
//!             → LDPC(174,91) encode → 174 code bits
//!             → bit-reversal interleave → 174 bits
//!             → 3-bit Gray→symbol mapping → 58 symbols
//!             → Costas insertion → 79 symbols
//!             → 8-FSK modulation → IQ samples
//! ```
//!
//! # Decoding chain (reverse)
//!
//! ```text
//! IQ → 8-FSK demodulation → 79 symbols
//!    → Costas strip → 58 symbols
//!    → Gray decode + unpack → 174 bits
//!    → bit-reversal deinterleave → 174 bits
//!    → LDPC decode → 91 bits (or None)
//!    → CRC-14 check → 77 bits (or None)
//!    → pack into bytes → Vec<u8>
//! ```
//!
//! # Type aliases
//!
//! [`Ft8Tx`] and [`Ft8Rx`] are thin type aliases around the generic
//! [`DigitalTxPipeline`] / [`DigitalRxPipeline`] with FT8 codecs inside.

use crate::coding::{crc, gray, interleave, ldpc};
use crate::complex::Complex;
use crate::ft8::frame::{self, DATA_SYMBOLS, FRAME_SYMBOLS};
use crate::modulation::fsk8::{Fsk8Demodulator, Fsk8Modulator};
use crate::pipeline::digital::{Decode, DigitalRxPipeline, DigitalTxPipeline, Encode};

/// FT8 sample rate (Hz).
const SAMPLE_RATE: f32 = 12_000.0;
/// FT8 symbol rate (baud).
const SYMBOL_RATE: f32 = 6.25;
/// FT8 tone spacing (Hz).
const TONE_SPACING: f32 = 6.25;

/// FT8 encoder: bytes → IQ samples.
///
/// Wraps an [`Fsk8Modulator`] and implements the complete FT8 encoding chain.
/// Create via [`Ft8Tx::new`] to get the full pipeline with optional IQ filters,
/// or use [`Ft8Encoder::new`] directly if you only need the codec.
pub struct Ft8Encoder {
    modulator: Fsk8Modulator,
}

impl Ft8Encoder {
    /// Create a new FT8 encoder with default FT8 parameters
    /// (12 kHz, 6.25 baud, 6.25 Hz tone spacing).
    pub fn new() -> Self {
        Self {
            modulator: Fsk8Modulator::new(SAMPLE_RATE, SYMBOL_RATE, TONE_SPACING),
        }
    }
}

impl Default for Ft8Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Encode for Ft8Encoder {
    /// Encode up to 77 bits from `data` (10 bytes, bottom 3 bits of last byte unused)
    /// into FT8 IQ samples.
    fn encode(&mut self, data: &[u8]) -> Vec<Complex<f32>> {
        // 1. Extract 77 payload bits (MSB first, missing bytes treated as zero).
        let mut msg_bits = [0u8; 77];
        for (i, bit) in msg_bits.iter_mut().enumerate() {
            let byte = i / 8;
            let bit_pos = 7 - (i % 8);
            if byte < data.len() {
                *bit = (data[byte] >> bit_pos) & 1;
            }
        }

        // 2. Compute CRC-14 and form 91-bit block [message | CRC].
        let checksum = crc::crc14_bits(&msg_bits);
        let mut info_bits = [0u8; 91];
        info_bits[..77].copy_from_slice(&msg_bits);
        for i in 0..14 {
            info_bits[77 + i] = ((checksum >> (13 - i)) & 1) as u8;
        }

        // 3. LDPC encode 91 → 174 bits.
        let codeword = ldpc::encode(&info_bits);

        // 4. Interleave.
        let interleaved = interleave::interleave(&codeword);

        // 5. Map to 58 Gray-coded 8-FSK symbols (3 bits per symbol).
        let mut data_symbols = [0f32; DATA_SYMBOLS];
        for i in 0..DATA_SYMBOLS {
            let b0 = interleaved[3 * i] as u32;
            let b1 = interleaved[3 * i + 1] as u32;
            let b2 = interleaved[3 * i + 2] as u32;
            let bin_sym = (b0 << 2) | (b1 << 1) | b2;
            data_symbols[i] = gray::encode(bin_sym) as f32;
        }

        // 6. Insert Costas synchronization arrays → 79-symbol frame.
        let mut frame = [0f32; FRAME_SYMBOLS];
        frame::insert_costas(&data_symbols, &mut frame);

        // 7. 8-FSK modulate.
        self.modulator.set_phase(0.0);
        self.modulator.modulate(&frame)
    }
}

/// FT8 decoder: IQ samples → bytes (fallible).
///
/// Wraps an [`Fsk8Demodulator`] and implements the complete FT8 decoding chain.
/// Returns `None` if the LDPC decoder fails or the CRC does not match.
pub struct Ft8Decoder {
    demodulator: Fsk8Demodulator,
}

impl Ft8Decoder {
    /// Create a new FT8 decoder with default FT8 parameters.
    pub fn new() -> Self {
        Self {
            demodulator: Fsk8Demodulator::new(SAMPLE_RATE, SYMBOL_RATE, TONE_SPACING),
        }
    }
}

impl Default for Ft8Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decode for Ft8Decoder {
    fn decode(&mut self, iq: &[Complex<f32>]) -> Option<Vec<u8>> {
        // 1. 8-FSK demodulate → 79 symbols.
        self.demodulator.reset();
        let symbols = self.demodulator.demodulate(iq);
        if symbols.len() < FRAME_SYMBOLS {
            return None;
        }

        // 2. Strip Costas arrays → 58 data symbols.
        let data_symbols = frame::extract_data(&symbols)?;

        // 3. Gray-decode each symbol and unpack to 174 bits.
        let mut interleaved = [0u8; 174];
        for i in 0..DATA_SYMBOLS {
            let gray_sym = data_symbols[i].round().clamp(0.0, 7.0) as u32;
            let bin_sym = gray::decode(gray_sym);
            interleaved[3 * i] = ((bin_sym >> 2) & 1) as u8;
            interleaved[3 * i + 1] = ((bin_sym >> 1) & 1) as u8;
            interleaved[3 * i + 2] = (bin_sym & 1) as u8;
        }

        // 4. Deinterleave.
        let codeword = interleave::deinterleave(&interleaved);

        // 5. LDPC decode → 91 information bits.
        let info_bits = ldpc::decode(&codeword)?;

        // 6. Verify CRC-14.
        let received_crc = {
            let mut c = 0u16;
            for i in 0..14 {
                c = (c << 1) | info_bits[77 + i] as u16;
            }
            c
        };
        let computed_crc = crc::crc14_bits(&info_bits[..77]);
        if received_crc != computed_crc {
            return None;
        }

        // 7. Pack 77 payload bits into 10 bytes (bottom 3 bits of last byte = 0).
        let mut out = vec![0u8; 10];
        for (i, &bit) in info_bits[..77].iter().enumerate() {
            let byte = i / 8;
            let bit_pos = 7 - (i % 8);
            out[byte] |= bit << bit_pos;
        }
        Some(out)
    }
}

/// FT8 transmit pipeline: [`Ft8Encoder`] plus an optional IQ filter chain.
pub type Ft8Tx = DigitalTxPipeline<Ft8Encoder>;

/// FT8 receive pipeline: an optional IQ filter chain plus [`Ft8Decoder`].
pub type Ft8Rx = DigitalRxPipeline<Ft8Decoder>;

#[cfg(test)]
mod tests {
    use super::*;

    /// A 10-byte input where the bottom 3 bits of the last byte are zero,
    /// so the roundtrip can recover it exactly.
    fn test_payload() -> [u8; 10] {
        [0xDE, 0xAD, 0xBE, 0xEF, 0x42, 0x13, 0x37, 0x55, 0xAA, 0xF8]
        //                                                         ^^ low 3 bits = 0
    }

    #[test]
    fn ft8_encode_output_length() {
        let mut enc = Ft8Encoder::new();
        let iq = enc.encode(&test_payload());
        // 79 symbols × 1920 samples/symbol = 151 680
        assert_eq!(iq.len(), 79 * (12_000.0_f32 / 6.25).round() as usize);
    }

    #[test]
    fn ft8_roundtrip() {
        let payload = test_payload();
        let mut enc = Ft8Encoder::new();
        let iq = enc.encode(&payload);

        let mut dec = Ft8Decoder::new();
        let recovered = dec.decode(&iq).expect("decode should succeed");

        assert_eq!(recovered, payload.to_vec());
    }

    #[test]
    fn ft8_pipeline_roundtrip() {
        let payload = test_payload();
        let mut tx = Ft8Tx::new(Ft8Encoder::new());
        let iq = tx.transmit(&payload);

        let mut rx = Ft8Rx::new(Ft8Decoder::new());
        let recovered = rx.receive(&iq).expect("pipeline receive should succeed");

        assert_eq!(recovered, payload.to_vec());
    }

    #[test]
    fn ft8_with_freq_shift_roundtrip() {
        let payload = test_payload();
        // Shift up by 1000 Hz, then back down before decoding.
        let mut tx = Ft8Tx::new(Ft8Encoder::new()).with_freq_shift(SAMPLE_RATE, 1_000.0);
        let iq = tx.transmit(&payload);

        let mut rx = Ft8Rx::new(Ft8Decoder::new()).with_freq_shift(SAMPLE_RATE, -1_000.0);
        let recovered = rx.receive(&iq).expect("freq-shifted pipeline should decode");

        assert_eq!(recovered, payload.to_vec());
    }
}
