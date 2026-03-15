//! LDPC(174, 91) encoder and decoder for FT8.
//!
//! FT8 uses a rate-91/174 LDPC code: 91 information bits (77-bit payload +
//! 14-bit CRC) are encoded into a 174-bit codeword by appending 83 parity bits.
//!
//! # Current status
//!
//! This module provides the correct interface for the FT8 LDPC codec. The
//! encoder appends parity bits computed by systematic encoding (all-zero
//! parity matrix as a placeholder), and the decoder performs a pass-through
//! that recovers the 91 systematic bits without error correction.
//!
//! A real implementation would embed the FT8 parity check matrix H (83×174)
//! and run min-sum belief propagation in the decoder. This placeholder is
//! sufficient for a noiseless encode/decode roundtrip.

/// Encode 91 information bits into a 174-bit systematic codeword.
///
/// The first 91 bits of the output are the unmodified information bits
/// (systematic form). The remaining 83 bits are parity bits.
///
/// # Placeholder behaviour
///
/// Parity bits are set to zero. A real encoder would compute them from
/// the FT8 generator matrix.
pub fn encode(info: &[u8; 91]) -> [u8; 174] {
    let mut codeword = [0u8; 174];
    codeword[..91].copy_from_slice(info);
    // parity bits [91..174] left as zero (placeholder)
    codeword
}

/// Decode a 174-bit received word to 91 information bits.
///
/// Returns `None` if decoding fails. In this placeholder implementation,
/// decoding always succeeds by returning the 91 systematic bits directly.
///
/// A real decoder would run min-sum belief propagation on the FT8 parity
/// check matrix for up to ~50 iterations, returning `None` if the syndrome
/// remains non-zero after all iterations.
pub fn decode(received: &[u8; 174]) -> Option<[u8; 91]> {
    let mut info = [0u8; 91];
    info.copy_from_slice(&received[..91]);
    Some(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_output_length() {
        let info = [0u8; 91];
        let codeword = encode(&info);
        assert_eq!(codeword.len(), 174);
    }

    #[test]
    fn systematic_bits_preserved() {
        let info: [u8; 91] = core::array::from_fn(|i| (i % 2) as u8);
        let codeword = encode(&info);
        assert_eq!(&codeword[..91], &info[..]);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let info: [u8; 91] = core::array::from_fn(|i| ((i * 3) % 2) as u8);
        let codeword = encode(&info);
        let recovered = decode(&codeword).expect("decode should succeed");
        assert_eq!(recovered, info);
    }
}
