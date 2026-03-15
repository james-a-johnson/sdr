//! CRC-14 integrity checking for FT8 frames.
//!
//! FT8 appends a 14-bit CRC to the 77-bit message payload, forming a 91-bit
//! block that is then LDPC-encoded. Polynomial: `0x2757`.

/// CRC-14 generator polynomial (without the implicit degree-14 leading term).
const POLY: u16 = 0x2757;

/// Compute CRC-14 over a slice of individual bits (each element is 0 or 1).
///
/// Processes bits in order (index 0 = first bit shifted in). Returns a 14-bit
/// value in the low bits of the result.
///
/// # Example
///
/// ```
/// use sdr::coding::crc;
///
/// let bits = [1u8, 0, 1, 1, 0, 0, 1, 0]; // 8 bits
/// let c1 = crc::crc14_bits(&bits);
/// assert_ne!(c1, 0); // non-zero for non-trivial input
/// // Verify the CRC is deterministic
/// assert_eq!(crc::crc14_bits(&bits), c1);
/// ```
pub fn crc14_bits(bits: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &bit in bits {
        let feedback = ((crc >> 13) & 1) ^ (bit as u16 & 1);
        crc = (crc << 1) & 0x3FFF;
        if feedback != 0 {
            crc ^= POLY;
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zeros_crc_is_zero() {
        let bits = [0u8; 77];
        assert_eq!(crc14_bits(&bits), 0);
    }

    #[test]
    fn single_bit_nonzero() {
        let mut bits = [0u8; 77];
        bits[0] = 1;
        let c = crc14_bits(&bits);
        assert_ne!(c, 0);
    }

    #[test]
    fn deterministic() {
        let bits: Vec<u8> = (0..77).map(|i| (i % 3) as u8 & 1).collect();
        assert_eq!(crc14_bits(&bits), crc14_bits(&bits));
    }

    #[test]
    fn different_inputs_different_crcs() {
        let bits_a: Vec<u8> = (0..77).map(|i| (i % 2) as u8).collect();
        let bits_b: Vec<u8> = (0..77).map(|i| (i % 3) as u8 & 1).collect();
        assert_ne!(crc14_bits(&bits_a), crc14_bits(&bits_b));
    }
}
