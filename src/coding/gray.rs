//! Gray code encoding and decoding for FSK symbol mapping.
//!
//! In 8-FSK systems like FT8, Gray coding maps binary symbol indices to
//! transmitted tones so that adjacent tones differ by only one bit, reducing
//! the bit-error impact of one-tone symbol errors.
//!
//! Only 3-bit symbols (values `0..=7`) are used in FT8, so both directions
//! are implemented as direct lookups into 8-entry compile-time tables.

/// Precomputed 3-bit Gray encode table.
///
/// `ENCODE_TABLE[n]` is the Gray code of binary value `n` for `n` in `0..8`.
/// Standard 3-bit Gray code sequence: 0, 1, 3, 2, 6, 7, 5, 4.
pub const ENCODE_TABLE: [u8; 8] = [0, 1, 3, 2, 6, 7, 5, 4];

/// Precomputed 3-bit Gray decode table (inverse of [`ENCODE_TABLE`]).
///
/// `DECODE_TABLE[g]` is the binary value whose Gray code is `g`, for `g` in `0..8`.
pub const DECODE_TABLE: [u8; 8] = [0, 1, 3, 2, 7, 6, 4, 5];

/// Encode a 3-bit binary value (`0..=7`) to its Gray code.
///
/// # Panics
///
/// Panics if `n >= 8`.
///
/// # Example
///
/// ```
/// use sdr::coding::gray;
/// assert_eq!(gray::encode(5), 7); // 101 → 111
/// assert_eq!(gray::encode(7), 4); // 111 → 100
/// ```
pub fn encode(n: u32) -> u32 {
    ENCODE_TABLE[n as usize] as u32
}

/// Decode a 3-bit Gray code value (`0..=7`) back to binary.
///
/// Inverse of [`encode`]: `decode(encode(n)) == n` for all `n` in `0..8`.
///
/// # Panics
///
/// Panics if `g >= 8`.
///
/// # Example
///
/// ```
/// use sdr::coding::gray;
/// assert_eq!(gray::decode(gray::encode(6)), 6);
/// ```
pub fn decode(g: u32) -> u32 {
    DECODE_TABLE[g as usize] as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_3bit_table() {
        // Standard 3-bit Gray code
        let expected = [0u32, 1, 3, 2, 6, 7, 5, 4];
        for (n, &g) in expected.iter().enumerate() {
            assert_eq!(encode(n as u32), g, "encode({n}) mismatch");
        }
    }

    #[test]
    fn decode_roundtrip() {
        for n in 0u32..8 {
            assert_eq!(decode(encode(n)), n, "roundtrip failed for {n}");
        }
    }

    #[test]
    fn adjacent_gray_codes_differ_by_one_bit() {
        for n in 0u32..7 {
            let diff = encode(n) ^ encode(n + 1);
            assert!(diff.count_ones() == 1, "n={n}: diff={diff:b} has more than one bit");
        }
    }
}
