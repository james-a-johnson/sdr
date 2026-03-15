//! Gray code encoding and decoding for FSK symbol mapping.
//!
//! In 8-FSK systems like FT8, Gray coding maps binary symbol indices to
//! transmitted tones so that adjacent tones differ by only one bit, reducing
//! the bit-error impact of one-tone symbol errors.

/// Encode a binary value to Gray code.
///
/// `n ^ (n >> 1)` maps consecutive integers to Gray codewords where
/// adjacent values differ by exactly one bit.
///
/// # Example
///
/// ```
/// use sdr::coding::gray;
/// assert_eq!(gray::encode(5), 7); // 101 → 111
/// assert_eq!(gray::encode(7), 4); // 111 → 100
/// ```
pub fn encode(n: u32) -> u32 {
    n ^ (n >> 1)
}

/// Decode a Gray code value back to binary.
///
/// Inverse of [`encode`]: `decode(encode(n)) == n` for all `n`.
///
/// # Example
///
/// ```
/// use sdr::coding::gray;
/// assert_eq!(gray::decode(gray::encode(6)), 6);
/// ```
pub fn decode(g: u32) -> u32 {
    let mut n = g;
    let mut mask = g >> 1;
    while mask != 0 {
        n ^= mask;
        mask >>= 1;
    }
    n
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
