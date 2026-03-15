//! Bit-reversal interleaver / deinterleaver for 174-bit FT8 codewords.
//!
//! The FT8 interleaver permutes 174 code bits using a 9-bit bit-reversal
//! permutation: for each source index `i` (in increasing order), compute
//! the 9-bit bit-reversal of `i`; if the result is less than 174, emit it
//! as the next destination index. This produces a bijection on `0..174`.

/// Build the 174-element bit-reversal permutation used by FT8.
///
/// `perm[i]` is the destination index for source bit `i`.
fn build_perm() -> [usize; 174] {
    let mut perm = [0usize; 174];
    let mut out_idx = 0usize;
    let mut in_idx: u32 = 0;
    while out_idx < 174 {
        // 9-bit bit-reversal: reverse all 32 bits then shift right by 23.
        let rev = (in_idx.reverse_bits() >> 23) as usize;
        if rev < 174 {
            perm[out_idx] = rev;
            out_idx += 1;
        }
        in_idx += 1;
    }
    perm
}

/// Interleave a 174-bit codeword.
///
/// `out[i] = bits[perm[i]]` — reads source bits in permuted order.
pub fn interleave(bits: &[u8; 174]) -> [u8; 174] {
    let perm = build_perm();
    let mut out = [0u8; 174];
    for i in 0..174 {
        out[i] = bits[perm[i]];
    }
    out
}

/// Deinterleave a 174-bit codeword (inverse of [`interleave`]).
///
/// `out[perm[i]] = bits[i]` — writes bits back to their original positions.
pub fn deinterleave(bits: &[u8; 174]) -> [u8; 174] {
    let perm = build_perm();
    let mut out = [0u8; 174];
    for i in 0..174 {
        out[perm[i]] = bits[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perm_is_valid_bijection() {
        let perm = build_perm();
        let mut seen = [false; 174];
        for &p in &perm {
            assert!(p < 174, "perm value {p} out of range");
            assert!(!seen[p], "duplicate perm value {p}");
            seen[p] = true;
        }
        assert!(seen.iter().all(|&s| s), "perm is not surjective");
    }

    #[test]
    fn interleave_deinterleave_roundtrip() {
        let bits: [u8; 174] = core::array::from_fn(|i| (i % 2) as u8);
        let interleaved = interleave(&bits);
        let recovered = deinterleave(&interleaved);
        assert_eq!(recovered, bits);
    }

    #[test]
    fn deinterleave_interleave_roundtrip() {
        let bits: [u8; 174] = core::array::from_fn(|i| ((i / 3) % 2) as u8);
        let deinterleaved = deinterleave(&bits);
        let recovered = interleave(&deinterleaved);
        assert_eq!(recovered, bits);
    }
}
