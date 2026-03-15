//! FT8 frame structure: Costas synchronization arrays and symbol layout.
//!
//! An FT8 frame contains 79 8-FSK symbols:
//!
//! ```text
//! [0..7]   Costas array #1  (7 symbols)
//! [7..36]  Data symbols     (29 symbols)
//! [36..43] Costas array #2  (7 symbols)
//! [43..72] Data symbols     (29 symbols)
//! [72..79] Costas array #3  (7 symbols)
//! ```
//!
//! Total: 58 data symbols × 3 bits/symbol = 174 coded bits.

/// FT8 Costas synchronization array (7 symbols, values 0–7).
///
/// Inserted at frame positions 0–6, 36–42, and 72–78.
pub const COSTAS: [u8; 7] = [3, 1, 4, 0, 6, 5, 2];

/// Total symbols in one FT8 frame.
pub const FRAME_SYMBOLS: usize = 79;

/// Number of data symbols per frame (excluding Costas).
pub const DATA_SYMBOLS: usize = 58;

/// Insert 58 data symbols and three Costas arrays into a 79-symbol frame.
///
/// Frame layout:
/// - `[0..7]`   ← Costas array
/// - `[7..36]`  ← `data[0..29]`
/// - `[36..43]` ← Costas array
/// - `[43..72]` ← `data[29..58]`
/// - `[72..79]` ← Costas array
pub fn insert_costas(data: &[f32; DATA_SYMBOLS], frame: &mut [f32; FRAME_SYMBOLS]) {
    for i in 0..7 {
        frame[i] = COSTAS[i] as f32;
        frame[36 + i] = COSTAS[i] as f32;
        frame[72 + i] = COSTAS[i] as f32;
    }
    frame[7..36].copy_from_slice(&data[..29]);
    frame[43..72].copy_from_slice(&data[29..]);
}

/// Extract the 58 data symbols from a 79-symbol frame, stripping Costas arrays.
///
/// Returns `None` if `symbols` is shorter than 79 elements.
pub fn extract_data(symbols: &[f32]) -> Option<[f32; DATA_SYMBOLS]> {
    if symbols.len() < FRAME_SYMBOLS {
        return None;
    }
    let mut data = [0f32; DATA_SYMBOLS];
    data[..29].copy_from_slice(&symbols[7..36]);
    data[29..].copy_from_slice(&symbols[43..72]);
    Some(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_then_extract_roundtrip() {
        let data: [f32; 58] = core::array::from_fn(|i| (i % 8) as f32);
        let mut frame = [0f32; 79];
        insert_costas(&data, &mut frame);
        let recovered = extract_data(&frame).expect("extract should succeed");
        assert_eq!(recovered, data);
    }

    #[test]
    fn costas_positions_correct() {
        let data = [0f32; 58];
        let mut frame = [0f32; 79];
        insert_costas(&data, &mut frame);
        for i in 0..7 {
            assert_eq!(frame[i], COSTAS[i] as f32, "Costas #1 pos {i}");
            assert_eq!(frame[36 + i], COSTAS[i] as f32, "Costas #2 pos {i}");
            assert_eq!(frame[72 + i], COSTAS[i] as f32, "Costas #3 pos {i}");
        }
    }

    #[test]
    fn extract_too_short_returns_none() {
        let short = [0f32; 70];
        assert!(extract_data(&short).is_none());
    }
}
