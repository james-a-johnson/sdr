//! Forward error correction and coding utilities for digital protocols.
//!
//! | Module | Description |
//! |---|---|
//! | [`gray`] | Gray code encoding and decoding for FSK symbol mapping |
//! | [`crc`] | Table-driven CRC-14 for FT8 frame integrity checking |
//! | [`interleave`] | Bit-reversal interleaver / deinterleaver for 174-bit FT8 codewords |
//! | [`ldpc`] | LDPC(174, 91) encoder and min-sum belief propagation decoder |

pub mod crc;
pub mod gray;
pub mod interleave;
pub mod ldpc;
