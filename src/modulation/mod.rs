//! Modulation and demodulation schemes.
//!
//! Currently provides FM (frequency modulation) via the [`fm`] module.
//! All modulators and demodulators operate on concrete `f32` samples, which
//! is the standard representation for SDR baseband processing.

pub mod fm;
