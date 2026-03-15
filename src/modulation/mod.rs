//! Modulation and demodulation schemes.
//!
//! Provides AM (amplitude modulation) via the [`am`] module and FM (frequency
//! modulation) via the [`fm`] module. All modulators and demodulators operate
//! on concrete `f32` samples, which is the standard representation for SDR
//! baseband processing.

pub mod am;
pub mod fm;
