//! A minimal software-defined radio (SDR) library.
//!
//! Provides the core building blocks for processing IQ sample streams:
//!
//! - [`complex`] — the [`Complex<T>`](complex::Complex) carrier type used throughout
//! - [`fft`] — radix-2 Cooley-Tukey FFT for spectrum analysis
//! - [`filter`] — streaming filters: biquad IIR, frequency shift, and decimation
//! - [`modulation`] — FM modulation and demodulation

pub mod complex;
pub mod fft;
pub mod filter;
pub mod modulation;
