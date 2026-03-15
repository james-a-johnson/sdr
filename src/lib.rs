//! A minimal software-defined radio (SDR) library.
//!
//! Provides the core building blocks for processing IQ sample streams:
//!
//! - [`complex`] — the [`Complex<T>`](complex::Complex) carrier type used throughout
//! - [`fft`] — radix-2 Cooley-Tukey FFT for spectrum analysis
//! - [`filter`] — streaming filters: biquad IIR, frequency shift, and decimation
//! - [`modulation`] — AM, FM, and 8-FSK modulation and demodulation
//! - [`iq_file`] — reading and writing raw interleaved f32 IQ files

pub mod coding;
pub mod complex;
pub mod fft;
pub mod filter;
pub mod ft8;
pub mod iq_file;
pub mod modulation;
pub mod pipeline;
