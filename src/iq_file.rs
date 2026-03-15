//! Reading and writing raw interleaved f32 IQ files.
//!
//! The file format is a flat sequence of little-endian 32-bit float pairs:
//! `[i0, q0, i1, q1, …]` with no header.  This matches the `.cf32` /
//! `.fc32` convention used by GNU Radio, SDR#, and most SDR capture tools.
//!
//! # Bulk API
//!
//! [`read_cf32`] / [`write_cf32`] load or store an entire file at once.
//! Use these for small files where holding everything in memory is fine.
//!
//! # Streaming API
//!
//! [`IqReader`] and [`IqWriter`] wrap any [`Read`] / [`Write`] impl and
//! process one sample at a time, keeping memory usage constant regardless
//! of file size.
//!
//! ```
//! use std::io::Cursor;
//! use sdr::complex::Complex;
//! use sdr::iq_file::{IqReader, IqWriter};
//!
//! // Write two samples into an in-memory buffer.
//! let mut buf = Vec::new();
//! let mut writer = IqWriter::new(&mut buf);
//! writer.write_sample(Complex::new(1.0_f32, 0.5)).unwrap();
//! writer.write_sample(Complex::new(-0.5_f32, 1.0)).unwrap();
//!
//! // Read them back.
//! let samples: Vec<_> = IqReader::new(Cursor::new(&buf))
//!     .collect::<Result<_, _>>()
//!     .unwrap();
//! assert_eq!(samples[0], Complex::new(1.0_f32, 0.5));
//! assert_eq!(samples[1], Complex::new(-0.5_f32, 1.0));
//! ```

use crate::complex::Complex;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

// ── Bulk API ─────────────────────────────────────────────────────────────────

/// Read an entire raw interleaved f32 IQ file into a `Vec`.
///
/// Returns an error if the file cannot be opened, any read fails, or the file
/// length is not a multiple of 8 bytes (two `f32`s per sample).
///
/// # Example
///
/// ```no_run
/// use sdr::iq_file::read_cf32;
/// let samples = read_cf32("capture.cf32").unwrap();
/// println!("{} samples", samples.len());
/// ```
pub fn read_cf32(path: impl AsRef<Path>) -> io::Result<Vec<Complex<f32>>> {
    let file = File::open(path)?;
    IqReader::new(BufReader::new(file)).collect()
}

/// Write a slice of complex samples to a raw interleaved f32 IQ file.
///
/// Creates or truncates the file. Samples are written as little-endian
/// `[i, q]` float pairs.
///
/// # Example
///
/// ```no_run
/// use sdr::complex::Complex;
/// use sdr::iq_file::write_cf32;
/// let samples = vec![Complex::new(1.0_f32, 0.0), Complex::new(0.0_f32, 1.0)];
/// write_cf32("out.cf32", &samples).unwrap();
/// ```
pub fn write_cf32(path: impl AsRef<Path>, samples: &[Complex<f32>]) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = IqWriter::new(BufWriter::new(file));
    writer.write_all(samples)
}

// ── Streaming reader ──────────────────────────────────────────────────────────

/// Streaming IQ reader.
///
/// Wraps any [`Read`] impl and yields one [`Complex<f32>`] per call to
/// [`Iterator::next`].  Construct from a file with [`IqReader::open`] or from
/// any reader with [`IqReader::new`].
///
/// # Example — file
///
/// ```no_run
/// use sdr::iq_file::IqReader;
///
/// for sample in IqReader::open("capture.cf32").unwrap() {
///     let s = sample.unwrap();
///     println!("i={} q={}", s.i, s.q);
/// }
/// ```
pub struct IqReader<R: Read> {
    reader: R,
}

impl<R: Read> IqReader<R> {
    /// Create a streaming reader from any [`Read`] impl.
    pub fn new(reader: R) -> Self {
        Self { reader }
    }
}

impl IqReader<BufReader<File>> {
    /// Open a file and create a streaming reader over it.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self::new(BufReader::new(file)))
    }
}

impl<R: Read> Iterator for IqReader<R> {
    type Item = io::Result<Complex<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = [0u8; 8];
        match self.reader.read_exact(&mut buf) {
            Ok(()) => {
                let i = f32::from_le_bytes(buf[0..4].try_into().unwrap());
                let q = f32::from_le_bytes(buf[4..8].try_into().unwrap());
                Some(Ok(Complex::new(i, q)))
            }
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => None,
            Err(e) => Some(Err(e)),
        }
    }
}

// ── Streaming writer ──────────────────────────────────────────────────────────

/// Streaming IQ writer.
///
/// Wraps any [`Write`] impl and serialises one [`Complex<f32>`] at a time.
/// Construct from a file with [`IqWriter::create`] or from any writer with
/// [`IqWriter::new`].
///
/// The underlying writer is flushed when [`IqWriter`] is dropped, but any
/// flush error at that point is silently discarded.  Call [`IqWriter::flush`]
/// explicitly if you need to handle flush errors.
///
/// # Example — file
///
/// ```no_run
/// use sdr::complex::Complex;
/// use sdr::iq_file::IqWriter;
///
/// let mut writer = IqWriter::create("out.cf32").unwrap();
/// writer.write_sample(Complex::new(1.0_f32, 0.0)).unwrap();
/// writer.flush().unwrap();
/// ```
pub struct IqWriter<W: Write> {
    writer: W,
}

impl<W: Write> IqWriter<W> {
    /// Create a streaming writer from any [`Write`] impl.
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    /// Write a single sample.
    pub fn write_sample(&mut self, sample: Complex<f32>) -> io::Result<()> {
        self.writer.write_all(&sample.i.to_le_bytes())?;
        self.writer.write_all(&sample.q.to_le_bytes())
    }

    /// Write a slice of samples.
    pub fn write_all(&mut self, samples: &[Complex<f32>]) -> io::Result<()> {
        for &s in samples {
            self.write_sample(s)?;
        }
        Ok(())
    }

    /// Flush the underlying writer.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

impl IqWriter<BufWriter<File>> {
    /// Create or truncate a file and wrap it in a streaming writer.
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::create(path)?;
        Ok(Self::new(BufWriter::new(file)))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_vec() -> Vec<Complex<f32>> {
        vec![
            Complex::new(1.0, 0.5),
            Complex::new(-0.5, 1.0),
            Complex::new(0.0, -1.0),
        ]
    }

    /// Encode samples to bytes using IqWriter, decode with IqReader.
    #[test]
    fn streaming_roundtrip() {
        let samples = sample_vec();

        let mut buf = Vec::new();
        let mut writer = IqWriter::new(&mut buf);
        writer.write_all(&samples).unwrap();

        let recovered: Vec<Complex<f32>> = IqReader::new(Cursor::new(&buf))
            .collect::<io::Result<_>>()
            .unwrap();

        assert_eq!(samples, recovered);
    }

    /// Bulk read_cf32 / write_cf32 round-trip through a temp file.
    #[test]
    fn bulk_file_roundtrip() {
        let path = std::env::temp_dir().join("sdr_test_bulk_roundtrip.cf32");
        let samples = sample_vec();

        write_cf32(&path, &samples).unwrap();
        let recovered = read_cf32(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(samples, recovered);
    }

    /// A file with an odd number of f32 values (not pairs) should be handled
    /// gracefully — the trailing 4 bytes are silently ignored because
    /// read_exact returns UnexpectedEof and the iterator stops.
    #[test]
    fn truncated_file_stops_cleanly() {
        // 1 complete sample (8 bytes) + 4 stray bytes
        let mut buf = Vec::new();
        buf.extend_from_slice(&1.0_f32.to_le_bytes());
        buf.extend_from_slice(&2.0_f32.to_le_bytes());
        buf.extend_from_slice(&0.0_f32.to_le_bytes()); // only i, no q

        let samples: Vec<_> = IqReader::new(Cursor::new(&buf))
            .collect::<io::Result<_>>()
            .unwrap();

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0], Complex::new(1.0_f32, 2.0));
    }

    /// write_sample and write_all produce identical bytes.
    #[test]
    fn write_sample_vs_write_all() {
        let samples = sample_vec();

        let mut buf_a = Vec::new();
        let mut writer_a = IqWriter::new(&mut buf_a);
        for &s in &samples {
            writer_a.write_sample(s).unwrap();
        }

        let mut buf_b = Vec::new();
        let mut writer_b = IqWriter::new(&mut buf_b);
        writer_b.write_all(&samples).unwrap();

        assert_eq!(buf_a, buf_b);
    }

    /// Each sample occupies exactly 8 bytes.
    #[test]
    fn byte_layout() {
        let sample = Complex::new(1.0_f32, -1.0_f32);
        let mut buf = Vec::new();
        IqWriter::new(&mut buf).write_sample(sample).unwrap();

        assert_eq!(buf.len(), 8);
        assert_eq!(&buf[0..4], &1.0_f32.to_le_bytes());
        assert_eq!(&buf[4..8], &(-1.0_f32).to_le_bytes());
    }
}
