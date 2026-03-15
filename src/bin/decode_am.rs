use std::f64::consts::FRAC_1_SQRT_2;

use sdr::complex::Complex;
use sdr::iq_file::read_cf32;
use sdr::modulation::am::AmDemodulator;
use sdr::pipeline::am::AmRx;
use sdr::pipeline::Receiver;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Load IQ file ---
    let samples = read_cf32("input.cf32")?;

    // --- Normalize IQ amplitude so carrier ≈ 1.0 ---
    // AmDemodulator computes (|iq| - 1.0) / depth, which requires
    // the mean envelope to be ~1.0.  Raw SDR samples are much smaller.
    let mean_amp = samples.iter().map(|s| s.abs()).sum::<f32>() / samples.len() as f32;
    let normalized: Vec<Complex<f32>> = samples
        .iter()
        .map(|s| Complex::new(s.i / mean_amp, s.q / mean_amp))
        .collect();

    // --- Build receive pipeline ---
    // Station is at 920 kHz, SDR tuned to 900 kHz → station is at +20 kHz.
    // Shift down by -20 kHz to bring it to DC before the envelope detector.
    //
    // TWO-STAGE LOWPASS:
    //  Stage 1 (400 kHz SR, 20 kHz cutoff): anti-alias before downsampling.
    //    Normalized ω_c = 20k/400k = 0.05 — numerically stable for a biquad.
    //    Rejects content above 24 kHz (the 48 kHz output Nyquist) before decimation.
    //  Stage 2 (48 kHz SR, 5 kHz cutoff): tight audio bandwidth filter.
    //    Normalized ω_c = 5k/48k ≈ 0.104 — biquad poles well away from z=1.
    //    Attenuates out-of-band noise/harmonics that survive stage 1.
    //
    // A single 5 kHz lowpass at 400 kHz (ω_c = 0.0125) would have its biquad
    // poles extremely close to z=1, causing numerical instability in f32.
    let mut rx = AmRx::new(AmDemodulator::new(0.8))
        .with_freq_shift(400_000.0, -20_000.0)              // bring station to DC
        .with_lowpass(400_000.0, 20_000.0, FRAC_1_SQRT_2)   // stage 1: anti-alias
        .with_rates(400_000, 48_000)                         // 400 kHz → 48 kHz
        .with_lowpass(48_000.0, 5_000.0, FRAC_1_SQRT_2);    // stage 2: audio BW

    // --- Demodulate ---
    let audio = rx.receive(&normalized);

    // --- Normalize audio to [-1, 1] before writing ---
    let max_abs = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 { 1.0 / max_abs } else { 1.0 };

    // --- Write WAV ---
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("output.wav", spec)?;
    for sample in &audio {
        let s = (sample * scale).clamp(-1.0, 1.0);
        writer.write_sample((s * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;

    println!("Wrote {} audio samples to output.wav", audio.len());
    Ok(())
}
