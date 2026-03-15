/// FFT demo: generates a mixture of sine tones whose amplitudes slowly pulse,
/// feeds them through the FFT every frame, and displays both the time-domain
/// waveform and the frequency spectrum in a TUI.
///
/// Run with:  cargo run --bin fft_demo
use std::f32::consts::PI;
use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, DisableMouseCapture, Event, KeyCode},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    symbols,
    text::Line,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType},
};

use sdr::{
    complex::Complex,
    fft::{fft, magnitude_db},
};

const SAMPLE_RATE: f32 = 48_000.0;
const FFT_SIZE: usize = 1024;
// Hz per FFT bin
const BIN_HZ: f32 = SAMPLE_RATE / FFT_SIZE as f32;
// Samples generated per frame (targets ~30 fps)
const SAMPLES_PER_FRAME: usize = 1600;
// How many samples to show in the time-domain view
const TIME_DISPLAY: usize = 256;

/// Three tones that each pulse independently at different rates.
struct Tone {
    freq_hz: f32,
    /// How fast this tone's amplitude oscillates (Hz in "animation space")
    pulse_rate: f32,
    /// Peak amplitude
    peak: f32,
}

impl Tone {
    fn amplitude(&self, t_anim: f32) -> f32 {
        // Smoothly oscillate between 0 and peak
        self.peak * 0.5 * (1.0 + (2.0 * PI * self.pulse_rate * t_anim).sin())
    }

    fn sample(&self, t_signal: f32, t_anim: f32) -> f32 {
        self.amplitude(t_anim) * (2.0 * PI * self.freq_hz * t_signal).sin()
    }
}

struct App {
    tones: Vec<Tone>,
    /// Total samples generated so far (drives signal phase)
    sample_counter: usize,
    /// Wall-clock seconds elapsed (drives amplitude pulsing)
    anim_time: f32,
    /// Most recent TIME_DISPLAY signal samples for the time-domain chart
    waveform: Vec<f32>,
    /// Most recent FFT magnitude spectrum (dBFS), length = FFT_SIZE
    spectrum: Vec<f32>,
    /// Top of the spectrum y-axis (dB)
    y_max: f32,
    /// Bottom of the spectrum y-axis (dB)
    y_min: f32,
    /// When true, y_max tracks the peak of the spectrum automatically
    auto_scale: bool,
}

impl App {
    fn new() -> Self {
        Self {
            tones: vec![
                Tone {
                    freq_hz: 2_000.0,
                    pulse_rate: 0.3,
                    peak: 0.8,
                },
                Tone {
                    freq_hz: 6_500.0,
                    pulse_rate: 0.7,
                    peak: 0.6,
                },
                Tone {
                    freq_hz: 11_200.0,
                    pulse_rate: 1.1,
                    peak: 0.5,
                },
                Tone {
                    freq_hz: 18_000.0,
                    pulse_rate: 0.5,
                    peak: 0.4,
                },
            ],
            sample_counter: 0,
            anim_time: 0.0,
            waveform: vec![0.0; TIME_DISPLAY],
            spectrum: vec![-120.0; FFT_SIZE],
            y_max: 0.0,
            y_min: -120.0,
            auto_scale: true,
        }
    }

    fn db_range(&self) -> f32 {
        self.y_max - self.y_min
    }

    fn tick(&mut self, dt: f32) {
        self.anim_time += dt;

        // Generate one frame of signal samples
        let mut frame: Vec<f32> = (0..SAMPLES_PER_FRAME)
            .map(|i| {
                let t_signal = (self.sample_counter + i) as f32 / SAMPLE_RATE;
                self.tones
                    .iter()
                    .map(|tone| tone.sample(t_signal, self.anim_time))
                    .sum()
            })
            .collect();
        self.sample_counter += SAMPLES_PER_FRAME;

        // Update rolling waveform display (keep the last TIME_DISPLAY samples)
        let tail = frame.len().min(TIME_DISPLAY);
        let new_tail = &frame[frame.len() - tail..];
        let keep = TIME_DISPLAY.saturating_sub(tail);
        self.waveform.drain(..TIME_DISPLAY - keep);
        self.waveform.extend_from_slice(new_tail);

        // FFT: use the last FFT_SIZE samples from frame (zero-pad if shorter)
        frame.resize(FFT_SIZE, 0.0);
        let fft_input: Vec<Complex<f32>> = frame[frame.len() - FFT_SIZE..]
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        let spectrum_complex = fft(&fft_input);
        self.spectrum = magnitude_db(&spectrum_complex);

        if self.auto_scale {
            // Find the peak dB in the positive-frequency half
            let peak = self.spectrum[..FFT_SIZE / 2]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            // Ceiling sits 6 dB above the peak, rounded up to nearest 10 dB
            let ceiling = ((peak + 6.0) / 10.0).ceil() * 10.0;
            // Smoothly chase the ceiling — fast attack, slow release
            let alpha = if ceiling > self.y_max {
                1.0
            } else {
                0.05_f32.powf(dt)
            };
            self.y_max = self.y_max + alpha * (ceiling - self.y_max);
            self.y_min = self.y_max - self.db_range();
        }
    }
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
) -> io::Result<()> {
    let frame_duration = Duration::from_millis(33);
    let mut last_frame = Instant::now();

    loop {
        let now = Instant::now();
        let dt = now.duration_since(last_frame).as_secs_f32();
        last_frame = now;

        app.tick(dt);

        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(45),
                    Constraint::Percentage(50),
                    Constraint::Percentage(5),
                ])
                .split(f.area());

            // ── Time domain ─────────────────────────────────────────────────
            let waveform_data: Vec<(f64, f64)> = app
                .waveform
                .iter()
                .enumerate()
                .map(|(i, &s)| (i as f64, s as f64))
                .collect();

            let time_chart = Chart::new(vec![
                Dataset::default()
                    .name("signal")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Cyan))
                    .data(&waveform_data),
            ])
            .block(Block::default().title("Time Domain").borders(Borders::ALL))
            .x_axis(
                Axis::default()
                    .bounds([0.0, TIME_DISPLAY as f64])
                    .labels(vec![Line::raw("0"), Line::raw("128"), Line::raw("256")]),
            )
            .y_axis(Axis::default().bounds([-2.0, 2.0]).labels(vec![
                Line::raw("-2"),
                Line::raw("0"),
                Line::raw("2"),
            ]));
            f.render_widget(time_chart, chunks[0]);

            // ── FFT spectrum ─────────────────────────────────────────────────
            // Only show the positive-frequency half (bins 0..FFT_SIZE/2)
            let half = FFT_SIZE / 2;
            let spectrum_data: Vec<(f64, f64)> = app.spectrum[..half]
                .iter()
                .enumerate()
                .map(|(bin, &db)| (bin as f64 * BIN_HZ as f64, db as f64))
                .collect();

            let nyquist = SAMPLE_RATE as f64 / 2.0;
            let y_min = app.y_min as f64;
            let y_max = app.y_max as f64;
            let y_mid = (y_min + y_max) / 2.0;
            let auto_indicator = if app.auto_scale { "auto" } else { "manual" };
            let fft_title = format!("FFT Spectrum (dBFS) [{auto_indicator}]");

            let fft_chart = Chart::new(vec![
                Dataset::default()
                    .name("dBFS")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Green))
                    .data(&spectrum_data),
            ])
            .block(Block::default().title(fft_title).borders(Borders::ALL))
            .x_axis(Axis::default().bounds([0.0, nyquist]).labels(vec![
                Line::raw("0 Hz"),
                Line::raw("12 kHz"),
                Line::raw("24 kHz"),
            ]))
            .y_axis(Axis::default().bounds([y_min, y_max]).labels(vec![
                Line::raw(format!("{:.0}", y_min)),
                Line::raw(format!("{:.0}", y_mid)),
                Line::raw(format!("{:.0}", y_max)),
            ]));
            f.render_widget(fft_chart, chunks[1]);

            // ── Status bar ───────────────────────────────────────────────────
            let tones_str = app
                .tones
                .iter()
                .map(|t| format!("{:.0}Hz={:.2}", t.freq_hz, t.amplitude(app.anim_time)))
                .collect::<Vec<_>>()
                .join("  ");
            let help = Block::default()
                .title(format!(
                    "[q]uit  [↑↓]shift  [+/-]range  [a]uto   {}",
                    tones_str
                ))
                .borders(Borders::NONE);
            f.render_widget(help, chunks[2]);
        })?;

        let elapsed = last_frame.elapsed();
        let timeout = frame_duration.saturating_sub(elapsed);
        if event::poll(timeout)?
            && let Event::Key(key) = event::read()?
        {
            match key.code {
                KeyCode::Char('q') => return Ok(()),
                // Shift window up/down by 10 dB
                KeyCode::Up => {
                    app.auto_scale = false;
                    app.y_max += 10.0;
                    app.y_min += 10.0;
                }
                KeyCode::Down => {
                    app.auto_scale = false;
                    app.y_max -= 10.0;
                    app.y_min -= 10.0;
                }
                // Expand/shrink the dB range (floor moves, ceiling stays)
                KeyCode::Char('+') | KeyCode::Char('=') => {
                    app.auto_scale = false;
                    app.y_min -= 20.0;
                }
                KeyCode::Char('-') => {
                    app.auto_scale = false;
                    if app.db_range() > 20.0 {
                        app.y_min += 20.0;
                    }
                }
                // Snap back to auto-scaling
                KeyCode::Char('a') => {
                    app.auto_scale = true;
                }
                _ => {}
            }
        }
    }
}

fn main() -> io::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, DisableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(info);
    }));

    let mut app = App::new();
    let result = run_app(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}
