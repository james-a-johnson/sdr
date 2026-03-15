use std::collections::VecDeque;
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
    filter::{Filter, iir::Iir},
    modulation::fm::{FmDemodulator, FmModulator},
};

const SAMPLE_RATE: f32 = 48000.0;
const DEVIATION: f32 = 5000.0;
const TONE_FREQ: f32 = 1000.0;
const FFT_SIZE: usize = 1024;
const SAMPLES_PER_FRAME: usize = 1600; // 48000 / 30

struct App {
    modulator: FmModulator,
    demodulator: FmDemodulator,
    lp_filter: Iir<f32>,
    iq_buffer: VecDeque<Complex<f32>>,
    spectrum: Vec<f32>,
    baseband_out: Vec<f32>,
    sample_counter: usize,
    scroll_offset: f64,
    zoom: f64,
}

impl App {
    fn new() -> Self {
        Self {
            modulator: FmModulator::new(SAMPLE_RATE, DEVIATION),
            demodulator: FmDemodulator::new(SAMPLE_RATE, DEVIATION),
            lp_filter: Iir::lowpass(SAMPLE_RATE as f64, 4000.0, 0.707),
            iq_buffer: VecDeque::with_capacity(FFT_SIZE),
            spectrum: vec![-120.0; FFT_SIZE],
            baseband_out: Vec::new(),
            sample_counter: 0,
            scroll_offset: 0.0,
            zoom: 1.0,
        }
    }

    fn tick(&mut self) {
        // Generate baseband sine samples
        let baseband: Vec<f32> = (0..SAMPLES_PER_FRAME)
            .map(|i| {
                let t = (self.sample_counter + i) as f32 / SAMPLE_RATE;
                (2.0 * PI * TONE_FREQ * t).sin()
            })
            .collect();
        self.sample_counter += SAMPLES_PER_FRAME;

        // FM modulate
        let iq = self.modulator.modulate(&baseband);

        // Update IQ buffer (keep last FFT_SIZE samples)
        for sample in &iq {
            if self.iq_buffer.len() >= FFT_SIZE {
                self.iq_buffer.pop_front();
            }
            self.iq_buffer.push_back(*sample);
        }

        // Compute FFT spectrum if we have enough samples
        if self.iq_buffer.len() == FFT_SIZE {
            let fft_input: Vec<Complex<f32>> = self.iq_buffer.iter().cloned().collect();
            let spectrum = fft(&fft_input);
            self.spectrum = magnitude_db(&spectrum);
        }

        // FM demodulate
        let demod = self.demodulator.demodulate(&iq);

        // IIR lowpass filter
        let complex_demod: Vec<Complex<f32>> =
            demod.iter().map(|&s| Complex::new(s, 0.0)).collect();
        let filtered = self.lp_filter.filter(&complex_demod);
        self.baseband_out = filtered.iter().map(|c| c.i).collect();
    }
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
) -> io::Result<()> {
    let frame_duration = Duration::from_millis(33);

    loop {
        let frame_start = Instant::now();

        app.tick();

        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(40),
                    Constraint::Percentage(55),
                    Constraint::Percentage(5),
                ])
                .split(f.area());

            // IQ Time Domain
            let display_len = (256.0 / app.zoom) as usize;
            let iq_points_i: Vec<(f64, f64)> = app
                .iq_buffer
                .iter()
                .take(display_len)
                .enumerate()
                .map(|(i, c)| (i as f64 + app.scroll_offset, c.i as f64))
                .collect();
            let iq_points_q: Vec<(f64, f64)> = app
                .iq_buffer
                .iter()
                .take(display_len)
                .enumerate()
                .map(|(i, c)| (i as f64 + app.scroll_offset, c.q as f64))
                .collect();

            let x_max = app.scroll_offset + display_len as f64;
            let iq_datasets = vec![
                Dataset::default()
                    .name("I")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Cyan))
                    .data(&iq_points_i),
                Dataset::default()
                    .name("Q")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Yellow))
                    .data(&iq_points_q),
            ];

            let iq_chart = Chart::new(iq_datasets)
                .block(
                    Block::default()
                        .title("IQ Time Domain")
                        .borders(Borders::ALL),
                )
                .x_axis(
                    Axis::default()
                        .bounds([app.scroll_offset, x_max])
                        .labels(vec![
                            Line::raw(format!("{:.0}", app.scroll_offset)),
                            Line::raw(format!("{:.0}", x_max)),
                        ]),
                )
                .y_axis(Axis::default().bounds([-1.2, 1.2]).labels(vec![
                    Line::raw("-1.0"),
                    Line::raw("0"),
                    Line::raw("1.0"),
                ]));
            f.render_widget(iq_chart, chunks[0]);

            // FFT Spectrum
            let spectrum_points: Vec<(f64, f64)> = app
                .spectrum
                .iter()
                .enumerate()
                .map(|(i, &db)| (i as f64, db as f64))
                .collect();

            let fft_datasets = vec![
                Dataset::default()
                    .name("dBFS")
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(Color::Green))
                    .data(&spectrum_points),
            ];

            let fft_chart = Chart::new(fft_datasets)
                .block(
                    Block::default()
                        .title("FFT Spectrum (dB)")
                        .borders(Borders::ALL),
                )
                .x_axis(Axis::default().bounds([0.0, FFT_SIZE as f64]).labels(vec![
                    Line::raw("0"),
                    Line::raw("512"),
                    Line::raw("1024"),
                ]))
                .y_axis(Axis::default().bounds([-120.0, 0.0]).labels(vec![
                    Line::raw("-120"),
                    Line::raw("-60"),
                    Line::raw("0"),
                ]));
            f.render_widget(fft_chart, chunks[1]);

            // Status bar
            let help = Block::default()
                .title("[q]uit  [←→] scroll  [↑↓] zoom")
                .borders(Borders::NONE);
            f.render_widget(help, chunks[2]);
        })?;

        // Poll for input
        let elapsed = frame_start.elapsed();
        let timeout = frame_duration.saturating_sub(elapsed);
        if event::poll(timeout)?
            && let Event::Key(key) = event::read()?
        {
            match key.code {
                KeyCode::Char('q') => return Ok(()),
                KeyCode::Left => app.scroll_offset = (app.scroll_offset - 16.0).max(0.0),
                KeyCode::Right => app.scroll_offset += 16.0,
                KeyCode::Up => app.zoom = (app.zoom * 1.25).min(8.0),
                KeyCode::Down => app.zoom = (app.zoom / 1.25).max(0.25),
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

    // Cleanup on panic
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_new_and_tick() {
        let mut app = App::new();
        for _ in 0..10 {
            app.tick();
        }
        assert!(!app.spectrum.is_empty());
        assert!(!app.baseband_out.is_empty());
        // No NaN in spectrum
        for &v in &app.spectrum {
            assert!(!v.is_nan(), "NaN in spectrum");
        }
    }
}
