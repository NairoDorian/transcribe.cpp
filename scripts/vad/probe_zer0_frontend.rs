#![allow(dead_code)]

mod denoise;
mod resampler;

use rubato::{Fft, FixedSync, Resampler};
use std::time::Duration;

fn main() {
    println!("geometry,input_hz,output_hz,requested_chunk,input_next,output_next,output_delay");
    for (input, output) in [
        (16000, 48000),
        (44100, 48000),
        (48000, 16000),
        (44100, 16000),
    ] {
        for chunk in [256, 512, 1024] {
            let r = Fft::<f32>::new(input, output, chunk, 1, FixedSync::Input).unwrap();
            println!(
                "geometry,{input},{output},{chunk},{},{},{}",
                r.input_frames_next(),
                r.output_frames_next(),
                r.output_delay()
            );
        }
    }
    println!("pipeline,input_hz,first_emit_source_ms,output_samples,impulse_peak_output_sample,impulse_extra_ms");
    for rate in [16000usize, 44100, 48000] {
        let mut input = vec![0.0f32; rate * 2];
        input[rate / 10] = 0.5;
        for mode in ["direct", "denoise_strength_zero"] {
            let mut output = Vec::new();
            let mut first_emit = None;
            let mut consumed = 0usize;
            let mut observe = |frame: &[f32], consumed: usize| {
                if first_emit.is_none() {
                    first_emit = Some(consumed);
                }
                output.extend_from_slice(frame);
            };
            let chunk = rate / 100; // synthetic 10 ms capture deliveries
            if mode == "direct" {
                let mut r = resampler::FrameResampler::new(rate, 16000, Duration::from_millis(16));
                for data in input.chunks(chunk) {
                    consumed += data.len();
                    r.push(data, |frame| observe(frame, consumed));
                }
                r.finish(|frame| observe(frame, consumed));
            } else {
                let mut r = denoise::DenoiseChain::new(rate, 16000, Duration::from_millis(16));
                let params = denoise::DenoiseParams {
                    strength: 0.0,
                    vad_threshold: 0.0,
                    vad_grace_ms: 200,
                };
                for data in input.chunks(chunk) {
                    consumed += data.len();
                    r.push(data, params, |_| {}, |frame, _| observe(frame, consumed));
                }
                r.finish(params, |_| {}, |frame, _| observe(frame, consumed));
            }
            let peak = output
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                .unwrap()
                .0;
            let extra_ms = (peak as f64 - 1600.0) / 16.0;
            println!(
                "{mode},{rate},{:.3},{},{peak},{extra_ms:.3}",
                first_emit.unwrap() as f64 / rate as f64 * 1000.0,
                output.len()
            );
        }
    }
}
