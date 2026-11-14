# RNNoise fixtures

Golden output of `nnnoiseless` 0.5.2 (`default-features = false`, the crate
and configuration ZER0 runs), used by `tests/rnnoise_unit.cpp`:

- `nnnoiseless.out.i16`: denoised samples, rounded and clamped to int16, raw
  little-endian;
- `nnnoiseless.vad.f32`: one speech probability per 480-sample frame, raw
  little-endian f32.

The input is not stored: `transcribe_rnnoise_unit --dump-input <path>` writes
it (raw f32, 48 kHz, int16 scale). It is `../earshot/ls-6930-75918-0000.wav`
(LibriSpeech, CC BY 4.0) held 3x to 48 kHz plus low-passed noise, so the
fixture is regenerated from a clip already in the tree.

To regenerate, build this harness against the crate and run it on the dumped
input:

```toml
# Cargo.toml
[package]
name = "rnref"
version = "0.1.0"
edition = "2021"

[dependencies]
nnnoiseless = { version = "=0.5.2", default-features = false }
```

```rust
// src/main.rs: rnref <in.f32> <out.i16> <vad.f32>
use nnnoiseless::DenoiseState;
use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let x: Vec<f32> = std::fs::read(&a[1])
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut st = DenoiseState::new();
    let mut out = vec![0f32; DenoiseState::FRAME_SIZE];
    let mut o16 = std::fs::File::create(&a[2]).unwrap();
    let mut ov = std::fs::File::create(&a[3]).unwrap();
    for f in x.chunks_exact(DenoiseState::FRAME_SIZE) {
        let p = st.process_frame(&mut out, f);
        for &s in &out {
            let v = s.round().clamp(-32768.0, 32767.0) as i16;
            o16.write_all(&v.to_le_bytes()).unwrap();
        }
        ov.write_all(&p.to_le_bytes()).unwrap();
    }
}
```

```sh
transcribe_rnnoise_unit --dump-input input.f32
cargo run --release -- input.f32 nnnoiseless.out.i16 nnnoiseless.vad.f32
```
