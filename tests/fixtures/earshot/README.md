# Earshot fixtures

- `ls-6930-75918-0000.wav`, `2086-149220-0033.wav`: LibriSpeech (CC BY 4.0),
  16 kHz mono, the two clips of audio.cpp's Earshot parity test.
- `*.scores.txt`: per-frame scores of the upstream Rust Earshot (rev ffd32d4,
  same network and weights as the 1.2.2 crate) for those files, recorded by
  audio.cpp (`tests/unittests/assets/earshot_vad/upstream_reference.json`).

Used by `tests/earshot_engine_unit.cpp`.
