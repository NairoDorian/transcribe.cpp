// Convert a WAV of any PCM/float format to the 16 kHz mono 16-bit PCM that
// transcribe.cpp accepts. Written because the ZER0 recordings are 48 kHz
// 32-bit WAVE_FORMAT_EXTENSIBLE and neither ffmpeg nor sox is installed here.
//
// Resampling is windowed-sinc low-pass then decimate, not sample dropping:
// dropping every 3rd sample aliases 8-24 kHz content back into the speech band,
// which would make the transcript a test of the resampler rather than of the
// model. The recordings are speech, but sibilants carry real energy up there.
//
// Usage: node wav-to-16k.js <in.wav> <out.wav>
const fs = require("fs");

const WAVE_FORMAT_PCM = 1;
const WAVE_FORMAT_IEEE_FLOAT = 3;
const WAVE_FORMAT_EXTENSIBLE = 0xfffe;

function parseWav(buf) {
  if (buf.toString("ascii", 0, 4) !== "RIFF" || buf.toString("ascii", 8, 12) !== "WAVE")
    throw new Error("not a RIFF/WAVE file");

  let p = 12;
  let fmt = null;
  let data = null;
  while (p + 8 <= buf.length) {
    const id = buf.toString("ascii", p, p + 4);
    const size = buf.readUInt32LE(p + 4);
    const body = p + 8;
    if (id === "fmt ") {
      let tag = buf.readUInt16LE(body);
      const channels = buf.readUInt16LE(body + 2);
      const sampleRate = buf.readUInt32LE(body + 4);
      const bits = buf.readUInt16LE(body + 14);
      if (tag === WAVE_FORMAT_EXTENSIBLE) {
        // The real format lives in the SubFormat GUID's first two bytes; the
        // 0xFFFE tag only says "look in the extension".
        tag = buf.readUInt16LE(body + 24);
      }
      fmt = { tag, channels, sampleRate, bits };
    } else if (id === "data") {
      data = buf.subarray(body, Math.min(body + size, buf.length));
    }
    p = body + size + (size % 2); // chunks are word-aligned
  }
  if (!fmt || !data) throw new Error("missing fmt or data chunk");
  return { fmt, data };
}

function decode(data, fmt) {
  const { tag, bits, channels } = fmt;
  const n = Math.floor(data.length / (bits / 8) / channels);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let c = 0; c < channels; c++) {
      const off = (i * channels + c) * (bits / 8);
      let v;
      if (tag === WAVE_FORMAT_IEEE_FLOAT) v = data.readFloatLE(off);
      else if (bits === 16) v = data.readInt16LE(off) / 32768;
      else if (bits === 32) v = data.readInt32LE(off) / 2147483648;
      else if (bits === 8) v = (data.readUInt8(off) - 128) / 128;
      else throw new Error("unsupported bit depth " + bits);
      acc += v;
    }
    out[i] = acc / channels; // downmix
  }
  return out;
}

// Windowed-sinc low-pass. Blackman window; cutoff just under the output
// Nyquist so the decimation step has nothing left to fold back.
function lowpass(x, cutoffHz, sampleRate) {
  const TAPS = 127;
  const M = TAPS - 1;
  const fc = cutoffHz / sampleRate; // cycles per sample
  const h = new Float64Array(TAPS);
  let sum = 0;
  for (let n = 0; n < TAPS; n++) {
    const k = n - M / 2;
    const sinc = k === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * k) / (Math.PI * k);
    const w =
      0.42 -
      0.5 * Math.cos((2 * Math.PI * n) / M) +
      0.08 * Math.cos((4 * Math.PI * n) / M);
    h[n] = sinc * w;
    sum += h[n];
  }
  for (let n = 0; n < TAPS; n++) h[n] /= sum; // unity DC gain

  const y = new Float32Array(x.length);
  const half = M >> 1;
  for (let i = 0; i < x.length; i++) {
    let acc = 0;
    for (let n = 0; n < TAPS; n++) {
      const j = i + n - half;
      if (j >= 0 && j < x.length) acc += x[j] * h[n];
    }
    y[i] = acc;
  }
  return y;
}

function resample(x, from, to) {
  if (from === to) return x;
  // Only integer decimation is needed here (48k -> 16k). Guard the assumption
  // rather than silently producing a wrong sample rate via naive decimation.
  if (from % to !== 0) throw new Error(`non-integer ratio ${from}->${to} unsupported`);
  const filtered = lowpass(x, to / 2 * 0.9, from);
  const factor = from / to;
  const n = Math.floor(x.length / factor);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = filtered[Math.round(i * factor)];
  return out;
}

function writeWav16(path, samples, sampleRate) {
  const dataBytes = samples.length * 2;
  const buf = Buffer.alloc(44 + dataBytes);
  buf.write("RIFF", 0, "ascii");
  buf.writeUInt32LE(36 + dataBytes, 4);
  buf.write("WAVE", 8, "ascii");
  buf.write("fmt ", 12, "ascii");
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(1, 22); // mono
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write("data", 36, "ascii");
  buf.writeUInt32LE(dataBytes, 40);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    buf.writeInt16LE(Math.round(v * 32767), 44 + i * 2);
  }
  fs.writeFileSync(path, buf);
  return buf.length;
}

const [inPath, outPath] = process.argv.slice(2);
const { fmt, data } = parseWav(fs.readFileSync(inPath));
const samples = decode(data, fmt);
const out = resample(samples, fmt.sampleRate, 16000);
const bytes = writeWav16(outPath, out, 16000);
console.log(
  `${inPath} -> ${outPath}  ${fmt.sampleRate}Hz/${fmt.bits}bit/${fmt.channels}ch ` +
    `(${(samples.length / fmt.sampleRate).toFixed(2)}s) -> 16000Hz/16bit/1ch ` +
    `(${(out.length / 16000).toFixed(2)}s, ${bytes} bytes)`,
);
