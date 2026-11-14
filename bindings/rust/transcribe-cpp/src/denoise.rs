//! Engine-native noise suppression (`transcribe_denoise_*`): classic RNNoise
//! with its bundled model on 10 ms frames of 48 kHz mono audio, plus the
//! wet/dry mix and speech-probability gate a capture chain applies around it.
//! The algorithm of the `nnnoiseless` crate, output within one int16 step of
//! it, about 1.4x faster on x86-64.

use transcribe_cpp_sys as sys;

use crate::error::{check, error_for_status, Result};

/// The only sample rate RNNoise is trained for.
pub const DENOISE_SAMPLE_RATE: u32 = 48_000;
/// Samples per frame: 10 ms at 48 kHz.
pub const DENOISE_FRAME_SAMPLES: usize = 480;
/// Longest grace period accepted.
pub const DENOISE_MAX_GRACE_MS: u32 = 5000;

/// The suppressor's tunables; all apply from the next frame on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DenoiseParams {
    /// Wet/dry mix, 0 ..= 1: 1 = RNNoise's output, 0 = the input untouched
    /// (one frame late, like the wet side).
    pub strength: f32,
    /// RNNoise's own speech probability below which frames are muted,
    /// 0 ..= 1; 0 turns the gate off. The gain slews instead of cutting.
    pub vad_threshold: f32,
    /// How long audio keeps passing after the last frame at or above the
    /// threshold, 0 ..= [`DENOISE_MAX_GRACE_MS`] (whole 10 ms frames).
    pub vad_grace_ms: u32,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            strength: 1.0,
            vad_threshold: 0.0,
            vad_grace_ms: 200,
        }
    }
}

impl DenoiseParams {
    /// Clamp into the accepted ranges, non-finite floats falling back to the
    /// defaults: for values straight from a settings slider.
    pub fn normalized(self) -> Self {
        let d = Self::default();
        let finite = |v: f32, fallback: f32| if v.is_finite() { v } else { fallback };
        Self {
            strength: finite(self.strength, d.strength).clamp(0.0, 1.0),
            vad_threshold: finite(self.vad_threshold, d.vad_threshold).clamp(0.0, 1.0),
            vad_grace_ms: self.vad_grace_ms.min(DENOISE_MAX_GRACE_MS),
        }
    }

    fn to_sys(self) -> sys::transcribe_denoise_params {
        let mut raw = std::mem::MaybeUninit::<sys::transcribe_denoise_params>::uninit();
        let mut raw = unsafe {
            sys::transcribe_denoise_params_init(raw.as_mut_ptr());
            raw.assume_init()
        };
        raw.strength = self.strength;
        raw.vad_threshold = self.vad_threshold;
        raw.vad_grace_ms = self.vad_grace_ms;
        raw
    }
}

/// One noise suppressor per audio source. Each call takes one
/// [`DENOISE_FRAME_SAMPLES`]-sample frame in [-1, 1] and returns one, one
/// frame behind (overlap-add synthesis): end a stream with
/// [`Denoiser::flush`]. Output is clamped to [-1, 1].
#[derive(Debug)]
pub struct Denoiser {
    ptr: *mut sys::transcribe_denoise,
}

unsafe impl Send for Denoiser {}

impl Drop for Denoiser {
    fn drop(&mut self) {
        unsafe { sys::transcribe_denoise_free(self.ptr) };
    }
}

impl Denoiser {
    /// Fails (`InvalidArg`) on a non-finite or out-of-range field; see
    /// [`DenoiseParams::normalized`].
    pub fn new(params: &DenoiseParams) -> Result<Self> {
        let raw = params.to_sys();
        let mut ptr = std::ptr::null_mut();
        check(
            unsafe { sys::transcribe_denoise_init(&raw, &mut ptr) },
            "transcribe_denoise_init",
        )?;
        if ptr.is_null() {
            return Err(error_for_status(
                sys::transcribe_status::TRANSCRIBE_ERR_OOM,
                "transcribe_denoise_init",
            ));
        }
        Ok(Self { ptr })
    }

    /// New params from the next frame on, mid-recording included.
    pub fn set_params(&mut self, params: &DenoiseParams) -> Result<()> {
        let raw = params.to_sys();
        check(
            unsafe { sys::transcribe_denoise_set_params(self.ptr, &raw) },
            "transcribe_denoise_set_params",
        )
    }

    /// Start a new recording: network state, dry delay and gate; keeps params.
    pub fn reset(&mut self) {
        unsafe { sys::transcribe_denoise_reset(self.ptr) };
    }

    /// One frame. Returns RNNoise's speech probability for `input` (0 when
    /// its 20 ms analysis window is near silent).
    pub fn process_frame(
        &mut self,
        input: &[f32; DENOISE_FRAME_SAMPLES],
        output: &mut [f32; DENOISE_FRAME_SAMPLES],
    ) -> Result<f32> {
        let mut prob = 0.0f32;
        check(
            unsafe {
                sys::transcribe_denoise_process_frame(
                    self.ptr,
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    &mut prob,
                )
            },
            "transcribe_denoise_process_frame",
        )?;
        Ok(prob)
    }

    /// [`Denoiser::process_frame`] in place.
    pub fn process_frame_in_place(
        &mut self,
        frame: &mut [f32; DENOISE_FRAME_SAMPLES],
    ) -> Result<f32> {
        let mut prob = 0.0f32;
        let ptr = frame.as_mut_ptr();
        check(
            unsafe { sys::transcribe_denoise_process_frame(self.ptr, ptr, ptr, &mut prob) },
            "transcribe_denoise_process_frame",
        )?;
        Ok(prob)
    }

    /// End of input: one silent frame pushes the last real frame out.
    pub fn flush(&mut self, output: &mut [f32; DENOISE_FRAME_SAMPLES]) -> Result<f32> {
        let mut prob = 0.0f32;
        check(
            unsafe { sys::transcribe_denoise_flush(self.ptr, output.as_mut_ptr(), &mut prob) },
            "transcribe_denoise_flush",
        )?;
        Ok(prob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(len: usize, hz: f32, amp: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / 48_000.0).sin() * amp)
            .collect()
    }

    fn frame(v: &[f32]) -> [f32; DENOISE_FRAME_SAMPLES] {
        v.try_into().unwrap()
    }

    #[test]
    fn strength_zero_passes_the_input_through_one_frame_late() {
        let mut d = Denoiser::new(&DenoiseParams {
            strength: 0.0,
            ..Default::default()
        })
        .unwrap();
        let first = frame(&sine(DENOISE_FRAME_SAMPLES, 1000.0, 0.5));
        let second = frame(&sine(DENOISE_FRAME_SAMPLES, 250.0, 0.25));
        let mut out = [0.0f32; DENOISE_FRAME_SAMPLES];
        d.process_frame(&first, &mut out).unwrap();
        assert!(out.iter().all(|&s| s == 0.0));
        d.process_frame(&second, &mut out).unwrap();
        for (x, y) in first.iter().zip(out.iter()) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn silence_in_place_flush_and_reset() {
        let mut d = Denoiser::new(&DenoiseParams::default()).unwrap();
        let mut f = [0.0f32; DENOISE_FRAME_SAMPLES];
        for _ in 0..20 {
            assert_eq!(d.process_frame_in_place(&mut f).unwrap(), 0.0);
            assert!(f.iter().all(|&s| s == 0.0));
        }
        let tone = frame(&sine(DENOISE_FRAME_SAMPLES, 300.0, 0.3));
        let mut a = [0.0f32; DENOISE_FRAME_SAMPLES];
        d.process_frame(&tone, &mut a).unwrap();
        let mut last = [0.0f32; DENOISE_FRAME_SAMPLES];
        let p = d.flush(&mut last).unwrap();
        assert!((0.0..=1.0).contains(&p));
        assert!(last
            .iter()
            .all(|s| s.is_finite() && (-1.0..=1.0).contains(s)));
        d.reset();
        let mut b = [0.0f32; DENOISE_FRAME_SAMPLES];
        d.process_frame(&[0.0; DENOISE_FRAME_SAMPLES], &mut b)
            .unwrap();
        assert!(b.iter().all(|&s| s == 0.0), "state leaked across reset");
    }

    #[test]
    fn validates_and_normalizes() {
        assert!(Denoiser::new(&DenoiseParams {
            strength: 1.5,
            ..Default::default()
        })
        .is_err());
        assert!(Denoiser::new(&DenoiseParams {
            vad_grace_ms: DENOISE_MAX_GRACE_MS + 1,
            ..Default::default()
        })
        .is_err());
        let p = DenoiseParams {
            strength: 7.0,
            vad_threshold: f32::NAN,
            vad_grace_ms: 999_999,
        }
        .normalized();
        assert_eq!(
            p,
            DenoiseParams {
                strength: 1.0,
                vad_threshold: 0.0,
                vad_grace_ms: DENOISE_MAX_GRACE_MS
            }
        );
        let mut d = Denoiser::new(&p).unwrap();
        assert!(d
            .set_params(&DenoiseParams {
                vad_threshold: -1.0,
                ..Default::default()
            })
            .is_err());
        let mut nan = [0.0f32; DENOISE_FRAME_SAMPLES];
        nan[3] = f32::NAN;
        let mut out = [0.0f32; DENOISE_FRAME_SAMPLES];
        assert!(d.process_frame(&nan, &mut out).is_err());
        assert_eq!(
            DENOISE_FRAME_SAMPLES,
            sys::TRANSCRIBE_DENOISE_FRAME_SAMPLES as usize
        );
        assert_eq!(DENOISE_SAMPLE_RATE, sys::TRANSCRIBE_DENOISE_SAMPLE_RATE);
    }
}
