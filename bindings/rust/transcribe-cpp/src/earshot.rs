//! Engine-native voice activity detection: the Earshot detector (SIMD
//! kernels, bundled weights, nothing to load), batch scoring, and the
//! capture-side speech gate.

use transcribe_cpp_sys as sys;

use crate::error::{check, error_for_status, Result};

/// Samples per Earshot frame: 16 ms at 16 kHz.
pub const EARSHOT_FRAME_SAMPLES: usize = 256;

/// Standalone frame-level detector (`transcribe_vad_*`): Earshot scores plus
/// a two-threshold speech flag with hangover.
#[derive(Debug)]
pub struct VoiceActivityDetector {
    ptr: *mut sys::transcribe_vad,
}

unsafe impl Send for VoiceActivityDetector {}

impl Drop for VoiceActivityDetector {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { sys::transcribe_vad_free(self.ptr) };
        }
    }
}

impl VoiceActivityDetector {
    /// Create a detector. `threshold` is the speech-entry score in
    /// [0.05, 0.95]; out of range (or NaN) takes the default 0.5. `None` only
    /// on an allocation failure.
    pub fn new(threshold: f32) -> Option<Self> {
        let ptr = unsafe { sys::transcribe_vad_init(threshold) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Speech probability [0.0, 1.0] of a 256-sample frame (16 ms @ 16 kHz).
    /// Advances the detector's recurrent state, not the speech flag. A frame
    /// with a non-finite sample scores 0 and leaves the state untouched.
    pub fn predict_frame(&mut self, frame_256: &[f32; 256]) -> f32 {
        unsafe { sys::transcribe_vad_predict_frame(self.ptr, frame_256.as_ptr()) }
    }

    /// Score a 256-sample frame and advance the speech flag: speech starts
    /// at a score >= threshold, continues while it stays >= 0.7 x threshold
    /// and for 1200 ms after it drops.
    ///
    /// Returns (is_speaking, frame_score); a frame with a non-finite sample
    /// returns (false, 0.0) and leaves the state untouched.
    pub fn process_frame(&mut self, frame_256: &[f32; 256]) -> (bool, f32) {
        let mut score = 0.0f32;
        let speaking =
            unsafe { sys::transcribe_vad_process_frame(self.ptr, frame_256.as_ptr(), &mut score) };
        (speaking, score)
    }

    /// Update the threshold in place without resetting recurrent state
    /// (clamped to [0.05, 0.95]; NaN is ignored).
    pub fn set_threshold(&mut self, threshold: f32) {
        unsafe { sys::transcribe_vad_set_threshold(self.ptr, threshold) };
    }

    /// Reset recurrent minGRU state and hysteresis counters.
    pub fn reset(&mut self) {
        unsafe { sys::transcribe_vad_reset(self.ptr) };
    }
}

/// Earshot scores for every complete 256-sample frame of a 16 kHz mono clip,
/// computed in batch over `threads` threads (the same values frame-by-frame
/// scoring gives). Samples are clamped to [-1, 1].
pub fn score_frames(pcm: &[f32], threads: i32) -> Result<Vec<f32>> {
    let mut scores = vec![0.0f32; pcm.len() / EARSHOT_FRAME_SAMPLES];
    let status = unsafe {
        sys::transcribe_vad_score_frames(
            pcm.as_ptr(),
            pcm.len() as i64,
            threads,
            scores.as_mut_ptr(),
        )
    };
    check(status, "transcribe_vad_score_frames")?;
    Ok(scores)
}

/// Settings of a [`VadGate`]; durations round up to whole 16 ms frames.
/// The defaults are a dictation capture path.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct VadGateParams {
    /// Enter speech at or above this score, 0.05 ..= 0.95.
    pub threshold: f32,
    /// Leave speech below this score; `None` = `threshold - 0.15`, at least 0.01.
    pub exit_threshold: Option<f32>,
    /// Consecutive voiced time needed to start speech.
    pub onset_ms: u32,
    /// Audio before the onset released with it.
    pub preroll_ms: u32,
    /// Audio kept flowing after the last voiced frame (~1650 for a streaming decoder).
    pub hangover_ms: u32,
    /// Pauses shorter than this are billed to the speech clock when speech resumes.
    pub speech_hold_ms: u32,
}

impl Default for VadGateParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            exit_threshold: None,
            onset_ms: 60,
            preroll_ms: 450,
            hangover_ms: 450,
            speech_hold_ms: 500,
        }
    }
}

/// What one [`VadGate::feed`] / [`VadGate::finish`] call produced.
#[derive(Debug, Clone, Copy)]
pub struct VadGateUpdate<'a> {
    /// Frames scored by this call.
    pub n_frames: i32,
    /// Smoothed state after the call (onset and hangover applied).
    pub speech: bool,
    /// Raw two-threshold decision of the last frame (what speech metrics count).
    pub voiced: bool,
    /// An onset fired during this call.
    pub speech_started: bool,
    /// A hangover ran out during this call.
    pub speech_ended: bool,
    /// Speech clock state.
    pub clock_speaking: bool,
    /// Earshot score of the last frame.
    pub last_score: f32,
    /// RMS level of the last frame, dBFS.
    pub level_dbfs: f32,
    /// Speech clock total since creation or the last reset.
    pub speech_ms: u64,
    /// Audio retained by this call (pre-roll, speech, hangover; original samples).
    pub audio: &'a [f32],
}

/// End-of-recording diagnostics: frames buffered but never released.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VadGateTail {
    pub withheld_frames: i32,
    pub withheld_voiced_frames: i32,
    pub speech: bool,
    pub onset_counter: i32,
    pub hangover_counter: i32,
}

/// Capture-side speech gate (`transcribe_vad_gate_*`): feed 16 kHz mono audio
/// of any length, get back the speech with pre-roll and hangover, plus a
/// speech clock. Earshot + hysteresis + onset debounce + pre-roll ring that
/// never releases a frame twice + hangover; one gate per audio source.
#[derive(Debug)]
pub struct VadGate {
    ptr: *mut sys::transcribe_vad_gate,
}

unsafe impl Send for VadGate {}

impl Drop for VadGate {
    fn drop(&mut self) {
        unsafe { sys::transcribe_vad_gate_free(self.ptr) };
    }
}

impl VadGate {
    pub fn new(params: &VadGateParams) -> Result<Self> {
        let mut raw = std::mem::MaybeUninit::<sys::transcribe_vad_gate_params>::uninit();
        let mut raw = unsafe {
            sys::transcribe_vad_gate_params_init(raw.as_mut_ptr());
            raw.assume_init()
        };
        raw.threshold = params.threshold;
        raw.exit_threshold = params.exit_threshold.unwrap_or(-1.0);
        raw.onset_ms = params.onset_ms;
        raw.preroll_ms = params.preroll_ms;
        raw.hangover_ms = params.hangover_ms;
        raw.speech_hold_ms = params.speech_hold_ms;
        let mut ptr = std::ptr::null_mut();
        check(
            unsafe { sys::transcribe_vad_gate_init(&raw, &mut ptr) },
            "transcribe_vad_gate_init",
        )?;
        if ptr.is_null() {
            return Err(error_for_status(
                sys::transcribe_status::TRANSCRIBE_ERR_OOM,
                "transcribe_vad_gate_init",
            ));
        }
        Ok(Self { ptr })
    }

    fn update<'a>(&'a self, raw: &sys::transcribe_vad_gate_update) -> VadGateUpdate<'a> {
        let audio = if raw.audio.is_null() || raw.n_audio <= 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(raw.audio, raw.n_audio as usize) }
        };
        VadGateUpdate {
            n_frames: raw.n_frames,
            speech: raw.speech,
            voiced: raw.voiced,
            speech_started: raw.speech_started,
            speech_ended: raw.speech_ended,
            clock_speaking: raw.clock_speaking,
            last_score: raw.last_score,
            level_dbfs: raw.level_dbfs,
            speech_ms: raw.speech_ms,
            audio,
        }
    }

    fn new_update() -> sys::transcribe_vad_gate_update {
        let mut raw = std::mem::MaybeUninit::<sys::transcribe_vad_gate_update>::uninit();
        unsafe {
            sys::transcribe_vad_gate_update_init(raw.as_mut_ptr());
            raw.assume_init()
        }
    }

    /// Feed samples in [-1, 1]; a partial frame is kept for the next call.
    /// The returned audio borrows the gate until the next call.
    pub fn feed(&mut self, pcm: &[f32]) -> Result<VadGateUpdate<'_>> {
        let mut raw = Self::new_update();
        check(
            unsafe {
                sys::transcribe_vad_gate_feed(self.ptr, pcm.as_ptr(), pcm.len() as i64, &mut raw)
            },
            "transcribe_vad_gate_feed",
        )?;
        Ok(self.update(&raw))
    }

    /// End of input: score a kept partial frame zero-padded.
    pub fn finish(&mut self) -> Result<VadGateUpdate<'_>> {
        let mut raw = Self::new_update();
        check(
            unsafe { sys::transcribe_vad_gate_finish(self.ptr, &mut raw) },
            "transcribe_vad_gate_finish",
        )?;
        Ok(self.update(&raw))
    }

    /// Start a new recording (detector, smoother, clock, buffers); keeps params.
    pub fn reset(&mut self) {
        unsafe { sys::transcribe_vad_gate_reset(self.ptr) };
    }

    /// New threshold from the next frame on, mid-recording included.
    pub fn set_threshold(&mut self, threshold: f32) -> Result<()> {
        check(
            unsafe { sys::transcribe_vad_gate_set_threshold(self.ptr, threshold) },
            "transcribe_vad_gate_set_threshold",
        )
    }

    pub fn set_hangover_ms(&mut self, hangover_ms: u32) -> Result<()> {
        check(
            unsafe { sys::transcribe_vad_gate_set_hangover_ms(self.ptr, hangover_ms) },
            "transcribe_vad_gate_set_hangover_ms",
        )
    }

    pub fn tail(&self) -> Result<VadGateTail> {
        let mut raw = std::mem::MaybeUninit::<sys::transcribe_vad_gate_tail>::uninit();
        let mut raw = unsafe {
            sys::transcribe_vad_gate_tail_init(raw.as_mut_ptr());
            raw.assume_init()
        };
        check(
            unsafe { sys::transcribe_vad_gate_get_tail(self.ptr, &mut raw) },
            "transcribe_vad_gate_get_tail",
        )?;
        Ok(VadGateTail {
            withheld_frames: raw.withheld_frames,
            withheld_voiced_frames: raw.withheld_voiced_frames,
            speech: raw.speech,
            onset_counter: raw.onset_counter,
            hangover_counter: raw.hangover_counter,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(seconds: f32, amp: f32) -> Vec<f32> {
        (0..(seconds * 16000.0) as usize)
            .map(|i| amp * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16000.0).sin())
            .collect()
    }

    #[test]
    fn silence_is_dropped_and_framing_does_not_matter() {
        let mut gate = VadGate::new(&VadGateParams::default()).unwrap();
        let silence = vec![0.0f32; 16000];
        let u = gate.feed(&silence).unwrap();
        assert!(u.audio.is_empty());
        assert!(!u.speech);
        assert_eq!(u.speech_ms, 0);

        let mut pcm = vec![0.0f32; 8000];
        pcm.extend(tone(2.0, 0.3));
        let run = |chunk: usize| {
            let mut g = VadGate::new(&VadGateParams::default()).unwrap();
            let mut out = Vec::new();
            for c in pcm.chunks(chunk) {
                out.extend_from_slice(g.feed(c).unwrap().audio);
            }
            out.extend_from_slice(g.finish().unwrap().audio);
            out
        };
        let a = run(256);
        assert_eq!(a, run(37));
        assert_eq!(a, run(4096));
        assert!(a.len() <= pcm.len());
    }

    #[test]
    fn validates_and_controls() {
        assert!(VadGate::new(&VadGateParams {
            threshold: 1.5,
            ..Default::default()
        })
        .is_err());
        let mut gate = VadGate::new(&VadGateParams::default()).unwrap();
        assert!(gate.set_threshold(0.7).is_ok());
        assert!(gate.set_threshold(0.0).is_err());
        assert!(gate.set_hangover_ms(1650).is_ok());
        assert!(gate.feed(&[0.0, f32::NAN]).is_err());
        let tail = gate.tail().unwrap();
        assert!(tail.withheld_voiced_frames <= tail.withheld_frames);
        gate.reset();
    }

    #[test]
    fn batch_scores_match_frame_by_frame() {
        let pcm = tone(1.0, 0.2);
        let batch = score_frames(&pcm, 4).unwrap();
        let mut vad = VoiceActivityDetector::new(0.5).unwrap();
        let live: Vec<f32> = pcm
            .chunks_exact(256)
            .map(|f| vad.predict_frame(f.try_into().unwrap()))
            .collect();
        assert_eq!(batch, live);
        assert!(score_frames(&[], 1).unwrap().is_empty());
    }
}
