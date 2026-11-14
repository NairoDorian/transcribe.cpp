//! Engine-native Voice Activity Detection (Earshot minGRU).

use transcribe_cpp_sys as sys;

/// Standalone Voice Activity Detector powered by clean-room embedded Earshot minGRU weights.
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
    /// Create a new detector instance with the specified sensitivity threshold [0.05, 0.95].
    pub fn new(threshold: f32) -> Option<Self> {
        let ptr = unsafe { sys::transcribe_vad_init(threshold) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Predict the raw speech probability score [0.0, 1.0] for a 256-sample frame (16 ms @ 16 kHz).
    pub fn predict_frame(&mut self, frame_256: &[f32; 256]) -> f32 {
        unsafe { sys::transcribe_vad_predict_frame(self.ptr, frame_256.as_ptr()) }
    }

    /// Process a 256-sample frame through energy pre-gate and hysteresis state machine.
    ///
    /// Returns (is_speaking, frame_score).
    pub fn process_frame(&mut self, frame_256: &[f32; 256]) -> (bool, f32) {
        let mut score = 0.0f32;
        let speaking =
            unsafe { sys::transcribe_vad_process_frame(self.ptr, frame_256.as_ptr(), &mut score) };
        (speaking, score)
    }

    /// Update sensitivity threshold in place without resetting recurrent state.
    pub fn set_threshold(&mut self, threshold: f32) {
        unsafe { sys::transcribe_vad_set_threshold(self.ptr, threshold) };
    }

    /// Reset recurrent minGRU state and hysteresis counters.
    pub fn reset(&mut self) {
        unsafe { sys::transcribe_vad_reset(self.ptr) };
    }
}
