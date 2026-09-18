//! Streaming data types and the stream-params builder.
//!
//! The [`Stream`](crate::Stream) handle itself lives in `session.rs` (it needs
//! the session internals); this module holds the owned option/result types it
//! exchanges.

use transcribe_cpp_sys as sys;

use crate::family::{StreamExtRaw, StreamExtension};
use crate::result::owned_str;
use crate::types::CommitPolicy;

/// Options for beginning a stream.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamOptions {
    /// When committed text grows. Default [`CommitPolicy::Auto`].
    pub commit_policy: CommitPolicy,
    /// Consecutive agreeing hypotheses required before a prefix commits
    /// (stable-prefix policies). 0 selects the library default (3).
    pub stable_prefix_agreement_n: u32,
    /// Optional family-specific stream extension.
    pub family: Option<StreamExtension>,
    /// Enable engine-native Earshot VAD. Defaults to true.
    pub enable_vad: bool,
    /// VAD sensitivity threshold in [0.05, 0.95]. Default 0.50.
    pub vad_threshold: f32,
    /// VAD speech onset prefill in ms. Default 450.
    pub vad_prefill_ms: u32,
    /// VAD hangover trailing duration in ms. Default 1200.
    pub vad_hangover_ms: u32,
}

impl Default for StreamOptions {
    fn default() -> Self {
        let mut params: sys::transcribe_stream_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_stream_params_init(&mut params) };
        Self {
            commit_policy: CommitPolicy::Auto,
            stable_prefix_agreement_n: params.stable_prefix_agreement_n,
            family: None,
            enable_vad: params.enable_vad,
            vad_threshold: params.vad_threshold,
            vad_prefill_ms: params.vad_prefill_ms,
            vad_hangover_ms: params.vad_hangover_ms,
        }
    }
}

/// Per-call change metadata from `feed`/`finalize`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StreamUpdate {
    /// Any observable property of the snapshot changed this call.
    pub result_changed: bool,
    /// True only on the `finalize` call's update.
    pub is_final: bool,
    /// Monotonic snapshot revision; diff against the previous value.
    pub revision: i32,
    /// Total audio received since begin (ms).
    pub input_received_ms: i64,
    /// Family-reported audio progress / drain hint (ms).
    pub audio_committed_ms: i64,
    /// Audio still buffered inside the family's streaming state (ms).
    pub buffered_ms: i64,
    /// `committed_text` changed this call.
    pub committed_changed: bool,
    /// `tentative_text` changed this call.
    pub tentative_changed: bool,
    /// Voice activity detected on the latest fed audio chunk.
    pub vad_speaking: bool,
    /// Accumulated duration of speech in the current active burst (ms).
    pub vad_speech_ms: u64,
    /// Raw probability score [0.0, 1.0] from the latest Earshot VAD evaluation.
    pub vad_last_score: f32,
    /// Current audio RMS energy level in dBFS.
    pub audio_level_dbfs: f32,
}

impl StreamUpdate {
    pub(crate) fn from_raw(raw: &sys::transcribe_stream_update) -> Self {
        StreamUpdate {
            result_changed: raw.result_changed,
            is_final: raw.is_final,
            revision: raw.revision,
            input_received_ms: raw.input_received_ms,
            audio_committed_ms: raw.audio_committed_ms,
            buffered_ms: raw.buffered_ms,
            committed_changed: raw.committed_changed,
            tentative_changed: raw.tentative_changed,
            vad_speaking: raw.vad_speaking,
            vad_speech_ms: raw.vad_speech_ms,
            vad_last_score: raw.vad_last_score,
            audio_level_dbfs: raw.audio_level_dbfs,
        }
    }
}

/// A UI-facing snapshot of the stream's text, fully owned.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StreamText {
    /// The raw current model hypothesis (authoritative; may rewrite anywhere).
    pub full: String,
    /// The append-only, flicker-free display/input prefix.
    pub committed: String,
    /// The volatile raw suffix after the committed prefix.
    pub tentative: String,
}

impl StreamText {
    /// `committed + tentative` — the flicker-free render most UIs want.
    pub fn display(&self) -> String {
        format!("{}{}", self.committed, self.tentative)
    }

    pub(crate) fn from_raw(raw: &sys::transcribe_stream_text) -> Self {
        StreamText {
            full: owned_str(raw.full_text),
            committed: owned_str(raw.committed_text),
            tentative: owned_str(raw.tentative_text),
        }
    }
}

/// Build `transcribe_stream_params`, returning the family-ext keepalive so its
/// backing struct outlives the `transcribe_stream_begin` call.
pub(crate) fn build_stream_params(
    o: &StreamOptions,
) -> (sys::transcribe_stream_params, Option<StreamExtRaw>) {
    let mut params: sys::transcribe_stream_params = unsafe { std::mem::zeroed() };
    unsafe { sys::transcribe_stream_params_init(&mut params) };
    params.commit_policy = o.commit_policy.to_raw();
    params.stable_prefix_agreement_n = o.stable_prefix_agreement_n;
    params.enable_vad = o.enable_vad;
    params.vad_threshold = o.vad_threshold;
    params.vad_prefill_ms = o.vad_prefill_ms;
    params.vad_hangover_ms = o.vad_hangover_ms;

    let family = o.family.as_ref().map(StreamExtension::materialize);
    params.family = family.as_ref().map_or(std::ptr::null(), |f| f.ext_ptr());
    (params, family)
}
