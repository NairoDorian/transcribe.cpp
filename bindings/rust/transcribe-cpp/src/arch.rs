//! Architecture-plugin loading for `TRANSCRIBE_ARCH_DL` builds.
//!
//! In a conventional build every model architecture is compiled into
//! libtranscribe and nothing here is needed. In an **`arch-dl`** build the core
//! carries NONE of them: each family (parakeet, granite, qwen3_asr, whisper, …)
//! ships as its own loadable module — `transcribe-arch-<family>.dll` on
//! Windows, `libtranscribe-arch-<family>.so` / `.dylib` elsewhere — and the
//! library resolves one at model-open time.
//!
//! A host normally does nothing at all: the library probes, in order, the model
//! file's directory and `<model dir>/arch`, every directory registered through
//! [`register_arch_dir`], `$TRANSCRIBE_ARCH_DIR`, and finally the directory
//! holding libtranscribe (plus its `arch/` subdirectory). Install the plugins
//! beside libtranscribe — which is what `cmake --install` does — and models
//! open with no host code.
//!
//! The two functions here are for the cases that layout does not cover:
//! pointing the loader at a plugin directory the library cannot guess
//! ([`register_arch_dir`]), and loading one specific file up front
//! ([`load_arch_plugin`]) so a failure surfaces at startup rather than on the
//! first model load.
//!
//! # Ordering
//!
//! Both calls mutate a process-global plugin registry. Complete them before
//! other threads load models; a model load that races a registration may miss
//! the plugin it needs and fail as if the architecture were unsupported.

use std::ffi::CString;
use std::path::Path;

use transcribe_cpp_sys as sys;

use crate::error::{check, Result};

/// Register a directory the architecture loader will search.
///
/// Adds `dir` to the process-global search list used when a model's
/// architecture is not compiled in and has not already been loaded. Registration
/// is additive, idempotent and cheap — it does not scan or load anything, so
/// registering a directory holding hundreds of plugins costs nothing until one
/// is actually needed. Call it BEFORE the first model load.
///
/// Errors with [`crate::Error::ModelFileNotFound`] when `dir` is not an
/// existing directory (the C layer classifies it as a not-found path, not a bad
/// argument), or [`crate::Error::InvalidArgument`] for a NUL in the path.
///
/// ```no_run
/// # use transcribe_cpp::{Model, RunOptions};
/// transcribe_cpp::register_arch_dir("/opt/my-app/arch")?;
/// let mut session = Model::load("model.gguf")?.session()?;
/// # Ok::<(), transcribe_cpp::Error>(())
/// ```
pub fn register_arch_dir(dir: impl AsRef<Path>) -> Result<()> {
    let dir = dir.as_ref();
    // Pass the path bytes through faithfully (Unix) / reject non-UTF-8 (Windows),
    // matching model loading and init_backends — never lossily mangle a path.
    let c_dir = CString::new(crate::model::path_bytes(dir)?)?;
    let status = unsafe { sys::transcribe_register_arch_dir(c_dir.as_ptr()) };
    check(status, "register_arch_dir")
}

/// Load one architecture plugin module from an explicit file path.
///
/// Use this to fail fast: the loader's own search is lazy, so a missing or
/// ABI-incompatible plugin otherwise surfaces as an "unsupported architecture"
/// error on the first model that needs it. Calling this at startup turns that
/// into an immediate, precisely-attributed error.
///
/// Loading a plugin that is already loaded is a no-op returning `Ok(())` — the
/// module is identified by its canonical path, so two spellings of one file
/// collapse to a single load. Call it BEFORE the first model load.
///
/// Errors are classified by the C layer: [`crate::Error::InvalidArgument`] for
/// a NUL in the path, [`crate::Error::ModelFileNotFound`] if no regular file is
/// there, and [`crate::Error::ModelLoad`] when the file is present but is not a
/// usable plugin — it failed to load, exports no `transcribe_arch_plugin_get`
/// entry point, reports a different plugin ABI version, or uses a different
/// architecture name than its filename implies.
///
/// ```no_run
/// # use transcribe_cpp::Model;
/// transcribe_cpp::load_arch_plugin("/opt/my-app/arch/transcribe-arch-whisper.so")?;
/// # Ok::<(), transcribe_cpp::Error>(())
/// ```
pub fn load_arch_plugin(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let c_path = CString::new(crate::model::path_bytes(path)?)?;
    let status = unsafe { sys::transcribe_load_arch_plugin(c_path.as_ptr()) };
    check(status, "load_arch_plugin")
}
