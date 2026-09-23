//! Build script for `transcribe-cpp-sys`.
//!
//! Source build is the primary path: the `cmake` crate drives the vendored
//! C++ tree with `TRANSCRIBE_INSTALL=ON`, and the link line is reconstructed
//! from NOTHING but the installed `lib/transcribe-link.json` manifest — the
//! same artifact the `link_smoke` CI lane compiles a toy C consumer against.
//! No per-platform link lists are hardcoded here (the whisper-rs drift class
//! this avoids).
//!
//! Prebuilt path: setting TRANSCRIBE_DIR (OPENSSL_DIR-style) to an install
//! prefix produced by `cmake --install` of a TRANSCRIBE_INSTALL=ON build
//! skips the source build entirely and links against that prefix's manifest
//! instead. Cargo features are inert there: the prebuilt already decided its
//! configuration (static/shared, backends), and the manifest records it.
//!
//! Cargo features map directly to CMake options:
//!   `shared`           -> TRANSCRIBE_BUILD_SHARED=ON (default: static)
//!   `dynamic-backends` -> the above + TRANSCRIBE_GGML_BACKEND_DL=ON (+ x86:
//!                         GGML_CPU_ALL_VARIANTS=ON, TRANSCRIBE_X86_CONSERVATIVE=ON)
//!   `metal`            -> TRANSCRIBE_METAL=ON   (Apple targets only; no-op elsewhere)
//!   `vulkan`           -> TRANSCRIBE_VULKAN=ON
//!   `cuda`             -> TRANSCRIBE_CUDA=ON
//!   `rocm`             -> TRANSCRIBE_HIP=ON
//!   `openmp`           -> TRANSCRIBE_USE_OPENMP=ON
//! Official-artifact hygiene flags (OpenMP/BLAS off) are deliberately NOT
//! forced here: a source build is the consumer's build (same philosophy as
//! the Python sdist).
//!
//! Escape hatch: anything else CMake accepts can be passed via the
//! TRANSCRIBE_CMAKE_ARGS (or CMAKE_ARGS) env var — see the passthrough at the
//! end of main(). This is the "no Cargo feature is a hard ceiling" guarantee.
//!
//! Windows: the native build runs through a short NTFS junction to OUT_DIR so
//! a stock machine builds the Vulkan backend from any checkout depth (MAX_PATH).

use std::env;
use std::path::{Path, PathBuf};

fn feature(name: &str) -> bool {
    env::var_os(format!("CARGO_FEATURE_{name}")).is_some()
}

/// Split a CMAKE_ARGS-style string into individual arguments, honoring simple
/// double-quotes so a value containing spaces survives (e.g. `-DFOO="a b"`).
/// Whitespace-separated otherwise; quotes are stripped from the emitted token.
fn split_cmake_args(s: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut has_token = false;
    for c in s.chars() {
        match c {
            '"' => {
                in_quotes = !in_quotes;
                has_token = true;
            }
            c if c.is_whitespace() && !in_quotes => {
                if has_token {
                    args.push(std::mem::take(&mut cur));
                    has_token = false;
                }
            }
            c => {
                cur.push(c);
                has_token = true;
            }
        }
    }
    if has_token {
        args.push(cur);
    }
    args
}

fn main() {
    // CARGO_MANIFEST_DIR for this crate is the repo root (the sys crate's
    // manifest lives there so the tarball can carry the whole C++ tree).
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    for p in [
        "CMakeLists.txt",
        "CMakePresets.json",
        "include",
        "src",
        "ggml",
        "cmake",
        "bindings/rust/sys/build.rs",
        "bindings/rust/sys/src",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(p).display());
    }

    // Prebuilt path: TRANSCRIBE_DIR (or TRANSCRIBE_PREBUILT_DIR) points at an existing install prefix (a
    // `cmake --install` tree from a TRANSCRIBE_INSTALL=ON configure). Skip the
    // source build and emit the link line from that prefix's manifest; the
    // manifest records the install's own posture (static/shared, backends), so
    // the Cargo features below never apply here.
    println!("cargo:rerun-if-env-changed=TRANSCRIBE_DIR");
    println!("cargo:rerun-if-env-changed=TRANSCRIBE_PREBUILT_DIR");
    let prebuilt_dir =
        env::var_os("TRANSCRIBE_DIR").or_else(|| env::var_os("TRANSCRIBE_PREBUILT_DIR"));
    if let Some(dir) = prebuilt_dir {
        let prefix = PathBuf::from(dir);
        let manifest = find_manifest(&prefix).unwrap_or_else(|| {
            panic!(
                "TRANSCRIBE_DIR is set but no lib/transcribe-link.json (or lib64/) exists \
                 under {}. It must point at an install prefix of this library, produced by \
                 `cmake -B build -DTRANSCRIBE_INSTALL=ON && cmake --build build && \
                 cmake --install build --prefix <dir>`. Unset TRANSCRIBE_DIR to build \
                 the vendored sources instead.",
                prefix.display()
            )
        });
        println!("cargo:rerun-if-changed={}", manifest.display());
        // Reinstalling an edited library need not change the link manifest.
        // Watch its artifacts too, otherwise Windows keeps the old staged DLLs.
        for subdir in ["bin", "lib", "lib64", "include"] {
            let artifacts = prefix.join(subdir);
            if artifacts.exists() {
                println!("cargo:rerun-if-changed={}", artifacts.display());
            }
        }
        emit_link_lines(&prefix, &manifest);
        return;
    }

    println!("cargo:rerun-if-env-changed=TRANSCRIBE_CACHE_DIR");
    println!("cargo:rerun-if-env-changed=TRANSCRIBE_FORCE_REBUILD");
    println!("cargo:rerun-if-env-changed=TRANSCRIBE_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=TRANSCRIBE_NO_CCACHE");
    println!("cargo:rerun-if-env-changed=TRANSCRIBE_CCACHE_PATH");
    println!("cargo:rerun-if-env-changed=TRANSCRIBE_NO_NINJA");
    println!("cargo:rerun-if-env-changed=TRANSCRIBE_NINJA_PATH");
    println!("cargo:rerun-if-env-changed=TRANSCRIBE_MODEL_SET");
    println!("cargo:rerun-if-env-changed=TRANSCRIBE_MODELS");

    // Explicit escape hatch: skip the persistent cache and compile from source.
    let force_rebuild = env::var_os("TRANSCRIBE_FORCE_REBUILD").is_some();
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));

    // `dynamic-backends` (loadable backend modules) requires a shared library,
    // so it implies `shared`. The Cargo manifest already encodes that implication
    // (`dynamic-backends = ["shared"]`), but treat it as load-bearing here too.
    let dynamic_backends = feature("DYNAMIC_BACKENDS");
    let arch_dl = feature("ARCH_DL");
    // Both loadable-plugin postures need a shared library: dynamic-backends
    // loads ggml backends, arch-dl loads architecture plugins, and each
    // resolves symbols out of libtranscribe at runtime. The Cargo manifests
    // already encode the implication (`… = ["shared"]`); treat it as
    // load-bearing here too, since CMake hard-errors on TRANSCRIBE_ARCH_DL
    // without a shared build.
    let shared = feature("SHARED") || dynamic_backends || arch_dl;
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let is_apple = matches!(target_os.as_str(), "macos" | "ios");
    let is_x86 = matches!(target_arch.as_str(), "x86" | "x86_64");

    let mut active_features = Vec::new();
    if shared {
        active_features.push("shared");
    }
    if dynamic_backends {
        active_features.push("dynamic-backends");
    }
    if arch_dl {
        active_features.push("arch-dl");
    }
    if feature("METAL") {
        active_features.push("metal");
    }
    if feature("VULKAN") {
        active_features.push("vulkan");
    }
    if feature("CUDA") {
        active_features.push("cuda");
    }
    if feature("ROCM") {
        active_features.push("rocm");
    }
    if feature("OPENMP") {
        active_features.push("openmp");
    }
    let model_set_env = env::var("TRANSCRIBE_MODEL_SET").unwrap_or_default();
    if feature("MINIMAL_MULTILINGUAL") {
        active_features.push("minimal-multilingual");
    } else if !model_set_env.is_empty() {
        active_features.push(&model_set_env);
    }

    let is_cuda = feature("CUDA");
    let cuda_arch = resolve_cuda_arch(is_cuda);

    // Cache key includes a source-tree fingerprint (max mtime across tracked
    // source dirs) and targeted CUDA arch so the persistent cache invalidates
    // automatically when transcribe.cpp sources change (git pull, local edit)
    // or when switching between dev (single-arch) and release (multi-arch).
    let cache_key = compute_cache_key(
        &root,
        &target_os,
        &target_arch,
        &target_env,
        &active_features,
        cuda_arch.as_deref(),
    );
    let cache_dir = get_cache_root().join(&cache_key);

    // Check for explicit prebuilt or persistent cache hit. On a cache hit we
    // copy the pre-built artifacts into OUT_DIR and skip CMake entirely — so
    // `bun run tauri dev` only rebuilds transcribe.cpp when its sources moved
    // (cache key differs → cache miss → CMake build).
    if !force_rebuild {
        if let Ok(prebuilt) = env::var("TRANSCRIBE_PREBUILT_DIR") {
            let prebuilt_path = PathBuf::from(prebuilt);
            if let Some(manifest) = find_manifest(&prebuilt_path) {
                println!(
                    "cargo:warning=transcribe-cpp-sys: [PREBUILT] Using prebuilt transcribe-cpp from {}",
                    prebuilt_path.display()
                );
                let _ = copy_dir_all(&prebuilt_path, &out_dir);
                emit_link_lines(&out_dir, &manifest);
                return;
            }
        }

        if let Some(manifest) = find_manifest(&cache_dir) {
            println!(
                "cargo:warning=transcribe-cpp-sys: [CACHE HIT] Using persistent cached build from {} (0s compile time)",
                cache_dir.display()
            );
            let _ = copy_dir_all(&cache_dir, &out_dir);
            emit_link_lines(&out_dir, &manifest);
            return;
        }

        let fallback_prebuilt = get_cache_root().join("prebuilt");
        if let Some(manifest) = find_manifest(&fallback_prebuilt) {
            println!(
                "cargo:warning=transcribe-cpp-sys: [CACHE HIT] Using prebuilt cache from {} (0s compile time)",
                fallback_prebuilt.display()
            );
            let _ = copy_dir_all(&fallback_prebuilt, &out_dir);
            emit_link_lines(&out_dir, &manifest);
            return;
        }
    }

    println!(
        "cargo:warning=transcribe-cpp-sys: [CACHE MISS] Compiling transcribe-cpp via CMake..."
    );

    let mut cfg = cmake::Config::new(&root);
    cfg.profile("Release") // a transcription library always wants an optimized native core
        .define("TRANSCRIBE_INSTALL", "ON")
        .define("TRANSCRIBE_BUILD_TESTS", "OFF")
        .define("TRANSCRIBE_BUILD_EXAMPLES", "OFF")
        .define("TRANSCRIBE_BUILD_TOOLS", "OFF")
        .define("TRANSCRIBE_BUILD_SHARED", if shared { "ON" } else { "OFF" })
        // Each enabled architecture is built as a separate loadable plugin
        // rather than compiled into libtranscribe. The plugin SET comes from
        // TRANSCRIBE_MODEL_SET (defined below); `_all_families` in
        // src/CMakeLists.txt is the authority for how many families that is,
        // so no count is restated here — it drifts with every new family.
        .define("TRANSCRIBE_ARCH_DL", if arch_dl { "ON" } else { "OFF" });

    // Ninja Generator Setup:
    // On Windows with MSVC, setup MSVC environment (via vcvars64.bat if needed)
    // and configure Ninja for fast parallel compilation if available.
    let ninja_disabled = env::var("TRANSCRIBE_NO_NINJA").is_ok()
        || env::var("CMAKE_GENERATOR").as_deref() == Ok("Visual Studio");

    let mut using_ninja = false;
    if !ninja_disabled {
        let msvc_ok = setup_msvc_environment();
        if msvc_ok {
            if let Some(ninja_path) = find_ninja() {
                println!(
                    "cargo:warning=transcribe-cpp-sys: [NINJA] Using Ninja generator ({})",
                    ninja_path.display()
                );
                cfg.generator("Ninja");
                if ninja_path.is_absolute() {
                    cfg.define(
                        "CMAKE_MAKE_PROGRAM",
                        ninja_path.to_string_lossy().replace('\\', "/"),
                    );
                }
                using_ninja = true;
            }
        }
    }

    if !using_ninja {
        println!("cargo:warning=transcribe-cpp-sys: [BUILD] Using default CMake generator");
    }

    // Force optimization on MSVC. `.profile("Release")` only selects the *config*
    // (and CRT) of the Visual Studio multi-config generator — it does NOT
    // guarantee optimization. cmake-rs strips `/O*` from the cc-derived flags
    // ("let cmake deal with optimization") and then, for the VS generator,
    // overwrites CMAKE_<LANG>_FLAGS_RELEASE with those stripped flags — dropping
    // CMake's default `/O2 /Ob2 /DNDEBUG`. The "Release" build then compiles ggml
    // UNOPTIMIZED with assertions live: ~7x slower on CPU (realtime collapses to
    // ~1x) and a pred-heavy ~3x slower decode on Vulkan. Flags passed via
    // cflag/cxxflag are injected verbatim (un-stripped) into that same
    // CMAKE_<LANG>_FLAGS_RELEASE — the var the VS generator actually respects — so
    // re-adding them here restores optimization. (Off-MSVC the GNU/Clang lanes get
    // -O from the profile and CMake's Release init flags aren't stripped.)
    if target_env == "msvc" {
        for flag in ["/O2", "/Ob2", "/DNDEBUG"] {
            cfg.cflag(flag);
            cfg.cxxflag(flag);
        }
    }

    // Dynamic backend modules: each compute backend becomes a loadable module
    // next to libtranscribe, picked at runtime by transcribe_init_backends().
    // The root CMakeLists validates BACKEND_DL => SHARED and force-sets
    // GGML_NATIVE=OFF, so this just flips the knobs. On x86, fan the CPU backend
    // out into one module per ISA tier (runtime feature scoring) over the
    // SIGILL-safe x86 floor — the same posture the Linux/Windows cpu-vulkan
    // wheel lane ships. ALL_VARIANTS is an x86 concept; on arm a DL build is a
    // single portable CPU module.
    if dynamic_backends {
        cfg.define("TRANSCRIBE_GGML_BACKEND_DL", "ON");
        if is_x86 {
            cfg.define("GGML_CPU_ALL_VARIANTS", "ON");
            cfg.define("TRANSCRIBE_X86_CONSERVATIVE", "ON");
        }
    }

    // Metal: on Apple, set TRANSCRIBE_METAL EXPLICITLY to track the `metal`
    // feature. CMake defaults TRANSCRIBE_METAL ON on Apple Silicon, so without an
    // explicit OFF a `--no-default-features` (metal off) build would still enable
    // Metal — breaking Cargo.toml's "pure CPU build on macOS is
    // default-features = false" contract. Off Apple the feature is a no-op (CMake
    // already defaults it OFF).
    if is_apple {
        if feature("METAL") {
            cfg.define("TRANSCRIBE_METAL", "ON");
            // Self-contained installed tree: embed the metallib instead of a
            // sidecar default.metallib next to the lib (matches the shipped macOS
            // wheel posture; what the shared-infra link-smoke uses).
            cfg.define("GGML_METAL_EMBED_LIBRARY", "ON");
        } else {
            cfg.define("TRANSCRIBE_METAL", "OFF");
        }
    }
    if feature("VULKAN") {
        cfg.define("TRANSCRIBE_VULKAN", "ON");
    }
    if is_cuda {
        cfg.define("TRANSCRIBE_CUDA", "ON");
        if let Some(arch) = &cuda_arch {
            println!(
                "cargo:warning=transcribe-cpp-sys: [CUDA DEV] Auto-detected local GPU -> targeting {arch} (fast single-arch build)"
            );
            cfg.define("CMAKE_CUDA_ARCHITECTURES", arch);
        } else {
            println!(
                "cargo:warning=transcribe-cpp-sys: [CUDA RELEASE] Targeting full distribution multi-arch set"
            );
        }
    }
    if feature("ROCM") {
        cfg.define("TRANSCRIBE_HIP", "ON");
    }

    // ccache: compiler caching to accelerate recompilation on cache misses
    let ccache_disabled =
        env::var("TRANSCRIBE_NO_CCACHE").is_ok() || env::var("GGML_CCACHE").as_deref() == Ok("OFF");
    if !ccache_disabled {
        if let Some(ccache) = find_ccache() {
            println!(
                "cargo:warning=transcribe-cpp-sys: [CCACHE] Enabling compiler caching launcher ({})",
                ccache.display()
            );
            let ccache_str = ccache.to_string_lossy();
            cfg.define("CMAKE_C_COMPILER_LAUNCHER", &*ccache_str);
            cfg.define("CMAKE_CXX_COMPILER_LAUNCHER", &*ccache_str);
            if is_cuda {
                cfg.define("CMAKE_CUDA_COMPILER_LAUNCHER", &*ccache_str);
            }
        } else {
            cfg.define("GGML_CCACHE", "OFF");
        }
    } else {
        cfg.define("GGML_CCACHE", "OFF");
    }

    // Keep OpenMP OFF unless explicitly opted in. TRANSCRIBE_USE_OPENMP already
    // defaults OFF in CMake (the native ggml threadpool is the default path); we
    // set it explicitly here so `--features openmp` is the single switch. We keep
    // it OFF by default because OpenMP's `-fopenmp` shows up only as a manifest
    // link_flag → a `cargo:rustc-link-arg` that does NOT propagate to downstream
    // binaries, so a static consumer link would fail with undefined GOMP_*/omp_*
    // symbols. A self-contained static build is the default; `--features openmp`
    // opts in (and then owns providing the OpenMP runtime).
    cfg.define(
        "TRANSCRIBE_USE_OPENMP",
        if feature("OPENMP") { "ON" } else { "OFF" },
    );
    // Windows no longer needs ggml's OpenMP. ggml's native CPU threadpool barrier
    // used to deadlock under MSVC (its spin `relax` compiled to a no-op, starving
    // an un-arrived worker), so OpenMP was the only multi-threaded CPU path there.
    // That barrier is fixed upstream (ggml-cpu.c: YieldProcessor relax + a
    // bounded-spin ggml_thread_yield fallback), so the native pool is correct on
    // MSVC — and avoids the vcomp pool that crashes at teardown when a host
    // dlopen/dlcloses this lib. We therefore leave GGML_OPENMP at its
    // CMakeLists-forced OFF on every platform; `--features openmp` is the only
    // opt-in. See the "OpenMP — CENTRAL POLICY" block in the root CMakeLists.txt.

    if feature("MINIMAL_MULTILINGUAL") {
        cfg.define("TRANSCRIBE_MODEL_SET", "minimal-multilingual");
    } else if !model_set_env.is_empty() {
        cfg.define("TRANSCRIBE_MODEL_SET", &model_set_env);
    }
    if let Ok(val) = env::var("TRANSCRIBE_MODELS") {
        cfg.define("TRANSCRIBE_MODELS", val);
    }

    // Escape hatch: forward arbitrary configure args so the curated features are
    // never a hard ceiling. Anything CMake accepts (-DGGML_*, a -DTRANSCRIBE_*
    // the features don't cover, a toolchain define, ...) can be passed via
    // TRANSCRIBE_CMAKE_ARGS / CMAKE_ARGS. Fed AFTER the feature-derived defines
    // so a user -D wins on the first configure. Unsupported/untested by design —
    // it exists so a consumer is never blocked. The link line is still
    // reconstructed from the regenerated manifest, so whatever this turns on is
    // linked correctly with no per-flag knowledge here.
    for var in ["TRANSCRIBE_CMAKE_ARGS", "CMAKE_ARGS"] {
        println!("cargo:rerun-if-env-changed={var}");
        if let Ok(extra) = env::var(var) {
            for arg in split_cmake_args(&extra) {
                cfg.configure_arg(arg);
            }
        }
    }

    // Windows: build through a short junction to OUT_DIR so the native build
    // stays under MAX_PATH on stock machines (see windows_short_out_dir).
    let short = windows_short_out_dir();
    if let Some(short) = &short {
        cfg.out_dir(short);
    }

    // Clean build directory if generator changed (e.g. Visual Studio -> Ninja)
    let build_dir = short.as_ref().unwrap_or(&out_dir).join("build");
    let cache_txt = build_dir.join("CMakeCache.txt");
    if cache_txt.is_file() {
        if let Ok(content) = std::fs::read_to_string(&cache_txt) {
            let has_ninja = content.contains("CMAKE_GENERATOR:INTERNAL=Ninja");
            if using_ninja != has_ninja {
                println!(
                    "cargo:warning=transcribe-cpp-sys: Generator changed -> cleaning build directory {}",
                    build_dir.display()
                );
                let _ = std::fs::remove_dir_all(&build_dir);
            }
        }
    }

    // Builds + installs into OUT_DIR; the returned path IS the install prefix.
    let prefix = cfg.build();
    // Emit downstream paths via the durable OUT_DIR, not the junction — cargo
    // caches them across builds, and the junction may be deleted between runs.
    let prefix = if short.is_some() {
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"))
    } else {
        prefix
    };

    let manifest = find_manifest(&prefix)
        .unwrap_or_else(|| panic!("transcribe-link.json not found under {}", prefix.display()));

    // Cache build artifacts to persistent storage for all future builds.
    if let Err(e) = copy_dir_all(&prefix, &cache_dir) {
        println!(
            "cargo:warning=transcribe-cpp-sys: failed to cache build to {}: {e}",
            cache_dir.display()
        );
    } else {
        println!(
            "cargo:warning=transcribe-cpp-sys: [CACHE SAVED] Cached transcribe-cpp artifacts to {}",
            cache_dir.display()
        );
    }

    emit_link_lines(&prefix, &manifest);
}

/// Windows MAX_PATH mitigation: a short NTFS junction (`%LOCALAPPDATA%\tcs\<hash>`,
/// no admin needed) resolving to OUT_DIR. The Vulkan ExternalProject nests ~185
/// chars past OUT_DIR, and MSBuild's FileTracker ignores LongPathsEnabled (FTK1011),
/// so the build root itself must be short. None = build in OUT_DIR (non-Windows,
/// or best-effort failure). Idempotent; hash-named per OUT_DIR so checkouts don't collide.
fn windows_short_out_dir() -> Option<PathBuf> {
    if !cfg!(windows) {
        return None;
    }
    // Backslash-normalize: mklink rejects forward slashes (MSYS-style CARGO_TARGET_DIR).
    let out_dir = PathBuf::from(env::var("OUT_DIR").ok()?.replace('/', "\\"));
    let Some(base) = env::var_os("LOCALAPPDATA")
        .or_else(|| env::var_os("TEMP"))
        .map(PathBuf::from)
    else {
        println!(
            "cargo:warning=transcribe-cpp-sys: neither LOCALAPPDATA nor TEMP is set; \
             building in OUT_DIR (may exceed Windows MAX_PATH in deep checkouts)"
        );
        return None;
    };
    let base = base.join("tcs");

    let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a: stable across rustc versions
    for b in out_dir.to_string_lossy().bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let link = base.join(format!("{hash:016x}"));

    warn_fallback(
        std::fs::create_dir_all(&out_dir),
        "create junction target",
        &out_dir,
    )?;
    // symlink_metadata (not exists()) so a dangling junction is detected and reclaimed.
    if std::fs::symlink_metadata(&link).is_ok() {
        match (
            std::fs::canonicalize(&link),
            std::fs::canonicalize(&out_dir),
        ) {
            (Ok(a), Ok(b)) if a == b => return Some(link),
            // remove_dir fails if something non-junction squats here (e.g. a
            // backup tool materialized it as a real tree); the warning names
            // the path so the user knows what to delete.
            _ => warn_fallback(std::fs::remove_dir(&link), "remove stale junction", &link)?,
        }
    }
    warn_fallback(
        std::fs::create_dir_all(&base),
        "create junction parent",
        &base,
    )?;

    // No std API creates junctions; mklink /J needs no extra deps.
    let output = std::process::Command::new("cmd")
        .arg("/C")
        .arg("mklink")
        .arg("/J")
        .arg(&link)
        .arg(&out_dir)
        .output();
    let created = output.as_ref().map(|o| o.status.success()).unwrap_or(false);
    let verified = created
        && matches!(
            (std::fs::canonicalize(&link), std::fs::canonicalize(&out_dir)),
            (Ok(a), Ok(b)) if a == b
        );
    if !verified {
        // Best-effort: fall back to building in the deep OUT_DIR.
        let detail = output
            .map(|o| {
                String::from_utf8_lossy(if o.stderr.is_empty() {
                    &o.stdout
                } else {
                    &o.stderr
                })
                .trim()
                .to_string()
            })
            .unwrap_or_else(|e| e.to_string());
        println!(
            "cargo:warning=transcribe-cpp-sys: could not create short build junction {} -> {} ({detail}); \
             building in OUT_DIR (may exceed Windows MAX_PATH in deep checkouts)",
            link.display(),
            out_dir.display()
        );
        return None;
    }
    Some(link)
}

/// Best-effort junction setup step: on failure, warn like the mklink branch
/// and bail to the deep-OUT_DIR fallback via `?`. Cargo hides build-script
/// warnings for registry crates unless the build fails — so this is silent on
/// success and visible exactly when a deep-path build dies of MAX_PATH.
fn warn_fallback<T, E: std::fmt::Display>(
    res: Result<T, E>,
    action: &str,
    path: &Path,
) -> Option<T> {
    match res {
        Ok(v) => Some(v),
        Err(e) => {
            println!(
                "cargo:warning=transcribe-cpp-sys: could not {action} {} ({e}); \
                 building in OUT_DIR (may exceed Windows MAX_PATH in deep checkouts)",
                path.display()
            );
            None
        }
    }
}

/// GNUInstallDirs picks `lib` or `lib64`; find the manifest under either.
fn find_manifest(prefix: &Path) -> Option<PathBuf> {
    for libdir in ["lib", "lib64"] {
        let p = prefix.join(libdir).join("transcribe-link.json");
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Convert a conventional absolute Unix library filename into the Cargo native
/// library kind/name pair. `rustc-link-lib` propagates through dependent Rust
/// crates; a raw `rustc-link-arg=/path/to/lib` does not.
fn cargo_library_name(path: &Path) -> Option<(&'static str, String)> {
    let file = path.file_name()?.to_str()?;
    let file = file.strip_prefix("lib")?;

    if let Some(name) = file.strip_suffix(".dll.a") {
        return Some(("dylib", name.to_owned()));
    }
    if let Some(name) = file.strip_suffix(".a") {
        return Some(("static", name.to_owned()));
    }
    for suffix in [".so", ".dylib", ".tbd"] {
        if let Some(name) = file.strip_suffix(suffix) {
            return Some(("dylib", name.to_owned()));
        }
    }
    None
}

fn emit_link_lines(prefix: &Path, manifest_path: &Path) {
    let text = std::fs::read_to_string(manifest_path).expect("read transcribe-link.json");
    let json: serde_json::Value = serde_json::from_str(&text).expect("parse transcribe-link.json");

    let strs = |key: &str| -> Vec<String> {
        json[key]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };

    let shared = json["shared"].as_bool().unwrap_or(false);
    let lib_dir = prefix.join(json["lib_dir"].as_str().unwrap_or("lib"));
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Archives (static) or the single shared lib. The manifest order is
    // single-pass-GNU-ld safe (each archive's undefined refs resolve in a
    // later one: transcribe -> ggml -> backends -> ggml-base).
    let kind = if shared { "dylib" } else { "static" };
    for name in strs("libraries") {
        println!("cargo:rustc-link-lib={kind}={name}");
    }

    // Absolute library paths (e.g. ROCm SDK libraries, compiler-rt, or a
    // find_package(BLAS) result). Express conventional library files as
    // search-dir + rustc-link-lib rather than a raw rustc-link-arg: native
    // library directives propagate from this -sys crate to final downstream
    // binaries, while link arguments only affect this crate's own artifacts.
    // Keep the exact-path fallback for an unusual linker input that cannot be
    // represented as a conventional native library.
    for path in strs("library_paths") {
        let path = Path::new(&path);
        if let (Some(parent), Some((kind, name))) = (path.parent(), cargo_library_name(path)) {
            println!("cargo:rustc-link-search=native={}", parent.display());
            println!("cargo:rustc-link-lib={kind}={name}");
        } else {
            println!("cargo:rustc-link-arg={}", path.display());
        }
    }
    // System libraries the C++/backend archives drag in.
    for name in strs("system_libs") {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    // Apple frameworks (Metal/Foundation/Accelerate...).
    for name in strs("frameworks") {
        println!("cargo:rustc-link-lib=framework={name}");
    }
    // Extra link flags (e.g. -fopenmp).
    for flag in strs("link_flags") {
        println!("cargo:rustc-link-arg={flag}");
    }
    // Shared posture: the installed libs carry $ORIGIN/@loader_path rpaths, but
    // the consumer binary still needs to find lib_dir itself. Unix uses an
    // rpath; Windows has no rpath (the DLL resolves via PATH / the exe dir), so
    // nothing is emitted there. NOTE: rustc-link-arg does NOT propagate to
    // downstream crates, so this rpath only reaches THIS crate's own artifacts
    // (the -sys smoke tests). The safe crate re-emits it for its tests/examples
    // — see bindings/rust/transcribe-cpp/build.rs.
    let is_windows = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    if shared && !is_windows {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }
    // Windows has no rpath: a shared-posture binary resolves transcribe.dll +
    // the ggml DLLs from its OWN directory. Stage the installed DLLs next to
    // every artifact cargo produces so tests/examples/bins run in place.
    if shared && is_windows {
        stage_windows_dlls(prefix);
    }

    // Metadata for the safe crate's build script (DEP_TRANSCRIBE_* because this
    // crate sets `links = "transcribe"`). The wrapper re-emits these one more hop
    // as DEP_TRANSCRIBE_CPP_* for downstream packaging — see its build.rs.
    println!(
        "cargo:include_dir={}",
        prefix
            .join(json["include_dir"].as_str().unwrap_or("include"))
            .display()
    );
    println!("cargo:lib_dir={}", lib_dir.display());

    // Runtime-artifact dirs for downstream PACKAGING. Only a shared posture
    // produces separate runtime files to ship next to a consumer's executable;
    // a static build bakes everything into the consumer binary, so these stay
    // unset and the consumer reads their absence as "nothing to bundle".
    if shared {
        let bin_dir = prefix.join("bin");
        // The ONE directory a consumer copies into their installer. Windows keeps
        // the runtime DLLs in bin/; on Unix the .so/.dylib live in lib_dir. A
        // single value so the consumer needs no per-OS cfg — `bin_dir` alone
        // would be too Windows-specific.
        let runtime_dir = if is_windows { &bin_dir } else { &lib_dir };
        println!("cargo:runtime_dir={}", runtime_dir.display());
        println!("cargo:bin_dir={}", bin_dir.display());
        // dynamic-backends: the loadable compute modules (a ggml backend subdir,
        // or bin/ on Windows). The manifest already records the install-relative
        // path; forward it so a consumer bundles the modules too — without them a
        // relocated DL build registers zero compute devices.
        if let Some(rel) = json["module_dir"].as_str() {
            println!("cargo:module_dir={}", prefix.join(rel).display());
        }
    }
}

/// Copy the installed runtime DLLs (`<prefix>/bin/*.dll` — transcribe.dll plus
/// the ggml DLLs) next to every artifact cargo will build, so a shared-posture
/// consumer's tests/examples/bins find them with no rpath (Windows has none)
/// and no PATH fiddling. Windows + `shared` only; a no-op anywhere else.
///
/// Windows resolves a process's DLLs from the EXE's own directory first, and
/// cargo emits tests into `deps/`, examples into `examples/`, and bins into the
/// profile root — so all three get the DLLs. This runs from the -sys build
/// script (it owns the native build), which is always in the dep graph, so it
/// covers the safe crate's artifacts too. Best-effort: copy failures warn
/// rather than fail the build.
fn stage_windows_dlls(prefix: &Path) {
    let bin_dir = prefix.join("bin");
    let dlls: Vec<PathBuf> = match std::fs::read_dir(&bin_dir) {
        Ok(entries) => entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("dll"))
            .collect(),
        Err(_) => Vec::new(),
    };
    if dlls.is_empty() {
        println!(
            "cargo:warning=transcribe-cpp-sys: no DLLs under {} to stage for the shared posture",
            bin_dir.display()
        );
        return;
    }
    // OUT_DIR = <target>/<profile>/build/<crate>-<hash>/out; the profile root is
    // four ancestors up. tests -> deps/, examples -> examples/, bins -> root.
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return;
    };
    for sub in ["", "deps", "examples"] {
        let dest = if sub.is_empty() {
            profile_dir.to_path_buf()
        } else {
            profile_dir.join(sub)
        };
        let _ = std::fs::create_dir_all(&dest);
        for dll in &dlls {
            if let Some(name) = dll.file_name() {
                let _ = std::fs::copy(dll, dest.join(name));
            }
        }
    }
}

/// Root directory for persistent out-of-tree transcribe.cpp build artifacts.
fn get_cache_root() -> PathBuf {
    if let Ok(dir) = env::var("TRANSCRIBE_PREBUILT_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = env::var("TRANSCRIBE_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    let base = if cfg!(windows) {
        env::var_os("LOCALAPPDATA")
            .or_else(|| env::var_os("APPDATA"))
            .or_else(|| env::var_os("USERPROFILE"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("C:\\ProgramData"))
    } else {
        env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .unwrap_or_else(|| PathBuf::from("/tmp"))
    };
    base.join("handy").join("transcribe_cpp_cache")
}

/// Compute a unique cache key for the target platform, enabled features, and
/// source tree fingerprint. The source fingerprint (max mtime across tracked
/// source dirs) ensures the cache invalidates when any source file changes.
fn compute_cache_key(
    root: &Path,
    target_os: &str,
    target_arch: &str,
    target_env: &str,
    features: &[&str],
    cuda_arch: Option<&str>,
) -> String {
    let mut s = format!("{target_os}-{target_arch}-{target_env}-release-");
    let mut sorted_features = features.to_vec();
    sorted_features.sort();
    for f in sorted_features {
        s.push_str(f);
        s.push('_');
    }
    if let Some(arch) = cuda_arch {
        s.push_str("cudaarch-");
        s.push_str(arch);
        s.push('_');
    }
    s.push_str(&format!("src-{}", max_source_mtime(root)));
    let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a: stable across rustc versions
    for b in s.bytes() {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{target_os}_{target_arch}_{target_env}_{hash:016x}")
}

/// Probe the local GPU compute capability via nvidia-smi.
fn probe_local_gpu_arch() -> Option<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next()?.trim();
    let mut parts = line.split('.');
    let major: u32 = parts.next()?.parse().ok()?;
    let minor: u32 = parts.next()?.parse().ok()?;

    let arch = format!("{major}{minor}");
    if major >= 12 {
        Some(format!("{arch}a-real"))
    } else {
        Some(format!("{arch}-real"))
    }
}

/// Resolve the CUDA target architecture policy.
///
/// Returns `Some(arch)` for a specific architecture (e.g. "89-real" for local dev),
/// or `None` when building the full multi-arch distribution set (for release builds).
fn resolve_cuda_arch(is_cuda: bool) -> Option<String> {
    if !is_cuda {
        return None;
    }

    if let Ok(val) =
        env::var("TRANSCRIBE_CUDA_ARCHITECTURES").or_else(|_| env::var("CMAKE_CUDA_ARCHITECTURES"))
    {
        let val = val.trim();
        if val == "default" {
            return None; // None signals CMake default / full multi-arch
        }
        if val == "auto" {
            return probe_local_gpu_arch();
        }
        if !val.is_empty() {
            return Some(val.to_string());
        }
    }

    // Default policy:
    // Dev build (`tauri dev`, `cargo run`, `cargo test` -> PROFILE == "debug" or DEBUG == "true"):
    // auto-detect local GPU for fast single-architecture compilation.
    // Release build (`tauri build`, `cargo build --release` -> PROFILE == "release"):
    // full multi-arch distribution set.
    let profile = env::var("PROFILE").unwrap_or_default();
    let debug_var = env::var("DEBUG").unwrap_or_default();
    let is_dev = profile == "debug" || debug_var == "true" || debug_var == "1" || debug_var == "2";

    if is_dev {
        probe_local_gpu_arch()
    } else {
        None
    }
}

/// Find ccache on PATH, from TRANSCRIBE_CCACHE_PATH, or in standard developer tool locations.
fn find_ccache() -> Option<PathBuf> {
    if let Ok(p) = env::var("TRANSCRIBE_CCACHE_PATH") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    if let Ok(output) = std::process::Command::new("ccache")
        .arg("--version")
        .output()
    {
        if output.status.success() {
            return Some(PathBuf::from("ccache"));
        }
    }
    if let Some(userprofile) = env::var_os("USERPROFILE") {
        let user_path = PathBuf::from(userprofile);
        let cargo_ccache = user_path.join(".cargo").join("bin").join("ccache.exe");
        if cargo_ccache.is_file() {
            return Some(cargo_ccache);
        }
        let local_ccache = user_path
            .join("AppData")
            .join("Local")
            .join("bin")
            .join("ccache.exe");
        if local_ccache.is_file() {
            return Some(local_ccache);
        }
    }
    None
}

/// Find ninja executable path on PATH, from TRANSCRIBE_NINJA_PATH, or in Visual Studio CMake directory.
fn find_ninja() -> Option<PathBuf> {
    if let Ok(p) = env::var("TRANSCRIBE_NINJA_PATH") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Some(pb);
        }
    }
    if let Ok(output) = std::process::Command::new("ninja")
        .arg("--version")
        .output()
    {
        if output.status.success() {
            return Some(PathBuf::from("ninja"));
        }
    }
    // Check known Visual Studio paths
    let mut vswhere_path =
        PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    if !vswhere_path.is_file() {
        if let Some(pf86) = env::var_os("ProgramFiles(x86)") {
            vswhere_path =
                PathBuf::from(pf86).join(r"Microsoft Visual Studio\Installer\vswhere.exe");
        }
    }
    if vswhere_path.is_file() {
        if let Ok(output) = std::process::Command::new(&vswhere_path)
            .args(["-latest", "-products", "*", "-property", "installationPath"])
            .output()
        {
            if output.status.success() {
                let vs_install = String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !vs_install.is_empty() {
                    let vs_ninja = PathBuf::from(vs_install)
                        .join(r"Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe");
                    if vs_ninja.is_file() {
                        return Some(vs_ninja);
                    }
                }
            }
        }
    }
    None
}

/// Setup MSVC environment on Windows if cl.exe is not in PATH.
/// Returns true if MSVC environment is active/ready.
fn setup_msvc_environment() -> bool {
    if !cfg!(windows) {
        return true;
    }
    // If cl.exe is already runnable, environment is already set up.
    if std::process::Command::new("cl.exe").output().is_ok() {
        return true;
    }

    // Locate vswhere.exe
    let mut vswhere_path =
        PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    if !vswhere_path.is_file() {
        if let Some(pf86) = env::var_os("ProgramFiles(x86)") {
            vswhere_path =
                PathBuf::from(pf86).join(r"Microsoft Visual Studio\Installer\vswhere.exe");
        }
    }
    if !vswhere_path.is_file() {
        println!(
            "cargo:warning=transcribe-cpp-sys: [MSVC] vswhere.exe not found at {}",
            vswhere_path.display()
        );
        return false;
    }

    let output = match std::process::Command::new(&vswhere_path)
        .args([
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
    {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            println!(
                "cargo:warning=transcribe-cpp-sys: [MSVC] vswhere failed with status {:?}",
                o.status
            );
            return false;
        }
        Err(e) => {
            println!("cargo:warning=transcribe-cpp-sys: [MSVC] vswhere execution error: {e}");
            return false;
        }
    };

    let vs_install = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    if vs_install.is_empty() {
        println!(
            "cargo:warning=transcribe-cpp-sys: [MSVC] vswhere returned empty installation path"
        );
        return false;
    }

    let vcvars_bat = PathBuf::from(&vs_install).join(r"VC\Auxiliary\Build\vcvars64.bat");
    if !vcvars_bat.is_file() {
        println!(
            "cargo:warning=transcribe-cpp-sys: [MSVC] vcvars64.bat not found at {}",
            vcvars_bat.display()
        );
        return false;
    }

    // Execute vcvars64.bat and capture environment
    #[cfg(windows)]
    use std::os::windows::process::CommandExt;

    let mut cmd = std::process::Command::new("cmd");
    cmd.args(["/d", "/s", "/c"]);
    #[cfg(windows)]
    cmd.raw_arg(format!("\"\"{}\" >nul && set\"", vcvars_bat.display()));

    let cmd_output = match cmd.output() {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            println!(
                "cargo:warning=transcribe-cpp-sys: [MSVC] vcvars64.bat failed with status {:?}: {}",
                o.status,
                stderr.trim()
            );
            return false;
        }
        Err(e) => {
            println!("cargo:warning=transcribe-cpp-sys: [MSVC] cmd.exe execution error: {e}");
            return false;
        }
    };

    let env_text = String::from_utf8_lossy(&cmd_output.stdout);
    let mut count = 0;
    for line in env_text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            let key = k.trim();
            if !key.is_empty() && key != "PROMPT" {
                env::set_var(key, v.trim());
                count += 1;
            }
        }
    }
    println!("cargo:warning=transcribe-cpp-sys: [MSVC] Loaded {count} environment variables from vcvars64.bat");
    true
}

/// Walk the source trees that participate in the native build and return the
/// maximum file modification time as a hex string. When any source file changes
/// (git pull, local edit) the max advances and the cache key changes, forcing a
/// recompile. mtime-based (not content hash) for speed.
fn max_source_mtime(root: &Path) -> String {
    let mut max: Option<std::time::SystemTime> = None;
    for p in [
        "CMakeLists.txt",
        "include",
        "src",
        "ggml",
        "cmake",
        "bindings/rust/sys",
    ] {
        walk_mtimes(&root.join(p), &mut max);
    }
    match max.and_then(|t| t.duration_since(std::time::SystemTime::UNIX_EPOCH).ok()) {
        Some(d) => format!("{:016x}", d.as_nanos()),
        None => "0".to_string(),
    }
}

fn walk_mtimes(path: &Path, max: &mut Option<std::time::SystemTime>) {
    if path.is_file() {
        if let Ok(meta) = path.metadata() {
            if let Ok(mtime) = meta.modified() {
                if max.map_or(true, |m| mtime > m) {
                    *max = Some(mtime);
                }
            }
        }
    } else if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name == "."
                    || name == ".."
                    || name.starts_with('.')
                    || name == "target"
                    || name == "build"
                {
                    continue;
                }
                walk_mtimes(&entry.path(), max);
            }
        }
    }
}

/// Recursively copy a directory tree.
fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    if !src.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &dst_path)?
        } else {
            let _ = std::fs::copy(entry.path(), dst_path);
        }
    }
    Ok(())
}
