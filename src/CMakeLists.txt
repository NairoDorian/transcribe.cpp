# transcribe library.
#
# Sources are listed explicitly. Per-family code lives under arch/<fam>/;
# shared scaffolding (loader, mel frontend, tokenizer, load-common,
# backend classification, flash-policy, ...) lives alongside the
# dispatcher at the top level of src/.
#
# Backend libraries (Metal, Vulkan, ...) are compiled into ggml by the
# top-level CMake via GGML_METAL / GGML_VULKAN. This target discovers
# TRANSCRIBE_SHARED_EMBED: build libtranscribe as SHARED while ggml stays
# STATIC, so ggml's .o (incl. CUDA/Vulkan backends) are embedded into the
# single libtranscribe.so/.dll at link time. Otherwise follow BUILD_SHARED_LIBS.
if(TRANSCRIBE_SHARED_EMBED OR TRANSCRIBE_ARCH_DL)
    set(_transcribe_lib_type SHARED)
else()
    set(_transcribe_lib_type "")
endif()

# -----------------------------------------------------------------------------
# BLAS probe (used by decoder and mel frontend)
# -----------------------------------------------------------------------------
set(_has_blas FALSE)
set(_transcribe_extra_libs "")
if(APPLE)
    set(_has_blas TRUE)
    list(APPEND _transcribe_extra_libs "-framework Accelerate")
elseif(TRANSCRIBE_USE_SYSTEM_BLAS)
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        include(CheckIncludeFile)
        check_include_file(cblas.h HAVE_CBLAS_H)
        if(HAVE_CBLAS_H)
            set(_has_blas TRUE)
            list(APPEND _transcribe_extra_libs ${BLAS_LIBRARIES})
            message(STATUS "transcribe: BLAS found — decoder will use cblas_sgemv")
        else()
            message(STATUS "transcribe: BLAS found but cblas.h missing — decoder uses scalar fallback (install libopenblas-dev)")
        endif()
    else()
        message(STATUS "transcribe: no BLAS found — decoder uses scalar fallback")
    endif()
else()
    message(STATUS "transcribe: TRANSCRIBE_USE_SYSTEM_BLAS=OFF — decoder uses scalar fallback (official-wheel posture; no system BLAS vendored)")
endif()

# -----------------------------------------------------------------------------
# Model composite resolution
# -----------------------------------------------------------------------------
set(_all_families
    parakeet
    cohere
    canary
    qwen3_asr
    moss
    sortformer
    voxtral
    voxtral_realtime
    canary_qwen
    whisper
    moonshine
    moonshine_streaming
    sensevoice
    funasr_nano
    gigaam
    granite
    granite5_ctc
    granite_nar
    medasr
)

if(TRANSCRIBE_MODEL_SET STREQUAL "minimal-multilingual")
    # granite5_ctc is deliberately absent: Granite Speech 5.0 is English-only,
    # and the whole point of this set is a compact MULTILINGUAL composite. The
    # granite family that belongs here is granite (Granite Speech 2B), which is
    # multilingual. Request granite5_ctc through `full` or `custom` instead.
    set(_enabled_families parakeet granite qwen3_asr)
elseif(TRANSCRIBE_MODEL_SET STREQUAL "none")
    set(_enabled_families "")
elseif(TRANSCRIBE_MODEL_SET STREQUAL "custom")
    string(REPLACE "," ";" _raw_families "${TRANSCRIBE_MODELS}")
    set(_enabled_families "")
    foreach(_fam IN LISTS _raw_families)
        string(STRIP "${_fam}" _fam_clean)
        if(_fam_clean)
            list(APPEND _enabled_families "${_fam_clean}")
        endif()
    endforeach()
elseif(TRANSCRIBE_MODEL_SET STREQUAL "full" OR NOT TRANSCRIBE_MODEL_SET)
    set(_enabled_families ${_all_families})
else()
    message(FATAL_ERROR "Unknown TRANSCRIBE_MODEL_SET: '${TRANSCRIBE_MODEL_SET}'. Must be full, minimal-multilingual, none, or custom.")
endif()

message(STATUS "transcribe model composite: ${TRANSCRIBE_MODEL_SET} [${_enabled_families}]")

set(TRANSCRIBE_SRCS
    transcribe.cpp
    transcribe-loader.cpp
    transcribe-arch.cpp
    transcribe-meta.cpp
    transcribe-kaldi-fbank.cpp
    transcribe-mel.cpp
    transcribe-model.cpp
    transcribe-tokenizer.cpp
    transcribe-unicode.cpp
    transcribe-unicode-data.cpp
    transcribe-debug.cpp
    transcribe-env.cpp
    transcribe-flash-policy.cpp
    transcribe-backend.cpp
    transcribe-load-common.cpp
    transcribe-weights-util.cpp
    transcribe-bin-loader.cpp
    transcribe-batch-util.cpp
    transcribe-stubs.cpp
)

# Per-family source definitions
#
# Each list must name EVERY shared helper the family's files link against —
# conformer/, sanm/, granite_conformer/, causal_lm/ — and not just the ones
# under arch/<family>/. Under TRANSCRIBE_ARCH_DL the list becomes the complete
# source set of a standalone transcribe-arch-<family> module (see the plugin
# block at the bottom of this file), so nothing else supplies those symbols.
# The static build hides the omission: there every enabled family's list is
# concatenated into TRANSCRIBE_SRCS and REMOVE_DUPLICATES leaves one copy of
# each helper, which resolves the references of every family that happens to
# be enabled alongside it.
set(_srcs_parakeet
    arch/parakeet/model.cpp
    arch/parakeet/multitalker.cpp
    arch/parakeet/capabilities.cpp
    arch/parakeet/weights.cpp
    arch/parakeet/encoder.cpp
    arch/parakeet/decoder.cpp
    conformer/conformer.cpp
)

set(_srcs_cohere
    arch/cohere/model.cpp
    arch/cohere/capabilities.cpp
    arch/cohere/weights.cpp
    arch/cohere/encoder.cpp
    arch/cohere/decoder.cpp
    conformer/conformer.cpp
    causal_lm/causal_lm.cpp
)

set(_srcs_canary
    arch/canary/model.cpp
    arch/canary/capabilities.cpp
    arch/canary/weights.cpp
    arch/canary/encoder.cpp
    arch/canary/decoder.cpp
    conformer/conformer.cpp
)

set(_srcs_qwen3_asr
    arch/qwen3_asr/model.cpp
    arch/qwen3_asr/capabilities.cpp
    arch/qwen3_asr/weights.cpp
    arch/qwen3_asr/encoder.cpp
    arch/qwen3_asr/decoder.cpp
    causal_lm/causal_lm.cpp
)

set(_srcs_moss
    arch/moss/model.cpp
    arch/moss/capabilities.cpp
    arch/moss/diarize.cpp
    arch/moss/weights.cpp
    arch/moss/encoder.cpp
    arch/moss/decoder.cpp
    conformer/conformer.cpp
    causal_lm/causal_lm.cpp
)

set(_srcs_sortformer
    arch/sortformer/model.cpp
    arch/sortformer/stream.cpp
    arch/sortformer/capabilities.cpp
    arch/sortformer/weights.cpp
    conformer/conformer.cpp
)

set(_srcs_voxtral
    arch/voxtral/model.cpp
    arch/voxtral/capabilities.cpp
    arch/voxtral/weights.cpp
    arch/voxtral/encoder.cpp
    arch/voxtral/decoder.cpp
    conformer/conformer.cpp
    causal_lm/causal_lm.cpp
)

set(_srcs_voxtral_realtime
    arch/voxtral_realtime/model.cpp
    arch/voxtral_realtime/capabilities.cpp
    arch/voxtral_realtime/weights.cpp
    arch/voxtral_realtime/encoder.cpp
    arch/voxtral_realtime/decoder.cpp
    conformer/conformer.cpp
    causal_lm/causal_lm.cpp
)

set(_srcs_canary_qwen
    arch/canary_qwen/model.cpp
    arch/canary_qwen/capabilities.cpp
    arch/canary_qwen/weights.cpp
    arch/canary_qwen/encoder.cpp
    arch/canary_qwen/decoder.cpp
    conformer/conformer.cpp
    causal_lm/causal_lm.cpp
)

set(_srcs_whisper
    arch/whisper/model.cpp
    arch/whisper/capabilities.cpp
    arch/whisper/weights.cpp
    arch/whisper/encoder.cpp
    arch/whisper/decoder.cpp
    arch/whisper/bin_load.cpp
    arch/whisper/public.cpp
    conformer/conformer.cpp
    third_party/miniz/miniz.c
)

set(_srcs_moonshine
    arch/moonshine/model.cpp
    arch/moonshine/capabilities.cpp
    arch/moonshine/weights.cpp
    arch/moonshine/encoder.cpp
    arch/moonshine/decoder.cpp
    conformer/conformer.cpp
)

set(_srcs_moonshine_streaming
    arch/moonshine_streaming/model.cpp
    arch/moonshine_streaming/capabilities.cpp
    arch/moonshine_streaming/weights.cpp
    arch/moonshine_streaming/encoder.cpp
    arch/moonshine_streaming/decoder.cpp
    conformer/conformer.cpp
)

set(_srcs_sensevoice
    arch/sensevoice/model.cpp
    arch/sensevoice/capabilities.cpp
    arch/sensevoice/weights.cpp
    arch/sensevoice/encoder.cpp
    sanm/sanm.cpp
    conformer/conformer.cpp
)

set(_srcs_funasr_nano
    arch/funasr_nano/model.cpp
    arch/funasr_nano/capabilities.cpp
    arch/funasr_nano/weights.cpp
    arch/funasr_nano/encoder.cpp
    arch/funasr_nano/decoder.cpp
    arch/funasr_nano/adaptor.cpp
    sanm/sanm.cpp
    conformer/conformer.cpp
    causal_lm/causal_lm.cpp
)

set(_srcs_gigaam
    arch/gigaam/model.cpp
    arch/gigaam/capabilities.cpp
    arch/gigaam/weights.cpp
    arch/gigaam/encoder.cpp
    arch/gigaam/decoder.cpp
    arch/gigaam/mel.cpp
    conformer/conformer.cpp
)

set(_srcs_granite
    arch/granite/model.cpp
    arch/granite/diarize.cpp
    arch/granite/capabilities.cpp
    arch/granite/weights.cpp
    arch/granite/encoder.cpp
    arch/granite/projector.cpp
    arch/granite/decoder.cpp
    granite_conformer/shaw_attn.cpp
    conformer/conformer.cpp
    causal_lm/causal_lm.cpp
)

# Granite Speech 5.0 CTC. Shares granite_conformer's Shaw attention with
# the other two granite families and pulls batchnorm / macaron / depthwise
# helpers out of conformer/, so both are named here even though only
# encoder.cpp references them.
set(_srcs_granite5_ctc
    arch/granite5_ctc/model.cpp
    arch/granite5_ctc/capabilities.cpp
    arch/granite5_ctc/weights.cpp
    arch/granite5_ctc/encoder.cpp
    arch/granite5_ctc/decoder.cpp
    granite_conformer/shaw_attn.cpp
    conformer/conformer.cpp
)

set(_srcs_granite_nar
    arch/granite_nar/model.cpp
    arch/granite_nar/capabilities.cpp
    arch/granite_nar/weights.cpp
    arch/granite_nar/encoder.cpp
    arch/granite_nar/projector.cpp
    arch/granite_nar/decoder.cpp
    granite_conformer/shaw_attn.cpp
    conformer/conformer.cpp
)

set(_srcs_medasr
    arch/medasr/model.cpp
    arch/medasr/capabilities.cpp
    arch/medasr/weights.cpp
    arch/medasr/encoder.cpp
    conformer/conformer.cpp
)

# parakeet and sortformer call into each other, so under TRANSCRIBE_ARCH_DL
# neither can be a module on its own — each one's plugin is built from this
# union instead (see the plugin block at the bottom of this file).
#
#   parakeet -> sortformer: parakeet's multitalker models embed a Sortformer
#     diarizer. multitalker.cpp calls sf::resolve_stream_params,
#     run_diar_streaming_core and probs_to_speaker_segments, and model.cpp
#     calls init_embedded_diarizer / fuse_embedded_diar_bn — all defined in
#     arch/sortformer/. The calls sit behind TRANSCRIBE_ENABLE_ARCH_SORTFORMER,
#     which the static build can honour because one binary holds both families.
#   sortformer -> parakeet: sortformer's encoder reuses parakeet's Conformer
#     frontend — ParakeetWeights / ParakeetHParams / ParakeetBlock (types) and
#     build_encoder_graph, defined in arch/parakeet/encoder.cpp. These calls
#     are unconditional.
#
# Compiling the sibling's sources also emits its `arch` object (defined
# unconditionally in each family's model.cpp). Nothing references it: a
# plugin's entry point (transcribe-plugin-entry.cpp) names exactly one
# namespace, the one this plugin is being built for.
set(_srcs_parakeet_sortformer
    ${_srcs_parakeet}
    ${_srcs_sortformer}
)
list(REMOVE_DUPLICATES _srcs_parakeet_sortformer)

if(NOT TRANSCRIBE_ARCH_DL)
    foreach(_fam IN LISTS _enabled_families)
        list(APPEND TRANSCRIBE_SRCS ${_srcs_${_fam}})
    endforeach()
    list(REMOVE_DUPLICATES TRANSCRIBE_SRCS)
endif()

# The families whose symbols are actually IN libtranscribe — which is what a
# test linking `transcribe` may reference. tests/CMakeLists.txt gates every
# family-specific test on this (`if("granite" IN_LIST ...)`), so an inert value
# here does not fail the tests: it builds them anyway and the link fails
# instead, in whichever configure happens to exclude that family.
#
# Under TRANSCRIBE_ARCH_DL the answer is the empty list even though families
# are enabled: the core holds no architecture compiled in, so a test that
# referenced family symbols would not link against it. Family-specific tests
# are a static-build affair.
#
# Exported to the parent scope because tests/ is added as a sibling subdirectory
# at the top level, after src/, and this file's scope ends with it.
if(TRANSCRIBE_ARCH_DL)
    set(TRANSCRIBE_ACTIVE_MODELS "")
else()
    set(TRANSCRIBE_ACTIVE_MODELS "${_enabled_families}")
endif()
set(TRANSCRIBE_ACTIVE_MODELS "${TRANSCRIBE_ACTIVE_MODELS}" PARENT_SCOPE)

add_library(transcribe ${_transcribe_lib_type} ${TRANSCRIBE_SRCS})

if(NOT TRANSCRIBE_ARCH_DL)
    foreach(_fam IN LISTS _enabled_families)
        string(TOUPPER "${_fam}" _fam_upper)
        target_compile_definitions(transcribe PUBLIC TRANSCRIBE_ENABLE_ARCH_${_fam_upper}=1)
    endforeach()
endif()

target_include_directories(transcribe
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        # Internal headers (transcribe-loader.h, transcribe-arch.h) live
        # alongside the .cpp files. PRIVATE so they never escape into the
        # public include path.
        ${CMAKE_CURRENT_SOURCE_DIR}
        # Generated build-provenance header (transcribe-build-info.h), produced
        # by the top-level CMakeLists.txt. PRIVATE: it is an internal detail of
        # the build-id string in transcribe.cpp, not part of the public API.
        ${CMAKE_BINARY_DIR}/generated
)

target_compile_definitions(transcribe
    PRIVATE
        TRANSCRIBE_BUILD
        # Git commit stamped at configure time; surfaced by
        # transcribe_version_commit(). Falls back to "unknown" in the source
        # (see transcribe.cpp) when the build tree has no git metadata.
        TRANSCRIBE_COMMIT="${TRANSCRIBE_BUILD_COMMIT}"
)

if(TRANSCRIBE_GGML_BACKEND_DL)
    target_compile_definitions(transcribe PRIVATE TRANSCRIBE_GGML_BACKEND_DL)
    if(CMAKE_DL_LIBS)
        target_link_libraries(transcribe PRIVATE ${CMAKE_DL_LIBS})
    endif()
endif()

# Lets the loader call the Metal capability query behind the AUTO-selection
# gate (see metal_backend_lacks_simdgroup_mm in transcribe-load-common.cpp).
if(GGML_METAL)
    target_compile_definitions(transcribe PRIVATE TRANSCRIBE_HAS_METAL=1)
endif()

if(TRANSCRIBE_ENABLE_VALIDATION_HOOKS)
    target_compile_definitions(transcribe PRIVATE TRANSCRIBE_ENABLE_VALIDATION_HOOKS)
endif()

# Static posture: tell transcribe.h to drop the Windows __declspec(dllimport/
# export) decoration. PUBLIC so it reaches both this archive's own TUs and any
# CMake consumer linking `transcribe` (C++ tests/examples) — without it a static
# MSVC consumer that includes transcribe.h links against __imp_* thunks the
# archive never provides (LNK2019). Bindgen/Python FFIs carry no __declspec and
# are unaffected. No-op off Windows/MSVC (the header guard is _WIN32-only).
#
# TRANSCRIBE_SHARED_EMBED builds a SHARED lib, so it must NOT define
# TRANSCRIBE_STATIC (the .dll needs dllexport to generate the .lib import lib).
if(NOT TRANSCRIBE_BUILD_SHARED AND NOT TRANSCRIBE_SHARED_EMBED AND NOT TRANSCRIBE_ARCH_DL)
    target_compile_definitions(transcribe PUBLIC TRANSCRIBE_STATIC)
endif()

# ggml is the runtime substrate. Linked PRIVATE because the public header
# stays ggml-free (callers never include <ggml.h>).
target_link_libraries(transcribe
    PRIVATE
        ggml
)

# TRANSCRIBE_SHARED_EMBED: ggml is STATIC, so its .o (incl. SYCL/CUDA backends)
# link into libtranscribe.so/.dll. Backend targets (ggml-sycl, ggml-cuda, ...)
# set their runtime deps (-fsycl, CUDA::cudart, IntelSYCL::SYCL_CXX, Level Zero,
# DNNL, MKL) as PRIVATE on themselves. `target_link_libraries(ggml PUBLIC
# <backend>)` propagates the backend's .o to ggml's consumers, but PRIVATE
# link deps/options do NOT cross that boundary — so the merged libtranscribe
# link is missing them.
#
# Fix: link the backend targets DIRECTLY to transcribe. This brings each
# backend's own PRIVATE link line (-fsycl, cudart, ...) onto transcribe's link
# step, exactly as a static consumer would need (see transcribe-link.json's
# static `libraries` list: transcribe + ggml + backends + ggml-base).
if(TRANSCRIBE_SHARED_EMBED)
    # transcribe_backend_kinds is defined in cmake/transcribe-backend-kinds.cmake
    # (included later by transcribe-install.cmake at the top level). Include it
    # here so we can resolve backend target names during the src/ configure pass.
    include("${CMAKE_SOURCE_DIR}/cmake/transcribe-backend-kinds.cmake")
    transcribe_backend_kinds(_embed_kinds _embed_backend_targets)
    foreach(_be ${_embed_backend_targets})
        if(TARGET ${_be})
            target_link_libraries(transcribe PRIVATE ${_be})
        endif()
    endforeach()
endif()

if(_has_blas)
    target_link_libraries(transcribe PRIVATE ${_transcribe_extra_libs})
    target_compile_definitions(transcribe PRIVATE TRANSCRIBE_HAS_BLAS=1)
endif()

if(PROJECT_VERSION_MAJOR EQUAL 0)
    # Pre-1.0 policy permits ABI changes only on minor releases. Encode that
    # boundary in the shared-library identity so an old raw-C binary fails to
    # load a new incompatible minor instead of silently binding to it.
    set(TRANSCRIBE_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")
else()
    set(TRANSCRIBE_SOVERSION "${PROJECT_VERSION_MAJOR}")
endif()

set_target_properties(transcribe PROPERTIES
    OUTPUT_NAME transcribe
    POSITION_INDEPENDENT_CODE ON
    # Versions the shared lib (for 0.2.0: libtranscribe.0.2.0.dylib plus the
    # libtranscribe.0.2.dylib compatibility link). Ignored for the static archive. Before 1.0 SOVERSION tracks
    # major.minor because the public ABI policy allows breaks on minor bumps;
    # from 1.0 onward it tracks the major conventionally.
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${TRANSCRIBE_SOVERSION}
)

# miniz (vendored, src/third_party/miniz) provides the deflate compressor for
# Whisper's temperature-fallback compression_ratio metric (HF
# generation_whisper.py:_retrieve_compression_ratio uses
# len(token_bytes) / len(deflate(token_bytes)) to detect decoder token-id
# loops). Replaces system zlib so there is no find_package(ZLIB) / vcpkg
# dependency and nothing extra to stage into the install link manifest.
#
# Trim flags reach both miniz.c and its only consumer (arch/whisper/model.cpp),
# so they are PRIVATE on the target, not per-source:
#   NO_INFLATE_APIS  compress-only (this also makes miniz.h auto-define
#                    MINIZ_NO_ARCHIVE_APIS, so ZIP is off too — defining the
#                    latter here as well would clash with that and warn).
#   NO_STDIO         no file I/O    NO_TIME      no wall-clock
#   NO_ZLIB_COMPATIBLE_NAMES  don't export global compress/crc32/Z_OK/... that
#                             could collide with a real zlib downstream; call
#                             sites use the mz_-prefixed native API instead.
target_compile_definitions(transcribe PRIVATE
    MINIZ_NO_INFLATE_APIS
    MINIZ_NO_STDIO
    MINIZ_NO_TIME
    MINIZ_NO_ZLIB_COMPATIBLE_NAMES
)

# miniz is third-party C; its style trips our -Wshadow/-Wpedantic/-Wextra set.
# Silence warnings for this one translation unit only (-w / MSVC /w wins over
# the project flags transcribe_apply_warnings appends to the whole target).
if(MSVC)
    set_source_files_properties(third_party/miniz/miniz.c PROPERTIES COMPILE_OPTIONS "/W0")
else()
    set_source_files_properties(third_party/miniz/miniz.c PROPERTIES COMPILE_OPTIONS "-w")
endif()

if(WIN32)
    # Windows toolchains hide M_PI and friends behind _USE_MATH_DEFINES
    # (used by the mel frontends and Whisper's bin loader). MSVC and MinGW
    # both honor it — MinGW's strict-C++17 mode hides M_PI the same way.
    target_compile_definitions(transcribe PRIVATE _USE_MATH_DEFINES)
endif()

# ──────────────────────────────────────────────────────────────────────────
# Embed a Windows VS_VERSION_INFO resource so the build version is viewable
# via file Properties > Details (right-click transcribe.dll) WITHOUT loading
# the DLL. Only the SHARED library gets it: a static archive has no
# file-level Properties view, and the resource would be dead weight there.
# Numeric FILEVERSION derives directly from the integer
# TRANSCRIBE_VERSION_MAJOR/MINOR/PATCH parsed in the top-level CMakeLists.txt
# (no describe-parse fallback needed); the string fields carry the full
# provenance (version | commit | branch | date) for tools that read strings.
# ──────────────────────────────────────────────────────────────────────────
if(WIN32 AND MSVC AND (TRANSCRIBE_BUILD_SHARED OR TRANSCRIBE_SHARED_EMBED))
    set(TRANSCRIBE_RC_FILEVERSION
        "${TRANSCRIBE_VERSION_MAJOR},${TRANSCRIBE_VERSION_MINOR},${TRANSCRIBE_VERSION_PATCH},0")
    set(TRANSCRIBE_RC_FILE "${CMAKE_BINARY_DIR}/generated/transcribe-version.rc")
    configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/transcribe-version.rc.in"
        "${TRANSCRIBE_RC_FILE}"
        @ONLY)
    target_sources(transcribe PRIVATE "${TRANSCRIBE_RC_FILE}")
endif()

transcribe_apply_warnings(transcribe)

# -----------------------------------------------------------------------------
# Dynamic Architecture Plugins (TRANSCRIBE_ARCH_DL)
# -----------------------------------------------------------------------------
if(TRANSCRIBE_ARCH_DL)
    set_target_properties(transcribe PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

    add_custom_target(transcribe-plugins COMMENT "Build all enabled architecture plugins")

    # The install rules live at the TOP level (cmake/transcribe-install.cmake,
    # included after add_subdirectory(src)), so the target list has to be
    # promoted out of this directory scope — add_subdirectory() gives this file
    # its own scope and nothing set here is visible there otherwise.
    set(_arch_plugin_targets "")
    # The installed file names (`transcribe-arch-<arch>`), in the same order as
    # the targets above. They are what the install rules must key on — see the
    # prune in cmake/transcribe-install.cmake — and they are NOT the target
    # names whenever a family's directory is not spelled like its architecture.
    set(_arch_plugin_names "")

    foreach(_fam IN LISTS _enabled_families)
        set(_fam_srcs ${_srcs_${_fam}})
        # The mutually-dependent pair shares one source set: a parakeet plugin
        # without sortformer's diarizer and a sortformer plugin without
        # parakeet's encoder would each fail to link.
        if(_fam STREQUAL "parakeet" OR _fam STREQUAL "sortformer")
            set(_fam_srcs ${_srcs_parakeet_sortformer})
        endif()

        # A plugin module's FILE NAME is its lookup key: the loader derives
        # `transcribe-arch-<name>.dll` from the GGUF's `general.architecture`
        # string (transcribe-arch.cpp, find_arch) and finds nothing when the
        # module is named after the source directory instead. Four families
        # are spelled differently on disk than in the GGUF — granite /
        # granite5_ctc / granite_nar / cohere are directories whose
        # architectures are granite_speech / granite_speech5_ctc /
        # granite_speech_nar / cohere_asr — so naming the module after the
        # directory leaves those families unloadable in every DL build (the
        # model fails at open with "unsupported architecture").
        #
        # Read the name out of the family's own model.cpp rather than repeating
        # it here: that string is the exact value the loader compares against,
        # so deriving the file name from it makes the two impossible to
        # desync. The formatting is clang-format's (see the Arch initializer in
        # any src/arch/<family>/model.cpp).
        set(_fam_model_cpp "${CMAKE_CURRENT_SOURCE_DIR}/arch/${_fam}/model.cpp")
        file(STRINGS "${_fam_model_cpp}" _arch_name_lines
            REGEX "^[ \t]*/\\*[ \t]*\\.name[ \t]*=[ \t]*\\*/[ \t]*\"[A-Za-z0-9_]+\"")
        list(LENGTH _arch_name_lines _arch_name_count)
        if(_arch_name_count EQUAL 0)
            message(FATAL_ERROR
                "transcribe: no architecture name found in ${_fam_model_cpp}. The "
                "dynamic plugin's file name must equal the string its Arch is registered "
                "under (the loader compares it against general.architecture), so it is "
                "read from that file's Arch initializer; expected a line of the form "
                "'    /* .name = */ \"family_name\",'.")
        endif()
        list(GET _arch_name_lines 0 _arch_name_line)
        string(REGEX MATCH "\"([A-Za-z0-9_]+)\"" _arch_name_ignored "${_arch_name_line}")
        set(_plugin_name "${CMAKE_MATCH_1}")

        add_library(transcribe-arch-${_fam} SHARED
            ${_fam_srcs}
            transcribe-plugin-entry.cpp
        )
        add_dependencies(transcribe-plugins transcribe-arch-${_fam})
        string(TOUPPER "${_fam}" _fam_upper)
        target_compile_definitions(transcribe-arch-${_fam} PRIVATE
            TRANSCRIBE_BUILD=1
            TRANSCRIBE_ARCH_PLUGIN=1
            # The architecture this module provides, as the loader will ask for
            # it. The entry point stringifies this into plugin->arch_name and
            # the loader stores it as the loaded module's name, which its
            # already-loaded lookup (find_arch step 2) compares against the
            # GGUF's architecture — so it has to be the arch name, not the
            # directory. The namespace below stays the directory's: it names the
            # `transcribe::<ns>::arch` symbol the entry point exports.
            TRANSCRIBE_PLUGIN_ARCH_NAME=${_plugin_name}
            TRANSCRIBE_PLUGIN_ARCH_NS=${_fam}
            TRANSCRIBE_ENABLE_ARCH_${_fam_upper}=1
        )
        if("sortformer" IN_LIST _enabled_families)
            target_compile_definitions(transcribe-arch-${_fam} PRIVATE TRANSCRIBE_ENABLE_ARCH_SORTFORMER=1)
        endif()
        target_include_directories(transcribe-arch-${_fam}
            PUBLIC
                $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${CMAKE_BINARY_DIR}/generated
        )
        target_link_libraries(transcribe-arch-${_fam} PRIVATE transcribe ggml)
        if(_has_blas)
            target_compile_definitions(transcribe-arch-${_fam} PRIVATE TRANSCRIBE_HAS_BLAS=1)
            target_link_libraries(transcribe-arch-${_fam} PRIVATE ${_transcribe_extra_libs})
        endif()
        if(_fam STREQUAL "whisper")
            target_compile_definitions(transcribe-arch-${_fam} PRIVATE
                MINIZ_NO_INFLATE_APIS
                MINIZ_NO_STDIO
                MINIZ_NO_TIME
                MINIZ_NO_ZLIB_COMPATIBLE_NAMES
            )
        endif()
        if(WIN32)
            target_compile_definitions(transcribe-arch-${_fam} PRIVATE _USE_MATH_DEFINES)
        endif()
        set_target_properties(transcribe-arch-${_fam} PROPERTIES
            OUTPUT_NAME "transcribe-arch-${_plugin_name}"
            PREFIX ""
            POSITION_INDEPENDENT_CODE ON
        )
        transcribe_apply_warnings(transcribe-arch-${_fam})
        list(APPEND _arch_plugin_targets transcribe-arch-${_fam})
        list(APPEND _arch_plugin_names "transcribe-arch-${_plugin_name}")
    endforeach()

    set(TRANSCRIBE_ARCH_PLUGIN_TARGETS "${_arch_plugin_targets}" PARENT_SCOPE)
    # File stems, for the install rules that glob the prefix (the prune) and for
    # the link manifest's arch_plugins list. Kept separate from the target list
    # above because the two differ wherever a family directory is not spelled
    # like its architecture.
    set(TRANSCRIBE_ARCH_PLUGIN_NAMES "${_arch_plugin_names}" PARENT_SCOPE)
endif()
