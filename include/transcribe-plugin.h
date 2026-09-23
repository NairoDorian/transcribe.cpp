// include/transcribe-plugin.h - Architecture Plugin Protocol for transcribe.cpp
//
// Allows speech-to-text model architectures (e.g. Parakeet, Granite, Qwen3-ASR)
// to be packaged and distributed as independent dynamically-loaded modules
// (transcribe-arch-<family>.dll / .so / .dylib).

#pragma once

#include "transcribe.h"

#include <stdint.h>

#define TRANSCRIBE_ARCH_PLUGIN_ABI_VERSION 1

#if defined(_WIN32)
#    define TRANSCRIBE_PLUGIN_EXPORT __declspec(dllexport)
#else
#    define TRANSCRIBE_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
namespace transcribe {
struct Arch;
}

extern "C" {
#endif

// Forward declaration of internal Arch trait
struct transcribe_arch;

typedef struct transcribe_arch_plugin {
    // ABI version for compatibility verification
    uint32_t     abi_version;
    // The canonical architecture name (e.g. "parakeet", "granite", "qwen3_asr")
    const char * arch_name;
    // Pointer to the family's Arch instance
    const void * arch;
} transcribe_arch_plugin;

// Entry point signature exported by all architecture plugin modules
typedef const transcribe_arch_plugin * (*transcribe_arch_plugin_entry_fn)(void);

#define TRANSCRIBE_ARCH_PLUGIN_ENTRY_NAME "transcribe_arch_plugin_get"

#define TRANSCRIBE_ARCH_PLUGIN_DEFINE(name_str, arch_ref)                                                 \
    extern "C" TRANSCRIBE_PLUGIN_EXPORT const transcribe_arch_plugin * transcribe_arch_plugin_get(void) { \
        static const transcribe_arch_plugin s_plugin = {                                                  \
            TRANSCRIBE_ARCH_PLUGIN_ABI_VERSION,                                                           \
            name_str,                                                                                     \
            reinterpret_cast<const void *>(&(arch_ref)),                                                  \
        };                                                                                                \
        return &s_plugin;                                                                                 \
    }

#ifdef __cplusplus
}
#endif
