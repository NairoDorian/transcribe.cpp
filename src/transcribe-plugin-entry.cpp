// transcribe-plugin-entry.cpp - Standardized entry point for architecture plugins.
#include "transcribe-arch.h"
#include "transcribe-plugin.h"

#ifndef TRANSCRIBE_PLUGIN_ARCH_NAME
#    error "TRANSCRIBE_PLUGIN_ARCH_NAME must be defined"
#endif

#ifndef TRANSCRIBE_PLUGIN_ARCH_NS
#    error "TRANSCRIBE_PLUGIN_ARCH_NS must be defined"
#endif

namespace transcribe {
namespace TRANSCRIBE_PLUGIN_ARCH_NS {
extern const Arch arch;
}  // namespace TRANSCRIBE_PLUGIN_ARCH_NS
}  // namespace transcribe

#define TRANSCRIBE_STR_IMPL(x) #x
#define TRANSCRIBE_STR(x)      TRANSCRIBE_STR_IMPL(x)

extern "C" TRANSCRIBE_PLUGIN_EXPORT const transcribe_arch_plugin * transcribe_arch_plugin_get(void) {
    static const transcribe_arch_plugin s_plugin = {
        TRANSCRIBE_ARCH_PLUGIN_ABI_VERSION,
        TRANSCRIBE_STR(TRANSCRIBE_PLUGIN_ARCH_NAME),
        &transcribe::TRANSCRIBE_PLUGIN_ARCH_NS::arch,
    };
    return &s_plugin;
}
