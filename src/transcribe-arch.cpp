// transcribe-arch.cpp - architecture registry and dynamic plugin loader.
//
// Supports both compiled-in static architectures and dynamically-loaded
// per-family plugin modules (transcribe-arch-<arch>.dll / .so / .dylib, where
// <arch> is the name the Arch is registered under — the same string it is
// matched against in a model's general.architecture — so a plugin module is
// named after its architecture and not after its source directory).

#include "transcribe-arch.h"

#include "transcribe-log.h"
#include "transcribe-plugin.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace transcribe {

// Per-family static Arch instances (when compiled-in)
#ifdef TRANSCRIBE_ENABLE_ARCH_PARAKEET
namespace parakeet {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_COHERE
namespace cohere {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_CANARY
namespace canary {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_QWEN3_ASR
namespace qwen3_asr {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_MOSS
namespace moss {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_VOXTRAL
namespace voxtral {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_VOXTRAL_REALTIME
namespace voxtral_realtime {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_CANARY_QWEN
namespace canary_qwen {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_WHISPER
namespace whisper {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_MOONSHINE
namespace moonshine {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_MOONSHINE_STREAMING
namespace moonshine_streaming {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_SENSEVOICE
namespace sensevoice {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_FUNASR_NANO
namespace funasr_nano {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_GIGAAM
namespace gigaam {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_GRANITE
namespace granite {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_GRANITE_NAR
namespace granite_nar {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_MEDASR
namespace medasr {
extern const Arch arch;
}
#endif

#ifdef TRANSCRIBE_ENABLE_ARCH_SORTFORMER
namespace sortformer {
extern const Arch arch;
}
#endif

namespace {

struct LoadedArchPlugin {
    std::string           name;
    std::filesystem::path path;
    void *                handle = nullptr;
    const Arch *          arch   = nullptr;
};

std::mutex                         g_plugin_mutex;
std::vector<LoadedArchPlugin>      g_loaded_plugins;
std::vector<std::filesystem::path> g_custom_dirs;

std::filesystem::path self_dir() {
#if defined(_WIN32)
    HMODULE    module = nullptr;
    const auto flags  = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    if (!GetModuleHandleExW(flags, reinterpret_cast<LPCWSTR>(&find_arch), &module)) {
        return {};
    }

    std::wstring path(1024, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
        if (n == 0) {
            return {};
        }
        if (n < path.size()) {
            path.resize(n);
            break;
        }
        if (path.size() >= 32768) {
            return {};
        }
        path.resize(path.size() * 2);
    }
    return std::filesystem::path(path).parent_path();
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&find_arch), &info) == 0 || info.dli_fname == nullptr ||
        info.dli_fname[0] == '\0') {
        return {};
    }
    std::filesystem::path path(info.dli_fname);
    std::error_code       ec;
    auto                  canonical = std::filesystem::weakly_canonical(path, ec);
    if (ec) {
        canonical = std::filesystem::absolute(path, ec);
    }
    if (ec) {
        canonical = path;
    }
    return canonical.parent_path();
#endif
}

const Arch * load_plugin_file(const std::filesystem::path & file_path, const char * expected_name) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(file_path, ec)) {
        return nullptr;
    }

    // Already loaded -> hand back the SAME Arch instead of dlopen-ing again.
    // Without this, a host that registers several plugin directories (the
    // documented layout: next to libtranscribe, <app_dir>/arch,
    // TRANSCRIBE_ARCH_DIR, ...) and scans each one loads a plugin visible in
    // more than one of them once PER directory: the module's refcount climbs,
    // g_loaded_plugins grows a duplicate entry per load, and every duplicate
    // re-runs the plugin's static initializers. The canonical path is the
    // identity, so two spellings of one file still collapse to one load.
    //
    // Caller must hold g_plugin_mutex. Both call sites (load_arch_plugin and
    // find_arch) take it and keep it held across this call, so this reads the
    // vector under the same lock rather than re-acquiring it — g_plugin_mutex
    // is a plain std::mutex, and a nested lock_guard here would self-deadlock.
    std::filesystem::path canonical = std::filesystem::weakly_canonical(file_path, ec);
    if (ec) {
        canonical = std::filesystem::absolute(file_path, ec);
    }
    if (ec) {
        canonical = file_path;
    }
    for (const auto & item : g_loaded_plugins) {
        if (item.path == canonical) {
            return item.arch;
        }
    }

    void * module_handle = nullptr;
#if defined(_WIN32)
    module_handle = LoadLibraryW(file_path.c_str());
    if (module_handle == nullptr) {
        return nullptr;
    }
    auto entry_fn = reinterpret_cast<transcribe_arch_plugin_entry_fn>(
        GetProcAddress(static_cast<HMODULE>(module_handle), TRANSCRIBE_ARCH_PLUGIN_ENTRY_NAME));
#else
    module_handle = dlopen(file_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module_handle == nullptr) {
        return nullptr;
    }
    auto entry_fn =
        reinterpret_cast<transcribe_arch_plugin_entry_fn>(dlsym(module_handle, TRANSCRIBE_ARCH_PLUGIN_ENTRY_NAME));
#endif

    if (entry_fn == nullptr) {
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(module_handle));
#else
        dlclose(module_handle);
#endif
        return nullptr;
    }

    const transcribe_arch_plugin * plugin = entry_fn();
    if (plugin == nullptr || plugin->abi_version != TRANSCRIBE_ARCH_PLUGIN_ABI_VERSION || plugin->arch == nullptr) {
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(module_handle));
#else
        dlclose(module_handle);
#endif
        return nullptr;
    }

    const auto * arch = static_cast<const Arch *>(plugin->arch);
    if (expected_name != nullptr && plugin->arch_name != nullptr) {
        if (std::strcmp(plugin->arch_name, expected_name) != 0 && std::strcmp(arch->name, expected_name) != 0) {
#if defined(_WIN32)
            FreeLibrary(static_cast<HMODULE>(module_handle));
#else
            dlclose(module_handle);
#endif
            return nullptr;
        }
    }

    LoadedArchPlugin loaded;
    loaded.name   = plugin->arch_name ? plugin->arch_name : (arch->name ? arch->name : "");
    // Store the canonical path — it is the identity the dedup guard above
    // compares against, so a plugin reached by two different spellings (a
    // relative one from a scan, an absolute one from an explicit load) still
    // resolves to a single entry.
    loaded.path   = canonical;
    loaded.handle = module_handle;
    loaded.arch   = arch;

    transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_INFO, "transcribe: loaded dynamic architecture plugin '%s' from %s\n",
                        loaded.name.c_str(), file_path.string().c_str());

    g_loaded_plugins.push_back(std::move(loaded));
    return arch;
}

}  // namespace

transcribe_status register_arch_dir(const char * dir) {
    if (dir == nullptr || dir[0] == '\0') {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    std::filesystem::path p(dir);
    std::error_code       ec;
    if (!std::filesystem::is_directory(p, ec)) {
        return TRANSCRIBE_ERR_FILE_NOT_FOUND;
    }
    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    g_custom_dirs.push_back(std::filesystem::canonical(p, ec));
    return TRANSCRIBE_OK;
}

transcribe_status load_arch_plugin(const char * path) {
    if (path == nullptr || path[0] == '\0') {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    std::filesystem::path p(path);
    std::error_code       ec;
    if (!std::filesystem::is_regular_file(p, ec)) {
        return TRANSCRIBE_ERR_FILE_NOT_FOUND;
    }
    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    const Arch *                arch = load_plugin_file(p, nullptr);
    if (arch == nullptr) {
        return TRANSCRIBE_ERR_UNSUPPORTED_ARCH;
    }
    return TRANSCRIBE_OK;
}

const Arch * find_arch(const char * name, const char * model_hint_path) {
    if (name == nullptr) {
        return nullptr;
    }

    // 1. Built-in static table
    static const Arch * const k_archs[] = {
#ifdef TRANSCRIBE_ENABLE_ARCH_PARAKEET
        &parakeet::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_COHERE
        &cohere::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_CANARY
        &canary::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_QWEN3_ASR
        &qwen3_asr::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_VOXTRAL
        &voxtral::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_VOXTRAL_REALTIME
        &voxtral_realtime::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_CANARY_QWEN
        &canary_qwen::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_WHISPER
        &whisper::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_MOONSHINE
        &moonshine::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_MOONSHINE_STREAMING
        &moonshine_streaming::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_SENSEVOICE
        &sensevoice::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_FUNASR_NANO
        &funasr_nano::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_GIGAAM
        &gigaam::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_GRANITE
        &granite::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_GRANITE_NAR
        &granite_nar::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_MEDASR
        &medasr::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_MOSS
        &moss::arch,
#endif
#ifdef TRANSCRIBE_ENABLE_ARCH_SORTFORMER
        &sortformer::arch,
#endif
        nullptr,
    };
    constexpr size_t k_n = sizeof(k_archs) / sizeof(k_archs[0]);

    for (size_t i = 0; i < k_n; ++i) {
        const Arch * a = k_archs[i];
        if (a == nullptr || a->name == nullptr) {
            continue;
        }
        if (std::strcmp(a->name, name) == 0) {
            return a;
        }
    }

    // 2. Previously loaded dynamic plugins
    std::lock_guard<std::mutex> lock(g_plugin_mutex);
    for (const auto & item : g_loaded_plugins) {
        if (item.name == name || (item.arch && item.arch->name && std::strcmp(item.arch->name, name) == 0)) {
            return item.arch;
        }
    }

    // 3. Search for dynamic plugin on disk
    std::vector<std::filesystem::path> search_dirs;

    if (model_hint_path != nullptr && model_hint_path[0] != '\0') {
        std::filesystem::path mp(model_hint_path);
        std::filesystem::path mdir = mp.parent_path();
        if (!mdir.empty()) {
            search_dirs.push_back(mdir);
            search_dirs.push_back(mdir / "arch");
        }
    }

    for (const auto & cd : g_custom_dirs) {
        search_dirs.push_back(cd);
    }

    if (const char * env_dir = std::getenv("TRANSCRIBE_ARCH_DIR")) {
        if (env_dir[0] != '\0') {
            search_dirs.emplace_back(env_dir);
        }
    }

    std::filesystem::path sdir = self_dir();
    if (!sdir.empty()) {
        search_dirs.push_back(sdir / "arch");
        search_dirs.push_back(sdir);
    }

    search_dirs.push_back(std::filesystem::current_path() / "arch");
    search_dirs.push_back(std::filesystem::current_path());

    std::vector<std::string> candidates;
#if defined(_WIN32)
    candidates.push_back(std::string("transcribe-arch-") + name + ".dll");
#elif defined(__APPLE__)
    candidates.push_back(std::string("libtranscribe-arch-") + name + ".dylib");
    candidates.push_back(std::string("transcribe-arch-") + name + ".dylib");
#else
    candidates.push_back(std::string("libtranscribe-arch-") + name + ".so");
    candidates.push_back(std::string("transcribe-arch-") + name + ".so");
#endif

    for (const auto & d : search_dirs) {
        for (const auto & c : candidates) {
            std::filesystem::path target = d / c;
            if (const Arch * arch = load_plugin_file(target, name)) {
                return arch;
            }
        }
    }

    return nullptr;
}

}  // namespace transcribe
