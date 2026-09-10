// transcribe-backend.cpp - internal backend selection helpers.
//
// See transcribe-backend.h for rationale. This file owns the
// device-classification rules: given a ggml_backend_dev_t, what
// library-level BackendKind does it correspond to?

#include "transcribe-backend.h"

#include "ggml-cpu.h"
#include "ggml.h"
#include "transcribe-batch-util.h"
#include "transcribe-log.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace transcribe {

const char * kind_name(BackendKind kind) {
    switch (kind) {
        case BackendKind::Cpu:
            return "cpu";
        case BackendKind::Metal:
            return "metal";
        case BackendKind::Vulkan:
            return "vulkan";
        case BackendKind::Cuda:
            return "cuda";
        case BackendKind::Rocm:
            return "rocm";
        case BackendKind::Sycl:
            return "sycl";
        case BackendKind::Accel:
            return "accel";
        case BackendKind::OtherGpu:
            return "gpu";
        case BackendKind::Unknown:
        default:
            return "unknown";
    }
}

// Return true if `reg_name` (the ggml backend registry name) starts
// with the given prefix. ggml's registry names look like "MTL",
// "Vulkan", "CUDA", "ROCm", "SYCL", "BLAS", "CPU", etc. Prefix matching is
// intentional: registry names can get version suffixes or device
// index suffixes in some ggml builds.
static bool reg_name_is(const char * reg_name, const char * prefix) {
    if (reg_name == nullptr || prefix == nullptr) {
        return false;
    }
    return std::strncmp(reg_name, prefix, std::strlen(prefix)) == 0;
}

BackendKind classify_backend_type(enum ggml_backend_dev_type dev_type, const char * reg_name) {
    if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return BackendKind::Cpu;
    }
    if (dev_type == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        return BackendKind::Accel;
    }
    if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU && dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return BackendKind::Unknown;
    }

    if (reg_name_is(reg_name, "MTL") || reg_name_is(reg_name, "Metal")) {
        return BackendKind::Metal;
    } else if (reg_name_is(reg_name, "Vulkan")) {
        return BackendKind::Vulkan;
    } else if (reg_name_is(reg_name, "CUDA")) {
        return BackendKind::Cuda;
    } else if (reg_name_is(reg_name, "ROCm")) {
        return BackendKind::Rocm;
    } else if (reg_name_is(reg_name, "SYCL")) {
        return BackendKind::Sycl;
    }

    return BackendKind::OtherGpu;
}

BackendKind classify_device(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return BackendKind::Unknown;
    }

    // First cut: ggml's device-type classification. This tells us CPU
    // vs GPU vs IGPU vs ACCEL without any name matching. For GPU and
    // IGPU devices, the registry name resolves the vendor-specific kind.
    const auto         dev_type = ggml_backend_dev_type(dev);
    ggml_backend_reg_t reg      = ggml_backend_dev_backend_reg(dev);
    const char *       reg_name = (reg != nullptr) ? ggml_backend_reg_name(reg) : nullptr;
    return classify_backend_type(dev_type, reg_name);
}

std::vector<size_t> gpu_probe_order(const std::vector<enum ggml_backend_dev_type> & dev_types) {
    std::vector<size_t> order;
    order.reserve(dev_types.size());
    for (size_t i = 0; i < dev_types.size(); ++i) {
        if (dev_types[i] == GGML_BACKEND_DEVICE_TYPE_GPU) {
            order.push_back(i);
        }
    }
    for (size_t i = 0; i < dev_types.size(); ++i) {
        if (dev_types[i] == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            order.push_back(i);
        }
    }
    return order;
}

namespace {

// Shared body for safe_* teardown wrappers. The test hook fires after the
// real free; present-but-empty is inert.
template <typename Fn> void contained_free(const char * what, Fn && do_free) noexcept {
    try {
        do_free();
        if (const char * hook = std::getenv("TRANSCRIBE_TEST_TEARDOWN_THROW"); hook != nullptr && hook[0] != '\0') {
            throw std::runtime_error("TRANSCRIBE_TEST_TEARDOWN_THROW fault injection");
        }
    } catch (const std::exception & e) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN, "%s threw during teardown (contained; resource may leak): %s", what,
                e.what());
    } catch (...) {
        log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                "%s threw an unknown exception during teardown (contained; resource may leak)", what);
    }
}

struct BackendTpEntry {
    ggml_threadpool_t      tp = nullptr;
    ggml_threadpool_params params{};
};

std::mutex                                         g_backend_tp_mutex;
std::unordered_map<ggml_backend_t, BackendTpEntry> g_backend_tps;

typedef ggml_threadpool_t (*pfn_threadpool_new)(ggml_threadpool_params *);
typedef void (*pfn_threadpool_free)(ggml_threadpool_t);
typedef void (*pfn_set_threadpool)(ggml_backend_t, ggml_threadpool_t);

static void * cpu_backend_proc(ggml_backend_t backend, const char * name) {
    ggml_backend_dev_t dev = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
    ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg != nullptr ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

}  // namespace

bool is_cpu_backend(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    return dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

void safe_set_cpu_backend_threadpool(ggml_backend_t backend, int n_threads) {
    if (!is_cpu_backend(backend)) {
        return;
    }
    auto tp_set  = reinterpret_cast<pfn_set_threadpool>(cpu_backend_proc(backend, "ggml_backend_cpu_set_threadpool"));
    auto tp_free = reinterpret_cast<pfn_threadpool_free>(cpu_backend_proc(backend, "ggml_threadpool_free"));
    auto tp_new  = reinterpret_cast<pfn_threadpool_new>(cpu_backend_proc(backend, "ggml_threadpool_new"));
    if (tp_set == nullptr || tp_free == nullptr || tp_new == nullptr) {
        return;
    }
    if (n_threads <= 0) {
        n_threads = default_n_threads();
    }
    const ggml_threadpool_params desired = make_threadpool_params(n_threads);

    std::lock_guard<std::mutex> lock(g_backend_tp_mutex);
    auto                        it = g_backend_tps.find(backend);
    if (it != g_backend_tps.end()) {
        if (ggml_threadpool_params_match(&it->second.params, &desired)) {
            return;
        }
        tp_set(backend, nullptr);
        if (it->second.tp != nullptr) {
            tp_free(it->second.tp);
        }
        g_backend_tps.erase(it);
    }

    ggml_threadpool_params params_copy = desired;
    ggml_threadpool_t      tp          = tp_new(&params_copy);
    if (tp != nullptr) {
        tp_set(backend, tp);
        g_backend_tps[backend] = { tp, desired };
    }
}

void cleanup_cpu_backend_threadpool(ggml_backend_t backend) noexcept {
    if (backend == nullptr) {
        return;
    }
    ggml_threadpool_t to_free = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_backend_tp_mutex);
        auto                        it = g_backend_tps.find(backend);
        if (it != g_backend_tps.end()) {
            to_free = it->second.tp;
            g_backend_tps.erase(it);
        }
    }
    if (to_free != nullptr) {
        try {
            if (is_cpu_backend(backend)) {
                if (auto tp_set = reinterpret_cast<pfn_set_threadpool>(cpu_backend_proc(backend, "ggml_backend_cpu_set_threadpool"))) {
                    tp_set(backend, nullptr);
                }
            }
            if (auto tp_free = reinterpret_cast<pfn_threadpool_free>(cpu_backend_proc(backend, "ggml_threadpool_free"))) {
                tp_free(to_free);
            }
        } catch (...) {
            // Teardown containment
        }
    }
}

void safe_backend_free(ggml_backend_t backend) noexcept {
    if (backend == nullptr) {
        return;
    }
    cleanup_cpu_backend_threadpool(backend);
    contained_free("ggml_backend_free", [&] { ggml_backend_free(backend); });
}

void safe_buffer_free(ggml_backend_buffer_t buffer) noexcept {
    if (buffer == nullptr) {
        return;
    }
    contained_free("ggml_backend_buffer_free", [&] { ggml_backend_buffer_free(buffer); });
}

void safe_sched_free(ggml_backend_sched_t sched) noexcept {
    if (sched == nullptr) {
        return;
    }
    contained_free("ggml_backend_sched_free", [&] { ggml_backend_sched_free(sched); });
}

void release_compute_scratch(ggml_backend_sched_t & sched, struct ggml_context *& compute_ctx) noexcept {
    safe_sched_free(sched);
    sched = nullptr;
    if (compute_ctx != nullptr) {
        ggml_free(compute_ctx);
        compute_ctx = nullptr;
    }
}

}  // namespace transcribe
