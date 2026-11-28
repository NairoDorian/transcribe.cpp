// transcribe-backend.h - internal backend selection types.
//
// INTERNAL, C++17. Not part of the public ABI.
//
// The public API exposes a small transcribe_backend_request enum.
// Internally the library needs two things that don't belong in the public
// header: a typed classification of the ggml backend it landed on
// (BackendKind, replacing string-matching on ggml_backend_name()), and a
// BackendPlan holding the request, primary backend handle, its kind, and
// the scheduler list in priority order. Every per-family load() resolves a
// plan and hands it to helpers that key off the primary backend (F16→F32
// conv promotion, flash-attn defaults).

#pragma once

#include "ggml-backend.h"
#include "transcribe.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
struct ggml_backend_device;
typedef struct ggml_backend_device * ggml_backend_dev_t;

namespace transcribe {

// Library-internal classification from the ggml backend device type plus
// registry name. Only the kinds the library branches on are listed; anything
// else is Unknown.
enum class BackendKind {
    Unknown = 0,
    Cpu,       // ggml CPU backend (strict system memory)
    Metal,     // Apple Metal
    Vulkan,    // Vulkan compute
    Cuda,      // NVIDIA CUDA
    Rocm,      // AMD ROCm
    Sycl,      // Intel oneAPI / SYCL
    Accel,     // BLAS / AMX / other host-memory accelerator
    OtherGpu,  // GPU/IGPU device we don't have a special case for
};

// Human-readable kind label for logs. Never nullptr.
const char * kind_name(BackendKind kind);

// Unit-testable classification core. Production code normally calls
// classify_device(), which fetches these two inputs from ggml.
BackendKind classify_backend_type(enum ggml_backend_dev_type dev_type, const char * reg_name);

// Classify a backend device into a BackendKind. Uses
// ggml_backend_dev_type for the GPU/IGPU/ACCEL/CPU dimension and the
// reg name ("MTL", "Vulkan", "CUDA", "ROCm", "SYCL", ...) to resolve
// the vendor. Never returns Unknown for a valid device pointer.
BackendKind classify_device(ggml_backend_dev_t dev);

// Probe order for GPU device selection. `dev_types` holds the ggml
// device type of every registry device, indexed by registry position;
// the returned indices are the candidates to try: every discrete GPU
// (registry order) before any integrated GPU (registry order), non-GPU
// devices excluded. Discrete-first matters because Vulkan enumeration
// on hybrid-graphics machines often lists the display iGPU before the
// dGPU. Unit-testable core for try_init_kind.
std::vector<size_t> gpu_probe_order(const std::vector<enum ggml_backend_dev_type> & dev_types);

// A resolved backend plan. Produced by load_common::init_backends
// from a transcribe_backend_request and consumed by every helper
// that needs to know where the graph will run.
//
//   requested:       the original caller request, preserved for
//                    logging / diagnostics.
//   primary:         the first (highest-priority) backend in the
//                    scheduler list. This is the backend that owns
//                    the weight buffer and runs most of the graph.
//   primary_kind:    classified kind of `primary`. Helpers check this
//                    directly instead of calling ggml_backend_name
//                    and string-matching.
//   scheduler_list:  every backend that should participate in the
//                    ggml scheduler, in priority order. Primary
//                    first, then ACCEL (when appropriate), then CPU
//                    last as the fallback. Cleaned up in reverse in
//                    the model destructor.
struct BackendPlan {
    transcribe_backend_request  requested    = TRANSCRIBE_BACKEND_AUTO;
    ggml_backend_t              primary      = nullptr;
    BackendKind                 primary_kind = BackendKind::Unknown;
    std::vector<ggml_backend_t> scheduler_list;
};

// Check if a backend device is the CPU backend. DL-safe (uses ggml_backend_dev_type).
bool is_cpu_backend(ggml_backend_t backend);

// Attach or update a persistent threadpool on a CPU backend.
// Reuses the existing threadpool if parameters match; otherwise frees the previous
// threadpool and instantiates a new one with normal OS scheduling.
void safe_set_cpu_backend_threadpool(ggml_backend_t backend, int n_threads);

// Park an idle pool without freeing its workers. The next configure reattaches
// it, and GGML resumes it when a graph is dispatched.
void pause_cpu_backend_threadpool(ggml_backend_t backend);

// Detach and cleanly free any threadpool associated with the backend.
void cleanup_cpu_backend_threadpool(ggml_backend_t backend) noexcept;

// No-throw wrappers for ggml backend teardown. Family destructors are
// implicitly noexcept, so raw backend frees must not appear in library code;
// tests/lint_teardown.cmake enforces this. NULL is a no-op.
//
// Test hook: non-empty TRANSCRIBE_TEST_TEARDOWN_THROW injects an internal
// throw after the real free, proving containment without leaking the handle.
void safe_backend_free(ggml_backend_t backend) noexcept;
void safe_buffer_free(ggml_backend_buffer_t buffer) noexcept;
void safe_sched_free(ggml_backend_sched_t sched) noexcept;

// Release a session's per-run compute scratch: the scheduler (whose
// allocator only ever grows) first, then the no_alloc graph context. Both
// are nulled; families re-create them lazily on the next run. Used by
// transcribe_session::release_scratch and the base destructor, keeping the
// sched / compute_ctx free order in one place. NULLs are no-ops.
void release_compute_scratch(ggml_backend_sched_t & sched, struct ggml_context *& compute_ctx) noexcept;

// Host memory for one graph's ggml metadata (its tensor and graph structs),
// kept across runs. ggml-cuda keys a captured CUDA graph by the address of the
// graph's first node, and a graph built in a freshly allocated context lands
// at new addresses every run: the backend then sees a new graph, runs it once
// without a CUDA graph, then captures and instantiates it from scratch.
// Rebuilt in the same arena, the structs land where they were, so the next
// run (or stream tick) finds the CUDA graph it captured last time: replayed as
// is when every node's buffers and shapes match, re-captured and updated in
// place (cudaGraphExecUpdate) when one moved. ggml-cuda compares every node
// before it replays, so only the addresses carry over: init() starts an empty
// context and every tensor is created afresh. One live context per arena:
// free the previous one before the next init(). For a graph computed many
// times per run (a decode step); a one-shot graph gains nothing from it.
class GraphArena {
  public:
    // A no_alloc context of `bytes` over the arena (grown if smaller); nullptr
    // if the memory cannot be had.
    struct ggml_context * init(size_t bytes);

  private:
    std::unique_ptr<uint8_t[]> buf_;
    size_t                     size_ = 0;
};

// --- reclaiming device memory on an allocation-failure path (A7) -------------
//
// Optional CUDA hooks are supplied by patches/ggml/0101. Reclamation is
// restricted to the caller-owned backend; sibling models may be computing or
// unloading concurrently and must never be trimmed via an unowned pointer.
//
// Return a CUDA/HIP backend's cached, idle pool memory to the driver. No-op on
// other backends. Safe on an idle backend; do not call with work in flight.
void trim_backend_pools(ggml_backend_t backend);

// Drop a backend's cached compiled-graph state for `graph`, so a later
// same-shape graph does not inherit the device memory the cached instance
// holds. Null-safe; no-op where CUDA/HIP graphs are not compiled in.
void evict_backend_graph_cache(ggml_backend_t backend, struct ggml_cgraph * graph);

// Allocate, then reclaim this idle backend's pools/graph on failure and retry
// exactly once. The caller must serialize access to its backend. No global
// backend enumeration or synchronization occurs on the successful fast path.
ggml_backend_buffer_t alloc_ctx_tensors_with_reclaim(ggml_backend_t       backend,
                                                     ggml_context *       ctx,
                                                     struct ggml_cgraph * graph = nullptr);

inline ggml_backend_buffer_t alloc_ctx_tensors_with_reclaim(ggml_context *       ctx,
                                                            ggml_backend_t       backend,
                                                            struct ggml_cgraph * graph = nullptr) {
    return alloc_ctx_tensors_with_reclaim(backend, ctx, graph);
}

// ggml_backend_sched_alloc_graph() plus a DEBUG-level scratch log. Returns the
// same bool as the ggml call.
//
// The log is the only direct read on graph memory versus audio length: process
// RSS misses GPU buffers, and the scheduler's per-backend buffer is what the
// backend must allocate. A line is emitted only when some graph raises that
// buffer's high-water mark, so per-token step reallocations stay silent.
// `tag`/`line` identify the call site; `extent` (>= 0) is an optional
// graph-length field to log alongside it (e.g. the encoder frame count).
bool alloc_sched_graph(ggml_backend_sched_t sched,
                       struct ggml_cgraph * graph,
                       const char *         tag,
                       int                  line,
                       long long            extent = -1);

#define TRANSCRIBE_ALLOC_SCHED_GRAPH(sched, graph, tag) \
    (::transcribe::alloc_sched_graph((sched), (graph), (tag), __LINE__))

// Same, with a graph-length field (e.g. encoder frames) in the log line.
#define TRANSCRIBE_ALLOC_SCHED_GRAPH_EXTENT(sched, graph, tag, extent) \
    (::transcribe::alloc_sched_graph((sched), (graph), (tag), __LINE__, (extent)))

}  // namespace transcribe
