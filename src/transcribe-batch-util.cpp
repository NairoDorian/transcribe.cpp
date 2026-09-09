// transcribe-batch-util.cpp - see transcribe-batch-util.h.

#include "transcribe-batch-util.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "transcribe-backend.h"
#include "transcribe-log.h"
#include "transcribe-session.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#    include <sched.h>

#    include <set>
#elif defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
#elif defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX  // else <windows.h>'s min/max macros clobber std::min/std::max
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

namespace transcribe {

// Number of CPUs the process is actually allowed to run on. Falls back to
// hardware_concurrency() when the platform query is unavailable or fails.
static int usable_cpu_count() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int n = CPU_COUNT(&set);
        if (n > 0) {
            return n;
        }
    }
#elif defined(_WIN32)
    DWORD_PTR proc_mask = 0, sys_mask = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask) && proc_mask != 0) {
        int n = 0;
        for (DWORD_PTR m = proc_mask; m != 0; m &= (m - 1)) {
            ++n;  // popcount
        }
        if (n > 0) {
            return n;
        }
    }
#endif
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 1;
}

namespace {

#if defined(__linux__)
// Highest CPU index read_cpu_list() will accept, so a malformed sysfs range
// cannot spin the parser.
constexpr int kMaxCpuIndex = 4096;

// Read a Linux CPU list ("0-5,8,10-11") into `out`. Returns false if the file
// is absent or unparsable, leaving `out` untouched.
bool read_cpu_list(const char * path, std::set<int> & out) {
    std::FILE * f = std::fopen(path, "r");
    if (f == nullptr) {
        return false;
    }
    char         buf[4096];
    const char * line = std::fgets(buf, sizeof(buf), f);
    std::fclose(f);
    if (line == nullptr) {
        return false;
    }
    std::set<int> parsed;
    const char *  p = buf;
    while (*p != 0) {
        char *     end = nullptr;
        const long lo  = std::strtol(p, &end, 10);
        if (end == p) {
            break;
        }
        long hi = lo;
        p       = end;
        if (*p == '-') {
            ++p;
            hi = std::strtol(p, &end, 10);
            if (end == p) {
                break;
            }
            p = end;
        }
        for (long c = lo; c <= hi && c >= 0 && c < kMaxCpuIndex; ++c) {
            parsed.insert(static_cast<int>(c));
        }
        if (*p == ',') {
            ++p;
        } else {
            break;
        }
    }
    if (parsed.empty()) {
        return false;
    }
    out = std::move(parsed);
    return true;
}

// Read one small non-negative integer out of a sysfs file. Returns -1 on failure.
int read_sysfs_int(const char * fmt, int cpu) {
    char path[256];
    std::snprintf(path, sizeof(path), fmt, cpu);
    std::FILE * f = std::fopen(path, "r");
    if (f == nullptr) {
        return -1;
    }
    long      v  = -1;
    const int ok = std::fscanf(f, "%ld", &v);
    std::fclose(f);
    return (ok == 1 && v >= 0) ? static_cast<int>(v) : -1;
}
#endif  // __linux__

struct CoreLogicalInfo {
    std::vector<int> primary;    // 1 primary logical CPU per physical core
    std::vector<int> secondary;  // SMT siblings on those same physical cores
};

#if defined(_WIN32)
static CoreLogicalInfo query_platform_cores() {
    CoreLogicalInfo res;
    DWORD_PTR       proc_mask = 0, sys_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask) || proc_mask == 0) {
        return res;
    }
    DWORD len = 0;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) {
        return res;
    }
    std::vector<unsigned char> buf(len);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data()), &len)) {
        return res;
    }

    constexpr DWORD kRecordHeader = 2 * sizeof(DWORD);                               // Relationship + Size
    constexpr DWORD kMinCoreSize  = kRecordHeader + sizeof(PROCESSOR_RELATIONSHIP);  // one group: 48 on x64

    struct CoreEntry {
        int              eff_class = 0;
        WORD             group     = 0;
        KAFFINITY        mask      = 0;
        std::vector<int> cpus;
    };

    std::vector<CoreEntry> cores;
    int                    best_class = -1;

    DWORD off = 0;
    while (off + kRecordHeader <= len) {
        auto * e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + off);
        if (e->Size < kMinCoreSize || e->Size > len - off) {
            break;
        }
        if (e->Relationship == RelationProcessorCore) {
            bool      usable = false;
            CoreEntry entry;
            entry.eff_class = static_cast<int>(e->Processor.EfficiencyClass);
            for (WORD g = 0; g < e->Processor.GroupCount; ++g) {
                const GROUP_AFFINITY & ga = e->Processor.GroupMask[g];
                if (ga.Group == 0) {
                    KAFFINITY active = ga.Mask & proc_mask;
                    if (active != 0) {
                        entry.group = ga.Group;
                        entry.mask  = active;
                        usable      = true;
                        for (int b = 0; b < 64; ++b) {
                            if ((active & (1ULL << b)) != 0) {
                                entry.cpus.push_back(b);
                            }
                        }
                        break;
                    }
                } else {
                    // Non-zero processor group (hosts with > 64 CPUs)
                    usable      = true;
                    entry.group = ga.Group;
                    entry.mask  = ga.Mask;
                    for (int b = 0; b < 64; ++b) {
                        if ((ga.Mask & (1ULL << b)) != 0) {
                            entry.cpus.push_back(b);
                        }
                    }
                    break;
                }
            }
            if (usable) {
                best_class = std::max(best_class, entry.eff_class);
                cores.push_back(std::move(entry));
            }
        }
        off += e->Size;
    }

    // Filter to performance cores (eff_class == best_class)
    std::vector<CoreEntry> p_cores;
    for (auto & c : cores) {
        if (c.eff_class == best_class) {
            p_cores.push_back(std::move(c));
        }
    }

    // Exclude 1st CPU core (Core 0 / core containing CPU 0) when >= 2 physical cores exist.
    // Core 0 is reserved for Windows system/OS duties, hardware interrupts, and DPCs.
    const bool exclude_first = (p_cores.size() >= 2);
    for (const auto & c : p_cores) {
        if (exclude_first && c.group == 0 && (c.mask & 1ULL) != 0) {
            continue;
        }
        if (!c.cpus.empty()) {
            res.primary.push_back(c.cpus[0]);
            for (size_t i = 1; i < c.cpus.size(); ++i) {
                res.secondary.push_back(c.cpus[i]);
            }
        }
    }
    return res;
}
#elif defined(__linux__)
static CoreLogicalInfo query_platform_cores() {
    CoreLogicalInfo res;
    cpu_set_t       set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return res;
    }
    std::set<int> fast;
    if (!read_cpu_list("/sys/devices/cpu_core/cpus", fast)) {
        int best_cap = -1;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &set)) {
                continue;
            }
            const int cap = read_sysfs_int("/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
            if (cap > best_cap) {
                best_cap = cap;
                fast.clear();
            }
            if (cap >= 0 && cap == best_cap) {
                fast.insert(cpu);
            }
        }
        if (best_cap < 0) {
            fast.clear();
        }
    }

    std::map<std::pair<int, int>, std::vector<int>> core_map;
    std::vector<int>                                untopologized;

    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &set)) {
            continue;
        }
        if (!fast.empty() && fast.count(cpu) == 0) {
            continue;
        }
        const int core = read_sysfs_int("/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        const int pkg  = read_sysfs_int("/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        if (core < 0 || pkg < 0) {
            untopologized.push_back(cpu);
            continue;
        }
        core_map[{ pkg, core }].push_back(cpu);
    }

    std::vector<std::vector<int>> distinct_cores;
    for (auto & kv : core_map) {
        std::sort(kv.second.begin(), kv.second.end());
        distinct_cores.push_back(std::move(kv.second));
    }
    for (int u : untopologized) {
        distinct_cores.push_back({ u });
    }

    const bool exclude_first = (distinct_cores.size() >= 2);
    for (const auto & c : distinct_cores) {
        if (exclude_first && !c.empty() && c[0] == 0) {
            continue;
        }
        if (!c.empty()) {
            res.primary.push_back(c[0]);
            for (size_t i = 1; i < c.size(); ++i) {
                res.secondary.push_back(c[i]);
            }
        }
    }
    return res;
}
#elif defined(__APPLE__)
static CoreLogicalInfo query_platform_cores() {
    CoreLogicalInfo res;
    int             n = 0;
    for (const char * key : { "hw.perflevel0.physicalcpu", "hw.physicalcpu" }) {
        int    v  = 0;
        size_t sz = sizeof(v);
        if (sysctlbyname(key, &v, &sz, nullptr, 0) == 0 && v > 0) {
            n = v;
            break;
        }
    }
    if (n <= 0) {
        n = usable_cpu_count();
    }
    if (n >= 2) {
        for (int i = 1; i < n; ++i) {
            res.primary.push_back(i);
        }
    } else if (n == 1) {
        res.primary.push_back(0);
    }
    return res;
}
#else
static CoreLogicalInfo query_platform_cores() {
    CoreLogicalInfo res;
    const int       n = usable_cpu_count();
    if (n >= 2) {
        for (int i = 1; i < n; ++i) {
            res.primary.push_back(i);
        }
    } else if (n == 1) {
        res.primary.push_back(0);
    }
    return res;
}
#endif

}  // namespace

std::vector<int> performance_cpu_ids(int n_threads) {
    CoreLogicalInfo info = query_platform_cores();
    if (info.primary.empty()) {
        const int n = usable_cpu_count();
        if (n >= 2) {
            for (int i = 1; i < n; ++i) {
                info.primary.push_back(i);
            }
        } else {
            info.primary.push_back(0);
        }
    }

    if (n_threads <= 0) {
        return info.primary;
    }

    std::vector<int> out;
    out.reserve(static_cast<size_t>(n_threads));

    // First, assign primary CPUs (1 per physical P-core)
    for (int cpu : info.primary) {
        if (static_cast<int>(out.size()) < n_threads) {
            out.push_back(cpu);
        }
    }
    // If more threads requested than physical P-cores, use SMT siblings on those same P-cores
    for (int cpu : info.secondary) {
        if (static_cast<int>(out.size()) < n_threads) {
            out.push_back(cpu);
        }
    }
    // If still more requested, wrap around available P-core logical CPUs (never touching Core 0 or E-cores)
    if (!out.empty()) {
        const size_t base_count = out.size();
        size_t       idx        = 0;
        while (static_cast<int>(out.size()) < n_threads) {
            out.push_back(out[idx % base_count]);
            ++idx;
        }
    }
    return out;
}

int performance_cpu_count() {
    return static_cast<int>(performance_cpu_ids(0).size());
}

int default_n_threads(int cap) {
    int n = performance_cpu_count();
    if (n < 1) {
        n = 1;
    }
    if (cap > 0 && n > cap) {
        n = cap;
    }
    return n;
}

struct ggml_threadpool_params make_threadpool_params(int n_threads) {
    if (n_threads <= 0) {
        n_threads = default_n_threads();
    }
    struct ggml_threadpool_params tpp;
    ggml_threadpool_params_init(&tpp, n_threads);
    tpp.strict_cpu = true;
    tpp.poll       = 50;
    tpp.prio       = GGML_SCHED_PRIO_NORMAL;

    const std::vector<int> cpus = performance_cpu_ids(n_threads);
    for (int cpu : cpus) {
        if (cpu >= 0 && cpu < GGML_MAX_N_THREADS) {
            tpp.cpumask[cpu] = true;
        }
    }
    return tpp;
}

void bind_thread_to_cpu(int cpu_id) {
    if (cpu_id < 0) {
        return;
    }
#if defined(_WIN32)
    if (cpu_id < 64) {
        const DWORD_PTR mask = static_cast<DWORD_PTR>(1ULL << cpu_id);
        SetThreadAffinityMask(GetCurrentThread(), mask);

#    if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;
        SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t));
#    endif
    }
#elif defined(__linux__)
    if (cpu_id < CPU_SETSIZE) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    }
#endif
}

thread_affinity_guard::thread_affinity_guard(int cpu_id) {
    if (cpu_id < 0) {
        return;
    }
#if defined(_WIN32)
    if (cpu_id < 64) {
        const DWORD_PTR mask = static_cast<DWORD_PTR>(1ULL << cpu_id);
        const DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), mask);
        prev_mask_           = static_cast<uint64_t>(prev);

#    if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;
        SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t));
#    endif
    }
#elif defined(__linux__)
    bind_thread_to_cpu(cpu_id);
#endif
}

thread_affinity_guard::~thread_affinity_guard() {
#if defined(_WIN32)
    if (prev_mask_ != 0) {
        SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(prev_mask_));
    }
#endif
}

// ---------------------------------------------------------------------------
// Per-node profiler (opt-in diagnostics).
//
// When TRANSCRIBE_PERF_DEBUG contains the substring "nodes", an eval
// callback is attached to every scheduler that passes through
// configure_sched_n_threads (the one choke point every family crosses right
// before compute) and a per-op / per-node time table is printed to stderr at
// process exit. This is how cross-family hotspots get attributed without
// per-family instrumentation (it found granite's im2col depthwise costing
// ~19% of its encode).
//
// Caveats: node-by-node evaluation suppresses fusion and adds two timestamps
// per node, so numbers are for RELATIVE attribution, not absolute cost — tiny
// elementwise nodes read inflated. The accumulator is an unsynchronized
// process global, so only the FIRST scheduler seen is instrumented (see
// node_prof_attach); in a concurrent host the report therefore covers one
// session, not the whole process. Inert (no callback installed) unless
// requested.
// ---------------------------------------------------------------------------
namespace {

struct NodeProf {
    int64_t                                            t0 = 0;
    std::map<std::string, std::pair<int64_t, int64_t>> by_name;
    int64_t                                            op_us[GGML_OP_COUNT]  = { 0 };
    int64_t                                            op_cnt[GGML_OP_COUNT] = { 0 };
};

NodeProf g_node_prof;

std::string node_prof_key(const ggml_tensor * t) {
    std::string collapsed;
    for (const char * c = t->name; *c != 0; ++c) {
        const char mapped = (*c >= '0' && *c <= '9') ? '#' : *c;
        if (mapped == '#' && !collapsed.empty() && collapsed.back() == '#') {
            continue;
        }
        collapsed.push_back(mapped);
    }
    char buf[96];
    std::snprintf(buf, sizeof(buf), " [%s %lldx%lldx%lldx%lld]", ggml_op_name(t->op), (long long) t->ne[0],
                  (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    return collapsed + buf;
}

bool node_prof_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * p = static_cast<NodeProf *>(ud);
    if (ask) {
        p->t0 = ggml_time_us();
        return true;
    }
    const int64_t dt = ggml_time_us() - p->t0;
    auto &        e  = p->by_name[node_prof_key(t)];
    e.first += dt;
    e.second += 1;
    const int op = static_cast<int>(t->op);
    if (op >= 0 && op < GGML_OP_COUNT) {
        p->op_us[op] += dt;
        p->op_cnt[op] += 1;
    }
    return true;
}

void node_prof_report() {
    const NodeProf & p     = g_node_prof;
    int64_t          total = 0;
    for (int o = 0; o < GGML_OP_COUNT; ++o) {
        total += p.op_us[o];
    }
    if (total == 0) {
        return;
    }
    std::fprintf(stderr, "[nodeprof] total=%.1f ms\n[nodeprof] --- by op ---\n", total / 1000.0);
    std::vector<int> ord(GGML_OP_COUNT);
    for (int o = 0; o < GGML_OP_COUNT; ++o) {
        ord[static_cast<size_t>(o)] = o;
    }
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return p.op_us[a] > p.op_us[b]; });
    for (int o : ord) {
        if (p.op_us[o] <= 0) {
            break;
        }
        std::fprintf(stderr, "[nodeprof]   %-18s %9.1f ms %5.1f%%  x%lld\n", ggml_op_name(static_cast<ggml_op>(o)),
                     p.op_us[o] / 1000.0, 100.0 * p.op_us[o] / total, (long long) p.op_cnt[o]);
    }
    std::fprintf(stderr, "[nodeprof] --- by node (top 45) ---\n");
    std::vector<std::pair<std::string, std::pair<int64_t, int64_t>>> v(p.by_name.begin(), p.by_name.end());
    std::sort(
        v.begin(), v.end(),
        [](const std::pair<std::string, std::pair<int64_t, int64_t>> & a,
           const std::pair<std::string, std::pair<int64_t, int64_t>> & b) { return a.second.first > b.second.first; });
    for (size_t i = 0; i < v.size() && i < 45; ++i) {
        std::fprintf(stderr, "[nodeprof]   %9.1f ms %5.1f%% x%-5lld %s\n", v[i].second.first / 1000.0,
                     100.0 * v[i].second.first / total, (long long) v[i].second.second, v[i].first.c_str());
    }
}

void node_prof_attach(ggml_backend_sched_t sched) {
    static const bool requested = [] {
        const char * v = std::getenv("TRANSCRIBE_PERF_DEBUG");
        return v != nullptr && std::strstr(v, "nodes") != nullptr;
    }();
    if (!requested || sched == nullptr) {
        return;
    }
    // Instrument exactly ONE scheduler. g_node_prof is a lock-free global, so
    // a second concurrently-computing scheduler (a multi-model host) would
    // race on it; claiming the slot keeps the accumulator single-writer and
    // makes the report attributable to one session instead of interleaved.
    // Re-attaching to the same sched (a family that reconfigures per run) is
    // a no-op, so the claim survives graph rebuilds.
    static std::atomic<ggml_backend_sched_t> owner{ nullptr };
    ggml_backend_sched_t                     expected = nullptr;
    if (!owner.compare_exchange_strong(expected, sched) && expected != sched) {
        static std::atomic<bool> warned{ false };
        if (!warned.exchange(true)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_WARN,
                    "[nodeprof] a second scheduler appeared; profiling only the "
                    "first (the accumulator is not thread-safe)");
        }
        return;
    }
    static const bool registered = [] {
        std::atexit(node_prof_report);
        return true;
    }();
    (void) registered;
    ggml_backend_sched_set_eval_callback(sched, node_prof_cb, &g_node_prof);
}

}  // namespace

int configure_sched_n_threads(ggml_backend_sched_t sched, int requested) {
    node_prof_attach(sched);
    const int n_threads = requested > 0 ? requested : default_n_threads();
    if (sched == nullptr) {
        return n_threads;
    }
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t     be  = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(be);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg == nullptr) {
            continue;
        }
        auto * fn = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (fn != nullptr) {
            fn(be, n_threads);
        }
        if (ggml_backend_is_cpu(be)) {
            safe_set_cpu_backend_threadpool(be, n_threads);
        }
    }
    return n_threads;
}

bool parallel_for_all(int n, int n_threads, const std::function<bool(int)> & work) {
    if (n <= 0) {
        return true;
    }
    if (n_threads <= 0) {
        n_threads = default_n_threads();
    }
    n_threads = std::max(1, std::min(n, n_threads));

    const std::vector<int> target_cpus = performance_cpu_ids(n_threads);

    std::atomic<int>  next{ 0 };
    std::atomic<bool> all_ok{ true };
    auto              worker = [&]() {
        int i;
        while ((i = next.fetch_add(1, std::memory_order_relaxed)) < n) {
            if (!work(i)) {
                all_ok.store(false, std::memory_order_relaxed);
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(n_threads - 1));
    for (int w = 0; w < n_threads - 1; ++w) {
        const int cpu = (w < static_cast<int>(target_cpus.size())) ? target_cpus[static_cast<size_t>(w)] : -1;
        pool.emplace_back([&worker, cpu]() {
            if (cpu >= 0) {
                bind_thread_to_cpu(cpu);
            }
            worker();
        });
    }

    const int             main_cpu = (!target_cpus.empty()) ? target_cpus.back() : -1;
    thread_affinity_guard guard(main_cpu);
    worker();  // the calling thread participates

    for (auto & th : pool) {
        th.join();
    }

    return all_ok.load(std::memory_order_relaxed);
}

void pack_pad_channel_major(std::vector<float> &                    dst,
                            const std::vector<std::vector<float>> & src,
                            const std::vector<int> &                lens,
                            int                                     n_ch,
                            int                                     T_max) {
    const int    n   = static_cast<int>(src.size());
    const size_t per = static_cast<size_t>(n_ch) * static_cast<size_t>(T_max);
    dst.assign(per * static_cast<size_t>(n), 0.0f);
    for (int b = 0; b < n; ++b) {
        const int     nb = lens[static_cast<size_t>(b)];
        const float * s  = src[static_cast<size_t>(b)].data();
        float *       d  = dst.data() + static_cast<size_t>(b) * per;
        for (int c = 0; c < n_ch; ++c) {
            std::copy(s + static_cast<size_t>(c) * nb, s + static_cast<size_t>(c) * nb + nb,
                      d + static_cast<size_t>(c) * T_max);
        }
    }
}

void fill_keypad_mask(ggml_tensor * mask, const std::vector<int> & real_lens, int T, int n) {
    if (mask == nullptr) {
        return;
    }
    // Keep fully masked rows finite on the manual softmax path; -inf would
    // produce NaNs. For rows with a valid key, exp(-1e30) still underflows to
    // zero, and F16 conversion preserves the flash path's -inf sentinel.
    const float        mask_neg = -1e30f;
    std::vector<float> buf(static_cast<size_t>(T) * n);
    for (int b = 0; b < n; ++b) {
        const int real = real_lens[static_cast<size_t>(b)];
        for (int k = 0; k < T; ++k) {
            buf[static_cast<size_t>(b) * T + k] = (k < real) ? 0.0f : mask_neg;
        }
    }
    ggml_backend_tensor_set(mask, buf.data(), 0, buf.size() * sizeof(float));
}

void fill_valid_frame_mask(ggml_tensor * mask, const std::vector<int> & real_lens, int T, int n) {
    if (mask == nullptr) {
        return;
    }
    std::vector<float> buf(static_cast<size_t>(T) * n);
    for (int b = 0; b < n; ++b) {
        const int real = real_lens[static_cast<size_t>(b)];
        for (int t = 0; t < T; ++t) {
            buf[static_cast<size_t>(b) * T + t] = (t < real) ? 1.0f : 0.0f;
        }
    }
    ggml_backend_tensor_set(mask, buf.data(), 0, buf.size() * sizeof(float));
}

transcribe_status decode_batch_slices(transcribe_session * session,
                                      int                  n,
                                      const float *        host_buf,
                                      std::size_t          utt_elems,
                                      int64_t              total_encode_us,
                                      int64_t              total_mel_us,
                                      const std::function<transcribe_status(int b, const float * slice)> & decode_fn) {
    const int64_t enc_per_utt = total_encode_us / std::max(1, n);
    const int64_t mel_per_utt = total_mel_us / std::max(1, n);
    for (int b = 0; b < n; ++b) {
        if (session->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        session->clear_result();
        const float *           slice = host_buf + static_cast<size_t>(b) * utt_elems;
        const transcribe_status st    = decode_fn(b, slice);
        auto                    rs    = session->capture_result(st);
        rs.t_mel_us                   = mel_per_utt;
        rs.t_encode_us                = enc_per_utt;
        session->batch_results.push_back(std::move(rs));
    }
    return TRANSCRIBE_OK;
}

transcribe_status run_batched_encdec_step_loop(transcribe_session *                session,
                                               ggml_backend_sched_t                sched,
                                               const EncDecRebuildFn &             rebuild,
                                               const std::vector<int32_t> &        prompt_ids,
                                               int                                 prompt_len,
                                               int                                 init_window,
                                               int                                 max_new,
                                               int                                 max_n_kv,
                                               int32_t                             eos_id,
                                               int                                 n_batch,
                                               const std::vector<char> &           valid,
                                               std::vector<std::vector<int32_t>> & generated,
                                               int *                               n_steps_out,
                                               std::vector<char> *                 truncated_out) {
    const int         n        = n_batch;
    const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());

    int          kv_window = init_window;
    EncDecStepIO io{};
    if (!rebuild(kv_window, io)) {
        return TRANSCRIBE_ERR_GGUF;
    }

    std::vector<ggml_fp16_t> smask(static_cast<size_t>(kv_window) * n, f16_ninf);
    std::vector<int32_t>     tok_buf(n, 0), pos_buf(n, 0), argmax_buf(n, 0);
    std::vector<int64_t>     kvidx_buf(n, 0);
    std::vector<char>        finished(n, 0);
    std::vector<int32_t>     next_tok(n, 0);
    for (int b = 0; b < n; ++b) {
        if (!valid[b]) {
            finished[b] = 1;
        }
    }

    int  n_steps  = 0;
    auto run_step = [&](int posv) -> transcribe_status {
        for (int b = 0; b < n; ++b) {
            pos_buf[b]                                       = posv;
            kvidx_buf[b]                                     = posv;
            smask[static_cast<size_t>(b) * kv_window + posv] = f16_zero;
        }
        ggml_backend_tensor_set(io.token_ids, tok_buf.data(), 0, n * sizeof(int32_t));
        ggml_backend_tensor_set(io.pos_ids, pos_buf.data(), 0, n * sizeof(int32_t));
        ggml_backend_tensor_set(io.kv_idx, kvidx_buf.data(), 0, n * sizeof(int64_t));
        ggml_backend_tensor_set(io.self_mask, smask.data(), 0, smask.size() * sizeof(ggml_fp16_t));
        if (ggml_backend_sched_graph_compute(sched, io.graph) != GGML_STATUS_SUCCESS) {
            return TRANSCRIBE_ERR_GGUF;
        }
        ggml_backend_tensor_get(io.argmax, argmax_buf.data(), 0, n * sizeof(int32_t));
        ++n_steps;
        return TRANSCRIBE_OK;
    };

    // Grow the read window (rebuild graph + widen mask) so position `posv` fits.
    auto ensure_window = [&](int posv) -> bool {
        if (posv + 1 <= kv_window) {
            return true;
        }
        int win = kv_window;
        while (win < posv + 1 && win < max_n_kv) {
            win *= 2;
        }
        if (win > max_n_kv) {
            win = max_n_kv;
        }
        if (win == kv_window) {
            return true;
        }
        std::vector<ggml_fp16_t> wider(static_cast<size_t>(win) * n, f16_ninf);
        for (int b = 0; b < n; ++b) {
            std::fill(wider.data() + static_cast<size_t>(b) * win, wider.data() + static_cast<size_t>(b) * win + posv,
                      f16_zero);
        }
        smask.swap(wider);
        kv_window = win;
        return rebuild(kv_window, io);
    };

    // Prompt feed: prompt_len sequential steps (uniform tokens across rows).
    int pos = 0;
    for (; pos < prompt_len; ++pos) {
        if (session->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        if (!ensure_window(pos)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "batched decode: step graph allocation failed — out of memory. "
                                "Lower transcribe_session_params.n_ctx or the batch size.");
            return TRANSCRIBE_ERR_OOM;
        }
        for (int b = 0; b < n; ++b) {
            tok_buf[b] = prompt_ids[static_cast<size_t>(pos)];
        }
        if (run_step(pos) != TRANSCRIBE_OK) {
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    // argmax from the last prompt position = first generated token.
    for (int b = 0; b < n; ++b) {
        if (finished[b]) {
            continue;
        }
        next_tok[b] = argmax_buf[b];
        if (next_tok[b] == eos_id) {
            finished[b] = 1;
        } else {
            generated[b].push_back(next_tok[b]);
        }
    }

    // Generation.
    for (int produced = 1; produced < max_new; ++produced, ++pos) {
        if (session->poll_abort()) {
            return TRANSCRIBE_ERR_ABORTED;
        }
        bool all_done = true;
        for (int b = 0; b < n; ++b) {
            if (!finished[b]) {
                all_done = false;
                break;
            }
        }
        if (all_done || pos + 1 > max_n_kv) {
            break;
        }
        if (!ensure_window(pos)) {
            transcribe::log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                                "batched decode: step graph allocation failed — out of memory. "
                                "Lower transcribe_session_params.n_ctx or the batch size.");
            return TRANSCRIBE_ERR_OOM;
        }
        for (int b = 0; b < n; ++b) {
            tok_buf[b] = finished[b] ? eos_id : next_tok[b];
        }
        if (run_step(pos) != TRANSCRIBE_OK) {
            return TRANSCRIBE_ERR_GGUF;
        }
        for (int b = 0; b < n; ++b) {
            if (finished[b]) {
                continue;
            }
            next_tok[b] = argmax_buf[b];
            if (next_tok[b] == eos_id) {
                finished[b] = 1;
            } else {
                generated[b].push_back(next_tok[b]);
            }
        }
    }

    if (n_steps_out != nullptr) {
        *n_steps_out = n_steps;
    }

    // A valid row that never reached eos was cut off at the generation budget
    // or the context window — report it as truncated so the family can return
    // per-utterance TRANSCRIBE_ERR_OUTPUT_TRUNCATED. See docs/input-limits.md.
    if (truncated_out != nullptr) {
        truncated_out->assign(n, 0);
        for (int b = 0; b < n; ++b) {
            (*truncated_out)[b] = (valid[b] && !finished[b]) ? 1 : 0;
        }
    }
    return TRANSCRIBE_OK;
}

}  // namespace transcribe
