// transcribe-batch-util.cpp - see transcribe-batch-util.h.

#include "transcribe-batch-util.h"

#include "ggml-backend.h"
#include "ggml.h"
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

// Number of *performance* physical cores the process may run on, or 0 when
// the platform query is unavailable (the caller falls back to
// usable_cpu_count()).
//
// "Performance" means one entry per physical core (SMT siblings collapsed)
// and, on a hybrid CPU, only the fastest core class. On a homogeneous CPU
// every core is in the fastest class, so this degenerates to the physical-
// core count.
//
// Why not the logical-CPU count: ggml's CPU backend splits each op's rows
// evenly across threads and joins on a spin barrier, so every thread waits
// for the slowest. An SMT sibling contributes far less than a full core, and
// an Intel E-core / ARM little core is several times slower than a P-core.
// Either asymmetry makes the barrier wait on the stragglers, so one thread
// per performance core beats one per logical CPU. Measured on an i9-13900H
// (6 P-cores + 8 E-cores, 20 logical) with the Parakeet v3 encoder: 6 threads
// pinned to P-cores 1.9 s, 6 threads pinned to E-cores 5.4 s, 12 threads on
// the 6 P-cores (SMT) 2.9 s.
//
// COUNT only — deliberately no affinity pinning. Measured on the same
// machine (parakeet-v3 encode, arms interleaved per round): a process
// hard-pinned to the 6 P-cores was SLOWER than unpinned in 5/5 rounds
// (~5-15%), while E-core-pinned was ~2.3x slower. Two conclusions: the
// OS's hybrid-aware scheduler already places these 6 threads on P-cores
// (else unpinned would sit near the E-pinned time), and pinning removes
// its freedom to migrate off a P-core occupied by another process — under
// background load one stalled thread holds up ggml's spin barrier. Do not
// add cpumask/strict_cpu to the ggml threadpools without beating the
// unpinned numbers on an interleaved benchmark.
int performance_cpu_count() {
#if defined(_WIN32)
    DWORD_PTR proc_mask = 0, sys_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask) || proc_mask == 0) {
        return 0;
    }
    DWORD len = 0;
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0) {
        return 0;
    }
    std::vector<unsigned char> buf(len);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data()), &len)) {
        return 0;
    }
    // Pass 0 finds the highest EfficiencyClass among the cores this process
    // may use; pass 1 counts the cores in that class. GetProcessAffinityMask
    // only describes group 0, so cores in other processor groups (hosts with
    // >64 logical CPUs) are counted unconditionally — matching
    // usable_cpu_count(), which falls back to hardware_concurrency() there.
    //
    // Records are variable-length and `Size` is the stride. Do NOT bound the
    // walk with sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX): that is the
    // size of the largest union member (GROUP_RELATIONSHIP, 80 bytes on x64)
    // while a RelationProcessorCore record is 48, so such a bound silently
    // drops the trailing record — one core, invisibly, on every homogeneous
    // CPU. Bound by `len` and validate `Size` instead.
    constexpr DWORD kRecordHeader = 2 * sizeof(DWORD);                               // Relationship + Size
    constexpr DWORD kMinCoreSize  = kRecordHeader + sizeof(PROCESSOR_RELATIONSHIP);  // one group: 48 on x64
    int             best_class    = -1;
    int             n_perf        = 0;
    for (int pass = 0; pass < 2; ++pass) {
        DWORD off = 0;
        while (off + kRecordHeader <= len) {
            auto * e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + off);
            if (e->Size < kMinCoreSize || e->Size > len - off) {
                break;
            }
            if (e->Relationship == RelationProcessorCore) {
                bool usable = false;
                for (WORD g = 0; g < e->Processor.GroupCount; ++g) {
                    const GROUP_AFFINITY & ga = e->Processor.GroupMask[g];
                    if (ga.Group != 0 || (static_cast<DWORD_PTR>(ga.Mask) & proc_mask) != 0) {
                        usable = true;
                        break;
                    }
                }
                if (usable) {
                    const int cls = static_cast<int>(e->Processor.EfficiencyClass);
                    if (pass == 0) {
                        best_class = std::max(best_class, cls);
                    } else if (cls == best_class) {
                        ++n_perf;
                    }
                }
            }
            off += e->Size;
        }
    }
    return n_perf;
#elif defined(__APPLE__)
    // perflevel0 is the fastest cluster on Apple silicon; absent on Intel Macs.
    for (const char * key : { "hw.perflevel0.physicalcpu", "hw.physicalcpu" }) {
        int    v  = 0;
        size_t sz = sizeof(v);
        if (sysctlbyname(key, &v, &sz, nullptr, 0) == 0 && v > 0) {
            return v;
        }
    }
    return 0;
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return 0;
    }
    // Restrict to the fastest core class when the kernel exposes one: Intel
    // hybrid publishes the P-core CPU list under the cpu_core PMU, ARM
    // big.LITTLE publishes a per-CPU capacity. An empty `fast` means "no
    // class information — every usable CPU counts".
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
    // Collapse SMT siblings: count distinct (package, core) pairs.
    std::set<std::pair<int, int>> cores;
    int                           n_untopologized = 0;
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
            ++n_untopologized;  // sysfs unavailable (some containers): count the CPU itself
            continue;
        }
        cores.insert({ pkg, core });
    }
    return static_cast<int>(cores.size()) + n_untopologized;
#else
    return 0;
#endif
}

}  // namespace

int default_n_threads(int cap) {
    // Performance cores when the platform can tell us, clamped by the affinity
    // mask so taskset/cpuset still wins; otherwise every usable CPU.
    int n = performance_cpu_count();
    if (n > 0) {
        n = std::min(n, usable_cpu_count());
    } else {
        n = usable_cpu_count();
    }
    if (n < 1) {
        n = 1;
    }
    if (cap > 0 && n > cap) {
        n = cap;
    }
    return n;
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
    }
    return n_threads;
}

bool parallel_for_all(int n, int n_threads, const std::function<bool(int)> & work) {
    if (n <= 0) {
        return true;
    }
    if (n_threads <= 0) {
        // Every CPU the process may run on, clamped to the batch size below.
        // Deliberately NOT default_n_threads(): this pool hands out items from
        // an atomic counter with no barrier, so an SMT sibling or an E-core
        // still adds throughput instead of stalling the join. The performance-
        // core restriction exists for ggml's barrier-synchronized op split,
        // which is the opposite situation.
        n_threads = usable_cpu_count();
    }
    n_threads = std::max(1, std::min(n, n_threads));

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
        pool.emplace_back(worker);
    }
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
