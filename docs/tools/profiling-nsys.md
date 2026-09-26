# Profiling transcribe.cpp with Nsight Systems (and friends)

A field guide for finding *why* a model is slow on a given backend and turning that
into a fix in the graph, a kernel, or a downstream ggml patch. It is written from a
real investigation (parakeet on an RTX 4070 Laptop, 2026-09-26 — see
`docs/porting/parakeet-optimization-2026-09-26.md`) where this exact method took
CUDA from 110× to 257× realtime in an afternoon, but the method is model-agnostic.

The one-line summary: **measure the phases, then ask the GPU what it actually ran,
then ask the scheduler where it ran it.** Most "slow backend" problems in ggml-based
code are not slow kernels; they are the GPU waiting.

---

## 0. The toolbox

| Tool | Question it answers | Cost |
|---|---|---|
| `transcribe-bench` | Which phase is slow? (mel / encode / decode, warm, averaged) | seconds |
| `GGML_SCHED_DEBUG=1/2` | Does the graph leave the GPU? Which op, which tensor? | seconds |
| **Nsight Systems** (`nsys`) | What did the GPU run, for how long, and how long was it idle? | a minute |
| Nsight Compute (`ncu`) | Why is *this one kernel* slow (occupancy, memory, tensor cores)? | minutes per kernel |
| `test-backend-ops` / unit tests | Is a new/changed kernel numerically correct on every backend? | minutes |
| `scripts/validate.py compare` | Is the whole model still numerically right vs the reference? | a minute |
| `scripts/wer/run.py` + `score.py` | Did accuracy move? | 5–20 min |

Always in this order. Never optimize before you know which phase and which op.

---

## 1. Measure the phases (transcribe-bench)

```bash
build-gpu/bin/Release/transcribe-bench.exe \
    --model models/parakeet-ultra-0.6b/parakeet-ultra-0.6b-Q8_0.gguf \
    --sample samples/german.wav --iters 5 --warmup 2 --backend cuda --quiet \
  > bench.json
```

`bench.json` (schema `transcribe-bench-v2`) has `per_iter[]` and `summary` with
`mel_ms`, `encode_ms`, `decode_ms`, `total_ms`, `wall_ms`, plus `load_ms` and
`rtf_wall_mean`. Read the **mean of the warm iterations**; the first run of a process
pays one-time costs (CUDA context/module load, cuBLAS handle, Vulkan pipeline
compilation — the latter can be seconds on the very first run and is cached on disk
afterwards).

Rules that save days:

* **Compare phases across backends, not totals.** On parakeet the decoder was 57 ms on
  CPU, CUDA *and* Vulkan — so the decoder was not the CUDA problem; the encoder was
  (194 ms CUDA vs 84 ms Vulkan).
* **Vary the weight format.** If CUDA takes the same time with F16, Q8_0, Q4_K and a
  2-bit ternary layout, the big matmuls are not the bottleneck — something fixed is.
  That single observation redirected the whole parakeet investigation.
* The CLI's `realtime:` line is a *single cold-ish run*; do not benchmark with it.
* Benchmark method for decisions: 4 runs, drop the first, average 3, interleave arms
  (A B A B …), and treat differences below ~3 % as noise.

---

## 2. Does the graph leave the GPU? (`GGML_SCHED_DEBUG`)

ggml's backend scheduler (`ggml_backend_sched`) assigns every node to a backend. If a
backend's `supports_op` rejects a node, the node runs on the **CPU** and the scheduler
inserts copies GPU→host→GPU around it, with a stream synchronization each time. In a
24-block encoder, one unsupported op per block means ~50–100 round trips per call.

```bash
# number of graph splits (1 == whole graph on one backend)
GGML_SCHED_DEBUG=1 transcribe-cli --backend cuda -m model.gguf samples/jfk.wav 2>&1 | grep -c "SPLIT #"

# which nodes land on the CPU, with their inputs
GGML_SCHED_DEBUG=2 transcribe-cli --backend cuda -m model.gguf samples/jfk.wav 2>&1 \
  | awk '/SPLIT #1: CPU/{f=1} /SPLIT #4:/{f=0} f'
```

Output reads like:

```
## SPLIT #1: CPU # 4 inputs
node #131 (FLASH_ATTN):   node_131 ( 552K) [  CPU ]
    CPU# (permuted)#0 ...  CPU# node_130#0 (297K)  <- the mask
## SPLIT #3: CPU # 1 inputs
node #144 (   SIGMOID):   node_144 ( 552K) [  CPU ]
    CPU# (view)#0 (1M)                              <- a strided view
```

Interpretation checklist for every CPU split on a GPU run:

1. **Which op?** Grep that backend's `supports_op` (`ggml-cuda/ggml-cuda.cu`:
   `ggml_backend_cuda_device_supports_op`; Vulkan: `ggml_backend_vk_device_supports_op`;
   Metal: `ggml_metal_device_supports_op`). For big ops there is usually a helper
   (e.g. CUDA flash attention: `ggml_cuda_flash_attn_ext_supported` →
   `ggml_cuda_get_best_fattn_kernel` in `ggml-cuda/fattn.cu`).
2. **Why rejected?** Typical reasons: tensor not contiguous (unary ops on CUDA), an
   unsupported type combination, a mask/broadcast shape the kernel does not handle
   (CUDA flash attention: `mask->ne[2] != 1`, i.e. no per-head masks), a head size
   without a compiled kernel, rows not padded.
3. **Fix at the cheapest level:**
   * *Graph level (transcribe.cpp)*: make the input contiguous, apply the op to the
     whole contiguous tensor and view the result, reorder a permute, or pick an
     equivalent op sequence the backend supports (manual `mul_mat + soft_max` instead
     of a fused kernel). Prefer **probing** support (`ggml_backend_dev_supports_op` on
     a tiny no-alloc context) over backend-name checks, so the choice stays right for
     backends you did not test. Example: `conformer::flash_supports_rel_pos_mask`.
   * *Kernel level (ggml patch)*: extend the kernel (e.g. per-head mask support) — only
     when the graph workaround costs real time. Ship as `patches/ggml/NNNN-*.patch`.

Target: **1 split** for single-backend runs. Re-count after every fix.

---

## 3. What did the GPU actually run? (Nsight Systems)

### 3.1 Install / locate

Nsight Systems ships with the CUDA toolkit and separately. On this machine:

```
C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.2\target-windows-x64\nsys.exe
```

(`ncu.exe` lives under `Nsight Compute 2026.x`.) Linux: `/usr/local/cuda/bin/nsys` or the
standalone package.

### 3.2 Capture

```bash
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.3.2/target-windows-x64/nsys.exe"
"$NSYS" profile \
    -t cuda,nvtx \
    --sample=none --cpuctxsw=none \
    -o prof --force-overwrite true \
    build-gpu/bin/Release/transcribe-bench.exe \
        --model models/parakeet-ultra-0.6b/parakeet-ultra-0.6b-Q8_0.gguf \
        --sample samples/german.wav --iters 3 --warmup 1 --backend cuda --quiet
```

* `-t cuda,nvtx`: trace CUDA runtime/driver API calls and kernels (+ NVTX ranges if
  the code emits any). Add `cublas` to see cuBLAS internals, `osrt` for OS runtime.
* `--sample=none --cpuctxsw=none`: skip CPU sampling and context-switch tracing. **On
  Windows these need administrator rights**; without admin, nsys prints
  "CPU sampling requires administrative privileges", and in our first attempt the
  capture then stalled with `Timeout has expired (75 sec)` and produced no report.
  Either run the shell elevated or disable those two collectors — for GPU questions
  you do not need them.
* Profile **`transcribe-bench`**, not the CLI: several warm iterations make one-time
  costs visible separately from steady state, and the per-phase timings in the JSON
  let you map GPU time to phases.
* Keep captures short (a 30 s clip, 3–4 iterations). Big reports are slow to process.

Output: `prof.nsys-rep` (open in the Nsight Systems GUI for a timeline) and the
intermediate `.qdstrm`.

### 3.3 Read it from the command line

```bash
"$NSYS" stats -r cuda_gpu_kern_sum -f csv -o kern prof.nsys-rep   # kernels
"$NSYS" stats -r cuda_api_sum      -f csv -o api  prof.nsys-rep   # host API calls
"$NSYS" stats -r cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum prof.nsys-rep  # copies
```

Useful reports (`nsys stats --help-reports` lists all): `cuda_gpu_kern_sum`,
`cuda_api_sum`, `cuda_gpu_mem_time_sum`, `cuda_gpu_mem_size_sum`, `cuda_kern_exec_sum`
(launch→start latency), `nvtx_sum`.

A tiny formatter (any Python):

```python
import csv, glob
rows = list(csv.DictReader(open(glob.glob("kern*.csv")[0])))
for r in rows[:25]:
    print(f"{float(r['Time (%)']):5.1f}% {float(r['Total Time (ns)'])/1e6:8.1f} ms "
          f"n={r['Instances']:>6} avg={float(r['Avg (ns)'])/1e3:7.1f} us  {r['Name'][:100]}")
```

### 3.4 The three numbers that matter

1. **GPU busy time per iteration** = Σ kernel time ÷ number of iterations captured
   (warmup + measured). Compare with the phase wall time from the bench JSON.
   *Parakeet/CUDA before the fix: ~57 ms of kernels per run vs ~250 ms per run →
   the GPU was idle ~75 % of the time.* When busy ≪ wall, stop looking at kernels.
2. **`cudaStreamSynchronize` count and total** (from `cuda_api_sum`). Hundreds per
   iteration means host round trips — almost always CPU-assigned graph nodes
   (go to §2) or host-side loops that read results every step. *Before: ~740 syncs
   and ~420 `cudaMemcpyAsync` per run.*
3. **Top kernels by total time**, only once the GPU is actually busy. Map kernel
   names to ggml ops:

   | Kernel name fragment | ggml op / path |
   |---|---|
   | `mul_mat_q<type,…>` + `mul_mat_q_stream_k_fixup` | quantized MUL_MAT via MMQ (int8 tensor cores) |
   | `mul_mat_vec_q` | quantized MUL_MAT, small batch (MMVQ) |
   | `cutlass…gemm`, `ampere_h1688gemm…`, `…sgemm` | cuBLAS: F16/F32 MUL_MAT or dequant+GEMM fallback |
   | `dequantize_block_*`, `convert_unary_*` | type conversion feeding cuBLAS |
   | `quantize_mmq_q8_1`, `quantize_q8_1` | activation quantization for MMQ/MMVQ |
   | `flash_attn_ext_*` / `flash_attn_tile*` | FLASH_ATTN_EXT |
   | `soft_max_f32` | SOFT_MAX |
   | `k_bin_bcast<op_add/op_mul…>` | ADD / MUL (broadcast) |
   | `norm_f32`, `rms_norm_f32` | NORM / RMS_NORM |
   | `conv2d_kernel`, `conv2d_dw_kernel`, `im2col_kernel` | CONV_2D / CONV_2D_DW / IM2COL |
   | `cpy_*`, `concat_*` | CPY / CONT / CONCAT (layout shuffles — reduce them) |
   | `unary_op_kernel<op_silu/op_relu…>` | unary activations |

   Look for: one-off huge kernels (e.g. `conv2d_kernel` 4.6 ms × 4 calls in the
   subsampling stem), many tiny launches (launch-bound; fuse or batch), conversion
   kernels feeding GEMMs (a native kernel or a better type would remove them), layout
   shuffles (`cpy_*`, `concat_*`) that a different permute order would avoid.

### 3.5 Timeline (GUI)

Open `prof.nsys-rep` in `nsys-ui.exe`. In the CUDA HW row, gaps between kernels are
idle GPU; zoom into one encoder call: gaps that line up with
`cudaStreamSynchronize` / `cudaMemcpyAsync` in the API row are scheduler round trips.
Evenly spaced tiny kernels with gaps are launch-bound code.

---

## 4. Why is this one kernel slow? (Nsight Compute)

Only after §2–3 point at a specific kernel:

```bash
NCU="/c/Program Files/NVIDIA Corporation/Nsight Compute 2026.3.0/ncu.exe"
"$NCU" --kernel-name regex:mul_mat_q --launch-skip 20 --launch-count 3 \
       --set full -o mmq build-gpu/bin/Release/transcribe-bench.exe \
       --model model.gguf --sample samples/jfk.wav --iters 1 --warmup 0 --backend cuda --quiet
```

Read: achieved occupancy, memory throughput vs peak (memory-bound?), tensor-core
utilization, warp stall reasons, shared-memory bank conflicts. Needs admin on
Windows for performance counters (or the NVIDIA control panel "allow access to GPU
performance counters to all users").

---

## 5. CPU backend: the equivalent questions

nsys does not help much on the CPU backend. Use:

* `transcribe-bench --backend cpu` phases, and **vary weight types** again: if Q8_0 is
  faster than a smaller format, the kernel is compute/decode-bound, not bandwidth-bound.
* Check what the CPU kernel does per call: ggml's CPU `mul_mat` calls `vec_dot` once
  per (weight row, activation column). Any per-element decode inside `vec_dot` is paid
  once **per activation column** — for an encoder with ~370 frames that is ~370×.
  The cure is ggml's **CPU_REPACK** buffer (interleaved tiles + GEMM kernels that
  decode once per tile): x86 AVX2 has repacked GEMMs for Q4_0, Q4_K, IQ4_NL, MXFP4;
  ARM dotprod/i8mm additionally Q5_K, Q6_K, Q8_0. transcribe.cpp opts weights into it
  via `load_common::alloc_cpu_repack_weights` (log line:
  `N weight tensor(s) (X MB) use CPU_REPACK GEMM kernels`).
* Windows `perf`-style sampling: Visual Studio profiler or Intel VTune / AMD uProf on
  `transcribe-bench`; Linux `perf record -g` + `perf report`.

---

## 6. From finding to fix — the loop

1. Hypothesis from §1–5 ("encoder on CUDA is idle because FLASH_ATTN runs on the CPU").
2. **Cheapest experiment first.** Existing env switches often test the hypothesis
   without code: `TRANSCRIBE_NO_FLASH=1` (manual attention), `TRANSCRIBE_CONV_{NO_,}DIRECT_{PW,DW,CONV0}`,
   `TRANSCRIBE_TERNARY_RUNTIME=q4_0|q2_0|native`, `TRANSCRIBE_NO_CPU_REPACK=1`,
   `GGML_CUDA_DISABLE_GRAPHS=1`, `GGML_CUDA_FORCE_CUBLAS=1` / `GGML_CUDA_FORCE_MMQ=1`.
   Parakeet: `TRANSCRIBE_NO_FLASH=1` alone moved CUDA encode 181 → 88 ms, proving the
   hypothesis in one run.
3. Implement the fix at the right level (graph > loader policy > kernel > new type),
   make it automatic and **backend-agnostic** (support probes, not name checks),
   keep an env override for A/B.
4. Re-run §2 (splits), §1 (phases), and the correctness gates:
   * `transcribe_ternary_tq1_g128_unit` / `test-backend-ops` for kernels,
   * `scripts/validate.py cpp|compare` (18/18 tensors, exact transcript),
   * `scripts/batch_parity.py` on every backend you touched (CPU must be byte-exact;
     GPU batched-vs-serial may flip borderline words because kernel choice depends on
     the batch shape — confirm with the CPU run that the *logic* is right),
   * a WER re-run on the changed paths.
5. If ggml changed: regenerate the downstream patch
   (`git diff --relative=ggml -- ggml > patches/ggml/NNNN-name.patch`, with
   `git add -N` for new files) and prove it applies to a pristine export of the
   vendored tree (`git archive HEAD:ggml | tar -x -C tmp && git -C tmp apply --check`).
6. Record the numbers (before/after table, env switch, measurement method) in the
   model's port record.

---

## 7. Pitfalls seen in practice

* nsys hangs without admin → `--sample=none --cpuctxsw=none` or elevated shell.
* Output paths with Git-Bash-style `/c/...` inside nsys arguments: pass Windows paths.
* First-run Vulkan pipeline compilation (seconds) and CUDA module loading pollute
  single-run timings — always warm up.
* A "slow kernel" in the summary can be the *victim*: e.g. `cudaMemcpyAsync` at 106 µs
  average was waiting on a queued kernel, not copying slowly.
* The scheduler silently falls back to CPU — nothing is logged at INFO level. Make
  the split count part of every GPU bring-up.
* Unary ops on strided views are the most common accidental CPU fallback on CUDA.
* Batch-parity "failures" on GPU can be numerics, not bugs — decide with a CPU run.
