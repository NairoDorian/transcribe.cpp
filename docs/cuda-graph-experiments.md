# CUDA graphs and audio.cpp optimizer experiments

Measured 2026-09-20 on the RTX 4070 Laptop / i9-13900H with the 11-second JFK fixture. Every cell is the arithmetic mean of runs 2 and 3 of exactly three resident runs; run 1 is excluded. Each profile measured both CPU and CUDA. No weights were downloaded.

All profiles use the same Release native/architecture binaries and the newly compiled GGML v0.24 CUDA module. The CPU module is the same feature-dispatched module used in the prior baseline. `GGML_CUDA_DISABLE_GRAPHS=1` disables capture in the graphs-capable binary; removing it enables capture. `TRANSCRIBE_GRAPH_OPTIMIZER=1` enables the compatible audio.cpp pass. `GGML_CUDA_GRAPH_OPT=1` enables GGML's separate CUDA scheduling optimizer. Those are distinct experiments, not interchangeable names for the same feature.

## Decision

Keep CUDA graphs, the audio.cpp pass and GGML's CUDA scheduling optimizer **off by default** for this Nemotron-focused build. CUDA graphs improve Granite substantially, but are not consistently faster for Nemotron/Parakeet. The audio.cpp subset removes three redundant copies from some graphs without a consistent wall-time benefit. Both new optimizers remain available for explicit experiments; no speculative kernel port was made. CUDA logs show repeated capture warm-up resets as streaming graph properties change.

The audio.cpp adaptation retains output/storage protections and Apache-2.0 attribution. Its numerical unit test computes a real graph with the pass off/on, checks exact results, and verifies view/output preservation. Every successful model case also checks measured-run and baseline transcript parity.

## CPU warm wall time (ms)

| Model / mode | Graphs off, pass off | Graphs on, pass off | Graphs on, pass on | Graphs off, pass on | Graphs on, CUDA scheduler opt |
|---|---:|---:|---:|---:|---:|
| Granite Q4 batch | 3439.5 | 4225.5 | 4152.1 | 4170.5 | 3542.2 |
| Qwen 1.7B Q5 batch | 4891.4 | 5035.8 | 5127.2 | 5004.4 | 4561.9 |
| Nemotron Q6 batch | 991.6 | 899.8 | 1106.7 | 996.1 | 917.2 |
| Nemotron Q6 stream | 1540.5 | 1719.3 | 1788.8 | 1750.4 | 1481.0 |
| Nemotron Q8 batch | 1065.4 | 987.2 | 1017.1 | 979.2 | 995.6 |
| Nemotron Q8 stream | 1647.7 | 1678.7 | 1633.0 | 1634.8 | 1682.4 |
| Parakeet Q4 batch | 900.3 | 821.5 | 856.8 | 893.4 | 895.6 |

## CUDA warm wall time (ms)

| Model / mode | Graphs off, pass off | Graphs on, pass off | Graphs on, pass on | Graphs off, pass on | Graphs on, CUDA scheduler opt |
|---|---:|---:|---:|---:|---:|
| Granite Q4 batch | 401.0 | 316.1 | 319.2 | 374.1 | 315.9 |
| Qwen 1.7B Q5 batch | 292.8 | 268.4 | 265.6 | 282.6 | 266.0 |
| Nemotron Q6 batch | 172.4 | 186.5 | 186.1 | 227.5 | 172.6 |
| Nemotron Q6 stream | 596.1 | 716.4 | 662.9 | 638.8 | 681.8 |
| Nemotron Q8 batch | 194.8 | 166.3 | 172.2 | 192.3 | 195.1 |
| Nemotron Q8 stream | 667.1 | 639.4 | 613.6 | 624.6 | 654.9 |
| Parakeet Q4 batch | 126.5 | 144.3 | 144.1 | 144.6 | 149.0 |

## Qwen CPU thread-budget sweep

Graphs and the audio.cpp pass were off. Explicit counts preserve ordinary OS scheduling; there is no CPU binding, P/E-core discovery or elevated priority. The automatic limit is now five, the fastest tested budget in this sweep; explicit caller counts still win.

| Workers | CPU (ms) | CUDA (ms) |
|---:|---:|---:|
| 2 | 7733.3 | 280.6 |
| 3 | 5878.9 | 284.2 |
| 4 | 4891.4 | 292.8 |
| 5 | 4144.5 | 274.1 |
| 6 | 4599.4 | 282.9 |
| 8 | 4198.4 | 283.0 |

## Regression gates and limits

Timing failures are retained, not suppressed or replaced with the fastest run. Even CPU cases whose code path is unchanged by the CUDA toggle varied across profiles. A laptop's power/thermal state and ordinary scheduling remain sources of noise after other busy apps close. The 15% gate therefore flags measurements for investigation; it does not prove causation. These results do not establish a universal optimum on other machines or replace full WER/long-stream testing.

- [graphs-off-opt-off](../build/diagnostics/graphs-off-opt-off/summary.json): 14 cases, 0 gate failures.
- [graphs-on-opt-off](../build/diagnostics/graphs-on-opt-off/summary.json): 14 cases, 2 gate failures.
  - `granite-speech-4.1-2b-Q4_K_M.cpu.batch`: timing regression.
  - `nemotron-3.5-asr-streaming-0.6b-Q6_K.cuda.stream`: timing regression.
- [graphs-on-opt-on](../build/diagnostics/graphs-on-opt-on/summary.json): 14 cases, 1 gate failures.
  - `nemotron-3.5-asr-streaming-0.6b-Q6_K.cpu.batch`: timing regression.
- [graphs-off-opt-on](../build/diagnostics/graphs-off-opt-on/summary.json): 14 cases, 2 gate failures.
  - `granite-speech-4.1-2b-Q4_K_M.cpu.batch`: timing regression.
  - `nemotron-3.5-asr-streaming-0.6b-Q6_K.cuda.batch`: timing regression.
- [graphs-on-scheduler-opt](../build/diagnostics/graphs-on-scheduler-opt/summary.json): 14 cases, 1 gate failures.
  - `nemotron-3.5-asr-streaming-0.6b-Q8_0.cuda.batch`: timing regression.

The selected final build and complete upstream comparison are documented in [the investigation](nemotron-performance-investigation.md). Use `scripts/bench/build-native.ps1 -CudaGraphs` to reproduce a graphs-capable backend; the default build keeps graphs off. The Windows helper targets AVX2 CPU and the selected CUDA architecture; enable `-CpuAvxVnni` only on a CPU supporting those instructions.
