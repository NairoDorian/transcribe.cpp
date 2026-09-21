# Upstream-only measurement branch

Branch `transcribe_benchmarks` starts at upstream main
`be7a8b35e9ba2df20298bd26e32d53407c3bcbcd`. Its parent history is unchanged.
Only benchmark tooling/documentation is added; there are no inference fixes.
The origin is the existing NairoDorian fork, not a new unrelated repository.

On Windows, in an x64 Developer PowerShell with CMake, Ninja and CUDA installed:

```powershell
./scripts/bench/build-native.ps1 -CudaArchitecture 89-real
uv run --no-project scripts/bench/suite.py `
  --library build/bench-native/install/bin/transcribe.dll `
  --wav samples/jfk.wav --output build/bench/upstream
```

Architecture 89 is this investigation's RTX 4070; select your own GPU's
architecture on other machines. CUDA graphs default off here to match the
existing app DLL baseline; `-CudaGraphs` is a separate experiment. Both CPU
and CUDA modules are compiled from this reference checkout's own GGML.

Every case keeps one model/session loaded for exactly three inferences,
discards run 1 from scores, and averages runs 2 and 3. Model loading and WAV
reading are separate metrics. No model downloads occur. Native wall/stage and
per-feed timing details are in the JSON, and backend logs are retained.
Use `--only nemotron parakeet` for a subset, or `pipeline.py --model <GGUF>`
for one installed weight file. Both CPU and CUDA are required by the suite.

Compare the fork with the same WAV, weight files, language, thread count,
device, build profile, graph settings and power state. Run sequentially on an
idle machine. `--baseline <summary.json>` checks warm averages and transcript
parity, with a default 15% timing budget. This short-file smoke does not replace
WER or long-stream testing. Upstream native stage timers retain upstream's
definitions; wall time is authoritative across instrumentation differences.

`scripts/bench/report.py --summary <dir>/summary.json` renders any suite summary
as `report.md` and `report.csv`, including the per-chunk stage columns. It reads
only the summary, so it renders this branch's and the fork's output alike, and
the app replay summaries too (the app runner writes the same field names). This
file is byte identical in this branch, in the fork and in `Handy_benchmarks`, so
the two sides cannot drift into separate reporting implementations.

```powershell
uv run --no-project scripts/bench/compare.py `
  --wav samples/jfk.wav --model <installed.gguf> --backend cpu `
  --arm upstream=build/bench-native/install/bin/transcribe.dll `
  --arm fork=../transcribe-fork/build/bench-native/install/bin/transcribe.dll
```

`compare.py` is the paired A/B runner and is the only thing that may be cited
for a claim about one tree versus another. `suite.py` runs one library's cases
back to back, so drift lands entirely inside whichever ran second: Nemotron Q6
CPU batch read 1090.3 ms here against 1208.4 ms in the fork, an apparent 10.8%
regression, while the same pair interleaved read 937.2 against 809.3 — the fork
13.6% faster, with this branch's own number moving 18% between the two runs.
Use `--baseline` to track this branch over time and `compare.py` to compare it
with anything else. Each arm loads through the bindings of the checkout that
owns its library by default (generated bindings are ABI specific, and a foreign
library fails at import on the first symbol it does not export); `--arm-bindings`
overrides that when a library lives outside its checkout.

Features gated at run time rather than build time are compared with
`--arm-env NAME=KEY=VALUE`, which sets an environment variable for one arm only
so the two arms need not be rebuilt; the gate each arm ran under is recorded in
the JSON. A gate compared this way proves nothing unless it actually fired — if
it found no work to do the arms ran identical code and any difference is noise.
Confirm the gate's own log line appears in the enabled arm and not in the
disabled one first (`TRANSCRIBE_GRAPH_OPTIMIZER` reports itself only when it
changed the graph), and use `--reps` of at least 3 for a decision: with two
repetitions the spread is the difference of two samples and the noise floor it
implies is far too small.
