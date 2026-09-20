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
