# Building and running transcribe.cpp on Android

How to build an optimized transcribe.cpp for Android phones (arm64-v8a), what makes it
fast on modern SoCs, and how to deploy and tune it. Verified by building the full tree
with NDK 30 on Windows 11 (2026-09-26). **Not yet run on a physical device** — the NEON
ternary kernel was verified bit-exact against the scalar reference by running the same
intrinsics through SIMDe on x86 — so treat on-device numbers as still to be measured.

## 1. What the optimized build does

The `android-arm64` CMake preset (`CMakePresets.json`) produces:

```
build-android-arm64/
  src/libtranscribe.so                      the library an app links / loads via JNI
  ggml/src/libggml.so, libggml-base.so
  bin/libggml-cpu-android_armv8.0_1.so      baseline NEON (every arm64 phone)
  bin/libggml-cpu-android_armv8.2_1.so      + dot product (sdot/udot)
  bin/libggml-cpu-android_armv8.2_2.so      + FP16 vector arithmetic
  bin/libggml-cpu-android_armv8.6_1.so      + int8 matrix multiply (i8mm / smmla)
  bin/libggml-cpu-android_armv9.0_1.so      + SVE2
  bin/libggml-cpu-android_armv9.2_1.so      + SVE + SME
  bin/libggml-cpu-android_armv9.2_2.so      + SVE2 + SME
  bin/libggml-vulkan.so                     Adreno / Mali / Xclipse / PowerVR GPUs
  bin/transcribe-cli, transcribe-bench …    command-line tools
```

* **Runtime CPU dispatch** (`GGML_BACKEND_DL=ON` + `GGML_CPU_ALL_VARIANTS=ON`): ggml
  loads the best CPU variant the SoC supports. A single APK gets dot-product and
  i8mm kernels on phones that have them and still runs on old cores. A fixed
  `-march=armv8-a` build would miss 2–4× on the matmuls.
* **CPU_REPACK GEMMs**: on ARM with dotprod/i8mm, ggml has repacked (interleaved tile)
  GEMM kernels for Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, IQ4_NL and MXFP4. transcribe.cpp places
  eligible encoder weights in the CPU_REPACK buffer automatically when the CPU is the
  primary backend (log: `… use CPU_REPACK GEMM kernels`). That includes every ultra
  quant and redux (its ternary weights run as Q4_0 on CPU, losslessly).
* **Vulkan** for mobile GPUs. Quality of mobile Vulkan drivers varies a lot; always
  compare `--backend vulkan` against `--backend cpu` on the target phone (below).
* **No OpenMP, no system BLAS** (neither is available on Android).

## 2. Prerequisites

* Android NDK r27 or newer (tested: 30.0.14904198), `ANDROID_NDK` pointing at it.
* CMake ≥ 3.21 and Ninja.
* For the Vulkan backend:
  * Vulkan SDK on the **host** (`VULKAN_SDK`): provides `vulkan.hpp`, SPIRV-Headers
    and `glslc` used to compile the shaders at build time. The NDK ships `libvulkan.so`
    and the C headers but not the C++ `vulkan.hpp`.
  * A **host** C++ compiler: ggml compiles the `vulkan-shaders-gen` tool for the build
    machine in a sub-build. On Linux/macOS the system compiler is found automatically.
    On Windows use Visual Studio's x64 developer environment (see §3.2).

## 3. Build

### 3.1 Linux / macOS host

```bash
export ANDROID_NDK=$HOME/Android/Sdk/ndk/30.0.14904198
SYSROOT=$ANDROID_NDK/toolchains/llvm/prebuilt/$(uname -s | tr A-Z a-z)-x86_64/sysroot
cmake --preset android-arm64 \
  -DVulkan_INCLUDE_DIR=$VULKAN_SDK/include \
  -DVulkan_LIBRARY=$SYSROOT/usr/lib/aarch64-linux-android/28/libvulkan.so \
  -DVulkan_GLSLC_EXECUTABLE=$VULKAN_SDK/bin/glslc \
  -DSPIRV-Headers_DIR=$VULKAN_SDK/lib/cmake/SPIRV-Headers
cmake --build build-android-arm64 -j
```

### 3.2 Windows host

Run from a Visual Studio x64 developer environment so the host tool compiles with
MSVC, and point the shader-generator sub-build at the host toolchain file shipped in
the repo (`cmake/toolchains/host-msvc.cmake`). Without it the sub-build inherits the
NDK clang and fails to link a Windows executable (`lld: unable to find library
-lkernel32 …`).

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set ANDROID_NDK=C:/Users/<you>/AppData/Local/Android/Sdk/ndk/30.0.14904198
set SYSROOT=%ANDROID_NDK%/toolchains/llvm/prebuilt/windows-x86_64/sysroot
cmake --preset android-arm64 ^
  -DVulkan_INCLUDE_DIR=%VULKAN_SDK%/Include ^
  -DVulkan_LIBRARY=%SYSROOT%/usr/lib/aarch64-linux-android/28/libvulkan.so ^
  -DVulkan_GLSLC_EXECUTABLE=%VULKAN_SDK%/Bin/glslc.exe ^
  -DSPIRV-Headers_DIR=%VULKAN_SDK%/Lib/cmake/SPIRV-Headers ^
  -DGGML_VULKAN_SHADERS_GEN_TOOLCHAIN=%CD%/cmake/toolchains/host-msvc.cmake
cmake --build build-android-arm64 -j 12
```

### 3.3 CPU-only build

Add `-DTRANSCRIBE_VULKAN=OFF`: no Vulkan SDK or host compiler needed, and the
package is ~50 MB smaller (`libggml-vulkan.so` is the largest file).

### 3.4 Checks

```bash
file build-android-arm64/bin/transcribe-cli      # ELF 64-bit ... ARM aarch64 ... Android 28
ls build-android-arm64/bin/libggml-cpu-android_*.so   # 7 variants
```

## 4. Run on a device (adb)

```bash
D=/data/local/tmp/transcribe
adb shell mkdir -p $D
adb push build-android-arm64/bin/*.so build-android-arm64/bin/transcribe-cli \
         build-android-arm64/bin/transcribe-bench \
         build-android-arm64/src/libtranscribe.so \
         build-android-arm64/ggml/src/libggml.so build-android-arm64/ggml/src/libggml-base.so $D/
adb push parakeet-redux-0.6b-TQ1_Q8_0.gguf samples/jfk.wav $D/
adb shell "cd $D && LD_LIBRARY_PATH=. ./transcribe-cli --list-devices"
adb shell "cd $D && LD_LIBRARY_PATH=. ./transcribe-bench --model parakeet-redux-0.6b-TQ1_Q8_0.gguf --sample jfk.wav --backend cpu"
adb shell "cd $D && LD_LIBRARY_PATH=. ./transcribe-bench --model parakeet-redux-0.6b-TQ1_Q8_0.gguf --sample jfk.wav --backend vulkan"
```

The backend modules are found next to `libtranscribe.so` (package-local loader); an
app that places them elsewhere passes that directory to `transcribe_init_backends(dir)`.

## 5. Choosing models for phones

| Model file | Download | Weights in RAM on CPU (computed) | Notes |
|---|---|---|---|
| `parakeet-redux-0.6b-TQ1_Q8_0.gguf` | 159 MB | ≈ 365 MB (ternary part as Q4_0: 604 M × 4.5 bit) | Smallest download; repacked Q4_0 GEMM on CPU |
| `parakeet-redux-0.6b-TQ1_Q4_K.gguf` | 157 MB | ≈ 365 MB | Same accuracy as TQ1_Q8_0 |
| `parakeet-ultra-0.6b-Q4_K_M.gguf` | 485 MB | ≈ 485 MB | Best accuracy per byte of the ultra set; Q4_K repacked on ARM dotprod |
| `parakeet-ultra-0.6b-Q8_0.gguf` | 740 MB | ≈ 740 MB | Q8_0 is repacked on ARM (not on x86) |

Weight memory only; activations add tens of MB for a 30 s clip. Measure peak RSS on
the target device before committing to a model.

Redux trades ~3.5 WER points (FLEURS-fr 9.9 % vs 6.4–6.8 %) for a 3× smaller download.
To keep the smallest in-memory footprint for redux on a memory-starved device, set
`TRANSCRIBE_TERNARY_RUNTIME=native` (1.75 bpw in RAM, slower) or `q2_0` (2.25 bpw).

## 6. Tuning on the device

* **Threads**: phones are big.LITTLE. Using all cores often *loses* to using only the
  performance cores, because the slow cores make every matmul wait. Benchmark
  `--threads` = number of big/prime cores (typically 2–4) against the default.
* **CPU vs Vulkan**: on flagship Adreno GPUs Vulkan usually wins for the encoder; on
  mid-range Mali GPUs the CPU with i8mm can be faster and more predictable. Measure
  both with `transcribe-bench` (warm, several iterations) and ship per-device-class
  defaults.
* **Thermals**: phones throttle within seconds under load. Benchmark with a cool device,
  report the steady-state value after several iterations, and prefer the backend that
  holds its speed over the one with the best first run.
* **First Vulkan run** compiles pipelines (can take seconds); later runs are cached.
* Verify the chosen CPU variant and repack are active in the log output
  (`load_backend: loaded CPU backend from …libggml-cpu-android_armv8.6_1.so`,
  `… use CPU_REPACK GEMM kernels`).

## 7. App integration notes

* Package `libtranscribe.so`, `libggml*.so` and the `libggml-cpu-android_*` / `libggml-vulkan`
  modules under `jniLibs/arm64-v8a/`, and make sure the backend directory passed to
  `transcribe_init_backends` is the app's native library directory
  (`context.getApplicationInfo().nativeLibraryDir`).
* Keep `android:extractNativeLibs="true"` (or load from the APK with a loader that
  supports it) so ggml can `dlopen` the variant modules by path.
* Minimum API level is 28 (the preset's `ANDROID_PLATFORM`); raise it if the app needs.
