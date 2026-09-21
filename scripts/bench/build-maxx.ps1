param(
    # Compiler for the host language. 'msvc' reproduces build-native.ps1's
    # toolchain; 'clang' switches to clang-cl, which is the only way to reach
    # AVX-VNNI on x86 (see the -CpuVnni switch).
    [ValidateSet('msvc', 'clang')]
    [string]$Compiler = 'clang',
    # Name of the build under build/; lets several variants coexist so they can
    # be A/B'd against each other with compare.py instead of rebuilt in place.
    [string]$Name = 'maxx',
    [string]$CudaArchitecture = '89-real',
    [int]$Jobs = 6,
    [switch]$Cuda,

    # --- CPU ---
    # Target the build host's own microarchitecture (and every SIMD tier below
    # it) instead of a conservative floor. clang only: MSVC has no -march=native.
    [switch]$Native,
    # AVX-VNNI: the vpdpbusd integer dot product that the Q4_K/Q8_0 repack
    # GEMM, the q8_0 dot product and the llamafile sgemm all branch onto. Off
    # in the distribution lanes (scripts/build_windows.ps1 pins
    # GGML_AVX_VNNI=OFF) because the resulting binary faults with SIGILL on any
    # host without it; enable it only for a machine known to have AVX-VNNI.
    #
    # What each compiler needs, measured on this toolchain rather than inferred
    # from ggml's CMake:
    #   - MSVC: ggml's x86 branch appends only the __AVXVNNI__/GGML_AVX_VNNI
    #     definitions and never an /arch flag. That turns out to be enough —
    #     cl.exe 14.51 declares _mm256_dpbusd_avx_epi32 in immintrin.h and
    #     emits vpdpbusd under a plain /arch:AVX2 (verified by compiling and
    #     disassembling), so the definition alone selects the VNNI path.
    #   - clang-cl: needs -mavxvnni on the command line, which is why this
    #     script passes it globally.
    [switch]$CpuVnni,

    # --- CUDA ---
    [switch]$CudaGraphs,
    [ValidateSet('size', 'speed', 'none')]
    [string]$CudaCompression = '',
    [switch]$FaAllQuants,
    [switch]$ForceMmq,
    [switch]$ForceCublas,

    # --- Linking ---
    # Compile the ggml backends into the library instead of building them as
    # runtime-loaded modules. Static linking lets the optimizer inline across
    # the backend/core boundary; GGML_BACKEND_DL forbids it, and ggml also
    # hard-errors on GGML_NATIVE together with a module build.
    [switch]$StaticBackends,
    [switch]$Lto,

    [string]$Target = 'transcribe-cli'
)
$ErrorActionPreference = 'Stop'
$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$buildRoot = Join-Path $sourceRoot "build/$Name"
$installRoot = Join-Path $buildRoot 'install'

# cl.exe merely being on PATH does not mean the toolchain is usable: a bare
# PATH entry, or a shell inherited without the developer environment, leaves
# INCLUDE/LIB unset and the build then dies with C1083 on <stdbool.h>. Only
# skip the dev shell when the SDK/CRT headers are genuinely visible. clang-cl
# needs the same INCLUDE/LIB, so this preamble is common to both compilers.
$toolchainReady = [bool](Get-Command cl.exe -ErrorAction SilentlyContinue) -and
                  [bool]($env:INCLUDE -split ';' | Where-Object { $_ -match 'Windows Kits|MSVC' })
if (-not $toolchainReady) {
    $vsRoot = $null
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (Test-Path $vswhere) {
        $vsRoot = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
    }
    if (-not $vsRoot) {
        foreach ($base in @("${env:ProgramFiles}\Microsoft Visual Studio", "${env:ProgramFiles(x86)}\Microsoft Visual Studio")) {
            if (-not (Test-Path $base)) { continue }
            $found = Get-ChildItem -Path $base -Directory |
                ForEach-Object { Get-ChildItem -Path $_.FullName -Directory } |
                Where-Object { Test-Path (Join-Path $_.FullName 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll') } |
                Sort-Object FullName -Descending | Select-Object -First 1
            if ($found) { $vsRoot = $found.FullName; break }
        }
    }
    if (-not $vsRoot) { throw 'An x64 Visual Studio C++ toolchain is required (checked vswhere and the standard install roots)' }
    $vsRoot = $vsRoot.Trim()
    Import-Module (Join-Path $vsRoot 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vsRoot -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
}

$extra = @()
if ($Compiler -eq 'clang') {
    $clangCl = 'C:/Program Files/LLVM/bin/clang-cl.exe'
    if (-not (Test-Path $clangCl)) { throw "clang-cl not found at $clangCl" }
    $extra += "-DCMAKE_C_COMPILER=$clangCl", "-DCMAKE_CXX_COMPILER=$clangCl"
} elseif ($Native) {
    throw '-Native requires -Compiler clang; MSVC has no -march=native'
}

# `-DCMAKE_<LANG>_FLAGS` REPLACES the value CMake initialised for this platform
# rather than extending it. Restating the platform defaults here is therefore
# mandatory, not cosmetic: dropping /DWIN32 and /D_WINDOWS changes which
# conditionals compile, and dropping /EHsc is worse still — CMakeLists.txt:559
# rewrites /EHsc into /EHs precisely so exceptions raised in ggml's C-ABI frames
# unwind into transcribe's public-entry guards, so a build without /EHsc
# silently loses that protection instead of failing loudly. The reference build
# (build-native.ps1) never overrides these and so keeps them, and any lane that
# is meant to be comparable to it has to keep them too.
$cFlags   = '/DWIN32 /D_WINDOWS'
$cxxFlags = '/DWIN32 /D_WINDOWS /EHsc'
if ($Compiler -eq 'clang') {
    # -march=native is the honest "optimize for THIS machine" switch and implies
    # every tier below it; the explicit x86 flags are belt-and-braces so the ISA
    # is on even in translation units outside ggml-cpu's own ARCH_FLAGS.
    if ($Native)   { $cFlags += ' -march=native'; $cxxFlags += ' -march=native' }
    # clang-cl needs -mavxvnni explicitly (see the -CpuVnni comment above).
    if ($CpuVnni)  { $cFlags += ' -mavxvnni';     $cxxFlags += ' -mavxvnni' }
    if ($Lto) {
        # clang-cl -flto emits LLVM bitcode, which MSVC's link.exe cannot
        # consume; lld-link is required.
        $cFlags += ' -flto -fuse-ld=lld'; $cxxFlags += ' -flto -fuse-ld=lld'
    }
}
if ($cFlags -ne '/DWIN32 /D_WINDOWS') {
    $extra += "-DCMAKE_C_FLAGS=$cFlags", "-DCMAKE_CXX_FLAGS=$cxxFlags"
}
# MSVC + -CpuVnni needs no flags at all: the GGML_AVX_VNNI definition below is
# what selects the intrinsic, and cl.exe emits it under the /arch:AVX2 ggml
# already passes.
if ($CpuVnni) { $extra += '-DGGML_AVX_VNNI=ON' }
if ($Lto -and $Compiler -eq 'msvc') { $extra += '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON' }
if ($Lto -and $Compiler -eq 'clang') { $extra += '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON' }

$backendDl = if ($StaticBackends) { 'OFF' } else { 'ON' }
$graphs = if ($CudaGraphs) { 'ON' } else { 'OFF' }
if ($CudaCompression) { $extra += "-DGGML_CUDA_COMPRESSION_MODE=$CudaCompression" }
if ($FaAllQuants) { $extra += '-DGGML_CUDA_FA_ALL_QUANTS=ON' }
if ($ForceMmq) { $extra += '-DGGML_CUDA_FORCE_MMQ=ON' }
if ($ForceCublas) { $extra += '-DGGML_CUDA_FORCE_CUBLAS=ON' }

# Not `$cuda`: PowerShell variable names are case-insensitive, so `$cuda` and
# the `[switch]$Cuda` above are the same variable and the assignment lands a
# string in a switch parameter.
$cudaEnabled = if ($Cuda) { 'ON' } else { 'OFF' }
$configure = @('-S', $sourceRoot, '-B', $buildRoot, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DTRANSCRIBE_BUILD_SHARED=ON',
    "-DTRANSCRIBE_GGML_BACKEND_DL=$backendDl", "-DTRANSCRIBE_CUDA=$cudaEnabled",
    '-DTRANSCRIBE_ARCH_DL=ON',
    '-DTRANSCRIBE_INSTALL=ON', '-DTRANSCRIBE_BUILD_TESTS=OFF',
    '-DGGML_NATIVE=OFF', '-DGGML_CPU_ALL_VARIANTS=OFF',
    '-DGGML_AVX=ON', '-DGGML_AVX2=ON', '-DGGML_FMA=ON', '-DGGML_F16C=ON',
    '-DGGML_CCACHE=OFF', '-DCMAKE_CUDA_COMPILER_LAUNCHER=',
    '-DCMAKE_C_COMPILER_LAUNCHER=', '-DCMAKE_CXX_COMPILER_LAUNCHER=',
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitecture", "-DGGML_CUDA_GRAPHS=$graphs") + $extra
& cmake @configure
if ($LASTEXITCODE) { throw 'CMake configure failed' }
& cmake --build $buildRoot --target $Target --parallel $Jobs
if ($LASTEXITCODE) { throw 'Build failed' }
if ($Target -ne 'transcribe-cli') { return }
& cmake --install $buildRoot --prefix $installRoot
if ($LASTEXITCODE) { throw 'Install failed' }
Write-Output "maxx install: $installRoot"
