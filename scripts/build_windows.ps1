<#
.SYNOPSIS
  Build transcribe.cpp locally on Windows with MSVC + optional CUDA.

.DESCRIPTION
  Mirrors audio.cpp/scripts/build_windows.ps1: auto-detects the CUDA toolkit,
  Visual Studio install, and the current GPU's compute capability, then drives
  CMake/Ninja. Presets bundle the common flag sets; -Target picks what to build.

  Examples:
    # CUDA release, build only the CLI, auto-detect GPU arch (sm_120 on a 5060 Ti)
    .\scripts\build_windows.ps1 -Preset windows-cuda-release -Target transcribe-cli -Ccache

    # CPU-only release, build the C ABI shared library for downstream consumers
    .\scripts\build_windows.ps1 -Preset windows-cpu-release -Target transcribe

    # Configure only (no build), inspect the generated build/<preset>/
    .\scripts\build_windows.ps1 -Preset windows-cuda-release -ConfigureOnly

.PARAMETER Preset
  windows-cpu-release   Release, CPU backend, native CPU + llamafile, single
                       self-contained transcribe.dll (SHARED_EMBED=ON, ggml
                       statically linked inside — the posture C ABI consumers
                       downstream of the publish-to-OSS pipeline expect).
  windows-cuda-release  Release, CUDA + CUDA graphs + native CPU + llamafile.
  windows-cuda-debug    Debug, CUDA + CUDA graphs, tests on, /O2 /Zi.

.PARAMETER Target
  CMake target to build. Common: transcribe-cli (the CLI), transcribe (the
  library), transcribe-bench, transcribe-quantize, or a test binary like
  transcribe_whisper_e2e_smoke. Empty = build the default target.

.PARAMETER CudaArchitectures
  "auto" (default): probe the current GPU via nvidia-smi and build only its
  arch (fastest; e.g. "120a-real" on a 5060 Ti).
  "default": the release multi-arch set (75;80;86;89;120a for distribution).
  Otherwise: a literal CMAKE_CUDA_ARCHITECTURES value like "120-real" or "75;86".

.PARAMETER CpuArch
  "", "native", "avx2", or "baseline". Overrides the preset's CPU SIMD profile.

.PARAMETER Ccache
  Enable ccache (auto-detected on PATH; pass this to force a clear message if
  missing). Unchanged sources skip recompilation on clean rebuilds.

.PARAMETER SharedEmbed
  "ON" / "OFF" — override TRANSCRIBE_SHARED_EMBED. With ON, the libtranscribe
  target is a single self-contained .dll/.so with ggml statically linked inside
  (the "prebuilt consumer" posture the capi CI matrix uses). CPU preset
  defaults to ON; CUDA presets default to OFF (developer builds).

.PARAMETER Jobs
  Parallel build jobs. Default: max(2, ProcessorCount).

.PARAMETER Clean
  Build --target clean for this preset's build dir, then exit.

.PARAMETER ConfigureOnly
  Configure and exit (do not build).

.PARAMETER VsInstall
  Explicit Visual Studio install path (skips auto-detection).
#>
[CmdletBinding()]
param(
    [string]$Preset = "windows-cuda-release",
    [string]$Target = "transcribe",
    [int]$Jobs = 0,
    [switch]$ConfigureOnly,
    [switch]$Clean,
    [string]$CudaArchitectures = "auto",
    [ValidateSet("", "native", "avx2", "baseline")]
    [string]$CpuArch = "",
    [ValidateSet("ON", "OFF")]
    [string]$NativeCpu = $null,
    [ValidateSet("ON", "OFF")]
    [string]$Llamafile = $null,
    [ValidateSet("ON", "OFF")]
    [string]$SharedEmbed = $null,
    [ValidateSet("", "full", "minimal-multilingual", "custom", "none")]
    [string]$ModelSet = "",
    [string]$Models = "",
    [ValidateSet("ON", "OFF")]
    [string]$ArchDl = $null,
    [switch]$Ccache,
    [switch]$RealModelTests,
    [string]$VsInstall = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter()][string[]]$Arguments = @()
    )
    Write-Host "> $FilePath $($Arguments -join ' ')"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Convert-ToCMakePath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ($Path -replace "\\", "/")
}

function Add-PathFront {
    param([Parameter(Mandatory = $true)][string]$Path)
    # [IO.Path]::PathSeparator is a static FIELD, but Set-StrictMode -Latest
    # on some PowerShell builds mis-parses it as a property lookup and throws.
    # The separator is ";" on Windows regardless — hardcode it.
    if ((Test-Path $Path) -and (($env:PATH -split ";") -notcontains $Path)) {
        $env:PATH = ($Path, $env:PATH) -join ";"
    }
}

function Add-EnvListFront {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Paths
    )
    $existing = [Environment]::GetEnvironmentVariable($Name, "Process")
    $items = @()
    foreach ($path in $Paths) {
        if ($path -and (Test-Path $path) -and ($items -notcontains $path)) { $items += $path }
    }
    if ($existing) {
        foreach ($path in ($existing -split ";")) {
            if ($path -and ($items -notcontains $path)) { $items += $path }
        }
    }
    [Environment]::SetEnvironmentVariable($Name, ($items -join ";"), "Process")
}

function Find-FirstFile {
    param([Parameter(Mandatory = $true)][string[]]$Patterns)
    foreach ($pattern in $Patterns) {
        $found = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
        if ($null -ne $found) { return $found.FullName }
    }
    return ""
}

function Find-CudaRoot {
    foreach ($root in @($env:CUDA_PATH, $env:CUDAToolkit_ROOT)) {
        if ($root -and (Test-Path (Join-Path $root "bin\nvcc.exe"))) {
            return (Resolve-Path $root).Path
        }
    }
    $nvcc = Find-FirstFile @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe",
        "C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\nvcc.exe"
    )
    if ($nvcc -ne "") { return (Resolve-Path (Join-Path (Split-Path $nvcc -Parent) "..")).Path }
    return ""
}

function Test-VsInstall {
    param([Parameter(Mandatory = $true)][string]$Path)
    return ($Path -ne "" -and (Test-Path (Join-Path $Path "VC\Tools\MSVC")))
}

function Find-VsInstall {
    param([string]$RequestedInstall = "")
    if ($RequestedInstall -ne "") {
        if (Test-VsInstall $RequestedInstall) { return (Resolve-Path $RequestedInstall).Path }
        throw "Requested Visual Studio install was not usable: $RequestedInstall"
    }
    $candidates = @()
    $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)}) | Where-Object { $_ }
    foreach ($root in $roots) {
        foreach ($year in @("2026", "2022", "2019")) {
            foreach ($edition in @("BuildTools", "Community", "Professional", "Enterprise")) {
                $candidates += (Join-Path $root "Microsoft Visual Studio\$year\$edition")
            }
        }
    }
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $found) { $candidates += $found }
    }
    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        if (Test-VsInstall $candidate) { return (Resolve-Path $candidate).Path }
    }
    return ""
}

function Find-MsvcCompiler {
    param([Parameter(Mandatory = $true)][string]$VsInstall)
    return Find-FirstFile @("$VsInstall\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe")
}

function Find-VsCMake {
    param([Parameter(Mandatory = $true)][string]$VsInstall)
    $cmake = Find-FirstFile @("$VsInstall\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe")
    if ($cmake -ne "") { return $cmake }
    $cmd = Get-Command "cmake.exe" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source } else { return "" }
}

function Find-VsNinja {
    param([Parameter(Mandatory = $true)][string]$VsInstall)
    $ninja = Find-FirstFile @("$VsInstall\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe")
    if ($ninja -ne "") { return $ninja }
    $cmd = Get-Command "ninja.exe" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source } else { return "" }
}

function Find-WindowsKitTool {
    param([Parameter(Mandatory = $true)][string]$Name)
    return Find-FirstFile @(
        "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\$Name",
        "C:\Program Files\Windows Kits\10\bin\*\x64\$Name"
    )
}

function Add-MsvcEnvironment {
    param(
        [Parameter(Mandatory = $true)][string]$VsInstall,
        [Parameter(Mandatory = $true)][string]$Cl,
        [Parameter(Mandatory = $true)][string]$SdkTool
    )
    $vcvars = Join-Path $VsInstall "VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $vcvars) {
        $cmd = "`"$vcvars`" >nul && set"
        foreach ($line in (& cmd.exe /d /s /c $cmd)) {
            if ($line -match "^([^=]+)=(.*)$") {
                [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
            }
        }
    }
    if ($Cl -notmatch "^(.*\\VC\\Tools\\MSVC\\[^\\]+)\\bin\\Hostx64\\x64\\cl\.exe$") {
        throw "Could not parse MSVC toolset root from $Cl"
    }
    $msvcRoot = $Matches[1]
    if ($SdkTool -notmatch "^(.*\\Windows Kits\\10)\\bin\\([^\\]+)\\x64\\[^\\]+\.exe$") {
        throw "Could not parse Windows SDK root from $SdkTool"
    }
    $sdkRoot = $Matches[1]
    $sdkVersion = $Matches[2]
    Add-PathFront (Join-Path $msvcRoot "bin\Hostx64\x64")
    Add-PathFront (Join-Path $sdkRoot "bin\$sdkVersion\x64")
    Add-EnvListFront "INCLUDE" @(
        (Join-Path $msvcRoot "include"),
        (Join-Path $sdkRoot "Include\$sdkVersion\ucrt"),
        (Join-Path $sdkRoot "Include\$sdkVersion\shared"),
        (Join-Path $sdkRoot "Include\$sdkVersion\um"),
        (Join-Path $sdkRoot "Include\$sdkVersion\winrt"),
        (Join-Path $sdkRoot "Include\$sdkVersion\cppwinrt")
    )
    Add-EnvListFront "LIB" @(
        (Join-Path $msvcRoot "lib\x64"),
        (Join-Path $sdkRoot "Lib\$sdkVersion\ucrt\x64"),
        (Join-Path $sdkRoot "Lib\$sdkVersion\um\x64")
    )
    Add-EnvListFront "LIBPATH" @((Join-Path $msvcRoot "lib\x64"))
}

function Resolve-CudaArchitectures {
    # "default" = the multi-arch release set for distribution artifacts.
    function Get-ReleaseCudaArchitectures {
        $nvcc = Join-Path $env:CUDA_PATH "bin\nvcc.exe"
        $supported = @()
        if (Test-Path -LiteralPath $nvcc) {
            $supported = & $nvcc --list-gpu-arch 2>$null
        }
        $wanted = @(
            @{ Compute = "compute_75";  Arch = "75-virtual"  },
            @{ Compute = "compute_80";  Arch = "80-virtual"  },
            @{ Compute = "compute_86";  Arch = "86-real"     },
            @{ Compute = "compute_89";  Arch = "89-real"     },
            @{ Compute = "compute_120"; Arch = "120a-real"   }
        )
        $archs = @()
        foreach ($item in $wanted) {
            if ($supported -contains $item.Compute) { $archs += $item.Arch }
        }
        if ($archs.Count -eq 0) { $archs = @("75-virtual", "80-virtual", "86-real") }
        return ($archs -join ";")
    }

    if ($CudaArchitectures -eq "default") { return Get-ReleaseCudaArchitectures }
    if ($CudaArchitectures -ne "" -and $CudaArchitectures -ne "auto") { return $CudaArchitectures }

    # "auto": probe the current GPU and build only its arch (fastest local build).
    $smi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if ($null -eq $smi) { return "" }
    $cap = (& $smi.Source --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
    if ($cap -notmatch "^(\d+)\.(\d+)") { return "" }
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    $arch = "$major$minor"
    # Blackwell (cc 12.x) needs the "a" suffix in CMAKE_CUDA_ARCHITECTURES.
    if ($major -ge 12) { return "${arch}a-real" }
    return "${arch}-real"
}

function Get-CpuArchSettings {
    param([AllowEmptyString()][string]$Name)
    switch ($Name) {
        "" {
            return @{ Label = "preset default"; Native = $null; CMakeArgs = @() }
        }
        "native" {
            return @{ Label = "native"; Native = "ON"; CMakeArgs = @() }
        }
        "avx2" {
            return @{ Label = "AVX2"; Native = "OFF"; CMakeArgs = @(
                "-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_AVX512=OFF",
                "-DGGML_AVX512_VBMI=OFF", "-DGGML_AVX512_VNNI=OFF",
                "-DGGML_AVX512_BF16=OFF", "-DGGML_AVX_VNNI=OFF"
            ) }
        }
        "baseline" {
            return @{ Label = "baseline"; Native = "OFF"; CMakeArgs = @(
                "-DGGML_AVX=OFF", "-DGGML_AVX2=OFF", "-DGGML_AVX512=OFF",
                "-DGGML_AVX512_VBMI=OFF", "-DGGML_AVX512_VNNI=OFF",
                "-DGGML_AVX512_BF16=OFF", "-DGGML_AVX_VNNI=OFF"
            ) }
        }
    }
}

function Get-PresetSettings {
    param([Parameter(Mandatory = $true)][string]$Name)
    switch ($Name) {
        "windows-cpu-release" {
            return @{
                BuildType = "Release"; BuildTests = "OFF"; BuildRealModelTests = "OFF"
                BuildExamples = "ON"; BuildTools = "OFF"
                Native = "ON"; Llamafile = "ON"
                EnableCuda = "OFF"; EnableCudaGraphs = "OFF"
                SharedEmbed = "ON"  # single self-contained transcribe.dll; matches docstring "C ABI shared library for downstream consumers"
                CFlagsDebug = ""; CxxFlagsDebug = ""
            }
        }
        "windows-cuda-release" {
            return @{
                BuildType = "Release"; BuildTests = "OFF"; BuildRealModelTests = "OFF"
                BuildExamples = "ON"; BuildTools = "OFF"
                Native = "ON"; Llamafile = "ON"
                EnableCuda = "ON"; EnableCudaGraphs = "ON"
                SharedEmbed = "OFF"  # dev build; pass -SharedEmbed ON to match capi CI posture
                CFlagsDebug = ""; CxxFlagsDebug = ""
            }
        }
        "windows-cuda-debug" {
            return @{
                BuildType = "Debug"; BuildTests = "ON"; BuildRealModelTests = "ON"
                BuildExamples = "ON"; BuildTools = "OFF"
                Native = "ON"; Llamafile = "ON"
                EnableCuda = "ON"; EnableCudaGraphs = "ON"
                SharedEmbed = "OFF"  # dev build
                CFlagsDebug = "/O2 /Zi"; CxxFlagsDebug = "/O2 /Zi"
            }
        }
        default {
            throw "Unsupported Windows preset '$Name'. Use windows-cpu-release, windows-cuda-release, or windows-cuda-debug."
        }
    }
}

# === Main flow =============================================================

$settings = Get-PresetSettings $Preset
$cpuArchSettings = Get-CpuArchSettings $CpuArch
if ($null -ne $cpuArchSettings.Native) { $settings.Native = $cpuArchSettings.Native }
if (-not [string]::IsNullOrEmpty($NativeCpu)) { $settings.Native = $NativeCpu }
if (-not [string]::IsNullOrEmpty($Llamafile)) { $settings.Llamafile = $Llamafile }
if (-not [string]::IsNullOrEmpty($SharedEmbed)) { $settings.SharedEmbed = $SharedEmbed }
if ($RealModelTests) {
    # Real-model tests are nested under TRANSCRIBE_BUILD_TESTS in tests/CMakeLists.txt,
    # so turning them on also requires the outer tests gate.
    $settings.BuildRealModelTests = "ON"
    $settings.BuildTests = "ON"
}
$isCudaPreset = $settings.EnableCuda -eq "ON"

if ($isCudaPreset) {
    $cudaRoot = Find-CudaRoot
    if ($cudaRoot -eq "") {
        throw "Official CUDA Toolkit was not found. Install it so nvcc exists under C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin."
    }
    Add-PathFront (Join-Path $cudaRoot "bin")
    $env:CUDA_PATH = $cudaRoot
    $env:CUDAToolkit_ROOT = $cudaRoot
} else {
    $cudaRoot = ""
}

$vsInstall = Find-VsInstall $VsInstall
if ($vsInstall -eq "") {
    throw "Visual Studio C++ workload was not found. Install VS 2022+ (or Build Tools) with MSVC, Windows SDK, CMake/Ninja."
}

$cl = Find-MsvcCompiler $vsInstall
$cmake = Find-VsCMake $vsInstall
$ninja = Find-VsNinja $vsInstall
$mt = Find-WindowsKitTool "mt.exe"
$rc = Find-WindowsKitTool "rc.exe"
if ($cl -eq "" -or $cmake -eq "" -or $ninja -eq "" -or $mt -eq "" -or $rc -eq "") {
    throw "Missing Build Tools component. Need cl.exe, cmake.exe, ninja.exe, mt.exe, and rc.exe."
}

Add-MsvcEnvironment $vsInstall $cl $mt
Add-PathFront (Split-Path $ninja -Parent)
$arch = if ($isCudaPreset) { Resolve-CudaArchitectures } else { "" }

Write-Host "Preset: $Preset"
if ($isCudaPreset) { Write-Host "CUDA: $cudaRoot" } else { Write-Host "CUDA: disabled" }
Write-Host "Visual Studio: $vsInstall"
Write-Host "MSVC: $cl"
Write-Host "CMake: $cmake"
Write-Host "Ninja: $ninja"
Write-Host "Windows SDK: $(Split-Path $mt -Parent)"
if ($arch -ne "") { Write-Host "CUDA architectures: $arch" }
Write-Host "CPU profile: $($cpuArchSettings.Label) (native=$($settings.Native))"
Write-Host "llamafile SGEMM: $($settings.Llamafile)"

if ($Clean) {
    $buildDirForClean = Join-Path (Join-Path (Split-Path $PSScriptRoot -Parent) "build") $Preset
    if (Test-Path $buildDirForClean) {
        Invoke-Checked $cmake @("--build", $buildDirForClean, "--target", "clean")
    }
    exit 0
}

$sourceDir = Split-Path $PSScriptRoot -Parent
$buildDir = Join-Path (Join-Path $sourceDir "build") $Preset

# /utf-8 is scoped via CMAKE_C/CXX_FLAGS (cl.exe host compiler only) and
# -Xcompiler=/utf-8 for nvcc. Passing /utf-8 to nvcc directly breaks CUDA
# builds ("single input file required" — nvcc reads it as a filename).
$configureArgs = @(
    "-S", $sourceDir,
    "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$($settings.BuildType)",
    "-DCMAKE_C_COMPILER=$(Convert-ToCMakePath $cl)",
    "-DCMAKE_CXX_COMPILER=$(Convert-ToCMakePath $cl)",
    "-DCMAKE_C_FLAGS=/utf-8",
    "-DCMAKE_CXX_FLAGS=/utf-8 /EHsc",
    "-DCMAKE_MAKE_PROGRAM=$(Convert-ToCMakePath $ninja)",
    "-DCMAKE_MT=$(Convert-ToCMakePath $mt)",
    "-DCMAKE_RC_COMPILER=$(Convert-ToCMakePath $rc)",
    "-DGGML_NATIVE=$($settings.Native)",
    "-DGGML_LLAMAFILE=$($settings.Llamafile)",
    "-DTRANSCRIBE_CUDA=$($settings.EnableCuda)",
    "-DTRANSCRIBE_VULKAN=OFF",
    "-DTRANSCRIBE_METAL=OFF",
    "-DTRANSCRIBE_BUILD_TESTS=$($settings.BuildTests)",
    "-DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=$($settings.BuildRealModelTests)",
    "-DTRANSCRIBE_BUILD_EXAMPLES=$($settings.BuildExamples)",
    "-DTRANSCRIBE_BUILD_TOOLS=$($settings.BuildTools)",
    "-DTRANSCRIBE_SHARED_EMBED=$($settings.SharedEmbed)"
)
if (-not [string]::IsNullOrEmpty($ModelSet)) {
    $configureArgs += "-DTRANSCRIBE_MODEL_SET=$ModelSet"
}
if (-not [string]::IsNullOrEmpty($Models)) {
    $configureArgs += "-DTRANSCRIBE_MODELS=$Models"
}
if (-not [string]::IsNullOrEmpty($ArchDl)) {
    $configureArgs += "-DTRANSCRIBE_ARCH_DL=$ArchDl"
}
$configureArgs += $cpuArchSettings.CMakeArgs
if ($settings.CFlagsDebug -ne "") { $configureArgs += "-DCMAKE_C_FLAGS_DEBUG=$($settings.CFlagsDebug)" }
if ($settings.CxxFlagsDebug -ne "") { $configureArgs += "-DCMAKE_CXX_FLAGS_DEBUG=$($settings.CxxFlagsDebug)" }
if ($isCudaPreset) {
    $configureArgs += "-DCUDAToolkit_ROOT=$(Convert-ToCMakePath $cudaRoot)"
    $configureArgs += "-DCMAKE_CUDA_HOST_COMPILER=$(Convert-ToCMakePath $cl)"
    $configureArgs += "-DCMAKE_CUDA_FLAGS=-Xcompiler=/utf-8"
    # Forward the preset's CUDA-graphs setting. The preset sets
    # EnableCudaGraphs but it was previously a dead value (read nowhere), so
    # the build silently shipped with GGML_CUDA_GRAPHS=OFF. On a clean
    # reconfigure the ggml cache default (OFF) took over, hiding the intent.
    $configureArgs += "-DGGML_CUDA_GRAPHS=$($settings.EnableCudaGraphs)"
}
if ($isCudaPreset -and $arch -ne "") {
    $configureArgs += "-DCMAKE_CUDA_ARCHITECTURES=$arch"
} elseif ($isCudaPreset) {
    $configureArgs += @("-U", "CMAKE_CUDA_ARCHITECTURES")
}

# ccache: cache .obj files so unchanged sources (vendored ggml) skip
# recompilation on clean rebuilds. Note: on Windows, ccache's nvcc wrapper has
# historically been unreliable; if CUDA builds fail with "single input file
# required", drop -Ccache (or GGML_CCACHE=OFF) for the CUDA preset.
$ccacheExe = $null
if ($Ccache) {
    $ccacheExe = Get-Command "ccache" -ErrorAction SilentlyContinue
    if (-not $ccacheExe) {
        Write-Host "WARNING: -Ccache specified but ccache not found on PATH. Continuing without cache."
    }
} else {
    $ccacheExe = Get-Command "ccache" -ErrorAction SilentlyContinue
}
if ($ccacheExe) {
    $ccachePath = $ccacheExe.Source
    Write-Host "ccache: $ccachePath"
    $configureArgs += "-DCMAKE_C_COMPILER_LAUNCHER=$ccachePath"
    $configureArgs += "-DCMAKE_CXX_COMPILER_LAUNCHER=$ccachePath"
    if ($isCudaPreset) {
        $configureArgs += "-DCMAKE_CUDA_COMPILER_LAUNCHER=$ccachePath"
    }
} else {
    # No ccache: ensure ggml's default-on GGML_CCACHE doesn't try to invoke a
    # missing binary (it would warn but not fail; explicit OFF keeps it clean).
    $configureArgs += "-DGGML_CCACHE=OFF"
}

Invoke-Checked $cmake $configureArgs

if ($ConfigureOnly) { exit 0 }

$effectiveJobs = if ($Jobs -gt 0) { $Jobs } else { [Math]::Max(2, [Environment]::ProcessorCount) }
$buildArgs = @("--build", $buildDir, "-j", $effectiveJobs.ToString())
if ($Target -ne "") { $buildArgs += @("--target", $Target) }

Write-Host "Build jobs: $effectiveJobs"
Write-Host "Target: $(if ($Target) { $Target } else { 'default' })"
Invoke-Checked $cmake $buildArgs
