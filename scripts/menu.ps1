<#
.SYNOPSIS
  Interactive Terminal Menu for transcribe.cpp Build Configuration
.DESCRIPTION
  Provides a user-friendly, interactive configuration interface to build:
    1. Modular Plugin Mode (Zero-Model Minimal Core + Downloadable DLL Plugins)
    2. Monolithic Mode (Single Binary with compiled-in models)
    3. Model Selection: minimal-multilingual, full, custom, or none
    4. Compute Backends: CPU (AVX2), CUDA (auto-detected compute capability incl. sm_120), Vulkan
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ANSI Colors
$ESC = [char]27
$C_RESET   = "$ESC[0m"
$C_BOLD    = "$ESC[1m"
$C_CYAN    = "$ESC[36m"
$C_GREEN   = "$ESC[32m"
$C_YELLOW  = "$ESC[33m"
$C_BLUE    = "$ESC[34m"
$C_MAGENTA = "$ESC[35m"
$C_RED     = "$ESC[31m"
$C_WHITE   = "$ESC[37m"

function Print-Banner {
    Clear-Host
    Write-Host "$C_CYAN$C_BOLD========================================================================$C_RESET"
    Write-Host "$C_CYAN$C_BOLD          TRANSCRIBE.CPP - MODULAR BUILD CONFIGURATION MENU             $C_RESET"
    Write-Host "$C_CYAN$C_BOLD========================================================================$C_RESET"
    Write-Host "$C_WHITE  High-performance speech-to-text engine with dynamic plugin architecture$C_RESET"
    Write-Host ""
}

function Prompt-Choice {
    param(
        [string]$Title,
        [string[]]$Options,
        [int]$DefaultIndex = 1
    )
    Write-Host "$C_YELLOW$C_BOLD[?] $Title$C_RESET"
    for ($i = 0; $i -lt $Options.Length; $i++) {
        $num = $i + 1
        $opt = $Options[$i]
        $defTag = if ($num -eq $DefaultIndex) { " $C_GREEN(Default)$C_RESET" } else { "" }
        Write-Host "  $C_BOLD$num)$C_RESET $opt$defTag"
    }
    while ($true) {
        $inputVal = Read-Host "  Enter choice [1-$($Options.Length)] (default $DefaultIndex)"
        if ([string]::IsNullOrWhiteSpace($inputVal)) {
            return $DefaultIndex
        }
        $val = 0
        if ([int]::TryParse($inputVal, [ref]$val) -and $val -ge 1 -and $val -le $Options.Length) {
            return $val
        }
        Write-Host "  $C_REDInvalid choice. Please choose between 1 and $($Options.Length).$C_RESET"
    }
}

# 1. Architecture Mode
Print-Banner
$modeChoice = Prompt-Choice -Title "Select Build Architecture Mode" -Options @(
    "Modular Dynamic Plugins (TRANSCRIBE_ARCH_DL=ON) - Zero-model core + independent .dll plugins [Recommended for Handy]",
    "Monolithic Static Build (TRANSCRIBE_ARCH_DL=OFF) - All selected models compiled directly into library"
) -DefaultIndex 1

$archDl = if ($modeChoice -eq 1) { "ON" } else { "OFF" }

# 2. Model Set
Print-Banner
$modelChoice = Prompt-Choice -Title "Select Model Preset / Composite" -Options @(
    "minimal-multilingual [Parakeet TDT 0.6B + Granite Speech 2B + Qwen3-ASR 1.7B] (Fastest, compact, multilingual)",
    "full [All 18 speech-to-text models + diarizers]",
    "custom [Select specific model families manually]",
    "none [Zero models built-in; ideal for dynamic plugin runtime]"
) -DefaultIndex 1

$modelSet = "minimal-multilingual"
$customModels = ""

switch ($modelChoice) {
    1 { $modelSet = "minimal-multilingual" }
    2 { $modelSet = "full" }
    3 {
        $modelSet = "custom"
        Write-Host ""
        Write-Host "$C_YELLOWAvailable families: parakeet, granite, qwen3_asr, whisper, cohere, canary, moss, sortformer, voxtral, voxtral_realtime, canary_qwen, moonshine, moonshine_streaming, sensevoice, funasr_nano, gigaam, granite_nar, medasr$C_RESET"
        $customModels = Read-Host "  Enter comma-separated list of families (e.g. parakeet,granite)"
        if ([string]::IsNullOrWhiteSpace($customModels)) {
            $customModels = "parakeet,granite,qwen3_asr"
        }
    }
    4 { $modelSet = "none" }
}

# 3. Compute Backend
Print-Banner
# Probe NVIDIA GPU
$gpuDetected = ""
$hasNvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
if ($hasNvidiaSmi) {
    try {
        $gpuOut = & nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>$null
        if ($gpuOut) {
            $gpuDetected = " (Detected: $($gpuOut.Trim()))"
        }
    } catch {}
}

$backendChoice = Prompt-Choice -Title "Select Compute Backend" -Options @(
    "CPU (Host SIMD AVX2/AVX-512, highly optimized scalar fallback)",
    "CUDA (NVIDIA GPU Acceleration)$gpuDetected",
    "Vulkan (Cross-vendor GPU acceleration)"
) -DefaultIndex 1

$enableCuda = "OFF"
$enableVulkan = "OFF"
$cudaArch = "auto"
if ($backendChoice -eq 2) {
    $enableCuda = "ON"
    Write-Host ""
    $cudaArchInput = Read-Host "  Enter CUDA architectures (default: 'auto' for current GPU, e.g. sm_120 for RTX 50-series)"
    if (-not [string]::IsNullOrWhiteSpace($cudaArchInput)) {
        $cudaArch = $cudaArchInput
    }
} elseif ($backendChoice -eq 3) {
    $enableVulkan = "ON"
}

# 4. Target
Print-Banner
$targetChoice = Prompt-Choice -Title "Select Target to Build" -Options @(
    "transcribe-cli (Command-line tool + all architecture plugins)",
    "transcribe (C ABI Shared Library / DLL)",
    "all (Build all targets, tests, tools, and plugins)"
) -DefaultIndex 1

$target = switch ($targetChoice) {
    1 { "transcribe-cli" }
    2 { "transcribe" }
    3 { "all" }
}

# Summary
Print-Banner
Write-Host "$C_GREEN$C_BOLDConfiguration Summary:$C_RESET"
Write-Host "  $C_BOLDPlugin Architecture (TRANSCRIBE_ARCH_DL):$C_RESET $archDl"
Write-Host "  $C_BOLDModel Composite (TRANSCRIBE_MODEL_SET):$C_RESET  $modelSet"
if ($modelSet -eq "custom") {
    Write-Host "  $C_BOLDCustom Models (TRANSCRIBE_MODELS):$C_RESET        $customModels"
}
Write-Host "  $C_BOLDBackend:$C_RESET                                 $(if ($enableCuda -eq 'ON') { "CUDA ($cudaArch)" } elseif ($enableVulkan -eq 'ON') { "Vulkan" } else { "CPU" })"
Write-Host "  $C_BOLDTarget:$C_RESET                                  $target"
Write-Host ""

$proceed = Prompt-Choice -Title "Start Build Now?" -Options @("Yes - Start CMake build", "No - Exit") -DefaultIndex 1
if ($proceed -ne 1) {
    Write-Host "$C_YELLOWBuild cancelled.$C_RESET"
    exit 0
}

# Determine Build Directory
$buildDir = if ($archDl -eq "ON") { "build-plugins" } else { "build" }

Write-Host ""
Write-Host "$C_CYAN>>> Configuring CMake in $buildDir...$C_RESET"

$cmakeArgs = @(
    "-B", $buildDir,
    "-DTRANSCRIBE_ARCH_DL=$archDl",
    "-DTRANSCRIBE_MODEL_SET=$modelSet"
)

if ($modelSet -eq "custom") {
    $cmakeArgs += "-DTRANSCRIBE_MODELS=$customModels"
}
if ($enableCuda -eq "ON") {
    $cmakeArgs += "-DTRANSCRIBE_CUDA=ON"
    if ($cudaArch -ne "auto") {
        $cmakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$cudaArch"
    }
} elseif ($enableVulkan -eq "ON") {
    $cmakeArgs += "-DTRANSCRIBE_VULKAN=ON"
}

Write-Host "> cmake $($cmakeArgs -join ' ')"
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "$C_RED[!] CMake configuration failed with code $LASTEXITCODE$C_RESET"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "$C_CYAN>>> Building target '$target' in $buildDir...$C_RESET"
$buildArgs = @("--build", $buildDir)
if ($target -ne "all") {
    $buildArgs += @("--target", $target)
}

Write-Host "> cmake $($buildArgs -join ' ')"
& cmake @buildArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "$C_RED[!] Build failed with code $LASTEXITCODE$C_RESET"
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "$C_GREEN$C_BOLD========================================================================$C_RESET"
Write-Host "$C_GREEN$C_BOLD                     BUILD COMPLETED SUCCESSFULLY!                      $C_RESET"
Write-Host "$C_GREEN$C_BOLD========================================================================$C_RESET"
Write-Host "Artifacts are located in: $C_CYAN$buildDir\bin\Debug\$C_RESET (or Release)"
if ($archDl -eq "ON") {
    Write-Host "Plugins generated:"
    Get-ChildItem -Path "$buildDir\bin\*\transcribe-arch-*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  - $($_.Name) ($([math]::Round($_.Length / 1MB, 2)) MB)"
    }
}
Write-Host ""
