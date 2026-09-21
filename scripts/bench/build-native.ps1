param(
    [string]$CudaArchitecture = '89-real',
    [int]$Jobs = 6,
    [switch]$CudaGraphs,
    [switch]$CpuAvxVnni,
    [string]$Target = 'transcribe-cli'
)
$ErrorActionPreference = 'Stop'
$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$buildRoot = Join-Path $sourceRoot 'build/bench-native'
$installRoot = Join-Path $buildRoot 'install'

# CPU and CUDA come from this checkout's own GGML. Never borrow a backend DLL
# from another GGML version. All outputs stay under this checkout's build/.
#
# cl.exe merely being on PATH does not mean the toolchain is usable: a bare
# PATH entry, or a shell inherited without the developer environment, leaves
# INCLUDE/LIB unset and the build then dies with C1083 on <stdbool.h>. Only
# skip the dev shell when the SDK/CRT headers are genuinely visible.
$toolchainReady = [bool](Get-Command cl.exe -ErrorAction SilentlyContinue) -and
                  [bool]($env:INCLUDE -split ';' | Where-Object { $_ -match 'Windows Kits|MSVC' })
if (-not $toolchainReady) {
    # vswhere is the documented locator, but some layouts (VS 18 preview/dev)
    # ship without it, so fall back to the standard install roots. DevShell.dll
    # under Common7/Tools is the only thing this actually needs.
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
$graphs = if ($CudaGraphs) { 'ON' } else { 'OFF' }
$vnni = if ($CpuAvxVnni) { 'ON' } else { 'OFF' }
$configure = @('-S', $sourceRoot, '-B', $buildRoot, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DTRANSCRIBE_BUILD_SHARED=ON',
    '-DTRANSCRIBE_GGML_BACKEND_DL=ON', '-DTRANSCRIBE_CUDA=ON',
    '-DTRANSCRIBE_ARCH_DL=ON',
    '-DTRANSCRIBE_INSTALL=ON', '-DTRANSCRIBE_BUILD_TESTS=OFF',
    '-DGGML_NATIVE=OFF', '-DGGML_CPU_ALL_VARIANTS=OFF',
    '-DGGML_AVX=ON', '-DGGML_AVX2=ON', '-DGGML_FMA=ON', '-DGGML_F16C=ON',
    "-DGGML_AVX_VNNI=$vnni",
    '-DGGML_CCACHE=OFF', '-DCMAKE_CUDA_COMPILER_LAUNCHER=',
    '-DCMAKE_C_COMPILER_LAUNCHER=', '-DCMAKE_CXX_COMPILER_LAUNCHER=',
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitecture", "-DGGML_CUDA_GRAPHS=$graphs")
& cmake @configure
if ($LASTEXITCODE) { throw 'CMake configure failed' }
& cmake --build $buildRoot --target $Target --parallel $Jobs
if ($LASTEXITCODE) { throw 'Native build failed' }
if ($Target -ne 'transcribe-cli') { return }
& cmake --install $buildRoot --prefix $installRoot
if ($LASTEXITCODE) { throw 'Native install failed' }
Write-Output "Benchmark native install: $installRoot"
