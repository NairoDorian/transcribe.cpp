param(
    [string]$CudaArchitecture = '89-real',
    [int]$Jobs = 6,
    [switch]$CudaGraphs
)
$ErrorActionPreference = 'Stop'
$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$buildRoot = Join-Path $sourceRoot 'build/bench-native'
$installRoot = Join-Path $buildRoot 'install'

# CPU and CUDA come from this checkout's own GGML. Never borrow a backend DLL
# from another GGML version. All outputs stay under this checkout's build/.
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsRoot) { throw 'An x64 Visual Studio C++ toolchain is required' }
    Import-Module (Join-Path $vsRoot 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vsRoot -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
}
$graphs = if ($CudaGraphs) { 'ON' } else { 'OFF' }
$configure = @('-S', $sourceRoot, '-B', $buildRoot, '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DTRANSCRIBE_BUILD_SHARED=ON',
    '-DTRANSCRIBE_GGML_BACKEND_DL=ON', '-DTRANSCRIBE_CUDA=ON',
    '-DTRANSCRIBE_INSTALL=ON', '-DTRANSCRIBE_BUILD_TESTS=OFF',
    '-DGGML_NATIVE=ON', '-DGGML_CPU_ALL_VARIANTS=OFF',
    '-DGGML_CCACHE=OFF', '-DCMAKE_CUDA_COMPILER_LAUNCHER=',
    '-DCMAKE_C_COMPILER_LAUNCHER=', '-DCMAKE_CXX_COMPILER_LAUNCHER=',
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitecture", "-DGGML_CUDA_GRAPHS=$graphs")
& cmake @configure
if ($LASTEXITCODE) { throw 'CMake configure failed' }
& cmake --build $buildRoot --target transcribe-cli --parallel $Jobs
if ($LASTEXITCODE) { throw 'Native build failed' }
& cmake --install $buildRoot --prefix $installRoot
if ($LASTEXITCODE) { throw 'Native install failed' }
Write-Output "Benchmark native install: $installRoot"
