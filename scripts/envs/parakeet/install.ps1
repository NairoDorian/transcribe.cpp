# Install/refresh the parakeet reference env and assert the PyTorch build.
#
#   powershell -File scripts/envs/parakeet/install.ps1
#
# The lock pins win32 torch to the nightly CUDA 13.4 index
# (https://download.pytorch.org/whl/nightly/cu134), so a sync is the install
# step; this script exists to fail loudly if the env ever ends up on a CPU or
# older-CUDA torch instead. The python check below deliberately avoids double
# quotes - PowerShell 5.1 strips them from native command arguments.

$ErrorActionPreference = "Stop"
$proj = $PSScriptRoot

uv sync --project $proj
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run --project $proj python -c @'
import platform
import sys

import torch

version = torch.__version__
cuda = torch.version.cuda
print('torch ' + version + ' (cuda ' + str(cuda) + ')')

if platform.system() == 'Windows':
    if not version.endswith('+cu134') or cuda != '13.4':
        sys.stderr.write('ERROR: expected nightly torch +cu134 on win32\n')
        sys.exit(1)
    if not torch.cuda.is_available():
        sys.stderr.write('ERROR: CUDA not available\n')
        sys.exit(1)
    print('device: ' + torch.cuda.get_device_name(0))
'@
exit $LASTEXITCODE
