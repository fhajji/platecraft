<#
.SYNOPSIS
    Installs the PlateCraft engine into this folder: portable Python, one
    environment per model, PyTorch with CUDA, ffmpeg (LGPL), model code and
    model weights.

.DESCRIPTION
    Everything lands inside engine\ and nothing is installed system-wide, so
    the whole PlateCraft folder can be copied to another Windows PC.
    Safe to re-run: finished steps are skipped unless -Force is given.

.PARAMETER Models
    Models to install. Comma-separated or an array.
      normalcrafter  Normals. Restricted commercial use (Stable Video Diffusion base).
      vda-s          Video Depth Anything Small. Apache 2.0.
      vda-l          Video Depth Anything Large. NON-COMMERCIAL (CC-BY-NC 4.0).
      marigold-n     Marigold Normals v1-1, per-frame normals. CreativeML Open RAIL++-M
                     (commercial use allowed; use-based restrictions apply).
      raft           RAFT optical flow (torchvision, BSD-3-Clause) for --temporal-filter.
      diffusion-renderer
                     Cosmos DiffusionRenderer (inverse, 7B): albedo, normal, depth, roughness,
                     metallic. Apache 2.0 code; NVIDIA Open Model License weights (commercial use
                     allowed). Needs Linux and >= 20 GB VRAM (~27 GB per 57-frame chunk). On
                     Windows this script only downloads its code and weights (~28 GB); build its
                     environment with engine/setup_engine.sh inside WSL2 or on a Linux PC.

.PARAMETER HfToken
    Hugging Face access token. Required for NormalCrafter, because its base
    model (stabilityai/stable-video-diffusion-img2vid-xt) is gated: accept its
    license on the model page first. Defaults to $env:HF_TOKEN.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File engine\setup_engine.ps1 -HfToken hf_xxx

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File engine\setup_engine.ps1 -Models vda-s -SkipWeights
#>
[CmdletBinding()]
param(
    [string[]]$Models = @('normalcrafter', 'vda-s', 'marigold-n', 'raft'),
    [string]$PythonVersion = '3.11',
    [string]$TorchVersion = '2.5.1',
    [string]$TorchvisionVersion = '0.20.1',
    [string]$XformersVersion = '0.0.28.post3',
    [string]$Cuda = 'cu124',
    [string]$HfToken = $env:HF_TOKEN,
    [switch]$SkipWeights,
    [switch]$SkipFfmpeg,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is very slow with the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# ---------------------------------------------------------------- settings --

# Pinned upstream code (commit SHAs checked 2026-10-29).
$Repos = @{
    'normalcrafter' = @{ Url = 'https://github.com/Binyr/NormalCrafter'; Sha = '75af9887a2cb14cd1ce3883c5773bc296565777c' }
    'vda'           = @{ Url = 'https://github.com/DepthAnything/Video-Depth-Anything'; Sha = '4f5ae23172ba60fd7bc11ef671cca678842c7072' }
    # Checked 2026-11-01 (main, 2025-10-02). Keep in sync with engine/setup_engine.sh.
    'diffusion-renderer' = @{ Url = 'https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer'; Sha = '0f3e2dc435032ecbad654c2fc2153df85384b138' }
}
# Hugging Face revisions of the DiffusionRenderer weights (checked 2026-11-01).
$DrInverseRev   = '41f63abccfb764b2516ff85b98af529c2662e2df'   # nvidia/Diffusion_Renderer_Inverse_Cosmos_7B
$DrTokenizerRev = 'b6af495317c76f287a4131e9299936b1533f5f9f'   # nvidia/Cosmos-Tokenize1-CV8x8x8-720p

$UvUrl     = 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'
# LGPL build: ffmpeg runs as a separate program, no GPL components bundled.
$FfmpegUrl = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl.zip'

# Packages per environment. torch/torchvision come from the PyTorch CUDA index first.
$CorePackages = @('numpy<2', 'opencv-python>=4.9,<5', 'pillow>=10', 'OpenEXR>=3.2,<4', 'tqdm', 'huggingface-hub>=0.30')
$EnvPackages = @{
    'normalcrafter' = @('numpy==1.26.4', 'diffusers==0.29.1', 'transformers==4.41.2', 'accelerate==0.30.1',
                        'huggingface-hub==0.30.1', 'tokenizers==0.19.1', 'safetensors>=0.4', 'einops>=0.8',
                        'opencv-python>=4.9,<5', 'pillow>=10', 'OpenEXR>=3.2,<4', 'tqdm')
    'vda'           = @('numpy<2', 'einops>=0.6', 'easydict', 'opencv-python>=4.9,<5', 'pillow>=10',
                        'OpenEXR>=3.2,<4', 'tqdm', 'huggingface-hub>=0.30')
}
# marigold-n uses the diffusers pipeline already in the normalcrafter env; raft uses torchvision.
$ModelEnv = @{ 'normalcrafter' = 'normalcrafter'; 'vda-s' = 'vda'; 'vda-l' = 'vda'; 'marigold-n' = 'normalcrafter'; 'raft' = 'vda'
               'diffusion-renderer' = 'cosmos' }
# Envs whose code repo has another name, and envs that can only be built on Linux (setup_engine.sh).
$EnvRepo = @{ 'cosmos' = 'diffusion-renderer' }
$LinuxOnlyEnvs = @('cosmos')

# ------------------------------------------------------------------- paths --

$EngineDir = $PSScriptRoot
$BinDir    = Join-Path $EngineDir 'bin'
$EnvsDir   = Join-Path $EngineDir 'envs'
$ReposDir  = Join-Path $EngineDir 'repos'
$ModelsDir = Join-Path $EngineDir 'models'
$TmpDir    = Join-Path $EngineDir 'cache\downloads'
$Uv        = Join-Path $BinDir 'uv.exe'

# Keep uv's Python installs, caches and Hugging Face files inside engine\.
$env:UV_PYTHON_INSTALL_DIR = Join-Path $EngineDir 'python'
$env:UV_CACHE_DIR          = Join-Path $EngineDir 'cache\uv'
$env:UV_PYTHON_PREFERENCE  = 'only-managed'
$env:UV_LINK_MODE          = 'hardlink'
$env:HF_HOME               = Join-Path $ModelsDir 'hf'
$env:TORCH_HOME            = Join-Path $ModelsDir 'torch'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
if ($HfToken) { $env:HF_TOKEN = $HfToken }

# --------------------------------------------------------------- helpers --

function Write-Step([string]$Text) { Write-Host "`n==> $Text" -ForegroundColor Cyan }
function Write-Ok([string]$Text)   { Write-Host "    $Text" -ForegroundColor Green }
function Write-Warn2([string]$Text){ Write-Host "    WARNING: $Text" -ForegroundColor Yellow }

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

function Get-File([string]$Url, [string]$Dest) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Dest) | Out-Null
    Write-Host "    downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
}

function Expand-Zip([string]$Zip, [string]$Dest) {
    if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
    Expand-Archive -Path $Zip -DestinationPath $Dest -Force
}

function Get-EnvPython([string]$Name) { Join-Path $EnvsDir "$Name\Scripts\python.exe" }

# A venv records its base Python as an absolute path (pyvenv.cfg "home", and inside uv's
# Scripts\python.exe launcher). After the PlateCraft folder is moved or renamed, rebuild the
# venv around its installed packages; console-script .exe files still hold the old path, but
# PlateCraft only runs "python -m".
function Repair-PyEnv([string]$EnvDir) {
    $cfg = Join-Path $EnvDir 'pyvenv.cfg'
    if (-not (Test-Path $cfg)) { return }
    $homeLine = Get-Content $cfg | Where-Object { $_ -match '^\s*home\s*=' } | Select-Object -First 1
    if (-not $homeLine) { return }
    $oldHome = ($homeLine -split '=', 2)[1].Trim()
    if ($oldHome.StartsWith($env:UV_PYTHON_INSTALL_DIR, [StringComparison]::OrdinalIgnoreCase)) { return }
    Invoke-Checked $Uv @('venv', $EnvDir, '--python', $PythonVersion, '--seed', '--allow-existing')
    Write-Ok "repaired moved environment $(Split-Path $EnvDir -Leaf) (was $oldHome)"
}

function New-PyEnv {
    param([string]$Name, [string[]]$Packages, [switch]$WithTorch)
    $envDir = Join-Path $EnvsDir $Name
    $py = Get-EnvPython $Name
    if ($Force -and (Test-Path $envDir)) { Remove-Item -Recurse -Force $envDir }
    Repair-PyEnv $envDir
    if (-not (Test-Path $py)) {
        Invoke-Checked $Uv @('venv', $envDir, '--python', $PythonVersion, '--seed')
    }
    if ($WithTorch) {
        $index = "https://download.pytorch.org/whl/$Cuda"
        Invoke-Checked $Uv @('pip', 'install', '--python', $py, '--index-url', $index,
                             "torch==$TorchVersion", "torchvision==$TorchvisionVersion")
        # xformers is optional: both models fall back to PyTorch attention without it.
        & $Uv pip install --python $py --index-url $index "xformers==$XformersVersion"
        if ($LASTEXITCODE -ne 0) { Write-Warn2 "xformers not installed for '$Name' (optional, slightly more VRAM use)." }
    }
    # torch is already satisfied, so these resolve from PyPI without replacing the CUDA build.
    Invoke-Checked $Uv (@('pip', 'install', '--python', $py) + $Packages)
    Write-Ok "environment '$Name' ready"
}

function Get-Repo([string]$Key) {
    $info = $Repos[$Key]
    $dest = Join-Path $ReposDir $Key
    $marker = Join-Path $dest '.platecraft-sha'
    if (-not $Force -and (Test-Path $marker) -and ((Get-Content $marker -Raw).Trim() -eq $info.Sha)) {
        Write-Ok "$Key code already at $($info.Sha.Substring(0,7))"; return
    }
    $zip = Join-Path $TmpDir "$Key.zip"
    Get-File "$($info.Url)/archive/$($info.Sha).zip" $zip
    $staging = Join-Path $TmpDir "$Key-src"
    Expand-Zip $zip $staging
    $inner = Get-ChildItem $staging -Directory | Select-Object -First 1
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    New-Item -ItemType Directory -Force -Path $ReposDir | Out-Null
    Move-Item $inner.FullName $dest
    Set-Content -Path $marker -Value $info.Sha
    Remove-Item -Recurse -Force $staging, $zip
    Write-Ok "$Key code at $($info.Sha.Substring(0,7))"
}

# Python snippet run inside a model env to fetch weights with huggingface_hub.
function Invoke-HfDownload([string]$EnvName, [string]$Code) {
    $py = Get-EnvPython $EnvName
    $script = Join-Path $TmpDir "hf_$EnvName.py"
    Set-Content -Path $script -Value $Code -Encoding UTF8
    & $py $script | Out-Host   # keep Python's output out of the return value
    $code = $LASTEXITCODE
    Remove-Item $script -ErrorAction SilentlyContinue
    return $code
}

# ------------------------------------------------------------------ checks --

$Models = @($Models | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim().ToLower() } | Where-Object { $_ })
foreach ($m in $Models) {
    if (-not $ModelEnv.ContainsKey($m)) { throw "Unknown model '$m'. Choose from: $($ModelEnv.Keys -join ', ')" }
}
if ([Environment]::Is64BitOperatingSystem -eq $false) { throw 'A 64-bit Windows is required.' }

Write-Host 'PlateCraft engine setup' -ForegroundColor White
Write-Host "  engine folder : $EngineDir"
Write-Host "  models        : $($Models -join ', ')"
Write-Host "  python/torch  : $PythonVersion / $TorchVersion+$Cuda"

if ($Models -contains 'vda-l') {
    Write-Warn2 'vda-l (Video Depth Anything Large) is CC-BY-NC 4.0: NON-COMMERCIAL use only.'
}
if ($Models -contains 'diffusion-renderer') {
    Write-Warn2 'diffusion-renderer: NVIDIA Open Model License. Commercial use is allowed; credit "Built on NVIDIA Cosmos"'
    Write-Warn2 'and do not bypass safety guardrails (that terminates the license). Needs Linux and >= 20 GB VRAM.'
}
if ($Models -contains 'normalcrafter') {
    Write-Warn2 'NormalCrafter is built on Stable Video Diffusion (Stability AI Community License):'
    Write-Warn2 'commercial use is free only under US$1M annual revenue; above that you need an Enterprise licence.'
}

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $gpu = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) | Select-Object -First 1
    Write-Host "  GPU           : $gpu"
} else {
    Write-Warn2 'nvidia-smi not found. Install a current NVIDIA driver (550 or newer for CUDA 12.4).'
}

New-Item -ItemType Directory -Force -Path $BinDir, $EnvsDir, $ReposDir, $ModelsDir, $TmpDir | Out-Null

# --------------------------------------------------------------------- uv --

Write-Step 'uv (Python and package manager)'
if ($Force -or -not (Test-Path $Uv)) {
    $zip = Join-Path $TmpDir 'uv.zip'
    Get-File $UvUrl $zip
    $staging = Join-Path $TmpDir 'uv'
    Expand-Zip $zip $staging
    Get-ChildItem $staging -Recurse -Filter 'uv*.exe' | ForEach-Object { Copy-Item $_.FullName $BinDir -Force }
    Remove-Item -Recurse -Force $staging, $zip
}
Write-Ok (& $Uv --version)

Write-Step "Portable Python $PythonVersion"
# --no-bin: keep python3.x.exe out of ~\.local\bin (nothing system-wide).
Invoke-Checked $Uv @('python', 'install', $PythonVersion, '--no-bin')

# ----------------------------------------------------------------- ffmpeg --

if (-not $SkipFfmpeg) {
    Write-Step 'ffmpeg (LGPL build)'
    $ff = Join-Path $BinDir 'ffmpeg.exe'
    if ($Force -or -not (Test-Path $ff)) {
        $zip = Join-Path $TmpDir 'ffmpeg.zip'
        Get-File $FfmpegUrl $zip
        $staging = Join-Path $TmpDir 'ffmpeg'
        Expand-Zip $zip $staging
        Get-ChildItem $staging -Recurse -Include 'ffmpeg.exe', 'ffprobe.exe' | ForEach-Object { Copy-Item $_.FullName $BinDir -Force }
        Get-ChildItem $staging -Recurse -Filter 'LICENSE*' | Select-Object -First 1 | ForEach-Object { Copy-Item $_.FullName (Join-Path $BinDir 'FFMPEG-LICENSE.txt') -Force }
        Remove-Item -Recurse -Force $staging, $zip
    }
    Write-Ok ((& $ff -hide_banner -version) | Select-Object -First 1)
}

# ----------------------------------------------------------- environments --

Write-Step 'Core environment (PlateCraft CLI)'
New-PyEnv -Name 'core' -Packages $CorePackages

$envsNeeded = @($Models | ForEach-Object { $ModelEnv[$_] } | Select-Object -Unique)
foreach ($e in $envsNeeded) {
    $repo = if ($EnvRepo.ContainsKey($e)) { $EnvRepo[$e] } else { $e }
    if ($LinuxOnlyEnvs -contains $e) {
        Write-Step "Model environment '$e' (Linux only)"
        Get-Repo $repo
        Write-Warn2 "'$e' needs Linux (transformer-engine). Code and weights are downloaded here; build the"
        Write-Warn2 'environment inside WSL2 (bash engine/setup_engine.sh from this folder) or on a Linux PC.'
        continue
    }
    Write-Step "Model environment '$e' (PyTorch $TorchVersion + $Cuda)"
    New-PyEnv -Name $e -Packages $EnvPackages[$e] -WithTorch
    Get-Repo $repo
}

# ---------------------------------------------------------------- weights --

$weightsFailed = @()
if (-not $SkipWeights) {
    foreach ($m in $Models) {
        Write-Step "Weights: $m"
        $target = (Join-Path $ModelsDir $m) -replace '\\', '/'
        switch ($m) {
            'normalcrafter' {
                $code = @"
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
try:
    snapshot_download('Yanrui95/NormalCrafter', local_dir='$target/normalcrafter')
    # Only the parts of SVD that NormalCrafter does not replace (it brings its own unet and vae).
    snapshot_download('stabilityai/stable-video-diffusion-img2vid-xt', local_dir='$target/svd-xt',
        allow_patterns=['model_index.json', 'feature_extractor/*', 'scheduler/*',
                        'image_encoder/config.json', 'image_encoder/*.fp16.safetensors'])
except (GatedRepoError, RepositoryNotFoundError) as e:
    print('GATED: stabilityai/stable-video-diffusion-img2vid-xt needs access.', file=sys.stderr)
    print('  1. Log in at https://huggingface.co and accept the license on', file=sys.stderr)
    print('     https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt', file=sys.stderr)
    print('  2. Create a read token at https://huggingface.co/settings/tokens', file=sys.stderr)
    print('  3. Re-run: setup_engine.ps1 -HfToken hf_xxx', file=sys.stderr)
    sys.exit(3)
"@
                $rc = Invoke-HfDownload 'normalcrafter' $code
            }
            'vda-s' {
                $rc = Invoke-HfDownload 'vda' "from huggingface_hub import hf_hub_download`nhf_hub_download('depth-anything/Video-Depth-Anything-Small', 'video_depth_anything_vits.pth', local_dir='$target')"
            }
            'vda-l' {
                $rc = Invoke-HfDownload 'vda' "from huggingface_hub import hf_hub_download`nhf_hub_download('depth-anything/Video-Depth-Anything-Large', 'video_depth_anything_vitl.pth', local_dir='$target')"
            }
            'marigold-n' {
                # fp16 weights only (~2.6 GB).
                $rc = Invoke-HfDownload 'normalcrafter' "from huggingface_hub import snapshot_download`nsnapshot_download('prs-eth/marigold-normals-v1-1', local_dir='$target', allow_patterns=['model_index.json', 'scheduler/*', 'tokenizer/*', '*/config.json', '*.fp16.safetensors'])"
            }
            'diffusion-renderer' {
                # Only what inverse rendering loads (~28 GB): the 7B DiT and the tokenizer's encoder/decoder.
                $rc = Invoke-HfDownload 'core' @"
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Cosmos-Tokenize1-CV8x8x8-720p', revision='$DrTokenizerRev', local_dir='$target/Cosmos-Tokenize1-CV8x8x8-720p',
    allow_patterns=['README.md', 'config.json', 'encoder.jit', 'decoder.jit', 'mean_std.pt', 'image_mean_std.pt'])
snapshot_download('nvidia/Diffusion_Renderer_Inverse_Cosmos_7B', revision='$DrInverseRev', local_dir='$target/Diffusion_Renderer_Inverse_Cosmos_7B',
    allow_patterns=['README.md', 'config.json', 'model.pt'])
"@
            }
            'raft' {
                # torchvision downloads into TORCH_HOME (engine\models\torch).
                $rc = Invoke-HfDownload 'vda' "from torchvision.models.optical_flow import raft_large, Raft_Large_Weights`nraft_large(weights=Raft_Large_Weights.DEFAULT)"
            }
        }
        if ($rc -eq 0) { Write-Ok "$m weights ready" } else { $weightsFailed += $m; Write-Warn2 "$m weights not downloaded (exit $rc)." }
    }
}

# ------------------------------------------------------------- manifest --

# Merge with an earlier run, so installing one more model keeps the others listed.
$EngineJson = Join-Path $EngineDir 'engine.json'
$prev = if ((Test-Path $EngineJson) -and -not $Force) { Get-Content $EngineJson -Raw | ConvertFrom-Json } else { $null }
$allModels = @(@($prev.models) + $Models | Where-Object { $_ } | Select-Object -Unique)
$builtEnvs = @($envsNeeded | Where-Object { $LinuxOnlyEnvs -notcontains $_ })
$allEnvs = @(@('core') + @($prev.envs) + $builtEnvs | Where-Object { $_ } | Select-Object -Unique)
$engineInfo = [ordered]@{
    installed_at   = (Get-Date).ToString('s')
    python         = $PythonVersion
    torch          = "$TorchVersion+$Cuda"
    models         = $allModels
    envs           = $allEnvs
    repos          = @{}
    weights_failed = $weightsFailed
}
foreach ($e in @($allEnvs + $envsNeeded | Select-Object -Unique)) {
    $repo = if ($EnvRepo.ContainsKey($e)) { $EnvRepo[$e] } else { $e }
    if ($Repos.ContainsKey($repo)) { $engineInfo.repos[$repo] = $Repos[$repo].Sha }
}
$engineInfo | ConvertTo-Json -Depth 4 | Set-Content -Path $EngineJson -Encoding UTF8

Write-Step 'Done'
if ($weightsFailed.Count -gt 0) {
    Write-Warn2 "Missing weights: $($weightsFailed -join ', '). Fix the issue above and re-run this script."
}
Write-Host '    Next: run  platecraft doctor  from the PlateCraft folder.'
