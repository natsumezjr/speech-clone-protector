param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173,
  [string]$HostAddress = "127.0.0.1",
  [switch]$DisableRealGuard,
  [switch]$DisableAsr,
  [string]$AsrModel = "openai-whisper:tiny"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $Root "backend\SemE2E"
$FrontendDir = Join-Path $Root "fro"
$LogDir = Join-Path $Root "logs"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Stop-PortProcess {
  param([int]$Port)

  $connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
  $processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique

  foreach ($processId in $processIds) {
    if (-not $processId -or $processId -eq $PID) {
      continue
    }

    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) {
      Write-Host "Killing process $processId ($($process.ProcessName)) on port $Port"
      Stop-Process -Id $processId -Force
    }
  }
}

function Resolve-CommandPath {
  param(
    [string[]]$Candidates,
    [string]$HelpText
  )

  foreach ($candidate in $Candidates) {
    $command = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($command) {
      return $command.Source
    }
  }

  throw $HelpText
}

$python = Resolve-CommandPath -Candidates @("python", "python3", "py") -HelpText "Python is required to start the backend."
$pnpm = Resolve-CommandPath -Candidates @("pnpm.cmd", "pnpm") -HelpText "pnpm is required to start the frontend."

$pythonLauncherArgs = @()
if ((Split-Path -Leaf $python) -eq "py.exe") {
  $pythonLauncherArgs = @("-3.11")
}
$pythonVersion = & $python @pythonLauncherArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $pythonVersion.Trim() -ne "3.11") {
  $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
  if (-not $pyLauncher) {
    throw "Python 3.11 is required by Coqui TTS 0.22.0. Install Python 3.11 or make it the active python command."
  }
  $python = $pyLauncher.Source
  $pythonLauncherArgs = @("-3.11")
  $pythonVersion = & $python @pythonLauncherArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
  if ($LASTEXITCODE -ne 0 -or $pythonVersion.Trim() -ne "3.11") {
    throw "Python 3.11 is required by Coqui TTS 0.22.0, but the 3.11 launcher is unavailable."
  }
}

$audioRuntimeJson = & $python @pythonLauncherArgs -c "import json, sys; sys.path.insert(0, sys.argv[1]); from audio_preprocess import audio_preprocess_capabilities; capabilities = audio_preprocess_capabilities(); print(json.dumps(capabilities)); raise SystemExit(0 if capabilities['recordingSupported'] else 2)" $BackendDir
if ($LASTEXITCODE -ne 0) {
  throw "Browser recording requires an FFmpeg decoder. Install Python 3.11 dependencies with: python -m pip install -r backend/SemE2E/requirements.txt, or set SEME2E_FFMPEG_PATH. Diagnostics: $audioRuntimeJson"
}
$audioRuntime = $audioRuntimeJson | ConvertFrom-Json
Write-Host "Recording decoder: $($audioRuntime.decoder.path) ($($audioRuntime.decoder.source))"

Stop-PortProcess -Port $BackendPort
Stop-PortProcess -Port $FrontendPort

$backendOut = Join-Path $LogDir "backend.out.log"
$backendErr = Join-Path $LogDir "backend.err.log"
$frontendOut = Join-Path $LogDir "frontend.out.log"
$frontendErr = Join-Path $LogDir "frontend.err.log"

Write-Host "Starting backend: http://localhost:$BackendPort"
$backendArgs = @($pythonLauncherArgs) + @("api_server.py")
$previousBackendPort = $env:SEME2E_API_PORT
$previousRealGuard = $env:SEME2E_API_REAL_GUARD
$previousEnableAsr = $env:SEME2E_ENABLE_ASR
$previousAsrModel = $env:SEME2E_ASR_MODEL
$env:SEME2E_API_PORT = [string]$BackendPort
$env:SEME2E_API_REAL_GUARD = if ($DisableRealGuard) { "0" } else { "1" }
$env:SEME2E_ENABLE_ASR = if ($DisableAsr) { "0" } else { "1" }
$env:SEME2E_ASR_MODEL = $AsrModel
$backend = Start-Process -FilePath $python -ArgumentList $backendArgs -WorkingDirectory $BackendDir -RedirectStandardOutput $backendOut -RedirectStandardError $backendErr -PassThru -WindowStyle Hidden
if ($null -eq $previousBackendPort) {
  Remove-Item Env:\SEME2E_API_PORT -ErrorAction SilentlyContinue
} else {
  $env:SEME2E_API_PORT = $previousBackendPort
}
if ($null -eq $previousRealGuard) {
  Remove-Item Env:\SEME2E_API_REAL_GUARD -ErrorAction SilentlyContinue
} else {
  $env:SEME2E_API_REAL_GUARD = $previousRealGuard
}
if ($null -eq $previousEnableAsr) {
  Remove-Item Env:\SEME2E_ENABLE_ASR -ErrorAction SilentlyContinue
} else {
  $env:SEME2E_ENABLE_ASR = $previousEnableAsr
}
if ($null -eq $previousAsrModel) {
  Remove-Item Env:\SEME2E_ASR_MODEL -ErrorAction SilentlyContinue
} else {
  $env:SEME2E_ASR_MODEL = $previousAsrModel
}

Write-Host "Starting frontend: http://localhost:$FrontendPort"
$frontendCommand = "`"$pnpm`" run dev -- --host $HostAddress --port $FrontendPort"
$frontend = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $frontendCommand) -WorkingDirectory $FrontendDir -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr -PassThru -WindowStyle Hidden

Write-Host ""
Write-Host "Backend PID:  $($backend.Id)"
Write-Host "Frontend PID: $($frontend.Id)"
Write-Host "Logs:"
Write-Host "  $backendOut"
Write-Host "  $backendErr"
Write-Host "  $frontendOut"
Write-Host "  $frontendErr"
Write-Host ""
Write-Host "Press Ctrl+C to stop both services."

try {
  while (-not $backend.HasExited -and -not $frontend.HasExited) {
    Start-Sleep -Seconds 1
    $backend.Refresh()
    $frontend.Refresh()
  }

  if ($backend.HasExited) {
    Write-Host "Backend exited with code $($backend.ExitCode). See $backendErr"
  }

  if ($frontend.HasExited) {
    Write-Host "Frontend exited with code $($frontend.ExitCode). See $frontendErr"
  }
} finally {
  foreach ($process in @($backend, $frontend)) {
    if ($process -and -not $process.HasExited) {
      Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    }
  }
}
