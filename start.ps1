param(
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173,
  [string]$HostAddress = "127.0.0.1",
  [switch]$AllowFallback,
  [switch]$DisableRealGuard
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

Stop-PortProcess -Port $BackendPort
Stop-PortProcess -Port $FrontendPort

$python = Resolve-CommandPath -Candidates @("python", "python3", "py") -HelpText "Python is required to start the backend."
$pnpm = Resolve-CommandPath -Candidates @("pnpm.cmd", "pnpm") -HelpText "pnpm is required to start the frontend."

$backendOut = Join-Path $LogDir "backend.out.log"
$backendErr = Join-Path $LogDir "backend.err.log"
$frontendOut = Join-Path $LogDir "frontend.out.log"
$frontendErr = Join-Path $LogDir "frontend.err.log"

Write-Host "Starting backend: http://localhost:$BackendPort"
$backendArgs = if ((Split-Path -Leaf $python) -eq "py.exe") { @("-3", "api_server.py") } else { @("api_server.py") }
$previousBackendPort = $env:SEME2E_API_PORT
$previousRealGuard = $env:SEME2E_API_REAL_GUARD
$previousAllowFallback = $env:SEME2E_API_ALLOW_FALLBACK
$env:SEME2E_API_PORT = [string]$BackendPort
$env:SEME2E_API_REAL_GUARD = if ($DisableRealGuard) { "0" } else { "1" }
$env:SEME2E_API_ALLOW_FALLBACK = if ($AllowFallback) { "1" } else { "0" }
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
if ($null -eq $previousAllowFallback) {
  Remove-Item Env:\SEME2E_API_ALLOW_FALLBACK -ErrorAction SilentlyContinue
} else {
  $env:SEME2E_API_ALLOW_FALLBACK = $previousAllowFallback
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
