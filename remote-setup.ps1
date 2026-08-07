param(
  [string]$SshHost = "pro",
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173,
  [string]$HostAddress = "localhost",
  [switch]$SkipFrontend,
  [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$FrontendDir = Join-Path $Root "fro"
$BackendDir = Join-Path $Root "backend\SemE2E"
$LogDir = Join-Path $Root "logs"
$StatePath = Join-Path $LogDir "remote-setup.json"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

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

function Add-ProcessId {
  param(
    [System.Collections.Generic.HashSet[int]]$Target,
    [object]$Value
  )

  if ($null -eq $Value) {
    return
  }

  $processId = 0
  if ([int]::TryParse([string]$Value, [ref]$processId) -and $processId -gt 0 -and $processId -ne $PID) {
    [void]$Target.Add($processId)
  }
}

function Stop-LocalProjectProcesses {
  $processIds = [System.Collections.Generic.HashSet[int]]::new()

  foreach ($port in @($BackendPort, $FrontendPort)) {
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    foreach ($connection in $connections) {
      Add-ProcessId -Target $processIds -Value $connection.OwningProcess
    }
  }

  $projectProcessNames = @("node.exe", "python.exe", "pythonw.exe", "cmd.exe")
  $forwardSpec = "${BackendPort}:127.0.0.1:${BackendPort}"
  $processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue
  foreach ($process in $processes) {
    if (-not $process.CommandLine -or $process.ProcessId -eq $PID) {
      continue
    }

    $isProjectProcess = $projectProcessNames -contains $process.Name -and (
      $process.CommandLine.IndexOf($Root, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $process.CommandLine.IndexOf($FrontendDir, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
      $process.CommandLine.IndexOf($BackendDir, [StringComparison]::OrdinalIgnoreCase) -ge 0
    )
    $isTunnel = $process.Name -eq "ssh.exe" -and (
      $process.CommandLine.IndexOf($forwardSpec, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
      $process.CommandLine.IndexOf($SshHost, [StringComparison]::OrdinalIgnoreCase) -ge 0
    )

    if ($isProjectProcess -or $isTunnel) {
      Add-ProcessId -Target $processIds -Value $process.ProcessId
    }
  }

  foreach ($processId in $processIds) {
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if (-not $process) {
      continue
    }

    Write-Host "Stopping local process $processId ($($process.ProcessName))"
    & taskkill.exe /PID $processId /T /F 2>$null | Out-Null
  }

  Start-Sleep -Milliseconds 500
}

function Get-RemoteHealth {
  param([string]$SshPath)

  $healthCommand = "curl --fail --silent --max-time 10 http://127.0.0.1:${BackendPort}/api/health 2>/dev/null"
  $previousErrorActionPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    $raw = (& $SshPath -o BatchMode=yes -o ConnectTimeout=15 $SshHost $healthCommand 2>$null | Out-String).Trim()
    $sshExitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $previousErrorActionPreference
  }

  if ($sshExitCode -ne 0 -or -not $raw) {
    return $null
  }

  try {
    $health = $raw | ConvertFrom-Json
  } catch {
    return $null
  }

  if ($health.ok -ne $true) {
    return $null
  }

  return $health
}

function Start-RemoteBackend {
  param([string]$SshPath)

  $scriptTemplate = @'
set -eu
project="$HOME/VoiceSheild"
backend="$project/backend/SemE2E"
venv="$project/.venv"
deploy_dir="$project/.runtime/deploy"
pidfile="$deploy_dir/backend.pid"
outlog="$deploy_dir/backend.out.log"
errlog="$deploy_dir/backend.err.log"
port=__BACKEND_PORT__

[ "$(readlink -f "$project")" = "$HOME/VoiceSheild" ]
[ -x "$venv/bin/python" ]
mkdir -p "$deploy_dir"

if [ -f "$pidfile" ]; then
  old_pid="$(cat "$pidfile")"
  if kill -0 "$old_pid" 2>/dev/null; then
    expected="$venv/bin/python -m uvicorn api_server:app --host 127.0.0.1 --port $port"
    actual="$(tr '\0' ' ' < "/proc/$old_pid/cmdline" | sed 's/ $//')"
    if [ "$actual" != "$expected" ]; then
      printf 'PID file points to an unexpected process: %s\n' "$actual" >&2
      exit 20
    fi
    kill "$old_pid"
    for unused in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$old_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$old_pid" 2>/dev/null; then
      printf 'Remote backend did not stop cleanly\n' >&2
      exit 21
    fi
  fi
fi

if ss -ltn 2>/dev/null | grep -q ":$port "; then
  printf 'Remote port %s is occupied by an unmanaged process\n' "$port" >&2
  exit 22
fi

cd "$backend"
SEME2E_RUNTIME_DIR="$project/seme2e-runtime" \
SEME2E_API_REAL_GUARD=1 \
SEME2E_ENABLE_ASR=1 \
SEME2E_API_DEVICE='cuda:0' \
PYTHONUNBUFFERED=1 \
nohup "$venv/bin/python" -m uvicorn api_server:app --host 127.0.0.1 --port "$port" > "$outlog" 2> "$errlog" < /dev/null &
new_pid=$!
printf '%s\n' "$new_pid" > "$pidfile"

for unused in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
  if curl --fail --silent --max-time 3 "http://127.0.0.1:$port/api/health" >/dev/null 2>&1; then
    printf 'Remote backend started: PID %s\n' "$new_pid"
    exit 0
  fi
  kill -0 "$new_pid" 2>/dev/null || {
    tail -80 "$errlog" >&2 || true
    exit 23
  }
  sleep 1
done

printf 'Remote backend health check timed out\n' >&2
tail -80 "$errlog" >&2 || true
exit 24
# end
'@

  $remoteScript = $scriptTemplate.Replace("__BACKEND_PORT__", [string]$BackendPort).Replace("`r", "")
  ($remoteScript + "`n") | & $SshPath -o BatchMode=yes $SshHost bash -s
  if ($LASTEXITCODE -ne 0) {
    throw "Remote backend failed to start."
  }
}

function Wait-HttpOk {
  param(
    [string]$Url,
    [int]$Attempts = 30
  )

  for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
    try {
      $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
      if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) {
        return $true
      }
    } catch {
      Start-Sleep -Seconds 1
    }
  }

  return $false
}

$ssh = Resolve-CommandPath -Candidates @("ssh.exe", "ssh") -HelpText "OpenSSH is required. Verify that 'ssh pro' works first."
$pnpm = $null
if (-not $SkipFrontend) {
  $pnpm = Resolve-CommandPath -Candidates @("pnpm.cmd", "pnpm") -HelpText "pnpm is required to start the frontend."
}

Write-Host "Stopping local project frontend/backend/tunnel processes..."
Stop-LocalProjectProcesses

Write-Host "Checking remote API through ssh $SshHost..."
$remoteHealth = Get-RemoteHealth -SshPath $ssh
$remoteBackendReused = $null -ne $remoteHealth
if ($remoteHealth) {
  Write-Host "Remote API is healthy; keeping the existing remote backend process."
} else {
  Write-Host "Remote API is unavailable; starting the remote backend..."
  Start-RemoteBackend -SshPath $ssh
  $remoteHealth = Get-RemoteHealth -SshPath $ssh
  if (-not $remoteHealth) {
    throw "Remote API is still unavailable after startup."
  }
}

$tunnelOut = Join-Path $LogDir "remote-tunnel.out.log"
$tunnelErr = Join-Path $LogDir "remote-tunnel.err.log"
$forwardSpec = "${BackendPort}:127.0.0.1:${BackendPort}"
$tunnelArgs = @(
  "-N",
  "-o", "BatchMode=yes",
  "-o", "ExitOnForwardFailure=yes",
  "-o", "ServerAliveInterval=30",
  "-o", "ServerAliveCountMax=3",
  "-L", $forwardSpec,
  $SshHost
)

Write-Host "Starting SSH tunnel: 127.0.0.1:$BackendPort -> remote 127.0.0.1:$BackendPort"
$tunnel = Start-Process -FilePath $ssh -ArgumentList $tunnelArgs -RedirectStandardOutput $tunnelOut -RedirectStandardError $tunnelErr -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 2
$tunnel.Refresh()
if ($tunnel.HasExited) {
  $message = Get-Content $tunnelErr -Raw -ErrorAction SilentlyContinue
  throw "SSH tunnel failed to start. $message"
}

$localApiUrl = "http://127.0.0.1:$BackendPort"
if (-not (Wait-HttpOk -Url "$localApiUrl/api/health" -Attempts 10)) {
  Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
  throw "The SSH tunnel started, but the local API health check failed. See $tunnelErr"
}

$frontend = $null
if (-not $SkipFrontend) {
  $frontendOut = Join-Path $LogDir "remote-frontend.out.log"
  $frontendErr = Join-Path $LogDir "remote-frontend.err.log"
  $previousApiBaseUrl = $env:VITE_API_BASE_URL
  $env:VITE_API_BASE_URL = $localApiUrl
  try {
    $frontendCommand = "`"$pnpm`" run dev -- --host $HostAddress --port $FrontendPort"
    Write-Host "Starting local frontend: http://${HostAddress}:$FrontendPort"
    $frontend = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $frontendCommand) -WorkingDirectory $FrontendDir -RedirectStandardOutput $frontendOut -RedirectStandardError $frontendErr -PassThru -WindowStyle Hidden
  } finally {
    if ($null -eq $previousApiBaseUrl) {
      Remove-Item Env:\VITE_API_BASE_URL -ErrorAction SilentlyContinue
    } else {
      $env:VITE_API_BASE_URL = $previousApiBaseUrl
    }
  }

  if (-not (Wait-HttpOk -Url "http://${HostAddress}:$FrontendPort" -Attempts 30)) {
    Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
    $message = Get-Content $frontendErr -Raw -ErrorAction SilentlyContinue
    throw "Frontend failed to start. $message"
  }
}

$state = [ordered]@{
  startedAt = (Get-Date).ToString("o")
  sshHost = $SshHost
  remoteBackendReused = $remoteBackendReused
  remoteDevice = $remoteHealth.device
  maxConcurrency = $remoteHealth.protectQueue.maxConcurrency
  tunnelPid = $tunnel.Id
  frontendPid = if ($frontend) { $frontend.Id } else { $null }
  apiUrl = $localApiUrl
  frontendUrl = if ($frontend) { "http://${HostAddress}:$FrontendPort/workspace" } else { $null }
}
$state | ConvertTo-Json | Set-Content -Path $StatePath -Encoding UTF8

Write-Host ""
Write-Host "Remote API:  $localApiUrl/api/health"
Write-Host "API docs:    $localApiUrl/docs"
if ($frontend) {
  $workspaceUrl = "http://${HostAddress}:$FrontendPort/workspace"
  Write-Host "Workspace:   $workspaceUrl"
  if (-not $NoBrowser) {
    Start-Process $workspaceUrl
  }
}
Write-Host "Tunnel PID:  $($tunnel.Id)"
if ($frontend) {
  Write-Host "Frontend PID: $($frontend.Id)"
}
Write-Host "State:       $StatePath"
