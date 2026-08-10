param(
  [string]$SshHost = "pro",
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173,
  [string]$HostAddress = "localhost",
  [switch]$SkipFrontend,
  [switch]$NoBrowser,
  [switch]$ForceRemoteRestart
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
min_free_gpu_mib=12288

[ "$(readlink -f "$project")" = "$HOME/VoiceSheild" ]
[ -x "$venv/bin/python" ]
mkdir -p "$deploy_dir"

expected="$venv/bin/python -m uvicorn api_server:app --host 127.0.0.1 --port $port"

process_cmdline() {
  local candidate_pid="$1"
  tr '\0' ' ' < "/proc/$candidate_pid/cmdline" 2>/dev/null | sed 's/ $//' || true
}

descendant_pids() {
  local parent_pid="$1"
  local children
  local child_pid
  children="$(ps -eo pid=,ppid= | awk -v parent="$parent_pid" '$2 == parent {print $1}')"
  for child_pid in $children; do
    descendant_pids "$child_pid"
    printf '%s\n' "$child_pid"
  done
}

normalize_pid_lines() {
  awk '/^[0-9]+$/ && $1 > 0 && !seen[$1]++ {print $1}'
}

managed_tree_pids() {
  local root_pid="$1"
  {
    descendant_pids "$root_pid"
    printf '%s\n' "$root_pid"
  } | normalize_pid_lines
}

signal_pid_lines() {
  local signal_name="$1"
  local pid_lines="$2"
  local candidate_pid
  printf '%s\n' "$pid_lines" | while IFS= read -r candidate_pid; do
    case "$candidate_pid" in
      ''|*[!0-9]*) continue ;;
    esac
    kill -"$signal_name" "$candidate_pid" 2>/dev/null || true
  done
}

living_pid_lines() {
  local pid_lines="$1"
  local candidate_pid
  printf '%s\n' "$pid_lines" | while IFS= read -r candidate_pid; do
    case "$candidate_pid" in
      ''|*[!0-9]*) continue ;;
    esac
    if kill -0 "$candidate_pid" 2>/dev/null; then
      printf '%s\n' "$candidate_pid"
    fi
  done
}

stop_pid_tree_snapshot() {
  local tree_pids="$1"
  local description="$2"
  local remaining
  local unused
  [ -n "$tree_pids" ] || return 0

  signal_pid_lines TERM "$tree_pids"
  remaining="$tree_pids"
  for unused in 1 2 3 4 5 6 7 8 9 10; do
    remaining="$(living_pid_lines "$tree_pids")"
    [ -n "$remaining" ] || break
    sleep 1
  done

  if [ -n "$remaining" ]; then
    printf '%s did not stop after 10 seconds; sending SIGKILL to remaining PIDs: %s\n' \
      "$description" "$(printf '%s' "$remaining" | tr '\n' ' ')" >&2
    signal_pid_lines KILL "$remaining"
    sleep 1
  fi

  remaining="$(living_pid_lines "$tree_pids")"
  if [ -n "$remaining" ]; then
    printf 'Failed to stop %s PIDs: %s\n' \
      "$description" "$(printf '%s' "$remaining" | tr '\n' ' ')" >&2
    return 1
  fi
  return 0
}

find_exact_backend_pid() {
  local found_pid=""
  local cmdline
  local actual
  local candidate_pid
  for cmdline in /proc/[0-9]*/cmdline; do
    [ -r "$cmdline" ] || continue
    actual="$(tr '\0' ' ' < "$cmdline" 2>/dev/null | sed 's/ $//' || true)"
    [ "$actual" = "$expected" ] || continue
    candidate_pid="${cmdline#/proc/}"
    candidate_pid="${candidate_pid%/cmdline}"
    if [ -n "$found_pid" ]; then
      printf 'Multiple managed backends match port %s; refusing to choose one.\n' "$port" >&2
      return 1
    fi
    found_pid="$candidate_pid"
  done
  printf '%s\n' "$found_pid"
}

old_pid=""
if [ -f "$pidfile" ]; then
  candidate_pid="$(cat "$pidfile")"
  case "$candidate_pid" in
    ''|*[!0-9]*)
      printf 'Invalid backend PID file: %s\n' "$pidfile" >&2
      exit 20
      ;;
  esac

  if kill -0 "$candidate_pid" 2>/dev/null; then
    actual="$(process_cmdline "$candidate_pid")"
    if [ "$actual" != "$expected" ]; then
      printf 'PID file points to an unexpected process: %s\n' "$actual" >&2
      exit 20
    fi
    old_pid="$candidate_pid"
  else
    rm -f "$pidfile"
  fi
fi

if [ -z "$old_pid" ]; then
  old_pid="$(find_exact_backend_pid)" || exit 21
fi

if [ -n "$old_pid" ]; then
  printf 'Stopping managed remote backend tree rooted at PID %s\n' "$old_pid"
  old_tree_pids="$(managed_tree_pids "$old_pid")"
  if ! stop_pid_tree_snapshot "$old_tree_pids" "Remote backend tree"; then
    exit 21
  fi
  rm -f "$pidfile"
fi

if ss -ltn 2>/dev/null | grep -q ":$port "; then
  printf 'Remote port %s is occupied by an unmanaged process\n' "$port" >&2
  exit 22
fi

printf '1\n' > "$project/seme2e-runtime/capabilities-refresh.flag"
cd "$backend"
export SEME2E_RUNTIME_DIR="$project/seme2e-runtime"
export SEME2E_API_REAL_GUARD=1
export SEME2E_ENABLE_ASR=1
export SEME2E_API_DEVICE='cuda:0'
export SEME2E_PROTECT_MAX_CONCURRENCY=1
export SEME2E_ASR_WORKER_MAX_CONCURRENCY=1
export SEME2E_SEMANTIC_WORKER_MAX_CONCURRENCY=1
export SEME2E_COQUI_TTS_WORKER_MAX_CONCURRENCY=1
export SEME2E_COSYVOICE_WORKER_MAX_CONCURRENCY=1
export SEME2E_CLONE_GPU_MAX_CONCURRENCY=1
export SEME2E_DNSMOS_WORKER_MAX_CONCURRENCY=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
gpt_worker_max_concurrency=1
unset SEME2E_GPT_SOVITS_GPU_POOL
unset SEME2E_GPT_SOVITS_CUDA_VISIBLE_DEVICES

if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t gpu_candidates < <(
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | awk -F',' -v minimum="$min_free_gpu_mib" '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); if (($2 + 0) >= minimum) print $2, $1}' \
      | sort -nr \
      | awk '{print $2}'
  )
  if [ "${#gpu_candidates[@]}" -eq 0 ]; then
    mapfile -t gpu_candidates < <(
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F',' '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); print $2, $1}' \
        | sort -nr \
        | awk '{print $2}'
    )
  fi
  if [ "${#gpu_candidates[@]}" -gt 0 ]; then
    gpt_gpu_primary="${gpu_candidates[0]}"
    gpt_gpu_secondary="${gpu_candidates[1]:-}"
    gpt_gpu_pool="$gpt_gpu_primary"
    if [ -n "$gpt_gpu_secondary" ] && [ "$gpt_gpu_secondary" != "$gpt_gpu_primary" ]; then
      gpt_gpu_pool="$gpt_gpu_primary,$gpt_gpu_secondary"
      gpt_worker_max_concurrency=2
    fi
    protect_gpu="${gpu_candidates[2]:-$gpt_gpu_primary}"
    asr_gpu="${gpu_candidates[3]:-$protect_gpu}"
    clone_gpu="${gpu_candidates[4]:-$asr_gpu}"
    export CUDA_VISIBLE_DEVICES="$protect_gpu"
    export SEME2E_ASR_CUDA_VISIBLE_DEVICES="$asr_gpu"
    export SEME2E_CLONE_ASR_CUDA_VISIBLE_DEVICES="$asr_gpu"
    export SEME2E_COQUI_TTS_CUDA_VISIBLE_DEVICES="$clone_gpu"
    export SEME2E_COSYVOICE_CUDA_VISIBLE_DEVICES="$clone_gpu"
    export SEME2E_GPT_SOVITS_GPU_POOL="$gpt_gpu_pool"
    printf 'GPU routing: protect=%s asr=%s clone=%s gpt-sovits-pool=%s\n' "$protect_gpu" "$asr_gpu" "$clone_gpu" "$gpt_gpu_pool"
  fi
fi
export SEME2E_GPT_SOVITS_WORKER_MAX_CONCURRENCY="$gpt_worker_max_concurrency"

new_pid=""
tracked_new_pids=""
startup_complete=0

remember_new_backend_tree() {
  local current_tree
  [ -n "$new_pid" ] || return 0
  current_tree="$(managed_tree_pids "$new_pid")"
  tracked_new_pids="$({
    printf '%s\n' "$tracked_new_pids"
    printf '%s\n' "$current_tree"
  } | normalize_pid_lines)"
}

cleanup_failed_startup() {
  local exit_code="${1:-$?}"
  trap - EXIT HUP INT TERM
  if [ "$startup_complete" -ne 1 ] && [ -n "$new_pid" ]; then
    remember_new_backend_tree
    if [ -n "$tracked_new_pids" ]; then
      printf 'Cleaning up failed backend startup tree rooted at PID %s\n' "$new_pid" >&2
      stop_pid_tree_snapshot "$tracked_new_pids" "Failed backend startup tree" || true
    fi
    if [ -f "$pidfile" ] && [ "$(cat "$pidfile" 2>/dev/null || true)" = "$new_pid" ]; then
      rm -f "$pidfile"
    fi
  fi
  exit "$exit_code"
}

trap cleanup_failed_startup EXIT
trap 'cleanup_failed_startup 129' HUP
trap 'cleanup_failed_startup 130' INT
trap 'cleanup_failed_startup 143' TERM
nohup "$venv/bin/python" -m uvicorn api_server:app --host 127.0.0.1 --port "$port" > "$outlog" 2> "$errlog" < /dev/null &
new_pid=$!
printf '%s\n' "$new_pid" > "$pidfile"
remember_new_backend_tree

for unused in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
  remember_new_backend_tree
  if ! kill -0 "$new_pid" 2>/dev/null; then
    tail -80 "$errlog" >&2 || true
    exit 23
  fi
  if curl --fail --silent --max-time 3 "http://127.0.0.1:$port/api/health" >/dev/null 2>&1; then
    actual="$(process_cmdline "$new_pid")"
    if [ "$actual" != "$expected" ]; then
      printf 'Health endpoint responded, but new backend PID %s no longer has the expected command line: %s\n' \
        "$new_pid" "$actual" >&2
      exit 25
    fi
    startup_complete=1
    trap - EXIT HUP INT TERM
    printf 'Remote backend started: PID %s\n' "$new_pid"
    exit 0
  fi
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
$remoteHealth = if ($ForceRemoteRestart) { $null } else { Get-RemoteHealth -SshPath $ssh }
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
