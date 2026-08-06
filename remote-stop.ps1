<#
.SYNOPSIS
Stops the managed VoiceSheild backend through SSH.

.EXAMPLE
.\remote-stop.ps1

.EXAMPLE
.\remote-stop.ps1 -WhatIf
#>
[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = "Medium")]
param(
  [string]$SshHost = "pro",
  [ValidateRange(1, 65535)]
  [int]$BackendPort = 8000
)

$ErrorActionPreference = "Stop"

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

if (-not $PSCmdlet.ShouldProcess("${SshHost}:127.0.0.1:${BackendPort}", "Stop the managed VoiceSheild backend")) {
  return
}

$ssh = Resolve-CommandPath -Candidates @("ssh.exe", "ssh") -HelpText "OpenSSH client was not found. Install OpenSSH or add ssh.exe to PATH."
$scriptTemplate = @'
set -eu

project="$HOME/VoiceSheild"
venv="$project/.venv"
deploy_dir="$project/.runtime/deploy"
pidfile="$deploy_dir/backend.pid"
port=__BACKEND_PORT__
expected="$venv/bin/python -m uvicorn api_server:app --host 127.0.0.1 --port $port"

[ "$(readlink -f "$project")" = "$HOME/VoiceSheild" ]

port_is_listening() {
  ss -ltnH "sport = :$port" 2>/dev/null | grep -q .
}

managed_pid=""
if [ -f "$pidfile" ]; then
  candidate="$(cat "$pidfile")"
  case "$candidate" in
    ''|*[!0-9]*)
      printf 'Invalid backend PID file: %s\n' "$pidfile" >&2
      exit 20
      ;;
  esac

  if kill -0 "$candidate" 2>/dev/null; then
    actual="$(tr '\0' ' ' < "/proc/$candidate/cmdline" | sed 's/ $//')"
    if [ "$actual" != "$expected" ]; then
      printf 'PID file points to an unexpected process: %s\n' "$actual" >&2
      exit 21
    fi
    managed_pid="$candidate"
  else
    rm -f "$pidfile"
  fi
fi

if [ -z "$managed_pid" ]; then
  for cmdline in /proc/[0-9]*/cmdline; do
    [ -r "$cmdline" ] || continue
    actual="$(tr '\0' ' ' < "$cmdline" 2>/dev/null | sed 's/ $//' || true)"
    [ "$actual" = "$expected" ] || continue
    candidate="${cmdline#/proc/}"
    candidate="${candidate%/cmdline}"
    if [ -n "$managed_pid" ]; then
      printf 'Multiple managed backends match port %s; refusing to choose one.\n' "$port" >&2
      exit 22
    fi
    managed_pid="$candidate"
  done
fi

if [ -z "$managed_pid" ]; then
  if port_is_listening; then
    printf 'Remote port %s is occupied by an unmanaged process; nothing was stopped.\n' "$port" >&2
    exit 23
  fi
  printf 'Remote backend is already stopped; port %s is free.\n' "$port"
  exit 0
fi

printf 'Stopping managed remote backend: PID %s, port %s\n' "$managed_pid" "$port"
kill "$managed_pid"
for unused in 1 2 3 4 5 6 7 8 9 10; do
  kill -0 "$managed_pid" 2>/dev/null || break
  sleep 1
done

if kill -0 "$managed_pid" 2>/dev/null; then
  printf 'Backend did not stop after 10 seconds; sending SIGKILL.\n' >&2
  kill -KILL "$managed_pid"
  sleep 1
fi

if kill -0 "$managed_pid" 2>/dev/null; then
  printf 'Failed to stop remote backend PID %s.\n' "$managed_pid" >&2
  exit 24
fi

rm -f "$pidfile"
if port_is_listening; then
  printf 'Backend process stopped, but remote port %s is still occupied.\n' "$port" >&2
  exit 25
fi

printf 'Remote backend stopped; port %s is free.\n' "$port"
exit 0
# end
'@

$remoteScript = $scriptTemplate.Replace("__BACKEND_PORT__", [string]$BackendPort).Replace("`r", "")
$remoteScriptBytes = [System.Text.Encoding]::UTF8.GetBytes($remoteScript + "`n")
$remoteScriptBase64 = [Convert]::ToBase64String($remoteScriptBytes)
$remoteCommand = "printf '%s' '$remoteScriptBase64' | base64 --decode | bash -s"
& $ssh -o BatchMode=yes -o ConnectTimeout=15 $SshHost $remoteCommand
if ($LASTEXITCODE -ne 0) {
  throw "Failed to stop the managed remote backend on port $BackendPort."
}
