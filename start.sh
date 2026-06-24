#!/usr/bin/env bash
set -euo pipefail

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
HOST_ADDRESS="${HOST_ADDRESS:-127.0.0.1}"
SEME2E_API_REAL_GUARD="${SEME2E_API_REAL_GUARD:-1}"
SEME2E_API_ALLOW_FALLBACK="${SEME2E_API_ALLOW_FALLBACK:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT/backend/SemE2E"
FRONTEND_DIR="$ROOT/fro"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR"

kill_port() {
  local port="$1"
  local pids=""

  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti "tcp:$port" 2>/dev/null || true)"
  elif command -v fuser >/dev/null 2>&1; then
    pids="$(fuser "${port}/tcp" 2>/dev/null || true)"
  elif command -v ss >/dev/null 2>&1; then
    pids="$(ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)"
  fi

  if [[ -z "$pids" ]]; then
    return
  fi

  echo "Killing process(es) on port $port: $pids"
  kill $pids 2>/dev/null || true
  sleep 1
  kill -9 $pids 2>/dev/null || true
}

find_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    echo "Python is required to start the backend." >&2
    exit 1
  fi
}

if ! command -v pnpm >/dev/null 2>&1; then
  echo "pnpm is required to start the frontend." >&2
  exit 1
fi

kill_port "$BACKEND_PORT"
kill_port "$FRONTEND_PORT"

PYTHON_BIN="$(find_python)"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "Starting backend: http://localhost:$BACKEND_PORT"
(
  cd "$BACKEND_DIR"
  SEME2E_API_PORT="$BACKEND_PORT" SEME2E_API_REAL_GUARD="$SEME2E_API_REAL_GUARD" SEME2E_API_ALLOW_FALLBACK="$SEME2E_API_ALLOW_FALLBACK" "$PYTHON_BIN" api_server.py
) >"$LOG_DIR/backend.out.log" 2>"$LOG_DIR/backend.err.log" &
BACKEND_PID="$!"

echo "Starting frontend: http://localhost:$FRONTEND_PORT"
(
  cd "$FRONTEND_DIR"
  pnpm run dev -- --host "$HOST_ADDRESS" --port "$FRONTEND_PORT"
) >"$LOG_DIR/frontend.out.log" 2>"$LOG_DIR/frontend.err.log" &
FRONTEND_PID="$!"

echo ""
echo "Backend PID:  $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Logs:"
echo "  $LOG_DIR/backend.out.log"
echo "  $LOG_DIR/backend.err.log"
echo "  $LOG_DIR/frontend.out.log"
echo "  $LOG_DIR/frontend.err.log"
echo ""
echo "Press Ctrl+C to stop both services."

wait -n "$BACKEND_PID" "$FRONTEND_PID"
