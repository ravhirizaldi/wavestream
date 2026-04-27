#!/usr/bin/env bash
# Wavestream launcher.
#
#   ./start.sh                # foreground (default; same as before)
#   ./start.sh -f             # foreground (explicit)
#   ./start.sh -d             # background (daemonize); writes PID + log
#   ./start.sh -s             # stop the background instance
#   ./start.sh -e <env-file>  # override env file (default: .env.production)
#   ./start.sh -h             # help
#
# In background mode the env file (if present) is sourced before launch, the
# server runs under nohup, the PID is written to .run/wavestream.pid and stdout
# + stderr are appended to .run/wavestream.log.
set -euo pipefail

cd "$(dirname "$0")"

ENV_FILE=".env.production"
RUN_DIR=".run"
PID_FILE="$RUN_DIR/wavestream.pid"
LOG_FILE="$RUN_DIR/wavestream.log"
MODE="foreground"
STOP_TIMEOUT_S=15

usage() {
  cat <<EOF
Usage: $0 [-d|-s|-f] [-e ENV_FILE]

  -d            run in background; PID -> $PID_FILE, logs -> $LOG_FILE
  -s            stop a background instance started with -d
  -f            run in foreground (default)
  -e ENV_FILE   env file to source before launch (default: $ENV_FILE)
  -h            show this help
EOF
}

while getopts "dsfe:h" opt; do
  case "$opt" in
    d) MODE="background" ;;
    s) MODE="stop" ;;
    f) MODE="foreground" ;;
    e) ENV_FILE="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

stop_running() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file at $PID_FILE; nothing to stop."
    return 0
  fi
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    echo "No live process for PID '${pid:-<empty>}'; removing stale PID file."
    rm -f "$PID_FILE"
    return 0
  fi
  echo "Stopping wavestream (PID $pid)..."
  kill -TERM "$pid"
  for ((i = 0; i < STOP_TIMEOUT_S * 2; i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "Stopped."
      return 0
    fi
    sleep 0.5
  done
  echo "Did not exit within ${STOP_TIMEOUT_S}s; sending SIGKILL."
  kill -KILL "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
}

if [[ "$MODE" == "stop" ]]; then
  stop_running
  exit 0
fi

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$ENV_FILE"; set +a
fi

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8880}"

if [[ "$MODE" == "background" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    existing="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$existing" ]] && kill -0 "$existing" 2>/dev/null; then
      echo "Already running (PID $existing). Use '$0 -s' to stop first." >&2
      exit 1
    fi
    rm -f "$PID_FILE"
  fi
  mkdir -p "$RUN_DIR"
  echo "Starting wavestream in background on http://$HOST:$PORT"
  echo "  env file : ${ENV_FILE}$([[ -f "$ENV_FILE" ]] || echo " (not found, using process env)")"
  echo "  pid file : $PID_FILE"
  echo "  log file : $LOG_FILE"
  nohup python3 -m uvicorn app:app --host "$HOST" --port "$PORT" \
    >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  disown 2>/dev/null || true
  echo "Started (PID $(cat "$PID_FILE"))."
  exit 0
fi

exec python3 -m uvicorn app:app --host "$HOST" --port "$PORT"
