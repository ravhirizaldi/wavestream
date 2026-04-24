#!/usr/bin/env bash
set -euo pipefail

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8880}"

exec python3 -m uvicorn app:app --host "$HOST" --port "$PORT"
