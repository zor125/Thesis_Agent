#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/Users/zor125/Projects/Paper_Agent"
LOG_DIR="$PROJECT_ROOT/logs"
if RUN_DATE="$(date -v-1d +%F 2>/dev/null)"; then
  :
else
  RUN_DATE="$(date -d "yesterday" +%F)"
fi
LAST_RUN_FILE="$LOG_DIR/last_run_$RUN_DATE"
RUN_LOG="$LOG_DIR/run_daily_$RUN_DATE.log"

mkdir -p "$LOG_DIR"
exec >> "$RUN_LOG" 2>&1

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Waiting 30 seconds for network readiness"
sleep 30
echo "[$(date "+%Y-%m-%d %H:%M:%S")] Paper_Agent daily run requested for $RUN_DATE"

cd "$PROJECT_ROOT" || {
  echo "Failed to enter project root: $PROJECT_ROOT"
  exit 1
}

if [[ -f "$LAST_RUN_FILE" ]]; then
  echo "Already ran for run date: $LAST_RUN_FILE"
  exit 0
fi

if [[ -f ".env" && -z "${OPENAI_API_KEY:-}" ]]; then
  env_key_line="$(grep -E "^OPENAI_API_KEY=" ".env" | tail -n 1 || true)"
  if [[ -n "$env_key_line" ]]; then
    OPENAI_API_KEY="${env_key_line#OPENAI_API_KEY=}"
    OPENAI_API_KEY="$(printf "%s" "$OPENAI_API_KEY" | tr -d "\r")"
    OPENAI_API_KEY="${OPENAI_API_KEY%\"}"
    OPENAI_API_KEY="${OPENAI_API_KEY#\"}"
    export OPENAI_API_KEY
  fi
fi

PYTHON_BIN=""
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif [[ -x "/opt/homebrew/bin/python3" ]]; then
  PYTHON_BIN="/opt/homebrew/bin/python3"
elif [[ -x "/usr/bin/python3" ]]; then
  PYTHON_BIN="/usr/bin/python3"
else
  echo "python3 not found"
  exit 1
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -m pip --version || echo "pip not available for $PYTHON_BIN"
echo "Starting daily.py for $RUN_DATE"
if "$PYTHON_BIN" daily.py --date "$RUN_DATE"; then
  touch "$LAST_RUN_FILE"
  echo "daily.py completed successfully; wrote $LAST_RUN_FILE"
  open -a "Obsidian" || echo "Obsidian could not be opened automatically"
else
  status=$?
  echo "daily.py failed with exit code $status"
  exit "$status"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Paper_Agent daily run finished"
