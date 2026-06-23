#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/Users/zor125/Projects/Paper_Agent"
LOG_DIR="$PROJECT_ROOT/logs"

resolve_date_offset() {
  local offset="$1"
  if date -v-"${offset}"d +%F >/dev/null 2>&1; then
    date -v-"${offset}"d +%F
  else
    date -d "${offset} days ago" +%F
  fi
}

START_DATE="$(resolve_date_offset 1)"
RUN_LOG="$LOG_DIR/run_daily_$START_DATE.log"

mkdir -p "$LOG_DIR"
exec >> "$RUN_LOG" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting 30 seconds for network readiness"
sleep 30
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Paper_Agent automatic daily run requested"
echo "Starting search from: $START_DATE"

cd "$PROJECT_ROOT" || {
  echo "Failed to enter project root: $PROJECT_ROOT"
  exit 1
}

if [[ -f ".env" && -z "${OPENAI_API_KEY:-}" ]]; then
  env_key_line="$(grep -E "^OPENAI_API_KEY=" ".env" | tail -n 1 || true)"
  if [[ -n "$env_key_line" ]]; then
    OPENAI_API_KEY="${env_key_line#OPENAI_API_KEY=}"
    OPENAI_API_KEY="$(printf "%s" "$OPENAI_API_KEY" | tr -d $'\r')"
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

for offset in 1 2 3 4 5 6 7; do
  RUN_DATE="$(resolve_date_offset "$offset")"
  LAST_RUN_FILE="$LOG_DIR/last_run_$RUN_DATE"

  echo ""
  echo "Checking date: $RUN_DATE"

  if [[ -f "$LAST_RUN_FILE" ]]; then
    echo "Already processed: $RUN_DATE"
    continue
  fi

  FETCH_OUTPUT="$("$PYTHON_BIN" fetch.py --date "$RUN_DATE" --max-results 100 --debug 2>&1)"
  FETCH_STATUS=$?
  printf '%s\n' "$FETCH_OUTPUT"

  if [[ "$FETCH_STATUS" -ne 0 ]]; then
    echo "fetch.py failed for $RUN_DATE with exit code $FETCH_STATUS. Trying previous date."
    continue
  fi

  FILTERED_COUNT="$(printf '%s\n' "$FETCH_OUTPUT" | awk -F': ' '/filtered count:/ {print $2; exit}')"
  if [[ -z "$FILTERED_COUNT" ]]; then
    echo "Could not determine filtered count for $RUN_DATE. Trying previous date."
    continue
  fi

  if [[ "$FILTERED_COUNT" == "0" ]]; then
    echo "No papers found."
    continue
  fi

  echo "Found $FILTERED_COUNT papers."
  echo "Running daily.py for $RUN_DATE"

  if "$PYTHON_BIN" daily.py --date "$RUN_DATE"; then
    touch "$LAST_RUN_FILE"
    echo "Success."
    echo "daily.py completed successfully; wrote $LAST_RUN_FILE"
    open -a "Obsidian" || echo "Obsidian could not be opened automatically"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Paper_Agent daily run finished for $RUN_DATE"
    exit 0
  else
    status=$?
    echo "daily.py failed for $RUN_DATE with exit code $status. Trying previous date."
  fi
done

echo ""
echo "No new paper date found. Skipping run."
exit 0
