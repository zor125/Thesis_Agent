#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Users/zor125/Projects/Paper_Agent"
LABEL="com.zor125.paperagent"
SOURCE_PLIST="$PROJECT_ROOT/scripts/$LABEL.plist"
TARGET_DIR="$HOME/Library/LaunchAgents"
TARGET_PLIST="$TARGET_DIR/$LABEL.plist"

mkdir -p "$TARGET_DIR"
mkdir -p "$PROJECT_ROOT/logs"

cp "$SOURCE_PLIST" "$TARGET_PLIST"
chmod +x "$PROJECT_ROOT/scripts/run_daily.sh"

launchctl unload "$TARGET_PLIST" 2>/dev/null || true
launchctl load "$TARGET_PLIST"

echo "Installed $LABEL"
echo "Manual test:"
echo "launchctl start $LABEL"
echo "Logs:"
echo "cat $PROJECT_ROOT/logs/launchd.out.log"
echo "cat $PROJECT_ROOT/logs/launchd.err.log"
