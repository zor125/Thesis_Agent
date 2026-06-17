#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Users/zor125/Projects/Paper_Agent"
LABEL="com.zor125.paperagent"
TARGET_PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl unload "$TARGET_PLIST" 2>/dev/null || true
rm -f "$TARGET_PLIST"

echo "Uninstalled $LABEL"
echo "Project root: $PROJECT_ROOT"
