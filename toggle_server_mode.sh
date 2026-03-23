#!/usr/bin/env bash
# toggle_server_mode.sh - Toggle between CLI and Server mode
# Updates config.toml and restarts the daemon

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.toml"

if [ ! -f "$CONFIG_FILE" ]; then
    notify-send "Whisper Mode" "Config file not found: $CONFIG_FILE" -u critical -t 2000
    exit 1
fi

# Get current mode from config
CURRENT_MODE=$(python3 -c "
from config_loader import load_config
c = load_config('$SCRIPT_DIR/config.toml')
print(c.daemon.mode)
" 2>/dev/null || echo "cli")

if [ "$CURRENT_MODE" = "server" ]; then
    # Switch to CLI mode
    sed -i 's/^mode = "server"/mode = "cli"/' "$CONFIG_FILE"
    NEW_MODE="CLI"
    ICON="●"
else
    # Switch to Server mode
    sed -i 's/^mode = "cli"/mode = "server"/' "$CONFIG_FILE"
    NEW_MODE="Server"
    ICON="◆"
fi

# Restart daemon
systemctl --user daemon-reload
systemctl --user restart whisper.service

sleep 1

if systemctl --user is-active --quiet whisper.service; then
    notify-send "Whisper Mode" "$ICON Switched to $NEW_MODE mode" -t 2000
else
    notify-send "Whisper Mode" "Failed to restart daemon" -u critical -t 3000
fi
