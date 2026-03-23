#!/bin/bash
# Toggle whisper dictation (press-to-record mode)
# Reads socket path from config.toml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Try to read socket path from config
SOCKET_PATH=$(python3 -c "
from config_loader import load_config
c = load_config('$SCRIPT_DIR/config.toml')
print(c.daemon.socket_path)
" 2>/dev/null || echo "/tmp/whisper_daemon.sock")

# Check if daemon is running
if [ ! -S "$SOCKET_PATH" ]; then
    notify-send -u critical "Whisper" "Daemon not running! Start it first."
    exit 1
fi

# Send toggle command to daemon
echo "TOGGLE" | ncat -U "$SOCKET_PATH"
exit 0
