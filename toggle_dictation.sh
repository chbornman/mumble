#!/usr/bin/env bash
# toggle_dictation.sh - Toggle press-to-talk dictation.
#
# Back-compat keybind wrapper: all logic lives in the mumble CLI (`mumble
# toggle`), which speaks the daemon's IPC protocol. Bind your key to either.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/mumble" toggle
