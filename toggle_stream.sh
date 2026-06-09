#!/usr/bin/env bash
# toggle_stream.sh - Toggle live streaming transcription.
#
# Back-compat keybind wrapper: all logic lives in the mumble CLI (`mumble
# stream`), which sends the STREAM verb to the daemon. The daemon owns the
# streaming lifecycle and dispatches to whichever engine is configured
# (backend.streaming_backend: "whisper-stream" or "nemotron-streaming").

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/mumble" stream
