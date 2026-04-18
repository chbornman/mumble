#!/usr/bin/env bash
# toggle_llm.sh - Load/unload the llama-server LLM used for dictation cleanup.
# When unloaded, mumble falls back to raw Whisper output (by design).

set -e

UNIT=llama-server.service
PORT=19999

if systemctl --user is-active --quiet "$UNIT"; then
    systemctl --user stop "$UNIT"
    notify-send "LLM" "Unloaded (VRAM freed)" -t 1500
else
    systemctl --user start "$UNIT"
    # Port-listening is a better readiness signal than systemd's "active"
    # state because llama.cpp takes a few seconds to mmap the model.
    for i in {1..60}; do
        if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
            if curl -s -o /dev/null -m 1 "http://127.0.0.1:$PORT/v1/models"; then
                notify-send "LLM" "Loaded" -t 1500
                exit 0
            fi
        fi
        sleep 0.25
    done
    notify-send "LLM" "Start timed out — check journalctl --user -u $UNIT" -u critical -t 3000
    exit 1
fi
