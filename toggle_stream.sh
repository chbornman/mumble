#!/usr/bin/env bash
# toggle_stream.sh - Toggle live streaming transcription mode
# Reads configuration from config.toml via the Python config helper.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Read config values via Python helper
read_config() {
    "$SCRIPT_DIR/.venv/bin/python" -c "
from config_loader import load_config
c = load_config('$SCRIPT_DIR/config.toml')
print(c.daemon.streaming_flag)
print(c.daemon.stream_pid_file)
print(str(c.paths.sound_dir))
print(str(c.whisper_stream_path))
print(str(c.model_path))
print(c.streaming.vad_threshold)
print(c.streaming.threads)
print(c.streaming.buffer_length)
print(c.streaming.keep)
print(c.streaming.step)
print(c.streaming.debug.stream_log)
print(c.streaming.debug.output_log)
print(c.model.language)
print(c.backend.type)
print(c.backend.vulkan.device)
print(c.wayland.display)
print(c.wayland.xdg_runtime_dir)
" 2>/dev/null
}

# Parse config
CONFIG_VALUES=$(read_config)
if [ -z "$CONFIG_VALUES" ]; then
    # Fallback to defaults if config loading fails
    STREAM_FLAG="/tmp/whisper_streaming"
    STREAM_PID_FILE="/tmp/whisper_stream.pid"
    SOUND_DIR="$HOME/projects/mumble/sounds"
    WHISPER_STREAM="$HOME/projects/whisper.cpp/build/bin/whisper-stream"
    MODEL="$HOME/projects/whisper.cpp/models/ggml-base.en.bin"
    VAD_THRESHOLD="0.6"
    THREADS="8"
    BUFFER_LENGTH="30000"
    KEEP="200"
    STEP="0"
    STREAM_LOG="/tmp/whisper_stream.log"
    OUTPUT_LOG="/tmp/whisper_stream_output.log"
    LANGUAGE="en"
    BACKEND="cpu"
    VULKAN_DEVICE="0"
    WAYLAND_DISPLAY_CFG="wayland-1"
    XDG_RUNTIME_DIR_CFG="/run/user/$(id -u)"
else
    STREAM_FLAG=$(echo "$CONFIG_VALUES" | sed -n '1p')
    STREAM_PID_FILE=$(echo "$CONFIG_VALUES" | sed -n '2p')
    SOUND_DIR=$(echo "$CONFIG_VALUES" | sed -n '3p')
    WHISPER_STREAM=$(echo "$CONFIG_VALUES" | sed -n '4p')
    MODEL=$(echo "$CONFIG_VALUES" | sed -n '5p')
    VAD_THRESHOLD=$(echo "$CONFIG_VALUES" | sed -n '6p')
    THREADS=$(echo "$CONFIG_VALUES" | sed -n '7p')
    BUFFER_LENGTH=$(echo "$CONFIG_VALUES" | sed -n '8p')
    KEEP=$(echo "$CONFIG_VALUES" | sed -n '9p')
    STEP=$(echo "$CONFIG_VALUES" | sed -n '10p')
    STREAM_LOG=$(echo "$CONFIG_VALUES" | sed -n '11p')
    OUTPUT_LOG=$(echo "$CONFIG_VALUES" | sed -n '12p')
    LANGUAGE=$(echo "$CONFIG_VALUES" | sed -n '13p')
    BACKEND=$(echo "$CONFIG_VALUES" | sed -n '14p')
    VULKAN_DEVICE=$(echo "$CONFIG_VALUES" | sed -n '15p')
    WAYLAND_DISPLAY_CFG=$(echo "$CONFIG_VALUES" | sed -n '16p')
    XDG_RUNTIME_DIR_CFG=$(echo "$CONFIG_VALUES" | sed -n '17p')
fi

# Ensure Wayland env is set for wtype
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-$WAYLAND_DISPLAY_CFG}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$XDG_RUNTIME_DIR_CFG}"

# Check if already streaming
if [ -f "$STREAM_FLAG" ]; then
    # Stop streaming
    if [ -f "$STREAM_PID_FILE" ]; then
        PID=$(cat "$STREAM_PID_FILE")
        kill $PID 2>/dev/null || true
        pkill -P $PID 2>/dev/null || true
        rm -f "$STREAM_PID_FILE"
    fi

    # Kill any lingering whisper-stream processes
    pkill -f "whisper-stream" 2>/dev/null || true

    rm -f "$STREAM_FLAG"

    if [ -f "$SOUND_DIR/hihat.wav" ]; then
        pw-play "$SOUND_DIR/hihat.wav" &
    fi

    notify-send "Whisper Stream" "Streaming stopped" -t 1500
else
    # Start streaming
    touch "$STREAM_FLAG"

    if [ -f "$SOUND_DIR/snare.wav" ]; then
        pw-play "$SOUND_DIR/snare.wav" &
    fi

    # Build GPU/device flags
    GPU_FLAGS=""
    if [ "$BACKEND" = "cpu" ]; then
        GPU_FLAGS="--no-gpu"
    else
        GPU_FLAGS="--device $VULKAN_DEVICE"
    fi

    # Launch streaming with Python deduplication
    (
        "$WHISPER_STREAM" \
            -m "$MODEL" \
            -l "$LANGUAGE" \
            $GPU_FLAGS \
            --step "$STEP" \
            --length "$BUFFER_LENGTH" \
            --keep "$KEEP" \
            -vth "$VAD_THRESHOLD" \
            -t "$THREADS" \
            2>"$STREAM_LOG" \
            | tee "$OUTPUT_LOG" \
            | "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/stream_dedup.py"
    ) &

    echo $! > "$STREAM_PID_FILE"

    notify-send "Whisper Stream" "Streaming started (VAD mode)" -t 1500
fi
