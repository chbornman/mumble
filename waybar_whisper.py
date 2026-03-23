#!/usr/bin/env python3
"""
Waybar module for Whisper dictation status.
Shows animated recording indicator in waybar.
All settings driven by config.toml.
"""

import json
import socket
import sys
import time
from pathlib import Path

from config_loader import load_config

# Load config once at startup
try:
    config = load_config()
except FileNotFoundError:
    # Fallback defaults if config not found
    config = None

# Resolve paths from config or use defaults
if config:
    SOCKET_PATH = config.daemon.socket_path
    RECORDING_FLAG = config.daemon.recording_flag
    STREAMING_FLAG = config.daemon.streaming_flag
    SERVICE_FILE = Path.home() / ".config/systemd/user/whisper.service"
    UPDATE_INTERVAL = config.waybar.update_interval
    ICONS = config.waybar.icons
else:
    SOCKET_PATH = "/tmp/whisper_daemon.sock"
    RECORDING_FLAG = "/tmp/whisper_recording"
    STREAMING_FLAG = "/tmp/whisper_streaming"
    SERVICE_FILE = Path.home() / ".config/systemd/user/whisper.service"
    UPDATE_INTERVAL = 0.5
    ICONS = None

frame_index = 0


def get_server_mode() -> bool:
    """Check if daemon is running in server mode."""
    if config:
        return config.daemon.mode == "server"
    try:
        if not SERVICE_FILE.exists():
            return False
        content = SERVICE_FILE.read_text()
        return "--server-mode" in content
    except Exception:
        return False


def is_streaming() -> bool:
    """Check if streaming mode is active."""
    return Path(STREAMING_FLAG).exists()


def get_icon(key: str) -> str:
    """Get icon string for a given state."""
    if ICONS:
        return getattr(ICONS, key, "~")
    # Hardcoded fallback
    defaults = {
        "cli_ready": "~",
        "cli_recording": "● dictation",
        "cli_processing": "dictation",
        "server_ready": "◆",
        "server_recording": "◆ dictation",
        "server_processing": "dictation",
        "streaming": "~ streaming",
        "error": "~",
    }
    return defaults.get(key, "~")


def get_current_model() -> str:
    """Get current model name from config or systemd service file."""
    if config:
        return config.model.name
    try:
        if not SERVICE_FILE.exists():
            return "unknown"
        content = SERVICE_FILE.read_text()
        for line in content.split("\n"):
            if "ExecStart=" in line and "--model" in line:
                parts = line.split("--model")
                if len(parts) > 1:
                    model_path = parts[1].strip().split()[0]
                    model_file = Path(model_path).name
                    if model_file.startswith("ggml-") and model_file.endswith(".bin"):
                        return model_file[5:-4]
        return "unknown"
    except Exception:
        return "unknown"


def get_daemon_status() -> str:
    """Check daemon status via socket."""
    if not Path(SOCKET_PATH).exists():
        return "error"

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        sock.connect(SOCKET_PATH)
        sock.send(b"STATUS")
        response = sock.recv(1024).decode().strip()
        sock.close()

        if response == "RECORDING":
            return "recording"
        elif response == "READY":
            return "ready"
        else:
            return "error"
    except Exception:
        return "error"


def get_waybar_output() -> str:
    """Generate waybar JSON output."""
    global frame_index

    streaming = is_streaming()

    if streaming:
        icon = get_icon("streaming")
        tooltip = (
            f"Streaming (VAD mode)\n"
            f"Model: {get_current_model()}\n"
            f"Backend: {config.backend.type if config else 'cpu'}\n"
            f"SUPER+Shift+D: stop stream"
        )
        css_class = "streaming"
    else:
        status = get_daemon_status()
        model = get_current_model()
        is_server = get_server_mode()
        backend = config.backend.type if config else "cpu"
        mode_text = "Server (model in memory)" if is_server else "CLI (loads each time)"

        if Path(RECORDING_FLAG).exists() and status == "ready":
            status = "processing"

        if status == "recording":
            prefix = "server" if is_server else "cli"
            icon = get_icon(f"{prefix}_recording")
            frame_index += 1
            tooltip = (
                f"Recording... (SUPER+D to stop)\n"
                f"Model: {model}\n"
                f"Mode: {mode_text}\n"
                f"Backend: {backend}\n"
                f"Right-click: switch model | SUPER+Shift+D: start stream"
            )
            css_class = "recording"
        elif status == "processing":
            prefix = "server" if is_server else "cli"
            icon = get_icon(f"{prefix}_processing")
            tooltip = (
                f"Processing transcription...\n"
                f"Model: {model}\n"
                f"Mode: {mode_text}\n"
                f"Backend: {backend}"
            )
            css_class = "processing"
        elif status == "ready":
            prefix = "server" if is_server else "cli"
            icon = get_icon(f"{prefix}_ready")
            tooltip = (
                f"Ready (SUPER+D to start)\n"
                f"Model: {model}\n"
                f"Mode: {mode_text}\n"
                f"Backend: {backend}\n"
                f"Right-click: switch model | SUPER+Shift+D: start stream"
            )
            css_class = "ready"
        else:
            icon = get_icon("error")
            tooltip = f"Daemon not running\nModel: {model}\nMode: {mode_text}"
            css_class = "error"

    output = {"text": icon, "tooltip": tooltip, "class": css_class}
    return json.dumps(output)


def main():
    """Main loop for waybar module."""
    try:
        while True:
            print(get_waybar_output(), flush=True)
            time.sleep(UPDATE_INTERVAL)
    except (KeyboardInterrupt, BrokenPipeError):
        pass


if __name__ == "__main__":
    main()
