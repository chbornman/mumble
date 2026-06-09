"""Internal named constants.

Non-user-tunable magic numbers live here so the logic that uses them reads
clearly. Anything a user might reasonably want to change belongs in
``config.toml`` instead (see config_loader.py), not here.
"""

# --- Nemotron streaming: mic capture + thread/proc lifecycle -----------------
# Mic capture block sent to the sidecar per step. The sidecar re-chunks to its
# own native frame internally; this only controls how often the daemon ships
# audio over the socket.
STREAM_AUDIO_CHUNK_SECONDS = 0.1
# Blocking get() on the capture queue; bounded so the send loop can notice a
# stop request promptly.
STREAM_AUDIO_QUEUE_TIMEOUT_SECONDS = 0.2
# Joining the reader/audio threads when streaming stops.
STREAM_THREAD_JOIN_TIMEOUT_SECONDS = 3.0
# wait() on a daemon-spawned helper process (sidecar autostart / legacy stream).
STREAM_PROC_WAIT_TIMEOUT_SECONDS = 3.0
# Brief settle after signaling stop (FLUSH) before sending BYE / tearing down.
STREAM_STOP_SETTLE_SECONDS = 0.3
# Backoff between connect attempts while waiting for the sidecar socket.
STREAM_CONNECT_RETRY_SECONDS = 0.1

# --- Clipboard paste path ----------------------------------------------------
# How long to wait after dispatching the paste keystroke before restoring the
# previous clipboard. Must outlast the focused app's clipboard read or the
# paste silently drops; see clipboard_paste.py.
CLIPBOARD_RESTORE_SETTLE_MS = 400

# --- Live partial typing (inject_mode = "live") ------------------------------
# Separator inserted after a committed FINAL so consecutive utterances don't
# run together.
LIVE_FINAL_SEPARATOR = " "

# --- Subprocess / IPC timeouts ------------------------------------------------
# Graceful wait for whisper-server to exit after terminate(), before kill().
WHISPER_SERVER_TERMINATE_TIMEOUT_SECONDS = 5
# Per-request probe of whisper-server's HTTP port during startup.
WHISPER_SERVER_HEALTH_TIMEOUT_SECONDS = 1
# playerctl status/pause/play calls (media auto-pause while dictating).
PLAYERCTL_TIMEOUT_SECONDS = 1
# Typing one transcribed segment via the injector in the legacy stream pipeline.
STREAM_TYPER_TIMEOUT_SECONDS = 5
# Floor for the sidecar handshake socket timeout, so a config connect_timeout
# of ~0 can't make the handshake racy.
STREAM_HANDSHAKE_MIN_TIMEOUT_SECONDS = 0.5
# notify-send desktop notifications (fire-and-forget).
NOTIFY_SEND_TIMEOUT_SECONDS = 1
# Voice-command mode: wl-paste read of the current selection.
SELECTION_CAPTURE_TIMEOUT_SECONDS = 1
# One recv() on the daemon's IPC socket; verbs are short single lines.
IPC_RECV_BUFFER_BYTES = 1024
# Poll interval while recording, to notice stop/max-duration promptly (ms).
RECORDING_POLL_MS = 100

# --- Text injection / clipboard / app context ----------------------------------
# Typing a whole transcription via the injector (long dictations take a while
# to synthesize keystroke-by-keystroke).
INJECTOR_TYPE_TIMEOUT_SECONDS = 10.0
# One batched edit / backspace / paste keystroke operation.
INJECTOR_EDIT_TIMEOUT_SECONDS = 5.0
# wl-copy / wl-paste clipboard snapshot+restore calls.
CLIPBOARD_OP_TIMEOUT_SECONDS = 1.0
# hyprctl activewindow probe for per-app context.
APP_CONTEXT_PROBE_TIMEOUT_SECONDS = 0.5
# Connecting to the mumble-stt sidecar socket.
SIDECAR_CONNECT_TIMEOUT_SECONDS = 5.0

# --- Waybar status module ------------------------------------------------------
# STATUS query to the daemon socket; keep short so the bar never stalls.
WAYBAR_SOCKET_TIMEOUT_SECONDS = 0.5
# Fallback refresh interval when config.toml is missing.
WAYBAR_FALLBACK_UPDATE_INTERVAL_SECONDS = 0.5
