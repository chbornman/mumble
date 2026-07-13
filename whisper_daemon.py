#!/usr/bin/env python3
"""
Whisper Daemon - Persistent dictation service with Unix socket IPC.
Supports CLI mode (model per request), server mode (persistent model),
and streaming mode (live VAD transcription with deduplication).

All settings are driven by config.toml — no magic numbers.
"""

import argparse
import json
import logging
import os
import queue
import re
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd

from app_context import (
    detect_app_context,
    format_context_block,
    select_app_style,
)
import constants
from clipboard_paste import paste_via_clipboard
from config_loader import Config, load_config
from text_injector import resolve_injector
from glossary import (
    WHISPER_PROMPT_CHAR_BUDGET,
    Glossary,
    format_whisper_prompt,
    load_glossary,
)
from llm_postprocess import LLMPostProcessor
from modes import available_modes, resolve_mode_block, resolve_mode_for_app
from stream_vad import SileroVad, StreamingVadGate

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def setup_logging(config: Config) -> logging.Logger:
    """Configure logging from config."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.daemon.log_file, mode="a"),
        ],
    )
    return logging.getLogger("whisper_daemon")


class StreamDeduplicator:
    """
    Character-based deduplication for whisper-stream output.

    whisper-stream uses a rolling audio buffer, so each transcription block
    overlaps with the previous one. This class tracks what text has already
    been typed ("committed") and extracts only the new portion from each
    transcription update.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        self.cfg = config.streaming
        self.logger = logger
        self.committed_text = ""
        self.fallback_count = 0

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison: lowercase, collapse whitespace."""
        return re.sub(r"\s+", " ", text.lower()).strip()

    def extract_new_text(self, current_full_text: str) -> str:
        """
        Given the full transcription from the current whisper-stream block,
        return only the text that hasn't been typed yet.
        """
        if not current_full_text:
            return ""

        if not self.committed_text:
            self.logger.debug("First transcription, returning all text")
            return current_full_text

        committed_norm = self._normalize(self.committed_text)
        current_norm = self._normalize(current_full_text)

        # Strategy 1: Find longest suffix of committed that matches a prefix of current
        committed_len = len(committed_norm)
        min_overlap = self.cfg.min_overlap_chars
        step = self.cfg.overlap_step

        for cut in range(0, committed_len - min_overlap, step):
            suffix = committed_norm[cut:]
            suffix_len = len(suffix)
            current_prefix = current_norm[:suffix_len]

            if suffix == current_prefix:
                # Map normalized position back to original text
                original_len = len(current_full_text)
                norm_len = len(current_norm)
                ratio_pos = (suffix_len * original_len) // norm_len if norm_len else 0

                new_text = current_full_text[ratio_pos:]
                new_text = self._trim_partial_word(
                    current_full_text, ratio_pos, new_text
                )
                new_text = new_text.lstrip()

                self.fallback_count = 0
                self.logger.debug(
                    f"Overlap found (cut={cut}, suffix_len={suffix_len}, "
                    f"ratio_pos={ratio_pos}) -> new: '{new_text}'"
                )
                return new_text

        # Strategy 2: Search for end of committed text within current text
        suffix_search_len = self.cfg.fallback_suffix_length
        if committed_len > suffix_search_len:
            search_suffix = committed_norm[-suffix_search_len:]
        else:
            search_suffix = committed_norm

        try:
            pos = current_norm.index(search_suffix)
            new_start = pos + len(search_suffix)
            original_len = len(current_full_text)
            norm_len = len(current_norm)
            ratio_pos = (new_start * original_len) // norm_len if norm_len else 0

            new_text = current_full_text[ratio_pos:]
            new_text = self._trim_partial_word(current_full_text, ratio_pos, new_text)
            new_text = new_text.lstrip()

            self.fallback_count = 0
            self.logger.debug(
                f"Committed suffix found at pos {pos}, ratio_pos={ratio_pos} "
                f"-> new: '{new_text}'"
            )
            return new_text
        except ValueError:
            pass

        # Strategy 3: Fallback — no overlap found
        self.fallback_count += 1
        self.logger.debug(f"No overlap found (fallback_count={self.fallback_count})")

        if self.fallback_count >= self.cfg.drift_reset_threshold:
            self.logger.debug("Resetting committed_text due to drift")
            new_text = self._extract_last_sentence(current_full_text)
            self.committed_text = current_full_text
            self.fallback_count = 0
            return new_text

        # Conservative: just the last sentence
        return self._extract_last_sentence(current_full_text)

    def _trim_partial_word(self, full_text: str, ratio_pos: int, new_text: str) -> str:
        """If we landed mid-word, skip to the next word boundary."""
        if not new_text:
            return new_text
        first_char = new_text[0]
        if first_char != " " and ratio_pos > 0:
            char_before = full_text[ratio_pos - 1]
            if char_before != " ":
                # Mid-word — skip to next space
                space_idx = new_text.find(" ")
                if space_idx != -1:
                    return new_text[space_idx:]
        return new_text

    def _extract_last_sentence(self, text: str) -> str:
        """Extract the last sentence as a conservative fallback."""
        # Split on sentence-ending punctuation followed by space
        match = re.search(r"[.!?]\s+", text[::-1])
        if match:
            boundary = len(text) - match.start()
            last = text[boundary:]
            if len(last) < self.cfg.max_fallback_sentence_length:
                return last
        return ""

    def commit(self, typed_text: str):
        """Record that text was successfully typed."""
        if self.committed_text:
            self.committed_text += " " + typed_text
        else:
            self.committed_text = typed_text

        # Trim if too long
        max_len = self.cfg.max_committed_length
        if len(self.committed_text) > max_len:
            self.committed_text = self.committed_text[-max_len:]
            # Trim to word boundary
            space_idx = self.committed_text.find(" ")
            if space_idx != -1:
                self.committed_text = self.committed_text[space_idx + 1 :]

    def reset(self):
        """Reset state (e.g., after long silence)."""
        self.committed_text = ""
        self.fallback_count = 0


class WhisperDaemon:
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Validate paths
        if not config.model_path.exists():
            self.logger.error(f"Model not found: {config.model_path}")
            sys.exit(1)
        if not config.whisper_cli_path.exists():
            self.logger.error(f"Whisper CLI not found: {config.whisper_cli_path}")
            sys.exit(1)

        # Load and prioritize vocab for Whisper --prompt.
        self.vocab_prompt = self._load_vocab()

        # LLM post-processing (optional, off by default). When enabled we also
        # retain the structured Glossary for LLM hints + deterministic
        # substitutions. Whisper prompt parsing uses the same format regardless.
        self.glossary: Glossary | None = None
        self.llm_processor: LLMPostProcessor | None = None
        if config.llm_postprocess.enabled:
            self.glossary = load_glossary(config.paths.vocab_file)
            self.llm_processor = LLMPostProcessor(config.llm_postprocess, self.logger)
            self.logger.info(
                f"LLM postprocess enabled: {config.llm_postprocess.endpoint} "
                f"(model={config.llm_postprocess.model})"
            )

        # Pluggable text injector (wtype/ydotool/xdotool/raw), resolved once
        # from wayland.typer. See text_injector.py.
        self.injector = resolve_injector(config.wayland.typer, self.logger)

        # State
        self.recording = False
        self.interrupted = False
        self.audio_queue = queue.Queue()
        self.server_socket = None
        self.whisper_server_process = None
        # Default mode for the session; overridden per-turn via IPC
        # (`START:mode=commit`) or by per-app config. `None` = base
        # cleanup prompt only.
        self.default_mode: str | None = None
        # Active mode for the current/next recording, consumed on stop.
        self.active_mode: str | None = None
        # Command-mode state. When the next recording is a voice command,
        # we capture the selection at START time, store it here, and the
        # transcription thread uses it as the "selected text" input to
        # the LLM transform. Cleared after each turn.
        self.command_mode_pending = False
        self.command_mode_selection: str = ""

        # MPRIS players paused on dictation start; resumed on stop. Tracked
        # per-player so already-paused media doesn't get woken up on stop.
        self._paused_players: list[str] = []

        # Streaming session state (daemon-owned, parallels recording state).
        # `streaming` is the single source of truth for "a stream is live".
        # `stream_thread` runs the sidecar reader loop (nemotron path) and is
        # gated by `stream_stop`; `stream_socket` is the client connection to
        # the mumble-stt sidecar; `legacy_stream_proc` holds the whisper-stream
        # | stream_dedup.py subprocess when the legacy backend is selected (or
        # fallen back to).
        self.streaming = False
        self.stream_thread: threading.Thread | None = None
        self.stream_stop = threading.Event()
        self.stream_socket: socket.socket | None = None
        # inject_mode="live": the text currently typed for the in-progress
        # utterance, so the reader loop can backspace+retype the revised tail.
        self._live_typed = ""
        self.legacy_stream_proc: subprocess.Popen | None = None
        # The stream_dedup.py stage of the legacy pipeline (downstream of
        # legacy_stream_proc); reaped alongside it in teardown.
        self._legacy_dedup_proc: subprocess.Popen | None = None

        # Server mode from config
        self.server_mode = config.daemon.mode == "server"

        # Audio feedback
        self.start_sound = None
        self.stop_sound = None
        self._preload_sounds()

        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("Whisper daemon initialized")
        self.logger.info(f"Model: {config.model_path}")
        self.logger.info(f"Backend: {config.backend.type}")
        self.logger.info(f"Threads: {config.backend.effective_threads}")
        self.logger.info(f"Mode: {'server' if self.server_mode else 'cli'}")
        if self.vocab_prompt:
            self.logger.info(f"Vocab prompt loaded ({len(self.vocab_prompt)} chars)")

    def _load_vocab(self) -> str | None:
        """Load a bounded Whisper prompt, favoring later personalized terms."""
        vocab_file = self.config.paths.vocab_file
        if not vocab_file or not vocab_file.exists():
            if vocab_file:
                self.logger.warning(f"Vocab file not found: {vocab_file}")
            return None

        try:
            glossary = load_glossary(vocab_file)
            prompt = format_whisper_prompt(
                glossary, max_chars=WHISPER_PROMPT_CHAR_BUDGET
            )
            self.logger.info(
                f"Loaded bounded vocab prompt from {vocab_file} "
                f"({len(prompt)}/{WHISPER_PROMPT_CHAR_BUDGET} chars)"
            )
            return prompt or None
        except Exception as e:
            self.logger.error(f"Failed to load vocab file: {e}")
            return None

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("Received shutdown signal")
        self.interrupted = True
        # Tear down any live streaming session so the reader thread, sidecar
        # socket, and legacy subprocess (incl. an autostarted multi-GB sidecar)
        # don't leak on shutdown.
        if self.streaming:
            try:
                self.stop_streaming()
            except Exception as e:
                self.logger.warning(f"Error stopping streaming on shutdown: {e}")
        if self.server_socket:
            self.server_socket.close()
        if self.whisper_server_process:
            self.logger.info("Stopping whisper server...")
            self.whisper_server_process.terminate()
            try:
                self.whisper_server_process.wait(
                    timeout=constants.WHISPER_SERVER_TERMINATE_TIMEOUT_SECONDS
                )
            except subprocess.TimeoutExpired:
                self.whisper_server_process.kill()
        sys.exit(0)

    def _preload_sounds(self):
        """Preload audio feedback sounds into memory."""
        try:
            start_file = self.config.paths.sound_dir / "snare.wav"
            stop_file = self.config.paths.sound_dir / "hihat.wav"

            if start_file.exists():
                _, self.start_sound = wavfile.read(start_file)
                self.logger.info(f"Loaded start sound: {start_file}")

            if stop_file.exists():
                _, self.stop_sound = wavfile.read(stop_file)
                self.logger.info(f"Loaded stop sound: {stop_file}")
        except Exception as e:
            self.logger.warning(f"Could not load sounds: {e}")

    def _play_sound(self, sound_data):
        """Play audio feedback."""
        if sound_data is not None:
            try:
                sd.play(sound_data, self.config.audio.sound_sample_rate)
                sd.wait()
            except Exception as e:
                self.logger.warning(f"Could not play sound: {e}")

    def _notify(self, message: str, urgency: str = "normal"):
        """Show desktop notification."""
        if not self.config.daemon.notifications:
            return
        try:
            timeout_ms = str(self.config.wayland.notification_timeout)
            subprocess.run(
                [
                    self.config.wayland.notifier,
                    "-u",
                    urgency,
                    "Whisper",
                    message,
                    "-t",
                    timeout_ms,
                ],
                timeout=constants.NOTIFY_SEND_TIMEOUT_SECONDS,
            )
        except Exception as e:
            self.logger.warning(f"Could not show notification: {e}")

    def _pause_media_players(self) -> None:
        """Pause MPRIS players that are currently Playing (YouTube, mpv, Spotify, etc.).

        Records which players were paused so `_resume_media_players` only
        resumes those — avoiding un-pausing media the user had already paused.
        Silently no-ops if `playerctl` is missing or errors; dictation must
        not block on media control.
        """
        self._paused_players = []
        try:
            listing = subprocess.run(
                ["playerctl", "-l"],
                capture_output=True,
                text=True,
                timeout=constants.PLAYERCTL_TIMEOUT_SECONDS,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            self.logger.debug(f"playerctl unavailable, skipping media pause: {e}")
            return
        if listing.returncode != 0:
            return
        for player in filter(None, (p.strip() for p in listing.stdout.splitlines())):
            try:
                status = subprocess.run(
                    ["playerctl", "-p", player, "status"],
                    capture_output=True,
                    text=True,
                    timeout=constants.PLAYERCTL_TIMEOUT_SECONDS,
                )
                if status.returncode == 0 and status.stdout.strip() == "Playing":
                    subprocess.run(
                        ["playerctl", "-p", player, "pause"],
                        timeout=constants.PLAYERCTL_TIMEOUT_SECONDS,
                    )
                    self._paused_players.append(player)
            except subprocess.TimeoutExpired as e:
                self.logger.debug(f"playerctl timed out for {player}: {e}")

    def _resume_media_players(self) -> None:
        """Resume players that `_pause_media_players` paused on dictation start."""
        for player in self._paused_players:
            try:
                subprocess.run(
                    ["playerctl", "-p", player, "play"],
                    timeout=constants.PLAYERCTL_TIMEOUT_SECONDS,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                self.logger.debug(f"playerctl resume failed for {player}: {e}")
        self._paused_players = []

    def start_recording(
        self, mode: str | None = None, command_mode: bool = False
    ) -> str:
        """Start audio recording. `mode` overrides the session default.

        When `command_mode=True`, the daemon captures the current Wayland
        selection immediately (so a slow user doesn't lose it before
        stopping) and routes the transcribed instruction through the
        LLM voice-command path on stop.
        """
        if self.recording:
            self.logger.warning("Already recording")
            return "ALREADY_RECORDING"

        # Press-to-talk and streaming share the mic and the _type_text injector,
        # so they must never run concurrently. Stop any live stream first.
        if self.streaming:
            self.logger.info("Stopping active stream before recording")
            self.stop_streaming()

        if command_mode:
            if not self.config.llm_postprocess.command_mode.enabled:
                self.logger.warning(
                    "COMMAND received but llm_postprocess.command_mode is disabled"
                )
                return "COMMAND_MODE_DISABLED"
            if self.llm_processor is None:
                self.logger.warning(
                    "COMMAND received but LLM postprocess is not enabled"
                )
                return "LLM_DISABLED"
            self.command_mode_selection = self._capture_selection()
            self.command_mode_pending = True
            self.logger.info(
                f"Command mode: captured {len(self.command_mode_selection)} "
                "chars of selection"
            )
        else:
            self.command_mode_pending = False
            self.command_mode_selection = ""

        self.active_mode = mode if mode is not None else self.default_mode
        self.recording = True
        Path(self.config.daemon.recording_flag).touch()

        self._pause_media_players()
        self._play_sound(self.start_sound)
        self._notify("Recording started... Press SUPER+D to stop")

        threading.Thread(target=self._record_audio, daemon=True).start()

        self.logger.info("Recording started")
        return "RECORDING"

    def stop_recording(self) -> str:
        """Stop audio recording."""
        if not self.recording:
            self.logger.warning("Not recording")
            return "NOT_RECORDING"

        self.recording = False
        Path(self.config.daemon.recording_flag).unlink(missing_ok=True)

        self._play_sound(self.stop_sound)
        self._resume_media_players()
        self._notify("Recording stopped - transcribing...")

        self.logger.info("Recording stopped")
        return "STOPPED"

    # ------------------------------------------------------------------
    # Streaming lifecycle (daemon-owned). Parallels the recording lifecycle:
    # one session at a time, gated by self.streaming, with a reader thread
    # (self.stream_thread) playing the role _record_audio plays for
    # press-to-talk. Config-gated by backend.streaming_backend.
    # ------------------------------------------------------------------

    # mumble-stt sidecar wire protocol (length-prefixed binary framing).
    # Every frame: TYPE(uint8) + LEN(uint32 big-endian) + PAYLOAD(LEN bytes).
    # JSON payloads are single-line UTF-8.
    _ST_HELLO = 0x01  # JSON  daemon->sidecar  session open + params
    _ST_AUDIO = 0x02  # bytes daemon->sidecar  raw s16le mono 16k PCM chunk
    _ST_FLUSH = 0x03  # JSON  daemon->sidecar  utterance boundary
    _ST_BYE = 0x04  # JSON  daemon->sidecar  end session
    _ST_READY = 0x10  # JSON  sidecar->daemon  model loaded, ready for audio
    _ST_PARTIAL = 0x11  # JSON sidecar->daemon  in-progress hypothesis
    _ST_FINAL = 0x12  # JSON  sidecar->daemon  finalized utterance text
    _ST_ERROR = 0x13  # JSON  sidecar->daemon  recoverable/fatal error

    def toggle_streaming(self) -> str:
        """Toggle the streaming session on/off (idempotent)."""
        if self.streaming:
            return self.stop_streaming()
        return self.start_streaming()

    def start_streaming(self) -> str:
        """Start a streaming session via the configured streaming backend.

        Refuses while a press-to-talk recording is active (they share the mic
        and injector). Branches on backend.streaming_backend; on a nemotron
        start failure, falls back to the legacy whisper-stream pipeline when
        nemotron.legacy_fallback is set.
        """
        if self.recording:
            self.logger.warning("Cannot start streaming while recording")
            return "BUSY_RECORDING"
        if self.streaming:
            self.logger.warning("Already streaming")
            return "ALREADY_STREAMING"

        backend = self.config.backend.streaming_backend
        self.stream_stop.clear()
        self._live_typed = ""

        started = False
        if backend == "nemotron-streaming":
            try:
                self._start_nemotron_stream()
                started = True
            except Exception as e:
                self.logger.error(f"Nemotron streaming start failed: {e}")
                self._teardown_stream_resources()
                if self.config.nemotron.legacy_fallback:
                    self.logger.info("Falling back to legacy whisper-stream pipeline")
                    try:
                        self._start_legacy_stream()
                        started = True
                    except Exception as e2:
                        self.logger.error(f"Legacy stream fallback failed: {e2}")
                        return "STREAM_START_FAILED"
                else:
                    return "STREAM_START_FAILED"
        elif backend == "whisper-stream":
            try:
                self._start_legacy_stream()
                started = True
            except Exception as e:
                self.logger.error(f"Legacy streaming start failed: {e}")
                return "STREAM_START_FAILED"
        else:
            self.logger.error(f"Unknown streaming_backend '{backend}'")
            return "UNKNOWN_STREAMING_BACKEND"

        if not started:
            return "STREAM_START_FAILED"

        self.streaming = True
        Path(self.config.daemon.streaming_flag).touch()
        self._pause_media_players()
        self._play_sound(self.start_sound)
        self._notify("Streaming started")
        self.logger.info(f"Streaming started (backend={backend})")
        return "STREAMING"

    def stop_streaming(self) -> str:
        """Stop the streaming session and tear down all owned resources."""
        if not self.streaming:
            self.logger.warning("Not streaming")
            return "NOT_STREAMING"

        # Mark user-initiated stop first so the reader loop, on socket close,
        # knows not to clear state itself (we own the teardown here).
        self.stream_stop.set()

        # Best-effort graceful end of the sidecar session: FLUSH the trailing
        # utterance, wait briefly for the matching FINAL the reader thread will
        # inject, then BYE. Wrapped — a dead sidecar must not block stop.
        sock = self.stream_socket
        if sock is not None:
            try:
                self._send_frame(sock, self._ST_FLUSH, b"{}")
                time.sleep(constants.STREAM_STOP_SETTLE_SECONDS)
                self._send_frame(sock, self._ST_BYE, b"{}")
            except OSError as e:
                self.logger.debug(f"Sidecar BYE/FLUSH failed (already gone?): {e}")

        self._teardown_stream_resources()

        Path(self.config.daemon.streaming_flag).unlink(missing_ok=True)
        self._play_sound(self.stop_sound)
        self._resume_media_players()
        self._notify("Streaming stopped")

        self.streaming = False
        self.logger.info("Streaming stopped")
        return "STREAM_STOPPED"

    def _teardown_stream_resources(self):
        """Close the sidecar socket, join the reader thread, kill the legacy
        subprocess. Safe to call multiple times; does NOT touch the
        streaming_flag / media / sounds (callers handle user-visible state)."""
        if self.stream_socket is not None:
            try:
                self.stream_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.stream_socket.close()
            except OSError:
                pass
            self.stream_socket = None

        if self.stream_thread is not None:
            # Don't join ourselves (reader loop may call into reset on socket
            # close); only join from a different thread.
            if self.stream_thread is not threading.current_thread():
                self.stream_thread.join(timeout=constants.STREAM_THREAD_JOIN_TIMEOUT_SECONDS)
            self.stream_thread = None

        if self.legacy_stream_proc is not None:
            proc = self.legacy_stream_proc
            self.legacy_stream_proc = None
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=constants.STREAM_PROC_WAIT_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as e:
                self.logger.debug(f"Error terminating legacy stream proc: {e}")

        if self._legacy_dedup_proc is not None:
            proc = self._legacy_dedup_proc
            self._legacy_dedup_proc = None
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=constants.STREAM_PROC_WAIT_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as e:
                self.logger.debug(f"Error terminating dedup proc: {e}")

    def _send_frame(self, sock: socket.socket, frame_type: int, payload: bytes):
        """Send one length-prefixed frame to the sidecar."""
        header = struct.pack(">BI", frame_type, len(payload))
        sock.sendall(header + payload)

    def _start_nemotron_stream(self):
        """Connect to the mumble-stt sidecar, complete the HELLO/READY
        handshake, start the mic capture + audio-send, and spawn the reader
        thread. Raises on any failure so start_streaming can fall back."""
        cfg = self.config.nemotron
        socket_path = cfg.socket_path

        # Optionally spawn the sidecar if it isn't listening yet.
        if cfg.sidecar_autostart and not os.path.exists(socket_path):
            if not cfg.sidecar_cmd:
                raise RuntimeError(
                    "sidecar_autostart is true but nemotron.sidecar_cmd is empty"
                )
            self.logger.info(f"Autostarting sidecar: {cfg.sidecar_cmd}")
            # Track under legacy_stream_proc so teardown reaps it too.
            self.legacy_stream_proc = subprocess.Popen(
                cfg.sidecar_cmd, shell=True
            )

        # Connect loop bounded by connect_timeout (the socket may appear a
        # moment after the sidecar process starts).
        deadline = time.time() + cfg.connect_timeout
        sock = None
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(socket_path)
                sock = s
                break
            except OSError as e:
                last_err = e
                try:
                    s.close()
                except OSError:
                    pass
                time.sleep(constants.STREAM_CONNECT_RETRY_SECONDS)
        if sock is None:
            raise RuntimeError(
                f"Could not connect to sidecar at {socket_path}: {last_err}"
            )

        self.stream_socket = sock

        # HELLO with the audio params (must match our capture: s16le mono 16k).
        hello = {
            "v": 1,
            "sample_rate": self.config.audio.sample_rate,
            "encoding": "s16le",
            "channels": self.config.audio.channels,
            "want_partials": cfg.inject_mode == "live",
            "session_id": str(int(time.time() * 1000)),
        }
        self._send_frame(sock, self._ST_HELLO, json.dumps(hello).encode("utf-8"))

        # Wait for READY (or ERROR) within connect_timeout. Frames are
        # length-prefixed; reuse the blocking frame reader with the deadline.
        sock.settimeout(
            max(cfg.connect_timeout, constants.STREAM_HANDSHAKE_MIN_TIMEOUT_SECONDS)
        )
        frame_type, payload = self._recv_frame(sock)
        if frame_type == self._ST_ERROR:
            self._teardown_stream_resources()
            raise RuntimeError(f"Sidecar ERROR on HELLO: {payload!r}")
        if frame_type != self._ST_READY:
            self._teardown_stream_resources()
            raise RuntimeError(f"Expected READY, got frame type {frame_type:#x}")
        self.logger.info(f"Sidecar READY: {payload!r}")
        sock.settimeout(None)

        # Spawn the reader thread (sidecar->daemon: PARTIAL/FINAL/ERROR) and
        # the audio capture thread (daemon->sidecar: AUDIO frames). Both exit
        # when stream_stop is set or the socket closes.
        self.stream_thread = threading.Thread(
            target=self._nemotron_reader_loop, args=(sock,), daemon=True
        )
        self.stream_thread.start()
        threading.Thread(
            target=self._nemotron_audio_loop, args=(sock,), daemon=True
        ).start()

    def _recv_frame(self, sock: socket.socket):
        """Blocking read of one length-prefixed frame. Returns (type, payload)
        or (None, b"") if the socket closed cleanly."""
        header = self._recv_exact(sock, 5)
        if header is None:
            return None, b""
        frame_type, length = struct.unpack(">BI", header)
        payload = self._recv_exact(sock, length) if length else b""
        if payload is None:
            return None, b""
        return frame_type, payload

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int):
        """Read exactly n bytes; return None if the peer closed first."""
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _nemotron_audio_loop(self, sock: socket.socket):
        """Capture mic audio and stream it to the sidecar as AUDIO frames.

        Mirrors _record_audio's capture (s16le mono 16k via sounddevice) but
        pushes ~100ms chunks to the sidecar instead of buffering. Exits when
        stream_stop is set or the socket dies."""
        sample_rate = self.config.audio.sample_rate
        channels = self.config.audio.channels
        audio_q: queue.Queue = queue.Queue()
        gate = None
        captured_bytes = 0
        sent_bytes = 0
        if self.config.nemotron.vad_enabled:
            try:
                if sample_rate != 16000 or channels != 1:
                    raise RuntimeError(
                        "Silero VAD requires 16 kHz mono capture "
                        f"(configured: {sample_rate} Hz, {channels} channels)"
                    )
                detector = SileroVad(
                    self.config.nemotron.vad_model,
                    enter=self.config.nemotron.vad_enter,
                    exit=self.config.nemotron.vad_exit,
                    hang_windows=self.config.nemotron.vad_hang_windows,
                )
                gate = StreamingVadGate(
                    detector,
                    sample_rate=sample_rate,
                    channels=channels,
                    pre_roll_ms=self.config.nemotron.vad_pre_roll_ms,
                    trailing_ms=self.config.nemotron.vad_trailing_ms,
                )
                self.logger.info(
                    "Streaming Silero VAD enabled "
                    f"(pre-roll={self.config.nemotron.vad_pre_roll_ms} ms, "
                    f"trailing={self.config.nemotron.vad_trailing_ms} ms)"
                )
            except Exception as e:
                # Fail open: a missing/broken optimization must not lose speech.
                self.logger.error(f"Streaming VAD unavailable; sending all audio: {e}")

        def audio_callback(indata, frames, time_info, status):
            if status:
                self.logger.warning(f"Stream audio callback status: {status}")
            if not self.stream_stop.is_set():
                audio_q.put(indata.copy())

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                callback=audio_callback,
                dtype="int16",
                blocksize=int(sample_rate * constants.STREAM_AUDIO_CHUNK_SECONDS),
            ):
                while not self.stream_stop.is_set():
                    try:
                        chunk = audio_q.get(timeout=constants.STREAM_AUDIO_QUEUE_TIMEOUT_SECONDS)
                    except queue.Empty:
                        continue
                    try:
                        pcm = chunk.tobytes()
                        captured_bytes += len(pcm)
                        outgoing = gate.feed(pcm) if gate is not None else [pcm]
                        for payload in outgoing:
                            self._send_frame(sock, self._ST_AUDIO, payload)
                            sent_bytes += len(payload)
                        if gate is not None and gate.failed:
                            self.logger.error(
                                "Streaming VAD failed during capture; gate is now fail-open"
                            )
                            gate = None
                    except OSError as e:
                        self.logger.debug(f"Audio send stopped (socket gone): {e}")
                        break
        except Exception as e:
            self.logger.error(f"Streaming audio capture error: {e}")
        finally:
            if self.config.nemotron.vad_enabled:
                seconds_sent = sent_bytes / max(1, sample_rate * channels * 2)
                seconds_withheld = max(0, captured_bytes - sent_bytes) / max(
                    1, sample_rate * channels * 2
                )
                self.logger.info(
                    f"Streaming VAD audio: sent {seconds_sent:.1f}s, "
                    f"withheld {seconds_withheld:.1f}s of sustained silence"
                )

    def _nemotron_reader_loop(self, sock: socket.socket):
        """Read sidecar->daemon frames; inject FINALs (and PARTIALs if enabled).

        Exits when stream_stop is set or the socket closes (recv -> b""). If the
        socket dies WITHOUT a user-initiated stop (stream_stop unset), the
        sidecar crashed mid-session — clear the streaming_flag and reset state
        so waybar / STATUS don't get stuck on STREAMING."""
        try:
            while not self.stream_stop.is_set():
                try:
                    frame_type, payload = self._recv_frame(sock)
                except OSError:
                    break
                if frame_type is None:
                    # Socket closed by sidecar.
                    break

                live = self.config.nemotron.inject_mode == "live"

                # In live mode, drain every frame already buffered on the socket
                # and process them as one batch: partials COALESCE (only the
                # newest is rendered) so a burst of revisions costs one
                # backspace+retype instead of one per partial; finals commit
                # immediately and supersede any pending partial. This is what
                # keeps the on-screen rewrite snappy during fast speech.
                batch = [(frame_type, payload)]
                if live:
                    while True:
                        readable, _, _ = select.select([sock], [], [], 0)
                        if not readable:
                            break
                        try:
                            ft, pl = self._recv_frame(sock)
                        except OSError:
                            ft = None
                        if ft is None:
                            break
                        batch.append((ft, pl))

                pending_partial = None
                stop = False
                for ft, pl in batch:
                    if ft == self._ST_FINAL:
                        text = self._parse_segment_text(pl)
                        if text is not None:
                            pending_partial = None
                            if live:
                                self._live_commit_final(text)
                            else:
                                self._inject_stream_final(text)
                    elif ft == self._ST_PARTIAL:
                        if live:
                            text = self._parse_segment_text(pl)
                            if text is not None:
                                pending_partial = text  # coalesce to latest
                    elif ft == self._ST_ERROR:
                        self.logger.warning(f"Sidecar ERROR frame: {pl!r}")
                        try:
                            info = json.loads(pl.decode("utf-8"))
                            if info.get("fatal"):
                                stop = True
                        except (ValueError, UnicodeDecodeError):
                            pass
                    # READY/HELLO etc. mid-session are ignored.

                if pending_partial is not None:
                    self._live_update(pending_partial)
                if stop:
                    break
        finally:
            if not self.stream_stop.is_set():
                # Unexpected sidecar death: reset so the indicator isn't stuck.
                self.logger.warning("Sidecar connection closed unexpectedly")
                Path(self.config.daemon.streaming_flag).unlink(missing_ok=True)
                self.streaming = False
                self.stream_stop.set()
                self.stream_socket = None
                try:
                    self._resume_media_players()
                except Exception:
                    pass

    def _parse_segment_text(self, payload: bytes) -> str | None:
        """Extract transcript text from a sidecar JSON segment payload.

        Accepts the protocol JSON form {"type":...,"text":...}; tolerates a
        bare string payload as a fallback. Returns None on parse failure."""
        try:
            obj = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        if isinstance(obj, dict):
            return obj.get("text", "")
        if isinstance(obj, str):
            return obj
        return None

    def _inject_stream_final(self, text: str):
        """Inject a FINAL segment. Nemotron emits clean, non-overlapping,
        punctuated+capitalized finals, so no dedup is needed — just strip,
        skip empties, clean artifacts, and append via the shared _type_text
        path (which handles wtype vs clipboard-paste)."""
        if not text:
            return
        cleaned = self.config.transcription.clean_text(text).strip()
        if not cleaned:
            return
        self.logger.info(f"Stream FINAL: {cleaned[:60]}")
        self._type_text(cleaned + " ")

    def _live_update(self, target: str):
        """Make the on-screen in-progress text equal `target` (inject_mode=live).

        Diffs `target` against what we've already typed for this utterance,
        backspaces the divergent tail, and types the new tail — so a revised
        partial corrects in place instead of duplicating. Cheap when the partial
        only grows (no backspaces, just the new suffix)."""
        current = self._live_typed
        common = 0
        limit = min(len(current), len(target))
        while common < limit and current[common] == target[common]:
            common += 1
        # One batched keystroke op: backspace the divergent tail + type the new
        # suffix (no-op when the partial only grew and common == len(current)).
        self.injector.edit(len(current) - common, target[common:])
        self._live_typed = target

    def _live_commit_final(self, text: str):
        """Reconcile the in-progress partial to the FINAL, commit a separator,
        and reset for the next utterance (inject_mode=live)."""
        cleaned = self.config.transcription.clean_text(text).strip()
        # Reconcile whatever partial is on screen to the final text (this also
        # erases a partial that the model decided was noise: final == "").
        self._live_update(cleaned)
        if cleaned:
            self.logger.info(f"Stream FINAL: {cleaned[:60]}")
            self.injector.type_text(constants.LIVE_FINAL_SEPARATOR)
        self._live_typed = ""

    def _start_legacy_stream(self):
        """Spawn the legacy whisper-stream | stream_dedup.py pipeline as a
        daemon-owned subprocess (the command toggle_stream.sh used to build).
        Config-derived to avoid drift. Stored in self.legacy_stream_proc."""
        whisper_stream = str(self.config.whisper_stream_path)
        if not Path(whisper_stream).exists():
            raise RuntimeError(f"whisper-stream not found at {whisper_stream}")

        scfg = self.config.streaming
        cmd = [
            whisper_stream,
            "-m",
            str(self.config.model_path),
            "-l",
            self.config.model.language,
            "--step",
            str(scfg.step),
            "--length",
            str(scfg.buffer_length),
            "--keep",
            str(scfg.keep),
            "-vth",
            str(scfg.vad_threshold),
            "-t",
            str(scfg.threads),
        ]
        if self.config.backend.type == "cpu":
            cmd.append("--no-gpu")
        else:
            cmd.extend(["--device", str(self.config.backend.vulkan.device)])

        dedup_script = str(self.config.paths.project_dir / "stream_dedup.py")
        venv_python = str(self.config.paths.project_dir / ".venv" / "bin" / "python")

        stream_log = open(scfg.debug.stream_log, "w")
        self.logger.info(f"Legacy stream command: {' '.join(cmd)}")

        # whisper-stream stdout -> stream_dedup.py stdin. Run the dedup stage in
        # the same process group via a shell pipe so one terminate cleans up
        # both. Mirror env (Wayland) so stream_dedup's wtype works.
        whisper_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stream_log,
        )
        dedup_proc = subprocess.Popen(
            [venv_python, dedup_script],
            stdin=whisper_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Allow whisper_proc to receive SIGPIPE if dedup dies.
        if whisper_proc.stdout:
            whisper_proc.stdout.close()

        # We track the head process; terminate() on it stops capture, and the
        # dedup stage exits on the resulting EOF. Keep a reference to the dedup
        # proc so it's reaped too.
        self.legacy_stream_proc = whisper_proc
        self._legacy_dedup_proc = dedup_proc

    def _record_audio(self):
        """Record audio in background thread."""
        self.logger.info("Recording thread started")
        recorded_chunks = []
        frames_recorded = 0

        def audio_callback(indata, frames, time_info, status):
            nonlocal frames_recorded
            if status:
                self.logger.warning(f"Audio callback status: {status}")
            if self.recording:
                recorded_chunks.append(indata.copy())
                frames_recorded += frames

        sample_rate = self.config.audio.sample_rate
        channels = self.config.audio.channels
        max_seconds = self.config.audio.max_recording_seconds
        max_frames = int(max_seconds * sample_rate) if max_seconds > 0 else 0

        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            callback=audio_callback,
            dtype="int16",
        ):
            while self.recording:
                sd.sleep(constants.RECORDING_POLL_MS)
                # Hard cap on recording length. Without this, a stuck recording
                # flag (missed toggle / dropped IPC) grows recorded_chunks
                # without bound until the kernel OOM-kills the daemon. Auto-stop
                # and transcribe whatever we captured.
                if max_frames and frames_recorded >= max_frames:
                    self.logger.warning(
                        f"Recording hit {max_seconds}s cap; auto-stopping"
                    )
                    self.recording = False
                    Path(self.config.daemon.recording_flag).unlink(missing_ok=True)
                    self._resume_media_players()
                    self._notify(
                        f"Recording hit {max_seconds}s limit - transcribing...",
                        urgency="critical",
                    )

        if recorded_chunks:
            audio_data = np.concatenate(recorded_chunks, axis=0)
            self._transcribe_and_type(audio_data)
        else:
            self.logger.warning("No audio recorded")

    def _handle_command_mode(self, voice_instruction: str):
        """Apply a voice instruction to the stored selection via the LLM.

        Posts the transformed text back over the selection using the
        clipboard path (wtype is unreliable for multi-line replacements).
        Clears command-mode state regardless of outcome.
        """
        selection = self.command_mode_selection
        self.command_mode_pending = False
        self.command_mode_selection = ""

        if not selection.strip():
            self.logger.warning("Command mode: empty selection; ignoring")
            self._notify("Command mode: empty selection", urgency="critical")
            return

        outcome = self.llm_processor.transform_command(
            selected_text=selection,
            voice_instruction=voice_instruction,
            temperature=self.config.llm_postprocess.command_mode.temperature,
        )
        if outcome.error:
            self.logger.warning(
                f"Command mode LLM failed ({outcome.error}); selection untouched"
            )
            self._notify("Command mode failed", urgency="critical")
            return

        self.logger.info(
            f"Command-mode transform in {outcome.latency_ms}ms: "
            f"instr='{voice_instruction[:40]}' "
            f"sel={len(selection)}ch -> out={len(outcome.cleaned)}ch"
        )
        paste_via_clipboard(
            outcome.cleaned,
            injector=self.injector,
            wl_copy=self.config.wayland.wl_copy,
            wl_paste=self.config.wayland.wl_paste,
            logger=self.logger,
        )
        self._notify(f"Command applied: {outcome.cleaned[:40]}...", urgency="low")

    def _capture_selection(self) -> str:
        """Read the current selection for command mode.

        `selection_source = "primary"` reads the Wayland PRIMARY selection
        (text highlighted with the mouse). `"clipboard"` uses the regular
        clipboard (user must Ctrl+C first). Returns "" on any failure.
        """
        source = self.config.llm_postprocess.command_mode.selection_source
        cmd = [self.config.wayland.wl_paste, "-n"]
        if source == "primary":
            cmd.append("-p")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=constants.SELECTION_CAPTURE_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
            self.logger.warning(f"wl-paste selection capture failed: {e}")
        return ""

    def _transcribe_and_type(self, audio_data):
        """Transcribe audio and type the result."""
        sample_rate = self.config.audio.sample_rate
        self.logger.info(f"Transcribing {len(audio_data) / sample_rate:.1f}s of audio")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_file = tmp.name
            wavfile.write(temp_file, sample_rate, audio_data)

        try:
            if self.server_mode:
                text = self._transcribe_server(temp_file)
            else:
                text = self._transcribe_cli(temp_file)

            # Clean artifacts (e.g., leading --)
            if text:
                text = self.config.transcription.clean_text(text)

            # Voice-command mode: treat the transcript as an instruction
            # applied to the selection captured at START time. Paste the
            # result over the selection. Branches before the cleanup path
            # because the command flow uses its own prompt + temperature.
            if (
                text
                and self.command_mode_pending
                and self.llm_processor is not None
                and self.config.llm_postprocess.command_mode.enabled
            ):
                self._handle_command_mode(text)
                return

            # Optional LLM cleanup pass (feature-flagged off by default).
            # On any failure the processor returns the original text, so the
            # user's dictation never hard-breaks on a misconfigured endpoint.
            if text and self.llm_processor is not None:
                context_block = None
                app_class: str | None = None
                style_hint: str | None = None
                if self.config.llm_postprocess.app_context.enabled:
                    ctx = detect_app_context()
                    app_class = ctx.app_class
                    context_block = (
                        format_context_block(
                            ctx,
                            max_title_chars=self.config.llm_postprocess.app_context.max_title_chars,
                        )
                        or None
                    )
                    style_hint = select_app_style(ctx, self.config.llm_postprocess.apps)

                # Mode precedence: explicit per-turn > session default > per-app default.
                chosen_mode = (
                    self.active_mode
                    or self.default_mode
                    or resolve_mode_for_app(app_class, self.config.llm_postprocess.apps)
                )
                mode_block = resolve_mode_block(chosen_mode)
                if chosen_mode and mode_block is None:
                    self.logger.warning(
                        f"Unknown mode '{chosen_mode}'; falling back to base prompt"
                    )
                if style_hint:
                    style_block = f"App-specific style hint: {style_hint}"
                    mode_block = (
                        style_block
                        if mode_block is None
                        else f"{mode_block}\n{style_block}"
                    )

                outcome = self.llm_processor.process(
                    text,
                    glossary=self.glossary,
                    context_block=context_block,
                    mode_block=mode_block,
                    audio_file=Path(temp_file).name,
                )
                if outcome.error:
                    self.logger.warning(
                        f"LLM postprocess failed ({outcome.error}); "
                        f"using raw transcript"
                    )
                else:
                    self.logger.info(
                        f"LLM cleaned in {outcome.latency_ms}ms: "
                        f"'{text[:40]}' -> '{outcome.cleaned[:40]}'"
                    )
                text = outcome.cleaned

            if text:
                self.logger.info(f"Transcribed: {text[:50]}...")
                self._type_text(text)
                self._notify(f"Typed: {text[:40]}...", urgency="low")
            else:
                self.logger.warning("No speech detected")
                self._notify("No speech detected", urgency="critical")

        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
        finally:
            os.unlink(temp_file)
            # Per-turn mode is cleared once the turn completes so the
            # override does not leak into the next recording.
            self.active_mode = None

    def _transcribe_cli(self, audio_file: str) -> str:
        """Transcribe using whisper-cli (loads model each time)."""
        cmd = [
            str(self.config.whisper_cli_path),
            "-m",
            str(self.config.model_path),
            "-f",
            audio_file,
            "-nt",  # No timestamps
            "--no-prints",  # Minimal output
            "-t",
            str(self.config.backend.effective_threads),
        ]

        if self.config.backend.type == "vulkan":
            cmd.extend(["--device", str(self.config.backend.vulkan.device)])
        elif self.config.backend.type == "cpu":
            cmd.append("--no-gpu")

        if self.config.model.language:
            cmd.extend(["-l", self.config.model.language])

        if self.vocab_prompt:
            cmd.extend(["--prompt", self.vocab_prompt])

        self.logger.info(f"CLI command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.transcription.cli_timeout,
        )

        if result.stderr:
            self.logger.info(f"CLI stderr: {result.stderr[:500]}")

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            text_lines = [
                line.strip()
                for line in lines
                if line.strip()
                and not line.startswith("whisper_")
                and not line.startswith("system_info")
                and not line.startswith("main:")
            ]
            return " ".join(text_lines).strip()
        else:
            self.logger.error(f"Transcription failed: {result.stderr}")
            return ""

    def _transcribe_server(self, audio_file: str) -> str:
        """Transcribe using whisper-server (model stays in memory)."""
        if not HAS_REQUESTS:
            self.logger.error("Server mode requires 'requests' library")
            return ""

        try:
            with open(audio_file, "rb") as f:
                files = {"file": ("audio.wav", f, "audio/wav")}
                data = {
                    "temperature": str(self.config.transcription.temperature),
                    "temperature_inc": str(
                        self.config.transcription.temperature_increment
                    ),
                    "response_format": self.config.transcription.response_format,
                }

                if self.vocab_prompt:
                    data["prompt"] = self.vocab_prompt

                port = self.config.daemon.server_port
                response = requests.post(
                    f"http://127.0.0.1:{port}/inference",
                    files=files,
                    data=data,
                    timeout=self.config.transcription.server_timeout,
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("text", "").strip()
                else:
                    self.logger.error(f"Server returned status {response.status_code}")
                    return ""
        except Exception as e:
            self.logger.error(f"Server transcription error: {e}")
            return ""

    def _type_text(self, text: str):
        """Type text using the configured Wayland typer.

        When the text is longer than `wayland.clipboard_paste_threshold`
        (and the threshold is enabled), mumble copies via `wl-copy`,
        synthesizes Ctrl+V, and restores the previous clipboard — wtype
        drops keystrokes on long inputs. Threshold 0 keeps the legacy
        behavior (always use the typer).
        """
        threshold = self.config.wayland.clipboard_paste_threshold
        if threshold > 0 and len(text) > threshold:
            self.logger.info(
                f"Text ({len(text)} chars) exceeds threshold ({threshold}); "
                "using clipboard paste path"
            )
            paste_via_clipboard(
                text,
                injector=self.injector,
                wl_copy=self.config.wayland.wl_copy,
                wl_paste=self.config.wayland.wl_paste,
                logger=self.logger,
            )
            return
        self.injector.type_text(text)

    def handle_command(self, command: str) -> str:
        """Handle IPC command.

        Wire format (backward compatible):
          START | STOP | STATUS | TOGGLE | STREAM
          STREAM                    (parameterless toggle of live streaming)
          START mode=commit         (per-turn preset override)
          TOGGLE mode=email
          SET_MODE commit           (sets the session default)
          SET_MODE none             (clears the session default)
        """
        raw = command.strip()
        if not raw:
            return "UNKNOWN_COMMAND"
        parts = raw.split(None, 1)
        verb = parts[0].upper()
        tail = parts[1].strip() if len(parts) > 1 else ""

        mode_override: str | None = None
        if tail.lower().startswith("mode="):
            mode_override = tail.split("=", 1)[1].strip() or None

        if verb == "START":
            return self.start_recording(mode=mode_override)
        if verb == "COMMAND":
            # Voice command: if we're already recording this is a press-
            # to-stop, same as STOP. Otherwise we begin a command-mode turn.
            if self.recording:
                return self.stop_recording()
            return self.start_recording(command_mode=True)
        if verb == "STOP":
            return self.stop_recording()
        if verb == "STREAM":
            # Parameterless toggle (mirrors the old toggle_stream.sh keybind).
            return self.toggle_streaming()
        if verb == "STATUS":
            if self.streaming:
                return "STREAMING"
            base = "RECORDING" if self.recording else "READY"
            if self.default_mode:
                base = f"{base} mode={self.default_mode}"
            return base
        if verb == "TOGGLE":
            if self.recording:
                return self.stop_recording()
            return self.start_recording(mode=mode_override)
        if verb == "SET_MODE":
            new_mode = tail.strip().lower() or None
            if new_mode in (None, "none", ""):
                self.default_mode = None
                return "MODE_CLEARED"
            if new_mode not in available_modes():
                return f"UNKNOWN_MODE {new_mode}"
            self.default_mode = new_mode
            return f"MODE_SET {new_mode}"
        return "UNKNOWN_COMMAND"

    def _handle_client(self, client_socket: socket.socket):
        """Handle client connection."""
        try:
            data = client_socket.recv(constants.IPC_RECV_BUFFER_BYTES).decode()
            response = self.handle_command(data)
            client_socket.send(response.encode())
        except Exception as e:
            self.logger.error(f"Client handling error: {e}")
        finally:
            client_socket.close()

    def _start_whisper_server(self):
        """Start whisper-server subprocess."""
        if not self.server_mode:
            return

        if not HAS_REQUESTS:
            self.logger.error(
                "Server mode requires 'requests' library. "
                "Install with: uv pip install requests"
            )
            self.logger.info("Falling back to CLI mode")
            self.server_mode = False
            return

        server_bin = self.config.whisper_server_path
        if not server_bin.exists():
            self.logger.error(f"whisper-server not found at {server_bin}")
            self.logger.info("Falling back to CLI mode")
            self.server_mode = False
            return

        port = self.config.daemon.server_port
        threads = self.config.backend.effective_threads
        processors = self.config.backend.processors

        cmd = [
            str(server_bin),
            "--model",
            str(self.config.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--threads",
            str(threads),
            "--processors",
            str(processors),
            "--no-timestamps",
        ]

        if self.config.backend.type == "vulkan":
            cmd.extend(["--device", str(self.config.backend.vulkan.device)])
        elif self.config.backend.type == "cpu":
            cmd.append("--no-gpu")

        self.logger.info(
            f"Starting whisper-server on port {port} "
            f"(threads={threads}, processors={processors})..."
        )
        self.whisper_server_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Wait for server to be ready
        timeout = self.config.daemon.server_startup_timeout
        interval = self.config.daemon.server_health_check_interval
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://127.0.0.1:{port}/",
                    timeout=constants.WHISPER_SERVER_HEALTH_TIMEOUT_SECONDS,
                )
                if response.status_code in [200, 404]:
                    self.logger.info("Whisper server started successfully")
                    return
            except requests.exceptions.RequestException:
                time.sleep(interval)

        self.logger.error("Whisper server failed to start")
        self.logger.info("Falling back to CLI mode")
        self.server_mode = False
        if self.whisper_server_process:
            self.whisper_server_process.kill()
            self.whisper_server_process = None

    def start(self):
        """Start the daemon."""
        self.logger.info("Starting Whisper daemon...")

        self._start_whisper_server()

        socket_path = self.config.daemon.socket_path
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(socket_path)
        self.server_socket.listen(5)

        self.logger.info(f"Daemon listening on {socket_path}")
        mode_str = (
            "SERVER (model in memory)"
            if self.server_mode
            else "CLI (load model each time)"
        )
        self.logger.info(f"Mode: {mode_str}")
        self.logger.info("Ready for commands")

        while not self.interrupted:
            try:
                client_socket, _ = self.server_socket.accept()
                threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True,
                ).start()
            except OSError:
                if not self.interrupted:
                    self.logger.error("Socket error")
                break


def main():
    parser = argparse.ArgumentParser(description="Whisper Daemon")
    parser.add_argument(
        "--config",
        "-c",
        help="Path to config.toml (default: auto-detect)",
    )
    # Legacy CLI args still supported as overrides
    parser.add_argument("--model", "-m", help="Override model name from config")
    parser.add_argument(
        "--server-mode", action="store_true", help="Override to server mode"
    )
    parser.add_argument(
        "--no-notifications", "-n", action="store_true", help="Disable notifications"
    )
    parser.add_argument("--vocab-file", "-v", help="Override vocab file path")
    parser.add_argument(
        "--mode",
        choices=available_modes() + ["none"],
        default=None,
        help=(
            "Default LLM post-processing mode for this session "
            "(overridden per-turn via IPC `START mode=<name>`)"
        ),
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply CLI overrides
    if args.model:
        config.model.name = args.model
    if args.server_mode:
        config.daemon.mode = "server"
    if args.no_notifications:
        config.daemon.notifications = False
    if args.vocab_file:
        config.paths.vocab_file = Path(os.path.expanduser(args.vocab_file))

    logger = setup_logging(config)

    daemon = WhisperDaemon(config=config, logger=logger)
    if args.mode and args.mode != "none":
        daemon.default_mode = args.mode
        logger.info(f"Default mode: {args.mode}")
    daemon.start()


if __name__ == "__main__":
    main()
