#!/usr/bin/env python3
"""
Whisper Daemon - Persistent dictation service with Unix socket IPC.
Supports CLI mode (model per request), server mode (persistent model),
and streaming mode (live VAD transcription with deduplication).

All settings are driven by config.toml — no magic numbers.
"""

import argparse
import logging
import os
import queue
import re
import signal
import socket
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
from config_loader import Config, load_config
from glossary import Glossary, format_whisper_prompt, load_glossary
from llm_postprocess import LLMPostProcessor

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

        # Load vocab (legacy flat prompt for Whisper --prompt)
        self.vocab_prompt = self._load_vocab()

        # LLM post-processing (optional, off by default). When enabled we also
        # parse vocab.txt into a structured Glossary for LLM hints + determ-
        # inistic substitutions; the Whisper --prompt string above is unchanged.
        self.glossary: Glossary | None = None
        self.llm_processor: LLMPostProcessor | None = None
        if config.llm_postprocess.enabled:
            self.glossary = load_glossary(config.paths.vocab_file)
            self.llm_processor = LLMPostProcessor(config.llm_postprocess, self.logger)
            self.logger.info(
                f"LLM postprocess enabled: {config.llm_postprocess.endpoint} "
                f"(model={config.llm_postprocess.model})"
            )

        # State
        self.recording = False
        self.interrupted = False
        self.audio_queue = queue.Queue()
        self.server_socket = None
        self.whisper_server_process = None

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
        """Load vocabulary prompt from file."""
        vocab_file = self.config.paths.vocab_file
        if not vocab_file or not vocab_file.exists():
            if vocab_file:
                self.logger.warning(f"Vocab file not found: {vocab_file}")
            return None

        try:
            with open(vocab_file) as f:
                words = []
                for line in f:
                    line = line.split("#")[0].strip()
                    if line:
                        words.extend(w.strip() for w in line.split(",") if w.strip())
                prompt = ", ".join(words)
                self.logger.info(f"Loaded {len(words)} vocab words from {vocab_file}")
                return prompt
        except Exception as e:
            self.logger.error(f"Failed to load vocab file: {e}")
            return None

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("Received shutdown signal")
        self.interrupted = True
        if self.server_socket:
            self.server_socket.close()
        if self.whisper_server_process:
            self.logger.info("Stopping whisper server...")
            self.whisper_server_process.terminate()
            try:
                self.whisper_server_process.wait(timeout=5)
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
                timeout=1,
            )
        except Exception as e:
            self.logger.warning(f"Could not show notification: {e}")

    def start_recording(self) -> str:
        """Start audio recording."""
        if self.recording:
            self.logger.warning("Already recording")
            return "ALREADY_RECORDING"

        self.recording = True
        Path(self.config.daemon.recording_flag).touch()

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
        self._notify("Recording stopped - transcribing...")

        self.logger.info("Recording stopped")
        return "STOPPED"

    def _record_audio(self):
        """Record audio in background thread."""
        self.logger.info("Recording thread started")
        recorded_chunks = []

        def audio_callback(indata, frames, time_info, status):
            if status:
                self.logger.warning(f"Audio callback status: {status}")
            if self.recording:
                recorded_chunks.append(indata.copy())

        sample_rate = self.config.audio.sample_rate
        channels = self.config.audio.channels

        with sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            callback=audio_callback,
            dtype="int16",
        ):
            while self.recording:
                sd.sleep(100)

        if recorded_chunks:
            audio_data = np.concatenate(recorded_chunks, axis=0)
            self._transcribe_and_type(audio_data)
        else:
            self.logger.warning("No audio recorded")

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

            # Optional LLM cleanup pass (feature-flagged off by default).
            # On any failure the processor returns the original text, so the
            # user's dictation never hard-breaks on a misconfigured endpoint.
            if text and self.llm_processor is not None:
                context_block = None
                style_block = None
                if self.config.llm_postprocess.app_context.enabled:
                    ctx = detect_app_context()
                    context_block = format_context_block(
                        ctx,
                        max_title_chars=self.config.llm_postprocess.app_context.max_title_chars,
                    ) or None
                    style = select_app_style(ctx, self.config.llm_postprocess.apps)
                    if style:
                        style_block = f"App-specific style hint: {style}"
                outcome = self.llm_processor.process(
                    text,
                    glossary=self.glossary,
                    context_block=context_block,
                    mode_block=style_block,
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
        """Type text using the configured Wayland typer."""
        typer = self.config.wayland.typer
        try:
            subprocess.run([typer, "-"], input=text, text=True, check=True, timeout=5)
            self.logger.info("Text typed successfully")
        except FileNotFoundError:
            self.logger.error(f"{typer} not found - install it for auto-typing")
        except Exception as e:
            self.logger.error(f"Typing error: {e}")

    def handle_command(self, command: str) -> str:
        """Handle IPC command."""
        command = command.strip().upper()

        if command == "START":
            return self.start_recording()
        elif command == "STOP":
            return self.stop_recording()
        elif command == "STATUS":
            return "RECORDING" if self.recording else "READY"
        elif command == "TOGGLE":
            if self.recording:
                return self.stop_recording()
            else:
                return self.start_recording()
        else:
            return "UNKNOWN_COMMAND"

    def _handle_client(self, client_socket: socket.socket):
        """Handle client connection."""
        try:
            data = client_socket.recv(1024).decode()
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
                response = requests.get(f"http://127.0.0.1:{port}/", timeout=1)
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
    daemon.start()


if __name__ == "__main__":
    main()
