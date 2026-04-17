#!/usr/bin/env python3
"""
Stream deduplication processor for whisper-stream output.

Reads whisper-stream stdout line by line, extracts transcription blocks,
deduplicates against previously typed text, and types only new content
via wtype. Replaces the previous 279-line bash implementation.

All tunable parameters come from config.toml.
"""

import logging
import re
import subprocess
import sys
from pathlib import Path

from config_loader import load_config
from whisper_daemon import StreamDeduplicator
from word_dedup import WordLevelDeduplicator

# Load config
try:
    config = load_config()
except FileNotFoundError:
    print("Error: config.toml not found", file=sys.stderr)
    sys.exit(1)

# Set up debug logging
debug_log = config.streaming.debug.log_file
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(debug_log, mode="a")],
)
logger = logging.getLogger("stream_dedup")

# Noise filter pattern from config
noise_re = re.compile(config.streaming.noise_filter_pattern, re.IGNORECASE)

# Wayland typer command from config
typer = config.wayland.typer

# Initialize deduplicator. Legacy character-overlap class remains the
# default; users opt into the word-level rewrite via
# `streaming.legacy_dedup = false`.
if config.streaming.legacy_dedup:
    dedup = StreamDeduplicator(config, logger)
    logger.info("stream dedup: using legacy StreamDeduplicator")
else:
    dedup = WordLevelDeduplicator(config, logger)
    logger.info("stream dedup: using WordLevelDeduplicator")


def type_text(text: str) -> bool:
    """Type text using the configured Wayland typer. Returns True on success."""
    try:
        result = subprocess.run(
            [typer, text + " "],
            timeout=5,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.debug(f"Typed: '{text} '")
            return True
        else:
            logger.error(f"Typer failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Typer error: {e}")
        return False


def process_stream():
    """Read whisper-stream output from stdin and process transcription blocks."""
    in_transcription = False
    current_full_text = ""

    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")

        # Detect start of transcription block
        if "### Transcription" in line and "START" in line:
            in_transcription = True
            current_full_text = ""
            continue

        # Detect end of transcription block
        if "### Transcription" in line and "END" in line:
            in_transcription = False

            if not current_full_text:
                continue

            logger.debug(f"current_full_text: '{current_full_text}'")
            logger.debug(f"committed_text: '{dedup.committed_text}'")

            # Extract new text via deduplication
            new_text = dedup.extract_new_text(current_full_text)

            if new_text:
                logger.debug(f"Typing: '{new_text}'")
                if type_text(new_text):
                    dedup.commit(new_text)
                    logger.debug("Type succeeded")
                else:
                    logger.debug("Type FAILED")
            else:
                logger.debug("No new text to type")

            continue

        # Capture transcription lines (timestamped text)
        if in_transcription and line.startswith("["):
            # Extract text after timestamp bracket: [00:00:00.000 --> 00:00:05.000]  text
            match = re.match(r"\[.*?\]\s*(.*)", line)
            if not match:
                continue

            text = match.group(1).strip()
            logger.debug(f"Captured: '{text}'")

            # Filter noise artifacts like (background noise), [music], *cough*
            if noise_re.match(text):
                logger.debug(f"Filtered noise: '{text}'")
                continue

            if text:
                if current_full_text:
                    current_full_text += " " + text
                else:
                    current_full_text = text


if __name__ == "__main__":
    try:
        process_stream()
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        pass
