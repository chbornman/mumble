"""Silero voice-activity detection and a loss-resistant streaming audio gate.

The gate intentionally does not decide utterance boundaries. It ships pre-roll
when speech begins and trailing silence after speech ends, leaving Nemotron's
endpointer enough received audio to produce FINAL events. Only sustained idle
silence is withheld.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 512
CONTEXT_SAMPLES = 64
STATE_SHAPE = (2, 1, 128)


@dataclass(frozen=True)
class VadResult:
    gate_open: bool
    speech_started: bool = False
    speech_ended: bool = False
    probability: float = 0.0


class VoiceDetector(Protocol):
    def process_pcm16(self, pcm: bytes) -> VadResult: ...


class SileroVad:
    """Silero VAD v5 ONNX runner using the model's required 64-sample context."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        enter: float = 0.5,
        exit: float = 0.35,
        hang_windows: int = 15,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "VAD is enabled but onnxruntime is not installed in mumble's venv"
            ) from exc

        path = Path(model_path)
        if not path.is_file():
            raise RuntimeError(f"Silero VAD model not found: {path}")
        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.enter = enter
        self.exit = exit
        self.hang_windows = hang_windows
        self.state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self.context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)
        self.in_speech = False
        self.below_exit = 0

    def _window_probability(self, window: np.ndarray) -> float:
        model_input = np.concatenate((self.context, window))[None, :]
        self.context = window[-CONTEXT_SAMPLES:].copy()
        probability, self.state = self.session.run(
            ["output", "stateN"],
            {
                "input": model_input,
                "state": self.state,
                "sr": np.asarray(SAMPLE_RATE, dtype=np.int64),
            },
        )
        return float(np.asarray(probability).reshape(-1)[0])

    def process_pcm16(self, pcm: bytes) -> VadResult:
        if len(pcm) % 2:
            raise ValueError("PCM s16le audio must contain complete samples")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if self.pending.size:
            samples = np.concatenate((self.pending, samples))

        started = False
        ended = False
        latest_probability = 0.0
        offset = 0
        while samples.size - offset >= WINDOW_SAMPLES:
            window = samples[offset : offset + WINDOW_SAMPLES]
            offset += WINDOW_SAMPLES
            latest_probability = self._window_probability(window)
            if self.in_speech:
                if latest_probability >= self.exit:
                    self.below_exit = 0
                else:
                    self.below_exit += 1
                    if self.below_exit >= self.hang_windows:
                        self.in_speech = False
                        self.below_exit = 0
                        ended = True
            elif latest_probability >= self.enter:
                self.in_speech = True
                self.below_exit = 0
                started = True

        self.pending = samples[offset:].copy()
        return VadResult(self.in_speech, started, ended, latest_probability)


class StreamingVadGate:
    """Select capture chunks to send, with pre-roll, hangover, and fail-open."""

    def __init__(
        self,
        detector: VoiceDetector,
        *,
        sample_rate: int = SAMPLE_RATE,
        channels: int = 1,
        pre_roll_ms: int = 1000,
        trailing_ms: int = 3000,
    ) -> None:
        self.detector = detector
        self.bytes_per_ms = sample_rate * channels * 2 / 1000
        self.pre_roll_bytes = int(self.bytes_per_ms * pre_roll_ms)
        self.trailing_bytes = int(self.bytes_per_ms * trailing_ms)
        self.pre_roll: deque[bytes] = deque()
        self.pre_roll_size = 0
        self.trailing_remaining = 0
        self.failed = False

    def _hold(self, chunk: bytes) -> None:
        if self.pre_roll_bytes == 0:
            return
        self.pre_roll.append(chunk)
        self.pre_roll_size += len(chunk)
        while self.pre_roll and self.pre_roll_size > self.pre_roll_bytes:
            self.pre_roll_size -= len(self.pre_roll.popleft())

    def feed(self, chunk: bytes) -> list[bytes]:
        """Return zero or more chunks to ship for this captured audio chunk."""
        if self.failed:
            return [chunk]
        try:
            result = self.detector.process_pcm16(chunk)
        except Exception:
            # Optimization must never turn a VAD fault into lost dictation.
            self.failed = True
            held = list(self.pre_roll)
            self.pre_roll.clear()
            self.pre_roll_size = 0
            return [*held, chunk]

        if result.gate_open:
            self.trailing_remaining = self.trailing_bytes
            held = list(self.pre_roll)
            self.pre_roll.clear()
            self.pre_roll_size = 0
            return [*held, chunk]

        if self.trailing_remaining > 0:
            self.trailing_remaining = max(0, self.trailing_remaining - len(chunk))
            return [chunk]

        self._hold(chunk)
        return []
