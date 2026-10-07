"""Tests for the silence-hallucination guards.

Whisper hallucinates on silent or near-silent audio. This used to reach the
user's keyboard verbatim: a press-to-talk toggle with no speech transcribed
"...", "C'mon, go ahead, go ahead...", "and the other one." and typed them.

The guards under test:
1. has_speech_content -- transcript-level drop for punctuation-only text.
2. WhisperDaemon._rms_dbfs -- the metric behind the pre-transcription gate.
3. WhisperDaemon._transcribe_and_type -- silent clips never reach the model.
4. _inject_stream_final / _live_commit_final -- punctuation-only streaming
   FINALs are dropped (finals mode) or erase the on-screen partial (live mode).
"""

import logging
import math
import unittest
from types import SimpleNamespace

import numpy as np

from whisper_daemon import WhisperDaemon, has_speech_content


class FakeInjector:
    def __init__(self):
        self.edits = []        # (backspaces, typed_text)
        self.typed = []        # type_text() payloads

    def edit(self, backspaces, text):
        self.edits.append((backspaces, text))
        return True

    def type_text(self, text):
        self.typed.append(text)
        return True


def make_daemon():
    """A bare object carrying only what the guarded methods touch."""
    d = WhisperDaemon.__new__(WhisperDaemon)
    d.logger = logging.getLogger("test_silence_guard")
    d.config = SimpleNamespace(
        audio=SimpleNamespace(sample_rate=16000),
        transcription=SimpleNamespace(
            clean_text=lambda t: t,
            min_rms_dbfs=-25.0,
        ),
    )
    d.server_mode = False
    d.command_mode_pending = False
    d.llm_processor = None
    d.active_mode = None
    d.default_mode = None
    d.notified = []
    d.typed = []
    d.transcribed = []
    d.injector = FakeInjector()
    d._live_typed = ""

    def _notify(msg, urgency=None):
        d.notified.append((msg, urgency))

    def _type_text(text):
        d.typed.append(text)
        return True

    def _transcribe_cli(path):
        d.transcribed.append(path)
        return "Hello world."

    d._notify = _notify
    d._type_text = _type_text
    d._transcribe_cli = _transcribe_cli
    return d


def square_wave(amp: int, n: int) -> np.ndarray:
    """An exact-RMS test signal: a +/-amp square wave has RMS == amp."""
    return np.tile(np.array([amp, -amp], dtype=np.int16), n // 2)


def room_noise_with_burst(n: int, quiet_amp: int, burst_amp: int) -> np.ndarray:
    """n samples of quiet square-wave noise with one loud 100ms window."""
    clip = square_wave(quiet_amp, n)
    clip[24000:25600] = square_wave(burst_amp, 1600)
    return clip


class TestHasSpeechContent(unittest.TestCase):
    def test_punctuation_only_is_not_speech(self):
        for t in ("", ".", "...", ". . .", "??", "—"):
            self.assertFalse(has_speech_content(t), repr(t))

    def test_letters_or_digits_are_speech(self):
        for t in ("a", "Hello.", "42", "and the other one.",
                  "C'mon, go ahead, go ahead."):
            self.assertTrue(has_speech_content(t), repr(t))


class TestClipRmsDbfs(unittest.TestCase):
    def test_silence_is_minus_infinity(self):
        d = make_daemon()
        self.assertEqual(
            d._clip_rms_dbfs(np.zeros(16000, dtype=np.int16)),
            (float("-inf"), float("-inf")))
        # 2-D (frames, channels) shape, as it arrives from the capture loop.
        self.assertEqual(
            d._clip_rms_dbfs(np.zeros((1600, 2), dtype=np.int16)),
            (float("-inf"), float("-inf")))

    def test_empty_clip_is_minus_infinity(self):
        d = make_daemon()
        self.assertEqual(
            d._clip_rms_dbfs(np.zeros(0, dtype=np.int16)),
            (float("-inf"), float("-inf")))

    def test_square_wave_level_is_exact(self):
        d = make_daemon()
        amp = 16000
        clip_dbfs, peak_dbfs = d._clip_rms_dbfs(square_wave(amp, 16000))
        want = 20.0 * math.log10(amp / 32768.0)
        self.assertAlmostEqual(clip_dbfs, want, places=5)
        self.assertAlmostEqual(peak_dbfs, want, places=5)

    def test_short_utterance_lifts_peak_but_not_clip(self):
        d = make_daemon()
        # 3s: room noise at -50.3 dBFS, one 100ms window of "speech" at
        # -12.2 dBFS. The whole-clip average sinks back below the -25 floor;
        # only the max-window metric sees the utterance.
        clip = room_noise_with_burst(48000, 100, 8000)
        clip_dbfs, peak_dbfs = d._clip_rms_dbfs(clip)
        self.assertLess(clip_dbfs, -25.0)
        self.assertGreater(peak_dbfs, -12.5)
        self.assertLess(peak_dbfs, -12.0)


class TestTranscribeAndTypeGate(unittest.TestCase):
    def test_silent_clip_never_reaches_model(self):
        d = make_daemon()
        silence = np.zeros(16000 * 3, dtype=np.int16)  # 3s of digital silence
        d._transcribe_and_type(silence)
        self.assertEqual(d.transcribed, [])
        self.assertEqual(d.typed, [])
        self.assertEqual(d.notified, [("No speech detected", "critical")])

    def test_clip_below_floor_is_gated(self):
        d = make_daemon()
        # 100/32768 -> -50.3 dBFS in every window: well under the -25 floor.
        d._transcribe_and_type(square_wave(100, 16000))
        self.assertEqual(d.transcribed, [])
        self.assertEqual(d.typed, [])

    def test_short_utterance_in_long_clip_passes_gate(self):
        d = make_daemon()
        n = 48000  # 3s of room noise with one 100ms speech window
        clip = square_wave(100, n)
        clip[24000:25600] = square_wave(8000, 1600)
        d._transcribe_and_type(clip)
        self.assertEqual(len(d.transcribed), 1)
        self.assertEqual(d.typed, ["Hello world."])

    def test_speech_level_clip_is_transcribed_and_typed(self):
        d = make_daemon()
        d._transcribe_and_type(square_wave(8000, 16000))  # ~ -12 dBFS
        self.assertEqual(len(d.transcribed), 1)
        self.assertEqual(d.typed, ["Hello world."])
        self.assertEqual(d.notified, [("Typed: Hello world....", "low")])

    def test_punctuation_only_transcript_is_dropped(self):
        d = make_daemon()

        def _hallucinating_cli(path):
            d.transcribed.append(path)
            return "..."

        d._transcribe_cli = _hallucinating_cli
        d._transcribe_and_type(square_wave(8000, 16000))
        self.assertEqual(len(d.transcribed), 1)  # the model ran
        self.assertEqual(d.typed, [])            # ...but nothing was typed
        self.assertEqual(d.notified, [("No speech detected", "critical")])


class TestStreamFinalGuards(unittest.TestCase):
    def test_finals_mode_skips_punctuation_only(self):
        d = make_daemon()
        d._inject_stream_final("...")
        d._inject_stream_final(".")
        self.assertEqual(d.typed, [])
        d._inject_stream_final("Hello.")
        self.assertEqual(d.typed, ["Hello. "])

    def test_live_mode_erases_partial_on_punctuation_final(self):
        d = make_daemon()
        d._live_update("C'mon go ahead")          # partial reached the screen
        d._live_commit_final("...")               # hallucinated final
        self.assertEqual(d._live_typed, "")
        # The partial was erased (backspace everything, type nothing) and no
        # separator was committed.
        self.assertEqual(d.injector.edits[-1], (len("C'mon go ahead"), ""))
        self.assertEqual(d.injector.typed, [])

    def test_live_mode_commits_real_final_with_separator(self):
        d = make_daemon()
        d._live_update("Hello world")
        d._live_commit_final("Hello world.")
        self.assertEqual(d.injector.typed, [" "])
        self.assertEqual(d._live_typed, "")


if __name__ == "__main__":
    unittest.main()
