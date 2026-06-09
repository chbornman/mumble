"""Tests for StreamingEngine._run_chunk event emission (no NeMo required).

The engine's heavy deps are injected lazily, so we build an instance via
__new__ with numpy real and torch/pipeline mocked, and drive _run_chunk with
synthetic pipeline outputs. The properties under test:

- finals are stripped; partials are LSTRIPPED at the source. NeMo detokenizes
  partials "without stripping", so the leading SentencePiece underscore becomes
  a leading space — if that ever reaches the daemon again, the live prefix-diff
  degenerates to a full delete+retype on every utterance (the bug fixed on
  2026-06-09).
- unchanged partials are deduplicated (no event), finals reset the dedup state.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from mumble_stt.engine import EngineError, StreamingEngine

CHUNK_SAMPLES = 2560
CHUNK_BYTES = bytes(CHUNK_SAMPLES * 2)  # silence, s16le


def make_engine(outputs):
    """Engine with mocked heavy deps; transcribe_step yields `outputs` in order."""
    eng = StreamingEngine.__new__(StreamingEngine)
    eng._np = np
    eng._torch = MagicMock()
    eng._Frame = MagicMock()
    eng._ASRRequestOptions = MagicMock()
    eng._pipeline = MagicMock()
    eng._pipeline.transcribe_step.side_effect = [[o] for o in outputs]
    eng._chunk_samples = CHUNK_SAMPLES
    eng._stream_id = 0
    eng._device = "cpu"
    eng._first_chunk_pending = True
    eng._last_partial = ""
    return eng


def out(partial=None, final=None):
    return SimpleNamespace(partial_transcript=partial, final_transcript=final)


class TestRunChunkEvents(unittest.TestCase):
    def test_partial_is_lstripped(self):
        """Leading SentencePiece space must not leave the engine."""
        eng = make_engine([out(partial=" And I guess")])
        ev = eng._run_chunk(CHUNK_BYTES, is_last=False)
        self.assertEqual(ev.kind, "partial")
        self.assertEqual(ev.text, "And I guess")

    def test_final_is_stripped_and_resets_partial_dedup(self):
        eng = make_engine([
            out(partial=" hello"),
            out(final="  Hello.  "),
            out(partial=" hello"),  # same raw partial again, new utterance
        ])
        self.assertEqual(eng._run_chunk(CHUNK_BYTES, is_last=False).text, "hello")
        ev = eng._run_chunk(CHUNK_BYTES, is_last=False)
        self.assertEqual((ev.kind, ev.text), ("final", "Hello."))
        # After a final, the identical partial must be emitted again.
        ev = eng._run_chunk(CHUNK_BYTES, is_last=False)
        self.assertEqual((ev.kind, ev.text), ("partial", "hello"))

    def test_unchanged_partial_is_deduplicated(self):
        eng = make_engine([out(partial=" same"), out(partial=" same")])
        self.assertIsNotNone(eng._run_chunk(CHUNK_BYTES, is_last=False))
        self.assertIsNone(eng._run_chunk(CHUNK_BYTES, is_last=False))

    def test_no_output_yields_none(self):
        eng = make_engine([out()])
        self.assertIsNone(eng._run_chunk(CHUNK_BYTES, is_last=False))

    def test_short_chunk_is_padded_not_crashed(self):
        eng = make_engine([out(partial=" tail")])
        ev = eng._run_chunk(CHUNK_BYTES[: CHUNK_SAMPLES], is_last=True)
        self.assertEqual(ev.text, "tail")

    def test_pipeline_exception_maps_to_engine_error(self):
        eng = make_engine([out()])
        eng._pipeline.transcribe_step.side_effect = RuntimeError("cuda oom")
        with self.assertRaises(EngineError):
            eng._run_chunk(CHUNK_BYTES, is_last=False)


if __name__ == "__main__":
    unittest.main()
