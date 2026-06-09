"""Tests for the live partial-typing path (inject_mode="live").

Exercises WhisperDaemon._live_update / _live_commit_final unbound, with a fake
injector recording the batched edit ops — no daemon construction, no audio.
The key property under test: the on-screen text is corrected with the MINIMAL
backspace+retype from the divergence point, and a FINAL that matches the
on-screen partial commits without rewriting the utterance.
"""

import logging
import unittest
from types import SimpleNamespace

import constants
from whisper_daemon import WhisperDaemon


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
    """A bare object carrying only what the live-injection methods touch."""
    d = WhisperDaemon.__new__(WhisperDaemon)
    d.injector = FakeInjector()
    d._live_typed = ""
    d.logger = logging.getLogger("test_live_injection")
    # clean_text passthrough: these tests target diffing, not artifact cleanup.
    d.config = SimpleNamespace(
        transcription=SimpleNamespace(clean_text=lambda t: t)
    )
    return d


class TestLiveUpdate(unittest.TestCase):
    def test_growing_partial_appends_only(self):
        d = make_daemon()
        d._live_update("Hello")
        d._live_update("Hello world")
        self.assertEqual(d.injector.edits, [(0, "Hello"), (0, " world")])
        self.assertEqual(d._live_typed, "Hello world")

    def test_revision_backspaces_from_divergence_point(self):
        d = make_daemon()
        d._live_update("Hello word")
        d._live_update("Hello world")
        # "Hello wor" is common; only "d" -> "ld" is corrected.
        self.assertEqual(d.injector.edits[-1], (1, "ld"))

    def test_identical_target_is_a_noop_edit(self):
        d = make_daemon()
        d._live_update("Same text")
        d._live_update("Same text")
        self.assertEqual(d.injector.edits[-1], (0, ""))

    def test_full_rewrite_only_when_first_char_differs(self):
        d = make_daemon()
        d._live_update("hello there")
        d._live_update("Hello there")
        # Case change at position 0 forces a full backspace+retype: this is
        # exactly why engine partials must arrive lstripped/consistent.
        self.assertEqual(d.injector.edits[-1], (len("hello there"), "Hello there"))


class TestLiveCommitFinal(unittest.TestCase):
    def test_final_matching_partial_does_not_rewrite(self):
        """Regression: an utterance whose FINAL equals the on-screen partial
        must commit with zero backspaces (the leading-space bug made this a
        full delete+retype every time)."""
        d = make_daemon()
        d._live_update("And I guess I'm just curious")
        d._live_commit_final("And I guess I'm just curious")
        self.assertEqual(d.injector.edits[-1], (0, ""))
        self.assertEqual(d.injector.typed, [constants.LIVE_FINAL_SEPARATOR])
        self.assertEqual(d._live_typed, "")  # reset for the next utterance

    def test_final_revises_tail_only(self):
        d = make_daemon()
        d._live_update("I saw to cats")
        d._live_commit_final("I saw two cats")
        backspaces, typed = d.injector.edits[-1]
        self.assertEqual((backspaces, typed), (len("o cats"), "wo cats"))
        self.assertEqual(d.injector.typed, [constants.LIVE_FINAL_SEPARATOR])

    def test_empty_final_erases_partial_and_skips_separator(self):
        """The model judged the audio as noise: wipe the partial, no separator."""
        d = make_daemon()
        d._live_update("uh")
        d._live_commit_final("")
        self.assertEqual(d.injector.edits[-1], (2, ""))
        self.assertEqual(d.injector.typed, [])
        self.assertEqual(d._live_typed, "")

    def test_final_is_cleaned_and_stripped(self):
        d = make_daemon()
        d.config.transcription.clean_text = lambda t: t.replace("[BLANK]", "")
        d._live_update("ok")
        d._live_commit_final("  ok[BLANK]  ")
        self.assertEqual(d.injector.edits[-1], (0, ""))
        self.assertEqual(d._live_typed, "")


if __name__ == "__main__":
    unittest.main()
