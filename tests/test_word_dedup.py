"""Unit tests for the word-level stream deduplicator."""

from __future__ import annotations

import logging
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from word_dedup import WordLevelDeduplicator


@dataclass
class _StreamCfg:
    word_overlap_lookback: int = 15
    immediate_repeat_window: int = 8
    max_committed_words: int = 200


class _Cfg:
    def __init__(self, **kw):
        self.streaming = _StreamCfg(**kw)


def make_dedup(**kw) -> WordLevelDeduplicator:
    return WordLevelDeduplicator(_Cfg(**kw), logging.getLogger("test"))


class TestFirstBlock(unittest.TestCase):
    def test_emits_all_on_empty_state(self):
        d = make_dedup()
        self.assertEqual(d.extract_new_text("hello world"), "hello world")


class TestSuffixPrefixOverlap(unittest.TestCase):
    def test_exact_rolling_buffer_overlap(self):
        d = make_dedup()
        d.commit("The quick brown fox")
        # Next block extends and repeats the last 2 words as the buffer overlap.
        out = d.extract_new_text("brown fox jumps over the lazy dog")
        self.assertEqual(out, "jumps over the lazy dog")

    def test_case_and_punctuation_insensitive_match(self):
        d = make_dedup()
        d.commit("hello there")
        out = d.extract_new_text("There, friend.")
        self.assertEqual(out, "friend.")

    def test_rewording_tolerated_via_tail_search(self):
        # Whisper rewrites one of the committed words; exact prefix match
        # fails but the tail-search fallback still finds the boundary.
        d = make_dedup()
        d.commit("the server is down")
        out = d.extract_new_text("down again shortly")
        self.assertEqual(out, "again shortly")


class TestImmediateRepeatSuppression(unittest.TestCase):
    def test_trims_leading_repeat_when_no_overlap(self):
        d = make_dedup()
        d.commit("and then we shipped the feature")
        # New block starts with a chunk that matches the committed tail —
        # the legacy class would re-emit "shipped the feature"; the word
        # dedup must suppress it.
        out = d.extract_new_text("shipped the feature on Friday")
        self.assertEqual(out, "on Friday")

    def test_pause_resume_dupe_failure_mode(self):
        # Stream-mode signature failure: "... and then" emits; user
        # resumes "and then we shipped"; second block transcribes
        # "and then we shipped".
        d = make_dedup()
        d.commit("we deployed the build and then")
        out = d.extract_new_text("and then we shipped")
        self.assertEqual(out, "we shipped")

    def test_no_false_suppression_on_unrelated_words(self):
        d = make_dedup()
        d.commit("alpha beta gamma")
        out = d.extract_new_text("delta epsilon")
        self.assertEqual(out, "delta epsilon")


class TestCommitAndReset(unittest.TestCase):
    def test_committed_text_joins_tokens(self):
        d = make_dedup()
        d.commit("one two")
        d.commit("three")
        self.assertEqual(d.committed_text, "one two three")

    def test_reset_clears_state(self):
        d = make_dedup()
        d.commit("one two three")
        d.reset()
        self.assertEqual(d.committed_text, "")
        # After reset, the next call emits all words.
        self.assertEqual(d.extract_new_text("anything"), "anything")

    def test_committed_trimmed_to_cap(self):
        d = make_dedup(max_committed_words=5)
        d.commit("a b c d e f g h")
        self.assertEqual(
            d.committed_text.split(),
            ["d", "e", "f", "g", "h"],
        )


if __name__ == "__main__":
    unittest.main()
