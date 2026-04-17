"""Unit tests for pure calibration helpers (no audio, no subprocess)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibrate import (
    compute_dupe_rate,
    write_config_local_overrides,
)


class TestComputeDupeRate(unittest.TestCase):
    def test_empty_hypothesis_is_zero(self):
        self.assertEqual(compute_dupe_rate("", "reference text"), 0.0)

    def test_perfect_match_is_zero(self):
        self.assertEqual(compute_dupe_rate("hello world", "hello world"), 0.0)

    def test_stream_dupe_failure_mode_flagged(self):
        # Classic stream-mode dupe: second "Inconsistent." has no ground-truth
        # justification and repeats an earlier word.
        self.assertGreater(
            compute_dupe_rate("Inconsistent. Inconsistent.", "Inconsistent."), 0.0
        )

    def test_legitimate_repeat_in_reference_is_allowed(self):
        # "the the" in the reference means two emissions are legitimate.
        hyp = "the the quick fox"
        ref = "the the quick fox"
        self.assertEqual(compute_dupe_rate(hyp, ref), 0.0)

    def test_excess_repeat_beyond_reference_is_counted(self):
        # Reference has one "the"; hypothesis has two.
        hyp = "the the quick fox"
        ref = "the quick fox"
        rate = compute_dupe_rate(hyp, ref)
        self.assertGreater(rate, 0.0)
        self.assertLess(rate, 1.0)

    def test_missing_word_is_not_a_dupe(self):
        # "world" missing entirely — that's a WER problem, not a dupe.
        self.assertEqual(compute_dupe_rate("hello", "hello world"), 0.0)


class TestWriteConfigLocalOverrides(unittest.TestCase):
    def test_writes_new_local_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_config_local_overrides(
                Path(tmpdir),
                {"keep": 250, "min_overlap_chars": 15},
            )
            text = path.read_text()
            self.assertIn("[streaming]", text)
            self.assertIn("keep = 250", text)
            self.assertIn("min_overlap_chars = 15", text)

    def test_rerun_refreshes_keys_without_duplicating(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_config_local_overrides(Path(tmpdir), {"keep": 100})
            path = write_config_local_overrides(Path(tmpdir), {"keep": 300})
            text = path.read_text()
            self.assertEqual(text.count("keep ="), 1)
            self.assertIn("keep = 300", text)
            self.assertNotIn("keep = 100", text)

    def test_preserves_unrelated_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "config.local.toml"
            p.write_text("[backend]\ntype = \"vulkan\"\n")
            write_config_local_overrides(Path(tmpdir), {"keep": 200})
            text = p.read_text()
            self.assertIn("[backend]", text)
            self.assertIn('type = "vulkan"', text)
            self.assertIn("[streaming]", text)
            self.assertIn("keep = 200", text)


if __name__ == "__main__":
    unittest.main()
