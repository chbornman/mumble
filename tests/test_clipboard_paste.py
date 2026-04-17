"""Unit tests for the clipboard-paste fallback (pure, no display server)."""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clipboard_paste import paste_via_clipboard


class FakeCompleted:
    def __init__(self, stdout: bytes = b"", returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


class TestPasteViaClipboard(unittest.TestCase):
    def _record_calls(self):
        calls: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            # wl-paste read path
            if cmd and cmd[0] == "wl-paste":
                return FakeCompleted(stdout=b"PREV_CLIP")
            return FakeCompleted()

        return calls, fake_run

    def test_copies_text_dispatches_ctrl_v_and_restores(self):
        calls, fake_run = self._record_calls()
        with patch("clipboard_paste.subprocess.run", side_effect=fake_run), \
             patch("clipboard_paste.time.sleep", return_value=None):
            ok = paste_via_clipboard(
                "long text " * 100,
                typer="wtype",
                wl_copy="wl-copy",
                wl_paste="wl-paste",
                logger=logging.getLogger("test"),
            )
        self.assertTrue(ok)
        # Order contract: snapshot, set clipboard to text, Ctrl+V, restore.
        self.assertEqual(calls[0][0], "wl-paste")
        self.assertEqual(calls[1][0], "wl-copy")
        self.assertEqual(calls[2][0], "wtype")
        self.assertIn("-M", calls[2])
        self.assertIn("ctrl", calls[2])
        self.assertIn("v", calls[2])
        self.assertEqual(calls[3][0], "wl-copy")  # restore

    def test_returns_false_when_wl_copy_missing(self):
        def fake_run(cmd, **kwargs):
            if cmd and cmd[0] == "wl-paste":
                return FakeCompleted(stdout=b"PREV")
            if cmd and cmd[0] == "wl-copy":
                raise FileNotFoundError("wl-copy")
            return FakeCompleted()

        with patch("clipboard_paste.subprocess.run", side_effect=fake_run):
            ok = paste_via_clipboard(
                "hi",
                typer="wtype",
                wl_copy="wl-copy",
                wl_paste="wl-paste",
                logger=logging.getLogger("test"),
            )
        self.assertFalse(ok)

    def test_wl_paste_unavailable_still_allows_paste(self):
        # wl-paste fails (no prior clipboard) — we should still set and
        # dispatch Ctrl+V and skip the restore without crashing.
        calls, _ = self._record_calls()

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            if cmd and cmd[0] == "wl-paste":
                raise FileNotFoundError("wl-paste")
            return FakeCompleted()

        with patch("clipboard_paste.subprocess.run", side_effect=fake_run), \
             patch("clipboard_paste.time.sleep", return_value=None):
            ok = paste_via_clipboard(
                "hi",
                typer="wtype",
                wl_copy="wl-copy",
                wl_paste="wl-paste",
                logger=logging.getLogger("test"),
            )
        self.assertTrue(ok)
        verbs = [c[0] for c in calls]
        # Restore skipped when snapshot returned None.
        self.assertEqual(verbs.count("wl-copy"), 1)


if __name__ == "__main__":
    unittest.main()
