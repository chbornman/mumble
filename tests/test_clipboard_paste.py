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


class FakeInjector:
    """Stand-in for text_injector.Injector — records paste keystrokes."""

    def __init__(self, name: str = "wtype", paste_ok: bool = True):
        self.name = name
        self._paste_ok = paste_ok
        self.paste_calls: list[tuple[str, str]] = []

    def paste(self, modifier: str = "ctrl", key: str = "v") -> bool:
        self.paste_calls.append((modifier, key))
        return self._paste_ok


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

    def test_copies_text_dispatches_paste_and_restores(self):
        calls, fake_run = self._record_calls()
        injector = FakeInjector()
        with (
            patch("clipboard_paste.subprocess.run", side_effect=fake_run),
            patch("clipboard_paste.time.sleep", return_value=None),
        ):
            ok = paste_via_clipboard(
                "long text " * 100,
                injector=injector,
                wl_copy="wl-copy",
                wl_paste="wl-paste",
                logger=logging.getLogger("test"),
            )
        self.assertTrue(ok)
        # Order contract: snapshot, set clipboard to text, then restore.
        self.assertEqual(calls[0][0], "wl-paste")
        self.assertEqual(calls[1][0], "wl-copy")
        self.assertEqual(calls[2][0], "wl-copy")  # restore
        # The paste keystroke is dispatched through the injector (Ctrl+V),
        # between the set and the restore.
        self.assertEqual(injector.paste_calls, [("ctrl", "v")])

    def test_returns_false_when_wl_copy_missing(self):
        def fake_run(cmd, **kwargs):
            if cmd and cmd[0] == "wl-paste":
                return FakeCompleted(stdout=b"PREV")
            if cmd and cmd[0] == "wl-copy":
                raise FileNotFoundError("wl-copy")
            return FakeCompleted()

        injector = FakeInjector()
        with patch("clipboard_paste.subprocess.run", side_effect=fake_run):
            ok = paste_via_clipboard(
                "hi",
                injector=injector,
                wl_copy="wl-copy",
                wl_paste="wl-paste",
                logger=logging.getLogger("test"),
            )
        self.assertFalse(ok)
        # Aborted before dispatching the paste.
        self.assertEqual(injector.paste_calls, [])

    def test_wl_paste_unavailable_still_allows_paste(self):
        # wl-paste fails (no prior clipboard) — we should still set and
        # dispatch the paste and skip the restore without crashing.
        calls, _ = self._record_calls()
        injector = FakeInjector()

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            if cmd and cmd[0] == "wl-paste":
                raise FileNotFoundError("wl-paste")
            return FakeCompleted()

        with (
            patch("clipboard_paste.subprocess.run", side_effect=fake_run),
            patch("clipboard_paste.time.sleep", return_value=None),
        ):
            ok = paste_via_clipboard(
                "hi",
                injector=injector,
                wl_copy="wl-copy",
                wl_paste="wl-paste",
                logger=logging.getLogger("test"),
            )
        self.assertTrue(ok)
        self.assertEqual(injector.paste_calls, [("ctrl", "v")])
        verbs = [c[0] for c in calls]
        # Restore skipped when snapshot returned None.
        self.assertEqual(verbs.count("wl-copy"), 1)


if __name__ == "__main__":
    unittest.main()
