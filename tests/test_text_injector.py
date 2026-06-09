"""Unit tests for the pluggable text injector (pure; no display server)."""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import text_injector
from text_injector import _INJECTORS, resolve_injector

LOG = logging.getLogger("test")


class TestRegistryArgv(unittest.TestCase):
    def test_every_injector_builds_type_paste_backspace(self):
        for name, spec in _INJECTORS.items():
            self.assertEqual(spec.type_argv[0], spec.bin, name)
            paste = spec.paste_argv("ctrl", "v")
            self.assertEqual(paste[0], spec.bin, name)
            bs = spec.backspace_argv(3)
            self.assertEqual(bs[0], spec.bin, name)

    def test_wtype_backspace_repeats_key(self):
        argv = _INJECTORS["wtype"].backspace_argv(3)
        self.assertEqual(argv, ["wtype", "-k", "BackSpace", "-k", "BackSpace", "-k", "BackSpace"])

    def test_xdotool_backspace_uses_repeat(self):
        argv = _INJECTORS["xdotool"].backspace_argv(5)
        self.assertEqual(argv, ["xdotool", "key", "--clearmodifiers", "--repeat", "5", "BackSpace"])

    def test_ydotool_backspace_uses_keycode(self):
        argv = _INJECTORS["ydotool"].backspace_argv(2)
        self.assertEqual(argv, ["ydotool", "key", "14:1", "14:0", "14:1", "14:0"])


class TestResolveInjector(unittest.TestCase):
    def test_explicit_known_and_present(self):
        with patch.object(text_injector.shutil, "which", return_value="/usr/bin/wtype"):
            inj = resolve_injector("wtype", LOG)
        self.assertEqual(inj.name, "wtype")

    def test_explicit_missing_falls_back_to_autodetect(self):
        # configured xdotool is absent; ydotool happens to be installed
        def which(b):
            return "/usr/bin/ydotool" if b == "ydotool" else None
        with patch.object(text_injector.shutil, "which", side_effect=which), \
             patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}, clear=True):
            inj = resolve_injector("xdotool", LOG)
        self.assertEqual(inj.name, "ydotool")

    def test_unknown_name_is_raw_command(self):
        inj = resolve_injector("mytyper", LOG)
        self.assertEqual(inj.name, "mytyper")  # raw passthrough, no spec

    def test_auto_prefers_session_appropriate(self):
        with patch.object(text_injector.shutil, "which", return_value="/usr/bin/x"), \
             patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=True):
            inj = resolve_injector("auto", LOG)
        self.assertEqual(inj.name, "xdotool")  # x11 session → xdotool first

    def test_backspace_noop_for_zero(self):
        with patch.object(text_injector.shutil, "which", return_value="/usr/bin/wtype"):
            inj = resolve_injector("wtype", LOG)
        with patch.object(text_injector.subprocess, "run") as run:
            self.assertTrue(inj.backspace(0))
            run.assert_not_called()

    def test_wtype_edit_is_single_batched_call(self):
        with patch.object(text_injector.shutil, "which", return_value="/usr/bin/wtype"):
            inj = resolve_injector("wtype", LOG)
        with patch.object(text_injector.subprocess, "run") as run:
            inj.edit(2, "ck")
            self.assertEqual(run.call_count, 1)  # one subprocess, not two
            argv = run.call_args[0][0]
            self.assertEqual(argv, ["wtype", "-k", "BackSpace", "-k", "BackSpace", "ck"])

    def test_xdotool_edit_chains_key_and_type(self):
        with patch.object(text_injector.shutil, "which", return_value="/usr/bin/xdotool"), \
             patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=True):
            inj = resolve_injector("xdotool", LOG)
        with patch.object(text_injector.subprocess, "run") as run:
            inj.edit(1, "x")
            argv = run.call_args[0][0]
            self.assertEqual(argv[0], "xdotool")
            self.assertIn("key", argv)
            self.assertIn("type", argv)

    def test_edit_noop_for_empty(self):
        with patch.object(text_injector.shutil, "which", return_value="/usr/bin/wtype"):
            inj = resolve_injector("wtype", LOG)
        with patch.object(text_injector.subprocess, "run") as run:
            self.assertTrue(inj.edit(0, ""))
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
