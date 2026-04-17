"""Unit tests for preset/mode composition."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modes import (
    BUILTIN_MODES,
    available_modes,
    resolve_mode_block,
    resolve_mode_for_app,
)


class TestResolveModeBlock(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(resolve_mode_block(None))
        self.assertIsNone(resolve_mode_block(""))
        self.assertIsNone(resolve_mode_block("  "))
        self.assertIsNone(resolve_mode_block("none"))
        self.assertIsNone(resolve_mode_block("NONE"))

    def test_unknown_returns_none(self):
        self.assertIsNone(resolve_mode_block("does-not-exist"))

    def test_known_modes_return_nonempty(self):
        for name in available_modes():
            block = resolve_mode_block(name)
            self.assertIsNotNone(block)
            self.assertTrue(len(block) > 0)

    def test_case_insensitive(self):
        self.assertEqual(resolve_mode_block("COMMIT"), BUILTIN_MODES["commit"])
        self.assertEqual(resolve_mode_block("Email"), BUILTIN_MODES["email"])

    def test_commit_mode_shape(self):
        block = resolve_mode_block("commit")
        # Commit preset must mention the conventional-commit format somehow.
        self.assertIn("Conventional Commit", block)


class TestResolveModeForApp(unittest.TestCase):
    def test_missing_config_returns_none(self):
        self.assertIsNone(resolve_mode_for_app("Alacritty", None))
        self.assertIsNone(resolve_mode_for_app("Alacritty", {}))
        self.assertIsNone(resolve_mode_for_app(None, {"x": {"mode": "commit"}}))

    def test_returns_configured_mode(self):
        apps = {"thunderbird": {"mode": "email"}}
        self.assertEqual(resolve_mode_for_app("thunderbird", apps), "email")

    def test_ignores_non_string_mode(self):
        apps = {"thunderbird": {"mode": 42}}
        self.assertIsNone(resolve_mode_for_app("thunderbird", apps))


if __name__ == "__main__":
    unittest.main()
