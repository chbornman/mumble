"""Tests for mumble_stt.config — the sidecar's [mumble_stt] table reader.

Covers the defaults (notably the repo-relative .venv-stt python, which must
never regress to a machine-specific path), config.local.toml deep-merge, the
%t socket-path convention, and att_context label mapping.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mumble_stt.config import (
    DEFAULT_EOU_SILENCE_MS,
    DEFAULT_DEVICE,
    DEFAULT_VENV_PYTHON,
    DEFAULT_VOCAB_BIASING_ENABLED,
    SidecarConfig,
    load_sidecar_config,
)

REPO_DIR = Path(__file__).resolve().parent.parent

MINIMAL_TOML = """
[daemon]
mode = "cli"
"""


def write_config(tmpdir: str, main: str, local: str | None = None) -> str:
    path = Path(tmpdir) / "config.toml"
    path.write_text(main)
    if local is not None:
        (Path(tmpdir) / "config.local.toml").write_text(local)
    return str(path)


class TestDefaults(unittest.TestCase):
    def test_default_venv_python_is_repo_relative(self):
        # Must derive from wherever the repo lives (portable), never from a
        # path outside the repo (the old machine-specific spike venv).
        self.assertEqual(
            DEFAULT_VENV_PYTHON, str(REPO_DIR / ".venv-stt" / "bin" / "python")
        )
        self.assertNotIn("nemotron-spike", DEFAULT_VENV_PYTHON)

    def test_missing_section_yields_defaults(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = load_sidecar_config(write_config(td, MINIMAL_TOML))
        self.assertEqual(cfg.model, "nvidia/nemotron-speech-streaming-en-0.6b")
        self.assertEqual(cfg.sample_rate, 16000)
        self.assertEqual(cfg.venv_python, DEFAULT_VENV_PYTHON)
        self.assertEqual(cfg.eou_silence_ms, DEFAULT_EOU_SILENCE_MS)
        self.assertEqual(cfg.device, DEFAULT_DEVICE)
        self.assertEqual(cfg.compute_dtype, "bfloat16")
        self.assertEqual(cfg.vocab_biasing_enabled, DEFAULT_VOCAB_BIASING_ENABLED)
        self.assertEqual(cfg.vocab_file, str(Path(td) / "vocab.txt"))
        self.assertEqual(cfg.vocab_biasing_max_phrases, 1024)

    def test_socket_path_defaults_to_runtime_dir(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {"XDG_RUNTIME_DIR": "/run/user/42"}):
                cfg = load_sidecar_config(write_config(td, MINIMAL_TOML))
        self.assertEqual(cfg.socket_path, "/run/user/42/mumble-stt.sock")


class TestOverrides(unittest.TestCase):
    def test_section_values_and_percent_t(self):
        toml = MINIMAL_TOML + """
[mumble_stt]
model = "other/model"
sample_rate = 8000
venv_python = "/opt/stt/bin/python"
eou_silence_ms = 500
device = "cpu"
vocab_biasing_enabled = true
vocab_file = "terms.txt"
vocab_biasing_context_score = 0.75
vocab_biasing_depth_scaling = 1.5
vocab_biasing_alpha = 0.5
vocab_biasing_max_phrases = 99
socket_path = "%t/custom.sock"
"""
        with tempfile.TemporaryDirectory() as td:
            with patch.dict(os.environ, {"XDG_RUNTIME_DIR": "/run/user/42"}):
                cfg = load_sidecar_config(write_config(td, toml))
        self.assertEqual(cfg.model, "other/model")
        self.assertEqual(cfg.sample_rate, 8000)
        self.assertEqual(cfg.venv_python, "/opt/stt/bin/python")
        self.assertEqual(cfg.eou_silence_ms, 500)
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.compute_dtype, "float32")
        self.assertTrue(cfg.vocab_biasing_enabled)
        self.assertEqual(cfg.vocab_file, str(Path(td) / "terms.txt"))
        self.assertEqual(cfg.vocab_biasing_context_score, 0.75)
        self.assertEqual(cfg.vocab_biasing_depth_scaling, 1.5)
        self.assertEqual(cfg.vocab_biasing_alpha, 0.5)
        self.assertEqual(cfg.vocab_biasing_max_phrases, 99)
        self.assertEqual(cfg.socket_path, "/run/user/42/custom.sock")

    def test_config_local_merges_over_main(self):
        local = """
[mumble_stt]
venv_python = "/local/override/python"
"""
        with tempfile.TemporaryDirectory() as td:
            cfg = load_sidecar_config(write_config(td, MINIMAL_TOML, local))
        self.assertEqual(cfg.venv_python, "/local/override/python")

    def test_invalid_device_is_rejected(self):
        local = """
[mumble_stt]
device = "vulkan"
"""
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "expected 'cuda' or 'cpu'"):
                load_sidecar_config(write_config(td, MINIMAL_TOML, local))


class TestAttContext(unittest.TestCase):
    def cfg(self, label):
        return SidecarConfig(
            socket_path="/tmp/x.sock", model="m", sample_rate=16000,
            venv_python="py", att_context=label, log_level="info",
            eou_silence_ms=800,
            device="cuda",
            vocab_biasing_enabled=False,
            vocab_file="/tmp/vocab.txt",
            vocab_biasing_context_score=1.0,
            vocab_biasing_depth_scaling=2.0,
            vocab_biasing_alpha=1.0,
            vocab_biasing_max_phrases=1024,
        )

    def test_default_label_maps_to_low_latency(self):
        for label in ("default", "low-latency", "[70,1]", "70,1"):
            self.assertEqual(self.cfg(label).att_context_size, [70, 1])

    def test_explicit_pair_is_parsed(self):
        self.assertEqual(self.cfg("70,13").att_context_size, [70, 13])

    def test_garbage_falls_back(self):
        self.assertEqual(self.cfg("wat").att_context_size, [70, 1])


if __name__ == "__main__":
    unittest.main()
