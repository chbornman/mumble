"""Sidecar configuration — reads the ``[mumble_stt]`` table from the SAME
``config.toml`` the daemon uses, so there is one control plane and no second
config file.

Why not reuse ``config_loader.load_config()`` directly? That builder validates
and constructs the daemon's full typed ``Config`` (whisper paths, wayland, etc.)
and does NOT model ``[mumble_stt]``. The sidecar only needs a handful of knobs
and must stay importable on its own. So we parse the TOML here (stdlib
``tomllib``), reuse ``config_loader``'s search + deep-merge helpers so
``config.local.toml`` overrides apply identically, and read only our table.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Reuse the daemon loader's file-finding and merge logic so behavior matches
# exactly (same search order, same config.local.toml override semantics).
_THIS_DIR = Path(__file__).resolve().parent
_REPO_DIR = _THIS_DIR.parent
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

from config_loader import _find_config_file, _deep_merge  # noqa: E402

DEFAULT_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
DEFAULT_SAMPLE_RATE = 16000
# The sidecar's heavy deps live in their own venv (see mumble_stt/requirements.txt);
# default to the conventional in-repo location, overridable via [mumble_stt].venv_python.
DEFAULT_VENV_PYTHON = str(_REPO_DIR / ".venv-stt" / "bin" / "python")
DEFAULT_ATT_CONTEXT = "default"
DEFAULT_LOG_LEVEL = "info"
# End-of-utterance silence (ms) before the model commits a FINAL. Lower = finals
# fire on shorter pauses (snappier, more fragmented); higher = waits for clearer
# sentence breaks. Tunable via [mumble_stt] eou_silence_ms.
DEFAULT_EOU_SILENCE_MS = 800


@dataclass
class SidecarConfig:
    socket_path: str          # resolved absolute path to the unix socket
    model: str
    sample_rate: int
    venv_python: str
    att_context: str          # latency profile label ("default" -> [70, 1])
    log_level: str            # "debug" | "info" | "warning" | "error"
    eou_silence_ms: int       # trailing silence before a FINAL commits

    @property
    def att_context_size(self) -> list[int]:
        """Map the att_context profile label to the model's [left, right] window.

        Only the verified low-latency profile is wired for Phase 1. "default"
        and the explicit "[70,1]"/"low-latency" labels all resolve to [70, 1]
        (right context 1 -> ~80-160ms lookahead, the lowest-latency setting
        this model exposes).
        """
        label = (self.att_context or "default").strip().lower()
        if label in ("default", "low-latency", "low_latency", "[70,1]", "70,1"):
            return [70, 1]
        # Allow an explicit "L,R" form for experimentation.
        if "," in label:
            try:
                left, right = (int(x) for x in label.split(","))
                return [left, right]
            except ValueError:
                pass
        return [70, 1]


def _resolve_socket_path(raw_value: Optional[str]) -> str:
    """Resolve the socket path, expanding %t / $XDG_RUNTIME_DIR conventions.

    Default (when unset) is ``$XDG_RUNTIME_DIR/mumble-stt.sock``. We accept the
    systemd-style ``%t`` token as an alias for XDG_RUNTIME_DIR so the config
    value can mirror the unit file. Falls back to ``/run/user/<uid>`` when the
    env var is missing (e.g. a bare shell launch for debugging).
    """
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    if not raw_value:
        return os.path.join(runtime_dir, "mumble-stt.sock")
    value = raw_value.replace("%t", runtime_dir)
    value = os.path.expanduser(os.path.expandvars(value))
    return value


def load_sidecar_config(config_path: Optional[str] = None) -> SidecarConfig:
    """Load the ``[mumble_stt]`` section, with config.local.toml merged on top."""
    path = _find_config_file(config_path).resolve()
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    local_path = path.parent / "config.local.toml"
    if local_path.exists():
        with open(local_path, "rb") as f:
            local_raw = tomllib.load(f)
        raw = _deep_merge(raw, local_raw)

    section = raw.get("mumble_stt", {})
    if not isinstance(section, dict):
        section = {}

    return SidecarConfig(
        socket_path=_resolve_socket_path(section.get("socket_path")),
        model=section.get("model", DEFAULT_MODEL),
        sample_rate=int(section.get("sample_rate", DEFAULT_SAMPLE_RATE)),
        venv_python=section.get("venv_python", DEFAULT_VENV_PYTHON),
        att_context=section.get("att_context", DEFAULT_ATT_CONTEXT),
        log_level=section.get("log_level", DEFAULT_LOG_LEVEL),
        eou_silence_ms=int(section.get("eou_silence_ms", DEFAULT_EOU_SILENCE_MS)),
    )
