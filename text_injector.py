"""Pluggable text injection — how transcribed text lands in the focused window.

mumble needs exactly ONE working injector. Which one fits depends on your
session, so this is a swappable backend like the ASR engines:

    wtype     Wayland (default)
    ydotool   Wayland *or* X11 (via the uinput kernel device; needs ydotoold)
    xdotool   X11

Choose with ``wayland.typer`` in config.toml — a tool name, or ``"auto"`` to
pick from the session type ($XDG_SESSION_TYPE / $WAYLAND_DISPLAY) and what's
installed. Add a new injector by appending one entry to ``_INJECTORS`` (the
command to type stdin, and the command to send a modified keystroke). Nothing
else in mumble needs to know how injection works.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

import constants


@dataclass(frozen=True)
class _Spec:
    bin: str                                     # binary that must be on PATH
    type_argv: list[str]                         # types stdin into the window
    paste_argv: Callable[[str, str], list[str]]  # (modifier, key) -> keystroke argv
    backspace_argv: Callable[[int], list[str]]   # n -> argv sending n Backspaces
    sessions: tuple[str, ...]                    # XDG_SESSION_TYPE values it suits


# Registry. type_argv reads the text on stdin; paste_argv builds a modified
# keystroke (e.g. Ctrl+V); backspace_argv sends N Backspaces (for live partial
# correction). To support another tool, add an entry here.
_INJECTORS: dict[str, _Spec] = {
    "wtype": _Spec(
        bin="wtype",
        type_argv=["wtype", "-"],
        paste_argv=lambda mod, key: ["wtype", "-M", mod, key, "-m", mod],
        backspace_argv=lambda n: ["wtype", *(["-k", "BackSpace"] * n)],
        sessions=("wayland",),
    ),
    "ydotool": _Spec(
        bin="ydotool",
        type_argv=["ydotool", "type", "--file", "-"],
        # ydotool uses key *names* via `key`; modifier+key chord syntax.
        paste_argv=lambda mod, key: ["ydotool", "key", f"{mod}+{key}"],
        # Linux keycode 14 = Backspace; "14:1 14:0" is press then release.
        backspace_argv=lambda n: ["ydotool", "key", *(["14:1", "14:0"] * n)],
        sessions=("wayland", "x11", "tty"),
    ),
    "xdotool": _Spec(
        bin="xdotool",
        type_argv=["xdotool", "type", "--clearmodifiers", "--file", "-"],
        paste_argv=lambda mod, key: ["xdotool", "key", "--clearmodifiers", f"{mod}+{key}"],
        backspace_argv=lambda n: ["xdotool", "key", "--clearmodifiers", "--repeat", str(n), "BackSpace"],
        sessions=("x11",),
    ),
}


class Injector:
    """A resolved text injector. Wraps one registry entry (or a raw command)."""

    def __init__(self, name: str, spec: Optional[_Spec], logger: logging.Logger):
        self.name = name
        self._spec = spec
        self._logger = logger

    def type_text(self, text: str, timeout: float = constants.INJECTOR_TYPE_TIMEOUT_SECONDS) -> bool:
        """Type ``text`` into the focused window. Returns success."""
        argv = self._spec.type_argv if self._spec else [self.name, "-"]
        try:
            subprocess.run(argv, input=text, text=True, check=True, timeout=timeout)
            self._logger.info("Text typed successfully")
            return True
        except FileNotFoundError:
            self._logger.error(
                f"Injector '{self.name}' not found — install it "
                f"(see README Requirements) or set wayland.typer to another."
            )
        except Exception as e:
            self._logger.error(f"Text injection failed ({self.name}): {e}")
        return False

    def edit(self, backspaces: int, text: str, timeout: float = constants.INJECTOR_EDIT_TIMEOUT_SECONDS) -> bool:
        """Backspace `backspaces` chars then type `text` in a SINGLE keystroke
        batch where the backend supports it — one subprocess spawn instead of
        two, which keeps live partial correction snappy. Falls back to separate
        backspace()+type_text() for backends that can't chain (ydotool, raw).
        """
        if backspaces <= 0 and not text:
            return True
        argv = None
        # wtype processes args left-to-right: each `-k BackSpace` then the
        # positional text. Skip the combined form if text could be read as a
        # flag (leading '-') — transcripts effectively never start with one.
        if self.name == "wtype" and (not text or not text.startswith("-")):
            argv = ["wtype", *(["-k", "BackSpace"] * backspaces)]
            if text:
                argv.append(text)
        elif self.name == "xdotool":
            argv = ["xdotool"]
            if backspaces > 0:
                argv += ["key", "--clearmodifiers", "--repeat", str(backspaces), "BackSpace"]
            if text:
                argv += ["type", "--clearmodifiers", "--", text]
        if argv is None:
            ok = self.backspace(backspaces)
            return (self.type_text(text) if text else True) and ok
        try:
            subprocess.run(argv, check=True, timeout=timeout)
            return True
        except FileNotFoundError:
            self._logger.error(f"Injector '{self.name}' not found — cannot edit")
        except Exception as e:
            self._logger.error(f"Live edit failed ({self.name}): {e}")
        return False

    def backspace(self, n: int, timeout: float = constants.INJECTOR_EDIT_TIMEOUT_SECONDS) -> bool:
        """Send `n` Backspace keystrokes. Used by live partial typing to erase
        the divergent tail before retyping a revised hypothesis. Returns success.
        """
        if n <= 0:
            return True
        if self._spec:
            argv = self._spec.backspace_argv(n)
        elif self.name == "wtype":
            argv = ["wtype", *(["-k", "BackSpace"] * n)]
        else:
            self._logger.error(
                f"Injector '{self.name}' can't synthesize Backspace; "
                f"inject_mode='live' needs wtype/xdotool/ydotool"
            )
            return False
        try:
            subprocess.run(argv, check=True, timeout=timeout)
            return True
        except FileNotFoundError:
            self._logger.error(f"Injector '{self.name}' not found — cannot backspace")
        except Exception as e:
            self._logger.error(f"Backspace failed ({self.name}): {e}")
        return False

    def paste(self, modifier: str = "ctrl", key: str = "v", timeout: float = constants.INJECTOR_EDIT_TIMEOUT_SECONDS) -> bool:
        """Synthesize a modified keystroke (default Ctrl+V). Returns success.

        ``modifier``/``key`` are parameterized so callers can send Ctrl+Shift+V
        for terminals later without touching this module.
        """
        if self._spec:
            argv = self._spec.paste_argv(modifier, key)
        elif self.name == "wtype":
            argv = ["wtype", "-M", modifier, key, "-m", modifier]
        else:
            self._logger.error(
                f"Injector '{self.name}' has no known paste-keystroke form"
            )
            return False
        try:
            subprocess.run(argv, check=True, timeout=timeout)
            return True
        except FileNotFoundError:
            self._logger.error(f"Injector '{self.name}' not found — cannot paste")
        except Exception as e:
            self._logger.error(f"Paste keystroke failed ({self.name}): {e}")
        return False


def resolve_injector(configured: str, logger: logging.Logger) -> Injector:
    """Pick a text injector from the ``wayland.typer`` config value.

    - A known tool name ("wtype"/"ydotool"/"xdotool") uses that, warning if its
      binary is missing.
    - "auto" detects from the session type and what's installed.
    - Any other value is treated as a raw typer command (back-compat): the text
      is piped to ``<cmd> -``.
    """
    name = (configured or "auto").strip()

    if name == "auto":
        return _autodetect(logger)

    spec = _INJECTORS.get(name)
    if spec is not None:
        if shutil.which(spec.bin) is None:
            logger.warning(
                f"Configured injector '{name}' is not installed; "
                f"falling back to autodetect."
            )
            return _autodetect(logger)
        return Injector(name, spec, logger)

    # Unknown name -> raw command, for forward/backward compatibility.
    logger.info(f"Using custom typer command '{name}' (raw stdin pipe).")
    return Injector(name, None, logger)


def _autodetect(logger: logging.Logger) -> Injector:
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if not session:
        session = "wayland" if os.environ.get("WAYLAND_DISPLAY") else "x11"

    # Prefer an injector that suits the session; among those, prefer the more
    # session-specific one (e.g. on X11 pick xdotool over the multi-session
    # ydotool, which needs a daemon + uinput perms). Then fall back to any.
    ordered = sorted(
        _INJECTORS.items(),
        key=lambda kv: (0 if session in kv[1].sessions else 1, len(kv[1].sessions)),
    )
    for n, spec in ordered:
        if shutil.which(spec.bin) is not None:
            logger.info(f"Auto-selected text injector '{n}' (session={session or '?'}).")
            return Injector(n, spec, logger)

    logger.error(
        "No text injector found. Install one of: wtype (Wayland), "
        "xdotool (X11), or ydotool. See README Requirements."
    )
    # Return a wtype injector so the daemon stays up; type_text will log the
    # not-found error per attempt rather than crashing the daemon.
    return Injector("wtype", _INJECTORS["wtype"], logger)
