"""
Clipboard-paste fallback for long transcriptions (Wayland).

`wtype` gets flaky when the input is thousands of characters long —
keystrokes drop, ordering drifts, trailing characters land in the wrong
order. Past a configurable threshold, mumble instead copies the text
with `wl-copy`, synthesizes Ctrl+V via wtype, and restores the user's
previous clipboard contents so nothing they'd staged is destroyed.

Feature-scoped: only consumed when
`wayland.clipboard_paste_threshold > 0`. The previous typer path
remains the default for backwards-compatibility.
"""

from __future__ import annotations

import logging
import subprocess
import time


def _snapshot_clipboard(wl_paste: str, timeout: float = 1.0) -> bytes | None:
    """Return the current clipboard bytes, or None if unavailable.

    Uses `-n` to avoid a trailing newline injection and `-t` is omitted so
    we get whatever mime type wl-paste picks — restoring the same bytes
    keeps simple text/URI/image clipboards working for the common case.
    """
    try:
        result = subprocess.run(
            [wl_paste, "-n"],
            capture_output=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _set_clipboard(wl_copy: str, data: bytes | str, timeout: float = 1.0) -> bool:
    """Write bytes/str to the clipboard via wl-copy. Returns success."""
    try:
        if isinstance(data, str):
            subprocess.run(
                [wl_copy],
                input=data,
                text=True,
                check=True,
                timeout=timeout,
            )
        else:
            subprocess.run(
                [wl_copy],
                input=data,
                check=True,
                timeout=timeout,
            )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        return False


def paste_via_clipboard(
    text: str,
    typer: str,
    wl_copy: str,
    wl_paste: str,
    logger: logging.Logger,
    ctrl_v_settle_ms: int = 80,
) -> bool:
    """Copy `text`, synthesize Ctrl+V, then restore the previous clipboard.

    Returns True if the Ctrl+V was dispatched; the restore step is
    best-effort (always attempted, but failure is logged rather than
    propagated since the paste already happened).
    """
    saved = _snapshot_clipboard(wl_paste)
    if not _set_clipboard(wl_copy, text):
        logger.error(f"{wl_copy} failed to set clipboard; aborting paste")
        return False

    try:
        # wtype -M / -m dispatches a modified keystroke. -M press, key, -m release.
        subprocess.run(
            [typer, "-M", "ctrl", "v", "-m", "ctrl"],
            check=True,
            timeout=5,
        )
    except FileNotFoundError:
        logger.error(f"{typer} not found — cannot synthesize Ctrl+V")
        # Best-effort clipboard restore below still runs.
    except Exception as e:
        logger.error(f"Ctrl+V synthesis failed: {e}")
    finally:
        # Give the focused app a moment to read the clipboard before we
        # overwrite it with the restored value.
        time.sleep(ctrl_v_settle_ms / 1000.0)
        if saved is not None and not _set_clipboard(wl_copy, saved):
            logger.warning("Could not restore previous clipboard contents")
    return True
