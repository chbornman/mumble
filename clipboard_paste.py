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

import constants


def _snapshot_clipboard(wl_paste: str, timeout: float = constants.CLIPBOARD_OP_TIMEOUT_SECONDS) -> bytes | None:
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


def _set_clipboard(wl_copy: str, data: bytes | str, timeout: float = constants.CLIPBOARD_OP_TIMEOUT_SECONDS) -> bool:
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
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        OSError,
    ):
        return False


def paste_via_clipboard(
    text: str,
    injector,
    wl_copy: str,
    wl_paste: str,
    logger: logging.Logger,
    ctrl_v_settle_ms: int = constants.CLIPBOARD_RESTORE_SETTLE_MS,
) -> bool:
    """Copy `text`, synthesize the paste keystroke, then restore the clipboard.

    `injector` is a text_injector.Injector — the paste keystroke is dispatched
    through it so this path honors the configured backend (wtype/xdotool/…)
    instead of assuming wtype. Returns True if the paste was dispatched; the
    restore step is best-effort.

    The settle delay must outlast the focused app's clipboard read — restoring
    too early races the paste and silently drops the text (the historical 80 ms
    default did exactly that), so it defaults conservatively to 400 ms.
    """
    saved = _snapshot_clipboard(wl_paste)
    if not _set_clipboard(wl_copy, text):
        logger.error(f"{wl_copy} failed to set clipboard; aborting paste")
        return False

    pasted = injector.paste(modifier="ctrl", key="v")
    # Give the focused app time to read the clipboard before we overwrite it.
    time.sleep(ctrl_v_settle_ms / 1000.0)
    if saved is not None and not _set_clipboard(wl_copy, saved):
        logger.warning("Could not restore previous clipboard contents")
    return pasted
