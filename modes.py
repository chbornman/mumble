"""
Preset / mode prompt transforms for LLM post-processing.

Each mode appends a short, composable instruction to the base cleanup
prompt — the layering pattern is lifted from VoiceTypr's
`build_enhancement_prompt`:

    final_prompt = base_prompt + mode_transform + "Transcribed text:" + text

Feature-scoped: only consumed when `[llm_postprocess]` is enabled. When
no mode is selected (None, "none", or an unknown name) the base cleanup
prompt is unchanged.
"""

from __future__ import annotations

# Built-in presets. Keep each transform a single short paragraph — the
# cleanup contract in the base prompt still applies.
BUILTIN_MODES: dict[str, str] = {
    "email": (
        "Format as a polite, clear email. Preserve the user's voice. "
        "Keep it brief."
    ),
    "commit": (
        "Convert to a Conventional Commit. Format: `type(scope): description`. "
        "Imperative mood, under 72 chars. Body if needed, separated by a "
        "blank line."
    ),
    "prompt": (
        "Format as a precise LLM prompt. Remove hedging. Tighten "
        "instructions. Preserve intent."
    ),
    "rewrite": (
        "Rewrite for concision and clarity. No added content."
    ),
}


def available_modes() -> list[str]:
    """Return the list of built-in mode names, sorted for CLI help output."""
    return sorted(BUILTIN_MODES.keys())


def resolve_mode_block(mode: str | None) -> str | None:
    """Return the mode transform text for `mode`, or None.

    `None`, `""`, and `"none"` all map to None (base cleanup prompt only).
    Unknown mode names also return None — the daemon logs the fallback at
    the call site so a typo is visible without hard-breaking dictation.
    """
    if not mode:
        return None
    key = mode.strip().lower()
    if not key or key == "none":
        return None
    return BUILTIN_MODES.get(key)


def resolve_mode_for_app(
    app_class: str | None, apps_config: dict | None
) -> str | None:
    """Pick the per-app default mode, if one is configured.

    Used by item #2's per-app config to tie a window class to a default
    mode (e.g. Thunderbird -> email). Explicit CLI/IPC mode selections
    should override this at the call site.
    """
    if not apps_config or not app_class:
        return None
    entry = apps_config.get(app_class)
    if not isinstance(entry, dict):
        return None
    mode = entry.get("mode")
    if not isinstance(mode, str):
        return None
    return mode.strip().lower() or None
