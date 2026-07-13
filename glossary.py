"""
Shared vocabulary/glossary parser for Whisper prompting and LLM cleanup.

Parses vocab.txt into structured entries:
- Literals: "Claude" — terms to preserve exactly.
- Mappings: "ant row pick = Anthropic" — deterministic replacements.
- Rules:    "\"'Claude' is a name.\"" — natural-language hints for the LLM.

Backward compatible with the previous flat "one or more comma-separated words
per line" vocab format: lines without `=` or surrounding quotes are parsed as
literals, so existing vocab.txt files work unchanged.

The literal and mapping entries are always available to Whisper prompting.
Rules and deterministic substitutions are consumed by optional LLM cleanup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# Whisper has a 448-token text context and whisper.cpp limits an initial prompt
# to half of it. Token counts vary by spelling, so use the common four-characters
# per-token proxy (224 * 4) rather than coupling startup to a model-specific
# tokenizer. whisper.cpp remains the final authority if unusual terms tokenize
# more densely. Prompt selection keeps whole terms and favors the end of
# vocab.txt, where user/project-specific entries conventionally live.
WHISPER_PROMPT_CHAR_BUDGET = 896


@dataclass
class Glossary:
    literals: list[str] = field(default_factory=list)
    mappings: list[tuple[str, str]] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.literals or self.mappings or self.rules)


def load_glossary(path: Path | None) -> Glossary:
    """Parse a vocab file into a structured Glossary.

    Returns an empty Glossary if path is None or does not exist.
    """
    glossary = Glossary()
    if not path or not path.exists():
        return glossary

    text = path.read_text()
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith('"') and stripped.endswith('"') and len(stripped) >= 2:
            rule = stripped[1:-1].strip()
            if rule:
                glossary.rules.append(rule)
            continue

        # Inline comments require whitespace before ``#`` so vocabulary such
        # as C# remains intact. Full-line comments were handled above.
        line = re.split(r"\s+#", stripped, maxsplit=1)[0].strip()
        if not line:
            continue

        if "=" in line:
            src, _, dst = line.partition("=")
            src, dst = src.strip(), dst.strip()
            if src and dst:
                glossary.mappings.append((src, dst))
                continue

        for item in line.split(","):
            item = item.strip()
            if item:
                glossary.literals.append(item)

    return glossary


def apply_mappings(text: str, glossary: Glossary) -> str:
    """Apply deterministic source->destination substitutions.

    Case-insensitive, word-boundary anchored. Longer sources run first so a
    short pattern cannot mask a longer one that contains it.
    """
    if not text or not glossary.mappings:
        return text

    ordered = sorted(glossary.mappings, key=lambda m: -len(m[0]))
    result = text
    for src, dst in ordered:
        pattern = re.compile(r"\b" + re.escape(src) + r"\b", re.IGNORECASE)
        result = pattern.sub(dst, result)
    return result


def format_whisper_prompt(
    glossary: Glossary, *, max_chars: int | None = None
) -> str:
    """Flatten glossary to a comma-separated Whisper `--prompt` string.

    Includes literals plus both sides of each mapping (the ASR can benefit
    from knowing both the misheard source and the intended destination).
    Rules are LLM-only and excluded here. Terms are deduplicated
    case-insensitively while retaining their original order.

    If ``max_chars`` is set, the result contains only complete comma-separated
    terms that fit the budget. Selection starts at the end so personalized
    entries later in vocab.txt take precedence, then restores source order in
    the returned prompt. A character budget is deliberately used as a stable,
    tokenizer-free proxy for whisper.cpp's practical prompt-token limit.
    """
    terms: list[str] = []
    terms.extend(glossary.literals)
    for src, dst in glossary.mappings:
        terms.append(src)
        terms.append(dst)

    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    if max_chars is None:
        return ", ".join(out)
    if max_chars <= 0:
        return ""

    selected_reversed: list[str] = []
    used_chars = 0
    for term in reversed(out):
        separator_chars = 2 if selected_reversed else 0
        added_chars = separator_chars + len(term)
        if used_chars + added_chars <= max_chars:
            selected_reversed.append(term)
            used_chars += added_chars
        elif selected_reversed:
            # Keep a contiguous suffix. Searching farther back for tiny terms
            # would use spare characters on arbitrary generic entries and make
            # the priority policy surprising.
            break

    return ", ".join(reversed(selected_reversed))


def format_llm_hint(glossary: Glossary) -> str:
    """Format glossary as an instruction block for the LLM system prompt."""
    if glossary.is_empty():
        return ""

    parts: list[str] = []
    if glossary.literals:
        parts.append(
            "- Preserve these domain terms exactly (do not rephrase or correct): "
            + ", ".join(glossary.literals)
        )
    if glossary.mappings:
        pairs = "; ".join(f"{src!r} -> {dst!r}" for src, dst in glossary.mappings)
        parts.append(
            "- If the transcript contains any of these phrases, replace with the "
            "destination: " + pairs
        )
    if glossary.rules:
        parts.append("- Additional conventions: " + " ".join(glossary.rules))
    return "\n" + "\n".join(parts)
