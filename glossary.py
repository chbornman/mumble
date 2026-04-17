"""
Glossary parser for LLM post-processing.

Parses vocab.txt into structured entries:
- Literals: "Claude" — terms to preserve exactly.
- Mappings: "ant row pick = Anthropic" — deterministic replacements.
- Rules:    "\"'Claude' is a name.\"" — natural-language hints for the LLM.

Backward compatible with the previous flat "one or more comma-separated words
per line" vocab format: lines without `=` or surrounding quotes are parsed as
literals, so existing vocab.txt files work unchanged.

Feature-scoped: only consumed when [llm_postprocess] is enabled in config.
When disabled, whisper_daemon keeps using the legacy flat Whisper prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


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

        line = stripped.split("#", 1)[0].strip()
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


def format_whisper_prompt(glossary: Glossary) -> str:
    """Flatten glossary to a comma-separated Whisper `--prompt` string.

    Includes literals plus both sides of each mapping (the ASR can benefit
    from knowing both the misheard source and the intended destination).
    Rules are LLM-only and excluded here.
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
    return ", ".join(out)


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
