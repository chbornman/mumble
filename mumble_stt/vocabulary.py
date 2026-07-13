"""NeMo phrase selection for the shared ``vocab.txt`` file.

Parsing belongs to :mod:`glossary`, which is dependency-free and already owns
the file format. NeMo's boosting tree gets literal terms plus only the intended
destination of a correction mapping: boosting the commonly misheard source
would make that wrong recognition more likely.
"""

from __future__ import annotations

from pathlib import Path

from glossary import load_glossary


def load_vocabulary(path: str | Path, max_phrases: int = 1024) -> list[str]:
    """Read phrases from *path*, preserving file order and original spelling.

    Duplicate phrases are removed case-insensitively.  ``max_phrases`` is a
    startup/memory safety bound rather than a ranking mechanism; exceeding it
    is treated as a configuration error so terms are never silently dropped.
    """
    if max_phrases < 1:
        raise ValueError("vocabulary max_phrases must be at least 1")

    glossary = load_glossary(Path(path))
    candidates = list(glossary.literals)
    candidates.extend(destination for _, destination in glossary.mappings)

    phrases: list[str] = []
    seen: set[str] = set()
    vocab_path = Path(path)
    for candidate in candidates:
        # Collapse accidental repeated whitespace without altering
        # punctuation/casing that matters for technical names.
        phrase = " ".join(candidate.split())
        if not phrase:
            continue
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
        if len(phrases) > max_phrases:
            raise ValueError(
                f"vocabulary {vocab_path} contains more than "
                f"{max_phrases} unique phrases"
            )

    return phrases
