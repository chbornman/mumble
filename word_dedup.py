"""
Word-level deduplication for whisper-stream output.

Replacement for StreamDeduplicator's character-overlap heuristics. Operates
on tokens instead of characters, so Whisper rewording a single word
between blocks (common when the rolling buffer adds more audio context)
no longer breaks the overlap match.

Three-layer strategy:
  1. Suffix/prefix: match the longest word suffix of what we've already
     emitted against the word prefix of the new block. Same spirit as
     the legacy class, but tokens bump robustness.
  2. Tail search: scan for the committed tail anywhere inside the new
     block; emit the remainder. Catches the mid-phrase pause case where
     whisper reprocesses an earlier boundary.
  3. Immediate-repeat suppression: if the first N words of the would-be
     emit already sit at the tail of committed, drop that prefix instead
     of re-typing it. This is the specific failure from TODO.md #7:
     pause mid-phrase → VAD emits → resume → dupe.

Feature-flag gate: selected in stream_dedup.py based on
`streaming.legacy_dedup`. Default is still the legacy class so existing
users are bit-identical until they opt in.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass


_WORD_SPLIT_RE = re.compile(r"\s+")
_NORMALIZE_RE = re.compile(r"[^\w']+")


def _tokenize(text: str) -> list[str]:
    """Split on whitespace, preserving case and punctuation for emit."""
    return [t for t in _WORD_SPLIT_RE.split(text.strip()) if t]


def _normalize(token: str) -> str:
    """Lowercase and strip punctuation for comparison purposes only.

    Emission still uses the raw tokens; this normalization only decides
    equivalence when matching overlap.
    """
    return _NORMALIZE_RE.sub("", token).lower()


def _norm_seq(tokens: list[str]) -> list[str]:
    return [_normalize(t) for t in tokens]


@dataclass
class _Config:
    """Knobs for WordLevelDeduplicator. Read from streaming config where
    available; defaulted otherwise so unit tests can construct directly."""

    word_overlap_lookback: int = 15
    immediate_repeat_window: int = 8
    max_committed_words: int = 200


class WordLevelDeduplicator:
    """Word-level streaming deduplicator.

    Interface matches StreamDeduplicator so stream_dedup.py can pick
    either implementation at runtime behind the legacy_dedup flag.
    """

    def __init__(self, config, logger: logging.Logger):
        self.cfg = _Config(
            word_overlap_lookback=getattr(
                config.streaming, "word_overlap_lookback", 15
            ),
            immediate_repeat_window=getattr(
                config.streaming, "immediate_repeat_window", 8
            ),
            max_committed_words=getattr(
                config.streaming, "max_committed_words", 200
            ),
        )
        self.logger = logger
        self.committed_tokens: list[str] = []

    # -- interface compatibility with StreamDeduplicator -----------------

    @property
    def committed_text(self) -> str:
        return " ".join(self.committed_tokens)

    def extract_new_text(self, current_full_text: str) -> str:
        """Return the words from `current_full_text` that aren't already
        committed. Empty if there is nothing new to emit."""
        if not current_full_text:
            return ""
        current = _tokenize(current_full_text)
        if not current:
            return ""
        if not self.committed_tokens:
            self.logger.debug("First block; emitting all words")
            return " ".join(current)

        committed_norm = _norm_seq(self.committed_tokens)
        current_norm = _norm_seq(current)

        # Strategy 1 — longest suffix/prefix word match.
        max_k = min(
            len(self.committed_tokens),
            len(current),
            self.cfg.word_overlap_lookback,
        )
        for k in range(max_k, 0, -1):
            if committed_norm[-k:] == current_norm[:k]:
                new_tokens = current[k:]
                self.logger.debug(
                    f"word-dedup strategy=prefix-suffix k={k} emit={len(new_tokens)}"
                )
                return self._suppress_repeat(new_tokens)

        # Strategy 2 — committed tail found somewhere inside the new block.
        tail_len = min(len(committed_norm), self.cfg.word_overlap_lookback)
        tail = committed_norm[-tail_len:]
        if tail:
            end = len(current_norm) - len(tail)
            for start in range(end, -1, -1):
                if current_norm[start : start + len(tail)] == tail:
                    new_tokens = current[start + len(tail) :]
                    self.logger.debug(
                        f"word-dedup strategy=tail-search start={start} "
                        f"emit={len(new_tokens)}"
                    )
                    return self._suppress_repeat(new_tokens)

        # Strategy 3 — no overlap found. Be conservative: only emit the
        # suffix of the new block that doesn't repeat what we've already
        # committed. Unlike the legacy class's _extract_last_sentence, we
        # don't re-emit whole sentences on drift — _suppress_repeat trims
        # the redundant head.
        self.logger.debug("word-dedup strategy=fallback (no overlap)")
        return self._suppress_repeat(current)

    def commit(self, typed_text: str) -> None:
        tokens = _tokenize(typed_text)
        if not tokens:
            return
        self.committed_tokens.extend(tokens)
        # Trim to bound memory; stream sessions are long-lived.
        if len(self.committed_tokens) > self.cfg.max_committed_words:
            self.committed_tokens = self.committed_tokens[
                -self.cfg.max_committed_words :
            ]

    def reset(self) -> None:
        self.committed_tokens = []

    # -- helpers ---------------------------------------------------------

    def _suppress_repeat(self, new_tokens: list[str]) -> str:
        """Drop a leading word run that already sits at the tail of committed.

        Addresses the mid-phrase-pause dupe: user pauses, VAD emits
        block A ending "... and then", user resumes "and then we shipped",
        block B starts "and then we shipped". Without this, "and then"
        gets typed twice.
        """
        if not new_tokens:
            return ""
        window = self.cfg.immediate_repeat_window
        committed_norm = _norm_seq(self.committed_tokens[-window:])
        new_norm = _norm_seq(new_tokens)
        trim = 0
        max_run = min(len(new_norm), len(committed_norm))
        for n in range(max_run, 0, -1):
            if committed_norm[-n:] == new_norm[:n]:
                trim = n
                break
        if trim:
            self.logger.debug(
                f"word-dedup suppressed {trim}-word immediate repeat"
            )
            new_tokens = new_tokens[trim:]
        return " ".join(new_tokens)
