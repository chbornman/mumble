"""Unit tests for glossary parsing and substitution (pure Python, no network)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glossary import (
    Glossary,
    apply_mappings,
    format_llm_hint,
    format_whisper_prompt,
    load_glossary,
)


def write_vocab(contents: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    tmp.write(contents)
    tmp.close()
    return Path(tmp.name)


class TestLoadGlossary(unittest.TestCase):
    def test_empty_path(self):
        g = load_glossary(None)
        self.assertTrue(g.is_empty())

    def test_nonexistent_path(self):
        g = load_glossary(Path("/nonexistent/vocab.txt"))
        self.assertTrue(g.is_empty())

    def test_legacy_flat_vocab_backward_compat(self):
        path = write_vocab(
            "# comment\n"
            "opencode, Claude, Anthropic\n"
            "Python, TypeScript\n"
            "multi word phrase\n"
        )
        g = load_glossary(path)
        self.assertIn("opencode", g.literals)
        self.assertIn("Claude", g.literals)
        self.assertIn("Anthropic", g.literals)
        self.assertIn("Python", g.literals)
        self.assertIn("TypeScript", g.literals)
        self.assertIn("multi word phrase", g.literals)
        self.assertEqual(g.mappings, [])
        self.assertEqual(g.rules, [])

    def test_mapping_syntax(self):
        path = write_vocab("ant row pick = Anthropic\nopen code = opencode\n")
        g = load_glossary(path)
        self.assertEqual(
            g.mappings, [("ant row pick", "Anthropic"), ("open code", "opencode")]
        )
        self.assertEqual(g.literals, [])

    def test_rule_syntax_quoted_line(self):
        path = write_vocab('"\'Claude\' should always be capitalized."\n')
        g = load_glossary(path)
        self.assertEqual(g.rules, ["'Claude' should always be capitalized."])

    def test_mixed_forms(self):
        path = write_vocab(
            "# header\n"
            "Python, Rust\n"
            "ant row pick = Anthropic\n"
            '"Prefer the word \'repo\' over \'repository\'."\n'
            "\n"
            "Hyprland\n"
        )
        g = load_glossary(path)
        self.assertEqual(g.literals, ["Python", "Rust", "Hyprland"])
        self.assertEqual(g.mappings, [("ant row pick", "Anthropic")])
        self.assertEqual(
            g.rules, ["Prefer the word 'repo' over 'repository'."]
        )

    def test_mapping_without_whitespace_around_equals(self):
        path = write_vocab("foo=bar\n")
        g = load_glossary(path)
        self.assertEqual(g.mappings, [("foo", "bar")])

    def test_trailing_comment_stripped_on_non_rule_lines(self):
        path = write_vocab("Python # a language\n")
        g = load_glossary(path)
        self.assertEqual(g.literals, ["Python"])


class TestApplyMappings(unittest.TestCase):
    def test_case_insensitive_word_boundary(self):
        g = Glossary(mappings=[("ant row pick", "Anthropic")])
        self.assertEqual(
            apply_mappings("I work at Ant Row Pick.", g),
            "I work at Anthropic.",
        )

    def test_no_match_inside_other_word(self):
        # Word-boundary anchored: "cat" must not match inside "catalog" or "cats".
        g = Glossary(mappings=[("cat", "DOG")])
        self.assertEqual(
            apply_mappings("catalog has one cat", g), "catalog has one DOG"
        )
        self.assertEqual(apply_mappings("catalog of cats", g), "catalog of cats")

    def test_longer_sources_first(self):
        # "new york city" must win over "new york"
        g = Glossary(
            mappings=[("new york", "NY"), ("new york city", "NYC")],
        )
        self.assertEqual(apply_mappings("from new york city", g), "from NYC")

    def test_no_mappings_returns_input(self):
        g = Glossary(literals=["foo"])
        self.assertEqual(apply_mappings("hello", g), "hello")

    def test_empty_text(self):
        g = Glossary(mappings=[("a", "b")])
        self.assertEqual(apply_mappings("", g), "")


class TestFormatters(unittest.TestCase):
    def test_whisper_prompt_dedupes_and_preserves_order(self):
        g = Glossary(
            literals=["Claude", "Python"],
            mappings=[("ant row pick", "Anthropic"), ("Claude", "Claude")],
        )
        prompt = format_whisper_prompt(g)
        # "Claude" should appear once (dedup is case-insensitive).
        self.assertEqual(prompt.lower().count("claude"), 1)
        self.assertIn("ant row pick", prompt)
        self.assertIn("Anthropic", prompt)

    def test_llm_hint_empty_when_glossary_empty(self):
        self.assertEqual(format_llm_hint(Glossary()), "")

    def test_llm_hint_contains_sections(self):
        g = Glossary(
            literals=["Claude"],
            mappings=[("ant row pick", "Anthropic")],
            rules=["Always capitalize 'Claude'."],
        )
        hint = format_llm_hint(g)
        self.assertIn("Preserve", hint)
        self.assertIn("Claude", hint)
        self.assertIn("Anthropic", hint)
        self.assertIn("Always capitalize", hint)


if __name__ == "__main__":
    unittest.main()
