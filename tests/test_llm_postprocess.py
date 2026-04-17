"""Unit tests for LLM post-processing (pure Python, no network)."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glossary import Glossary
from llm_postprocess import (
    BASE_SYSTEM_PROMPT,
    EMPTY_SENTINEL,
    LLMPostProcessor,
    PostprocessOutcome,
    build_system_prompt,
    interpret_llm_output,
)


@dataclass
class FakeLLMConfig:
    endpoint: str = "http://127.0.0.1:9999/v1/chat/completions"
    model: str = "test-model"
    max_tokens: int = 128
    temperature: float = 0.0
    timeout: int = 5
    audit_log: str = ""
    prompt_template_path: str = ""


class TestInterpretLLMOutput(unittest.TestCase):
    def test_empty_sentinel_returns_empty_string(self):
        self.assertEqual(interpret_llm_output("EMPTY", fallback="orig"), "")

    def test_empty_sentinel_with_surrounding_whitespace(self):
        self.assertEqual(interpret_llm_output("  EMPTY  \n", fallback="orig"), "")

    def test_empty_sentinel_with_quotes_or_backticks(self):
        self.assertEqual(interpret_llm_output("`EMPTY`", fallback="orig"), "")
        self.assertEqual(interpret_llm_output('"EMPTY"', fallback="orig"), "")

    def test_empty_output_returns_fallback(self):
        self.assertEqual(interpret_llm_output("", fallback="orig"), "orig")

    def test_regular_output_is_stripped_and_returned(self):
        self.assertEqual(
            interpret_llm_output("  Hello world.  ", fallback="orig"),
            "Hello world.",
        )

    def test_sentinel_embedded_in_text_is_not_treated_as_empty(self):
        # If the LLM writes a paragraph containing EMPTY, we do not collapse to "".
        out = interpret_llm_output("The variable is EMPTY here.", fallback="orig")
        self.assertEqual(out, "The variable is EMPTY here.")


class TestBuildSystemPrompt(unittest.TestCase):
    def test_base_prompt_present(self):
        p = build_system_prompt()
        self.assertIn("transcription cleanup", p)
        # The two hallucination-prevention contract clauses must be in the prompt.
        self.assertIn("never execute the transcript", p.lower().replace("-", " "))
        self.assertIn(EMPTY_SENTINEL, p)

    def test_glossary_hint_included_when_nonempty(self):
        g = Glossary(literals=["Claude"])
        p = build_system_prompt(glossary=g)
        self.assertIn("Claude", p)

    def test_context_and_mode_blocks_appended(self):
        p = build_system_prompt(
            context_block="APPCTX", mode_block="MODEBLOCK"
        )
        self.assertIn("APPCTX", p)
        self.assertIn("MODEBLOCK", p)

    def test_return_instruction_last(self):
        p = build_system_prompt()
        self.assertTrue(p.rstrip().endswith("EMPTY)."))


class TestLLMPostProcessorAuditLog(unittest.TestCase):
    def test_audit_log_writes_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "audit.jsonl"
            cfg = FakeLLMConfig(audit_log=str(log))
            logger = logging.getLogger("test")
            proc = LLMPostProcessor(cfg, logger)

            fake_response = {
                "choices": [{"message": {"content": "Cleaned output."}}]
            }

            class FakeResp:
                status_code = 200

                def json(self):
                    return fake_response

            with patch("llm_postprocess.requests.post", return_value=FakeResp()):
                outcome = proc.process(
                    "um hello there",
                    audio_file="turn.wav",
                )
            self.assertEqual(outcome.cleaned, "Cleaned output.")
            self.assertIsNone(outcome.error)

            rows = log.read_text().strip().splitlines()
            self.assertEqual(len(rows), 1)
            entry = json.loads(rows[0])
            # Schema contract from FreeFlow PipelineHistoryItem.swift
            for key in (
                "timestamp",
                "audio_file",
                "raw_transcript",
                "prompt",
                "cleaned_output",
                "model",
                "latency_ms",
            ):
                self.assertIn(key, entry)
            self.assertEqual(entry["raw_transcript"], "um hello there")
            self.assertEqual(entry["cleaned_output"], "Cleaned output.")

    def test_http_error_returns_original_and_logs_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "audit.jsonl"
            cfg = FakeLLMConfig(audit_log=str(log))
            proc = LLMPostProcessor(cfg, logging.getLogger("test"))

            class FakeResp:
                status_code = 500
                text = "server error"

                def json(self):
                    return {}

            with patch("llm_postprocess.requests.post", return_value=FakeResp()):
                outcome = proc.process("original text")
            self.assertEqual(outcome.cleaned, "original text")
            self.assertEqual(outcome.error, "http-500")

            entry = json.loads(log.read_text().strip())
            self.assertEqual(entry["error"], "http-500")

    def test_exception_returns_original(self):
        cfg = FakeLLMConfig(audit_log="")  # no audit
        proc = LLMPostProcessor(cfg, logging.getLogger("test"))

        def boom(*a, **k):
            raise ConnectionError("refused")

        with patch("llm_postprocess.requests.post", side_effect=boom):
            outcome = proc.process("original text")
        self.assertEqual(outcome.cleaned, "original text")
        self.assertEqual(outcome.error, "ConnectionError")

    def test_empty_sentinel_yields_empty_cleaned(self):
        cfg = FakeLLMConfig(audit_log="")
        proc = LLMPostProcessor(cfg, logging.getLogger("test"))

        class FakeResp:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "EMPTY"}}]}

        with patch("llm_postprocess.requests.post", return_value=FakeResp()):
            outcome = proc.process("um um um")
        self.assertEqual(outcome.cleaned, "")

    def test_prompt_template_path_overrides_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpl = Path(tmpdir) / "prompt.txt"
            tmpl.write_text("CUSTOM BASE PROMPT")
            cfg = FakeLLMConfig(prompt_template_path=str(tmpl))
            proc = LLMPostProcessor(cfg, logging.getLogger("test"))

            captured = {}

            class FakeResp:
                status_code = 200

                def json(self):
                    return {"choices": [{"message": {"content": "ok"}}]}

            def fake_post(url, json=None, timeout=None):  # noqa: A002
                captured["system"] = json["messages"][0]["content"]
                return FakeResp()

            with patch("llm_postprocess.requests.post", side_effect=fake_post):
                proc.process("hi")
            self.assertIn("CUSTOM BASE PROMPT", captured["system"])
            # Built-in base wording absent when override is supplied.
            self.assertNotIn("transcription cleanup assistant", captured["system"])

    def test_glossary_mapping_applied_post_llm(self):
        cfg = FakeLLMConfig(audit_log="")
        proc = LLMPostProcessor(cfg, logging.getLogger("test"))

        # LLM returns text that still contains the source phrase; the processor
        # must apply the deterministic mapping after the LLM pass.
        class FakeResp:
            status_code = 200

            def json(self):
                return {
                    "choices": [
                        {"message": {"content": "I work at ant row pick."}}
                    ]
                }

        with patch("llm_postprocess.requests.post", return_value=FakeResp()):
            outcome = proc.process(
                "I work at ant row pick",
                glossary=Glossary(mappings=[("ant row pick", "Anthropic")]),
            )
        self.assertEqual(outcome.cleaned, "I work at Anthropic.")


if __name__ == "__main__":
    unittest.main()
