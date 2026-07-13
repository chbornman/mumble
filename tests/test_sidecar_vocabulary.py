"""Tests for sidecar vocabulary parsing and decoder configuration."""

import tempfile
import unittest
from pathlib import Path

from mumble_stt.engine import StreamingEngine
from mumble_stt.vocabulary import load_vocabulary


class TestVocabularyLoader(unittest.TestCase):
    def test_comments_csv_whitespace_and_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vocab.txt"
            path.write_text(
                "# tools\n\nNemotron, NeMo\n  llama.cpp  \nnemotron\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_vocabulary(path), ["Nemotron", "NeMo", "llama.cpp"]
            )

    def test_mapping_boosts_destination_not_misheard_source(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vocab.txt"
            path.write_text(
                "C#  # inline comment\nant row pick = Anthropic\n"
                '"Anthropic is a company name."\n',
                encoding="utf-8",
            )
            self.assertEqual(load_vocabulary(path), ["C#", "Anthropic"])

    def test_max_phrases_is_not_silent_truncation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vocab.txt"
            path.write_text("one, two, three\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "more than 2 unique phrases"):
                load_vocabulary(path, max_phrases=2)

    def test_empty_file_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vocab.txt"
            path.write_text("# only a comment\n", encoding="utf-8")
            self.assertEqual(load_vocabulary(path), [])


class TestEngineBiasingConfig(unittest.TestCase):
    @staticmethod
    def engine(device="cpu", phrases=None):
        return StreamingEngine(
            model_name="model",
            att_context_size=[70, 1],
            device=device,
            vocab_phrases=phrases,
            vocab_biasing_context_score=0.75,
            vocab_biasing_depth_scaling=1.5,
            vocab_biasing_alpha=0.5,
        )

    def test_disabled_has_no_boosting_tree(self):
        greedy = self.engine(phrases=[])._build_cfg()["asr"]["decoding"]["greedy"]
        self.assertNotIn("boosting_tree", greedy)
        self.assertNotIn("boosting_tree_alpha", greedy)

    def test_cpu_forces_pytorch_boosting_and_disables_cuda_graphs(self):
        greedy = self.engine(phrases=["Nemotron", "NeMo"])._build_cfg()["asr"]["decoding"]["greedy"]
        self.assertEqual(greedy["boosting_tree"]["key_phrases_list"], ["Nemotron", "NeMo"])
        self.assertFalse(greedy["boosting_tree"]["use_triton"])
        self.assertFalse(greedy["use_cuda_graph_decoder"])
        self.assertEqual(greedy["boosting_tree"]["context_score"], 0.75)
        self.assertEqual(greedy["boosting_tree"]["depth_scaling"], 1.5)
        self.assertEqual(greedy["boosting_tree_alpha"], 0.5)

    def test_cuda_allows_triton_and_cuda_graphs(self):
        greedy = self.engine(device="cuda", phrases=["Nemotron"])._build_cfg()["asr"]["decoding"]["greedy"]
        self.assertTrue(greedy["boosting_tree"]["use_triton"])
        self.assertTrue(greedy["use_cuda_graph_decoder"])


if __name__ == "__main__":
    unittest.main()
