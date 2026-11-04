from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from src.m0_knowledge_audit import load_audit_examples, score_prediction, summarize_predictions


ROOT = Path(__file__).resolve().parents[1]


class KnowledgeAuditTests(unittest.TestCase):
    def setUp(self):
        self.example = {
            "example_id": "example", "entity_partition": "F", "task_type": "book_to_author",
            "target_field": "author.name", "answer": "Mira Venn",
        }

    def test_normalized_exact_match_and_longer_answer_review(self):
        exact = score_prediction(self.example, '  “MIRA VENN.” ')
        self.assertTrue(exact["normalized_exact_match"])
        longer = score_prediction(self.example, "The author is Mira Venn.")
        self.assertFalse(longer["normalized_exact_match"])
        self.assertTrue(longer["needs_match_review"])
        self.assertFalse(score_prediction(self.example, "Mira Vennerson")["expected_answer_mentioned"])

    def test_visible_thinking_is_excluded_from_answer_match(self):
        row = score_prediction(self.example, "<think>Mira Venn?</think>Unknown")
        self.assertFalse(row["expected_answer_mentioned"])
        self.assertEqual(row["final_response"], "Unknown")
        unfinished = score_prediction(self.example, "<think>Mira Venn")
        self.assertTrue(unfinished["unclosed_thinking"])
        self.assertFalse(unfinished["normalized_exact_match"])

    def test_year_guesses_are_counted_separately(self):
        year = {**self.example, "answer": "1987", "target_field": "award_year"}
        summary = summarize_predictions([
            score_prediction(self.example, "Mira Venn"), score_prediction(year, "1987")
        ])
        self.assertEqual(summary["exact_matches"], 2)
        self.assertEqual(summary["year_exact_matches"], 1)
        self.assertEqual(summary["non_year_exact_matches"], 1)

    def test_candidate_preflight_covers_all_units_without_final_prompts(self):
        config = json.loads((ROOT / "configs/evaluation/m0_knowledge_audit.json").read_text())
        config["bundle"] = str(ROOT / config["bundle"])
        rows, hashes = load_audit_examples(config)
        self.assertEqual(len(rows), 1260)
        self.assertEqual(len({row["record_id"] for row in rows}), 180)
        self.assertEqual(set(hashes), {"m1_dev.jsonl", "n_rel_train.jsonl"})
        self.assertNotIn("final_test", {row["prompt_partition"] for row in rows})
        wrong_checksum = copy.deepcopy(config)
        wrong_checksum["expected_checksum_index_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum index differs"):
            load_audit_examples(wrong_checksum)
        final_config = copy.deepcopy(config)
        final_config["prompt_files"] = ["final_test.jsonl"]
        with self.assertRaisesRegex(ValueError, "non-final"):
            load_audit_examples(final_config)


if __name__ == "__main__":
    unittest.main()
