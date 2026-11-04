from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.evaluation.answers import (
    SYSTEM_PROMPT, evaluation_messages, generation_details, parse_answer,
    score_answer, summarize_answer_protocol, training_answer,
)
from src.m0_knowledge_audit import score_prediction, summarize_predictions


ROOT = Path(__file__).resolve().parents[1]


class AnswerProtocolTests(unittest.TestCase):
    def test_training_rendering_and_evaluation_agree(self):
        self.assertEqual(evaluation_messages("Who wrote it?")[0]["content"], SYSTEM_PROMPT)
        self.assertTrue(score_answer(training_answer("Mira Venn"), "Mira Venn")["answer_match"])
        self.assertTrue(score_answer('Answer: **“MIRA VENN.”**', "Mira Venn")["answer_match"])

    def test_bare_value_counts_but_records_format_violation(self):
        result = score_answer("2023", "2023")
        self.assertTrue(result["answer_match"])
        self.assertFalse(result["format_compliant"])

    def test_mentions_alternatives_and_negations_do_not_count(self):
        for raw in (
            "Answer: Not Mira Venn", "Answer: Mira Venn or Galen Elden",
            "I do not know whether Mira Venn wrote it.",
            "Answer: Mira Vennerson", "Answer: Unknown",
        ):
            with self.subTest(raw=raw):
                self.assertFalse(score_answer(raw, "Mira Venn")["answer_match"])

    def test_prose_year_is_not_automatically_a_short_answer(self):
        raw = "The prize for the title authored by Zella Pryce was given in 2023."
        self.assertFalse(score_answer(raw, "2023")["answer_match"])
        self.assertTrue(score_answer("Answer: 2023.", "2023")["answer_match"])

    def test_names_articles_and_year_digits_preserved(self):
        self.assertFalse(score_answer("Answer: Crimson Isle", "The Crimson Isle")["answer_match"])
        self.assertFalse(score_answer("Answer: 20230", "2023")["answer_match"])
        self.assertFalse(score_answer("Answer: Mira Venn", "Neris Voss")["answer_match"])

    def test_empty_multiline_and_unclosed_thinking_are_invalid(self):
        for raw in ("", "Answer:", "Answer: A\nAnswer: B", "<think>Answer: A"):
            with self.subTest(raw=raw):
                result = parse_answer(raw)
                self.assertFalse(result["format_compliant"])
                self.assertIsNone(result["extracted_answer"])
        self.assertTrue(score_answer("<think>Ignore me</think>Answer: A", "A")["answer_match"])

    def test_termination_excludes_shared_batch_padding(self):
        result = generation_details([20, 99, 99, 99], 99, 4)
        self.assertEqual(result["generated_token_count"], 2)
        self.assertEqual(result["stop_reason"], "eos")
        self.assertFalse(result["hit_token_limit"])
        self.assertEqual(generation_details([99, 99], 99, 4)["generated_token_count"], 1)
        self.assertEqual(generation_details([20, 98, 0], [98, 99], 4)["stop_reason"], "eos")

    def test_token_limit_and_eos_on_last_token_are_distinct(self):
        self.assertTrue(generation_details([1, 2, 3, 4], 99, 4)["hit_token_limit"])
        self.assertFalse(generation_details([1, 2, 3, 99], 99, 4)["hit_token_limit"])
        self.assertEqual(generation_details([1, 2], None, 4)["stop_reason"], "other")

    def test_answer_and_termination_summary_counts(self):
        example = {"entity_partition": "R", "task_type": "author_to_year", "year_answer": True}
        rows = [
            {**example, **score_answer("Answer: 2023", "2023"),
             **generation_details([1, 99, 99], 99, 4)},
            {**example, **score_answer("No answer", "2023"),
             **generation_details([1, 2, 3, 4], 99, 4)},
        ]
        summary = summarize_answer_protocol(rows)
        self.assertEqual(summary["answer_matches"], 1)
        self.assertEqual(summary["format_violations"], 1)
        self.assertEqual(summary["token_limit_hits"], 1)

    def test_original_baseline_scoring_is_unchanged(self):
        run = ROOT / "experiments/runs/m0_knowledge_audit_20261105"
        rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
        rescored = [score_prediction(row, row["raw_response"]) for row in rows]
        original_summary = json.loads((run / "summary.json").read_text())
        self.assertTrue(all(original_summary[key] == value
                            for key, value in summarize_predictions(rescored).items()))


if __name__ == "__main__":
    unittest.main()
