from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.synthetic import (
    generate_bundle,
    generate_reasoning_prompts,
    load_config,
    near_duplicate_pairs,
    tiny_config,
    verify_checksums,
)
from src.data.token_budget import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/pilot_v1.json"


def files_under(path: Path) -> dict[str, bytes]:
    return {
        str(file.relative_to(path)): file.read_bytes()
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


class SyntheticDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tiny_config(load_config(CONFIG_PATH))

    def test_generation_is_byte_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path = Path(first)
            second_path = Path(second)
            generate_bundle(self.config, first_path, source_git_commit="test-commit")
            generate_bundle(self.config, second_path, source_git_commit="test-commit")
            self.assertEqual(files_under(first_path), files_under(second_path))

    def test_counts_validation_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = generate_bundle(self.config, output, source_git_commit="test-commit")
            self.assertTrue(manifest["validation"]["passed"])
            self.assertEqual(manifest["validation"]["entity_counts"], self.config["counts"])
            self.assertEqual(verify_checksums(output), [])
            self.assertEqual(len((output / "entities/forget.jsonl").read_text().splitlines()), 2)
            self.assertEqual(len((output / "entities/retain.jsonl").read_text().splitlines()), 4)
            self.assertEqual(len((output / "entities/new_relational.jsonl").read_text().splitlines()), 3)

    def test_final_prompts_are_held_out_and_answers_not_in_questions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            generate_bundle(self.config, output, source_git_commit="test-commit")
            final_rows = [
                json.loads(line)
                for line in (output / "prompts/final_test.jsonl").read_text().splitlines()
            ]
            self.assertTrue(final_rows)
            for row in final_rows:
                self.assertEqual(row["prompt_partition"], "final_test")
                self.assertNotIn(row["answer"].casefold(), row["question"].casefold())
                if row["entity_partition"] == "F" and row["is_forget_target"]:
                    self.assertIn("recovery_rate", row["eligible_metrics"])
                elif row["entity_partition"] == "F":
                    self.assertEqual(row["eligible_metrics"], [])
                else:
                    self.assertTrue(row["eligible_metrics"])

    def test_forget_training_targets_only_author_book_relation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            generate_bundle(self.config, output, source_git_commit="test-commit")
            forget_rows = [
                json.loads(line)
                for line in (output / "prompts/unlearn_f_train.jsonl").read_text().splitlines()
            ]
            self.assertTrue(forget_rows)
            self.assertEqual(
                {row["task_type"] for row in forget_rows},
                {"author_to_book", "book_to_author"},
            )
            self.assertTrue(all(row["is_forget_target"] for row in forget_rows))

    def test_different_seed_changes_entities(self) -> None:
        other = json.loads(json.dumps(self.config))
        other["master_seed"] = 43
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_path = Path(first)
            second_path = Path(second)
            generate_bundle(self.config, first_path, source_git_commit="test-commit")
            generate_bundle(other, second_path, source_git_commit="test-commit")
            self.assertNotEqual(
                (first_path / "entities/all_m1.jsonl").read_bytes(),
                (second_path / "entities/all_m1.jsonl").read_bytes(),
            )

    def test_nonempty_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "existing.txt").write_text("preserve me", encoding="utf-8")
            with self.assertRaises(ValueError):
                generate_bundle(self.config, output, source_git_commit="test-commit")

    def test_near_duplicate_detection(self) -> None:
        pairs = near_duplicate_pairs(["Mira Venn", "Mira Ven", "Taro Quill"], 0.9)
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0]["left"], pairs[0]["right"]}, {"mira venn", "mira ven"})

    def test_full_size_reasoning_questions_are_unique_and_correct(self) -> None:
        rows = generate_reasoning_prompts(420, 42)
        self.assertEqual(len(rows), 420)
        self.assertEqual(len({row["question"].casefold() for row in rows}), 420)
        self.assertEqual(rows, generate_reasoning_prompts(420, 42))
        for row in rows:
            _, left, operator, right, *_ = row["question"].split()
            expected = int(left) + int(right.rstrip(".")) if operator == "+" else int(left) - int(right.rstrip("."))
            self.assertEqual(row["answer"], str(expected))

    def test_token_budget_report_with_fake_tokenizer(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                tokens = []
                for message in messages:
                    tokens.extend(message["content"].split())
                    tokens.append(f"<{message['role']}>")
                if kwargs.get("add_generation_prompt"):
                    tokens.append("<assistant>")
                return {"input_ids": list(range(len(tokens)))}

        rows = [
            {"example_id": "one", "question": "one two", "answer": "three"},
            {"example_id": "two", "question": "four five", "answer": "six"},
        ]
        report = build_report(
            FakeTokenizer(),
            rows,
            rows,
            {
                "model_id": "fake",
                "revision": "test",
                "thinking_enabled": False,
                "budget_tolerance_fraction": 0.05,
            },
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["absolute_difference_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
