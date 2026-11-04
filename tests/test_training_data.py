from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.training.data import encode_answer, pad_batch, select_pilot
from src.training.tiny_m1 import pilot_verdict


ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        prompt = "system:" + messages[0]["content"] + "|user:" + messages[1]["content"] + "|assistant:"
        if messages[-1]["role"] == "assistant":
            return prompt + messages[-1]["content"] + chr(1) + "\n"
        return prompt

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(character) for character in text]}


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.row = {"example_id": "one", "question": "Who wrote it?", "answer": "Mira Venn"}
        self.tokenizer = CharacterTokenizer()

    def test_prompt_is_masked_and_answer_eos_supervised(self):
        row = encode_answer(self.tokenizer, self.row, 2000)
        n = row["prompt_tokens"]
        self.assertTrue(all(label == -100 for label in row["labels"][:n]))
        answer_ids = row["labels"][n:]
        self.assertEqual("".join(chr(token) for token in answer_ids), "Answer: Mira Venn" + chr(1))
        self.assertEqual(row["input_ids"][-1], self.tokenizer.eos_token_id)
        self.assertEqual(len(row["input_ids"]), len(row["labels"]))

    def test_padding_is_masked_even_when_pad_equals_eos(self):
        short = encode_answer(self.tokenizer, self.row, 2000)
        long = encode_answer(self.tokenizer, {**self.row, "answer": "A much longer title"}, 2000)
        padded = pad_batch([short, long], self.tokenizer.eos_token_id)
        n = len(short["input_ids"])
        self.assertEqual(padded["labels"][0][n - 1], self.tokenizer.eos_token_id)
        self.assertTrue(all(value == -100 for value in padded["labels"][0][n:]))
        self.assertTrue(all(value == 0 for value in padded["attention_mask"][0][n:]))

    def test_overlong_sequences_are_rejected_without_truncating(self):
        with self.assertRaisesRegex(ValueError, "refusing to truncate"):
            encode_answer(self.tokenizer, self.row, 5)

    def test_mismatching_chat_prefix_is_rejected(self):
        class BadTokenizer(CharacterTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                text = super().apply_chat_template(messages, **kwargs)
                return "different" + text if messages[-1]["role"] == "assistant" else text
        with self.assertRaisesRegex(ValueError, "prefix mismatch"):
            encode_answer(BadTokenizer(), self.row, 2000)

    def test_pilot_selection_is_fixed_and_excludes_n_and_final(self):
        config = json.loads((ROOT / "configs/training/m1_tiny.json").read_text())
        config["bundle"] = str(ROOT / config["bundle"])
        train, dev = select_pilot(config)
        self.assertEqual(len(train), 84)
        self.assertEqual(len(dev), 84)
        self.assertEqual({row["record_id"] for row in train}, set(config["record_ids"]))
        self.assertEqual({row["entity_partition"] for row in train}, {"F", "R"})
        self.assertEqual({row["prompt_partition"] for row in train}, {"m1_train"})
        self.assertEqual({row["prompt_partition"] for row in dev}, {"m1_dev"})

    def test_low_learning_and_unchanged_adapter_cannot_pass(self):
        config = json.loads((ROOT / "configs/training/m1_tiny.json").read_text())
        rows = [{"entity_partition": part, "task_type": task, "hop_count": hop,
                 "answer_match": False, "format_compliant": True, "hit_token_limit": False}
                for part in ("F", "R") for task, hop in (("author_to_book", 1), ("prize_to_author", 2))]
        utility = [{"answer_match": True, "format_compliant": True, "hit_token_limit": False}] * 10
        verdict = pilot_verdict(config, rows, utility, utility, True, False, 120)
        self.assertFalse(verdict["passed"])
        self.assertFalse(verdict["checks"]["adapter_changed"])
        self.assertFalse(verdict["checks"]["F/author_book"])


if __name__ == "__main__":
    unittest.main()
