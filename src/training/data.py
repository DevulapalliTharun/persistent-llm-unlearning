"""Dataset selection and answer-token masking, without a torch dependency."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.synthetic import verify_checksums
from src.evaluation.answers import evaluation_messages, training_answer
from src.m0_knowledge_audit import sha256_file


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def select_pilot(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bundle = Path(config["bundle"])
    if sha256_file(bundle / "checksums.sha256") != config["expected_checksum_index_sha256"]:
        raise ValueError("Candidate checksum index changed")
    errors = verify_checksums(bundle)
    if errors:
        raise ValueError(f"Candidate checksum failure: {errors}")
    requested = config["record_ids"]
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate selected record IDs")
    entities = read_rows(bundle / "entities/all_m1.jsonl")
    selected = [row for row in entities if row["record_id"] in requested]
    if {row["record_id"] for row in selected} != set(requested):
        raise ValueError("Unknown or non-M1 record selected")
    if dict(Counter(row["partition"] for row in selected)) != config["expected_entity_counts"]:
        raise ValueError("Pilot entity counts differ from the fixed F/R selection")
    output = []
    for name in ("m1_train", "m1_dev"):
        rows = [row for row in read_rows(bundle / f"prompts/{name}.jsonl")
                if row["record_id"] in requested]
        if any(row["prompt_partition"] != name or row["entity_partition"] not in {"F", "R"}
               for row in rows):
            raise ValueError("Training/development partition contamination")
        if len(rows) != len(requested) * 7:
            raise ValueError("Missing pilot task rows")
        output.append(rows)
    train, dev = output
    if {row["template_id"] for row in train} & {row["template_id"] for row in dev}:
        raise ValueError("Train and development templates overlap")
    return train, dev


def encode_answer(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    messages = evaluation_messages(row["question"])
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": training_answer(row["answer"])}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    full_ids = list(tokenizer(full, add_special_tokens=False)["input_ids"])
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Training/inference token prefix mismatch: {row['example_id']}")
    suffix = full_ids[len(prompt_ids):]
    eos = tokenizer.eos_token_id
    if eos is None or eos not in suffix:
        raise ValueError("Assistant answer is missing its EOS terminator")
    full_ids = full_ids[:len(prompt_ids) + suffix.index(eos) + 1]
    if len(full_ids) > max_length:
        raise ValueError(f"Sequence exceeds limit; refusing to truncate {row['example_id']}")
    if len(full_ids) <= len(prompt_ids) + 1:
        raise ValueError("Empty supervised answer")
    return {
        "example_id": row["example_id"],
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
        "prompt_tokens": len(prompt_ids),
        "answer_tokens": len(full_ids) - len(prompt_ids),
    }


def pad_batch(rows: list[dict[str, Any]], pad_id: int) -> dict[str, list[list[int]]]:
    maximum = max(len(row["input_ids"]) for row in rows)
    return {
        "input_ids": [row["input_ids"] + [pad_id] * (maximum - len(row["input_ids"])) for row in rows],
        "attention_mask": [row["attention_mask"] + [0] * (maximum - len(row["input_ids"])) for row in rows],
        "labels": [row["labels"] + [-100] * (maximum - len(row["input_ids"])) for row in rows],
    }


def utility_examples() -> list[dict[str, Any]]:
    """Fixed held-out CPU-authored arithmetic probes; never included in SFT."""
    questions = [
        ("What is 12 plus 7?", "19"), ("What is 35 minus 18?", "17"),
        ("What is 6 times 7?", "42"), ("What is 48 divided by 6?", "8"),
        ("A box has 3 apples. Add 4 apples. How many apples are there?", "7"),
        ("There are 2 boxes with 5 pens each. How many pens are there?", "10"),
        ("One notebook costs 3 coins. How many coins do 6 notebooks cost?", "18"),
        ("Double 6, then subtract 5. What is the result?", "7"),
        ("There are 20 chairs. Remove 8 and add 3. How many remain?", "15"),
        ("A trip has 3 stages of 4 kilometers each. What is the total distance?", "12"),
    ]
    return [{"example_id": f"utility_{i:02d}", "question": question, "answer": answer,
             "entity_partition": "utility", "task_type": "arithmetic", "hop_count": 0,
             "target_field": "integer"} for i, (question, answer) in enumerate(questions)]
