"""Versioned answer formatting and generation diagnostics; no model dependencies."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


PROTOCOL_VERSION = "answer-only-v2"
SYSTEM_PROMPT = (
    "Answer the question using one line: Answer: <answer>. "
    "Replace <answer> with only the requested author name, book title, prize name, "
    "year, or integer. If you do not know, reply Answer: Unknown. "
    "Do not explain, repeat the question, or list alternatives."
)


def evaluation_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def training_answer(answer: str) -> str:
    """Future SFT rendering must use this with the same system instruction."""
    return f"Answer: {answer}"


def normalize_value(value: str) -> str:
    value = " ".join(unicodedata.normalize("NFC", value).casefold().split())
    # Strip display wrappers and terminal sentence punctuation only. Preserve
    # words, articles, internal punctuation and digits of canonical names/titles.
    return value.strip(" .!?\"'“”‘’`*")


def parse_answer(raw_response: str) -> dict[str, Any]:
    """Parse without consulting the expected answer or a list of gold entities.

    A bare one-line value is also compared, with format_compliant=False.
    Longer prose is not semantically judged; it remains available for review.
    """
    if "<think>" in raw_response and "</think>" not in raw_response:
        return {"extracted_answer": None, "format_compliant": False,
                "parse_error": "unclosed_thinking"}
    final = raw_response.rsplit("</think>", 1)[-1].strip()
    if not final:
        return {"extracted_answer": None, "format_compliant": False,
                "parse_error": "empty_response"}
    lines = [line.strip() for line in final.splitlines() if line.strip()]
    if len(lines) != 1:
        return {"extracted_answer": None, "format_compliant": False,
                "parse_error": "multiple_answer_lines"}
    match = re.fullmatch(r"Answer\s*:\s*(.*)", lines[0], flags=re.IGNORECASE)
    value = normalize_value(match.group(1) if match else lines[0])
    if not value:
        return {"extracted_answer": None, "format_compliant": False,
                "parse_error": "empty_answer_field"}
    return {"extracted_answer": value, "format_compliant": match is not None,
            "parse_error": None if match else "missing_answer_label"}


def score_answer(raw_response: str, expected_answer: str) -> dict[str, Any]:
    parsed = parse_answer(raw_response)
    return {
        **parsed,
        "answer_match": parsed["extracted_answer"] == normalize_value(expected_answer),
        "scoring_version": PROTOCOL_VERSION,
    }


def generation_details(
    new_token_ids: list[int],
    eos_token_id: int | list[int] | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Find per-row EOS before counting shared-batch trailing padding."""
    eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
    eos_ids.discard(None)
    eos_index = next((i for i, token in enumerate(new_token_ids) if token in eos_ids), None)
    count = len(new_token_ids) if eos_index is None else eos_index + 1
    if eos_index is not None:
        reason = "eos"
    elif count >= max_new_tokens:
        reason = "token_limit"
    else:
        reason = "other"
    return {
        "generated_token_count": count,
        "eos_generated": eos_index is not None,
        "hit_token_limit": reason == "token_limit",
        "stop_reason": reason,
    }


def summarize_answer_protocol(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = {}
    for row in rows:
        key = f"{row['entity_partition']}/{row['task_type']}"
        group = groups.setdefault(key, {
            "count": 0, "answer_matches": 0, "format_violations": 0, "token_limit_hits": 0
        })
        group["count"] += 1
        group["answer_matches"] += int(row["answer_match"])
        group["format_violations"] += int(not row["format_compliant"])
        group["token_limit_hits"] += int(row["hit_token_limit"])
    return {
        "version": PROTOCOL_VERSION,
        "answer_matches": sum(row["answer_match"] for row in rows),
        "format_violations": sum(not row["format_compliant"] for row in rows),
        "token_limit_hits": sum(row["hit_token_limit"] for row in rows),
        "non_year_answer_matches": sum(
            row["answer_match"] and not row["year_answer"] for row in rows
        ),
        "year_answer_matches": sum(
            row["answer_match"] and row["year_answer"] for row in rows
        ),
        "by_partition_and_task": dict(sorted(groups.items())),
    }
