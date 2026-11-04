"""Audit N_rel and N_reason token budgets with a pinned Hugging Face tokenizer."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _token_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    token_ids = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return list(token_ids)


def count_rows(
    tokenizer: Any,
    rows: Iterable[dict[str, Any]],
    enable_thinking: bool = False,
) -> dict[str, Any]:
    row_reports = []
    for row in rows:
        user_message = {"role": "user", "content": row["question"]}
        assistant_message = {"role": "assistant", "content": row["answer"]}
        prompt_ids = _token_ids(
            tokenizer,
            [user_message],
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        full_ids = _token_ids(
            tokenizer,
            [user_message, assistant_message],
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        answer_tokens = max(0, len(full_ids) - len(prompt_ids))
        row_reports.append(
            {
                "example_id": row["example_id"],
                "prompt_tokens": len(prompt_ids),
                "answer_tokens": answer_tokens,
                "total_tokens": len(full_ids),
            }
        )
    total = sum(row["total_tokens"] for row in row_reports)
    prompt = sum(row["prompt_tokens"] for row in row_reports)
    answer = sum(row["answer_tokens"] for row in row_reports)
    count = len(row_reports)
    return {
        "rows": count,
        "prompt_tokens": prompt,
        "answer_tokens": answer,
        "total_tokens": total,
        "mean_total_tokens": round(total / count, 6) if count else 0.0,
        "row_reports": row_reports,
    }


def build_report(
    tokenizer: Any,
    n_rel_rows: list[dict[str, Any]],
    n_reason_rows: list[dict[str, Any]],
    tokenizer_config: dict[str, Any],
) -> dict[str, Any]:
    enable_thinking = bool(tokenizer_config.get("thinking_enabled", False))
    n_rel = count_rows(tokenizer, n_rel_rows, enable_thinking=enable_thinking)
    n_reason = count_rows(tokenizer, n_reason_rows, enable_thinking=enable_thinking)
    target = n_rel["total_tokens"]
    observed = n_reason["total_tokens"]
    difference = observed - target
    fraction = abs(difference) / target if target else 0.0
    tolerance = float(tokenizer_config["budget_tolerance_fraction"])
    return {
        "passed": fraction <= tolerance,
        "tokenizer": {
            "model_id": tokenizer_config["model_id"],
            "revision": tokenizer_config["revision"],
            "class": tokenizer.__class__.__name__,
            "thinking_enabled": enable_thinking,
        },
        "tolerance_fraction": tolerance,
        "absolute_difference_tokens": difference,
        "absolute_difference_fraction": round(fraction, 8),
        "n_rel": n_rel,
        "n_reason": n_reason,
        "recommendation": (
            "Budgets are within tolerance"
            if fraction <= tolerance
            else "Adjust N_reason count or prompt length, regenerate, and rerun this audit"
        ),
    }


def main() -> int:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Install requirements/data-audit.txt before running the tokenizer audit.")
        return 2

    config = json.loads(args.config.read_text(encoding="utf-8"))
    tokenizer_config = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["model_id"],
        revision=tokenizer_config["revision"],
    )
    report = build_report(
        tokenizer,
        read_jsonl(args.bundle / "prompts/n_rel_train.jsonl"),
        read_jsonl(args.bundle / "prompts/n_reason_train.jsonl"),
        tokenizer_config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key not in {"n_rel", "n_reason"}}
    summary["n_rel"] = {key: value for key, value in report["n_rel"].items() if key != "row_reports"}
    summary["n_reason"] = {key: value for key, value in report["n_reason"].items() if key != "row_reports"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
