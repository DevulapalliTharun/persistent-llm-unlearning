"""Collect base-model predictions on candidate facts for manual review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.synthetic import verify_checksums
from src.evaluation.answers import (
    PROTOCOL_VERSION, SYSTEM_PROMPT, evaluation_messages, generation_details,
    score_answer, summarize_answer_protocol,
)
from src.m0_smoke_test import git_commit, installed_packages, package_version, write_json


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_answer(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split()).strip(
        " .!?\"'“”‘’`"
    )


def score_prediction(example: dict[str, Any], raw_response: str) -> dict[str, Any]:
    unclosed_thinking = "<think>" in raw_response and "</think>" not in raw_response
    final_response = "" if unclosed_thinking else raw_response.rsplit("</think>", 1)[-1].strip()
    normalized_expected = normalize_answer(example["answer"])
    normalized_response = normalize_answer(final_response)
    mentioned = bool(re.search(
        r"(?<!\w)" + re.escape(normalized_expected) + r"(?!\w)",
        normalized_response,
    ))
    exact = bool(normalized_expected) and normalized_expected == normalized_response
    return {
        **example,
        "raw_response": raw_response,
        "final_response": final_response,
        "normalized_exact_match": exact,
        "expected_answer_mentioned": mentioned,
        "year_answer": example["target_field"] == "award_year",
        "needs_match_review": exact or mentioned,
        "unclosed_thinking": unclosed_thinking,
    }


def load_audit_examples(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    bundle = Path(config["bundle"])
    checksum_index = bundle / "checksums.sha256"
    if sha256_file(checksum_index) != config["expected_checksum_index_sha256"]:
        raise ValueError("Candidate checksum index differs from the configured audit target")
    errors = verify_checksums(bundle)
    if errors:
        raise ValueError(f"Candidate checksum verification failed: {errors}")
    examples = []
    hashes = {}
    allowed_files = {"m1_dev.jsonl", "n_rel_train.jsonl"}
    for filename in config["prompt_files"]:
        if filename not in allowed_files:
            raise ValueError("Audit inputs must use the designated non-final prompt files")
        path = bundle / "prompts" / filename
        hashes[filename] = sha256_file(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row["prompt_partition"] not in {"m1_dev", "n_rel_train"}:
                raise ValueError("Unexpected prompt partition in audit input")
        examples.extend(rows)
    if not examples or len({row["example_id"] for row in examples}) != len(examples):
        raise ValueError("Audit input is empty or contains duplicate example IDs")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    expected_records = sum(manifest["counts"].values())
    if len({row["record_id"] for row in examples}) != expected_records:
        raise ValueError("Audit inputs do not cover all candidate knowledge units")
    return examples, hashes


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {}
    for row in rows:
        key = f"{row['entity_partition']}/{row['task_type']}"
        group = groups.setdefault(key, {"count": 0, "exact_matches": 0, "answer_mentions": 0})
        group["count"] += 1
        group["exact_matches"] += int(row["normalized_exact_match"])
        group["answer_mentions"] += int(row["expected_answer_mentioned"])
    return {
        "predictions": len(rows),
        "exact_matches": sum(row["normalized_exact_match"] for row in rows),
        "non_year_exact_matches": sum(
            row["normalized_exact_match"] and not row["year_answer"] for row in rows
        ),
        "year_exact_matches": sum(
            row["normalized_exact_match"] and row["year_answer"] for row in rows
        ),
        "review_flags": sum(row["needs_match_review"] for row in rows),
        "empty_final_responses": sum(not row["final_response"] for row in rows),
        "by_partition_and_task": dict(sorted(groups.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true", help="Verify inputs without loading a model")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    protocol = config.get("protocol_version", "legacy-v1")
    if protocol not in {"legacy-v1", PROTOCOL_VERSION}:
        raise ValueError(f"Unknown audit protocol: {protocol}")
    if protocol == PROTOCOL_VERSION and config.get("system_prompt") != SYSTEM_PROMPT:
        raise ValueError("Versioned system prompt differs from the shared answer protocol")
    if not re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]):
        raise ValueError("Use an immutable 40-character model revision")
    if config["batch_size"] < 1 or config["max_new_tokens"] < 1:
        raise ValueError("Batch size and generation length must be positive")
    if config["thinking_enabled"] or config["do_sample"]:
        raise ValueError("This audit protocol requires non-thinking, greedy decoding")
    examples, input_hashes = load_audit_examples(config)
    plan = {
        "config": config,
        "config_sha256": sha256_file(args.config),
        "input_sha256": input_hashes,
        "git_commit": git_commit(),
        "planned_predictions": len(examples),
        "knowledge_units": len({row["record_id"] for row in examples}),
        "by_entity_partition": dict(Counter(row["entity_partition"] for row in examples)),
        "model_executed": False,
        "protocol_version": protocol,
        "runner_sha256": sha256_file(Path(__file__)),
        "answer_protocol_sha256": sha256_file(
            Path(__file__).parent / "evaluation" / "answers.py"
        ),
    }
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is nonempty; choose a new run directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "plan.json", plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Connect to a Kaggle GPU kernel to execute this audit")
    set_seed(config["seed"])
    started = datetime.now(timezone.utc).isoformat()
    clock_start = time.perf_counter()
    # T4 uses FP16. BF16 is selected only on hardware with native BF16 capability.
    major, _ = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if major >= 8 and torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["model_revision"], dtype=dtype
    ).to(device).eval()
    torch.cuda.reset_peak_memory_stats(0)
    generation_config = GenerationConfig(
        max_new_tokens=config["max_new_tokens"], do_sample=False,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, use_cache=True,
    )

    predictions = []
    with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8") as output:
        for offset in range(0, len(examples), config["batch_size"]):
            batch = examples[offset : offset + config["batch_size"]]
            formatted = [tokenizer.apply_chat_template(
                evaluation_messages(row["question"])
                if protocol == PROTOCOL_VERSION
                else [{"role": "user", "content": row["question"]}],
                tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ) for row in batch]
            inputs = tokenizer(
                formatted, add_special_tokens=False, padding=True, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=generation_config)
            prompt_width = inputs["input_ids"].shape[1]
            new_ids = generated[:, prompt_width:].tolist()
            responses = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            for index, (example, response) in enumerate(zip(batch, responses)):
                prediction = score_prediction(example, response)
                if protocol == PROTOCOL_VERSION:
                    prediction.update(score_answer(response, example["answer"]))
                    prediction.update(generation_details(
                        new_ids[index], generation_config.eos_token_id,
                        config["max_new_tokens"],
                    ))
                    prediction["needs_match_review"] = (
                        prediction["needs_match_review"] or prediction["answer_match"]
                        or not prediction["format_compliant"] or prediction["hit_token_limit"]
                    )
                predictions.append(prediction)
                output.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            print(f"Audited {len(predictions)}/{len(examples)} prompts", flush=True)

    torch.cuda.synchronize(0)
    summary = {
        **summarize_predictions(predictions),
        "execution_complete": len(predictions) == len(examples),
        "dataset_acceptance": "pending_manual_review",
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": round(time.perf_counter() - clock_start, 4),
        "plan": plan,
        "model_executed": True,
        "runtime": {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "huggingface_hub": package_version("huggingface-hub"),
            "dtype": str(dtype), "device": str(device),
            "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
            "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated(0) / 1024**3, 4),
        },
        "predictions_sha256": sha256_file(args.output_dir / "predictions.jsonl"),
        "protocol_version": protocol,
        "interpretation": "Matches are review flags, not proof of pretraining knowledge; year matches can occur by chance. No automatic exclusion or dataset change.",
    }
    if protocol == PROTOCOL_VERSION:
        summary["answer_scoring"] = summarize_answer_protocol(predictions)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "matches_for_review.json", {
        "matches": [row for row in predictions if row["needs_match_review"]]
    })
    (args.output_dir / "environment.freeze.txt").write_text(
        "\n".join(installed_packages()) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "plan"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
