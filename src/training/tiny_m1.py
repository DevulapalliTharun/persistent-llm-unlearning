"""Prepare or execute a bounded 12-unit LoRA learning diagnostic."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.answers import (
    PROTOCOL_VERSION, evaluation_messages, generation_details,
    score_answer, summarize_answer_protocol,
)
from src.m0_knowledge_audit import score_prediction, sha256_file
from src.m0_smoke_test import git_commit, installed_packages, package_version, write_json
from src.training.data import encode_answer, pad_batch, select_pilot, utility_examples


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            for row in rows), encoding="utf-8")


def learning_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for partition in ("F", "R"):
        for category in ("author_book", "multi_hop"):
            subset = [r for r in rows if r["entity_partition"] == partition and
                      (r["task_type"] in {"author_to_book", "book_to_author"}
                       if category == "author_book" else r["hop_count"] > 1)]
            correct = sum(r["answer_match"] for r in subset)
            metrics[f"{partition}/{category}"] = {
                "correct": correct, "count": len(subset),
                "accuracy": correct / len(subset) if subset else 0.0,
            }
    return metrics


def pilot_verdict(config: dict[str, Any], dev: list[dict[str, Any]],
                  base_utility: list[dict[str, Any]], utility: list[dict[str, Any]],
                  reload_matches: bool, adapter_changed: bool, updates: int) -> dict[str, Any]:
    gates = config["pilot_gates"]
    metrics = learning_metrics(dev)
    checks = {}
    for key, result in metrics.items():
        threshold = (gates["direct_author_book_accuracy_min"] if key.endswith("author_book")
                     else gates["multi_hop_accuracy_min"])
        checks[key] = result["accuracy"] >= threshold
    base_acc = sum(r["answer_match"] for r in base_utility) / len(base_utility)
    new_acc = sum(r["answer_match"] for r in utility) / len(utility)
    checks["utility_baseline_measurable"] = base_acc >= gates["utility_base_accuracy_min"]
    checks["utility_preserved"] = new_acc + 1e-12 >= base_acc - gates["utility_accuracy_drop_max"]
    checks["format"] = sum(not r["format_compliant"] for r in dev + utility) <= gates["max_format_violations"]
    checks["generation_limit"] = sum(r["hit_token_limit"] for r in dev + utility) <= gates["max_token_limit_hits"]
    checks["save_reload"] = reload_matches
    checks["adapter_changed"] = adapter_changed
    checks["optimizer_budget"] = updates == config["optimizer_steps"]
    return {"passed": all(checks.values()), "checks": checks, "dev": metrics,
            "base_utility_accuracy": base_acc, "post_utility_accuracy": new_acc,
            "scope": "tiny diagnostic only; not a canonical M1 or evidence for either safeguard"}


def evaluate(model: Any, tokenizer: Any, rows: list[dict[str, Any]],
             config: dict[str, Any]) -> list[dict[str, Any]]:
    import torch
    from transformers import GenerationConfig
    model.eval()
    settings = config["evaluation"]
    generation = GenerationConfig(
        max_new_tokens=settings["max_new_tokens"], do_sample=False,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, use_cache=True,
    )
    output = []
    for offset in range(0, len(rows), settings["batch_size"]):
        batch = rows[offset:offset + settings["batch_size"]]
        texts = [tokenizer.apply_chat_template(evaluation_messages(row["question"]),
                 tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in batch]
        inputs = tokenizer(texts, add_special_tokens=False, padding=True,
                           return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            sequences = model.generate(**inputs, generation_config=generation)
        new_ids = sequences[:, inputs["input_ids"].shape[1]:].tolist()
        responses = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        for row, response, token_ids in zip(batch, responses, new_ids):
            scored = score_prediction(row, response)
            scored.update(score_answer(response, row["answer"]))
            scored.update(generation_details(token_ids, generation.eos_token_id,
                                             settings["max_new_tokens"]))
            output.append(scored)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("plan", "prepare", "train"), default="plan")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["run_type"] != "diagnostic_m1_tiny" or config["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Use the fixed tiny diagnostic config and v2 evaluation protocol")
    if len(config["record_ids"]) != 12 or config["optimizer_steps"] != 120:
        raise ValueError("This runner is bounded to the reviewed 12-unit / 120-step diagnostic")
    if config["evaluation"] != {"batch_size": 16, "max_new_tokens": 128,
                               "do_sample": False, "thinking_enabled": False}:
        raise ValueError("Evaluation must match the verified v2 protocol")
    train, dev = select_pilot(config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory is nonempty; use a fresh run directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {"config": config, "config_sha256": sha256_file(args.config),
            "source_git_commit": git_commit(), "training_rows": len(train),
            "development_rows": len(dev), "utility_rows": len(utility_examples()),
            "model_executed": False, "scope": "12-unit infrastructure/learning diagnostic",
            "source_sha256": {name: sha256_file(Path(__file__).with_name(name))
                              for name in ("tiny_m1.py", "data.py")}}
    write_json(args.output_dir / "plan.json", plan)
    if args.mode == "plan":
        print(json.dumps(plan, indent=2))
        return 0

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["model_revision"], padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_answer(tokenizer, row, config["max_sequence_length"]) for row in train]
    report = {"rows": len(encoded), "max_tokens": max(len(row["input_ids"]) for row in encoded),
              "prompt_tokens": sum(row["prompt_tokens"] for row in encoded),
              "supervised_answer_tokens": sum(row["answer_tokens"] for row in encoded),
              "prefixes_verified": True, "truncated_examples": 0}
    write_json(args.output_dir / "masking_report.json", report)
    write_rows(args.output_dir / "selected_train.jsonl", train)
    write_rows(args.output_dir / "selected_dev.jsonl", dev)
    if args.mode == "prepare":
        print(json.dumps(report, indent=2))
        return 0

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires the Kaggle CUDA kernel; use --mode prepare locally")
    set_seed(config["seed"])
    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            config["model_id"], revision=config["model_revision"], dtype=torch.float16
        ).to("cuda:0")

    base = load_base()
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **config["lora"]))
    model.peft_config["default"].revision = config["model_revision"]
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name or param.dtype != torch.float32 for name, param in trainable):
        raise RuntimeError("Expected only FP32 LoRA parameters to be trainable")
    trainable_count = sum(param.numel() for _, param in trainable)
    initial = {name: param.detach().cpu().clone() for name, param in trainable}
    base_utility = evaluate(model, tokenizer, utility_examples(), config)
    base_dev = evaluate(model, tokenizer, dev, config)
    write_rows(args.output_dir / "base_dev.jsonl", base_dev)
    write_rows(args.output_dir / "base_utility.jsonl", base_utility)

    def collate(rows):
        return {key: torch.tensor(value, dtype=torch.long, device="cuda:0")
                for key, value in pad_batch(rows, tokenizer.pad_token_id).items()}

    loader = torch.utils.data.DataLoader(encoded, batch_size=config["batch_size"], shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"]), collate_fn=collate, num_workers=0)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW([param for _, param in trainable],
                                  lr=config["learning_rate"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    model.train()
    updates = 0
    losses = []
    torch.cuda.reset_peak_memory_stats(0)
    with (args.output_dir / "training.jsonl").open("x", encoding="utf-8") as log:
        for attempt in range(config["optimizer_steps"]):
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            for _ in range(config["gradient_accumulation_steps"]):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                labels = batch.pop("labels")
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**batch, use_cache=False).logits
                    target = labels[:, 1:]
                    token_loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1].float().transpose(1, 2), target,
                        ignore_index=-100, reduction="none")
                    lengths = target.ne(-100).sum(dim=1)
                    loss = (token_loss.sum(dim=1) / lengths).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss; preserving partial run for diagnosis")
                micro_losses.append(float(loss.detach()))
                scaler.scale(loss / config["gradient_accumulation_steps"]).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], config["max_grad_norm"])
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            applied = scaler.get_scale() >= scale_before
            updates += int(applied)
            record = {"attempt": attempt + 1, "optimizer_updates": updates,
                      "mean_answer_ce": sum(micro_losses) / len(micro_losses),
                      "update_applied": applied, "grad_norm": float(norm) if math.isfinite(float(norm)) else None}
            losses.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if (attempt + 1) % 10 == 0:
                print(f"Tiny training attempt {attempt + 1}/120; answer CE={record['mean_answer_ce']:.4f}", flush=True)

    adapter_changed = any(not torch.equal(initial[name], param.detach().cpu()) for name, param in trainable)
    post_dev = evaluate(model, tokenizer, dev, config)
    post_utility = evaluate(model, tokenizer, utility_examples(), config)
    write_rows(args.output_dir / "post_dev.jsonl", post_dev)
    write_rows(args.output_dir / "post_utility.jsonl", post_utility)
    adapter_path = args.output_dir / "adapter"
    model.save_pretrained(adapter_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    # A final optimizer state is available for recovery/inspection, not a full base-model checkpoint.
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "optimizer_updates": updates, "seed": config["seed"]}, args.output_dir / "optimizer_state.pt")
    memory_peak = torch.cuda.max_memory_allocated(0) / 1024**3
    del optimizer, trainable, initial, logits, loss, batch, labels, token_loss, model, base
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(load_base(), adapter_path).eval()
    restored_dev = evaluate(reloaded, tokenizer, dev, config)
    restored_utility = evaluate(reloaded, tokenizer, utility_examples(), config)
    write_rows(args.output_dir / "reloaded_dev.jsonl", restored_dev)
    write_rows(args.output_dir / "reloaded_utility.jsonl", restored_utility)
    reload_matches = ([r["raw_response"] for r in post_dev + post_utility] ==
                      [r["raw_response"] for r in restored_dev + restored_utility])
    verdict = pilot_verdict(config, post_dev, base_utility, post_utility,
                            reload_matches, adapter_changed, updates)
    summary = {"execution_complete": True, "pilot": verdict, "plan": plan,
        "base_dev": learning_metrics(base_dev), "post_dev": summarize_answer_protocol(post_dev),
        "reload_predictions_identical": reload_matches, "adapter_changed": adapter_changed,
        "trainable_parameters": trainable_count,
        "optimizer_updates": updates, "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "seconds": round(time.perf_counter() - start_clock, 4), "gpu": torch.cuda.get_device_name(0),
        "dtype": "torch.float16 base / torch.float32 trainable LoRA", "peak_gpu_memory_gib": memory_peak,
        "versions": {name: package_version(name) for name in ("torch", "transformers", "peft", "accelerate")},
        "artifact_files_sha256": {str(p.relative_to(args.output_dir)): sha256_file(p)
                                  for p in sorted(adapter_path.rglob("*")) if p.is_file()},
        "canonical_m1": False, "artifact_persistence": "local runtime only until private upload verified"}
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "environment.freeze.txt").write_text("\n".join(installed_packages()) + "\n")
    print(json.dumps({"execution_complete": True, "pilot": verdict,
                      "reload_predictions_identical": reload_matches, "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
