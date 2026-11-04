"""Reproducible M0 load-and-inference smoke test.

This script performs inference only. It does not train, alter, merge, or upload
model weights.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--revision",
        default="main",
        help="Requested Hub revision. It is resolved to an immutable commit before loading.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m0-smoke"),
    )
    parser.add_argument(
        "--prompt",
        default="In one short sentence, state that the model loaded successfully.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit a CPU-only run. The project M0 gate still requires a later CUDA run.",
    )
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def installed_packages() -> list[str]:
    packages = {
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return sorted(packages, key=str.lower)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        import transformers
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as error:
        print(f"Missing dependency: {error}", file=sys.stderr)
        print("Install requirements/m0-smoke.txt and retry.", file=sys.stderr)
        return 2

    cuda_available = torch.cuda.is_available()
    if not cuda_available and not args.allow_cpu:
        print(
            "CUDA is unavailable. Enable a Kaggle GPU accelerator or pass --allow-cpu "
            "for a non-gating diagnostic run.",
            file=sys.stderr,
        )
        return 3

    set_seed(args.seed)
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map: str | None = "auto"
    else:
        dtype = torch.float32
        device_map = None

    started_at = datetime.now(timezone.utc).isoformat()
    resolved_revision = HfApi().model_info(
        args.model_id,
        revision=args.revision,
    ).sha
    if not resolved_revision:
        raise RuntimeError("Hugging Face did not return an immutable model revision.")

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=resolved_revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=resolved_revision,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )
    if cuda_available:
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started

    input_length = inputs["input_ids"].shape[-1]
    output_ids = generated[0, input_length:]
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    if not response:
        raise RuntimeError("The model loaded but generated an empty response.")

    result: dict[str, Any] = {
        "success": True,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "model": {
            "id": args.model_id,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "class": model.__class__.__name__,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "dtype": str(dtype),
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "chat_template_present": bool(tokenizer.chat_template),
            "thinking_enabled": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": package_version("accelerate"),
            "huggingface_hub": package_version("huggingface-hub"),
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "gpu_count": torch.cuda.device_count() if cuda_available else 0,
            "gpus": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if cuda_available
            else [],
            "peak_gpu_memory_gib": round(
                torch.cuda.max_memory_allocated() / (1024**3), 4
            )
            if cuda_available
            else None,
        },
        "inference": {
            "seed": args.seed,
            "prompt": args.prompt,
            "response": response,
            "input_tokens": int(input_length),
            "generated_tokens": int(output_ids.shape[-1]),
            "max_new_tokens": args.max_new_tokens,
            "load_seconds": round(load_seconds, 4),
            "generation_seconds": round(generation_seconds, 4),
            "decoding": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            },
        },
    }

    result_path = args.output_dir / "result.json"
    environment_path = args.output_dir / "environment.freeze.txt"
    write_json(result_path, result)
    environment_path.write_text(
        "\n".join(installed_packages()) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Result written to: {result_path}")
    print(f"Environment written to: {environment_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
