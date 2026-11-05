"""Compare two utility instructions on the pinned base and saved tiny adapter.

Inference only: no optimizer, training, final-test data or baseline replacement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from src.evaluation.answers import (
    SYSTEM_PROMPT, generation_details, score_answer, summarize_answer_protocol,
)
from src.m0_knowledge_audit import score_prediction, sha256_file
from src.m0_smoke_test import git_commit, installed_packages, package_version, write_json
from src.training.data import read_rows, utility_examples


ARITHMETIC_PROMPT = (
    "Solve the arithmetic question. Return exactly one line: Answer: <integer>. "
    "Give only the integer after the label, without units or explanation."
)
VARIANTS = {"original-v2": SYSTEM_PROMPT, "arithmetic-v1": ARITHMETIC_PROMPT}


def diagnostic_plan(config: dict[str, Any]) -> dict[str, Any]:
    if config['run_type'] != 'utility_prompt_diagnostic_v1':
        raise ValueError('Use the versioned utility diagnostic config')
    if config['evaluation'] != {'batch_size': 16, 'max_new_tokens': 128,
                                'do_sample': False, 'thinking_enabled': False}:
        raise ValueError('Keep the original tiny-run decoding settings')
    return {'config': config, 'system_prompts': VARIANTS,
            'questions': utility_examples(), 'question_count': 10,
            'planned_prediction_count': 40, 'training_updates': 0,
            'scope': 'Utility prompt sensitivity only; original pilot gates remain failed.',
            'interpretation': 'The full system instruction changes; this does not isolate any single clause.'}


def compare_reference(actual: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, Any]:
    expected_inputs = utility_examples()
    for rows in (actual, reference):
        if len(rows) != len(expected_inputs):
            raise ValueError('Utility reference must cover the fixed ten questions')
        for row, expected in zip(rows, expected_inputs):
            if any(row[key] != value for key, value in expected.items()):
                raise ValueError('Utility reference input/order differs')
    differences = [a['example_id'] for a, b in zip(actual, reference)
                   if a['raw_response'] != b['raw_response']]
    return {'raw_predictions_identical': not differences, 'different_example_ids': differences}


def evaluate(model: Any, tokenizer: Any, variant: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    import torch
    from transformers import GenerationConfig

    model.eval()
    rows = utility_examples()
    prompts = [tokenizer.apply_chat_template(
        [{'role': 'system', 'content': VARIANTS[variant]},
         {'role': 'user', 'content': row['question']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in rows]
    inputs = tokenizer(prompts, add_special_tokens=False, padding=True,
                       return_tensors='pt').to('cuda:0')
    generation = GenerationConfig(
        max_new_tokens=config['evaluation']['max_new_tokens'], do_sample=False,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, use_cache=True)
    with torch.inference_mode():
        sequences = model.generate(**inputs, generation_config=generation)
    ids = sequences[:, inputs['input_ids'].shape[1]:].tolist()
    responses = tokenizer.batch_decode(ids, skip_special_tokens=True)
    predictions = []
    for row, prompt, response, tokens in zip(rows, prompts, responses, ids):
        scored = score_prediction(row, response)
        scored.update(score_answer(response, row['answer']))
        scored.update(generation_details(tokens, generation.eos_token_id, generation.max_new_tokens))
        scored.update({'prompt_variant': variant, 'rendered_prompt': prompt,
                       'generated_token_ids': tokens[:scored['generated_token_count']]})
        predictions.append(scored)
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('plan', 'evaluate'), default='plan')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    plan = diagnostic_plan(config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Output directory is nonempty; choose a fresh directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan.update({'source_git_commit': git_commit(), 'config_sha256': sha256_file(args.config),
                 'source_sha256': {str(p): sha256_file(p) for p in [
                     Path(__file__), Path('src/evaluation/answers.py'),
                     Path('src/training/data.py'), Path('src/m0_knowledge_audit.py')]},
                 'model_executed': False})
    write_json(args.output_dir / 'plan.json', plan)
    if args.mode == 'plan':
        print(json.dumps(plan, indent=2))
        return 0

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from peft import PeftModel
    from peft.import_utils import is_torchao_available
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    is_torchao_available()  # Fail early on an incompatible optional backend.
    if not torch.cuda.is_available():
        raise RuntimeError('Use the Kaggle CUDA kernel; --mode plan works locally')
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('Attach HF_TOKEN through Kaggle Secrets; never put it in source')
    artifact = config['artifact']
    api = HfApi(token=token)
    if not api.repo_info(artifact['repo_id'], repo_type='model').private:
        raise ValueError('Expected the configured private artifact repository')
    restore = args.output_dir / 'restored_source'
    verified = {}

    def download(name: str, expected: str) -> Path:
        path = Path(hf_hub_download(
            repo_id=artifact['repo_id'], filename=artifact['path'] + '/' + name,
            revision=artifact['revision'], token=token, local_dir=restore))
        if sha256_file(path) != expected:
            raise ValueError(f'Private artifact checksum mismatch: {name}')
        verified[name] = expected
        return path

    paths = {name: download(name, expected) for name, expected in artifact['evidence_sha256'].items()}
    original_summary = json.loads(paths['summary.json'].read_text())
    original_config = original_summary['plan']['config']
    if any(config[key] != original_config[key] for key in ('model_id', 'model_revision', 'evaluation', 'seed')):
        raise ValueError('Model or decoding settings differ from the original diagnostic')
    adapter_hashes = original_summary['artifact_files_sha256']
    if set(adapter_hashes) != {'adapter/README.md', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'}:
        raise ValueError('Unexpected adapter file manifest')
    for name, expected in adapter_hashes.items():
        download(name, expected)
    adapter_path = restore / artifact['path'] / 'adapter'
    adapter_config = json.loads((adapter_path / 'adapter_config.json').read_text())
    if (adapter_config['base_model_name_or_path'] != config['model_id'] or
            adapter_config['revision'] != config['model_revision']):
        raise ValueError('Adapter/base lineage differs')
    write_json(args.output_dir / 'source_artifact_verification.json', {
        'artifact': artifact, 'files_sha256_verified': verified})
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'],
                                              padding_side='left', token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    set_seed(config['seed'])
    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'], revision=config['model_revision'], dtype=torch.float16, token=token).to('cuda:0').eval()
    groups, controls = {}, {}
    for checkpoint in ('base', 'adapter'):
        if checkpoint == 'adapter':
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False).eval()
        for variant in VARIANTS:
            rows = evaluate(model, tokenizer, variant, config)
            filename = f'{checkpoint}_{variant}.jsonl'
            with (args.output_dir / filename).open('x') as output:
                for row in rows:
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            metrics = summarize_answer_protocol(rows)
            metrics['abstentions'] = sum(r['extracted_answer'] == 'unknown' for r in rows)
            metrics['count'] = len(rows)
            groups[f'{checkpoint}/{variant}'] = metrics
            if variant == 'original-v2':
                reference_name = 'base_utility.jsonl' if checkpoint == 'base' else 'post_utility.jsonl'
                controls[checkpoint] = compare_reference(rows, read_rows(paths[reference_name]))
            print(f"{checkpoint}/{variant}: {metrics['answer_matches']}/10 strict correct; "
                  f"{metrics['abstentions']} Unknown", flush=True)
    summary = {'execution_complete': True, 'model_executed': True, 'training_updates': 0,
               'canonical_m1': False, 'original_pilot_result_unchanged': True,
               'plan': plan, 'groups': groups, 'original_v2_reproduction': controls,
               'paired_comparison_ready': all(c['raw_predictions_identical'] for c in controls.values()),
               'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
               'seconds': round(time.perf_counter() - start_clock, 4),
               'gpu': torch.cuda.get_device_name(0), 'dtype': 'torch.float16',
               'versions': {p: package_version(p) for p in ('torch', 'transformers', 'peft', 'accelerate')},
               'prediction_files_sha256': {p.name: sha256_file(p) for p in sorted(args.output_dir.glob('*.jsonl'))},
               'interpretation': 'Inspect raw answers. If original-v2 controls differ, diagnose reproduction before attributing changes to instructions. This is not a new pilot pass or a trace evaluation.'}
    write_json(args.output_dir / 'summary.json', summary)
    (args.output_dir / 'environment.freeze.txt').write_text('\n'.join(installed_packages()) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'plan'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
