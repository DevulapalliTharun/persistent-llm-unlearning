"""Bounded inference-only check of thinking mode, with matched sampling controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from src.evaluation.answers import SYSTEM_PROMPT, generation_details, score_answer, summarize_answer_protocol
from src.evaluation.utility_diagnostic import ARITHMETIC_PROMPT, compare_reference, evaluate as evaluate_utility
from src.m0_knowledge_audit import score_prediction, sha256_file
from src.m0_smoke_test import git_commit, installed_packages, package_version, write_json
from src.training.data import read_rows, utility_examples
from src.training.tiny_m1 import evaluate as evaluate_dev, learning_metrics


SAMPLING = {'do_sample': True, 'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0}


def assistant_parts(generated_text: str, thinking: bool, prompt: str = '') -> dict[str, Any]:
    """Expose generated text, never count a target inside an unfinished trace."""
    full = generated_text
    if thinking and prompt.rstrip().endswith('<think>'):
        full = '<think>\n' + full
    if not thinking:
        return {'raw_response': full, 'visible_reasoning': None,
                'final_answer_text': full, 'thinking_closed': None, 'boundary_error': None}
    valid = (full.count('<think>') == full.count('</think>') == 1 and
             not full.split('<think>', 1)[0].strip())
    if not valid:
        return {'raw_response': full, 'visible_reasoning': full,
                'final_answer_text': '', 'thinking_closed': False,
                'boundary_error': 'missing_or_malformed_thinking_boundary'}
    trace, final = full.split('<think>', 1)[1].split('</think>', 1)
    return {'raw_response': full, 'visible_reasoning': trace.strip(),
            'final_answer_text': final.strip(), 'thinking_closed': True, 'boundary_error': None}


def score_generated(row: dict[str, Any], generated_text: str, thinking: bool, prompt: str = '') -> dict[str, Any]:
    parts = assistant_parts(generated_text, thinking, prompt)
    scored = score_prediction(row, parts['raw_response'])
    scored.update(score_answer(parts['raw_response'], row['answer']))
    if parts['boundary_error']:
        scored.update(extracted_answer=None, answer_match=False, format_compliant=False,
                      parse_error=parts['boundary_error'])
    return {**scored, **parts, 'generated_text': generated_text}


def decode_generation(tokenizer: Any, tokens: list[int], eos_ids: Any, cap: int):
    details = generation_details(tokens, eos_ids, cap)
    saved_ids = tokens[:details['generated_token_count']]
    content_ids = saved_ids[:-1] if details['eos_generated'] else saved_ids
    # Preserve thinking delimiters even if the tokenizer marks them special.
    text = tokenizer.decode(content_ids, skip_special_tokens=False)
    return text, saved_ids, details


def prepare_plan(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if config['run_type'] != 'thinking_mode_diagnostic_v1':
        raise ValueError('Use the fixed thinking-mode diagnostic config')
    if config['sampling'] != SAMPLING or config['sampled_max_new_tokens'] != 1024:
        raise ValueError('Use the prespecified matched sampling and bounded output budget')
    if config['batch_size'] != 16 or config['seed'] != 42:
        raise ValueError('Use the fixed batching and seed')
    reference = Path(config['reference_dir'])
    for name, expected in config['reference_sha256'].items():
        if sha256_file(reference / name) != expected:
            raise ValueError(f'Reference hash differs: {name}')
    tiny_summary = json.loads((reference / 'summary.json').read_text())
    if any(config[k] != tiny_summary['plan']['config'][k] for k in ('model_id', 'model_revision')):
        raise ValueError('Keep the original pinned base')
    dev = read_rows(reference / 'selected_dev.jsonl')
    if len(dev) != 84 or any(r['prompt_partition'] != 'm1_dev' for r in dev):
        raise ValueError('Use all 84 original development rows; no final-test data')
    for checkpoint in ('base', 'adapter'):
        control_file = Path(config[f'{checkpoint}_arithmetic_reference'])
        if sha256_file(control_file) != config[f'{checkpoint}_arithmetic_reference_sha256']:
            raise ValueError('Utility prompt diagnostic reference changed')
    receipt = json.loads((reference / 'persistence.json').read_text())
    expected_artifact = {'repo_id': receipt['repo_id'], 'revision': receipt['artifact_revision'], 'path': receipt['path']}
    if config['artifact'] != expected_artifact:
        raise ValueError('Use the original verified tiny adapter revision')
    plan = {'config': config, 'training_updates': 0, 'model_executed': False,
            'base_suites': {'utility': 10}, 'adapter_suites': {'utility': 10, 'development': 84},
            'arms': ['greedy_reference', 'sampled_no_thinking', 'sampled_thinking'],
            'prediction_count': 312, 'thinking_prediction_count': 104,
            'system_prompts': {'utility': ARITHMETIC_PROMPT, 'development': SYSTEM_PROMPT},
            'scope': 'Development-only mode feasibility; no canonical M1 or retrospective gate changes.',
            'limits': 'One sampling seed, 1024-token cap; visible text is not a complete record of internal reasoning.'}
    return plan, dev


def sampled_evaluation(model: Any, tokenizer: Any, rows: list[dict[str, Any]],
                       thinking: bool, config: dict[str, Any], output_path: Path | None = None) -> list[dict[str, Any]]:
    import torch
    from transformers import GenerationConfig, set_seed

    output = []
    model.eval()
    generation = GenerationConfig(**SAMPLING, max_new_tokens=config['sampled_max_new_tokens'],
        eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    for offset in range(0, len(rows), config['batch_size']):
        batch = rows[offset:offset + config['batch_size']]
        prompts = [tokenizer.apply_chat_template([
            {'role': 'system', 'content': ARITHMETIC_PROMPT if r['entity_partition'] == 'utility' else SYSTEM_PROMPT},
            {'role': 'user', 'content': r['question']}], tokenize=False,
            add_generation_prompt=True, enable_thinking=thinking) for r in batch]
        inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda:0')
        batch_seed = config['seed'] + offset
        set_seed(batch_seed)
        started = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**inputs, generation_config=generation)[:, inputs['input_ids'].shape[1]:].tolist()
        seconds = time.perf_counter() - started
        for row, prompt, tokens in zip(batch, prompts, ids):
            text, saved_ids, details = decode_generation(tokenizer, tokens, generation.eos_token_id, generation.max_new_tokens)
            scored = score_generated(row, text, thinking, prompt)
            scored.update(details)
            scored.update(rendered_prompt=prompt, generated_token_ids=saved_ids,
                          thinking_enabled=thinking, batch_seed=batch_seed, batch_seconds=round(seconds, 4))
            output.append(scored)
        if output_path is not None:
            with output_path.open('x' if offset == 0 else 'a') as log:
                for scored in output[-len(batch):]:
                    log.write(json.dumps(scored, ensure_ascii=False, sort_keys=True) + '\n')
        print(f"Thinking={thinking}: generated {len(output)}/{len(rows)} predictions", flush=True)
    return output


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = summarize_answer_protocol(rows)
    result.update(count=len(rows), missing_thinking_boundaries=sum(r.get('boundary_error') is not None for r in rows),
                  abstentions=sum(r['extracted_answer'] == 'unknown' for r in rows))
    if rows[0]['entity_partition'] != 'utility':
        result['learning_metrics'] = learning_metrics(rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('plan', 'evaluate'), default='plan')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    plan, dev = prepare_plan(config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a fresh output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = ['src/evaluation/mode_diagnostic.py', 'src/evaluation/utility_diagnostic.py',
               'src/evaluation/answers.py', 'src/training/data.py', 'src/training/tiny_m1.py', 'src/m0_knowledge_audit.py']
    plan.update(source_git_commit=git_commit(), config_sha256=sha256_file(args.config),
                source_sha256={name: sha256_file(Path(name)) for name in sources})
    write_json(args.output_dir / 'plan.json', plan)
    if args.mode == 'plan':
        print(json.dumps(plan, indent=2))
        return 0

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from peft import PeftModel
    from peft.import_utils import is_torchao_available
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_torchao_available()
    if not torch.cuda.is_available():
        raise RuntimeError('Use a Kaggle CUDA kernel; --mode plan is local-only')
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('Attach HF_TOKEN using Kaggle Secrets')
    artifact = config['artifact']
    if not HfApi(token=token).repo_info(artifact['repo_id'], repo_type='model').private:
        raise ValueError('Expected private artifact storage')
    original = Path(config['reference_dir'])
    original_summary = json.loads((original / 'summary.json').read_text())
    restore = args.output_dir / 'restored_source'
    hashes = original_summary['artifact_files_sha256']
    if set(hashes) != {'adapter/README.md', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'}:
        raise ValueError('Unexpected adapter manifest')
    for name, expected in hashes.items():
        file = Path(hf_hub_download(repo_id=artifact['repo_id'], filename=artifact['path'] + '/' + name,
                    revision=artifact['revision'], token=token, local_dir=restore))
        if sha256_file(file) != expected:
            raise ValueError(f'Adapter hash mismatch: {name}')
    adapter_path = restore / artifact['path'] / 'adapter'
    adapter_config = json.loads((adapter_path / 'adapter_config.json').read_text())
    if adapter_config['base_model_name_or_path'] != config['model_id'] or adapter_config['revision'] != config['model_revision']:
        raise ValueError('Adapter/base lineage differs')
    write_json(args.output_dir / 'source_artifact_verification.json', {'artifact': artifact, 'verified_sha256': hashes})
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'],
                                              padding_side='left', token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(config['model_id'], revision=config['model_revision'],
                                               dtype=torch.float16, token=token).to('cuda:0').eval()
    write_json(args.output_dir / 'generation_metadata.json', {
        'eos_token_id': model.generation_config.eos_token_id, 'pad_token_id': tokenizer.pad_token_id,
        'think_open_token_id': tokenizer.convert_tokens_to_ids('<think>'),
        'think_close_token_id': tokenizer.convert_tokens_to_ids('</think>'),
        'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        'sampling': SAMPLING, 'sampled_max_new_tokens': config['sampled_max_new_tokens']})
    groups, controls = {}, {}
    greedy_config = {'evaluation': {'batch_size': 16, 'max_new_tokens': 128, 'do_sample': False, 'thinking_enabled': False}}
    for checkpoint in ('base', 'adapter'):
        if checkpoint == 'adapter':
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False).eval()
        suites = {'utility': utility_examples()}
        if checkpoint == 'adapter':
            suites['development'] = dev
        for suite, rows in suites.items():
            for arm in plan['arms']:
                name = f'{checkpoint}_{suite}_{arm}'
                output_path = args.output_dir / (name + '.jsonl')
                if arm == 'greedy_reference':
                    if suite == 'utility':
                        predictions = evaluate_utility(model, tokenizer, 'arithmetic-v1', greedy_config)
                        if checkpoint == 'base':
                            reference = read_rows(Path(config['base_arithmetic_reference']))
                        else:
                            reference = read_rows(Path(config['adapter_arithmetic_reference']))
                        controls[f'{checkpoint}/{suite}'] = compare_reference(predictions, reference)
                    else:
                        predictions = evaluate_dev(model, tokenizer, rows, greedy_config)
                        reference = read_rows(original / 'post_dev.jsonl')
                        differences = [r['example_id'] for r, old in zip(predictions, reference) if r['raw_response'] != old['raw_response']]
                        controls[f'{checkpoint}/{suite}'] = {'raw_predictions_identical': not differences, 'different_example_ids': differences}
                else:
                    predictions = sampled_evaluation(model, tokenizer, rows, arm == 'sampled_thinking', config, output_path)
                if arm == 'greedy_reference':
                    with output_path.open('x') as output:
                        for row in predictions:
                            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
                groups[name] = metrics(predictions)
                print(f"{name}: {groups[name]['answer_matches']}/{len(rows)} strict correct; "
                      f"{groups[name]['token_limit_hits']} caps; {groups[name]['missing_thinking_boundaries']} boundary failures", flush=True)
    summary = {'execution_complete': True, 'model_executed': True, 'training_updates': 0,
               'canonical_m1': False, 'original_pilot_result_unchanged': True, 'plan': plan,
               'groups': groups, 'greedy_reproduction': controls,
               'controls_reproduced': all(c['raw_predictions_identical'] for c in controls.values()),
               'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
               'seconds': round(time.perf_counter() - start_clock, 4), 'gpu': torch.cuda.get_device_name(0),
               'versions': {p: package_version(p) for p in ('torch', 'transformers', 'peft', 'accelerate')},
               'prediction_files_sha256': {p.name: sha256_file(p) for p in sorted(args.output_dir.glob('*.jsonl'))},
               'interpretation': 'Compare sampled thinking/non-thinking within checkpoint and suite; one seed and cap limit conclusions. Greedy controls test reproducibility, not an isolated mode effect. Visible traces are diagnostic text, not certified internal reasoning or post-unlearning leakage.'}
    write_json(args.output_dir / 'summary.json', summary)
    (args.output_dir / 'environment.freeze.txt').write_text('\n'.join(installed_packages()) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'plan'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
