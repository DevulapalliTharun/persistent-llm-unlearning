"""Fresh-base full candidate M0 audit in fixed non-thinking/thinking modes."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

from src.evaluation.answers import PROTOCOL_VERSION, SYSTEM_PROMPT, evaluation_messages, summarize_answer_protocol
from src.evaluation.mode_diagnostic import SAMPLING, decode_generation, score_generated
from src.m0_knowledge_audit import load_audit_examples, normalize_answer, sha256_file, summarize_predictions
from src.m0_smoke_test import git_commit, installed_packages, package_version, write_json
from src.training.model_capacity import MODEL_ID, MODEL_REVISION


def prepare_plan(config):
    if (config['run_type'] != 'qwen17_full_m0_dual_mode_v1' or config['model_id'] != MODEL_ID or
            config['model_revision'] != MODEL_REVISION or config['protocol_version'] != PROTOCOL_VERSION or
            config['system_prompt'] != SYSTEM_PROMPT):
        raise ValueError('Use the fixed pinned 1.7B full M0 protocol')
    if (config['prompt_files'] != ['m1_dev.jsonl', 'n_rel_train.jsonl'] or
            config['batch_size'] != 16 or config['seed'] != 42 or
            config['answer_only_max_new_tokens'] != 128 or config['thinking_max_new_tokens'] != 1024 or
            config['sampling'] != SAMPLING):
        raise ValueError('Keep non-final inputs and declared decoding/batching')
    rows, hashes = load_audit_examples(config)
    if len(rows) != 1260 or len({r['record_id'] for r in rows}) != 180:
        raise ValueError('Audit must cover all 180 units and 1260 non-final prompts')
    reviews = {}
    for key in ('structural_review', 'collision_review'):
        path = Path(config[key])
        if sha256_file(path) != config[key + '_sha256']:
            raise ValueError('Frozen candidate review report changed')
        reviews[key] = json.loads(path.read_text())
    flagged = [unit['record_id'] for unit in reviews['collision_review']['sampled_units'] if unit['findings']]
    if not set(flagged) <= {r['record_id'] for r in rows}:
        raise ValueError('Unknown externally flagged candidate record')
    plan = {'config': config, 'input_sha256': hashes, 'planned_predictions': 2520,
            'prompts_per_mode': len(rows), 'knowledge_units': 180,
            'by_entity_partition_per_mode': dict(Counter(r['entity_partition'] for r in rows)),
            'batching': 'Independent per input view; reset seed offsets per file, never mix N_rel with the last F/R dev batch.',
            'externally_flagged_record_ids': flagged, 'training_updates': 0, 'adapters_loaded': False,
            'model_executed': False, 'dataset_accepted': False,
            'scope': 'Candidate baseline evidence only. No final-test generation, filtering, regeneration or full-M1 authorization.'}
    return plan, rows


def score_audit_row(row, text, thinking, prompt, flagged_records):
    scored = score_generated(row, text, thinking, prompt)
    trace = normalize_answer(scored.get('visible_reasoning') or '')
    target = normalize_answer(row['answer'])
    trace_mention = bool(target and re.search(r'(?<!\w)' + re.escape(target) + r'(?!\w)', trace))
    scored['expected_answer_in_visible_reasoning'] = trace_mention
    scored['external_surface_review_flag'] = row['record_id'] in flagged_records
    scored['potential_answer_review'] = bool(scored['answer_match'] or scored['expected_answer_mentioned'] or trace_mention)
    scored['protocol_issue'] = bool(not scored['format_compliant'] or scored.get('boundary_error'))
    scored['needs_match_review'] = bool(scored['potential_answer_review'] or scored['external_surface_review_flag'] or scored['protocol_issue'] or scored.get('boundary_error'))
    return scored


def summarize_mode(rows):
    return {'answer_scoring': summarize_answer_protocol(rows), 'legacy': summarize_predictions(rows),
            'reasoning_answer_mentions': sum(r['expected_answer_in_visible_reasoning'] for r in rows),
            'external_surface_review_rows': sum(r['external_surface_review_flag'] for r in rows),
            'missing_thinking_boundaries': sum(r.get('boundary_error') is not None for r in rows),
            'potential_answer_review_rows': sum(r['potential_answer_review'] for r in rows),
            'protocol_issue_rows': sum(r['protocol_issue'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('plan', 'audit'), default='plan')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    plan, examples = prepare_plan(config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a fresh output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = ['src/evaluation/full_m0.py', 'src/evaluation/answers.py', 'src/evaluation/mode_diagnostic.py',
               'src/m0_knowledge_audit.py', 'src/data/synthetic.py', 'src/training/model_capacity.py']
    plan.update(config_sha256=sha256_file(args.config), source_git_commit=git_commit(),
                source_sha256={name: sha256_file(Path(name)) for name in sources})
    write_json(args.output_dir / 'plan.json', plan)
    if args.mode == 'plan':
        print(json.dumps(plan, indent=2))
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError('Run the fresh-base audit on Kaggle CUDA; plan works locally')
    token = os.environ.get('HF_TOKEN')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, padding_side='left', token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    set_seed(config['seed'])
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                dtype=torch.float16, token=token).to('cuda:0').eval()
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(0)
    write_json(args.output_dir / 'generation_metadata.json', {
        'eos_token_id': model.generation_config.eos_token_id, 'pad_token_id': tokenizer.pad_token_id,
        'think_open_token_id': tokenizer.convert_tokens_to_ids('<think>'),
        'think_close_token_id': tokenizer.convert_tokens_to_ids('</think>'),
        'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()})
    all_predictions, mode_results = [], {}
    for mode in ('answer_only', 'thinking'):
        thinking = mode == 'thinking'
        generation = GenerationConfig(
            **(SAMPLING if thinking else {'do_sample': False}),
            max_new_tokens=config['thinking_max_new_tokens'] if thinking else config['answer_only_max_new_tokens'],
            eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        predictions = []
        with (args.output_dir / (mode + '_predictions.jsonl')).open('x') as log:
            for view in ('m1_dev', 'n_rel_train'):
                rows = [r for r in examples if r['prompt_partition'] == view]
                for offset in range(0, len(rows), config['batch_size']):
                    batch = rows[offset:offset + config['batch_size']]
                    prompts = [tokenizer.apply_chat_template(evaluation_messages(r['question']), tokenize=False,
                        add_generation_prompt=True, enable_thinking=thinking) for r in batch]
                    inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda:0')
                    set_seed(config['seed'] + offset)
                    with torch.inference_mode():
                        ids = model.generate(**inputs, generation_config=generation)[:, inputs['input_ids'].shape[1]:].tolist()
                    for row, prompt, tokens in zip(batch, prompts, ids):
                        text, saved_ids, details = decode_generation(tokenizer, tokens, generation.eos_token_id, generation.max_new_tokens)
                        scored = score_audit_row(row, text, thinking, prompt, set(plan['externally_flagged_record_ids']))
                        scored.update(details)
                        scored.update(mode=mode, thinking_enabled=thinking, batch_seed=config['seed'] + offset,
                                      rendered_prompt=prompt, generated_token_ids=saved_ids)
                        scored['protocol_issue'] = bool(not scored['format_compliant'] or scored['hit_token_limit'] or scored.get('boundary_error'))
                        scored['needs_match_review'] = bool(scored['potential_answer_review'] or scored['external_surface_review_flag'] or scored['protocol_issue'])
                        predictions.append(scored)
                        log.write(json.dumps(scored, ensure_ascii=False, sort_keys=True) + '\n')
                    log.flush()
                    print(f'{mode}/{view}: {offset + len(batch)}/{len(rows)}; {len(predictions)}/1260 mode predictions', flush=True)
        mode_results[mode] = summarize_mode(predictions)
        all_predictions.extend(predictions)
        print(mode, json.dumps(mode_results[mode]), flush=True)
    write_json(args.output_dir / 'potential_answers_for_review.json', {
        'rows': [r for r in all_predictions if r['potential_answer_review']],
        'note': 'Final/trace mentions are review signals, not automatic correct associations or leakage-after-unlearning metrics.'})
    write_json(args.output_dir / 'protocol_issues_for_review.json', {'rows': [r for r in all_predictions if r['protocol_issue']]})
    write_json(args.output_dir / 'external_collision_rows_for_review.json', {'rows': [r for r in all_predictions if r['external_surface_review_flag']]})
    summary = {'execution_complete': len(all_predictions) == 2520, 'model_executed': True,
               'training_updates': 0, 'adapters_loaded': False, 'weights_updated': False,
               'dataset_accepted': False, 'plan': plan, 'modes': mode_results,
               'prediction_files_sha256': {p.name: sha256_file(p) for p in args.output_dir.glob('*_predictions.jsonl')},
               'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
               'seconds': round(time.perf_counter() - clock, 4), 'gpu': torch.cuda.get_device_name(0),
               'dtype': 'torch.float16', 'peak_gpu_memory_gib': torch.cuda.max_memory_allocated(0) / 1024**3,
               'versions': {p: package_version(p) for p in ('torch', 'transformers', 'accelerate')},
               'interpretation': 'Fresh-base evidence only. Preserve matches/mentions, prose, failures and collision flags; no automatic target filtering, dataset acceptance or claim of absent internal knowledge.'}
    write_json(args.output_dir / 'summary.json', summary)
    (args.output_dir / 'environment.freeze.txt').write_text('\n'.join(installed_packages()) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'plan'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
