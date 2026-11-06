"""Refresh whole changed M0 batches, retaining unchanged-batch evidence with provenance."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from src.evaluation.answers import evaluation_messages, generation_details
from src.evaluation.full_m0 import prepare_plan as parent_plan, score_audit_row, summarize_mode
from src.evaluation.mode_diagnostic import decode_generation
from src.m0_knowledge_audit import load_audit_examples, sha256_file
from src.m0_smoke_test import git_commit, installed_packages, package_version, write_json
from src.training.data import read_rows


MODES = ('answer_only', 'thinking')
VIEWS = ('m1_dev', 'n_rel_train')


def read(path):
    return json.loads(Path(path).read_text())


def batch_specs(original, corrected, batch_size):
    """Changes to labels also invalidate the whole original inference batch."""
    if [r['example_id'] for r in original] != [r['example_id'] for r in corrected]:
        raise ValueError('Refresh must preserve all input IDs and ordering')
    affected, controls = [], []
    for view in VIEWS:
        old = [r for r in original if r['prompt_partition'] == view]
        new = [r for r in corrected if r['prompt_partition'] == view]
        if not old:
            raise ValueError('Missing required input view')
        for offset in range(0, len(old), batch_size):
            before, after = old[offset:offset + batch_size], new[offset:offset + batch_size]
            spec = {'view': view, 'offset': offset, 'count': len(after),
                    'example_ids': [r['example_id'] for r in after]}
            if before != after:
                affected.append(spec)
            elif not any(c['view'] == view for c in controls):
                controls.append(spec)
    if len(controls) != len(VIEWS) or not affected:
        raise ValueError('Need changed batches and an unchanged control batch per view')
    return affected, controls


def prepare_plan(config):
    if config['run_type'] != 'qwen17_corrected_m0_batch_refresh_v1':
        raise ValueError('Unknown corrected audit protocol')
    old_config_path = Path(config['parent_config'])
    if sha256_file(old_config_path) != config['parent_config_sha256']:
        raise ValueError('Parent audit config changed')
    old_config = read(old_config_path)
    for key, value in old_config.items():
        if key not in {'run_type', 'bundle', 'expected_checksum_index_sha256'} and config.get(key) != value:
            raise ValueError(f'Preserve original model/prompt/decoding/review policy: {key}')
    old_plan, old_inputs = parent_plan(old_config)
    new_inputs, hashes = load_audit_examples(config)
    bundle = Path(config['bundle'])
    correction_path = bundle / 'reports/correction.json'
    if sha256_file(correction_path) != config['correction_report_sha256']:
        raise ValueError('Correction report changed')
    correction = read(correction_path)
    patch = correction['correction']
    if correction['parent_checksum_index_sha256'] != old_config['expected_checksum_index_sha256']:
        raise ValueError('Correction refers to another parent candidate')
    if len(new_inputs) != 1260:
        raise ValueError('Incomplete corrected coverage')
    changed = []
    for old, new in zip(old_inputs, new_inputs):
        if old != new:
            expected = deepcopy(old)
            for key in ('question', 'answer'):
                expected[key] = expected[key].replace(patch['old_value'], patch['new_value'])
            if expected != new or old['record_id'] != patch['record_id']:
                raise ValueError('Unexpected change outside documented title substitution')
            changed.append(new['example_id'])
    if sorted(changed) != sorted(correction['changed_prompt_ids']['m1_dev']):
        raise ValueError('Changed audit inputs differ from correction report')
    run = Path(config['parent_run'])
    if sha256_file(run / 'artifact_verification.json') != config['parent_verification_sha256']:
        raise ValueError('Frozen parent verification report changed')
    verification = read(run / 'artifact_verification.json')
    if not verification['verified'] or not verification['offline_tokenizer_render_and_decode_replayed']:
        raise ValueError('Parent evidence was not independently verified')
    for name, digest in verification['original_files_sha256'].items():
        if sha256_file(run / name) != digest:
            raise ValueError(f'Parent evidence changed: {name}')
    summary = read(run / 'summary.json')
    if (not summary['execution_complete'] or summary['training_updates'] != 0 or summary['adapters_loaded']
            or summary['weights_updated'] or summary['plan']['config'] != old_config):
        raise ValueError('Parent is not the complete frozen fresh-base audit')
    for name, digest in summary['plan']['source_sha256'].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f'Original audit implementation changed: {name}')
    if summary['plan']['input_sha256'] != old_plan['input_sha256']:
        raise ValueError('Parent input provenance differs')
    parent_rows = {}
    for mode in MODES:
        rows = read_rows(run / f'{mode}_predictions.jsonl')
        if len(rows) != 1260 or summarize_mode(rows) != summary['modes'][mode]:
            raise ValueError('Parent summary/coverage differs')
        for row, example in zip(rows, old_inputs):
            if any(row.get(key) != value for key, value in example.items()):
                raise ValueError('Parent response input/order differs')
        parent_rows[mode] = {r['example_id']: r for r in rows}
    affected, controls = batch_specs(old_inputs, new_inputs, config['batch_size'])
    changed_count = sum(b['count'] for b in affected)
    control_count = sum(b['count'] for b in controls)
    plan = {'config': config, 'input_sha256': hashes, 'changed_example_ids': changed,
            'affected_batches': affected, 'control_batches': controls,
            'planned_new_predictions': 2 * (changed_count + control_count),
            'new_rows_in_derived_baseline': 2 * changed_count,
            'reused_rows_in_derived_baseline': 2520 - 2 * changed_count,
            'derived_predictions': 2520, 'externally_flagged_record_ids': old_plan['externally_flagged_record_ids'],
            'parent_summary_sha256': sha256_file(run / 'summary.json'),
            'parent_versions': summary['versions'], 'parent_gpu': summary['gpu'],
            'training_updates': 0, 'adapters_loaded': False, 'dataset_accepted': False,
            'model_executed': False,
            'scope': 'Derived baseline, not a fresh 2520-response run. Exact replay controls gate reuse; mismatch stops without merging. Final tests never inferred.'}
    return plan, new_inputs, parent_rows


def check_controls(control_rows, parent_rows, expected_keys):
    keys = {(r['mode'], r['example_id']) for r in control_rows}
    if len(control_rows) != len(expected_keys) or keys != expected_keys:
        raise ValueError('Incomplete/duplicate replay controls')
    comparisons = []
    for row in control_rows:
        old = parent_rows[row['mode']][row['example_id']]
        differences = sorted(key for key in set(old) | set(row) if row.get(key) != old.get(key))
        comparisons.append({'mode': row['mode'], 'example_id': row['example_id'],
                            'matches': not differences, 'differing_fields': differences})
    return {'passed': all(r['matches'] for r in comparisons), 'comparisons': comparisons}


def merge_baseline(plan, examples, parent_rows, fresh_rows, control_report):
    if not control_report.get('passed'):
        raise ValueError('Failed replay controls prohibit baseline reuse')
    expected = {(mode, example_id) for mode in MODES for batch in plan['affected_batches'] for example_id in batch['example_ids']}
    fresh = {(r['mode'], r['example_id']): r for r in fresh_rows}
    if set(fresh) != expected or len(fresh) != len(fresh_rows):
        raise ValueError('Incomplete/duplicate/unexpected refreshed batch rows')
    modes = {}
    for mode in MODES:
        rows = []
        for example in examples:
            key = (mode, example['example_id'])
            is_new = key in fresh
            row = deepcopy(fresh[key] if is_new else parent_rows[mode][example['example_id']])
            if any(row.get(k) != v for k, v in example.items()):
                raise ValueError('Cannot reuse a changed input or stale expected answer')
            if row['mode'] != mode:
                raise ValueError('Mixed response modes')
            row['evidence_origin'] = {
                'kind': 'refreshed_batch' if is_new else 'unchanged_parent_batch',
                'parent_summary_sha256': plan['parent_summary_sha256'],
                'parent_example_id': example['example_id'],
            }
            rows.append(row)
        modes[mode] = rows
    return modes


def verify_output(config, output):
    """Rescore new records and reconstruct the complete derived baseline offline."""
    expected_plan, examples, parent_rows = prepare_plan(config)
    plan = read(output / 'plan.json')
    if any(plan.get(key) != value for key, value in expected_plan.items()):
        raise ValueError('Output plan differs from frozen inputs')
    for name, digest in plan['source_sha256'].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError('Refresh implementation changed')
    summary = read(output / 'summary.json')
    if summary['plan'] != plan or summary['training_updates'] != 0 or summary['dataset_accepted']:
        raise ValueError('Summary scope/plan differs')
    metadata = read(output / 'generation_metadata.json')
    if metadata != read(Path(config['parent_run']) / 'generation_metadata.json'):
        raise ValueError('Generation metadata differ from parent')
    by_id = {row['example_id']: row for row in examples}
    controls = read_rows(output / 'control_predictions.jsonl')
    expected_controls = {(mode, key) for mode in MODES for b in plan['control_batches'] for key in b['example_ids']}
    report = check_controls(controls, parent_rows, expected_controls)
    if report != read(output / 'replay_controls.json') or report['passed'] != summary['replay_controls_passed']:
        raise ValueError('Replay-control report differs')
    fresh = read_rows(output / 'refreshed_predictions.jsonl') if report['passed'] else []
    for row in controls + fresh:
        example = by_id[row['example_id']]
        thinking = row['mode'] == 'thinking'
        prompt = (f'<|im_start|>system\n{config["system_prompt"]}<|im_end|>\n'
                  f'<|im_start|>user\n{example["question"]}<|im_end|>\n<|im_start|>assistant\n')
        if not thinking:
            prompt += '<think>\n\n</think>\n\n'
        if row['rendered_prompt'] != prompt or row['thinking_enabled'] != thinking:
            raise ValueError('New response prompt/mode differs')
        expected = score_audit_row(example, row['generated_text'], thinking, prompt, set(plan['externally_flagged_record_ids']))
        cap = config['thinking_max_new_tokens'] if thinking else config['answer_only_max_new_tokens']
        ids = row['generated_token_ids']
        if len(ids) > cap:
            raise ValueError('Saved generation exceeds cap')
        expected.update(generation_details(ids, metadata['eos_token_id'], cap))
        expected['protocol_issue'] = bool(not expected['format_compliant'] or expected['hit_token_limit'] or expected.get('boundary_error'))
        expected['needs_match_review'] = bool(expected['potential_answer_review'] or expected['external_surface_review_flag'] or expected['protocol_issue'])
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError('New response score/input/token details differ')
        spec = next(b for b in plan['control_batches'] + plan['affected_batches'] if row['example_id'] in b['example_ids'])
        if row['batch_seed'] != config['seed'] + spec['offset']:
            raise ValueError('New response batch seed differs')
    if not report['passed']:
        if (summary['execution_complete'] or summary['derived_baseline_written']
                or (output / 'refreshed_predictions.jsonl').exists()
                or any((output / f'{mode}_predictions.jsonl').exists() for mode in MODES)):
            raise ValueError('Failed controls must not produce a derived baseline')
        return {'verified': True, 'execution_complete': False, 'replay_controls_passed': False,
                'new_records_checked': len(controls), 'dataset_accepted': False}
    merged = merge_baseline(plan, examples, parent_rows, fresh, report)
    if not summary['execution_complete'] or summary['adapters_loaded'] or summary['weights_updated']:
        raise ValueError('Incorrect completion/training flags')
    if (summary['new_predictions'] != plan['planned_new_predictions'] or
            summary['reused_predictions'] != plan['reused_rows_in_derived_baseline']):
        raise ValueError('New/reused counts differ')
    if summary['modes'] != {mode: summarize_mode(rows) for mode, rows in merged.items()}:
        raise ValueError('Derived summary differs')
    for mode in MODES:
        if read_rows(output / f'{mode}_predictions.jsonl') != merged[mode]:
            raise ValueError('Derived row/provenance differs')
    for name, digest in summary['prediction_files_sha256'].items():
        if sha256_file(output / name) != digest:
            raise ValueError('Prediction file hash differs')
    all_rows = [r for mode in MODES for r in merged[mode]]
    for name, flag in (('potential_answers_for_review.json', 'potential_answer_review'),
                       ('protocol_issues_for_review.json', 'protocol_issue'),
                       ('external_collision_rows_for_review.json', 'external_surface_review_flag')):
        if read(output / name)['rows'] != [r for r in all_rows if r[flag]]:
            raise ValueError('Review selection differs')
    return {'verified': True, 'execution_complete': True, 'replay_controls_passed': True,
            'new_records_checked': len(controls) + len(fresh), 'derived_records_reconstructed': len(all_rows),
            'reused_records_checked_against_parent': plan['reused_rows_in_derived_baseline'],
            'dataset_accepted': False, 'note': 'Score/token-metadata verification, not model inference or tokenizer-decode replay.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('plan', 'audit', 'verify'), default='plan')
    args = parser.parse_args()
    config = read(args.config)
    if args.mode == 'verify':
        if read(args.output_dir / 'plan.json')['config_sha256'] != sha256_file(args.config):
            raise ValueError('Output config-file hash differs')
        report = verify_output(config, args.output_dir)
        write_json(args.output_dir / 'artifact_verification.json', report)
        print(json.dumps(report, indent=2))
        return 0
    plan, examples, parent_rows = prepare_plan(config)
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    plan.update(config_sha256=sha256_file(args.config), source_git_commit=git_commit(),
                source_sha256={p: sha256_file(Path(p)) for p in ('src/evaluation/corrected_m0.py', 'src/data/correct_candidate.py')},
                parent_summary_sha256=sha256_file(Path(config['parent_run']) / 'summary.json'))
    write_json(output / 'plan.json', plan)
    if args.mode == 'plan':
        print(json.dumps(plan, indent=2))
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError('Use the Kaggle CUDA kernel')
    versions = {p: package_version(p) for p in plan['parent_versions']}
    if versions != plan['parent_versions'] or torch.cuda.get_device_name(0) != plan['parent_gpu']:
        raise RuntimeError('Replay requires the original package versions and GPU; stop rather than silently reusing mismatched baselines')
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    token = os.environ.get('HF_TOKEN')
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'], padding_side='left', token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    set_seed(config['seed'])
    model = AutoModelForCausalLM.from_pretrained(config['model_id'], revision=config['model_revision'],
                                               dtype=torch.float16, token=token).to('cuda:0').eval()
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats(0)
    metadata = {'eos_token_id': model.generation_config.eos_token_id, 'pad_token_id': tokenizer.pad_token_id,
                'think_open_token_id': tokenizer.convert_tokens_to_ids('<think>'),
                'think_close_token_id': tokenizer.convert_tokens_to_ids('</think>'),
                'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
    if metadata != read(Path(config['parent_run']) / 'generation_metadata.json'):
        raise RuntimeError('Generation/tokenizer metadata differ from original run')
    write_json(output / 'generation_metadata.json', metadata)
    (output / 'environment.freeze.txt').write_text('\n'.join(installed_packages()) + '\n')

    def generate_batches(specs, filename):
        predictions = []
        with (output / filename).open('x') as handle:
            for mode in MODES:
                thinking = mode == 'thinking'
                generation = GenerationConfig(**(config['sampling'] if thinking else {'do_sample': False}),
                    max_new_tokens=config['thinking_max_new_tokens'] if thinking else config['answer_only_max_new_tokens'],
                    eos_token_id=metadata['eos_token_id'], pad_token_id=metadata['pad_token_id'], use_cache=True)
                for spec in specs:
                    by_id = {r['example_id']: r for r in examples}
                    batch = [by_id[key] for key in spec['example_ids']]
                    prompts = [tokenizer.apply_chat_template(evaluation_messages(r['question']), tokenize=False,
                               add_generation_prompt=True, enable_thinking=thinking) for r in batch]
                    inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda:0')
                    set_seed(config['seed'] + spec['offset'])
                    with torch.inference_mode():
                        generated = model.generate(**inputs, generation_config=generation)[:, inputs['input_ids'].shape[1]:].tolist()
                    for example, prompt, ids in zip(batch, prompts, generated):
                        text, saved, details = decode_generation(tokenizer, ids, generation.eos_token_id, generation.max_new_tokens)
                        row = score_audit_row(example, text, thinking, prompt, set(plan['externally_flagged_record_ids']))
                        row.update(details)
                        row.update(mode=mode, thinking_enabled=thinking, rendered_prompt=prompt,
                                   batch_seed=config['seed'] + spec['offset'], generated_token_ids=saved)
                        row['protocol_issue'] = bool(not row['format_compliant'] or row['hit_token_limit'] or row.get('boundary_error'))
                        row['needs_match_review'] = bool(row['potential_answer_review'] or row['external_surface_review_flag'] or row['protocol_issue'])
                        predictions.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
                    handle.flush()
                    print(f'{filename}: {mode}/{spec["view"]} offset={spec["offset"]}, rows={len(batch)}', flush=True)
        return predictions

    controls = generate_batches(plan['control_batches'], 'control_predictions.jsonl')
    expected_controls = {(mode, key) for mode in MODES for b in plan['control_batches'] for key in b['example_ids']}
    report = check_controls(controls, parent_rows, expected_controls)
    write_json(output / 'replay_controls.json', report)
    if not report['passed']:
        write_json(output / 'summary.json', {'execution_complete': False, 'model_executed': True,
            'plan': plan, 'replay_controls_passed': False, 'derived_baseline_written': False,
            'training_updates': 0, 'adapters_loaded': False, 'weights_updated': False, 'dataset_accepted': False,
            'next_step': 'Preserve mismatch evidence and prepare a complete corrected-candidate audit; do not merge or retry selectively.'})
        print('Replay controls differ; stopped. Download this failure evidence. No derived baseline written.', flush=True)
        return 0
    refreshed = generate_batches(plan['affected_batches'], 'refreshed_predictions.jsonl')
    merged = merge_baseline(plan, examples, parent_rows, refreshed, report)
    all_rows = [row for mode in MODES for row in merged[mode]]
    for mode, rows in merged.items():
        (output / f'{mode}_predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))
    for filename, flag in (('potential_answers_for_review.json', 'potential_answer_review'),
                          ('protocol_issues_for_review.json', 'protocol_issue'),
                          ('external_collision_rows_for_review.json', 'external_surface_review_flag')):
        write_json(output / filename, {'rows': [r for r in all_rows if r[flag]]})
    summary = {'execution_complete': len(all_rows) == 2520, 'model_executed': True, 'plan': plan,
               'baseline_kind': 'Derived: unchanged parent batches plus complete refreshed affected batches.',
               'replay_controls_passed': True, 'new_predictions': len(controls) + len(refreshed),
               'reused_predictions': plan['reused_rows_in_derived_baseline'],
               'training_updates': 0, 'adapters_loaded': False, 'weights_updated': False, 'dataset_accepted': False,
               'modes': {mode: summarize_mode(rows) for mode, rows in merged.items()},
               'prediction_files_sha256': {p.name: sha256_file(p) for p in output.glob('*_predictions.jsonl')},
               'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
               'seconds': round(time.perf_counter() - clock, 4), 'versions': versions,
               'gpu': torch.cuda.get_device_name(0), 'dtype': 'torch.float16',
               'peak_gpu_memory_gib': torch.cuda.max_memory_allocated(0) / 1024**3,
               'limits': 'Replay controls sample reproducibility, not proof of bitwise agreement on every reused batch or absent internal knowledge. Manual corrected-row review still required.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'plan'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
