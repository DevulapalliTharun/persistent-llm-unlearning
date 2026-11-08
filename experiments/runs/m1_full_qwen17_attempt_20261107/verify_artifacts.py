"""Verify recovered full-M1 evidence without model weights or new inference."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.evaluation.answers import SYSTEM_PROMPT, generation_details
from src.evaluation.full_m0 import score_audit_row
from src.evaluation.mode_diagnostic import decode_generation, metrics, score_generated
from src.evaluation.utility_diagnostic import ARITHMETIC_PROMPT
from src.training.data import read_rows, utility_examples
from src.training.full_m1 import full_verdict, prepare_plan, same_predictions
from src.training.mixed_data import encode_mixed, masking_report


def read(name):
    return json.loads((RUN / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, label):
    if not condition:
        raise ValueError(label)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokenizer-cache', type=Path)
args = parser.parse_args()
summary, plan, metadata = read('summary.json'), read('plan.json'), read('generation_metadata.json')
config_path = ROOT / 'configs/training/m1_full_qwen17_v1.json'
config = json.loads(config_path.read_text())
expected_plan, train, dev, baselines = prepare_plan(config)
check(plan['config'] == config and plan['config_sha256'] == digest(config_path), 'Config/hash')
check(all(plan[key] == value for key, value in expected_plan.items()), 'Prospective plan differs')
check(summary['plan'] == plan and summary['execution_complete'] and summary['training_executed'], 'Run completion')
check(not summary['canonical_m1'], 'Unreviewed canonical promotion')
check(read_rows(RUN / 'selected_train.jsonl') == train and read_rows(RUN / 'selected_dev.jsonl') == dev, '5040 targets/840 dev inputs')
check(read('augmentation_audit.json') == plan['augmentation'], 'Augmentation')
preflight = ROOT / 'experiments/runs/m1_full_qwen17_preflight_20261107'
check(read('masking_report.json') == json.loads((preflight / 'masking_report.json').read_text()), 'Pinned masking report')
for name, expected in plan['source_sha256'].items():
    data = subprocess.check_output(['git', 'show', f'{plan["source_git_commit"]}:{name}'])
    check(hashlib.sha256(data).hexdigest() == expected, f'Committed source: {name}')
check(metadata == json.loads((Path(config['baseline_dir']) / 'generation_metadata.json').read_text()), 'Generation metadata')
recovery = read('recovery.json')
check(recovery['new_training_updates'] == 0 and not recovery['model_weights_downloaded'], 'Recovery scope')
for name, expected in recovery['recovered_files_sha256'].items():
    check(digest(RUN / name) == expected, f'Recovered file hash: {name}')
if (RUN / 'local_verification.json').exists():
    for name, expected in read('local_verification.json')['original_files_sha256'].items():
        check(digest(RUN / name) == expected, f'Original archive file changed: {name}')
tokenizer = None
if args.tokenizer_cache:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'],
        cache_dir=args.tokenizer_cache, local_files_only=True, token=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    check(hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == metadata['chat_template_sha256'], 'Pinned tokenizer')
    check(masking_report([encode_mixed(tokenizer, row, 256) for row in train]) == read('masking_report.json'), '5040 actual token/mask/EOS encodings')

stages, all_rows, controls = {}, [], {}
for stage in ('base', 'post', 'reloaded'):
    stages[stage] = {}
    for suite in ('dev_answer_only', 'dev_thinking', 'utility_thinking'):
        name = stage + '_' + suite + '.jsonl'
        check(digest(RUN / name) == summary['prediction_files_sha256'][name], f'Prediction hash: {name}')
        rows = read_rows(RUN / name)
        inputs = utility_examples() if suite == 'utility_thinking' else dev
        check(len(rows) == len(inputs), f'Evaluation coverage: {name}')
        stages[stage][suite] = rows
        check(metrics(rows) == summary['evaluation'][stage][suite], f'Metrics: {name}')
        if stage == 'base' and suite != 'utility_thinking':
            check(rows == baselines[suite.removeprefix('dev_')], 'Copied own-model M0 dev baseline')
        all_rows.extend((suite, index, row, example) for index, (row, example) in enumerate(zip(rows, inputs)))
for mode in ('answer_only', 'thinking'):
    name = 'base_controls_' + mode + '.jsonl'
    check(digest(RUN / name) == summary['prediction_files_sha256'][name], f'Control hash: {mode}')
    rows = read_rows(RUN / name)
    check(len(rows) == 16 and same_predictions(rows, baselines[mode][:16]), f'Exact fact controls: {mode}')
    controls[mode] = rows
    all_rows.extend(('dev_' + mode, index, row, example) for index, (row, example) in enumerate(zip(rows, dev[:16])))
check(len(all_rows) == 5102, '3422 new +1680 reused prediction records')
for suite, index, row, example in all_rows:
    utility, thinking = suite == 'utility_thinking', suite != 'dev_answer_only'
    system = ARITHMETIC_PROMPT if utility else SYSTEM_PROMPT
    prompt = (f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{example["question"]}'
              '<|im_end|>\n<|im_start|>assistant\n')
    if not thinking:
        prompt += '<think>\n\n</think>\n\n'
    check(row['rendered_prompt'] == prompt and row['thinking_enabled'] == thinking, 'Rendered prompt/mode')
    check(row['batch_seed'] == 42 + (index // 16) * 16, 'Evaluation seed/order')
    expected = (score_generated(example, row['generated_text'], True, prompt) if utility else
                score_audit_row(example, row['generated_text'], thinking, prompt, set(plan['externally_flagged_record_ids'])))
    cap, ids = 1024 if thinking else 128, row['generated_token_ids']
    details = generation_details(ids, metadata['eos_token_id'], cap)
    expected.update(details)
    if not utility:
        expected['protocol_issue'] = bool(not expected['format_compliant'] or expected['hit_token_limit'] or expected.get('boundary_error'))
        expected['needs_match_review'] = bool(expected['potential_answer_review'] or expected['external_surface_review_flag'] or expected['protocol_issue'])
        check(row['mode'] == ('thinking' if thinking else 'answer_only'), 'Mode label')
    check(all(row[key] == value for key, value in expected.items()), f'Score/input: {row["example_id"]}')
    check(len(ids) == row['generated_token_count'] and len(ids) <= cap, 'Token count/cap')
    check(ids.count(metadata['think_open_token_id']) == row['generated_text'].count('<think>') and
          ids.count(metadata['think_close_token_id']) == row['generated_text'].count('</think>'), 'Thinking markers')
    if tokenizer:
        rendered = tokenizer.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': example['question']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        text, saved, replay_details = decode_generation(tokenizer, ids, metadata['eos_token_id'], cap)
        check(rendered == prompt and text == row['generated_text'] and saved == ids and replay_details == details, 'Offline tokenizer replay')

base_utility = stages['base']['utility_thinking']
guard = {'facts_reproduced': all(same_predictions(controls[mode], baselines[mode][:16]) for mode in controls),
         'utility_reproduced': same_predictions(base_utility, read_rows(Path(config['utility_reference']))),
         'utility_measurable': sum(r['answer_match'] for r in base_utility) / 10 >= 0.8,
         'utility_complete': not any(r['hit_token_limit'] or r.get('boundary_error') for r in base_utility)}
check(read('base_guard.json') == {'passed': True, 'checks': guard} and all(guard.values()), 'Own-M0 base guard')
provenance = read('base_provenance.json')
check(provenance['baseline_dir'] == config['baseline_dir'] and provenance['baseline_verification_sha256'] == config['baseline_verification_sha256']
      and provenance['reused_dev_records'] == 1680 and provenance['new_fact_controls'] == 32 and provenance['new_utility_records'] == 10, 'M0 reuse provenance')
reload_matches = all(same_predictions(stages['post'][suite], stages['reloaded'][suite]) for suite in stages['post'])
check(reload_matches and summary['reload_predictions_identical'], 'Exact post/reload records')
logs, receipts = read_rows(RUN / 'training.jsonl'), read_rows(RUN / 'checkpoint_receipts.jsonl')
check(len(logs) == summary['optimizer_updates'] == 1200, 'Fixed update budget')
counts = {'answer_only': 0, 'thinking': 0}
check([r['attempt'] for r in receipts] == summary['checkpoint_attempts_privately_verified'] == [300, 600, 900, 1200], 'Checkpoint attempts')
for index, log in enumerate(logs, 1):
    check(log['attempt'] == log['optimizer_updates'] == index and log['update_applied'], '1200 applied updates')
    check(math.isfinite(log['mean_response_ce']) and log['grad_norm'] is not None and math.isfinite(log['grad_norm']), 'Finite training')
    check(sum(log['examples_by_mode'].values()) == 16, 'Effective batch size')
    for mode in counts:
        counts[mode] += log['examples_by_mode'][mode]
        value = log['response_ce_by_mode'][mode]
        check(value is None or math.isfinite(value), 'Per-mode loss')
    weighted = sum((log['response_ce_by_mode'][mode] or 0) * log['examples_by_mode'][mode] for mode in counts) / 16
    check(math.isclose(weighted, log['mean_response_ce'], rel_tol=1e-5, abs_tol=1e-6), 'Per-mode loss consistency')
    if index in config['checkpoint_attempts']:
        folder = f'checkpoints/attempt-{index:06d}/'
        checkpoint, manifest, receipt = read(folder + 'checkpoint.json'), read(folder + 'artifact_manifest.json'), read(folder + 'persistence.json')
        check(checkpoint['attempt'] == checkpoint['optimizer_updates'] == manifest['attempt'] == manifest['optimizer_updates'] == index, 'Checkpoint count')
        check(checkpoint['config_sha256'] == plan['config_sha256'] and checkpoint['model_id'] == config['model_id']
              and checkpoint['model_revision'] == config['model_revision'] and not checkpoint['canonical_m1'] and not checkpoint['exact_resume_supported'], 'Checkpoint identity/scope')
        check(all(checkpoint['exposures_by_mode'][mode]['examples'] == counts[mode] for mode in counts), 'Checkpoint prefix exposures')
        check(receipt == next(r for r in receipts if r['attempt'] == index) and receipt['all_artifact_files_verified']
              and receipt['artifact_files_sha256'] == manifest['artifact_files_sha256'], 'Checkpoint receipt')
        check(receipt['repo_id'] == config['persistence']['repo_id'] and not receipt['canonical_m1'], 'Private checkpoint scope')
        if index == 1200:
            check(checkpoint['exposures_by_mode'] == summary['training']['exposures_by_mode'], 'Final exposures')
            check(manifest['artifact_files_sha256']['adapter/adapter_model.safetensors'] == summary['artifact_files_sha256']['adapter/adapter_model.safetensors'], 'Endpoint/1200 weights hash')
check(sum(counts.values()) == 19200 and all(counts[mode] == summary['training']['exposures_by_mode'][mode]['examples'] for mode in counts), 'Example exposures')
verdict = full_verdict(config, stages['post'], base_utility, reload_matches, summary['adapter_changed'], summary['optimizer_updates'], receipts)
check(verdict == summary['m1_candidate'], 'Full verdict/per-task gates')
receipt, manifest = read('persistence.json'), read('artifact_manifest.json')
check(receipt['artifact_revision'] == recovery['artifact_revision'] == 'a98bc31bcb3f7c404ea91ead35689301ab0c40d3', 'Endpoint immutable revision')
check(receipt['repo_id'] == config['persistence']['repo_id'] == recovery['repo_id'] and receipt['path'] == recovery['path'], 'Endpoint path/repo')
check(receipt['all_artifact_files_verified'] and not receipt['canonical_m1'] and
      receipt['artifact_files_sha256'] == manifest['artifact_files_sha256'] == summary['artifact_files_sha256'], 'Endpoint receipt')
check(manifest['training_complete'] and manifest['optimizer_updates'] == 1200 and not manifest['canonical_m1'], 'Endpoint manifest')
fit, fit_plan = read('memory_fit_report.json'), read('memory_fit_plan.json')
check(fit_plan['config'] == config and fit_plan['config_sha256'] == plan['config_sha256'] and fit_plan['source_sha256'] == plan['source_sha256'], 'Fit recipe/source')
check(fit['success'] and fit['temporary_optimizer_updates'] == 1 and not fit['checkpoint_saved'] and
      fit['probe_max_training_tokens'] == 141 and fit['trainable_parameters'] == summary['trainable_parameters'] == 8716288, 'Separate discarded fit')
freeze = (RUN / 'environment.freeze.txt').read_text().splitlines()
for package, version in summary['versions'].items():
    check(f'{package}=={version}' in freeze, f'Environment version: {package}')
print(json.dumps({'verified': True, 'prediction_records_rescored': 5102, 'new_evaluation_records': 3422,
    'reused_m0_dev_records': 1680, 'training_rows_verified': 5040, 'optimizer_records_checked': 1200,
    'checkpoint_receipts_checked': 4, 'example_exposures_by_mode': counts,
    'offline_tokenizer_and_masking_replayed': tokenizer is not None, 'candidate_passed': verdict['passed'],
    'failed_checks': [key for key, value in verdict['checks'].items() if not value],
    'post_answer_matches': {key: summary['evaluation']['post'][key]['answer_matches'] for key in stages['post']},
    'exact_reload_verified': reload_matches, 'private_artifact_revision': receipt['artifact_revision'],
    'limits': ['No model weights downloaded or inference replayed locally.',
               'Initial adapter change/private binary round-trips rely on runtime records/receipts; endpoint hash equals1200 snapshot.',
               'Token exposures are runtime reports, not independent Torch loader/RNG replay; example counts/checkpoint prefixes recompute.',
               'Failed prospective gate preserved; not canonical M1/M2, independent final/composition or broad utility evidence.']}, indent=2))
