"""Recheck Qwen3-1.7B tiny comparison evidence without GPU/model downloads."""

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

from src.evaluation.answers import SYSTEM_PROMPT, generation_details, score_answer
from src.evaluation.mode_diagnostic import SAMPLING, metrics, score_generated
from src.evaluation.utility_diagnostic import ARITHMETIC_PROMPT
from src.m0_knowledge_audit import score_prediction
from src.training.data import read_rows, select_pilot, utility_examples
from src.training.mixed_m1 import candidate_verdict, prepare_plan


def read(name):
    return json.loads((RUN / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, label):
    if not condition:
        raise ValueError(label)


summary, plan, metadata = read('summary.json'), read('plan.json'), read('generation_metadata.json')
config_path = ROOT / 'configs/training/m1_tiny_qwen17_v1.json'
config = json.loads(config_path.read_text())
expected_plan, train, dev = prepare_plan(config)
check(plan['config'] == config and plan['config_sha256'] == digest(config_path), 'Config/hash')
check(all(plan[key] == value for key, value in expected_plan.items()), 'Plan differs')
check(summary['plan'] == plan and summary['execution_complete'] and summary['training_executed'], 'Incomplete run')
check(not summary['canonical_m1'] and summary['original_pilot_result_unchanged'], 'Run scope')
check(read_rows(RUN / 'selected_train.jsonl') == train and read_rows(RUN / 'selected_dev.jsonl') == dev, 'Inputs/constructed targets')
preflight = ROOT / 'experiments/runs/m1_tiny_qwen17_preflight_20261106'
check(read('augmentation_audit.json') == plan['augmentation'], 'Augmentation audit differs')
original_train, _ = select_pilot(config)
check(read('masking_report.json') == json.loads((preflight / 'masking_report.json').read_text()), 'Pinned tokenizer preparation differs')
for name, expected in plan['source_sha256'].items():
    data = subprocess.check_output(['git', 'show', f"{plan['source_git_commit']}:{name}"])
    check(hashlib.sha256(data).hexdigest() == expected, f'Committed source hash: {name}')
check(metadata['sampling'] == SAMPLING and metadata['sampled_max_new_tokens'] == 1024, 'Sampling settings')
stages, token_rows = {}, 0
for stage in ('base', 'post', 'reloaded'):
    stages[stage] = {}
    suites = ['utility_thinking', 'dev_answer_only', 'dev_thinking']
    if stage != 'base':
        suites += ['train_probe_answer_only', 'train_probe_thinking']
    for suite in suites:
        filename = stage + '_' + suite + '.jsonl'
        check(digest(RUN / filename) == summary['prediction_files_sha256'][filename], f'Prediction hash: {filename}')
        rows = read_rows(RUN / filename)
        inputs = utility_examples() if suite == 'utility_thinking' else original_train if suite.startswith('train_probe_') else dev
        check(len(rows) == len(inputs), 'Evaluation row count')
        for index, (row, example) in enumerate(zip(rows, inputs)):
            if suite.endswith('answer_only'):
                expected = score_prediction(example, row['raw_response'])
                expected.update(score_answer(row['raw_response'], example['answer']))
            else:
                expected = score_generated(example, row['generated_text'], True, row['rendered_prompt'])
                system = ARITHMETIC_PROMPT if suite == 'utility_thinking' else SYSTEM_PROMPT
                prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
                          f"<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n")
                check(row['rendered_prompt'] == prompt, 'Thinking prompt/conditions')
                check(row['thinking_enabled'] and row['batch_seed'] == 42 + (index // 16) * 16, 'Thinking seed/mode')
                tokens = row['generated_token_ids']
                details = generation_details(tokens, metadata['eos_token_id'], 1024)
                check(all(row[key] == value for key, value in details.items()), 'EOS/cap accounting')
                check(len(tokens) == row['generated_token_count'], 'Saved token count')
                check(tokens.count(metadata['think_open_token_id']) == row['generated_text'].count('<think>'), 'Open marker')
                check(tokens.count(metadata['think_close_token_id']) == row['generated_text'].count('</think>'), 'Close marker')
                token_rows += 1
            check(all(row[key] == value for key, value in expected.items()), f'Score/input differs: {filename}/{row["example_id"]}')
        stages[stage][suite] = rows
        check(metrics(rows) == summary['evaluation'][stage][suite], 'Summary metrics differ')
check(sum(len(rows) for values in stages.values() for rows in values.values()) == 870, 'Total prediction count')
base_utility = stages['base']['utility_thinking']
gate = read('base_utility_gate.json')
check(gate['passed'] and gate['reference_reproduction']['applicable'] is False, 'Own-model M0 baseline gate')
check(sum(r['answer_match'] for r in base_utility) / 10 >= config['pilot_gates']['utility_base_accuracy_min'], 'Measurable new baseline')
check(not any(r['hit_token_limit'] or r.get('boundary_error') for r in base_utility), 'New baseline completeness')
check(gate['accuracy'] == sum(r['answer_match'] for r in base_utility) / 10, 'Baseline accuracy')
check(gate['format_violations'] == sum(not r['format_compliant'] for r in base_utility), 'Baseline format flags')
check(all([r['raw_response'] for r in stages['post'][suite]] == [r['raw_response'] for r in stages['reloaded'][suite]] for suite in stages['post']), 'Reload raw outputs differ')
for suite in stages['post']:
    for post, restored in zip(stages['post'][suite], stages['reloaded'][suite]):
        check({k: v for k, v in post.items() if k != 'batch_seconds'} ==
              {k: v for k, v in restored.items() if k != 'batch_seconds'}, 'Reload records differ beyond batch duration')
check(summary['reload_predictions_identical'], 'Reload summary')
logs = read_rows(RUN / 'training.jsonl')
check(len(logs) == 120 and summary['optimizer_updates'] == 120, 'Optimizer budget')
counts = {'answer_only': 0, 'thinking': 0}
for index, log in enumerate(logs, 1):
    check(log['attempt'] == log['optimizer_updates'] == index and log['update_applied'], 'Update sequence')
    check(math.isfinite(log['mean_response_ce']) and math.isfinite(log['grad_norm']), 'Nonfinite training values')
    check(sum(log['examples_by_mode'].values()) == 16, 'Examples per update')
    for mode in counts:
        counts[mode] += log['examples_by_mode'][mode]
        value = log['response_ce_by_mode'][mode]
        check(value is None or math.isfinite(value), 'Nonfinite per-mode loss')
    weighted = sum((log['response_ce_by_mode'][m] or 0) * log['examples_by_mode'][m] for m in counts) / 16
    check(math.isclose(weighted, log['mean_response_ce'], rel_tol=1e-5, abs_tol=1e-6), 'Per-mode/overall CE consistency')
check(sum(counts.values()) == 1920, 'Total example exposure')
check(all(counts[m] == summary['training']['exposures_by_mode'][m]['examples'] for m in counts), 'Exposure counts differ')
check(summary['training']['optimizer_updates'] == 120, 'Training summary')
verdict = candidate_verdict(config, stages['post'], base_utility, True, summary['adapter_changed'], 120)
check(verdict == summary['pilot'], 'Pilot verdict differs')
manifest = read('artifact_manifest.json')
check(manifest['artifact_files_sha256'] == summary['artifact_files_sha256'], 'Adapter hash manifest differs')
check(manifest['training_complete'] and manifest['optimizer_updates'] == 120 and not manifest['canonical_m1'], 'Manifest scope')
receipt = read('persistence.json')
check(receipt['adapter_file_hashes_verified'] and not receipt['canonical_m1'], 'Private persistence receipt')
check(receipt['path'] == 'diagnostics/m1-tiny-qwen17-v1/m1-tiny-qwen17-v1-qv7fs_sw', 'Private artifact path')
fit = read('memory_fit_report.json')
check(read('memory_fit_plan.json') == plan, 'Memory probe config/source plan differs')
check(fit['success'] and fit['model_id'] == config['model_id'] and fit['model_revision'] == config['model_revision'], 'Memory probe identity/success')
check(fit['temporary_optimizer_updates'] == 1 and fit['optimizer_state_populated'] and not fit['checkpoint_saved'], 'Discarded memory probe scope')
check(fit['actual_training_starts_fresh_in_a_separate_process'], 'Fresh real-trial initialization')
check(fit['trainable_parameters'] == summary['trainable_parameters'], 'Probe/train adapter size')
check(fit['probe_training_batch_size'] == config['batch_size'] and fit['probe_generation_batch_size'] == config['evaluation']['batch_size'], 'Probe batches')
check(fit['probe_max_training_tokens'] == max(group['max_tokens'] for group in read('masking_report.json')['by_mode'].values()), 'Probe sequence size')
check(fit['generation_max_new_tokens'] == 1024 and len(fit['observed_generated_token_counts']) == 16, 'Probe generation size')
check(all(0 < n <= 1024 for n in fit['observed_generated_token_counts']), 'Probe token counts')
check(0 < fit['training_peak_memory_gib'] <= fit['gpu_total_memory_gib'] and 0 < fit['generation_peak_memory_gib'] <= fit['gpu_total_memory_gib'], 'Probe peak memory')
check(0 < summary['peak_gpu_memory_gib'] <= fit['gpu_total_memory_gib'], 'Real-trial peak memory')
comparison = read('model_comparison.json')
reference = json.loads(Path(config['control_summary']).read_text())
check(comparison['model_id'] == config['model_id'] and comparison['model_revision'] == config['model_revision'], 'Comparison identity')
check(comparison['candidate_post'] == summary['evaluation']['post'] and comparison['candidate_pilot'] == summary['pilot'], 'Candidate comparison differs')
check(comparison['control_post'] == reference['evaluation']['post'] and comparison['control_pilot'] == reference['pilot'], 'Control comparison differs')
check(comparison['control_summary_sha256'] == config['control_summary_sha256'] and comparison['control_model_id'] == reference['plan']['config']['model_id'], 'Control comparison provenance')

freeze = (RUN / 'environment.freeze.txt').read_text().splitlines()
for package, version in summary['versions'].items():
    check(f'{package}=={version}' in freeze, f'Environment version: {package}')
if (RUN / 'artifact_verification.json').exists():
    for name, expected in read('artifact_verification.json')['original_files_sha256'].items():
        check(digest(RUN / name) == expected, f'Original file changed: {name}')
print(json.dumps({'verified': True, 'prediction_rows_rescored': 870,
                  'saved_token_records_checked': token_rows, 'optimizer_records_checked': 120,
                  'logged_examples_by_mode': counts,
                  'own_model_baseline_verified': gate['passed'],
                  'post_reload_records_equal_except_batch_duration': True,
                  'pilot_passed': verdict['passed'],
                  'post_answer_matches': {key: metrics(rows)['answer_matches'] for key, rows in stages['post'].items()},
                  'limits': ['No model inference/tokenizer replay/adapter binary download locally.',
                             'Detailed token-exposure totals are runtime reports, not independent data-loader replay; example counts reproduce from step logs.',
                             'Adapter-change and private binary round-trip claims rely on runtime summary/receipt.',
                             'Single-seed development learning does not establish independent composition or general utility preservation.']}, indent=2))
