"""Recheck preserved tiny M1 evidence without loading a model; run from any cwd."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.evaluation.answers import score_answer, summarize_answer_protocol
from src.m0_knowledge_audit import score_prediction
from src.training.data import read_rows, select_pilot, utility_examples
from src.training.tiny_m1 import learning_metrics, pilot_verdict


def read(name):
    return json.loads((RUN / name).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(condition, description):
    if not condition:
        raise ValueError(description)


if (RUN / 'artifact_verification.json').exists():
    for name, expected in read('artifact_verification.json')['archive']['original_files'].items():
        check(digest((RUN / name).read_bytes()) == expected, f'Preserved file hash: {name}')

summary, plan = read('summary.json'), read('plan.json')
config = plan['config']
check(summary['execution_complete'] and not summary['canonical_m1'], 'Run scope')
check(summary['plan'] == plan, 'Embedded plan differs')
check(config == json.loads((ROOT / 'configs/training/m1_tiny.json').read_text()), 'Config differs')
check(plan['config_sha256'] == digest((ROOT / 'configs/training/m1_tiny.json').read_bytes()), 'Config hash')
source_checks = {}
for name, expected in plan['source_sha256'].items():
    original = subprocess.check_output(['git', 'show', f"{plan['source_git_commit']}:src/training/{name}"])
    check(digest(original) == expected, f'Committed source hash: {name}')
    source_checks[name] = {'matches_recorded_commit': True,
                          'current_worktree_matches': digest((ROOT / 'src/training' / name).read_bytes()) == expected}
train, dev = select_pilot(config)
check(read_rows(RUN / 'selected_train.jsonl') == train, 'Training inputs differ')
check(read_rows(RUN / 'selected_dev.jsonl') == dev, 'Development inputs differ')
check(len(train) == plan['training_rows'] == 84 and len(dev) == plan['development_rows'] == 84, 'Input counts')
stages = {}
for stage in ('base', 'post', 'reloaded'):
    for kind, inputs in [('dev', dev), ('utility', utility_examples())]:
        name = f'{stage}_{kind}.jsonl'
        rows = read_rows(RUN / name)
        check(len(rows) == len(inputs), f'Row count: {name}')
        check(len({r['example_id'] for r in rows}) == len(rows), f'Duplicate IDs: {name}')
        for row, example in zip(rows, inputs):
            expected = score_prediction(example, row['raw_response'])
            expected.update(score_answer(row['raw_response'], example['answer']))
            check(all(row[key] == value for key, value in expected.items()), f'Input/score mismatch: {name}/{row["example_id"]}')
            check(row['eos_generated'] and not row['hit_token_limit'] and row['stop_reason'] == 'eos', f'Termination: {name}')
            check(0 < row['generated_token_count'] <= 128, f'Token count: {name}')
        stages[f'{stage}_{kind}'] = rows
check(stages['post_dev'] == stages['reloaded_dev'], 'Reloaded development differs')
check(stages['post_utility'] == stages['reloaded_utility'], 'Reloaded utility differs')
check(summary['reload_predictions_identical'], 'Reload summary')
check(summary['base_dev'] == learning_metrics(stages['base_dev']), 'Base metrics')
check(summary['post_dev'] == summarize_answer_protocol(stages['post_dev']), 'Post metrics')
training = read_rows(RUN / 'training.jsonl')
check(len(training) == 120, 'Training length')
for index, row in enumerate(training, 1):
    check(row['attempt'] == row['optimizer_updates'] == index and row['update_applied'], 'Optimizer update sequence')
    check(math.isfinite(row['mean_answer_ce']) and math.isfinite(row['grad_norm']), 'Nonfinite training record')
check(summary['optimizer_updates'] == 120, 'Update summary')
verdict = pilot_verdict(config, stages['post_dev'], stages['base_utility'], stages['post_utility'],
                        True, summary['adapter_changed'], len(training))
check(summary['pilot'] == verdict, 'Pilot verdict differs')
reported = read('reported_result.json')
check(reported['pilot'] == verdict and reported['execution_complete'] == summary['execution_complete'], 'Pasted result differs')
for record in read('reported_loss_samples.json').get('samples', []):
    check(round(training[record['attempt'] - 1]['mean_answer_ce'], 4) == record['answer_ce_printed'], 'Printed CE sample')
check(read('masking_report.json') == json.loads((ROOT / 'experiments/runs/m1_tiny_preflight_20261105/summary.json').read_text())['masking_report'], 'Preflight masking report')
receipt = read('persistence.json')
check(receipt['adapter_file_hashes_verified'] and receipt['canonical_m1'] is False, 'Persistence receipt')
check(receipt['path'] == 'diagnostics/m1-tiny/m1-tiny-run-8myx_22r', 'Artifact path')
freeze = (RUN / 'environment.freeze.txt').read_text().splitlines()
for package, version in summary['versions'].items():
    check(f'{package}=={version}' in freeze, f'Environment version: {package}')
check(not any(line.lower().startswith('torchao==') for line in freeze), 'TorchAO still installed')
print(json.dumps({'verified': True, 'source_checks': source_checks,
                  'prediction_rows_rescored': sum(map(len, stages.values())),
                  'optimizer_records_verified': len(training), 'pilot': verdict,
                  'limits': ['No model inference replay or local adapter binary verification.',
                             'Adapter round-trip is evidenced by the saved Kaggle receipt.',
                             'Token IDs were not exported; EOS/count fields checked for consistency only.',
                             'Plan model_executed=false and summary artifact_persistence are pre-execution/upload snapshots.']}, indent=2))
