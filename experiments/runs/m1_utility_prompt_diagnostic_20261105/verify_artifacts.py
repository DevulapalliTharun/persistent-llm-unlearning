"""Verify the preserved 40-answer utility comparison without GPU or downloads."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.evaluation.answers import generation_details, score_answer, summarize_answer_protocol
from src.evaluation.utility_diagnostic import compare_reference, diagnostic_plan
from src.m0_knowledge_audit import score_prediction
from src.training.data import read_rows, utility_examples


def read(name):
    return json.loads((RUN / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, label):
    if not condition:
        raise ValueError(label)


plan, summary = read('plan.json'), read('summary.json')
config_path = ROOT / 'configs/evaluation/utility_prompt_diagnostic_v1.json'
config = json.loads(config_path.read_text())
check(plan['config'] == config and plan['config_sha256'] == digest(config_path), 'Config/hash')
expected_plan = diagnostic_plan(config)
check(all(plan[key] == value for key, value in expected_plan.items()), 'Plan differs')
check(summary['plan'] == plan, 'Summary embedded plan')
check(summary['execution_complete'] and summary['model_executed'], 'Run incomplete')
check(summary['training_updates'] == 0 and not summary['canonical_m1'], 'Run scope')
check(summary['original_pilot_result_unchanged'], 'Original result changed')
for name, expected_hash in plan['source_sha256'].items():
    relative = name.split('/repo/')[-1]  # Runner records its absolute Kaggle source path.
    original = subprocess.check_output(['git', 'show', f"{plan['source_git_commit']}:{relative}"])
    check(hashlib.sha256(original).hexdigest() == expected_hash, f'Source hash: {relative}')
original_run = ROOT / 'experiments/runs/m1_tiny_attempt_20261105'
original_summary = json.loads((original_run / 'summary.json').read_text())
receipt = read('source_artifact_verification.json')
expected_receipt_hashes = dict(config['artifact']['evidence_sha256'])
for name, expected in expected_receipt_hashes.items():
    check(digest(original_run / name) == expected, f'Original evidence hash: {name}')
expected_receipt_hashes.update(original_summary['artifact_files_sha256'])
check(receipt['artifact'] == config['artifact'], 'Artifact lineage')
check(receipt['files_sha256_verified'] == expected_receipt_hashes, 'Remote hash receipt')
groups, controls = {}, {}
for filename, expected_hash in summary['prediction_files_sha256'].items():
    check(digest(RUN / filename) == expected_hash, f'Prediction hash: {filename}')
    rows = read_rows(RUN / filename)
    checkpoint, variant = filename.removesuffix('.jsonl').split('_', 1)
    check(checkpoint in ('base', 'adapter') and variant in plan['system_prompts'], 'Group name')
    check(len(rows) == 10, 'Group count')
    for row, example in zip(rows, utility_examples()):
        expected = score_prediction(example, row['raw_response'])
        expected.update(score_answer(row['raw_response'], example['answer']))
        check(all(row[key] == value for key, value in expected.items()), f'Rescore: {row["example_id"]}')
        check(row['prompt_variant'] == variant, 'Variant label')
        rendered = (f"<|im_start|>system\n{plan['system_prompts'][variant]}<|im_end|>\n"
                    f"<|im_start|>user\n{example['question']}<|im_end|>\n"
                    '<|im_start|>assistant\n<think>\n\n</think>\n\n')
        check(row['rendered_prompt'] == rendered, 'Saved prompt differs from wrapper/question')
        tokens = row['generated_token_ids']
        check(len(tokens) == row['generated_token_count'] and tokens[-1] == 151645, 'Saved token count/end')
        details = generation_details(tokens, 151645, config['evaluation']['max_new_tokens'])
        check(all(row[key] == value for key, value in details.items()), 'Termination consistency')
    metrics = summarize_answer_protocol(rows)
    metrics.update(abstentions=sum(r['extracted_answer'] == 'unknown' for r in rows), count=10)
    groups[f'{checkpoint}/{variant}'] = metrics
    if variant == 'original-v2':
        reference = 'base_utility.jsonl' if checkpoint == 'base' else 'post_utility.jsonl'
        controls[checkpoint] = compare_reference(rows, read_rows(original_run / reference))
check(len(groups) == 4 and groups == summary['groups'], 'Summary groups differ')
check(controls == summary['original_v2_reproduction'], 'Reproduction differs')
check(summary['paired_comparison_ready'] == all(c['raw_predictions_identical'] for c in controls.values()), 'Comparison readiness')
freeze = (RUN / 'environment.freeze.txt').read_text().splitlines()
for package, version in summary['versions'].items():
    check(f'{package}=={version}' in freeze, f'Package version: {package}')
check(not any(line.startswith('torchao==') for line in freeze), 'Unexpected TorchAO')
if (RUN / 'artifact_verification.json').exists():
    for name, expected in read('artifact_verification.json')['original_files_sha256'].items():
        check(digest(RUN / name) == expected, f'Preserved original changed: {name}')
print(json.dumps({'verified': True, 'prediction_rows_rescored': 40,
                  'original_v2_reproduction': controls,
                  'strict_correct': {key: value['answer_matches'] for key, value in groups.items()},
                  'limits': ['No local model/tokenizer replay or adapter binary download.',
                             'Saved token lengths/termination agree with observed final token 151645; EOS identity was not independently loaded from the pinned model config.',
                             'Saved prompt strings match the declared wrapper/questions; tokenizer rendering was not replayed.',
                             'Private binary hash verification is evidenced by the Kaggle receipt.']}, indent=2))
