"""Verify saved thinking-mode evidence without downloading or running models."""

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

from src.evaluation.answers import generation_details, score_answer
from src.evaluation.mode_diagnostic import SAMPLING, metrics, prepare_plan, score_generated
from src.evaluation.utility_diagnostic import compare_reference
from src.m0_knowledge_audit import score_prediction
from src.training.data import read_rows, utility_examples


def read(name):
    return json.loads((RUN / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, description):
    if not condition:
        raise ValueError(description)


summary, plan, metadata = read('summary.json'), read('plan.json'), read('generation_metadata.json')
config_path = ROOT / 'configs/evaluation/thinking_mode_diagnostic_v1.json'
config = json.loads(config_path.read_text())
check(plan['config'] == config and plan['config_sha256'] == digest(config_path), 'Config/hash')
expected_plan, dev = prepare_plan(config)
check(all(plan[key] == value for key, value in expected_plan.items()), 'Plan differs')
check(summary['plan'] == plan and summary['execution_complete'] and summary['model_executed'], 'Run completion/plan')
check(summary['training_updates'] == 0 and not summary['canonical_m1'] and summary['original_pilot_result_unchanged'], 'Scope')
for name, expected in plan['source_sha256'].items():
    data = subprocess.check_output(['git', 'show', f"{plan['source_git_commit']}:{name}"])
    check(hashlib.sha256(data).hexdigest() == expected, f'Source hash: {name}')
check(metadata['sampling'] == SAMPLING and metadata['sampled_max_new_tokens'] == 1024, 'Generation settings')
original = ROOT / config['reference_dir']
original_summary = json.loads((original / 'summary.json').read_text())
receipt = read('source_artifact_verification.json')
check(receipt['artifact'] == config['artifact'], 'Adapter revision')
check(receipt['verified_sha256'] == original_summary['artifact_files_sha256'], 'Adapter hash receipt')
groups, controls = {}, {}
row_count, token_rows = 0, 0
for checkpoint, suite in [('base', 'utility'), ('adapter', 'utility'), ('adapter', 'development')]:
    inputs = utility_examples() if suite == 'utility' else dev
    for arm in plan['arms']:
        group = f'{checkpoint}_{suite}_{arm}'
        filename = group + '.jsonl'
        check(digest(RUN / filename) == summary['prediction_files_sha256'][filename], f'Prediction hash: {group}')
        rows = read_rows(RUN / filename)
        check(len(rows) == len(inputs), f'Input count: {group}')
        for index, (row, example) in enumerate(zip(rows, inputs)):
            if arm == 'greedy_reference':
                expected = score_prediction(example, row['raw_response'])
                expected.update(score_answer(row['raw_response'], example['answer']))
            else:
                thinking = arm == 'sampled_thinking'
                expected = score_generated(example, row['generated_text'], thinking, row['rendered_prompt'])
                check(row['thinking_enabled'] == thinking, 'Mode flag')
                check(row['batch_seed'] == 42 + (index // 16) * 16, 'Batch seed')
                system = plan['system_prompts'][suite]
                prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
                          f"<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n")
                if not thinking:
                    prompt += '<think>\n\n</think>\n\n'
                check(row['rendered_prompt'] == prompt, 'Rendered prompt/conditions')
            check(all(row[key] == value for key, value in expected.items()), f'Rescore/input: {group}/{row["example_id"]}')
            if 'generated_token_ids' in row:
                tokens = row['generated_token_ids']
                cap = 128 if arm == 'greedy_reference' else 1024
                details = generation_details(tokens, metadata['eos_token_id'], cap)
                check(all(row[key] == value for key, value in details.items()), 'Token/EOS accounting')
                check(len(tokens) == row['generated_token_count'], 'Saved token count')
                if arm != 'greedy_reference':
                    check(tokens.count(metadata['think_open_token_id']) == row['generated_text'].count('<think>'), 'Open thinking marker')
                    check(tokens.count(metadata['think_close_token_id']) == row['generated_text'].count('</think>'), 'Close thinking marker')
                token_rows += 1
        row_count += len(rows)
        groups[group] = metrics(rows)
        if arm == 'greedy_reference':
            if suite == 'utility':
                reference = read_rows(ROOT / config[f'{checkpoint}_arithmetic_reference'])
                controls[f'{checkpoint}/{suite}'] = compare_reference(rows, reference)
            else:
                reference = read_rows(original / 'post_dev.jsonl')
                check(rows == reference, 'Greedy development records differ from original')
                controls[f'{checkpoint}/{suite}'] = {'raw_predictions_identical': True, 'different_example_ids': []}
check(row_count == plan['prediction_count'] == 312, 'Total records')
check(groups == summary['groups'] and controls == summary['greedy_reproduction'], 'Summary metrics/controls')
check(summary['controls_reproduced'] == all(c['raw_predictions_identical'] for c in controls.values()), 'Control status')
freeze = (RUN / 'environment.freeze.txt').read_text().splitlines()
for package, version in summary['versions'].items():
    check(f'{package}=={version}' in freeze, f'Package version: {package}')
if (RUN / 'artifact_verification.json').exists():
    for name, expected in read('artifact_verification.json')['original_files_sha256'].items():
        check(digest(RUN / name) == expected, f'Preserved original changed: {name}')
print(json.dumps({'verified': True, 'prediction_rows_rescored': row_count,
                  'saved_token_records_checked': token_rows, 'greedy_reproduction': controls,
                  'answer_matches': {key: value['answer_matches'] for key, value in groups.items()},
                  'limits': ['No model inference or tokenizer decode/render replay locally.',
                             'Token counts/EOS/think markers checked against saved generation metadata; 84 original greedy development rows lack saved token IDs.',
                             'Adapter binary hashes verified by saved Kaggle receipt; no local binary download.',
                             'One sampling seed and bounded generations do not establish general utility preservation or unlearning.']}, indent=2))
