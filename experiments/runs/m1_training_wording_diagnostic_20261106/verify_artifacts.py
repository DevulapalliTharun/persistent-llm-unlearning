"""Verify the fixed-adapter wording comparison without models, GPUs or downloads."""

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

from src.evaluation.answers import SYSTEM_PROMPT, generation_details, score_answer
from src.evaluation.mode_diagnostic import metrics, score_generated
from src.evaluation.utility_diagnostic import ARITHMETIC_PROMPT
from src.m0_knowledge_audit import score_prediction
from src.training.data import read_rows, select_pilot, utility_examples


def read(name):
    return json.loads((RUN / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, label):
    if not condition:
        raise ValueError(label)


summary, plan, metadata = read('summary.json'), read('plan.json'), read('generation_metadata.json')
config = json.loads((ROOT / 'configs/evaluation/training_wording_diagnostic_v1.json').read_text())
check(plan['config'] == config, 'Config snapshot')
check(plan['config_canonical_json_sha256'] == hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(), 'Config hash')
check(summary['plan'] == plan and summary['execution_complete'] and summary['model_executed'], 'Completion/plan')
check(summary['training_updates'] == 0 and not summary['canonical_m1'] and summary['original_pilot_result_unchanged'], 'Scope')
check(plan['planned_predictions'] == 346 and plan['source_git_commit'] == config['helper_git_commit'], 'Plan count/source')
for name, expected in plan['source_sha256'].items():
    data = subprocess.check_output(['git', 'show', f"{config['helper_git_commit']}:{name}"])
    check(hashlib.sha256(data).hexdigest() == expected, f'Helper source hash: {name}')
training_config_path = ROOT / config['training_config']
check(digest(training_config_path) == config['training_config_sha256'], 'Training config')
training_config = json.loads(training_config_path.read_text())
train, dev = select_pilot(training_config)
key = lambda row: (row['record_id'], row['task_type'])
train_map = {key(row): row for row in train}
train = [train_map[key(row)] for row in dev]
check(read_rows(RUN / 'selected_training_questions.jsonl') == train, 'Training questions/order')
check(read_rows(RUN / 'selected_development_questions.jsonl') == dev, 'Development questions/order')
check(len(train) == len(dev) == 84 and len(train_map) == 84, 'Pair count')
for a, b in zip(train, dev):
    check(a['answer'] == b['answer'] and a['question'] != b['question'], 'Pair semantics/wording')
check(not ({r['template_id'] for r in train} & {r['template_id'] for r in dev}), 'Template overlap')
original = ROOT / 'experiments/runs/m1_tiny_mixed_attempt_20261105'
for name, expected in config['artifact']['evidence_sha256'].items():
    check(digest(original / name) == expected, f'Original private reference hash: {name}')
original_summary = json.loads((original / 'summary.json').read_text())
receipt = read('source_artifact_verification.json')
expected_receipt_hashes = {**config['artifact']['evidence_sha256'], **original_summary['artifact_files_sha256']}
check(receipt['artifact'] == config['artifact'] and receipt['verified_sha256'] == expected_receipt_hashes, 'Adapter/reference hash receipt')
old_metadata = json.loads((original / 'generation_metadata.json').read_text())
check(all(old_metadata[k] == v for k, v in metadata.items()), 'Generation metadata')
all_rows, groups, controls = {}, {}, {}
token_records = 0
for suite in ('development_answer_only', 'development_thinking', 'utility_thinking', 'training_answer_only', 'training_thinking'):
    filename = suite + '.jsonl'
    check(digest(RUN / filename) == summary['prediction_files_sha256'][filename], f'Prediction hash: {suite}')
    rows = read_rows(RUN / filename)
    inputs = utility_examples() if suite == 'utility_thinking' else train if suite.startswith('training_') else dev
    check(len(rows) == len(inputs), 'Suite count')
    for index, (row, example) in enumerate(zip(rows, inputs)):
        if suite.endswith('answer_only'):
            expected = score_prediction(example, row['raw_response'])
            expected.update(score_answer(row['raw_response'], example['answer']))
        else:
            expected = score_generated(example, row['generated_text'], True, row['rendered_prompt'])
            system = ARITHMETIC_PROMPT if suite == 'utility_thinking' else SYSTEM_PROMPT
            prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
                      f"<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n")
            check(row['rendered_prompt'] == prompt, 'Thinking prompt')
            check(row['thinking_enabled'] and row['batch_seed'] == 42 + (index // 16) * 16, 'Mode/seed')
            ids = row['generated_token_ids']
            details = generation_details(ids, metadata['eos_token_id'], 1024)
            check(all(row[k] == v for k, v in details.items()), 'EOS/cap consistency')
            check(len(ids) == row['generated_token_count'], 'Token length')
            check(ids.count(metadata['think_open_token_id']) == row['generated_text'].count('<think>'), 'Open marker')
            check(ids.count(metadata['think_close_token_id']) == row['generated_text'].count('</think>'), 'Close marker')
            token_records += 1
        check(all(row[k] == v for k, v in expected.items()), f'Score/input: {suite}/{row["example_id"]}')
    all_rows[suite] = rows
    groups[suite] = metrics(rows)
    if not suite.startswith('training_'):
        old_name = {'development_answer_only': 'post_dev_answer_only.jsonl', 'development_thinking': 'post_dev_thinking.jsonl', 'utility_thinking': 'post_utility_thinking.jsonl'}[suite]
        previous = read_rows(original / old_name)
        differences = [r['example_id'] for r, old in zip(rows, previous) if r['raw_response'] != old['raw_response']]
        controls[suite] = {'raw_predictions_identical': not differences, 'different_example_ids': differences}
check(sum(map(len, all_rows.values())) == 346 and groups == summary['groups'], 'All scores/groups')
check(controls == read('reference_reproduction.json') == summary['controls'], 'Reference control report')
check(all(c['raw_predictions_identical'] for c in controls.values()), 'Original outputs differ')
paired = {}
for mode in ('answer_only', 'thinking'):
    counts = {'correct_both': 0, 'training_only_correct': 0, 'development_only_correct': 0, 'wrong_both': 0}
    pairs = []
    for a, b in zip(all_rows['training_' + mode], all_rows['development_' + mode]):
        check(key(a) == key(b) and a['answer'] == b['answer'], 'Pair alignment')
        t, d = a['answer_match'], b['answer_match']
        category = 'correct_both' if t and d else 'training_only_correct' if t else 'development_only_correct' if d else 'wrong_both'
        counts[category] += 1
        pairs.append({'record_id': a['record_id'], 'task_type': a['task_type'], 'entity_partition': a['entity_partition'], 'answer': a['answer'], 'training_example_id': a['example_id'], 'development_example_id': b['example_id'], 'training_question': a['question'], 'development_question': b['question'], 'training_raw_response': a['raw_response'], 'development_raw_response': b['raw_response'], 'training_correct': t, 'development_correct': d, 'category': category})
    paired[mode] = {'counts': counts, 'pairs': pairs}
check(paired == read('paired_wording_review.json'), 'Paired response/category report')
freeze = (RUN / 'environment.freeze.txt').read_text().splitlines()
for package, version in summary['versions'].items():
    check(f'{package}=={version}' in freeze, f'Environment: {package}')
if (RUN / 'artifact_verification.json').exists():
    for name, expected in read('artifact_verification.json')['original_files_sha256'].items():
        check(digest(RUN / name) == expected, f'Preserved file changed: {name}')
print(json.dumps({'verified': True, 'prediction_rows_rescored': 346, 'saved_token_records_checked': token_records,
                  'original_controls_reproduced': True,
                  'answer_matches': {key: value['answer_matches'] for key, value in groups.items()},
                  'paired_counts': {mode: values['counts'] for mode, values in paired.items()},
                  'limits': ['No local model inference/tokenizer replay or private binary download.',
                             'One sampled seed; observed wording sensitivity is not an isolated universal causal claim.',
                             'Training-question accuracy is diagnostic, not held-out performance or a new pilot pass.']}, indent=2))
