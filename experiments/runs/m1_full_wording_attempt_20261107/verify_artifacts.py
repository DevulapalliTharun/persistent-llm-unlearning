"""Recheck saved-endpoint wording evidence, optionally replaying cached tokenizer outputs."""

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.evaluation.answers import SYSTEM_PROMPT, generation_details, summarize_answer_protocol
from src.evaluation.full_m0 import score_audit_row
from src.evaluation.full_wording_diagnostic import build_overlay, diagnostic_plan, paired_results
from src.evaluation.mode_diagnostic import decode_generation, score_generated
from src.evaluation.utility_diagnostic import ARITHMETIC_PROMPT
from src.training.data import read_rows, select_pilot, utility_examples
from src.training.full_m1 import same_predictions


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
config = json.loads((ROOT / 'configs/evaluation/m1_full_wording_v1.json').read_text())
training_config = json.loads((ROOT / config['training_config']).read_text())
source = ROOT / 'experiments/runs/m1_full_qwen17_attempt_20261107'
source_summary, source_plan = json.loads((source / 'summary.json').read_text()), json.loads((source / 'plan.json').read_text())
for name, expected in config['artifact']['evidence_sha256'].items():
    check(digest(source / name) == expected, f'Original source evidence hash: {name}')
check(digest(ROOT / config['training_config']) == config['training_config_sha256'], 'Training config hash')
training, dev = select_pilot(training_config)
expected_plan, overlay = diagnostic_plan(config, training_config, source_summary, training, dev)
summary, plan, metadata = read('summary.json'), read('plan.json'), read('generation_metadata.json')
check(all(plan[key] == value for key, value in expected_plan.items()), 'Plan differs')
check(summary['plan'] == plan and summary['execution_complete'] and summary['model_executed'], 'Completion/plan')
check(summary['training_updates'] == 0 and not summary['weights_updated'] and not summary['canonical_m1'] and
      not summary['original_candidate_passed'] and summary['original_candidate_result_unchanged'], 'Inference-only scope')
check(plan['config_canonical_json_sha256'] == hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), 'Embedded config hash')
helper = '\n\n'.join(inspect.getsource(f) for f in (build_overlay, paired_results, diagnostic_plan))
check(plan['embedded_helpers_sha256'] == hashlib.sha256(helper.encode()).hexdigest(), 'Embedded helper hash')
check(plan['helper_git_commit'] == config['helper_git_commit'], 'Published helper commit')
check(plan['diagnostic_script_sha256'] == digest(RUN / 'diagnostic_source.py'), 'Actual child script hash')
nb = json.loads((ROOT / 'experiments/notebooks/14_m1_full_wording_diagnostic.ipynb').read_text())
parsed = ast.parse(''.join(nb['cells'][3]['source']))
script = next(ast.literal_eval(node.value) for node in parsed.body if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'diagnostic_script')
check((RUN / 'diagnostic_source.py').read_bytes() == script.encode(), 'Executed child script differs from issued notebook')
check(read_rows(RUN / 'selected_overlay.jsonl') == overlay, '840-row overlay')
check(metadata == json.loads((source / 'generation_metadata.json').read_text()), 'Generation metadata')
verification = read('source_artifact_verification.json')
check(verification['artifact'] == config['artifact'] and verification['verified_sha256'] ==
      {**config['artifact']['evidence_sha256'], **config['artifact']['adapter_files_sha256']}, 'Endpoint/source restore hashes')
check(summary['versions'] == source_summary['versions'] and summary['gpu'] == source_summary['gpu'], 'Source package versions/GPU')
freeze = (RUN / 'environment.freeze.txt').read_text().splitlines()
for name, version in summary['versions'].items():
    check(f'{name}=={version}' in freeze, f'Package freeze: {name}')
if (RUN / 'local_verification.json').exists():
    for name, expected in read('local_verification.json')['original_files_sha256'].items():
        check(digest(RUN / name) == expected, f'Original file changed: {name}')
tokenizer = None
if args.tokenizer_cache:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'],
        cache_dir=args.tokenizer_cache, local_files_only=True, token=False)
    check(hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == metadata['chat_template_sha256'], 'Pinned tokenizer')

previous, outputs, controls, records = {}, {}, {}, []
for mode in ('answer_only', 'thinking'):
    previous[mode] = read_rows(source / f'post_dev_{mode}.jsonl')
    outputs[mode] = read_rows(RUN / f'overlay_{mode}.jsonl')
    controls[mode] = read_rows(RUN / f'control_{mode}.jsonl')
    check(len(outputs[mode]) == 840 and len(controls[mode]) == 16, 'Overlay/control coverage')
    records.extend((mode, i, row, example) for i, (row, example) in enumerate(zip(outputs[mode], overlay)))
    records.extend((mode, i, row, example) for i, (row, example) in enumerate(zip(controls[mode], dev[:16])))
utility = read_rows(RUN / 'control_utility_thinking.jsonl')
check(len(utility) == 10, 'Utility coverage')
records.extend(('utility', i, row, example) for i, (row, example) in enumerate(zip(utility, utility_examples())))
check(len(records) == 1722, 'Total new predictions')
for mode, index, row, example in records:
    thinking = mode != 'answer_only'
    system = ARITHMETIC_PROMPT if mode == 'utility' else SYSTEM_PROMPT
    prompt = (f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{example["question"]}'
              '<|im_end|>\n<|im_start|>assistant\n')
    if not thinking:
        prompt += '<think>\n\n</think>\n\n'
    check(row['rendered_prompt'] == prompt and row['thinking_enabled'] == thinking, 'Prompt/mode')
    check(row['batch_seed'] == 42 + (index // 16) * 16, 'Original batch seed/order')
    expected = (score_generated(example, row['generated_text'], True, prompt) if mode == 'utility' else
                score_audit_row(example, row['generated_text'], thinking, prompt, set(source_plan['externally_flagged_record_ids'])))
    ids, cap = row['generated_token_ids'], 1024 if thinking else 128
    details = generation_details(ids, metadata['eos_token_id'], cap)
    expected.update(details)
    if mode != 'utility':
        check(row['mode'] == mode, 'Saved mode label')
        expected['protocol_issue'] = bool(not expected['format_compliant'] or expected['hit_token_limit'] or expected.get('boundary_error'))
        expected['needs_match_review'] = bool(expected['potential_answer_review'] or expected['external_surface_review_flag'] or expected['protocol_issue'])
    check(all(row[key] == value for key, value in expected.items()), f'Input/score/flag: {row["example_id"]}')
    check(len(ids) == row['generated_token_count'] and len(ids) <= cap, 'Token count/cap')
    if tokenizer:
        rendered = tokenizer.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': example['question']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        text, saved, replay_details = decode_generation(tokenizer, ids, metadata['eos_token_id'], cap)
        check(rendered == prompt and text == row['generated_text'] and saved == ids and replay_details == details, 'Offline tokenizer replay')

guard = {mode: same_predictions(controls[mode], previous[mode][:16]) for mode in controls}
guard['utility_thinking'] = same_predictions(utility, read_rows(source / 'post_utility_thinking.jsonl'))
check(all(guard.values()) and read('reference_reproduction.json') == {'passed': True, 'checks': guard} and summary['controls'] == guard, '42 exact controls')
pairs = {mode: paired_results(previous[mode], outputs[mode]) for mode in outputs}
check(read('paired_wording_review.json') == pairs, 'All240 target pairs/filler drift')
check(summary['paired_counts'] == {mode: pairs[mode]['by_partition'] for mode in pairs}, 'Pair summary')
check(summary['changed_fillers'] == {mode: pairs[mode]['changed_filler_count'] for mode in pairs}, 'Filler drift summary')
def target_metrics(rows):
    selected = [row for row in rows if row['task_type'] == 'prize_to_author']
    result = summarize_answer_protocol(selected)
    result.update(count=len(selected), missing_thinking_boundaries=sum(row.get('boundary_error') is not None for row in selected))
    return result
for mode in outputs:
    check(summary['original_development_metrics'][mode] == target_metrics(previous[mode]) and
          summary['training_wording_diagnostic_metrics'][mode] == target_metrics(outputs[mode]), 'Target metrics')
    fillers = [i for i, row in enumerate(overlay) if row['task_type'] != 'prize_to_author']
    check(all(same_predictions([previous[mode][i]], [outputs[mode][i]]) for i in fillers), 'All720 unchanged filler records match exactly')
for name, expected in summary['prediction_files_sha256'].items():
    check(digest(RUN / name) == expected, f'Prediction file hash: {name}')
print(json.dumps({'verified': True, 'prediction_records_rescored': 1722, 'exact_control_records': 42,
    'paired_target_records': 240, 'unchanged_filler_records_identical': 1440,
    'offline_tokenizer_replayed': tokenizer is not None, 'training_updates': 0,
    'original_full_m1_passed': False, 'paired_counts': summary['paired_counts'],
    'limits': ['No model weights or new inference replayed locally.',
               'Source adapter binary hash relies on runtime immutable restore verification.',
               'One-seed training-wording recall is diagnostic only, not replacement acceptance or a proven causal/general-intelligence result.']}, indent=2))
