"""Verify full M0 audit; optional cached-tokenizer replay, never model inference."""

import argparse
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

from src.evaluation.answers import evaluation_messages, generation_details
from src.evaluation.full_m0 import prepare_plan, score_audit_row, summarize_mode
from src.evaluation.mode_diagnostic import decode_generation
from src.training.data import read_rows


def read(name):
    return json.loads((RUN / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, label):
    if not condition:
        raise ValueError(label)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokenizer-cache', type=Path, help='Optional offline cache of the exact pinned tokenizer')
args = parser.parse_args()
summary, plan, metadata = read('summary.json'), read('plan.json'), read('generation_metadata.json')
config_path = ROOT / 'configs/evaluation/m0_qwen17_full_v1.json'
config = json.loads(config_path.read_text())
expected_plan, inputs = prepare_plan(config)
check(plan['config'] == config and plan['config_sha256'] == digest(config_path), 'Config/hash')
check(all(plan[k] == v for k, v in expected_plan.items()), 'Plan differs')
check(summary['plan'] == plan and summary['execution_complete'] and summary['model_executed'], 'Completion/embedded plan')
check(summary['training_updates'] == 0 and summary['adapters_loaded'] is False and summary['weights_updated'] is False and summary['dataset_accepted'] is False, 'Audit scope')
for name, expected in plan['source_sha256'].items():
    data = subprocess.check_output(['git', 'show', f"{plan['source_git_commit']}:{name}"])
    check(hashlib.sha256(data).hexdigest() == expected, f'Source hash: {name}')
tokenizer = None
if args.tokenizer_cache:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'],
        cache_dir=args.tokenizer_cache, local_files_only=True, token=False)
    check(hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == metadata['chat_template_sha256'], 'Pinned tokenizer chat template')
    check(tokenizer.pad_token_id == metadata['pad_token_id'] and tokenizer.eos_token_id in metadata['eos_token_id'], 'Tokenizer EOS/pad')
    check(tokenizer.convert_tokens_to_ids('<think>') == metadata['think_open_token_id'] and tokenizer.convert_tokens_to_ids('</think>') == metadata['think_close_token_id'], 'Tokenizer thinking marker IDs')
all_predictions, modes = [], {}
for mode in ('answer_only', 'thinking'):
    filename = mode + '_predictions.jsonl'
    check(digest(RUN / filename) == summary['prediction_files_sha256'][filename], f'Prediction hash: {mode}')
    rows = read_rows(RUN / filename)
    check(len(rows) == len(inputs) == 1260 and len({r['example_id'] for r in rows}) == 1260, 'Input count/uniqueness')
    view_indices = {}
    thinking = mode == 'thinking'
    cap = config['thinking_max_new_tokens'] if thinking else config['answer_only_max_new_tokens']
    for row, example in zip(rows, inputs):
        view = example['prompt_partition']
        index = view_indices.get(view, 0)
        view_indices[view] = index + 1
        check(row['mode'] == mode and row['thinking_enabled'] == thinking, 'Mode flag')
        check(row['batch_seed'] == 42 + (index // 16) * 16, 'Per-view batch seed')
        prompt = (f"<|im_start|>system\n{config['system_prompt']}<|im_end|>\n"
                  f"<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n")
        if not thinking:
            prompt += '<think>\n\n</think>\n\n'
        check(row['rendered_prompt'] == prompt, 'Rendered prompt/only question inputs')
        expected = score_audit_row(example, row['generated_text'], thinking, prompt, set(plan['externally_flagged_record_ids']))
        ids = row['generated_token_ids']
        details = generation_details(ids, metadata['eos_token_id'], cap)
        expected.update(details)
        expected['protocol_issue'] = bool(not expected['format_compliant'] or expected['hit_token_limit'] or expected.get('boundary_error'))
        expected['needs_match_review'] = bool(expected['potential_answer_review'] or expected['external_surface_review_flag'] or expected['protocol_issue'])
        check(all(row[k] == v for k, v in expected.items()), f'Score/input/flags: {mode}/{example["example_id"]}')
        check(len(ids) == row['generated_token_count'] and len(ids) <= cap, 'Token count/cap')
        check(ids.count(metadata['think_open_token_id']) == row['generated_text'].count('<think>') and ids.count(metadata['think_close_token_id']) == row['generated_text'].count('</think>'), 'Token/text thinking markers')
        if tokenizer:
            rendered = tokenizer.apply_chat_template(evaluation_messages(example['question']), tokenize=False,
                add_generation_prompt=True, enable_thinking=thinking)
            decoded, saved_ids, replay_details = decode_generation(tokenizer, ids, metadata['eos_token_id'], cap)
            check(rendered == prompt and decoded == row['generated_text'] and saved_ids == ids and replay_details == details, 'Pinned tokenizer prompt/decode replay')
    modes[mode] = summarize_mode(rows)
    all_predictions.extend(rows)
check(modes == summary['modes'] and len(all_predictions) == 2520, 'Full summary groups/count')
expected_review = [r for r in all_predictions if r['potential_answer_review']]
check(read('potential_answers_for_review.json')['rows'] == expected_review, 'Potential-answer review selection')
check(read('protocol_issues_for_review.json')['rows'] == [r for r in all_predictions if r['protocol_issue']], 'Protocol-issue review selection')
check(read('external_collision_rows_for_review.json')['rows'] == [r for r in all_predictions if r['external_surface_review_flag']], 'External collision review selection')
freeze = (RUN / 'environment.freeze.txt').read_text().splitlines()
for package, version in summary['versions'].items():
    check(f'{package}=={version}' in freeze, f'Environment version: {package}')
if (RUN / 'artifact_verification.json').exists():
    for name, expected in read('artifact_verification.json')['original_files_sha256'].items():
        check(digest(RUN / name) == expected, f'Original file changed: {name}')
print(json.dumps({'verified': True, 'prediction_rows_rescored': 2520,
                  'saved_token_records_checked': 2520, 'offline_tokenizer_render_and_decode_replayed': tokenizer is not None,
                  'mode_counts': {mode: {'answer_matches': m['answer_scoring']['answer_matches'],
                    'non_year_matches': m['answer_scoring']['non_year_answer_matches'],
                    'format_violations': m['answer_scoring']['format_violations'],
                    'token_limit_hits': m['answer_scoring']['token_limit_hits'],
                    'missing_thinking_boundaries': m['missing_thinking_boundaries'],
                    'reasoning_mentions': m['reasoning_answer_mentions']} for mode, m in modes.items()},
                  'potential_answer_review_rows': len(expected_review),
                  'protocol_issue_rows': sum(r['protocol_issue'] for r in all_predictions),
                  'external_review_rows': sum(r['external_surface_review_flag'] for r in all_predictions),
                  'limits': ['No model-weight/inference replay or proof of absent internal knowledge.',
                             'Year matches/trace mentions are not evidence of the complete synthetic graph.',
                             'Execution completion is not dataset acceptance; confirmed surface collision remains unresolved.']}, indent=2))
