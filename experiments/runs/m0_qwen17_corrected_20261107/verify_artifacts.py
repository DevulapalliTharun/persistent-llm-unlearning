"""Verify the preserved derived baseline; optional offline tokenizer replay, no weights."""

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

from src.evaluation.answers import evaluation_messages
from src.evaluation.corrected_m0 import verify_output
from src.evaluation.mode_diagnostic import decode_generation
from src.m0_knowledge_audit import sha256_file
from src.training.data import read_rows


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tokenizer-cache', type=Path)
args = parser.parse_args()
config_path = ROOT / 'configs/evaluation/m0_qwen17_corrected_v1_1.json'
config = json.loads(config_path.read_text())
plan = json.loads((RUN / 'plan.json').read_text())
summary = json.loads((RUN / 'summary.json').read_text())
assert plan['config_sha256'] == sha256_file(config_path)
for name, digest in plan['source_sha256'].items():
    recorded = subprocess.check_output(['git', 'show', f'{plan["source_git_commit"]}:{name}'])
    assert hashlib.sha256(recorded).hexdigest() == digest, name
report = verify_output(config, RUN)
assert report == json.loads((RUN / 'artifact_verification.json').read_text())
assert summary['versions'] == plan['parent_versions'] and summary['gpu'] == plan['parent_gpu']
assert summary['dtype'] == 'torch.float16'
freeze = (RUN / 'environment.freeze.txt').read_text().splitlines()
for package, version in summary['versions'].items():
    assert f'{package}=={version}' in freeze
local_report = RUN / 'local_verification.json'
if local_report.exists():
    for name, digest in json.loads(local_report.read_text())['original_files_sha256'].items():
        assert sha256_file(RUN / name) == digest, name
tokenizer_records = 0
if args.tokenizer_cache:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'],
        cache_dir=args.tokenizer_cache, local_files_only=True, token=False)
    metadata = json.loads((RUN / 'generation_metadata.json').read_text())
    assert hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == metadata['chat_template_sha256']
    for name in ('control_predictions.jsonl', 'refreshed_predictions.jsonl',
                 'answer_only_predictions.jsonl', 'thinking_predictions.jsonl'):
        for row in read_rows(RUN / name):
            thinking = row['mode'] == 'thinking'
            cap = config['thinking_max_new_tokens'] if thinking else config['answer_only_max_new_tokens']
            prompt = tokenizer.apply_chat_template(evaluation_messages(row['question']), tokenize=False,
                add_generation_prompt=True, enable_thinking=thinking)
            text, ids, details = decode_generation(tokenizer, row['generated_token_ids'], metadata['eos_token_id'], cap)
            assert prompt == row['rendered_prompt'] and text == row['generated_text'] and ids == row['generated_token_ids']
            assert all(row[key] == value for key, value in details.items())
            tokenizer_records += 1
    assert tokenizer_records == 2600  # 80 fresh/control plus 2520 derived, with intentional overlap.
report.update(local_verification_passed=True, offline_tokenizer_records_replayed=tokenizer_records,
              source_commit_verified=plan['source_git_commit'],
              mode_counts={mode: {key: data['answer_scoring'][key] for key in (
                  'answer_matches', 'non_year_answer_matches', 'format_violations', 'token_limit_hits')}
                  for mode, data in summary['modes'].items()})
print(json.dumps(report, indent=2))
