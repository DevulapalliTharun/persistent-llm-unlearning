from pathlib import Path
import os, sys, tempfile
remote_root = Path('/kaggle/working')
sys.path.insert(0, str(Path.cwd()))
import hashlib, json
from collections import Counter
from copy import deepcopy
from src.training.data import read_rows, select_pilot, utility_examples
from src.training.full_m1 import prepare_plan as prepare_full_plan, generate_facts, same_predictions
from src.evaluation.mode_diagnostic import sampled_evaluation
from src.evaluation.answers import summarize_answer_protocol
from src.m0_smoke_test import write_json, installed_packages, package_version
from huggingface_hub import HfApi, hf_hub_download

config = json.loads(r"""{
  "run_type": "full_m1_prize_author_wording_v1",
  "helper_git_commit": "35b961e914a57a0fc09cd4c9d02e634670778d73",
  "model_id": "Qwen/Qwen3-1.7B",
  "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
  "training_config": "configs/training/m1_full_qwen17_v1.json",
  "training_config_sha256": "01d8006068c3fa33c754d531895cb2c904b1230dd92b2cc3a4c6785f6933c8ea",
  "task_type": "prize_to_author",
  "batch_size": 16,
  "seed": 42,
  "answer_only_max_new_tokens": 128,
  "thinking_max_new_tokens": 1024,
  "fact_control_rows_per_mode": 16,
  "utility_control_rows": 10,
  "sampling": {
    "do_sample": true,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0
  },
  "artifact": {
    "repo_id": "tharundevulapalli/persistent-llm-unlearning-checkpoints",
    "revision": "a98bc31bcb3f7c404ea91ead35689301ab0c40d3",
    "path": "shared/m1-full-qwen17-v1/m1-full-qwen17-v1-hpcajqfx",
    "adapter_files_sha256": {
      "adapter/README.md": "7580bf8de69547f09962e6e9ac8f70b2d0dcfb140bbdc2e97aa6237e950ff449",
      "adapter/adapter_config.json": "de91b1a2868059df44e853881c82995a2d1d9864ac62abb5d5c14ce6c8ba4ddf",
      "adapter/adapter_model.safetensors": "674aa28ee0bfd806829447844e8eb50c23f768ff1233cfc702b82a5542d12a99"
    },
    "evidence_sha256": {
      "summary.json": "ce0ae0222f440cc6acb9eaa5da06ee1af59d028615db2af601b591638542a270",
      "plan.json": "89f8d951c0bee13383b3e05e2a6d1d524b04b4ec44a76029f20a6d09ed1a4e2a",
      "artifact_manifest.json": "decb1d436293a488f835840952cea6e765be346705c147d58f74916fb65a7f8e",
      "selected_train.jsonl": "183c903045428de92b41ae04f8f36dd0b691f6c707ae045fae8dafae227c00ec",
      "selected_dev.jsonl": "dca17e13f011902db82c7e19aeb9e3e9f707dc6d7dfdfd8f82729476c7807449",
      "post_dev_answer_only.jsonl": "7559250824047e8dc31a26d78f9b2e8e67cc203fff0ecaa1c04050ebc222a12b",
      "post_dev_thinking.jsonl": "aaff8222456c5359d94d971f7559007f7f2bcf630f578d8d504c9e331e9184a8",
      "post_utility_thinking.jsonl": "a93f98947c9bbd20ef19a489e218299083c4218cf090c77359b20b3cd5eb9666",
      "generation_metadata.json": "4d5103ff770cf5350f074ff7847ea0c89709819f4eb6e77ca040721466ab6897"
    }
  }
}""")

# Embedded diagnostic-only helpers, tested locally; generated from src/evaluation/full_wording_diagnostic.py.
def build_overlay(training, development):
    if len(training) != 840 or len(development) != 840:
        raise ValueError('Use all840 original F/R training and development prompts')
    key = lambda row: (row['record_id'], row['task_type'])
    train = {key(row): row for row in training}
    if len(train) != 840 or len({key(row) for row in development}) != 840 or set(train) != {key(row) for row in development}:
        raise ValueError('Require unique matching record/task coverage')
    if any(row['prompt_partition'] != 'm1_train' or row['entity_partition'] not in ('F', 'R') for row in training):
        raise ValueError('Only original F/R training wording is diagnostic input')
    if any(row['prompt_partition'] != 'm1_dev' or row['entity_partition'] not in ('F', 'R') for row in development):
        raise ValueError('Only original F/R development view is a reference')
    if {r['template_id'] for r in training} & {r['template_id'] for r in development}:
        raise ValueError('Training/development template overlap')
    overlay, replacements = [], []
    for index, dev in enumerate(development):
        original = train[key(dev)]
        if any(original[field] != dev[field] for field in ('answer', 'entity_partition', 'target_field', 'hop_count', 'path', 'source_entity_ids')):
            raise ValueError('Paired questions disagree on canonical association')
        if original['question'] == dev['question']:
            raise ValueError('Require distinct training/development wording')
        target = dev['task_type'] == 'prize_to_author'
        overlay.append(deepcopy(original if target else dev))
        if target:
            replacements.append({'index': index, 'batch_offset': (index // 16) * 16,
                'batch_position': index % 16, 'record_id': dev['record_id'], 'entity_partition': dev['entity_partition'],
                'development_example_id': dev['example_id'], 'training_example_id': original['example_id']})
    if len(replacements) != 120 or Counter(r['entity_partition'] for r in replacements) != {'F': 30, 'R': 90}:
        raise ValueError('Require all120 prize-to-author pairs,30 F/90 R')
    return overlay, {'replacements': replacements, 'replaced_questions': 120, 'unchanged_fillers': 720,
                     'full_rows_per_mode': 840, 'target_rows_by_partition': {'F': 30, 'R': 90},
                     'batching': 'Original840-row dev order, batch16, seed42+original offset; only120 target questions replaced.'}


def paired_results(previous, probed):
    if len(previous) != 840 or len(probed) != 840:
        raise ValueError('Pair full840-row reference and overlay outputs')
    categories = ('correct_both', 'training_only_correct', 'development_only_correct', 'wrong_both')
    groups = {part: {category: 0 for category in categories} for part in ('F', 'R')}
    pairs, filler_changes = [], []
    for old, new in zip(previous, probed):
        if any(old[key] != new[key] for key in ('record_id', 'task_type', 'answer', 'entity_partition')):
            raise ValueError('Prediction pair identity/answer differs')
        if old['task_type'] != 'prize_to_author':
            if old['example_id'] != new['example_id'] or old['question'] != new['question']:
                raise ValueError('Unchanged filler input was modified')
            if old['generated_token_ids'] != new['generated_token_ids'] or old['raw_response'] != new['raw_response']:
                filler_changes.append(old['example_id'])
            continue
        if old['prompt_partition'] != 'm1_dev' or new['prompt_partition'] != 'm1_train':
            raise ValueError('Target must compare development versus original training wording')
        t, d = new['answer_match'], old['answer_match']
        category = 'correct_both' if t and d else 'training_only_correct' if t else 'development_only_correct' if d else 'wrong_both'
        groups[old['entity_partition']][category] += 1
        pairs.append({'record_id': old['record_id'], 'entity_partition': old['entity_partition'], 'answer': old['answer'],
                      'development_example_id': old['example_id'], 'training_example_id': new['example_id'],
                      'development_question': old['question'], 'training_question': new['question'],
                      'development_generated_text': old['generated_text'], 'training_generated_text': new['generated_text'],
                      'development_correct': d, 'training_correct': t, 'category': category})
    if len(pairs) != 120 or any(sum(groups[part].values()) != count for part, count in (('F', 30), ('R', 90))):
        raise ValueError('Incomplete paired target denominator')
    return {'by_partition': groups, 'pairs': pairs, 'changed_filler_count': len(filler_changes),
            'changed_filler_example_ids': filler_changes,
            'scope': 'Training-wording recall diagnostic, not held-out accuracy or a replacement full-M1 gate. One seed; padding/batch effects can accompany wording changes.'}


def diagnostic_plan(config, training_config, source_summary, training, development):
    fixed = {'run_type': 'full_m1_prize_author_wording_v1',
             'helper_git_commit': '35b961e914a57a0fc09cd4c9d02e634670778d73',
             'task_type': 'prize_to_author', 'batch_size': 16, 'seed': 42,
             'answer_only_max_new_tokens': 128, 'thinking_max_new_tokens': 1024,
             'fact_control_rows_per_mode': 16, 'utility_control_rows': 10}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError('Keep the fixed original-batch wording diagnostic protocol')
    if (config['model_id'] != training_config['model_id'] or config['model_revision'] != training_config['model_revision']
            or source_summary['plan']['config'] != training_config
            or source_summary['plan']['source_git_commit'] != config['helper_git_commit']
            or source_summary['plan']['config_sha256'] != config['training_config_sha256']):
        raise ValueError('Use the original full-M1 model/config/source')
    if (not source_summary['execution_complete'] or source_summary['optimizer_updates'] != 1200
            or source_summary['m1_candidate']['passed'] or source_summary['canonical_m1']):
        raise ValueError('Use the verified failed1200 endpoint; never substitute another checkpoint')
    artifact = config['artifact']
    if (artifact['revision'] != 'a98bc31bcb3f7c404ea91ead35689301ab0c40d3'
            or artifact['repo_id'] != 'tharundevulapalli/persistent-llm-unlearning-checkpoints'
            or artifact['path'] != 'shared/m1-full-qwen17-v1/m1-full-qwen17-v1-hpcajqfx'):
        raise ValueError('Pin the original private endpoint')
    expected_adapter = {k: v for k, v in source_summary['artifact_files_sha256'].items() if k.startswith('adapter/')}
    if artifact['adapter_files_sha256'] != expected_adapter:
        raise ValueError('Saved adapter hashes differ')
    if config['sampling'] != {'do_sample': True, 'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0}:
        raise ValueError('Preserve the source thinking sampler')
    overlay, audit = build_overlay(training, development)
    plan = {'config': config, 'training_updates': 0, 'model_executed': False, 'canonical_m1': False,
            'original_candidate_passed': False, 'original_candidate_result_unchanged': True,
            'control_predictions': 42, 'overlay_predictions': 1680, 'planned_predictions': 1722,
            'source_dev_records_reused': 1680, 'target_pairs_per_mode': 120, 'audit': audit,
            'scope': 'Saved adapter inference only, no optimizer, SFT, new model checkpoint, dataset/gate edits or final-test input.'}
    return plan, overlay


embedded_helpers_sha256 = 'f240b7880fa3553797781c513444becafe5e29a0f1b649da0d36511f1105255d'

hf_token = os.environ['HF_TOKEN']
artifact = config['artifact']
api = HfApi(token=hf_token)
assert api.repo_info(artifact['repo_id'], repo_type='model').private
restore_dir = Path(tempfile.mkdtemp(prefix='full-wording-restore-', dir=remote_root))
verified = {}
def restore_file(name, expected):
    path = Path(hf_hub_download(repo_id=artifact['repo_id'], repo_type='model', filename=artifact['path'] + '/' + name, revision=artifact['revision'], token=hf_token, local_dir=restore_dir))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    verified[name] = expected
    return path
references = {name: restore_file(name, expected) for name, expected in artifact['evidence_sha256'].items()}
assert hashlib.sha256(Path(config['training_config']).read_bytes()).hexdigest() == config['training_config_sha256']
training_config = json.loads(Path(config['training_config']).read_text())
source_summary = json.loads(references['summary.json'].read_text())
source_plan = json.loads(references['plan.json'].read_text())
assert source_summary['plan'] == source_plan
expected_full_plan, expected_train, expected_dev, _ = prepare_full_plan(training_config)
assert all(source_plan[key] == value for key, value in expected_full_plan.items())
assert read_rows(references['selected_train.jsonl']) == expected_train
assert read_rows(references['selected_dev.jsonl']) == expected_dev
for name, expected in source_plan['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
original_train, dev_questions = select_pilot(training_config)
plan, overlay = diagnostic_plan(config, training_config, source_summary, original_train, dev_questions)
source_predictions = {mode: read_rows(references['post_dev_' + mode + '.jsonl']) for mode in ['answer_only', 'thinking']}
for mode, rows in source_predictions.items():
    assert len(rows) == 840 and all(all(row.get(key) == value for key, value in example.items()) for row, example in zip(rows, dev_questions))
for name, expected in artifact['adapter_files_sha256'].items():
    restore_file(name, expected)
adapter_path = restore_dir / artifact['path'] / 'adapter'
adapter_config = json.loads((adapter_path / 'adapter_config.json').read_text())
assert adapter_config['base_model_name_or_path'] == config['model_id'] and adapter_config['revision'] == config['model_revision']
output_dir = Path(os.environ['WORDING_OUTPUT_DIR'])
output_dir.mkdir(parents=True, exist_ok=True)
plan.update(config_canonical_json_sha256=hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), helper_git_commit=config['helper_git_commit'], embedded_helpers_sha256=embedded_helpers_sha256, diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
write_json(output_dir / 'plan.json', plan)
write_json(output_dir / 'source_artifact_verification.json', {'artifact': artifact, 'verified_sha256': verified})
(output_dir / 'selected_overlay.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in overlay))
print('Restored saved endpoint with matching hashes; token is not displayed.')
print('Diagnostic outputs:', output_dir)
print('Planned1722 inference outputs; original full-M1 verdict stays failed; zero training updates.')

import gc, time
from datetime import datetime, timezone
import torch
from peft import PeftModel
from peft.import_utils import is_torchao_available
from transformers import AutoModelForCausalLM, AutoTokenizer

is_torchao_available()
assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == source_summary['gpu']
versions = {name: package_version(name) for name in source_summary['versions']}
assert versions == source_summary['versions'], 'Source environment changed; stop before comparison.'
started = datetime.now(timezone.utc).isoformat()
clock = time.perf_counter()
base = model = None
try:
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'], padding_side='left', token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(config['model_id'], revision=config['model_revision'], dtype=torch.float16, token=hf_token).to('cuda:0')
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False).eval()
    model.requires_grad_(False)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    metadata = {'eos_token_id': model.generation_config.eos_token_id, 'pad_token_id': tokenizer.pad_token_id, 'think_open_token_id': tokenizer.convert_tokens_to_ids('<think>'), 'think_close_token_id': tokenizer.convert_tokens_to_ids('</think>'), 'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
    assert metadata == json.loads(references['generation_metadata.json'].read_text())
    write_json(output_dir / 'generation_metadata.json', metadata)
    controls = {mode: generate_facts(model, tokenizer, dev_questions[:16], training_config, source_plan, mode, output_dir / ('control_' + mode + '.jsonl')) for mode in ['answer_only', 'thinking']}
    utility = sampled_evaluation(model, tokenizer, utility_examples(), True, {'batch_size': 16, 'seed': 42, 'sampled_max_new_tokens': 1024}, output_dir / 'control_utility_thinking.jsonl')
    guard = {mode: same_predictions(controls[mode], source_predictions[mode][:16]) for mode in controls}
    guard['utility_thinking'] = same_predictions(utility, read_rows(references['post_utility_thinking.jsonl']))
    write_json(output_dir / 'reference_reproduction.json', {'passed': all(guard.values()), 'checks': guard})
    if not all(guard.values()):
        summary = {'execution_complete': False, 'model_executed': True, 'training_updates': 0, 'canonical_m1': False, 'original_candidate_result_unchanged': True, 'plan': plan, 'controls': guard, 'stop_reason': 'Saved endpoint controls did not reproduce; no wording inference started. Preserve evidence and inspect environment/adapter before retrying.'}
    else:
        results = {mode: generate_facts(model, tokenizer, overlay, training_config, source_plan, mode, output_dir / ('overlay_' + mode + '.jsonl')) for mode in ['answer_only', 'thinking']}
        paired = {mode: paired_results(source_predictions[mode], results[mode]) for mode in results}
        write_json(output_dir / 'paired_wording_review.json', paired)
        def target_metrics(rows):
            selected = [row for row in rows if row['task_type'] == 'prize_to_author']
            result = summarize_answer_protocol(selected)
            result['count'] = len(selected)
            result['missing_thinking_boundaries'] = sum(row.get('boundary_error') is not None for row in selected)
            return result
        summary = {'execution_complete': True, 'model_executed': True, 'training_updates': 0, 'weights_updated': False, 'canonical_m1': False, 'original_candidate_passed': False, 'original_candidate_result_unchanged': True, 'plan': plan, 'controls': guard, 'original_development_metrics': {mode: target_metrics(source_predictions[mode]) for mode in results}, 'training_wording_diagnostic_metrics': {mode: target_metrics(results[mode]) for mode in results}, 'paired_counts': {mode: paired[mode]['by_partition'] for mode in paired}, 'changed_fillers': {mode: paired[mode]['changed_filler_count'] for mode in paired}, 'prediction_files_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output_dir.glob('*.jsonl') if path.name.startswith(('control_', 'overlay_'))}, 'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'seconds': round(time.perf_counter() - clock, 4), 'gpu': torch.cuda.get_device_name(0), 'versions': versions, 'interpretation': 'Original failed gate preserved. Training-wording recall diagnoses access; no held-out acceptance, causal mechanism or final-test result is established.'}
    write_json(output_dir / 'summary.json', summary)
    (output_dir / 'environment.freeze.txt').write_text('\n'.join(installed_packages()) + '\n')
finally:
    del model, base
    gc.collect()
    torch.cuda.empty_cache()
