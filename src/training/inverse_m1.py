"""Prospective inverse-book auxiliary SFT, preserving original full-M1 evaluation."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path

from src.m0_knowledge_audit import sha256_file
from src.m0_smoke_test import git_commit, installed_packages, write_json
from src.training.data import read_rows
from src.training.evidence_zip import archive_evidence, publish_zip
from src.training.full_artifacts import publish_verified
from src.training.full_m1 import prepare_plan as reference_plan, read, run_training
from src.training.inverse_data import combine_with_original, inverse_examples
from src.training.mixed_data import encode_mixed, masking_report
from src.training.tiny_m1 import write_rows


def prepare_plan(config):
    reference_path = Path(config['reference_full_config'])
    if sha256_file(reference_path) != config['reference_full_config_sha256']:
        raise ValueError('Original full-M1 configuration changed')
    reference = read(reference_path)
    changed_fields = {'run_type', 'optimizer_steps', 'checkpoint_attempts', 'checkpoint_selection', 'persistence'}
    for key, value in reference.items():
        if key not in changed_fields and config.get(key) != value:
            raise ValueError(f'Keep original facts/recipe/evaluation/gates: {key}')
    if (config['run_type'] != 'full_m1_qwen17_inverse_support_v2' or config['optimizer_steps'] != 1372
            or config['checkpoint_attempts'] != [343, 686, 1029, 1372]
            or config['checkpoint_selection'] != 'Only the endpoint at1372 attempts is eligible; no best-dev/early-stop selection.'
            or config['initialization_policy'] != 'fresh_public_m0_new_lora'
            or config['budget_policy'] != 'Ceiling of v1 steps times5760/5040; near-equal nominal per-row exposure.'
            or config['evidence_path_prefix'] != 'evidence/m1-full-inverse-v2'):
        raise ValueError('Keep the declared inverse-support/1372-attempt endpoint protocol')
    expected_persistence = {**reference['persistence'], 'path_prefix': 'shared/m1-full-qwen17-inverse-v2'}
    if config['persistence'] != expected_persistence:
        raise ValueError('Use a distinct private v2 checkpoint namespace')
    plan, original, dev, baselines = reference_plan(reference)
    control = Path(config['control_dir'])
    if sha256_file(control / 'local_verification.json') != config['control_verification_sha256']:
        raise ValueError('Verified v1 control report changed')
    verification = read(control / 'local_verification.json')
    if not verification['verified'] or verification['candidate_passed']:
        raise ValueError('Preserve the verified failed v1 comparison')
    for name, expected in verification['original_files_sha256'].items():
        if sha256_file(control / name) != expected:
            raise ValueError(f'Original control evidence changed: {name}')
    control_summary = read(control / 'summary.json')
    if (control_summary['plan']['config'] != reference or not control_summary['execution_complete']
            or control_summary['optimizer_updates'] != 1200 or control_summary['m1_candidate']['passed']):
        raise ValueError('Control is not the completed failed original1200 trial')
    bank_path = Path(config['inverse_training_bank'])
    if sha256_file(bank_path) != config['inverse_training_bank_sha256']:
        raise ValueError('Auxiliary template bank changed')
    entities = read_rows(Path(config['bundle']) / 'entities/all_m1.jsonl')
    auxiliary, audit = inverse_examples(entities, read(bank_path))
    train = combine_with_original(original, auxiliary, config['bundle'])
    steps = (reference['optimizer_steps'] * len(train) + len(original) - 1) // len(original)
    if steps != config['optimizer_steps']:
        raise ValueError('Budget no longer matches near-equal row exposure')
    effective_batch = config['batch_size'] * config['gradient_accumulation_steps']
    original_passes = reference['optimizer_steps'] * effective_batch / len(original)
    candidate_passes = steps * effective_batch / len(train)
    comparison = {'control_dir': str(control), 'control_summary_sha256': sha256_file(control / 'summary.json'),
                  'reference_config_sha256': config['reference_full_config_sha256'],
                  'original_rows': len(original), 'auxiliary_rows': len(auxiliary), 'total_rows': len(train),
                  'original_attempts': 1200, 'candidate_attempts': steps,
                  'original_nominal_passes': original_passes, 'candidate_nominal_passes': candidate_passes,
                  'relative_row_exposure_difference': candidate_passes / original_passes - 1,
                  'old_targets_preserved_as_prefix': train[:5040] == original,
                  'evaluation_inputs_and_gates_unchanged': True,
                  'limit': 'Auxiliary task plus near-matched row exposure; more updates/tokens/compute, one seed. Not a token/compute-matched causal study or guaranteed repair.'}
    plan = deepcopy(plan)
    plan.update(config=config, training_rows=5760, unique_training_questions=2880,
                training_rows_by_mode=dict(Counter(r['training_mode'] for r in train)),
                training_rows_by_partition=dict(Counter(r['entity_partition'] for r in train)),
                augmentation={'original_seven_task_audit': plan['augmentation'], 'inverse_support': audit,
                              'training_rows': 5760, 'unique_training_questions': 2880,
                              'original_target_prefix_unchanged': True, 'original_bundle_unchanged': True},
                auxiliary_rows=720, original_training_rows=5040,
                planned_update_attempts=steps, required_applied_updates=steps,
                nominal_examples_exposed=steps * effective_batch, nominal_dataset_passes=candidate_passes,
                attempt_budget_rationale='Ceil(1200*5760/5040)=1372; integer budget keeps nominal per-row exposure within0.05% of v1.',
                checkpoint_attempts=config['checkpoint_attempts'], checkpoint_selection=config['checkpoint_selection'],
                baseline_versions=control_summary['versions'],
                comparison=comparison, model_executed=False, canonical_m1=False,
                initialization='Fresh pinned public1.7B base/new LoRA; original failed M1 is comparison only, never resumed.')
    return plan, train, dev, baselines


def backup_existing(config, output):
    token = os.environ.get('HF_TOKEN')
    if not token or not (output / 'plan.json').is_file() or read(output / 'plan.json')['config'] != config:
        raise ValueError('Require this exact run plan and existing private token for backup')
    if (output / 'artifact_manifest.json').is_file():
        return publish_verified(output, config['persistence'], config['persistence']['path_prefix'] + '/' + output.name, token)
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    if not api.repo_info(config['persistence']['repo_id'], repo_type='model').private:
        raise ValueError('Partial evidence destination must be private')
    remote = config['persistence']['path_prefix'] + '/' + output.name
    commit = api.upload_folder(repo_id=config['persistence']['repo_id'], repo_type='model', folder_path=str(output),
                              path_in_repo=remote, commit_message='Preserve incomplete inverse-support M1 attempt')
    receipt = {'repo_id': config['persistence']['repo_id'], 'path': remote, 'artifact_revision': commit.oid,
               'execution_complete': False, 'canonical_m1': False, 'endpoint_roundtrip_verified': False}
    write_json(output / 'partial_persistence.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('plan', 'prepare', 'fit', 'train', 'backup', 'export'), default='plan')
    parser.add_argument('--fit-dir', type=Path)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--tokenizer-cache', type=Path)
    args = parser.parse_args()
    config = read(args.config)
    plan, train, dev, baselines = prepare_plan(config)
    output = args.output_dir
    if args.mode == 'backup':
        print(json.dumps(backup_existing(config, output), indent=2))
        return 0
    if args.mode == 'export':
        if not (output / 'plan.json').is_file() or read(output / 'plan.json')['config'] != config:
            raise ValueError('Export only a run with the matching recorded plan')
        archive = archive_evidence(output)
        print('DOWNLOAD THIS ZIP:', archive, flush=True)
        receipt = publish_zip(archive, config['persistence'], config['evidence_path_prefix'], os.environ.get('HF_TOKEN'))
        print(json.dumps(receipt, indent=2))
        return 0
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    sources = ['src/training/inverse_m1.py', 'src/training/inverse_data.py', 'src/training/evidence_zip.py',
               'src/training/full_m1.py', 'src/training/full_artifacts.py', 'src/training/mixed_m1.py',
               'src/training/paraphrase_data.py', 'src/training/mixed_data.py', 'src/training/data.py',
               'src/training/tiny_m1.py', 'src/training/model_capacity.py', 'src/evaluation/answers.py',
               'src/evaluation/mode_diagnostic.py', 'src/evaluation/full_m0.py', 'src/evaluation/corrected_m0.py',
               'src/evaluation/utility_diagnostic.py', config['paraphrase_bank'], config['inverse_training_bank']]
    plan.update(config_sha256=sha256_file(args.config), source_git_commit=git_commit(),
                source_sha256={name: sha256_file(Path(name)) for name in sources})
    write_json(output / 'plan.json', plan)
    write_json(output / 'augmentation_audit.json', plan['augmentation'])
    write_rows(output / 'selected_train.jsonl', train)
    write_rows(output / 'selected_dev.jsonl', dev)
    if args.mode == 'plan':
        print(json.dumps({key: value for key, value in plan.items() if key not in ('config', 'augmentation', 'source_sha256')}, indent=2))
        return 0
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'],
        padding_side='left', token=os.environ.get('HF_TOKEN'), local_files_only=args.local_files_only, cache_dir=args.tokenizer_cache)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_mixed(tokenizer, row, config['max_sequence_length']) for row in train]
    write_json(output / 'masking_report.json', masking_report(encoded))
    if args.mode == 'prepare':
        print(json.dumps(masking_report(encoded), indent=2))
        return 0
    if args.mode == 'fit':
        from src.training.model_capacity import memory_fit_probe
        report = memory_fit_probe(config, tokenizer, encoded, dev, local_files_only=args.local_files_only)
        write_json(output / 'fit_report.json', report)
        (output / 'environment.freeze.txt').write_text('\n'.join(installed_packages()) + '\n')
        print(json.dumps(report, indent=2))
        return 0 if report['success'] else 2
    if args.fit_dir is None:
        raise ValueError('Require a separate disposable fit directory before training')
    try:
        result = run_training(config, plan, train, dev, baselines, encoded, tokenizer, output, args.fit_dir, args.local_files_only)
        summary = read(output / 'summary.json')
        if summary['execution_complete']:
            original = read(Path(config['control_dir']) / 'summary.json')
            write_json(output / 'comparison_v1.json', {'control_summary_sha256': plan['comparison']['control_summary_sha256'],
                'exposure_policy': plan['comparison'], 'control_post': original['evaluation']['post'],
                'candidate_post': summary['evaluation']['post'], 'control_verdict': original['m1_candidate'],
                'candidate_verdict': summary['m1_candidate'], 'canonical_m1': False})
        return result
    except Exception as error:
        message = str(error)
        token = os.environ.get('HF_TOKEN')
        if token:
            message = message.replace(token, '[redacted]')
        write_json(output / 'failure.json', {'execution_complete': False, 'error_type': type(error).__name__,
            'message': message, 'canonical_m1': False, 'action': 'Preserve partial records/receipts; no automatic resume or retry.'})
        raise


if __name__ == '__main__':
    main()
