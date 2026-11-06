"""Prospective full F30/R90 M1 candidate, reusing the passing mixed-mode recipe."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from src.data.synthetic import TASKS, normalize_text
from src.evaluation.answers import evaluation_messages
from src.evaluation.corrected_m0 import verify_output
from src.evaluation.full_m0 import score_audit_row
from src.evaluation.mode_diagnostic import SAMPLING, decode_generation, metrics, sampled_evaluation
from src.m0_knowledge_audit import sha256_file
from src.m0_smoke_test import git_commit, installed_packages, package_version, write_json
from src.training.data import read_rows, utility_examples
from src.training.full_artifacts import artifact_hashes, publish_verified
from src.training.mixed_data import encode_mixed, masking_report
from src.training.mixed_m1 import candidate_verdict, train_updates
from src.training.paraphrase_data import augmented_examples
from src.training.tiny_m1 import write_rows


MODES = ('answer_only', 'thinking')


def read(path):
    return json.loads(Path(path).read_text())


def prepare_plan(config):
    reference_path = Path(config['reference_tiny_config'])
    if sha256_file(reference_path) != config['reference_tiny_config_sha256']:
        raise ValueError('Passing tiny recipe changed')
    reference = read(reference_path)
    for key in ('model_id', 'model_revision', 'protocol_version', 'seed', 'max_sequence_length', 'batch_size',
                'gradient_accumulation_steps', 'learning_rate', 'max_grad_norm', 'lora', 'evaluation',
                'pilot_gates', 'training_representation', 'sampled_max_new_tokens', 'evaluation_seed',
                'paraphrase_bank', 'paraphrase_bank_sha256', 'training_variants_per_task'):
        if config[key] != reference[key]:
            raise ValueError(f'Preserve passing tiny recipe: {key}')
    if (config['run_type'] != 'full_m1_qwen17_mixed_v1' or config['optimizer_steps'] != 1200
            or config['checkpoint_attempts'] != [300, 600, 900, 1200]
            or config['primary_research_mode'] != 'thinking' or config['evaluate_modes'] != list(MODES)
            or config['expected_entity_counts'] != {'F': 30, 'R': 90}
            or config['per_task_gates'] != {'one_hop_accuracy_min': 0.9, 'multi_hop_accuracy_min': 0.75}
            or config['baseline_policy'] != 'verified_corrected_m0_with_fresh_replay_controls'):
        raise ValueError('Use the prospective full120-unit/1200-attempt protocol, both modes and fixed gates')
    if config['checkpoint_selection'] != 'Only the endpoint at1200 attempts is eligible; no best-dev/early-stop selection.':
        raise ValueError('No checkpoint selection using development scores')
    if config['persistence'] != {'repo_id': 'tharundevulapalli/persistent-llm-unlearning-checkpoints',
            'path_prefix': 'shared/m1-full-qwen17-v1', 'require_private': True, 'checkpoint_roundtrip_required': True}:
        raise ValueError('Keep the declared private durable checkpoint policy')
    if sha256_file(Path(config['data_review'])) != config['data_review_sha256']:
        raise ValueError('Frozen scoped data review changed')
    review = read(config['data_review'])
    if (not review['review_passed'] or review['bundle'] != config['bundle']
            or review['checksum_index_sha256'] != config['expected_checksum_index_sha256']):
        raise ValueError('Candidate differs from reviewed v1.1')
    baseline = Path(config['baseline_dir'])
    if sha256_file(baseline / 'local_verification.json') != config['baseline_verification_sha256']:
        raise ValueError('Corrected baseline verification changed')
    verified = read(baseline / 'local_verification.json')
    if not verified['local_verification_passed'] or verified['offline_tokenizer_records_replayed'] != 2600:
        raise ValueError('Require verified corrected baseline and offline tokenizer evidence')
    for name, digest in verified['original_files_sha256'].items():
        if sha256_file(baseline / name) != digest:
            raise ValueError(f'Corrected baseline file changed: {name}')
    old_config = read(baseline / 'plan.json')['config']
    if old_config['bundle'] != config['bundle'] or old_config['expected_checksum_index_sha256'] != config['expected_checksum_index_sha256']:
        raise ValueError('Baseline is for another candidate')
    if not verify_output(old_config, baseline)['execution_complete']:
        raise ValueError('Corrected baseline is incomplete')
    if sha256_file(Path(config['utility_reference'])) != config['utility_reference_sha256']:
        raise ValueError('Own-model M0 utility reference changed')
    all_m1 = read_rows(Path(config['bundle']) / 'entities/all_m1.jsonl')
    if config['record_ids'] != [r['record_id'] for r in all_m1] or len(all_m1) != 120:
        raise ValueError('Select all and only the120 F/R units in canonical order')
    train, dev, audit = augmented_examples(config, expected_units=120)
    reserved = {normalize_text(r['question']) for path in (Path(config['bundle']) / 'prompts').glob('*.jsonl')
                if path.stem != 'm1_train' for r in read_rows(path)}
    if {normalize_text(r['question']) for r in train} & reserved:
        raise ValueError('Training question overlaps a reserved prompt view')
    if len(train) != 5040 or len(dev) != 840 or Counter(r['training_mode'] for r in train) != {'answer_only': 2520, 'thinking': 2520}:
        raise ValueError('Incorrect full training/development coverage')
    baseline_rows = {mode: [r for r in read_rows(baseline / f'{mode}_predictions.jsonl') if r['prompt_partition'] == 'm1_dev'] for mode in MODES}
    for rows in baseline_rows.values():
        if len(rows) != 840 or any(any(row.get(k) != v for k, v in example.items()) for row, example in zip(rows, dev)):
            raise ValueError('M0 baseline dev input order/content differ')
    original_utility = read_rows(Path(config['utility_reference']))
    if len(original_utility) != 10 or any(any(row.get(k) != v for k, v in example.items()) for row, example in zip(original_utility, utility_examples())):
        raise ValueError('M0 utility probe input differs')
    plan = {'config': config, 'training_rows': 5040, 'unique_training_questions': 2520, 'development_rows': 840,
            'training_rows_by_mode': audit['rows_by_mode'], 'training_rows_by_partition': audit['rows_by_partition'],
            'augmentation': audit, 'planned_update_attempts': 1200, 'required_applied_updates': 1200,
            'nominal_examples_exposed': 19200, 'nominal_dataset_passes': 19200 / 5040,
            'attempt_budget_rationale': '10x rows and10x attempts vs tiny pilot; same nominal exposure per row, not a guarantee of scaled learning.',
            'checkpoint_attempts': config['checkpoint_attempts'], 'new_base_control_answers': 32,
            'new_base_utility_answers': 10, 'reused_base_dev_records': 1680,
            'post_answers': 1690, 'reload_answers': 1690, 'planned_new_evaluation_answers': 3422,
            'baseline_versions': read(baseline / 'summary.json')['versions'], 'baseline_gpu': 'Tesla T4',
            'externally_flagged_record_ids': read(baseline / 'plan.json')['externally_flagged_record_ids'],
            'initialization': 'Fresh pinned public1.7B base/new LoRA; no tiny adapter restoration.',
            'checkpoint_selection': config['checkpoint_selection'], 'model_executed': False, 'canonical_m1': False,
            'limits': 'One seed; supervised composite chains and dev wording checks, not unseen-composition proof. Ten reused arithmetic probes are diagnostic, not independent broad reasoning evidence. Final model tests remain held out.'}
    return plan, train, dev, baseline_rows


def same_predictions(left, right):
    ignored = {'batch_seconds', 'evidence_origin'}
    return len(left) == len(right) and all(
        {k: v for k, v in a.items() if k not in ignored} == {k: v for k, v in b.items() if k not in ignored}
        for a, b in zip(left, right))


def full_verdict(config, post, base_utility, reload_matches, changed, updates, receipts):
    if (set(post) != {'dev_answer_only', 'dev_thinking', 'utility_thinking'} or len(base_utility) != 10
            or len(post.get('utility_thinking', [])) != 10):
        return {'passed': False, 'checks': {'coverage': False}}
    expected = read_rows(Path(config['bundle']) / 'prompts/m1_dev.jsonl')
    for mode in MODES:
        rows = post['dev_' + mode]
        if len(rows) != 840 or any(any(row.get(k) != v for k, v in example.items()) for row, example in zip(rows, expected)):
            return {'passed': False, 'checks': {'coverage': False}}
    verdict = candidate_verdict(config, post, base_utility, reload_matches, changed, updates)
    for result in verdict['by_fact_mode'].values():
        result['scope'] = 'Original family thresholds applied prospectively to full development rows; separate per-task full gates also required.'
    checks = verdict['checks']
    per_task = {}
    for mode in MODES:
        for partition in ('F', 'R'):
            for task, spec in TASKS.items():
                rows = [r for r in post['dev_' + mode] if r['entity_partition'] == partition and r['task_type'] == task]
                accuracy = sum(r['answer_match'] for r in rows) / len(rows)
                threshold = config['per_task_gates']['one_hop_accuracy_min' if spec['hop_count'] == 1 else 'multi_hop_accuracy_min']
                key = mode + '/' + partition + '/' + task
                per_task[key] = {'correct': sum(r['answer_match'] for r in rows), 'count': len(rows), 'accuracy': accuracy, 'minimum': threshold}
                checks['per_task/' + key] = accuracy >= threshold
    attempts = [r['attempt'] for r in receipts]
    checks['periodic_private_roundtrip'] = (attempts == config['checkpoint_attempts'] and
                                          all(r['all_artifact_files_verified'] for r in receipts))
    verdict.update(passed=all(checks.values()), per_task=per_task,
                   scope='Full-M1 candidate development gates. Canonical promotion still requires independent evidence/private-persistence review; not final-test or safeguard success.')
    return verdict


def generate_facts(model, tokenizer, rows, config, plan, mode, output):
    import torch
    from transformers import GenerationConfig, set_seed
    thinking = mode == 'thinking'
    model.eval()
    generation = GenerationConfig(**(SAMPLING if thinking else {'do_sample': False}),
        max_new_tokens=config['sampled_max_new_tokens'] if thinking else config['evaluation']['max_new_tokens'],
        eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    predictions = []
    with output.open('x') as handle:
        for offset in range(0, len(rows), config['evaluation']['batch_size']):
            batch = rows[offset:offset + config['evaluation']['batch_size']]
            prompts = [tokenizer.apply_chat_template(evaluation_messages(r['question']), tokenize=False,
                       add_generation_prompt=True, enable_thinking=thinking) for r in batch]
            inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda:0')
            set_seed(config['evaluation_seed'] + offset)
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=generation)[:, inputs['input_ids'].shape[1]:].tolist()
            for example, prompt, tokens in zip(batch, prompts, generated):
                text, saved, details = decode_generation(tokenizer, tokens, generation.eos_token_id, generation.max_new_tokens)
                row = score_audit_row(example, text, thinking, prompt, set(plan['externally_flagged_record_ids']))
                row.update(details)
                row.update(mode=mode, thinking_enabled=thinking, rendered_prompt=prompt,
                           batch_seed=config['evaluation_seed'] + offset, generated_token_ids=saved)
                row['protocol_issue'] = bool(not row['format_compliant'] or row['hit_token_limit'] or row.get('boundary_error'))
                row['needs_match_review'] = bool(row['potential_answer_review'] or row['external_surface_review_flag'] or row['protocol_issue'])
                predictions.append(row)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            handle.flush()
            print(f'{output.name}: {len(predictions)}/{len(rows)}', flush=True)
    return predictions


def evaluation_stage(model, tokenizer, dev, config, plan, output, stage):
    result = {'dev_' + mode: generate_facts(model, tokenizer, dev, config, plan, mode,
               output / f'{stage}_dev_{mode}.jsonl') for mode in MODES}
    result['utility_thinking'] = sampled_evaluation(model, tokenizer, utility_examples(), True,
        {'batch_size': 16, 'seed': 42, 'sampled_max_new_tokens': 1024}, output / f'{stage}_utility_thinking.jsonl')
    return result


def run_training(config, plan, train, dev, baseline_rows, encoded, tokenizer, output, fit_dir, local_files_only):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from peft.import_utils import is_torchao_available
    from transformers import AutoModelForCausalLM, set_seed

    is_torchao_available()
    if not torch.cuda.is_available():
        raise RuntimeError('Use Kaggle CUDA for full M1')
    versions = {p: package_version(p) for p in plan['baseline_versions']}
    if versions != plan['baseline_versions'] or torch.cuda.get_device_name(0) != plan['baseline_gpu']:
        raise RuntimeError('Baseline replay requires original T4/package versions; no silent fallback')
    fit = read(fit_dir / 'fit_report.json')
    fit_plan = read(fit_dir / 'plan.json')
    if (fit_plan['config'] != config or fit_plan['config_sha256'] != plan['config_sha256']
            or fit_plan['source_sha256'] != plan['source_sha256']
            or not fit['success'] or fit['temporary_optimizer_updates'] != 1
            or fit['checkpoint_saved'] or fit['model_id'] != config['model_id'] or fit['model_revision'] != config['model_revision']):
        raise ValueError('Require successful separate disposable fit on this exact config')
    if (fit['probe_max_training_tokens'] != max(len(row['input_ids']) for row in encoded)
            or fit['probe_training_batch_size'] != config['batch_size']):
        raise ValueError('Fit did not cover the actual longest full-training examples')
    from shutil import copyfile
    copyfile(fit_dir / 'fit_report.json', output / 'memory_fit_report.json')
    copyfile(fit_dir / 'plan.json', output / 'memory_fit_plan.json')
    token = os.environ.get('HF_TOKEN')
    from huggingface_hub import HfApi
    if not token or not HfApi(token=token).repo_info(config['persistence']['repo_id'], repo_type='model').private:
        raise RuntimeError('Private HF_TOKEN access required before training')
    remote = config['persistence']['path_prefix'] + '/' + output.name
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    (output / 'environment.freeze.txt').write_text('\n'.join(installed_packages()) + '\n')
    def load_base():
        return AutoModelForCausalLM.from_pretrained(config['model_id'], revision=config['model_revision'],
            dtype=torch.float16, token=token, local_files_only=local_files_only).to('cuda:0').eval()
    set_seed(config['seed'])
    base = load_base()
    metadata = {'eos_token_id': base.generation_config.eos_token_id, 'pad_token_id': tokenizer.pad_token_id,
                'think_open_token_id': tokenizer.convert_tokens_to_ids('<think>'),
                'think_close_token_id': tokenizer.convert_tokens_to_ids('</think>'),
                'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
    if metadata != read(Path(config['baseline_dir']) / 'generation_metadata.json'):
        raise RuntimeError('Baseline generation metadata differ')
    write_json(output / 'generation_metadata.json', metadata)
    controls = {mode: generate_facts(base, tokenizer, dev[:16], config, plan, mode,
                output / f'base_controls_{mode}.jsonl') for mode in MODES}
    utility = sampled_evaluation(base, tokenizer, utility_examples(), True,
        {'batch_size': 16, 'seed': 42, 'sampled_max_new_tokens': 1024}, output / 'base_utility_thinking.jsonl')
    guard = {'facts_reproduced': all(same_predictions(controls[mode], baseline_rows[mode][:16]) for mode in MODES),
             'utility_reproduced': same_predictions(utility, read_rows(Path(config['utility_reference']))),
             'utility_measurable': sum(r['answer_match'] for r in utility) / 10 >= config['pilot_gates']['utility_base_accuracy_min'],
             'utility_complete': not any(r['hit_token_limit'] or r.get('boundary_error') for r in utility)}
    write_json(output / 'base_guard.json', {'passed': all(guard.values()), 'checks': guard})
    if not all(guard.values()):
        write_json(output / 'summary.json', {'execution_complete': False, 'training_executed': False, 'optimizer_updates': 0,
            'canonical_m1': False, 'plan': plan, 'stop_reason': 'Fresh base replay/utility guard failed before adapter creation.', 'base_guard': guard})
        return 2
    for mode in MODES:
        write_rows(output / f'base_dev_{mode}.jsonl', baseline_rows[mode])
    write_json(output / 'base_provenance.json', {'kind': 'Reused verified M0 development records with fresh exact32-answer replay control.',
        'baseline_dir': config['baseline_dir'], 'baseline_verification_sha256': config['baseline_verification_sha256'],
        'reused_dev_records': 1680, 'new_fact_controls': 32, 'new_utility_records': 10})
    set_seed(config['seed'])
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', **config['lora']))
    model.peft_config['default'].revision = config['model_revision']
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not parameters or any('lora_' not in name or p.dtype != torch.float32 for name, p in parameters):
        raise RuntimeError('Expected FP32 LoRA parameters only')
    initial = {name: p.detach().cpu().clone() for name, p in parameters}
    count = sum(p.numel() for _, p in parameters)
    receipts = []
    tokenizer.save_pretrained(output / 'tokenizer')
    def checkpoint(model, optimizer, scaler, record, exposures):
        attempt = record['attempt']
        if attempt not in config['checkpoint_attempts']:
            return
        directory = output / 'checkpoints' / f'attempt-{attempt:06d}'
        directory.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(directory / 'adapter', safe_serialization=True)
        tokenizer.save_pretrained(directory / 'tokenizer')
        torch.save({'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                    'attempt': attempt, 'optimizer_updates': record['optimizer_updates'], 'seed': config['seed']}, directory / 'optimizer_state.pt')
        write_json(directory / 'checkpoint.json', {'attempt': attempt, 'optimizer_updates': record['optimizer_updates'],
            'config_sha256': plan['config_sha256'], 'exposures_by_mode': deepcopy(exposures),
            'model_id': config['model_id'], 'model_revision': config['model_revision'], 'canonical_m1': False,
            'exact_resume_supported': False})
        # Preserve the complete training prefix, configuration and tokenizer masks with each checkpoint.
        from shutil import copyfile
        for name in ('training.jsonl', 'plan.json', 'masking_report.json', 'augmentation_audit.json', 'environment.freeze.txt'):
            copyfile(output / name, directory / name)
        write_json(directory / 'artifact_manifest.json', {'artifact_files_sha256': artifact_hashes(directory),
            'attempt': attempt, 'optimizer_updates': record['optimizer_updates'], 'canonical_m1': False})
        receipt = publish_verified(directory, config['persistence'], remote + '/checkpoints/' + directory.name, token)
        receipt['attempt'] = attempt
        receipts.append(receipt)
        write_json(directory / 'persistence.json', receipt)
        with (output / 'checkpoint_receipts.jsonl').open('a') as handle:
            handle.write(json.dumps(receipt, sort_keys=True) + '\n')
        print(f'Checkpoint{attempt}: private immutable round-trip verified.', flush=True)
    torch.cuda.reset_peak_memory_stats(0)
    set_seed(config['seed'])
    training = train_updates(model, encoded, tokenizer, config, output, checkpoint_callback=checkpoint)
    changed = any(not torch.equal(initial[name], p.detach().cpu()) for name, p in parameters)
    model.save_pretrained(output / 'adapter', safe_serialization=True)
    write_json(output / 'artifact_manifest.json', {'artifact_files_sha256': artifact_hashes(output),
        'training_complete': True, 'optimizer_updates': training['optimizer_updates'], 'canonical_m1': False})
    post = evaluation_stage(model, tokenizer, dev, config, plan, output, 'post')
    del parameters, initial, model, base
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(load_base(), output / 'adapter', is_trainable=False).eval()
    restored = evaluation_stage(reloaded, tokenizer, dev, config, plan, output, 'reloaded')
    reload_matches = all(same_predictions(post[key], restored[key]) for key in post)
    verdict = full_verdict(config, post, utility, reload_matches, changed, training['optimizer_updates'], receipts)
    summary = {'execution_complete': True, 'training_executed': True, 'canonical_m1': False, 'plan': plan,
        'm1_candidate': verdict, 'training': training, 'optimizer_updates': training['optimizer_updates'],
        'adapter_changed': changed, 'trainable_parameters': count, 'reload_predictions_identical': reload_matches,
        'checkpoint_attempts_privately_verified': [r['attempt'] for r in receipts],
        'evaluation': {'base': {key: metrics(rows) for key, rows in {'dev_answer_only': baseline_rows['answer_only'],
                       'dev_thinking': baseline_rows['thinking'], 'utility_thinking': utility}.items()},
                       'post': {key: metrics(rows) for key, rows in post.items()},
                       'reloaded': {key: metrics(rows) for key, rows in restored.items()}},
        'artifact_files_sha256': artifact_hashes(output),
        'artifact_persistence': 'Four checkpoint round-trips verified; endpoint/evaluation folder needs final notebook backup receipt.',
        'prediction_files_sha256': {p.name: sha256_file(p) for p in output.glob('*.jsonl') if p.name.startswith(('base_', 'post_', 'reloaded_'))},
        'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'seconds': round(time.perf_counter() - clock, 4), 'gpu': torch.cuda.get_device_name(0),
        'peak_gpu_memory_gib': torch.cuda.max_memory_allocated(0) / 1024**3,
        'versions': {p: package_version(p) for p in ('torch', 'transformers', 'peft', 'accelerate')}}
    write_json(output / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key not in ('plan', 'evaluation')}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('plan', 'prepare', 'fit', 'train'), default='plan')
    parser.add_argument('--fit-dir', type=Path)
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--tokenizer-cache', type=Path)
    args = parser.parse_args()
    config = read(args.config)
    plan, train, dev, baseline_rows = prepare_plan(config)
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    sources = ['src/training/full_m1.py', 'src/training/full_artifacts.py', 'src/training/mixed_m1.py',
        'src/training/paraphrase_data.py', 'src/training/mixed_data.py', 'src/training/data.py',
        'src/training/tiny_m1.py', 'src/training/model_capacity.py', 'src/evaluation/answers.py',
        'src/evaluation/mode_diagnostic.py', 'src/evaluation/full_m0.py', 'src/evaluation/corrected_m0.py',
        'src/evaluation/utility_diagnostic.py', config['paraphrase_bank']]
    plan.update(config_sha256=sha256_file(args.config), source_git_commit=git_commit(),
                source_sha256={name: sha256_file(Path(name)) for name in sources})
    write_json(output / 'plan.json', plan)
    write_json(output / 'augmentation_audit.json', plan['augmentation'])
    write_rows(output / 'selected_train.jsonl', train)
    write_rows(output / 'selected_dev.jsonl', dev)
    if args.mode == 'plan':
        print(json.dumps(plan, indent=2))
        return 0
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'],
        padding_side='left', token=os.environ.get('HF_TOKEN'), local_files_only=args.local_files_only,
        cache_dir=args.tokenizer_cache)
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
        return 0 if report['success'] else 2
    if args.fit_dir is None:
        raise ValueError('Full training requires the separate completed --fit-dir')
    try:
        return run_training(config, plan, train, dev, baseline_rows, encoded, tokenizer, output, args.fit_dir, args.local_files_only)
    except Exception as error:
        # Save a concise failure record; never serialize the environment or credential.
        message = str(error)
        token = os.environ.get('HF_TOKEN')
        if token:
            message = message.replace(token, '[redacted]')
        write_json(output / 'failure.json', {'execution_complete': False, 'error_type': type(error).__name__,
            'message': message, 'canonical_m1': False, 'action': 'Preserve checkpoint receipts/partial evidence; no automatic retry, resume or budget extension.'})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
