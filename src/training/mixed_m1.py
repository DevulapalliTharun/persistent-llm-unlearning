"""Bounded, fresh-base tiny M1 retry with answer-only and constructed fact-chain SFT."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from src.evaluation.mode_diagnostic import SAMPLING, metrics, sampled_evaluation
from src.evaluation.utility_diagnostic import compare_reference
from src.m0_knowledge_audit import sha256_file
from src.m0_smoke_test import git_commit, installed_packages, package_version, write_json
from src.training.data import pad_batch, read_rows, select_pilot, utility_examples
from src.training.mixed_data import encode_mixed, masking_report, mixed_examples
from src.training.tiny_m1 import evaluate, pilot_verdict, write_rows


FACT_MODES = ('dev_answer_only', 'dev_thinking')


def prepare_plan(config: dict[str, Any]):
    if config['run_type'] == 'diagnostic_m1_tiny_paraphrase_v1':
        from src.training.paraphrase_data import prepare_augmented_plan
        return prepare_augmented_plan(config, prepare_plan)
    original = json.loads(Path('configs/training/m1_tiny.json').read_text())
    fixed = ['bundle', 'expected_checksum_index_sha256', 'record_ids', 'expected_entity_counts',
             'model_id', 'model_revision', 'seed', 'max_sequence_length', 'batch_size',
             'gradient_accumulation_steps', 'optimizer_steps', 'learning_rate', 'max_grad_norm', 'lora', 'evaluation', 'protocol_version']
    if any(config[k] != original[k] for k in fixed):
        raise ValueError('Keep the original tiny dataset, model and optimizer/LoRA settings')
    if (config['run_type'] != 'diagnostic_m1_tiny_mixed_v1' or
            config['training_representation'] != 'answer_only_plus_canonical_chain_v1' or
            config['sampled_max_new_tokens'] != 1024 or config['evaluation_seed'] != 42 or
            config['require_base_reference_reproduction'] is not True):
        raise ValueError('Use the fixed mixed-mode diagnostic protocol')
    if config['pilot_gates'] != {**original['pilot_gates'], 'max_missing_thinking_boundaries': 0}:
        raise ValueError('Do not weaken the prespecified pilot gates')
    if sha256_file(Path(config['base_utility_reference'])) != config['base_utility_reference_sha256']:
        raise ValueError('Original thinking utility reference changed')
    train, dev = mixed_examples(config)
    if len(train) != 168 or Counter(r['training_mode'] for r in train) != {'answer_only': 84, 'thinking': 84}:
        raise ValueError('Use exactly 84 examples per training mode')
    plan = {'config': config, 'training_rows': len(train), 'development_rows': len(dev),
            'training_rows_by_mode': dict(Counter(r['training_mode'] for r in train)),
            'utility_rows': 10, 'planned_full_evaluation_predictions': 534,
            'initialization': 'Fresh pinned base and newly initialized LoRA; no prior adapter loaded.',
            'rationale_source': 'Deterministic canonical training facts; not model-generated or observed internal reasoning.',
            'evaluation': {'facts_answer_only': original['evaluation'],
                           'facts_and_utility_thinking': {'sampling': SAMPLING, 'max_new_tokens': 1024,
                                                         'batch_size': 16, 'seed': 42}},
            'model_executed': False, 'canonical_m1': False,
            'scope': 'Versioned tiny diagnostic. Original answer-only pilot and its failed gates remain unchanged.',
            'comparison_limit': 'Same attempted updates but 168 rows and longer supervised responses; not equal token/compute exposure or an isolated causal ablation.'}
    return plan, train, dev


def candidate_verdict(config, post, base_utility, reload_matches, changed, updates):
    by_mode = {mode: pilot_verdict(config, post[mode], base_utility, post['utility_thinking'],
                                  reload_matches, changed, updates) for mode in FACT_MODES}
    checks = {mode + '/' + name: passed for mode, verdict in by_mode.items()
              for name, passed in verdict['checks'].items()}
    thinking_rows = post['dev_thinking'] + post['utility_thinking']
    checks['thinking_boundaries'] = sum(r.get('boundary_error') is not None for r in thinking_rows) <= config['pilot_gates']['max_missing_thinking_boundaries']
    return {'passed': all(checks.values()), 'checks': checks, 'by_fact_mode': by_mode,
            'scope': 'Mixed-mode tiny diagnostic, not canonical M1 or a replacement for the original pilot.'}


def evaluation_stage(model, tokenizer, dev, config, output_dir, stage, baseline_guard=False):
    sampling = {'batch_size': config['evaluation']['batch_size'], 'seed': config['evaluation_seed'],
                'sampled_max_new_tokens': config['sampled_max_new_tokens']}
    utility = sampled_evaluation(model, tokenizer, utility_examples(), True, sampling,
                                output_dir / f'{stage}_utility_thinking.jsonl')
    output = {'utility_thinking': utility}
    if baseline_guard:
        control = compare_reference(utility, read_rows(Path(config['base_utility_reference'])))
        ready = (sum(r['answer_match'] for r in utility) / len(utility) >= config['pilot_gates']['utility_base_accuracy_min']
                 and control['raw_predictions_identical'] and
                 not any(r['hit_token_limit'] or r.get('boundary_error') for r in utility))
        write_json(output_dir / 'base_utility_gate.json', {'passed': ready, 'reference_reproduction': control,
                    'accuracy': sum(r['answer_match'] for r in utility) / len(utility),
                    'format_violations': sum(not r['format_compliant'] for r in utility),
                    'interpretation': 'Bare exact values are accepted with flags under the unchanged parser; the base-format flags are reported, not used to redefine its existing score.'})
        if not ready:
            return output
    output['dev_answer_only'] = evaluate(model, tokenizer, dev, config)
    write_rows(output_dir / f'{stage}_dev_answer_only.jsonl', output['dev_answer_only'])
    output['dev_thinking'] = sampled_evaluation(model, tokenizer, dev, True, sampling,
                                               output_dir / f'{stage}_dev_thinking.jsonl')
    if config['run_type'] == 'diagnostic_m1_tiny_paraphrase_v1' and stage != 'base':
        # Diagnose original training-wording recall in the same run; never use it as held-out accuracy.
        original_train, _ = select_pilot(config)
        output['train_probe_answer_only'] = evaluate(model, tokenizer, original_train, config)
        write_rows(output_dir / f'{stage}_train_probe_answer_only.jsonl', output['train_probe_answer_only'])
        output['train_probe_thinking'] = sampled_evaluation(model, tokenizer, original_train, True, sampling,
            output_dir / f'{stage}_train_probe_thinking.jsonl')
    return output


def train_updates(model, encoded, tokenizer, config, output_dir):
    import torch

    parameters = [p for p in model.parameters() if p.requires_grad]
    def collate(rows):
        tensors = {key: torch.tensor(value, dtype=torch.long, device='cuda:0')
                   for key, value in pad_batch(rows, tokenizer.pad_token_id).items()}
        return tensors, [{'mode': r['training_mode'], 'input_tokens': len(r['input_ids']),
                          'response_tokens': r['answer_tokens']} for r in rows]
    loader = torch.utils.data.DataLoader(encoded, batch_size=config['batch_size'], shuffle=True,
        generator=torch.Generator().manual_seed(config['seed']), collate_fn=collate, num_workers=0)
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=0.0)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    updates = 0
    exposures = {mode: {'examples': 0, 'input_tokens': 0, 'response_tokens': 0}
                 for mode in ('answer_only', 'thinking')}
    started = time.perf_counter()
    model.train()
    model.config.use_cache = False
    with (output_dir / 'training.jsonl').open('x') as log:
        for attempt in range(config['optimizer_steps']):
            optimizer.zero_grad(set_to_none=True)
            losses, by_mode = [], {mode: [] for mode in exposures}
            for _ in range(config['gradient_accumulation_steps']):
                try:
                    batch, batch_meta = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch, batch_meta = next(iterator)
                labels = batch.pop('labels')
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(**batch, use_cache=False).logits
                    target = labels[:, 1:]
                    tokens = torch.nn.functional.cross_entropy(logits[:, :-1].float().transpose(1, 2),
                        target, ignore_index=-100, reduction='none')
                    per_example = tokens.sum(dim=1) / target.ne(-100).sum(dim=1)
                    loss = per_example.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite response loss; partial logs preserved')
                losses.append(float(loss.detach()))
                for meta, value in zip(batch_meta, per_example.detach().cpu().tolist()):
                    mode = meta['mode']
                    by_mode[mode].append(value)
                    exposures[mode]['examples'] += 1
                    exposures[mode]['input_tokens'] += meta['input_tokens']
                    exposures[mode]['response_tokens'] += meta['response_tokens']
                scaler.scale(loss / config['gradient_accumulation_steps']).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config['max_grad_norm'])
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            applied = scaler.get_scale() >= scale
            updates += int(applied)
            record = {'attempt': attempt + 1, 'optimizer_updates': updates, 'update_applied': applied,
                      'mean_response_ce': sum(losses) / len(losses),
                      'response_ce_by_mode': {mode: sum(values) / len(values) if values else None for mode, values in by_mode.items()},
                      'examples_by_mode': {mode: len(values) for mode, values in by_mode.items()},
                      'grad_norm': float(norm) if math.isfinite(float(norm)) else None}
            log.write(json.dumps(record) + '\n')
            log.flush()
            if (attempt + 1) % 10 == 0:
                print(f"Mixed training {attempt + 1}/120; response CE={record['mean_response_ce']:.4f}", flush=True)
    torch.save({'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'optimizer_updates': updates, 'seed': config['seed']}, output_dir / 'optimizer_state.pt')
    return {'optimizer_updates': updates, 'exposures_by_mode': exposures,
            'training_seconds': round(time.perf_counter() - started, 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('plan', 'prepare', 'train'), default='plan')
    parser.add_argument('--local-files-only', action='store_true', help='Use already-cached tokenizer/model files only')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    plan, train, dev = prepare_plan(config)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a fresh output directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = ['src/training/mixed_m1.py', 'src/training/mixed_data.py', 'src/training/tiny_m1.py',
               'src/training/data.py', 'src/evaluation/answers.py', 'src/evaluation/mode_diagnostic.py',
               'src/evaluation/utility_diagnostic.py', 'src/m0_knowledge_audit.py']
    if config['run_type'] == 'diagnostic_m1_tiny_paraphrase_v1':
        sources += ['src/training/paraphrase_data.py', config['paraphrase_bank']]
    plan.update(config_sha256=sha256_file(args.config), source_git_commit=git_commit(),
                source_sha256={name: sha256_file(Path(name)) for name in sources})
    write_json(args.output_dir / 'plan.json', plan)
    write_rows(args.output_dir / 'selected_train.jsonl', train)
    write_rows(args.output_dir / 'selected_dev.jsonl', dev)
    if 'augmentation' in plan:
        write_json(args.output_dir / 'augmentation_audit.json', plan['augmentation'])
    if args.mode == 'plan':
        print(json.dumps(plan, indent=2))
        return 0
    from transformers import AutoTokenizer
    token = os.environ.get('HF_TOKEN')
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'],
        padding_side='left', token=token, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_mixed(tokenizer, row, config['max_sequence_length']) for row in train]
    report = masking_report(encoded)
    write_json(args.output_dir / 'masking_report.json', report)
    if args.mode == 'prepare':
        print(json.dumps(report, indent=2))
        return 0
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from peft.import_utils import is_torchao_available
    from transformers import AutoModelForCausalLM, set_seed

    is_torchao_available()
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires Kaggle CUDA; use plan/prepare locally')
    (args.output_dir / 'environment.freeze.txt').write_text('\n'.join(installed_packages()) + '\n')
    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.perf_counter()
    def load_base():
        return AutoModelForCausalLM.from_pretrained(config['model_id'], revision=config['model_revision'],
            dtype=torch.float16, token=token, local_files_only=args.local_files_only).to('cuda:0')
    set_seed(config['seed'])
    base = load_base().eval()
    write_json(args.output_dir / 'generation_metadata.json', {
        'eos_token_id': base.generation_config.eos_token_id, 'pad_token_id': tokenizer.pad_token_id,
        'think_open_token_id': tokenizer.convert_tokens_to_ids('<think>'),
        'think_close_token_id': tokenizer.convert_tokens_to_ids('</think>'),
        'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        'sampling': SAMPLING, 'sampled_max_new_tokens': 1024})
    baseline = evaluation_stage(base, tokenizer, dev, config, args.output_dir, 'base', baseline_guard=True)
    if len(baseline) != 3:
        write_json(args.output_dir / 'summary.json', {'execution_complete': False, 'training_executed': False,
            'optimizer_updates': 0, 'canonical_m1': False, 'plan': plan, 'stop_reason': 'Base utility/reference gate failed before LoRA creation.',
            'base_utility_gate': json.loads((args.output_dir / 'base_utility_gate.json').read_text())})
        print('Base utility/reference gate failed. No adapter created or training performed; preserve evidence.', flush=True)
        return 2
    set_seed(config['seed'])
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', **config['lora']))
    model.peft_config['default'].revision = config['model_revision']
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not parameters or any('lora_' not in n or p.dtype != torch.float32 for n, p in parameters):
        raise RuntimeError('Expected only FP32 LoRA parameters to be trainable')
    initial = {n: p.detach().cpu().clone() for n, p in parameters}
    count = sum(p.numel() for _, p in parameters)
    torch.cuda.reset_peak_memory_stats(0)
    set_seed(config['seed'])
    training = train_updates(model, encoded, tokenizer, config, args.output_dir)
    changed = any(not torch.equal(initial[n], p.detach().cpu()) for n, p in parameters)
    adapter = args.output_dir / 'adapter'
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir / 'tokenizer')
    artifact_hashes = {str(p.relative_to(args.output_dir)): sha256_file(p) for p in sorted(adapter.rglob('*')) if p.is_file()}
    write_json(args.output_dir / 'artifact_manifest.json', {'artifact_files_sha256': artifact_hashes,
        'training_complete': True, 'canonical_m1': False, 'optimizer_updates': training['optimizer_updates']})
    post = evaluation_stage(model, tokenizer, dev, config, args.output_dir, 'post')
    del parameters, initial, model, base
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(load_base(), adapter, is_trainable=False).eval()
    restored = evaluation_stage(reloaded, tokenizer, dev, config, args.output_dir, 'reloaded')
    reload_matches = all([r['raw_response'] for r in post[key]] == [r['raw_response'] for r in restored[key]] for key in post)
    verdict = candidate_verdict(config, post, baseline['utility_thinking'], reload_matches, changed, training['optimizer_updates'])
    summary = {'execution_complete': True, 'training_executed': True, 'canonical_m1': False,
               'original_pilot_result_unchanged': True, 'plan': plan, 'pilot': verdict,
               'training': training, 'optimizer_updates': training['optimizer_updates'],
               'adapter_changed': changed, 'trainable_parameters': count,
               'reload_predictions_identical': reload_matches,
               'evaluation': {stage: {key: metrics(rows) for key, rows in values.items()}
                              for stage, values in [('base', baseline), ('post', post), ('reloaded', restored)]},
               'artifact_files_sha256': artifact_hashes, 'artifact_persistence': 'Runtime only until private upload verified.',
               'prediction_files_sha256': {p.name: sha256_file(p) for p in sorted(args.output_dir.glob('*.jsonl')) if p.name.startswith(('base_', 'post_', 'reloaded_'))},
               'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
               'seconds': round(time.perf_counter() - start_clock, 4), 'gpu': torch.cuda.get_device_name(0),
               'peak_gpu_memory_gib': torch.cuda.max_memory_allocated(0) / 1024**3,
               'versions': {p: package_version(p) for p in ('torch', 'transformers', 'peft', 'accelerate')}}
    write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('plan', 'evaluation')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
