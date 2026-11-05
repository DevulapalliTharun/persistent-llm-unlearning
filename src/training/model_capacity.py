"""Fixed same-recipe Qwen3-1.7B trial and disposable GPU memory probe."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable

from src.evaluation.answers import SYSTEM_PROMPT, generation_details
from src.evaluation.mode_diagnostic import SAMPLING
from src.evaluation.utility_diagnostic import ARITHMETIC_PROMPT
from src.m0_knowledge_audit import sha256_file
from src.training.data import pad_batch, utility_examples


MODEL_ID = 'Qwen/Qwen3-1.7B'
MODEL_REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
RUN_TYPE = 'diagnostic_m1_qwen17_paraphrase_v1'


def prepare_capacity_plan(config: dict[str, Any], original_prepare: Callable):
    if config['model_id'] != MODEL_ID or config['model_revision'] != MODEL_REVISION:
        raise ValueError('Use the pinned Qwen3-1.7B comparison model')
    if config['baseline_policy'] != 'measure_new_model_m0' or config['require_base_reference_reproduction'] is not False:
        raise ValueError('Measure this model baseline; never reuse 0.6B response expectations')
    if 'base_utility_reference' in config or 'base_utility_reference_sha256' in config:
        raise ValueError('A different-model baseline must not reuse 0.6B reference responses')
    reference_path = Path(config['reference_model_config'])
    if sha256_file(reference_path) != config['reference_model_config_sha256']:
        raise ValueError('The 0.6B recipe/config control changed')
    reference = json.loads(reference_path.read_text())
    exceptions = {'run_type', 'model_id', 'model_revision', 'base_utility_reference',
                  'base_utility_reference_sha256', 'require_base_reference_reproduction'}
    for key, value in reference.items():
        if key not in exceptions and config[key] != value:
            raise ValueError(f'Keep the same data/optimizer/evaluation/gates as the 0.6B control: {key}')
    metadata_path = Path(config['model_metadata'])
    if sha256_file(metadata_path) != config['model_metadata_sha256']:
        raise ValueError('Pinned model metadata changed')
    metadata = json.loads(metadata_path.read_text())
    if metadata['model_id'] != MODEL_ID or metadata['model_revision'] != MODEL_REVISION:
        raise ValueError('Model metadata pin differs')
    control_path = Path(config['control_summary'])
    if sha256_file(control_path) != config['control_summary_sha256']:
        raise ValueError('Preserved 0.6B result control changed')
    control = json.loads(control_path.read_text())
    if control['plan']['config'] != reference or not control['execution_complete']:
        raise ValueError('Control is not the verified completed same-recipe 0.6B run')
    plan, train, dev = original_prepare(reference)
    plan.update(config=config, model_metadata=metadata,
                baseline_policy='Own fresh 1.7B M0 development/utility measurements before adapter training.',
                control_model={'model_id': reference['model_id'], 'model_revision': reference['model_revision'],
                               'summary_sha256': config['control_summary_sha256']},
                comparison_limit='Same 504 rows, rank and 120 attempted updates; different weights/model/LoRA parameter counts, possible tokenization differences and compute. One-seed feasibility comparison, not proof that size alone caused outcomes.')
    return plan, train, dev


def memory_fit_probe(config, tokenizer, encoded, dev, local_files_only=False):
    """One disposable optimizer step and representative generation, not M1 learning."""
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.import_utils import is_torchao_available
    from transformers import AutoModelForCausalLM, GenerationConfig, set_seed

    is_torchao_available()
    if not torch.cuda.is_available():
        raise RuntimeError('Run the memory probe on the Kaggle CUDA kernel')
    import os
    token = os.environ.get('HF_TOKEN')
    started = time.perf_counter()
    report = {'purpose': 'Disposable memory feasibility only; not an M1 result.',
              'model_id': config['model_id'], 'model_revision': config['model_revision'],
              'success': False, 'temporary_optimizer_updates': 0, 'checkpoint_saved': False,
              'actual_training_starts_fresh_in_a_separate_process': True,
              'gpu': torch.cuda.get_device_name(0),
              'gpu_total_memory_gib': torch.cuda.get_device_properties(0).total_memory / 1024**3,
              'limits': 'Representative actual-data batch/generation probe, not a guarantee against worst-case full-run OOM. No learning metrics are reported.'}
    try:
        set_seed(config['seed'])
        base = AutoModelForCausalLM.from_pretrained(config['model_id'], revision=config['model_revision'],
            dtype=torch.float16, token=token, local_files_only=local_files_only).to('cuda:0')
        set_seed(config['seed'])
        model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', **config['lora']))
        parameters = [p for n, p in model.named_parameters() if p.requires_grad]
        if not parameters or any(p.dtype != torch.float32 for p in parameters):
            raise RuntimeError('Expected FP32 LoRA probe parameters')
        report['trainable_parameters'] = sum(p.numel() for p in parameters)
        longest = sorted(encoded, key=lambda row: len(row['input_ids']), reverse=True)[:config['batch_size']]
        batch = {key: torch.tensor(value, device='cuda:0', dtype=torch.long)
                 for key, value in pad_batch(longest, tokenizer.pad_token_id).items()}
        labels = batch.pop('labels')
        report['probe_training_batch_size'] = len(longest)
        report['probe_max_training_tokens'] = max(len(r['input_ids']) for r in longest)
        optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=0.0)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024)
        model.train()
        torch.cuda.reset_peak_memory_stats(0)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config['gradient_accumulation_steps']):
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(**batch, use_cache=False).logits
                target = labels[:, 1:]
                token_loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().transpose(1, 2),
                    target, ignore_index=-100, reduction='none')
                loss = (token_loss.sum(1) / target.ne(-100).sum(1)).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite provisional memory-probe loss')
            scaler.scale(loss / config['gradient_accumulation_steps']).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, config['max_grad_norm'])
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        report['temporary_optimizer_updates'] = int(scaler.get_scale() >= scale)
        report['optimizer_state_populated'] = bool(optimizer.state)
        report['training_peak_memory_gib'] = torch.cuda.max_memory_allocated(0) / 1024**3
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        examples = dev[:8] + utility_examples()[:8]
        prompts = [tokenizer.apply_chat_template([
            {'role': 'system', 'content': ARITHMETIC_PROMPT if row['entity_partition'] == 'utility' else SYSTEM_PROMPT},
            {'role': 'user', 'content': row['question']}], tokenize=False,
            add_generation_prompt=True, enable_thinking=True) for row in examples]
        inputs = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda:0')
        generation = GenerationConfig(**SAMPLING, max_new_tokens=config['sampled_max_new_tokens'],
            eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        set_seed(config['evaluation_seed'])
        torch.cuda.reset_peak_memory_stats(0)
        with torch.inference_mode():
            ids = model.generate(**inputs, generation_config=generation)[:, inputs['input_ids'].shape[1]:].tolist()
        counts = [generation_details(row, generation.eos_token_id, generation.max_new_tokens)['generated_token_count'] for row in ids]
        report.update(probe_generation_batch_size=len(examples), generation_max_new_tokens=1024,
                      observed_generated_token_counts=counts,
                      generation_peak_memory_gib=torch.cuda.max_memory_allocated(0) / 1024**3)
        report['success'] = report['temporary_optimizer_updates'] == 1 and report['optimizer_state_populated']
    except torch.cuda.OutOfMemoryError as error:
        report.update(success=False, error_type='CUDAOutOfMemoryError', error_message=str(error),
                      action='Stop; do not silently change batching, precision or quantization in the comparison.')
    report['seconds'] = round(time.perf_counter() - started, 4)
    return report
