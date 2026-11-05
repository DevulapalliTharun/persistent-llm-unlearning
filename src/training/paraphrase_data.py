"""Train-only paraphrase overlay; canonical bundle and evaluation views stay fixed."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from string import Formatter
from typing import Any, Callable

from src.data.synthetic import TASKS, normalize_text, similarity_ratio
from src.evaluation.answers import training_answer
from src.m0_knowledge_audit import sha256_file
from src.training.data import read_rows, select_pilot
from src.training.mixed_data import canonical_chain


def validate_bank(bank: dict[str, Any]) -> dict[str, Any]:
    if (bank['version'] != 'tiny_m1_train_paraphrases_v1' or
            bank['additional_templates_per_task'] != 2 or
            bank['reserved_template_similarity_max'] != 0.92 or
            set(bank['templates']) != set(TASKS)):
        raise ValueError('Use the fixed seven-task train-only paraphrase bank')
    reserved = [template for task in TASKS.values() for template in task['templates']]
    new_templates, audit = [], []
    for task, templates in bank['templates'].items():
        if len(templates) != 2:
            raise ValueError('Require two additional training templates per task')
        for index, template in enumerate(templates, 1):
            fields = {field for _, field, _, _ in Formatter().parse(template) if field is not None}
            if fields != {TASKS[task]['source']}:
                raise ValueError('Paraphrase questions must contain only the task source entity')
            closest = max(similarity_ratio(template, old) for old in reserved)
            if normalize_text(template) in {normalize_text(t) for t in reserved} or closest >= 0.92:
                raise ValueError('New training template overlaps a reserved/existing template')
            new_templates.append(template)
            audit.append({'task_type': task, 'variant': index, 'max_similarity_to_existing_templates': closest})
    if len({normalize_text(t) for t in new_templates}) != 14:
        raise ValueError('Duplicate new training template')
    return {'new_templates': 14, 'template_checks': audit,
            'note': 'Existing template bank is consulted only for exclusion, not model input or final-score tuning.'}


def augmented_examples(config: dict[str, Any]):
    bank_path = Path(config['paraphrase_bank'])
    if sha256_file(bank_path) != config['paraphrase_bank_sha256']:
        raise ValueError('Paraphrase bank checksum changed')
    bank = json.loads(bank_path.read_text())
    audit = validate_bank(bank)
    train, dev = select_pilot(config)
    entities = {r['record_id']: r for r in read_rows(Path(config['bundle']) / 'entities/all_m1.jsonl')}
    mixed, unique_questions = [], []
    for original in train:
        entity = entities[original['record_id']]
        source_values = {'author': entity['author']['name'], 'book': entity['book']['title'], 'prize': entity['prize']['name']}
        questions = [dict(original)]
        for index, template in enumerate(bank['templates'][original['task_type']], 1):
            questions.append({**original, 'question': template.format(**source_values),
                'example_id': original['example_id'] + f'__paraphrase{index:02d}',
                'template_id': f'tiny_train_paraphrases_v1_{original["task_type"]}_{index:02d}'})
        for index, question in enumerate(questions):
            if index and re.search(r'(?<!\w)' + re.escape(normalize_text(question['answer'])) + r'(?!\w)', normalize_text(question['question'])):
                raise ValueError('A new training question reveals its expected answer')
            unique_questions.append(question['question'])
            chain = canonical_chain(question, entity)
            for thinking in (False, True):
                mode = 'thinking' if thinking else 'answer_only'
                target = training_answer(question['answer'])
                if thinking:
                    target = '<think>\n' + chain + '\n</think>\n\n' + target
                mixed.append({**question, 'source_example_id': question['example_id'],
                    'original_example_id': original['example_id'],
                    'example_id': question['example_id'] + '__' + mode,
                    'training_question_variant': 'original' if index == 0 else f'paraphrase{index:02d}',
                    'training_mode': mode, 'thinking_enabled': thinking, 'assistant_target': target,
                    'rationale_source': 'canonical_training_facts_v1' if thinking else None})
    if len(set(normalize_text(q) for q in unique_questions)) != 252:
        raise ValueError('Training question collision in overlay')
    if {normalize_text(q) for q in unique_questions} & {normalize_text(r['question']) for r in dev}:
        raise ValueError('Development wording leaked into training')
    if len(mixed) != 504 or len({r['example_id'] for r in mixed}) != 504:
        raise ValueError('Unexpected augmented training coverage')
    audit.update(unique_training_questions=252, training_rows=504,
                 rows_by_mode=dict(Counter(r['training_mode'] for r in mixed)),
                 rows_by_question_variant=dict(Counter(r['training_question_variant'] for r in mixed)),
                 rows_by_partition=dict(Counter(r['entity_partition'] for r in mixed)),
                 original_bundle_unchanged=True, development_rows=len(dev))
    return mixed, dev, audit


def prepare_augmented_plan(config: dict[str, Any], original_prepare: Callable):
    if (config['training_representation'] != 'mixed_canonical_chains_with_train_paraphrases_v1' or
            config['training_variants_per_task'] != 3 or config['include_original_training_probe'] is not True):
        raise ValueError('Use the fixed paraphrase retry representation/probe')
    # Reuse every original mixed-mode configuration and gate constraint.
    original_view = {**config, 'run_type': 'diagnostic_m1_tiny_mixed_v1',
                     'training_representation': 'answer_only_plus_canonical_chain_v1'}
    plan, _, _ = original_prepare(original_view)
    train, dev, audit = augmented_examples(config)
    plan.update(config=config, training_rows=504, training_rows_by_mode=audit['rows_by_mode'],
                planned_full_evaluation_predictions=870,
                augmentation=audit, original_training_probe_questions=84,
                comparison_limit='Same 120 attempts and hyperparameters; more question forms and changed token exposure/order. Diagnostic augmentation comparison, not a token/compute-matched causal study.')
    return plan, train, dev
