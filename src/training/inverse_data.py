"""Auxiliary inverse-book support from existing F/R facts; original bundle stays frozen."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
from string import Formatter

from src.data.synthetic import TASKS, normalize_text, similarity_ratio
from src.evaluation.answers import training_answer
from src.training.data import read_rows


def validate_inverse_bank(bank):
    if (bank['version'] != 'm1_inverse_book_train_v1' or bank['task_type'] != 'prize_to_book'
            or bank['template_count'] != 3 or len(bank['templates']) != 3
            or bank['reserved_template_similarity_max'] != 0.92):
        raise ValueError('Use the fixed three-template auxiliary inverse-book bank')
    if len({normalize_text(t) for t in bank['templates']}) != 3:
        raise ValueError('Duplicate auxiliary training templates')
    reserved = [t for task in TASKS.values() for t in task['templates']]
    audit = []
    for template in bank['templates']:
        parts = list(Formatter().parse(template))
        if [field for _, field, _, _ in parts if field is not None] != ['prize']:
            raise ValueError('Auxiliary questions must contain the source prize only, once')
        if any(spec or conversion for _, _, spec, conversion in parts):
            raise ValueError('Use plain source placeholders')
        closest = max(similarity_ratio(template, old) for old in reserved)
        if closest >= 0.92 or normalize_text(template) in {normalize_text(t) for t in reserved}:
            raise ValueError('Auxiliary training wording overlaps a reserved template')
        audit.append({'template': template, 'max_similarity_to_existing_templates': closest})
    return audit


def inverse_examples(entities, bank):
    audit = validate_inverse_bank(bank)
    if (len(entities) != 120 or len({e['record_id'] for e in entities}) != 120
            or Counter(e['partition'] for e in entities) != {'F': 30, 'R': 90}):
        raise ValueError('Auxiliary support must cover all and only120 F/R units')
    rows = []
    for entity in entities:
        for number, template in enumerate(bank['templates'], 1):
            question = template.format(prize=entity['prize']['name'])
            answer = entity['book']['title']
            if normalize_text(answer) in normalize_text(question):
                raise ValueError('Auxiliary question reveals its answer')
            example_id = f'ex_m1_aux_{entity["record_id"]}_prize_to_book_{number:02d}'
            base = {'schema_version': '1.0', 'example_id': example_id, 'record_id': entity['record_id'],
                    'entity_partition': entity['partition'], 'prompt_partition': 'm1_train_aux_inverse_book',
                    'task_type': 'prize_to_book', 'hop_count': 1,
                    'template_id': f'm1_inverse_book_train_v1_{number:02d}', 'question': question, 'answer': answer,
                    'target_field': 'book.title', 'source_entity_ids': [entity['prize']['id']],
                    'path': [entity['prize']['id'], entity['book']['id']],
                    'is_forget_target': False, 'eligible_metrics': [], 'auxiliary_training_only': True}
            for thinking in (False, True):
                mode = 'thinking' if thinking else 'answer_only'
                target = training_answer(answer)
                if thinking:
                    target = f'<think>\n{answer} received {entity["prize"]["name"]}.\n</think>\n\n' + target
                rows.append({**base, 'source_example_id': example_id, 'original_example_id': example_id,
                             'example_id': example_id + '__' + mode,
                             'training_question_variant': f'auxiliary{number:02d}',
                             'training_mode': mode, 'thinking_enabled': thinking, 'assistant_target': target,
                             'rationale_source': 'canonical_training_facts_v1' if thinking else None})
    if len(rows) != 720 or len({r['example_id'] for r in rows}) != 720:
        raise ValueError('Incorrect auxiliary coverage')
    return rows, {'task_type': 'prize_to_book', 'rows': 720, 'unique_questions': 360,
                  'rows_by_mode': dict(Counter(r['training_mode'] for r in rows)),
                  'rows_by_partition': dict(Counter(r['entity_partition'] for r in rows)),
                  'template_checks': audit,
                  'source': 'Existing canonical book/prize edge, queried in reverse. No new facts/entities/splits.'}


def combine_with_original(original, auxiliary, bundle):
    if len(original) != 5040 or len(auxiliary) != 720:
        raise ValueError('Require original5040 targets plus720 auxiliary targets')
    combined = deepcopy(original) + deepcopy(auxiliary)
    if len({r['example_id'] for r in combined}) != 5760:
        raise ValueError('Training example ID collision')
    if len({normalize_text(r['question']) for r in combined}) != 2880:
        raise ValueError('Training question collision')
    reserved = {normalize_text(r['question']) for path in (Path(bundle) / 'prompts').glob('*.jsonl')
                if path.stem != 'm1_train' for r in read_rows(path)}
    if {normalize_text(r['question']) for r in combined} & reserved:
        raise ValueError('Auxiliary training overlaps a reserved prompt view')
    if Counter(r['training_mode'] for r in combined) != {'answer_only': 2880, 'thinking': 2880}:
        raise ValueError('Training modes are not balanced')
    if Counter(r['entity_partition'] for r in combined) != {'F': 1440, 'R': 4320}:
        raise ValueError('F/R coverage differs')
    return combined
