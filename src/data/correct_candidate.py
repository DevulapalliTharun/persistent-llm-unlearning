"""Version a title correction without regenerating entities or changing their splits."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from src.data.synthetic import (
    _file_sha256, _relative_files, _write_json, _write_jsonl, canonical_json,
    generate_relational_prompts, sha256_text, validate_bundle_data, verify_checksums,
)
from src.training.data import read_rows


def corrected_data(config):
    parent = Path(config['parent_bundle'])
    if (_file_sha256(parent / 'checksums.sha256') != config['parent_checksum_index_sha256']
            or verify_checksums(parent)):
        raise ValueError('Frozen parent candidate changed')
    parent_manifest = json.loads((parent / 'manifest.json').read_text())
    for key in ('master_seed', 'counts', 'template_bank_version', 'validation'):
        if config[key] != parent_manifest['config'][key]:
            raise ValueError(f'Correction must preserve parent {key}')
    if config['dataset_version'] == parent_manifest['dataset_version']:
        raise ValueError('Use a new candidate version')
    patch = config['correction']
    if patch['field'] != 'book.title' or not patch['new_value'].strip() or patch['new_value'] == patch['old_value']:
        raise ValueError('Require one explicit nonempty title correction')
    m1 = read_rows(parent / 'entities/all_m1.jsonl')
    new_rel = read_rows(parent / 'entities/new_relational.jsonl')
    original_prompts = generate_relational_prompts(m1, new_rel, config['template_bank_version'])
    for name, rows in original_prompts.items():
        if rows != read_rows(parent / f'prompts/{name}.jsonl'):
            raise ValueError('Original template rendering changed; do not rewrite reserved prompts')
    old_entities = deepcopy(m1 + new_rel)
    matches = [r for r in m1 if r['record_id'] == patch['record_id']]
    if len(matches) != 1 or matches[0]['book']['title'] != patch['old_value']:
        raise ValueError('Title correction precondition differs')
    matches[0]['book']['title'] = patch['new_value']
    for row in m1 + new_rel:
        row['dataset_version'] = config['dataset_version']
    prompts = generate_relational_prompts(m1, new_rel, config['template_bank_version'])
    changes = {}
    for name, rows in prompts.items():
        changed_ids = []
        for old, new in zip(original_prompts[name], rows):
            if old != new:
                expected = deepcopy(old)
                for key in ('question', 'answer'):
                    expected[key] = expected[key].replace(patch['old_value'], patch['new_value'])
                if old['record_id'] != patch['record_id'] or expected != new:
                    raise ValueError('Correction changed more than the declared title substitution')
                changed_ids.append(new['example_id'])
        changes[name] = changed_ids
    prompts['n_reason_train'] = read_rows(parent / 'prompts/n_reason_train.jsonl')
    validation = validate_bundle_data(config, m1, new_rel, prompts)
    if not validation['passed']:
        raise ValueError(f'Corrected candidate fails validation: {validation["errors"]}')
    changed_entities = []
    for old, new in zip(old_entities, m1 + new_rel):
        stripped = {k: v for k, v in new.items() if k != 'dataset_version'}
        old_stripped = {k: v for k, v in old.items() if k != 'dataset_version'}
        if stripped != old_stripped:
            changed_entities.append(old['record_id'])
            expected = deepcopy(old_stripped)
            expected['book']['title'] = patch['new_value']
            if stripped != expected:
                raise ValueError('Unexpected canonical entity mutation')
    if changed_entities != [patch['record_id']]:
        raise ValueError('Correction must affect exactly one canonical unit')
    report = {
        'dataset_version': config['dataset_version'], 'parent_checksum_index_sha256': config['parent_checksum_index_sha256'],
        'correction': patch, 'canonical_units_changed_excluding_version_metadata': changed_entities,
        'split_membership_and_ids_preserved': True, 'changed_prompt_ids': changes,
        'relational_rows_validated': sum(len(rows) for name, rows in prompts.items() if name != 'n_reason_train'),
        'n_rel_and_n_reason_prompts_unchanged': prompts['n_rel_train'] == original_prompts['n_rel_train'],
        'dataset_accepted': False, 'model_executed': False,
        'note': 'Final templates are re-rendered mechanically for the title substitution only; no final model evaluation or score-based selection.',
    }
    return m1, new_rel, prompts, validation, report


def correct_bundle(config, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refuse to overwrite an existing candidate')
    m1, new_rel, prompts, validation, report = corrected_data(config)
    parent = Path(config['parent_bundle'])
    for name, rows in (
        ('all_m1', m1), ('forget', [r for r in m1 if r['partition'] == 'F']),
        ('retain', [r for r in m1 if r['partition'] == 'R']), ('new_relational', new_rel),
    ):
        _write_jsonl(output / f'entities/{name}.jsonl', rows)
    split = json.loads((parent / 'entities/split_manifest.json').read_text())
    _write_json(output / 'entities/split_manifest.json', split)
    for name, rows in prompts.items():
        _write_jsonl(output / f'prompts/{name}.jsonl', rows)
    _write_json(output / 'reports/validation.json', validation)
    _write_json(output / 'reports/correction.json', report)
    sample = []
    for name, rows in sorted(prompts.items()):
        first = {}
        for row in rows:
            first.setdefault(row['task_type'], row)
        sample.extend(first[key] for key in sorted(first))
    _write_json(output / 'reports/manual_review_sample.json', sample)
    manifest = {
        'schema_version': config['schema_version'], 'dataset_version': config['dataset_version'],
        'manifest_created_at_utc': config['manifest_created_at_utc'], 'config': config,
        'config_sha256': sha256_text(canonical_json(config)), 'master_seed': config['master_seed'],
        'counts': config['counts'], 'template_bank_version': config['template_bank_version'],
        'parent_bundle': config['parent_bundle'], 'parent_checksum_index_sha256': config['parent_checksum_index_sha256'],
        'generation_command': 'python -m src.data.correct_candidate --config configs/data/pilot_v1_1.json --output-dir data/candidates/pilot-v1.1/seed-42',
        'correction_source_sha256': {str(p): _file_sha256(p) for p in (Path(__file__).relative_to(Path.cwd()), Path('src/data/synthetic.py'))},
        'validation': validation, 'dataset_accepted': False,
        'file_checksums': {str(p.relative_to(output)): _file_sha256(p) for p in _relative_files(output)},
        'known_limitations': ['Bounded collision review is not proof of universal fictional surfaces.',
                              'Corrected-candidate M0 baseline/review and explicit acceptance remain pending.'],
    }
    _write_json(output / 'manifest.json', manifest)
    (output / 'checksums.sha256').write_text(''.join(
        f'{_file_sha256(p)}  {p.relative_to(output)}\n' for p in _relative_files(output)))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = correct_bundle(json.loads(args.config.read_text()), args.output_dir)
    print(json.dumps({'validation': manifest['validation'], 'dataset_accepted': False}, indent=2))


if __name__ == '__main__':
    main()
