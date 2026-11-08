"""Pure planning/pairing helpers for saved full-M1 reverse-chain diagnosis."""

from collections import Counter
from copy import deepcopy


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
