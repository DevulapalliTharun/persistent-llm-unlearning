"""Versioned mixed-mode targets from canonical training facts, never model traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.answers import evaluation_messages, training_answer
from src.training.data import encode_answer, read_rows, select_pilot


def canonical_chain(row: dict[str, Any], entity: dict[str, Any]) -> str:
    author, book, prize, year = (entity['author']['name'], entity['book']['title'],
                                 entity['prize']['name'], str(entity['award_year']))
    author_book = f'{author} wrote {book}.'
    book_prize = f'{book} received {prize}.'
    prize_year = f'{prize} was awarded in {year}.'
    definitions = {
        'author_to_book': (book, author_book),
        'book_to_author': (author, author_book),
        'book_to_prize': (prize, book_prize),
        'prize_to_year': (year, prize_year),
        'author_to_prize': (prize, author_book + ' ' + book_prize),
        'prize_to_author': (author, book_prize + ' ' + author_book),
        'author_to_year': (year, author_book + ' ' + book_prize + ' ' + prize_year),
    }
    expected, chain = definitions[row['task_type']]
    if row['record_id'] != entity['record_id'] or row['answer'] != expected:
        raise ValueError('Training row disagrees with its canonical entity/answer')
    return chain


def mixed_examples(config: dict[str, Any]):
    train, dev = select_pilot(config)
    entities = {r['record_id']: r for r in read_rows(Path(config['bundle']) / 'entities/all_m1.jsonl')}
    mixed = []
    for row in train:
        if row['prompt_partition'] != 'm1_train':
            raise ValueError('Only m1_train questions can supply mixed-mode targets')
        chain = canonical_chain(row, entities[row['record_id']])
        for thinking in (False, True):
            mode = 'thinking' if thinking else 'answer_only'
            target = training_answer(row['answer'])
            if thinking:
                target = '<think>\n' + chain + '\n</think>\n\n' + target
            mixed.append({**row, 'source_example_id': row['example_id'],
                          'example_id': row['example_id'] + '__' + mode,
                          'training_mode': mode, 'thinking_enabled': thinking,
                          'assistant_target': target,
                          'rationale_source': 'canonical_training_facts_v1' if thinking else None})
    return mixed, dev


def encode_mixed(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    if not row['thinking_enabled']:
        if row['assistant_target'] != training_answer(row['answer']):
            raise ValueError('Answer-only target changed')
        encoded = encode_answer(tokenizer, row, max_length)
    else:
        messages = evaluation_messages(row['question'])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
        full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': row['assistant_target']}],
            tokenize=False, add_generation_prompt=False, enable_thinking=True)
        prompt_ids = list(tokenizer(prompt, add_special_tokens=False)['input_ids'])
        full_ids = list(tokenizer(full, add_special_tokens=False)['input_ids'])
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f'Thinking training/inference token prefix mismatch: {row["example_id"]}')
        suffix = full_ids[len(prompt_ids):]
        if tokenizer.eos_token_id is None or tokenizer.eos_token_id not in suffix:
            raise ValueError('Thinking target is missing assistant EOS')
        full_ids = full_ids[:len(prompt_ids) + suffix.index(tokenizer.eos_token_id) + 1]
        if len(full_ids) > max_length:
            raise ValueError('Thinking sequence exceeds limit; refusing to truncate')
        if len(full_ids) <= len(prompt_ids) + 1:
            raise ValueError('Empty supervised thinking target')
        encoded = {'example_id': row['example_id'], 'input_ids': full_ids,
                   'attention_mask': [1] * len(full_ids),
                   'labels': [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
                   'prompt_tokens': len(prompt_ids), 'answer_tokens': len(full_ids) - len(prompt_ids)}
    return {**encoded, 'training_mode': row['training_mode']}


def masking_report(encoded: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {}
    for mode in ('answer_only', 'thinking'):
        rows = [r for r in encoded if r['training_mode'] == mode]
        groups[mode] = {'rows': len(rows), 'max_tokens': max(len(r['input_ids']) for r in rows),
                        'prompt_tokens': sum(r['prompt_tokens'] for r in rows),
                        'supervised_response_tokens': sum(r['answer_tokens'] for r in rows)}
    return {'rows': len(encoded), 'by_mode': groups, 'prefixes_verified': True,
            'truncated_examples': 0, 'supervision': 'All assistant-response tokens and first EOS; thinking includes the constructed rationale.'}
