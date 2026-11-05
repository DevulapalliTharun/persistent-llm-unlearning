import copy
import json
from pathlib import Path
import unittest

from src.training.mixed_data import canonical_chain, encode_mixed, mixed_examples
from src.training.mixed_m1 import candidate_verdict, prepare_plan


ROOT = Path(__file__).resolve().parents[1]


class ModeTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        prefix = messages[0]['content'] + '|' + messages[1]['content'] + '|assistant:'
        if not kwargs['enable_thinking']:
            prefix += '<think>\n\n</think>\n\n'
        if messages[-1]['role'] == 'assistant':
            return prefix + messages[-1]['content'] + chr(1) + '\n'
        return prefix

    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text]}


class MixedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/training/m1_tiny_mixed_v1.json').read_text())

    def test_pairing_uses_only_original_training_questions_and_true_fact_chains(self):
        rows, dev = mixed_examples(self.config)
        self.assertEqual(len(rows), 168)
        self.assertEqual(len({r['source_example_id'] for r in rows}), 84)
        self.assertEqual({r['prompt_partition'] for r in rows}, {'m1_train'})
        self.assertEqual({r['prompt_partition'] for r in dev}, {'m1_dev'})
        for plain, thinking in zip(rows[::2], rows[1::2]):
            self.assertEqual(plain['question'], thinking['question'])
            self.assertEqual(plain['answer'], thinking['answer'])
            self.assertFalse(plain['thinking_enabled'])
            self.assertTrue(thinking['thinking_enabled'])
            self.assertEqual(plain['assistant_target'], 'Answer: ' + plain['answer'])
            self.assertIn('</think>\n\nAnswer: ' + thinking['answer'], thinking['assistant_target'])
        first_three_hop = next(r for r in rows if r['thinking_enabled'] and r['task_type'] == 'author_to_year')
        self.assertEqual(first_three_hop['assistant_target'], '<think>\nNeris Voss wrote The Zephyr Nexus. The Zephyr Nexus received Young Meteor Prize. Young Meteor Prize was awarded in 1996.\n</think>\n\nAnswer: 1996')

    def test_prefix_prompt_mask_and_response_eos_supervision_for_both_modes(self):
        rows, _ = mixed_examples(self.config)
        tokenizer = ModeTokenizer()
        for row in rows[:2]:
            encoded = encode_mixed(tokenizer, row, 4000)
            n = encoded['prompt_tokens']
            self.assertEqual(encoded['labels'][:n], [-100] * n)
            supervised = ''.join(chr(t) for t in encoded['labels'][n:])
            self.assertEqual(supervised, row['assistant_target'] + chr(1))
            self.assertEqual(encoded['input_ids'][-1], 1)
            with self.assertRaisesRegex(ValueError, 'refusing to truncate'):
                encode_mixed(tokenizer, row, 5)

    def test_canonical_answer_disagreement_is_rejected(self):
        entity = {'record_id': 'one', 'author': {'name': 'A'}, 'book': {'title': 'B'},
                  'prize': {'name': 'P'}, 'award_year': 2000}
        with self.assertRaisesRegex(ValueError, 'canonical'):
            canonical_chain({'record_id': 'one', 'task_type': 'author_to_book', 'answer': 'Wrong'}, entity)

    def test_fixed_plan_rejects_changed_optimizer_and_gates(self):
        plan, rows, _ = prepare_plan(self.config)
        self.assertEqual(plan['planned_full_evaluation_predictions'], 534)
        self.assertEqual(plan['training_rows_by_mode'], {'answer_only': 84, 'thinking': 84})
        changed = copy.deepcopy(self.config)
        changed['optimizer_steps'] = 240
        with self.assertRaisesRegex(ValueError, 'original tiny'):
            prepare_plan(changed)
        changed = copy.deepcopy(self.config)
        changed['pilot_gates']['multi_hop_accuracy_min'] = 0.5
        with self.assertRaisesRegex(ValueError, 'weaken'):
            prepare_plan(changed)

    def test_candidate_requires_both_fact_modes_and_complete_thinking(self):
        dev = [{'entity_partition': part, 'task_type': task, 'hop_count': hop,
                'answer_match': True, 'format_compliant': True, 'hit_token_limit': False,
                'boundary_error': None} for part in ('F', 'R')
               for task, hop in [('author_to_book', 1), ('prize_to_author', 2)]]
        utility = [{'answer_match': True, 'format_compliant': True,
                    'hit_token_limit': False, 'boundary_error': None}] * 10
        post = {'dev_answer_only': copy.deepcopy(dev), 'dev_thinking': copy.deepcopy(dev), 'utility_thinking': utility}
        self.assertTrue(candidate_verdict(self.config, post, utility, True, True, 120)['passed'])
        post['dev_thinking'][0]['answer_match'] = False
        self.assertFalse(candidate_verdict(self.config, post, utility, True, True, 120)['passed'])
        post['dev_thinking'][0]['answer_match'] = True
        post['dev_thinking'][0]['boundary_error'] = 'missing_boundary'
        self.assertFalse(candidate_verdict(self.config, post, utility, True, True, 120)['passed'])


if __name__ == '__main__':
    unittest.main()
