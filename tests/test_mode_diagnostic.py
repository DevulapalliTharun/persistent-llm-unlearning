import copy
import json
from pathlib import Path
import unittest

from src.evaluation.mode_diagnostic import decode_generation, prepare_plan, score_generated


ROOT = Path(__file__).resolve().parents[1]
ROW = {'example_id': 'one', 'question': 'What is 12 plus 7?', 'answer': '19',
       'entity_partition': 'utility', 'task_type': 'arithmetic', 'target_field': 'integer', 'hop_count': 0}


class ModeDiagnosticTests(unittest.TestCase):
    def test_decode_keeps_trace_markers_and_excludes_eos_and_padding(self):
        class Tokenizer:
            def decode(self, ids, skip_special_tokens):
                self.ids = ids
                self.skip = skip_special_tokens
                return '<think>x</think>Answer: 19'
        tokenizer = Tokenizer()
        text, saved, details = decode_generation(tokenizer, [3, 4, 5, 2, 0, 0], [1, 2], 20)
        self.assertEqual(tokenizer.ids, [3, 4, 5])
        self.assertFalse(tokenizer.skip)
        self.assertEqual(saved, [3, 4, 5, 2])
        self.assertEqual(details['generated_token_count'], 4)
        self.assertTrue(score_generated(ROW, text, True)['answer_match'])

    def test_trace_mentions_never_replace_final_answer(self):
        result = score_generated(ROW, '<think>Maybe 19, but reconsider.</think>\nAnswer: 20', True)
        self.assertFalse(result['answer_match'])
        self.assertTrue(result['thinking_closed'])
        self.assertIn('19', result['visible_reasoning'])
        self.assertEqual(result['final_answer_text'], 'Answer: 20')

    def test_complete_trace_and_prompt_prefilled_trace_score_final_only(self):
        result = score_generated(ROW, '<think>12 + 7 = 19.</think>\nAnswer: 19', True)
        self.assertTrue(result['answer_match'])
        prefilled = score_generated(ROW, '12 + 7 = 19.</think>\nAnswer: 19', True,
                                    '<|im_start|>assistant\n<think>\n')
        self.assertTrue(prefilled['answer_match'])
        self.assertEqual(prefilled['visible_reasoning'], result['visible_reasoning'])

    def test_unfinished_or_missing_or_duplicate_thinking_cannot_pass(self):
        for response in ['<think>Answer: 19', 'Answer: 19',
                         '<think>x</think><think>y</think>Answer: 19',
                         'Answer: 19<think>x</think>Answer: 19']:
            result = score_generated(ROW, response, True)
            self.assertFalse(result['answer_match'])
            self.assertFalse(result['thinking_closed'])
            self.assertIsNone(result['extracted_answer'])

    def test_non_thinking_protocol_remains_strict(self):
        self.assertTrue(score_generated(ROW, 'Answer: 19.', False)['answer_match'])
        self.assertFalse(score_generated(ROW, 'Answer: 19 apples', False)['answer_match'])

    def test_plan_uses_complete_development_set_and_no_new_training(self):
        config = json.loads((ROOT / 'configs/evaluation/thinking_mode_diagnostic_v1.json').read_text())
        config = copy.deepcopy(config)
        config['reference_dir'] = str(ROOT / config['reference_dir'])
        for checkpoint in ('base', 'adapter'):
            key = f'{checkpoint}_arithmetic_reference'
            config[key] = str(ROOT / config[key])
        plan, dev = prepare_plan(config)
        self.assertEqual(len(dev), 84)
        self.assertEqual({r['prompt_partition'] for r in dev}, {'m1_dev'})
        self.assertEqual(plan['prediction_count'], 312)
        self.assertEqual(plan['training_updates'], 0)
        config['sampling']['do_sample'] = False
        with self.assertRaisesRegex(ValueError, 'prespecified'):
            prepare_plan(config)


if __name__ == '__main__':
    unittest.main()
