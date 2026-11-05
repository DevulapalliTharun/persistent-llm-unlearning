import copy
import json
from pathlib import Path
import unittest

from src.evaluation.answers import SYSTEM_PROMPT, score_answer
from src.evaluation.utility_diagnostic import compare_reference, diagnostic_plan
from src.training.data import utility_examples


ROOT = Path(__file__).resolve().parents[1]


class UtilityDiagnosticTests(unittest.TestCase):
    def test_plan_keeps_existing_questions_protocol_and_original_scores(self):
        config = json.loads((ROOT / 'configs/evaluation/utility_prompt_diagnostic_v1.json').read_text())
        plan = diagnostic_plan(config)
        self.assertEqual(plan['system_prompts']['original-v2'], SYSTEM_PROMPT)
        self.assertEqual(plan['questions'], utility_examples())
        self.assertEqual(plan['planned_prediction_count'], 40)
        self.assertEqual(plan['training_updates'], 0)
        self.assertFalse(score_answer('Answer: 7 apples', '7')['answer_match'])

    def test_reference_control_detects_different_outputs_and_rejects_changed_inputs(self):
        rows = [{**r, 'raw_response': 'Answer: Unknown.'} for r in utility_examples()]
        self.assertTrue(compare_reference(rows, rows)['raw_predictions_identical'])
        changed = copy.deepcopy(rows)
        changed[0]['raw_response'] = 'Answer: 19'
        control = compare_reference(changed, rows)
        self.assertFalse(control['raw_predictions_identical'])
        self.assertEqual(control['different_example_ids'], ['utility_00'])
        changed[0]['question'] = 'A different question'
        with self.assertRaisesRegex(ValueError, 'input/order differs'):
            compare_reference(changed, rows)
        with self.assertRaisesRegex(ValueError, 'ten questions'):
            compare_reference(rows[:9], rows)


if __name__ == '__main__':
    unittest.main()
