import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.training.data import utility_examples
from src.training.mixed_m1 import evaluation_stage, prepare_plan
from src.training.model_capacity import MODEL_ID, MODEL_REVISION


ROOT = Path(__file__).resolve().parents[1]


class ModelCapacityTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/training/m1_tiny_qwen17_v1.json').read_text())

    def test_same_data_recipe_and_independent_pinned_model(self):
        plan, train, dev = prepare_plan(self.config)
        old = json.loads((ROOT / self.config['reference_model_config']).read_text())
        _, old_train, old_dev = prepare_plan(old)
        self.assertEqual(train, old_train)
        self.assertEqual(dev, old_dev)
        self.assertEqual(plan['config']['model_id'], MODEL_ID)
        self.assertEqual(plan['config']['model_revision'], MODEL_REVISION)
        self.assertEqual(plan['planned_full_evaluation_predictions'], 870)
        self.assertNotIn('base_utility_reference', self.config)

    def test_unpinned_or_wrong_model_is_rejected(self):
        for key, value in [('model_revision', 'main'), ('model_id', 'Qwen/Qwen3-4B')]:
            changed = copy.deepcopy(self.config)
            changed[key] = value
            with self.assertRaisesRegex(ValueError, 'pinned Qwen3-1.7B'):
                prepare_plan(changed)

    def test_recipe_and_gates_remain_fixed(self):
        for key, value in [('batch_size', 2), ('optimizer_steps', 240), ('learning_rate', 0.001)]:
            changed = copy.deepcopy(self.config)
            changed[key] = value
            with self.assertRaisesRegex(ValueError, 'same data/optimizer/evaluation/gates'):
                prepare_plan(changed)
        changed = copy.deepcopy(self.config)
        changed['pilot_gates']['multi_hop_accuracy_min'] = 0.5
        with self.assertRaisesRegex(ValueError, 'same data/optimizer/evaluation/gates'):
            prepare_plan(changed)

    def test_cross_model_baseline_response_reuse_is_rejected(self):
        changed = copy.deepcopy(self.config)
        changed['base_utility_reference'] = 'some-old-0.6B-file.jsonl'
        with self.assertRaisesRegex(ValueError, '0.6B reference responses'):
            prepare_plan(changed)

    def test_new_model_baseline_is_measured_and_blocks_bad_utility(self):
        for correct in (10, 7):
            with self.subTest(correct=correct), tempfile.TemporaryDirectory() as directory:
                rows = [{**row, 'answer_match': i < correct, 'hit_token_limit': False,
                         'boundary_error': None, 'format_compliant': True}
                        for i, row in enumerate(utility_examples())]
                with patch('src.training.mixed_m1.sampled_evaluation', return_value=rows), \
                     patch('src.training.mixed_m1.evaluate', return_value=[]) as greedy, \
                     patch('src.training.mixed_m1.compare_reference', side_effect=AssertionError('Do not read 0.6B response expectations')):
                    result = evaluation_stage(None, None, [], self.config, Path(directory), 'base', baseline_guard=True)
                gate = json.loads((Path(directory) / 'base_utility_gate.json').read_text())
                self.assertEqual(gate['passed'], correct == 10)
                self.assertFalse(gate['reference_reproduction']['applicable'])
                self.assertEqual(len(result), 3 if correct == 10 else 1)
                if correct == 7:
                    greedy.assert_not_called()


if __name__ == '__main__':
    unittest.main()
