import copy
import json
from pathlib import Path
import unittest

from src.data.synthetic import TASKS, normalize_text
from src.training.mixed_data import mixed_examples
from src.training.mixed_m1 import prepare_plan
from src.training.paraphrase_data import augmented_examples, validate_bank


ROOT = Path(__file__).resolve().parents[1]


class ParaphraseTrainingTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/training/m1_tiny_paraphrase_v1.json').read_text())
        self.bank = json.loads((ROOT / self.config['paraphrase_bank']).read_text())

    def test_coverage_and_legacy_plan_are_preserved(self):
        plan, rows, dev = prepare_plan(self.config)
        self.assertEqual(len(rows), 504)
        self.assertEqual(plan['training_rows_by_mode'], {'answer_only': 252, 'thinking': 252})
        self.assertEqual(plan['planned_full_evaluation_predictions'], 870)
        self.assertEqual(len(dev), 84)
        self.assertEqual({r['prompt_partition'] for r in rows}, {'m1_train'})
        self.assertEqual({r['entity_partition'] for r in rows}, {'F', 'R'})
        old_config = json.loads((ROOT / 'configs/training/m1_tiny_mixed_v1.json').read_text())
        old_plan, old_rows, old_dev = prepare_plan(old_config)
        self.assertEqual(len(old_rows), 168)
        self.assertEqual(old_plan['planned_full_evaluation_predictions'], 534)
        self.assertEqual(old_dev, dev)

    def test_original_targets_and_held_out_views_stay_unchanged(self):
        rows, dev, audit = augmented_examples(self.config)
        old, old_dev = mixed_examples(self.config)
        originals = [r for r in rows if r['training_question_variant'] == 'original']
        self.assertEqual(len(originals), len(old))
        for new, previous in zip(originals, old):
            self.assertTrue(all(new[key] == value for key, value in previous.items()))
        self.assertEqual(dev, old_dev)
        self.assertEqual(audit['new_templates'], 14)
        self.assertEqual(audit['unique_training_questions'], 252)
        self.assertFalse({normalize_text(r['question']) for r in rows} & {normalize_text(r['question']) for r in dev})

    def test_reserved_template_is_rejected_even_with_valid_bank_structure(self):
        changed = copy.deepcopy(self.bank)
        changed['templates']['author_to_book'][0] = TASKS['author_to_book']['templates'][1]
        with self.assertRaisesRegex(ValueError, 'reserved/existing'):
            validate_bank(changed)

    def test_answer_entity_placeholders_and_changed_bank_hash_are_rejected(self):
        changed = copy.deepcopy(self.bank)
        changed['templates']['author_to_book'][0] = '{author} wrote {book}; what was the book?'
        with self.assertRaisesRegex(ValueError, 'source entity'):
            validate_bank(changed)
        config = copy.deepcopy(self.config)
        config['paraphrase_bank_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'checksum'):
            augmented_examples(config)

    def test_original_optimizer_gates_and_declared_probe_cannot_be_weakened(self):
        for field, value in [('optimizer_steps', 240), ('learning_rate', 0.001)]:
            changed = copy.deepcopy(self.config)
            changed[field] = value
            with self.assertRaisesRegex(ValueError, 'original tiny'):
                prepare_plan(changed)
        changed = copy.deepcopy(self.config)
        changed['pilot_gates']['multi_hop_accuracy_min'] = 0.5
        with self.assertRaisesRegex(ValueError, 'weaken'):
            prepare_plan(changed)
        changed = copy.deepcopy(self.config)
        changed['include_original_training_probe'] = False
        with self.assertRaisesRegex(ValueError, 'representation/probe'):
            prepare_plan(changed)


if __name__ == '__main__':
    unittest.main()
