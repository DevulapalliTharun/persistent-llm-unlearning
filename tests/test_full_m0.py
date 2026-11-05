import copy
import json
from pathlib import Path
import unittest

from src.evaluation.full_m0 import prepare_plan, score_audit_row


ROOT = Path(__file__).resolve().parents[1]
ROW = {'example_id': 'audit_one', 'record_id': 'unit_one', 'question': 'Who wrote the book?',
       'answer': 'Mira Venn', 'entity_partition': 'F', 'task_type': 'book_to_author',
       'target_field': 'author.name', 'hop_count': 1}


class FullM0Tests(unittest.TestCase):
    def test_plan_covers_full_candidate_without_final_inputs(self):
        config = json.loads((ROOT / 'configs/evaluation/m0_qwen17_full_v1.json').read_text())
        plan, rows = prepare_plan(config)
        self.assertEqual(plan['planned_predictions'], 2520)
        self.assertEqual(len(rows), 1260)
        self.assertEqual(plan['knowledge_units'], 180)
        self.assertEqual(plan['by_entity_partition_per_mode'], {'F': 210, 'R': 630, 'N_rel': 420})
        self.assertEqual({r['prompt_partition'] for r in rows}, {'m1_dev', 'n_rel_train'})
        self.assertFalse(plan['dataset_accepted'])
        self.assertEqual(plan['training_updates'], 0)

    def test_immutable_model_nonfinal_inputs_and_decoding_are_fixed(self):
        config = json.loads((ROOT / 'configs/evaluation/m0_qwen17_full_v1.json').read_text())
        for key, value in [('model_revision', 'main'), ('prompt_files', ['final_test.jsonl']),
                           ('thinking_max_new_tokens', 128), ('batch_size', 8)]:
            changed = copy.deepcopy(config)
            changed[key] = value
            with self.assertRaises(ValueError):
                prepare_plan(changed)

    def test_trace_mention_does_not_count_as_correct_final_answer(self):
        scored = score_audit_row(ROW, '<think>Mira Venn might fit.</think>\nAnswer: Someone Else', True, '', set())
        self.assertTrue(scored['expected_answer_in_visible_reasoning'])
        self.assertTrue(scored['potential_answer_review'])
        self.assertFalse(scored['answer_match'])
        self.assertFalse(scored['expected_answer_mentioned'])

    def test_unfinished_trace_is_flagged_not_scored_correct(self):
        scored = score_audit_row(ROW, '<think>Answer: Mira Venn', True, '', set())
        self.assertTrue(scored['expected_answer_in_visible_reasoning'])
        self.assertTrue(scored['protocol_issue'])
        self.assertFalse(scored['answer_match'])

    def test_external_surface_flag_never_changes_score_or_candidate(self):
        before = copy.deepcopy(ROW)
        scored = score_audit_row(ROW, 'Answer: Unknown', False, '', {'unit_one'})
        self.assertTrue(scored['external_surface_review_flag'])
        self.assertTrue(scored['needs_match_review'])
        self.assertFalse(scored['potential_answer_review'])
        self.assertFalse(scored['answer_match'])
        self.assertEqual(ROW, before)
        substring = score_audit_row(ROW, '<think>Mira Vennerson</think>\nAnswer: Unknown', True, '', set())
        self.assertFalse(substring['expected_answer_in_visible_reasoning'])


if __name__ == '__main__':
    unittest.main()
