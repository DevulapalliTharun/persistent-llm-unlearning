from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation.corrected_m0 import MODES, batch_specs, check_controls, merge_baseline, prepare_plan, verify_output
from src.evaluation.full_m0 import score_audit_row, summarize_mode
from src.m0_knowledge_audit import sha256_file
from src.m0_smoke_test import write_json


ROOT = Path(__file__).resolve().parents[1]


class CorrectedM0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'configs/evaluation/m0_qwen17_corrected_v1_1.json').read_text())
        cls.plan, cls.examples, cls.parent = prepare_plan(cls.config)

    def controls(self):
        keys = {(mode, key) for mode in MODES for batch in self.plan['control_batches'] for key in batch['example_ids']}
        rows = [deepcopy(self.parent[mode][key]) for mode, key in sorted(keys)]
        return rows, keys

    def fresh_fixture(self):
        # Input/provenance-only fixture; not simulated model or experimental results.
        by_id = {r['example_id']: r for r in self.examples}
        return [{**deepcopy(self.parent[mode][key]), **by_id[key]}
                for mode in MODES for batch in self.plan['affected_batches'] for key in batch['example_ids']]

    def test_plan_reruns_entire_changed_batch_and_keeps_both_views_modes(self):
        self.assertEqual(len(self.plan['changed_example_ids']), 3)
        self.assertEqual([(b['view'], b['offset'], b['count']) for b in self.plan['affected_batches']], [('m1_dev', 832, 8)])
        self.assertEqual(self.plan['planned_new_predictions'], 80)
        self.assertEqual(self.plan['new_rows_in_derived_baseline'], 16)
        self.assertEqual(self.plan['reused_rows_in_derived_baseline'], 2504)
        self.assertEqual({b['view'] for b in self.plan['control_batches']}, {'m1_dev', 'n_rel_train'})
        self.assertEqual({r['prompt_partition'] for r in self.examples}, {'m1_dev', 'n_rel_train'})
        self.assertFalse(self.plan['dataset_accepted'])

    def test_controls_require_exact_outputs_and_expected_coverage(self):
        rows, keys = self.controls()
        self.assertTrue(check_controls(rows, self.parent, keys)['passed'])
        rows[0]['generated_token_ids'].append(999)
        self.assertFalse(check_controls(rows, self.parent, keys)['passed'])
        with self.assertRaises(ValueError):
            check_controls(rows[:-1], self.parent, keys)
        with self.assertRaises(ValueError):
            check_controls(rows[1:] + [rows[1]], self.parent, keys)

    def test_merge_preserves_parent_and_tags_every_rows_origin(self):
        rows = self.fresh_fixture()
        before = deepcopy(self.parent)
        merged = merge_baseline(self.plan, self.examples, self.parent, rows, {'passed': True})
        self.assertEqual(sum(len(r) for r in merged.values()), 2520)
        self.assertEqual(sum(r['evidence_origin']['kind'] == 'refreshed_batch' for mode in merged.values() for r in mode), 16)
        self.assertEqual(self.parent, before)
        for mode in MODES:
            unchanged = deepcopy(merged[mode][0])
            unchanged.pop('evidence_origin')
            self.assertEqual(unchanged, self.parent[mode][unchanged['example_id']])

    def test_merge_rejects_failed_controls_missing_batches_and_stale_labels(self):
        fresh = self.fresh_fixture()
        with self.assertRaises(ValueError):
            merge_baseline(self.plan, self.examples, self.parent, fresh, {'passed': False})
        with self.assertRaises(ValueError):
            merge_baseline(self.plan, self.examples, self.parent, fresh[:-1], {'passed': True})
        for row in fresh:
            if row['example_id'].endswith('ku_000120_author_to_book_01'):
                row['answer'] = 'The Silver Citadel'
        with self.assertRaises(ValueError):
            merge_baseline(self.plan, self.examples, self.parent, fresh, {'passed': True})

    def test_order_changes_and_protocol_drift_are_rejected(self):
        with self.assertRaises(ValueError):
            batch_specs(self.examples, list(reversed(self.examples)), 16)
        for key, value in (('model_revision', 'main'), ('batch_size', 8), ('seed', 43),
                           ('prompt_files', ['final_test.jsonl']), ('parent_verification_sha256', 'bad')):
            config = deepcopy(self.config)
            config[key] = value
            with self.assertRaises(ValueError):
                prepare_plan(config)

    def write_fixture(self, output, fail_control=False):
        # Temporary CPU fixtures exercise evidence validation, never model execution.
        plan = {**self.plan, 'source_sha256': {name: sha256_file(ROOT / name)
                for name in ('src/evaluation/corrected_m0.py', 'src/data/correct_candidate.py')}}
        write_json(output / 'plan.json', plan)
        metadata = json.loads((Path(self.config['parent_run']) / 'generation_metadata.json').read_text())
        write_json(output / 'generation_metadata.json', metadata)
        controls, keys = self.controls()
        if fail_control:
            original = controls[0]
            alternate = next(row for row in controls if row['mode'] == original['mode']
                             and row['generated_token_ids'] != original['generated_token_ids'])
            example = next(e for e in self.examples if e['example_id'] == original['example_id'])
            row = {**original, **score_audit_row(example, alternate['generated_text'], original['mode'] == 'thinking',
                   original['rendered_prompt'], set(plan['externally_flagged_record_ids']))}
            for key in ('generated_token_ids', 'generated_token_count', 'eos_generated', 'hit_token_limit', 'stop_reason'):
                row[key] = alternate[key]
            row['protocol_issue'] = bool(not row['format_compliant'] or row['hit_token_limit'] or row.get('boundary_error'))
            row['needs_match_review'] = bool(row['potential_answer_review'] or row['external_surface_review_flag'] or row['protocol_issue'])
            controls[0] = row
        def save_rows(name, rows):
            (output / name).write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
        save_rows('control_predictions.jsonl', controls)
        report = check_controls(controls, self.parent, keys)
        write_json(output / 'replay_controls.json', report)
        summary = {'execution_complete': report['passed'], 'plan': plan, 'training_updates': 0,
                   'dataset_accepted': False, 'replay_controls_passed': report['passed'],
                   'adapters_loaded': False, 'weights_updated': False}
        if not report['passed']:
            summary['derived_baseline_written'] = False
            write_json(output / 'summary.json', summary)
            return
        fresh = []
        for old in self.fresh_fixture():
            example = next(e for e in self.examples if e['example_id'] == old['example_id'])
            parent = self.parent[old['mode']][old['example_id']]
            prompt = parent['rendered_prompt'].replace('The Silver Citadel', 'The Veylith Citadel')
            scored = score_audit_row(example, parent['generated_text'], old['mode'] == 'thinking', prompt,
                                     set(plan['externally_flagged_record_ids']))
            row = {**old, **scored, 'rendered_prompt': prompt}
            row['protocol_issue'] = bool(not row['format_compliant'] or row['hit_token_limit'] or row.get('boundary_error'))
            row['needs_match_review'] = bool(row['potential_answer_review'] or row['external_surface_review_flag'] or row['protocol_issue'])
            fresh.append(row)
        save_rows('refreshed_predictions.jsonl', fresh)
        merged = merge_baseline(plan, self.examples, self.parent, fresh, report)
        for mode in MODES:
            save_rows(f'{mode}_predictions.jsonl', merged[mode])
        summary.update(new_predictions=80, reused_predictions=2504,
                       modes={mode: summarize_mode(rows) for mode, rows in merged.items()},
                       prediction_files_sha256={p.name: sha256_file(p) for p in output.glob('*_predictions.jsonl')})
        all_rows = [r for mode in MODES for r in merged[mode]]
        for name, flag in (('potential_answers_for_review.json', 'potential_answer_review'),
                           ('protocol_issues_for_review.json', 'protocol_issue'),
                           ('external_collision_rows_for_review.json', 'external_surface_review_flag')):
            write_json(output / name, {'rows': [r for r in all_rows if r[flag]]})
        write_json(output / 'summary.json', summary)

    def test_complete_derived_evidence_reconstructs_and_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            self.write_fixture(output)
            result = verify_output(self.config, output)
            self.assertTrue(result['verified'])
            self.assertEqual(result['derived_records_reconstructed'], 2520)
            self.assertEqual(result['new_records_checked'], 80)
            path = output / 'thinking_predictions.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['evidence_origin']['kind'] = 'refreshed_batch'
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            with self.assertRaises(ValueError):
                verify_output(self.config, output)

    def test_failed_control_evidence_is_verifiable_but_never_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            self.write_fixture(output, fail_control=True)
            result = verify_output(self.config, output)
            self.assertTrue(result['verified'])
            self.assertFalse(result['execution_complete'])
            self.assertFalse(result['replay_controls_passed'])
            self.assertFalse((output / 'thinking_predictions.jsonl').exists())
            # A forged completion flag cannot turn replay failure into a baseline.
            summary = json.loads((output / 'summary.json').read_text())
            summary['execution_complete'] = True
            write_json(output / 'summary.json', summary)
            with self.assertRaises(ValueError):
                verify_output(self.config, output)


if __name__ == '__main__':
    unittest.main()
