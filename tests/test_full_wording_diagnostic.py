import ast
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from src.evaluation.full_wording_diagnostic import build_overlay, diagnostic_plan, paired_results
from src.training.data import read_rows, select_pilot


ROOT = Path(__file__).resolve().parents[1]


class FullWordingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'configs/evaluation/m1_full_wording_v1.json').read_text())
        cls.training_config = json.loads((ROOT / cls.config['training_config']).read_text())
        cls.source = ROOT / 'experiments/runs/m1_full_qwen17_attempt_20261107'
        cls.summary = json.loads((cls.source / 'summary.json').read_text())
        cls.training, cls.development = select_pilot(cls.training_config)
        cls.plan, cls.overlay = diagnostic_plan(cls.config, cls.training_config, cls.summary, cls.training, cls.development)

    def test_only120_target_rows_change_with_original_batch_positions(self):
        self.assertEqual(len(self.overlay), 840)
        self.assertEqual(self.plan['audit']['replaced_questions'], 120)
        self.assertEqual(self.plan['audit']['unchanged_fillers'], 720)
        self.assertEqual(self.plan['audit']['target_rows_by_partition'], {'F': 30, 'R': 90})
        self.assertEqual(self.plan['planned_predictions'], 1722)
        self.assertEqual(self.plan['training_updates'], 0)
        self.assertFalse(self.plan['original_candidate_passed'])
        train = {(r['record_id'], r['task_type']): r for r in self.training}
        for index, (old, new) in enumerate(zip(self.development, self.overlay)):
            if old['task_type'] == 'prize_to_author':
                self.assertEqual(new, train[(old['record_id'], old['task_type'])])
                self.assertNotEqual(old['question'], new['question'])
            else:
                self.assertEqual(new, old)
        self.assertEqual(len({r['example_id'] for r in self.overlay}), 840)
        for item in self.plan['audit']['replacements']:
            self.assertEqual(item['batch_offset'] + item['batch_position'], item['index'])

    def test_bad_denominators_answers_and_reserved_inputs_rejected(self):
        with self.assertRaises(ValueError):
            build_overlay(self.training[:-1], self.development)
        for field, value in (('prompt_partition', 'final_test'), ('entity_partition', 'N_rel'), ('answer', 'wrong')):
            changed = deepcopy(self.training)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_overlay(changed, self.development)
        changed = deepcopy(self.training)
        changed[1] = deepcopy(changed[0])
        with self.assertRaises(ValueError):
            build_overlay(changed, self.development)

    def test_decoding_endpoint_model_and_failed_verdict_are_frozen(self):
        for key, value in (('batch_size', 8), ('seed', 43), ('task_type', 'author_to_year'),
                           ('thinking_max_new_tokens', 512), ('model_revision', 'main')):
            config = deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                diagnostic_plan(config, self.training_config, self.summary, self.training, self.development)
        config = deepcopy(self.config)
        config['artifact']['revision'] = 'other-checkpoint'
        with self.assertRaises(ValueError):
            diagnostic_plan(config, self.training_config, self.summary, self.training, self.development)
        summary = deepcopy(self.summary)
        summary['m1_candidate']['passed'] = True
        with self.assertRaises(ValueError):
            diagnostic_plan(self.config, self.training_config, summary, self.training, self.development)

    def fixture_predictions(self):
        previous = read_rows(self.source / 'post_dev_thinking.jsonl')
        # CPU pairing fixture only; no new model predictions or experiment results.
        probed = [{**deepcopy(row), **overlay} for row, overlay in zip(previous, self.overlay)]
        return previous, probed

    def test_pair_categories_and_filler_drift_are_separate(self):
        previous, probed = self.fixture_predictions()
        wrong = next(i for i, r in enumerate(previous) if r['entity_partition'] == 'F' and r['task_type'] == 'prize_to_author' and not r['answer_match'])
        right = next(i for i, r in enumerate(previous) if r['entity_partition'] == 'F' and r['task_type'] == 'prize_to_author' and r['answer_match'])
        probed[wrong]['answer_match'] = True
        probed[right]['answer_match'] = False
        probed[0]['generated_token_ids'].append(999)
        report = paired_results(previous, probed)
        self.assertEqual(report['by_partition']['F']['training_only_correct'], 1)
        self.assertEqual(report['by_partition']['F']['development_only_correct'], 1)
        self.assertEqual(sum(report['by_partition']['F'].values()), 30)
        self.assertEqual(sum(report['by_partition']['R'].values()), 90)
        self.assertEqual(len(report['pairs']), 120)
        self.assertEqual(report['changed_filler_count'], 1)
        self.assertFalse(previous[wrong]['answer_match'])

    def test_prediction_pair_mismatch_or_changed_filler_input_rejected(self):
        previous, probed = self.fixture_predictions()
        with self.assertRaises(ValueError):
            paired_results(previous, probed[:-1])
        probed[0]['question'] = 'Changed filler question'
        with self.assertRaises(ValueError):
            paired_results(previous, probed)

    def test_notebook_embedded_helpers_and_configuration_match_tested_sources(self):
        nb = json.loads((ROOT / 'experiments/notebooks/14_m1_full_wording_diagnostic.ipynb').read_text())
        text = ''.join(nb['cells'][3]['source'])
        parsed = ast.parse(text)
        functions = {node.name: node for node in parsed.body if isinstance(node, ast.FunctionDef)}
        for function in (build_overlay, paired_results, diagnostic_plan):
            expected = ast.parse(inspect.getsource(function)).body[0]
            self.assertEqual(ast.dump(functions[function.__name__], include_attributes=False), ast.dump(expected, include_attributes=False))
        helper_source = '\n\n'.join(inspect.getsource(function) for function in (build_overlay, paired_results, diagnostic_plan))
        for node in parsed.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == 'embedded_helpers_sha256':
                    self.assertEqual(ast.literal_eval(node.value), hashlib.sha256(helper_source.encode()).hexdigest())
                if node.targets[0].id == 'config':
                    self.assertEqual(json.loads(ast.literal_eval(node.value.args[0])), self.config)
                if node.targets[0].id == 'diagnostic_script':
                    child = ast.parse(ast.literal_eval(node.value))
                    child_functions = {item.name: item for item in child.body if isinstance(item, ast.FunctionDef)}
                    for function in (build_overlay, paired_results, diagnostic_plan):
                        expected = ast.parse(inspect.getsource(function)).body[0]
                        self.assertEqual(ast.dump(child_functions[function.__name__], include_attributes=False), ast.dump(expected, include_attributes=False))
                    child_config = next(item.value for item in child.body if isinstance(item, ast.Assign)
                                        and isinstance(item.targets[0], ast.Name) and item.targets[0].id == 'config')
                    self.assertEqual(json.loads(ast.literal_eval(child_config.args[0])), self.config)
                    compile(ast.literal_eval(node.value), 'diagnostic_child', 'exec')
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                # Saved execution outputs are valid research evidence; compile the unchanged source.
                compile(''.join(cell['source']), 'notebook14', 'exec')

    def test_fresh_process_zip_backup_records_verified_receipt_without_credential(self):
        nb = json.loads((ROOT / 'experiments/notebooks/14_m1_full_wording_diagnostic.ipynb').read_text())
        parsed = ast.parse(''.join(nb['cells'][-1]['source']))
        script = next(ast.literal_eval(node.value) for node in parsed.body if isinstance(node, ast.Assign)
                      and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'backup_script')
        class Api:
            def __init__(self, **kwargs): pass
            def repo_info(self, *args, **kwargs): return SimpleNamespace(private=True)
            def upload_file(self, **kwargs):
                self.__class__.data = Path(kwargs['path_or_fileobj']).read_bytes()
                self.__class__.path = kwargs['path_in_repo']
                return SimpleNamespace(oid='immutable-test-revision')
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'm1-full-wording-test-evidence.zip'
            archive.write_bytes(b'temporary CPU ZIP-backup test fixture')
            def download(**kwargs):
                self.assertEqual(kwargs['revision'], 'immutable-test-revision')
                self.assertEqual(kwargs['filename'], Api.path)
                restored = Path(temp) / 'restored.zip'
                restored.write_bytes(Api.data)
                return str(restored)
            hub = ModuleType('huggingface_hub')
            hub.HfApi, hub.hf_hub_download = Api, download
            import contextlib, io, os
            with patch.dict('sys.modules', {'huggingface_hub': hub}), patch.dict(os.environ, {'HF_TOKEN': 'temporary-test-credential', 'WORDING_ARCHIVE': str(archive)}), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(script, 'zip_backup_child', 'exec'), {})
            receipt_text = Path(str(archive) + '-persistence.json').read_text()
            receipt = json.loads(receipt_text)
            self.assertTrue(receipt['zip_roundtrip_verified'])
            self.assertEqual(receipt['training_updates'], 0)
            self.assertEqual(receipt['sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertNotIn('temporary-test-credential', receipt_text)


if __name__ == '__main__':
    unittest.main()
