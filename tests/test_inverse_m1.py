from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

from src.data.synthetic import TASKS
from src.training.data import read_rows
from src.training.evidence_zip import archive_evidence, publish_zip
from src.training.full_m1 import full_verdict, prepare_plan as reference_plan, run_training
from src.training.inverse_data import combine_with_original, inverse_examples, validate_inverse_bank
from src.training.inverse_m1 import prepare_plan, run_training as inverse_training


ROOT = Path(__file__).resolve().parents[1]


class InverseM1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'configs/training/m1_full_qwen17_inverse_v2.json').read_text())
        cls.reference = json.loads((ROOT / cls.config['reference_full_config']).read_text())
        cls.base_plan, cls.original, cls.old_dev, cls.old_baseline = reference_plan(cls.reference)
        cls.plan, cls.train, cls.dev, cls.baseline = prepare_plan(cls.config)
        cls.bank = json.loads((ROOT / cls.config['inverse_training_bank']).read_text())
        cls.entities = read_rows(ROOT / cls.config['bundle'] / 'entities/all_m1.jsonl')

    def test_old5040_targets_dev_and_baseline_remain_exact(self):
        self.assertEqual(self.train[:5040], self.original)
        self.assertEqual(self.original, read_rows(ROOT / self.config['control_dir'] / 'selected_train.jsonl'))
        self.assertEqual(self.dev, self.old_dev)
        self.assertEqual(self.baseline, self.old_baseline)
        self.assertIs(inverse_training, run_training)
        self.assertNotIn('prize_to_book', TASKS)  # Frozen source generator/evaluation is not expanded.

    def test_auxiliary_answers_paths_and_targets_match_each_canonical_entity(self):
        extra = self.train[5040:]
        self.assertEqual(len(extra), 720)
        self.assertEqual(self.plan['training_rows_by_mode'], {'answer_only': 2880, 'thinking': 2880})
        self.assertEqual(self.plan['training_rows_by_partition'], {'F': 1440, 'R': 4320})
        entities = {e['record_id']: e for e in self.entities}
        for plain, thinking in zip(extra[::2], extra[1::2]):
            entity = entities[plain['record_id']]
            self.assertEqual(plain['answer'], entity['book']['title'])
            self.assertEqual(plain['source_entity_ids'], [entity['prize']['id']])
            self.assertEqual(plain['path'], [entity['prize']['id'], entity['book']['id']])
            self.assertIn(entity['prize']['name'], plain['question'])
            self.assertNotIn(entity['book']['title'], plain['question'])
            self.assertEqual(plain['assistant_target'], 'Answer: ' + entity['book']['title'])
            self.assertEqual(thinking['assistant_target'], '<think>\n' + entity['book']['title'] + ' received ' + entity['prize']['name'] + '.\n</think>\n\nAnswer: ' + entity['book']['title'])
            self.assertFalse(plain['is_forget_target'])
            self.assertFalse(plain['eligible_metrics'])
            self.assertEqual(plain['prompt_partition'], 'm1_train_aux_inverse_book')
            self.assertEqual(plain['question'], thinking['question'])
        self.assertEqual({r['entity_partition'] for r in extra}, {'F', 'R'})
        self.assertEqual(len({r['record_id'] for r in extra}), 120)
        self.assertEqual(len({r['example_id'] for r in self.train}), 5760)

    def test_budget_near_matches_exposure_and_has_four_fixed_checkpoints(self):
        comparison = self.plan['comparison']
        self.assertEqual(self.plan['training_rows'], 5760)
        self.assertEqual(self.plan['planned_update_attempts'], 1372)
        self.assertEqual(self.plan['checkpoint_attempts'], [343, 686, 1029, 1372])
        self.assertEqual(self.plan['nominal_examples_exposed'], 21952)
        self.assertLess(abs(comparison['relative_row_exposure_difference']), 0.0005)
        self.assertEqual(self.config['pilot_gates'], self.reference['pilot_gates'])
        self.assertEqual(self.config['per_task_gates'], self.reference['per_task_gates'])
        self.assertEqual(self.plan['planned_new_evaluation_answers'], 3422)
        self.assertEqual(self.plan['baseline_versions']['peft'], '0.21.2')
        self.assertFalse(self.plan['model_executed'])

    def test_recipe_budget_mode_data_and_storage_drift_rejected(self):
        for key, value in (('optimizer_steps', 1200), ('optimizer_steps', 2400),
                           ('checkpoint_attempts', [1372]), ('initialization_policy', 'resume_old_m1'),
                           ('learning_rate', 0.001), ('evaluate_modes', ['thinking']), ('model_revision', 'main'),
                           ('control_verification_sha256', 'bad'), ('inverse_training_bank_sha256', 'bad')):
            config = deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                prepare_plan(config)
        config = deepcopy(self.config)
        config['persistence'] = self.reference['persistence']
        with self.assertRaises(ValueError):
            prepare_plan(config)

    def test_bad_template_target_leak_reserved_or_n_data_rejected(self):
        for replacement in ('Which book {book} received {prize}?', TASKS['prize_to_author']['templates'][0]):
            bank = deepcopy(self.bank)
            bank['templates'][0] = replacement
            with self.assertRaises(ValueError):
                validate_inverse_bank(bank)
        bank = deepcopy(self.bank)
        bank['templates'][1] = bank['templates'][0]
        with self.assertRaises(ValueError):
            validate_inverse_bank(bank)
        entities = deepcopy(self.entities)
        entities[0]['partition'] = 'N_rel'
        with self.assertRaises(ValueError):
            inverse_examples(entities, self.bank)
        with self.assertRaises(ValueError):
            combine_with_original(self.original, self.train[5040:-1], self.config['bundle'])

    def test_unchanged_gate_function_accepts_only_new_full_budget_and_receipts(self):
        dev = [{**r, 'answer_match': True, 'format_compliant': True, 'hit_token_limit': False, 'boundary_error': None} for r in self.dev]
        utility = [{'answer_match': True, 'format_compliant': True, 'hit_token_limit': False, 'boundary_error': None} for _ in range(10)]
        post = {'dev_answer_only': dev, 'dev_thinking': deepcopy(dev), 'utility_thinking': utility}
        receipts = [{'attempt': i, 'all_artifact_files_verified': True} for i in self.config['checkpoint_attempts']]
        self.assertTrue(full_verdict(self.config, post, utility, True, True, 1372, receipts)['passed'])
        self.assertFalse(full_verdict(self.config, post, utility, True, True, 1200, receipts)['passed'])
        self.assertFalse(full_verdict(self.config, post, utility, True, True, 1372, receipts[:-1])['passed'])
        bad = [r for r in post['dev_thinking'] if r['entity_partition'] == 'F' and r['task_type'] == 'prize_to_author']
        for r in bad[:9]:
            r['answer_match'] = False
        verdict = full_verdict(self.config, post, utility, True, True, 1372, receipts)
        self.assertFalse(verdict['checks']['per_task/thinking/F/prize_to_author'])
        self.assertFalse(verdict['passed'])


class EvidenceZipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / 'trial'
        for name in ('plan.json', 'training.jsonl', 'diagnostic_source.py', 'optimizer_state.pt',
                     'adapter/adapter_model.safetensors', 'checkpoints/attempt-000343/checkpoint.json',
                     'checkpoints/attempt-000343/artifact_manifest.json', 'checkpoints/attempt-000343/persistence.json',
                     'checkpoints/attempt-000343/optimizer_state.pt'):
            p = self.directory / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'temporary text-or-binary test fixture')

    def tearDown(self):
        self.temp.cleanup()

    def test_archive_keeps_text_and_checkpoint_metadata_but_no_model_binaries(self):
        archive = archive_evidence(self.directory)
        first = archive.read_bytes()
        self.assertEqual(archive_evidence(self.directory).read_bytes(), first)
        with zipfile.ZipFile(archive) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(len(z.namelist()), 6)
            self.assertIn('checkpoints/attempt-000343/checkpoint.json', z.namelist())
            self.assertFalse(any(name.endswith(('.pt', '.safetensors')) for name in z.namelist()))
        (self.directory / 'plan.json').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            archive_evidence(self.directory)
        self.assertEqual(archive.read_bytes(), first)

    def test_private_zip_roundtrip_receipt_excludes_credential_and_detects_corruption(self):
        archive = archive_evidence(self.directory)
        class Api:
            def repo_info(self, *args, **kwargs): return SimpleNamespace(private=True)
            def upload_file(self, **kwargs): return SimpleNamespace(oid='immutable-test-revision')
        def download(**kwargs):
            self.assertEqual(kwargs['revision'], 'immutable-test-revision')
            return str(archive)
        settings = {'repo_id': 'test/private', 'require_private': True}
        receipt = publish_zip(archive, settings, 'evidence/test', 'test-credential', Api(), download)
        self.assertTrue(receipt['zip_roundtrip_verified'])
        self.assertEqual(receipt['sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertNotIn('test-credential', json.dumps(receipt))
        corrupt = Path(self.temp.name) / 'corrupt.zip'
        corrupt.write_bytes(b'wrong')
        with self.assertRaises(ValueError):
            publish_zip(archive, settings, 'evidence/test', 'test-credential', Api(), lambda **kwargs: str(corrupt))
        with self.assertRaises(ValueError):
            publish_zip(archive, settings, 'evidence/test', None, Api(), download)
        class PublicApi(Api):
            def repo_info(self, *args, **kwargs): return SimpleNamespace(private=False)
            def upload_file(self, **kwargs): raise AssertionError('Public destination must not receive evidence')
        with self.assertRaises(ValueError):
            publish_zip(archive, settings, 'evidence/test', 'test-credential', PublicApi(), download)


if __name__ == '__main__':
    unittest.main()
