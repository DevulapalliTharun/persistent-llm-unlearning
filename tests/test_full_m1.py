from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from src.training.data import read_rows
from src.training.full_artifacts import artifact_hashes, publish_verified
from src.training.full_m1 import full_verdict, prepare_plan, same_predictions
from src.training.mixed_m1 import prepare_plan as tiny_plan
from src.training.paraphrase_data import augmented_examples


ROOT = Path(__file__).resolve().parents[1]


class FullM1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'configs/training/m1_full_qwen17_v1.json').read_text())
        cls.plan, cls.train, cls.dev, cls.baseline = prepare_plan(cls.config)

    def passed_fixture(self):
        dev = [{**r, 'answer_match': True, 'format_compliant': True,
                'hit_token_limit': False, 'boundary_error': None} for r in self.dev]
        utility = [{'answer_match': True, 'format_compliant': True,
                    'hit_token_limit': False, 'boundary_error': None} for _ in range(10)]
        post = {'dev_answer_only': deepcopy(dev), 'dev_thinking': deepcopy(dev), 'utility_thinking': utility}
        receipts = [{'attempt': attempt, 'all_artifact_files_verified': True} for attempt in (300, 600, 900, 1200)]
        return post, utility, receipts

    def verdict(self, post, utility, receipts, **kwargs):
        return full_verdict(self.config, post, utility, kwargs.get('reload', True), kwargs.get('changed', True),
                            kwargs.get('updates', 1200), receipts)

    def test_full_plan_has_all_fr_units_and_excludes_later_data(self):
        self.assertEqual(len(self.train), 5040)
        self.assertEqual(len(self.dev), 840)
        self.assertEqual(self.plan['training_rows_by_mode'], {'answer_only': 2520, 'thinking': 2520})
        self.assertEqual(self.plan['training_rows_by_partition'], {'F': 1260, 'R': 3780})
        self.assertEqual({r['prompt_partition'] for r in self.train}, {'m1_train'})
        self.assertEqual({r['entity_partition'] for r in self.train}, {'F', 'R'})
        self.assertEqual(len({r['record_id'] for r in self.train}), 120)
        self.assertEqual(self.plan['planned_new_evaluation_answers'], 3422)
        self.assertEqual(self.plan['reused_base_dev_records'], 1680)
        self.assertEqual(self.plan['nominal_dataset_passes'], 120 * 16 / 504)
        self.assertFalse(self.plan['model_executed'])
        for mode in ('answer_only', 'thinking'):
            self.assertEqual(len(self.baseline[mode]), 840)

    def test_tiny_defaults_and_saved_targets_are_unchanged(self):
        config = json.loads((ROOT / 'configs/training/m1_tiny_qwen17_v1.json').read_text())
        _, train, dev = tiny_plan(config)
        self.assertEqual(len(train), 504)
        self.assertEqual(len(dev), 84)
        self.assertEqual(train, read_rows(ROOT / 'experiments/runs/m1_tiny_qwen17_attempt_20261106/selected_train.jsonl'))
        with self.assertRaises(ValueError):
            augmented_examples(self.config)  # Default API remains a twelve-unit diagnostic.

    def test_budget_modes_gates_baseline_and_storage_cannot_drift(self):
        mutations = [('optimizer_steps', 2400), ('checkpoint_attempts', [1200]),
                     ('checkpoint_selection', 'best_dev'), ('evaluate_modes', ['thinking']),
                     ('primary_research_mode', 'answer_only'), ('model_revision', 'main'),
                     ('per_task_gates', {'one_hop_accuracy_min': 0.5, 'multi_hop_accuracy_min': 0.5}),
                     ('data_review_sha256', 'bad'), ('baseline_verification_sha256', 'bad'),
                     ('utility_reference_sha256', 'bad'), ('record_ids', self.config['record_ids'][:-1])]
        for key, value in mutations:
            config = deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare_plan(config)
        config = deepcopy(self.config)
        config['persistence']['require_private'] = False
        with self.assertRaises(ValueError):
            prepare_plan(config)

    def test_per_task_gate_detects_failure_hidden_by_aggregate(self):
        post, utility, receipts = self.passed_fixture()
        self.assertTrue(self.verdict(post, utility, receipts)['passed'])
        affected = [r for r in post['dev_thinking'] if r['entity_partition'] == 'F' and r['task_type'] == 'author_to_year']
        for row in affected[:8]:
            row['answer_match'] = False
        verdict = self.verdict(post, utility, receipts)
        self.assertFalse(verdict['passed'])
        self.assertTrue(verdict['checks']['dev_thinking/F/multi_hop'])
        self.assertFalse(verdict['checks']['per_task/thinking/F/author_to_year'])

    def test_nonforget_one_hop_attributes_must_be_learned(self):
        post, utility, receipts = self.passed_fixture()
        affected = [r for r in post['dev_answer_only'] if r['entity_partition'] == 'R' and r['task_type'] == 'book_to_prize']
        for row in affected[:10]:
            row['answer_match'] = False
        verdict = self.verdict(post, utility, receipts)
        self.assertFalse(verdict['passed'])
        self.assertFalse(verdict['checks']['per_task/answer_only/R/book_to_prize'])

    def test_coverage_cap_boundary_update_reload_and_backup_failures_block(self):
        post, utility, receipts = self.passed_fixture()
        for key, value in (('updates', 1199), ('reload', False), ('changed', False)):
            self.assertFalse(self.verdict(post, utility, receipts, **{key: value})['passed'])
        self.assertFalse(self.verdict(post, utility, receipts[:-1])['passed'])
        for field, value in (('hit_token_limit', True), ('boundary_error', 'unclosed'), ('format_compliant', False)):
            altered = deepcopy(post)
            altered['dev_thinking'][0][field] = value
            self.assertFalse(self.verdict(altered, utility, receipts)['passed'])
        post['dev_thinking'].pop()
        self.assertFalse(self.verdict(post, utility, receipts)['passed'])

    def test_reload_and_replay_compare_tokens_not_only_final_answers(self):
        old = [{'example_id': 'one', 'generated_token_ids': [1, 2], 'batch_seconds': 1.0, 'evidence_origin': {'kind': 'parent'}}]
        new = [{'example_id': 'one', 'generated_token_ids': [1, 2], 'batch_seconds': 2.0}]
        self.assertTrue(same_predictions(old, new))
        new[0]['generated_token_ids'] = [1, 3]
        self.assertFalse(same_predictions(old, new))


class FakeApi:
    def __init__(self, private=True):
        self.private = private
        self.files = {}
        self.uploads = 0

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(private=self.private)

    def upload_folder(self, **kwargs):
        self.uploads += 1
        folder = Path(kwargs['folder_path'])
        self.files = {kwargs['path_in_repo'] + '/' + str(p.relative_to(folder)): p.read_bytes()
                      for p in folder.rglob('*') if p.is_file()}
        return SimpleNamespace(oid='immutable-test-revision')

    def upload_file(self, **kwargs):
        self.files[kwargs['path_in_repo']] = Path(kwargs['path_or_fileobj']).read_bytes()

    def download(self, **kwargs):
        path = Path(kwargs['local_dir']) / kwargs['filename']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[kwargs['filename']])
        return str(path)


class FullArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        for name in ('adapter/adapter_model.safetensors', 'adapter/adapter_config.json',
                     'tokenizer/tokenizer_config.json', 'optimizer_state.pt'):
            path = self.folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'temporary CPU test fixture, not model weights')
        self.hashes = artifact_hashes(self.folder)
        (self.folder / 'artifact_manifest.json').write_text(json.dumps({'artifact_files_sha256': self.hashes}))
        self.settings = {'repo_id': 'test/private', 'path_prefix': 'shared/test',
                         'require_private': True, 'checkpoint_roundtrip_required': True}

    def tearDown(self):
        self.temp.cleanup()

    def test_private_roundtrip_checks_all_saved_artifacts_and_records_revision(self):
        api = FakeApi()
        receipt = publish_verified(self.folder, self.settings, 'shared/test/run/step', 'test-credential', api, api.download)
        self.assertTrue(receipt['all_artifact_files_verified'])
        self.assertEqual(receipt['artifact_files_sha256'], self.hashes)
        self.assertEqual(receipt['artifact_revision'], 'immutable-test-revision')
        self.assertNotIn('test-credential', json.dumps(receipt))
        self.assertIn('shared/test/run/step/persistence.json', api.files)

    def test_public_repo_wrong_prefix_and_changed_artifacts_never_upload(self):
        api = FakeApi(private=False)
        with self.assertRaises(ValueError):
            publish_verified(self.folder, self.settings, 'shared/test/run', 'test-credential', api, api.download)
        self.assertEqual(api.uploads, 0)
        api = FakeApi()
        with self.assertRaises(ValueError):
            publish_verified(self.folder, self.settings, 'other/run', 'test-credential', api, api.download)
        (self.folder / 'optimizer_state.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            publish_verified(self.folder, self.settings, 'shared/test/run', 'test-credential', api, api.download)
        self.assertEqual(api.uploads, 0)

    def test_corrupt_remote_bytes_or_missing_tokenizer_cannot_verify(self):
        api = FakeApi()
        def corrupt(**kwargs):
            path = Path(api.download(**kwargs))
            path.write_bytes(b'corrupt')
            return str(path)
        with self.assertRaises(ValueError):
            publish_verified(self.folder, self.settings, 'shared/test/run', 'test-credential', api, corrupt)
        (self.folder / 'tokenizer/tokenizer_config.json').unlink()
        with self.assertRaises(ValueError):
            artifact_hashes(self.folder)


if __name__ == '__main__':
    unittest.main()
