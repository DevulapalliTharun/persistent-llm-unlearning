from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from src.data.correct_candidate import correct_bundle, corrected_data
from src.data.synthetic import verify_checksums
from src.m0_knowledge_audit import sha256_file
from src.training.data import read_rows


ROOT = Path(__file__).resolve().parents[1]


class CandidateCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'configs/data/pilot_v1_1.json').read_text())
        cls.parent = Path(cls.config['parent_bundle'])
        cls.parent_hash = sha256_file(cls.parent / 'checksums.sha256')
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / 'candidate'
        correct_bundle(cls.config, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_only_declared_title_changes_and_splits_stay_fixed(self):
        for name in ('all_m1', 'new_relational', 'forget', 'retain'):
            old = read_rows(self.parent / f'entities/{name}.jsonl')
            new = read_rows(self.output / f'entities/{name}.jsonl')
            expected = deepcopy(old)
            for row in expected:
                row['dataset_version'] = self.config['dataset_version']
                if row['record_id'] == 'ku_000120':
                    row['book']['title'] = 'The Veylith Citadel'
            self.assertEqual(new, expected)
        self.assertEqual((self.parent / 'entities/split_manifest.json').read_bytes(),
                         (self.output / 'entities/split_manifest.json').read_bytes())
        self.assertEqual(sha256_file(self.parent / 'checksums.sha256'), self.parent_hash)
        self.assertEqual(verify_checksums(self.output), [])

    def test_all_prompts_preserve_ids_and_only_substitute_title(self):
        for path in (self.parent / 'prompts').glob('*.jsonl'):
            old = read_rows(path)
            new = read_rows(self.output / 'prompts' / path.name)
            expected = deepcopy(old)
            for row in expected:
                if row['record_id'] == 'ku_000120':
                    for field in ('question', 'answer'):
                        row[field] = row[field].replace('The Silver Citadel', 'The Veylith Citadel')
            self.assertEqual(new, expected)
        for name in ('n_rel_train', 'n_reason_train'):
            self.assertEqual((self.parent / f'prompts/{name}.jsonl').read_bytes(),
                             (self.output / f'prompts/{name}.jsonl').read_bytes())

    def test_tiny_selected_facts_are_unchanged(self):
        ids = set(json.loads((ROOT / 'configs/training/m1_tiny_qwen17_v1.json').read_text())['record_ids'])
        for name in ('m1_train', 'm1_dev'):
            before = [r for r in read_rows(self.parent / f'prompts/{name}.jsonl') if r['record_id'] in ids]
            after = [r for r in read_rows(self.output / f'prompts/{name}.jsonl') if r['record_id'] in ids]
            self.assertEqual(before, after)

    def test_byte_reproducibility_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            other = Path(temp) / 'second'
            correct_bundle(self.config, other)
            first_files = {str(p.relative_to(self.output)): p.read_bytes() for p in self.output.rglob('*') if p.is_file()}
            other_files = {str(p.relative_to(other)): p.read_bytes() for p in other.rglob('*') if p.is_file()}
            self.assertEqual(first_files, other_files)
        with self.assertRaises(ValueError):
            correct_bundle(self.config, self.output)

    def test_bad_parent_seed_precondition_and_new_collision_are_rejected(self):
        for key, value in (('parent_checksum_index_sha256', 'bad'), ('master_seed', 43)):
            config = deepcopy(self.config)
            config[key] = value
            with self.assertRaises(ValueError):
                corrected_data(config)
        for key, value in (('old_value', 'Wrong old title'), ('new_value', 'The Copper Yearbook')):
            config = deepcopy(self.config)
            config['correction'][key] = value
            with self.assertRaises(ValueError):
                corrected_data(config)


if __name__ == '__main__':
    unittest.main()
