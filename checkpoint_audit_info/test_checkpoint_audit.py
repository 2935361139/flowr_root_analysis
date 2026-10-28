"""Developer verification only; the user needs only inspect_flowr_checkpoint.py."""
import argparse
import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch
import inspect_flowr_checkpoint as audit


def write_sentinel(path):
    Path(path).write_text('EXECUTED')
    return {}


class ExecutableObject:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return write_sentinel, (self.path,)


class AuditTests(unittest.TestCase):
    def test_false_null_zero_and_missing_are_distinct(self):
        result = audit.extract_config({'hyper_parameters': {
            'mask_fixed_discrete_loss': False, 'type_loss_weight': 0,
            'bond_loss_weight': None}})['fields']
        self.assertIs(result['mask_fixed_discrete_loss']['occurrences'][0]['value'], False)
        self.assertEqual(result['type_loss_weight']['occurrences'][0]['value'], 0)
        self.assertIsNone(result['bond_loss_weight']['occurrences'][0]['value'])
        self.assertEqual(result['inpaint_self_condition']['status'], 'not_recorded')
        self.assertEqual(result['inpaint_self_condition']['occurrences'], [])

    def test_nested_conflicts_are_not_silently_resolved(self):
        r = audit.extract_config({'hyper_parameters': {'self_condition': True,
            'config': {'self_condition': False}}, 'hparams': {'self_condition': True}})
        f = r['fields']['self_condition']
        self.assertTrue(f['conflicting_values'])
        self.assertEqual(len(f['occurrences']), 3)
        self.assertEqual(r['fields']['self_cond']['status'], 'not_recorded')

    def test_plain_and_namespace_lightning_checkpoints_with_real_torch(self):
        with tempfile.TemporaryDirectory() as td:
            for hp in ({'self_condition': True}, argparse.Namespace(self_condition=True)):
                p = Path(td)/'small.ckpt'
                torch.save({'hyper_parameters': hp, 'state_dict': {
                    'gen.ligand_dec.inv_emb.atom_emb.0.weight': torch.ones(4, 6)},
                    'optimizer_states': [{'state': {0: {'momentum': torch.ones(3)}}}]}, p)
                original = audit.sha256(p)
                r = audit.inspect_checkpoint(torch, p, expected_sha=original)
                self.assertEqual(r['status'], 'inspected')
                self.assertIs(r['configuration']['fields']['self_condition']['occurrences'][0]['value'], True)
                shape = r['configuration']['selected_parameter_shapes'][0]['metadata']
                self.assertEqual(shape['shape'], [4, 6])
                self.assertEqual(audit.sha256(p), original)

    def test_wrong_weight_is_never_deserialized(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'wrong.ckpt';p.write_bytes(b'wrong weight')
            with patch.object(audit, 'metadata_load', side_effect=AssertionError('must not load')):
                r = audit.inspect_checkpoint(torch, p)
            self.assertFalse(r['hash_matches'])
            self.assertEqual(r['status'], 'hash_mismatch_not_loaded')

    def test_unsupported_pickle_does_not_execute_or_trigger_unsafe_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'unsafe.ckpt';sentinel = Path(td)/'sentinel'
            torch.save({'hyper_parameters': {'self_condition': True},
                        'object': ExecutableObject(sentinel)}, p)
            r = audit.inspect_checkpoint(torch, p, expected_sha=audit.sha256(p))
            self.assertEqual(r['status'], 'metadata_read_failed')
            self.assertFalse(sentinel.exists())
            self.assertIn('unsupported_pickle_globals', r)

    def test_missing_and_corrupt_lfs_are_not_called_available(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td);p = repo/'record.pt';blob = b'known';objhash = audit.hashlib.sha256(blob).hexdigest()
            p.write_text(f'version https://git-lfs.github.com/spec/v1\noid sha256:{objhash}\nsize 5\n')
            resolver = audit.Files(repo)
            self.assertEqual(resolver.inspect('record.pt')[0]['status'], 'lfs_pointer_only')
            obj = repo/'.git/lfs/objects'/objhash[:2]/objhash[2:4]/objhash
            obj.parent.mkdir(parents=True);obj.write_bytes(b'wrong')
            row, actual = resolver.inspect('record.pt', checksum=True)
            self.assertIsNone(actual)
            self.assertEqual(row['status'], 'lfs_hash_mismatch')
            obj.write_bytes(blob)
            row, actual = resolver.inspect('record.pt', checksum=True)
            self.assertEqual(actual, obj)
            self.assertTrue(row['content_verified'])

    def test_failed_runs_create_separate_small_bundles_and_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td);(repo/'flowr').mkdir();(repo/'checkpoints').mkdir()
            p = repo/'checkpoints/flowr_root_v2.2.ckpt';p.write_bytes(b'wrong')
            source = repo/'flowr/train.py';source.write_text('# local modification\n')
            before = {x: x.read_bytes() for x in [p, source]}
            for _ in range(2):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = audit.main(['--repo', str(repo)])
                self.assertEqual(code, 2)
            dirs = list((repo/'outputs').iterdir())
            self.assertEqual(len(dirs), 2)
            for d in dirs:
                r = json.loads((d/'audit.json').read_text())
                self.assertEqual(r['replay_count'], 0)
                self.assertFalse(r['checkpoint']['hash_matches'])
                self.assertEqual(r['checkpoint']['configuration']['status'], 'not_read')
                with zipfile.ZipFile(d/'checkpoint_audit_result.zip') as z:
                    self.assertIsNone(z.testzip())
                    self.assertIn('run.log', z.namelist())
                    self.assertFalse(any(n.endswith(('.pt', '.ckpt')) for n in z.namelist()))
                    self.assertEqual(z.read('source_snapshot/flowr/train.py'), before[source])
            for f, value in before.items():
                self.assertEqual(f.read_bytes(), value)

    def test_checkpoint_without_hparams_is_not_given_defaults(self):
        r = audit.extract_config({'state_dict': {}})
        self.assertEqual(r['status'], 'no_hyperparameter_mapping')
        self.assertTrue(all(v['status']=='not_recorded' for v in r['fields'].values()))

    def test_matching_checkpoint_end_to_end_reports_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td);(repo/'flowr').mkdir();(repo/'checkpoints').mkdir()
            p = repo/'checkpoints/flowr_root_v2.2.ckpt'
            torch.save({'hyper_parameters': {'mask_fixed_discrete_loss': False,
                        'type_loss_weight': 0.0, 'self_condition': True},
                        'state_dict': {'w': torch.zeros(2)}}, p)
            real_inspect = audit.inspect_checkpoint
            expected = audit.sha256(p)
            with patch.object(audit, 'inspect_checkpoint',
                              side_effect=lambda t, path: real_inspect(t, path, expected_sha=expected)):
                with contextlib.redirect_stdout(io.StringIO()):
                    audit.main(['--repo', str(repo)])
            d = next((repo/'outputs').iterdir())
            r = json.loads((d/'audit.json').read_text())
            self.assertEqual(r['checkpoint']['status'], 'inspected')
            f = r['checkpoint']['configuration']['fields']
            self.assertEqual(f['inpaint_self_condition']['status'], 'not_recorded')
            self.assertIs(f['mask_fixed_discrete_loss']['occurrences'][0]['value'], False)
            self.assertIn('mask_fixed_discrete_loss,recorded', (d/'configuration.csv').read_text())
            self.assertEqual(audit.sha256(p), expected)

    def test_wrong_project_has_no_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    audit.main(['--repo', td])
            self.assertEqual(list(Path(td).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
