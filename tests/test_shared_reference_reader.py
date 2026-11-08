import contextlib
import io
import shutil
import sys
from unittest.mock import patch
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import zarr
from src.corrdiff.diagnostics.reference_reader import ReferenceReader, CODES


def make(root):
    contract = {'coverage_state':'provisional_geometric_region'}
    norm = dict(channels=['a','b'], input_mean=[10,20], input_std=[2,4])
    for name, obj in [('contract.json', contract), ('normalization.json', norm)]:
        (root/name).write_text(json.dumps(obj))
    (root/'summary.json').write_text(json.dumps(dict(complete=True, channels=['a','b'],
        artifact_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                        for name in ['contract.json','normalization.json']})))
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    z = zarr.open_group(str(root/'train.zarr'), mode='w')
    rows, cols = list(range(0,39,8))+[38], list(range(0,53,8))+[52]
    pos = [(r,c) for r in rows for c in cols]
    x = np.stack([np.full((70,84),14),np.full((70,84),28)]).astype('f4')
    y = np.zeros((1,70,84), dtype='f4'); y[0,10,10]=.4
    m = np.ones_like(y, dtype=bool);m[0,0,0]=False
    for key,v in [('input',x),('target',y),('mask',m)]:
        z.create_dataset(key,data=np.stack([v[:,r:r+32,c:c+32] for r,c in pos]))
    for key, values in dict(timestamps=[1000]*48, source_timestamps=[880]*48,
        offset_seconds=[-120]*48,patch_row=[r for r,c in pos],patch_col=[c for r,c in pos],
        source_sha256=np.array(['a'*64]*48,dtype='S64')).items():
        z.create_dataset(key,data=np.asarray(values))
    for k in ['lat','lon','support']:
        z.create_dataset('grid_'+k,data=np.ones((70,84)))
    z.attrs.update(complete=True,version='reference-zarr-v1', split='train',
        contract_sha256=digest,input_normalized=False,target_transform='target_code / 50',channels=['a','b'])
    zarr.consolidate_metadata(str(root/'train.zarr'))
    return z


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.z=make(self.root)

    def test_physical_and_standardized_channel_subset(self):
        np.testing.assert_array_equal(ReferenceReader(self.root)[0]['input'][:,1,1],[14,28])
        r=ReferenceReader(self.root,input_mode='standardized',channels=['b','a'])
        np.testing.assert_array_equal(r[0]['input'][:,1,1],[2,2])

    def test_mask_and_echo_boundary(self):
        b=ReferenceReader(self.root)[0]
        self.assertTrue(np.isnan(b['target_code'][0,0,0]))
        self.assertFalse(b['no_echo'][0,0,0])
        self.assertTrue(b['no_echo'][0,1,1])
        self.assertTrue(ReferenceReader.echo_at_least(b,20)[0,10,10])
        self.assertFalse(ReferenceReader.echo_at_least(b,25)[0,10,10])

    def test_hour_reconstruction_unique_points_edges(self):
        h=ReferenceReader(self.root).hour(1000)
        self.assertEqual(h['mask'].sum(),5879)
        self.assertEqual((h['patch_count']>0).sum(),5880)
        self.assertEqual(h['patch_count'].sum(),48*32*32)
        self.assertEqual(h['target_code'][0,10,10],20)

    def test_overlap_disagreement(self):
        self.z['input'][1,0,1,1]=99
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            ReferenceReader(self.root).hour(1000)

    def test_unknown_placeholder_rejected(self):
        self.z['target'][0,0,0,0]=.4
        with self.assertRaisesRegex(ValueError,'placeholder'):
            ReferenceReader(self.root)[0]

    def test_temporal_offset_rejected(self):
        self.z['offset_seconds'][0]=0
        with self.assertRaisesRegex(ValueError,'offset'):
            ReferenceReader(self.root)[0]

    def test_hash_and_contract_rejected(self):
        (self.root/'normalization.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'hash'):
            ReferenceReader(self.root)

    def test_cli_checks_four_splits(self):
        from scripts.check_reference_reader import main
        for split in ('validation','test_primary','test_stress_ood'):
            shutil.copytree(self.root/'train.zarr',self.root/(split+'.zarr'))
            z=zarr.open_group(str(self.root/(split+'.zarr')),mode='a')
            z.attrs['split']=split
            zarr.consolidate_metadata(str(self.root/(split+'.zarr')))
        with tempfile.TemporaryDirectory() as output:
            path=Path(output)/'check'
            with patch.object(sys,'argv',['check','--dataset-dir',str(self.root),
                    '--output-dir',str(path)]), contextlib.redirect_stdout(io.StringIO()):
                main()
            report=json.loads((path/'summary.json').read_text())
            self.assertTrue(report['reader_check_passed'])
            self.assertEqual(len(report['splits']),4)

    def test_batch_bounds_and_iteration(self):
        r=ReferenceReader(self.root)
        self.assertEqual(sum(len(b['input']) for b in r.iter_batches(7)),48)
        with self.assertRaises(IndexError):r[-1]
        with self.assertRaises(ValueError):ReferenceReader(self.root,split='../train')
        with self.assertRaises(ValueError):ReferenceReader(self.root,input_mode='log1p')


if __name__=='__main__':unittest.main()
