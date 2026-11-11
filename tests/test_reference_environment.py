import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec=importlib.util.spec_from_file_location('reference_environment',Path(__file__).resolve().parents[1]/'scripts/reference_environment.py')
env=importlib.util.module_from_spec(spec); spec.loader.exec_module(env)

class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); (self.root/'wheels').mkdir()
        wheel=self.root/'wheels/demo-1.0-py3-none-any.whl'
        with zipfile.ZipFile(wheel,'w') as z: z.writestr('demo-1.0.dist-info/METADATA','Name: demo\nVersion: 1.0\n')
        self.record=env.wheel_record(wheel)
        (self.root/'requirements.lock').write_text('demo==1.0 --hash=sha256:'+self.record['sha256']+'\n')
        self.manifest=dict(schema='reference-wheel-bundle-v1',host=env.host(),wheels=[self.record],lock_sha256=env.digest(self.root/'requirements.lock'))
        self.save()
    def save(self): (self.root/'manifest.json').write_text(json.dumps(self.manifest))
    def test_valid_bundle(self): self.assertEqual(env.verify_bundle(self.root)['wheels'][0]['version'],'1.0')
    def test_wheel_mutation(self):
        (self.root/'wheels'/self.record['file']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Wheel hash'): env.verify_bundle(self.root)
    def test_lock_mutation(self):
        (self.root/'requirements.lock').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Lock hash'): env.verify_bundle(self.root)
    def test_architecture_mismatch(self):
        self.manifest['host']['machine']='different'; self.save()
        with self.assertRaisesRegex(ValueError,'same Python'): env.verify_bundle(self.root)
    def test_path_escape_rejected(self):
        self.manifest['wheels'][0]['file']='../escape.whl'; self.save()
        with self.assertRaisesRegex(ValueError,'Unsafe'): env.verify_bundle(self.root)
    def test_exact_freeze_and_physicsnemo_rebuild(self):
        self.assertEqual(env.pinned_requirements('torch==2.13.0+cu129\nnvidia-physicsnemo==2.2.2\n'),['torch==2.13.0+cu129'])
    def test_unpinned_sources_rejected(self):
        for text in ('-e /tmp/source','foo @ file:///tmp/foo','foo>=1',''):
            with self.assertRaises(ValueError): env.pinned_requirements(text)
    def test_existing_environment_untouched(self):
        class Args: pass
        args=Args(); args.bundle=self.root; args.venv=self.root/'existing'
        args.venv.mkdir(); sentinel=args.venv/'keep'; sentinel.write_text('preserve')
        with self.assertRaisesRegex(ValueError,'existing environment'): env.install(args)
        self.assertEqual(sentinel.read_text(),'preserve')
    def test_commit_mismatch(self):
        with patch.object(env,'output',side_effect=['wrong','']):
            with self.assertRaisesRegex(ValueError,'differs'): env.checkout_info(self.root,'expected')

if __name__=='__main__': unittest.main()
