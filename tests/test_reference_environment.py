"""Offline unit tests for Fase 18 reference_environment utilities."""
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'reference_environment.py'
if not SCRIPT.is_file():
    SCRIPT = ROOT.parent / 'scripts' / 'reference_environment.py'
spec = importlib.util.spec_from_file_location('reference_environment', SCRIPT)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def wheel_file(where, name='nvidia-cudnn-cu12', version='9.20.0.48',
               native=('nvidia/cudnn/lib/libcudnn.so.9', b'expected')):
    filename = f'{name.replace("-", "_")}-{version}-py3-none-any.whl'
    path = where / filename
    info = f'{name.replace("-", "_")}-{version}.dist-info/METADATA'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(info, f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
        if native:
            z.writestr(*native)
    return path


class SnapshotTests(unittest.TestCase):
    def test_direct_remote_requires_explicit_url(self):
        url = 'https://github.com/NVIDIA/warp/releases/download/v1.19.0/warp_lang-1.19.0+cu12-py3-none-any.whl'
        with patch.object(ref.md, 'version', return_value='1.19.0+cu12'):
            with self.assertRaises(ValueError):
                ref.pinned_requirements('warp-lang @ '+url)
            pins, origins = ref.pinned_requirements('warp-lang @ '+url, {'warp-lang': url})
        self.assertEqual(pins, ['warp-lang==1.19.0+cu12'])
        self.assertEqual(origins[0]['origin'], url)

    def test_approved_direct_url_with_pip_freeze_sha256(self):
        url = ('https://github.com/NVIDIA/warp/releases/download/v1.19.0/'
               'warp_lang-1.19.0+cu12-py3-none-manylinux_2_28_x86_64.whl')
        hash_value = 'f95d9581ed12663df272e45a50101b171e143fde1ef7b2433a8bfc9fba096598'
        frozen = f'warp-lang @ {url}#sha256={hash_value}'
        with patch.object(ref.md, 'version', return_value='1.19.0+cu12'):
            pins, converted = ref.pinned_requirements(frozen, {'warp-lang': url})
        self.assertEqual(pins, ['warp-lang==1.19.0+cu12'])
        self.assertEqual(converted[0]['origin'], url + '#sha256=' + hash_value)
        self.assertEqual(converted[0]['source_sha256'], hash_value)

    def test_reject_changed_direct_url_and_unsupported_fragment(self):
        base = 'https://example.org/files/warp_lang-1.19.0+cu12-py3-none-any.whl'
        with patch.object(ref.md, 'version', return_value='1.19.0+cu12'):
            for origin in (base + '#sha256=abc', base + '#egg=warp-lang',
                           'https://other.example.org/files/warp_lang-1.19.0+cu12-py3-none-any.whl'):
                with self.subTest(origin=origin), self.assertRaises(ValueError):
                    ref.pinned_requirements('warp-lang @ ' + origin, {'warp-lang': base})

    def test_refuse_unsafe_urls_and_editables(self):
        for case in ('bad::https://example.org/a.whl',
                     'warp-lang=http://example.org/a.whl',
                     'warp-lang=https://a.example/p.whl?token=abc'):
            with self.subTest(case=case), self.assertRaises(ValueError):
                ref.named_wheel_urls([case])
        with self.assertRaises(ValueError):
            ref.pinned_requirements('-e git+https://example.org/pkg.git')

    def test_local_conda_reference(self):
        with patch.object(ref.md, 'version', return_value='1.2.3'):
            pins, converted = ref.pinned_requirements(
                'asttokens @ file:///home/conda/build_artifacts/foo\npython-dateutil==2.9.0')
        self.assertEqual(pins[0], 'asttokens==1.2.3')
        self.assertEqual(len(converted), 1)


class NativeAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.bundle = self.base / 'bundle'
        (self.bundle / 'wheels').mkdir(parents=True)
        self.wheel = wheel_file(self.bundle / 'wheels')
        self.manifest = {'wheels': [ref.wheel_record(self.wheel)]}
        self.site = self.base / 'site-packages'
        (self.site / 'nvidia/cudnn/lib').mkdir(parents=True)
        self.library = self.site / 'nvidia/cudnn/lib/libcudnn.so.9'

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_native_library(self):
        result = ref.native_audit(self.bundle, self.manifest, self.site)
        self.assertFalse(result['passed'])
        self.assertEqual(result['problems'][0]['type'], 'missing')

    def test_finalize_checks_offline_dependency_preflight(self):
        physics = wheel_file(self.bundle / 'wheels', name='nvidia-physicsnemo',
                             version='2.3.0a0', native=None)
        requirements = ['nvidia-cudnn-cu12==9.20.0.48']
        (self.bundle/'source-freeze.txt').write_text('fake-freeze\n')
        (self.bundle/'dependencies.txt').write_text('\n'.join(requirements) + '\n')
        info = {'commit':'abc', 'dirty':False}
        with patch.object(ref, 'preflight') as preflight:
            ref.finish_bundle(self.bundle, info, requirements, [])
        preflight.assert_called_once()
        manifest = ref.verify_bundle(self.bundle)
        self.assertEqual(manifest['schema'], 'reference-wheel-bundle-v2')
        self.assertTrue(manifest['preflight_passed'])
        self.assertEqual(manifest['native_library_count'], 1)

    def test_finish_rejects_direct_url_hash_mismatch(self):
        physics = wheel_file(self.bundle / 'wheels', name='nvidia-physicsnemo',
                             version='2.3.0a0', native=None)
        (self.bundle/'source-freeze.txt').write_text('fake-freeze\n')
        (self.bundle/'dependencies.txt').write_text('nvidia-cudnn-cu12==9.20.0.48\n')
        conversion = [{'package': 'nvidia-cudnn-cu12',
                       'origin': 'https://example.org/wheel.whl#sha256=' + ('a' * 64),
                       'resolved_version': '9.20.0.48', 'source_sha256': 'a' * 64}]
        with self.assertRaisesRegex(ValueError, 'Direct wheel SHA256 mismatch'):
            ref.finish_bundle(self.bundle, {'commit': 'abc'},
                              ['nvidia-cudnn-cu12==9.20.0.48'], conversion)
        self.assertFalse((self.bundle/'manifest.json').exists())

    def test_hash_matches(self):
        self.library.write_bytes(b'expected')
        result = ref.native_audit(self.bundle, self.manifest, self.site)
        self.assertTrue(result['passed'])
        self.assertEqual(result['native_files_checked'], 1)

    def test_replaced_and_orphan_libraries(self):
        self.library.write_bytes(b'from-other-wheel')
        (self.library.parent/'libcudnn_ext.so.9').write_bytes(b'unowned')
        result = ref.native_audit(self.bundle, self.manifest, self.site)
        self.assertFalse(result['passed'])
        self.assertEqual({p['type'] for p in result['problems']}, {'mismatch', 'orphan'})

    def test_quarantine_does_not_delete(self):
        self.library.write_bytes(b'wrong')
        orphan = self.library.parent / 'libcudnn_ext.so.9'
        orphan.write_bytes(b'unowned')
        result = ref.native_audit(self.bundle, self.manifest, self.site)
        venv = self.base / 'venv'
        py = venv / 'bin/python'
        with patch.object(ref, 'run') as runner:
            actions = ref.repair_native_from_bundle(self.bundle, self.manifest, py, self.site, result)
        self.assertEqual(len(actions), 2)
        runner.assert_called_once()
        self.assertFalse(orphan.exists())
        self.assertTrue((venv/'reference_native_quarantine/nvidia/cudnn/lib/libcudnn_ext.so.9').exists())

    def test_top_level_cpython_extension_accepted(self):
        """A root-level .so is a valid wheel module, not an unsafe path."""
        top = '_duckdb.cpython-312-x86_64-linux-gnu.so'
        other = wheel_file(self.bundle / 'wheels', name='duckdb',
                           version='1.4.0', native=(top, b'duckdb-native'))
        manifest = {'wheels': [ref.wheel_record(other)]}
        members = ref.wheel_native_members(self.bundle, manifest)
        self.assertEqual(members[top], other.name)
        (self.site / top).write_bytes(b'duckdb-native')
        audited = ref.native_audit(self.bundle, manifest, self.site)
        self.assertTrue(audited['passed'])
        self.assertEqual(audited['native_files_checked'], 1)

    def test_top_level_cpython_extension_hash_mismatch_detected(self):
        top = '_duckdb.cpython-312-x86_64-linux-gnu.so'
        other = wheel_file(self.bundle / 'wheels', name='duckdb',
                           version='1.4.0', native=(top, b'duckdb-native'))
        (self.site / top).write_bytes(b'changed')
        audited = ref.native_audit(self.bundle,
                                  {'wheels': [ref.wheel_record(other)]}, self.site)
        self.assertFalse(audited['passed'])
        self.assertEqual(audited['problems'][0]['type'], 'mismatch')

    def test_unsafe_native_paths_still_rejected(self):
        for posix_name in ('../escape.so', '/tmp/escape.so',
                           'pkg/../escape.so', 'pkg//escape.so',
                           'pkg\\escape.so', 'bad.data/platlib/libhack.so'):
            with self.subTest(member=posix_name):
                p = wheel_file(self.bundle / 'wheels', name='unsafe',
                               version='1.0.0', native=(posix_name, b'bad'))
                with self.assertRaisesRegex(ValueError, 'Unsupported/unsafe'):
                    ref.wheel_native_members(self.bundle,
                        {'wheels': [ref.wheel_record(p)]})

    def test_top_level_extension_collisions_still_rejected(self):
        top = '_collision.cpython-312-x86_64-linux-gnu.so'
        first = wheel_file(self.bundle / 'wheels', name='first',
                           version='1.0.0', native=(top, b'first'))
        second = wheel_file(self.bundle / 'wheels', name='second',
                            version='1.0.0', native=(top, b'second'))
        with self.assertRaisesRegex(ValueError, 'Overlapping native library'):
            ref.wheel_native_members(self.bundle, {'wheels': [
                ref.wheel_record(first), ref.wheel_record(second)]})

    def test_collision_rejected(self):
        other = wheel_file(self.bundle/'wheels', name='other', version='1.0.0')
        self.manifest['wheels'].append(ref.wheel_record(other))
        with self.assertRaisesRegex(ValueError, 'Overlapping native library'):
            ref.wheel_native_members(self.bundle, self.manifest)

    def test_wheel_reuse_hardlink(self):
        manifest = dict(schema='reference-wheel-bundle-v1', host=ref.host(),
                        checkout={'commit':'abc'}, wheels=self.manifest['wheels'],
                        lock_sha256='abc')
        with patch.object(ref, 'verify_bundle', return_value=manifest):
            dest = self.base/'reused'
            dest.mkdir()
            result = ref.reuse_wheels(self.bundle, dest,
                                      {'nvidia-cudnn-cu12':'9.20.0.48'}, 'abc')
        self.assertEqual(result['nvidia-cudnn-cu12'], 'hardlink')
        self.assertEqual((dest / self.wheel.name).stat().st_ino, self.wheel.stat().st_ino)


class CudaTargetProjectionTests(unittest.TestCase):
    def test_mixed_cuda_snapshot_requires_explicit_target(self):
        pins = ['torch==2.13.0+cu129', 'nvidia-cudnn-cu12==9.20.0.48',
                'nvidia-cudnn-cu13==9.24.0.43', 'nvidia-curand-cu13==10.4.0',
                'cupy-cuda13x==14.2.0', 'nvidia-dali-cuda130==2.3.0',
                'warp-lang==1.19.0+cu12']
        with self.assertRaisesRegex(ValueError, 'CUDA 13 distributions'):
            ref.cuda12_projection(pins)
        target, excluded = ref.cuda12_projection(pins, 'cu12')
        self.assertEqual(target, [pins[0], pins[1], pins[-1]])
        self.assertEqual(len(excluded), 4)

    def test_cuda13_torch_fails_cuda12_target(self):
        with self.assertRaisesRegex(ValueError, 'CUDA 13 PyTorch'):
            ref.cuda12_projection(['torch==2.13.0+cu130'], 'cu12')

    def test_unlink_cuda13_wheel_preserves_other_bundle_hardlink(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'old').mkdir()
            (path / 'new' / 'wheels').mkdir(parents=True)
            original = wheel_file(path / 'old', name='nvidia-cudnn-cu13', version='9.24.0.43')
            linked = path / 'new' / 'wheels' / original.name
            import os
            os.link(original, linked)
            removed = ref.remove_excluded_wheels(path / 'new', [
                {'package': 'nvidia-cudnn-cu13', 'version': '9.24.0.43'}])
            self.assertEqual(len(removed), 1)
            self.assertTrue(original.exists())
            self.assertFalse(linked.exists())

    def test_finalize_cuda12_projection_keeps_source_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'bundle'
            wheels = root / 'wheels'
            wheels.mkdir(parents=True)
            wheel_file(wheels, name='nvidia-cudnn-cu12', version='9.20.0.48')
            wheel_file(wheels, name='nvidia-cudnn-cu13', version='9.24.0.43')
            wheel_file(wheels, name='nvidia-physicsnemo', version='2.3.0a0', native=None)
            frozen = 'nvidia-cudnn-cu12==9.20.0.48\nnvidia-cudnn-cu13==9.24.0.43'
            (root / 'source-freeze.txt').write_text(frozen + '\n')
            (root / 'dependencies.txt').write_text(frozen + '\n')
            (root / 'local-origin-resolutions.json').write_text('[]\n')
            (root / 'explicit-wheel-urls.json').write_text('{}\n')
            from types import SimpleNamespace
            args = SimpleNamespace(bundle=str(root), physicsnemo_dir='/physicsnemo',
                                   expected_commit='abc', cuda_family='cu12')
            with patch.object(ref, 'run'), patch.object(ref, 'output', return_value=frozen), \
                 patch.object(ref, 'checkout_info', return_value={'commit':'abc', 'dirty':False}), \
                 patch.object(ref, 'preflight') as preflight:
                code = ref.finalize(args)
            self.assertEqual(code, 0)
            self.assertFalse((wheels / 'nvidia_cudnn_cu13-9.24.0.43-py3-none-any.whl').exists())
            self.assertTrue((root / 'manifest.json').exists())
            result = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(result['cuda_family'], 'cu12')
            self.assertEqual(result['excluded_packages'][0]['package'], 'nvidia-cudnn-cu13')
            self.assertIn('nvidia-cudnn-cu12==9.20.0.48', (root / 'requirements.lock').read_text())
            self.assertNotIn('cu13', (root / 'requirements.lock').read_text())
            self.assertEqual((root / 'source-freeze.txt').read_text(), frozen + '\n')
            preflight.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
