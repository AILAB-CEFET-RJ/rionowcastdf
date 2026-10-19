"""Synthetic CPU tests against the real NVIDIA base class and Hydra configs."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
import zarr
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from adapter.conditioning import RAW_CHANNELS, ALL20, derive_batch


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='session')
def corrdiff():
    path = Path(os.environ['CORRDIFF_DIR']).resolve()
    assert (path / 'datasets/base.py').is_file()
    sys.path.insert(0, str(path))
    return path


@pytest.fixture
def fixture_data(tmp_path):
    data, splits, ablations, stats = [tmp_path / n for n in ['data', 'splits', 'ablations', 'stats']]
    for path in (data, splits, ablations):
        path.mkdir()
    root = zarr.open_group(str(data / 'train.zarr'), mode='w')
    root.attrs.update(channels=RAW_CHANNELS, target_transform='log1p')
    raw = np.random.default_rng(42).normal(size=(60, 12, 32, 32)).astype('f4')
    raw[12:] += 10000
    root.create_dataset('input', data=raw)
    root.create_dataset('target', data=np.full((60, 1, 32, 32), np.log1p(20), dtype='f4'))
    root.create_dataset('mask', data=np.ones((60, 1, 32, 32), dtype='f4'))
    ts = np.repeat(np.array(['2015-06-01', '2021-06-01', '2023-06-01', '2024-06-01', '2020-12-31'], dtype='datetime64[s]').astype('i8'), 12)
    root.create_dataset('timestamps', data=ts)
    root.create_dataset('patch_row', data=np.tile(np.repeat([0, 16, 32], 4), 5))
    root.create_dataset('patch_col', data=np.tile([0, 16, 32, 48], 15))
    for i, name in enumerate(['train', 'validation', 'test_primary', 'test_stress_ood', 'excluded_purge']):
        np.save(splits / f'{name}_patch_indices.npy', np.arange(i * 12, (i + 1) * 12))
    catalog = json.loads((PROJECT / 'configs/phase17_channel_catalog.json').read_text())
    experiments = json.loads((PROJECT / 'configs/phase17_experiments.json').read_text())['experiments']
    for exp in experiments:
        wanted = sum([catalog['families'][f] for f in exp['raw_families']], [])
        wanted += sum([catalog['derived_families'][f] for f in exp['derived_families']], [])
        exp['channels'] = [c for c in ALL20 if c in wanted]
    (ablations / 'ablation_matrix.json').write_text(json.dumps({'experiments': experiments}))
    subprocess.run([sys.executable, str(PROJECT/'scripts/18_prepare_conditioning_stats.py'),
        '--dataset-dir', str(data), '--phase15-dir', str(splits), '--phase17-dir', str(ablations),
        '--output-dir', str(stats), '--chunk-size', '5'], check=True)
    return dict(data_path=str(data/'train.zarr'), split_indices_path=str(splits/'train_patch_indices.npy'),
        ablation_matrix_path=str(ablations/'ablation_matrix.json'), experiment_id='C00',
        stats_path=str(stats/'C00.npz'), split_name='train')


def dataset(corrdiff, kwargs):
    return load_script('18_run_native_smoke').load_dataset(dict(
        type=str(PROJECT/'adapter/rionowcast_corrdiff_dataset.py')+'::RioNowcastCorrDiffDataset', **kwargs), corrdiff)


def test_physical_derived_values():
    raw = np.zeros((1, 12, 1, 1))
    raw[:, 2] = 3; raw[:, 3] = 4
    raw[:, 1] = 290; raw[:, 4] = 280; raw[:, 8] = 250
    np.testing.assert_allclose(derive_batch(raw).reshape(-1), [5, 0, 0, -30, -10, 0, 5, 0])


def test_train_only_stats_and_all_ablation_channels(corrdiff, fixture_data):
    from datasets.base import DownscalingDataset
    args = fixture_data
    raw = zarr.open_group(args['data_path'], mode='r')['input'][:12]
    with np.load(args['stats_path']) as stats:
        np.testing.assert_allclose(stats['mean'], raw.astype('f8').mean(axis=(0, 2, 3)), atol=1e-6)
    for i in range(11):
        exp = f'C{i:02}'
        ds = dataset(corrdiff, dict(args, experiment_id=exp, stats_path=str(Path(args['stats_path']).with_name(exp+'.npz'))))
        assert isinstance(ds, DownscalingDataset)
        y, x = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))
        assert y.shape == (2, 1, 32, 32)
        assert x.shape == (2, len(ds.channels), 32, 32) and torch.isfinite(x).all()
        np.testing.assert_allclose(ds.denormalize_input(ds.normalize_input(np.ones_like(ds.mean))), 1, atol=1e-5)
        assert np.isnan(ds.latitude()).all() and ds.sample_metadata(1)['source_index'] == 1


@pytest.mark.parametrize('name', ['test_primary', 'test_stress_ood'])
def test_locked_splits_refused_before_data_open(corrdiff, name):
    with pytest.raises(RuntimeError, match='Locked split'):
        dataset(corrdiff, dict(data_path='/not/read', split_indices_path='/not/read',
            ablation_matrix_path='/not/read', experiment_id='C00', stats_path='/not/read', split_name=name))


def test_invalid_mask_is_not_silently_trained(corrdiff, fixture_data):
    zarr.open_group(fixture_data['data_path'], mode='a')['mask'][0, 0, 0, 0] = 0
    ds = dataset(corrdiff, fixture_data)
    with pytest.raises(RuntimeError, match='unmasked'):
        ds[0]


def test_wrong_calendar_refused(corrdiff, fixture_data):
    zarr.open_group(fixture_data['data_path'], mode='a')['timestamps'][0] = np.datetime64('2023-02-01', 's').astype('i8')
    with pytest.raises(RuntimeError, match='calendar'):
        dataset(corrdiff, fixture_data)


def test_wrong_channel_order_refused(corrdiff, fixture_data):
    zarr.open_group(fixture_data['data_path'], mode='a').attrs['channels'] = RAW_CHANNELS[::-1]
    with pytest.raises(RuntimeError, match='channel metadata'):
        dataset(corrdiff, fixture_data)


def test_real_hydra_composition_and_constructor(corrdiff, fixture_data, tmp_path, monkeypatch):
    installer, runner = load_script('18_install_nvidia_smoke_configs'), load_script('18_run_native_smoke')
    monkeypatch.setenv('CORRDIFF_REGRESSION_CKPT', '/compose-only.mdlus')
    args = fixture_data
    for exp in ('C00', 'C07'):
        for stage in ('regression', 'diffusion'):
            file = corrdiff/'conf'/f'config_training_rionowcast_{exp}_{stage}_smoke.yaml'
            assert not file.exists(), 'Use clean NVIDIA checkout: tests will not overwrite existing configs'
            file.write_text(installer.config_text(stage=stage, experiment_id=exp,
                adapter_path=PROJECT/'adapter/rionowcast_corrdiff_dataset.py', data_path=Path(args['data_path']),
                train_idx=Path(args['split_indices_path']), val_idx=Path(args['split_indices_path']).with_name('validation_patch_indices.npy'),
                ablation_matrix=Path(args['ablation_matrix_path']), stats_path=Path(args['stats_path']).with_name(exp+'.npz'),
                output_dir=tmp_path/(exp+stage), training_duration=64, batch_size=2))
            try:
                cfg = runner.compose_config(corrdiff, exp, stage)
                assert cfg['training']['io']['save_checkpoint_freq'] == 64
                assert cfg['training']['perf']['fp_optimizations'] == 'fp32'
                ds = runner.load_dataset(cfg['dataset'], corrdiff)
                val = runner.load_dataset(dict(cfg['dataset'], **cfg['validation']), corrdiff)
                assert len(ds) == 12 and len(val) == 12
                assert ds[0][1].shape[0] == (12 if exp == 'C00' else 20)
            finally:
                file.unlink()


def test_incomplete_evidence_never_passes(tmp_path):
    report = load_script('18_summarize_integration').evaluate(tmp_path)
    assert report['phase18_status'] == 'BLOCKED' and report['stage_r_ready'] is False


@pytest.mark.parametrize('seed', [42, 43, 44])
def test_native_seed_bridge_reproducible(seed, tmp_path):
    from types import SimpleNamespace
    def native_seed(value):
        np.random.seed(value)
        torch.manual_seed(value)
    helpers = SimpleNamespace(set_seed=native_seed)
    datasets = SimpleNamespace(init_train_valid_datasets_from_config=lambda **kwargs: kwargs['seed'])
    load_script('18_native_entrypoint').install_seed_hooks(helpers, datasets, seed, tmp_path/'seed.json')
    helpers.set_seed(0)
    first = torch.rand(4)
    helpers.set_seed(0)
    assert torch.equal(first, torch.rand(4))
    assert datasets.init_train_valid_datasets_from_config(seed=0) == seed
    report = json.loads((tmp_path/'seed.json').read_text())
    assert report['model_seeds'] == [seed, seed] and report['sampler_seeds'] == [seed]


def test_full_dataset_smoke_cli(corrdiff, fixture_data, tmp_path):
    args = fixture_data
    out = tmp_path/'dataset_smoke_audit.json'
    subprocess.run([sys.executable, str(PROJECT/'scripts/18_smoke_dataset_adapter.py'),
        '--corrdiff-dir', str(corrdiff), '--adapter-path', str(PROJECT/'adapter/rionowcast_corrdiff_dataset.py'),
        '--dataset-dir', str(Path(args['data_path']).parent), '--phase15-dir', str(Path(args['split_indices_path']).parent),
        '--phase17-dir', str(Path(args['ablation_matrix_path']).parent), '--stats-dir', str(Path(args['stats_path']).parent),
        '--output', str(out)], check=True)
    audit = json.loads(out.read_text())
    assert audit['status'] == 'PASS' and audit['all_train_validation_masks_valid']
    assert len(audit['fingerprint']) == 64


def test_stats_from_another_manifest_refused(corrdiff, fixture_data):
    path = Path(fixture_data['stats_path'])
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays['train_indices_sha256'] = np.array('wrong-manifest')
    np.savez(path, **arrays)
    with pytest.raises(RuntimeError, match='another train manifest'):
        dataset(corrdiff, fixture_data)
