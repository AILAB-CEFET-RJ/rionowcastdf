"""Read-only integrity audit of reference-zarr-v1. Run from the repository root."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import pandas as pd
import zarr
from src.corrdiff.builder.contract_common import (read, dump, sha, ro, inside,
    load_config, inputs, digest_obj, SPLITS, validate_target)
from src.corrdiff.builder.contract_zarr import patches, normalization, cache_index
from src.corrdiff.builder.contract_cache import decode_job


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(config, output, png_samples_per_policy=2):
    started = time.monotonic()
    c = load_config(config)
    root = Path(c['output_dir']).resolve()
    output = Path(output).resolve()
    # Never write reports inside any source tree.
    for key in ('output_dir', 'radar_cache_dir', 'radar_dir', 'era5_root', 'temporal_dir', 'pairing_dir', 'reuse_14j_dir'):
        if c.get(key):
            p = Path(c[key]).resolve()
            require(output != p and p not in output.parents and output not in p.parents,
                    'Audit output must be separate from inputs: ' + key)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(version='reference-integrity-v1', complete=False, integrity_passed=False,
                  production_ready=False, splits={}, raw_png_samples=0)
    dump(output/'summary.json', report)
    try:
        summary = read(root/'summary.json')
        require(summary.get('complete') and summary.get('zarr_created'), 'Incomplete dataset')
        for name, digest in summary['artifact_sha256'].items():
            require(sha(inside(root, name)) == digest, 'Artifact checksum: '+name)
        df, g, semantic = inputs(c)  # Frozen manifest, local/UTC, purge and geometry checks.
        index = cache_index(c, semantic)
        semantic['era5_inventory_sha256'] = sha(Path(c['pairing_dir'])/'era5_inventory_snapshot.sqlite')
        semantic['inventory_config'] = read(Path(c['pairing_dir'])/'run_config.json')['inventory_config']
        semantic['radar_cache_index_sha256'] = sha(Path(c['radar_cache_dir'])/'index.sqlite')
        require(read(root/'contract.json') == semantic, 'Contract/config/code changed')
        stored = pd.read_parquet(root/'selected_manifest.parquet')
        for col in ('timestamp_utc','source_utc'):
            stored[col] = pd.to_datetime(stored[col], utc=True)
        pd.testing.assert_frame_equal(stored.reset_index(drop=True), df.reset_index(drop=True), check_dtype=False)
        require(len(df) == summary['selected_hours'], 'Selected hours mismatch')
        stages = []
        locations = {}
        for month, rows in df.groupby(df.timestamp_utc.dt.strftime('%Y%m'), sort=True):
            folder = root/'staging'/month
            rec = read(folder/'complete.json')
            require(rec['hours'] == len(rows), 'Staging hours '+month)
            for name in ('input.npy','target_code.npy','mask.npy'):
                require(sha(folder/name) == rec['sha256'][name], 'Staging checksum '+month+'/'+name)
            stages.append((month, rows, folder))
            for i, stamp in enumerate(rows.timestamp_utc):
                locations[stamp] = (folder, i)
        norm = normalization(stages, c['channels'])
        saved = read(root/'normalization.json')
        for key, value in norm.items():
            if key in ('input_mean','input_std','input_std_unfloored'):
                require(np.allclose(value, saved[key], rtol=1e-10, atol=1e-10), 'Normalization '+key)
            else:
                require(value == saved[key], 'Normalization '+key)
        with np.load(root/'normalization.npz', allow_pickle=False) as nz:
            for key in ('input_mean','input_std','target_mean','target_std'):
                require(np.array_equal(nz[key], np.asarray(saved[key],np.float32)), 'NPZ normalization '+key)
        print('Manifest, contracts, staging and train-only normalization: PASS', flush=True)
        checked_objects = set()
        for split in SPLITS:
            z = zarr.open_group(str(root/(split+'.zarr')), mode='r')
            rec = read(root/(split+'.complete.json'))
            rows = df[df.split == split]
            total = rec['patches']
            require(z.attrs.get('complete') and z.attrs.get('input_normalized') is False, 'Zarr attributes '+split)
            require(z.attrs.get('contract_sha256') == digest_obj(semantic), 'Zarr contract '+split)
            require(z.attrs.get('channels') == c['channels'] and z.attrs.get('target_transform') == 'target_code / 50', 'Zarr encoding '+split)
            shapes = {'input':(total,len(c['channels']),32,32),'target':(total,1,32,32),'mask':(total,1,32,32)}
            types = {'input':'float32','target':'float32','mask':'bool','timestamps':'int64','source_timestamps':'int64',
                     'offset_seconds':'int16','patch_row':'int16','patch_col':'int16','source_sha256':'|S64'}
            hashes = {}
            for key, dtype in types.items():
                a = z[key]
                require(a.shape == shapes.get(key,(total,)) and a.dtype == np.dtype(dtype), 'Shape/dtype '+split+'/'+key)
                dims = ['sample'] + ([('input_channel' if key=='input' else 'target_channel'),'y','x'] if key in shapes else [])
                require(a.attrs.get('_ARRAY_DIMENSIONS') == dims and a.fill_value is None, 'Dimensions/fill value '+split+'/'+key)
                h = hashlib.sha256(); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); hashes[key] = h
            for key in ('lat','lon','support'):
                a = z['grid_'+key]
                require(np.array_equal(a[:],g[key]), 'Grid '+key)
                h = hashlib.sha256(); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(np.ascontiguousarray(a[:]).tobytes())
                hashes['grid_'+key] = h
            cursor = supported = zeros = used = 0
            loaded = None
            for hour, r in enumerate(rows.itertuples(index=False), 1):
                folder, i = locations[r.timestamp_utc]
                if folder != loaded:
                    xx, yy, mm = [np.load(folder/(key+'.npy'),mmap_mode='r') for key in ('input','target_code','mask')]
                    loaded = folder
                rel, digest, policy = index[r.sha]
                require(policy == r.policy_status, 'Cache policy')
                obj = inside(c['radar_cache_dir'],rel)
                if r.sha not in checked_objects:
                    require(sha(obj) == digest, 'Cache checksum '+r.sha)
                    checked_objects.add(r.sha)
                with np.load(obj,allow_pickle=False) as cached:
                    code, mask = cached['target_code'], cached['mask']
                    require(str(cached['source_sha256']) == r.sha, 'Cache source hash')
                    validate_target(code,mask,g['support'])
                    require(np.array_equal(code,yy[i,0],equal_nan=True) and np.array_equal(mask,mm[i,0]), 'Cache/staging target mismatch')
                pos = patches(mask)
                n = len(pos)
                if not n:
                    continue
                used += 1
                values = {k:z[k][cursor:cursor+n] for k in types}
                expected = {
                    'input':np.stack([xx[i,:,a:a+32,b:b+32] for a,b in pos]),
                    'mask':np.stack([mask[None,a:a+32,b:b+32] for a,b in pos]),
                    'target':np.stack([np.where(mask,code/50,0)[None,a:a+32,b:b+32] for a,b in pos]).astype('f4'),
                    'timestamps':np.full(n,int(r.timestamp_utc.timestamp()),'i8'),
                    'source_timestamps':np.full(n,int(r.source_utc.timestamp()),'i8'),
                    'offset_seconds':np.full(n,r.offset_seconds,'i2'),
                    'patch_row':np.array([p[0] for p in pos],'i2'),
                    'patch_col':np.array([p[1] for p in pos],'i2'),
                    'source_sha256':np.full(n,r.sha.encode(),'S64')}
                for key, arr in values.items():
                    require(np.array_equal(arr,expected[key]), f'{split} {r.timestamp_utc} mismatch: {key}')
                    hashes[key].update(np.ascontiguousarray(arr).tobytes())
                require(np.isfinite(values['input']).all() and np.isfinite(values['target']).all(), 'Nonfinite tensor')
                supported += int(values['mask'].sum())
                zeros += int((values['mask'] & (values['target']==0)).sum())
                cursor += n
                if hour % 500 == 0:
                    print(json.dumps(dict(split=split,hours_checked=hour,total_hours=len(rows),patches_checked=cursor)),flush=True)
            counts = dict(hours_selected=len(rows),hours_with_patches=used,patches=cursor,
                          supported_pixels_with_overlap=supported,no_echo_pixels_with_overlap=zeros)
            for key,value in counts.items():
                require(rec[key] == value and summary['splits'][split][key] == value, 'Count mismatch '+split+'/'+key)
            require({k:h.hexdigest() for k,h in hashes.items()} == rec['array_sha256'], 'Array checksum '+split)
            report['splits'][split] = counts
            dump(output/'summary.json',report)
            print(split+': PASS',flush=True)
        # Raw-source confirmation is explicitly sampled, stratified by split and policy.
        samples = []
        for _, rows in df.groupby(['split','policy_status']):
            unique = rows.drop_duplicates('sha').sort_values('timestamp_utc')
            idx = np.unique(np.linspace(0,len(unique)-1,min(png_samples_per_policy,len(unique)),dtype=int))
            samples.extend(unique.iloc[idx].to_dict('records'))
        with tempfile.TemporaryDirectory(dir=output) as temp:
            raw_c = dict(c,reuse_14j_dir=None)
            for r in samples:
                result = decode_job((r['sha'],r['path'],r['policy_status'],None),raw_c,g,Path(temp))
                with np.load(Path(temp)/result[1],allow_pickle=False) as raw, np.load(inside(c['radar_cache_dir'],index[r['sha']][0]),allow_pickle=False) as cached:
                    require(np.array_equal(raw['target_code'],cached['target_code'],equal_nan=True) and np.array_equal(raw['mask'],cached['mask']), 'PNG/cache mismatch '+r['path'])
                report['raw_png_samples'] += 1
        dump(output/'png_samples.json', [{k:str(r[k]) for k in ('timestamp_utc','split','policy_status','path','sha')} for r in samples])
        report.update(complete=True,integrity_passed=True,selected_hours=len(df),unique_cache_objects_checked=len(checked_objects),
            normalization_recomputed=True,all_patch_tensors_compared_to_staging=True,
            all_targets_compared_to_cache=True,era5_raw_recomputed=False,
            coverage_state=summary['coverage_state'],spatial_validity_confirmed=summary['spatial_validity_confirmed'],
            note='Technical integrity only. PNG comparison is sampled; ERA5 raw interpolation and physical coverage are not independently validated.')
    except Exception as exc:
        report['error'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        report['elapsed_seconds'] = round(time.monotonic()-started,2)
        report['script_sha256'] = sha(__file__)
        dump(output/'summary.json',report)
        print(json.dumps(report,indent=2),flush=True)
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract-config',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--png-samples-per-policy',type=int,default=2)
    args=p.parse_args()
    if args.png_samples_per_policy<1:p.error('PNG samples must be >=1')
    audit(args.contract_config,args.output_dir,args.png_samples_per_policy)
