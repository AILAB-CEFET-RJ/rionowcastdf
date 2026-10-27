#!/usr/bin/env python3
"""Read-only exact scan of stored log1p(dBZ) patches; UTC-contiguous episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = '14b-extreme-episodes-v1'
THRESHOLDS = (30, 40, 45)


def reconstruct(patches, masks, rows, cols):
    """Deduplicate the covered footprint; reject contradictory overlapping values."""
    h, w = patches.shape[-2:]
    shape = (int(max(rows)) + h, int(max(cols)) + w)
    field = np.zeros(shape, np.float32)
    valid = np.zeros(shape, bool)
    seen = np.zeros(shape, bool)
    for p, m, r, c in zip(patches, masks, rows, cols):
        sl = np.s_[r:r+h, c:c+w]
        old, ok, was = field[sl], valid[sl], seen[sl]
        if np.any(was & (ok != m)) or np.any(was & ok & m & (old != p)):
            raise ValueError('Conflicting target/mask in overlapping patches; inspect geometry/data.')
        old[:] = p
        ok[:] = m
        was[:] = True
    return field, valid, seen


def summarize_frame(t, patches, masks, rows, cols):
    order = np.lexsort((cols, rows))
    patches, masks, rows, cols = (a[order] for a in (patches, masks, rows, cols))
    if len(set(zip(rows.tolist(), cols.tolist()))) != len(rows):
        raise ValueError('Duplicate patch anchor within timestamp')
    field, valid, seen = reconstruct(patches, masks, rows, cols)
    digest = hashlib.blake2b(digest_size=16)
    for a in (rows.astype('<i8'), cols.astype('<i8'), patches.astype('<f4'), masks.astype('u1')):
        digest.update(a.tobytes())
    result = dict(timestamp=t, n_patches=len(patches), patch_valid=int(masks.sum()),
                  unique_valid=int(valid.sum()), covered_pixels=int(seen.sum()),
                  field_hash=digest.hexdigest(), max_dbz=float(np.expm1(field[valid].max())) if valid.any() else np.nan)
    for threshold in THRESHOLDS:
        # Compare in stored float32 domain; avoid inverse-transform threshold rounding.
        cut = np.float32(np.log1p(threshold))
        for scope, a, m in [('patch', patches, masks), ('unique', field, valid)]:
            count = int(((a >= cut) & m).sum())
            result[f'{scope}_count_{threshold}'] = count
            result[f'{scope}_fraction_{threshold}'] = count / m.sum() if m.any() else np.nan
    return result


def assign_episodes(df, threshold):
    active = df[f'unique_count_{threshold}'].gt(0)
    consecutive = df.timestamp.diff().eq(pd.Timedelta(hours=1))
    start = active & (~active.shift(fill_value=False) | ~consecutive)
    return start.cumsum().where(active, 0).astype(int)


def scan(root, batch):
    ts = np.asarray(root['timestamps'][:])
    if ts.dtype.kind not in 'iu':
        raise ValueError('Expected builder Unix integer seconds in timestamps')
    times = pd.to_datetime(ts, unit='s', utc=True)
    if not times.is_monotonic_increasing or times.isna().any():
        raise ValueError('Timestamps must be ordered valid Unix seconds')
    rows, cols = (np.asarray(root[k][:], dtype=np.int64) for k in ('patch_row', 'patch_col'))
    if len(rows) != len(ts) or len(cols) != len(ts) or np.any(rows < 0) or np.any(cols < 0):
        raise ValueError('Invalid patch anchors')
    shape = root['target'].shape
    if len(shape) != 4 or shape[1] != 1 or shape != root['mask'].shape or shape[0] != len(ts) or not len(ts):
        raise ValueError('Expected matching nonempty [N,1,H,W] target/mask')
    result, pending, current = [], [], None
    for lo in range(0, len(ts), batch):
        hi = min(lo+batch, len(ts))
        yy = np.asarray(root['target'][lo:hi, 0], dtype=np.float32)
        mm = np.asarray(root['mask'][lo:hi, 0])
        if not np.isin(mm, [0, 1]).all() or not np.isfinite(yy).all() or (yy < 0).any():
            raise ValueError('Expected finite nonnegative log1p target and binary masks')
        for j, i in enumerate(range(lo, hi)):
            if current is not None and ts[i] != current:
                idx, p, m = zip(*pending)
                idx = np.asarray(idx)
                record = summarize_frame(times[idx[0]], np.stack(p), np.stack(m), rows[idx], cols[idx])
                record.update(first_row=int(idx[0]), end_row=int(idx[-1])+1)
                result.append(record)
                pending = []
            current = ts[i]
            pending.append((i, yy[j].copy(), mm[j].astype(bool)))
        if lo % (batch*100) == 0:
            print(f'Scanned {hi}/{len(ts)} patches', flush=True)
    idx, p, m = zip(*pending)
    idx = np.asarray(idx)
    record = summarize_frame(times[idx[0]], np.stack(p), np.stack(m), rows[idx], cols[idx])
    record.update(first_row=int(idx[0]), end_row=int(idx[-1])+1)
    result.append(record)
    df = pd.DataFrame(result)
    df['year'] = df.timestamp.dt.year
    df['same_field_previous_exact_1h'] = df.field_hash.eq(df.field_hash.shift()) & df.timestamp.diff().eq(pd.Timedelta(hours=1))
    return df


def episode_table(df, threshold):
    records = []
    total = df[f'unique_count_{threshold}'].sum()
    for eid, g in df[df.episode_id.gt(0)].groupby('episode_id'):
        peak = g.loc[g[f'unique_count_{threshold}'].idxmax()]
        count = int(g[f'unique_count_{threshold}'].sum())
        records.append(dict(episode_id=eid, start=g.timestamp.min(), end=g.timestamp.max(),
            observed_hours=len(g), pixel_hours=count, area_km2_hours=count*df.attrs['pixel_km']**2,
            contribution_all=count/total, peak_timestamp=peak.timestamp,
            peak_unique_fraction=peak[f'unique_fraction_{threshold}'], peak_year=int(peak.year),
            identical_1h_transitions=int(g.same_field_previous_exact_1h.sum())))
    columns = ['episode_id','start','end','observed_hours','pixel_hours','area_km2_hours','contribution_all','peak_timestamp','peak_unique_fraction','peak_year','identical_1h_transitions']
    return pd.DataFrame(records, columns=columns).sort_values('pixel_hours', ascending=False)


def yearly(df, scenario, removed):
    records = []
    for year, g0 in df.groupby('year'):
        g = g0[~g0.episode_id.isin(removed)]
        for threshold in THRESHOLDS:
            for scope in ('patch', 'unique'):
                denominator = g[f'{scope}_valid'].sum()
                records.append(dict(scenario=scenario, year=year, threshold_dbz=threshold, scope=scope,
                    original_hours=len(g0), retained_hours=len(g), removed_hours=len(g0)-len(g),
                    event_rate=float(g[f'{scope}_count_{threshold}'].gt(0).mean()),
                    mean_timestamp_fraction=float(g[f'{scope}_fraction_{threshold}'].mean()),
                    pooled_pixel_fraction=float(g[f'{scope}_count_{threshold}'].sum()/denominator) if denominator else np.nan))
    return records


def inspect_selected(root, df, episodes, args, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = set(episodes.head(args.top_episodes).episode_id)
    selected.update(episodes[episodes.peak_year < args.focus_year].head(args.historical_episodes).episode_id)
    selected.update(episodes[episodes.peak_year == args.focus_year].head(args.top_episodes).episode_id)
    cache_checks, positions = [], []
    lookup = {t: i for i, t in enumerate(df.timestamp)}
    for eid in sorted(selected):
        ep = episodes[episodes.episode_id.eq(eid)].iloc[0]
        wanted = pd.date_range(ep.peak_timestamp-pd.Timedelta(hours=2), periods=5, freq='h')
        fig, axes = plt.subplots(2, 5, figsize=(17, 7), squeeze=False)
        for col, t in enumerate(wanted):
            for ax in axes[:,col]: ax.set_axis_off()
            axes[0,col].set_title(t.strftime('%Y-%m-%d %H UTC'), fontsize=8)
            if t not in lookup:
                axes[0,col].text(.1,.5,'Timestamp missing', transform=axes[0,col].transAxes)
                continue
            row = df.loc[lookup[t]]
            sl = slice(int(row.first_row), int(row.end_row))
            p = np.asarray(root['target'][sl,0], np.float32)
            m = np.asarray(root['mask'][sl,0], bool)
            rr, cc = (np.asarray(root[k][sl], int) for k in ('patch_row','patch_col'))
            field, valid, covered = reconstruct(p,m,rr,cc)
            dbz = np.where(valid, np.expm1(field), np.nan)
            axes[0,col].imshow(dbz, vmin=0, vmax=55, cmap='turbo', origin='upper')
            axes[0,col].set_axis_on()
            axes[0,col].set_ylabel('Stored covered footprint')
            for j, (r,c) in enumerate(zip(rr,cc)):
                positions.append(dict(episode_id=eid, timestamp=t, patch_row=int(r),patch_col=int(c),
                    event_pixels=int(((p[j]>=np.float32(np.log1p(args.threshold_dbz))) & m[j]).sum()), valid_pixels=int(m[j].sum())))
            if not args.radar_cache_dir:
                axes[1,col].text(.1,.5,'Cache not supplied', transform=axes[1,col].transAxes)
                continue
            path = args.radar_cache_dir / f'{t:%Y%m%d_%H_%M}.npy'
            check = dict(episode_id=int(eid), timestamp=t, cache_file=str(path))
            try:
                raw = np.load(path, allow_pickle=False).astype(np.float32)
                if raw.ndim != 2 or raw.shape[0]<field.shape[0] or raw.shape[1]<field.shape[1]:
                    raise ValueError('Cache grid does not cover stored anchors')
                crop = raw[:field.shape[0],:field.shape[1]]
                expected = np.log1p(np.clip(np.nan_to_num(crop,nan=0,posinf=0,neginf=0),0,None)).astype(np.float32)
                check.update(status='ok', target_mismatch_pixels=int((covered & (expected != field)).sum()),
                    mask_mismatch_pixels=int((covered & (np.isfinite(crop) != valid)).sum()),
                    max_abs_log1p_difference=float(np.abs(expected[covered]-field[covered]).max()))
                axes[1,col].imshow(np.where(np.isfinite(raw),np.maximum(raw,0),np.nan),vmin=0,vmax=55,cmap='turbo',origin='upper')
                axes[1,col].set_axis_on()
                axes[1,col].set_ylabel('Full cache grid')
            except Exception as exc:
                check.update(status='missing' if not path.exists() else 'error', detail=str(exc))
                axes[1,col].text(.1,.5,check['status'], transform=axes[1,col].transAxes)
            cache_checks.append(check)
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        fig.colorbar(ScalarMappable(norm=Normalize(0,55), cmap='turbo'), ax=axes.ravel().tolist(), fraction=.015, label='dBZ')
        fig.suptitle(f'Episode {eid}: peak +/-2h | dBZ color range 0-55 | grid row/column axes')
        fig.subplots_adjust(top=.86, right=.88, hspace=.4, wspace=.35)
        fig.savefig(out/f'episode_{eid:05d}.png',dpi=140)
        plt.close(fig)
    pd.DataFrame(cache_checks, columns=list(dict.fromkeys(['episode_id','timestamp','cache_file','status','target_mismatch_pixels','mask_mismatch_pixels','max_abs_log1p_difference','detail']))).to_csv(out/'cache_checks.csv',index=False)
    pd.DataFrame(positions).to_csv(out/'selected_patch_positions.csv',index=False)
    fig, ax = plt.subplots(figsize=(15,4))
    ax.plot(df.timestamp, df[f'unique_fraction_{args.threshold_dbz}'], '.', ms=1)
    ax.set(ylabel=f'Covered unique pixel fraction >= {args.threshold_dbz} dBZ', xlabel='UTC')
    fig.tight_layout(); fig.savefig(out/'timeline.png',dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--radar-cache-dir',type=Path)
    p.add_argument('--threshold-dbz',type=int,choices=THRESHOLDS,default=45)
    p.add_argument('--batch-samples',type=int,default=512)
    p.add_argument('--pixel-km',type=float,default=2)
    p.add_argument('--focus-year',type=int,default=2024)
    p.add_argument('--top-episodes',type=int,default=10)
    p.add_argument('--historical-episodes',type=int,default=5)
    a = p.parse_args()
    if a.batch_samples<1 or a.pixel_km<=0 or min(a.top_episodes,a.historical_episodes)<0:
        p.error('Invalid batch, resolution or episode counts')
    import zarr
    root = zarr.open_group(str(a.dataset_dir/'train.zarr'),mode='r')
    for key in ('target','mask','timestamps','patch_row','patch_col'):
        if key not in root: raise ValueError(f'Missing required array: {key}')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError('Output directory must be new or empty; existing outputs are preserved')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    df = scan(root,a.batch_samples)
    df.attrs['pixel_km'] = a.pixel_km
    df['episode_id'] = assign_episodes(df,a.threshold_dbz)
    df.to_csv(a.output_dir/'timestamp_metrics.csv',index=False)
    episodes = episode_table(df,a.threshold_dbz)
    episodes.to_csv(a.output_dir/'episodes_ranked.csv',index=False)
    sensitivity = yearly(df,'all',set())
    for k in (1,5,10):
        sensitivity += yearly(df,f'without_top_{k}_global',set(episodes.head(k).episode_id))
        focus_ids = set(df.loc[df.year.eq(a.focus_year),'episode_id']) - {0}
        focus = episodes[episodes.episode_id.isin(focus_ids)]
        sensitivity += yearly(df,f'without_top_{k}_touching_{a.focus_year}',set(focus.head(k).episode_id))
    pd.DataFrame(sensitivity).to_csv(a.output_dir/'annual_sensitivity.csv',index=False)
    repeated = df[df.same_field_previous_exact_1h & df[f'unique_count_{a.threshold_dbz}'].gt(0)]
    repeated.to_csv(a.output_dir/'identical_extreme_transitions.csv',index=False)
    inspect_selected(root,df,episodes,a,a.output_dir)
    summary = dict(version=VERSION, arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        n_timestamps=len(df),n_patches=int(df.n_patches.sum()),n_episodes=len(episodes),
        identical_extreme_1h_transitions=len(repeated),
        threshold_rule='float32 target >= float32(log1p(threshold_dBZ)) with mask == 1',
        scope='Unique metrics cover stored anchors only, not the complete radar domain.',
        episode_rule='At least one covered valid event pixel; exact 1h UTC pairs only; gaps/dry hours split episodes. Not storm tracking.',
        sensitivity_rule='Diagnostic removal of whole ranked episodes; no dataset modification; rankings use unique pixel-hours.',
        cache_scope='Selected peak +/-2h frames only; no claim of full cache validation.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__ == '__main__':
    main()
