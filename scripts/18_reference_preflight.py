#!/usr/bin/env python3
"""Read-only reference C00/C07 data preflight; never reads locked test splits."""
import argparse
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from adapter.reference_conditioning import VALID_CODES, reference_fingerprint


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, default=Path('datasets/corrdiff_reference_v1'))
    p.add_argument('--stats-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(argv)
    root=a.dataset_dir.expanduser().resolve()
    output=a.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if root==output or root in output.parents:
        raise ValueError('Cannot write into dataset directory')
    import numpy as np
    import torch
    from adapter.reference_corrdiff_dataset import ReferenceCorrDiffDataset
    from adapter.reference_derived_dataset import ReferenceDerivedDataset
    report=dict(schema='phase18-reference-data-preflight-v1',status='FAIL',
                target_transform='target_code / 50',locked_splits_accessed=False,
                reference_fingerprint=reference_fingerprint(root),datasets={})
    start=time.monotonic()
    try:
        for exp in ('C00','C07'):
            for split in ('train','validation'):
                if exp=='C00':
                    ds=ReferenceCorrDiffDataset(root,split)
                else:
                    ds=ReferenceDerivedDataset(root,split,stats_path=a.stats_file,allow_sampled_stats=True)
                checked=[]
                for idx in sorted(set([0,len(ds)//2,len(ds)-1])):
                    sample=ds[idx]
                    x,y,m=(sample[k] for k in ('input','target','mask'))
                    if (x.shape!=(12 if exp=='C00' else 20,32,32) or
                            y.shape!=(1,32,32) or m.shape!=y.shape or
                            m.dtype!=torch.bool or not torch.isfinite(x).all() or
                            not torch.isfinite(y[m]).all() or torch.count_nonzero(m)==0):
                        raise ValueError(f'Invalid sample {exp}/{split}/{idx}')
                    # Independent class-code check; invalid pixels are placeholders.
                    vals=y[m].numpy()
                    if not np.isin(vals,VALID_CODES).all() or (y[~m]!=0).any():
                        raise ValueError('Invalid target codes or masked placeholders')
                    checked.append(dict(index=idx,valid_pixels=int(m.sum())))
                report['datasets'][f'{exp}_{split}']=dict(length=len(ds),samples_checked=checked)
        # Verify constructors refuse test access BEFORE opening any test Zarr.
        for locked in ('test_primary','test_stress_ood'):
            for cls,kw in ((ReferenceCorrDiffDataset,{}),
                           (ReferenceDerivedDataset,dict(stats_path=a.stats_file,allow_sampled_stats=True))):
                try:
                    cls(root,locked,**kw)
                except ValueError:
                    continue
                raise RuntimeError(f'Locked split constructor accepted {locked}')
        report['status']='PASS'
    except Exception as exc:
        report['reason']=f'{type(exc).__name__}: {exc}'
    report['elapsed_seconds']=round(time.monotonic()-start,3)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'status':report['status'],'output':str(output),'reason':report.get('reason')}))
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
