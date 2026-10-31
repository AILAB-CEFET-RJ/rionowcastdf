#!/usr/bin/env python3
"""Run repository phase 13/14/15 with event flags from Phase 6 pixel counts.

The original file is not modified. This avoids the float32 expm1 threshold bug.
The executed script hash and override are logged beside the requested output.
"""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
from radar_qc import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',type=int,choices=[13,14,15],required=True)
    p.add_argument('--repo-dir',type=Path,default=Path('.'))
    a,extra=p.parse_known_args()
    if extra and extra[0]=='--':extra=extra[1:]
    names={13:('13_compute_meteorological_regimes.py','prepare_radar_metrics'),
           14:('14_compute_distribution_shift.py','prepare_radar'),
           15:('15_compute_formal_splits.py','prep_radar')}
    filename,function=names[a.phase];script=a.repo_dir/'scripts'/filename
    sys.path.insert(0,str(script.parent.resolve()))
    spec=importlib.util.spec_from_file_location('qc_wrapped_phase',script)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    original=getattr(module,function)
    def corrected(*args,**kwargs):
        out=original(*args,**kwargs)
        if 'valid_pixels' not in out or (out.valid_pixels<=0).any():
            raise ValueError('Phase 6 contains unobserved timestamps; use masked QC audit and resolve source coverage first')
        event_ids=[c[6:] for c in out.columns if c.startswith('event_') and c[6:] in ['gt_0','ge_20','ge_30','ge_40','ge_45']]
        for eid in event_ids:
            col='positive_pixels' if eid=='gt_0' else 'event_pixels_'+eid
            if col not in out:raise ValueError(f'Phase 6 missing {col}; regenerate it on the QC dataset')
            out['event_'+eid]=(out[col].to_numpy()>0).astype(np.int8)
        return out
    setattr(module,function,corrected)
    sys.argv=[str(script),*extra]
    module.main()
    if '--output-dir' in extra:
        outdir=Path(extra[extra.index('--output-dir')+1])
        (outdir/'qc_event_override.json').write_text(json.dumps(dict(script_sha256=sha256(script),
            wrapper_sha256=sha256(__file__),rule='timestamp event = Phase6 masked event_pixels > 0',
            source_file_unchanged=True,production_ready=False),indent=2))
if __name__=='__main__':main()
