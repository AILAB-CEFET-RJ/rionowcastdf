import json
import subprocess
from pathlib import Path
import numpy as np
from .common import csv,dump,sha,derived


class StageBlocked(RuntimeError):
    pass


def phase16(ctx,out):
    """Simple pixelwise ridge and no-echo baselines, trained on train only."""
    r=ctx.reader('train');rng=np.random.default_rng(ctx.c['seed']);p=len(r.channels)+1
    xx=np.zeros((p,p));xy=np.zeros(p);n=0;sy=0.
    for t,h in ctx.fields('train',ctx.c['baseline_train_hours']):
        m=h['mask'][0]&np.isfinite(h['input']).all(0);points=np.flatnonzero(m)
        points=rng.choice(points,min(len(points),ctx.c['sample_pixels_per_hour']),replace=False)
        if not len(points):continue
        x=((h['input']-r.mean)/r.std).reshape(p-1,-1)[:,points].T.astype('f8')
        x=np.column_stack([np.ones(len(x)),x]);y=h['target'][0].ravel()[points].astype('f8')
        xx+=x.T@x;xy+=x.T@y;n+=len(y);sy+=y.sum()
    if n==0:raise ValueError('No valid train points for baseline')
    penalty=np.eye(p)*ctx.c['ridge_alpha'];penalty[0,0]=0
    beta=np.linalg.solve(xx/n+penalty,xy/n);constant=sy/n
    np.savez(out/'ridge_train_fit.npz',beta=beta,channels=r.channels,input_mean=r.mean,input_std=r.std,train_mean_target=constant)
    rows=[]
    for split in ('train','validation'):
        totals={name:dict(n=0,se=0.,ae=0.,hits=0,misses=0,false_alarms=0) for name in ('no_echo','train_mean','ridge')}
        hours=0
        for t,h in ctx.fields(split,ctx.c['baseline_eval_hours']):
            m=h['mask'][0]&np.isfinite(h['input']).all(0);truth=h['target'][0][m].astype('f8')
            x=((h['input']-r.mean)/r.std)[:,m].T.astype('f8')
            predictions={'no_echo':np.zeros_like(truth),'train_mean':np.full_like(truth,constant),
                         'ridge':np.clip(beta[0]+x@beta[1:],0,1)}
            for name,pred in predictions.items():
                q=totals[name];e=pred-truth;q['n']+=len(e);q['se']+=float((e*e).sum());q['ae']+=float(abs(e).sum())
                a,b=pred>=float(np.float32(.4)),truth>=float(np.float32(.4))
                q['hits']+=int((a&b).sum());q['misses']+=int((~a&b).sum());q['false_alarms']+=int((a&~b).sum())
            hours+=1
        for name,q in totals.items():
            n_eval=q['n'];den=q['hits']+q['misses']+q['false_alarms']
            rows.append(dict(split=split,model=name,hours=hours,valid_pixels=n_eval,mse=q['se']/n_eval if n_eval else None,
                mae=q['ae']/n_eval if n_eval else None,csi20=q['hits']/den if den else None,**{k:q[k] for k in ('hits','misses','false_alarms')}))
    csv(out/'baseline_metrics.csv',rows)
    return dict(passed=True,scope='pixelwise baselines; sampled train fit and train/validation evaluation',train_points=n,
                target_units='code / 50',ridge_prediction_clip=[0,1],native_corrdiff=False,test_evaluation=False)


def phase17(ctx,out):
    configs=Path(ctx.c['ablation_config_dir'])
    catalog=json.loads((configs/'phase17_channel_catalog.json').read_text())
    experiments=json.loads((configs/'phase17_experiments.json').read_text())['experiments']
    available=set(ctx.reader('train').channels)|set(derived(np.zeros((12,1,1)),ctx.reader('train').channels))
    cases=[]
    for e in experiments:
        channels=[]
        for f in e['raw_families']:channels+=catalog['families'][f]
        for f in e.get('derived_families',[]):channels+=catalog['derived_families'][f]
        if len(set(channels))!=len(channels) or not set(channels)<=available:raise ValueError('Invalid ablation channels')
        canonical=catalog['raw_channels_canonical_order']+catalog['derived_channels_canonical_order']
        channels=[name for name in canonical if name in channels]
        for seed in ctx.c['experiment_seeds']:
            for mode in ('regression','regression_diffusion'):
                cases.append(dict(experiment_id=e['experiment_id'],name=e['name'],channels=channels,seed=seed,mode=mode,
                    dataset_dir=ctx.c['dataset_dir'],target_transform='target_code / 50',patch_size=32,patch_stride=8,
                    train_split='train',validation_split='validation',tests_locked=True,mask_required=True,
                    derived_channel_statistics='fit on train only before execution' if e.get('derived_families') else 'frozen reference normalization',
                    runnable_native=False))
    dump(out/'experiment_manifest.json',dict(version='reference-ablation-plan-v1',cases=cases,training_executed=False))
    return dict(passed=True,scope='ablation planning only; original experiment definitions retained',runs=len(cases),training_executed=False)


def phase18(ctx,out):
    """Native integration hook. Never run incompatible historical scripts."""
    import sys
    native=ctx.c.get('native',{})
    checkout=native.get('corrdiff_dir')
    report=dict(complete=False,native_training_passed=False,target_transform='target_code / 50',mask_required=True,
                reason='Native masked integration has not been configured and verified.',python=sys.executable)
    if checkout:
        checkout=Path(checkout)
        report['checkout_exists']=checkout.is_dir()
        if checkout.is_dir():
            p=subprocess.run(['git','-C',str(checkout),'rev-parse','HEAD'],capture_output=True,text=True)
            report['checkout_commit']=p.stdout.strip() if p.returncode==0 else None
            report['train_py_exists']=(checkout/'train.py').is_file()
    dump(out/'native_preflight.json',report)
    command=native.get('command')
    if not command:
        raise StageBlocked('Phase 18 needs an explicitly configured native command with masked regression/diffusion loss. See native_preflight.json; historical log1p/unmasked entrypoints are not run.')
    if not isinstance(command,list) or not all(isinstance(v,str) for v in command):raise ValueError('native.command must be an argv list')
    expected=native.get('expected_commit')
    if not expected or report.get('checkout_commit')!=expected:raise StageBlocked('Pin and verify the native checkout commit before execution')
    if not native.get('masked_loss_verified'):raise StageBlocked('native.masked_loss_verified must be explicitly confirmed for the configured integration')
    values=dict(dataset_dir=ctx.c['dataset_dir'],output_dir=str(out.resolve()),experiment_manifest=str((ctx.directory(17)/'experiment_manifest.json').resolve()))
    argv=[arg.format(**values) for arg in command]
    dump(out/'native_command.json',dict(argv=argv,expected_commit=expected))
    with (out/'native.log').open('w') as log:
        result=subprocess.run(argv,cwd=checkout,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Native command failed with exit {result.returncode}')
    evidence=json.loads((out/'native_result.json').read_text())
    required=('complete','native_regression_tested','native_diffusion_tested','masked_loss_tested','checkpoint_reload_tested','all_steps_finite','parameters_changed','test_splits_untouched')
    if any(evidence.get(k) is not True for k in required):raise StageBlocked('Native result lacks required evidence flags')
    if evidence.get('contract_sha256')!=sha(Path(ctx.c['dataset_dir'])/'contract.json'):raise ValueError('Native evidence refers to a different contract')
    if evidence.get('target_transform')!='target_code / 50':raise ValueError('Native target transform mismatch')
    return dict(passed=True,scope='external pinned native integration; evidence supplied by configured command',native_result=evidence,independently_verified_training=False)
