#!/usr/bin/env python3
"""Train-only masked smoke using native PhysicsNeMo regression/EDM models.

This is a dedicated adapter loop, NOT NVIDIA's unmodified train.py. Partial
patches condition on support; unsupported residual/noise are zero. No sampler
or production training performance is certified by this test.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from dataset import RadarDataset,masked_mse,residual_inputs
from pair_contract import digest,dump

UPSTREAM='42fa19c3fa4c80d76a621b2454511326cde8a177'

def models():
    try:
        from physicsnemo.models.diffusion_unets import CorrDiffRegressionUNet
        from physicsnemo.diffusion.preconditioners import EDMPrecondSuperResolution
    except ImportError as exc:
        raise RuntimeError('Native API unavailable. Use an isolated PhysicsNeMo environment matching the API inspected at '+UPSTREAM+'. No fallback to TinyRegressor.') from exc
    return CorrDiffRegressionUNet,EDMPrecondSuperResolution

def step_check(loss,model,optimizer):
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Missing/nonfinite gradients')
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite model after update')

def run(a):
    out=Path(a.output_dir).resolve()
    if out.exists():raise ValueError('Use a new smoke output directory')
    if a.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; repair compatible environment or use --device cpu')
    ds=RadarDataset(a.data_dir,'train',fully_supported=a.fully_supported_only)
    if not len(ds):raise ValueError('No train patches')
    # Deterministic diagnostic subset spanning training, not production sampling.
    ids=sorted(set(torch.linspace(0,len(ds)-1,min(a.max_patches,len(ds))).long().tolist()))
    gen=torch.Generator().manual_seed(a.seed)
    loader=DataLoader(Subset(ds,ids),batch_size=a.batch_size,shuffle=True,generator=gen,num_workers=0)
    R,D=models();torch.manual_seed(a.seed);torch.set_num_threads(a.cpu_threads)
    kw=dict(img_resolution=[32,32],img_in_channels=15,img_out_channels=1,
      model_type='SongUNet',model_channels=32,channel_mult=[1,2,2],num_blocks=1,
      attn_resolutions=[],dropout=0.,use_fp16=False)
    regression=R(**kw).to(a.device);diffusion=D(**kw,sigma_data=a.sigma_data).to(a.device)
    out.mkdir(parents=True)
    dump(out/'summary.json',{'complete':False,'production_ready':False,'corrdiff_native_tested':False})
    results={}
    for stage,model in [('regression',regression),('diffusion',diffusion)]:
        model.train();opt=torch.optim.Adam(model.parameters(),lr=a.lr)
        before=[p.detach().clone() for p in model.parameters()];iterator=iter(loader);records=[]
        for k in range(a.steps):
            try:b=next(iterator)
            except StopIteration:iterator=iter(loader);b=next(iterator)
            y=b['target'].to(a.device);mask=b['mask'].to(a.device)
            cond=torch.cat((b['conditioning'],b['coordinates'],b['mask'].float()),dim=1).to(a.device)
            opt.zero_grad(set_to_none=True)
            if stage=='regression':
                pred=model(torch.zeros_like(y),cond);loss=masked_mse(pred,y,mask)
            else:
                regression.eval()
                with torch.no_grad():mean=regression(torch.zeros_like(y),cond)
                sigma=(torch.randn((len(y),1,1,1),device=y.device)*1.2-1.2).exp()
                residual,noisy=residual_inputs(y,mean,mask,sigma,torch.randn_like(y))
                pred=model(noisy,cond,sigma)
                weight=(sigma.square()+a.sigma_data**2)/(sigma*a.sigma_data).square()
                loss=masked_mse(pred,residual,mask,weight)
            step_check(loss,model,opt)
            records.append(float(loss.detach().cpu()));print(f'{stage} {k+1}/{a.steps} loss={records[-1]:.6g}',flush=True)
        changed=any(not torch.equal(p,q.detach()) for p,q in zip(before,model.parameters()))
        if not changed:raise ValueError('No parameter changed')
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'model_kwargs':kw,'sigma_data':a.sigma_data,'stage':stage,'smoke_only':True},out/f'{stage}_smoke.pt')
        results[stage]={'losses':records,'all_steps_finite':True,'parameters_changed':changed}
    dump(out/'run_config.json',{'args':vars(a),'upstream_inspected_commit':UPSTREAM,
      'torch_version':torch.__version__,'model_module':str(R.__module__),
      'prepared_summary_sha256':digest(Path(a.data_dir)/'summary.json'),
      'adapter_loop':True,'conditioning_channels':'12 ERA5 + 2 static coordinates + 1 support mask',
      'sigma_data':'explicit smoke hyperparameter, not a fitted residual statistic'})
    dump(out/'summary.json',{'version':'14o-native-smoke-v1','complete':True,'technical_smoke_passed':True,
      'corrdiff_native_tested':True,'stock_training_entrypoint_tested':False,'generation_tested':False,
      'production_ready':False,'coverage_state':ds.summary['coverage_state'],'selected_train_patches':len(ids),
      'fully_supported_only':a.fully_supported_only,'stages':results,
      'note':'Native models with a small architecture and custom masked loop; no validation/test, no skill claim, no sampler validation.'})
    print((out/'summary.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--steps',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--max-patches',type=int,default=48)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--sigma-data',type=float,default=.5)
    p.add_argument('--fully-supported-only',action='store_true');a=p.parse_args()
    if min(a.steps,a.batch_size,a.max_patches,a.cpu_threads)<=0 or not 0<a.sigma_data<100 or not 0<a.lr<1:p.error('Invalid numeric parameters')
    run(a)
