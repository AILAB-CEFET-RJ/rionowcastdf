"""Reference-contract diagnostics. All thresholds use stored float32 code/50."""
import numpy as np
import pandas as pd
from scipy import ndimage,stats
from .common import CODES,SPLITS,csv,dump,describe,derived,corr


def phase1(ctx,out):
    r=ctx.reader('train');rng=np.random.default_rng(ctx.c['seed']);samples=[];ys=[]
    for t,h in ctx.fields():
        points=np.flatnonzero(np.isfinite(h['input']).all(axis=0))
        take=rng.choice(points,min(len(points),ctx.c['sample_pixels_per_hour']),replace=False)
        samples.append(h['input'].reshape(len(r.channels),-1)[:,take].T)
        ys.append(h['target_code'][h['mask']])
    if not samples:raise ValueError('No train sample')
    x=np.concatenate(samples);y=np.concatenate(ys)
    rows=[dict(channel=c,**describe(x[:,j]),skew=stats.skew(x[:,j]),excess_kurtosis=stats.kurtosis(x[:,j])) for j,c in enumerate(r.channels)]
    csv(out/'input_distributions.csv',rows)
    csv(out/'target_distribution.csv',[dict(variable='target_code_including_no_echo',**describe(y)),dict(variable='displayed_echo_dbz_only',**describe(y[y>0]))])
    csv(out/'sampled_target_classes.csv',[dict(code=int(c),pixels=int((y==c).sum()),denominator=len(y)) for c in CODES])
    np.savez_compressed(out/'train_point_sample.npz',input=x,channels=r.channels)
    return dict(passed=True,scope='train sampled unique hours and spatial points',hours=len(ctx.selected()),points=len(x),target_zero_is_category=True)


def phase2(ctx,out):
    values={};names=ctx.reader('train').channels
    for t,h in ctx.fields():
        for k,v in derived(h['input'],names).items():
            values.setdefault(k,[]).append(v[np.isfinite(v)])
    units={k:('K' if k.startswith('delta_t') else 'percentage_points' if k.startswith('delta_r') else 'm/s') for k in values}
    csv(out/'derived_statistics.csv',[dict(variable=k,unit=units[k],**describe(np.concatenate(v))) for k,v in values.items()])
    dump(out/'definitions.json',dict(note='Wind differences are vector differences, not vertical shear per metre. Temperatures K; RH percentage points. No clipping.',variables=list(values)))
    return dict(passed=True,scope='train sampled unique grid hours',hours=len(ctx.selected()))


def phase3(ctx,out):
    d=ctx.table().query("split == 'train'");names=ctx.reader('train').channels+list(derived(np.zeros((12,1,1)),ctx.reader('train').channels))
    d=d[d.valid_pixels>0].copy();d['echo_fraction']=d.ge_20/d.valid_pixels
    rows=[]
    for name in names:
        for target in ['echo_fraction','mean_target_code','max_code']:
            rows.append(dict(predictor=name,target=target,hours=len(d),pearson=corr(d[name].to_numpy(),d[target].to_numpy()),spearman=stats.spearmanr(d[name],d[target]).statistic))
    csv(out/'hourly_relationships.csv',rows)
    return dict(passed=True,scope='train hourly spatial means; descriptive associations, not causal or pixelwise')


def phase4(ctx,out):
    d=ctx.table();rows=[];smd=[];train=d[d.split=='train']
    for split,g in d.groupby('split'):
        for c in CODES[1:]:
            event=g[f'ge_{int(c)}']>0
            rows.append(dict(split=split,threshold_dbz=int(c),event_hours=int(event.sum()),observed_hours=int((g.valid_pixels>0).sum()),event_pixels=int(g[f'ge_{int(c)}'].sum()),valid_pixels=int(g.valid_pixels.sum())))
    for c in (30,40,45,50):
        event=train[f'ge_{c}']>0
        for name in ctx.reader('train').channels:
            a,b=train.loc[event,name],train.loc[~event,name]
            scale=np.sqrt((a.var(ddof=0)+b.var(ddof=0))/2)
            smd.append(dict(threshold_dbz=c,predictor=name,event_hours=len(a),non_event_hours=len(b),smd=float((a.mean()-b.mean())/scale) if len(a)>1 and len(b)>1 and scale>0 else None))
    csv(out/'extreme_rates.csv',rows);csv(out/'train_extreme_smd.csv',smd)
    return dict(passed=True,scope='unique hourly/pixel denominators; fixed palette thresholds',thresholds_fit_on_test=False)


def calendar_phase(ctx,out,phase):
    d=ctx.table();t=pd.to_datetime(d.timestamp_utc,unit='s',utc=True)
    frames=[]
    for clock in (['UTC','America/Sao_Paulo'] if phase==6 else ['UTC']):
        local=t.dt.tz_convert(clock);d=d.copy()
        if phase==5:d['group']=local.dt.month.map({12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'})
        else:d['group']=local.dt.hour
        for (split,group),g in d.groupby(['split','group']):
            row=dict(split=split,group=group,clock=clock,selected_hours=len(g),valid_pixels=int(g.valid_pixels.sum()),no_echo_pixels=int(g.no_echo_pixels.sum()))
            for c in (20,30,40,45,50):
                row[f'ge_{c}_pixels']=int(g[f'ge_{c}'].sum());row[f'ge_{c}_hours']=int((g[f'ge_{c}']>0).sum())
            for name in ctx.reader('train').channels:row[name+'_mean']=g[name].mean()
            frames.append(row)
    csv(out/'calendar_metrics.csv',frames)
    return dict(passed=True,scope='observed unique hours, conditioned on availability; not unconditional climatology',local_timezone='America/Sao_Paulo' if phase==6 else None)


def phase7(ctx,out):
    d=ctx.table();rows=[]
    for split,g in d.groupby('split'):
        g=g[g.valid_pixels>0].set_index('timestamp_utc');v=g.ge_20/g.valid_pixels
        for lag in ctx.c['lags_hours']:
            previous=v.reindex(v.index-lag*3600).to_numpy();current=v.to_numpy();m=np.isfinite(previous)&np.isfinite(current)
            p,q=previous[m]>0,current[m]>0
            p1=q[p].mean() if p.any() else np.nan;p0=q[~p].mean() if (~p).any() else np.nan
            rows.append(dict(split=split,lag_hours=lag,pairs=int(m.sum()),pearson=corr(previous,current),phi=corr(p.astype(float),q.astype(float)),risk_ratio=p1/p0 if p0>0 else None))
    csv(out/'lag_metrics.csv',rows)
    return dict(passed=True,scope='exact UTC time joins within each split; gaps never bridged')


def phase8(ctx,out):
    rows=[]
    for split in ('train','validation'):
        r=ctx.reader(split);available=set(r.hours.tolist())
        for lag in ctx.c['lags_hours']:
            sums=dict(n=0,se=0.,ae=0.,hits=0,misses=0,false_alarms=0,correct_negatives=0);pairs=0
            for t in ctx.selected(split):
                if int(t)-lag*3600 not in available:continue
                a=r.hour(int(t)-lag*3600);b=r.hour(int(t));m=a['mask']&b['mask']
                pred=a['target'][m];truth=b['target'][m];e=pred-truth
                sums['n']+=len(e);sums['se']+=float(np.square(e.astype('f8')).sum());sums['ae']+=float(abs(e).sum())
                p,q=pred>=np.float32(.4),truth>=np.float32(.4)
                for k,v in [('hits',p&q),('misses',~p&q),('false_alarms',p&~q),('correct_negatives',~p&~q)]:sums[k]+=int(v.sum())
                pairs+=1
            n=sums.pop('n');se=sums.pop('se');ae=sums.pop('ae')
            rows.append(dict(split=split,lag_hours=lag,pairs=pairs,valid_pixel_pairs=n,mse_code_over_50=se/n if n else None,mae_code_over_50=ae/n if n else None,**sums))
    csv(out/'persistence_metrics.csv',rows)
    return dict(passed=True,scope='sampled train/validation exact-lag pairs; mask intersection; tests held out')


def phase9(ctx,out):
    variogram={};morph=[]
    for t,h in ctx.fields():
        y=h['target'][0].astype('f8');mask=h['mask'][0]
        for axis in (0,1):
            for lag in (1,2,4,8,16):
                a=[slice(None),slice(None)];b=list(a);a[axis]=slice(None,-lag);b[axis]=slice(lag,None)
                a,b=tuple(a),tuple(b);m=mask[a]&mask[b];delta=y[a][m]-y[b][m]
                acc=variogram.setdefault((axis,lag),[0.,0]);acc[0]+=float(np.square(delta).sum());acc[1]+=len(delta)
        for c in (20,30,40,45):
            event=mask&(y>=np.float32(c/50));labels,n=ndimage.label(event)
            sizes=np.bincount(labels.ravel())[1:];touch=set(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]))-{0}
            adjacent=ndimage.binary_dilation(~mask)&event
            mask_touch=set(labels[adjacent])-{0}
            morph.append(dict(timestamp_utc=t,threshold_dbz=c,components=n,max_component_pixels=int(sizes.max()) if len(sizes) else 0,border_components=len(touch),mask_adjacent_components=len(mask_touch)))
    csv(out/'semivariogram.csv',[dict(axis='north_south' if a==0 else 'east_west',lag_grid_points=l,pairs=n,semivariance_code_over_50=s/(2*n) if n else None) for (a,l),(s,n) in variogram.items()])
    csv(out/'morphology.csv',morph)
    return dict(passed=True,scope='sampled train full fields; 4-neighbour components; masked pairwise variogram',distance_unit='grid points; angular grid is not square kilometres')


def phase10(ctx,out):
    r=ctx.reader('train');shape=(len(r.channels),70,84)
    n=np.zeros(shape);sx=np.zeros(shape);sy=np.zeros(shape);sxx=np.zeros(shape);syy=np.zeros(shape);sxy=np.zeros(shape)
    for t,h in ctx.fields():
        x=h['input'].astype('f8');y=h['target'].astype('f8');m=h['mask']&np.isfinite(x)
        n+=m;sx+=np.where(m,x,0);sy+=np.where(m,y,0);sxx+=np.where(m,x*x,0);syy+=np.where(m,y*y,0);sxy+=np.where(m,x*y,0)
    with np.errstate(divide='ignore',invalid='ignore'):
        cov=sxy-sx*sy/n;den=np.sqrt(np.maximum(sxx-sx*sx/n,0)*np.maximum(syy-sy*sy/n,0));rho=np.where((n>2)&(den>0),cov/den,np.nan)
    np.savez_compressed(out/'pixelwise_correlations.npz',correlation=rho,pairs=n,lat=r.lat,lon=r.lon,channels=r.channels)
    return dict(passed=True,scope='sampled train temporal correlations at each grid point; paired masks',hours=len(ctx.selected()))


def phase11(ctx,out):
    rows=[]
    for t,h in ctx.fields():
        for scale in (1,2,4,8,16):
            sums=[];counts=[]
            for r in range(0,70,scale):
                for c in range(0,84,scale):
                    m=h['mask'][0,r:r+scale,c:c+scale];y=h['target'][0,r:r+scale,c:c+scale]
                    if m.any():sums.append(float(y[m].mean()));counts.append(int(m.sum()))
            if sums:
                mean=np.average(sums,weights=counts);var=np.average((np.asarray(sums)-mean)**2,weights=counts)
            else:mean=var=np.nan
            rows.append(dict(timestamp_utc=t,block_grid_points=scale,valid_blocks=len(sums),valid_pixels=sum(counts),mean_code_over_50=mean,block_mean_variance=var))
    csv(out/'scale_variance.csv',rows)
    return dict(passed=True,scope='nonoverlapping full-grid blocks, including smaller edge blocks; mask-weighted code averages',note='Numerical code averages are not linear-Z reflectivity averages or rainfall estimates.')


def phase12(ctx,out):
    d=ctx.table().query("split == 'train'");names=ctx.reader('train').channels;x=d[names].to_numpy('f8')
    mu=x.mean(axis=0);sd=x.std(axis=0);sd=np.where(sd>0,sd,1);z=(x-mu)/sd
    _,s,v=np.linalg.svd(z,full_matrices=False);variance=s*s/max(len(x)-1,1);ratio=variance/variance.sum() if variance.sum() else np.zeros_like(variance)
    csv(out/'pca_variance.csv',[dict(component=i+1,variance=var,explained_fraction=ratio[i]) for i,var in enumerate(variance)])
    pd.DataFrame(v,columns=names).to_csv(out/'pca_loadings.csv',index=False)
    np.savez(out/'pca_train_fit.npz',mean=mu,std=sd,components=v,channels=names)
    return dict(passed=True,scope='PCA of train hourly predictor means; fit only train',hours=len(x))


def phase13(ctx,out):
    from sklearn.cluster import KMeans
    d=ctx.table();names=ctx.reader('train').channels;train=d[d.split=='train'];x=train[names].to_numpy('f8')
    mu=x.mean(0);sd=x.std(0);sd=np.where(sd>0,sd,1)
    k=min(ctx.c['regime_clusters'],len(x));model=KMeans(n_clusters=k,random_state=ctx.c['seed'],n_init=10)
    if len(x)>ctx.c['regime_fit_max_hours']:
        ix=np.random.default_rng(ctx.c['seed']).choice(len(x),ctx.c['regime_fit_max_hours'],replace=False);fit=x[ix]
    else:fit=x
    model.fit((fit-mu)/sd)
    assignments=d[['timestamp_utc','split']].copy();assignments['regime']=model.predict((d[names].to_numpy()-mu)/sd)
    assignments.to_parquet(out/'regime_assignments.parquet',index=False)
    np.savez(out/'train_regime_fit.npz',mean=mu,std=sd,centers=model.cluster_centers_,channels=names)
    return dict(passed=True,scope='train-fit KMeans on hourly ERA5 means, applied unchanged to all splits',clusters=k,fit_hours=len(fit))


def phase14(ctx,out):
    d=ctx.table();train=d[d.split=='train'];rows=[]
    for split in SPLITS[1:]:
        other=d[d.split==split]
        for name in ctx.reader('train').channels:
            a,b=train[name].to_numpy(),other[name].to_numpy();sd=a.std()
            rows.append(dict(split=split,channel=name,train_hours=len(a),other_hours=len(b),mean_shift_train_std=(b.mean()-a.mean())/sd if sd>0 else None,ks_statistic=stats.ks_2samp(a,b).statistic if len(a) and len(b) else None))
    csv(out/'predictor_shift.csv',rows)
    reg=pd.read_parquet(ctx.directory(13)/'regime_assignments.parquet')
    reg.groupby(['split','regime']).size().rename('hours').reset_index().to_csv(out/'regime_composition.csv',index=False)
    return dict(passed=True,scope='descriptive held-out distribution audit; no fitting or tuning on test',ks_pvalues_omitted_due_to_temporal_dependence=True)
