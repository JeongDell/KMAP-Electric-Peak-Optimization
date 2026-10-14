"""중·장기 공유 리드타임 회귀의 입력·분할·위험평가 계약."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import boosting_common as bc
import forecast_common as fc
ROOT=bc.ROOT
PROFILES=('medium_4h','medium_8h','medium_12h','medium_24h','long_3d','long_4d')
LONG=('long_3d','long_4d')
HORIZON={'medium_4h':16,'medium_8h':32,'medium_12h':48,'medium_24h':96,'long_3d':288,'long_4d':384}

def base_features(df,profile):
    if profile not in PROFILES:raise ValueError(f'지원profile: {PROFILES}')
    f,_,g=bc.rf.build_features(df)
    obs=[c for c in g['F1'] if c.startswith('observed_')]
    groups={'F1':obs,'F2':obs+[c for c in g['F2'] if c not in g['F1']],
        'F3':[c for c in g['F3'] if not c.startswith('h')]}
    if profile in LONG:
        f['power_lag_14d']=df.power_15min.shift(1344)
        for stat in ['mean','max','std']:
            rolling=df.power_15min.rolling(1344)
            f[f'power_14d_{stat}']=rolling.std(ddof=0) if stat=='std' else getattr(rolling,stat)()
        groups['F3']+=['power_lag_14d','power_14d_mean','power_14d_max','power_14d_std']
    return f,groups

def prepare_profile(df,profile,fold):
    fc.check_frame(df)
    base,groups=base_features(df,profile)
    spec=fc.PROFILES[profile];H=spec['horizon_steps']
    if not np.isfinite(df[list(bc.rf.OBSERVED)].to_numpy(dtype=float)).all():raise ValueError('외생값 결측/무한값')
    values=df.power_15min.to_numpy(dtype=float)
    windows=np.lib.stride_tricks.sliding_window_view(values,H+1)
    origins=df.index[:-H]
    complete=np.isfinite(base.loc[origins,groups['F3']].to_numpy(dtype=float)).all(axis=1)&np.isfinite(windows).all(axis=1)
    cadence=(origins.hour*60+origins.minute)%(spec['refresh_steps']*15)==0
    start,end=map(pd.Timestamp,fc.FOLDS[fold]);ends=origins+pd.Timedelta(minutes=15*H)
    train=complete&cadence&(ends<start)
    valid=complete&cadence&(origins>=start)&(ends<end)
    if not train.any() or not valid.any():raise ValueError(f'{profile}/{fold}: 유효원점없음')
    return dict(train_origins=origins[train],valid_origins=origins[valid],y_train=windows[train,1:].copy(),
        y_valid=windows[valid,1:].copy(),base=base,groups=groups,profile=profile,fold=fold)

def training_pairs(origins,H,max_rows=60000,seed=42):
    if not len(origins) or H<1 or max_rows<len(origins):raise ValueError('원점마다 최소1개리드타임이필요')
    count=min(H,max_rows//len(origins))
    if count==H:return np.broadcast_to(np.arange(1,H+1),(len(origins),H)).copy()
    rng=np.random.default_rng(seed)
    return np.sort(np.argsort(rng.random((len(origins),H)),axis=1)[:,:count]+1,axis=1)

def build_inputs(df,origins,profile,feature,scenario,leads=None,base=None,groups=None):
    if feature not in ['F1','F2','F3'] or scenario not in ['observed','future']:raise ValueError('F/외생조건오류')
    if base is None:base,groups=base_features(df,profile)
    H=HORIZON[profile]
    if leads is None:leads=np.broadcast_to(np.arange(1,H+1),(len(origins),H))
    leads=np.asarray(leads,dtype=int)
    if leads.ndim!=2 or leads.shape[0]!=len(origins) or not ((leads>=1)&(leads<=H)).all():raise ValueError('리드타임shape/범위오류')
    if any(len(np.unique(row))!=len(row) for row in leads):raise ValueError('원점내리드타임중복')
    m=leads.shape[1];expanded=origins.repeat(m);h=leads.ravel()
    ends=expanded+pd.to_timedelta(15*h,unit='min');starts=ends-pd.Timedelta(minutes=15)
    index=pd.MultiIndex.from_arrays([expanded,h],names=['origin','horizon_step'])
    X=pd.DataFrame(index=index)
    for col in bc.rf.OBSERVED:
        X[f'exogenous_{col}']=df[col].reindex(ends).to_numpy() if scenario=='future' else np.repeat(base.loc[origins,f'observed_{col}'].to_numpy(),m)
    for col in groups[feature]:
        if not col.startswith('observed_'):X[col]=np.repeat(base.loc[origins,col].to_numpy(),m)
    for day in range(7):X[f'target_weekday_{day}']=(starts.dayofweek==day).astype(float)
    X['target_month']=starts.month.astype(float)
    X['target_night']=((starts.hour>=18)|(starts.hour<9)).astype(float)
    X['target_workday']=df.day_type.reindex(ends).eq(0).to_numpy(dtype=float)
    X['target_quarter']=(starts.hour*4+starts.minute//15).astype(float)
    X['horizon_step']=h.astype(float)
    if feature=='F3':
        cyclic=expanded+pd.to_timedelta((((h-1)%96)+1-96)*15,unit='min')
        week=ends-pd.Timedelta(days=7)
        if (cyclic>expanded).any() or (week>expanded).any():raise ValueError('계절패턴이원점미래를참조')
        X['power_previous_day_cycle']=df.power_15min.reindex(cyclic).to_numpy()
        X['power_previous_week']=df.power_15min.reindex(week).to_numpy()
        if profile in LONG:X['power_previous_14d']=df.power_15min.reindex(ends-pd.Timedelta(days=14)).to_numpy()
    if X.index.has_duplicates or not np.isfinite(X.to_numpy(dtype=float)).all():raise ValueError(f'{profile}/{feature}/{scenario}: 입력키중복/결측')
    return X.astype(np.float32)

def scores(actual,predicted,profile,resolution_steps=1):
    W=96 if profile in LONG else 4
    W//=resolution_steps
    a,p=np.asarray(actual),np.asarray(predicted)
    if a.shape!=p.shape or a.ndim!=2 or a.shape[1]%W:raise ValueError('경로/위험window shape오류')
    out=bc.evaluate_paths(a.reshape(-1,W),p.reshape(-1,W))
    out['n_risk_windows']=out['n_origins'];out['n_origins']=len(a)
    return out
