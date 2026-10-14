"""RF continuous2와 같은 학습 지원/평가 가중치 및 전체 C 부분경사."""
import numpy as np
import pandas as pd

WEIGHT_RULE='rf_continuous2_calendar_support_v1'


def thresholds(power, initial=170.):
    if not isinstance(power.index,pd.DatetimeIndex) or power.index.has_duplicates or not power.index.is_monotonic_increasing:
        raise ValueError('유일하고 시간순인 DatetimeIndex 필요')
    if not np.isfinite(power.to_numpy()).all() or initial<=0:raise ValueError('유한 전력/양수 초기값 필요')
    # 종료00:00 셀은 직전 날짜의 관측이다. 원점 월 첫00:00은 새 월 초기값.
    month=(power.index-pd.Timedelta(nanoseconds=1)).to_period('M')
    running=power.groupby(month).cummax().to_numpy(float)
    running[month!=power.index.to_period('M')]=-np.inf
    return pd.Series(np.maximum(initial,running),index=power.index,name='B_origin')


def weights(y,b,origins=None,training=False):
    y=np.asarray(y,float);b=np.asarray(b,float)
    if y.ndim!=2 or not y.size or b.shape!=(len(y),) or not np.isfinite(y).all() or not np.isfinite(b).all() or (b<=0).any():
        raise ValueError('유한 (N,H) 정답 및 양수 (N,) 기준 필요')
    w=1+np.clip((y/b[:,None]-.8)/.2,0,1)
    if training:
        origins=pd.DatetimeIndex(origins)
        if len(origins)!=len(y):raise ValueError('원점 수 불일치')
        next_month=origins.to_period('M').to_timestamp()+pd.offsets.MonthBegin(1)
        supported=(origins>=pd.Timestamp('2021-02-01'))&(origins+pd.Timedelta(minutes=15*y.shape[1])<next_month)
        w[~supported]=1
    return w


def loss_gradient(y,p,w,width,objective='C'):
    y=np.asarray(y,float);p=np.asarray(p,float);w=np.asarray(w,float)
    if y.ndim!=2 or not y.size or p.shape!=y.shape or w.shape!=y.shape or not all(np.isfinite(a).all() for a in [y,p,w]) or (w<0).any():
        raise ValueError('정답/예측/비음수 가중치 동일 유한 (N,H) 필요')
    if not isinstance(width,(int,np.integer)) or width<1 or y.shape[1]%width:raise ValueError('H를 나누는 양의 정수 창 필요')
    d=p-y;blocks=d.reshape(-1,width);r=np.sqrt(np.mean(blocks**2,axis=1));k=max(1,int(np.ceil(.2*len(r))))
    # RF 코드와 같은 안정 오름차순 마지막 K: 동률에서는 뒤의 블록 선택.
    top=np.argsort(r,kind='stable')[-k:]
    under=np.minimum(d,0);peak=float(np.sqrt(np.mean(w*under**2)))
    mean=float(r.mean());tail=float(r[top].mean());rmse=float(np.sqrt(np.mean(d**2)))
    metric=dict(Loss_C=.5*mean+.3*tail+.2*peak,Loss_A=rmse,Loss_B=.6*mean+.4*tail,
        L_mean=mean,L_CVaR=tail,L_peak=peak,RMSE=rmse,MAE=float(np.abs(d).mean()),
        under_RMSE=float(np.sqrt(np.mean(under**2))),over_RMSE=float(np.sqrt(np.mean(np.maximum(d,0)**2))),bias=float(d.mean()))
    inv=np.divide(1.,width*r,out=np.zeros_like(r),where=r>0)
    gm=(.5/len(r)*inv[:,None]*blocks).reshape(y.shape)
    select=np.zeros(len(r));select[top]=.3/k
    gt=(select[:,None]*inv[:,None]*blocks).reshape(y.shape)
    gp=.2*w*under/(y.size*peak) if peak>0 else np.zeros_like(y)
    if objective=='C_without_peak':gp=np.zeros_like(y)
    elif objective=='C_without_tail':gt=np.zeros_like(y)
    elif objective=='A':
        gm=d/(y.size*rmse) if rmse>0 else np.zeros_like(y);gt=np.zeros_like(y);gp=np.zeros_like(y)
    elif objective!='C':raise ValueError('지원하지 않는 학습 목적')
    value={'C':metric['Loss_C'],'A':rmse,'C_without_peak':.5*mean+.3*tail,'C_without_tail':.5*mean+.2*peak}[objective]
    return metric,gm+gt+gp,dict(value=float(value),tail_blocks=top.tolist(),components=dict(mean=gm,tail=gt,peak=gp))
