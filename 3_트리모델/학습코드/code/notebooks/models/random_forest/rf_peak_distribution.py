"""RF 잎에 속한 실제 학습 경로의 경험분포와 피크 위험 계산.

Native 다중출력에 QRF 잎 분포 원리를 확장한다. RF 나무의 평균 예측을
표본 경로로 간주하지 않는다. 가중 학습과 균등 leaf 분포 질량을 구분한다.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logit

QUANTILES=np.array([.05,.10,.20,.50,.80,.90,.95])


def thresholds(power,initial,mode='dynamic'):
    if not isinstance(power.index,pd.DatetimeIndex) or power.index.has_duplicates or not power.index.is_monotonic_increasing:
        raise ValueError('관측 종료시각은 유일하고 시간순이어야 합니다.')
    if not np.isfinite(initial) or initial<=0 or not np.isfinite(power.to_numpy(float)).all():
        raise ValueError('피크 초기값은 양수, 관측 전력은 유한해야 합니다.')
    if mode=='fixed':return pd.Series(float(initial),index=power.index,name='peak_threshold')
    if mode!='dynamic':raise ValueError(f'알 수 없는 피크 기준 방식: {mode}')
    # [t-15min,t) 전력은 t에 이용 가능. 자정에 끝난 전월 구간을 새 월에 넣지 않는다.
    observation_month=(power.index-pd.Timedelta(minutes=15)).to_period('M')
    origin_month=power.index.to_period('M')
    current=power.groupby(observation_month).cummax().to_numpy(float)
    b=np.where(observation_month==origin_month,np.maximum(initial,current),initial)
    return pd.Series(b,index=power.index,name='peak_threshold')


def time_masks(origins,horizon,start,end):
    idx=pd.DatetimeIndex(origins);start=pd.Timestamp(start);end=pd.Timestamp(end)
    if horizon<1 or start>=end:raise ValueError('양수 예측거리와 증가하는 기간 필요')
    last=idx+pd.Timedelta(minutes=15*horizon)
    return np.asarray(last<start),np.asarray((idx>=start)&(last<end))


def cell_weights(actual,b,reference_probability,cap=2):
    a=np.asarray(actual,float);b=np.asarray(b,float);p=np.asarray(reference_probability,float)
    if a.ndim!=2 or b.shape!=(len(a),) or p.shape!=a.shape:
        raise ValueError('정답·기준 확률은 (원점,H), B는 원점별 한 값이어야 합니다.')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (b<=0).any():
        raise ValueError('유한 정답과 양수 피크 기준 필요')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isfinite(cap) or cap<1:
        raise ValueError('확률은 0~1, 가중치 상한은 1 이상이어야 합니다.')
    g=np.clip((a/b[:,None]-.8)/.2,0,1)
    return 1+(cap-1)*g*(1+p)/2


def point_metrics(actual,predicted,weights,window_steps):
    a=np.asarray(actual,float);p=np.asarray(predicted,float);w=np.asarray(weights,float)
    if a.ndim!=2 or p.shape!=a.shape or w.shape!=a.shape or not len(a):
        raise ValueError('비어 있지 않은 같은 크기 전력 경로와 셀 가중치 필요')
    if not np.isfinite(a).all() or not np.isfinite(p).all() or not np.isfinite(w).all() or (w<0).any():
        raise ValueError('전력·예측·비음수 가중치는 유한해야 합니다.')
    if window_steps<1 or a.shape[1]%window_steps:raise ValueError('손실 시간 창이 H를 나누어야 합니다.')
    error=a-p
    windows=np.sqrt(np.mean(error.reshape(-1,window_steps)**2,axis=1))
    mean=float(windows.mean());tail=float(np.sort(windows)[-int(np.ceil(.2*len(windows))):].mean())
    peak=float(np.sqrt(np.mean(w*np.maximum(error,0)**2)))
    under=float(np.sqrt(np.mean(np.maximum(error,0)**2)))
    return dict(RMSE=float(np.sqrt(np.mean(error**2))),MAE=float(np.abs(error).mean()),
                L_mean=mean,L_CVaR=tail,L_peak=peak,L_peak_w1=under,
                Loss_C=.5*mean+.3*tail+.2*peak,Loss_C_w1=.5*mean+.3*tail+.2*under,
                bias=float((p-a).mean()),over_RMSE=float(np.sqrt(np.mean(np.maximum(-error,0)**2))))


class LeafPathDistribution:
    """한 나무의 각 leaf에서 실제 학습 경로를 균등하게 선택한 평균 분포."""
    def __init__(self,forest,x_train,y_train):
        self.forest=forest
        self.targets=np.asarray(y_train,float)
        if self.targets.ndim==1:self.targets=self.targets[:,None]
        if self.targets.ndim!=2 or not np.isfinite(self.targets).all() or len(self.targets)!=len(x_train):
            raise ValueError('분포 학습 경로가 RF 학습 행과 맞아야 합니다.')
        leaves=forest.apply(np.asarray(x_train,dtype=np.float32))
        self.n_trees=leaves.shape[1]
        nodes=np.array([tree.tree_.node_count for tree in forest.estimators_],dtype=np.int64)
        self.offset=np.r_[0,np.cumsum(nodes[:-1])]
        self.n_nodes=int(nodes.sum())
        cols=(leaves+self.offset).ravel()
        rows=np.repeat(np.arange(len(leaves)),self.n_trees)
        self.train_members=sparse.csr_matrix((np.ones(len(cols)),(rows,cols)),shape=(len(leaves),self.n_nodes))
        self.count=np.bincount(cols,minlength=self.n_nodes).astype(float)
        self.sorted_targets=np.argsort(self.targets,axis=0,kind='stable')

    def weights(self,x):
        x=np.asarray(x,dtype=np.float32)
        leaves=self.forest.apply(x)
        cols=(leaves+self.offset).ravel()
        if (self.count[cols]<=0).any():raise ValueError('평가 leaf에 학습 정답이 없습니다.')
        rows=np.repeat(np.arange(len(leaves)),self.n_trees)
        values=1/(self.n_trees*self.count[cols])
        query=sparse.csr_matrix((values,(rows,cols)),shape=(len(leaves),self.n_nodes))
        mass=(query@self.train_members.T).toarray()
        if not np.allclose(mass.sum(axis=1),1,rtol=1e-10,atol=1e-10):
            raise ValueError('조건부 경로 확률 질량 합이 1이 아닙니다.')
        return mass

    def predict(self,x,b,quantiles=False,initial=None,batch_size=48):
        x=np.asarray(x,dtype=np.float32);b=np.asarray(b,float)
        if b.shape!=(len(x),) or not np.isfinite(b).all():raise ValueError('원점별 유한 피크 기준 필요')
        n=len(x);h=self.targets.shape[1]
        point=np.empty((n,h));period=np.empty(n)
        q=np.empty((n,h,len(QUANTILES))) if quantiles else None
        initial_point=np.empty((n,h)) if initial is not None else None
        initial_period=np.empty(n) if initial is not None else None
        initial_event=(self.targets>initial).astype(float) if initial is not None else None
        for first in range(0,n,batch_size):
            stop=min(n,first+batch_size);mass=self.weights(x[first:stop]);values=b[first:stop]
            for value in np.unique(values):
                chosen=np.flatnonzero(values==value)
                event=(self.targets>value).astype(float)
                point[first+chosen]=mass[chosen]@event
                period[first+chosen]=mass[chosen]@event.max(axis=1)
            if initial is not None:
                initial_point[first:stop]=mass@initial_event
                initial_period[first:stop]=mass@initial_event.max(axis=1)
            if quantiles:
                for lead in range(h):
                    order=self.sorted_targets[:,lead]
                    cumulative=np.cumsum(mass[:,order],axis=1)
                    cumulative[:,-1]=1
                    for j,level in enumerate(QUANTILES):
                        which=(cumulative>=level).argmax(axis=1)
                        q[first:stop,lead,j]=self.targets[order[which],lead]
        point=np.clip(point,0,1);period=np.clip(period,0,1)
        if (period+1e-10<point.max(axis=1)).any():raise ValueError('기간 확률이 시점 확률보다 작습니다.')
        return dict(p_point=point,p_period=period,quantiles=q,p_initial=initial_point,p_initial_period=initial_period)


class MonotoneCalibrator:
    """과거 OOF 사건에서 적합한 2계수 단조 sigmoid 보정. 경로 분포와 별개."""
    def __init__(self):
        self.slope=1.;self.intercept=0.;self.status='identity_unfitted'

    def fit(self,p,y,weights=None):
        p=np.asarray(p,float).ravel();y=np.asarray(y,float).ravel()
        w=np.ones(len(p)) if weights is None else np.asarray(weights,float).ravel()
        if p.shape!=y.shape or w.shape!=y.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.isin(y,[0,1]).all():
            raise ValueError('유한 확률과 대응되는 0/1 정답 필요')
        if not np.isfinite(w).all() or (w<0).any() or (len(w) and w.sum()<=0):raise ValueError('보정 가중치 오류')
        self.n_samples=len(p);self.n_positive=int(y.sum())
        if not len(p):self.status='identity_no_past_oof';return self
        if len(np.unique(y))<2:self.status='identity_single_class';return self
        z=logit(np.clip(p,1e-5,1-1e-5));w=w/w.sum()
        def objective(theta):
            slope,intercept=theta;v=slope*z+intercept
            penalty=1e-3*((slope-1)**2+intercept**2)
            loss=np.sum(w*(np.logaddexp(0,v)-y*v))+penalty
            d=w*(expit(v)-y)
            grad=np.array([np.sum(d*z)+2e-3*(slope-1),d.sum()+2e-3*intercept])
            return loss,grad
        fit=minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(0,None),(None,None)])
        if not fit.success:raise RuntimeError(f'확률 보정 최적화 실패: {fit.message}')
        self.slope=float(fit.x[0]);self.intercept=float(fit.x[1]);self.status='fitted_past_oof'
        return self

    def predict(self,p):
        p=np.asarray(p,float)
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('보정 입력 확률 오류')
        if self.status.startswith('identity'):return p.copy()
        return expit(self.slope*logit(np.clip(p,1e-5,1-1e-5))+self.intercept)


def coherent_period_probability(point,period):
    """보정한 union 확률을 주변 확률의 가능한 범위로 투영한다.

    max(p_h) <= P(any peak) <= min(1,sum(p_h)). 보정된 joint 경로
    분포를 적합한 것은 아니며, 원시 leaf 경로 분포는 변경하지 않는다.
    """
    point=np.asarray(point,float);period=np.asarray(period,float)
    if point.ndim!=2 or point.shape[1]<1 or period.shape!=(len(point),):
        raise ValueError('시점 확률은 (원점,H), 기간 확률은 (원점,) 필요')
    if not np.isfinite(point).all() or not np.isfinite(period).all() or ((point<0)|(point>1)).any() or ((period<0)|(period>1)).any():
        raise ValueError('유한 0~1 확률 필요')
    return np.clip(period,point.max(axis=1),np.minimum(1,point.sum(axis=1)))


def binary_metrics(event,p):
    y=np.asarray(event,bool).ravel();p=np.asarray(p,float).ravel()
    if not len(y) or y.shape!=p.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('유효 사건과 0~1 확률 필요')
    clipped=np.clip(p,1e-10,1-1e-10);alarm=p>=.5
    tp=int((y&alarm).sum());fn=int((y&~alarm).sum());fp=int((~y&alarm).sum())
    return dict(n_events=len(y),n_positive=int(y.sum()),positive_rate=float(y.mean()),
                Brier=float(np.mean((p-y)**2)),logloss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
                recall=tp/(tp+fn) if tp+fn else np.nan,precision=tp/(tp+fp) if tp+fp else np.nan,
                false_positive=fp,false_negative=fn,mean_probability=float(p.mean()))


def reliability(event,p,bins=10):
    y=np.asarray(event,bool).ravel();p=np.asarray(p,float).ravel()
    bucket=np.minimum((p*bins).astype(int),bins-1)
    rows=[]
    for k in range(bins):
        chosen=bucket==k
        if chosen.any():rows.append(dict(bin=k,n=int(chosen.sum()),mean_probability=float(p[chosen].mean()),actual_rate=float(y[chosen].mean())))
    return rows
