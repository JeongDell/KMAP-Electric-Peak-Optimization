"""RF·부스팅 공통 잔차 경로 분포(E0). docs/residual_distribution_peak_probability_20261006.md.

기존 C 전력 예측 경로에 과거 미래 예측 오차 경로 전체를 더해 전력 경로 분포를 만든다.
시점·기간 확률, 분위수, 경로 표본은 모두 같은 오차 경로와 같은 질량(1/N_pool)을 쓴다.
시점 독립 곱 1-prod(1-p_h)를 사용하지 않으며 정규분포·평균0·음수 절단을 강제하지 않는다.
"""
import numpy as np
import pandas as pd

QUANTILES=np.array([.05,.10,.20,.50,.80,.90,.95])
DISTRIBUTION_VERSION='residual_path_E0_uniform_v1'


def _paths(values,name,horizon=None):
    values=np.asarray(values,float)
    if values.ndim!=2 or not values.size or not np.isfinite(values).all():raise ValueError(f'{name}: 유한 (N,H) 배열 필요')
    if horizon is not None and values.shape[1]!=horizon:raise ValueError(f'{name}: H={horizon} 필요, 실제 {values.shape[1]}')
    return values


def build_residual_pool(origins,actual,predicted,clean,horizon,before):
    """완전 경로의 예측 오차 e=y-ŷ를 모은다. 경로 끝(origin+15분×H) < before만 허용.

    복구·대체 정답을 포함한 경로는 실제 예측 오차가 아니므로 제외하고 건수를 남긴다.
    """
    origins=pd.DatetimeIndex(origins);before=pd.Timestamp(before)
    actual=_paths(actual,'actual',horizon);predicted=_paths(predicted,'predicted',horizon)
    clean=np.asarray(clean,bool)
    if origins.has_duplicates or not origins.is_monotonic_increasing:raise ValueError('오차 원점은 유일·시간순이어야 함')
    if actual.shape!=predicted.shape or len(origins)!=len(actual) or clean.shape!=(len(actual),):raise ValueError('원점/정답/예측/마스크 정렬 불일치')
    end=origins+pd.Timedelta(minutes=15*horizon)
    in_time=np.asarray(end<before);keep=in_time&clean
    audit=dict(candidates=len(origins),excluded_target_end_not_before=int((~in_time).sum()),
               excluded_repaired_target=int((in_time&~clean).sum()),kept=int(keep.sum()),
               last_target_end=str(end[keep].max()) if keep.any() else None)
    return (actual-predicted)[keep],origins[keep],audit


class ResidualPathDistribution:
    """과거 완전 오차 경로의 균등 경험분포. 분포를 저장하는 것이며 새 트리를 학습하지 않는다."""

    def fit(self,residual_paths_kw,residual_keys,metadata=None):
        e=_paths(residual_paths_kw,'residual_paths_kw')
        keys=pd.DatetimeIndex(residual_keys)
        if len(keys)!=len(e) or keys.has_duplicates:raise ValueError('오차 경로 ID(원점)는 경로마다 유일해야 함')
        self.residual_paths_kw=e;self.residual_origins=keys;self.horizon=e.shape[1];self.n_paths=len(e)
        # 상수 덧셈은 순서를 보존하므로 lead별 정렬 오차로 분위수를 계산한다.
        self.sorted_residuals=np.sort(e,axis=0)
        cumulative=np.arange(1,self.n_paths+1)/self.n_paths
        # 가중 이산 분위수: 누적 질량이 처음 level 이상이 되는 경로(inverted CDF).
        self.quantile_index=np.minimum(np.searchsorted(cumulative,QUANTILES-1e-12,side='left'),self.n_paths-1)
        self.metadata=dict(metadata or {},distribution_version=DISTRIBUTION_VERSION,mass='uniform 1/N_pool',
                           n_paths=self.n_paths,horizon=self.horizon,
                           last_residual_target_end=str(keys.max()+pd.Timedelta(minutes=15*self.horizon)),
                           output='raw',calibrated_joint_distribution=False)
        return self

    def _query(self,predicted_kw,threshold_kw):
        if not hasattr(self,'residual_paths_kw'):raise ValueError('fit 이후 사용')
        if self.n_paths<1:raise ValueError('오차 풀이 비어 있음')
        p=_paths(predicted_kw,'predicted_kw',self.horizon);b=np.asarray(threshold_kw,float)
        if b.shape!=(len(p),) or not np.isfinite(b).all() or (b<=0).any():raise ValueError('원점별 양수 B 필요')
        return p,b

    def predict(self,predicted_kw,query_threshold_kw,batch_cells=10_000_000):
        """p_point_raw (N,H), p_any_raw (N,), q_raw_kw (N,H,7), interval90_raw_kw (N,H,2)."""
        p,b=self._query(predicted_kw,query_threshold_kw)
        n,h=p.shape;point=np.empty((n,h));period=np.empty(n);negative=np.zeros(n,np.int64)
        step=max(1,batch_cells//(self.n_paths*h))
        for first in range(0,n,step):
            stop=min(n,first+step)
            # 정의 그대로 Y^(i)=ŷ+e_i 를 만든 뒤 엄격한 > B로 비교한다.
            paths=p[first:stop,None,:]+self.residual_paths_kw[None,:,:]
            event=paths>b[first:stop,None,None]
            point[first:stop]=event.mean(axis=1);period[first:stop]=event.any(axis=2).mean(axis=1)
            negative[first:stop]=(paths<0).any(axis=2).sum(axis=1)
            del paths,event
        q=p[:,:,None]+self.sorted_residuals[self.quantile_index].T[None,:,:]
        return dict(p_point_raw=point,p_any_raw=period,q_raw_kw=q,interval90_raw_kw=q[:,:,[0,-1]],
                    n_negative_paths=negative,metadata=dict(self.metadata,query_threshold='per-origin B'))

    def sample_paths(self,predicted_kw,n_paths=1000,seed=42):
        """같은 분포에서 오차 경로 ID 전체(H개 묶음)를 추출한다. 직접 계산의 Monte Carlo 근사."""
        if not isinstance(n_paths,(int,np.integer)) or n_paths<1:raise ValueError('양의 정수 경로 수 필요')
        p=_paths(predicted_kw,'predicted_kw',self.horizon)
        ids=np.random.default_rng(seed).integers(0,self.n_paths,size=(len(p),n_paths))
        return dict(sample_paths_raw_kw=p[:,None,:]+self.residual_paths_kw[ids],sample_residual_ids=ids,
                    sample_residual_origins=self.residual_origins.asi8[ids],seed=seed,n_samples=int(n_paths))

    def simulate(self,predicted_kw,query_threshold_kw,n_paths=1000,seed=42,batch_cells=10_000_000):
        """시뮬레이션 기반 시점·기간 확률과 표본 분위수. sample_paths와 같은 ID를 쓰되 경로를 배치로만 만든다."""
        if not isinstance(n_paths,(int,np.integer)) or n_paths<1:raise ValueError('양의 정수 경로 수 필요')
        p,b=self._query(predicted_kw,query_threshold_kw)
        n,h=p.shape
        ids=np.random.default_rng(seed).integers(0,self.n_paths,size=(n,n_paths))
        point=np.empty((n,h));period=np.empty(n);q=np.empty((n,h,len(QUANTILES)))
        step=max(1,batch_cells//(n_paths*h))
        for first in range(0,n,step):
            stop=min(n,first+step)
            paths=p[first:stop,None,:]+self.residual_paths_kw[ids[first:stop]]
            event=paths>b[first:stop,None,None]
            point[first:stop]=event.mean(axis=1);period[first:stop]=event.any(axis=2).mean(axis=1)
            q[first:stop]=np.quantile(paths,QUANTILES,axis=1,method='inverted_cdf').transpose(1,2,0)
            del paths,event
        return dict(p_point_sim=point,p_any_sim=period,q_sim_kw=q,sample_residual_ids=ids.astype(np.int32),
                    seed=seed,n_samples=int(n_paths))


def mixture_quantiles(distributions,predicted_list,tol=1e-4):
    """같은 무게 혼합(모델 k의 경로 ŷ_k+e_k,i에 질량 1/(K·N_k))의 lead별 분위수.

    혼합 CDF F(x)=mean_k G_k(x-ŷ_k)를 이분법으로 풀어 inverted-CDF 분위수를 tol kW 이내로 근사한다.
    """
    K=len(distributions)
    if K<1 or K!=len(predicted_list):raise ValueError('혼합 구성원/예측 수 불일치')
    preds=[_paths(p,'predicted_kw',d.horizon) for d,p in zip(distributions,predicted_list)]
    if any(p.shape!=preds[0].shape for p in preds):raise ValueError('구성원 예측 형태 불일치')
    n,h=preds[0].shape
    lo=np.min([p+d.sorted_residuals[0][None,:] for d,p in zip(distributions,preds)],axis=0)-1.
    hi=np.max([p+d.sorted_residuals[-1][None,:] for d,p in zip(distributions,preds)],axis=0)
    def cdf(x):
        out=np.zeros((n,h))
        for d,p in zip(distributions,preds):
            for j in range(h):out[:,j]+=np.searchsorted(d.sorted_residuals[:,j],x[:,j]-p[:,j],side='right')/d.n_paths
        return out/K
    q=np.empty((n,h,len(QUANTILES)))
    for k,level in enumerate(QUANTILES):
        a=lo.copy();b=hi.copy()
        while (b-a).max()>tol:
            mid=(a+b)/2;ok=cdf(mid)>=level-1e-12
            b=np.where(ok,mid,b);a=np.where(ok,a,mid)
        q[:,:,k]=b
    return q


def mixture_sample_quantiles(distributions,predicted_list,ids_list,batch_cells=10_000_000):
    """구성원별 시뮬레이션 경로(같은 표본 수)를 합친 표본 분위수. 결과 앙상블의 시뮬레이션 기반."""
    preds=[_paths(p,'predicted_kw',d.horizon) for d,p in zip(distributions,predicted_list)]
    n,h=preds[0].shape;m=sum(ids.shape[1] for ids in ids_list)
    q=np.empty((n,h,len(QUANTILES)));step=max(1,batch_cells//(m*h))
    for first in range(0,n,step):
        stop=min(n,first+step)
        paths=np.concatenate([p[first:stop,None,:]+d.residual_paths_kw[ids[first:stop]] for d,p,ids in zip(distributions,preds,ids_list)],axis=1)
        q[first:stop]=np.quantile(paths,QUANTILES,axis=1,method='inverted_cdf').transpose(1,2,0)
        del paths
    return q


def history_probabilities(y,b):
    """과거 실제 경로에 현재 B를 적용한 시점·기간 초과빈도(기존 기준선과 같은 정의)."""
    y=_paths(y,'history');b=np.asarray(b,float)
    point=np.empty((len(b),y.shape[1]));period=np.empty(len(b))
    for value in np.unique(b):
        chosen=b==value;event=y>value
        point[chosen]=event.mean(axis=0);period[chosen]=event.any(axis=1).mean()
    return point,period


def binary_scores(event,p):
    from sklearn.metrics import average_precision_score
    event=np.asarray(event,bool).ravel();p=np.asarray(p,float).ravel()
    if not len(event) or event.shape!=p.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('정답/확률 계약 오류')
    clipped=np.clip(p,1e-10,1-1e-10)
    return dict(Brier=float(np.mean((p-event)**2)),
                logloss=float(-np.mean(event*np.log(clipped)+(1-event)*np.log(1-clipped))),
                AP=float(average_precision_score(event,p)) if event.any() else None,
                n=int(event.size),n_positive=int(event.sum()),mean_probability=float(p.mean()))


def quantile_scores(actual,q):
    actual=_paths(actual,'actual');q=np.asarray(q,float)
    if q.shape!=(*actual.shape,len(QUANTILES)):raise ValueError('분위수 배열 형태 오류')
    out={}
    for j,level in enumerate(QUANTILES):
        error=actual-q[:,:,j]
        out[f'pinball_Q{int(round(level*100)):02d}']=float(np.maximum(level*error,(level-1)*error).mean())
        out[f'coverage_Q{int(round(level*100)):02d}']=float((actual<=q[:,:,j]).mean())
    out['mean_pinball']=float(np.mean([out[f'pinball_Q{int(round(l*100)):02d}'] for l in QUANTILES]))
    out['coverage90']=float(((actual>=q[:,:,0])&(actual<=q[:,:,-1])).mean())
    out['interval90_width_kw']=float((q[:,:,-1]-q[:,:,0]).mean())
    return out
