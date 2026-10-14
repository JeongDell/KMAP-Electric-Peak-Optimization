"""전체 경로 C 부분경사를 단일 native 트리에 투영하는 외부 부스팅.

native MSE는 음의 경사 회귀용이다. 학습 C의 세 항이 업데이트를 결정하며,
native custom objective의 행별 가산/대각 Hessian 가정을 요구하지 않는다.
"""
from time import perf_counter
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from full_c_loss import loss_gradient


def tree(library,params):
    depth=params.get('depth',4);l2=params.get('l2',1);threads=params.get('threads',4)
    if library=='xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=1,learning_rate=1,max_depth=depth,min_child_weight=params.get('child',1),
            reg_lambda=l2,base_score=0,tree_method='hist',n_jobs=threads,random_state=42,subsample=1,colsample_bytree=1)
    if library=='lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=1,learning_rate=1,max_depth=depth,num_leaves=min(params.get('leaves',2**depth),2**depth),
            min_child_samples=params.get('leaf',20),reg_lambda=l2,boost_from_average=False,n_jobs=threads,
            random_state=42,verbosity=-1,subsample=1,colsample_bytree=1)
    if library=='catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=1,learning_rate=1,depth=depth,l2_leaf_reg=l2,loss_function='RMSE',
            bootstrap_type='No',boosting_type='Plain',boost_from_average=False,random_strength=0,
            thread_count=threads,random_seed=42,allow_writing_files=False,verbose=False)
    raise ValueError(f'지원하지 않는 라이브러리: {library}')


class CBoosting:
    def __init__(self,library,params,objective='C'):
        self.library=library;self.params=dict(params);self.objective=objective
        self.trees=[];self.coefficients=[];self.trace=[];self.stop_reason='not_fitted'

    def _check(self,x,y=None):
        x=np.asarray(x,np.float32)
        if x.ndim!=2 or not np.isfinite(x).all() or not len(x):raise ValueError('유한 2D 피처 필요')
        if hasattr(self,'n_features') and x.shape[1]!=self.n_features:raise ValueError('피처 수 불일치')
        if not self.direct and len(x)%self.horizon:raise ValueError('pooled 입력은 원점별 전체 H 필요')
        if y is not None and len(x)!=(len(y) if self.direct else y.size):raise ValueError('입력/전체 경로 정렬 불일치')
        return x

    def _direction(self,models,x):
        def prediction(model):
            return model.booster_.predict(x,num_threads=self.params.get('threads',4)) if self.library=='lightgbm' else model.predict(x)
        if self.direct:return np.column_stack([prediction(m) for m in models])
        return prediction(models[0]).reshape(-1,self.horizon)

    def fit(self,x,y,w,width,x_valid=None,y_valid=None,w_valid=None,direct=False,deadline=None):
        self.trees=[];self.coefficients=[];self.trace=[];self.stop_reason='not_fitted'
        y=np.asarray(y,float);self.horizon=y.shape[1];self.direct=direct;self.n_features=np.asarray(x).shape[1]
        x=self._check(x,y);deadline=float('inf') if deadline is None else deadline
        if x_valid is not None:x_valid=self._check(x_valid,y_valid)
        self.initial=y.mean(axis=0) if direct else np.full(self.horizon,y.mean())
        p=np.broadcast_to(self.initial,y.shape).copy()
        pv=None if x_valid is None else np.broadcast_to(self.initial,np.asarray(y_valid).shape).copy()
        scale=y.size*max(float(y.std()),1.)
        best=float('inf');best_round=0;stale=0;rounds=int(self.params.get('rounds',100));nu=self.params.get('shrinkage',.1)
        if not 0<nu<=1 or rounds<0:raise ValueError('비음수 rounds 및 0<shrinkage<=1 필요')
        for round_id in range(rounds+1):
            score,gradient,details=loss_gradient(y,p,w,width,self.objective)
            entry=dict(round=round_id,train_C=score['Loss_C'],objective_value=details['value'],tail_blocks=details['tail_blocks'])
            entry.update({key:score[key] for key in ['L_mean','L_CVaR','L_peak','RMSE','MAE']})
            entry.update({f'gradient_norm_{k}':float(np.linalg.norm(v)) for k,v in details['components'].items()})
            if round_id:entry.update(lambda_=chosen_lambda,n_trees=len(models),seconds=perf_counter()-started)
            if pv is not None and (round_id%5==0 or round_id==rounds):
                v=loss_gradient(y_valid,pv,w_valid,width)[0]['Loss_C'];entry['valid_C']=v
                if v<best-1e-4:best=v;best_round=round_id;stale=0
                else:stale+=1
            self.trace.append(entry)
            if round_id==rounds:self.stop_reason='max_rounds';break
            stop_file=os.environ.get('KAMP_BOOSTING_STOP_FILE')
            if stop_file and Path(stop_file).is_file():self.stop_reason='external_stop';break
            if stale>=8:self.stop_reason='validation_patience';break
            if perf_counter()>=deadline:self.stop_reason='budget';break
            if not np.any(gradient):self.stop_reason='zero_gradient';break
            started=perf_counter();target=-scale*gradient
            models=[]
            for lead in range(self.horizon if direct else 1):
                estimator=tree(self.library,self.params)
                estimator.fit(x,target[:,lead] if direct else target.ravel())
                models.append(estimator)
            direction=self._direction(models,x)
            # 설계 순서: 비축소 방향의 최적lambda → shrinkage 적용 → 실제 감소 확인.
            def value(step):return loss_gradient(y,p+step*direction,w,width,self.objective)[2]['value']
            result=minimize_scalar(value,bounds=(0.,1.),method='bounded',options=dict(xatol=1e-3,maxiter=24))
            options=[(details['value'],0.),(value(1.),1.),(float(result.fun),float(result.x))]
            next_value,chosen_lambda=min(options)
            applied_value=value(nu*chosen_lambda)
            entry.update(search_lambda=chosen_lambda,unshrunk_objective=next_value,after_shrinkage_objective=applied_value)
            if applied_value>=details['value']-max(1e-10,1e-10*abs(details['value'])):
                self.stop_reason='no_descent';entry.update(lambda_=0.,attempted_lambda=chosen_lambda,rejected_objective=applied_value);break
            self.trees.append(models);self.coefficients.append(nu*chosen_lambda)
            p+=nu*chosen_lambda*direction
            if pv is not None:pv+=nu*chosen_lambda*self._direction(models,x_valid)
        # 마지막 비5배수의 실제 checkpoint도 평가하여 저장 모델의 선택 근거를 남긴다.
        if pv is not None and 'valid_C' not in self.trace[-1]:
            v=loss_gradient(y_valid,pv,w_valid,width)[0]['Loss_C'];self.trace[-1]['valid_C']=v
            if v<best-1e-4:best_round=len(self.trees);best=v
        self.fitted_rounds=len(self.trees)
        self.selected_round=best_round if pv is not None else len(self.trees)
        if pv is not None:
            self.trees=self.trees[:best_round];self.coefficients=self.coefficients[:best_round]
        return self

    def predict(self,x):
        x=self._check(x);n=len(x) if self.direct else len(x)//self.horizon
        p=np.broadcast_to(self.initial,(n,self.horizon)).copy()
        for models,coefficient in zip(self.trees,self.coefficients):p+=coefficient*self._direction(models,x)
        return p
