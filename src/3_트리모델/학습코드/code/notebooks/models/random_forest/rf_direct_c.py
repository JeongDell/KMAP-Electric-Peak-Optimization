"""정확한 전체 C로 공동 leaf 출력값과 terminal split을 최적화하는 숲.

표준 sklearn RF의 분할 초기화는 보존한다. 전체 C의 직접 학습과
수치적 최적해 보장은 별개이며, solver 상태와 best feasible iterate를 기록한다.
"""
# 역할: 실행기가 호출하는 학습 모듈이다. 이 파일 자체에는 CLI 실행부가 없다.
# 흐름: sklearn RF 초기화 → 잎 출력 공동 C 최적화 → C를 줄이는 추가 분할 탐색.
# 배열 기준: N=예측 원점 수, H=미래 15분 목표 수, L=숲 전체 잎 출력 슬롯 수.
# y/p/w는 (N,H), theta는 (L,H), A는 (N,L), 숲의 예측은 p=A@theta다.
from dataclasses import dataclass, field
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, minimize


def c_value_gradient(actual, predicted, weights, window_steps):
    # 전체 C와 예측값 p에 대한 부분경사 dC/dp를 함께 반환한다.
    # w는 실제값·원점별 피크 기준으로 미리 계산한 고정값이며 여기서 학습하지 않는다.
    y=np.asarray(actual,float);p=np.asarray(predicted,float);w=np.asarray(weights,float)
    if y.ndim!=2 or not y.size or p.shape!=y.shape or w.shape!=y.shape:
        raise ValueError('정답·예측·셀가중치는 비어 있지 않은 동일 (N,H)이어야 합니다.')
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(w).all() or (w<0).any():
        raise ValueError('유한 값과 비음수 셀가중치가 필요합니다.')
    if not isinstance(window_steps,(int,np.integer)) or window_steps<1 or y.shape[1]%window_steps:
        raise ValueError('손실창은 H를 나누는 양의 정수여야 합니다.')
    # 원점별 미래 경로를 겹치지 않는 손실 블록으로 나눈다.
    # H가 window_steps로 나뉘므로 reshape가 서로 다른 원점의 경로를 섞지 않는다.
    e=y-p;blocks=e.reshape(-1,window_steps)
    r=np.sqrt(np.mean(blocks**2,axis=1));k=int(np.ceil(.2*len(r)))
    # 최악 20%는 올림 개수로 선택한다. 동률은 안정 정렬로 재현 가능하게 처리한다.
    top=np.argsort(r,kind='stable')[-k:]
    mean=float(r.mean());tail=float(r[top].mean());under=np.maximum(e,0)
    # L_peak: 과소오차만 제곱·가중한 뒤 전체 N*H 셀 평균에 마지막 루트를 씌운다.
    # 과대예측은 under=0이지만 분모에 포함한다. sum(w)로 나누지 않는다.
    peak=float(np.sqrt(np.mean(w*under**2)))
    # 평균·최악 블록 항의 경사를 합친다. 오차 0의 블록에는 0 부분경사를 쓴다.
    coeff=np.full(len(r),.5/len(r));coeff[top]+=.3/k
    inverse=np.divide(coeff,window_steps*r,out=np.zeros_like(r),where=r>0)
    gradient=(-inverse[:,None]*blocks).reshape(y.shape)
    # y-p의 p 미분은 -1이므로 과소예측 셀의 경사는 음수다.
    # L_peak=0이면 이 항의 0 부분경사를 선택해 0으로 나누는 일을 피한다.
    if peak>0:gradient-=.2*w*under/(y.size*peak)
    metrics=dict(Loss_C=.5*mean+.3*tail+.2*peak,L_mean=mean,L_CVaR=tail,L_peak=peak,
        RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.abs(e).mean()),
        over_RMSE=float(np.sqrt(np.mean(np.maximum(-e,0)**2))),bias=float((p-y).mean()))
    return metrics,gradient


@dataclass
class ForestCModel:
    # base는 초기 sklearn 숲, theta는 각 잎의 H개 미래 출력값이다.
    # initial은 ±delta 출력 제약의 기준이며 최적화 중 갱신하지 않는다.
    # 추가 분할은 splits에 별도로 저장하여 원래 sklearn tree 구조를 보존한다.
    base: object
    theta: np.ndarray
    initial: np.ndarray
    node_to_column: list
    splits: list=field(default_factory=list)
    lower: object=None
    upper: object=None

    @classmethod
    def from_forest(cls,forest):
        # 나무마다 서로 다른 leaf 번호를 숲 전체의 연속 출력 슬롯으로 연결한다.
        if not hasattr(forest,'estimators_') or not forest.estimators_:
            raise ValueError('적합된 회귀 RF 필요')
        arrays=[];maps=[];offset=0
        for estimator in forest.estimators_:
            tree=estimator.tree_;leaves=np.flatnonzero(tree.children_left<0)
            mapping=np.full(tree.node_count,-1,dtype=np.int64)
            mapping[leaves]=offset+np.arange(len(leaves));maps.append(mapping)
            arrays.append(tree.value[leaves,:,0]);offset+=len(leaves)
        theta=np.concatenate(arrays).astype(float)
        return cls(forest,theta.copy(),theta.copy(),maps)

    @property
    def estimators_(self):
        # 기존 잎 경로 분포 모듈이 요구하는 node_count 계약을 제공한다.
        # 원래 sklearn estimator를 반환하는 대신 추가 자식 수를 반영한 메타정보다.
        counts=np.bincount([s['tree'] for s in self.splits],minlength=len(self.base.estimators_))
        return [SimpleNamespace(tree_=SimpleNamespace(node_count=t.tree_.node_count+2*int(c)))
                for t,c in zip(self.base.estimators_,counts)]

    def apply(self,x):
        # 학습·추론에 같은 분기 규칙을 적용한다. 실제 미래 전력은 입력하지 않는다.
        x=np.asarray(x,np.float32)
        if x.ndim!=2 or x.shape[1]!=self.base.n_features_in_ or not np.isfinite(x).all():
            raise ValueError('RF 입력 스키마와 유한 값을 확인하세요.')
        original=self.base.apply(x);nodes=original.copy()
        for s in self.splits:
            keep=original[:,s['tree']]==s['parent_node']
            nodes[keep,s['tree']]=np.where(x[keep,s['feature']]<=s['threshold'],s['left_node'],s['right_node'])
        return nodes

    def design(self,x):
        # 한 원점은 각 나무의 잎 하나에 도달한다. 각 잎에 1/나무수 질량을 주어
        # sparse A@theta가 sklearn RF와 같은 나무 평균을 계산하게 한다.
        nodes=self.apply(x);columns=np.empty_like(nodes)
        for t,mapping in enumerate(self.node_to_column):columns[:,t]=mapping[nodes[:,t]]
        if (columns<0).any():raise ValueError('예측 leaf와 출력 컬럼 대응 실패')
        rows=np.repeat(np.arange(len(nodes)),nodes.shape[1])
        return sparse.csr_matrix((np.full(columns.size,1/nodes.shape[1]),(rows,columns.ravel())),
            shape=(len(nodes),len(self.theta)))

    # 현재 분기 구조와 학습된 잎 출력값으로 H개 미래 전력을 한 번에 반환한다.
    def predict(self,x):return np.asarray(self.design(x)@self.theta)

    def add_split(self,tree,parent_node,feature,threshold):
        # 초기 terminal leaf 하나를 두 자식으로 확장한다. 자식의 재분할은 지원하지 않는다.
        if any(s['tree']==tree and s['parent_node']==parent_node for s in self.splits):
            raise ValueError('동일 초기 terminal leaf의 중복 분할은 지원하지 않습니다.')
        if not 0<=tree<len(self.base.estimators_) or not 0<=feature<self.base.n_features_in_ or not np.isfinite(threshold):
            raise ValueError('유효한 나무·피처·임계값 필요')
        mapping=self.node_to_column[tree];parent=int(mapping[parent_node])
        if parent<0:raise ValueError('terminal leaf만 확장할 수 있습니다.')
        node=len(mapping);left=len(self.theta);right=left+1
        self.node_to_column[tree]=np.r_[mapping,left,right]
        # 두 자식을 부모와 같은 출력으로 시작하면 분할 전 예측을 그대로 재현한다.
        self.theta=np.vstack([self.theta,self.theta[parent],self.theta[parent]])
        self.initial=np.vstack([self.initial,self.initial[parent],self.initial[parent]])
        # 자식도 부모의 최초 출력 허용 범위를 상속하여 분할마다 제약이 느슨해지지 않는다.
        if self.lower is not None:
            self.lower=np.vstack([self.lower,self.lower[parent],self.lower[parent]])
            self.upper=np.vstack([self.upper,self.upper[parent],self.upper[parent]])
        self.splits.append(dict(tree=int(tree),parent_node=int(parent_node),parent_column=parent,
            feature=int(feature),threshold=float(threshold),left_node=node,right_node=node+1,
            left_column=left,right_column=right))
        return left,right


# SciPy 계산 중 시간 예산을 넘으면 최적화 루프만 빠져나오기 위한 내부 예외다.
class _BudgetStop(Exception):pass


def optimize_leaves(model,x,y,w,window_steps,delta=None,max_iter=120,deadline=None):
    # 분기 구조를 고정한 채 모든 나무·잎·예측거리 출력 theta를 공동 최적화한다.
    # delta=3/10은 초기 출력에서 허용하는 kW 변경폭이며 가중치 상한이 아니다.
    before=perf_counter();deadline=float('inf') if deadline is None else deadline
    if delta is not None:
        if not np.isfinite(delta) or delta<0:raise ValueError('출력 변경 한계는 비음수 유한 kW여야 합니다.')
        model.lower=np.maximum(0,model.initial-delta);model.upper=model.initial+delta
    if model.lower is None:raise ValueError('초기 최적화에는 delta가 필요합니다.')
    if max_iter<1:raise ValueError('양수 반복 한계 필요')
    # 구조가 고정되어 A는 최적화 중 변하지 않는다. theta만 최적화 변수다.
    a=model.design(x);shape=model.theta.shape
    theta=np.clip(model.theta,model.lower,model.upper)
    initial,_=c_value_gradient(y,a@theta,w,window_steps)
    best=dict(value=initial['Loss_C'],theta=theta.copy());curve=[dict(iteration=0,seconds=0.,**initial)]

    def objective(flat):
        # SciPy의 1차원 변수 벡터를 (L,H)로 복원한다.
        # 연쇄법칙: dC/dtheta = A.T @ (dC/dp).
        if perf_counter()>=deadline:raise _BudgetStop()
        candidate=flat.reshape(shape);score,g=c_value_gradient(y,a@candidate,w,window_steps)
        if score['Loss_C']<best['value']:
            best.update(value=score['Loss_C'],theta=candidate.copy())
        return score['Loss_C'],np.asarray(a.T@g).ravel()

    def callback(flat):
        # 반복별 전체 C와 세 구성항을 저장해 학습 과정의 절충을 추적한다.
        score,_=c_value_gradient(y,a@flat.reshape(shape),w,window_steps)
        curve.append(dict(iteration=len(curve),seconds=perf_counter()-before,**score))

    try:
        # 출력 제약이 있는 L-BFGS-B를 사용한다. top-k·루트의 비매끄러운 지점은
        # 위에서 선택한 부분경사로 처리하며 solver 성공을 전역 최적 인증으로 해석하지 않는다.
        result=minimize(objective,theta.ravel(),jac=True,method='L-BFGS-B',
            bounds=Bounds(model.lower.ravel(),model.upper.ravel()),callback=callback,
            options=dict(maxiter=int(max_iter),maxcor=5,maxls=30,ftol=1e-9,gtol=1e-6))
        status=int(result.status);message=str(result.message);success=bool(result.success);n_iter=int(result.nit)
    except _BudgetStop:
        status=99;message='시간 예산으로 중단; best feasible iterate 저장';success=False;n_iter=len(curve)-1
    # 마지막 반복보다 앞선 값이 더 좋을 수 있어 실행 중 가장 낮은 C의 출력을 저장한다.
    # 시간 중단도 숨기지 않고 solver_success=False와 상태 99로 기록한다.
    model.theta=best['theta'];final,g=c_value_gradient(y,a@model.theta,w,window_steps)
    grad=np.asarray(a.T@g)
    # 경계 제약을 반영한 잔여 경사 크기다. 비수렴 여부를 판단할 보조 진단으로 남긴다.
    projected=model.theta-np.clip(model.theta-grad,model.lower,model.upper)
    curve.append(dict(iteration=len(curve),seconds=perf_counter()-before,saved_best=True,**final))
    return dict(curve=curve,solver_status=status,solver_message=message,solver_success=success,
        n_iterations=n_iter,elapsed_seconds=perf_counter()-before,initial_C=initial['Loss_C'],final_C=final['Loss_C'],
        projected_gradient_inf=float(np.max(np.abs(projected))),n_parameters=int(model.theta.size),
        n_nonzero_routes=int(a.nnz),delta_kw=delta,
        max_leaf_change_kw=float(np.max(np.abs(model.theta-model.initial))))


def refine_splits(model,x,y,w,window_steps,min_leaf=20,max_splits=4,proposals=12,max_iter=60,seed=42,deadline=None):
    # 공동 잎 학습 이후, 초기 잎 일부를 나누어 전체 학습 C가 줄어드는 구조만 채택한다.
    # sklearn의 국소 MSE 분할을 C 분할로 재학습하는 대신 별도 terminal routing을 추가한다.
    if model.lower is None:raise ValueError('공동 leaf 학습과 출력 범위를 먼저 설정하세요.')
    if min_leaf<1 or max_splits<0 or proposals<1:raise ValueError('분할 탐색 크기를 확인하세요.')
    before=perf_counter();deadline=float('inf') if deadline is None else deadline
    rng=np.random.default_rng(seed);x=np.asarray(x,np.float32);y=np.asarray(y,float)
    original=model.base.apply(x);logs=[];joint=[]
    for round_id in range(max_splits):
        if perf_counter()>=deadline:break
        p=model.predict(x);score,g=c_value_gradient(y,p,w,window_steps);current=score['Loss_C']
        used={(s['tree'],s['parent_node']) for s in model.splits};eligible=[]
        # 전체 C 경사가 큰 원점이 많이 도달하는 잎을 우선 탐색한다.
        # 후보를 정하는 기준일 뿐, 채택 여부는 전체 학습 셀의 정확한 C로 결정한다.
        impact=np.abs(g).sum(axis=1)
        for tree in range(original.shape[1]):
            for node in np.unique(original[:,tree]):
                if (tree,int(node)) in used:continue
                ids=np.flatnonzero(original[:,tree]==node)
                if len(ids)>=2*min_leaf:eligible.append((float(impact[ids].sum()),tree,int(node),ids))
        eligible.sort(key=lambda item:item[0],reverse=True)
        if not eligible:break
        best=None;round_logs=[]
        # 영향이 큰 상위12 leaf에서 후보를 분산하고 손실은 전체 학습 셀로 계산한다.
        for trial in range(proposals):
            if perf_counter()>=deadline:break
            _,tree,node,ids=eligible[trial%min(len(eligible),12)]
            variable=np.flatnonzero(np.ptp(x[ids],axis=0)>0)
            if not len(variable):continue
            # 값이 변하는 피처 하나와 잎 내부 25/50/75% 위치의 임계값을 제안한다.
            feature=int(rng.choice(variable));quantile=[.25,.5,.75][trial%3]
            threshold=float(np.quantile(x[ids,feature],quantile))
            left_ids=ids[x[ids,feature]<=threshold];right_ids=ids[x[ids,feature]>threshold]
            entry=dict(round=round_id,trial=trial,tree=tree,parent_node=node,feature=feature,
                threshold=threshold,n_left=len(left_ids),n_right=len(right_ids),C_before=current,accepted=False)
            if min(len(left_ids),len(right_ids))<min_leaf:
                entry.update(status='최소 leaf 표본 미달',C_after=current);round_logs.append(entry);continue
            parent=int(model.node_to_column[tree][node]);parent_value=model.theta[parent].copy()
            initial=np.vstack([parent_value,parent_value]);lb=np.tile(model.lower[parent],(2,1));ub=np.tile(model.upper[parent],(2,1))
            local_best=dict(value=current,theta=initial.copy())

            def objective(flat):
                # 후보 평가에서는 다른 잎을 고정하고 두 자식의 H개 출력만 최적화한다.
                # 해당 나무의 변화량을 1/나무수로 반영하되 C는 전체 학습 경로에서 계산한다.
                if perf_counter()>=deadline:raise _BudgetStop()
                values=flat.reshape(initial.shape);candidate=p.copy()
                candidate[left_ids]+=(values[0]-parent_value)/original.shape[1]
                candidate[right_ids]+=(values[1]-parent_value)/original.shape[1]
                metrics,gradient=c_value_gradient(y,candidate,w,window_steps)
                if metrics['Loss_C']<local_best['value']:local_best.update(value=metrics['Loss_C'],theta=values.copy())
                derivatives=np.vstack([gradient[left_ids].sum(axis=0),gradient[right_ids].sum(axis=0)])/original.shape[1]
                return metrics['Loss_C'],derivatives.ravel()

            try:
                result=minimize(objective,initial.ravel(),jac=True,method='L-BFGS-B',bounds=Bounds(lb.ravel(),ub.ravel()),
                    options=dict(maxiter=max_iter,maxcor=5,maxls=25,ftol=1e-9,gtol=1e-7))
                entry.update(status=str(result.message),solver_status=int(result.status),n_iterations=int(result.nit))
            except _BudgetStop:entry.update(status='시간 예산 중단',solver_status=99,n_iterations=None)
            entry['C_after']=local_best['value'];round_logs.append(entry)
            # 수치 오차 수준을 넘는 감소가 있는 후보 중 가장 낮은 C만 채택한다.
            if local_best['value']<current-1e-9 and (best is None or local_best['value']<best['value']):
                best=dict(value=local_best['value'],theta=local_best['theta'],entry=entry)
        if best is None:
            logs.extend(round_logs);break
        chosen=best['entry'];chosen['accepted']=True
        left,right=model.add_split(chosen['tree'],chosen['parent_node'],chosen['feature'],chosen['threshold'])
        model.theta[[left,right]]=best['theta']
        # 가상 후보의 C가 실제 저장된 새 routing에서도 같은지 즉시 검증한다.
        actual,_=c_value_gradient(y,model.predict(x),w,window_steps)
        if not np.isclose(actual['Loss_C'],best['value'],atol=1e-8,rtol=1e-9):
            raise RuntimeError('채택 분할의 새 route와 후보 C가 일치하지 않습니다.')
        # 채택된 구조에서 전체 잎을 다시 공동 최적화해 나무 평균 간 상호작용을 반영한다.
        post=optimize_leaves(model,x,y,w,window_steps,max_iter=min(60,max_iter),deadline=deadline)
        chosen['C_after_joint']=post['final_C'];joint.append(dict(round=round_id,**post));logs.extend(round_logs)
    return dict(proposals=logs,joint_optimizations=joint,accepted_splits=len(model.splits),
        elapsed_seconds=perf_counter()-before,budget_expired=perf_counter()>=deadline)
