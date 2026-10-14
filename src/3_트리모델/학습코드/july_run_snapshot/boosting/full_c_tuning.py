"""고정 continuous2 전체 C 학습, F1→F2→F3 독립 튜닝, 동결 후9월 평가.

실제 후보마다 독립 MD를 작성한다. 완료 자료는 새 AGENTS final 계약으로
사본을 묶으며 원본 데이터와 활성 코드를 이동하지 않는다.
"""
import argparse
from datetime import datetime
import gzip
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
from time import perf_counter
import traceback
from uuid import uuid4
from zoneinfo import ZoneInfo
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from full_c_boosting import CBoosting
from full_c_data import Dataset, PROFILES, ACTIVE_PROFILES, FOLDS, REPAIR_SOURCE, load_frame
from full_c_loss import WEIGHT_RULE, loss_gradient
import boosting_common as bc

ROOT=bc.ROOT
KST=ZoneInfo('Asia/Seoul')
LIBRARIES=['xgboost','lightgbm'] # CatBoost는 2026-10-06 사용자 지시로 신규 실행에서 제외.
BASE=dict(depth=4,l2=1,rounds=100,shrinkage=.1,threads=4)
CANDIDATES={
    'xgboost':[dict(BASE),dict(BASE,depth=3,l2=5,child=5),dict(BASE,depth=6,l2=5,child=10)],
    'lightgbm':[dict(BASE,leaves=16,leaf=20),dict(BASE,depth=3,leaves=8,leaf=40,l2=5),dict(BASE,depth=6,leaves=32,leaf=40,l2=5)]}
FEATURES=['F1','F2','F3']
SCENARIOS=['observed','future']
TUNING_POLICY='independent_feature_scenario_v2'


def selection_key(library,profile,feature,scenario):
    return f'{library}/{profile}/{feature}/{scenario}'


def build_plan(models=None,profiles=None,features=None,scenarios=None,a_control=True):
    def checked(values,allowed,label):
        values=list(allowed) if values is None else list(values)
        if not values or len(values)!=len(set(values)) or not set(values)<=set(allowed):
            raise ValueError(f'{label}: 비어 있지 않은 중복 없는 {list(allowed)} 목록 필요')
        return [v for v in allowed if v in values]
    models=checked(models,LIBRARIES,'models');profiles=checked(ACTIVE_PROFILES if profiles is None else profiles,PROFILES,'profiles')
    features=checked(features,FEATURES,'features');scenarios=checked(scenarios,SCENARIOS,'scenarios')
    contexts=[dict(library=lib,profile=profile,feature=feature,scenario=scenario)
        for feature in features for profile in profiles for scenario in scenarios for lib in models]
    c_fits=sum(len(CANDIDATES[c['library']])*len(FOLDS) for c in contexts)
    a_fits=len(contexts)*len(FOLDS) if a_control else 0
    return dict(tuning_policy=TUNING_POLICY,models=models,profiles=profiles,feature_order=features,
        scenarios=scenarios,a_control=bool(a_control),contexts=contexts,selection_contexts=len(contexts),
        C_cv_fits=c_fits,A_control_cv_fits=a_fits,final_fits=len(contexts),total_fits=c_fits+a_fits+len(contexts),
        validation_months=[7,8],selection_metric='equal-month mean Loss_C; lower is better',
        September_used_for_selection=False)


def reproduction_arguments(plan,budget,rounds):
    args=['--budget-seconds',str(budget),'--rounds',str(rounds)]
    for option,key in [('models','models'),('profiles','profiles'),('features','feature_order'),('scenarios','scenarios')]:
        args+=['--'+option]+plan[key]
    if not plan['a_control']:args+=['--skip-a-control']
    return args


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(KST).isoformat()
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def table(rows):return pd.DataFrame(rows).round(4).to_markdown(index=False) if rows else '실제 완료 결과 없음.'


def fit_status(library,experiment_id,fold,entry):
    return (f"FIT {library} {experiment_id}/{fold} C={entry['Loss_C']:.4f} "
            f"RMSE={entry['RMSE']:.4f} kW rounds={entry['selected_round']} seconds={entry['seconds']:.1f}")


def check_input_versions(config):
    stop_file=config.get('external_stop_file')
    if stop_file and Path(stop_file).is_file():raise TimeoutError('자원 감시 중단 요청: 메모리 보호')
    for path_key,hash_key in [('source_path','source_sha256'),('repair_source_path','repair_source_sha256')]:
        if path_key in config and digest(config[path_key])!=config[hash_key]:
            raise ValueError(f'실행 중 입력 변경: {config[path_key]}; 서로 다른 버전의 결과 혼합 금지')


def clean_target_scores(data,indices,predicted):
    keep=~data.has_repaired_target[indices]
    result=dict(n_clean_target_origins=int(keep.sum()),n_repaired_target_origins=int((~keep).sum()))
    if keep.any():
        scores=loss_gradient(data.y[indices][keep],predicted[keep],data.w_eval[indices][keep],data.width)[0]
        result.update({f'clean_targets_{k}':scores[k] for k in ['Loss_C','RMSE','MAE','L_peak']})
    else:result.update({f'clean_targets_{k}':None for k in ['Loss_C','RMSE','MAE','L_peak']})
    return result


def select_candidate(rows):
    frame=pd.DataFrame(rows);frame=frame.loc[frame.fold.isin(FOLDS)].copy()
    if frame.empty or frame.duplicated(['candidate','fold']).any():raise ValueError('검증 후보/월 키 없음 또는중복')
    for column in ['library','profile','feature','scenario']:
        if column in frame and frame[column].nunique(dropna=False)!=1:
            raise ValueError(f'후보 선정에 서로 다른 {column} 조건을 섞을 수 없음')
    choices=[]
    for candidate,group in frame.groupby('candidate'):
        if set(group.fold)!=set(FOLDS) or not np.isfinite(group.Loss_C).all():raise ValueError('완료된 7·8월만 선정 가능')
        choice=dict(candidate=candidate,Loss_C=float(group.Loss_C.mean()))
        if 'RMSE' in group:
            if not np.isfinite(group.RMSE).all():raise ValueError('유한 RMSE 필요')
            choice['RMSE_month_mean']=float(group.RMSE.mean())
        choices.append(choice)
    return min(choices,key=lambda x:(x['Loss_C'],x['candidate']))


def selection_flags(config,status,selections):
    key=selection_key(**{k:config[k] for k in ['library','profile','feature','scenario']})
    chosen=selections.get(key)
    if chosen is None and config['feature']=='F2' and config['scenario']=='observed':
        chosen=selections.get(f"{config['library']}/{config['profile']}") # 이전 기록의 검증 호환
    validation_selected=(status=='완료' and config['phase']=='cv'
        and (chosen or {}).get('candidate')==config['candidate'])
    return validation_selected,status=='완료' and config['phase']=='final'


def comparison_rows(rows,selections):
    result=[]
    for chosen in selections.values():
        context={k:chosen[k] for k in ['library','profile','feature','scenario']}
        eligible=[r for r in rows if r['phase']=='cv' and all(r[k]==v for k,v in context.items())]
        baseline=[r for r in eligible if r['candidate']=='c00']
        picked=[r for r in eligible if r['candidate']==chosen['candidate']]
        control=[r for r in eligible if r['candidate']=='a_control']
        if len(baseline)!=2 or len(picked)!=2:continue
        base=float(np.mean([r['Loss_C'] for r in baseline]));value=chosen['Loss_C']
        a=float(np.mean([r['Loss_C'] for r in control])) if len(control)==2 else float('nan')
        result.append(dict(**context,selected_candidate=chosen['candidate'],c00_C=base,A_control_C=a,selected_C=value,
            tuning_change_pct=(value/base-1)*100 if base else float('nan'),
            C_base_vs_A_pct=(base/a-1)*100 if a else float('nan'),
            seasonal_week_C=float(np.mean([r['seasonal_week_C'] for r in picked])),
            improved_months_vs_c00=sum(p['Loss_C']<b['Loss_C'] for p,b in
                zip(sorted(picked,key=lambda x:x['fold']),sorted(baseline,key=lambda x:x['fold'])))))
    return result


def condition_scores(data,vi,predicted):
    """조건별 셀 오차/표본 수. 임의 셀 부분집합으로 block-C를 재정의하지 않는다."""
    origins=data.origins[vi];H=data.horizon
    target_end=origins.repeat(H)+pd.to_timedelta(np.tile(np.arange(1,H+1)*15,len(vi)),unit='min')
    start=target_end-pd.Timedelta(minutes=15)
    observed=data.observed_production.loc[origins].to_numpy(float)
    frame=pd.DataFrame(dict(origin=origins.repeat(H),day=start.normalize(),actual=data.y[vi].ravel(),
        predicted=predicted.ravel(),B=np.repeat(data.b[vi],H),hour=start.hour,weekday=start.dayofweek,
        production=np.repeat(np.where(observed>0,'관측생산있음','관측생산없음'),H)))
    out=[]
    for axis in ['hour','weekday','production']:
        for label,g in frame.groupby(axis):
            d=(g.predicted-g.actual).to_numpy();peak=(g.actual>g.B).to_numpy();alarm=(g.predicted>=g.B).to_numpy()
            out.append(dict(condition=axis,label=label,n_cells=len(g),n_origins=g.origin.nunique(),n_days=g.day.nunique(),
                RMSE=float(np.sqrt(np.mean(d**2))),MAE=float(np.mean(np.abs(d))),
                under_RMSE=float(np.sqrt(np.mean(np.minimum(d,0)**2))),over_RMSE=float(np.sqrt(np.mean(np.maximum(d,0)**2))),
                n_peak_cells=int(peak.sum()),peak_recall=float((peak&alarm).sum()/peak.sum()) if peak.any() else None))
    return out


def record(path,config,status,rows,error=None):
    metrics=[{k:r[k] for k in ['fold','n_train','n_valid','Loss_C','RMSE','L_peak','selected_round','seconds']} for r in rows]
    text=f'''# 실험 {config['experiment_id']}

- 상태: {status}
- 목적과 가설: {config['purpose']}
- 시작·종료 시각 및 시간대: {config['started']} / {now()}; Asia/Seoul
- 모델·라이브러리 버전: {config['library']} {version(config['library'])}
- 데이터 위치·버전 또는 해시·사용 범위: {bc.rf.SOURCE}; SHA256 {config['source_sha256']}; {config['phase']} 원점별 완전 경로.
- 목표값·단위·예측 구간: power_15min, kW, {config['profile']}.
- 학습·검증·테스트 분할과 누수 점검: 7월 이전/7월 및8월 이전/8월. 마지막 학습 목표<검증 시작; 검증 마지막 목표<다음 월. 9월은 동결 후 재평가만. 8월은 개발에 포함.
- 전처리·특성·하이퍼파라미터·seed: 기존 전처리 v1; {config['feature']}/{config['scenario']}; {json.dumps(config['params'],ensure_ascii=False)}; seed42.
- 가중치: {WEIGHT_RULE}; RF 동일 식·학습 달력지원 mask, 주 평가 모든 셀 cap2. 초기170kW/원점 월 최대 고정. 분모는전체셀. RF 확률 사용 없음.
- 이전 또는 기준 실험 링크와 변경 사항: 같은 실행의 c00 기준 및 A 제어와 비교. 기존 RF 연결 C 숫자와 직접 비교 불가.
- 실행 명령·환경·코드 버전 또는 코드 식별 정보: python -B -X utf8 notebooks/_shared/full_c_tuning.py; config.json의 code_sha256 및 environment.json.
- 실제 실행 시간·자원 사용량: 아래 fold별 seconds; CPU threads4/BLAS1, RSS 미측정.
- 평가 지표·구간별 또는 fold별 결과:

{table(metrics)}

- 기준모델 대비 변화와 비교 가능 여부: 같은 split·유효 원점·평가 cap2 내에서 비교. 다른 F/외생 조건은 시나리오별 비교.
- 오차가 큰 조건·해석·한계: 원점 경로가 중첩한다. 검증 선택·checkpoint 선택에 같은 두 달 사용. 170kW에 과거 전체 q90 정보 포함. future는사후실측대체이며 실제 예보가 아님.
- 오류·경고·실패 또는 중단 원인: {error or '없음; 개별 optimizer stop_reason은 trace에 기록.'}
- 예측·모델·표·그래프·로그의 저장 경로: {config['artifact_folder']}; keys/actual/predicted/B의 npz, 모델joblib, trace.json.gz.
- 결론과 다음 실험: {status}; 실제 후보 선정은 summary/selection.json에 기록.
'''
    Path(path).write_text(text,encoding='utf-8')


def save_data(data,ids,x,folder):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists():return
    np.savez_compressed(folder/'features.npz',X=x.to_numpy(np.float32),origins=data.origins[ids].asi8)
    np.savez_compressed(folder/'targets.npz',y=data.y[ids],origins=data.origins[ids].asi8)
    np.savez_compressed(folder/'weights.npz',B=data.b[ids],train=data.w_train[ids],evaluation=data.w_eval[ids],origins=data.origins[ids].asi8)
    dump(folder/'manifest.json',dict(profile=data.profile,feature=data.feature,scenario=data.scenario,
        feature_columns=list(x.columns),direct=data.direct,horizon=data.horizon,width=data.width,
        n_origins=len(ids),n_feature_rows=len(x),n_target_cells=data.y[ids].size,
        input_order='origin ascending, lead1..H ascending (pooled); direct uses one origin per row',
        target_timestamp_rule='origin + lead*15 minutes',source_sha256=digest(bc.rf.SOURCE),
        feature_sha256=digest(folder/'features.npz'),target_sha256=digest(folder/'targets.npz'),weight_sha256=digest(folder/'weights.npz'),weight_rule=WEIGHT_RULE))


def fit_fold(data,library,params,objective,fold,folder,deadline,final=False):
    ti,vi=data.split(fold);x=data.inputs(ti);xv=data.inputs(vi)
    before=perf_counter()
    model=CBoosting(library,params,objective)
    with threadpool_limits(limits=1,user_api='blas'):
        model.fit(x.to_numpy(),data.y[ti],data.w_train[ti],data.width,
            None if final else xv.to_numpy(),None if final else data.y[vi],None if final else data.w_eval[vi],
            direct=data.direct,deadline=deadline)
    if model.stop_reason in ['budget','external_stop']:
        folder.mkdir(parents=True,exist_ok=False)
        joblib.dump(dict(model=model,feature_columns=list(x.columns),status='중단'),folder/'interrupted_model.joblib',compress=3)
        with gzip.open(folder/'trace.json.gz','wt',encoding='utf-8') as stream:json.dump(model.trace,stream)
        raise TimeoutError(f'학습 중 중단({model.stop_reason}): 완료round {model.fitted_rounds}, 실제 {perf_counter()-before:.1f}초; 미완료 후보 선정 금지')
    predicted=model.predict(xv.to_numpy());scores,_,_=loss_gradient(data.y[vi],predicted,data.w_eval[vi],data.width)
    scores.update(clean_target_scores(data,vi,predicted))
    initial=np.broadcast_to(model.initial,data.y[vi].shape)
    initial_C=loss_gradient(data.y[vi],initial,data.w_eval[vi],data.width)[0]['Loss_C']
    seasonal=np.column_stack([data.df.power_15min.reindex(data.origins[vi]+pd.Timedelta(minutes=15*h)-pd.Timedelta(days=7)).to_numpy()
                              for h in range(1,data.horizon+1)])
    seasonal_C=loss_gradient(data.y[vi],seasonal,data.w_eval[vi],data.width)[0]['Loss_C']
    events=data.y[vi]>data.b[vi,None];alarm=predicted>=data.b[vi,None]
    scores.update(n_peak_cells=int(events.sum()),peak_recall=float((events&alarm).sum()/events.sum()) if events.any() else None,
        false_peak_cells=int((~events&alarm).sum()),initial_C=initial_C,seasonal_week_C=seasonal_C,
        path_peak_MAE=float(np.abs(data.y[vi].max(axis=1)-predicted.max(axis=1)).mean()),
        peak_time_MAE_minutes=float(np.abs(data.y[vi].argmax(axis=1)-predicted.argmax(axis=1)).mean()*15))
    folder.mkdir(parents=True,exist_ok=False)
    payload=dict(model=model,feature_columns=list(x.columns),profile=data.profile,feature=data.feature,scenario=data.scenario,weight_rule=WEIGHT_RULE)
    joblib.dump(payload,folder/'model.joblib',compress=3)
    loaded=joblib.load(folder/'model.joblib')['model']
    small=xv.iloc[:min(3,len(vi))] if data.direct else xv.iloc[:min(3,len(vi))*data.horizon]
    np.testing.assert_allclose(loaded.predict(small.to_numpy()),predicted[:min(3,len(vi))],atol=1e-6,rtol=0)
    np.savez_compressed(folder/'predictions.npz',origins=data.origins[vi].asi8,actual=data.y[vi],predicted=predicted,B=data.b[vi],evaluation_weights=data.w_eval[vi],keep_clean_targets=~data.has_repaired_target[vi])
    pd.DataFrame(condition_scores(data,vi,predicted)).to_csv(folder/'condition_metrics.csv',index=False)
    with gzip.open(folder/'trace.json.gz','wt',encoding='utf-8') as stream:json.dump(model.trace,stream,ensure_ascii=False)
    row=dict(fold=fold,n_train=len(ti),n_valid=len(vi),selected_round=model.selected_round,
        fitted_rounds=model.fitted_rounds,stop_reason=model.stop_reason,seconds=perf_counter()-before,
        train_last_target=str(data.origins[ti[-1]]+pd.Timedelta(minutes=15*data.horizon)),
        valid_first_origin=str(data.origins[vi[0]]),valid_last_target=str(data.origins[vi[-1]]+pd.Timedelta(minutes=15*data.horizon)),
        **scores)
    dump(folder/'metrics.json',row);return row


class Suite:
    def __init__(self,budget,rounds,plan=None):
        if budget<0 or rounds<1:raise ValueError('budget>=0, rounds>=1 필요; budget=0은 명시적 무제한')
        self.plan=build_plan() if plan is None else plan
        self.run_id=datetime.now(KST).strftime('%Y%m%dT%H%M%S_KST')+'_'+uuid4().hex[:6]+'_full_c'
        self.start=perf_counter();self.deadline=self.start+budget if budget else float('inf');self.rounds=rounds
        self.root=ROOT/'artifacts/boosting/runs'/self.run_id;self.root.mkdir(parents=True,exist_ok=False)
        self.data_root=ROOT/'data/processed'/f'full_c_inputs_{self.run_id}'
        self.rows=[];self.records=[];self.selections={};self.config=dict(run_id=self.run_id,status='실행 중',started=now(),
            budget_seconds=budget,weight_rule=WEIGHT_RULE,cap=2,validation_months=[7,8],development_fold_count=2,
            step_policy='unshrunk_line_search_then_shrinkage_v1',
            tuning_policy=TUNING_POLICY,selection_key_fields=['library','profile','feature','scenario'],
            execution_plan=self.plan,budget_policy='stop_and_record_without_transferring_or_skipping_contexts',
            final_round_rule='floor(median(July,August best checkpoint rounds))',
            profiles={k:PROFILES[k] for k in self.plan['profiles']},
            feature_tuning='F1 -> F2 -> F3; independent by model/profile/feature/scenario',
            September_used_for_selection=False,source_path=str(bc.rf.SOURCE.resolve()),source_sha256=digest(bc.rf.SOURCE),rounds=rounds)
        self.config.update(repair_source_path=str(REPAIR_SOURCE.resolve()),repair_source_sha256=digest(REPAIR_SOURCE),
            clean_target_policy='secondary only: exclude whole paths containing repaired targets; primary selection remains full C',
            external_stop_file=os.environ.get('KAMP_BOOSTING_STOP_FILE'))
        self.environment=dict(python=platform.python_version(),packages={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost','lightgbm','joblib','tabulate']})
        dump(self.root/'environment.json',self.environment)
        code=self.root/'code';code.mkdir()
        paths=[ROOT/'notebooks/_shared'/f for f in ['full_c_loss.py','full_c_boosting.py','full_c_data.py','full_c_tuning.py','boosting_common.py','performance_profiles.py','forecast_common.py']]
        paths+=[ROOT/'notebooks/models/random_forest'/f for f in ['rf_current_data.py','rf_direct_c.py','rf_direct_c_experiments.py']]
        paths+=[ROOT/'tests/test_full_c_boosting.py']
        paths+=[ROOT/'tests/test_full_c_feature_tuning.py']
        paths+=[ROOT/'tests/test_full_c_delivery.py']
        paths+=[ROOT/'artifacts/boosting'/f for f in ['infer_full_c.py','verify_full_c_delivery.py',
            'verify_full_c_inference.py','summarize_full_c_delivery.py','plot_full_c_delivery.py']]
        paths+=[ROOT/'docs/research'/f for f in ['2026-10-03-load-forecasting-tuning-evidence.md',
            '2026-10-04-boosting-full-c-objective-evidence.md','2026-10-04-peak-loss-weighting-literature-review.md']]
        snapshot_root=Path(__file__).resolve().parents[2]
        paths=[snapshot_root/p.relative_to(ROOT) for p in paths]
        self.config['code_sha256']={str(p.relative_to(snapshot_root)):digest(p) for p in paths}
        for path in paths:
            destination=code/path.relative_to(snapshot_root);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        dump(code/'environment.json',self.environment);self.flush()

    def flush(self):
        dump(self.root/'config.json',self.config)
        if self.rows:pd.DataFrame(self.rows).to_csv(self.root/'all_metrics.csv',index=False)
        dump(self.root/'records.json',self.records)
        if self.selections:dump(self.root/'selection_in_progress.json',self.selections)

    def experiment(self,data,library,candidate,params,objective='C',phase='cv',folds=None):
        experiment_id=f'{data.profile}_{data.scenario}_{data.feature}_{candidate}_{phase}'
        directory=ROOT/'experiments'/library/self.run_id/'records';directory.mkdir(parents=True,exist_ok=True)
        md=directory/f'{experiment_id}.md';folder=self.root/library/experiment_id
        if md.exists():raise ValueError('동일 실험 덮어쓰기 금지')
        config=dict(experiment_id=experiment_id,candidate=candidate,library=library,profile=data.profile,feature=data.feature,scenario=data.scenario,
            params=params,objective=objective,phase=phase,started=now(),purpose='각 F·외생 조건의 전체 C 독립 튜닝 및 설정 동결 후 재평가',
            source_sha256=self.config['source_sha256'],artifact_folder=str(folder))
        entries=[];record(md,config,'실행 중',entries);dump(directory/f'{experiment_id}.json',config)
        self.records.append(dict(library=library,experiment_id=experiment_id,md=str(md),config=config,status='실행 중'))
        item=self.records[-1];self.flush()
        try:
            check_input_versions(self.config)
            for fold in (folds or list(FOLDS)):
                if perf_counter()>=self.deadline:raise TimeoutError('전체 예산 소진')
                ti,vi=data.split(fold);ids=np.r_[ti,vi]
                # 데이터는 후보별 중복하지 않고 profile/F/시나리오/분할별 공통 저장.
                data_folder=self.data_root/data.profile/data.scenario/data.feature/fold
                if not (data_folder/'manifest.json').exists():save_data(data,ids,data.inputs(ids),data_folder)
                entry=fit_fold(data,library,params,objective,fold,folder/fold,self.deadline,final=phase=='final')
                check_input_versions(self.config)
                entry.update(library=library,profile=data.profile,feature=data.feature,scenario=data.scenario,candidate=candidate,phase=phase,
                    experiment_id=experiment_id,objective=objective,artifact_folder=str(folder/fold))
                entries.append(entry);self.rows.append(entry)
                record(md,config,'실행 중',entries);self.flush()
                print(fit_status(library,experiment_id,fold,entry),flush=True)
            item['status']='완료';record(md,config,'완료',entries);self.flush();return entries
        except Exception as error:
            item['status']='중단' if isinstance(error,TimeoutError) else '실패'
            record(md,config,item['status'],entries,traceback.format_exc());self.flush();raise

    def run(self):
        completed_variants=[]
        try:
            check_input_versions(self.config)
            dev=load_frame()
            check_input_versions(self.config)
            for context in self.plan['contexts']:
                library=context['library'];profile=context['profile']
                feature=context['feature'];scenario=context['scenario']
                data=Dataset(dev,profile,feature,scenario)
                candidate_rows=[]
                for ix,parameters in enumerate(CANDIDATES[library]):
                    params=dict(parameters,rounds=self.rounds)
                    candidate_rows+=self.experiment(data,library,f'c{ix:02}',params)
                chosen=select_candidate(candidate_rows);key=selection_key(**context)
                params=dict(CANDIDATES[library][int(chosen['candidate'][1:])],rounds=self.rounds)
                checkpoints=[r['selected_round'] for r in candidate_rows if r['candidate']==chosen['candidate']]
                self.selections[key]=dict(**chosen,**context,params=params,final_rounds=int(np.median(checkpoints)))
                completed_variants.append((library,profile,feature,scenario,chosen['candidate']))
                self.flush();print(f'SELECT {key} {chosen}',flush=True)
                if self.plan['a_control']:
                    self.experiment(data,library,'a_control',dict(BASE,rounds=self.rounds),objective='A')
            # 모든 요청 조건의 독립 튜닝이 끝나야 설정 동결 및 9월 평가를 진행한다.
            if len(self.selections)!=self.plan['selection_contexts']:
                raise ValueError('요청 조건 전체의 독립 튜닝 미완료: 9월 평가 금지')
            dump(self.root/'selection.json',self.selections)
            self.config['selection_sha256']=digest(self.root/'selection.json')
            self.config['selection_frozen_at']=now();self.config['development_frozen_at']=now()
            self.config['development_complete_variants']=completed_variants;self.flush()
            full=load_frame('2021-09-15 00:01')
            check_input_versions(self.config)
            for library,profile,feature,scenario,candidate in completed_variants:
                chosen=self.selections[selection_key(library,profile,feature,scenario)]
                params=dict(chosen['params'],rounds=chosen['final_rounds'])
                self.experiment(Dataset(full,profile,feature,scenario),library,candidate,params,phase='final',folds=['september'])
            self.config.update(status='완료',ended=now(),elapsed_seconds=perf_counter()-self.start);self.flush()
        except Exception:
            self.config.update(status='중단/실패',ended=now(),error=traceback.format_exc(),elapsed_seconds=perf_counter()-self.start);self.flush()
            raise
        finally:self.package()
        return self.root

    def package(self):
        replay=' '.join(reproduction_arguments(self.plan,self.config['budget_seconds'],self.rounds))
        for library in LIBRARIES:
            records=[r for r in self.records if r['library']==library]
            if not records:continue
            target=ROOT/'final'/library/self.run_id;target.mkdir(parents=True,exist_ok=True)
            for sub in ['records','model','results']:(target/sub).mkdir(exist_ok=True)
            shutil.copytree(self.root/'code',target/'code',dirs_exist_ok=True)
            shutil.copy2(self.root/'config.json',target/'code/config.json')
            if (self.root/'selection.json').exists():shutil.copy2(self.root/'selection.json',target/'results/selection.json')
            for entry in records:shutil.copy2(entry['md'],target/'records'/Path(entry['md']).name)
            rows=[r for r in self.rows if r['library']==library]
            summary=[]
            for entry in records:
                group=[r for r in rows if r['experiment_id']==entry['experiment_id']]
                selected,final_saved=selection_flags(entry['config'],entry['status'],self.selections)
                summary.append(dict(experiment_id=entry['experiment_id'],status=entry['status'],
                    phase=entry['config']['phase'],change=entry['config']['params'],split_id='7_8_months' if entry['config']['phase']=='cv' else 'september_reevaluation',
                    Loss_C=float(np.mean([r['Loss_C'] for r in group])) if group else None,
                    RMSE=float(np.mean([r['RMSE'] for r in group])) if group else None,
                    seconds=sum(r['seconds'] for r in group),record_path=f"{entry['experiment_id']}.md",
                    selected=selected,final_model_saved=final_saved))
            pd.DataFrame(summary).to_csv(target/'records/summary.csv',index=False)
            pd.DataFrame(rows).to_csv(target/'results/fold_metrics.csv',index=False)
            conditions=[]
            for row in rows:
                condition_file=Path(row['artifact_folder'])/'condition_metrics.csv'
                if condition_file.exists():
                    cond=pd.read_csv(condition_file)
                    for key in ['profile','feature','scenario','candidate','phase','fold']:cond[key]=row[key]
                    conditions.append(cond)
            if conditions:pd.concat(conditions,ignore_index=True).to_csv(target/'results/condition_metrics.csv',index=False)
            comparisons=comparison_rows(rows,{k:v for k,v in self.selections.items() if v['library']==library})
            pd.DataFrame(comparisons).to_csv(target/'results/candidate_comparison.csv',index=False)
            finals=[r for r in rows if r['phase']=='final']
            for row in finals:
                name=f"{row['profile']}_{row['scenario']}_{row['feature']}"
                dest=target/'model'/name;dest.mkdir(exist_ok=True)
                source=Path(row['artifact_folder'])
                shutil.copy2(source/'model.joblib',dest/'model.joblib')
                shutil.copy2(source/'metrics.json',dest/'metrics.json')
                shutil.copy2(source/'predictions.npz',target/'results'/f'{name}_september_predictions.npz')
            report=f'''# {library} 전체 C 직접 학습·제한 튜닝 보고서

- 상태: {self.config['status']}; 실행 {self.run_id}; {self.config.get('elapsed_seconds',0):.1f}초.
- 목적: 피크 과소예측·전체 구간·최악20%를 공동 최적화하고 모델별 제한 후보를 선택.
- 학습/선정: 각 검증 월 이전 expanding 자료,7·8월 월 동일평균C. 학습 마지막 목표<검증 시작, 완전 경로만 사용. 8월은 개발 검증에 포함.
- 데이터: {bc.rf.SOURCE}; SHA256 {self.config['source_sha256']}; v1,15분kW. 원본변경 없음. 데이터별 피처/키/정답/가중치는 {self.data_root}의 manifest.json.
- 조건: F1→F2→F3 순서로 모델·예측구간·외생조건마다 3개 C 후보를 독립 탐색. F2 설정 전이 없음. 기준 A 목적 대조 포함 여부: {self.plan["a_control"]}.
- 주가중치: RF 동일 continuous2 및학습지원마스크; 주평가 모든셀. C계수 .5/.3/.2, L_peak 분모 전체셀. 실제값 기반 가중치를 모델 입력으로 전달하지 않음.
- 원점 cadence: 단기15분/중기1h/장기1d. H=4/16/32/48/288, 위험창4/4/4/4/96. 이력지원7/14/28일, F3 장기에14일통계/시차.
- 미래 외생 실측 대체는 zero-error 예보 시나리오. 관측 조건과 별도 해석. 미래 전력은 모든 입력에서 제외. day_type은 사전 달력 이용 가능을 가정.
- 연구근거: [기존튜닝연구검토](../code/docs/research/2026-10-03-load-forecasting-tuning-evidence.md). depth3/4/6, leaf/child,L2를 비용에맞춰제한한 프로젝트후보이며 논문최적값이 아님. 외부학습률.1/최대{self.rounds}round, native내부1tree학습률1.
- 최종round: 각 모델·예측구간·F·외생조건의 선정 후보별 7·8월 최적checkpoint 중앙값을 내림해 독립 동결. 다른 F의 round를 사용하지 않음.
- 확률·경보: 이번 실험은 점예측/C 최적화. peak_recall은 predicted>=B의 점경보이며 확률예측/확률보정모델을 학습했다는 뜻이 아님.
- 한계: 월2개뿐이고 overlapping경로/동일월checkpoint및HPO 선택의 개발편향. 전체q90에서정한170kW에9월정보포함,9월은 이미본탐색적재평가이며 blindtest아님. RF의별도24h/4d·피처설정과 자동직접비교불가. CPU4,BLAS1,메모리최대미측정. 전역최적화보장없음.

## 실제 동결 후보의9월 결과

{table([{k:r[k] for k in ['profile','feature','scenario','Loss_C','RMSE','L_peak','peak_recall','n_valid','seasonal_week_C']} for r in finals])}

RMSE는 평가한 모든 원점·예측 리드 셀의 `sqrt(mean((실제−예측)^2))`이며 가중치 없는 kW 오차다. 경로가 중첩되면 같은 시각도 원점·리드별 예측으로 각각 평가한다. 선정 출력의 RMSE_month_mean은 7·8월 RMSE의 동일 비중 평균이며, 두 달을 합친 pooled RMSE와 구분한다. 후보 선정에는 평균 C만 사용한다.

RF와 같은 보정 정답 민감도: fold_metrics의 clean_targets_*는 전주 대체된 전력 정답을 하나라도 포함하는 경로를 제외한 보조 평가다. n_clean_target_origins/n_repaired_target_origins를 함께 기록한다. 주 선정 full C는 그대로 유지하며 보정 입력까지 제외한 clean_targets_inputs는 이번 구현 범위에 없다. 원천·보정 대조 파일의 실행 중 해시 변경은 차단한다.

## 기준·튜닝·조건별 전체 검증

{table(comparisons)}

주 지표는 작을수록 좋다. 선정 후보는 동일한7·8월 평균 C가 가장 낮은 완료 후보다. 다른 후보는 이 기준으로 기각했으며 최적 파라미터를 전역 탐색한 결과는 아니다. tuning_change_pct는 전체 C 기본 설정(c00) 대비 파라미터 탐색의 변화다. C_base_vs_A_pct는 같은 기본설정에서 목적만 바꾼 대조이며 선정 C와 A의 차이는 목적과파라미터의 복합 효과다.

[전체 fold 결과](fold_metrics.csv) · [후보별 기록과 선정](../records/summary.csv) · [동일 기준 비교](candidate_comparison.csv) · [시간대·요일·원점 관측 생산 유무별 셀 오차](condition_metrics.csv). 조건별 표에 셀/원점/일수와 피크 셀 수를 함께 기록했다. 적은 사건이나 중첩 셀의 조건별 비율은 불확실성이 크다. 미실행 조건은 결과표에 넣지 않는다.

## 재현과산출물

- [코드설정](../code/config.json),[환경](../code/environment.json),[전체후보기록](../records/summary.csv).
- 보관 사본을 새 결과 폴더로 재실행: `python -B -X utf8 <묶음>/code/notebooks/_shared/full_c_tuning.py --project-root <새결과폴더> --source "{bc.rf.SOURCE}" {replay}`. 활성 소스를 대신 실행할 필요가 없다. 원본 final 묶음은 수정하지 않는다.
- model/<profile>_<scenario>_<F>/model.joblib에는적합ensemble/컬럼순서/스키마. results의npz는원점(actual,predicted,B,w)과전체H를보존하고 target=origin+15min*lead로복원.
- 중간CV모델/trace/예측은 {self.root/library}, 공통대용량자료중복하지않음.
'''
            (target/'results/final_report.md').write_text(report,encoding='utf-8')
            (target/'README.md').write_text(f'# {library} / {self.run_id}\n\n상태: {self.config["status"]}.\n\n[보고서](results/final_report.md) · [후보summary](records/summary.csv) · [실행설정](code/config.json).\n\n선정은7·8월 평균C만사용, F별 독립 선정, 전체 개발 조건 동결 후9월탐색평가. 정확한학습/평가모델은model및fold_metrics참조. 소스SHA {self.config["source_sha256"]}. 공통입력 {self.data_root}.\n',encoding='utf-8')
        index=ROOT/'final/README.md';index.parent.mkdir(exist_ok=True)
        if not index.exists():index.write_text('# 최종 모델 실험 묶음\n\n선정에사용한검증과최종평가상태는각보고서에명시한다.\n',encoding='utf-8')
        with index.open('a',encoding='utf-8') as stream:
            for library in LIBRARIES:
                if (ROOT/'final'/library/self.run_id/'README.md').exists():stream.write(f'\n- [{library} {self.run_id}]({library}/{self.run_id}/README.md): {self.config["status"]};7·8월선정/9월탐색재평가.\n')
        expindex=ROOT/'experiments/README.md'
        with expindex.open('a',encoding='utf-8') as stream:stream.write(f'\n\n## {self.run_id}\n\n전체C직접부스팅/F별·외생조건별 독립 튜닝/고정RFcontinuous2. 상태{self.config["status"]}; [최종인덱스](../final/README.md). 공통입력 `{self.data_root}`.\n')


def main(argv=None):
    global ROOT
    parser=argparse.ArgumentParser(description='F1→F2→F3 독립 전체 C 튜닝; 기존 final 묶음은 보존')
    parser.add_argument('--budget-seconds',type=int,default=14400,help='전체 실행 시간 제한(초); 0은 명시적 무제한')
    parser.add_argument('--rounds',type=int,default=60)
    parser.add_argument('--models',nargs='+',choices=LIBRARIES,default=LIBRARIES)
    parser.add_argument('--profiles',nargs='+',choices=list(PROFILES),default=list(ACTIVE_PROFILES))
    parser.add_argument('--features',nargs='+',choices=FEATURES,default=FEATURES)
    parser.add_argument('--scenarios',nargs='+',choices=SCENARIOS,default=SCENARIOS)
    parser.add_argument('--skip-a-control',action='store_true',help='A 목적 대조군만 제외; 모든 F의 C 튜닝은 유지')
    parser.add_argument('--dry-run',action='store_true',help='실행 계획 출력만; 원자료 로딩·실험폴더 생성·학습 없음')
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--source',type=Path,default=bc.rf.SOURCE)
    args=parser.parse_args(argv)
    if args.budget_seconds<0 or args.rounds<1:parser.error('budget>=0, rounds>=1 필요')
    plan=build_plan(args.models,args.profiles,args.features,args.scenarios,not args.skip_a_control)
    plan.update(max_rounds=args.rounds,budget_seconds=args.budget_seconds,
        candidates={lib:[dict(p,rounds=args.rounds) for p in CANDIDATES[lib]] for lib in plan['models']})
    if args.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return plan
    ROOT=args.project_root.resolve();bc.rf.SOURCE=args.source.resolve()
    suite=Suite(args.budget_seconds,args.rounds,plan)
    print(f'RUN {suite.root}',flush=True);print(suite.run(),flush=True)
    return suite.root


if __name__=='__main__':main()
