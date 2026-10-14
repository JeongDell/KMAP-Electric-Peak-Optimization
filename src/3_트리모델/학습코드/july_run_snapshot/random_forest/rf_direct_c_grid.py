"""순수 연속형2·7/8월 시간순 C 그리드. 통합 노트북과 CLI에서 같은 함수를 쓴다."""
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import platform
import sys
import types
import traceback
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits
import rf_direct_c as direct
import rf_direct_c_experiments as runner
import rf_feature_sequence as feature_sequence

ROOT=runner.ROOT
RF_DIR=Path(__file__).resolve().parent
OLD_RUN=ROOT/'artifacts/random_forest/runs/20261004T155233_KST_e8d1e6_rf_direct_c'
PAIRED=['rf_mse','c_leaf_d10']
FAMILIES=runner.FAMILIES

def build_profile_for_config(df,repair,profile,config):
    return feature_sequence.profile_for_config(df,repair,profile,config)

def restore_embedded_modules(sources):
    # loky 초기화 때 노트북과 같은 소스를 등록한다. 이름 기반 직렬화도 이 버전을 찾는다.
    for name,source in sources.items():
        module=types.ModuleType(name);module.__file__=str(RF_DIR/f'{name}.py')
        module.__embedded_source__=source;sys.modules[name]=module
        exec(compile(source,module.__file__,'exec'),module.__dict__)

def dispatch_job(sources,*args):
    # joblib1.4의 worker 초기화 API에 의존하지 않고 각 작업 시작 때 내장 코드를 등록한다.
    # 병렬 프로세스는 작업을 순차 실행하므로 동일 프로세스의 모듈 교체가 겹치지 않는다.
    if sources:
        restore_embedded_modules(sources)
        return sys.modules['rf_direct_c_grid'].fit_job(*args)
    return fit_job(*args)

def core_grid():
    # 먼저 트리 수를 고정하고 피처 비율·최소 잎 표본·최대 잎 수의18조합을 비교한다.
    common=dict(n_estimators=64,max_depth=None,max_samples=None,bootstrap=True,
                criterion='squared_error',random_state=42,n_jobs=4)
    return [{**common,**p} for p in ParameterGrid(dict(min_samples_leaf=[5,10,20],
            max_features=[1/3,.5,1.0],max_leaf_nodes=[32,64]))]

def refinement_grid(anchor):
    return [{**anchor,**p} for p in ParameterGrid(dict(n_estimators=[64,128,256],max_samples=[.7,None]))]

def parameter_id(params):
    return 'p'+hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:10]

def unique_parameters(params):
    return list({parameter_id(p):p for p in params}.values())

def evaluation_weights(y,b):
    return 1+np.clip((np.asarray(y,float)/np.asarray(b,float)[:,None]-.8)/.2,0,1)

def pure_training_weights(y,b,supported):
    # 초기 RF에는 평균w, C 공동 학습에는 시점별w를 쓴다. RF 확률은 참조하지 않는다.
    supported=np.asarray(supported,bool)
    weights=evaluation_weights(y,b);weights[~supported]=1
    return weights,supported

def select_models(metrics):
    rows=metrics.loc[metrics.phase.eq('cv')&metrics.scope.eq('full')&metrics.fold.isin(['m07','m08'])]
    if rows.empty:raise ValueError('7·8월 완료 지표 필요')
    if rows.duplicated(['profile','candidate','fold']).any():raise ValueError('중복 CV 키')
    selected={}
    for profile,part in rows.groupby('profile',sort=False):
        options=[]
        for candidate,group in part.groupby('candidate'):
            if set(group.fold)!=set(['m07','m08']) or not np.isfinite(group.Loss_C).all():
                raise ValueError(f'{profile}/{candidate}:두 fold의 유한 C 필요')
            family=group.family.iloc[0]
            options.append(dict(candidate=candidate,param_id=group.param_id.iloc[0] if 'param_id' in group else candidate,
                family=family,cv_C=float(group.Loss_C.mean()),priority=FAMILIES.index(family)))
        selected[profile]=min(options,key=lambda x:(x['cv_C'],x['priority'],x['candidate']))
    return selected

def _timestamp():return datetime.now(runner.previous.common.KST).isoformat()

def _record_path(suite,phase,profile,candidate):
    return ROOT/'experiments/random_forest'/f'{suite.name}_{phase}_{profile}_{candidate}.md'

def write_candidate_record(suite,config,phase,profile,candidate,params,rows,status,error=None):
    path=_record_path(suite,phase,profile,candidate);path.parent.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    table=runner.previous.common.md_table(frame[['fold','scope','n_origins','Loss_C','L_mean','L_CVaR','L_peak','RMSE']]) if len(frame) else '실제 평가 결과 없음.'
    text=f'''# 실험 {path.stem}

- 상태: {status}
- 목적과 가설: 순수 연속형2에서 RF 구조/트리 수 튜닝 및 전체C 직접 학습 비교.
- 시작·종료 시각 및 시간대: {config['started']} / {_timestamp()}, Asia/Seoul. 후보별 실제 fit 시각은 fold_config.json.
- 모델·라이브러리 버전: Python {config['python']}; {config['versions']}.
- 데이터 위치·버전 또는 해시·사용 범위: {runner.previous.common.SOURCE}; {config['source_sha256']}.
- 목표값·단위·예측 구간: power_15min(kW), {profile},15분 출력; 입력창 {config['windows'][profile]}h.
- 학습·검증·테스트 분할과 누수 점검: 7월=1~6월,8월=1~7월;9월=동결후1~8월. 완전경로 시간분할. phase={phase}.
- 전처리·특성·하이퍼파라미터·seed: native 피처{config.get('feature_sets',{}).get(profile,'이전 F2')},공통지원{config.get('common_feature_support','이전지원')}; 스케일링 없음,미래생산/기상실측=오차0예보. params={params};C=.5mean+.3topceil20%+.2peak;w=1+clip((y/b-.8)/.2,0,1),RF확률인자제외. 1월/월경계 학습w=1.
- 이전 또는 기준 실험 링크와 변경 사항: [공통가중치 설계](../../docs/superpowers/specs/2026-10-04-rf-direct-c-design.md). 기존64trees/64leaves/leaf20/features1/3을포함.
- 실행 명령·환경·코드 버전 또는 코드 식별 정보: {config['command']};code hashes={config['code_hashes']}.
- 실제 실행 시간·자원 사용량(측정한 경우): fold_config.json의family 단계시간·RF 적합시간·공유 RF 시간·재사용여부를구분한다.후보2프로세스×RF4스레드·BLAS1. peak RAM 미측정.
- 평가 지표·구간별 또는 fold별 결과: 아래 실제결과. 선정은7·8월full C의월동일평균.
- 기준모델 대비 변화와 비교 가능 여부: 같은params의rf_mse와paired 비교;이전고정설정 대비튜닝효과는별도보고.
- 오차가 큰 조건·해석·한계: 보정제외scope 병기;중첩경로;전체q90에9월포함;고정solver시간에따른비수렴한계.
- 오류·경고·실패 또는 중단 원인: {error or '치명오류없음;solver는optimizer.json 참조'}.
- 예측·모델·표·그래프·로그의 저장 경로: {suite/phase/profile/candidate}.
- 결론과 다음 실험: 9월로재튜닝하지않으며실제완료범위와모든후보를보고한다.

{table}
'''
    path.write_text(text,encoding='utf-8')
    return path

def fit_job(suite,config,data,fold,params,families,deadline):
    """하나의 파라미터·fold. 독립 폴더에 저장하고 부모에 지표/오류를 반환한다."""
    suite=Path(suite);profile=data['profile'];pid=parameter_id(params)
    phase='final' if fold=='september' else 'cv'
    start=pd.Timestamp('2021-09-01') if phase=='final' else pd.Timestamp(2021,int(fold[-2:]),1)
    end=pd.Timestamp('2021-09-15') if phase=='final' else start+pd.offsets.MonthBegin(1)
    base_dir=suite/'fit_cache'/phase/profile/pid/fold;base_dir.mkdir(parents=True,exist_ok=True)
    rows=[];fits=0;reused=0;began=perf_counter();failed_family=None
    job_identity=dict(profile=profile,param_id=pid,fold=fold,phase=phase,families=families)
    try:
        ti,vi=runner.split_indices(data['origins'],data['spec']['horizon_steps'],start,end)
        x=data['features'].iloc[ti].to_numpy(np.float32);y=data['targets'].iloc[ti].to_numpy(float)
        xv=data['features'].iloc[vi].to_numpy(np.float32);actual=data['targets'].iloc[vi].to_numpy(float)
        b=data['threshold'].iloc[ti].to_numpy();bv=data['threshold'].iloc[vi].to_numpy()
        supported=np.isfinite(runner.support_reference(data)[ti]).all(axis=1)
        w,known=pure_training_weights(y,b,supported);eval_w=evaluation_weights(actual,bv)
        identity=dict(params=params,train_origins_sha=hashlib.sha256(data['origins'][ti].asi8.tobytes()).hexdigest(),
            valid_origins_sha=hashlib.sha256(data['origins'][vi].asi8.tobytes()).hexdigest(),
            source_sha256=config['source_sha256'],weight_mode='continuous',cap=2,
            feature_set=data.get('feature_set','F2'),features=list(data['features']))
        forest_path=base_dir/'base_rf.joblib'
        with threadpool_limits(limits=1):
            if forest_path.is_file():
                if runner.previous.read(base_dir/'identity.json')!=identity:raise ValueError('재사용설정불일치')
                forest=joblib.load(forest_path);reused=1
                base_meta=runner.previous.read(base_dir/'base_fit.json')
            else:
                if perf_counter()>=deadline:raise TimeoutError('실험시간예산소진')
                base_start=_timestamp();base_clock=perf_counter()
                forest=RandomForestRegressor(**params).fit(x,y,sample_weight=w.mean(axis=1));fits=1
                base_meta=dict(fit_started=base_start,fit_ended=_timestamp(),fit_seconds=perf_counter()-base_clock)
                runner.previous.dump(base_dir/'base_fit.json',base_meta)
                joblib.dump(forest,forest_path,compress=3);runner.previous.dump(base_dir/'identity.json',identity)
                np.savez_compressed(base_dir/'training_inputs.npz',x=x,y=y,w=w,b=b,supported=known,
                    origins=data['origins'][ti].asi8,x_valid=xv,origins_valid=data['origins'][vi].asi8)
            base=direct.ForestCModel.from_forest(forest);models={};base_out=None
            for family in families:
                failed_family=family
                candidate=pid+'_'+family;folder=suite/phase/profile/candidate/fold
                if (folder/'fold_config.json').is_file():
                    rows.extend(pd.read_csv(folder/'metrics.csv').to_dict('records'))
                    models[family]=joblib.load(folder/'model_bundle.joblib')['model'];continue
                if perf_counter()>=deadline:raise TimeoutError('실험시간예산소진')
                folder.mkdir(parents=True,exist_ok=True);fit_started=_timestamp();t0=perf_counter();curves=[]
                if family=='rf_mse':model=base;optimizer=dict(solver_success=None)
                elif family.startswith('c_leaf'):
                    model=deepcopy(base);delta=int(family.rsplit('d',1)[1])
                    optimizer=direct.optimize_leaves(model,x,y,w,data['spec']['loss_steps'],delta=delta,
                        max_iter=120,deadline=min(deadline,perf_counter()+16))
                    curves=optimizer['curve']
                else:
                    model=deepcopy(models[family.replace('c_split','c_leaf')])
                    optimizer=direct.refine_splits(model,x,y,w,data['spec']['loss_steps'],min_leaf=params['min_samples_leaf'],
                        max_splits=4,proposals=12,max_iter=60,seed=42,deadline=min(deadline,perf_counter()+20))
                    for item in optimizer['joint_optimizations']:curves.extend(item['curve'])
                models[family]=model;prediction=model.predict(xv)
                train,_=direct.c_value_gradient(y,model.predict(x),w,data['spec']['loss_steps'])
                if not curves:curves=[dict(iteration=0,seconds=perf_counter()-t0,**train)]
                pd.DataFrame(curves).to_csv(folder/'training_curve.csv',index=False)
                runner.previous.dump(folder/'optimizer.json',optimizer)
                out=None
                if phase=='final':
                    if not family.startswith('c_split'):
                        if base_out is None:base_out=runner.previous.risk.LeafPathDistribution(base,x,y).predict(xv,bv,quantiles=True,initial=170.)
                        out=base_out
                    else:out=runner.previous.risk.LeafPathDistribution(model,x,y).predict(xv,bv,quantiles=True,initial=170.)
                masks={'full':np.ones(len(vi),bool),'clean_targets':~data['has_repaired_target'][vi],
                       'clean_targets_inputs':~(data['has_repaired_target'][vi]|data['has_repaired_input'][vi])}
                local=[]
                for scope,keep in masks.items():
                    row=dict(profile=profile,candidate=candidate,param_id=pid,family=family,phase=phase,fold=fold,
                        scope=scope,n_origins=int(keep.sum()),n_excluded_origins=int((~keep).sum()),n_predictions=int(keep.sum()*y.shape[1]))
                    if keep.any():
                        row.update(runner.previous.risk.point_metrics(actual[keep],prediction[keep],eval_w[keep],data['spec']['loss_steps']))
                        event=actual[keep]>bv[keep,None];signal=prediction[keep]>=bv[keep,None]
                        row.update(n_actual_power_peak=int(event.sum()),n_power_missed_peak=int((event&~signal).sum()),
                            n_power_false_peak=int((~event&signal).sum()),power_peak_recall=float((event&signal).sum()/event.sum()) if event.any() else np.nan,
                            predicted_peak_rate=float(signal.mean()))
                        if out is not None:
                            row.update(point_Brier_raw=float(np.mean((out['p_point'][keep]-event)**2)),
                                period_Brier_raw=float(np.mean((out['p_period'][keep]-event.any(axis=1))**2)))
                    local.append(row)
                h=y.shape[1];stamps=data['origins'][vi].to_numpy()[:,None]+np.arange(1,h+1)[None,:]*np.timedelta64(15,'m')
                if out is not None:
                    runner.previous.export_predictions(folder,data,vi,actual,prediction,out,runner.previous.risk.MonotoneCalibrator(),eval_w)
                else:
                    pd.DataFrame(dict(origin=np.repeat(data['origins'][vi],h),target_end=stamps.ravel(),actual=actual.ravel(),
                        predicted=prediction.ravel(),common_peak_weight=eval_w.ravel(),peak_threshold=np.repeat(bv,h),
                        keep_clean_targets=np.repeat(masks['clean_targets'],h),keep_clean_targets_inputs=np.repeat(masks['clean_targets_inputs'],h))).to_csv(folder/'predictions.csv.gz',index=False,compression='gzip')
                joblib.dump(dict(model=model,features=list(data['features']),profile=profile,spec=data['spec'],
                    feature_set=data.get('feature_set','F2'),common_feature_support=data.get('common_feature_support'),
                    params=params,context_window_hours=data['window'],training_targets=y,training_features=x,
                    initial_peak_kw=170.,threshold_mode='dynamic',raw_probability_only=True),folder/'model_bundle.joblib',compress=3)
                pd.DataFrame(local).to_csv(folder/'metrics.csv',index=False)
                runner.previous.dump(folder/'fold_config.json',dict(fit_started=fit_started,fit_ended=_timestamp(),phase=phase,fold=fold,
                    profile=profile,candidate=candidate,params=params,n_train=len(ti),n_valid=len(vi),train_C=train['Loss_C'],
                    train_label_max=str(data['origins'][ti[-1]]+pd.Timedelta(minutes=15*h)),valid_origin_min=str(data['origins'][vi[0]]),
                    elapsed_seconds=perf_counter()-t0+(base_meta['fit_seconds'] if family=='rf_mse' and not reused else 0),
                    family_stage_seconds=perf_counter()-t0,shared_initial_rf_fit_seconds=base_meta['fit_seconds'],
                    rf_fit_seconds=base_meta['fit_seconds'] if family=='rf_mse' and not reused else 0,
                    rf_fit_started=base_meta['fit_started'],rf_fit_ended=base_meta['fit_ended'],
                    rf_reused=bool(reused),n_supported_origins=int(known.sum()),weight_min=float(w.min()),weight_max=float(w.max())))
                rows.extend(local)
                failed_family=None
        return dict(**job_identity,rows=rows,new_rf_fits=fits,reused_rf_fits=reused,error=None,failed_family=None,seconds=perf_counter()-began)
    except Exception as exc:
        (base_dir/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        return dict(**job_identity,rows=rows,new_rf_fits=fits,reused_rf_fits=reused,
            error=f'{type(exc).__name__}: {exc}',failed_family=failed_family,seconds=perf_counter()-began)

def run_grid(profiles=None,budget_seconds=14400,workers=2,smoke=False,feature_selection=None):
    if feature_selection is None:raise ValueError('완료된 피처 선정 JSON이 필요합니다. F1→F2→F3 비교가 먼저입니다.')
    if workers<1 or workers*4>(os.cpu_count() or 1):raise ValueError('프로세스×RF4스레드가CPU수초과')
    profiles=profiles or runner.ACTIVE_PROFILES
    if any(p not in runner.ACTIVE_PROFILES for p in profiles):raise ValueError('활성6구간만지원')
    feature_info,feature_path=feature_sequence.load_completed_selection(feature_selection,profiles,runner.previous.common.digest(runner.previous.common.SOURCE))
    started=perf_counter();deadline=started+budget_seconds
    identity=datetime.now(runner.previous.common.KST).strftime('%Y%m%dT%H%M%S_KST_')+uuid4().hex[:6]+'_rf_integrated_grid'+('_smoke' if smoke else '')
    suite=ROOT/'artifacts/random_forest/runs'/identity;suite.mkdir(parents=True,exist_ok=False)
    source_files=[Path(__file__),Path(feature_sequence.__file__),RF_DIR/'rf_feature_first_workflow.py',Path(direct.__file__),Path(runner.__file__),Path(runner.previous.__file__),
        Path(runner.previous.risk.__file__),Path(runner.previous.common.__file__),Path(runner.previous.grid.__file__),
        Path(runner.previous.iw.__file__),Path(runner.previous.ml.__file__),Path(runner.previous.final.__file__)]
    source_files.extend([RF_DIR/'rf_future_proxy.py',RF_DIR/'rf_august_confirmation.py'])
    source_files.extend(ROOT/'notebooks/_shared'/name for name in
        ['forecast_common.py','boosting_common.py','performance_profiles.py'])
    embedded_sources={name:module.__embedded_source__ for name,module in
        [('rf_direct_c',direct),('rf_direct_c_experiments',runner),('rf_direct_c_grid',sys.modules[__name__])]
        if hasattr(module,'__embedded_source__')}
    code_bytes={p.name:(embedded_sources[p.stem].encode('utf-8') if p.stem in embedded_sources else p.read_bytes()) for p in source_files}
    config=dict(run_id=identity,status='실행 중',started=_timestamp(),python=platform.python_version(),
        versions={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','nbformat','nbclient','matplotlib']},
        source_sha256=runner.previous.common.digest(runner.previous.common.SOURCE),code_hashes={name:hashlib.sha256(value).hexdigest() for name,value in code_bytes.items()},
        legacy_source=str(runner.previous.final.august.LEGACY),legacy_sha256=runner.previous.common.digest(runner.previous.final.august.LEGACY),
        code_paths={p.name:str(p.relative_to(ROOT)) for p in source_files},
        embedded_module_names=list(embedded_sources),
        windows={p:feature_info['windows'][p] for p in profiles},profiles=profiles,weight_policy='continuous2_only',eval_mode='continuous',
        feature_sets={p:feature_info['selected'][p]['feature_set'] for p in profiles},feature_selection_sha256=feature_sequence.digest(feature_path),
        feature_selection_source=str(feature_path),common_feature_support='F3',stage_order=['F1','F2','F3','feature_selection','tuning'],
        train_support='Feb onward & last target before next month',initial_peak_kw=170.,independent_test=False,
        global_q90_includes_september=True,future_exogenous='actual_as_zero_error_forecast',raw_probability_only=True,
        command=(('python notebooks/models/random_forest/run_rf_integrated_grid_notebook.py --execute' if embedded_sources else
            f'python notebooks/models/random_forest/rf_direct_c_grid.py --workers {workers}')+
            f' --feature-selection "{feature_path}" --budget-seconds {budget_seconds}'+(' --smoke' if smoke else '')),
        budget_seconds=budget_seconds,workers=workers,smoke_only=smoke,parameters={},new_rf_fits=0,reused_rf_fits=0)
    if config['source_sha256']!=feature_info['source_sha256']:raise ValueError('피처 선정 이후 원자료 해시 변경')
    snap=suite/'code_snapshot';snap.mkdir()
    for name,value in code_bytes.items():(snap/name).write_bytes(value)
    runner.previous.dump(suite/'suite_config.json',config)
    (suite/'feature_selection.json').write_bytes(feature_path.read_bytes())
    runner.previous.dump(RF_DIR/'integrated_grid_active.json',dict(run_id=identity,path=str(suite),started=config['started']))
    frame=pd.DataFrame();registry={};grids={};anchors={};job_errors={}
    def batch(data,params,families,phase='cv'):
        nonlocal frame
        params=unique_parameters(params);folds=['september'] if phase=='final' else (['m07'] if smoke else ['m07','m08'])
        for p in params:
            pid=parameter_id(p);config['parameters'][pid]=p
            for family in families:
                candidate=pid+'_'+family;key=(phase,data['profile'],candidate)
                if key not in registry:
                    registry[key]=p;write_candidate_record(suite,config,*key,p,[], '실행 중')
        jobs=(joblib.delayed(dispatch_job)(embedded_sources,suite,config,data,f,p,families,deadline) for p in params for f in folds)
        failures=[]
        with joblib.parallel_config(backend='loky',inner_max_num_threads=1):
            results=joblib.Parallel(n_jobs=workers,return_as='generator_unordered',pre_dispatch=workers)(jobs)
            for result in results:
                if result['rows']:
                    frame=pd.concat([frame,pd.DataFrame(result['rows'])],ignore_index=True)
                    frame=frame.drop_duplicates(['phase','profile','candidate','fold','scope'],keep='last')
                    frame.to_csv(suite/'metrics.csv',index=False,encoding='utf-8-sig')
                config['new_rf_fits']+=result['new_rf_fits'];config['reused_rf_fits']+=result['reused_rf_fits']
                config['elapsed_seconds']=perf_counter()-started;runner.previous.dump(suite/'suite_config.json',config)
                for key,p in registry.items():
                    affected=[result['param_id']+'_'+f for f in result['families']]
                    if key[0]!=result['phase'] or key[1]!=result['profile'] or key[2] not in affected:continue
                    subset=frame.loc[frame.phase.eq(key[0])&frame.profile.eq(key[1])&frame.candidate.eq(key[2])] if len(frame) else frame
                    complete=len(subset)>0 and set(subset.fold)==set(folds)
                    if result['error'] and (not result['failed_family'] or key[2].endswith('_'+result['failed_family'])):
                        job_errors[key]=f"{result['fold']}: {result['error']}"
                    write_candidate_record(suite,config,*key,p,subset.to_dict('records'),
                        '완료' if complete else ('실패' if key in job_errors else '실행 중'),job_errors.get(key))
                if result['error']:failures.append(result['error'])
                runner.previous.log(suite,f"JOB profile={data['profile']} phase={phase} rows={len(result['rows'])} seconds={result['seconds']:.1f} total={config['elapsed_seconds']:.1f} error={result['error']}")
        if failures:raise RuntimeError('; '.join(failures))
    try:
        df,repair,audit=runner.previous.final.load_final_data();config['data_audit']=audit
        for profile in profiles:
            data=build_profile_for_config(df,repair,profile,config)
            grid=core_grid()[:1] if smoke else core_grid();grids[profile]=grid
            runner.previous.log(suite,f'STAGE1 {profile} parameters={len(grid)}')
            batch(data,grid,PAIRED)
            if smoke:continue
            part=frame.loc[frame.profile.eq(profile)&frame.family.eq('c_leaf_d10')]
            anchor=select_models(part)[profile];params=config['parameters'][anchor['param_id']]
            extra=[p for p in refinement_grid(params) if parameter_id(p) not in {parameter_id(q) for q in grid}]
            runner.previous.log(suite,f'STAGE2 {profile} new_parameters={len(extra)}')
            batch(data,extra,PAIRED);grids[profile]=unique_parameters(grid+extra)
            best=select_models(frame.loc[frame.profile.eq(profile)])[profile]
            anchors[profile]=config['parameters'][best['param_id']]
            runner.previous.log(suite,f'STAGE3 {profile} five_learning_methods')
            batch(data,[anchors[profile]],FAMILIES)
        if not smoke:
            selection=select_models(frame)
            runner.previous.dump(suite/'selection.json',dict(selected=selection,selected_at=_timestamp(),
                selection_rule='7·8월full C월동일평균최소;동률기존family순서',september_scores_read=False))
            config['selection_sha256']=runner.previous.common.digest(suite/'selection.json');config['selected']=selection
            runner.previous.dump(suite/'suite_config.json',config)
            for profile in profiles:
                data=build_profile_for_config(df,repair,profile,config)
                runner.previous.log(suite,f'FINAL_ALL {profile} parameters={len(grids[profile])}')
                batch(data,grids[profile],PAIRED,'final')
                batch(data,[anchors[profile]],FAMILIES,'final')
        config.update(status='스모크 완료' if smoke else '학습 완료·검증 대기',ended=_timestamp(),elapsed_seconds=perf_counter()-started)
    except Exception as exc:
        config.update(status='중단 또는 실패',ended=_timestamp(),error=f'{type(exc).__name__}: {exc}',elapsed_seconds=perf_counter()-started)
        (suite/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        for key,p in registry.items():
            subset=frame.loc[frame.phase.eq(key[0])&frame.profile.eq(key[1])&frame.candidate.eq(key[2])] if len(frame) else frame
            needed={'september'} if key[0]=='final' else ({'m07'} if smoke else {'m07','m08'})
            complete=len(subset) and set(subset.fold)==needed
            write_candidate_record(suite,config,*key,p,subset.to_dict('records'),'완료' if complete else '중단',
                None if complete else job_errors.get(key,'전체 실행 중단으로 미완료; 이 후보 자체의 오류는 확인되지 않음'))
        runner.previous.dump(suite/'suite_config.json',config);raise
    finally:
        runner.previous.dump(suite/'suite_config.json',config)
        runner.previous.dump(suite/'records.json',[dict(path=str(_record_path(suite,*key)),phase=key[0],profile=key[1],candidate=key[2]) for key in registry])
        with (ROOT/'experiments/README.md').open('a',encoding='utf-8') as stream:
            stream.write(f'\n## {identity}: RF 통합C 그리드 ({config["status"]})\n\n')
            for key in registry:
                path=_record_path(suite,*key);stream.write(f'- [{path.stem}](random_forest/{path.name})\n')
    print('SUITE',suite,flush=True)
    return suite

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--budget-seconds',type=int,default=14400);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--profiles',nargs='+',choices=runner.ACTIVE_PROFILES)
    parser.add_argument('--feature-selection',required=True,help='F1→F2→F3 완료 selection.json')
    run_grid(**vars(parser.parse_args()))
