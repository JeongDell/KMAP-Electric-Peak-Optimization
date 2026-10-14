"""RF 정의를 보존한 F1/F2/F3 부스팅 비교용 데이터·학습·평가 보조."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
RF_DIR=ROOT/'notebooks/models/random_forest'
if str(RF_DIR) not in sys.path:
    sys.path.insert(0,str(RF_DIR))
import rf_current_data as rf

METRICS=['Loss_A','MAE','L_mean','L_CVaR','L_peak','Loss_B','Loss_C']
FEATURE_SETS=('F1','F2','F3')
SCENARIOS={'observed':'observed','future':'future_actual_as_forecast_zero_error'}

def build_designs(df):
    features,targets,groups=rf.build_features(df)
    future={}
    proxy_columns=[]
    for h in range(1,5):
        for col in rf.OBSERVED:
            name=f'h{h}_forecast_proxy_{col}'
            features[name]=df[col].shift(-h)
            proxy_columns.append(name)
    for feature in FEATURE_SETS:
        future[feature]=[col for col in groups[feature] if not col.startswith('observed_')]+proxy_columns
    support=list(dict.fromkeys(groups['F3']+future['F3']))
    designs={}
    for scenario in SCENARIOS:
        for feature in FEATURE_SETS:
            columns=groups[feature] if scenario=='observed' else future[feature]
            designs[f'{scenario}_{feature}']=dict(X=features[columns].copy(),y=targets.copy(),
                full_features=features,support=support,columns=list(columns),feature_set=feature,scenario=SCENARIOS[scenario])
    return designs

def collect_fold(design,fold):
    train,valid=rf.fold_masks(design['full_features'],design['y'],design['support'],fold)
    X,y=design['X'],design['y']
    return dict(X_train=X.loc[train].copy(),y_train=y.loc[train].copy(),
                X_valid=X.loc[valid].copy(),y_valid=y.loc[valid].copy(),
                train_origins=X.index[train],valid_origins=X.index[valid])

def validate_training_inputs(X_train,y_train,X_valid):
    if not X_train.index.equals(y_train.index):
        raise ValueError('X/y index 정렬이 일치하지 않습니다.')
    if X_train.index.has_duplicates or X_valid.index.has_duplicates:
        raise ValueError('입력 index 중복이 있습니다.')
    if list(X_train.columns)!=list(X_valid.columns):
        raise ValueError('학습/검증 column 열 순서가 일치하지 않습니다.')
    if list(y_train.columns)!=['h1','h2','h3','h4']:
        raise ValueError('정답 열은 h1/h2/h3/h4 순서여야 합니다.')
    for label,frame in [('X_train',X_train),('y_train',y_train),('X_valid',X_valid)]:
        if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f'{label}: 비어 있거나 NaN/무한값이 있습니다.')

def evaluate_paths(actual,predicted,weights=1.0):
    return rf.risk_scores(actual,predicted,weights)

def prediction_rows(fold,origins,actual,predicted):
    evaluate_paths(actual,predicted)
    if len(origins)!=len(actual) or actual.shape[1]!=4:
        raise ValueError('원점 수와 4점 정답 경로가 일치해야 합니다.')
    distances=np.tile([15,30,45,60],len(origins))
    ends=origins.repeat(4)+pd.to_timedelta(distances,unit='min')
    result=pd.DataFrame(dict(fold=fold,origin=origins.repeat(4),target_start=ends-pd.Timedelta(minutes=15),
        target_end=ends,horizon_minutes=distances,actual=np.asarray(actual).ravel(),predicted=np.asarray(predicted).ravel()))
    if result.duplicated(['fold','origin','target_end','horizon_minutes']).any():
        raise ValueError('발행 예측 식별키가 중복됐습니다.')
    return result

def peak_path_rows(actual,predicted):
    evaluate_paths(actual,predicted)
    a,p=np.asarray(actual),np.asarray(predicted)
    am,pm=a.max(axis=1),p.max(axis=1)
    return pd.DataFrame(dict(actual_peak=am,predicted_peak=pm,magnitude_error=pm-am,
        time_error_minutes=np.abs(a.argmax(axis=1)-p.argmax(axis=1))*15.,
        peak_underprediction=np.maximum(0,am-pm),actual_peak_tie=(a==am[:,None]).sum(axis=1)>1))

def macro_scores(metrics):
    if len(metrics)!=4 or set(metrics.fold)!=set(rf.FOLDS):
        raise ValueError('완료된 4fold 결과만 평균할 수 있습니다.')
    cols=[c for c in METRICS if c in metrics]
    if not np.isfinite(metrics[cols].to_numpy()).all():
        raise ValueError('완료 fold 지표에 NaN/무한값이 있습니다.')
    return metrics[cols].mean().to_dict()

def default_params(model_name):
    if model_name=='xgboost':
        return dict(objective='reg:squarederror',eval_metric='rmse',tree_method='hist',booster='gbtree',
            n_estimators=200,learning_rate=0.05,max_depth=6,min_child_weight=1,reg_lambda=1,reg_alpha=0,
            gamma=0,subsample=1.,colsample_bytree=1.,random_state=42,n_jobs=4)
    if model_name=='lightgbm':
        return dict(objective='regression',metric='rmse',boosting_type='gbdt',n_estimators=200,
            learning_rate=0.05,num_leaves=31,min_child_samples=20,max_depth=-1,reg_lambda=0,
            reg_alpha=0,subsample=1.,subsample_freq=0,colsample_bytree=1.,random_state=42,n_jobs=4,
            deterministic=True,force_col_wise=True,verbosity=-1)
    if model_name=='catboost':
        return dict(loss_function='RMSE',eval_metric='RMSE',iterations=200,learning_rate=0.05,depth=6,
            l2_leaf_reg=3,bootstrap_type='Bayesian',bagging_temperature=1,random_strength=1,
            boosting_type='Plain',random_seed=42,thread_count=4,task_type='CPU',
            allow_writing_files=False,verbose=False)
    if model_name=='random_forest':
        return dict(rf.RF_PARAMS,n_jobs=4)
    raise ValueError(f'지원하지 않는 모델: {model_name}')

def build_model(model_name,params):
    if model_name=='xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(**params)
    if model_name=='lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)
    if model_name=='catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params)
    if model_name=='random_forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)
    raise ValueError(f'지원하지 않는 모델: {model_name}')

def fit_direct_models(model_name,params,X_train,y_train,X_valid,deadline=None,on_horizon_done=None):
    from time import perf_counter
    import warnings
    validate_training_inputs(X_train,y_train,X_valid)
    models,predictions,logs=[],[],[]
    for h in y_train.columns:
        if deadline is not None and perf_counter()>=deadline:
            raise TimeoutError(f'{model_name}/{h}: 실행 시간 상한 소진')
        before=perf_counter()
        model=build_model(model_name,dict(params))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(X_train,y_train[h])
            if on_horizon_done is not None:
                on_horizon_done(h,model,None,dict(horizon=h,fit_seconds=perf_counter()-before,
                    warnings=[f'{type(w.message).__name__}: {w.message}' for w in caught]))
            p=np.asarray(model.predict(X_valid),dtype=float)
        if p.shape!=(len(X_valid),) or not np.isfinite(p).all():
            raise ValueError(f'{model_name}/{h}: 예측 shape 또는 유한값 오류')
        models.append(model)
        predictions.append(p)
        logs.append(dict(horizon=h,fit_predict_seconds=perf_counter()-before,
                         warnings=[f'{type(w.message).__name__}: {w.message}' for w in caught]))
    return models,np.column_stack(predictions),dict(horizons=logs)

def predict_direct(models,X):
    if len(models)!=4:
        raise ValueError('h1~h4 모델 4개가 필요합니다.')
    return np.column_stack([np.asarray(m.predict(X,num_threads=4) if m.__class__.__name__=='Booster' else m.predict(X),dtype=float) for m in models])

def save_models(model_name,models,folder,fold):
    folder=Path(folder)
    paths=[]
    for h,model in enumerate(models,1):
        ext={'xgboost':'json','lightgbm':'txt','catboost':'cbm'}[model_name]
        dest=folder/f'model_{fold}_h{h}.{ext}'
        if model_name=='lightgbm':
            # Windows native file I/O can reject Unicode paths; Python handles the text I/O.
            dest.write_text(model.booster_.model_to_string(),encoding='utf-8')
        else:
            model.save_model(str(dest))
        paths.append(str(dest))
    return paths

def load_models(model_name,paths):
    models=[]
    for filename in paths:
        if model_name=='lightgbm':
            import lightgbm as lgb
            model=lgb.Booster(model_str=Path(filename).read_text(encoding='utf-8'))
        elif model_name=='xgboost':
            from xgboost import XGBRegressor
            model=XGBRegressor(n_jobs=4)
            model.load_model(str(filename))
        elif model_name=='catboost':
            from catboost import CatBoostRegressor
            model=CatBoostRegressor(thread_count=4)
            model.load_model(str(filename))
        else:
            raise ValueError(f'지원하지 않는 저장 모델: {model_name}')
        models.append(model)
    return models



def _json(path,value):
    import json
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def _write_record(record):
    from datetime import datetime
    import platform
    c=record['config']
    metrics=pd.DataFrame(record['metrics'])
    table='미완료' if metrics.empty else rf.md_table(metrics[['fold','n_train','n_origins',*METRICS]])
    text=f'''# 실험 {c['experiment_id']}

- 상태: {record['status']}
- 목적과 가설: {c['model']}/{c['feature_set']}/{c['scenario']} 고정파라미터 피처 비교. 모든 후보에 동일7일 지원 원점.
- 시작·종료 시각 및 시간대: {c['started']} / {datetime.now(rf.KST).isoformat() if record['status']!='실행 중' else '미완료'}, Asia/Seoul.
- 모델·라이브러리 버전: {c['model']}; {c['versions']}.
- 데이터 위치·버전 또는 해시·사용 범위: {c['source']}; SHA256 {c['source_sha256']}; 완료시각<2021-08-01 중 유효원점; 원본 미변경.
- 목표값·단위·예측 구간: power_15min kW, 원점 뒤15/30/45/60분의4점 경로; stride15분.
- 학습·검증·테스트 분할과 누수 점검: apr/may/jun/jul expanding; train target_end<검증시작, valid target_end<다음월. 8월·9월 모델 평가 미실행. 미래 전력 입력금지.
- 전처리·특성·하이퍼파라미터·seed: 원단위, scaler없음; {len(c['features'])}열; {c['params']}; support=F3 observed/future union; 외생조건={c['scenario']}; 통계1h 고정.
- 이전 또는 기준 실험 링크와 변경 사항: [새 실행계획](../../docs/superpowers/plans/2026-10-03-boosting-f123-training-plan.md); [비교 인덱스](../README.md); RF 재사용={c.get('reuse_source')}.
- 실행 명령·환경·코드 버전 또는 코드 식별 정보: {c['command']}; Python {platform.python_version()}; interpreter={c['interpreter']}; code SHA256 {c['code_sha256']}; RF 코드 SHA256 {c['rf_code_sha256']}; Git없는workspace.
- 실제 실행 시간·자원 사용량(측정한 경우): {record['elapsed_seconds']:.3f}초; 완료fit {record['new_fits']}회, 재사용fold {record['reused_folds']}개; CPU4threads; fold경계 관측RSS최대 {record['max_observed_rss_bytes']/1024**2:.1f}MiB. 실제 peak RSS는 미측정.
- 평가 지표·구간별 또는 fold별 결과: 아래표 단위kW. 주지표 월별동일가중RMSE; window1h, worst20% ceil, L_peak 가중치1 전체셀 평균후root. LossB/C는 평가이며 직접학습손실이 아님.
- 기준모델 대비 변화와 비교 가능 여부: 같은외생조건·공통원점·정답 내에서 RF/기준선과 비교. 24h지원RF core의 숫자를 피처효과로 직접 비교하지 않음.
- 오차가 큰 조건·해석·한계: horizon/condition/peak표 참조; 단일seed42, 중첩셀 독립아님. 미래조건은 사후외생실측=예보오차0 가정이며 실제운영예보 검증아님. B(t)경보성능 미측정.
- 오류·경고·실패 또는 중단 원인: {record.get('error','없음')}; Python 경고={record.get('warnings',[])}.
- 예측·모델·표·그래프·로그의 저장 경로: {record['artifact']}; config/status, predictions/fold/horizon/condition/peak CSV, model files, run.log.
- 결론과 다음 실험: 동일조건에서만 F효과 판단. 별도 튜닝과8월확인 후 동결,9월최종테스트는 후속 단계.

{table}
'''
    record['record_path'].write_text(text,encoding='utf-8')
    _json(record['artifact']/'status.json',{k:record.get(k) for k in ['status','new_fits','reused_folds','elapsed_seconds','error','max_observed_rss_bytes']})

def _reuse_rf(design,part,fold,reuse_source,config):
    import json,joblib
    old_config=json.loads((reuse_source/'config.json').read_text(encoding='utf-8'))
    if old_config['source_sha256']!=config['source_sha256'] or old_config['code_sha256']!=config['rf_code_sha256']:
        raise ValueError('RF 재사용 데이터/코드 SHA256 불일치')
    if old_config['features']!=design['columns'] or old_config['support']!='F3':
        raise ValueError('RF 재사용 피처/7일지원 불일치')
    model=joblib.load(reuse_source/f'model_{fold}.joblib')
    model.set_params(n_jobs=4)
    for key,val in config['params'].items():
        if model.get_params()[key]!=val:
            raise ValueError(f'RF 재사용 파라미터 불일치: {key}')
    old_metrics=pd.read_csv(reuse_source/'fold_metrics.csv')
    if int(old_metrics.loc[old_metrics.fold.eq(fold),'n_train'].iloc[0])!=len(part['X_train']):
        raise ValueError('RF 재사용 학습 원점 수 불일치')
    historical=ROOT/'artifacts/random_forest/runs/20261003T131206_KST_cbba11'
    prior=json.loads((historical/'suite.json').read_text(encoding='utf-8'))
    if prior['source_sha256']!=config['source_sha256']:
        raise ValueError('RF 원실행 데이터 불일치')
    # Same RF code, data, F3 support and sorted complete timeline reconstruct the exact train origins.
    old=pd.read_csv(reuse_source/'predictions.csv',parse_dates=['origin','target_end'])
    old=old.loc[old.fold.eq(fold)].sort_values(['origin','horizon_minutes'])
    predicted=np.asarray(model.predict(part['X_valid']))
    np.testing.assert_array_equal(old.origin.drop_duplicates().to_numpy(),part['valid_origins'].to_numpy())
    np.testing.assert_array_equal(old.actual.to_numpy().reshape(-1,4),part['y_valid'].to_numpy())
    np.testing.assert_allclose(old.predicted.to_numpy().reshape(-1,4),predicted,rtol=1e-12,atol=1e-9)
    return model,predicted

def run_feature_suite(model_name,root=ROOT,source=rf.SOURCE,budget_seconds=2700,scenarios=('observed','future')):
    from datetime import datetime
    from time import perf_counter
    from uuid import uuid4
    from importlib.metadata import version
    import joblib,psutil,warnings
    root=Path(root)
    before=perf_counter()
    suite_id=datetime.now(rf.KST).strftime('%Y%m%dT%H%M%S_KST')+'_'+uuid4().hex[:6]+'_f123'
    suite=root/'artifacts'/model_name/'runs'/suite_id
    suite.mkdir(parents=True,exist_ok=False)
    records_dir=root/'experiments'/model_name
    records_dir.mkdir(parents=True,exist_ok=True)
    versions={p:version(p) for p in ['numpy','pandas','scikit-learn','joblib','nbclient']}
    if model_name!='random_forest':
        versions[model_name]=version(model_name)
    df=rf.load_data(source)
    designs=build_designs(df)
    params=default_params(model_name)
    deadline=before+budget_seconds
    all_metrics,all_macro,records=[],[],[]
    status='완료'
    nfits=0
    fatal_error=None
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f'외생 시나리오 오류: {scenario}')
        for feature in FEATURE_SETS:
            key=f'{scenario}_{feature}'
            design=designs[key]
            artifact=suite/key
            artifact.mkdir(exist_ok=False)
            reuse_source=None
            if model_name=='random_forest' and scenario=='observed' and feature in ['F2','F3']:
                old_name='F2_week_support' if feature=='F2' else 'F3'
                reuse_source=ROOT/'artifacts/random_forest/runs/20261003T131206_KST_cbba11'/old_name
            config=dict(experiment_id=f'{suite_id}_{key}',model=model_name,feature_set=feature,
                scenario=design['scenario'],support='7d_common_all_F',params=params,features=design['columns'],
                source=str(Path(source).resolve()),source_sha256=rf.digest(source),code_sha256=rf.digest(__file__),
                rf_code_sha256=rf.digest(rf.__file__),versions=versions,interpreter=sys.executable,
                started=datetime.now(rf.KST).isoformat(),folds=rf.FOLDS,budget_seconds=budget_seconds,
                command=f'Jupyter {model_name}_rf_features.ipynb Run All',
                reuse_source=str(reuse_source) if reuse_source else None)
            record=dict(config=config,artifact=artifact,record_path=records_dir/f'{config["experiment_id"]}.md',
                metrics=[],status='실행 중',elapsed_seconds=0.,new_fits=0,reused_folds=0,max_observed_rss_bytes=0,warnings=[])
            records.append(record)
            _json(artifact/'config.json',config)
            _write_record(record)
            started=perf_counter()
            predictions,horizons,conditions,peaks,logs=[],[],[],[],[]
            try:
                for fold in rf.FOLDS:
                    if perf_counter()>=deadline:
                        raise TimeoutError(f'{model_name}/{key}/{fold}: 지정 실행 예산 소진')
                    part=collect_fold(design,fold)
                    validate_training_inputs(part['X_train'],part['y_train'],part['X_valid'])
                    process=psutil.Process()
                    rss_before=process.memory_info().rss
                    if rss_before>psutil.virtual_memory().total*.7:
                        raise MemoryError('fit 시작 전 process RSS가 물리 RAM의70%를 넘었습니다.')
                    t0=perf_counter()
                    if model_name=='random_forest':
                        if reuse_source:
                            model,predicted=_reuse_rf(design,part,fold,reuse_source,config)
                            record['reused_folds']+=1
                        else:
                            model=build_model(model_name,params)
                            model.fit(part['X_train'],part['y_train'])
                            record['new_fits']+=1
                            record['elapsed_seconds']=perf_counter()-started
                            _write_record(record)
                            predicted=model.predict(part['X_valid'])
                        model_path=artifact/f'model_{fold}.joblib'
                        joblib.dump(model,model_path,compress=3)
                        np.testing.assert_allclose(joblib.load(model_path).predict(part['X_valid']),predicted,rtol=1e-6,atol=1e-6)
                    else:
                        def on_fit(h,model,p,info):
                            record['new_fits']+=1
                            record['warnings'].extend(info['warnings'])
                            record['elapsed_seconds']=perf_counter()-started
                            _write_record(record)
                            _json(artifact/f'fit_completed_{fold}_{h}.json',info)
                        models,predicted,info=fit_direct_models(model_name,params,part['X_train'],part['y_train'],part['X_valid'],deadline=deadline,on_horizon_done=on_fit)
                        paths=save_models(model_name,models,artifact,fold)
                        np.testing.assert_allclose(predict_direct(load_models(model_name,paths),part['X_valid']),predicted,rtol=1e-6,atol=1e-6)
                        _json(artifact/f'fit_details_{fold}.json',info)
                    actual=part['y_valid'].to_numpy(dtype=float)
                    elapsed=perf_counter()-t0
                    record['max_observed_rss_bytes']=max(record['max_observed_rss_bytes'],rss_before,process.memory_info().rss)
                    m=dict(model=model_name,feature_set=feature,scenario=design['scenario'],fold=fold,n_train=len(part['X_train']),
                           fit_predict_save_seconds=elapsed,**evaluate_paths(actual,predicted))
                    record['metrics'].append(m)
                    frame=prediction_rows(fold,part['valid_origins'],actual,predicted).assign(model=model_name,feature_set=feature,scenario=design['scenario'])
                    predictions.append(frame)
                    pd.DataFrame({'origin':part['train_origins']}).to_csv(artifact/f'train_origins_{fold}.csv',index=False)
                    for h in range(4):
                        horizons.append(dict(fold=fold,horizon_minutes=15*(h+1),**evaluate_paths(actual[:,h:h+1],predicted[:,h:h+1])))
                    for col,vals in [('hour',frame.target_start.dt.hour),('weekday',frame.target_start.dt.dayofweek),('is_night',(frame.target_start.dt.hour>=18)|(frame.target_start.dt.hour<9)),('is_workday',np.column_stack([design['full_features'].loc[part['valid_origins'],f'h{h}_workday'] for h in range(1,5)]).ravel())]:
                        for value in sorted(pd.unique(vals)):
                            selected=frame.loc[np.asarray(vals)==value]
                            conditions.append(dict(fold=fold,condition=col,value=str(value),n_cells=len(selected),
                                **evaluate_paths(selected.actual.to_numpy()[:,None],selected.predicted.to_numpy()[:,None])))
                    peak=peak_path_rows(actual,predicted)
                    peaks.append(dict(fold=fold,n_origins=len(peak),path_max_MAE=float(np.abs(peak.magnitude_error).mean()),
                        path_max_under_RMSE=float(np.sqrt(np.mean(peak.peak_underprediction**2))),
                        argmax_timing_MAE_minutes=float(peak.time_error_minutes.mean()),actual_max_tie_origins=int(peak.actual_peak_tie.sum())))
                    line=f'{model_name}/{key}/{fold}: RMSE={m["Loss_A"]:.4f}, L_peak={m["L_peak"]:.4f}, {elapsed:.1f}s'
                    logs.append(line)
                    print(line,flush=True)
                    for filename,items in [('fold_metrics',record['metrics']),('horizon_metrics',horizons),('condition_metrics',conditions),('peak_path_metrics',peaks)]:
                        pd.DataFrame(items).to_csv(artifact/f'{filename}.csv',index=False,encoding='utf-8-sig')
                    pd.concat(predictions,ignore_index=True).to_csv(artifact/'predictions.csv',index=False,encoding='utf-8-sig')
                    (artifact/'run.log').write_text('\n'.join(logs)+'\n',encoding='utf-8')
                record['status']='완료'
                values=macro_scores(pd.DataFrame(record['metrics']))
                all_macro.append(dict(model=model_name,feature_set=feature,scenario=design['scenario'],**values,
                    n_origins=sum(m['n_origins'] for m in record['metrics']),n_predictions=sum(m['n_predictions'] for m in record['metrics']),
                    fit_predict_save_seconds=sum(m['fit_predict_save_seconds'] for m in record['metrics']),
                    reused_folds=record['reused_folds'],record_path=str(record['record_path'])))
            except (Exception,KeyboardInterrupt) as exc:
                record['status']='중단' if isinstance(exc,(TimeoutError,MemoryError,KeyboardInterrupt)) else '실패'
                record['error']=f'{type(exc).__name__}: {exc}'
                (artifact/'run.log').write_text('\n'.join(logs)+'\n'+record['error'],encoding='utf-8')
                status=record['status']
                if not isinstance(exc,(TimeoutError,MemoryError,KeyboardInterrupt)):
                    fatal_error=exc
            record['elapsed_seconds']=perf_counter()-started
            nfits+=record['new_fits']
            _write_record(record)
            all_metrics.extend(record['metrics'])
            if record['status']!='완료':
                break
        if status!='완료':
            break
    metrics,macro=pd.DataFrame(all_metrics),pd.DataFrame(all_macro)
    metrics.to_csv(suite/'comparison_folds.csv',index=False,encoding='utf-8-sig')
    macro.to_csv(suite/'comparison_macro.csv',index=False,encoding='utf-8-sig')
    _json(suite/'suite.json',dict(suite_id=suite_id,model=model_name,status=status,completed_candidates=len(macro),
        planned_candidates=len(scenarios)*3,new_fits=nfits,reused_folds=sum(r['reused_folds'] for r in records),
        elapsed_seconds=perf_counter()-before,source_sha256=rf.digest(source),code_sha256=rf.digest(__file__),
        records=[str(r['record_path']) for r in records],august_september='not evaluated',scenarios=list(scenarios)))
    print(f'SAVED {suite}',flush=True)
    if fatal_error is not None:
        raise fatal_error
    return dict(output_dir=suite,metrics=metrics,macro=macro,status=status)
