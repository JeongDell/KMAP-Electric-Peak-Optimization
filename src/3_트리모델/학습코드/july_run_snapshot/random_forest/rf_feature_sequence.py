"""F1→F2→F3 고정 설정 비교와 이후 단계에 넘길 피처 선정.

입력 창·시점·정답은 세 묶음에 공통이다. execute=False는 학습도 파일도 만들지 않는다.
"""
from pathlib import Path
from datetime import datetime
from time import perf_counter
from uuid import uuid4
import hashlib,json,platform,traceback
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits
import rf_direct_c_experiments as runner
import rf_direct_c as direct

ROOT=runner.ROOT
FEATURE_ORDER=('F1','F2','F3')
WINDOWS=dict(short_1h=1,medium_4h=1,medium_8h=48,medium_12h=48,medium_24h=48,long_4d=24)
FIXED_PARAMS=dict(n_estimators=64,min_samples_leaf=20,max_features=1/3,max_leaf_nodes=64,max_depth=None,
                 max_samples=None,bootstrap=True,criterion='squared_error',random_state=42,n_jobs=4)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def feature_source_files():
    """피처 비교에 실제로 사용되는 입력·분할·학습 코드의 전체 의존성."""
    from importlib import import_module
    import rf_direct_c_grid as grid
    return [Path(__file__),Path(__file__).parent/'rf_feature_first_workflow.py',Path(grid.__file__),Path(direct.__file__),Path(runner.__file__),Path(runner.previous.__file__),
        Path(runner.previous.common.__file__),Path(runner.previous.risk.__file__),Path(runner.previous.grid.__file__),
        Path(runner.previous.iw.__file__),Path(runner.previous.ml.__file__),Path(runner.previous.final.__file__),
        Path(runner.previous.grid.proxy.__file__),Path(runner.previous.ml.august.__file__),
        *(Path(import_module(name).__file__) for name in ['forecast_common','boosting_common','performance_profiles'])]

def preview_plan():
    return dict(status='미실행',comparison_order=list(FEATURE_ORDER),profiles=list(WINDOWS),windows=WINDOWS.copy(),
                fixed_params=FIXED_PARAMS.copy(),folds=['m07','m08'],selection_metric='7·8월 full Loss_C 월동일평균',
                common_support='F3 필요 입력의 유한 공통 원점',test_in_feature_stage=False)

def _feature_frames(base,df,origins,window,repair=None,base_repair=None):
    """일간·주간 목표시각 참조를 이용 가능한 과거로 후퇴시킨다."""
    h=runner.previous.PROFILES[base.attrs['profile']]['horizon_steps'];power=df.power_15min.astype(float)
    simple=[c for c in base if not ('power' in c or 'peak_threshold' in c)]
    extras={};offsets=[0,1,2,4,672]
    for name,offset in [('lag_15min',1),('lag_30min',2),('lag_7d',672)]:extras[f'power_{name}']=power.shift(offset).reindex(origins).to_numpy()
    for label,steps in [('4h',16),('24h',96)]:
        if window==steps//4:continue # 기존 입력창의 통계와 중복되는 열은 추가하지 않는다.
        rolling=power.rolling(steps)
        for stat in ['mean','max','std']:
            values=rolling.std(ddof=0) if stat=='std' else getattr(rolling,stat)()
            extras[f'power_extra_{label}_{stat}']=values.reindex(origins).to_numpy()
    for label,offset in [('15min',1),('1h',4)]:extras[f'power_change_{label}']=(power-power.shift(offset)).reindex(origins).to_numpy()
    for lead in range(1,h+1):
        for label,period in [('day',96),('week',672)]:
            offset=period*((lead+period-1)//period)-lead
            assert offset>=0
            extras[f'h{lead}_previous_{label}_available']=power.shift(offset).reindex(origins).to_numpy();offsets.append(offset)
    frames={'F1':base.loc[:,simple].copy(),'F2':base.copy(),'F3':pd.concat([base,pd.DataFrame(extras,index=origins)],axis=1)}
    masks={key:np.zeros(len(origins),bool) for key in FEATURE_ORDER}
    if repair is not None:
        masks['F2']=np.asarray(base_repair,bool)
        extra=repair.rolling(96,min_periods=1).max().astype(bool)
        for offset in set(offsets):extra|=repair.shift(offset,fill_value=False)
        masks['F3']=masks['F2']|extra.reindex(origins).to_numpy(bool)
    return frames,masks

def prepare_feature_sets(df,repair,profile,window):
    base=runner.previous.build_profile(df,repair,profile,window,170.,'dynamic')
    base['features'].attrs['profile']=profile
    frames,masks=_feature_frames(base['features'],df,base['origins'],window,repair,base['has_repaired_input'])
    complete=np.isfinite(frames['F3'].to_numpy(float)).all(axis=1)
    result={}
    for key,x in frames.items():
        chosen=base['origins'][complete]
        result[key]={**base,'feature_set':key,'features':x.loc[chosen].copy(),'targets':base['targets'].loc[chosen].copy(),
            'threshold':base['threshold'].loc[chosen].copy(),'origins':chosen,'has_repaired_target':base['has_repaired_target'][complete],
            'has_repaired_input':masks[key][complete],'common_feature_support':'F3'}
    return result

def build_query_features(df,origins,profile,window,feature_set,initial=170.,mode='dynamic'):
    if feature_set not in FEATURE_ORDER:raise ValueError('F1/F2/F3 피처 묶음 필요')
    origins=pd.DatetimeIndex(origins)
    if not len(origins) or not origins.isin(df.index).all():raise ValueError('데이터에 존재하는 예측 원점 필요')
    if profile=='short_1h':
        base,_,columns=runner.previous.grid.build_window_features(df,window);base=base.loc[origins,columns]
    else:base,_,_=runner.previous.iw.build_features(df,runner.previous.PROFILES[profile]['horizon_steps'],window,origins)
    past=df.power_15min.loc[df.index<=origins.max()]
    b=runner.previous.risk.thresholds(past,initial,mode).reindex(origins)
    base=base.assign(peak_threshold_origin=b,power_latest_to_peak=base.power_latest/b)
    base.attrs['profile']=profile
    frames,_=_feature_frames(base,df,base.index,window)
    x=frames[feature_set]
    if not np.isfinite(x.to_numpy(float)).all():raise ValueError('선정 피처에 필요한 과거 이력/미래 외생 부족')
    return x,b

def _comparison_config(profiles,run,stamp,budget_seconds,files,versions):
    """학습 전에 구성하는 실행 메타데이터. 계획의 상태/구간을 명시적으로 갱신한다."""
    legacy=ROOT/'data/processed/timeseries_all.csv'
    return {**preview_plan(),'status':'실행 중','run_id':run.name,'started':stamp.isoformat(),'profiles':profiles,
        'source_sha256':digest(runner.previous.common.SOURCE),'source':str(runner.previous.common.SOURCE),
        'legacy_source':str(legacy),'legacy_sha256':digest(legacy),'budget_seconds':budget_seconds,
        'code_hashes':{p.name:digest(p) for p in files},'code_paths':{p.name:p.relative_to(ROOT).as_posix() for p in files},
        'versions':versions,'python':platform.python_version(),'new_rf_fits':0}

def select_features(metrics):
    cv=metrics.loc[metrics.fold.isin(['m07','m08'])&metrics.scope.eq('full')]
    if cv.empty or cv.duplicated(['profile','feature_set','fold']).any():raise ValueError('유일한 7·8월 피처 비교 필요')
    selected={};pairs=[]
    for profile,part in cv.groupby('profile',sort=False):
        scores={}
        for key in FEATURE_ORDER:
            rows=part.loc[part.feature_set.eq(key)]
            if set(rows.fold)!={'m07','m08'} or not np.isfinite(rows.Loss_C.to_numpy()).all():raise ValueError('F1/F2/F3 모두의 7·8월 평가 필요')
            scores[key]=float(rows.Loss_C.mean())
        choice=min(FEATURE_ORDER,key=lambda k:(scores[k],FEATURE_ORDER.index(k)))
        selected[profile]=dict(feature_set=choice,cv_C=scores[choice],scores=scores)
        for old,key in [('F1','F2'),('F2','F3')]:
            pairs.append(dict(profile=profile,previous=old,feature_set=key,previous_C=scores[old],current_C=scores[key],
                delta_C=scores[key]-scores[old],change_pct=100*(scores[key]/scores[old]-1) if scores[old] else np.nan))
    return selected,pd.DataFrame(pairs)

def load_completed_selection(path,profiles,source_sha256):
    if path is None:raise ValueError('완료된 피처 선정 JSON이 필요합니다. F1→F2→F3 비교를 먼저 실행하세요.')
    path=Path(path)
    if path.is_dir():path=path/'selection.json'
    data=runner.previous.read(path)
    if data.get('status')!='feature_selection_complete' or data.get('comparison_order')!=list(FEATURE_ORDER):raise ValueError('완료된 F1→F2→F3 피처 선정만 허용')
    if data.get('folds')!=['m07','m08'] or data.get('test_used_for_selection') is not False:raise ValueError('7·8월 검증만 사용한 피처 선정 필요')
    if data.get('source_sha256')!=source_sha256:raise ValueError('피처 선정 데이터 해시 불일치')
    if data.get('feature_code_sha256')!=digest(__file__):raise ValueError('피처 생성 코드 버전 불일치; 선정 재검토 필요')
    expected=feature_source_files()
    if set(data.get('code_hashes',{}))!={p.name for p in expected}:raise ValueError('피처 선정 의존성 코드 해시 누락')
    for file in expected:
        if data['code_hashes'][file.name]!=digest(file):raise ValueError(f'피처 선정 의존성 코드 버전 불일치: {file.name}')
    for profile in profiles:
        if data.get('selected',{}).get(profile,{}).get('feature_set') not in FEATURE_ORDER:raise ValueError('모든 구간의 피처 선정 필요')
        if data.get('windows',{}).get(profile)!=WINDOWS[profile]:raise ValueError('선정 입력창 불일치')
    return data,path

def profile_for_config(df,repair,profile,config):
    if config.get('feature_sets'):
        return prepare_feature_sets(df,repair,profile,config['windows'][profile])[config['feature_sets'][profile]]
    # Historical report/inference compatibility only; new training entry points are gated.
    return runner.previous.build_profile(df,repair,profile,config['windows'][profile],170.,'dynamic')

def query_for_bundle(df,origins,profile,bundle):
    key=bundle.get('feature_set')
    if key:return build_query_features(df,origins,profile,bundle['context_window_hours'],key)
    return runner.previous.build_query_features(df,origins,profile,bundle['context_window_hours'],170.,'dynamic')

def _record(run,config,profile,key,status,rows,error=None):
    eid=f'{profile}_{key}';dest=run/'records'/f'{eid}.md'
    text=f'''# 실험 {eid}

- 상태: {status}
- 목적과 가설: F1기준→F2전력이력추가→F3주간·변화량추가, 동일설정/공통원점에서 순차 비교.
- 시작·종료 시각 및 시간대: {config['started']} / {datetime.now(runner.previous.common.KST).isoformat() if status!='실행 중' else '미완료'}; Asia/Seoul.
- 모델·라이브러리 버전: Python{platform.python_version()}; suite_config.json 환경 버전.
- 데이터 위치·버전 또는 해시·사용 범위: {runner.previous.common.SOURCE}, SHA256 {config['source_sha256']}; 1~8월 범위의 유효 학습/검증 경로.
- 목표값·단위·예측 구간: power_15min kW; {runner.previous.PROFILES[profile]}; 원점부터 미래전력H점.
- 학습·검증·테스트 분할과 누수 점검: 7·8월 expanding; 학습 전체정답 끝<평가시작; 검증 끝<다음월. 동일F3지원. 9월 예측·평가 미실행.
- 전처리·특성·하이퍼파라미터·seed: {key}, window{WINDOWS[profile]}h; {FIXED_PARAMS}; scaler없음, 연속형2; 초기RF→C10공동잎(max120iter/16초), 피처비교동안 고정. schema.json 피처순서.
- 이전 또는 기준 실험 링크와 변경 사항: [설계](../../../../docs/superpowers/specs/2026-10-05-rf-feature-first.md); 이전F2튜닝 모델을 이번 피처선정의결론으로 재사용하지않음.
- 실행 명령·환경·코드 버전 또는 코드 식별 정보: rf_feature_first_workflow.py --execute; suite_config.json code_hashes/code_snapshot. Git없음.
- 실제 실행 시간·자원 사용량: 아래fold별,RF4threads/BLAS1/순차;RSS미측정.
- 평가 지표·구간별 또는 fold별 결과: 주C·보조RMSE/MAE/Lpeak,full/clean_targets/clean_targets_inputs. 아래full표 및metrics.csv.
- 기준모델 대비 변화와 비교 가능 여부: F2-F1/F3-F2 단계별효과는pairwise.csv; 같은profile의같은원점·정답만비교.
- 오차가 큰 조건·해석·한계: 입력창은기존F2조건,미래외생실측=오차0가정,q90은전체자료정보포함;기존9월열람으로이후9월은탐색평가. repaired정답은참값복구가아님. window별초과학습제한 기록.
- 오류·경고·실패 또는 중단 원인: {error or '없음'}.
- 예측·모델·표·그래프·로그의 저장 경로: ../{profile}/{key}/; 모델/예측/설정; ../metrics.csv; ../pairwise.csv; ../selection.json.
- 결론과 다음 실험: 피처선정완료 JSON이 있어야만 다음튜닝에진입. 최종9월및확률은이번피처비교단계에서수행하지않음.

{pd.DataFrame(rows).to_markdown(index=False) if rows else '실제 완료 fold 없음.'}
'''
    dest.write_text(text,encoding='utf-8')

def run_feature_comparison(execute=False,budget_seconds=14400,profiles=None):
    if not execute:return preview_plan()
    profiles=profiles or list(WINDOWS)
    if any(p not in WINDOWS for p in profiles):raise ValueError('활성6구간만 지원')
    import rf_direct_c_grid as grid
    from importlib.metadata import version
    start=perf_counter();deadline=start+budget_seconds;stamp=datetime.now(runner.previous.common.KST)
    run=ROOT/'experiments/random_forest'/(stamp.strftime('%Y%m%dT%H%M%S_KST_')+uuid4().hex[:6]+'_feature_sequence')
    run.mkdir(parents=True);(run/'records').mkdir();snap=run/'code_snapshot';snap.mkdir()
    print('FEATURE_RUN='+str(run),flush=True)
    files=feature_source_files()
    config=_comparison_config(profiles,run,stamp,budget_seconds,files,
        {p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib']})
    for path in files:(snap/path.name).write_bytes(path.read_bytes())
    runner.previous.dump(run/'suite_config.json',config);rows=[];partial={}
    try:
        df,repair,audit=runner.previous.final.load_final_data();runner.previous.dump(run/'data_audit.json',audit)
        if digest(runner.previous.common.SOURCE)!=config['source_sha256']:raise ValueError('피처 비교 시작 중 데이터 변경')
        all_data={p:prepare_feature_sets(df,repair,p,WINDOWS[p]) for p in profiles}
        with threadpool_limits(limits=1):
            for key in FEATURE_ORDER:
                for profile in profiles:
                    data=all_data[profile][key];partial[(profile,key)]=[];_record(run,config,profile,key,'실행 중',[])
                    for month in [7,8]:
                        if perf_counter()>=deadline:raise TimeoutError('피처 비교 시간 예산 소진')
                        print(f'FEATURE_START {key} {profile} m{month:02}',flush=True)
                        before=perf_counter();start_month=pd.Timestamp(2021,month,1);end=start_month+pd.offsets.MonthBegin(1)
                        ti,vi=runner.split_indices(data['origins'],data['spec']['horizon_steps'],start_month,end)
                        x=data['features'].iloc[ti].to_numpy(np.float32);y=data['targets'].iloc[ti].to_numpy();b=data['threshold'].iloc[ti].to_numpy()
                        supported=np.isfinite(runner.support_reference(data)[ti]).all(axis=1);weights,_=grid.pure_training_weights(y,b,supported)
                        forest=RandomForestRegressor(**FIXED_PARAMS).fit(x,y,sample_weight=weights.mean(axis=1));config['new_rf_fits']+=1
                        model=direct.ForestCModel.from_forest(forest)
                        optimizer=direct.optimize_leaves(model,x,y,weights,data['spec']['loss_steps'],delta=10,max_iter=120,deadline=min(deadline,perf_counter()+16))
                        predicted=model.predict(data['features'].iloc[vi].to_numpy(np.float32));actual=data['targets'].iloc[vi].to_numpy();vb=data['threshold'].iloc[vi].to_numpy()
                        folder=run/profile/key/f'm{month:02}';folder.mkdir(parents=True)
                        joblib.dump(dict(model=model,features=list(data['features']),feature_set=key,context_window_hours=WINDOWS[profile],profile=profile),folder/'model_bundle.joblib',compress=3)
                        runner.previous.dump(folder/'schema.json',dict(features=list(data['features']),feature_set=key))
                        h=data['spec']['horizon_steps'];origins=data['origins'][vi];targets=origins.to_numpy()[:,None]+np.arange(1,h+1)*np.timedelta64(15,'m')
                        pd.DataFrame(dict(origin=np.repeat(origins,h),target_end=targets.ravel(),actual_kw=actual.ravel(),predicted_kw=predicted.ravel())).to_csv(folder/'predictions.csv.gz',index=False,compression='gzip')
                        ew=grid.evaluation_weights(actual,vb);masks=dict(full=np.ones(len(vi),bool),clean_targets=~data['has_repaired_target'][vi],clean_targets_inputs=~(data['has_repaired_target'][vi]|data['has_repaired_input'][vi]))
                        for scope,keep in masks.items():
                            row=dict(profile=profile,feature_set=key,fold=f'm{month:02}',scope=scope,n_origins=int(keep.sum()),elapsed_seconds=perf_counter()-before,solver_success=optimizer['solver_success'])
                            if keep.any():row.update(runner.previous.risk.point_metrics(actual[keep],predicted[keep],ew[keep],data['spec']['loss_steps']))
                            rows.append(row)
                            if scope=='full':partial[(profile,key)].append(row)
                        runner.previous.dump(folder/'fold_config.json',dict(params=FIXED_PARAMS,feature_set=key,train_label_max=str((data['origins'][ti]+pd.Timedelta(minutes=15*h)).max()),valid_origin_min=str(origins.min()),valid_label_max=str(targets.max()),optimizer=optimizer))
                        pd.DataFrame(rows).to_csv(run/'metrics.csv',index=False);_record(run,config,profile,key,'실행 중',partial[(profile,key)])
                        config['elapsed_seconds']=perf_counter()-start;runner.previous.dump(run/'suite_config.json',config)
                        print(f'FEATURE_DONE {key} {profile} m{month:02} C={partial[(profile,key)][-1]["Loss_C"]:.6f} seconds={perf_counter()-before:.1f}',flush=True)
                    _record(run,config,profile,key,'완료',partial[(profile,key)])
        if digest(runner.previous.common.SOURCE)!=config['source_sha256']:raise ValueError('피처 비교 중 데이터 변경')
        chosen,pairs=select_features(pd.DataFrame(rows));pairs.to_csv(run/'pairwise.csv',index=False)
        selection=dict(status='feature_selection_complete',comparison_order=list(FEATURE_ORDER),folds=['m07','m08'],selected=chosen,
            source_sha256=config['source_sha256'],feature_code_sha256=digest(__file__),code_hashes=config['code_hashes'],
            code_paths=config['code_paths'],windows=WINDOWS,test_used_for_selection=False,common_support='F3',
            feature_run=str(run),selected_at=datetime.now(runner.previous.common.KST).isoformat())
        runner.previous.dump(run/'selection.json',selection)
        pd.DataFrame([dict(experiment_id=f'{p}_{k}',status='완료',profile=p,feature_set=k,cv_C=v['scores'][k],selected=k==v['feature_set'],record_path=f'records/{p}_{k}.md') for p,v in chosen.items() for k in FEATURE_ORDER]).to_csv(run/'summary.csv',index=False)
        config.update(status='피처 선정 완료',ended=datetime.now(runner.previous.common.KST).isoformat(),elapsed_seconds=perf_counter()-start)
    except Exception:
        config.update(status='실패',error=traceback.format_exc(),elapsed_seconds=perf_counter()-start)
        for (p,k),values in partial.items():_record(run,config,p,k,'실패' if len(values)<2 else '완료',values,config['error'])
        raise
    finally:
        runner.previous.dump(run/'suite_config.json',config)
        with (ROOT/'experiments/README.md').open('a',encoding='utf-8') as stream:
            stream.write(f'\n## RF 피처 순차 비교 {run.name} ({config["status"]})\n\n[설정](random_forest/{run.name}/suite_config.json) · [지표](random_forest/{run.name}/metrics.csv)\n')
            for p,k in partial:stream.write(f'- [{p}_{k}](random_forest/{run.name}/records/{p}_{k}.md)\n')
    return run/'selection.json'
