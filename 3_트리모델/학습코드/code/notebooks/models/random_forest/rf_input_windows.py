"""중장기 RF 입력 통계 창만 비교. 같은 원점, CV 선정 후8월 확인."""
from pathlib import Path
from datetime import datetime
from time import perf_counter
from uuid import uuid4
from importlib.metadata import version
import gc
import json
import platform
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import rf_current_data as common
import rf_grid_search as grid
import rf_future_proxy as proxy
import rf_august_confirmation as august
import rf_medium_long as ml

ROOT=common.ROOT
WINDOWS={'medium_4h':[1,4,24],'medium_8h':[1,24,48],
         'medium_12h':[1,24,48],'long_3d':[1,24,168]}
METRICS=grid.METRICS
OLD_RUN=ROOT/'artifacts/random_forest/runs/20261003T181854_KST_b27d86_rf_medium_long'


def build_features(df,horizon,window_hours,origins=None):
    if not isinstance(window_hours,int) or not 1<=window_hours<=168:
        raise ValueError('입력 통계 창은1~168시간 정수여야 합니다.')
    x,y,columns=ml.build_features(df,horizon,origins)
    roll=df.power_15min.astype(float).rolling(window_hours*4)
    rename={f'power_1h_{s}':f'power_roll_{s}' for s in ['mean','max','std']}
    x=x.rename(columns=rename)
    for stat in ['mean','max','std']:
        series=roll.std(ddof=0) if stat=='std' else getattr(roll,stat)()
        x[f'power_roll_{stat}']=series.reindex(x.index).to_numpy()
    return x,y,[rename.get(c,c) for c in columns]


def prepare_profile(df,repaired,profile,start,end):
    if pd.Timestamp(end)>august.END: raise ValueError('9월 최종 테스트는 보존합니다.')
    data=ml.prepare_profile(df,repaired,profile,start,end)
    origins=data['features'].index
    features={}; complete=np.isfinite(data['targets'].to_numpy()).all(axis=1)
    for w in WINDOWS[profile]:
        x,y,cols=build_features(data['df'],data['spec']['horizon_steps'],w,origins)
        np.testing.assert_array_equal(y.to_numpy(),data['targets'].to_numpy())
        features[w]=x
        complete &= np.isfinite(x.to_numpy()).all(axis=1)
    data['train'] &= complete; data['valid'] &= complete
    if not data['train'].any() or not data['valid'].any(): raise ValueError(f'{profile}: 공통 원점 없음')
    # All candidate rolling windows end at origin and share a single repair sensitivity mask.
    tags=data['tags'].copy()
    touched=data['repaired'].rolling(max(WINDOWS[profile])*4,min_periods=1).max().astype(bool)
    tags['has_repaired_rf_input'] |= touched
    tags['has_repaired_any_input']=tags.has_repaired_rf_input|tags.has_repaired_daily_input|tags.has_repaired_weekly_input
    flags=tags.reindex(origins)
    data.update(features_by_window=features,columns=cols,tags=tags,
        scopes=dict(full=data['valid'].copy(),
            clean_targets=data['valid']&~flags.has_repaired_target,
            clean_targets_inputs=data['valid']&~flags.has_repaired_target&~flags.has_repaired_any_input),
        common_window_hours=max(WINDOWS[profile]))
    return data


def select_windows(metrics):
    cv=metrics.loc[metrics.fold.isin(common.FOLDS)&metrics.scope.eq('full')&metrics.candidate.str.startswith('w')].copy()
    selected={}
    for profile,part in cv.groupby('profile',sort=False):
        counts=part.groupby('candidate').fold.agg(lambda s:set(s))
        if not counts.map(lambda s:s==set(common.FOLDS)).all() or part.duplicated(['candidate','fold']).any():
            raise ValueError('각 입력 창의4~7월 전체 CV 결과가 필요합니다.')
        score_columns=['Loss_A','L_peak','Loss_C']
        macro=part.groupby(['candidate','window_hours'])[score_columns].mean().reset_index()
        if not np.isfinite(macro[score_columns].to_numpy()).all(): raise ValueError('선정 점수가 유한하지 않습니다.')
        best=macro.sort_values(['Loss_A','window_hours'],kind='stable').iloc[0]
        selected[profile]=dict(candidate=str(best.candidate),window_hours=int(best.window_hours),
            cv_Loss_A=float(best.Loss_A),cv_L_peak=float(best.L_peak),cv_Loss_C=float(best.Loss_C))
    if not selected: raise ValueError('선정할 CV 결과가 없습니다.')
    return selected


def write_record(rec):
    c=rec['config']; spec=ml.PROFILES[c['profile']]
    table=common.md_table(pd.DataFrame(rec['metrics'])) if rec['metrics'] else '완료 평가 없음.'
    baseline=rec['window_hours'] is None
    feature_note=('모델 적합0회, 전력 반복 기준선, 외생 입력 없음' if baseline else
        f'F2의 평균·최대·std만 최근{rec["window_hours"]}시간으로 대체; std ddof=0; 통계3열 고정')
    rec['record'].write_text(f'''# 실험 {c['experiment_id']}

- 상태: {rec['status']}
- 목적과 가설: {c['profile']} 입력 통계 창 {rec['window_hours']}시간의 효과를 같은 표본에서 확인; {c['candidate']}.
- 시작·종료 시각 및 시간대: {rec['started']} / {rec.get('ended','미완료')}, Asia/Seoul.
- 모델·라이브러리 버전: Python{platform.python_version()}; {c['versions']}.
- 데이터 위치·버전 또는 해시·사용 범위: {c['source']}, SHA256 {c['source_sha256']}; 1~8월. 보정이력 대조 {c['legacy']}, SHA256 {c['legacy_sha256']}. 원본 미변경.
- 목표값·단위·예측 구간: power_15min kW, 미래{spec['horizon_steps']*15}min의15분 원경로, 갱신{spec['cadence_steps']*15}min.
- 학습·검증·테스트 분할과 누수 점검: 4~7월 expanding CV; 전체 학습 정답 끝<평가월 시작, 전체 평가 정답 끝<다음월. 모든 후보는 가장 긴 창을 포함한 공통 학습·평가 원점. 월 내 모델 고정. 선정은 CV만 사용,8월은 선정 창 및1시간 창 확인,9월 미실행.
- 전처리·특성·하이퍼파라미터·seed: {feature_note}; {c['params']}; scaler없음. native multioutput RF; 발행 후 미래 전력 재주입 없음. 미래 생산·기상 실측=오차0예보. 보정표식은 피처가 아님.
- 이전 또는 기준 실험 링크와 변경 사항: [기존 중장기 RF](../../notebooks/models/random_forest/MEDIUM_LONG_RESULTS_20261003.md), [창 비교 계획](../../notebooks/models/random_forest/INPUT_WINDOW_PLAN.md). 입력 통계 창만 비교하고 다른 RF 설정·손실 정의는 고정. 이번 동일지원1시간 창과 비교; 이전 숫자는 학습지원이 달라질 수 있음.
- 실행 명령·환경·코드 버전 또는 코드 식별 정보: run_input_window_notebook.py / rf_input_window_experiments.ipynb Run All; Git없는workspace. 엔진 SHA256 {c['code_sha256']}; 공통 및 보호파일 해시는suite_config.json.
- 실제 실행 시간·자원 사용량(측정한 경우): 완료/중단fold누적{rec['elapsed_seconds']:.3f}초, 완료RFfit{rec['n_completed_rf_fits']}회; CPU4workers,순차forest,최대5400초. fold_config에process_peak_rss_bytes(Windows측정). 부분forest는 점수로 쓰지 않음.
- 평가 지표·구간별 또는 fold별 결과: Loss A 전체 RMSE; 원점 내부W={spec['loss_steps']*15}min 블록 RMSE 평균 및 최악ceil20% 평균; L_peak 전체 유효셀 과소예측제곱 평균 후 마지막root,w=1; B=.6mean+.4CVaR,C=.5mean+.3CVaR+.2peak. RF squared_error 학습,설계손실은 평가만.
- 기준모델 대비 변화와 비교 가능 여부: 같은profile/fold/scope의 창 및 단순기준선과 비교. CV는4개월동일가중. 모든 창과 기준선의 보정 입력 참조를 포함한 공통 민감도 마스크.
- 오차가 큰 조건·해석·한계: 외생예보오차0,단일seed42,중첩원점 독립아님. 장기자정 학습사례 적음. 보정제외는표본변경이며 원참값복구 아님. 이미 확인한8월은최종독립테스트가 아님. 월/리드타임/요일/야간/피크 크기·시점 결과 별도 저장.
- 오류·경고·실패 또는 중단 원인: {rec.get('error','없음')}.
- 예측·모델·표·그래프·로그의 저장 경로: {rec['artifact']}; fold별train_origins.csv,origin_masks.csv,fold_config.json,predictions.csv.gz,RFmodel.joblib; metrics,horizon,condition,peak,lead_band CSV; run.log.
- 결론과 다음 실험: 아래는 실제 실행 값. 선정 여부는suite의selection.json. 선정되지 않은 창의8월은미실행. RF의 최적 창이 부스팅의 최적 창이라는 의미는 아님.

{table}
''',encoding='utf-8')
    (rec['artifact']/'status.json').write_text(json.dumps({k:rec.get(k) for k in ['status','started','ended','error','elapsed_seconds','n_completed_rf_fits']},ensure_ascii=False,indent=2),encoding='utf-8')


def new_record(suite,root,profile,candidate,window_hours,params,metadata):
    artifact=Path(suite)/profile/candidate; artifact.mkdir(parents=True,exist_ok=False)
    path=Path(root)/'experiments/random_forest'/f'{Path(suite).name}_{profile}_{candidate}.md'
    path.parent.mkdir(parents=True,exist_ok=True)
    cfg=dict(experiment_id=path.stem,profile=profile,candidate=candidate,window_hours=window_hours,params=params,
        code_sha256=common.digest(__file__),versions={p:version(p) for p in ['numpy','pandas','scikit-learn','joblib','nbclient']},**metadata)
    rec=dict(config=cfg,artifact=artifact,record=path,window_hours=window_hours,status='실행 중',
        started=datetime.now(common.KST).isoformat(),elapsed_seconds=0.,n_completed_rf_fits=0,
        metrics=[],horizons=[],conditions=[],peaks=[],bands=[])
    (artifact/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    write_record(rec)
    return rec


def evaluate_fold(rec,data,fold,deadline):
    c=rec['config']; spec=data['spec']; H=spec['horizon_steps']; w=rec['window_hours']
    folder=rec['artifact']/fold; folder.mkdir(exist_ok=False)
    origins=data['targets'].index[data['valid']]; train_idx=data['targets'].index[data['train']]
    actual=data['targets'].loc[origins].to_numpy()
    tags=data['tags'].reindex(origins).assign(**{f'keep_{s}':mask.reindex(origins) for s,mask in data['scopes'].items()})
    tags.to_csv(folder/'origin_masks.csv',index_label='origin',encoding='utf-8-sig')
    pd.DataFrame({'origin':train_idx}).to_csv(folder/'train_origins.csv',index=False,encoding='utf-8-sig')
    cfg=dict(profile=data['profile'],candidate=c['candidate'],fold=fold,window_hours=w,
        common_window_hours=data['common_window_hours'],n_train=len(train_idx) if w is not None else 0,
        n_train_comparison_support=len(train_idx),n_valid=len(origins),features=data['columns'] if w is not None else [],
        train_first=str(train_idx[0]),train_last=str(train_idx[-1]),train_label_max=str(train_idx[-1]+pd.Timedelta(minutes=H*15)),
        valid_first=str(origins[0]),valid_last=str(origins[-1]),valid_label_max=str(origins[-1]+pd.Timedelta(minutes=H*15)),
        start=str(data['start']),end=str(data['end']),spec=spec,scopes={s:int(m.sum()) for s,m in data['scopes'].items()})
    (folder/'fold_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    before=perf_counter()
    try:
        if perf_counter()>=deadline: raise TimeoutError('입력 창 비교 실행 예산 소진')
        if w is not None:
            x=data['features_by_window'][w].loc[train_idx].astype(np.float32); y=data['targets'].loc[train_idx]
            params=c['params']; total=params['n_estimators']
            model=RandomForestRegressor(**{**params,'n_estimators':min(50,total),'warm_start':True})
            for trees in range(min(50,total),total+50,50):
                if perf_counter()>=deadline: raise TimeoutError(f'예산 소진:{len(getattr(model,"estimators_",[]))}/{total}trees')
                model.set_params(n_estimators=min(trees,total)); model.fit(x,y)
                print(f'TREES {data["profile"]}/{c["candidate"]}/{fold}: {model.n_estimators}/{total}, {perf_counter()-before:.1f}s',flush=True)
                if model.n_estimators==total: break
            model.set_params(warm_start=False)
            predicted=model.predict(data['features_by_window'][w].loc[origins])
            joblib.dump(model,folder/'model.joblib',compress=3)
            rec['n_completed_rf_fits']+=1
            del model,x,y; gc.collect()
        else:
            predicted=ml.baseline(data['df'],origins,H,c['candidate'])
        target=origins.to_numpy()[:,None]+np.arange(1,H+1)[None,:]*np.timedelta64(15,'m')
        frame=pd.DataFrame(dict(origin=np.repeat(origins,H),target_end=target.ravel(),
            horizon_minutes=np.tile(np.arange(1,H+1)*15,len(origins)),actual=actual.ravel(),predicted=predicted.ravel()))
        frame['target_start']=frame.target_end-pd.Timedelta(minutes=15)
        frame['is_night']=((frame.target_start.dt.hour>=18)|(frame.target_start.dt.hour<9)).astype(int)
        frame['is_workday']=data['df'].day_type.reindex(frame.target_end).eq(0).to_numpy().astype(int)
        frame['actual_was_repaired']=data['repaired'].reindex(frame.target_end).to_numpy()
        frame.assign(profile=data['profile'],candidate=c['candidate'],window_hours=w,fold=fold,scenario=proxy.SCENARIO).to_csv(folder/'predictions.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
        for scope,mask in data['scopes'].items():
            keep=mask.reindex(origins).to_numpy(bool)
            if not keep.any(): raise ValueError(f'{scope}: 평가 원점 없음')
            a,p=actual[keep],predicted[keep]; part=frame.loc[np.repeat(keep,H)]
            tag=dict(profile=data['profile'],candidate=c['candidate'],window_hours=w,fold=fold,scope=scope)
            rec['metrics'].append(dict(**tag,n_train=cfg['n_train'],n_train_comparison_support=len(train_idx),
                n_excluded_origins=len(origins)-int(keep.sum()),n_unique_target_times=int(part.target_end.nunique()),**ml.losses(a,p,spec['loss_steps'])))
            for h in range(H):
                e=a[:,h]-p[:,h]
                rec['horizons'].append(dict(**tag,horizon_minutes=15*(h+1),n_origins=len(a),
                    Loss_A=float(np.sqrt(np.mean(e**2))),MAE=float(np.abs(e).mean()),L_peak=float(np.sqrt(np.mean(np.maximum(0,e)**2)))))
            for dimension,values in [('is_night',[0,1]),('is_workday',[0,1]),('weekday',range(7))]:
                for value in values:
                    chosen=part.loc[part.target_start.dt.dayofweek.eq(value) if dimension=='weekday' else part[dimension].eq(value)]
                    if len(chosen):
                        e=chosen.actual.to_numpy()-chosen.predicted.to_numpy()
                        rec['conditions'].append(dict(**tag,dimension=dimension,condition=int(value),n_predictions=len(chosen),
                            n_unique_target_times=int(chosen.target_end.nunique()),Loss_A=float(np.sqrt(np.mean(e**2))),
                            MAE=float(np.abs(e).mean()),L_peak=float(np.sqrt(np.mean(np.maximum(0,e)**2))),mean_signed_error=float(-e.mean())))
            for grouping,steps in [('path',H),('loss_window',spec['loss_steps'])]:
                rec['peaks'].append(dict(**tag,grouping=grouping,window_minutes=steps*15,**ml.peak_scores(a.reshape(-1,steps),p.reshape(-1,steps))))
            chunk=96 if data['profile']=='long_3d' else 16
            for begin in range(0,H,chunk):
                e=a[:,begin:min(H,begin+chunk)]-p[:,begin:min(H,begin+chunk)]
                rec['bands'].append(dict(**tag,lead_start_minutes=15*begin,lead_end_minutes=15*min(H,begin+chunk),
                    n_predictions=e.size,Loss_A=float(np.sqrt(np.mean(e**2))),MAE=float(np.abs(e).mean()),L_peak=float(np.sqrt(np.mean(np.maximum(0,e)**2)))))
        try:
            import psutil
            cfg['process_peak_rss_bytes']=psutil.Process().memory_info().peak_wset
        except (ImportError,AttributeError): cfg['process_peak_rss_bytes']=None
        cfg['elapsed_seconds']=perf_counter()-before
        (folder/'fold_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
        for name,key in [('metrics','metrics'),('horizon_metrics','horizons'),('condition_metrics','conditions'),('peak_path_metrics','peaks'),('lead_band_metrics','bands')]:
            pd.DataFrame(rec[key]).to_csv(rec['artifact']/f'{name}.csv',index=False,encoding='utf-8-sig')
    except Exception as exc:
        rec.update(status='중단' if isinstance(exc,TimeoutError) else '실패',error=f'{type(exc).__name__}: {exc}',ended=datetime.now(common.KST).isoformat())
        rec['elapsed_seconds']+=perf_counter()-before; write_record(rec)
        with (rec['artifact']/'run.log').open('a',encoding='utf-8') as log:log.write(f'{fold}: {rec["error"]}\n')
        raise
    rec['elapsed_seconds']+=perf_counter()-before; write_record(rec)
    with (rec['artifact']/'run.log').open('a',encoding='utf-8') as log:log.write(f'{fold}: 완료,{perf_counter()-before:.3f}s,n_train={cfg["n_train"]},n_valid={len(origins)}\n')
    row=rec['metrics'][-len(data['scopes'])]
    print(f'DONE {data["profile"]}/{c["candidate"]}/{fold}: RMSE={row["Loss_A"]:.4f}, peak={row["L_peak"]:.4f}, {perf_counter()-before:.1f}s',flush=True)


def run_experiments(budget_seconds=5400):
    before=perf_counter(); deadline=before+budget_seconds
    suite=ROOT/'artifacts/random_forest/runs'/(datetime.now(common.KST).strftime('%Y%m%dT%H%M%S_KST')+'_'+uuid4().hex[:6]+'_rf_input_windows')
    suite.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((august.SELECTION/'selected_config.json').read_text(encoding='utf-8'))
    for key,path in [('source_sha256',common.SOURCE),('shared_code_sha256',common.__file__),('proxy_code_sha256',proxy.__file__),('code_sha256',grid.__file__)]:
        if common.digest(path)!=frozen[key]: raise ValueError(f'動結파일불일치:{key}')
    df,repaired=august.load_august_data()
    metadata=dict(source=str(common.SOURCE),source_sha256=common.digest(common.SOURCE),legacy=str(august.LEGACY),legacy_sha256=common.digest(august.LEGACY))
    cfg=dict(suite_id=suite.name,status='실행 중',started=datetime.now(common.KST).isoformat(),budget_seconds=budget_seconds,
        windows=WINDOWS,profiles=ml.PROFILES,periods=ml.PERIODS,selected_params=frozen['rf_params'],
        n_planned_rf_fits_max=56,n_cv_rf_fits=48,september_evaluated=False,tuning_after_august=False,
        selection_rule='4~7월 full 월별동일가중 Loss_A 최소,동률짧은창',scenario=proxy.SCENARIO,**metadata,
        code_sha256=common.digest(__file__),protected_code_sha256={Path(p).name:common.digest(p) for p in [common.__file__,grid.__file__,proxy.__file__,august.__file__,ml.__file__]})
    (suite/'suite_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    records=[]
    for profile,windows in WINDOWS.items():
        records.extend(new_record(suite,ROOT,profile,f'w{w:03d}h',w,frozen['rf_params'],metadata) for w in windows)
        records.extend(new_record(suite,ROOT,profile,name,None,{},metadata) for name in ml.MODEL_NAMES[1:])
    print(f'SUITE {suite}',flush=True)
    selected={}
    try:
        for profile in WINDOWS:
            for fold,(start,end) in common.FOLDS.items():
                data=prepare_profile(df,repaired,profile,start,end)
                print(f'SPLIT {profile}/{fold}: train={int(data["train"].sum())}, valid={int(data["valid"].sum())}, columns={len(data["columns"])}',flush=True)
                for rec in records:
                    if rec['config']['profile']==profile:evaluate_fold(rec,data,fold,deadline)
                del data;gc.collect()
        metrics=pd.DataFrame([row for rec in records for row in rec['metrics']])
        metrics.to_csv(suite/'cv_before_august.csv',index=False,encoding='utf-8-sig')
        selected=select_windows(metrics)
        decision=dict(selected=selected,selected_at=datetime.now(common.KST).isoformat(),selection_rule=cfg['selection_rule'],
            cv_source_sha256=common.digest(suite/'cv_before_august.csv'),august_used_for_selection=False,september_used_for_selection=False)
        (suite/'selection.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
        cfg['selection_sha256']=common.digest(suite/'selection.json')
        print(f'CV_SELECTED {json.dumps(selected,ensure_ascii=False)}',flush=True)
        for profile in WINDOWS:
            data=prepare_profile(df,repaired,profile,*ml.PERIODS['aug'])
            allowed={selected[profile]['candidate'],'w001h',*ml.MODEL_NAMES[1:]}
            for rec in records:
                if rec['config']['profile']==profile and rec['config']['candidate'] in allowed:evaluate_fold(rec,data,'aug',deadline)
            del data;gc.collect()
        for rec in records:
            rec.update(status='완료',ended=datetime.now(common.KST).isoformat());write_record(rec)
        cfg['status']='완료'
    except Exception as exc:
        cfg.update(status='중단' if isinstance(exc,TimeoutError) else '실패',error=f'{type(exc).__name__}: {exc}')
        for rec in records:
            if rec['status']=='실행 중':rec.update(status='중단',ended=datetime.now(common.KST).isoformat(),error=cfg['error']);write_record(rec)
        raise
    finally:
        metrics=pd.DataFrame([row for rec in records for row in rec['metrics']])
        metrics.to_csv(suite/'comparison_scopes.csv',index=False,encoding='utf-8-sig')
        for name,key in [('comparison_horizons','horizons'),('comparison_conditions','conditions'),('comparison_peak_paths','peaks'),('comparison_lead_bands','bands')]:
            pd.DataFrame([row for rec in records for row in rec[key]]).to_csv(suite/f'{name}.csv',index=False,encoding='utf-8-sig')
        cfg.update(ended=datetime.now(common.KST).isoformat(),elapsed_seconds=perf_counter()-before,
            n_completed_rf_fits=sum(r['n_completed_rf_fits'] for r in records),n_candidate_records=len(records),selected=selected)
        (suite/'suite_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
        with (ROOT/'experiments/README.md').open('a',encoding='utf-8') as stream:
            stream.write(f'\n## {suite.name}: RF 구간별 입력 통계 윈도우\n\n상태:{cfg["status"]}, 4~7월선정·8월확인·9월보존. 창별공통학습원점,RF·손실고정.\n\n')
            for rec in records:stream.write(f'- [{rec["config"]["profile"]}/{rec["config"]["candidate"]}]({rec["record"].relative_to(ROOT/"experiments").as_posix()}): {rec["status"]}\n')
    return dict(output_dir=suite,records=records,metrics=metrics,config=cfg,selected=selected)


if __name__=='__main__':run_experiments()
