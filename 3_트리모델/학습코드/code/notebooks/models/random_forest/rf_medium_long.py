"""동결 RF의 중기4/8/12h·장기3d. 원경로 유지,9월 보존,후보별 기록."""
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

ROOT=common.ROOT
PROFILES={
    'medium_4h':dict(horizon_steps=16,cadence_steps=4,loss_steps=4),
    'medium_8h':dict(horizon_steps=32,cadence_steps=4,loss_steps=4),
    'medium_12h':dict(horizon_steps=48,cadence_steps=4,loss_steps=4),
    'long_3d':dict(horizon_steps=288,cadence_steps=96,loss_steps=96),
}
PERIODS={**common.FOLDS,'aug':('2021-08-01','2021-09-01')}
SCOPES=august.SCOPES
MODEL_NAMES=['RF_selected','persistence','daily','weekly']


def build_features(df,horizon,origins=None):
    common.validate_frame(df)
    if not isinstance(horizon,int) or horizon<1 or horizon>384:
        raise ValueError('horizon은1~384개의15분 구간이어야 합니다.')
    idx=df.index if origins is None else pd.DatetimeIndex(origins)
    if not idx.isin(df.index).all(): raise ValueError('입력 원점이 데이터 시간축에 없습니다.')
    columns={}; quarters={}; forecasts={}; targets={}
    for h in range(1,horizon+1):
        start=idx+pd.Timedelta(minutes=15*(h-1))
        for day in range(1,8):
            columns[f'h{h}_day_{day}']=(start.dayofweek+1==day).astype(float)
        for name,values in [('month',start.month),('night',(start.hour>=18)|(start.hour<9)),
                            ('workday',df.day_type.shift(-h).reindex(idx).eq(0))]:
            columns[f'h{h}_{name}']=np.asarray(values,dtype=float)
        quarters[f'h{h}_quarter']=start.hour*4+start.minute//15
        targets[f'h{h}']=df.power_15min.astype(float).shift(-h).reindex(idx).to_numpy()
        for col in common.OBSERVED:
            forecasts[f'h{h}_forecast_proxy_{col}']=df[col].shift(-h).reindex(idx).to_numpy()
    columns.update(quarters)
    power=df.power_15min.astype(float)
    for name,offset in [('latest',0),('lag_1h',4),('lag_24h',96)]:
        columns[f'power_{name}']=power.shift(offset).reindex(idx).to_numpy()
    rolling=power.rolling(4)
    for stat in ['mean','max','std']:
        series=rolling.std(ddof=0) if stat=='std' else getattr(rolling,stat)()
        columns[f'power_1h_{stat}']=series.reindex(idx).to_numpy()
    columns.update(forecasts)
    return pd.DataFrame(columns,index=idx),pd.DataFrame(targets,index=idx),list(columns)


def baseline(df,origins,horizon,kind):
    origins=pd.DatetimeIndex(origins)
    if kind=='persistence':
        return np.repeat(df.power_15min.reindex(origins).to_numpy()[:,None],horizon,axis=1)
    if kind not in ['daily','weekly']: raise ValueError(f'알 수 없는 기준선: {kind}')
    period=96 if kind=='daily' else 672
    offsets=(np.arange(horizon)%period)+1-period
    # h>1d인 daily도 원점 이전에 완료된 마지막 하루만 반복한다.
    stamps=origins.to_numpy()[:,None]+offsets[None,:]*np.timedelta64(15,'m')
    return df.power_15min.reindex(pd.DatetimeIndex(stamps.ravel())).to_numpy().reshape(len(origins),horizon)


def prepare_profile(df,repaired,profile,start,end):
    spec=PROFILES[profile]; horizon=spec['horizon_steps']
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if end>august.END: raise ValueError('9월은 최종 테스트로 보존합니다.')
    df=df.loc[df.index<end].copy()
    repaired=repaired.reindex(df.index)
    if repaired.isna().any(): raise ValueError('보정 표식 시점 대응 누락')
    repaired=repaired.astype(bool)
    cadence=(df.index.hour*60+df.index.minute)%(spec['cadence_steps']*15)==0
    origins=df.index[cadence]
    features,targets,columns=build_features(df,horizon,origins)
    complete=np.isfinite(features.to_numpy()).all(axis=1)&np.isfinite(targets.to_numpy()).all(axis=1)
    train=pd.Series(complete&(origins+pd.Timedelta(minutes=15*horizon)<start),index=origins)
    valid=pd.Series(complete&(origins>=start)&(origins+pd.Timedelta(minutes=15*horizon)<end),index=origins)
    for kind in MODEL_NAMES[1:]:
        valid &= np.isfinite(baseline(df,origins,horizon,kind)).all(axis=1)
    if not train.any() or not valid.any(): raise ValueError(f'{profile}: 학습/평가 원점 없음')
    target=pd.Series(False,index=df.index)
    daily=pd.Series(False,index=df.index); weekly=pd.Series(False,index=df.index)
    for h in range(1,horizon+1):
        target |= repaired.shift(-h,fill_value=False)
    for period,tags in [(96,daily),(672,weekly)]:
        for offset in set(period-(np.arange(horizon)%period+1)):
            tags |= repaired.shift(int(offset),fill_value=False)
    rf=repaired|repaired.shift(4,fill_value=False)|repaired.shift(96,fill_value=False)
    rf |= repaired.rolling(4,min_periods=1).max().astype(bool)
    tags=pd.DataFrame(dict(has_repaired_target=target,has_repaired_rf_input=rf,
        has_repaired_daily_input=daily,has_repaired_weekly_input=weekly,
        has_repaired_any_input=rf|daily|weekly),index=df.index)
    flags=tags.reindex(origins)
    scopes=dict(full=valid,clean_targets=valid&~flags.has_repaired_target,
                clean_targets_inputs=valid&~flags.has_repaired_target&~flags.has_repaired_any_input)
    return dict(df=df,repaired=repaired,features=features,targets=targets,columns=columns,
                train=train,valid=valid,scopes=scopes,tags=tags,profile=profile,spec=spec,start=start,end=end)


def losses(actual,predicted,window_steps):
    a,p=np.asarray(actual,float),np.asarray(predicted,float)
    if a.ndim!=2 or a.shape!=p.shape or not len(a) or not isinstance(window_steps,int) or window_steps<1 or a.shape[1]%window_steps:
        raise ValueError('같은 크기 원경로와 길이를 나누는 양의 정수 손실 window가 필요합니다.')
    result=common.risk_scores(a.reshape(-1,window_steps),p.reshape(-1,window_steps))
    result.update(n_origins=len(a),n_predictions=a.size,n_loss_windows=a.size//window_steps)
    return result


def peak_scores(a,p):
    unique=((a==a.max(axis=1)[:,None]).sum(axis=1)==1)&((p==p.max(axis=1)[:,None]).sum(axis=1)==1)
    return dict(n_paths=len(a),max_MAE=float(np.abs(a.max(axis=1)-p.max(axis=1)).mean()),
        max_under_RMSE=float(np.sqrt(np.mean(np.maximum(0,a.max(axis=1)-p.max(axis=1))**2))),
        first_argmax_timing_MAE_minutes=float(np.abs(a.argmax(axis=1)-p.argmax(axis=1)).mean()*15),
        actual_max_ties=int(((a==a.max(axis=1)[:,None]).sum(axis=1)>1).sum()),
        n_unique_max_timing=int(unique.sum()),
        unique_max_timing_MAE_minutes=float(np.abs(a[unique].argmax(axis=1)-p[unique].argmax(axis=1)).mean()*15) if unique.any() else np.nan)


def write_record(rec):
    cfg=rec['config']; spec=PROFILES[cfg['profile']]
    table=common.md_table(pd.DataFrame(rec['metrics'])) if rec['metrics'] else '완료 평가 없음.'
    rec['record'].write_text(f'''# 실험 {cfg['experiment_id']}

- 상태: {rec['status']}
- 목적과 가설: 단기 동결 RF 설정을 {cfg['profile']}로 전이해 단순 기준선 대비 성능 측정. 별도 구간 최적 튜닝 없음.
- 시작·종료 시각 및 시간대: {rec['started']} / {rec.get('ended','미완료')}; Asia/Seoul.
- 모델·라이브러리 버전: {cfg['candidate']}; Python{platform.python_version()}; {cfg['versions']}.
- 데이터 위치·버전 또는 해시·사용 범위: {cfg['source']}; SHA256 {cfg['source_sha256']}; 1~8월. 기존 보정대조 {cfg['legacy']}; SHA256 {cfg['legacy_sha256']}.
- 목표값·단위·예측 구간: power_15min kW; 미래{spec['horizon_steps']*15}min,15분{spec['horizon_steps']}점; 원점 갱신{spec['cadence_steps']*15}min.
- 학습·검증·테스트 분할과 누수 점검: 4~7월 expanding,1~7월 재학습·8월 확인. 모든 학습 target_end<평가 시작, 모든 평가 target_end<다음월. 월 내 모델 고정. 9월 모델 점수 없음. 3d daily는 관측 완료 하루 반복. 학습 원점도 평가와 같은 cadence.
- 전처리·특성·하이퍼파라미터·seed: 미래 외생 오차0 F2 확장,최근1h 통계 유지,scaler 없음,native multioutput; {cfg['params']}. full W 입력열 목록·시간 범위는 fold_config.json. 보정 표식은 입력 아님.
- 이전 또는 기준 실험 링크와 변경 사항: [동결 설정](../../notebooks/models/random_forest/BASELINE_GRID_RESULTS_20261003.md), [중장기 계획](../../notebooks/models/random_forest/MEDIUM_LONG_PLAN.md). H와기존cadence/W 확장; 추가 피처·파라미터 탐색0회. 단기에서 선정한 설정의 전이이며 중장기 최적 설정을 의미하지 않음.
- 실행 명령·환경·코드 버전 또는 코드 식별 정보: run_medium_long_notebook.py 또는 rf_medium_long_experiments.ipynb Run All. Git 없는workspace; 엔진 SHA256 {cfg['code_sha256']}, 공통 {cfg['shared_code_sha256']}; 실행 환경·protected hashes는 suite_config.json.
- 실제 실행 시간·자원 사용량(측정한 경우): {rec.get('elapsed_seconds','미완료')}초(완료 fold 누적); worker4,순차 forest,총1800s 예산; 최대 RSS는fold_config.json(측정). 예산 점검50tree,완료는500tree 모두 사용.
- 평가 지표·구간별 또는 fold별 결과: Loss A 전체 RMSE; W={spec['loss_steps']*15}min를 원점 내부에서 나눔. mean 및 최악ceil20% CVaR는fold의 전체블록; L_peak=전체 유효셀의과소예측제곱평균후root(w=1), B=.6mean+.4CVaR, C=.5mean+.3CVaR+.2peak. RF squared_error 학습, B/C는평가만.
- 기준모델 대비 변화와 비교 가능 여부: 동일 profile/fold/민감도 마스크에서4모델 비교. 다른H/W는 단일순위로합치지 않음. CV 평균은4개월동일가중.
- 오차가 큰 조건·해석·한계: 미래 생산·기상 실측을 예보로 간주한 오차0 조건. 원점중첩,장기자정 학습사례 적음,고유시점/표본수 병기. 보정제외는 다른표본이며원참값복구 아님. peak_path_metrics.csv는원경로/손실window 최대·시점 오차를분리.
- 오류·경고·실패 또는 중단 원인: {rec.get('error','없음')}.
- 예측·모델·표·그래프·로그의 저장 경로: {rec['artifact']}; fold별 predictions.csv.gz,model.joblib(RF),origin_masks.csv,fold_config.json; metrics.csv,horizon_metrics.csv,condition_metrics.csv,peak_path_metrics.csv,run.log.
- 결론과 다음 실험: 실제 값은 아래. 8월 추가튜닝 없음. 후속부스팅/피처비교는같은구간계약 사용;9월은모든설정동결후최종테스트.

{table}
''',encoding='utf-8')
    (rec['artifact']/'status.json').write_text(json.dumps({k:rec.get(k) for k in ['status','started','ended','error','elapsed_seconds','n_completed_rf_fits']},ensure_ascii=False,indent=2),encoding='utf-8')


def new_record(suite,root,profile,candidate,params,metadata):
    artifact=Path(suite)/profile/candidate
    artifact.mkdir(parents=True,exist_ok=False)
    record=Path(root)/'experiments/random_forest'/f'{Path(suite).name}_{profile}_{candidate}.md'
    record.parent.mkdir(parents=True,exist_ok=True)
    cfg=dict(experiment_id=record.stem,profile=profile,candidate=candidate,params=params,**metadata,
        code_sha256=common.digest(__file__),shared_code_sha256=common.digest(common.__file__),
        versions={p:version(p) for p in ['numpy','pandas','scikit-learn','joblib','nbclient']},scenario=proxy.SCENARIO)
    rec=dict(config=cfg,artifact=artifact,record=record,status='실행 중',started=datetime.now(common.KST).isoformat(),
             elapsed_seconds=0.,metrics=[],horizons=[],conditions=[],peaks=[],n_completed_rf_fits=0)
    (artifact/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    write_record(rec)
    return rec


def evaluate_fold(rec,data,fold,deadline=None):
    spec=data['spec']; H=spec['horizon_steps']; candidate=rec['config']['candidate']
    folder=rec['artifact']/fold; folder.mkdir(exist_ok=False)
    origins=data['features'].index[data['valid']]
    actual=data['targets'].loc[data['valid']].to_numpy()
    mask_frame=data['tags'].reindex(origins).assign(**{f'keep_{scope}':mask.reindex(origins) for scope,mask in data['scopes'].items()})
    mask_frame.to_csv(folder/'origin_masks.csv',index_label='origin',encoding='utf-8-sig')
    train_idx=data['features'].index[data['train']]
    fold_config=dict(profile=data['profile'],fold=fold,n_train=len(train_idx),n_valid=len(origins),features=data['columns'],
        train_first=str(train_idx[0]),train_last=str(train_idx[-1]),train_label_max=str(train_idx[-1]+pd.Timedelta(minutes=15*H)),
        valid_first=str(origins[0]),valid_last=str(origins[-1]),valid_label_max=str(origins[-1]+pd.Timedelta(minutes=15*H)),
        start=str(data['start']),end=str(data['end']),spec=spec,scopes={scope:int(mask.sum()) for scope,mask in data['scopes'].items()})
    (folder/'fold_config.json').write_text(json.dumps(fold_config,ensure_ascii=False,indent=2),encoding='utf-8')
    before=perf_counter(); rows=[]
    try:
        if deadline is not None and perf_counter()>=deadline: raise TimeoutError('중장기 RF 총 실행 예산 소진')
        if candidate=='RF_selected':
            params=rec['config']['params']; total=params['n_estimators']
            model=RandomForestRegressor(**{**params,'n_estimators':min(50,total),'warm_start':True})
            x=data['features'].loc[data['train']].astype(np.float32)
            y=data['targets'].loc[data['train']]
            for trees in range(min(50,total),total+50,50):
                if deadline is not None and perf_counter()>=deadline: raise TimeoutError(f'예산 소진: {len(getattr(model,"estimators_",[]))}/{total}trees')
                model.set_params(n_estimators=min(trees,total))
                model.fit(x,y)
                print(f'TREES {data["profile"]}/{fold}: {model.n_estimators}/{total}, {perf_counter()-before:.1f}s',flush=True)
                if model.n_estimators==total: break
            model.set_params(warm_start=False)
            predicted=model.predict(data['features'].loc[data['valid']])
            joblib.dump(model,folder/'model.joblib',compress=3)
            rec['n_completed_rf_fits']+=1
            del model,x,y
            gc.collect()
        else:
            predicted=baseline(data['df'],origins,H,candidate)
        target_end=origins.to_numpy()[:,None]+np.arange(1,H+1)[None,:]*np.timedelta64(15,'m')
        frame=pd.DataFrame(dict(origin=np.repeat(origins,H),target_end=target_end.ravel(),
            horizon_minutes=np.tile(np.arange(1,H+1)*15,len(origins)),actual=actual.ravel(),predicted=predicted.ravel()))
        frame['target_start']=frame.target_end-pd.Timedelta(minutes=15)
        frame['is_night']=((frame.target_start.dt.hour>=18)|(frame.target_start.dt.hour<9)).astype(int)
        frame['is_workday']=data['df'].day_type.reindex(frame.target_end).eq(0).to_numpy().astype(int)
        frame['actual_was_repaired']=data['repaired'].reindex(frame.target_end).to_numpy()
        frame.assign(profile=data['profile'],candidate=candidate,fold=fold,scenario=proxy.SCENARIO).to_csv(folder/'predictions.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
        for scope,mask in data['scopes'].items():
            keep=mask.reindex(origins).to_numpy(bool)
            if not keep.any(): raise ValueError(f'{scope}: 유효 평가 원점 없음')
            a,p=actual[keep],predicted[keep]; part=frame.loc[np.repeat(keep,H)]
            tag=dict(profile=data['profile'],candidate=candidate,fold=fold,scope=scope)
            row=dict(**tag,n_train=len(train_idx),n_excluded_origins=len(origins)-int(keep.sum()),
                     n_unique_target_times=int(part.target_end.nunique()),**losses(a,p,spec['loss_steps']))
            rows.append(row); rec['metrics'].append(row)
            for h in range(H):
                rec['horizons'].append(dict(**tag,horizon_minutes=15*(h+1),n_origins=len(a),
                    Loss_A=float(np.sqrt(np.mean((a[:,h]-p[:,h])**2))),MAE=float(np.abs(a[:,h]-p[:,h]).mean()),
                    L_peak=float(np.sqrt(np.mean(np.maximum(0,a[:,h]-p[:,h])**2)))))
            for dimension,values in [('is_night',[0,1]),('is_workday',[0,1]),('weekday',range(7))]:
                for value in values:
                    chosen=part.loc[part.target_start.dt.dayofweek.eq(value) if dimension=='weekday' else part[dimension].eq(value)]
                    if len(chosen):
                        error=chosen.actual.to_numpy()-chosen.predicted.to_numpy()
                        rec['conditions'].append(dict(**tag,dimension=dimension,condition=int(value),n_predictions=len(chosen),
                            n_unique_target_times=int(chosen.target_end.nunique()),Loss_A=float(np.sqrt(np.mean(error**2))),
                            MAE=float(np.abs(error).mean()),L_peak=float(np.sqrt(np.mean(np.maximum(0,error)**2))),
                            mean_signed_error=float(-error.mean())))
            for grouping,steps in [('path',H),('loss_window',spec['loss_steps'])]:
                rec['peaks'].append(dict(**tag,grouping=grouping,window_minutes=steps*15,
                    **peak_scores(a.reshape(-1,steps),p.reshape(-1,steps))))
        try:
            import psutil
            fold_config['process_peak_rss_bytes']=psutil.Process().memory_info().peak_wset
        except (ImportError,AttributeError): fold_config['process_peak_rss_bytes']=None
        fold_config['elapsed_seconds']=perf_counter()-before
        (folder/'fold_config.json').write_text(json.dumps(fold_config,ensure_ascii=False,indent=2),encoding='utf-8')
        for name,key in [('metrics','metrics'),('horizon_metrics','horizons'),('condition_metrics','conditions'),('peak_path_metrics','peaks')]:
            pd.DataFrame(rec[key]).to_csv(rec['artifact']/f'{name}.csv',index=False,encoding='utf-8-sig')
    except Exception as exc:
        rec.update(status='중단' if isinstance(exc,TimeoutError) else '실패',error=f'{type(exc).__name__}: {exc}',ended=datetime.now(common.KST).isoformat())
        rec['elapsed_seconds']+=perf_counter()-before
        write_record(rec)
        with (rec['artifact']/'run.log').open('a',encoding='utf-8') as log: log.write(f'{fold}: {rec["error"]}\n')
        raise
    rec['elapsed_seconds']+=perf_counter()-before
    write_record(rec)
    with (rec['artifact']/'run.log').open('a',encoding='utf-8') as log:
        log.write(f'{fold}: 완료, {perf_counter()-before:.3f}s, n_train={len(train_idx)}, n_valid={len(origins)}\n')
    print(f'DONE {data["profile"]}/{fold}/{candidate}: RMSE={rows[0]["Loss_A"]:.4f}, peak={rows[0]["L_peak"]:.4f}, {perf_counter()-before:.1f}s',flush=True)
    return rows


def run_experiments(budget_seconds=1800):
    before=perf_counter()
    suite=ROOT/'artifacts/random_forest/runs'/(datetime.now(common.KST).strftime('%Y%m%dT%H%M%S_KST')+'_'+uuid4().hex[:6]+'_rf_medium_long')
    suite.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((august.SELECTION/'selected_config.json').read_text(encoding='utf-8'))
    for key,path in [('source_sha256',common.SOURCE),('shared_code_sha256',common.__file__),('proxy_code_sha256',proxy.__file__),('code_sha256',grid.__file__)]:
        if common.digest(path)!=frozen[key]: raise ValueError(f'동결 파일과 불일치: {key}')
    df,repaired=august.load_august_data()
    metadata=dict(source=str(common.SOURCE),source_sha256=common.digest(common.SOURCE),legacy=str(august.LEGACY),legacy_sha256=common.digest(august.LEGACY))
    cfg=dict(suite_id=suite.name,status='실행 중',started=datetime.now(common.KST).isoformat(),profiles=PROFILES,periods=PERIODS,
        budget_seconds=budget_seconds,n_planned_rf_fits=20,selected_params=frozen['rf_params'],selected_reference=str(august.SELECTION/'selected_config.json'),
        september_evaluated=False,tuning_after_august=False,scenario=proxy.SCENARIO,**metadata,
        code_sha256=common.digest(__file__),protected_code_sha256={Path(p).name:common.digest(p) for p in [common.__file__,grid.__file__,proxy.__file__,august.__file__]})
    (suite/'suite_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    records=[new_record(suite,ROOT,profile,model,frozen['rf_params'] if model=='RF_selected' else {},metadata) for profile in PROFILES for model in MODEL_NAMES]
    print(f'SUITE {suite}',flush=True)
    try:
        for profile in PROFILES:
            for fold,(start,end) in PERIODS.items():
                print(f'PREPARE {profile}/{fold}',flush=True)
                data=prepare_profile(df,repaired,profile,start,end)
                print(f'SPLIT {profile}/{fold}: train={int(data["train"].sum())}, valid={int(data["valid"].sum())}, columns={len(data["columns"])}',flush=True)
                for rec in records:
                    if rec['config']['profile']==profile: evaluate_fold(rec,data,fold,before+budget_seconds)
                del data
                gc.collect()
            for rec in records:
                if rec['config']['profile']==profile:
                    rec.update(status='완료',ended=datetime.now(common.KST).isoformat()); write_record(rec)
        cfg['status']='완료'
    except Exception as exc:
        cfg.update(status='중단' if isinstance(exc,TimeoutError) else '실패',error=f'{type(exc).__name__}: {exc}')
        for rec in records:
            if rec['status']=='실행 중':
                rec.update(status='중단',ended=datetime.now(common.KST).isoformat(),error=f'전체 실행 종료: {cfg["error"]}'); write_record(rec)
        raise
    finally:
        metrics=pd.DataFrame([row for rec in records for row in rec['metrics']])
        metrics.to_csv(suite/'comparison_scopes.csv',index=False,encoding='utf-8-sig')
        cfg.update(ended=datetime.now(common.KST).isoformat(),elapsed_seconds=perf_counter()-before,
            n_completed_rf_fits=sum(r['n_completed_rf_fits'] for r in records),n_candidate_records=len(records))
        (suite/'suite_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
        with (ROOT/'experiments/README.md').open('a',encoding='utf-8') as stream:
            stream.write(f'\n## {suite.name}: RF 중기·장기 동결 기준\n\n4~7월 검증·8월 확인,9월 보존,추가튜닝 없음。상태: {cfg["status"]}\n\n')
            for rec in records:
                stream.write(f'- [{rec["config"]["profile"]}/{rec["config"]["candidate"]}]({rec["record"].relative_to(ROOT/"experiments").as_posix()}): {rec["status"]}\n')
            if len(metrics):
                macro=metrics.loc[metrics.fold.ne('aug')&metrics.scope.eq('full')].groupby(['profile','candidate'])[grid.METRICS].mean().reset_index()
                stream.write('\n4~7월 동일 월가중 평균(kW):\n\n'+common.md_table(macro)+'\n')
                stream.write('\n8월 전체(kW):\n\n'+common.md_table(metrics.loc[metrics.fold.eq('aug')&metrics.scope.eq('full'),['profile','candidate','n_origins','Loss_A','L_peak','Loss_C']])+'\n')
    return dict(output_dir=suite,records=records,metrics=metrics,config=cfg)


if __name__=='__main__': run_experiments()
