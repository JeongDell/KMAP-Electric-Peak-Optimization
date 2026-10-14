"""전체 q90 초기 피크, 시간순 5-fold, 확률 연계 RF 가중치 실험.

전체 q90은 사용자 지정 탐색 기준이며 9월 독립 holdout을 주장하지 않는다.
모델·가중치·확률 보정 선택은 4~8월만 사용한다. 과거 결과를 덮어쓰지 않는다.
"""
from pathlib import Path
from datetime import datetime
from time import perf_counter
from uuid import uuid4
from importlib.metadata import version
import argparse
import gc
import json
import platform
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import rf_current_data as common
import rf_grid_search as grid
import rf_input_windows as iw
import rf_medium_long as ml
import rf_september_final as final
import rf_peak_distribution as risk

ROOT=common.ROOT
PROFILES={'short_1h':dict(horizon_steps=4,cadence_steps=1,loss_steps=4),**ml.PROFILES}
CV_FOLDS=[f'm{m:02}' for m in range(4,9)]
CAPS=[1,2,4]


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def log(suite,message):
    line=f'{datetime.now(common.KST).isoformat()} {message}'
    print(line,flush=True)
    with (Path(suite)/'run.log').open('a',encoding='utf-8') as stream:stream.write(line+'\n')


def build_profile(df,repair,profile,window,initial,mode):
    spec=PROFILES[profile];h=spec['horizon_steps']
    if profile=='short_1h':
        x,y,cols=grid.build_window_features(df,window);x=x[cols]
    else:
        cadence=(df.index.hour*60+df.index.minute)%(spec['cadence_steps']*15)==0
        x,y,cols=iw.build_features(df,h,window,df.index[cadence])
    b=risk.thresholds(df.power_15min,initial,mode).reindex(x.index)
    # Both additions are known at the origin and common to all cap candidates.
    x=x.assign(peak_threshold_origin=b,power_latest_to_peak=x.power_latest/b)
    complete=np.isfinite(x.to_numpy(float)).all(axis=1)&np.isfinite(y.to_numpy(float)).all(axis=1)
    x=x.loc[complete];y=y.loc[complete];b=b.loc[complete]
    origins=x.index
    stamps=origins.to_numpy()[:,None]+np.arange(1,h+1)[None,:]*np.timedelta64(15,'m')
    target_repair=repair.reindex(pd.DatetimeIndex(stamps.ravel())).to_numpy(bool).reshape(-1,h).any(axis=1)
    input_repair=repair|repair.shift(4,fill_value=False)|repair.shift(96,fill_value=False)
    input_repair|=repair.rolling(window*4,min_periods=1).max().astype(bool)
    input_repair=input_repair.reindex(origins).to_numpy(bool)
    return dict(profile=profile,spec=spec,window=window,features=x,targets=y,origins=origins,
                threshold=b,has_repaired_target=target_repair,has_repaired_input=input_repair,
                initial=initial,threshold_mode=mode)


def build_query_features(df,origins,profile,window,initial,mode):
    """관측 과거 전력과 알려진 미래 외생·달력으로 추론 입력 구성. 미래 전력은 불필요."""
    origins=pd.DatetimeIndex(origins)
    if not len(origins) or not origins.isin(df.index).all():raise ValueError('데이터에 존재하는 예측 원점 필요')
    if profile=='short_1h':
        x,_,cols=grid.build_window_features(df,window);x=x.loc[origins,cols]
    else:x,_,_=iw.build_features(df,PROFILES[profile]['horizon_steps'],window,origins)
    past=df.power_15min.loc[df.index<=origins.max()]
    b=risk.thresholds(past,initial,mode).reindex(origins)
    x=x.assign(peak_threshold_origin=b,power_latest_to_peak=x.power_latest/b)
    if not np.isfinite(x.to_numpy(float)).all():raise ValueError('추론 과거 이력/미래 외생 입력이 부족합니다.')
    return x,b


def forecast_from_bundle(bundle,features,b):
    """저장된 forest·경로 분포·보정기를 재적합 없이 사용한다."""
    x=features.loc[:,bundle['features']].to_numpy(np.float32)
    out=bundle['distribution'].predict(x,np.asarray(b,float),quantiles=True,initial=bundle['threshold_initial'])
    out['point_prediction']=bundle['distribution'].forest.predict(x)
    out['p_point_calibrated']=bundle['calibrator'].predict(out['p_point'])
    out['p_period_calibrated']=risk.coherent_period_probability(out['p_point_calibrated'],bundle['calibrator'].predict(out['p_period']))
    choice=bundle.get('probability_output','raw')
    out['p_point_selected']=out['p_point_calibrated'] if choice=='calibrated' else out['p_point']
    out['p_period_selected']=out['p_period_calibrated'] if choice=='calibrated' else out['p_period']
    return out


def fit_past_calibrator(history,start):
    all_p=[];all_y=[];all_w=[];ends=[];n_origins=0
    for entry in history:
        origins=pd.DatetimeIndex(entry['origins']);h=entry['horizon']
        label_end=origins+pd.Timedelta(minutes=15*h)
        keep=np.asarray(label_end<pd.Timestamp(start))
        if 'clean_targets' in entry:keep &= entry['clean_targets']
        if not keep.any():continue
        p=entry['p_point'][keep];period=entry['p_period'][keep];y=entry['events'][keep]
        all_p.extend([p.ravel(),period]);all_y.extend([y.ravel(),y.any(axis=1)])
        all_w.extend([np.full(p.size,.5/h),np.full(len(period),.5)])
        ends.extend(label_end[keep]);n_origins+=int(keep.sum())
    p=np.concatenate(all_p) if all_p else np.array([])
    y=np.concatenate(all_y) if all_y else np.array([])
    w=np.concatenate(all_w) if all_w else np.array([])
    calibrator=risk.MonotoneCalibrator().fit(p,y,w)
    return calibrator,dict(status=calibrator.status,n_oof_origins=n_origins,n_events=len(y),
        n_positive=int(y.sum()),max_target_end=str(max(ends)) if ends else None,
        slope=calibrator.slope,intercept=calibrator.intercept,cutoff=str(start),
        point_period_weight='each contributes half per origin; same monotone map',
        repaired_calibration_targets_excluded=True)


def select_cv(metrics):
    cv=metrics.loc[metrics.fold.isin(CV_FOLDS)&metrics.scope.eq('full')&metrics.candidate.str.match(r'^cap[124]$')]
    selected={}
    for profile,part in cv.groupby('profile',sort=False):
        if part.duplicated(['candidate','fold']).any():raise ValueError('CV 후보·월 중복')
        for candidate in ['cap1','cap2','cap4']:
            if set(part.loc[part.candidate.eq(candidate),'fold'])!=set(CV_FOLDS):
                raise ValueError(f'{profile}/{candidate}: 전체 5-fold 결과가 필요합니다.')
        columns=['Loss_C','RMSE','L_peak','point_Brier_raw','period_Brier_raw','point_Brier_cal','period_Brier_cal']
        macro=part.groupby('candidate')[columns].mean()
        if not np.isfinite(macro.to_numpy()).all():raise ValueError('CV 선정 점수가 유한하지 않습니다.')
        ranked=macro.assign(cap=[int(c.removeprefix('cap')) for c in macro.index]).sort_values(['Loss_C','cap'],kind='stable')
        candidate=str(ranked.index[0]);row=ranked.iloc[0]
        raw=(row.point_Brier_raw+row.period_Brier_raw)/2
        calibrated=(row.point_Brier_cal+row.period_Brier_cal)/2
        selected[profile]=dict(candidate=candidate,cap=int(row.cap),probability_output='calibrated' if calibrated<raw else 'raw',
                               cv_mean_common_C=float(row.Loss_C),cv_RMSE=float(row.RMSE),
                               cv_point_Brier_raw=float(row.point_Brier_raw),cv_period_Brier_raw=float(row.period_Brier_raw),
                               cv_point_Brier_cal=float(row.point_Brier_cal),cv_period_Brier_cal=float(row.period_Brier_cal))
    if not selected:raise ValueError('선정 가능한 CV 결과 없음')
    return selected


def new_record(suite,profile,candidate,phase,config):
    folder=suite/phase/profile/candidate;folder.mkdir(parents=True,exist_ok=False)
    experiment_id=f'{suite.name}_{phase}_{profile}_{candidate}'
    record=dict(experiment_id=experiment_id,profile=profile,candidate=candidate,phase=phase,
                artifact=folder,record=ROOT/'experiments/random_forest'/f'{experiment_id}.md',
                started=datetime.now(common.KST).isoformat(),status='실행 중',fits=[],metrics=[],config=config)
    write_record(record)
    return record


def write_record(rec):
    c=rec['config'];profile=rec['profile'];spec=PROFILES[profile]
    frame=pd.DataFrame(rec['metrics'])
    columns=['fold','scope','n_origins','RMSE','Loss_C','point_Brier_raw','point_Brier_cal','period_Brier_raw','period_Brier_cal','n_positive_point','n_positive_period']
    table=common.md_table(frame[[col for col in columns if col in frame]]) if len(frame) else '완료 평가 없음.'
    fits=common.md_table(pd.DataFrame(rec['fits'])) if rec['fits'] else '완료 RF 적합 0회.'
    elapsed=sum(f['elapsed_seconds'] for f in rec['fits'])
    rec['record'].write_text(f'''# 실험 {rec['experiment_id']}

- 상태: {rec['status']}
- 목적과 가설: {profile}/{rec['candidate']}의 피크 확률 연계 연속 가중 RF와 실제 학습 경로 분포 평가. 무가중 cap1 및 무조건 역사 분포와 비교.
- 시작·종료 시각 및 시간대: {rec['started']} / {rec.get('ended','미완료')}; Asia/Seoul.
- 모델·라이브러리 버전: Python {platform.python_version()}; {c['versions']}; RF native leaf-path distribution, {rec['candidate']}.
- 데이터 위치·버전 또는 해시·사용 범위: {common.SOURCE}; SHA256 {c['source_sha256']}; 원본 미변경. 전체 q90 계산 24,672행, 분할·예측은 종료 경계 행 제외.
- 목표값·단위·예측 구간: power_15min kW; 미래 {spec['horizon_steps']*15}min, 15분 {spec['horizon_steps']}점; 원점 갱신 {spec['cadence_steps']*15}min. 사건 y>B(origin), 기간 사건 max(y)>B(origin).
- 학습·검증·테스트 분할과 누수 점검: 4~8월 시간순 expanding 5-fold. 경로 마지막 학습 정답<월 시작, 평가 정답<다음 월. 2·3월은 OOF warm-up/보정 자료. 보정기는 과거 OOF 완전 경로만 사용. final 단계는 1~8월 재학습·9월 재평가이며 선택 동결 후 실행. 전체 q90에 9월이 들어가므로 독립 테스트로 주장하지 않음.
- 전처리·특성·하이퍼파라미터·seed: 기존 F2 미래 외생 오차0 가정·입력 통계 창 {c['windows'][profile]}h, 원점 B와 최근전력/B 두 공통 피처 추가, scaler 없음. {c['rf_params']}; 설정된 {c['rf_params']['n_estimators']} trees 완성 forest만 평가. smoke_only={c['smoke_only']}이며 smoke는 전체 실험 성능으로 해석하지 않음.
- 가중치·확률: g=clip((y/B-.8)/.2,0,1), w=1+(상한-1)*g*(1+p_ref_OOF)/2; 원점 평균을 sample_weight로 사용. OOF 없는 warm-up/월경계 경로는 w=1. 평가 C는 모든 후보 공통 상한2/동일 기준 확률. RF는 가중 squared_error이며 C·과소 오차를 직접 학습하지 않음. 잎 내 실제 학습 경로 질량은 균등, 나무 평균, 같은 경로로 시점·기간 원시 확률과 분위수를 구함. 과거 OOF 단조 sigmoid 보정 확률은 별도 출력하고 기간 보정 확률은 시점 최대~min(1,시점 합) 사이로 투영한다. 보정 joint 경로 분포는 재적합하지 않는다.
- 이전 또는 기준 실험 링크와 변경 사항: [기존 RF 전체 시나리오](../../notebooks/models/random_forest/RF_SCENARIO_OVERVIEW.md), [새 설계](../../docs/superpowers/specs/2026-10-04-rf-peak-probability-design.md). 4→5-fold, 전체 q90 초기 피크, 확률 연계 가중치·경로 분포 확장. 기존 보고서와 직접 동일 조건 비교 아님.
- 실행 명령·환경·코드 버전 또는 코드 식별 정보: python -B -X utf8 notebooks/models/random_forest/rf_peak_probability.py --budget-seconds 14400; Git 없음; {c['code_hashes']}.
- 실제 실행 시간·자원 사용량(측정한 경우): RF 적합 누적 {elapsed:.3f}s, 완료 {len(rec['fits'])}회; CPU4workers, 순차 forest, 실행 총예산14400s. 분포·평가·직렬화 포함 총시간은 suite_config.json. 최대 RSS 미측정.
- 평가 지표·구간별 또는 fold별 결과: 아래 실제 완료 결과. C는 평균/최악ceil20% 손실블록/가중 과소RMSE=.5/.3/.2, 전체 셀 평균 후 root. Loss_C_w1도 저장. Brier/logloss·Pinball·구간coverage·조건부 과소/과대·신뢰도·거리별 결과는 CSV. 경보0.5는 진단용이며 운영 최적 임계값 아님.
- 기준모델 대비 변화와 비교 가능 여부: 같은profile/fold/scope의 cap1/2/4 공통 평가 가중치로 비교; 5개월 동일 가중. 확률 기준선은 같은 학습 경로의 무조건 경험분포. 다른 H의 점수를 하나의 순위로 합치지 않음.
- 오차가 큰 조건·해석·한계: 전체 q90은 미래 정보 포함 탐색 기준. 미래 외생 실측=오차0 예보. 중첩 원점은 독립 아님, 동적 기준의 양성 사건 수·일수 보고. 0양성의 recall은 NaN. 보정 전력은 참값 복구가 아니며 민감도별 범위 분리. native leaf-path 확장은 scalar QRF 이론을 다변량으로 입증한 것이 아님. 보정 확률은 보정된 joint 경로에서 생성한 값이 아니며 원시 분위수·경로 확률과 구분.
- 오류·경고·실패 또는 중단 원인: {rec.get('error','없음')}.
- 예측·모델·표·그래프·로그의 저장 경로: {rec['artifact']}; model_bundle.joblib, predictions.csv.gz, path_predictions.csv.gz, oof.npz, fold_config.json. suite metrics/reliability/horizon/condition CSV, 보고서·그림·노트북.
- 결론과 다음 실험: 선정은 모든5-fold 완료 후 selection.json에 동결. 9월 점수로 재튜닝하지 않는다. 이후 실제 예보와 새 기간 검증이 필요하다.

{table}

## 실제 적합 및 OOF 경계

{fits}
''',encoding='utf-8')
    dump(rec['artifact']/'status.json',dict(status=rec['status'],started=rec['started'],ended=rec.get('ended'),n_fits=len(rec['fits']),fits=rec['fits'],error=rec.get('error')))
    if len(frame):frame.to_csv(rec['artifact']/'metrics.csv',index=False,encoding='utf-8-sig')


def fit_forest(x,y,weights,params,deadline,suite,tag):
    before=perf_counter();total=params['n_estimators']
    forest=RandomForestRegressor(**{**params,'n_estimators':min(100,total),'warm_start':True})
    for trees in range(min(100,total),total+100,100):
        if perf_counter()>=deadline:raise TimeoutError(f'전체4시간 예산 소진: {tag}')
        forest.set_params(n_estimators=min(trees,total));forest.fit(x,y,sample_weight=weights)
        log(suite,f'TREES {tag} {forest.n_estimators}/{total} {perf_counter()-before:.1f}s')
        if forest.n_estimators==total:break
    forest.set_params(warm_start=False)
    return forest,perf_counter()-before


def save_oof(folder,entry):
    np.savez_compressed(folder/'oof.npz',origins=pd.DatetimeIndex(entry['origins']).asi8,
        horizon=entry['horizon'],p_point=entry['p_point'],p_period=entry['p_period'],events=entry['events'],clean_targets=entry['clean_targets'])


def load_history(folder):
    history=[]
    for path in sorted(Path(folder).glob('m*/oof.npz')):
        with np.load(path,allow_pickle=False) as f:
            history.append(dict(origins=pd.to_datetime(f['origins']),horizon=int(f['horizon']),
                p_point=f['p_point'],p_period=f['p_period'],events=f['events'],clean_targets=f['clean_targets']))
    return history


def export_predictions(folder,data,indices,actual,predicted,out,calibrator,common_weights):
    origins=data['origins'][indices];h=actual.shape[1]
    stamps=origins.to_numpy()[:,None]+np.arange(1,h+1)[None,:]*np.timedelta64(15,'m')
    cal=calibrator.predict(out['p_point']);cal_period=risk.coherent_period_probability(cal,calibrator.predict(out['p_period']))
    if (cal_period+1e-10<cal.max(axis=1)).any():raise ValueError('보정 기간 확률 관계 오류')
    f=pd.DataFrame(dict(origin=np.repeat(origins,h),target_end=stamps.ravel(),lead_minutes=np.tile(np.arange(1,h+1)*15,len(origins)),
        actual=actual.ravel(),predicted=predicted.ravel(),peak_threshold=np.repeat(data['threshold'].iloc[indices].to_numpy(),h),
        p_peak_raw=out['p_point'].ravel(),p_peak_calibrated=cal.ravel(),p_initial_q90=out['p_initial'].ravel(),
        common_peak_weight=common_weights.ravel(),keep_clean_targets=np.repeat(~data['has_repaired_target'][indices],h),
        keep_clean_targets_inputs=np.repeat(~(data['has_repaired_target'][indices]|data['has_repaired_input'][indices]),h)))
    for j,q in enumerate(risk.QUANTILES):f[f'Q{int(q*100):02}']=out['quantiles'][:,:,j].ravel()
    f.to_csv(folder/'predictions.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')
    paths=pd.DataFrame(dict(origin=origins,target_end=stamps[:,-1],peak_threshold=data['threshold'].iloc[indices].to_numpy(),
        actual_max=actual.max(axis=1),predicted_max=predicted.max(axis=1),p_period_raw=out['p_period'],p_period_calibrated=cal_period,
        p_initial_period=out['p_initial_period'],keep_clean_targets=~data['has_repaired_target'][indices],
        keep_clean_targets_inputs=~(data['has_repaired_target'][indices]|data['has_repaired_input'][indices])))
    paths.to_csv(folder/'path_predictions.csv.gz',index=False,encoding='utf-8-sig',compression='gzip')


def evaluate(data,indices,predicted,out,calibrator,reference,tag):
    a=data['targets'].iloc[indices].to_numpy();b=data['threshold'].iloc[indices].to_numpy()
    weights=risk.cell_weights(a,b,reference,2);origins=data['origins'][indices]
    masks=dict(full=np.ones(len(a),bool),clean_targets=~data['has_repaired_target'][indices],
               clean_targets_inputs=~(data['has_repaired_target'][indices]|data['has_repaired_input'][indices]))
    rows=[];curves=[];horizons=[];conditions=[]
    for scope,keep in masks.items():
        row=dict(**tag,scope=scope,n_origins=int(keep.sum()),n_excluded_origins=int((~keep).sum()),n_predictions=int(keep.sum()*a.shape[1]))
        if not keep.any():row.update(status='평가 불가: 표본 0');rows.append(row);continue
        y=a[keep];p=predicted[keep];threshold=b[keep];event=y>threshold[:,None]
        raw=out['p_point'][keep];period=out['p_period'][keep];cal=calibrator.predict(raw)
        period_cal=risk.coherent_period_probability(cal,calibrator.predict(period))
        if (period_cal+1e-10<cal.max(axis=1)).any():raise ValueError('보정 확률 관계 오류')
        score=risk.point_metrics(y,p,weights[keep],data['spec']['loss_steps']);row.update(score)
        point_raw=risk.binary_metrics(event,raw);point_cal=risk.binary_metrics(event,cal)
        path_raw=risk.binary_metrics(event.any(axis=1),period);path_cal=risk.binary_metrics(event.any(axis=1),period_cal)
        for label,metrics in [('point_raw',point_raw),('point_cal',point_cal),('period_raw',path_raw),('period_cal',path_cal)]:
            row.update({f'{label}_{k}':v for k,v in metrics.items()})
        # Stable column names used in selection/report.
        row.update(point_Brier_raw=point_raw['Brier'],point_Brier_cal=point_cal['Brier'],period_Brier_raw=path_raw['Brier'],period_Brier_cal=path_cal['Brier'],
            n_positive_point=int(event.sum()),n_positive_period=int(event.any(axis=1).sum()),n_days=int(pd.DatetimeIndex(origins[keep]).normalize().nunique()))
        stamps=origins[keep].to_numpy()[:,None]+np.arange(1,y.shape[1]+1)[None,:]*np.timedelta64(15,'m')
        row['n_unique_target_times']=int(pd.DatetimeIndex(stamps.ravel()).nunique())
        row['n_unique_peak_times']=int(pd.DatetimeIndex(stamps[event]).nunique())
        q=out['quantiles'][keep]
        pinball=[]
        for j,level in enumerate(risk.QUANTILES):
            error=y-q[:,:,j];loss=float(np.maximum(level*error,(level-1)*error).mean())
            row[f'Pinball_Q{int(level*100):02}']=loss;pinball.append(loss)
        row.update(mean_pinball=float(np.mean(pinball)),coverage90=float(((y>=q[:,:,0])&(y<=q[:,:,-1])).mean()),
                   interval90_width=float((q[:,:,-1]-q[:,:,0]).mean()),Q50_RMSE=float(np.sqrt(np.mean((y-q[:,:,3])**2))),status='완료')
        for label,y_event,prob in [('point_raw',event,raw),('point_cal',event,cal),('period_raw',event.any(axis=1),period),('period_cal',event.any(axis=1),period_cal)]:
            curves.extend(dict(**tag,scope=scope,probability_kind=label,**item) for item in risk.reliability(y_event,prob))
        for h in range(y.shape[1]):
            horizons.append(dict(**tag,scope=scope,lead_minutes=(h+1)*15,n_origins=len(y),RMSE=float(np.sqrt(np.mean((y[:,h]-p[:,h])**2))),
                n_positive=int(event[:,h].sum()),Brier_raw=float(np.mean((raw[:,h]-event[:,h])**2)),Brier_cal=float(np.mean((cal[:,h]-event[:,h])**2))))
        for label,chosen in [('peak',event),('near_peak',y>=.8*threshold[:,None])]:
            error=y-p;n=int(chosen.sum())
            conditions.append(dict(**tag,scope=scope,condition=label,n_predictions=n,
                under_RMSE=float(np.sqrt(np.mean(np.maximum(error[chosen],0)**2))) if n else np.nan,
                over_RMSE=float(np.sqrt(np.mean(np.maximum(-error[chosen],0)**2))) if n else np.nan,
                RMSE=float(np.sqrt(np.mean(error[chosen]**2))) if n else np.nan))
        rows.append(row)
    return rows,curves,horizons,conditions,weights


def history_distribution(y,b,initial):
    n=len(b);h=y.shape[1];point=np.empty((n,h));period=np.empty(n)
    for value in np.unique(b):
        keep=b==value;event=y>value
        point[keep]=event.mean(axis=0);period[keep]=event.any(axis=1).mean()
    q=np.quantile(y,risk.QUANTILES,axis=0,method='inverted_cdf').T
    initial_event=y>initial
    return dict(p_point=point,p_period=period,quantiles=np.repeat(q[None,:,:],n,axis=0),
        p_initial=np.repeat(initial_event.mean(axis=0)[None,:],n,axis=0),p_initial_period=np.full(n,initial_event.any(axis=1).mean()))


def run_suite(budget_seconds=14400,threshold_mode='dynamic',n_estimators=None,profiles=None,smoke=False):
    started=datetime.now(common.KST).isoformat();before=perf_counter();deadline=before+budget_seconds
    run_id=datetime.now(common.KST).strftime('%Y%m%dT%H%M%S_KST')+'_'+uuid4().hex[:6]+('_rf_peak_smoke' if smoke else '_rf_peak_probability')
    suite=ROOT/'artifacts/random_forest/runs'/run_id;suite.mkdir(parents=True,exist_ok=False)
    selected_old=read(final.GRID_RUN/'selected_config.json');windows_old=read(final.WINDOW_RUN/'selection.json')
    params=dict(selected_old['rf_params']);params['n_jobs']=4
    if n_estimators is not None:params['n_estimators']=n_estimators
    windows={'short_1h':selected_old['spec']['window_hours'],**{p:s['window_hours'] for p,s in windows_old['selected'].items()}}
    config=dict(run_id=run_id,status='실행 중',started=started,budget_seconds=budget_seconds,threshold_mode=threshold_mode,
        rf_params=params,windows=windows,profiles=profiles or list(PROFILES),smoke_only=smoke,
        versions={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','nbformat','nbclient','matplotlib']},
        source_sha256=common.digest(common.SOURCE),code_hashes={Path(p).name:common.digest(p) for p in [__file__,risk.__file__,common.__file__,grid.__file__,iw.__file__,ml.__file__]},
        selection_rule='4~8월5fold 같은평가w2 C 월동일평균 최소; 같은후보 raw/cal Brier평균으로출력선택',
        global_threshold_is_exploratory=True,september_scores_used_for_candidate_selection=False,september_data_used_for_initial_q90=True,
        production_weather='future_actual_as_forecast_zero_error',probability_weight_mix=.5,weight_rho=.8,
        calibration='past OOF monotone sigmoid; same map for point/path; not recalibrated joint trajectories',
        loss_C_directly_trained=False,recorded_scope='native_RF_only')
    dump(suite/'suite_config.json',config)
    snapshot=suite/'code_snapshot';snapshot.mkdir()
    for path in [__file__,risk.__file__,common.__file__,grid.__file__,iw.__file__,ml.__file__]:
        original=Path(path);(snapshot/original.name).write_bytes(original.read_bytes())
    records=[];all_metrics=[];all_curves=[];all_horizons=[];all_conditions=[]
    def update_tables():
        for name,rows in [('metrics',all_metrics),('reliability',all_curves),('horizon_metrics',all_horizons),('condition_metrics',all_conditions)]:
            if rows:pd.DataFrame(rows).to_csv(suite/f'{name}.csv',index=False,encoding='utf-8-sig')
    try:
        raw=pd.read_csv(common.SOURCE,usecols=['power_15min'])
        initial=float(raw.power_15min.quantile(.9))
        df,repair,audit=final.load_final_data();config.update(initial_peak_kw=initial,data_audit=audit)
        dump(suite/'suite_config.json',config);log(suite,f'START initial_q90={initial}kW mode={threshold_mode} budget={budget_seconds}s')
        for profile in config['profiles']:
            data=build_profile(df,repair,profile,windows[profile],initial,threshold_mode)
            n=len(data['origins']);h=data['spec']['horizon_steps']
            reference=np.full((n,h),np.nan)
            history={cap:[] for cap in CAPS}
            recs={cap:new_record(suite,profile,f'cap{cap}','cv',config) for cap in CAPS}
            hist_rec=new_record(suite,profile,'history_distribution','cv',config)
            records.extend([*recs.values(),hist_rec])
            for month in range(2,9):
                start=pd.Timestamp(2021,month,1);end=start+pd.offsets.MonthBegin(1);fold=f'm{month:02}'
                train,valid=risk.time_masks(data['origins'],h,start,end)
                ti=np.flatnonzero(train);vi=np.flatnonzero(valid)
                if not len(ti) or not len(vi):raise ValueError(f'{profile}/{fold}: 빈 학습/평가')
                x=data['features'].iloc[ti].to_numpy(np.float32);y=data['targets'].iloc[ti].to_numpy()
                actual=data['targets'].iloc[vi].to_numpy();b=data['threshold'].iloc[vi].to_numpy()
                xv=data['features'].iloc[vi].to_numpy(np.float32)
                base_out=None
                for cap in CAPS:
                    rec=recs[cap];folder=rec['artifact']/fold;folder.mkdir(exist_ok=False)
                    if month==2 and cap!=1:
                        # All training origins lack past OOF here, so all cap models are exactly the same.
                        entry=history[1][-1];history[cap].append(entry);save_oof(folder,entry)
                        dump(folder/'reuse.json',dict(source=str(recs[1]['artifact']/fold),reason='all_training_weights_1_in_initial_warmup',new_model_fits=0))
                        continue
                    calibrator,cal_meta=fit_past_calibrator(history[cap],start)
                    sw=np.ones(len(ti));known=np.isfinite(reference[ti]).all(axis=1)
                    if cap!=1 and known.any():
                        sw[known]=risk.cell_weights(y[known],data['threshold'].iloc[ti].to_numpy()[known],reference[ti][known],cap).mean(axis=1)
                    forest,elapsed=fit_forest(x,y,sw,params,deadline,suite,f'{profile}/{fold}/cap{cap}')
                    dist=risk.LeafPathDistribution(forest,x,y)
                    out=dist.predict(xv,b,quantiles=month>=4,initial=initial)
                    predicted=forest.predict(xv)
                    if cap==1:base_out=out;reference[vi]=out['p_point']
                    entry=dict(origins=data['origins'][vi],horizon=h,p_point=out['p_point'],p_period=out['p_period'],
                        events=actual>b[:,None],clean_targets=~data['has_repaired_target'][vi])
                    history[cap].append(entry);save_oof(folder,entry)
                    info=dict(fold=fold,n_train=len(ti),n_valid=len(vi),n_weighted=int((sw>1+1e-12).sum()),
                        n_oof_supported=int(known.sum()),n_unweighted_warmup_gap=int((~known).sum()),weight_min=float(sw.min()),weight_max=float(sw.max()),weight_mean=float(sw.mean()),
                        elapsed_seconds=elapsed,train_label_max=str(data['origins'][ti[-1]]+pd.Timedelta(minutes=15*h)),
                        valid_label_max=str(data['origins'][vi[-1]]+pd.Timedelta(minutes=15*h)),n_trees=len(forest.estimators_))
                    rec['fits'].append(info)
                    dump(folder/'fold_config.json',dict(**info,start=str(start),end=str(end),features=list(data['features']),calibration=cal_meta,
                        distribution_mass='uniform_actual_paths_per_leaf_not_sample_weight',comparison_reference='same current cap1 model for every candidate'))
                    joblib.dump(dict(distribution=dist,calibrator=calibrator,features=list(data['features']),
                        threshold_initial=initial,threshold_mode=threshold_mode,profile=profile,context_window_hours=windows[profile],spec=data['spec'],cap=cap,fit=info),folder/'model_bundle.joblib',compress=3)
                    if month>=4:
                        tag=dict(profile=profile,candidate=f'cap{cap}',fold=fold,phase='cv')
                        rows,curves,horizons,conditions,eval_w=evaluate(data,vi,predicted,out,calibrator,base_out['p_point'],tag)
                        rec['metrics'].extend(rows);all_metrics.extend(rows);all_curves.extend(curves);all_horizons.extend(horizons);all_conditions.extend(conditions)
                        export_predictions(folder,data,vi,actual,predicted,out,calibrator,eval_w)
                        update_tables()
                        full=rows[0];log(suite,f'SCORE {profile}/{fold}/cap{cap} C={full["Loss_C"]:.4f} RMSE={full["RMSE"]:.4f} Brier={full["point_Brier_raw"]:.6f} peaks={full["n_positive_point"]}/{actual.size}')
                    write_record(rec);del forest,dist;gc.collect()
                if month>=4:
                    folder=hist_rec['artifact']/fold;folder.mkdir(exist_ok=False)
                    out=history_distribution(y,b,initial);predicted=np.repeat(y.mean(axis=0)[None,:],len(vi),axis=0)
                    calibrator=risk.MonotoneCalibrator();tag=dict(profile=profile,candidate='history_distribution',fold=fold,phase='cv')
                    rows,curves,horizons,conditions,w=evaluate(data,vi,predicted,out,calibrator,base_out['p_point'],tag)
                    hist_rec['metrics'].extend(rows);all_metrics.extend(rows);all_curves.extend(curves);all_horizons.extend(horizons);all_conditions.extend(conditions)
                    export_predictions(folder,data,vi,actual,predicted,out,calibrator,w);write_record(hist_rec);update_tables()
            for rec in [*recs.values(),hist_rec]:rec.update(status='완료',ended=datetime.now(common.KST).isoformat());write_record(rec)
            del data,reference,history;gc.collect()
        selection=select_cv(pd.DataFrame(all_metrics))
        selection_document=dict(selected=selection,selected_at=datetime.now(common.KST).isoformat(),folds=CV_FOLDS,
            criterion=config['selection_rule'],september_scores_read=False,global_q90_includes_september=True)
        dump(suite/'selection.json',selection_document);config['selection_sha256']=common.digest(suite/'selection.json')
        dump(suite/'suite_config.json',config);log(suite,'SELECTION_FROZEN '+json.dumps(selection,ensure_ascii=False))
        for profile in config['profiles']:
            data=build_profile(df,repair,profile,windows[profile],initial,threshold_mode);h=data['spec']['horizon_steps']
            start=pd.Timestamp('2021-09-01');end=pd.Timestamp('2021-09-15')
            train,valid=risk.time_masks(data['origins'],h,start,end);ti=np.flatnonzero(train);vi=np.flatnonzero(valid)
            x=data['features'].iloc[ti].to_numpy(np.float32);y=data['targets'].iloc[ti].to_numpy();xv=data['features'].iloc[vi].to_numpy(np.float32)
            actual=data['targets'].iloc[vi].to_numpy();b=data['threshold'].iloc[vi].to_numpy()
            reference=np.full((len(data['origins']),h),np.nan)
            for entry in load_history(suite/'cv'/profile/'cap1'):
                pos=data['origins'].get_indexer(entry['origins'])
                if (pos<0).any():raise ValueError('OOF 원점 식별자 대응 실패')
                reference[pos]=entry['p_point']
            selected_cap=selection[profile]['cap'];base_out=None
            for cap in sorted(set([1,selected_cap])):
                rec=new_record(suite,profile,f'cap{cap}','final',config);records.append(rec)
                folder=rec['artifact']/'september';folder.mkdir(exist_ok=False)
                history=load_history(suite/'cv'/profile/f'cap{cap}');calibrator,cal_meta=fit_past_calibrator(history,start)
                known=np.isfinite(reference[ti]).all(axis=1);sw=np.ones(len(ti))
                if cap!=1 and known.any():sw[known]=risk.cell_weights(y[known],data['threshold'].iloc[ti].to_numpy()[known],reference[ti][known],cap).mean(axis=1)
                forest,elapsed=fit_forest(x,y,sw,params,deadline,suite,f'{profile}/september/cap{cap}')
                dist=risk.LeafPathDistribution(forest,x,y);out=dist.predict(xv,b,quantiles=True,initial=initial);predicted=forest.predict(xv)
                if cap==1:base_out=out
                tag=dict(profile=profile,candidate=f'cap{cap}',fold='september',phase='final')
                rows,curves,horizons,conditions,w=evaluate(data,vi,predicted,out,calibrator,base_out['p_point'],tag)
                rec['metrics'].extend(rows);all_metrics.extend(rows);all_curves.extend(curves);all_horizons.extend(horizons);all_conditions.extend(conditions)
                export_predictions(folder,data,vi,actual,predicted,out,calibrator,w)
                fit=dict(fold='september',n_train=len(ti),n_valid=len(vi),n_weighted=int((sw>1+1e-12).sum()),n_oof_supported=int(known.sum()),
                    weight_min=float(sw.min()),weight_max=float(sw.max()),weight_mean=float(sw.mean()),elapsed_seconds=elapsed,
                    train_label_max=str(data['origins'][ti[-1]]+pd.Timedelta(minutes=15*h)),valid_label_max=str(data['origins'][vi[-1]]+pd.Timedelta(minutes=15*h)),n_trees=len(forest.estimators_))
                rec['fits'].append(fit);dump(folder/'fold_config.json',dict(**fit,start=str(start),end=str(end),features=list(data['features']),calibration=cal_meta,
                    selected=cap==selected_cap,probability_output=selection[profile]['probability_output'] if cap==selected_cap else 'reference_raw',selection_sha256=config['selection_sha256']))
                joblib.dump(dict(distribution=dist,calibrator=calibrator,features=list(data['features']),threshold_initial=initial,
                    threshold_mode=threshold_mode,profile=profile,context_window_hours=windows[profile],spec=data['spec'],cap=cap,fit=fit,probability_output=selection[profile]['probability_output']),folder/'model_bundle.joblib',compress=3)
                rec.update(status='완료',ended=datetime.now(common.KST).isoformat());write_record(rec);update_tables()
                log(suite,f'FINAL {profile}/cap{cap} RMSE={rows[0]["RMSE"]:.4f} point_Brier={rows[0]["point_Brier_raw"]:.6f} period_Brier={rows[0]["period_Brier_raw"]:.6f}')
                del forest,dist;gc.collect()
            rec=new_record(suite,profile,'history_distribution','final',config);records.append(rec)
            folder=rec['artifact']/'september';folder.mkdir(exist_ok=False)
            out=history_distribution(y,b,initial);predicted=np.repeat(y.mean(axis=0)[None,:],len(vi),axis=0);calibrator=risk.MonotoneCalibrator()
            tag=dict(profile=profile,candidate='history_distribution',fold='september',phase='final')
            rows,curves,horizons,conditions,w=evaluate(data,vi,predicted,out,calibrator,base_out['p_point'],tag)
            rec['metrics'].extend(rows);all_metrics.extend(rows);all_curves.extend(curves);all_horizons.extend(horizons);all_conditions.extend(conditions)
            export_predictions(folder,data,vi,actual,predicted,out,calibrator,w)
            rec.update(status='완료',ended=datetime.now(common.KST).isoformat());write_record(rec);update_tables()
            del data,reference;gc.collect()
        config.update(status='학습 완료·독립검증 대기',ended=datetime.now(common.KST).isoformat(),elapsed_seconds=perf_counter()-before,
            n_completed_rf_fits=sum(len(r['fits']) for r in records),n_experiment_records=len(records),selected=selection)
        dump(suite/'suite_config.json',config);log(suite,f'TRAINING_COMPLETE fits={config["n_completed_rf_fits"]} seconds={config["elapsed_seconds"]:.1f}')
    except Exception as exc:
        config.update(status='실패' if not isinstance(exc,TimeoutError) else '중단',error=f'{type(exc).__name__}: {exc}',
            ended=datetime.now(common.KST).isoformat(),elapsed_seconds=perf_counter()-before,n_completed_rf_fits=sum(len(r['fits']) for r in records))
        dump(suite/'suite_config.json',config);log(suite,traceback.format_exc())
        for rec in records:
            if rec['status']=='실행 중':rec.update(status=config['status'],error=config['error'],ended=config['ended']);write_record(rec)
        raise
    finally:
        update_tables()
        with (ROOT/'experiments/README.md').open('a',encoding='utf-8') as stream:
            stream.write(f'\n## {run_id}: RF 피크 확률 연계 연속 가중치\n\n상태: {config["status"]}. 전체 q90 초기 기준,4~8월5-fold,9월 탐색 재평가. 신규 적합 {config.get("n_completed_rf_fits",0)}회.\n\n')
            for rec in records:stream.write(f'- [{rec["profile"]}/{rec["candidate"]}/{rec["phase"]}]({rec["record"].relative_to(ROOT/"experiments").as_posix()}): {rec["status"]}\n')
        dump(suite/'records.json',[dict(experiment_id=r['experiment_id'],record=str(r['record']),artifact=str(r['artifact']),status=r['status'],n_fits=len(r['fits'])) for r in records])
    return suite


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--budget-seconds',type=int,default=14400)
    parser.add_argument('--threshold-mode',choices=['dynamic','fixed'],default='dynamic')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    suite=run_suite(args.budget_seconds,args.threshold_mode,n_estimators=5 if args.smoke else None,
                    profiles=['short_1h'] if args.smoke else None,smoke=args.smoke)
    print(f'RUN_DIR={suite}',flush=True)
