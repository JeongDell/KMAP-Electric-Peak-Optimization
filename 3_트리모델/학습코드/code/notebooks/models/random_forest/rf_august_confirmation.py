"""동결 RF 설정의 8월 확인. 9월 목표는 학습·평가에 사용하지 않는다."""
from pathlib import Path
from datetime import datetime
from importlib.metadata import version
from time import perf_counter
from uuid import uuid4
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

ROOT = common.ROOT
LEGACY = ROOT/'data/processed/timeseries_all.csv'
SELECTION = ROOT/'artifacts/random_forest/runs/20261003T160916_KST_c2f466_rf_grid'
START, END = pd.Timestamp('2021-08-01'), pd.Timestamp('2021-09-01')
METRICS = grid.METRICS
SCOPES = dict(full='전체 전처리본', clean_targets='보정 정답 경로 제외',
              clean_targets_inputs='보정 정답·모든 비교 모델의 보정 전력 입력 경로 제외')


def load_august_data(source=common.SOURCE, legacy=LEGACY):
    df = pd.read_csv(source, parse_dates=['timestamp','timestamp_end'])
    required = ['timestamp','timestamp_end','power_15min','day_type',*common.OBSERVED]
    if set(required)-set(df):
        raise ValueError(f'필수 컬럼 누락: {sorted(set(required)-set(df))}')
    # 9월은 수치 검사·피처 구성 전에 제외한다.
    df = df.loc[df.timestamp_end < END].set_index('timestamp_end',drop=False).copy()
    common.validate_frame(df)
    if not np.isfinite(df[['power_15min',*common.OBSERVED]].to_numpy(float)).all():
        raise ValueError('1~8월 입력·정답에 결측/무한값이 있습니다. 자동 대체하지 않습니다.')
    if not df.groupby(df.timestamp.dt.floor('h'))[common.OBSERVED].nunique().eq(1).all().all():
        raise ValueError('시간별 반복 외생 피처가 일치하지 않습니다.')
    if not df.day_type.isin([0,1,2,3]).all():
        raise ValueError('day_type은 0/1/2/3이어야 합니다.')
    prior = pd.read_csv(legacy,usecols=['timestamp','power_15min'],parse_dates=['timestamp'])
    prior = prior.loc[prior.timestamp < END].set_index('timestamp').power_15min
    if prior.index.has_duplicates:
        raise ValueError('보정 이력 대조용 기존 종료시각에 중복 키가 있습니다.')
    aligned = prior.reindex(df.index)
    if not np.isfinite(aligned.to_numpy(float)).all():
        raise ValueError('보정 이력 결합에 매칭 실패/결측이 있습니다.')
    repaired = df.power_15min.ne(aligned).rename('power_repaired')
    if repaired.any():
        if not aligned.loc[repaired].eq(0).all():
            raise ValueError('전력 변경 중 기존 0값 대체 이외의 변경이 있습니다.')
        previous = prior.reindex(df.index[repaired]-pd.Timedelta(days=7)).to_numpy()
        if not np.isfinite(previous).all() or not (previous > 0).all():
            raise ValueError('보정 전력의 전주 원본 참조값이 유효하지 않습니다.')
        if not np.array_equal(df.loc[repaired,'power_15min'].to_numpy(),previous):
            raise ValueError('보정 전력이 정확히 7일 전 원본값과 일치하지 않습니다.')
    return df,repaired


def prepare_august(df, repaired, window_hours=1, expected_features=None):
    df = df.loc[df.index < END].copy()
    repaired = repaired.reindex(df.index)
    if repaired.isna().any():
        raise ValueError('보정 표식의 시점 대응이 누락됐습니다.')
    repaired = repaired.astype(bool)
    features,targets,columns = grid.build_window_features(df,window_hours)
    if expected_features is not None and columns != expected_features:
        raise ValueError('동결한70열 피처 순서와 현재 입력이 다릅니다.')
    complete = np.isfinite(features[columns].to_numpy()).all(axis=1)
    complete &= np.isfinite(targets.to_numpy()).all(axis=1)
    train = pd.Series(complete & (df.index+pd.Timedelta(hours=1) < START),index=df.index)
    valid = pd.Series(complete & (df.index >= START) & (df.index+pd.Timedelta(hours=1) < END),index=df.index)
    if not train.any() or not valid.any():
        raise ValueError('8월 학습/평가 유효 원점이 없습니다.')
    target_repaired = pd.Series(False,index=df.index)
    daily_repaired = pd.Series(False,index=df.index)
    for h in range(1,5):
        target_repaired |= repaired.shift(-h,fill_value=False)
        daily_repaired |= repaired.shift(96-h,fill_value=False)
    rf_repaired = repaired | repaired.shift(4,fill_value=False) | repaired.shift(96,fill_value=False)
    rf_repaired |= repaired.rolling(window_hours*4,min_periods=1).max().astype(bool)
    tags = pd.DataFrame(dict(has_repaired_target=target_repaired,has_repaired_rf_input=rf_repaired,
                             has_repaired_daily_input=daily_repaired,
                             has_repaired_any_input=rf_repaired|daily_repaired),index=df.index)
    scopes = dict(full=valid, clean_targets=valid & ~target_repaired,
                  clean_targets_inputs=valid & ~target_repaired & ~tags.has_repaired_any_input)
    return dict(df=df,features=features,targets=targets,columns=columns,train=train,valid=valid,
                repaired=repaired,tags=tags,scopes=scopes,window_hours=window_hours)


def write_record(record, message='없음'):
    cfg=record['config']
    table=common.md_table(pd.DataFrame(record['metrics'])) if record['metrics'] else '실제 완료 점수 없음.'
    record['record'].write_text(f'''# 실험 {cfg['experiment_id']}

- 상태: {record['status']}
- 목적과 가설: 4~7월에서 동결한 RF의 최근8월 성능 확인. 기존 RF·단순 기준선과 비교하되8월 결과로 설정을 변경하지 않는다.
- 시작·종료 시각 및 시간대: {record['started']} / {datetime.now(common.KST).isoformat() if record['status']!='실행 중' else '미완료'}; Asia/Seoul.
- 모델·라이브러리 버전: {cfg['model_kind']}; Python{platform.python_version()}; {cfg['versions']}.
- 데이터 위치·버전 또는 해시·사용 범위: {cfg['source']}, SHA256 {cfg['source_sha256']};1~8월 종료시각<2021-09-01. 보정 대조 {cfg['legacy']}, SHA256 {cfg['legacy_sha256']}.
- 목표값·단위·예측 구간: power_15min kW, 미래1h의15min4점. 원점15min마다 갱신.
- 학습·검증·테스트 분할과 누수 점검: 학습{cfg['n_train']}원점, 학습 정답 종료<2021-08-01;8월{cfg['n_valid']}원점, 정답 종료<2021-09-01. 9월 평가 없음. 모델은8월 동안 고정, 원점까지 관측된 전력만 입력.
- 전처리·특성·하이퍼파라미터·seed: F2 미래 외생 오차0 시나리오70열, scaler 없음, 최근 전력 통계{cfg['window_hours']}h, {cfg['params']}. RF 학습 squared_error, 설계 Loss는 평가에 사용.
- 이전 또는 기준 실험 링크와 변경 사항: [동결 설정 선정](../../notebooks/models/random_forest/BASELINE_GRID_RESULTS_20261003.md). 동일 설정으로1~7월 재학습,8월 확인만 추가. 기준 설정 출처 {cfg['reference_config']}.
- 실행 명령·환경·코드 버전 또는 코드 식별 정보: Jupyter rf_august_confirmation.ipynb Run All; run_august_notebook.py. Git 없는 workspace; 코드 SHA256 {cfg['code_sha256']}, 공통 {cfg['shared_code_sha256']}, proxy {cfg['proxy_code_sha256']}, grid {cfg['grid_code_sha256']}.
- 실제 실행 시간·자원 사용량(측정한 경우): {record.get('elapsed_seconds','미완료')}초; RF worker4개, 후보 순차. 메모리 미측정. 
- 평가 지표·구간별 또는 fold별 결과: 아래3범위,1h4점 window, stride15min, CVaR 최악ceil20%, L_peak 전체 유효 셀의 과소예측 제곱 평균 후root(w=1), Loss B=.6mean+.4CVaR, C=.5mean+.3CVaR+.2peak. 
- 기준모델 대비 변화와 비교 가능 여부: 모든 후보 동일 원점·정답·민감도 마스크. 마스크는 예측 전에 고정하고 보정 표식을 피처로 사용하지 않는다. 
- 오차가 큰 조건·해석·한계: 조건·horizon·최대전력 표 참조. 미래 외생 실측을 예측값으로 간주한 오차0 가정. 보정 정답은 참값 확보가 아니며 각 민감도는 다른 표본이다. 중첩 셀은 독립 사례가 아님. 
- 오류·경고·실패 또는 중단 원인: {message}.
- 예측·모델·표·그래프·로그의 저장 경로: {record['artifact']}; predictions.csv, model.joblib(RF), config.json, status.json, scope_metrics.csv, horizon_metrics.csv, condition_metrics.csv, peak_path_metrics.csv, run.log. 
- 결론과 다음 실험: 이번8월 결과로 재튜닝하지 않는다. 부스팅 비교와 모든 모델 설정 동결 후9월 최종 테스트. 실제 수치는 아래 표와 suite 결과 문서 참조. 

{table}
''',encoding='utf-8')


def evaluate_candidate(candidate,kind,params,data,suite,root=ROOT,deadline=None,reference_config=None):
    root,suite=Path(root),Path(suite)
    artifact=suite/candidate
    artifact.mkdir(parents=True,exist_ok=False)
    record_path=root/'experiments/random_forest'/f'{suite.name}_{candidate}.md'
    record_path.parent.mkdir(parents=True,exist_ok=True)
    train,valid=data['train'],data['valid']
    origins=data['features'].index[valid]
    actual=data['targets'].loc[valid].to_numpy()
    cfg=dict(experiment_id=f'{suite.name}_{candidate}',candidate=candidate,model_kind=kind,params=params,
             reference_config=str(reference_config) if reference_config else None,
             source=data['source'],source_sha256=data['source_sha256'],legacy=data['legacy'],legacy_sha256=data['legacy_sha256'],
             features=data['columns'] if kind=='random_forest' else ['power_latest' if kind=='persistence' else 'target-aligned previous day'],
             window_hours=data['window_hours'],n_train=int(train.sum()),n_valid=int(valid.sum()),
             train_first=str(data['features'].index[train][0]),train_last_origin=str(data['features'].index[train][-1]),
             valid_first=str(origins[0]),valid_last_origin=str(origins[-1]),
             code_sha256=common.digest(__file__),shared_code_sha256=common.digest(common.__file__),
             proxy_code_sha256=common.digest(proxy.__file__),grid_code_sha256=common.digest(grid.__file__),
             versions={p:version(p) for p in ['numpy','pandas','scikit-learn','joblib','nbclient']},
             scenario=proxy.SCENARIO,peak_weights=1,forecast_error_assumption=0,scope_rules=SCOPES)
    record=dict(config=cfg,artifact=artifact,record=record_path,metrics=[],status='실행 중',new_fits=0,
                started=datetime.now(common.KST).isoformat())
    (artifact/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    write_record(record)
    before=perf_counter()
    try:
        if deadline is not None and perf_counter()>=deadline:
            raise TimeoutError('8월 확인 실행 예산 소진')
        if kind=='random_forest':
            model=RandomForestRegressor(**params)
            model.fit(data['features'].loc[train,data['columns']],data['targets'].loc[train])
            record['new_fits']=1
            predicted=model.predict(data['features'].loc[valid,data['columns']])
            joblib.dump(model,artifact/'model.joblib',compress=3)
            del model
            gc.collect()
        elif kind=='persistence':
            predicted=np.repeat(data['df'].power_15min.reindex(origins).to_numpy()[:,None],4,axis=1)
        elif kind=='daily':
            predicted=np.column_stack([data['df'].power_15min.reindex(origins+pd.Timedelta(minutes=15*h)-pd.Timedelta(days=1)).to_numpy() for h in range(1,5)])
        else:
            raise ValueError(f'지원하지 않는 모델: {kind}')
        frame=common.prediction_frame(actual,predicted,origins,data['features']).assign(candidate=candidate,period='august',scenario=proxy.SCENARIO)
        for col in data['tags']:
            frame[col]=np.repeat(data['tags'].loc[origins,col].to_numpy(),4)
        frame['actual_was_repaired']=data['repaired'].reindex(frame.target_end).to_numpy()
        frame.to_csv(artifact/'predictions.csv',index=False,encoding='utf-8-sig')
        horizons,conditions,peaks=[] ,[],[]
        for scope,mask in data['scopes'].items():
            subset=mask.loc[origins].to_numpy(bool)
            if not subset.any():
                raise ValueError(f'{scope}: 유효 평가 원점 없음')
            a,p=actual[subset],predicted[subset]
            chosen=frame.loc[np.repeat(subset,4)]
            record['metrics'].append(dict(candidate=candidate,scope=scope,n_excluded_origins=len(origins)-int(subset.sum()),
                n_unique_target_times=int(chosen.target_end.nunique()),**common.risk_scores(a,p)))
            for h in range(4):
                horizons.append(dict(candidate=candidate,scope=scope,horizon_minutes=15*(h+1),**common.risk_scores(a[:,h:h+1],p[:,h:h+1])))
            for col,labels in [('is_night',['day','night']),('is_workday',['non_workday','workday'])]:
                for value,label in enumerate(labels):
                    part=chosen.loc[chosen[col].eq(value)]
                    if len(part):
                        conditions.append(dict(candidate=candidate,scope=scope,dimension=col,condition=label,
                            mean_signed_error=float((part.predicted-part.actual).mean()),
                            **common.risk_scores(part.actual.to_numpy()[:,None],part.predicted.to_numpy()[:,None])))
            for day in range(7):
                part=chosen.loc[chosen.target_start.dt.dayofweek.eq(day)]
                if len(part):
                    conditions.append(dict(candidate=candidate,scope=scope,dimension='weekday',condition=f'day_{day+1}',
                        mean_signed_error=float((part.predicted-part.actual).mean()),
                        **common.risk_scores(part.actual.to_numpy()[:,None],part.predicted.to_numpy()[:,None])))
            unique=((a==a.max(axis=1)[:,None]).sum(axis=1)==1)&((p==p.max(axis=1)[:,None]).sum(axis=1)==1)
            peaks.append(dict(candidate=candidate,scope=scope,n_origins=len(a),
                path_max_MAE=float(np.abs(a.max(axis=1)-p.max(axis=1)).mean()),
                path_max_under_RMSE=float(np.sqrt(np.mean(np.maximum(0,a.max(axis=1)-p.max(axis=1))**2))),
                first_argmax_timing_MAE_minutes=float(np.abs(a.argmax(axis=1)-p.argmax(axis=1)).mean()*15),
                actual_max_tie_origins=int(((a==a.max(axis=1)[:,None]).sum(axis=1)>1).sum()),
                n_unique_max_timing_origins=int(unique.sum()),
                unique_max_timing_MAE_minutes=float(np.abs(a[unique].argmax(axis=1)-p[unique].argmax(axis=1)).mean()*15) if unique.any() else np.nan))
        for name,rows in [('scope_metrics',record['metrics']),('horizon_metrics',horizons),('condition_metrics',conditions),('peak_path_metrics',peaks)]:
            pd.DataFrame(rows).to_csv(artifact/f'{name}.csv',index=False,encoding='utf-8-sig')
        record['status']='완료'
    except Exception as exc:
        record['status']='중단' if isinstance(exc,TimeoutError) else '실패'
        record['error']=f'{type(exc).__name__}: {exc}'
    record['elapsed_seconds']=perf_counter()-before
    message=record.get('error','없음')
    write_record(record,message)
    (artifact/'status.json').write_text(json.dumps(dict(status=record['status'],new_fits=record['new_fits'],
        elapsed_seconds=record['elapsed_seconds'],error=record.get('error')),ensure_ascii=False,indent=2),encoding='utf-8')
    (artifact/'run.log').write_text(f"{candidate}: {record['status']}, {record['elapsed_seconds']:.3f}s\n{message}\n",encoding='utf-8')
    if record['metrics']:
        print(f"{candidate}: {record['status']}, RMSE={record['metrics'][0]['Loss_A']:.4f}, peak={record['metrics'][0]['L_peak']:.4f}, {record['elapsed_seconds']:.1f}s",flush=True)
    return record


def run_confirmation(budget_seconds=600):
    before=perf_counter()
    suite_id=datetime.now(common.KST).strftime('%Y%m%dT%H%M%S_KST')+'_'+uuid4().hex[:6]+'_rf_august'
    suite=ROOT/'artifacts/random_forest/runs'/suite_id
    suite.mkdir(parents=True,exist_ok=False)
    frozen=json.loads((SELECTION/'selected_config.json').read_text(encoding='utf-8'))
    baseline=json.loads((SELECTION/'c001/config.json').read_text(encoding='utf-8'))
    for key,path in [('source_sha256',common.SOURCE),('shared_code_sha256',common.__file__),
                     ('proxy_code_sha256',proxy.__file__),('code_sha256',grid.__file__)]:
        if common.digest(path)!=frozen[key]:
            raise ValueError(f'동결 설정의 {key}와 현재 파일이 다릅니다.')
    if frozen['spec']['window_hours']!=baseline['spec']['window_hours']:
        raise ValueError('8월 비교의 공통 입력 민감도는 동일 통계 window를 요구합니다.')
    df,repaired=load_august_data()
    data=prepare_august(df,repaired,frozen['spec']['window_hours'],frozen['features'])
    data.update(source=str(common.SOURCE),source_sha256=common.digest(common.SOURCE),legacy=str(LEGACY),legacy_sha256=common.digest(LEGACY))
    split=dict(n_train=int(data['train'].sum()),n_valid=int(data['valid'].sum()),
               train_label_end_exclusive=str(START),valid_label_end_exclusive=str(END),
               n_repaired_power_before_september=int(repaired.sum()),
               scopes={name:dict(n_origins=int(mask.sum()),n_predictions=int(mask.sum())*4,
                       n_excluded_origins=int(data['valid'].sum()-mask.sum())) for name,mask in data['scopes'].items()})
    config=dict(suite_id=suite_id,status='실행 중',started=datetime.now(common.KST).isoformat(),budget_seconds=budget_seconds,
                n_planned_rf_fits=2,rf_jobs=4,selection_source=str(SELECTION),september_evaluated=False,
                tuning_after_august=False,split=split,scope_rules=SCOPES,
                selected_params=frozen['rf_params'],baseline_params=baseline['rf_params'],
                source_sha256=data['source_sha256'],legacy_sha256=data['legacy_sha256'])
    (suite/'suite_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    data['tags'].loc[data['valid']].assign(**{f'keep_{name}':mask.loc[data['valid']] for name,mask in data['scopes'].items()}).to_csv(suite/'origin_masks.csv',index_label='origin',encoding='utf-8-sig')
    df.loc[repaired,['timestamp','timestamp_end']].to_csv(suite/'repaired_power_times.csv',index=False,encoding='utf-8-sig')
    print(f'SUITE {suite}\nPREDECLARED_SPLIT {json.dumps(split)}',flush=True)
    records=[]
    for name,kind,params,reference in [('RF_selected','random_forest',frozen['rf_params'],SELECTION/'selected_config.json'),
        ('RF_original','random_forest',baseline['rf_params'],SELECTION/'c001/config.json'),
        ('persistence','persistence',{},None),('daily','daily',{},None)]:
        print(f'START {name}: {json.dumps(params)}',flush=True)
        record=evaluate_candidate(name,kind,params,data,suite,deadline=before+budget_seconds,reference_config=reference)
        records.append(record)
        if record['status']!='완료':
            break
    metrics=pd.DataFrame([row for record in records for row in record['metrics']])
    metrics.to_csv(suite/'comparison_scopes.csv',index=False,encoding='utf-8-sig')
    config.update(status='완료' if len(records)==4 and all(r['status']=='완료' for r in records) else '미완료',
                  ended=datetime.now(common.KST).isoformat(),elapsed_seconds=perf_counter()-before,
                  n_new_rf_fits=sum(r['new_fits'] for r in records),n_candidate_records=len(records))
    (suite/'suite_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    index=ROOT/'experiments/README.md'
    with index.open('a',encoding='utf-8') as stream:
        stream.write(f'\n## {suite_id}: RF 동결 설정8월 확인\n\n1~7월 학습·8월 확인,9월 미평가. 8월 재튜닝 없음. \n\n')
        for record in records:
            stream.write(f"- [{record['config']['candidate']}]({record['record'].relative_to(ROOT/'experiments').as_posix()}): {record['status']}\n")
        if len(metrics):
            stream.write('\n'+common.md_table(metrics[['candidate','scope','n_origins','Loss_A','L_peak','Loss_C']])+'\n')
    if config['status']!='완료':
        raise RuntimeError(f'8월 확인 미완료: {records[-1].get("error","예산 또는 실행 실패")}; {suite}')
    return dict(output_dir=suite,records=records,metrics=metrics,config=config)


if __name__=='__main__':
    run_confirmation()
