"""현재 데이터 RF의 단계별 시간순 그리드 탐색과 후보별 결과 보존."""
from pathlib import Path
from datetime import datetime
from time import perf_counter
from importlib.metadata import version
from uuid import uuid4
import gc
import json
import platform
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

import rf_current_data as common
import rf_future_proxy as proxy

ROOT = common.ROOT
REFERENCE = ROOT/'artifacts/random_forest/runs/20261003T134710_KST_97e184_F2_future_proxy'
METRICS = ['Loss_A', 'MAE', 'L_mean', 'L_CVaR', 'L_peak', 'Loss_B', 'Loss_C']
DEFAULT = dict(min_samples_leaf=2, max_features=1.0, max_depth=None,
               window_hours=1, n_estimators=200, random_state=42)
SPEC_KEYS = tuple(DEFAULT)


def build_window_features(df, hours):
    if hours not in [1, 4, 24]:
        raise ValueError(f'지원 입력 통계 구간은 1/4/24h입니다: {hours}')
    features, targets, _, columns = proxy.build_proxy_features(df)
    old = ['power_1h_mean', 'power_1h_max', 'power_1h_std']
    new = [f'power_{hours}h_{stat}' for stat in ['mean', 'max', 'std']]
    rolling = df.power_15min.astype(float).rolling(hours*4)
    for stat, col in zip(['mean', 'max', 'std'], new):
        features[col] = rolling.std(ddof=0) if stat == 'std' else getattr(rolling, stat)()
    columns = [new[old.index(c)] if c in old else c for c in columns]
    return features, targets, columns


def prepare_data(source=common.SOURCE):
    df = common.load_data(source)
    windows, all_columns = {}, []
    base, targets, _ = build_window_features(df, 1)
    for hours in [1, 4, 24]:
        features, y, columns = build_window_features(df, hours)
        pd.testing.assert_frame_equal(targets, y)
        windows[hours] = dict(features=features, columns=columns, X_train={}, X_valid={})
        for col in columns:
            base[col] = features[col]
        all_columns.extend(columns)
    support = list(dict.fromkeys(all_columns))
    folds = {}
    for fold in common.FOLDS:
        train, valid = common.fold_masks(base, targets, support, fold)
        prior_train, prior_valid = common.fold_masks(base, targets, windows[1]['columns'], fold)
        if not np.array_equal(train, prior_train) or not np.array_equal(valid, prior_valid):
            raise ValueError(f'{fold}: 윈도우 변경으로 기존 F2 지원 원점이 달라졌습니다.')
        folds[fold] = dict(train_origins=base.index[train], origins=base.index[valid],
                           y_train=targets.loc[train], actual=targets.loc[valid].to_numpy())
        for window in windows.values():
            window['X_train'][fold] = window['features'].loc[train, window['columns']]
            window['X_valid'][fold] = window['features'].loc[valid, window['columns']]
    return dict(df=df, windows=windows, folds=folds, source=str(Path(source).resolve()),
                source_sha256=common.digest(source))


def spec_key(spec):
    return tuple(spec[k] for k in SPEC_KEYS)


def choose_best(metrics, eligible):
    rows = []
    for candidate, part in metrics.loc[metrics.candidate.isin(eligible)].groupby('candidate'):
        if len(part) != len(common.FOLDS) or set(part.fold) != set(common.FOLDS):
            continue
        rows.append(dict(candidate=candidate,
                         runtime=part.fit_predict_save_seconds.mean() if 'fit_predict_save_seconds' in part else np.inf,
                         **part[['Loss_A', 'L_peak', 'Loss_C']].mean().to_dict()))
    if not rows:
        raise ValueError('완료된 4fold 후보가 없어 설정을 선택할 수 없습니다.')
    return pd.DataFrame(rows).sort_values(['Loss_A', 'L_peak', 'Loss_C', 'runtime', 'candidate']).iloc[0].candidate


def write_record(record, status, metrics, elapsed, message='없음'):
    cfg = record['config']
    columns = ['fold', 'n_train', 'n_origins', *METRICS]
    table = common.md_table(pd.DataFrame(metrics)[columns]) if metrics else '실제 완료 fold 없음.'
    record['record'].write_text(f"""# 실험 {cfg['experiment_id']}

- 상태: {status}
- 목적과 가설: RF 베이스라인의 단계별 그리드 탐색. 입력 통계 구간과 핵심 파라미터의 효과를 동일 원점에서 비교한다.
- 시작·종료 시각 및 시간대: {record['started']} / {datetime.now(common.KST).isoformat() if status != '실행 중' else '미완료'}; Asia/Seoul.
- 모델·라이브러리 버전: RandomForestRegressor; {cfg['versions']}.
- 데이터 위치·버전 또는 해시·사용 범위: {cfg['source']}; SHA256 {cfg['source_sha256']}; 종료시각 2021-01-01~2021-07-31 중 유효 원점.
- 목표값·단위·예측 구간: power_15min kW, 미래1h의15min4점. 원점은 최신 완료 전력의 종료시각.
- 학습·검증·테스트 분할과 누수 점검: 4~7월 expanding; 학습 정답 종료<검증 시작, 검증 정답 종료<다음 월. 모든 후보 동일24h 지원 원점. 미래 전력 입력 금지. 8월·9월 모델 평가 없음.
- 전처리·특성·하이퍼파라미터·seed: 원단위, scaler 없음, 미래 외생5종×4점의 오차0 시나리오 유지. 입력70열. 최근 통계구간 {cfg['spec']['window_hours']}h. {cfg['rf_params']}.
- 이전 또는 기준 실험 링크와 변경 사항: [튜닝 범위](../../notebooks/models/random_forest/BASELINE_TUNING_SCOPE.md); [기존 미래 외생 F2](20261003T134710_KST_97e184_F2_future_proxy.md). 재사용 출처: {cfg['reuse_run'] or '새 학습'}.
- 실행 명령·환경·코드 버전 또는 코드 식별 정보: Jupyter rf_grid_search.ipynb Run All; Python {platform.python_version()}; 코드 해시 {cfg['code_sha256']}; 공통 코드 해시 {cfg['shared_code_sha256']}; proxy 코드 해시 {cfg['proxy_code_sha256']}; Git 없는 workspace.
- 실제 실행 시간·자원 사용량(측정한 경우): {elapsed:.3f}초; CPU worker 최대4, 후보·fold 순차. 메모리 미측정.
- 평가 지표·구간별 또는 fold별 결과: 아래 표, 단위kW. Loss A 주지표, 월별 동일 가중 평균. 미래4점=1h window, stride15min, CVaR 상위ceil20%, L_peak 전체 유효 셀의 가중 과소예측 제곱 평균 후 마지막 root; 초기w=1.
- 기준모델 대비 변화와 비교 가능 여부: 같은 데이터·목표·지원 원점의 현재 F2와 비교. 입력 구간·표의 파라미터만 변경. 재사용은 해시·입력 열·원점·정답·저장 모델 예측 재현을 검증하고 표시한다.
- 오차가 큰 조건·해석·한계: horizon/condition/peak 표 참조. 미래 외생 실측을 예보로 간주한 오차0 조건부 결과. 중첩 원점은 독립 표본이 아님. RF 학습은 squared_error, Loss B/C 직접 최적화 없음. seed를 성능에 맞춰 선택하지 않음.
- 오류·경고·실패 또는 중단 원인: {message}.
- 예측·모델·표·그래프·로그의 저장 경로: {record['artifact']}; config.json, status.json, predictions.csv, fold_metrics.csv, horizon_metrics.csv, condition_metrics.csv, peak_path_metrics.csv, run.log, model_<fold>.joblib.
- 결론과 다음 실험: 완료된4fold만 후보 선택에 사용. 최종 RF 설정을 보존하고8월 별도 확인·9월 최종 테스트는 후속 단계로 둔다.

{table}
""", encoding='utf-8')


def evaluate_candidate(spec, data, suite_dir, candidate, root=ROOT, deadline=None, reuse_run=None, stage='custom'):
    root, suite_dir = Path(root), Path(suite_dir)
    artifact = suite_dir/candidate
    artifact.mkdir(parents=True, exist_ok=False)
    records = root/'experiments/random_forest'
    records.mkdir(parents=True, exist_ok=True)
    params = dict(n_jobs=4, criterion='squared_error', bootstrap=True, max_samples=None,
                  min_samples_split=2, **{k: v for k, v in spec.items() if k != 'window_hours'})
    window = data['windows'][spec['window_hours']]
    config = dict(experiment_id=f'{suite_dir.name}_{candidate}', candidate=candidate, stage=stage,
                  spec=spec, rf_params=params, features=window['columns'], source=data['source'],
                  source_sha256=data['source_sha256'], code_sha256=common.digest(__file__),
                  shared_code_sha256=common.digest(common.__file__),
                  proxy_code_sha256=common.digest(proxy.__file__),
                  versions={pkg: version(pkg) for pkg in ['numpy','pandas','scikit-learn','joblib','nbclient']},
                  scenario=proxy.SCENARIO, forecast_error_assumption=0, folds=common.FOLDS,
                  window_minutes=60, stride_minutes=15, peak_weights=1,
                  reuse_run=str(reuse_run) if reuse_run else None,
                  reused_model_runtime_change='n_jobs -1 to4; learned trees unchanged' if reuse_run else None)
    record = dict(config=config, artifact=artifact, record=records/f'{config["experiment_id"]}.md',
                  started=datetime.now(common.KST).isoformat(), metrics=[], new_fits=0, reused_folds=0,
                  status='실행 중')
    (artifact/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    write_record(record, '실행 중', [], 0.)
    before = perf_counter()
    predictions, horizons, conditions, peaks, logs = [], [], [], [], []
    cached, cached_cfg, cached_metrics = None, None, None
    try:
        if reuse_run:
            cached_cfg = json.loads((Path(reuse_run)/'config.json').read_text(encoding='utf-8'))
            for key in ['source_sha256', 'shared_code_sha256']:
                if config[key] != cached_cfg[key]:
                    raise ValueError(f'기존 F2 재사용 {key} 불일치')
            if config['proxy_code_sha256'] != cached_cfg['code_sha256']:
                raise ValueError('기존 F2 재사용 proxy 코드 불일치')
            if spec_key(spec) != spec_key(DEFAULT) or config['features'] != cached_cfg['features']:
                raise ValueError('현재 기준 설정만 동일 입력으로 재사용할 수 있습니다.')
            cached = pd.read_csv(Path(reuse_run)/'predictions.csv', parse_dates=['origin','target_end'])
            cached_metrics = pd.read_csv(Path(reuse_run)/'comparison_folds.csv')
        for fold, fold_data in data['folds'].items():
            if deadline is not None and perf_counter() >= deadline:
                raise TimeoutError('45분 탐색 예산 또는 지정 실행 예산 소진')
            origins, actual = fold_data['origins'], fold_data['actual']
            started = perf_counter()
            if reuse_run:
                model = joblib.load(Path(reuse_run)/f'model_{fold}.joblib')
                model.set_params(n_jobs=4)
                for key, value in params.items():
                    if model.get_params()[key] != value:
                        raise ValueError(f'{fold}: 기준 모델 파라미터 {key} 불일치')
                old = cached.loc[cached.fold.eq(fold)].sort_values(['origin','horizon_minutes'])
                np.testing.assert_array_equal(old.origin.drop_duplicates().to_numpy(), origins.to_numpy())
                expected_ends = origins.repeat(4)+pd.to_timedelta(np.tile([15,30,45,60], len(origins)), unit='min')
                np.testing.assert_array_equal(old.target_end.to_numpy(), expected_ends.to_numpy())
                np.testing.assert_array_equal(old.actual.to_numpy().reshape(-1,4), actual)
                prior = cached_metrics.loc[(cached_metrics.fold == fold) &
                                           (cached_metrics.candidate == 'F2_future_proxy')].iloc[0]
                if prior.n_train != len(fold_data['train_origins']):
                    raise ValueError(f'{fold}: 기준 모델 학습 원점 수 불일치')
                predicted = old.predicted.to_numpy().reshape(-1,4)
                np.testing.assert_allclose(model.predict(window['X_valid'][fold]), predicted, rtol=1e-12, atol=1e-9)
                record['reused_folds'] += 1
            else:
                model = RandomForestRegressor(**params)
                model.fit(window['X_train'][fold], fold_data['y_train'])
                predicted = model.predict(window['X_valid'][fold])
                record['new_fits'] += 1
            joblib.dump(model, artifact/f'model_{fold}.joblib', compress=3)
            elapsed = perf_counter()-started
            scores = dict(candidate=candidate, fold=fold, n_train=len(fold_data['train_origins']),
                          train_first=str(fold_data['train_origins'][0]),
                          train_last_origin=str(fold_data['train_origins'][-1]), valid_first=str(origins[0]),
                          valid_last_origin=str(origins[-1]), fit_predict_save_seconds=elapsed,
                          **common.risk_scores(actual, predicted))
            record['metrics'].append(scores)
            frame = common.prediction_frame(actual, predicted, origins, window['features']).assign(
                fold=fold, candidate=candidate, scenario=proxy.SCENARIO)
            predictions.append(frame)
            for h in range(4):
                horizons.append(dict(candidate=candidate, fold=fold, horizon_minutes=(h+1)*15,
                                     **common.risk_scores(actual[:,h:h+1], predicted[:,h:h+1])))
            for column, labels in [('is_night',['day','night']),('is_workday',['non_workday','workday'])]:
                for value, label in enumerate(labels):
                    part = frame.loc[frame[column].eq(value)]
                    if len(part):
                        conditions.append(dict(candidate=candidate, fold=fold, condition=label,
                                               **common.risk_scores(part.actual.to_numpy()[:,None],
                                                                    part.predicted.to_numpy()[:,None])))
            peaks.append(dict(candidate=candidate, fold=fold, n_origins=len(origins),
                              path_max_MAE=float(np.abs(actual.max(axis=1)-predicted.max(axis=1)).mean()),
                              path_max_under_RMSE=float(np.sqrt(np.mean(np.maximum(0,actual.max(axis=1)-predicted.max(axis=1))**2))),
                              argmax_timing_MAE_minutes=float(np.abs(actual.argmax(axis=1)-predicted.argmax(axis=1)).mean()*15),
                              actual_max_tie_origins=int(((actual==actual.max(axis=1)[:,None]).sum(axis=1)>1).sum())))
            line = f'{candidate}/{fold}: RMSE={scores["Loss_A"]:.4f}, peak={scores["L_peak"]:.4f}, {elapsed:.1f}s'
            logs.append(line)
            print(line, flush=True)
            for name, items in [('fold_metrics',record['metrics']),('horizon_metrics',horizons),
                                ('condition_metrics',conditions),('peak_path_metrics',peaks)]:
                pd.DataFrame(items).to_csv(artifact/f'{name}.csv', index=False, encoding='utf-8-sig')
            pd.concat(predictions, ignore_index=True).to_csv(artifact/'predictions.csv', index=False, encoding='utf-8-sig')
            (artifact/'run.log').write_text('\n'.join(logs)+'\n', encoding='utf-8')
            del model
            gc.collect()
        record['status'] = '완료'
    except Exception as exc:
        record['status'] = '중단' if isinstance(exc, TimeoutError) else '실패'
        record['error'] = f'{type(exc).__name__}: {exc}'
        (artifact/'run.log').write_text('\n'.join(logs)+'\n'+record['error']+'\n', encoding='utf-8')
        write_record(record, record['status'], record['metrics'], perf_counter()-before, record['error'])
        (artifact/'status.json').write_text(json.dumps(dict(status=record['status'], new_fits=record['new_fits'],
            reused_folds=record['reused_folds'], error=record['error']), ensure_ascii=False, indent=2), encoding='utf-8')
        if not isinstance(exc, TimeoutError):
            raise
    record['elapsed_seconds'] = perf_counter()-before
    write_record(record, record['status'], record['metrics'], record['elapsed_seconds'], record.get('error','없음'))
    (artifact/'status.json').write_text(json.dumps(dict(status=record['status'], new_fits=record['new_fits'],
        reused_folds=record['reused_folds'], elapsed_seconds=record['elapsed_seconds'],
        error=record.get('error')), ensure_ascii=False, indent=2), encoding='utf-8')
    return record


def run_search(budget_seconds=2700):
    before = perf_counter()
    suite_id = datetime.now(common.KST).strftime('%Y%m%dT%H%M%S_KST')+'_'+uuid4().hex[:6]+'_rf_grid'
    suite = ROOT/'artifacts/random_forest/runs'/suite_id
    suite.mkdir(parents=True, exist_ok=False)
    suite_cfg = dict(suite_id=suite_id, status='실행 중', budget_seconds=budget_seconds,
                     search='staged grids, 23 conditions maximum', candidate_workers=1, rf_jobs=4,
                     purpose='RF baseline performance with bounded resource use',
                     august_september='not evaluated', started=datetime.now(common.KST).isoformat())
    (suite/'search_config.json').write_text(json.dumps(suite_cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'SUITE {suite}', flush=True)
    data = prepare_data()
    cache, records = {}, []
    stopped = False
    baseline_id, selected_id = None, None

    def metric_frame():
        return pd.DataFrame([row for record in records for row in record['metrics']])

    def best():
        eligible = [r['config']['candidate'] for r in records
                    if r['status']=='완료' and r['config']['spec']['random_state']==42]
        return choose_best(metric_frame(), eligible)

    def evaluate(spec, stage):
        nonlocal stopped
        key = spec_key(spec)
        if key in cache:
            return cache[key]
        candidate = f'c{len(records)+1:03d}'
        print(f'START {candidate} {stage}: {json.dumps(spec)}', flush=True)
        record = evaluate_candidate(spec, data, suite, candidate, deadline=before+budget_seconds,
                                    reuse_run=REFERENCE if spec_key(spec)==spec_key(DEFAULT) else None, stage=stage)
        records.append(record)
        cache[key] = record
        stopped = record['status'] != '완료'
        (suite/'progress.json').write_text(json.dumps(dict(candidate=candidate, stage=stage,
            completed_candidates=sum(r['status']=='완료' for r in records),
            completed_folds=sum(len(r['metrics']) for r in records),
            new_fits=sum(r['new_fits'] for r in records), elapsed_seconds=perf_counter()-before,
            latest_status=record['status']), ensure_ascii=False, indent=2), encoding='utf-8')
        metric_frame().to_csv(suite/'comparison_folds.csv', index=False, encoding='utf-8-sig')
        return record

    baseline = evaluate(dict(DEFAULT), 'reference')
    baseline_id = baseline['config']['candidate']
    for values in ParameterGrid(dict(min_samples_leaf=[1,2,5,10], max_features=[1/3,0.5,1.0])):
        if stopped:
            break
        evaluate(dict(DEFAULT, **values), 'leaf_features')
    selected_id = best()
    selected_spec = next(r['config']['spec'] for r in records if r['config']['candidate']==selected_id)
    print(f'STAGE1 BEST {selected_id}: {selected_spec}', flush=True)
    for values in ParameterGrid(dict(max_depth=[12,20,None], window_hours=[1,4,24])):
        if stopped:
            break
        evaluate(dict(selected_spec, **values), 'depth_window')
    selected_id = best()
    selected_spec = next(r['config']['spec'] for r in records if r['config']['candidate']==selected_id)
    print(f'STAGE2 BEST {selected_id}: {selected_spec}', flush=True)
    if not stopped:
        evaluate(dict(selected_spec, n_estimators=500), 'tree_count')
    selected_id = best()
    selected_spec = next(r['config']['spec'] for r in records if r['config']['candidate']==selected_id)
    # seed42로 설정을 동결한 뒤 다른 seed는 안정성 확인에만 사용한다.
    for seed in [7, 2026]:
        if stopped:
            break
        evaluate(dict(selected_spec, random_state=seed), 'seed_stability')
    return finish_search(suite, records, selected_id, baseline_id, before, stopped, data)


def finish_search(suite, records, selected_id, baseline_id, started, stopped, data):
    folds = pd.DataFrame([row for r in records for row in r['metrics']])
    rows = []
    for record in records:
        cfg = record['config']
        row = dict(candidate=cfg['candidate'], stage=cfg['stage'], status=record['status'],
                   n_complete_folds=len(record['metrics']), new_fits=record['new_fits'],
                   reused_folds=record['reused_folds'], elapsed_seconds=record['elapsed_seconds'],
                   **cfg['spec'])
        if record['status']=='완료':
            row.update(pd.DataFrame(record['metrics'])[METRICS].mean().to_dict())
        rows.append(row)
    macro = pd.DataFrame(rows)
    macro.to_csv(suite/'comparison_macro.csv', index=False, encoding='utf-8-sig')
    selected = next(r for r in records if r['config']['candidate']==selected_id)
    baseline = next(r for r in records if r['config']['candidate']==baseline_id)
    summary = dict(status='예산 소진으로 탐색 종료' if stopped else '완료',
                   elapsed_seconds=perf_counter()-started, selected_candidate=selected_id,
                   selected_spec=selected['config']['spec'], baseline_candidate=baseline_id,
                   n_candidate_records=len(records), n_new_fits=sum(r['new_fits'] for r in records),
                   n_reused_folds=sum(r['reused_folds'] for r in records),
                   completed_candidates=sum(r['status']=='완료' for r in records),
                   source_sha256=data['source_sha256'], code_sha256=common.digest(__file__),
                   shared_code_sha256=common.digest(common.__file__), proxy_code_sha256=common.digest(proxy.__file__),
                   august_september_evaluated=False, seed_selection='42 fixed; 7/2026 stability only',
                   baseline_reuse_source=str(REFERENCE))
    (suite/'search_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (suite/'selected_config.json').write_text(json.dumps(selected['config'], ensure_ascii=False, indent=2), encoding='utf-8')
    index = ROOT/'experiments/README.md'
    text = f'\n## {suite.name}: RF 단계별 그리드 탐색\n\n4~7월 검증, 8·9월 미평가. 월별 동일 가중 평균, 단위kW. 최종 선정 {selected_id}; seed42 고정.\n\n'
    for record in records:
        relative = record['record'].relative_to(ROOT/'experiments').as_posix()
        text += f'- [{record["config"]["candidate"]}]({relative}): {record["status"]}, {record["config"]["spec"]}\n'
    text += '\n'+common.md_table(macro[['candidate','status','window_hours','min_samples_leaf',
                                      'max_features','max_depth','n_estimators','random_state','Loss_A','L_peak','Loss_C']])+'\n'
    text += f'\n산출물: {suite}. 단계 탐색·중첩 원점·오차0 외생 시나리오의 한계는 후보 기록 참조.\n'
    with index.open('a', encoding='utf-8') as stream:
        stream.write(text)
    print(f'SELECTED {selected_id}: {selected["config"]["spec"]}', flush=True)
    print(f'SAVED {suite}', flush=True)
    return dict(output_dir=suite, macro=macro, fold_metrics=folds,
                summary=summary, selected=selected, baseline=baseline)


if __name__ == '__main__':
    run_search()
