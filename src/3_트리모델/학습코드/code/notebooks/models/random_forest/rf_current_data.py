"""현재 전처리 데이터의 인과적 RF 피처 실험. 기존 노트북 공통 계약은 보존한다."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from time import perf_counter
from importlib.metadata import version
import argparse
import hashlib
import json
import platform
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'notebooks/_shared'))
from forecast_common import FOLDS

KST = timezone(timedelta(hours=9))
OBSERVED = ['생산량', '기온', '풍속', '습도', '강수량']
SOURCE = ROOT / 'data/processed/preprocessed_20261002/timeseries_15min_v1_previous_week.csv'
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=2, random_state=42,
                 n_jobs=-1, criterion='squared_error')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path=SOURCE):
    df = pd.read_csv(path, parse_dates=['timestamp', 'timestamp_end'])
    required = ['timestamp', 'timestamp_end', 'power_15min', 'day_type', *OBSERVED]
    missing = set(required) - set(df)
    if missing:
        raise ValueError(f'필수 컬럼 누락: {sorted(missing)}')
    if not (df.timestamp_end - df.timestamp).eq(pd.Timedelta(minutes=15)).all():
        raise ValueError('timestamp는 구간 시작, timestamp_end는 15분 후 종료여야 합니다.')
    df = df.set_index('timestamp_end', drop=False)
    validate_frame(df)
    if not np.isfinite(df[['power_15min', *OBSERVED]].to_numpy(dtype=float)).all():
        raise ValueError('현재 실험은 전력과 관측 피처의 유한한 값이 필요합니다. 자동 대체하지 않습니다.')
    if not df.groupby(df.timestamp.dt.floor('h'))[OBSERVED].nunique().eq(1).all().all():
        raise ValueError('시간별로 반복되는 관측 피처가 일치하지 않습니다.')
    if not df.day_type.isin([0, 1, 2, 3]).all():
        raise ValueError('day_type은 0/1/2/3이어야 합니다.')
    # 실행에서 8월 확인·9월 테스트 목표값은 사용하지 않는다.
    return df.loc[df.index < pd.Timestamp('2021-08-01')].copy()


def validate_frame(df):
    idx = df.index
    if (not isinstance(idx, pd.DatetimeIndex) or idx.hasnans or idx.has_duplicates
            or not idx.is_monotonic_increasing
            or not idx.to_series().diff().dropna().eq(pd.Timedelta(minutes=15)).all()):
        raise ValueError('중복·누락 없는 오름차순 15분 종료시각이 필요합니다.')
    if not (df.timestamp.to_numpy() + np.timedelta64(15, 'm') == idx.to_numpy()).all():
        raise ValueError('시작시각과 종료 인덱스 정렬이 맞지 않습니다.')


def build_features(df):
    validate_frame(df)
    idx = df.index
    power = df.power_15min.astype(float)
    features = pd.DataFrame(index=idx)
    # 시간 구간 [hour, hour+1h)가 완료된 뒤 사용할 수 있다고 가정한다.
    hourly = df.loc[df.timestamp.dt.minute.eq(0)].set_index('timestamp')[OBSERVED]
    source_hour = idx.floor('h') - pd.Timedelta(hours=1)
    observed = hourly.reindex(source_hour)
    for col in OBSERVED:
        features[f'observed_{col}'] = observed[col].to_numpy()
    base = list(features)
    time_cols, daily_cols, weekly_cols = [], [], []
    targets = pd.DataFrame(index=idx)
    for h in range(1, 5):
        start = idx + pd.Timedelta(minutes=15*(h-1))
        for day in range(1, 8):
            name = f'h{h}_day_{day}'
            features[name] = (start.dayofweek+1 == day).astype(float)
            base.append(name)
        for name, values in [('month', start.month), ('night', (start.hour >= 18) | (start.hour < 9)),
                             ('workday', df.day_type.shift(-h).eq(0))]:
            col = f'h{h}_{name}'
            features[col] = np.asarray(values, dtype=float)
            base.append(col)
        name = f'h{h}_quarter'
        features[name] = start.hour*4+start.minute//15
        time_cols.append(name)
        targets[f'h{h}'] = power.shift(-h)
        for period, offset, collection in [('day', 96, daily_cols), ('week', 672, weekly_cols)]:
            col = f'h{h}_previous_{period}'
            features[col] = power.shift(offset-h)
            collection.append(col)
    recent_cols = []
    for name, offset in [('latest', 0), ('lag_1h', 4), ('lag_24h', 96)]:
        col = f'power_{name}'
        features[col] = power.shift(offset)
        recent_cols.append(col)
    for label, series in [('mean', power.rolling(4).mean()), ('max', power.rolling(4).max()),
                          ('std', power.rolling(4).std(ddof=0))]:
        col = f'power_1h_{label}'
        features[col] = series
        recent_cols.append(col)
    extra_cols = []
    for name, offset in [('lag_15min', 1), ('lag_30min', 2), ('lag_7d', 672)]:
        col = f'power_{name}'
        features[col] = power.shift(offset)
        extra_cols.append(col)
    for label, steps in [('4h', 16), ('24h', 96)]:
        for stat in ['mean', 'max', 'std']:
            col = f'power_{label}_{stat}'
            rolling = power.rolling(steps)
            features[col] = rolling.std(ddof=0) if stat == 'std' else getattr(rolling, stat)()
            extra_cols.append(col)
    for label, steps in [('15min', 1), ('1h', 4)]:
        col = f'power_change_{label}'
        features[col] = power - power.shift(steps)
        extra_cols.append(col)
    groups = {'F0': base, 'F1': base+time_cols, 'F2': base+time_cols+recent_cols,
              'F3': base+time_cols+recent_cols+extra_cols+daily_cols+weekly_cols}
    return features, targets, groups


def fold_masks(features, targets, support_columns, fold):
    start, end = map(pd.Timestamp, FOLDS[fold])
    complete = np.isfinite(features[support_columns].to_numpy()).all(axis=1)
    complete &= np.isfinite(targets.to_numpy()).all(axis=1)
    origins = features.index
    ends = origins + pd.Timedelta(hours=1)
    train = complete & (ends < start)
    valid = complete & (origins >= start) & (ends < end)
    if not train.any() or not valid.any():
        raise ValueError(f'{fold}: 학습/검증 유효 원점이 없습니다.')
    return train, valid


def risk_scores(actual, predicted, weights=1.0):
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.ndim != 2 or a.shape != p.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError('실제/예측은 동일 크기의 비어 있지 않은 유한한 2차원 경로여야 합니다.')
    w = np.broadcast_to(np.asarray(weights, dtype=float), a.shape)
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('피크 가중치는 유한한 비음수여야 합니다.')
    error = a-p
    windows = np.sqrt(np.mean(error**2, axis=1))
    tail_n = max(1, int(np.ceil(len(windows)*.2)))
    mean = float(windows.mean())
    cvar = float(np.sort(windows)[-tail_n:].mean())
    peak = float(np.sqrt(np.mean(w*np.maximum(0, error)**2)))
    return dict(Loss_A=float(np.sqrt(np.mean(error**2))), MAE=float(np.mean(np.abs(error))),
                L_mean=mean, L_CVaR=cvar, L_peak=peak, Loss_B=.6*mean+.4*cvar,
                Loss_C=.5*mean+.3*cvar+.2*peak, tail_windows=tail_n,
                n_origins=len(a), n_predictions=a.size)


def md_table(frame, digits=4):
    def val(x):
        return f'{x:.{digits}f}' if isinstance(x, (float, np.floating)) else str(x)
    cols = list(frame.columns)
    return '\n'.join(['| '+' | '.join(cols)+' |', '| '+' | '.join(['---']*len(cols))+' |'] +
                     ['| '+' | '.join(val(x) for x in row)+' |' for row in frame.itertuples(index=False, name=None)])


def prediction_frame(actual, predicted, origins, features):
    n, h = actual.shape
    distance = np.tile(np.arange(1, h+1)*15, n)
    ends = origins.repeat(h) + pd.to_timedelta(distance, unit='min')
    return pd.DataFrame(dict(origin=origins.repeat(h), target_start=ends-pd.Timedelta(minutes=15),
        target_end=ends, horizon_minutes=distance, actual=actual.ravel(), predicted=predicted.ravel(),
        is_night=np.column_stack([features.loc[origins, f'h{i}_night'] for i in range(1, 5)]).ravel(),
        is_workday=np.column_stack([features.loc[origins, f'h{i}_workday'] for i in range(1, 5)]).ravel()))


def write_record(record, status, metrics=None, message='', times=None):
    config, path = record['config'], record['md_path']
    table = '' if metrics is None else '\n\n'+md_table(metrics)
    text = f'''# 실험 {config['experiment_id']}

- 상태: {status}
- 목적과 가설: {config['candidate']}의 시계열 피처 추가 효과. {config['comparison_group']} 그룹 안에서 비교.
- 시작·종료 시각 및 시간대: {record['started']} / {datetime.now(KST).isoformat() if status != '실행 중' else '미완료'} (Asia/Seoul).
- 모델·라이브러리 버전: {config['model']}; {json.dumps(config['versions'], ensure_ascii=False)}.
- 데이터 위치·버전 또는 해시·사용 범위: {SOURCE}; SHA256 {config['source_sha256']}; 2021-01-01~2021-07-31 중 분할 조건을 만족하는 표본만 사용.
- 목표값·단위·예측 구간: power_15min, kW, 다음 1시간의 15분 전력 4개. 원점은 마지막 완료 전력의 종료시각.
- 학습·검증·테스트 분할과 누수 점검: 4~7월 expanding folds; 학습 정답 종료 < 검증 시작; 검증 정답 종료 < 다음 달 시작. 8월 별도 확인·9월 최종 테스트 미실행. 관측 외생값은 최근 완료 시간값, 미래 실측 미사용.
- 전처리·특성·하이퍼파라미터·seed: 입력 원단위, scaler 없음; {config['candidate']}; 입력 {len(config['features'])}열; RF {config['rf_params']}; support {config['support']}.
- 이전 또는 기준 실험 링크와 변경 사항: [비교 인덱스](../README.md); [실험 계획](../../notebooks/models/random_forest/EXPERIMENT_PLAN.md). core=24h 공통 이력, weekly=7d 공통 이력.
- 실행 명령·환경·코드 버전 또는 코드 식별 정보: `{config['command']}`; Python {platform.python_version()}; rf_current_data.py SHA256 {config['code_sha256']}; Git 사용하지 않는 workspace.
- 실제 실행 시간·자원 사용량(측정한 경우): {times if times is not None else '미완료'}; 메모리 미측정.
- 평가 지표·구간별 또는 fold별 결과: 아래 표. Loss A가 주 비교 지표. 단위 kW. window=origin별 향후 4점, stride 15min, 중첩; CVaR 최악 ceil(20%×원점수); L_peak 가중치1, 전체 표본 평균 후 root.
- 기준모델 대비 변화와 비교 가능 여부: 동일 그룹 내 표본·정답 동일. 다른 그룹의 train support가 달라 직접 피처 효과 비교 금지. 월별 macro 비교는 인덱스 참조.
- 오차가 큰 조건·해석·한계: 조건별 표와 horizon 표 산출물 참조. 단일 seed, 월 내 고정 모델, 중첩 예측 사례는 독립 표본이 아님. 모든 피크 근접도가 같은 w=1, 실제 B(t) 경보 성능 아님. RF는 squared_error로 학습, B/C 직접 최적화·OOF 재가중 미실행.
- 오류·경고·실패 또는 중단 원인: {message or '기록된 오류 없음'}.
- 예측·모델·표·그래프·로그의 저장 경로: `{record['artifact']}`; predictions.csv, fold_metrics.csv, horizon_metrics.csv, condition_metrics.csv, peak_path_metrics.csv, config.json, run.log. RF model_<fold>.joblib.
- 결론과 다음 실험: 실제 월별/인덱스 결과로 판단. 8월 별도 확인 후 설정을 동결하고 9월 최종 테스트; 다른 예측 길이와 OOF 재가중은 별도 실험.
{table}
'''
    path.write_text(text, encoding='utf-8')


def run_suite(root=ROOT, source=SOURCE, budget_seconds=600, execution_context=None):
    import joblib
    root = Path(root)
    df = load_data(source)
    features, targets, groups = build_features(df)
    suite_id = datetime.now(KST).strftime('%Y%m%dT%H%M%S_KST')+'_'+uuid4().hex[:6]
    suite_dir = root/'artifacts/random_forest/runs'/suite_id
    suite_dir.mkdir(parents=True, exist_ok=False)
    versions = {pkg: version(pkg) for pkg in ['numpy', 'pandas', 'scikit-learn', 'joblib']}
    data_hash, code_hash = digest(source), digest(__file__)
    command = execution_context or f'"{sys.executable}" -B -X utf8 "{Path(__file__).resolve()}" --budget-seconds {budget_seconds}'
    definitions = [('F0', 'F0', 'F2', 'core'), ('F1', 'F1', 'F2', 'core'), ('F2', 'F2', 'F2', 'core'),
                   ('F2_week_support', 'F2', 'F3', 'weekly'), ('F3', 'F3', 'F3', 'weekly')]
    for group, support in [('core', 'F2'), ('weekly', 'F3')]:
        definitions.extend([(f'persistence_{group}', 'persistence', support, group),
                            (f'daily_{group}', 'daily', support, group)])
    all_metrics, records = [], []
    before = perf_counter()
    for candidate, feature_key, support, comparison in definitions:
        model_name = 'random_forest' if feature_key in groups else feature_key
        exp_id = suite_id+'_'+candidate
        artifact = suite_dir/candidate
        artifact.mkdir()
        record_dir = root/'experiments'/model_name
        record_dir.mkdir(parents=True, exist_ok=True)
        config = dict(experiment_id=exp_id, candidate=candidate, model=model_name,
            comparison_group=comparison, source_sha256=data_hash, code_sha256=code_hash,
            versions=versions, command=command, rf_params=RF_PARAMS if model_name=='random_forest' else {},
            features=groups.get(feature_key, []), support=support,
            support_features=groups[support], folds=FOLDS, peak_weights=1,
            window_length_minutes=60, window_stride_minutes=15, tail_rounding='ceil',
            test_period='2021-09-01..2021-09-14 (reserved)', august='reserved confirmation')
        (artifact/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        record = dict(config=config, artifact=artifact, md_path=record_dir/(exp_id+'.md'),
                      started=datetime.now(KST).isoformat())
        records.append(record)
        write_record(record, '실행 중')
        fold_metrics, predictions, horizons, conditions, peaks, logs = [], [], [], [], [], []
        candidate_started = perf_counter()
        try:
            for fold in FOLDS:
                if perf_counter()-before > budget_seconds:
                    raise TimeoutError(f'실험 실행 예산 {budget_seconds}초 초과')
                train, valid = fold_masks(features, targets, groups[support], fold)
                orig = features.index[valid]
                actual = targets.loc[valid].to_numpy()
                started = perf_counter()
                if model_name == 'random_forest':
                    cols = groups[feature_key]
                    model = RandomForestRegressor(**RF_PARAMS)
                    model.fit(features.loc[train, cols], targets.loc[train])
                    predicted = model.predict(features.loc[valid, cols])
                    joblib.dump(model, artifact/f'model_{fold}.joblib', compress=3)
                elif model_name == 'persistence':
                    predicted = np.repeat(df.loc[orig, 'power_15min'].to_numpy()[:, None], 4, axis=1)
                else:
                    predicted = np.column_stack([df.power_15min.reindex(
                        orig+pd.Timedelta(minutes=15*h)-pd.Timedelta(days=1)).to_numpy() for h in range(1, 5)])
                elapsed = perf_counter()-started
                score = dict(candidate=candidate, comparison_group=comparison, fold=fold,
                    n_train=int(train.sum()), train_first=str(features.index[train][0]),
                    train_last_origin=str(features.index[train][-1]), valid_first=str(orig[0]),
                    valid_last_origin=str(orig[-1]), fit_predict_save_seconds=elapsed,
                    **risk_scores(actual, predicted))
                fold_metrics.append(score)
                frame = prediction_frame(actual, predicted, orig, features).assign(fold=fold, candidate=candidate)
                predictions.append(frame)
                for h in range(4):
                    horizons.append(dict(candidate=candidate, fold=fold, horizon_minutes=(h+1)*15,
                        **risk_scores(actual[:, h:h+1], predicted[:, h:h+1])))
                for column, labels in [('is_night', ['day', 'night']), ('is_workday', ['non_workday', 'workday'])]:
                    for value, label in enumerate(labels):
                        selected = frame.loc[frame[column] == value]
                        if len(selected):
                            conditions.append(dict(candidate=candidate, fold=fold, condition=label,
                                **risk_scores(selected.actual.to_numpy()[:, None], selected.predicted.to_numpy()[:, None])))
                peaks.append(dict(candidate=candidate, fold=fold, n_origins=len(orig),
                    path_max_MAE=float(np.abs(actual.max(axis=1)-predicted.max(axis=1)).mean()),
                    path_max_under_RMSE=float(np.sqrt(np.mean(np.maximum(0, actual.max(axis=1)-predicted.max(axis=1))**2))),
                    argmax_timing_MAE_minutes=float(np.abs(actual.argmax(axis=1)-predicted.argmax(axis=1)).mean()*15),
                    actual_max_tie_origins=int(((actual==actual.max(axis=1)[:, None]).sum(axis=1)>1).sum())))
                line = f'{candidate}/{fold}: train={train.sum()}, valid={valid.sum()}, RMSE={score["Loss_A"]:.4f}, Loss_C={score["Loss_C"]:.4f}, {elapsed:.1f}s'
                logs.append(line)
                print(line, flush=True)
                # fold 완료마다 실제 산출물을 저장하여 중단 시에도 완료 범위를 보존한다.
                for name, items in [('fold_metrics', fold_metrics), ('horizon_metrics', horizons),
                                    ('condition_metrics', conditions), ('peak_path_metrics', peaks)]:
                    pd.DataFrame(items).to_csv(artifact/f'{name}.csv', index=False, encoding='utf-8-sig')
                pd.concat(predictions, ignore_index=True).to_csv(artifact/'predictions.csv', index=False, encoding='utf-8-sig')
                (artifact/'run.log').write_text('\n'.join(logs)+'\n', encoding='utf-8')
            metrics = pd.DataFrame(fold_metrics)
            all_metrics.extend(fold_metrics)
            record['status'] = '완료'
            write_record(record, '완료', metrics[['fold', 'n_train', 'n_origins', 'Loss_A', 'MAE', 'L_mean', 'L_CVaR', 'L_peak', 'Loss_B', 'Loss_C']],
                         times=f'{perf_counter()-candidate_started:.3f}초 (학습·추론·저장 포함)')
        except Exception as exc:
            record['status'] = '중단' if isinstance(exc, TimeoutError) else '실패'
            write_record(record, record['status'], pd.DataFrame(fold_metrics) if fold_metrics else None,
                         message=f'{type(exc).__name__}: {exc}', times=f'{perf_counter()-candidate_started:.3f}초')
            (artifact/'run.log').write_text('\n'.join(logs)+f'\n{type(exc).__name__}: {exc}\n', encoding='utf-8')
            update_index(root, records, pd.DataFrame(all_metrics), suite_dir)
            raise
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(suite_dir/'comparison_folds.csv', index=False, encoding='utf-8-sig')
    macro = metrics.groupby(['comparison_group', 'candidate'])[['Loss_A', 'MAE', 'L_mean', 'L_CVaR', 'L_peak', 'Loss_B', 'Loss_C']].mean().reset_index()
    macro.to_csv(suite_dir/'comparison_macro.csv', index=False, encoding='utf-8-sig')
    update_index(root, records, metrics, suite_dir)
    (suite_dir/'suite.json').write_text(json.dumps(dict(suite_id=suite_id, elapsed_seconds=perf_counter()-before,
        source_sha256=data_hash, code_sha256=code_hash, n_rf_fits=20,
        n_candidate_records=len(records), versions=versions), indent=2), encoding='utf-8')
    print(f'SAVED {suite_dir}', flush=True)
    return dict(output_dir=suite_dir, fold_metrics=metrics, macro=macro)


def update_index(root, records, metrics, suite_dir):
    index = root/'experiments/README.md'
    if not index.exists():
        index.write_text('# 실험 비교 인덱스\n\n수치는 실제 실행 결과만 기록한다. 월별 검증과 최종 테스트를 구분한다.\n', encoding='utf-8')
    content = f'\n## {suite_dir.name}: RF 현재 데이터 피처 비교\n\n'
    content += '4~7월 expanding 검증. 8월 별도 확인·9월 최종 테스트는 미실행. 4개 월 점수의 동일 가중 평균. 단위 kW.\n\n'
    for record in records:
        rel = record['md_path'].relative_to(root/'experiments').as_posix()
        content += f'- [{record["config"]["candidate"]}]({rel}): {record.get("status", "실행 중")}\n'
    if len(metrics):
        macro = metrics.groupby(['comparison_group', 'candidate'])[['Loss_A', 'MAE', 'L_CVaR', 'L_peak', 'Loss_C']].mean().reset_index()
        content += '\n'+md_table(macro)+'\n'
        content += '\ncore와 weekly는 학습 원점 지원 범위가 달라 그룹 안에서만 피처 효과를 비교한다. 중첩 원점의 점수는 독립 표본에 대한 추정이 아니다. 단일 seed의 탐색 결과이며 최종 모델은 확정하지 않았다.\n'
    content += f'\n산출물: `{suite_dir}`.\n'
    with index.open('a', encoding='utf-8') as stream:
        stream.write(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget-seconds', type=int, default=600)
    args = parser.parse_args()
    run_suite(budget_seconds=args.budget_seconds)
