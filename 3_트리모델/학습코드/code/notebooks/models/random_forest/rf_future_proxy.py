"""미래 외생 실측값을 오차0 예보로 간주하는 사용자 승인 RF 시나리오."""
from pathlib import Path
from datetime import datetime
from time import perf_counter
from importlib.metadata import version
from uuid import uuid4
import json
import sys
import platform

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import rf_current_data as common

ROOT = common.ROOT
BASELINE_RUN = ROOT/'artifacts/random_forest/runs/20261003T131206_KST_cbba11'
SCENARIO = 'future_actual_as_forecast_zero_error'


def build_proxy_features(df):
    features, targets, groups = common.build_features(df)
    observed_columns = groups['F2']
    proxy_columns = [c for c in observed_columns if not c.startswith('observed_')]
    for h in range(1, 5):
        for col in common.OBSERVED:
            name = f'h{h}_forecast_proxy_{col}'
            features[name] = df[col].shift(-h)
            proxy_columns.append(name)
    return features, targets, observed_columns, proxy_columns


def write_record(path, config, status, started, elapsed=None, metrics=None, message='없음'):
    table = '' if metrics is None else common.md_table(metrics[['fold', 'n_train', 'n_origins', 'Loss_A', 'MAE', 'L_mean', 'L_CVaR', 'L_peak', 'Loss_B', 'Loss_C']])
    text = f'''# 실험 {config['experiment_id']}

- 상태: {status}
- 목적과 가설: 사용자 요청에 따라 미래 생산량·기온·풍속·습도·강수량을 예측값으로 간주하여 F2 RF의 외생 입력 효과를 비교한다.
- 시작·종료 시각 및 시간대: {started} / {datetime.now(common.KST).isoformat() if status != '실행 중' else '미완료'}; Asia/Seoul.
- 모델·라이브러리 버전: RandomForestRegressor; {config['versions']}.
- 데이터 위치·버전 또는 해시·사용 범위: {common.SOURCE}; SHA256 {config['source_sha256']}; 2021-01-01~2021-07-31 중 유효 원점.
- 목표값·단위·예측 구간: power_15min kW, 미래1h의15분4점, 15min 원점 갱신.
- 학습·검증·테스트 분할과 누수 점검: 4~7월 expanding; 학습 정답 종료<검증 시작, 검증 정답 종료<다음 월. 8월·9월 모델 평가 미실행. 이 시나리오는 미래 외생 실측 입력을 명시적으로 허용하며, 미래 전력 정답은 입력 금지.
- 전처리·특성·하이퍼파라미터·seed: scaler 없음; F2 관측 외생5열을 각 목표구간 외생20열로 교체, 총70열. 달력과 과거 전력 피처는 유지. {config['rf_params']}.
- 이전 또는 기준 실험 링크와 변경 사항: [기존 관측 F2](20261003T131206_KST_cbba11_F2.md); [시나리오 계획](../../notebooks/models/random_forest/FUTURE_PROXY_PLAN.md). 학습/검증 원점·정답·RF 설정을 유지하고 외생 가용성만 변경. 기존 모델과 예측 재사용 검증.
- 실행 명령·환경·코드 버전 또는 코드 식별 정보: `{config['command']}`; Python {platform.python_version()}; rf_future_proxy.py SHA256 {config['code_sha256']}; rf_current_data.py SHA256 {config['shared_code_sha256']}.
- 실제 실행 시간·자원 사용량(측정한 경우): {elapsed if elapsed is not None else '미완료'}초; 메모리 미측정.
- 평가 지표·구간별 또는 fold별 결과: 아래 표. RMSE 주 지표, MAE/Loss B/C 보조. 1h4점 origin별 window, stride15min, CVaR 상위ceil20%; L_peak 가중치1, 전체 예측 셀의 과소예측 제곱 평균 후 root.
- 기준모델 대비 변화와 비교 가능 여부: 동일 지원 원점의 기존 F2와 비교. baseline.csv는 이전 실험 예측의 사본이며 새로 학습하지 않음.
- 오차가 큰 조건·해석·한계: 실제 과거 발행 예보 없음. 미래 실측을 예보로 대체한 오차0 가정의 검증 결과이며 운영 시 예측 성능으로 해석하지 않는다. 단일seed, window 중첩. 실제 예보의 오차·지연은 반영하지 않았다. RF는 squared_error 학습, B/C 직접 최적화 미실행.
- 오류·경고·실패 또는 중단 원인: {message}.
- 예측·모델·표·그래프·로그의 저장 경로: {config['artifact_dir']}; predictions.csv, baseline.csv, comparison_folds.csv, comparison_macro.csv, horizon_metrics.csv, model_<fold>.joblib, config.json, run.log, notebook 실행본.
- 결론과 다음 실험: 아래 실제 결과와 비교표 확인. 본 시나리오를 실제 예보 이용 모델과 구분한다. 8월 별도 확인·9월 최종 테스트는 보존.

{table}
'''
    path.write_text(text, encoding='utf-8')


def run_future_proxy(budget_seconds=300, execution_context=None):
    before = perf_counter()
    started = datetime.now(common.KST).isoformat()
    exp_id = datetime.now(common.KST).strftime('%Y%m%dT%H%M%S_KST')+'_'+uuid4().hex[:6]+'_F2_future_proxy'
    artifact = ROOT/'artifacts/random_forest/runs'/exp_id
    artifact.mkdir(parents=True, exist_ok=False)
    record = ROOT/'experiments/random_forest'/(exp_id+'.md')
    config = dict(experiment_id=exp_id, scenario=SCENARIO, forecast_error_assumption=0,
                  production_weather_source='future actuals; assumed available at origin; no issued forecast archive',
                  source_sha256=common.digest(common.SOURCE), code_sha256=common.digest(__file__),
                  shared_code_sha256=common.digest(common.__file__), rf_params=common.RF_PARAMS,
                  versions={pkg: version(pkg) for pkg in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'nbclient']},
                  command=execution_context or 'Jupyter rf_future_proxy.ipynb Run All',
                  artifact_dir=str(artifact), baseline_run=str(BASELINE_RUN), folds=common.FOLDS,
                  test='September reserved; no August/September model evaluation',
                  budget_seconds=budget_seconds, window_minutes=60, stride_minutes=15, peak_weights=1)
    (artifact/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    write_record(record, config, '실행 중', started)
    logs, rows, new_predictions, baseline_parts, horizon_rows = [], [], [], [], []
    try:
        prior_config = json.loads((BASELINE_RUN/'F2/config.json').read_text(encoding='utf-8'))
        if prior_config['source_sha256'] != config['source_sha256'] or prior_config['code_sha256'] != config['shared_code_sha256']:
            raise ValueError('기존 F2와 현재 데이터/공통 코드 지문이 달라 동일 실험 비교를 보장할 수 없습니다.')
        if prior_config['rf_params'] != common.RF_PARAMS:
            raise ValueError('기존 F2와 RF 설정이 다릅니다.')
        df = common.load_data()
        f, y, observed_cols, proxy_cols = build_proxy_features(df)
        support_cols = list(dict.fromkeys(observed_cols+proxy_cols))
        config.update(features=proxy_cols, n_features=len(proxy_cols), support_features=support_cols)
        (artifact/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        prior = pd.read_csv(BASELINE_RUN/'F2/predictions.csv', parse_dates=['origin', 'target_start', 'target_end'])
        prior_metrics = pd.read_csv(BASELINE_RUN/'F2/fold_metrics.csv')
        for fold in common.FOLDS:
            if perf_counter()-before > budget_seconds:
                raise TimeoutError(f'실험 예산 {budget_seconds}초 초과')
            train, valid = common.fold_masks(f, y, support_cols, fold)
            original_train, original_valid = common.fold_masks(f, y, observed_cols, fold)
            if not np.array_equal(train, original_train) or not np.array_equal(valid, original_valid):
                raise ValueError(f'{fold}: 새 외생 입력의 유효 원점이 기존 F2와 다릅니다.')
            origins = f.index[valid]
            actual = y.loc[valid].to_numpy()
            previous = prior.loc[prior.fold.eq(fold)].sort_values(['origin', 'horizon_minutes']).reset_index(drop=True)
            np.testing.assert_array_equal(previous.origin.drop_duplicates().to_numpy(), origins.to_numpy())
            np.testing.assert_array_equal(previous.actual.to_numpy().reshape(-1, 4), actual)
            original_model = joblib.load(BASELINE_RUN/'F2'/f'model_{fold}.joblib')
            recomputed = original_model.predict(f.loc[valid, observed_cols])
            np.testing.assert_allclose(recomputed, previous.predicted.to_numpy().reshape(-1, 4), rtol=1e-10, atol=1e-10)
            prior_row = prior_metrics.loc[prior_metrics.fold.eq(fold)].iloc[0]
            assert int(train.sum()) == int(prior_row.n_train)
            previous = previous.assign(candidate='F2_observed_reference', scenario='observed_only')
            baseline_parts.append(previous)
            rows.append(dict(candidate='F2_observed_reference', scenario='observed_only', fold=fold,
                n_train=int(train.sum()), **common.risk_scores(actual, recomputed)))
            fit_started = perf_counter()
            model = RandomForestRegressor(**common.RF_PARAMS)
            model.fit(f.loc[train, proxy_cols], y.loc[train])
            predicted = model.predict(f.loc[valid, proxy_cols])
            joblib.dump(model, artifact/f'model_{fold}.joblib', compress=3)
            elapsed = perf_counter()-fit_started
            row = dict(candidate='F2_future_proxy', scenario=SCENARIO, fold=fold,
                       n_train=int(train.sum()), fit_predict_save_seconds=elapsed,
                       train_first=str(f.index[train][0]), train_last_origin=str(f.index[train][-1]),
                       valid_first=str(origins[0]), valid_last_origin=str(origins[-1]),
                       **common.risk_scores(actual, predicted))
            rows.append(row)
            frame = common.prediction_frame(actual, predicted, origins, f).assign(
                fold=fold, candidate='F2_future_proxy', scenario=SCENARIO,
                assumed_forecast_error=0)
            for col in common.OBSERVED:
                frame[f'forecast_proxy_{col}'] = np.column_stack([f.loc[origins, f'h{h}_forecast_proxy_{col}'] for h in range(1, 5)]).ravel()
            new_predictions.append(frame)
            for name, values in [('F2_observed_reference', recomputed), ('F2_future_proxy', predicted)]:
                for h in range(4):
                    horizon_rows.append(dict(candidate=name, fold=fold, horizon_minutes=15*(h+1),
                        **common.risk_scores(actual[:, h:h+1], values[:, h:h+1])))
            line = f'{fold}: observed RMSE={rows[-2]["Loss_A"]:.4f}; future_proxy RMSE={row["Loss_A"]:.4f}, Loss_C={row["Loss_C"]:.4f}, {elapsed:.1f}s'
            logs.append(line)
            print(line, flush=True)
            pd.DataFrame(rows).to_csv(artifact/'comparison_folds.csv', index=False, encoding='utf-8-sig')
            pd.concat(new_predictions, ignore_index=True).to_csv(artifact/'predictions.csv', index=False, encoding='utf-8-sig')
            pd.concat(baseline_parts, ignore_index=True).to_csv(artifact/'baseline.csv', index=False, encoding='utf-8-sig')
            pd.DataFrame(horizon_rows).to_csv(artifact/'horizon_metrics.csv', index=False, encoding='utf-8-sig')
            (artifact/'run.log').write_text('\n'.join(logs)+'\n', encoding='utf-8')
        metrics = pd.DataFrame(rows)
        macro = metrics.groupby('candidate')[['Loss_A', 'MAE', 'L_mean', 'L_CVaR', 'L_peak', 'Loss_B', 'Loss_C']].mean().reset_index()
        macro.to_csv(artifact/'comparison_macro.csv', index=False, encoding='utf-8-sig')
        new_metrics = metrics.loc[metrics.candidate.eq('F2_future_proxy')]
        duration = perf_counter()-before
        write_record(record, config, '완료', started, elapsed=round(duration, 3), metrics=new_metrics)
        index = ROOT/'experiments/README.md'
        content = f'\n## {exp_id}: 미래 외생 실측을 예보로 간주\n\n'
        content += f'- [F2_future_proxy](random_forest/{exp_id}.md): 완료, 새RF4회 학습.\n'
        content += '- [관측 F2 기준](random_forest/20261003T131206_KST_cbba11_F2.md): 기존 모델/예측 재사용 검증.\n\n'
        content += '4~7월 동일 지원 원점 비교, 월별 동일 가중 평균, 단위kW. 미래 생산·기상 실측을 오차0 예보로 대체한 시나리오이며 실제 예보 성능이 아니다. 8월·9월 모델 평가 없음.\n\n'
        content += common.md_table(macro)+'\n'
        with index.open('a', encoding='utf-8') as stream:
            stream.write(content)
        (artifact/'suite.json').write_text(json.dumps(dict(experiment_id=exp_id, scenario=SCENARIO,
            elapsed_seconds=duration, n_rf_fits=4, n_candidate_records=1, n_features=len(proxy_cols),
            source_sha256=config['source_sha256'], code_sha256=config['code_sha256'],
            shared_code_sha256=config['shared_code_sha256']), indent=2), encoding='utf-8')
        print(f'SAVED {artifact}', flush=True)
        return dict(output_dir=artifact, fold_metrics=metrics, macro=macro, record=record)
    except Exception as exc:
        status = '중단' if isinstance(exc, TimeoutError) else '실패'
        metrics = pd.DataFrame(rows)
        done = metrics.loc[metrics.candidate.eq('F2_future_proxy')] if len(metrics) else None
        write_record(record, config, status, started, elapsed=round(perf_counter()-before, 3),
                     metrics=done, message=f'{type(exc).__name__}: {exc}')
        (artifact/'run.log').write_text('\n'.join(logs)+f'\n{type(exc).__name__}: {exc}\n', encoding='utf-8')
        with (ROOT/'experiments/README.md').open('a', encoding='utf-8') as stream:
            stream.write(f'\n- [미래 외생 시나리오 {exp_id}](random_forest/{exp_id}.md): {status}\n')
        raise
