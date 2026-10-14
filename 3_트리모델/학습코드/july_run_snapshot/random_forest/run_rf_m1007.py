"""10/07 회의 반영 RF 재실험 — 7월 검증(고르기)까지.

변경점(지시서 docs/meeting_20261007_rerun_plan.md):
- 월 칸 제거(no_month.py 규칙), 미래 외생은 future(실측=오차0 예보)만.
- F2 통일: 모든 구간 최근 1시간 통계(window=1), b 칸(peak_threshold_origin, power_latest_to_peak) 제거.
- 순서: 고정 설정(FIXED_PARAMS, C10 잎 조정)으로 F1→F2→F3 7월 C 비교 → 구간별 최저 F 선택
  → 그 F에서 기존 그리드(core 18 → refinement → 5 학습방식) 튜닝, 7월 C로 선정 → 선정 구조로 RF A판(무가중·잎 조정 없음).
- 학습 = 7/1 이전에 정답 경로가 끝나는 원점, 평가 = 7월. 8·9월 미사용. 잎 조정은 학습 데이터 C로만 수행.
사용: python -u -B -X utf8 notebooks/models/random_forest/run_rf_m1007.py [--smoke] [--workers 2]
"""
import argparse
import json
import shutil
import sys
import traceback
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import platform

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT/'notebooks/_shared')]
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
import m1007_common as mc  # noqa: E402
import no_month  # noqa: E402
import rf_feature_sequence as fs  # noqa: E402
import rf_direct_c_grid as grid  # noqa: E402
import rf_direct_c_experiments as runner  # noqa: E402

B_COLUMNS = ['peak_threshold_origin', 'power_latest_to_peak']
WINDOW_HOURS = 1
_original_frames = fs._feature_frames


def _frames_m1007(*args, **kwargs):
    frames, masks = _original_frames(*args, **kwargs)
    out = {}
    for key, x in frames.items():
        x = no_month.drop_month(x)
        out[key] = x.drop(columns=[c for c in B_COLUMNS if c in x.columns])
        if any(c in out[key].columns for c in B_COLUMNS):
            raise ValueError('b 칸 잔존')
    return out, masks


fs._feature_frames = _frames_m1007


def split_inclusive(origins, horizon, start, end):
    """경로 마지막 칸의 종료 시각이 경계와 같으면(예: 7/31 23:45~24:00) 그 달에 포함. 팀 발행 시각과 일치시킨다.
    학습 = 마지막 정답 종료 ≤ start, 평가 = 원점 ≥ start 이고 마지막 정답 종료 ≤ end. 누수 없음(경계 칸은 start 이전 관측)."""
    idx = pd.DatetimeIndex(origins)
    last = idx + pd.Timedelta(minutes=15*horizon)
    ti = np.flatnonzero(last <= pd.Timestamp(start))
    vi = np.flatnonzero((idx >= pd.Timestamp(start)) & (last <= pd.Timestamp(end)))
    if not len(ti) or not len(vi) or np.intersect1d(ti, vi).size:
        raise ValueError('빈 분할/원점 중복')
    return ti, vi


runner.split_indices = split_inclusive


def fit_job_m1007(*args):
    """loky 작업자에서도 같은 분할 규칙을 쓰도록 작업 시작 때 다시 적용한다."""
    import rf_direct_c_experiments as worker_runner
    worker_runner.split_indices = split_inclusive
    return grid.fit_job(*args)


def load_data():
    df, repair, audit = runner.previous.final.load_final_data()
    keep = df.index <= mc.DATA_END
    return df.loc[keep].copy(), repair.loc[keep].copy(), audit


def rows_full(result):
    if result['error']:
        raise RuntimeError(f"{result['profile']}/{result['param_id']}: {result['error']}")
    return [r for r in result['rows'] if r['scope'] == 'full']


def run_jobs(jobs, workers):
    with joblib.parallel_config(backend='loky', inner_max_num_threads=1):
        return list(joblib.Parallel(n_jobs=workers)(jobs))


def prediction_file(folder):
    return Path(folder)/'predictions.csv.gz'


def a_version(data, params, folder):
    """RF A판: 같은 구조 파라미터, 가중치 없음(순수 squared_error), 잎 조정 없음."""
    ti, vi = runner.split_indices(data['origins'], data['spec']['horizon_steps'], *mc.JULY)
    x = data['features'].iloc[ti].to_numpy(np.float32)
    y = data['targets'].iloc[ti].to_numpy(float)
    xv = data['features'].iloc[vi].to_numpy(np.float32)
    actual = data['targets'].iloc[vi].to_numpy(float)
    b = data['threshold'].iloc[vi].to_numpy(float)
    t0 = perf_counter()
    with threadpool_limits(limits=1):
        forest = RandomForestRegressor(**params).fit(x, y)
    predicted = forest.predict(xv)
    folder.mkdir(parents=True, exist_ok=False)
    joblib.dump(dict(model=forest, features=list(data['features']), feature_set=data['feature_set'], params=params,
                     objective='A_unweighted_squared_error'), folder/'model.joblib', compress=3)
    h = actual.shape[1]
    origins = data['origins'][vi]
    pd.DataFrame(dict(origin=np.repeat(origins, h),
                      target_end=(origins.to_numpy()[:, None] + np.arange(1, h + 1)*np.timedelta64(15, 'm')).ravel(),
                      actual=actual.ravel(), predicted=predicted.ravel())).to_csv(prediction_file(folder), index=False, compression='gzip')
    return dict(**mc.score(actual, predicted, b, data['profile']), n_train=len(ti), n_origins=len(vi), seconds=perf_counter() - t0)


def record(run, eid, status, cfg, rows, extra=''):
    fields = {
        '상태': status,
        '목적과 가설': cfg['purpose'],
        '시작·종료 시각 및 시간대': f"{cfg['started']} / {mc.now()}; Asia/Seoul",
        '모델·라이브러리 버전': f"Python {platform.python_version()}, scikit-learn {version('scikit-learn')}",
        '데이터 위치·버전 또는 해시·사용 범위': f"{mc.SOURCE.relative_to(ROOT)}; SHA256 {cfg['source_sha256']}; timestamp_end ≤ 2021-08-01 00:00",
        '목표값·단위·예측 구간': f"power_15min kW, {cfg['profile']} ({mc.HORIZON[cfg['profile']]}칸)",
        '학습·검증·테스트 분할과 누수 점검': '학습=정답 경로 마지막 칸 종료 ≤ 2021-07-01 00:00, 검증=7월 원점 중 마지막 칸 종료 ≤ 8/1 00:00(7/31 23:45~24:00 칸 포함). 8·9월 미사용. 잎 조정·분할 수정은 학습 데이터 C로만.',
        '전처리·특성·하이퍼파라미터·seed': f"{cfg['feature_set']}, 월 칸 제거, b 칸 제거, 최근 통계 창 1h, future 외생; params={cfg['params']}; 학습방식={cfg['family']}; seed 42",
        '이전 또는 기준 실험 링크와 변경 사항': f"기준: 이전 RF 통합 그리드(20261006T171055_KST_e7e556) 대비 월 제거·F2 통일·7월만 선정. 지시서 {mc.DECISION_DOC}",
        '실행 명령·환경·코드 버전 또는 코드 식별 정보': f"python notebooks/models/random_forest/run_rf_m1007.py; 코드 해시는 {run.name}/config.json",
        '실제 실행 시간·자원 사용량(측정한 경우)': cfg.get('seconds', '아래 표 참조; RAM 미측정'),
        '평가 지표·구간별 또는 fold별 결과': '아래 표(7월, full scope). 주 지표 공통 C(낮을수록 좋음), 보조 RMSE kW.',
        '기준모델 대비 변화와 비교 가능 여부': '같은 구간의 같은 7월 원점·정답으로 비교. 기준선은 7월 요약표에서 비교.',
        '오차가 큰 조건·해석·한계': '7월 한 달만으로 선정 → 선택 편향 가능. future 외생은 사후 실측(예보 오차 0). 170kW 바닥값에 과거 전체 q90 정보.',
        '오류·경고·실패 또는 중단 원인': cfg.get('error', '없음'),
        '예측·모델·표·그래프·로그의 저장 경로': cfg['artifact'],
        '결론과 다음 실험': extra or status,
    }
    mc.write_record(run/'records'/f'{eid}.md', fields, rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--profiles', nargs='+', choices=mc.PROFILES, default=mc.PROFILES)
    args = parser.parse_args(argv)
    profiles = ['medium_4h'] if args.smoke else args.profiles
    run_name = mc.run_id('rf_m1007' + ('_smoke' if args.smoke else ''))
    run = ROOT/'experiments/random_forest'/run_name
    suite = ROOT/'artifacts/random_forest/runs'/run_name
    (run/'records').mkdir(parents=True)
    suite.mkdir(parents=True)
    code = [Path(__file__), Path(mc.__file__), Path(no_month.__file__), Path(fs.__file__), Path(grid.__file__),
            Path(runner.__file__), Path(grid.direct.__file__)]
    (run/'code').mkdir()
    for p in code:
        shutil.copy2(p, run/'code'/p.name)
    config = dict(run_id=run_name, status='실행 중', started=mc.now(), decision_doc=mc.DECISION_DOC,
                  source=str(mc.SOURCE), source_sha256=mc.digest(mc.SOURCE), profiles=profiles,
                  feature_order=mc.FEATURE_ORDER, window_hours=WINDOW_HOURS, removed_columns=['*month'] + B_COLUMNS,
                  scenario='future', fixed_params=fs.FIXED_PARAMS, selection='7월 full 공통 C 최소(동률 시 기존 family 순서)',
                  python=platform.python_version(), versions={p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib']},
                  code_sha256={p.name: mc.digest(p) for p in code}, workers=args.workers, smoke=args.smoke)
    mc.dump(run/'config.json', config)
    log = lambda m: print(f'{mc.now()} {m}', flush=True)  # noqa: E731
    start = perf_counter()
    deadline = float('inf')
    summary = []
    try:
        df, repair, audit = load_data()
        config['data_audit'] = audit
        sets = {}
        for profile in profiles:
            sets[profile] = fs.prepare_feature_sets(df, repair, profile, WINDOW_HOURS)
            config.setdefault('n_features', {})[profile] = {k: v['features'].shape[1] for k, v in sets[profile].items()}
            config.setdefault('feature_columns', {})[profile] = {k: list(v['features'].columns) for k, v in sets[profile].items()}
        mc.dump(run/'config.json', config)
        # 1단계: F1→F2→F3 고정 설정 비교
        fixed = dict(fs.FIXED_PARAMS)
        jobs, keys = [], []
        for feature in mc.FEATURE_ORDER:
            for profile in profiles:
                keys.append((profile, feature))
                jobs.append(joblib.delayed(fit_job_m1007)(suite/'feature_stage'/feature, config, sets[profile][feature], 'm07',
                                                         fixed, ['c_leaf_d10'], deadline))
        log(f'FEATURE_STAGE jobs={len(jobs)}')
        feature_scores = {}
        for (profile, feature), result in zip(keys, run_jobs(jobs, args.workers)):
            row = rows_full(result)[0]
            feature_scores.setdefault(profile, {})[feature] = row['Loss_C']
            eid = f'{profile}_{feature}_fixed_c_leaf_d10'
            pid = grid.parameter_id(fixed)
            artifact = suite/'feature_stage'/feature/'cv'/profile/f'{pid}_c_leaf_d10'/'m07'
            record(run, eid, '완료', dict(purpose='F1→F2→F3 고정 설정 7월 비교(피처 선택 단계)', started=config['started'],
                   source_sha256=config['source_sha256'], profile=profile, feature_set=feature, params=fixed, family='c_leaf_d10',
                   artifact=str(artifact.relative_to(ROOT))), [{k: row[k] for k in ['fold', 'n_origins', 'Loss_C', 'RMSE', 'L_mean', 'L_CVaR', 'L_peak']}])
            summary.append(dict(experiment_id=eid, stage='feature', profile=profile, feature_set=feature, family='c_leaf_d10',
                                param_id=pid, Loss_C_july=row['Loss_C'], RMSE_july=row['RMSE'], prediction=str(prediction_file(artifact).relative_to(ROOT))))
            log(f'FEATURE {profile} {feature} C={row["Loss_C"]:.4f} RMSE={row["RMSE"]:.4f}')
        chosen_f = {p: min(mc.FEATURE_ORDER, key=lambda k: (s[k], mc.FEATURE_ORDER.index(k))) for p, s in feature_scores.items()}
        config.update(feature_scores=feature_scores, chosen_feature=chosen_f)
        mc.dump(run/'config.json', config)
        pd.DataFrame(summary).to_csv(run/'summary.csv', index=False)
        # 2단계: 선택 F에서 그리드 튜닝 (7월만)
        selection = {}
        for profile in profiles:
            data = sets[profile][chosen_f[profile]]
            tune = suite/'tune'
            frame = []

            def batch(param_list, families, data=data, tune=tune, frame=frame):
                params = grid.unique_parameters(param_list)
                results = run_jobs([joblib.delayed(fit_job_m1007)(tune, config, data, 'm07', p, families, deadline) for p in params], args.workers)
                for p, result in zip(params, results):
                    for row in rows_full(result):
                        row['params'] = p
                        frame.append(row)

            core = grid.core_grid()[:2] if args.smoke else grid.core_grid()
            log(f'TUNE {profile} {chosen_f[profile]} core={len(core)}')
            batch(core, grid.PAIRED)
            leaf = [r for r in frame if r['family'] == 'c_leaf_d10']
            anchor = min(leaf, key=lambda r: (r['Loss_C'], r['candidate']))['params']
            extra = [] if args.smoke else [p for p in grid.refinement_grid(anchor) if grid.parameter_id(p) not in {grid.parameter_id(q) for q in core}]
            log(f'TUNE {profile} refinement={len(extra)}')
            if extra:
                batch(extra, grid.PAIRED)
            best = min(frame, key=lambda r: (r['Loss_C'], grid.FAMILIES.index(r['family']), r['candidate']))
            log(f'TUNE {profile} five_families anchor={grid.parameter_id(best["params"])}')
            batch([best['params']], grid.FAMILIES)
            seen = set()
            for row in frame:
                if row['candidate'] in seen:
                    continue
                seen.add(row['candidate'])
                eid = f'{profile}_{chosen_f[profile]}_{row["candidate"]}'
                artifact = tune/'cv'/profile/row['candidate']/'m07'
                record(run, eid, '완료', dict(purpose='선택 F에서 RF 구조·학습방식 튜닝(7월 C 선정)', started=config['started'],
                       source_sha256=config['source_sha256'], profile=profile, feature_set=chosen_f[profile], params=row['params'],
                       family=row['family'], artifact=str(artifact.relative_to(ROOT))),
                       [{k: row[k] for k in ['fold', 'n_origins', 'Loss_C', 'RMSE', 'L_mean', 'L_CVaR', 'L_peak']}])
                summary.append(dict(experiment_id=eid, stage='tune', profile=profile, feature_set=chosen_f[profile], family=row['family'],
                                    param_id=row['param_id'], Loss_C_july=row['Loss_C'], RMSE_july=row['RMSE'],
                                    prediction=str(prediction_file(artifact).relative_to(ROOT))))
            pick = min(frame, key=lambda r: (r['Loss_C'], grid.FAMILIES.index(r['family']), r['candidate']))
            selection[profile] = dict(feature_set=chosen_f[profile], candidate=pick['candidate'], family=pick['family'],
                                      params=pick['params'], Loss_C_july=pick['Loss_C'], RMSE_july=pick['RMSE'],
                                      prediction=str(prediction_file(tune/'cv'/profile/pick['candidate']/'m07').relative_to(ROOT)))
            # 3단계: RF A판
            folder = suite/'a_version'/profile
            metrics = a_version(data, pick['params'], folder)
            eid = f'{profile}_{chosen_f[profile]}_{grid.parameter_id(pick["params"])}_rf_A_unweighted'
            record(run, eid, '완료', dict(purpose='RF A판(무가중 RMSE 학습·잎 조정 없음): C 학습 효과 비교용', started=config['started'],
                   source_sha256=config['source_sha256'], profile=profile, feature_set=chosen_f[profile], params=pick['params'],
                   family='A_unweighted', artifact=str(folder.relative_to(ROOT)), seconds=f"{metrics['seconds']:.1f}초"),
                   [dict(fold='m07', **{k: metrics[k] for k in ['n_origins', 'Loss_C', 'RMSE', 'L_mean', 'L_CVaR', 'L_peak']})])
            summary.append(dict(experiment_id=eid, stage='A_version', profile=profile, feature_set=chosen_f[profile], family='A_unweighted',
                                param_id=grid.parameter_id(pick['params']), Loss_C_july=metrics['Loss_C'], RMSE_july=metrics['RMSE'],
                                prediction=str(prediction_file(folder).relative_to(ROOT))))
            selection[profile]['A_version'] = dict(Loss_C_july=metrics['Loss_C'], RMSE_july=metrics['RMSE'],
                                                   prediction=str(prediction_file(folder).relative_to(ROOT)))
            log(f'SELECT {profile} {selection[profile]["candidate"]} C={pick["Loss_C"]:.4f} | A C={metrics["Loss_C"]:.4f}')
            mc.dump(run/'selection.json', selection)
            pd.DataFrame(summary).to_csv(run/'summary.csv', index=False)
        config.update(status='완료', ended=mc.now(), elapsed_seconds=perf_counter() - start)
    except Exception:
        config.update(status='실패', ended=mc.now(), error=traceback.format_exc(), elapsed_seconds=perf_counter() - start)
        raise
    finally:
        mc.dump(run/'config.json', config)
        if summary:
            frame = pd.DataFrame(summary)
            frame['selected'] = [any(s.get('candidate') and r.experiment_id.endswith(s['candidate']) and r.profile == p for p, s in selection.items())
                                 for r in frame.itertuples()] if 'selection' in locals() else False
            frame.to_csv(run/'summary.csv', index=False)
        log(f'RUN {run} {config["status"]}')


if __name__ == '__main__':
    main()
