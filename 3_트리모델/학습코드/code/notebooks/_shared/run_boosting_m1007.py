"""10/07 회의 반영 부스팅 재실험 — 7월 검증(고르기)까지. 라이브러리 하나를 한 프로세스로 실행(3종 병렬).

변경점(지시서 docs/meeting_20261007_rerun_plan.md):
- 월 칸 제거(no_month.patch_boosting), future만, F1→F2→F3를 c00 설정으로 비교 → 구간별 최저 F에서 c00~c02 튜닝 → 같은 F·BASE로 A 대조.
- 나무 수: ① 학습 끝(7/1)에서 거꾸로 복사 안 된 날 14일이 드는 확인 구간을 떼고 그 앞까지 학습, 확인 구간 손실 최저 round R
  (확인 창 중 정답 칸이 전부 복사된 날인 창은 제외; C 학습은 공통 C, A 학습은 RMSE로 멈춤)
  ② 7/1 이전 전체로 R round 재학습 → 7월 예측. 7월 점수는 나무 수 결정에 쓰지 않는다.
- 선정: 7월 공통 C 최소. 8·9월 미사용.
사용: python -u -B -X utf8 notebooks/_shared/run_boosting_m1007.py --library xgboost --run-id <id> [--smoke]
"""
import argparse
import gzip
import json
import platform
import shutil
import sys
import traceback
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402
import m1007_common as mc  # noqa: E402
import no_month  # noqa: E402

no_month.patch_boosting()
import full_c_boosting as fb  # noqa: E402
import full_c_data as fd  # noqa: E402
import full_c_tuning as tuning  # noqa: E402

ROOT = mc.ROOT
THREADS = 3  # 3개 라이브러리 병렬(8 논리 CPU)
CANDIDATES = dict(tuning.CANDIDATES)
CANDIDATES['catboost'] = [dict(tuning.BASE), dict(tuning.BASE, depth=3, l2=5), dict(tuning.BASE, depth=6, l2=10)]  # e7991c와 동일
MAX_ROUNDS = 60

# 확인 구간 멈춤 지표: 검증 호출(인자 4개)에서만 Loss_C 자리에 선택 지표를 넣는다. 학습 경사 호출(인자 5개)은 그대로.
_loss = fb.loss_gradient
STOP = dict(metric='Loss_C')


def _loss_for_stopping(y, p, w, width, objective=None, *rest):
    if objective is not None:
        return _loss(y, p, w, width, objective, *rest)
    metric, gradient, details = _loss(y, p, w, width)
    if STOP['metric'] != 'Loss_C':
        metric = dict(metric, Loss_C=metric[STOP['metric']])
    return metric, gradient, details


fb.loss_gradient = _loss_for_stopping


def fit_two_stage(data, library, params, objective, copied, folder, period=mc.JULY):
    """period=(평가 시작, 끝). 7월 고르기는 기본값, 8월 검증은 (8/1, 9/1)로 같은 절차를 쓴다."""
    ti, vi = mc.july_split(data.origins, data.horizon, *period)
    hs = mc.holdout_start(period[0], copied)
    last = data.origins + pd.Timedelta(minutes=15*data.horizon)
    inner_train = ti[last[ti] <= hs]
    inner_valid = ti[(data.origins[ti] >= hs) & ~mc.all_copied_target(data.origins[ti], data.horizon, copied)]
    if not len(inner_train) or not len(inner_valid):
        raise ValueError('확인 구간 분할 비어 있음')
    t0 = perf_counter()
    STOP['metric'] = 'RMSE' if objective == 'A' else 'Loss_C'
    params = dict(params, rounds=MAX_ROUNDS, threads=THREADS)
    x, xv = data.inputs(inner_train), data.inputs(inner_valid)
    with threadpool_limits(limits=1, user_api='blas'):
        stage1 = fb.CBoosting(library, params, objective).fit(
            x.to_numpy(), data.y[inner_train], data.w_train[inner_train], data.width,
            xv.to_numpy(), data.y[inner_valid], data.w_eval[inner_valid], direct=data.direct)
    rounds = int(stage1.selected_round)
    seconds1 = perf_counter() - t0
    x_all, x_july = data.inputs(ti), data.inputs(vi)
    with threadpool_limits(limits=1, user_api='blas'):
        model = fb.CBoosting(library, dict(params, rounds=rounds), objective).fit(
            x_all.to_numpy(), data.y[ti], data.w_train[ti], data.width, direct=data.direct)
    predicted = model.predict(x_july.to_numpy())
    b = data.b[vi]
    metrics = mc.score(data.y[vi], predicted, b, data.profile)
    metrics.update(holdout_start=str(hs.date()), n_inner_train=len(inner_train), n_inner_valid=len(inner_valid),
                   inner_stop_metric=STOP['metric'], selected_round=rounds, stage1_fitted_rounds=stage1.fitted_rounds,
                   stage1_stop=stage1.stop_reason, final_fitted_rounds=model.fitted_rounds, final_stop=model.stop_reason,
                   n_train=len(ti), n_origins=len(vi), train_last_target=str(last[ti].max()),
                   seconds_stage1=seconds1, seconds_total=perf_counter() - t0, params=params, objective=objective)
    folder.mkdir(parents=True, exist_ok=False)
    joblib.dump(dict(model=model, feature_columns=list(x_all.columns), profile=data.profile, feature=data.feature,
                     scenario=data.scenario, params=params, objective=objective, rounds=rounds), folder/'model.joblib', compress=3)
    loaded = joblib.load(folder/'model.joblib')['model']
    n = min(3, len(vi))
    small = x_july.iloc[:n] if data.direct else x_july.iloc[:n*data.horizon]
    np.testing.assert_allclose(loaded.predict(small.to_numpy()), predicted[:n], atol=1e-6, rtol=0)
    np.savez_compressed(folder/'predictions.npz', origins=data.origins[vi].asi8, actual=data.y[vi], predicted=predicted, B=b)
    for name, m in [('trace_stage1', stage1), ('trace_final', model)]:
        with gzip.open(folder/f'{name}.json.gz', 'wt', encoding='utf-8') as stream:
            json.dump(m.trace, stream, ensure_ascii=False, default=float)
    mc.dump(folder/'metrics.json', metrics)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True, choices=['xgboost', 'lightgbm', 'catboost'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--profiles', nargs='+', choices=mc.PROFILES, default=mc.PROFILES)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    library = args.library
    profiles = ['short_1h'] if args.smoke else args.profiles
    art = ROOT/'artifacts/boosting/m1007_runs'/args.run_id/library
    rec = ROOT/'experiments'/library/args.run_id
    art.mkdir(parents=True, exist_ok=False)
    (rec/'records').mkdir(parents=True, exist_ok=False)
    code = [Path(__file__), Path(mc.__file__), Path(no_month.__file__), Path(fb.__file__), Path(fd.__file__),
            Path(tuning.__file__), HERE/'full_c_loss.py', HERE/'performance_profiles.py', HERE/'boosting_common.py',
            ROOT/'notebooks/models/random_forest/rf_current_data.py']
    (rec/'code').mkdir()
    for p in code:
        shutil.copy2(p, rec/'code'/p.name)
    config = dict(run_id=args.run_id, library=library, status='실행 중', started=mc.now(), decision_doc=mc.DECISION_DOC,
                  source=str(mc.SOURCE), source_sha256=mc.digest(mc.SOURCE), profiles=profiles, scenario='future',
                  feature_order=mc.FEATURE_ORDER, candidates=CANDIDATES[library], max_rounds=MAX_ROUNDS, threads=THREADS,
                  rounds_rule='학습 끝 복사 안 된 날 14일 확인 구간 최저 round → 7/1 이전 전체 재학습',
                  selection='7월 공통 C 최소(동률 후보 ID 순)', python=platform.python_version(),
                  versions={p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', library]},
                  code_sha256={p.name: mc.digest(p) for p in code}, smoke=args.smoke)
    mc.dump(rec/'config.json', config)
    log = lambda m: print(f'{mc.now()} {library} {m}', flush=True)  # noqa: E731
    start = perf_counter()
    summary, selection = [], {}

    def run_one(data, cand, params, objective, stage, copied):
        eid = f'{data.profile}_future_{data.feature}_{cand}'
        md = rec/'records'/f'{eid}.md'
        started = mc.now()
        fields = {'상태': '실행 중', '목적과 가설': stage, '시작·종료 시각 및 시간대': f'{started} / 미완료; Asia/Seoul'}
        mc.write_record(md, fields)
        folder = art/eid
        try:
            m = fit_two_stage(data, library, params, objective, copied, folder)
            status, error = '완료', '없음'
        except Exception:
            status, error, m = '실패', traceback.format_exc(), None
        fields = {
            '상태': status, '목적과 가설': stage,
            '시작·종료 시각 및 시간대': f'{started} / {mc.now()}; Asia/Seoul',
            '모델·라이브러리 버전': f"Python {platform.python_version()}, {library} {version(library)}; 외부 C 부분경사 부스팅(full_c_boosting)",
            '데이터 위치·버전 또는 해시·사용 범위': f"{mc.SOURCE.relative_to(ROOT)}; SHA256 {config['source_sha256']}; timestamp_end ≤ 2021-08-01 00:00",
            '목표값·단위·예측 구간': f'power_15min kW, {data.profile} ({data.horizon}칸)',
            '학습·검증·테스트 분할과 누수 점검': (f"① 학습: 정답 끝 < 확인 구간 시작 {m['holdout_start'] if m else '?'}, 확인: 그 이후~6/30 원점(정답 전부 복사된 날인 창 제외) "
                                     '→ 최저 round ② 정답 끝 ≤ 7/1 00:00 전체 재학습 → 7월. 7월 점수는 round·학습에 미사용. 8·9월 미사용.'),
            '전처리·특성·하이퍼파라미터·seed': f'{data.feature}, future, 월 칸 제거; objective={objective}; {json.dumps(dict(params, rounds=MAX_ROUNDS, threads=THREADS))}; seed 42',
            '이전 또는 기준 실험 링크와 변경 사항': f'이전 e7991c·08eb6e(7·8월 checkpoint 중앙값) 대비 월 제거·future만·확인 구간 round·7월 선정. 지시서 {mc.DECISION_DOC}',
            '실행 명령·환경·코드 버전 또는 코드 식별 정보': f'python notebooks/_shared/run_boosting_m1007.py --library {library} --run-id {args.run_id}; 코드 해시 config.json',
            '실제 실행 시간·자원 사용량(측정한 경우)': f"{m['seconds_total']:.1f}초(①{m['seconds_stage1']:.1f}초), threads {THREADS}; RAM 미측정" if m else '미완료',
            '평가 지표·구간별 또는 fold별 결과': '아래 표(7월). 주 지표 공통 C, 보조 RMSE kW.',
            '기준모델 대비 변화와 비교 가능 여부': '같은 구간·7월 원점에서 비교. 기준선은 7월 요약표.',
            '오차가 큰 조건·해석·한계': '7월 한 달 선정. future 외생은 사후 실측(예보 오차 0). 확인 구간 원본 날만 14일.',
            '오류·경고·실패 또는 중단 원인': error,
            '예측·모델·표·그래프·로그의 저장 경로': str(folder.relative_to(ROOT)),
            '결론과 다음 실험': status,
        }
        table = [{k: m[k] for k in ['n_origins', 'Loss_C', 'RMSE', 'L_mean', 'L_CVaR', 'L_peak', 'selected_round', 'holdout_start', 'n_inner_valid']}] if m else None
        mc.write_record(md, fields, table)
        row = dict(experiment_id=eid, status=status, stage=stage, profile=data.profile, feature_set=data.feature, candidate=cand,
                   objective=objective, Loss_C_july=m and m['Loss_C'], RMSE_july=m and m['RMSE'], rounds=m and m['selected_round'],
                   seconds=m and m['seconds_total'], record_path=f'records/{eid}.md',
                   prediction=str((folder/'predictions.npz').relative_to(ROOT)))
        summary.append(row)
        pd.DataFrame(summary).to_csv(rec/'summary.csv', index=False)
        log(f"{eid} {status} C={row['Loss_C_july']} RMSE={row['RMSE_july']} R={row['rounds']}")
        if m is None:
            raise RuntimeError(f'{eid} 실패')
        return row

    try:
        dev = fd.load_frame('2021-08-01 00:01')
        power = dev.set_index('timestamp').power_15min
        copied = mc.copied_days(power)
        config.update(copied_days=[str(d.date()) for d in copied], n_copied_days=len(copied),
                      holdout_start_july=str(mc.holdout_start(mc.JULY[0], copied).date()))
        mc.dump(rec/'config.json', config)
        cands = CANDIDATES[library][:2] if args.smoke else CANDIDATES[library]
        for profile in profiles:
            scores = {}
            datasets = {}
            for feature in mc.FEATURE_ORDER:
                datasets[feature] = fd.Dataset(dev, profile, feature, 'future')
                scores[feature] = run_one(datasets[feature], 'c00', cands[0], 'C', 'F 비교(c00)', copied)
            best_f = min(mc.FEATURE_ORDER, key=lambda f: (scores[f]['Loss_C_july'], mc.FEATURE_ORDER.index(f)))
            for f in mc.FEATURE_ORDER:
                if f != best_f:
                    datasets.pop(f)
            data = datasets[best_f]
            rows = [scores[best_f]]
            for ix, params in enumerate(cands[1:], start=1):
                rows.append(run_one(data, f'c{ix:02}', params, 'C', f'선택 F({best_f}) 튜닝', copied))
            pick = min(rows, key=lambda r: (r['Loss_C_july'], r['candidate']))
            control = run_one(data, 'a_control', tuning.BASE, 'A', f'선택 F({best_f}) A 대조(BASE)', copied)
            selection[profile] = dict(feature_set=best_f, feature_scores={f: scores[f]['Loss_C_july'] for f in scores},
                                      candidate=pick['candidate'], params=cands[int(pick['candidate'][1:])],
                                      Loss_C_july=pick['Loss_C_july'], RMSE_july=pick['RMSE_july'], rounds=pick['rounds'],
                                      prediction=pick['prediction'], A_control=dict(Loss_C_july=control['Loss_C_july'],
                                      RMSE_july=control['RMSE_july'], rounds=control['rounds'], prediction=control['prediction']))
            mc.dump(rec/'selection.json', selection)
            log(f"SELECT {profile} {best_f} {pick['candidate']} C={pick['Loss_C_july']:.4f} | A C={control['Loss_C_july']:.4f}")
            del datasets, data
        config.update(status='완료', ended=mc.now(), elapsed_seconds=perf_counter() - start)
    except Exception:
        config.update(status='실패', ended=mc.now(), error=traceback.format_exc(), elapsed_seconds=perf_counter() - start)
        raise
    finally:
        if summary:
            frame = pd.DataFrame(summary)
            frame['selected'] = [selection.get(r.profile, {}).get('candidate') == r.candidate and selection[r.profile]['feature_set'] == r.feature_set
                                 for r in frame.itertuples()]
            frame.to_csv(rec/'summary.csv', index=False)
        mc.dump(rec/'config.json', config)
        log(f'RUN {rec} {config["status"]}')


if __name__ == '__main__':
    main()
