"""10/07 회의 반영: 7월에 고른 설정 그대로 1~7월 재학습 → 8월 예측 → 7월 마진 표를 8월에 적용해 검증.

8월로 모델·설정·마진을 고르지 않는다. 9월 행은 원자료를 읽은 직후 제외한다(timestamp_end ≤ 2021-09-01 00:00만 사용).
판정 방식은 docs/meeting_20261007_rerun_plan.md "8월 마진 검증 판정 방식"(8월 결과 열람 전 기록)을 따른다.
사용:
  python -B -X utf8 notebooks/_shared/m1007_august.py rf --rf-run <dir> --out <dir>
  python -B -X utf8 notebooks/_shared/m1007_august.py boost --library xgboost --boost-run <id> --out <dir>
  python -B -X utf8 notebooks/_shared/m1007_august.py margin --july-report <dir> --out <dir>
"""
import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
RF_DIR = HERE.parents[1]/'notebooks/models/random_forest'
sys.path[:0] = [str(HERE), str(RF_DIR)]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import m1007_common as mc  # noqa: E402

ROOT = mc.ROOT
AUGUST = (pd.Timestamp('2021-08-01'), pd.Timestamp('2021-09-01'))
DATA_END = pd.Timestamp('2021-09-01')
VACATION = (pd.Timestamp('2021-08-02'), pd.Timestamp('2021-08-14'))  # 8/2~8/13 휴무·재가동 주
NAMES = {'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost'}


def record(out, eid, model, profile, feature, setting, metrics, folder, extra):
    mc.write_record(Path(out)/'records'/f'{eid}.md', {
        '상태': '완료', '목적과 가설': '7월에 고른 설정 그대로 1~7월 재학습, 8월 마진 검증용 예측(8월로 고르지 않음)',
        '시작·종료 시각 및 시간대': f'{mc.now()}; Asia/Seoul',
        '모델·라이브러리 버전': f'Python {platform.python_version()}, {model}',
        '데이터 위치·버전 또는 해시·사용 범위': f'{mc.SOURCE.relative_to(ROOT)}; SHA256 {mc.digest(mc.SOURCE)}; timestamp_end ≤ 2021-09-01 00:00 (9월 미사용)',
        '목표값·단위·예측 구간': f'power_15min kW, {profile}',
        '학습·검증·테스트 분할과 누수 점검': f'학습 = 마지막 정답 칸 종료 ≤ 8/1 00:00, 평가 = 8월 원점(마지막 칸 종료 ≤ 9/1 00:00). {extra}',
        '전처리·특성·하이퍼파라미터·seed': f'{feature}, future, 월 칸 제거; {json.dumps(setting, ensure_ascii=False, default=str)}; seed 42',
        '이전 또는 기준 실험 링크와 변경 사항': '7월 고르기 run과 같은 설정, 학습 기간만 1~7월로 확장',
        '실행 명령·환경·코드 버전 또는 코드 식별 정보': f'python notebooks/_shared/m1007_august.py; 코드 SHA256 {mc.digest(__file__)}',
        '실제 실행 시간·자원 사용량(측정한 경우)': f"{metrics.get('seconds', '미측정')}",
        '평가 지표·구간별 또는 fold별 결과': '아래 표(8월, 참고용; 선정에 쓰지 않음)',
        '기준모델 대비 변화와 비교 가능 여부': '8월 기준선은 margin 단계 요약표 참조',
        '오차가 큰 조건·해석·한계': '8/2~8/13 휴무·재가동 주 포함. future는 사후 실측.',
        '오류·경고·실패 또는 중단 원인': '없음',
        '예측·모델·표·그래프·로그의 저장 경로': str(Path(folder).relative_to(ROOT)),
        '결론과 다음 실험': '8월 마진 검증 입력',
    }, [{k: metrics[k] for k in ['n_origins', 'Loss_C', 'RMSE', 'L_peak'] if k in metrics}])


def run_rf(args):
    import run_rf_m1007 as rf  # 월·b 칸 제거, F2 1h, 경계 포함 분할 패치가 import 시 적용된다
    from time import perf_counter
    out = Path(args.out)
    selection = json.loads((Path(args.rf_run)/'selection.json').read_text(encoding='utf-8'))
    df, repair, _ = rf.runner.previous.final.load_final_data()
    keep = df.index <= DATA_END
    df, repair = df.loc[keep].copy(), repair.loc[keep].copy()
    suite = ROOT/'artifacts/random_forest/runs'/(out.name + '_rf_august')
    rows = []
    for profile, s in selection.items():
        data = rf.fs.prepare_feature_sets(df, repair, profile, rf.WINDOW_HOURS)[s['feature_set']]
        families = list(dict.fromkeys([s['family'].replace('c_split', 'c_leaf'), s['family']]))
        t0 = perf_counter()
        result = rf.fit_job_m1007(suite, dict(source_sha256=mc.digest(mc.SOURCE)), data, 'm08', s['params'], families, float('inf'))
        if result['error']:
            raise RuntimeError(result['error'])
        row = [r for r in result['rows'] if r['scope'] == 'full' and r['family'] == s['family']][0]
        folder = suite/'cv'/profile/f"{rf.grid.parameter_id(s['params'])}_{s['family']}"/'m08'
        metrics = dict(n_origins=row['n_origins'], Loss_C=row['Loss_C'], RMSE=row['RMSE'], L_peak=row['L_peak'], seconds=perf_counter() - t0)
        eid = f"{profile}_{s['feature_set']}_{s['candidate']}_august"
        record(out, eid, f"scikit-learn {version('scikit-learn')}", profile, s['feature_set'], dict(params=s['params'], family=s['family']), metrics, folder,
               '잎 조정은 학습 데이터(1~7월) C로만.')
        rows.append(dict(model='RF', profile=profile, feature_set=s['feature_set'], candidate=s['candidate'], experiment_id=eid,
                         prediction=str((folder/'predictions.csv.gz').relative_to(ROOT)), **metrics))
        print(mc.now(), 'RF', profile, f"C={row['Loss_C']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(out/'members_rf.csv', index=False)


def run_boost(args):
    import run_boosting_m1007 as rb
    import full_c_data as fd
    from time import perf_counter
    out = Path(args.out)
    lib = args.library
    selection = json.loads((ROOT/'experiments'/lib/args.boost_run/'selection.json').read_text(encoding='utf-8'))
    dev = fd.load_frame('2021-09-01 00:01')
    copied = mc.copied_days(dev.set_index('timestamp').power_15min)
    art = ROOT/'artifacts/boosting/m1007_runs'/(out.name + '_august')/lib
    rows = []
    for profile, s in selection.items():
        data = fd.Dataset(dev, profile, s['feature_set'], 'future')
        eid = f"{lib}_{profile}_future_{s['feature_set']}_{s['candidate']}_august"
        t0 = perf_counter()
        m = rb.fit_two_stage(data, lib, rb.CANDIDATES[lib][int(s['candidate'][1:])], 'C', copied, art/eid, period=AUGUST)
        m['seconds'] = perf_counter() - t0
        record(out, eid, f'{lib} {version(lib)}', profile, s['feature_set'], dict(candidate=s['candidate'], params=m['params']), m, art/eid,
               f"나무 수: 확인 구간 {m['holdout_start']}~7/31(복사 안 된 날 14일) 최저 round {m['selected_round']} → 1~7월 재학습.")
        rows.append(dict(model=NAMES[lib], profile=profile, feature_set=s['feature_set'], candidate=s['candidate'], experiment_id=eid,
                         prediction=str((art/eid/'predictions.npz').relative_to(ROOT)), n_origins=m['n_origins'], Loss_C=m['Loss_C'],
                         RMSE=m['RMSE'], L_peak=m['L_peak'], rounds=m['selected_round'], holdout_start=m['holdout_start'], seconds=m['seconds']))
        print(mc.now(), lib, profile, f"C={m['Loss_C']:.4f} R={m['selected_round']}", flush=True)
    pd.DataFrame(rows).to_csv(out/f'members_{lib}.csv', index=False)


def episodes(times, near, window=pd.Timedelta(minutes=60)):
    """경보 시각을 60분 묶음으로 센다. 묶음 [시작, 시작+60분) 안에 실제 피크 근처 시각이 있으면 맞은 경보."""
    times = pd.DatetimeIndex(sorted(set(times)))
    near = pd.DatetimeIndex(sorted(set(near)))
    out, i = [], 0
    while i < len(times):
        start = times[i]
        j = i
        while j < len(times) and times[j] < start + window:
            j += 1
        hit = bool(((near >= start) & (near < start + window)).any())
        out.append(dict(start=start, n_alarm_times=j - i, true=hit))
        i = j
    return pd.DataFrame(out, columns=['start', 'n_alarm_times', 'true'])


def evaluate_alarm(cell, margin, near_times, days):
    alarm = cell.yhat >= (.9 - margin)*cell.b_origin
    alarmed = cell.loc[alarm, 'target_time']
    ep = episodes(alarmed, near_times)
    near_days = pd.DatetimeIndex(near_times).normalize().unique()
    caught = pd.DatetimeIndex(sorted(set(alarmed) & set(near_times))).normalize().unique()
    near_cell = cell.y >= .9*cell.b_t
    return dict(n_days=len(days), near_peak_days=len(near_days), caught_days=len(caught),
                day_recall=len(caught)/len(near_days) if len(near_days) else np.nan,
                near_peak_times=len(set(near_times)), caught_times=len(set(alarmed) & set(near_times)),
                cell_recall=float((alarm & near_cell).sum()/near_cell.sum()) if near_cell.any() else np.nan,
                alarm_episodes=len(ep), false_episodes=int((~ep.true).sum()) if len(ep) else 0,
                false_per_day=float((~ep.true).sum()/len(days)) if len(ep) else 0.)


def run_margin(args):
    import m1007_july_report as jr
    import full_c_data as fd
    import full_c_loss
    out = Path(args.out)
    july = Path(args.july_report)
    reps = json.loads((july/'representatives.json').read_text(encoding='utf-8'))
    table = pd.read_csv(july/'margin_table_july.csv')
    members = pd.concat([pd.read_csv(out/f'members_{k}.csv') for k in ['rf', 'xgboost', 'lightgbm', 'catboost']], ignore_index=True)
    dev = fd.load_frame('2021-09-01 00:01')
    b_series = full_c_loss.thresholds(dev.power_15min)
    summary, scores, preds = [], [], []
    for profile in mc.PROFILES:
        if reps[profile]['model'] != '단순평균 앙상블':
            raise ValueError(f'{profile}: 7월 대표가 앙상블이 아님 — 이 스크립트는 앙상블 대표만 처리')
        h = mc.HORIZON[profile]
        data = fd.Dataset(dev, profile, 'F2', 'future')
        _, vi = mc.july_split(data.origins, h, *AUGUST)
        origins, y, b = pd.DatetimeIndex(data.origins[vi]), data.y[vi], data.b[vi]
        member_preds = {}
        for r in members.loc[members.profile == profile].itertuples():
            if r.model == 'RF':
                o, p = jr.read_rf_prediction(ROOT/r.prediction, h)
            else:
                z = np.load(ROOT/r.prediction)
                o, p = pd.DatetimeIndex(z['origins']), z['predicted']
            member_preds[r.model], _ = jr.aligned(origins, o, p, r.experiment_id)
        if sorted(member_preds) != sorted(['RF', 'XGBoost', 'LightGBM', 'CatBoost']):
            raise ValueError(f'{profile}: 앙상블 구성원 누락 {sorted(member_preds)}')
        ens = np.mean([member_preds[k] for k in sorted(member_preds)], axis=0)
        cands = dict(member_preds, **{'단순평균 앙상블': ens}, **jr.baselines(dev, origins, h))
        for name, p in cands.items():
            s = mc.score(y, p, b, profile)
            scores.append(dict(horizon=profile, model=name, n_origins=len(origins), Loss_C_august=s['Loss_C'], RMSE_august=s['RMSE'], L_peak=s['L_peak']))
            preds.append(mc.long_predictions(origins, y, p, b, model=name, horizon=profile, period='8월'))
        cell = mc.long_predictions(origins, y, ens, b)
        cell['b_t'] = b_series.reindex(pd.DatetimeIndex(cell.target_time)).to_numpy()
        jr.label_conditions(cell, dev)
        tab = table.loc[table.profile == profile]
        base_m = float(tab.loc[tab['axes'] == '기본', 'margin'].iloc[0])
        cond = tab.loc[tab['axes'] != '기본']
        if len(cond):
            axes = cond['axes'].iloc[0].split(' × ')
            lookup = dict(zip(cond.group, cond.margin))
            # 7월에 합쳐진 수준(예: 'A+B')은 구성 수준으로 펼친다
            levels = {a: {} for a in axes}
            for g in cond.group:
                for a, lv in zip(axes, g.split(' / ')):
                    for part in lv.split('+'):
                        levels[a][part] = lv
            keys = [' / '.join(levels[a].get(v, v) for a, v in zip(axes, vals)) for vals in zip(*[cell[a] for a in axes])]
            cell['margin_cond'] = [lookup.get(k, base_m) for k in keys]
            cell['margin_group'] = keys
        else:
            cell['margin_cond'] = base_m
            cell['margin_group'] = '전체'
        near_times = cell.loc[cell.y >= .9*cell.b_t, 'target_time']
        cell['day'] = cell.target_time.dt.normalize()
        for scope, mask in [('8월 전체', np.ones(len(cell), bool)),
                            ('8/2~8/13 휴무·재가동', (cell.day >= VACATION[0]) & (cell.day < VACATION[1])),
                            ('8월 나머지', ~((cell.day >= VACATION[0]) & (cell.day < VACATION[1])))]:
            part = cell.loc[mask]
            days = pd.DatetimeIndex(part.day.unique())
            near_scope = near_times[near_times.dt.normalize().isin(days)]
            for variant, m in [('마진 없음(m=0)', 0.), ('기본 m만', base_m), ('조건별 마진', part.margin_cond)]:
                res = evaluate_alarm(part, m, near_scope, days)
                res.update(horizon=profile, scope=scope, variant=variant, base_m=base_m,
                           passed=bool(res['day_recall'] >= .8 and res['false_per_day'] <= 1) if np.isfinite(res['day_recall']) else None)
                summary.append(res)
        cell.assign(horizon=profile).to_csv(out/f'alarm_cells_{profile}.csv.gz', index=False, compression='gzip')
        print(mc.now(), 'MARGIN', profile, flush=True)
    cols = ['horizon', 'scope', 'variant', 'base_m', 'n_days', 'near_peak_days', 'caught_days', 'day_recall', 'false_per_day',
            'alarm_episodes', 'false_episodes', 'near_peak_times', 'caught_times', 'cell_recall', 'passed']
    pd.DataFrame(summary)[cols].to_csv(out/'august_margin_validation.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(scores).to_csv(out/'august_point_scores.csv', index=False, encoding='utf-8-sig')
    pd.concat(preds, ignore_index=True).to_csv(out/'predictions_august.csv.gz', index=False, compression='gzip')
    print(out)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['rf', 'boost', 'margin'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--rf-run')
    parser.add_argument('--boost-run')
    parser.add_argument('--library', choices=list(NAMES))
    parser.add_argument('--july-report')
    args = parser.parse_args(argv)
    Path(args.out, 'records').mkdir(parents=True, exist_ok=True)
    {'rf': run_rf, 'boost': run_boost, 'margin': run_margin}[args.part](args)


if __name__ == '__main__':
    main()
