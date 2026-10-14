"""마진 게이트: 전력 예측·마진은 그대로 두고 '마진을 켤지'만 학습한다(규칙: docs/meeting_20261007_rerun_plan.md).

단계
  oos-rf    : RF 7월 선정 설정으로 3~6월 각 달 예측(그 달 시작 전 정답만 학습, 잎 조정은 학습 자료 C)
  oos-boost : 부스팅 7월 선정 설정으로 3~6월 각 달 예측(나무 수 = 그 달 시작 전 복사 안 된 날 14일 확인 구간)
  gate      : 3~7월 4모델 평균 예측으로 게이트 학습(달 하나씩 빼는 OOF로 τ) → 8월 평가
사용: python -B -X utf8 notebooks/_shared/m1007_gate.py <단계> --out <dir> [--library ...]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]/'notebooks/models/random_forest')]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import m1007_common as mc  # noqa: E402

ROOT = mc.ROOT
PROFILES = ['short_1h', 'medium_4h', 'medium_8h']
OOS_MONTHS = [3, 4, 5, 6]
RF_RUN = ROOT/'experiments/random_forest/20261007T081917_KST_f4847a_rf_m1007'
BOOST_RUN = '20261007T090243_KST_m1007_boost'
JULY_REPORT = ROOT/'reports/20261007T094057_KST_d80d75_m1007_july_selection'
AUGUST = ROOT/'reports/20261007T102741_KST_m1007_august_margin'
NAMES = {'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost'}
GATE_PARAMS = dict(n_estimators=300, learning_rate=.03, num_leaves=15, min_child_samples=100, subsample=.8, subsample_freq=1,
                   colsample_bytree=.8, reg_lambda=1., random_state=42, n_jobs=4, verbosity=-1)


def month_period(m):
    start = pd.Timestamp(2021, m, 1)
    return start, start + pd.offsets.MonthBegin(1)


def oos_rf(args):
    import run_rf_m1007 as rf
    out = Path(args.out)
    selection = json.loads((RF_RUN/'selection.json').read_text(encoding='utf-8'))
    df, repair, _ = rf.load_data()  # timestamp_end ≤ 8/1
    suite = ROOT/'artifacts/random_forest/runs'/(out.name + '_rf_oos')
    rows = []
    for profile in PROFILES:
        s = selection[profile]
        data = rf.fs.prepare_feature_sets(df, repair, profile, rf.WINDOW_HOURS)[s['feature_set']]
        families = list(dict.fromkeys([s['family'].replace('c_split', 'c_leaf'), s['family']]))
        for m in OOS_MONTHS:
            fold = f'm{m:02}'
            result = rf.fit_job_m1007(suite, dict(source_sha256=mc.digest(mc.SOURCE)), data, fold, s['params'], families, float('inf'))
            if result['error']:
                raise RuntimeError(result['error'])
            row = [r for r in result['rows'] if r['scope'] == 'full' and r['family'] == s['family']][0]
            path = suite/'cv'/profile/f"{rf.grid.parameter_id(s['params'])}_{s['family']}"/fold/'predictions.csv.gz'
            rows.append(dict(model='RF', profile=profile, month=m, Loss_C=row['Loss_C'], RMSE=row['RMSE'], n_origins=row['n_origins'],
                             prediction=str(path.relative_to(ROOT))))
            print(mc.now(), 'RF', profile, fold, f"C={row['Loss_C']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out/'oos_rf.csv', index=False)


def oos_boost(args):
    import run_boosting_m1007 as rb
    import full_c_data as fd
    out = Path(args.out)
    lib = args.library
    selection = json.loads((ROOT/'experiments'/lib/BOOST_RUN/'selection.json').read_text(encoding='utf-8'))
    dev = fd.load_frame('2021-08-01 00:01')
    copied = mc.copied_days(dev.set_index('timestamp').power_15min)
    art = ROOT/'artifacts/boosting/m1007_runs'/(out.name + '_oos')/lib
    rows = []
    for profile in PROFILES:
        s = selection[profile]
        data = fd.Dataset(dev, profile, s['feature_set'], 'future')
        for m in OOS_MONTHS:
            eid = f'{lib}_{profile}_{s["feature_set"]}_{s["candidate"]}_m{m:02}'
            if (art/eid/'metrics.json').exists():  # 중단 후 재실행: 완료된 적합은 저장 결과 재사용
                r = json.loads((art/eid/'metrics.json').read_text(encoding='utf-8'))
            else:
                try:
                    r = rb.fit_two_stage(data, lib, rb.CANDIDATES[lib][int(s['candidate'][1:])], 'C', copied, art/eid, period=month_period(m))
                except ValueError as error:
                    if '확인 구간 분할 비어 있음' not in str(error):
                        raise
                    # 4d 3·4월: 발행 가능 시작(1/29) 이후 확인 구간 앞에 학습 원점이 없다 → 그 달은 게이트 자료에서 제외
                    rows.append(dict(model=NAMES[lib], profile=profile, month=m, status='불가: 확인 구간 앞 학습 원점 없음'))
                    print(mc.now(), lib, profile, m, 'SKIP 확인 구간 앞 학습 원점 없음', flush=True)
                    continue
            rows.append(dict(model=NAMES[lib], profile=profile, month=m, status='완료', Loss_C=r['Loss_C'], RMSE=r['RMSE'], n_origins=r['n_origins'],
                             rounds=r['selected_round'], holdout_start=r['holdout_start'], prediction=str((art/eid/'predictions.npz').relative_to(ROOT))))
            print(mc.now(), lib, profile, m, f"C={r['Loss_C']:.3f} R={r['selected_round']} hs={r['holdout_start']}", flush=True)
            pd.DataFrame(rows).to_csv(out/f'oos_{lib}.csv', index=False)
    pd.DataFrame(rows).to_csv(out/f'oos_{lib}.csv', index=False)


def read_pred(path, h):
    import m1007_july_report as jr
    path = ROOT/path
    if str(path).endswith('.npz'):
        z = np.load(path)
        return pd.DatetimeIndex(z['origins']), z['predicted']
    return jr.read_rf_prediction(path, h)


def member_paths(out):
    """(profile, month) → {model: 예측 경로}. 3~6월 OOS, 7월 = 7월 고르기 선정 예측, 8월 = 8월 검증 예측."""
    paths = {}
    for f in ['oos_rf.csv'] + [f'oos_{lib}.csv' for lib in NAMES]:
        for r in pd.read_csv(Path(out)/f).itertuples():
            if getattr(r, 'status', '완료') == '완료' and isinstance(r.prediction, str):
                paths.setdefault((r.profile, r.month), {})[r.model] = r.prediction
    rf_sel = json.loads((RF_RUN/'selection.json').read_text(encoding='utf-8'))
    for profile in PROFILES:
        paths.setdefault((profile, 7), {})['RF'] = rf_sel[profile]['prediction']
        for lib in NAMES:
            sel = json.loads((ROOT/'experiments'/lib/BOOST_RUN/'selection.json').read_text(encoding='utf-8'))
            paths[(profile, 7)][NAMES[lib]] = sel[profile]['prediction']
    for f in ['members_rf.csv'] + [f'members_{lib}.csv' for lib in NAMES]:
        for r in pd.read_csv(AUGUST/f).itertuples():
            if r.profile in PROFILES:
                paths.setdefault((r.profile, 8), {})[r.model] = r.prediction
    return paths


def build_cells(dev, profile, month, preds, copied, b_series):
    import full_c_data as fd
    h = mc.HORIZON[profile]
    data = fd.Dataset(dev, profile, 'F2', 'future')
    _, vi = mc.july_split(data.origins, h, *month_period(month))
    origins, y, b = pd.DatetimeIndex(data.origins[vi]), data.y[vi], data.b[vi]
    import m1007_july_report as jr
    members = {}
    for model, path in preds.items():
        o, p = read_pred(path, h)
        members[model], _ = jr.aligned(origins, o, p, f'{model}/{profile}/{month}')
    if sorted(members) != sorted(['RF', 'XGBoost', 'LightGBM', 'CatBoost']):
        raise ValueError(f'{profile}/{month}: 구성원 {sorted(members)}')
    stack = np.stack([members[k] for k in sorted(members)])
    ens = stack.mean(axis=0)
    cell = mc.long_predictions(origins, y, ens, b)
    cell['month'] = month
    cell['b_t'] = b_series.reindex(pd.DatetimeIndex(cell.target_time)).to_numpy()
    cell['day'] = cell.target_time.dt.normalize()
    cell['near'] = cell.y >= .9*cell.b_t
    cell['copied_day'] = cell.day.isin(copied)
    # 게이트 입력(발행 시점에 알 수 있는 값)
    power = dev.power_15min
    o = pd.DatetimeIndex(cell.origin)
    cell['f_yhat_b'] = cell.yhat/cell.b_origin
    cell['f_spread_b'] = stack.std(axis=0).ravel()/cell.b_origin
    cell['f_maxmember_b'] = stack.max(axis=0).ravel()/cell.b_origin
    cell['f_latest_b'] = power.reindex(o).to_numpy()/cell.b_origin
    cell['f_max1h_b'] = power.rolling(4).max().reindex(o).to_numpy()/cell.b_origin
    cell['f_max24h_b'] = power.rolling(96).max().reindex(o).to_numpy()/cell.b_origin
    cell['f_b'] = cell.b_origin
    cell['f_lead'] = cell.lead
    cell['f_hour'] = cell.target_time.dt.hour + cell.target_time.dt.minute/60
    cell['f_weekday'] = cell.target_time.dt.dayofweek
    end = pd.DatetimeIndex(cell.target_time + pd.Timedelta(minutes=15))
    cell['f_daytype'] = dev.day_type.reindex(end).to_numpy()
    hourly = dev.set_index('timestamp')['생산량']
    cell['f_prod_target'] = hourly.reindex(cell.target_time.dt.floor('h')).to_numpy()
    daily = hourly.groupby(hourly.index.normalize()).sum()
    cell['f_prevday_prod_zero'] = (daily.reindex(cell.day - pd.Timedelta(days=1)).to_numpy() == 0).astype(float)
    cell['f_day_of_month'] = o.day + o.hour/24
    return cell


FEATURES = ['f_yhat_b', 'f_spread_b', 'f_maxmember_b', 'f_latest_b', 'f_max1h_b', 'f_max24h_b', 'f_b', 'f_lead', 'f_hour',
            'f_weekday', 'f_daytype', 'f_prod_target', 'f_prevday_prod_zero', 'f_day_of_month']


def alarms(cell, p, tau, m, mode):
    base = cell.yhat >= .9*cell.b_origin
    widened = cell.yhat >= (.9 - m)*cell.b_origin
    if mode == '마진 항상':
        return widened
    if mode == '마진 없음':
        return base
    if mode == '게이트만':
        return p >= tau
    return base | ((p >= tau) & widened)


def score(cell, alarm, counting):
    import m1007_august as ma
    times = cell.loc[alarm, 'target_time']
    near_times = cell.loc[cell.near, 'target_time']
    days = pd.DatetimeIndex(cell.day.unique())
    ep = ma.episodes(times, near_times) if counting == 'window60' else ma.episodes_run(times, near_times, pd.Timedelta(minutes=60))
    near_days = pd.DatetimeIndex(near_times).normalize().unique()
    caught = pd.DatetimeIndex(sorted(set(times) & set(near_times))).normalize().unique()
    n_ep = len(ep)
    false = int((~ep.true).sum()) if n_ep else 0
    return dict(near_peak_days=len(near_days), caught_days=len(caught), day_recall=len(caught)/len(near_days) if len(near_days) else np.nan,
                false_per_day=false/len(days), alarm_episodes=n_ep, precision=1 - false/n_ep if n_ep else np.nan,
                alarm_hours_per_day=len(set(times))*.25/len(days))


def gate(args):
    import full_c_data as fd
    import full_c_loss
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    out = Path(args.out)
    dev = fd.load_frame('2021-09-01 00:01')
    copied = mc.copied_days(dev.set_index('timestamp').power_15min)
    b_series = full_c_loss.thresholds(dev.power_15min)
    margins = pd.read_csv(JULY_REPORT/'margin_table_july.csv')
    paths = member_paths(out)
    results, taus, aucs, importances = [], [], [], []
    for profile in PROFILES:
        m = float(margins.loc[(margins.profile == profile) & (margins['axes'] == '기본'), 'margin'].iloc[0])
        # 4모델 예측이 모두 있는 달만 학습(4d는 3·4월 부스팅 확인 구간 불가 → 5~7월)
        months = [mo for mo in [3, 4, 5, 6, 7] if len(paths.get((profile, mo), {})) == 4]
        cells = {mo: build_cells(dev, profile, mo, paths[(profile, mo)], copied, b_series) for mo in months + [8]}
        train = pd.concat([cells[mo] for mo in months], ignore_index=True)
        fit_rows = train.loc[~train.copied_day]
        # OOF: 달 하나씩 빼고 학습 → 그 달 확률
        oof = {}
        for mo in months:
            tr = fit_rows.loc[fit_rows.month != mo]
            model = LGBMClassifier(**GATE_PARAMS).fit(tr[FEATURES], tr.near)
            oof[mo] = model.predict_proba(cells[mo][FEATURES])[:, 1]
        # τ: OOF 3~7월에서 날 재현율 ≥ 80% 중 하루 경보 시간 최소
        grid = np.unique(np.quantile(np.concatenate(list(oof.values())), np.linspace(.5, .999, 120)))
        best = None
        for tau in grid:
            caught = near = 0
            hours = days = 0.
            for mo in months:
                c = cells[mo]
                a = alarms(c, oof[mo], tau, m, '게이트+마진')
                near_times = c.loc[c.near & ~c.copied_day, 'target_time']
                nd = pd.DatetimeIndex(near_times).normalize().unique()
                cd = pd.DatetimeIndex(sorted(set(c.loc[a, 'target_time']) & set(near_times))).normalize().unique()
                caught += len(cd)
                near += len(nd)
                hours += len(set(c.loc[a, 'target_time']))*.25
                days += c.day.nunique()
            rec = caught/near if near else np.nan
            taus.append(dict(horizon=profile, tau=tau, oof_day_recall=rec, oof_alarm_hours_per_day=hours/days))
            if rec >= .8 and (best is None or hours/days < best[1]):
                best = (tau, hours/days, rec)
        if best is None:
            raise ValueError(f'{profile}: OOF에서 날 재현율 80%를 만드는 τ 없음')
        tau = best[0]
        final = LGBMClassifier(**GATE_PARAMS).fit(fit_rows[FEATURES], fit_rows.near)
        importances += [dict(horizon=profile, feature=f, gain=g) for f, g in zip(FEATURES, final.booster_.feature_importance('gain'))]
        aug = cells[8]
        p = final.predict_proba(aug[FEATURES])[:, 1]
        busy = aug.y >= .6*aug.b_t
        aucs.append(dict(horizon=profile, auc_all_yhat=roc_auc_score(aug.near, aug.f_yhat_b), auc_all_gate=roc_auc_score(aug.near, p),
                         auc_busy_yhat=roc_auc_score(aug.near[busy], aug.f_yhat_b[busy]), auc_busy_gate=roc_auc_score(aug.near[busy], p[busy]),
                         oof_auc_busy_gate=float(np.mean([roc_auc_score(cells[mo].near[cells[mo].y >= .6*cells[mo].b_t], oof[mo][cells[mo].y >= .6*cells[mo].b_t])
                                                          for mo in months if cells[mo].near[cells[mo].y >= .6*cells[mo].b_t].nunique() > 1])),
                         tau=tau, oof_day_recall=best[2], oof_alarm_hours=best[1], margin_m=m,
                         n_train_cells=len(fit_rows), n_train_near=int(fit_rows.near.sum()), train_months=','.join(map(str, months))))
        for mode in ['마진 없음', '마진 항상', '게이트만', '게이트+마진']:
            for counting in ['window60', 'run60']:
                res = score(aug, alarms(aug, p, tau, m, mode), counting)
                results.append(dict(horizon=profile, mode=mode, counting=counting, **res,
                                    passed=bool(res['day_recall'] >= .8 and res['false_per_day'] <= 1)))
        aug.assign(gate_p=p, horizon=profile).to_csv(out/f'gate_cells_aug_{profile}.csv.gz', index=False, compression='gzip')
        pd.concat([cells[mo].assign(gate_oof=oof[mo]) for mo in months]).assign(horizon=profile).to_csv(
            out/f'gate_cells_train_{profile}.csv.gz', index=False, compression='gzip')
        print(mc.now(), 'GATE', profile, f'tau={tau:.4f}', flush=True)
    pd.DataFrame(results).to_csv(out/'gate_august_results.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(aucs).to_csv(out/'gate_auc.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(taus).to_csv(out/'gate_tau_search_oof.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(importances).to_csv(out/'gate_feature_importance.csv', index=False, encoding='utf-8-sig')
    print(out)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['oos-rf', 'oos-boost', 'gate'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--library', choices=list(NAMES))
    parser.add_argument('--profiles', nargs='+', choices=mc.PROFILES, default=PROFILES)
    args = parser.parse_args(argv)
    PROFILES[:] = args.profiles  # 1h·4h·8h(기본) 또는 12h·24h·4d 확장 실행
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {'oos-rf': oos_rf, 'oos-boost': oos_boost, 'gate': gate}[args.part](args)


if __name__ == '__main__':
    main()
