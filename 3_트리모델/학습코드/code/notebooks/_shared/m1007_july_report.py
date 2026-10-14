"""10/07 회의 반영 재실험의 7월 고르기 표·기준선 4개·단순평균 앙상블·7월 마진 표.

입력: RF run(experiments/random_forest/<rf_run>), 부스팅 run(experiments/<lib>/<boost_run>).
모든 후보를 같은 7월 원점·같은 칸·같은 b로 다시 채점한다(공통 C, RMSE). 8·9월은 다루지 않는다.
마진 규칙(사용자 사전 확정, docs/meeting_20261007_rerun_plan.md):
  피크 근처 칸 y ≥ 0.9·b_origin, r = max(0, y−ŷ)/b_origin, 기본 m = r의 95% 지점.
  조건 후보: 요일(평일/토/일), 시간대(주간 09~18 / 야간), lead 묶음, 강수 여부, 생산량(>487 많음, 0 포함 ≤487 적음).
  조건 수준별 최소 크기 20칸 + 3일, 미달 수준은 같은 축의 다른 수준과 합침.
  뚜렷한 차이 = 칸 단위 Welch t-검정(수준 vs 나머지) p < 0.05 (사용자 결정 10/07 08:20; 칸 간 의존으로 p 과소 가능). 축당 최소 p, 최대 3개 축.
사용: python -B -X utf8 notebooks/_shared/m1007_july_report.py --rf-run <dir> --boost-run <run_id>
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]/'notebooks/models/random_forest')]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import m1007_common as mc  # noqa: E402
import full_c_data as fd  # noqa: E402

ROOT = mc.ROOT
LIBS = ['xgboost', 'lightgbm', 'catboost']
MIN_CELLS, MIN_DAYS, ALPHA, MAX_AXES = 20, 3, .05, 3
PRODUCTION_CUT = 487.0
# lead 묶음(7월 오차 열람 전 고정): 시간 단위 경계
LEAD_BINS = [0, 1, 4, 12, 24, 96]
LEAD_LABELS = ['≤1h', '1–4h', '4–12h', '12–24h', '>24h']


def july_reference(dev, profile):
    data = fd.Dataset(dev, profile, 'F2', 'future')
    _, vi = mc.july_split(data.origins, data.horizon)
    return pd.DatetimeIndex(data.origins[vi]), data.y[vi], data.b[vi]


def baselines(dev, origins, horizon, k_similar=4):
    power = dev.power_15min  # index = timestamp_end
    stamps = origins.to_numpy()[:, None] + np.arange(1, horizon + 1)[None, :]*np.timedelta64(15, 'm')
    out = {'직전값': np.repeat(power.reindex(origins).to_numpy()[:, None], horizon, axis=1)}
    for name, period in [('어제 같은 시각', 96), ('지난주 같은 시각', 672)]:
        offsets = (np.arange(horizon) % period) + 1 - period
        ref = origins.to_numpy()[:, None] + offsets[None, :]*np.timedelta64(15, 'm')
        out[name] = power.reindex(pd.DatetimeIndex(ref.ravel())).to_numpy().reshape(len(origins), horizon)
    out['비슷한 날 중앙값'] = similar_day(dev, origins, stamps, k_similar)
    return out


def day_group(day_type):
    # 팀원 기준선과 일치 확인한 정의: day_type 그대로(0 평일·1 토·2 일·3 공휴일), 최근 4일.
    # 7월 공통 C/RMSE 1h 22.41/23.01, 4d 25.06/23.24로 팀원 표와 동일(24h 22.54 vs 22.60).
    return np.asarray(day_type)


def similar_day(dev, origins, stamps, k):
    """맞힐 칸과 같은 날 종류(평일/토/일/공휴일)의 과거 날 중, 같은 시각 칸이 발행 시각까지 관측된 최근 k일의 중앙값."""
    power = dev.power_15min
    day_of = (dev.index - pd.Timedelta(minutes=15)).normalize()
    types = pd.Series(day_group(dev.day_type.to_numpy()), index=day_of).groupby(level=0).first()
    days = types.index
    flat = pd.DatetimeIndex(stamps.ravel())
    origin_flat = np.repeat(origins.to_numpy(), stamps.shape[1])
    target_day = (flat - pd.Timedelta(minutes=15)).normalize()
    tod = flat - (flat - pd.Timedelta(minutes=15)).normalize()
    tgt_type = types.reindex(target_day).to_numpy()
    result = np.full(len(flat), np.nan)
    by_type = {t: days[types.to_numpy() == t] for t in [0, 1, 2, 3]}
    for i in range(len(flat)):
        pool = by_type[tgt_type[i]]
        cand = pool[pool < target_day[i]]
        cand = cand[(cand + tod[i]).to_numpy() <= origin_flat[i]][-k:]
        if len(cand):
            result[i] = np.median(power.reindex(cand + tod[i]).to_numpy())
    return result.reshape(stamps.shape)


def load_rf(rf_run):
    rf_run = Path(rf_run)
    summary = pd.read_csv(rf_run/'summary.csv')
    selection = pd.read_json(rf_run/'selection.json')
    rows = []
    for r in summary.itertuples():
        rows.append(dict(model='RF', features=r.feature_set, loss='A' if r.family == 'A_unweighted' else 'C', stage=r.stage,
                         candidate=r.experiment_id, family=r.family, path=ROOT/r.prediction,
                         selected=bool(r.selected) if 'selected' in summary else False))
    return rows, selection


def read_rf_prediction(path, horizon):
    f = pd.read_csv(path, parse_dates=['origin', 'target_end'])
    f = f.sort_values(['origin', 'target_end'])
    origins = pd.DatetimeIndex(f.origin.unique())
    if len(f) != len(origins)*horizon:
        raise ValueError(f'{path}: 경로 길이 불일치')
    return origins, f.predicted.to_numpy().reshape(len(origins), horizon)


def load_boosting(run_id):
    rows = []
    for lib in LIBS:
        folder = ROOT/'experiments'/lib/run_id
        if not (folder/'summary.csv').exists():
            continue
        s = pd.read_csv(folder/'summary.csv')
        for r in s.itertuples():
            if r.status != '완료':
                continue
            rows.append(dict(model={'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost'}[lib], features=r.feature_set,
                             loss='A' if r.objective == 'A' else 'C', stage=r.stage, candidate=r.experiment_id, family=r.candidate,
                             path=ROOT/r.prediction, selected=bool(getattr(r, 'selected', False)), profile=r.profile))
    return rows


def aligned(origins_ref, origins, pred, name):
    pos = pd.Index(origins).get_indexer(origins_ref)
    if (pos < 0).any():
        raise ValueError(f'{name}: 7월 공통 원점 {int((pos < 0).sum())}개 누락')
    extra = len(origins) - len(origins_ref)
    return pred[pos], extra


def label_conditions(cell, dev):
    """맞힐 칸(target_time = 칸 시작 시각, lead)의 조건 라벨. 7월 마진 표와 8월 적용이 같은 함수를 쓴다."""
    dtype = dev.day_type.reindex(pd.DatetimeIndex(cell.target_time + pd.Timedelta(minutes=15))).to_numpy()
    cell['요일'] = np.select([dtype == 0, dtype == 1], ['평일', '토'], '일·공휴일')
    hour = cell.target_time.dt.hour
    cell['시간대'] = np.where((hour >= 9) & (hour < 18), '주간', '야간')
    hours = cell.lead*.25
    cell['lead 묶음'] = pd.cut(hours, LEAD_BINS, labels=LEAD_LABELS).astype(str)
    hourly = dev.set_index('timestamp')
    key = cell.target_time.dt.floor('h')
    cell['강수'] = np.where(hourly['강수량'].reindex(key).to_numpy() > 0, '강수 있음', '강수 없음')
    cell['생산량'] = np.where(hourly['생산량'].reindex(key).to_numpy() > PRODUCTION_CUT, '많음', '적음')
    return cell


def margin_analysis(profile, origins, y, p, b, dev):
    h = y.shape[1]
    lead = np.tile(np.arange(1, h + 1), len(origins))
    origin = np.repeat(origins, h)
    end = origin + pd.to_timedelta(lead*15, unit='min')
    start = end - pd.Timedelta(minutes=15)
    cell = pd.DataFrame(dict(origin=origin, lead=lead, target_time=start, y=y.ravel(), yhat=p.ravel(), b=np.repeat(b, h)))
    cell = cell.loc[cell.y >= .9*cell.b].copy()
    cell['r'] = np.maximum(0, cell.y - cell.yhat)/cell.b
    cell['day'] = cell.target_time.dt.normalize()
    label_conditions(cell, dev)
    base_m = float(np.quantile(cell.r, .95)) if len(cell) else np.nan
    tests = []
    merged_axes = {}
    for axis in ['요일', '시간대', 'lead 묶음', '강수', '생산량']:
        labels = cell[axis].copy()
        # 최소 크기 미달 수준은 가장 작은 다른 수준과 합침(반복)
        while True:
            size = labels.groupby(labels).agg('size')
            days = cell.groupby(labels).day.nunique()
            small = [lv for lv in size.index if size[lv] < MIN_CELLS or days[lv] < MIN_DAYS]
            if not small or len(size) <= 1:
                break
            lv = small[0]
            other = size.drop(lv).sort_values().index[0]
            labels = labels.replace({lv: f'{other}+{lv}', other: f'{other}+{lv}'})
        merged_axes[axis] = labels
        for lv in labels.unique():
            if labels.nunique() <= 1:
                break
            # 사용자 결정(2026-10-07 08:20): 칸 단위 Welch t-검정. 같은 날·겹치는 경로 칸이 독립이 아니어서 p값이 작게 나올 수 있음(보고서에 한계 명시).
            inside = cell.r[labels == lv]
            rest = cell.r[labels != lv]
            if len(inside) < 2 or len(rest) < 2:
                continue
            t = stats.ttest_ind(inside, rest, equal_var=False)
            tests.append(dict(profile=profile, axis=axis, level=lv, n_cells=int((labels == lv).sum()), n_days=int(cell.day[labels == lv].nunique()),
                              mean_r=float(cell.r[labels == lv].mean()), mean_r_rest=float(cell.r[labels != lv].mean()),
                              q95_r=float(np.quantile(cell.r[labels == lv], .95)), p_value=float(t.pvalue)))
    tests = pd.DataFrame(tests)
    chosen = []
    if len(tests):
        best = tests.loc[tests.p_value < ALPHA].groupby('axis').p_value.min().sort_values()
        chosen = list(best.index[:MAX_AXES])
    table = []
    if chosen:
        combo = cell[[]].copy()
        combo['group'] = merged_axes[chosen[0]]
        for axis in chosen[1:]:
            combo['group'] = combo.group + ' / ' + merged_axes[axis]
        for g, part in cell.groupby(combo.group):
            ok = len(part) >= MIN_CELLS and part.day.nunique() >= MIN_DAYS
            table.append(dict(profile=profile, axes=' × '.join(chosen), group=g, n_cells=len(part), n_days=part.day.nunique(),
                              margin=float(np.quantile(part.r, .95)) if ok else base_m, source='조건 95% 지점' if ok else '최소 크기 미달→기본 m'))
    table.append(dict(profile=profile, axes='기본', group='전체', n_cells=len(cell), n_days=cell.day.nunique(), margin=base_m, source='전체 95% 지점'))
    return cell, tests, pd.DataFrame(table), chosen


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--rf-run', required=True)
    parser.add_argument('--boost-run', required=True)
    parser.add_argument('--profiles', nargs='+', default=mc.PROFILES)
    args = parser.parse_args(argv)
    out = ROOT/'reports'/mc.run_id('m1007_july_selection')
    out.mkdir(parents=True)
    dev = fd.load_frame('2021-08-01 00:01')
    rf_rows, _ = load_rf(args.rf_run)
    boost_rows = load_boosting(args.boost_run)
    table, preds, margins, tests_all, notes = [], [], [], [], []
    representatives = {}
    for profile in args.profiles:
        h = mc.HORIZON[profile]
        origins, y, b = july_reference(dev, profile)
        cands = {}
        for name, p in baselines(dev, origins, h).items():
            if not np.isfinite(p).all():
                raise ValueError(f'{profile}/{name}: 기준선 결측')
            cands[(name, '-', '-', 'baseline')] = p
        for r in [r for r in rf_rows if r['candidate'].startswith(profile + '_')]:
            o, p = read_rf_prediction(r['path'], h)
            p, extra = aligned(origins, o, p, r['candidate'])
            cands[('RF', r['features'], r['loss'], r['candidate'])] = p
            if extra:
                notes.append(f'{profile} {r["candidate"]}: 공통 원점 밖 RF 원점 {extra}개는 채점 제외')
        for r in [r for r in boost_rows if r['profile'] == profile]:
            z = np.load(r['path'])
            p, _ = aligned(origins, pd.DatetimeIndex(z['origins']), z['predicted'], r['candidate'])
            np.testing.assert_allclose(aligned(origins, pd.DatetimeIndex(z['origins']), z['actual'], 'y')[0], y)
            cands[(r['model'], r['features'], r['loss'], r['candidate'])] = p
        # 모델별 7월 대표(C 학습 중 최저) 단순 평균
        best = {}
        for (model, feat, loss, cid), p in cands.items():
            if loss == 'C':
                s = mc.score(y, p, b, profile)['Loss_C']
                if model not in best or s < best[model][0]:
                    best[model] = (s, cid, p)
        if len(best) >= 2:
            members = sorted(best)
            cands[('단순평균 앙상블', 'mixed', 'C', '+'.join(best[m][1] for m in members))] = np.mean([best[m][2] for m in members], axis=0)
        for (model, feat, loss, cid), p in cands.items():
            s = mc.score(y, p, b, profile)
            table.append(dict(horizon=profile, model=model, features=feat, loss=loss, candidate=cid, n_origins=len(origins),
                              n_cells=y.size, Loss_C_july=s['Loss_C'], RMSE_july=s['RMSE'], MAE_july=s['MAE'], L_peak=s['L_peak']))
            preds.append(mc.long_predictions(origins, y, p, b, model=model, features=feat, loss=loss, horizon=profile,
                                             seed=42 if loss != '-' else None, period='7월', candidate=cid))
        frame = pd.DataFrame([t for t in table if t['horizon'] == profile])
        models = frame.loc[frame.loss != '-']
        rep = models.sort_values(['Loss_C_july', 'candidate']).iloc[0]
        base_best = frame.loc[frame.loss == '-'].sort_values('Loss_C_july').iloc[0]
        representatives[profile] = dict(model=rep.model, features=rep.features, loss=rep.loss, candidate=rep.candidate,
                                        Loss_C_july=rep.Loss_C_july, RMSE_july=rep.RMSE_july,
                                        best_baseline=base_best.model, best_baseline_C=base_best.Loss_C_july,
                                        vs_baseline_pct=100*(rep.Loss_C_july/base_best.Loss_C_july - 1))
        cell, tests, margin, chosen = margin_analysis(profile, origins, y, cands[(rep.model, rep.features, rep.loss, rep.candidate)], b, dev)
        representatives[profile]['margin_axes'] = chosen
        margins.append(margin)
        tests_all.append(tests)
        cell.assign(horizon=profile).to_csv(out/f'near_peak_cells_{profile}.csv.gz', index=False, compression='gzip')
    table = pd.DataFrame(table)
    table['is_representative'] = [representatives[t.horizon]['candidate'] == t.candidate for t in table.itertuples()]
    table.to_csv(out/'july_selection_table.csv', index=False, encoding='utf-8-sig')
    pd.concat(preds, ignore_index=True).to_csv(out/'predictions_july.csv.gz', index=False, compression='gzip')
    pd.concat(margins, ignore_index=True).to_csv(out/'margin_table_july.csv', index=False, encoding='utf-8-sig')
    pd.concat(tests_all, ignore_index=True).to_csv(out/'margin_factor_tests_july.csv', index=False, encoding='utf-8-sig')
    mc.dump(out/'representatives.json', representatives)
    (out/'notes.txt').write_text('\n'.join(notes) or '없음', encoding='utf-8')
    print(out)
    return out


if __name__ == '__main__':
    main()
