"""드문 피크 2차: 새 피크 변수 + LightGBM 설정 탐색 + seed 평균. 규칙: docs/meeting_20261007_rerun_plan.md
"드문 피크 2차(장시간)"(2026-10-08 04:55 KST). 9월은 검증으로 설정을 고른 뒤 한 번만 적용.
사용:
  python -u -B -X utf8 notebooks/_shared/m1007_rare_peak_v2.py --out <dir>
"""
import argparse
import itertools
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]/'notebooks/models/random_forest')]
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
import m1007_common as mc  # noqa: E402
import m1007_rare_peak_models as v1  # noqa: E402

ROOT = mc.ROOT
NEW = ['yday_max_b', 'recent5_opmax_b', 'days_since_b_up', 'ramp1h_b', 'prev_month_max_b']
GRID = list(itertools.product([7, 15, 31], [20, 50, 100], ['base', 'plus']))
SEEDS = [42, 43, 44, 45, 46]


def peak_features(t, dev):
    """원점(timestamp_end)까지 관측된 전력으로 새 피크 관련 변수."""
    import full_c_loss
    p = dev.power_15min  # index = timestamp_end
    day = (p.index - pd.Timedelta(minutes=15)).normalize()
    daily_max = p.groupby(day).max()
    dtype = pd.Series(np.asarray(dev.day_type), index=day).groupby(level=0).first()
    b_series = full_c_loss.thresholds(p)
    up = b_series.diff().fillna(0) > 0
    last_up = pd.Series(np.where(up, b_series.index, pd.NaT), index=b_series.index).ffill()
    month = (p.index - pd.Timedelta(minutes=15)).to_period('M')
    month_max = p.groupby(month).max()
    o = t.index
    oday = (o - pd.Timedelta(minutes=15)).normalize()
    b = t.b_origin.to_numpy()
    op_days = dtype.index[dtype.to_numpy() == 0]
    rec5 = []
    for d in oday:
        prev = op_days[op_days < d][-5:]
        rec5.append(np.median(daily_max.reindex(prev).to_numpy()) if len(prev) else np.nan)
    pm = (o - pd.Timedelta(minutes=15)).to_period('M') - 1
    out = pd.DataFrame(index=o)
    out['yday_max_b'] = daily_max.reindex(oday - pd.Timedelta(days=1)).to_numpy()/b
    out['recent5_opmax_b'] = np.asarray(rec5)/b
    lu = pd.to_datetime(last_up.reindex(o).to_numpy())
    out['days_since_b_up'] = ((o - lu).total_seconds()/86400).fillna(31).to_numpy()
    out['ramp1h_b'] = (p.reindex(o).to_numpy() - p.reindex(o - pd.Timedelta(hours=1)).to_numpy())/b
    out['prev_month_max_b'] = month_max.reindex(pm).to_numpy()/b
    return out


def fit_prob(train, test, feats, leaves, child):
    y = train.event.astype(int)
    if y.sum() < 3:
        return np.full(len(test), y.mean())
    ps = []
    for s in SEEDS:
        m = LGBMClassifier(**dict(v1.LGB, num_leaves=leaves, min_child_samples=child, random_state=s)).fit(train[feats], y)
        ps.append(m.predict_proba(test[feats])[:, 1])
    return np.mean(ps, axis=0)


def run(args):
    import full_c_data as fd
    import full_c_loss
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dev = fd.load_frame('2021-09-15 00:01')
    copied = mc.copied_days(dev.set_index('timestamp').power_15min)
    b_series = full_c_loss.thresholds(dev.power_15min)
    rows, chosen, preds = [], {}, []
    for profile in args.profiles:
        t0 = perf_counter()
        tab = v1.load_dev_cells(profile)
        tab = tab[~tab.copied]
        tab = tab.join(peak_features(tab, dev))
        sep = v1.sept_cells(profile, dev, copied, b_series)
        sep = sep[~sep.copied].join(peak_features(sep[~sep.copied], dev))
        res = v1.residual_probs(profile)
        val = []
        for leaves, child, fs in GRID:
            feats = v1.FEATURES + (NEW if fs == 'plus' else [])
            parts = []
            for month in v1.VALID[profile]:
                test = tab[tab.month == month]
                if test.empty:
                    continue
                p = fit_prob(tab[tab.month < month], test, feats, leaves, child)
                parts.append(test[['event', 'b170']].assign(p=p, month=month, 현행잔차=res.reindex(test.index).to_numpy()))
            d = pd.concat(parts)
            s = v1.scores(d.event, d.p, d.b170)
            rows.append(dict(구간=profile, 단계='검증', leaves=leaves, child=child, 변수=fs, **s))
            val.append((s['Brier'], leaves, child, fs, d))
            print(mc.now(), profile, leaves, child, fs, f"Brier={s['Brier']:.5f} AP_hard={s['AP_hard']}", flush=True)
        val.sort(key=lambda r: (r[0], r[3] == 'plus', r[1], -r[2]))
        best = val[0]
        chosen[profile] = dict(leaves=best[1], child=best[2], 변수=best[3], 검증Brier=best[0])
        d = best[4]
        d = d[d.현행잔차.notna()]
        rows.append(dict(구간=profile, 단계='검증(선정, 현행 비교 원점)', leaves=best[1], child=best[2], 변수=best[3], **v1.scores(d.event, d.p, d.b170)))
        rows.append(dict(구간=profile, 단계='검증(현행 잔차, 같은 원점)', **v1.scores(d.event, d.현행잔차, d.b170)))
        feats = v1.FEATURES + (NEW if best[3] == 'plus' else [])
        p9 = fit_prob(tab, sep, feats, best[1], best[2])
        r9 = res.reindex(sep.index).to_numpy()
        rows.append(dict(구간=profile, 단계='9월(선정 설정)', leaves=best[1], child=best[2], 변수=best[3], **v1.scores(sep.event, p9, sep.b170)))
        rows.append(dict(구간=profile, 단계='9월(현행 잔차)', **v1.scores(sep.event, r9, sep.b170)))
        preds.append(sep[['event', 'b170', 'b_origin']].assign(p=p9, 현행잔차=r9, 구간=profile).reset_index())
        models = [LGBMClassifier(**dict(v1.LGB, num_leaves=best[1], min_child_samples=best[2], random_state=s)).fit(tab[feats], tab.event.astype(int))
                  for s in SEEDS] if tab.event.sum() >= 3 else None
        joblib.dump(dict(models=models, features=feats, profile=profile, config=chosen[profile], train='3~8월'),
                    out/f'model_v2_{profile}.joblib', compress=3)
        pd.DataFrame(rows).to_csv(out/'v2_scores.csv', index=False, encoding='utf-8-sig')
        mc.dump(out/'v2_selection.json', chosen)
        print(mc.now(), profile, 'done', f'{perf_counter() - t0:.0f}s', flush=True)
    pd.concat(preds).to_csv(out/'v2_september_predictions.csv.gz', index=False, compression='gzip')
    mc.dump(out/'config.json', dict(decision_doc=mc.DECISION_DOC, new_features=NEW, grid=GRID, seeds=SEEDS,
                                    code=str(Path(__file__).relative_to(ROOT)), code_sha256=mc.digest(Path(__file__)),
                                    python_executable=sys.executable))
    print(mc.now(), 'ALL DONE', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--profiles', nargs='+', default=mc.PROFILES, choices=mc.PROFILES)
    run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
