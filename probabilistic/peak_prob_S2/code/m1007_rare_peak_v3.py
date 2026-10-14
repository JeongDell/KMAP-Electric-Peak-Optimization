"""드문 피크 3차: S1 쉬운 사건 분리, S2 혼합. 규칙: docs/meeting_20261007_rerun_plan.md
"드문 피크 3차 — 쉬운 사건 분리·혼합"(2026-10-08 기록). 9월은 1회 적용·보고만.
사용:
  python -u -B -X utf8 notebooks/_shared/m1007_rare_peak_v3.py --out <dir>
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
import m1007_rare_peak_models as v1  # noqa: E402
import m1007_rare_peak_v2 as v2  # noqa: E402

ROOT = mc.ROOT
V2_RUN = ROOT/'reports/20261008T045108_KST_b05309_m1007_rare_peak_v2'
S1_FEATS = v1.FEATURES + v2.NEW
S1_LEAVES, S1_CHILD = 7, 50


def run(args):
    import full_c_data as fd
    import full_c_loss
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((V2_RUN/'v2_selection.json').read_text(encoding='utf-8'))
    dev = fd.load_frame('2021-09-15 00:01')
    copied = mc.copied_days(dev.set_index('timestamp').power_15min)
    b_series = full_c_loss.thresholds(dev.power_15min)
    rows, preds = [], []
    for profile in args.profiles:
        tab = v1.load_dev_cells(profile)
        tab = tab[~tab.copied].join(v2.peak_features(tab[~tab.copied], dev))
        sep = v1.sept_cells(profile, dev, copied, b_series)
        sep = sep[~sep.copied].join(v2.peak_features(sep[~sep.copied], dev))
        res = v1.residual_probs(profile)
        cfg = sel[profile]
        v2_feats = v1.FEATURES + (v2.NEW if cfg['변수'] == 'plus' else [])
        for phase, months in [('검증', v1.VALID[profile]), ('9월', [9])]:
            parts = []
            for month in months:
                train = tab[tab.month < month]
                test = tab[tab.month == month] if month < 9 else sep
                if test.empty:
                    continue
                r = res.reindex(test.index).to_numpy()
                p_v2 = v2.fit_prob(train, test, v2_feats, cfg['leaves'], cfg['child'])
                easy_tr = train.b170 > 0
                easy_te = (test.b170 > 0).to_numpy()
                p_hard = v2.fit_prob(train[~easy_tr], test, S1_FEATS, S1_LEAVES, S1_CHILD)
                pool = tab[(tab.month < month) & (tab.month >= month - 2)]
                parts.append(test[['event', 'b170', 'b_origin']].assign(
                    month=month, 현행잔차=r, v2=p_v2, S1=np.where(easy_te, r, p_hard), S2=.5*p_v2 + .5*r,
                    상수=pool.event.mean() if len(pool) else np.nan, 학습_어려운사건=int(train[~easy_tr].event.sum())))
            d = pd.concat(parts)
            d = d[d.현행잔차.notna()]
            for name in ['현행잔차', 'v2', 'S1', 'S2', '상수']:
                rows.append(dict(구간=profile, 단계=phase, 모델=name, **v1.scores(d.event, d[name], d.b170)))
            preds.append(d.assign(구간=profile, 단계=phase).reset_index())
        print(mc.now(), profile, flush=True)
        pd.DataFrame(rows).to_csv(out/'v3_scores.csv', index=False, encoding='utf-8-sig')
    pd.concat(preds).to_csv(out/'v3_predictions.csv.gz', index=False, compression='gzip')
    mc.dump(out/'config.json', dict(decision_doc=mc.DECISION_DOC, s1_features=S1_FEATS, s1=dict(leaves=S1_LEAVES, child=S1_CHILD, seeds=v2.SEEDS),
                                    s2='0.5*v2 + 0.5*현행잔차', v2_selection=sel, code=str(Path(__file__).relative_to(ROOT)),
                                    code_sha256=mc.digest(Path(__file__)), python_executable=sys.executable))
    print(mc.now(), 'ALL DONE', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--profiles', nargs='+', default=mc.PROFILES, choices=mc.PROFILES)
    run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
