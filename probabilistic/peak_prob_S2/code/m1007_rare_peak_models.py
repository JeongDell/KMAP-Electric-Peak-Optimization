"""드문 피크 사건용 새 확률 모델(M1 LightGBM 직접 분류, M2 로지스틱, M3 EVT 꼬리). 규칙:
docs/meeting_20261007_rerun_plan.md "드문 피크 사건용 새 확률 모델"(2026-10-08 04:48 KST, 실행 전 기록).

사건: 원점의 기간 사건 = 경로 중 한 번이라도 y > b_origin. 원점 단위 입력은 게이트 칸 변수를 요약한다(발행 시점 정보만).
검증: 달별 굴리기 5~8월(4d 7·8월). 9월: 3~8월로 학습해 9월 A판에 한 번 적용.
사용:
  python -u -B -X utf8 notebooks/_shared/m1007_rare_peak_models.py --out <dir>
"""
import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]/'notebooks/models/random_forest')]
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import m1007_common as mc  # noqa: E402
import m1007_ops_score as ops  # noqa: E402

ROOT = mc.ROOT
VALID = {p: ([7, 8] if p == 'long_4d' else [5, 6, 7, 8]) for p in mc.PROFILES}
RES_RUN = ROOT/'reports/20261007T235513_KST_36f6ca_m1007_residual_prob'
FEATURES = ['yhat_b_max', 'yhat_b_mean', 'maxmember_b_max', 'spread_b_mean', 'latest_b', 'max1h_b', 'max24h_b', 'b', 'b170',
            'day_of_month', 'hour', 'weekday', 'op_ratio', 'prod_max', 'prevday_zero']
LGB = dict(n_estimators=300, learning_rate=.03, num_leaves=15, min_child_samples=50, subsample=.8, subsample_freq=1,
           colsample_bytree=.8, reg_lambda=1., random_state=42, n_jobs=4, verbosity=-1)
TAIL_Q, MIN_EXCESS = .90, 10


def origin_table(cells):
    """칸 단위 게이트 표 → 원점 단위 입력·사건·경로 최대."""
    c = cells.copy()
    c['ev'] = c.y > c.b_origin
    g = c.groupby('origin', sort=True)
    t = pd.DataFrame(dict(
        month=g.month.first(), event=g.ev.any(), copied=g.copied_day.any(),
        y_max=g.y.max(), yhat_max=g.yhat.max(), b_origin=g.b_origin.first(),
        yhat_b_max=g.f_yhat_b.max(), yhat_b_mean=g.f_yhat_b.mean(), maxmember_b_max=g.f_maxmember_b.max(),
        spread_b_mean=g.f_spread_b.mean(), latest_b=g.f_latest_b.first(), max1h_b=g.f_max1h_b.first(),
        max24h_b=g.f_max24h_b.first(), b=g.f_b.first(), day_of_month=g.f_day_of_month.first(),
        op_ratio=g.f_daytype.apply(lambda s: float((s == 0).mean())), prod_max=g.f_prod_target.max(),
        prevday_zero=g.f_prevday_prod_zero.first()))
    t.index = pd.DatetimeIndex(t.index)
    t['b170'] = np.isclose(t.b, 170).astype(float)
    t['hour'] = t.index.hour + t.index.minute/60
    t['weekday'] = t.index.dayofweek.astype(float)
    return t


def load_dev_cells(profile):
    folder = ROOT/'reports'/ops.GATE[profile]
    tr = pd.read_csv(folder/f'gate_cells_train_{profile}.csv.gz')
    au = pd.read_csv(folder/f'gate_cells_aug_{profile}.csv.gz').assign(month=8, copied_day=False)
    c = pd.concat([tr, au], ignore_index=True)
    c['origin'] = pd.to_datetime(c.origin, format='mixed')
    return origin_table(c)


def sept_cells(profile, dev, copied, b_series):
    """9월 A판 4모델 예측으로 같은 게이트 칸 변수를 만든다(m1007_gate.build_cells)."""
    import m1007_gate as gm
    preds = {}
    for k in ['rf', 'xgboost', 'lightgbm', 'catboost']:
        m = pd.read_csv(RES_RUN/f'sept_members_{k}_A.csv')
        r = m.loc[m.profile == profile].iloc[0]
        preds[r.model] = r.prediction
    c = gm.build_cells(dev, profile, 9, preds, copied, b_series)
    c['month'] = 9
    return origin_table(c)


def fit_predict_clf(kind, train, test):
    x, y = train[FEATURES], train.event.astype(int)
    if y.sum() < 3 or y.sum() == len(y):
        return np.full(len(test), y.mean()), dict(fallback='학습 사건 3개 미만 → 학습 발생률', n_pos=int(y.sum()))
    if kind == 'M1':
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**LGB).fit(x, y)
        return model.predict_proba(test[FEATURES])[:, 1], dict(n_pos=int(y.sum()), model=model)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    model.fit(x, y)
    return model.predict_proba(test[FEATURES])[:, 1], dict(n_pos=int(y.sum()), model=model)


def evt_prob(pool, test):
    """POT: z = max y − max ŷ. d = b − max ŷ. d ≤ u이면 경험분포, 아니면 꼬리 확률 × GPD 생존함수."""
    z = (pool.y_max - pool.yhat_max).to_numpy(float)
    d = (test.b_origin - test.yhat_max).to_numpy(float)
    u = np.quantile(z, TAIL_Q)
    exc = z[z > u] - u
    emp = (z[None, :] > d[:, None]).mean(axis=1)
    if len(exc) < MIN_EXCESS:
        return emp, dict(u=float(u), n_excess=int(len(exc)), gpd=None)
    xi, _, sigma = stats.genpareto.fit(exc, floc=0)
    tail = (z > u).mean()*stats.genpareto.sf(d - u, xi, loc=0, scale=sigma)
    return np.where(d <= u, emp, tail), dict(u=float(u), n_excess=int(len(exc)), xi=float(xi), sigma=float(sigma))


def residual_probs(profile):
    v = pd.read_csv(RES_RUN/'validation_predictions_identified.csv.gz')
    v = v[(v.구간 == profile) & (v.창 == '직전 2달') & (v.lead == 1)]
    s = pd.read_csv(RES_RUN/'test_september_predictions.csv.gz')
    s = s[(s.구간 == profile) & (s.판 == 'A') & (s.lead == 1)]
    out = pd.concat([v[['origin', 'p_any']], s[['origin', 'p_any']]])
    out['origin'] = pd.to_datetime(out.origin, format='mixed')
    return out.drop_duplicates('origin').set_index('origin').p_any


def scores(ev, p, b170):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ev, p, hard = np.asarray(ev, bool), np.asarray(p, float), ~np.asarray(b170, bool)
    out = dict(n=len(ev), n_event=int(ev.sum()), Brier=float(np.mean((p - ev)**2)), mean_p=float(p.mean()),
               AP=float(average_precision_score(ev, p)) if ev.any() else np.nan,
               AUC=float(roc_auc_score(ev, p)) if 0 < ev.sum() < len(ev) else np.nan,
               hi_n=int((p >= .5).sum()), hi_hit=int(ev[p >= .5].sum()), n_hard_event=int(ev[hard].sum()),
               Brier_hard=float(np.mean((p[hard] - ev[hard])**2)) if hard.any() else np.nan,
               AP_hard=float(average_precision_score(ev[hard], p[hard])) if ev[hard].any() else np.nan,
               AUC_hard=float(roc_auc_score(ev[hard], p[hard])) if 0 < ev[hard].sum() < hard.sum() else np.nan)
    return out


def run(args):
    import full_c_data as fd
    import full_c_loss
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_all = perf_counter()
    dev = fd.load_frame('2021-09-15 00:01')
    copied = mc.copied_days(dev.set_index('timestamp').power_15min)
    b_series = full_c_loss.thresholds(dev.power_15min)
    rows, preds, meta = [], [], []
    for profile in args.profiles:
        t0 = perf_counter()
        tab = load_dev_cells(profile)
        tab = tab[~tab.copied]
        res = residual_probs(profile)
        sep = sept_cells(profile, dev, copied, b_series)
        sep = sep[~sep.copied]
        for phase, months in [('검증', VALID[profile]), ('9월', [9])]:
            for month in months:
                train = tab[tab.month < month]
                test = tab[tab.month == month] if month < 9 else sep
                pool = tab[(tab.month < month) & (tab.month >= month - 2)]
                if test.empty:
                    continue
                pr = dict(현행잔차=res.reindex(test.index).to_numpy(), 상수=np.full(len(test), pool.event.mean() if len(pool) else np.nan))
                for kind in ['M1', 'M2']:
                    pr[kind], info = fit_predict_clf(kind, train, test)
                    if month == 9 and 'model' in info:
                        joblib.dump(dict(model=info['model'], features=FEATURES, profile=profile, kind=kind, train_months='3~8'),
                                    out/f'model_{kind}_{profile}.joblib', compress=3)
                    meta.append(dict(구간=profile, 단계=phase, 채점달=month, 모델=kind, 학습원점=len(train), 학습사건=info.get('n_pos'),
                                     대체=info.get('fallback')))
                if len(pool):
                    pr['M3'], info = evt_prob(pool, test)
                    meta.append(dict(구간=profile, 단계=phase, 채점달=month, 모델='M3', 학습원점=len(pool), 학습사건=int(pool.event.sum()), **info))
                frame = test[['month', 'event', 'b170', 'b_origin', 'yhat_max', 'y_max']].copy()
                for k, v in pr.items():
                    frame[k] = v
                frame['구간'], frame['단계'] = profile, phase
                preds.append(frame.reset_index())
        allp = pd.concat([p for p in preds if p.구간.iloc[0] == profile])
        for phase, part in allp.groupby('단계'):
            part = part[part.현행잔차.notna()] if phase == '검증' else part
            for name in ['현행잔차', '상수', 'M1', 'M2', 'M3']:
                if name in part and part[name].notna().all():
                    rows.append(dict(구간=profile, 단계=phase, 모델=name, **scores(part.event, part[name], part.b170)))
        print(mc.now(), profile, f'{perf_counter() - t0:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(out/'rare_peak_scores.csv', index=False, encoding='utf-8-sig')
    pd.concat(preds).to_csv(out/'rare_peak_predictions.csv.gz', index=False, compression='gzip')
    pd.DataFrame(meta).to_csv(out/'rare_peak_fit_meta.csv', index=False, encoding='utf-8-sig')
    mc.dump(out/'config.json', dict(decision_doc=mc.DECISION_DOC, features=FEATURES, lgb=LGB, tail_q=TAIL_Q, min_excess=MIN_EXCESS,
                                    code=str(Path(__file__).relative_to(ROOT)), code_sha256=mc.digest(Path(__file__)),
                                    seconds=perf_counter() - t_all, python_executable=sys.executable,
                                    versions={k: version(k) for k in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'scipy']},
                                    python=platform.python_version()))
    print(mc.now(), 'done', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--profiles', nargs='+', default=mc.PROFILES, choices=mc.PROFILES)
    run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
