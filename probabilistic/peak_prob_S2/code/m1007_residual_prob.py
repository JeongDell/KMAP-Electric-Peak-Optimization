"""서브 출력: m1007 4모델 평균 ŷ의 잔차 경로 분포로 피크 초과확률. 규칙은 실행 전 기록한
docs/meeting_20261007_rerun_plan.md "서브 출력: 잔차 경로 분포 피크 초과확률"(2026-10-07 23:50 KST).

- 주 사건: 시점 y(t+h) > b_origin, 기간 = 경로 중 한 번이라도 > b_origin. 보조: y ≥ 0.9·b_origin.
- 분포: residual_peak_distribution.ResidualPathDistribution(E0 raw, 균등 질량, 보정 없음). 시뮬레이션은 계산하지 않음(표시용).
- 복사된 날: 경로의 정답 칸(시작 시각의 날짜) 중 하나라도 복사된 날이면 그 원점을 풀·채점 모두에서 제외(완전 경로만).
- 검증: 1h~24h 5·6·7·8월, 4d 7·8월. 풀 창 (가) 이전 전체 (나) 직전 2달. 구간별 합친 score 낮은 창, 동률이면 (가).
사용:
  python -u -B -X utf8 notebooks/_shared/m1007_residual_prob.py validate --out <dir>
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import m1007_common as mc  # noqa: E402
import m1007_ops_score as ops  # noqa: E402
import residual_peak_distribution as rpd  # noqa: E402

ROOT = mc.ROOT
VALID_MONTHS = {p: ([7, 8] if p == 'long_4d' else [5, 6, 7, 8]) for p in mc.PROFILES}
WINDOWS = {'이전 전체': None, '직전 2달': 2}
NEAR = .9
BINS = [0, .01, .05, .1, .2, .3, .5, .7, 1.0000001]


def copied_set():
    import full_c_data as fd
    dev = fd.load_frame('2021-12-31')
    return mc.copied_days(dev.set_index('timestamp').power_15min)


def to_paths(cells, horizon, copied):
    """한 줄 = 한 칸 → 원점별 (N,H) 경로. 칸 정렬·b 일정·완전성을 확인한다."""
    c = cells.sort_values(['origin', 'lead'])
    origins = pd.DatetimeIndex(c.origin.drop_duplicates())
    n = len(origins)
    if len(c) != n*horizon or not (c.lead.to_numpy().reshape(n, horizon) == np.arange(1, horizon + 1)).all():
        raise ValueError(f'불완전 경로: 칸 {len(c)} / 원점 {n} × H {horizon}')
    y = c.y.to_numpy(float).reshape(n, horizon)
    yhat = c.yhat.to_numpy(float).reshape(n, horizon)
    b = c.b_origin.to_numpy(float).reshape(n, horizon)
    if not np.allclose(b, b[:, :1]):
        raise ValueError('원점 안에서 b_origin이 달라짐')
    starts = origins.to_numpy()[:, None] + np.arange(horizon)[None, :]*np.timedelta64(15, 'm')
    days = pd.DatetimeIndex(starts.ravel()).normalize()
    clean = ~np.asarray(days.isin(copied)).reshape(n, horizon).any(axis=1)
    return dict(origins=origins, y=y, yhat=yhat, b=b[:, 0], clean=clean,
                end=origins + pd.Timedelta(minutes=15*horizon))


def load_oof(profile, copied):
    """달별 굴리기 OOF: 3~7월(4d 5~7월) = 각 달 이전 정답으로 학습한 4모델 평균, 8월 = 1~7월 재학습 모델."""
    folder = ROOT/'reports'/ops.GATE[profile]
    cols = ['origin', 'lead', 'target_time', 'y', 'yhat', 'b_origin']
    out = {}
    tr = pd.read_csv(folder/f'gate_cells_train_{profile}.csv.gz', usecols=cols + ['month'])
    au = pd.read_csv(folder/f'gate_cells_aug_{profile}.csv.gz', usecols=cols).assign(month=8)
    for frame in (tr, au):
        for col in ['origin', 'target_time']:
            frame[col] = pd.to_datetime(frame[col], format='mixed')
        for month, part in frame.groupby('month'):
            out[int(month)] = to_paths(part, mc.HORIZON[profile], copied)
    return out, [folder/f'gate_cells_train_{profile}.csv.gz', folder/f'gate_cells_aug_{profile}.csv.gz']


def pool_for(oof, month, window):
    months = sorted(m for m in oof if m < month)
    if window:
        months = [m for m in months if m >= month - window]
    start = pd.Timestamp(2021, month, 1)
    e, y, bs, origins, audit = [], [], [], [], dict(months=','.join(map(str, months)), candidates=0, excluded_copied=0, excluded_end=0)
    for m in months:
        p = oof[m]
        in_time = np.asarray(p['end'] <= start)
        keep = in_time & p['clean']
        audit['candidates'] += len(keep)
        audit['excluded_end'] += int((~in_time).sum())
        audit['excluded_copied'] += int((in_time & ~p['clean']).sum())
        e.append((p['y'] - p['yhat'])[keep])
        y.append(p['y'][keep])
        bs.append(p['b'][keep])
        origins.append(p['origins'][keep])
    origins = pd.DatetimeIndex(np.concatenate([o.to_numpy() for o in origins]))
    audit['kept'] = len(origins)
    audit['last_target_end'] = str(origins.max() + pd.Timedelta(minutes=15*e[0].shape[1])) if len(origins) else None
    if len(origins) and origins.max() + pd.Timedelta(minutes=15*e[0].shape[1]) > start:
        raise ValueError('누수: 풀 경로가 채점 달 시작 뒤에 끝남')
    return np.concatenate(e), (np.concatenate(y), np.concatenate(bs)), origins, audit


def strict_above(threshold):
    """y ≥ x 를 엄격한 > 비교로 쓰기 위한 바로 아래 값."""
    return np.nextafter(np.asarray(threshold, float), -np.inf)


def predict_month(dist, hist, target, factor):
    """사건 y > factor·b(factor<1이면 y ≥ factor·b). hist = (풀 실제 경로, 풀 원점별 b).
    과거 빈도 = 풀 실제 경로에 현재 b 적용, 상수 발생률 = 풀 원점이 각자의 b로 초과한 비율(2026-10-08 사후 추가 기준)."""
    def thr(b):
        return b if factor == 1 else strict_above(factor*b)
    threshold = thr(target['b'])
    out = dist.predict(target['yhat'], threshold)
    hist_y, hist_b = hist
    hp, hq = rpd.history_probabilities(hist_y, threshold)
    pool_event = hist_y > thr(hist_b)[:, None]
    shape = target['yhat'].shape
    return dict(p_point=out['p_point_raw'], p_any=out['p_any_raw'], q=out['q_raw_kw'], h_point=hp, h_any=hq,
                c_point=np.full(shape, pool_event.mean()), c_any=np.full(shape[0], pool_event.any(axis=1).mean()),
                event=target['y'] > threshold[:, None])


def method_rows(base, main, aux, extra=None):
    """주 사건(> b)·보조 사건(≥ 0.9b)별 잔차 분포·과거 빈도·상수 발생률·항상 0 점수 행."""
    rows = []
    for 사건, r in [('> b', main), ('≥ 0.9b', aux)]:
        ep, ea = r['event'], r['event'].any(axis=1)
        for 방법, pp, pa in [('잔차 분포', r['p_point'], r['p_any']), ('과거 빈도', r['h_point'], r['h_any']),
                            ('상수 발생률', r['c_point'], r['c_any']), ('항상 0', np.zeros_like(r['p_point']), np.zeros_like(r['p_any']))]:
            row = dict(base, 방법=방법, 사건=사건, **scores(ep, ea, pp, pa))
            if 방법 == '잔차 분포' and 사건 == '> b' and extra:
                row.update(extra)
            rows.append(row)
    return rows


def stack(parts):
    """채점 달별 predict_month 결과를 합친다."""
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def scores(event_point, event_any, p_point, p_any):
    sp, sa = rpd.binary_scores(event_point, p_point), rpd.binary_scores(event_any, p_any)
    return dict(score=.5*sp['Brier'] + .5*sa['Brier'], Brier_point=sp['Brier'], Brier_period=sa['Brier'],
                logloss_point=sp['logloss'], logloss_period=sa['logloss'], AP_point=sp['AP'], AP_period=sa['AP'],
                n_cells=sp['n'], n_event_cells=sp['n_positive'], n_origins=sa['n'], n_event_origins=sa['n_positive'],
                mean_p_point=sp['mean_probability'], mean_p_period=sa['mean_probability'])


def reliability(event, p):
    event, p = np.asarray(event, bool).ravel(), np.asarray(p, float).ravel()
    bins = pd.cut(p, BINS, right=False, include_lowest=True)
    t = pd.DataFrame(dict(bin=bins, p=p, e=event)).groupby('bin', observed=False).agg(n=('e', 'size'), mean_p=('p', 'mean'), event_rate=('e', 'mean'))
    return t.reset_index().assign(bin=lambda d: d['bin'].astype(str))


def run_validate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = perf_counter()
    copied = copied_set()
    rows, pools, cells_out, rel_rows, sources = [], [], {}, [], {}
    for profile in args.profiles:
        oof, files = load_oof(profile, copied)
        sources[profile] = {str(f.relative_to(ROOT)): mc.digest(f) for f in files}
        h = mc.HORIZON[profile]
        for window, span in WINDOWS.items():
            mains, auxs, ys = [], [], []
            for month in VALID_MONTHS[profile]:
                t0 = perf_counter()
                e, hist, pool_origins, audit = pool_for(oof, month, span)
                target = oof[month]
                keep = target['clean'] & np.asarray(target['end'] <= pd.Timestamp(2021, month, 1) + pd.offsets.MonthBegin(1))
                tgt = {k: v[keep] for k, v in target.items()}
                dist = rpd.ResidualPathDistribution().fit(e, pool_origins)
                main = predict_month(dist, hist, tgt, 1)
                aux = predict_month(dist, hist, tgt, NEAR)
                starts = tgt['origins'].to_numpy()[:, None] + np.arange(h)[None, :]*np.timedelta64(15, 'm')
                base = dict(구간=profile, 창=window, 채점달=month, 풀달=audit['months'], 풀경로=audit['kept'],
                            풀제외_복사=audit['excluded_copied'], 채점원점=int(keep.sum()), 채점제외=int((~keep).sum()),
                            초과날=pd.DatetimeIndex(starts[main['event']]).normalize().nunique())
                q = rpd.quantile_scores(tgt['y'], main['q'])
                rows += method_rows(base, main, aux, dict(coverage90=q['coverage90'], interval90_width_kw=q['interval90_width_kw'],
                                                          seconds=perf_counter() - t0))
                pools.append(dict(구간=profile, 창=window, 채점달=month, **audit))
                mains.append(main), auxs.append(aux), ys.append(tgt['y'])
                cells_out[(profile, window, month)] = mc.long_predictions(tgt['origins'], tgt['y'], tgt['yhat'], tgt['b']).assign(
                    p_point=main['p_point'].ravel(), p_any=np.repeat(main['p_any'], h), p_point_near=aux['p_point'].ravel(),
                    p_any_near=np.repeat(aux['p_any'], h), 창=window, 채점달=month)
                print(mc.now(), profile, window, month, f"pool={audit['kept']}", flush=True)
            main, aux, y = stack(mains), stack(auxs), np.concatenate(ys)
            q = rpd.quantile_scores(y, main['q'])
            rows += method_rows(dict(구간=profile, 창=window, 채점달='합침', 채점원점=len(y)), main, aux,
                                dict(coverage90=q['coverage90'], interval90_width_kw=q['interval90_width_kw']))
            for kind, ev, p in [('시점', main['event'], main['p_point']), ('기간', main['event'].any(1), main['p_any'])]:
                rel_rows.append(reliability(ev, p).assign(구간=profile, 창=window, 확률=kind))
    table = pd.DataFrame(rows)
    table.to_csv(out/'validation_scores.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(pools).to_csv(out/'pool_audit.csv', index=False, encoding='utf-8-sig')
    pd.concat(rel_rows).to_csv(out/'reliability.csv', index=False, encoding='utf-8-sig')
    pd.concat(cells_out.values()).to_csv(out/'validation_predictions.csv.gz', index=False, compression='gzip')
    selection = select(table)
    mc.dump(out/'selection.json', selection)
    mc.dump(out/'config.json', dict(decision_doc=mc.DECISION_DOC, sources=sources, valid_months=VALID_MONTHS, windows=list(WINDOWS),
                                    copied_days=[str(d.date()) for d in copied], code=str(Path(__file__).relative_to(ROOT)),
                                    code_sha256=mc.digest(Path(__file__)), seconds=perf_counter() - t_all,
                                    python_executable=sys.executable,
                                    versions=dict(python=platform.python_version(), numpy=version('numpy'), pandas=version('pandas'),
                                                  scikit_learn=version('scikit-learn'))))
    print(json.dumps(selection, ensure_ascii=False, indent=1))


def select(table):
    """구간별 합친 score(> b, 잔차 분포)가 낮은 창. 동률이면 이전 전체. 과거 빈도보다 나쁘면 미제공."""
    t = table[(table.채점달 == '합침') & (table.사건 == '> b')]
    out = {}
    for profile, part in t.groupby('구간'):
        res = part[part.방법 == '잔차 분포'].set_index('창').score
        hist = part[part.방법 == '과거 빈도'].set_index('창').score
        const = part[part.방법 == '상수 발생률'].set_index('창').score
        best = '이전 전체' if res['이전 전체'] <= res['직전 2달'] else '직전 2달'
        fails = [name for name, ref in [('과거 빈도', hist[best]), ('상수 발생률', const[best])] if not res[best] < ref]
        out[profile] = dict(창=best, score=float(res[best]), 과거빈도=float(hist[best]), 상수발생률=float(const[best]),
                            상태='제공' if not fails else f"미제공({'·'.join(fails)}보다 낮지 않음; 대체 제공 방식 보류)",
                            기존판정_과거빈도만='제공' if res[best] < hist[best] else '미제공',
                            다른창_score=float(res.drop(best).iloc[0]))
    return out


SEPT = (pd.Timestamp('2021-09-01'), pd.Timestamp('2021-09-15'))
SEPT_FRAME_END = '2021-09-15 00:01'
AUGUST_RUN = ROOT/'reports/20261007T102741_KST_m1007_august_margin'
NAMES = {'xgboost': 'XGBoost', 'lightgbm': 'LightGBM', 'catboost': 'CatBoost'}
AUG = (pd.Timestamp('2021-08-01'), pd.Timestamp('2021-09-01'))


def sept_rf(args):
    """RF 9월 예측. A = 1~8월 재학습(fold m09: 학습 정답 끝 ≤ 9/1), B = 1~7월 모델(8월 검증 모델) 재사용.
    9월 점수는 출력하지 않는다(시험 단계에서 한 번 계산)."""
    import joblib
    import run_rf_m1007 as rf
    out = Path(args.out).resolve()
    members = pd.read_csv(AUGUST_RUN/'members_rf.csv')
    selection = json.loads((ROOT/'experiments/random_forest/20261007T081917_KST_f4847a_rf_m1007/selection.json').read_text(encoding='utf-8'))
    df, repair, _ = rf.runner.previous.final.load_final_data()
    keep = df.index <= SEPT[1]
    df, repair = df.loc[keep].copy(), repair.loc[keep].copy()
    suite = ROOT/'artifacts/random_forest/runs'/(out.name + '_rf_sept')
    rows = []
    for profile in args.profiles:
        s = selection[profile]
        h = mc.HORIZON[profile]
        data = rf.fs.prepare_feature_sets(df, repair, profile, rf.WINDOW_HOURS)[s['feature_set']]
        t0 = perf_counter()
        if args.variant == 'A':
            families = list(dict.fromkeys([s['family'].replace('c_split', 'c_leaf'), s['family']]))
            result = rf.fit_job_m1007(suite, dict(source_sha256=mc.digest(mc.SOURCE)), data, 'm09', s['params'], families, float('inf'))
            if result['error']:
                raise RuntimeError(result['error'])
            path = suite/'cv'/profile/f"{rf.grid.parameter_id(s['params'])}_{s['family']}"/'m09'/'predictions.csv.gz'
        else:
            aug_pred = Path(members.loc[members.profile == profile, 'prediction'].iloc[0])
            model = joblib.load(ROOT/aug_pred.parent/'model_bundle.joblib')['model']
            origins = pd.DatetimeIndex(data['origins'])
            # 재사용 점검: 새로 만든 입력으로 8월을 다시 예측해 저장된 8월 예측과 같아야 한다(입력 생성의 미래 정보 혼입 점검).
            _, va = rf.split_inclusive(origins, h, *AUG)
            o_saved, p_saved = __import__('m1007_july_report').read_rf_prediction(ROOT/aug_pred, h)
            p_again = model.predict(data['features'].iloc[va].to_numpy(np.float32))
            if not (origins[va].equals(o_saved) and np.allclose(p_again, p_saved, atol=1e-6)):
                raise ValueError(f'{profile}: 8월 재예측 불일치 — 입력 생성이 달라짐')
            _, vi = rf.split_inclusive(origins, h, *SEPT)
            predicted = model.predict(data['features'].iloc[vi].to_numpy(np.float32))
            path = out/'sept_members'/'B'/f'rf_{profile}.csv.gz'
            path.parent.mkdir(parents=True, exist_ok=True)
            o = origins[vi]
            pd.DataFrame(dict(origin=np.repeat(o, h), target_end=(o.to_numpy()[:, None] + np.arange(1, h + 1)*np.timedelta64(15, 'm')).ravel(),
                              predicted=predicted.ravel())).to_csv(path, index=False, compression='gzip')
        rows.append(dict(model='RF', profile=profile, variant=args.variant, feature_set=s['feature_set'], candidate=s['candidate'],
                         prediction=str(path.relative_to(ROOT)), seconds=perf_counter() - t0))
        print(mc.now(), 'RF', args.variant, profile, f'{perf_counter() - t0:.0f}s', flush=True)
    frame = pd.DataFrame(rows)
    target = out/f'sept_members_rf_{args.variant}.csv'
    if target.exists():
        frame = pd.concat([pd.read_csv(target), frame]).drop_duplicates(['profile'], keep='last')
    frame.to_csv(target, index=False)


def sept_boost(args):
    """부스팅 9월 예측. A = 1~8월 재학습(나무 수: 9/1 직전 복사 안 된 날 14일 확인 구간), B = 1~7월 모델 재사용."""
    import joblib
    import run_boosting_m1007 as rb
    import full_c_data as fd
    out = Path(args.out).resolve()
    lib = args.library
    members = pd.read_csv(AUGUST_RUN/f'members_{lib}.csv')
    dev = fd.load_frame(SEPT_FRAME_END)
    copied = mc.copied_days(dev.set_index('timestamp').power_15min)
    art = ROOT/'artifacts/boosting/m1007_runs'/(out.name + '_sept')/lib
    rows = []
    for r in members.itertuples():
        if r.profile not in args.profiles:
            continue
        data = fd.Dataset(dev, r.profile, r.feature_set, 'future')
        t0 = perf_counter()
        eid = f'{lib}_{r.profile}_future_{r.feature_set}_{r.candidate}_sept{args.variant}'
        if args.variant == 'A':
            m = rb.fit_two_stage(data, lib, rb.CANDIDATES[lib][int(r.candidate[1:])], 'C', copied, art/eid, period=SEPT)
            path = art/eid/'predictions.npz'
            extra = dict(rounds=m['selected_round'], holdout_start=m['holdout_start'], train_last_target=m['train_last_target'])
        else:
            bundle = joblib.load(ROOT/Path(r.prediction).parent/'model.joblib')
            # 재사용 점검: 같은 입력 생성으로 8월을 다시 예측해 저장값과 비교
            _, va = mc.july_split(data.origins, data.horizon, *AUG)
            xa = data.inputs(va)
            if list(xa.columns) != bundle['feature_columns']:
                raise ValueError(f'{eid}: 입력 칸 불일치')
            saved = np.load(ROOT/r.prediction)
            if not np.allclose(bundle['model'].predict(xa.to_numpy()), saved['predicted'], atol=1e-6):
                raise ValueError(f'{eid}: 8월 재예측 불일치')
            _, vi = mc.july_split(data.origins, data.horizon, *SEPT)
            predicted = bundle['model'].predict(data.inputs(vi).to_numpy())
            path = out/'sept_members'/'B'/f'{lib}_{r.profile}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, origins=data.origins[vi].asi8, predicted=predicted)
            extra = dict(rounds=bundle['rounds'])
        rows.append(dict(model=NAMES[lib], profile=r.profile, variant=args.variant, feature_set=r.feature_set, candidate=r.candidate,
                         prediction=str(path.relative_to(ROOT)), seconds=perf_counter() - t0, **extra))
        print(mc.now(), lib, args.variant, r.profile, f'{perf_counter() - t0:.0f}s', flush=True)
    frame = pd.DataFrame(rows)
    target = out/f'sept_members_{lib}_{args.variant}.csv'
    if target.exists():
        frame = pd.concat([pd.read_csv(target), frame]).drop_duplicates(['profile'], keep='last')
    frame.to_csv(target, index=False)


def sept_ensemble(out, variant, profile, dev):
    """9월 원점·정답·b와 4모델 단순평균(구성원 원점 대조)."""
    import full_c_data as fd
    import m1007_july_report as jr
    h = mc.HORIZON[profile]
    data = fd.Dataset(dev, profile, 'F2', 'future')
    _, vi = mc.july_split(data.origins, h, *SEPT)
    origins, y, b = pd.DatetimeIndex(data.origins[vi]), data.y[vi], data.b[vi]
    preds = {}
    for lib in ['rf', 'xgboost', 'lightgbm', 'catboost']:
        m = pd.read_csv(out/f'sept_members_{lib}_{variant}.csv')
        r = m.loc[m.profile == profile].iloc[0]
        if lib == 'rf':
            o, p = jr.read_rf_prediction(ROOT/r.prediction, h)
        else:
            z = np.load(ROOT/r.prediction)
            o, p = pd.DatetimeIndex(z['origins']), z['predicted']
        preds[r.model], _ = jr.aligned(origins, o, p, f'{lib}_{profile}_{variant}')
    if sorted(preds) != ['CatBoost', 'LightGBM', 'RF', 'XGBoost']:
        raise ValueError(f'{profile} {variant}: 구성원 누락')
    return origins, y, b, np.mean([preds[k] for k in sorted(preds)], axis=0)


def check_b(profile, dev):
    """9월 b(fd.Dataset)가 검증에 쓴 b_origin과 같은 정의인지 8월로 대조."""
    import full_c_data as fd
    data = fd.Dataset(dev, profile, 'F2', 'future')
    _, va = mc.july_split(data.origins, mc.HORIZON[profile], *AUG)
    cells = pd.read_csv(ROOT/'reports'/ops.GATE[profile]/f'gate_cells_aug_{profile}.csv.gz', usecols=['origin', 'lead', 'b_origin'])
    gate_b = cells.loc[cells.lead == 1].assign(origin=lambda d: pd.to_datetime(d.origin, format='mixed')).set_index('origin').b_origin
    mine = pd.Series(data.b[va], index=pd.DatetimeIndex(data.origins[va]))
    if not (mine.index.equals(gate_b.index) and np.allclose(mine.to_numpy(), gate_b.to_numpy())):
        raise ValueError(f'{profile}: 8월 b 정의 불일치')


def run_test(args):
    """9월 시험 한 번. 설정(창·사건·E0 raw)은 selection.json 그대로, 9월로 바꾸지 않는다."""
    import full_c_data as fd
    out = Path(args.out).resolve()
    selection = json.loads((out/'selection.json').read_text(encoding='utf-8'))
    dev = fd.load_frame(SEPT_FRAME_END)
    copied = copied_set()
    rows, rel_rows, cells = [], [], []
    for profile in mc.PROFILES:
        check_b(profile, dev)
        h = mc.HORIZON[profile]
        oof, _ = load_oof(profile, copied)
        span = WINDOWS[selection[profile]['창']]
        e, hist, pool_origins, audit = pool_for(oof, 9, span)
        dist = rpd.ResidualPathDistribution().fit(e, pool_origins)
        for variant in ['A', 'B']:
            origins, y, b, yhat = sept_ensemble(out, variant, profile, dev)
            starts = origins.to_numpy()[:, None] + np.arange(h)[None, :]*np.timedelta64(15, 'm')
            clean = ~np.asarray(pd.DatetimeIndex(starts.ravel()).normalize().isin(copied)).reshape(len(origins), h).any(axis=1)
            tgt = dict(origins=origins[clean], y=y[clean], yhat=yhat[clean], b=b[clean])
            main = predict_month(dist, hist, tgt, 1)
            aux = predict_month(dist, hist, tgt, NEAR)
            ep = main['event']
            base = dict(구간=profile, 판=variant, 창=selection[profile]['창'], 검증상태=selection[profile]['상태'], 풀달=audit['months'],
                        풀경로=audit['kept'], 채점원점=int(clean.sum()), 채점제외=int((~clean).sum()),
                        초과날=pd.DatetimeIndex(starts[clean][ep]).normalize().nunique())
            q = rpd.quantile_scores(tgt['y'], main['q'])
            rows += method_rows(base, main, aux, dict(coverage90=q['coverage90'], interval90_width_kw=q['interval90_width_kw']))
            for kind, ev, p in [('시점', ep, main['p_point']), ('기간', ep.any(1), main['p_any'])]:
                rel_rows.append(reliability(ev, p).assign(구간=profile, 판=variant, 확률=kind))
            cells.append(mc.long_predictions(tgt['origins'], tgt['y'], tgt['yhat'], tgt['b'], 구간=profile, 판=variant).assign(
                p_point=main['p_point'].ravel(), p_any=np.repeat(main['p_any'], h), p_point_near=aux['p_point'].ravel(),
                p_any_near=np.repeat(aux['p_any'], h), q05=main['q'][:, :, 0].ravel(), q50=main['q'][:, :, 3].ravel(), q95=main['q'][:, :, -1].ravel()))
    pd.DataFrame(rows).to_csv(out/'test_september_scores.csv', index=False, encoding='utf-8-sig')
    pd.concat(rel_rows).to_csv(out/'test_september_reliability.csv', index=False, encoding='utf-8-sig')
    pd.concat(cells).to_csv(out/'test_september_predictions.csv.gz', index=False, compression='gzip')
    print(mc.now(), '9월 시험 완료', flush=True)


def run_records(args):
    """AGENTS 8장: 후보(구간×창)·시험(구간×판)별 기록과 summary.csv."""
    out = Path(args.out).resolve()
    cfg = json.loads((out/'config.json').read_text(encoding='utf-8'))
    sel = json.loads((out/'selection.json').read_text(encoding='utf-8'))
    v = pd.read_csv(out/'validation_scores.csv')
    data_line = f"{mc.SOURCE.relative_to(ROOT)} SHA256 {mc.digest(mc.SOURCE)}; OOF 칸 파일 {json.dumps(cfg['sources'], ensure_ascii=False)}"
    common = {'모델·라이브러리 버전': f"Python {cfg['versions']['python']}, numpy {cfg['versions']['numpy']}, pandas {cfg['versions']['pandas']}, "
                                    f"scikit-learn {cfg['versions'].get('scikit_learn')}; 분포 {rpd.DISTRIBUTION_VERSION}",
              '목표값·단위·예측 구간': '주 사건 y(t+h) > b_origin(시점) / 경로 중 한 번이라도(기간), b = max(170kW, 이번 달 발행 전 최고); 보조 y ≥ 0.9·b. 15분 전력 kW',
              '전처리·특성·하이퍼파라미터·seed': 'E0 raw 균등 질량, 보정·조건 없음, 시뮬레이션 미계산. 복사된 날이 경로에 하나라도 있으면 원점 제외. 난수 없음',
              '실행 명령·환경·코드 버전 또는 코드 식별 정보': f"{cfg['python_executable']}; {cfg['code']} SHA256 {cfg['code_sha256']}",
              '데이터 위치·버전 또는 해시·사용 범위': data_line}
    summary = []
    for (profile, window), part in v[(v.방법 == '잔차 분포') & (v.사건 == '> b')].groupby(['구간', '창']):
        eid = f'{profile}_{"all" if window == "이전 전체" else "last2"}_valid'
        pooled = part[part.채점달 == '합침'].iloc[0]
        hist = v[(v.구간 == profile) & (v.창 == window) & (v.방법 == '과거 빈도') & (v.사건 == '> b') & (v.채점달 == '합침')].score.iloc[0]
        zero = v[(v.구간 == profile) & (v.창 == window) & (v.방법 == '항상 0') & (v.사건 == '> b') & (v.채점달 == '합침')].score.iloc[0]
        const = v[(v.구간 == profile) & (v.창 == window) & (v.방법 == '상수 발생률') & (v.사건 == '> b') & (v.채점달 == '합침')].score.iloc[0]
        chosen = sel[profile]['창'] == window
        mc.write_record(out/'records'/f'{eid}.md', dict(common, **{
            '상태': '완료', '목적과 가설': f'풀 창 "{window}"의 잔차 경로 분포가 과거 빈도보다 나은 초과확률을 주는지(검증 후보)',
            '시작·종료 시각 및 시간대': f"{mc.now()} 기록; Asia/Seoul",
            '학습·검증·테스트 분할과 누수 점검': f'달별 굴리기, 채점 달 {VALID_MONTHS[profile]}. 풀 = 채점 달 이전({window}) OOF 오차, 경로 끝 ≤ 채점 달 시작(코드에서 assert). 9월 미사용',
            '이전 또는 기준 실험 링크와 변경 사항': '기준 = 같은 풀의 과거 실제 경로 빈도, 항상 0. 이전 잔차 실험 35fce8과 달리 m1007 4모델 평균·달별 굴리기 OOF 사용',
            '실제 실행 시간·자원 사용량(측정한 경우)': f"채점 달 합 {part.seconds.sum():.1f}초",
            '평가 지표·구간별 또는 fold별 결과': 'score = 0.5×시점 Brier + 0.5×기간 Brier (낮을수록 좋음). 아래 표 = 채점 달별·합침',
            '기준모델 대비 변화와 비교 가능 여부': f"합침 score {pooled.score:.5f} vs 과거 빈도 {hist:.5f}, 상수 발생률 {const:.5f}(2026-10-08 검증 열람 후 추가 기준), 항상 0 {zero:.5f}. 같은 원점·b·정답",
            '오차가 큰 조건·해석·한계': '초과 날이 달마다 0~2일(4d 5일 이하)로 매우 적음. 6월은 복사된 날로 채점 원점이 적음. 4d 7월 풀은 경로 2개',
            '오류·경고·실패 또는 중단 원인': '없음',
            '예측·모델·표·그래프·로그의 저장 경로': 'validation_scores.csv, validation_predictions.csv.gz, reliability.csv, pool_audit.csv',
            '결론과 다음 실험': ('선정 — ' if chosen else '미선정 — ') + sel[profile]['상태']}),
            part[['채점달', '풀경로', '채점원점', '초과날', 'n_event_origins', 'score', 'Brier_point', 'Brier_period', 'AP_period', 'coverage90']])
        summary.append(dict(experiment_id=eid, 상태='완료', 주요변경=f'풀 창 {window}', 평가세트='5~8월 굴리기(4d 7·8월)', 주지표='score(↓)',
                            score=pooled.score, 과거빈도=hist, 상수발생률=const, 항상0=zero, AP_기간=pooled.AP_period, 기록=f'records/{eid}.md', 선정=chosen))
    test_path = out/'test_september_scores.csv'
    if test_path.exists():
        t = pd.read_csv(test_path)
        for (profile, variant), part in t.groupby(['구간', '판']):
            eid = f'{profile}_sept{variant}_test'
            res = part[(part.방법 == '잔차 분포') & (part.사건 == '> b')].iloc[0]
            hist = part[(part.방법 == '과거 빈도') & (part.사건 == '> b')].score.iloc[0]
            zero = part[(part.방법 == '항상 0') & (part.사건 == '> b')].score.iloc[0]
            const = part[(part.방법 == '상수 발생률') & (part.사건 == '> b')].score.iloc[0]
            model_line = ('1~8월 재학습 4모델(7월 선정 F·설정, 부스팅 나무 수 = 8/18~8/31 확인 구간)' if variant == 'A'
                          else '1~7월 모델(8월 검증에 쓴 모델) 재사용, 8월 재예측이 저장값과 일치함을 확인')
            mc.write_record(out/'records'/f'{eid}.md', dict(common, **{
                '상태': '완료', '목적과 가설': f'9월 시험 {variant}판: {model_line}',
                '시작·종료 시각 및 시간대': f"{mc.now()} 기록; Asia/Seoul",
                '학습·검증·테스트 분할과 누수 점검': f"풀 = {res.풀달}월 OOF({res.창}), 경로 끝 ≤ 9/1. 9월 원점 9/1~(정답 끝 ≤ 9/15). 설정은 검증 selection.json 그대로, 9월로 바꾸지 않음",
                '이전 또는 기준 실험 링크와 변경 사항': f"검증 후보 records/{profile}_{'all' if res.창 == '이전 전체' else 'last2'}_valid.md",
                '실제 실행 시간·자원 사용량(측정한 경우)': '미측정(채점 수 초)',
                '평가 지표·구간별 또는 fold별 결과': 'score = 0.5×시점 Brier + 0.5×기간 Brier. 아래 표',
                '기준모델 대비 변화와 비교 가능 여부': f"score {res.score:.5f} vs 과거 빈도 {hist:.5f}, 상수 발생률 {const:.5f}, 항상 0 {zero:.5f}",
                '오차가 큰 조건·해석·한계': ('검증 상태: ' + res.검증상태 + '. 9월은 이전 잔차 실험의 9월 결과를 본 뒤의 설계라 준독립. 초과 날이 적음'
                                       + ('. B판은 2달 앞 예측인데 풀은 1달 앞 오차라 확률 과소 가능' if variant == 'B' else '')),
                '오류·경고·실패 또는 중단 원인': '없음',
                '예측·모델·표·그래프·로그의 저장 경로': f'test_september_scores.csv, test_september_predictions.csv.gz, sept_members_*_{variant}.csv',
                '결론과 다음 실험': '보고만 함. 9월 결과로 창·판·설정을 고르지 않음'}),
                part[['방법', '사건', '채점원점', '초과날', 'n_event_origins', 'score', 'Brier_point', 'Brier_period', 'AP_period']])
            summary.append(dict(experiment_id=eid, 상태='완료', 주요변경=f'9월 {variant}판', 평가세트='9월 1회', 주지표='score(↓)',
                                score=res.score, 과거빈도=hist, 상수발생률=const, 항상0=zero, AP_기간=res.AP_period, 기록=f'records/{eid}.md', 선정=None))
    pd.DataFrame(summary).to_csv(out/'summary.csv', index=False, encoding='utf-8-sig')
    print('기록', len(summary))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('validate')
    v.add_argument('--out', required=True)
    v.add_argument('--profiles', nargs='+', default=mc.PROFILES, choices=mc.PROFILES)
    r = sub.add_parser('sept-rf')
    r.add_argument('--out', required=True)
    r.add_argument('--variant', required=True, choices=['A', 'B'])
    r.add_argument('--profiles', nargs='+', default=mc.PROFILES, choices=mc.PROFILES)
    b = sub.add_parser('sept-boost')
    b.add_argument('--out', required=True)
    b.add_argument('--variant', required=True, choices=['A', 'B'])
    b.add_argument('--library', required=True, choices=list(NAMES))
    b.add_argument('--profiles', nargs='+', default=mc.PROFILES, choices=mc.PROFILES)
    t = sub.add_parser('test')
    t.add_argument('--out', required=True)
    c = sub.add_parser('records')
    c.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.cmd == 'records':
        run_records(args)
    elif args.cmd == 'test':
        run_test(args)
    elif args.cmd == 'validate':
        run_validate(args)
    elif args.cmd == 'sept-rf':
        sept_rf(args)
    elif args.cmd == 'sept-boost':
        sept_boost(args)


if __name__ == '__main__':
    main()
