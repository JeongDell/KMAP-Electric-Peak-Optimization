"""대표 선정, 월별 성능 집계, 안전마진 입력 및 제출 CSV 저장."""
import hashlib
import warnings

import numpy as np
import pandas as pd
import torch

import config as C
from core import data as D, peak as PK, pipeline as P, train as T

RATE_START_KW = 180.0
RATE_HOURS = (9, 23)
RATE_MONTHS = (12, 1, 2, 7, 8, 9)
SIMILAR_DAYS = 4
BASELINES = {             # 학습 없는 단순 예측 — 발행 시각까지 알려진 값만 쓴다
    "기준선 직전값": "발행 직전 15분 값을 모든 칸에 그대로",
    "기준선 어제": "어제 같은 시각 (24시간보다 먼 칸은 발행 전에 끝난 가장 최근 날의 같은 시각)",
    "기준선 지난주": "지난주(7일 전) 같은 시각",
    "기준선 비슷한날": f"같은 날 종류(평일·토·일·공휴일)의 최근 {SIMILAR_DAYS}일 같은 시각 값의 중앙값 (발행한 날 이전 날만)",
}


def rate_before(df):
    """칸마다 '그 칸 직전까지' 알려진 요금 기준 B (kW). 발행 시각 t0의 B = rate[t0], 대상 칸 T의 B = rate[T]."""
    hour = df["ts"].dt.hour
    p = df["power"].where((hour >= RATE_HOURS[0]) & (hour < RATE_HOURS[1]), -np.inf)
    mid = pd.Series(PK.month_id(df), index=df.index)
    now = p.groupby(mid).cummax().groupby(mid).shift(1).fillna(-np.inf)   # 이번 달, 그 칸 앞까지
    top = p.groupby(mid).max()
    past = {m: max([top[k] for k in top.index if m - 11 <= k < m and k % 12 + 1 in RATE_MONTHS], default=-np.inf)
            for m in top.index}
    out = np.maximum(np.maximum(now.to_numpy(), mid.map(past).to_numpy()), RATE_START_KW)
    return out.astype(np.float32)


def is_tou(df):
    hour = df["ts"].dt.hour.to_numpy()
    return (hour >= RATE_HOURS[0]) & (hour < RATE_HOURS[1])


def day_kind(df):
    """칸별 날 종류 번호 (core/data.py의 DAY_TYPES: 0 평일, 1 토요일, 2 일요일, 3 공휴일)."""
    return df[[f"daytype_{k}" for k in D.DAY_TYPES]].to_numpy().argmax(1)


def baselines(df, t0, H):
    """기준선 4개 → {이름: (발행 수, H) kW}. 0번 행 = 1/1 00:00이라 칸 번호 // 96 = 날 번호."""
    y = df["power"].to_numpy(np.float32)
    k = np.arange(H)
    tgt = t0[:, None] + k                                   # 대상 칸 번호
    week = y[tgt - 672]
    out = {"기준선 직전값": np.repeat(y[t0 - 1][:, None], H, axis=1),
           "기준선 어제": y[tgt - 96 * (k // 96 + 1)],       # 발행 시각 앞에서 가장 가까운 '같은 시각'
           "기준선 지난주": week}
    kind = day_kind(df)[::96]                               # 날별 종류
    origin_day = np.broadcast_to((t0 // 96)[:, None], tgt.shape)
    tgt_kind, slot = kind[tgt // 96], tgt % 96
    vals = np.full(tgt.shape + (SIMILAR_DAYS,), np.nan, np.float32)
    for kd in np.unique(kind):
        days = np.flatnonzero(kind == kd)
        sel = tgt_kind == kd
        pos = np.searchsorted(days, origin_day[sel])        # 발행한 날보다 앞선 같은 종류 날의 수
        for j in range(SIMILAR_DAYS):
            d = days[np.maximum(pos - 1 - j, 0)]
            vals[sel, j] = np.where(pos - 1 - j >= 0, y[d * 96 + slot[sel]], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # 같은 종류 날이 아직 없으면 NaN → 지난주 값으로
        med = np.nanmedian(vals, axis=2)
    out["기준선 비슷한날"] = np.where(np.isnan(med), week, med)
    return {n: v.astype(np.float32) for n, v in out.items()}


def model_candidates(best, g, month, n):
    """(모델, 손실) 최고 조합의 시드 평균 예측 + 손실별 세 모델 평균 → {이름: 정보}. 이름 = '모델 손실' (significance와 같음)."""
    out, by_loss = {}, {}
    for _, b in best[best["group"] == g].iterrows():
        paths = {s: P.pred_path(g, b["model"], b["loss"], b["params"], month, s) for s in C.SEEDS}
        seeds = list(C.SEEDS)
        if not all(p.exists() for p in paths.values()):
            raise FileNotFoundError(f"{g} {month} {b['model']} {b['loss']}: 시드 예측이 누락되었습니다.")
        if not seeds:
            continue
        pred = np.mean([np.load(paths[s]) for s in seeds], axis=0).astype(np.float32)
        if pred.shape != (n, C.GROUPS[g]["horizon"]) or not np.isfinite(pred).all():
            raise SystemExit(f"{g} {month} {b['model']} {b['loss']}: 예측 {pred.shape[0]}줄 ≠ 발행 시각 {n}개 "
                             "— 설정을 바꾸기 전의 옛 예측 파일인지 확인 (results/를 지우고 다시 돌려야 함)")
        info = {"pred": pred, "model": b["model"], "loss": b["loss"], "params": b["params"], "seeds": seeds}
        out[f"{b['model']} {b['loss']}"] = info
        by_loss.setdefault(b["loss"], []).append(info)
    for loss, items in by_loss.items():
        if len(items) == len(C.MODELS_TO_RUN) > 1:
            out[f"ENSEMBLE {loss}"] = {
                "pred": np.mean([x["pred"] for x in items], axis=0).astype(np.float32), "model": "ENSEMBLE", "loss": loss,
                "params": " + ".join(f"{x['model']} {x['params']}" for x in items),
                "seeds": sorted(set.intersection(*[set(x["seeds"]) for x in items]))}
    return out


def month_block(df, best, g, month, role):
    """그룹·달 하나: 후보 예측, 채점 칸, 피크 기준 b, 지표 행. 예측 파일이 하나도 없으면 None."""
    G = C.GROUPS[g]
    H = G["horizon"]
    val_start, val_end, _ = P.split(df, month)
    fold = D.Fold(df, val_start, P.DEVICE)
    t0 = D.origins(df, val_start, val_end, H, G["eval_stride"])
    fi, thr = P.targets(fold, t0, H)
    cands = model_candidates(best, g, month, len(t0))
    if not cands:
        return None
    preds = {name: c["pred"] for name, c in cands.items()}
    preds.update(baselines(df, t0, H))
    y, m = fold.y_kw[fi], fold.score[fi]
    rows = []
    for name, p in preds.items():
        c = cands.get(name, {})
        kind = "기준선" if name in BASELINES else "앙상블" if c.get("model") == "ENSEMBLE" else "모델"
        for r in T.report(torch.as_tensor(p, device=fi.device), y, m, thr, G["report"], G["block"]):
            h = G["report"][r["horizon"]]
            rows.append({"구간": r["horizon"], "그룹": g, "칸수": h, "달": month, "역할": role, "후보": name, "종류": kind,
                         "손실": c.get("loss", "-"), "시드수": len(c["seeds"]) if c else 0,
                         "RMSE(kW)": r["RMSE"], "MAE(kW)": r["MAE"], "공통C(kW)": r["LossC"],
                         "최악20%구간RMSE(kW)": r["WinRMSE_worst20"], "피크칸수(실제≥b)": r["Peak_n"],
                         "피크칸_과소예측평균(kW)": r["Peak_under"], "피크칸_재현율": r["Peak_recall"],
                         "채점칸수": int(m[:, :h].sum()), "발행수": len(t0)})
    return {"t0": t0, "fi": fi.cpu().numpy(), "b": thr[:, 0].cpu().numpy(), "score": m.cpu().numpy(),
            "preds": preds, "cands": cands, "rows": rows}


def compare_to_baseline(perf):
    """(구간, 달)마다 가장 좋은 기준선(공통 C 기준)과 개선율 열을 붙인다."""
    key = ["구간", "달"]
    base = perf[perf["종류"] == "기준선"]
    top = base.loc[base.groupby(key)["공통C(kW)"].idxmin(), key + ["후보", "공통C(kW)", "RMSE(kW)"]]
    top.columns = key + ["기준선최고", "기준선최고_공통C(kW)", "기준선최고_RMSE(kW)"]
    perf = perf.merge(top, on=key, how="left")
    perf["공통C_개선율(%)"] = 100 * (1 - perf["공통C(kW)"] / perf["기준선최고_공통C(kW)"])
    perf["RMSE_개선율(%)"] = 100 * (1 - perf["RMSE(kW)"] / perf["기준선최고_RMSE(kW)"])
    return perf


def ranking(perf, g, select, loss):
    """그룹 전체 구간에서 그 손실로 학습한 후보의 고르기 달 공통 C (낮은 순). 고르기 달 모두에 예측이 있는 모델·앙상블만."""
    full = list(C.GROUPS[g]["report"])[-1]
    part = perf[(perf["그룹"] == g) & (perf["구간"] == full) & (perf["종류"] != "기준선") & perf["달"].isin(select)
                & (perf["손실"] == loss)]
    n = part.groupby("후보")["달"].nunique()
    part = part[part["후보"].isin(n[n == len(select)].index)]
    return part.groupby("후보")["공통C(kW)"].mean().sort_values()


def run_id(g, info):
    if info["model"] == "ENSEMBLE":
        return f"{g}_ENSEMBLE_loss{info['loss']}"
    return f"{g}_{info['model']}_loss{info['loss']}_{hashlib.md5(info['params'].encode()).hexdigest()[:8]}"


def path_frame(df, blk, rate):
    """한 그룹·달의 (발행 시각 × 몇 칸 앞) 공통 열."""
    t0, fi = blk["t0"], blk["fi"]
    n, H = fi.shape
    ts, flat = df["ts"].to_numpy(), fi.ravel()
    return pd.DataFrame({
        "origin": np.repeat(ts[t0], H), "lead": np.tile(np.arange(1, H + 1), n), "target_time": ts[flat],
        "y": df["power"].to_numpy()[flat], "B_origin": np.repeat(rate[t0], H), "B_target": rate[flat],
        "b_loss": np.repeat(blk["b"], H), "scored": blk["score"].ravel().astype(int),
        "tou": is_tou(df)[flat].astype(int), "copied_day": df["copied"].to_numpy()[flat].astype(int),
        "outage": df["outage"].to_numpy()[flat].astype(int), "prod_fix": df["prod_fix"].to_numpy()[flat].astype(int)})


def periods_table(df, roles):
    day = pd.Timedelta(days=1)
    data_end = df["ts"].iloc[-1].normalize() + day
    rows = []
    for month, role in roles.items():
        val_start, val_end, es = P.split(df, month)
        fit_end = val_start                              # 학습 끝 (그 전날까지)
        if month == C.TEST_MONTH and C.CONFIRM_MONTHS:   # 9월 = 확인 월과 같은 최종 모델 (pipeline.run_final)
            fit_end, _, es = P.split(df, C.CONFIRM_MONTHS[0])
        f = lambda t: t.strftime("%Y-%m-%d")
        rows.append({"달": month, "역할": role,
                     "①최고에폭_찾기_학습": f"{C.TRAIN_FIRST_ORIGIN} ~ {f(es - day)}",
                     "①조기종료_확인": f"{f(es)} ~ {f(fit_end - day)} (복사 안 된 날 {C.ES_DAYS}일이 드는 구간, "
                                      f"{C.CHECK_EVAL} 칸으로 채점)",
                     "②다시학습(채점할 모델)": f"{C.TRAIN_FIRST_ORIGIN} ~ {f(fit_end - day)}",
                     "평가": f"{f(val_start)} ~ {f(min(val_end, data_end) - day)} (예측 경로 전체가 이 안에 드는 발행만)",
                     "정규화_기준": f"① {f(es)} 이전 · ② {f(fit_end)} 이전 값으로만"})
    return pd.DataFrame(rows)



def collect(df, best, months, picks=None):
    """월별 결과를 모은다. 대표는 선정 월로만 고르고 테스트에서는 이미 확정한 대표를 사용한다."""
    blocks = {}
    for group in C.GROUPS_TO_RUN:
        for month in months:
            role = '고르기' if month in C.FOLDS else '확인' if month in C.CONFIRM_MONTHS else '시험'
            block = month_block(df, best, group, month, role)
            if block is None:
                raise ValueError(f'{group} {month}: 완료된 모델 예측이 없습니다.')
            blocks[group, month] = block
    performance = compare_to_baseline(pd.DataFrame([row for block in blocks.values() for row in block['rows']]))
    if picks is None:
        picks = {}
        for group in C.GROUPS_TO_RUN:
            for loss in C.LOSSES_TO_RUN:
                rank = ranking(performance, group, C.FOLDS, loss)
                if rank.empty:
                    raise ValueError(f'{group} {loss}: 대표 선정 후보가 없습니다.')
                picks[group, loss] = rank.index[0]
    else:
        picks = dict(picks)
    predictions = []
    rate = rate_before(df)
    for (group, month), block in blocks.items():
        frame = path_frame(df, block, rate)
        for loss in C.LOSSES_TO_RUN:
            name = picks[group, loss]
            if name not in block['cands']:
                raise ValueError(f'{group} {month}: 확정된 대표 {name}의 예측이 없습니다.')
            candidate = block['cands'][name]
            part = frame[['origin', 'lead', 'target_time', 'y', 'b_loss']].copy()
            part['yhat'] = block['preds'][name].ravel()
            part['run_id'] = run_id(group, candidate)
            part['model'] = candidate['model']
            part['loss'] = loss
            part['horizon'] = group
            part['seed'] = 'mean(세 모델)' if candidate['model'] == 'ENSEMBLE' else 'mean(' + ','.join(map(str, C.SEEDS)) + ')'
            part['period'] = str(int(month[-2:])) + '월'
            part['대표'] = loss + ' 최고'
            # Preserve the precision used by the original CSV-based margin calculation.
            for column in ['y', 'yhat', 'b_loss']:
                part[column] = part[column].map(lambda value: float(f'{value:.3f}'))
            predictions.append(part)
    performance['최고'] = [loss + ' 최고' if picks.get((group, loss)) == name else ''
                          for group, loss, name in zip(performance['그룹'], performance['손실'], performance['후보'])]
    return performance, pd.concat(predictions, ignore_index=True), picks


def export(df, best, performance, predictions, picks):
    """제출 CSV와 성능 요약을 한 번 저장한다. 내부 전달용 CSV나 압축 파일은 만들지 않는다."""
    out = C.RESULT_DIR
    selection = out / 'selection'
    selection.mkdir(parents=True, exist_ok=True)
    best.to_csv(selection / 'best_params.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{'group': group, 'loss': loss, 'representative': name, 'selection_month': ','.join(C.FOLDS)}
                  for (group, loss), name in picks.items()]).to_csv(selection / 'representatives.csv', index=False, encoding='utf-8-sig')
    periods_table(df, {month: '고르기' if month in C.FOLDS else '확인' if month in C.CONFIRM_MONTHS else '시험'
                       for month in C.FOLDS + C.CONFIRM_MONTHS + [C.TEST_MONTH]}).to_csv(out / 'periods.csv', index=False, encoding='utf-8-sig')
    performance.to_csv(out / 'performance.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    staged = out / 'test_predictions'
    staged.mkdir(parents=True, exist_ok=True)
    test_period = str(int(C.TEST_MONTH[-2:])) + '월'
    columns = ['model', 'loss', 'horizon', 'seed', 'period', 'origin', 'lead', 'target_time', 'y', 'yhat', 'b_origin', '대표']
    for group in C.GROUPS_TO_RUN:
        part = predictions[predictions['horizon'].eq(group) & predictions['period'].eq(test_period)]
        part = part.rename(columns={'b_loss': 'b_origin'})[columns]
        if part.empty:
            raise ValueError(f'{group}: 저장할 테스트 예측이 없습니다.')
        part.to_csv(staged / (group + '.csv'), index=False, encoding='utf-8-sig', float_format='%.3f', date_format='%Y-%m-%d %H:%M:%S')
    summary = '\n'.join([
        '# 실행 결과', '',
        f'- 선정 검증: {", ".join(C.FOLDS)}',
        f'- 최종 모델 확인 검증: {", ".join(C.CONFIRM_MONTHS)}',
        f'- 테스트: {C.TEST_MONTH}',
        '- 미래 입력: 실제 생산량·날씨를 아는 다앎 조건',
        '- 테스트에서 모델·대표·마진을 다시 선정하지 않음',
        '- test_predictions/: 대표별 테스트 예측',
        '- performance.csv: 모든 후보·대표·기준선의 월별 성능',
        '- selection/: 확정된 모델 설정과 대표',
        '- margin/: 7월 마진 산출, 8월 검증, 9월 시험',
        '- models/, preds/: 최종 가중치 및 시드별 예측',
    ]) + '\n'
    (out / 'README.md').write_text(summary, encoding='utf-8')
    return staged


def verify(df, best, picks, directory):
    """구간별 제출 결과를 원자료 및 시드 평균 배열과 대조한다. 모델 종류를 고정하지 않는다."""
    expected_groups = set(C.GROUPS_TO_RUN)
    if {p.stem for p in directory.glob('*.csv')} != expected_groups:
        raise ValueError('제출 테스트 예측의 구간 목록이 현재 설정과 다릅니다.')
    actual = pd.Series(df['power'].to_numpy(), index=df['ts'])
    monthly = df['power'].groupby(df['ts'].dt.to_period('M')).transform(lambda values: values.cummax().shift(1))
    peaks = pd.Series(monthly.fillna(C.PEAK_FLOOR).clip(lower=C.PEAK_FLOOR).to_numpy(), index=df['ts'])
    columns = {'model', 'loss', 'horizon', 'seed', 'period', 'origin', 'lead', 'target_time', 'y', 'yhat', 'b_origin', '대표'}
    for group in C.GROUPS_TO_RUN:
        G = C.GROUPS[group]
        frame = pd.read_csv(directory / (group + '.csv'), encoding='utf-8-sig')
        start = pd.Timestamp(C.TEST_MONTH + '-01')
        t0 = D.origins(df, start, start + pd.offsets.MonthBegin(1), G['horizon'], G['eval_stride'])
        candidates = model_candidates(best, group, C.TEST_MONTH, len(t0))
        origins = np.repeat(df['ts'].iloc[t0].to_numpy(), G['horizon'])
        leads = np.tile(np.arange(1, G['horizon'] + 1), len(t0))
        if set(frame.columns) != columns or set(frame['loss']) != set(C.LOSSES_TO_RUN):
            raise ValueError(f'{group}: 제출 열 또는 손실 종류가 다릅니다.')
        if not np.isfinite(frame[['y', 'yhat', 'b_origin']].to_numpy()).all():
            raise ValueError(f'{group}: 예측 결과에 비정상 값이 있습니다.')
        for loss in C.LOSSES_TO_RUN:
            part = frame[frame['loss'].eq(loss)].sort_values(['origin', 'lead'])
            name = picks[group, loss]
            candidate = candidates[name]
            origin, target = pd.to_datetime(part['origin']), pd.to_datetime(part['target_time'])
            checks = [len(part) == len(origins),
                      not part.duplicated(['origin', 'lead']).any(),
                      np.array_equal(origin.to_numpy(), origins), np.array_equal(part['lead'].to_numpy(), leads),
                      target.eq(origin + pd.to_timedelta((part['lead'] - 1) * 15, unit='min')).all(),
                      part['model'].eq(candidate['model']).all(), part['대표'].eq(loss + ' 최고').all(),
                      part['horizon'].eq(group).all(), part['period'].eq(str(int(C.TEST_MONTH[-2:])) + '월').all()]
            if not all(checks):
                raise ValueError(f'{group}/{loss}: 예측 칸 누락·중복 또는 메타데이터가 맞지 않습니다.')
            for label, observed, expected, tolerance in [
                    ('실제값', part['y'], actual.reindex(target).to_numpy(), 0.00051),
                    ('피크 기준', part['b_origin'], peaks.reindex(origin).to_numpy(), 0.00051),
                    ('예측값', part['yhat'], candidate['pred'].ravel(), 0.0006)]:
                if not np.allclose(observed, expected, atol=tolerance, rtol=0):
                    raise ValueError(f'{group}/{loss}: {label}가 저장된 원자료·배열과 다릅니다.')
        print(f'PASS {group}: {len(frame):,}행, 발행 시각·실제값·피크 기준·예측값 일치', flush=True)


def publish(staged):
    import shutil
    target = C.BASE_DIR / 'test_predictions'
    target.mkdir(parents=True, exist_ok=True)
    for source in staged.glob('*.csv'):
        temporary = target / (source.name + '.tmp')
        shutil.copyfile(source, temporary)
        temporary.replace(target / source.name)
    temporary = C.BASE_DIR / 'results_summary.csv.tmp'
    shutil.copyfile(C.RESULT_DIR / 'performance.csv', temporary)
    temporary.replace(C.BASE_DIR / 'results_summary.csv')
