"""7월 예측으로 안전마진을 만들고 8월 검증·9월 시험을 수행한다. main.py에서 자동 실행."""
import math
from datetime import datetime

import numpy as np
import pandas as pd

import config as C   # 설정 파일만 읽는다 (학습 코드·torch는 불러오지 않음)

LOSSES = ["A", "C"]          # 구간마다 A 최고 · C 최고 둘 다
MAKE, CHECK, TEST = "2021-07", "2021-08", "2021-09"   # 마진을 만드는 달 → 확인하는 달 → 시험
NEAR = 0.9                   # 피크 근처 = 실제 ≥ 0.9b, 경보 선 = (0.9 − 마진) × b
BASE_Q = 0.95                # 기본 마진 = r의 95% 지점
MIN_CELLS, MIN_DAYS = 20, 3  # 결정 1: 조건 값 하나에 이만큼 없으면 합침
ALPHA = 0.05                 # 결정 3: 뚜렷함 기준
MAX_CONDS = 3                # 메모: 조건은 2~3개까지
PASS_DAY_RECALL = 0.8        # 결정 4 (팀 확정): 피크 근처 날의 80% 이상 미리 경보
MAX_FA = 1.0                 # 하루 헛경보 1번 이하
SUPPRESS_MIN = 60            # 같은 경보로 묶는 시간 (분)
# 주 결과 (⑥ 8월 세부 조정, 10/07 1차 실행 뒤 — 조정기록.csv에 남음). 다른 쪽 결과도 csv에 그대로 나온다
MAIN_METHOD = "기본 마진만"  # "기본 마진만" 또는 "조건별 마진". 1차에서 조건별 마진이 4h~24h 피크를 놓쳐 메모 4-4대로 단순하게
MAIN_COUNT = "대상"          # 헛경보 세는 법 "대상" 또는 "발행" — 매시 발행하는 4h~24h가 같은 피크 경보를 매시간 새로 세지 않도록.
                             # 대상 기준은 경보 선이 낮으면 긴 시간대가 경보 하나가 되므로 경보적중률·하루경보시간을 같이 본다
BREAK = ("2021-08-02", "2021-08-13")    # 8월 휴무 주 + 재가동 주 (메모 4-4: 따로 보여 줌)
LEAD_BINS = [0, 1, 4, 12, 24, 96]       # 몇 시간 앞 묶음 경계 (시간)
LEAD_NAMES = ["~1h", "1~4h", "4~12h", "12~24h", "1~4일"]
CONDS = ["요일", "시간대", "앞", "강수", "생산량"]


def to_time(s):
    """csv 시각 열 → 시각. 자정은 '2021-07-05'처럼 날짜만 저장되기도 해서 맞춰 읽는다 (pandas 1.x · 2.x 공통)."""
    s = s.astype(str)
    return pd.to_datetime(s.where(s.str.len() > 10, s + " 00:00:00"), format="%Y-%m-%d %H:%M:%S")


def raw_table():
    """원자료 → 칸별 조건 재료 (시각 index): 요일·시간대·강수·생산량, 생산량 경계."""
    raw = pd.read_csv(C.DATA_PATH, encoding="utf-8-sig")
    raw.index = pd.to_datetime(raw["timestamp"])   # core/data.py와 같은 방식 (원자료는 '2021-01-01 0:00' 형식)
    prod = raw["생산량"].astype(float)
    cut = float(prod[(raw.index < pd.Timestamp(MAKE + "-01")) & (prod > 0)].median())   # 1~6월, 0 제외
    day = raw["day_type"].map({0: "평일", 1: "토요일", 2: "일·공휴일", 3: "일·공휴일"})
    cond = pd.DataFrame({"요일": day, "시간대": np.where(raw["is_night"] == 1, "야간", "주간"),
                         "강수": np.where(raw["강수량"].astype(float) > 0, "비", "안 옴"),
                         "생산량": np.where(prod > cut, "많음", "적음")}, index=raw.index)
    return cond, cut


def prepare(predictions, cond):
    """대표 예측 DataFrame → 칸(발행 × 몇 칸 앞)마다 예측·실제·b·r·피크 근처·조건."""
    cols = ["run_id", "loss", "horizon", "origin", "lead", "target_time", "y", "yhat", "b_loss"]
    d = predictions[cols].copy()
    d["origin"], d["target_time"] = to_time(d["origin"]), to_time(d["target_time"])
    d["달"] = d["origin"].dt.strftime("%Y-%m")
    d["후보"] = d["run_id"].str.split("_").str[1] + " " + d["loss"]
    d["r"] = (d["y"] - d["yhat"]).clip(lower=0) / d["b_loss"]
    d["근처"] = d["y"] >= NEAR * d["b_loss"]
    d["앞"] = pd.cut(d["lead"] / 4, LEAD_BINS, labels=LEAD_NAMES).astype(str)   # 몇 시간 앞 (lead 1 = 0.25시간)
    for c in ["요일", "시간대", "강수", "생산량"]:
        d[c] = d["target_time"].map(cond[c])
    return d


def welch_p(a, b):
    """두 표본 평균 차이 검정(Welch) 양측 p값. scipy가 없으면 정규 근사."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    if va + vb == 0:
        return np.nan
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), dof))
    except ImportError:
        return float(math.erfc(abs(t) / math.sqrt(2)))


def merged(near, c):
    """조건 c의 값마다 칸·날 수 → 작은 값은 '합침'으로. 반환: 칸마다 쓸 값 (합쳐도 작으면 '기본')."""
    unit = near.drop_duplicates(["target_time", c])
    size = unit.groupby(c).agg(칸=("target_time", "size"), 날=("target_time", lambda t: t.dt.normalize().nunique()))
    small = size[(size["칸"] < MIN_CELLS) | (size["날"] < MIN_DAYS)].index
    v = near[c].where(~near[c].isin(small), "합침")
    if v.eq("합침").any():
        m = unit[unit[c].isin(small)]
        if len(m) < MIN_CELLS or m["target_time"].dt.normalize().nunique() < MIN_DAYS:
            v = v.replace("합침", "기본")
    return v


def analyze(near, base):
    """③ 요인 분석 (7월 피크 근처 칸) → (조건·값별 표, 고른 조건 목록). 검정 표본 = 칸마다 r 평균
    (같은 칸을 여러 발행이 예측하므로. 조건이 '앞'이면 칸 × 앞 묶음마다)."""
    rows, min_p = [], {}
    for c in CONDS:
        v = merged(near, c)
        unit = near.assign(값=v).groupby(["target_time", "값"])["r"].mean().reset_index()
        values = [x for x in unit["값"].unique() if x != "기본"]
        for x in values if len(values) >= 2 else []:   # 값이 하나뿐이면 비교할 나머지가 없음
            p = welch_p(unit.loc[unit["값"] == x, "r"].to_numpy(), unit.loc[unit["값"] != x, "r"].to_numpy())
            cells = near[v == x]
            rows.append({"조건": c, "값": x, "칸수": cells["target_time"].nunique(),
                         "날수": cells["target_time"].dt.normalize().nunique(), "예측수": len(cells),
                         "r평균": cells["r"].mean(), "나머지_r평균": near.loc[v != x, "r"].mean(),
                         "r_분위수": cells["r"].quantile(BASE_Q), "기본마진": base, "p값": p, "뚜렷함": bool(p < ALPHA)})
            if p < ALPHA:
                min_p[c] = min(min_p.get(c, 1.0), p)
    chosen = sorted(min_p, key=min_p.get)[:MAX_CONDS]   # 뚜렷한 조건 중 p값이 작은 순
    table = pd.DataFrame(rows, columns=["조건", "값", "칸수", "날수", "예측수", "r평균", "나머지_r평균", "r_분위수",
                                        "기본마진", "p값", "뚜렷함"])
    table["고른조건"] = table["조건"].isin(chosen)
    return table, chosen


def combo_keys(df, near_ref, chosen):
    """칸마다 고른 조건 값의 조합 이름 (예: '주간 · 많음'). 값은 7월의 합치기를 그대로 따른다 (7월에 없던 값 = '합침')."""
    parts = []
    for c in chosen:
        kept = set(merged(near_ref, c).unique()) - {"합침", "기본"}
        parts.append(df[c].where(df[c].isin(kept), "합침").astype(str))
    return pd.concat(parts, axis=1).agg(" · ".join, axis=1)


def margin_table(near, chosen, base):
    """④ 마진 표: 고른 조건들의 조합마다 그 칸들의 r 분위수 (조건끼리 겹치는 효과를 두 번 세지 않도록 더하지 않고 조합으로 잰다).
    칸이 MIN_CELLS칸 · MIN_DAYS일보다 모자란 조합은 기본 마진."""
    rows = [{"조합": "기본 (조건 없음 · 모자란 조합)", "칸수": near["target_time"].nunique(),
             "날수": near["target_time"].dt.normalize().nunique(), "마진": base, "가감": 0.0}]
    if chosen:
        keys = combo_keys(near, near, chosen)
        for k, idx in keys.groupby(keys).groups.items():
            cells = near.loc[idx]
            n, days = cells["target_time"].nunique(), cells["target_time"].dt.normalize().nunique()
            ok = n >= MIN_CELLS and days >= MIN_DAYS
            q = cells["r"].quantile(BASE_Q) if ok else base
            rows.append({"조합": k, "칸수": n, "날수": days, "마진": q, "가감": q - base, "비고": "" if ok else "칸 부족 → 기본"})
    return pd.DataFrame(rows)


def apply_margin(d, table, base, near_ref, chosen):
    """칸마다 마진 = 그 칸 조건 조합의 마진 (표에 없는 조합은 기본), 0 ~ NEAR − 0.01."""
    if not chosen:
        return pd.Series(base, index=d.index)
    m = combo_keys(d, near_ref, chosen).map(table.set_index("조합")["마진"]).fillna(base)
    return m.clip(0, NEAR - 0.01)


def verdict(n_ev, day_rate, fa):
    if not n_ev:
        return "사례 없음"
    if day_rate >= PASS_DAY_RECALL:
        return "통함" if fa <= MAX_FA else "헛경보 많음(마진 넓음)"
    return "놓침 많음(마진 좁음)" if fa <= MAX_FA else "둘 다 못 미침"


def check(a, m_cell):
    """⑤ 한 달(또는 기간)의 칸들 + 칸별 마진 → 재현율·헛경보. m_cell = 0이면 '마진 없이'.
    헛경보는 두 가지로 센다 (어느 쪽으로 판정할지는 팀이 정함):
      발행 기준 = 경보가 울린 발행을 시각순으로 SUPPRESS_MIN분 묶음 → 묶음의 경로에 피크 근처 칸이 없으면 헛경보.
                  매시 발행하는 4h~24h는 같은 피크를 가리키는 경보도 매시간 새로 센다
      대상 기준 = 경보가 가리킨 칸(예측 ≥ 경보 선)을 시각순으로 이어 SUPPRESS_MIN분 넘게 끊기면 다른 경보 →
                  그 시간대에 실제 피크 근처 칸이 없으면 헛경보. 같은 시간대를 여러 발행이 가리켜도 한 번. 경보 선이 낮으면
                  근무 시간 전체가 경보 하나로 이어진다"""
    a = a.assign(경보칸=a["yhat"] >= (NEAR - m_cell) * a["b_loss"])
    per = a.groupby("origin").agg(경보=("경보칸", "any"), 근처=("근처", "any")).sort_index()
    alarms = per[per["경보"]]
    start, groups = None, []
    for t in alarms.index:
        if start is None or t - start >= pd.Timedelta(minutes=SUPPRESS_MIN):
            start = t
        groups.append(start)
    grouped = alarms.assign(묶음=groups).groupby("묶음")["근처"].any()
    ev = a.loc[a["근처"], "target_time"].drop_duplicates()
    caught = set(a.loc[a["근처"] & a["경보칸"], "target_time"])      # 그 칸을 어떤 발행이든 예측 ≥ 경보 선으로 짚음
    hit = ev.isin(caught)
    ev_days = ev.dt.normalize()
    days = a["origin"].dt.normalize().nunique()
    day_rate = ev_days[hit].nunique() / ev_days.nunique() if len(ev) else np.nan
    fa = (~grouped).sum() / days if days else np.nan
    cells = a.loc[a["경보칸"], "target_time"].drop_duplicates().sort_values()   # 대상 기준
    gid = (cells.diff() > pd.Timedelta(minutes=SUPPRESS_MIN)).cumsum().to_numpy()
    t_hit = cells.isin(set(ev)).groupby(gid).any() if len(cells) else pd.Series(dtype=bool)
    fa_t = (~t_hit).sum() / days if days else np.nan
    span = cells.groupby(gid).agg(["min", "max"]) if len(cells) else pd.DataFrame(columns=["min", "max"])
    hours = ((span["max"] - span["min"]).dt.total_seconds().sum() / 3600 + len(span) * 0.25) if len(span) else 0.0
    return {"피크근처_칸": len(ev), "칸재현율": hit.mean() if len(ev) else np.nan,
            "피크근처_날": ev_days.nunique(), "날재현율": day_rate, "날수": days,
            "경보(발행)": len(grouped), "헛경보(발행)": int((~grouped).sum()), "하루헛경보(발행)": fa,
            "판정(발행)": verdict(len(ev), day_rate, fa),
            "경보(대상)": len(t_hit), "헛경보(대상)": int((~t_hit).sum()), "하루헛경보(대상)": fa_t,
            "경보적중률(대상)": t_hit.mean() if len(t_hit) else np.nan,   # 경보 중 실제 피크 근처가 있던 비율
            "하루경보시간(대상,h)": hours / days if days else np.nan,     # 경보가 켜져 있는 시간 — 길수록 너그러운 경보
            "판정(대상)": verdict(len(ev), day_rate, fa_t)}


def periods(month, d):
    """확인 달을 메모 4-4대로 나눈 기간 (발행 날짜 기준)."""
    day = d["origin"].dt.normalize()
    out = {"전체": d}
    if month == CHECK:
        brk = (day >= pd.Timestamp(BREAK[0])) & (day <= pd.Timestamp(BREAK[1]))
        out[f"휴무·재가동 주({BREAK[0][5:]}~{BREAK[1][5:]})"] = d[brk]
        out["나머지"] = d[~brk]
    return out


def run(predictions, include_test=False):
    results = C.RESULT_DIR   # 조건 결과 폴더 (다앎 = results/future_known)
    out = results / "margin"
    cond, cut = raw_table()
    d = prepare(predictions, cond)
    months = sorted(d["달"].unique())
    if MAKE not in months:
        raise ValueError(f"{MAKE} 대표 예측이 없습니다. 완료된 월: {months}")
    apply_to = [m for m in [CHECK] + ([TEST] if include_test else []) if m in months]
    if include_test and TEST not in months:
        raise ValueError(f"{TEST} 테스트 예측이 없습니다.")
    print(f"조건 다앎 | 만드는 달 {MAKE} → 적용 {', '.join(apply_to) or '(없음)'} | 피크 근처 ≥ {NEAR}b · "
          f"분위수 {BASE_Q:.0%} · 최소 {MIN_CELLS}칸·{MIN_DAYS}일 · p < {ALPHA} · 생산량 경계 {cut:g}(1~6월, 0 제외)", flush=True)
    analysis, tables, checks = [], [], []
    for (h, loss), part in d[d["loss"].isin(LOSSES)].groupby(["horizon", "loss"], sort=False):
        mk = part[part["달"] == MAKE]
        near = mk[mk["근처"]]
        name = mk["후보"].iloc[0]
        tag = {"구간": h, "손실": loss, "후보": name}
        if len(near) == 0:
            print(f"{h} {loss}: 7월 피크 근처 칸이 없어 건너뜀", flush=True)
            continue
        base = near["r"].quantile(BASE_Q)
        table, chosen = analyze(near, base)
        mtab = margin_table(near, chosen, base)
        is_base = mtab["조합"].str.startswith("기본")
        short = mtab["비고"].eq("칸 부족 → 기본") if "비고" in mtab else False
        mtab["적용"] = np.where(is_base if MAIN_METHOD == "기본 마진만" else is_base | ~short, "✓", "")   # 주 결과에 쓴 마진
        analysis.append(table.assign(**tag))
        tables.append(mtab.assign(**tag, 고른조건=" + ".join(chosen) or "-"))
        for month in apply_to:
            ap = part[part["달"] == month]
            m_cell = apply_margin(ap, mtab, base, near, chosen)
            for period, sub in periods(month, ap.assign(마진=m_cell)).items():
                for how, m in [("조건별 마진", sub["마진"]), ("기본 마진만", base), ("마진 없이", 0.0)]:
                    checks.append({**tag, "달": month, "기간": period, "방식": how,
                                   "마진%(최소~최대)": f"{np.min(m):.1%}~{np.max(m):.1%}" if np.ndim(m) else f"{m:.1%}",
                                   **check(sub, m)})
    out.mkdir(parents=True, exist_ok=True)
    stale = out / "④9월_시험.csv"
    if TEST not in apply_to and stale.exists():   # 9월을 안 볼 때 예전 9월 표가 남아 섞이지 않게
        stale.unlink()
    save = dict(index=False, encoding="utf-8-sig", float_format="%.4f")
    first = lambda t, cols: t[cols + [c for c in t.columns if c not in cols]]
    if analysis:
        first(pd.concat(analysis), ["구간", "손실", "후보"]).to_csv(out / "①7월_오차분석.csv", **save)
    if not tables or not checks:
        raise ValueError("7월 피크 근처 예측 또는 확인 월 예측이 없어 안전마진을 계산할 수 없습니다.")
    mt = first(pd.concat(tables).sort_values("적용", ascending=False, kind="stable"), ["구간", "손실", "후보", "적용"])
    mt.to_csv(out / "②마진표.csv", **save)
    res = pd.DataFrame(checks)
    res.insert(res.columns.get_loc("방식") + 1, "주결과", np.where(res["방식"] == MAIN_METHOD, "✓", ""))
    res["판정"] = res[f"판정({MAIN_COUNT})"]
    for month in apply_to:   # 주 결과 줄이 먼저
        part = first(res[res["달"] == month].sort_values("주결과", ascending=False, kind="stable"),
                     ["구간", "손실", "후보", "달", "기간", "방식", "주결과", "판정"])
        part.to_csv(out / ("③8월_검증.csv" if month == CHECK else "④9월_시험.csv"), **save)
    keep = ["구간", "손실", "후보", "달", "방식", "주결과", "마진%(최소~최대)", "피크근처_날", "날재현율", "하루헛경보(대상)",
            "경보적중률(대상)", "하루경보시간(대상,h)", "하루헛경보(발행)", "판정"]
    log = res.loc[(res["기간"] == "전체") & (res["방식"] != "마진 없이"), keep].assign(
        실행=datetime.now().strftime("%Y-%m-%d %H:%M"), 설정=f"NEAR {NEAR}, Q {BASE_Q}, 최소 {MIN_CELLS}칸·{MIN_DAYS}일, "
                                                         f"p<{ALPHA}, 조건≤{MAX_CONDS}, 조합표, 주 결과 {MAIN_METHOD}, "
                                                         f"헛경보 {MAIN_COUNT} 기준")
    log_file = out / "조정기록.csv"   # ⑥ 설정·결과가 바뀔 때마다 남는 기록 (같은 설정으로 다시 돌린 같은 결과는 처음 것만)
    old = pd.read_csv(log_file, encoding="utf-8-sig") if log_file.exists() else None
    if old is not None and list(old.columns) != list(log.columns):
        stamp = datetime.fromtimestamp(log_file.stat().st_mtime).strftime("%m%d_%H%M")
        log_file.rename(out / f"조정기록_이전형식_{stamp}.csv")   # 칸이 바뀐 옛 기록은 시각을 붙여 따로 둔다 (덮어쓰지 않음)
        old = None
    if old is not None:
        log = pd.concat([old, log], ignore_index=True)
        same = [c for c in log.columns if c != "실행"]
        # 파일에 적힌 자릿수(소수 4자리)로 맞추고, 다시 읽으며 빈칸이 된 값("주결과"의 '')도 같게 맞춰 비교 (pandas 1.x~3.x 공통)
        key = log[same].round(4).fillna("").apply(lambda col: col.map(str)).agg("|".join, axis=1)
        log = log[~key.duplicated(keep="first")]
    log.to_csv(log_file, **save)

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(f"\n② 마진 표 (b 대비) — ✓ = 주 결과({MAIN_METHOD})에 적용한 마진, 나머지 = 참고\n"
              + mt[["구간", "손실", "후보", "적용", "고른조건", "조합", "칸수", "날수", "마진", "가감"]].to_string(index=False))
        for month in apply_to:
            show = res.loc[(res["달"] == month) & (res["기간"] == "전체") & (res["방식"] != "마진 없이"), keep]
            show = show.sort_values("주결과", ascending=False, kind="stable")
            print(f"\n{'③ 8월 검증' if month == CHECK else '④ 9월 시험'} — 전체 기간, 판정 = {MAIN_COUNT} 기준, "
                  f"✓ = 주 결과({MAIN_METHOD}) (기간별·마진 없이·발행 기준 판정은 csv)\n"
                  + show.drop(columns=["달"]).to_string(index=False))
            n = show[show["주결과"] == "✓"]
            print(f"주 결과 통함: {(n['판정'] == '통함').sum()}/{len(n)}")
    print(f"\n저장: {out}")


