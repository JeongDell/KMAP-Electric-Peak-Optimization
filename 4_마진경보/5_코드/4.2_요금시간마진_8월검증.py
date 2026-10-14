"""4.2 요금 시간대 마진 — 8월 검증 (동수 대표 예측).

규칙 (계산 전에 확정, 2026-10-08 사용자 "진행해"):
- 요금 시간대 = 일요일 · 공휴일이 아닌 날의 09:00~22:59 (4.1과 같음).
- 피크 근처 = 요금 시간대 칸 중 실제 ≥ 0.9 × b(발행 때). 경보도 요금 시간대 칸을 가리킬 때만 센다.
- 마진 = 7월 요금 시간대 피크 근처 칸 r의 95% 지점. 주 결과 = 생산량 채운 날(7/13 · 7/15) 제외판, 포함판 함께.
- 판정 = 동수 margin.check와 같다: 대상 기준 헛경보(60분), 날 재현율 ≥ 80% 그리고 하루 헛경보 ≤ 1번,
  하루 = 발행 날 수(주말 포함). 하루 경보 시간 · 경보 적중률 · 발행 기준 헛경보도 같이 낸다.
- 비교: 원래 규칙(모든 시각, 기본 마진) — 동수 ③8월_검증.csv와 같은 값이 나오는지로 코드 확인.
8월 예측만 읽는다. 9월은 읽지 않는다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent                                 # 제출 폴더 최상위
import sys; sys.path.insert(0, str(ROOT)); import paths as P   # 경로는 paths.py 한곳에서 정한다
PRED = P.DL_PRED / "예측_7_8월"
RAW = P.PRE_FINAL / "전처리_v1_label.csv"
HORIZONS = ["1h", "4h", "8h", "12h", "24h", "4d"]
NEAR, Q, SUPPRESS = 0.9, 0.95, pd.Timedelta(minutes=60)
PASS_DAY, MAX_FA = 0.8, 1.0
FILLED = ["07-13", "07-15"]


def mid(name):
    sub = {"csv": "csv", "txt": "로그"}[name.rsplit(".", 1)[1]]
    d = CH / "6_중간산출물" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def verdict(n_ev, day_rate, fa):
    if not n_ev:
        return "사례 없음"
    if day_rate >= PASS_DAY:
        return "통함" if fa <= MAX_FA else "헛경보 많음"
    return "놓침 많음" if fa <= MAX_FA else "둘 다 못 미침"


def check(a, m, bill_only):
    """동수 margin.check와 같은 셈. bill_only면 피크 근처 · 경보칸을 요금 시간대 칸으로 한정."""
    mask = a["요금"] if bill_only else pd.Series(True, index=a.index)
    a = a.assign(근처=(a["y"] >= NEAR * a["b_origin"]) & mask,
                 경보칸=(a["yhat"] >= (NEAR - m) * a["b_origin"]) & mask)
    days = a["origin"].dt.normalize().nunique()
    per = a.groupby("origin").agg(경보=("경보칸", "any"), 근처=("근처", "any")).sort_index()
    start, groups = None, []
    for t in per.index[per["경보"]]:
        if start is None or t - start >= SUPPRESS:
            start = t
        groups.append(start)
    grouped = per[per["경보"]].assign(묶음=groups).groupby("묶음")["근처"].any()
    ev = a.loc[a["근처"], "target_time"].drop_duplicates()
    hit = ev.isin(set(a.loc[a["근처"] & a["경보칸"], "target_time"]))
    ev_days = ev.dt.normalize()
    day_rate = ev_days[hit].nunique() / ev_days.nunique() if len(ev) else np.nan
    cells = a.loc[a["경보칸"], "target_time"].drop_duplicates().sort_values()
    gid = (cells.diff() > SUPPRESS).cumsum().to_numpy()
    t_hit = cells.isin(set(ev)).groupby(gid).any() if len(cells) else pd.Series(dtype=bool)
    span = cells.groupby(gid).agg(["min", "max"]) if len(cells) else pd.DataFrame(columns=["min", "max"])
    hours = (span["max"] - span["min"]).dt.total_seconds().sum() / 3600 + len(span) * 0.25 if len(span) else 0.0
    fa_t = (~t_hit).sum() / days
    return {"피크근처_칸": len(ev), "피크근처_날": ev_days.nunique(), "잡은날": ev_days[hit].nunique(),
            "날재현율": day_rate, "하루헛경보(대상)": fa_t, "경보적중률(대상)": t_hit.mean() if len(t_hit) else np.nan,
            "하루경보시간(h)": hours / days, "하루헛경보(발행)": (~grouped).sum() / days,
            "판정": verdict(len(ev), day_rate, fa_t)}


def main():
    raw = pd.read_csv(RAW, encoding="utf-8-sig", parse_dates=["시각"]).set_index("시각")
    bill_cell = (raw["daytype"] != 0) & (raw.index.hour >= 9) & (raw.index.hour < 23)
    rows = []
    for h in HORIZONS:
        d = pd.read_csv(PRED / f"{h}.csv", encoding="utf-8-sig", low_memory=False)
        d = d[d["대표"].notna()].copy()
        d["origin"], d["target_time"] = pd.to_datetime(d["origin"]), pd.to_datetime(d["target_time"])
        d["요금"] = d["target_time"].map(bill_cell).fillna(False).astype(bool)
        d["r"] = (d["y"] - d["yhat"]).clip(lower=0) / d["b_origin"]
        for loss, g in d.groupby("loss"):
            jul, aug = g[g["period"] == "7월"], g[g["period"] == "8월"]
            near = jul[jul["y"] >= NEAR * jul["b_origin"]]
            nb = near[near["요금"]]
            m = {"원래": near["r"].quantile(Q),
                 "요금(채운 날 포함)": nb["r"].quantile(Q),
                 "요금(채운 날 제외)": nb.loc[~nb["target_time"].dt.strftime("%m-%d").isin(FILLED), "r"].quantile(Q)}
            plans = [("① 원래 규칙 (모든 시각, 기본 마진)", m["원래"], False),
                     ("② 요금 시간대 판정 + 원래 마진", m["원래"], True),
                     ("③ 요금 시간대 + 요금 마진(채운 날 포함)", m["요금(채운 날 포함)"], True),
                     ("④ 요금 시간대 + 요금 마진(채운 날 제외) ← 주", m["요금(채운 날 제외)"], True),
                     ("(참고) 요금 시간대 + 마진 없이", 0.0, True)]
            for name, mm, bo in plans:
                rows.append({"구간": h, "손실": loss, "규칙": name, "마진": mm, **check(aug, mm, bo)})
    res = pd.DataFrame(rows)
    res.to_csv(mid("4.2_8월검증.csv"), index=False, encoding="utf-8-sig", float_format="%.4f")
    lines = []
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
        for name in res["규칙"].unique():
            s = res[res["규칙"] == name]
            lines += ["", f"{name} : 통함 {(s['판정'] == '통함').sum()}/{len(s)}",
                      s.drop(columns="규칙").to_string(index=False)]
    mid("4.2_log.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
