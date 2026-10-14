"""4.1 요금 시간대 마진 · 저생산 피크 칸 확인 (7월만).

① DL 대표 예측(7월)으로 기본 마진(피크 근처 r의 95% 지점)을 다시 계산해 DL 값과 맞는지 본다.
② 요금 시간대 피크 근처 칸만으로 마진을 다시 낸다[결정 2026-10-08]. 요금 시간대 = 일요일 · 공휴일이 아닌 날의 09:00~22:59
  (계시별 요금 가정: 경부하 23~09시와 일요일 · 공휴일 하루 전체는 요금적용전력에서 빠짐, 5번_분석/data/tariff_research.md).
③ 시간대 묶음(09~18 · 18~23 · 23~09 · 일공휴)별 칸 수와 r 95% 지점.
⑤ 요금 시간대에서 생산량 채운 날(7/13 · 7/15)을 뺀 마진 (참고).
④ 생산량이 적은(1~6월 0 제외 중앙값 이하) 피크 근처 칸 목록 — 날짜 · 시각 · 요일 · 전력 · 생산량 · 앞뒤 생산.
8·9월은 읽지 않는다.
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
DONGSU_BASE = {("1h", "A"): .2761, ("1h", "C"): .2589, ("4h", "A"): .3442, ("4h", "C"): .3413,
               ("8h", "A"): .3529, ("8h", "C"): .3649, ("12h", "A"): .3773, ("12h", "C"): .3748,
               ("24h", "A"): .3838, ("24h", "C"): .3760, ("4d", "A"): .3643, ("4d", "C"): .3928}
NEAR, Q = 0.9, 0.95
FILLED = ["07-13", "07-15"]   # 1.6: 생산량 하루 0 → 지난주 같은 요일 값으로 채운 날 (f_생산량대체)


def mid(name):
    sub = {"csv": "csv", "txt": "로그"}[name.rsplit(".", 1)[1]]
    d = CH / "6_중간산출물" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def window(t, daytype):
    """칸 시작 시각 → 시간대 묶음. 일요일 · 공휴일(daytype 0)은 하루 전체 '일공휴'."""
    h = t.dt.hour
    w = np.where((h >= 9) & (h < 18), "09~18", np.where((h >= 18) & (h < 23), "18~23", "23~09"))
    return np.where(t.map(daytype).to_numpy() == 0, "일공휴", w)


def main():
    raw = pd.read_csv(RAW, encoding="utf-8-sig", parse_dates=["시각"]).set_index("시각")
    dt = raw["daytype"]
    prod = raw["생산량"].astype(float)
    cut = float(prod[(prod.index < "2021-07-01") & (prod > 0)].median())
    out, lines = [], [f"생산량 경계(1~6월, 0 제외 중앙값) = {cut:g}"]
    lows = []
    for h in HORIZONS:
        d = pd.read_csv(PRED / f"{h}.csv", encoding="utf-8-sig", low_memory=False)
        d = d[(d["period"] == "7월") & d["대표"].notna()].copy()
        d["target_time"] = pd.to_datetime(d["target_time"])
        d["r"] = (d["y"] - d["yhat"]).clip(lower=0) / d["b_origin"]
        near = d[d["y"] >= NEAR * d["b_origin"]].copy()
        near["창"] = window(near["target_time"], dt)
        near["생산"] = near["target_time"].map(prod)
        for loss, g in near.groupby("loss"):
            base = g["r"].quantile(Q)
            bill = g[g["창"].isin(["09~18", "18~23"])]
            row = {"구간": h, "손실": loss, "기본마진_재계산": base, "기본마진_DL": DONGSU_BASE[(h, loss)],
                   "요금시간마진": bill["r"].quantile(Q),
                   "요금시간_생산많음_마진": bill.loc[bill["생산"] > cut, "r"].quantile(Q),
                   # 참고: 생산량이 원자료 결함(하루 0)이라 전처리에서 채운 날(7/13 · 7/15) 제외
                   "요금시간마진_채운날제외": bill.loc[~bill["target_time"].dt.strftime("%m-%d").isin(FILLED), "r"].quantile(Q)}
            for w in ["09~18", "18~23", "23~09", "일공휴"]:
                gw = g[g["창"] == w]
                row[f"{w}_칸"] = gw["target_time"].nunique()
                row[f"{w}_날"] = gw["target_time"].dt.normalize().nunique()
                row[f"{w}_r95"] = gw["r"].quantile(Q) if len(gw) else np.nan
            out.append(row)
            if loss == "C":
                lows.append(g[g["생산"] <= cut].assign(구간=h))
    res = pd.DataFrame(out)
    res.to_csv(mid("4.1_요금시간마진.csv"), index=False, encoding="utf-8-sig")
    lines.append(res.round(3).to_string())

    # ④ 저생산 피크 근처 칸 (손실 C 대표 기준, 구간 합쳐 칸 하나씩)
    low = pd.concat(lows).groupby("target_time").agg(
        전력=("y", "first"), b=("b_origin", "min"), 구간=("구간", lambda s: ",".join(sorted(set(s))))).reset_index()
    t = low["target_time"]
    low["요일"] = t.dt.dayofweek.map(dict(enumerate("월화수목금토일")))
    low["창"] = window(t, dt)
    low["생산"] = t.map(prod)
    low["1시간전_생산"] = (t - pd.Timedelta(hours=1)).map(prod)
    low["1시간후_생산"] = (t + pd.Timedelta(hours=1)).map(prod)
    low["daytype"] = t.map(dt)
    low["그날_생산합"] = t.dt.normalize().map(prod.resample("D").sum() / 4)
    low.to_csv(mid("4.1_저생산피크칸.csv"), index=False, encoding="utf-8-sig")
    lines += ["", f"저생산 피크 근처 칸 (손실 C 대표, 구간 합침): {len(low)}칸, {t.dt.normalize().nunique()}일",
              low.to_string(index=False)]
    by_day = low.groupby(t.dt.date).agg(칸=("전력", "size"), 시각=("target_time", lambda s: f"{s.min():%H:%M}~{s.max():%H:%M}"),
                                       창=("창", lambda s: ",".join(sorted(set(s)))), 최고전력=("전력", "max"),
                                       생산범위=("생산", lambda s: f"{s.min():g}~{s.max():g}"),
                                       그날_생산합=("그날_생산합", "first"))
    lines += ["", "날짜별 요약", by_day.to_string()]
    mid("4.1_log.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
