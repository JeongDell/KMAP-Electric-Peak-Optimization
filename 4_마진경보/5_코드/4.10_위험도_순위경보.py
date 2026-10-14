"""4.10 위험도 숫자 · 하루 상위 시간 경보 [참고, 8월 결과를 본 뒤의 탐색].

목적: "경보 선을 넘었나"(하루 수 시간 켜짐) 대신, 점 예측의 순위로 위험을 나눌 수 있는지 본다.

규칙 (계산 전에 적음, 2026-10-08):
- 칸마다 예측 하나만 쓴다: 구간 H의 "H시간 앞" 예측 (1h = lead 4, 4~24h = lead 4H−3 ~ 4H 중 그 칸을 덮는 하나).
  → 각 칸에 정확히 한 번, 피크보다 H시간 앞서 낸 예측.
- 위험 점수 s = 예측 ÷ b(발행 때). 피크 근처 = 실제 ≥ 0.9·b(발행 때).
- 대상 칸 = 요금 시간대(일 · 공휴일 아닌 날 09~23시) 가동 칸(ops_judge 정의), 7/13 · 7/15 제외.
① 순위가 의미 있나: 8월 칸에서 s로 피크 근처를 가려내는 AUC (0.5 = 동전, 1 = 완벽).
② 위험도 숫자: 7월 칸으로 s 구간별 "실제 피크 근처 비율"을 만들고, 8월 같은 구간의 비율과 비교.
   구간: <0.70, 0.70~0.75, 0.75~0.80, 0.80~0.85, ≥0.85.
③ 하루 상위 시간 경보: 가동일마다 s가 높은 칸 상위 4칸(1시간) · 8칸(2시간)만 경보.
   지표: 하루 경보 시간(1h · 2h 고정), 피크 근처 날 중 잡은 날, 켜진 칸 적중률.
8월을 평소 날 / 재가동 주(8/9~8/13)로 나눠 본다. 동수 · 정현 손실 C, 1h~24h. 9월은 읽지 않는다.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m48", HERE / "4.8_운영판정_재채점.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
oj, V = M.oj, M.V
BINS = [-np.inf, .70, .75, .80, .85, np.inf]
LABELS = ["<0.70", "0.70~0.75", "0.75~0.80", "0.80~0.85", "≥0.85"]


def auc(score, label):
    """순위 기반 AUC (동점은 평균 순위)."""
    label = np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    rank = pd.Series(score).rank().to_numpy()
    return (rank[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def one_lead(d, h):
    """각 칸에 'H시간 앞' 예측 하나."""
    H = int(h[:-1])
    lo = 4 * H - 3 if H > 1 else 4
    return d[(d["lead"] >= lo) & (d["lead"] <= 4 * H)].drop_duplicates("target_time", keep="first")


def main():
    src = pd.read_csv(M.CAL_SRC, encoding="utf-8-sig", parse_dates=["timestamp"])
    cal = oj.build_calendar(src["timestamp"], src["power_15min"], src["생산량"], end="2021-09-01")
    lab = pd.read_csv(M.LABEL, encoding="utf-8-sig", parse_dates=["시각"]).set_index("시각")
    bill = (lab["daytype"] != 0) & (lab.index.hour >= 9) & (lab.index.hour < 23)
    ok = cal["op"] & bill.reindex(cal.index).fillna(False)
    ok = ok[ok & ~cal.index.strftime("%m-%d").isin(V.FILLED)]
    rs = lambda t: (t >= pd.Timestamp(M.RESTART[0])) & (t < pd.Timestamp(M.RESTART[1]))
    rows_a, rows_b, rows_c = [], [], []
    for who, folder in M.PRED.items():
        for h in M.HORIZONS:
            d = pd.read_csv(folder / f"{h}.csv", encoding="utf-8-sig", low_memory=False)
            d = d[d["대표"].notna() & (d["loss"] == "C")].copy()
            d["target_time"] = pd.to_datetime(d["target_time"])
            d = one_lead(d.sort_values(["target_time", "lead"]), h)
            d = d[d["target_time"].isin(ok.index)].copy()
            d["s"] = d["yhat"] / d["b_origin"]
            d["근처"] = d["y"] >= V.NEAR * d["b_origin"]
            d["구간_s"] = pd.cut(d["s"], BINS, labels=LABELS)
            d["day"] = d["target_time"].dt.normalize()
            jul, aug = d[d["period"] == "7월"], d[d["period"] == "8월"]
            # ① AUC
            for part, a in [("7월", jul), ("8월 전체", aug), ("8월 평소 날", aug[~rs(aug["target_time"])]),
                            ("8월 재가동 주", aug[rs(aug["target_time"])])]:
                rows_a.append({"모델": who, "구간": h, "기간": part, "칸": len(a), "피크근처_칸": int(a["근처"].sum()),
                               "AUC": auc(a["s"], a["근처"])})
            # ② 위험도 표
            for part, a in [("7월", jul), ("8월 평소 날", aug[~rs(aug["target_time"])]), ("8월 재가동 주", aug[rs(aug["target_time"])])]:
                g = a.groupby("구간_s", observed=False)["근처"].agg(["size", "sum"])
                for b_, r in g.iterrows():
                    rows_b.append({"모델": who, "구간": h, "기간": part, "위험점수": b_, "칸": int(r["size"]),
                                   "피크근처": int(r["sum"]), "비율": r["sum"] / r["size"] if r["size"] else np.nan})
            # ③ 하루 상위 시간
            for k in [4, 8]:
                top = aug.sort_values("s", ascending=False).groupby("day").head(k)
                for part, sel_t, sel_a in [("8월 평소 날", ~rs(top["target_time"]), ~rs(aug["target_time"])),
                                           ("8월 재가동 주", rs(top["target_time"]), rs(aug["target_time"]))]:
                    t, a = top[sel_t], aug[sel_a]
                    ev_days = a.loc[a["근처"], "day"].unique()
                    caught = t.loc[t["근처"], "day"].unique()
                    rows_c.append({"모델": who, "구간": h, "기간": part, "하루_경보": f"{k/4:g}h", "피크근처_날": len(ev_days),
                                   "잡은_날": len(caught), "켜진칸_적중률": t["근처"].mean() if len(t) else np.nan,
                                   "가동일": a["day"].nunique()})
    A, B, C = pd.DataFrame(rows_a), pd.DataFrame(rows_b), pd.DataFrame(rows_c)
    for df, name in [(A, "4.10a_AUC.csv"), (B, "4.10b_위험도표.csv"), (C, "4.10c_하루상위경보.csv")]:
        df.to_csv(V.mid(name), index=False, encoding="utf-8-sig", float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_rows", 500,
                           "display.float_format", "{:.3f}".format):
        text = "\n\n".join(["① AUC", A.to_string(index=False), "② 위험도 표", B.to_string(index=False),
                            "③ 하루 상위 시간 경보", C.to_string(index=False)])
    V.mid("4.10_log.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
