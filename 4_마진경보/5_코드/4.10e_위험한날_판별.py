"""4.10e 아침 예보로 "오늘이 위험한 날인가" 판별 [참고].

규칙 (계산 전에 적음, 2026-10-08):
- 24h 모델 08:00 발행 하나. 그날 요금 시간대(09:00~22:45) 예측 중 최고 위험 점수 S = max(예측 ÷ b).
- 위험한 날 = 그날 요금 시간대에 피크 근처(실제 ≥ 0.9·b) 칸이 있는 날. 일 · 공휴일 · 7/13 · 7/15 제외.
- 기준선 T는 7월로만 정한다: 7월 위험한 날의 S 중 하위 10% 지점(= 7월 위험한 날의 90%를 잡는 선).
- 8월에 그대로 적용: S ≥ T인 날만 "위험한 날"로 알리고, 그날만 상위 k칸(2 · 3시간)을 위험 시간으로 지정.
- 보고: 날 단위 AUC, 알린 날 수, 위험한 날 중 알린 날, 그 안에서 위험 시간이 피크를 잡은 날. 7월 · 8월 평소 날 · 8월 재가동 주.
- 동수 · 정현 손실 C. 9월은 읽지 않는다.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m410", HERE / "4.10_위험도_순위경보.py")
M10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M10)
M, V = M10.M, M10.V


def main():
    lab = pd.read_csv(M.LABEL, encoding="utf-8-sig", parse_dates=["시각"]).set_index("시각")
    bill = (lab["daytype"] != 0) & (lab.index.hour >= 9) & (lab.index.hour < 23)
    rs = lambda t: (t >= pd.Timestamp(M.RESTART[0])) & (t < pd.Timestamp(M.RESTART[1]))
    rows, days_out = [], []
    for who, folder in M.PRED.items():
        d = pd.read_csv(folder / "24h.csv", encoding="utf-8-sig", low_memory=False)
        d = d[d["대표"].notna() & (d["loss"] == "C")].copy()
        d["origin"], d["target_time"] = pd.to_datetime(d["origin"]), pd.to_datetime(d["target_time"])
        d = d[(d["origin"].dt.hour == 8) & (d["origin"].dt.minute == 0)]
        d = d[d["target_time"].dt.normalize() == d["origin"].dt.normalize()]
        d = d[d["target_time"].map(bill).fillna(False).astype(bool)]
        d = d[~d["target_time"].dt.strftime("%m-%d").isin(V.FILLED)].copy()
        d["s"] = d["yhat"] / d["b_origin"]
        d["근처"] = d["y"] >= V.NEAR * d["b_origin"]
        d["day"] = d["target_time"].dt.normalize()
        day = d.groupby("day").agg(S=("s", "max"), 위험=("근처", "any"), 달=("period", "first"))
        T = day.loc[(day["달"] == "7월") & day["위험"], "S"].quantile(0.10)
        for k in [8, 12]:
            top = d.sort_values("s", ascending=False).groupby("day").head(k)
            hit = top.groupby("day")["근처"].any()
            day[f"잡음_{k}"] = hit.reindex(day.index).fillna(False)
        day["알림"] = day["S"] >= T
        for part, sel in [("7월", day["달"] == "7월"), ("8월 평소 날", (day["달"] == "8월") & ~rs(day.index)),
                          ("8월 재가동 주", (day["달"] == "8월") & rs(day.index))]:
            x = day[sel]
            r = {"모델": who, "기간": part, "기준선T": T, "날": len(x), "위험한_날": int(x["위험"].sum()),
                 "알린_날": int(x["알림"].sum()), "위험한날_중_알림": int((x["위험"] & x["알림"]).sum()),
                 "날AUC": M10.auc(x["S"], x["위험"])}
            for k in [8, 12]:
                r[f"알림+위험시간{k/4:g}h로_잡은날"] = int((x["위험"] & x["알림"] & x[f"잡음_{k}"]).sum())
            rows.append(r)
        days_out.append(day.assign(모델=who))
    res = pd.DataFrame(rows)
    res.to_csv(V.mid("4.10e_위험한날.csv"), index=False, encoding="utf-8-sig", float_format="%.4f")
    pd.concat(days_out).to_csv(V.mid("4.10e_날별.csv"), encoding="utf-8-sig", float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        text = res.to_string(index=False)
    V.mid("4.10e_log.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
