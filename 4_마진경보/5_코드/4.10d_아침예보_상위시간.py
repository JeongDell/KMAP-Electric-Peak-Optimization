"""4.10d 아침 예보로 '오늘 가장 위험한 시간' 지정 [참고, 8월 결과를 본 뒤의 탐색].

4.10의 ③(하루 상위 시간)은 하루 동안 다른 시각에 낸 예측을 모아 순위를 매겨, 실제로는 미리 알 수 없는 정보를 썼다.
여기서는 운영에서 그대로 할 수 있게, 하루 한 번 아침에 낸 예측 하나로 순위를 매긴다.

규칙 (계산 전에 적음, 2026-10-08):
- 24h 모델의 08:00 발행 하나로 그날 요금 시간대(09:00~22:45 시작 칸) 예측을 본다. 일 · 공휴일 제외.
- 그 예측 중 위험 점수 s = 예측 ÷ b(발행 때)가 가장 높은 칸 상위 k개를 "오늘의 위험 시간"으로 지정. k = 4(1시간) · 8(2시간) · 12(3시간).
- 피크 근처 = 실제 ≥ 0.9·b(발행 때). 잡은 날 = 그날 피크 근처 칸이 위험 시간 안에 하나라도 있음.
- 같은 방식으로 12h 모델 09:00 발행(09:00~20:45)도 본다.
- 7월(마진 · 설정을 고른 달)과 8월(평소 날 / 재가동 주)을 나눠 보고한다. 동수 · 정현 손실 C. 9월은 읽지 않는다.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m48", HERE / "4.8_운영판정_재채점.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
V = M.V
PLANS = [("24h", 8), ("12h", 9)]   # (모델 구간, 발행 시각)


def main():
    lab = pd.read_csv(M.LABEL, encoding="utf-8-sig", parse_dates=["시각"]).set_index("시각")
    bill = (lab["daytype"] != 0) & (lab.index.hour >= 9) & (lab.index.hour < 23)
    rs = lambda t: (t >= pd.Timestamp(M.RESTART[0])) & (t < pd.Timestamp(M.RESTART[1]))
    rows = []
    for who, folder in M.PRED.items():
        for h, hour in PLANS:
            d = pd.read_csv(folder / f"{h}.csv", encoding="utf-8-sig", low_memory=False)
            d = d[d["대표"].notna() & (d["loss"] == "C")].copy()
            d["origin"], d["target_time"] = pd.to_datetime(d["origin"]), pd.to_datetime(d["target_time"])
            d = d[(d["origin"].dt.hour == hour) & (d["origin"].dt.minute == 0)]
            d = d[d["target_time"].dt.normalize() == d["origin"].dt.normalize()]
            d = d[d["target_time"].map(bill).fillna(False).astype(bool)]
            d = d[~d["target_time"].dt.strftime("%m-%d").isin(V.FILLED)].copy()
            d["s"] = d["yhat"] / d["b_origin"]
            d["근처"] = d["y"] >= V.NEAR * d["b_origin"]
            d["day"] = d["target_time"].dt.normalize()
            for k in [4, 8, 12]:
                top = d.sort_values("s", ascending=False).groupby("day").head(k)
                for part, sel_all, sel_top in [
                        ("7월", d["period"] == "7월", top["period"] == "7월"),
                        ("8월 평소 날", (d["period"] == "8월") & ~rs(d["target_time"]), (top["period"] == "8월") & ~rs(top["target_time"])),
                        ("8월 재가동 주", (d["period"] == "8월") & rs(d["target_time"]), (top["period"] == "8월") & rs(top["target_time"]))]:
                    a, t = d[sel_all], top[sel_top]
                    ev = a.loc[a["근처"], "day"].unique()
                    caught = t.loc[t["근처"], "day"].unique()
                    first = t[t["근처"]].groupby("day")["target_time"].min()
                    rows.append({"모델": who, "예보": f"{h} {hour:02d}:00 발행", "위험시간": f"{k/4:g}h", "기간": part,
                                 "날": a["day"].nunique(), "피크근처_날": len(ev), "잡은_날": len(caught),
                                 "켜진칸_적중률": t["근처"].mean() if len(t) else np.nan,
                                 "첫 적중까지_h_중앙값": float(((first - (first.index + pd.Timedelta(hours=hour)))
                                                          .dt.total_seconds() / 3600).median()) if len(first) else np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(V.mid("4.10d_아침예보_상위시간.csv"), index=False, encoding="utf-8-sig", float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
        text = res.to_string(index=False)
    V.mid("4.10d_log.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
