"""4.13 R1 마진의 9월 경보 평가 (딥러닝 앙상블 C, 1h~24h) — 고정 적용, 다시 고르지 않음.

규칙(계산 전에 고정, 4.2 · 4.8과 같은 정의):
- 마진은 7월 예측에서만 만든다. 9월 결과로 마진·규칙을 바꾸지 않는다.
  ① 원래 마진 = 7월 모든 시각 피크 근처 칸 r의 95% 지점 (예전 큰 마진, 비교용)
  ④ R1 = 7월 요금 시간대 피크 근처 칸에서 7/13 · 7/15를 뺀 r의 95% 지점 (기본 마진)
- 경보 = 예측 ≥ (0.9 − 마진) × b(발행 때). 판정은 요금 시간대 칸(일요일 · 공휴일 아닌 날 09~23시)만.
- 판정 두 가지:
  팀 기준 = 4.2 check (날 재현율 ≥ 80% 그리고 대상 기준 하루 헛경보 ≤ 1번)
  운영 판정 v2 = ops_judge.judge 요금 시간대 판 (4.8과 같음)
- 9월 재가동 기간(생산량 0인 날 3일 이상 뒤 첫 5일)이 있는지도 확인해 함께 적는다.
읽는 파일: 7월은 ②예측_7_8월, 9월은 ③예측_9월_대표_마지막에열기 (9월은 이미 채점에 쓴 파일, 여기서도 고르기에 쓰지 않음).
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent                                 # 제출 폴더 최상위
import sys; sys.path.insert(0, str(ROOT)); import paths as P   # 경로는 paths.py 한곳에서 정한다


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


oj = load("ops_judge", "ops_judge.py")
V = load("v42", "4.2_요금시간마진_8월검증.py")

JUL = P.DL_PRED / "예측_7_8월"
SEP = P.DL_PRED / "예측_9월"
CAL_SRC = P.MODEL_INPUT
LABEL = P.PRE_FINAL / "전처리_v1_label.csv"
HORIZONS = ["1h", "4h", "8h", "12h", "24h"]


def read(folder, h):
    d = pd.read_csv(folder / f"{h}.csv", encoding="utf-8-sig", low_memory=False)
    d = d[d["대표"].notna() & (d["loss"] == "C")].copy()
    d["origin"], d["target_time"] = pd.to_datetime(d["origin"]), pd.to_datetime(d["target_time"])
    return d


def restart_days(src):
    day = src.groupby(src["timestamp"].dt.normalize())["생산량"].sum()
    out, run = [], 0
    for t, v in day.items():
        if v == 0:
            run += 1
            continue
        if run >= 3:
            out += list(day.index[(day.index >= t)][:5])
        run = 0
    return pd.DatetimeIndex(out)


def main():
    src = pd.read_csv(CAL_SRC, encoding="utf-8-sig", parse_dates=["timestamp"])
    cal = oj.build_calendar(src["timestamp"], src["power_15min"], src["생산량"], end=None)
    lab = pd.read_csv(LABEL, encoding="utf-8-sig", parse_dates=["시각"]).set_index("시각")
    bill = (lab["daytype"] != 0) & (lab.index.hour >= 9) & (lab.index.hour < 23)
    op_bill = cal["op"].to_numpy() & bill.reindex(cal.index).fillna(False).to_numpy()
    rs = restart_days(src)
    print("재가동 기간(전체):", [t.strftime("%m-%d") for t in rs])
    rows = []
    for h in HORIZONS:
        jul = read(JUL, h)
        jul = jul[jul["period"] == "7월"]
        sep = read(SEP, h)
        for d in (jul, sep):
            d["요금"] = d["target_time"].map(bill).fillna(False).astype(bool)
            d["r"] = (d["y"] - d["yhat"]).clip(lower=0) / d["b_origin"]
        near = jul[jul["y"] >= V.NEAR * jul["b_origin"]]
        nb = near[near["요금"]]
        plans = {"① 원래 마진": near["r"].quantile(V.Q),
                 "④ R1": nb.loc[~nb["target_time"].dt.strftime("%m-%d").isin(V.FILLED), "r"].quantile(V.Q)}
        sep = sep.reset_index(drop=True)
        for name, m in plans.items():
            team = V.check(sep, m, True)
            alarm = sep["yhat"].to_numpy() >= (V.NEAR - m) * sep["b_origin"].to_numpy()
            keep = sep["요금"].to_numpy()
            cal["op"] = op_bill
            v2 = oj.judge(sep.loc[keep, ["target_time", "y", "b_origin"]].reset_index(drop=True), alarm[keep])
            rows.append({"구간": h, "규칙": name, "마진": m,
                         **{f"팀_{k}": v for k, v in team.items()}, **{f"v2_{k}": v for k, v in v2.items()}})
    res = pd.DataFrame(rows)
    res.to_csv(V.mid("4.13_R1마진_9월평가.csv"), index=False, encoding="utf-8-sig", float_format="%.4f")
    keep = ["구간", "규칙", "마진", "팀_피크근처_날", "팀_잡은날", "팀_날재현율", "팀_하루헛경보(대상)", "팀_하루경보시간(h)",
            "팀_판정", "v2_헛경보3이하_가동일비율", "v2_헛경보시간비율", "v2_헛경보시간_기준", "v2_운영판정_합격"]
    with pd.option_context("display.width", 300, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        text = res[keep].to_string(index=False)
    V.mid("4.13_log.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
