"""4.8 ML 운영 판정 v2로 4.2 · 4.3 결과 재채점 (8월, 1h~24h, 손실 C 대표, DL · ML).

판정 코드: 같은 폴더 ops_judge.py (ML 단독 실행판, 로직 수정 없음).
달력: 1.전처리/3_최종데이터/timeseries_15min_v1_previous_week.csv (전력 · 생산량, 7/13 · 7/15 생산량 = 지난주 복사).
마진: 4.2(요금 시간대 · 원래 · 7/13 · 7/15 제외) · 4.3(분리 마진)과 같은 값을 다시 계산한다.
두 판으로 채점한다.
  하루 전체 판 = ML 정의 그대로.
  요금 시간대 판 = 예측 행을 요금 시간대 칸(일 · 공휴일 아닌 날 09~23시)으로 한정하고, 가동 칸도 요금 시간대로 한정.
8월만 쓴다. 9월은 읽지 않는다.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent                                 # 제출 폴더 최상위
import sys; sys.path.insert(0, str(ROOT)); import paths as P   # 경로는 paths.py 한곳에서 정한다
spec = importlib.util.spec_from_file_location("ops_judge", HERE / "ops_judge.py")
oj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oj)
spec = importlib.util.spec_from_file_location("v42", HERE / "4.2_요금시간마진_8월검증.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

CAL_SRC = P.MODEL_INPUT
LABEL = P.PRE_FINAL / "전처리_v1_label.csv"
PRED = {"DL": P.DL_PRED / "예측_7_8월",
        "ML": P.TREE_PRED}
HORIZONS = ["1h", "4h", "8h", "12h", "24h"]
RESTART = ("2021-08-09", "2021-08-14")   # 재가동 주 8/9~8/13 (4.3 정의)


def main():
    src = pd.read_csv(CAL_SRC, encoding="utf-8-sig", parse_dates=["timestamp"])
    cal = oj.build_calendar(src["timestamp"], src["power_15min"], src["생산량"], end="2021-09-01")
    lab = pd.read_csv(LABEL, encoding="utf-8-sig", parse_dates=["시각"]).set_index("시각")
    bill = (lab["daytype"] != 0) & (lab.index.hour >= 9) & (lab.index.hour < 23)
    op_full = cal["op"].copy()
    op_bill = op_full & bill.reindex(cal.index).fillna(False).to_numpy()
    rows = []
    for who, folder in PRED.items():
        for h in HORIZONS:
            d = pd.read_csv(folder / f"{h}.csv", encoding="utf-8-sig", low_memory=False)
            d = d[d["대표"].notna() & (d["loss"] == "C")].copy()
            d["target_time"] = pd.to_datetime(d["target_time"])
            d["요금"] = d["target_time"].map(bill).fillna(False).astype(bool)
            d["r"] = (d["y"] - d["yhat"]).clip(lower=0) / d["b_origin"]
            jul, aug = d[d["period"] == "7월"], d[d["period"] == "8월"].reset_index(drop=True)
            near = jul[jul["y"] >= V.NEAR * jul["b_origin"]]
            nb = near[near["요금"]]
            m_base = near["r"].quantile(V.Q)
            m_bill = nb["r"].quantile(V.Q)
            m_small = nb.loc[~nb["target_time"].dt.strftime("%m-%d").isin(V.FILLED), "r"].quantile(V.Q)
            rs = (aug["target_time"] >= RESTART[0]) & (aug["target_time"] < RESTART[1])
            plans = {"① 원래 마진": np.full(len(aug), m_base),
                     "③ 요금 마진(7/13·7/15 포함)": np.full(len(aug), m_bill),
                     "④ 작은 마진(7/13·7/15 제외)": np.full(len(aug), m_small),
                     "⑤ 분리 마진(평소 ④ · 재가동 주 ①) [참고]": np.where(rs, m_base, m_small)}
            for name, m in plans.items():
                alarm = aug["yhat"].to_numpy() >= (V.NEAR - m) * aug["b_origin"].to_numpy()
                for ver in ["하루 전체", "요금 시간대"]:
                    if ver == "하루 전체":
                        cal["op"] = op_full
                        a, al = aug, alarm
                    else:
                        cal["op"] = op_bill
                        keep = aug["요금"].to_numpy()
                        a, al = aug[keep].reset_index(drop=True), alarm[keep]
                    r = oj.judge(a[["target_time", "y", "b_origin"]], al)
                    rows.append({"모델": who, "구간": h, "규칙": name, "판": ver, "마진_평소": float(np.min(m)),
                                 "마진_최대": float(np.max(m)), **r})
    cal["op"] = op_full
    res = pd.DataFrame(rows)
    res.to_csv(V.mid("4.8_운영판정_재채점.csv"), index=False, encoding="utf-8-sig", float_format="%.4f")
    keep = ["모델", "구간", "규칙", "판", "잡은_날", "피크근처_날", "헛경보3이하_가동일비율", "최악일_헛경보", "헛경보시간비율",
            "헛경보시간_기준", "경보시간비율_가동", "합격_재현율", "합격_날별헛경보", "합격_헛경보시간", "운영판정_합격"]
    with pd.option_context("display.width", 300, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        text = res[keep].to_string(index=False)
    V.mid("4.8_log.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
