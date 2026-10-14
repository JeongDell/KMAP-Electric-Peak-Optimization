"""4.14 위험 시간(예측 상위 12칸) vs 고정 시간대 기준선 — 같은 위험한 날 조건에서 비교 (7·8·9월).

목적: 통합점검 zip 표 3-2(7월 14·13일, 8월 5·4일, 9월 8·8일)를 우리 코드로 다시 확인한다.
규칙(7.9와 같음): 딥러닝 24h 앙상블 C, 08:00 발행 하나, 요금 산정 칸(일요일·공휴일 아닌 날 09:00~22:45),
7/13·7/15 제외, 위험 점수 S = 그날 예측 ÷ b 최댓값, 기준선 T = 7월 피크 근접 날 S의 하위 10% 지점,
위험한 날 = S ≥ T 또는 재가동 기간(생산량 0인 날 3일 이상 뒤 생산 있는 첫 5일).
위험 시간 = 그날 예측 ÷ b 상위 12칸(동점은 이른 시각 우선).
고정 시간대 = 7월 요금 산정 칸에서 피크 근접 칸이 가장 많았던 3개 시(時)의 12칸. 7월로만 정한다.
비교는 같은 '알린 날'에서 한다. 지표: 피크 근접 날 중 잡은 날, 경보 칸 중 실제 피크 근접 칸.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent                                 # 제출 폴더 최상위
import sys; sys.path.insert(0, str(ROOT)); import paths as P   # 경로는 paths.py 한곳에서 정한다
SRC = P.DL_PRED
HOL = set(pd.to_datetime(["2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-03-01", "2021-05-05",
                          "2021-05-19", "2021-06-06", "2021-08-15", "2021-08-16"]))


def mid(name):
    p = CH / "6_중간산출물" / "csv"
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def restart_days(f):
    day = f.groupby(f["ts"].dt.normalize())["생산량"].sum()
    zero, idx, out, i = (day == 0).to_numpy(), day.index, set(), 0
    while i < len(idx):
        if zero[i]:
            j = i
            while j < len(idx) and zero[j]:
                j += 1
            if j - i >= 3:
                k, cnt = j, 0
                while k < len(idx) and cnt < 5:
                    if not zero[k]:
                        out.add(idx[k])
                        cnt += 1
                    k += 1
            i = j
        else:
            i += 1
    return out


def prep(d):
    d = d[d["loss"] == "C"].copy()
    d["origin"], d["target_time"] = pd.to_datetime(d["origin"]), pd.to_datetime(d["target_time"])
    d = d[(d["origin"].dt.hour == 8) & (d["origin"].dt.minute == 0)]
    d = d[d["target_time"].dt.normalize() == d["origin"].dt.normalize()]
    tn = d["target_time"].dt.normalize()
    off = (tn.dt.weekday == 6) | tn.isin(HOL)
    d = d[~off & (d["target_time"].dt.hour >= 9) & (d["target_time"].dt.hour < 23)].copy()
    d = d[~d["target_time"].dt.strftime("%m-%d").isin(["07-13", "07-15"])]
    d["s"], d["근처"], d["dday"] = d["yhat"] / d["b_origin"], d["y"] >= 0.9 * d["b_origin"], d["target_time"].dt.normalize()
    return d


def main():
    f = pd.read_csv(P.MODEL_INPUT, encoding="utf-8-sig")
    f["ts"] = pd.to_datetime(f["timestamp"])
    restart = restart_days(f)
    a = pd.read_csv(SRC / "예측_7_8월" / "24h.csv", low_memory=False)
    a = prep(a[(a["model"] == "ENSEMBLE") & a["대표"].notna()])
    b9 = prep(pd.read_csv(SRC / "예측_9월" / "24h.csv", low_memory=False))
    d = pd.concat([a, b9]).sort_values(["dday", "target_time"])
    dd = d.groupby("dday").agg(S=("s", "max"), 위험=("근처", "any"), 달=("period", "first"))
    T = dd.loc[(dd["달"] == "7월") & dd["위험"], "S"].quantile(0.10)
    jul = d[d["period"] == "7월"]
    hours = jul[jul["근처"]].groupby(jul["target_time"].dt.hour).size().sort_values(ascending=False, kind="stable")
    fixed = sorted(hours.index[:3])
    print("기준선 T", round(T, 4), "| 고정 시간대(7월 피크 근접 칸 많은 3개 시)", fixed, "| 7월 시별 칸 수", hours.head(5).to_dict())
    top = d.sort_values(["s", "target_time"], ascending=[False, True], kind="stable").groupby("dday").head(12)
    fix = d[d["target_time"].dt.hour.isin(fixed)]
    dd["알림"] = (dd["S"] >= T) | dd.index.isin(restart)
    rows = []
    for per in ["7월", "8월", "9월"]:
        x = dd[dd["달"] == per]
        alarm_days = x.index[x["알림"]]
        r = {"달": per, "피크근접_날": int(x["위험"].sum()), "알린_날": len(alarm_days)}
        for name, cells in [("상위12칸", top), ("고정시간", fix)]:
            c = cells[cells["dday"].isin(alarm_days)]
            caught = c.groupby("dday")["근처"].any()
            r[f"{name}_잡은날"] = int((caught & x.loc[caught.index, "위험"]).sum())
            r[f"{name}_적중칸"] = int(c["근처"].sum())
            r[f"{name}_경보칸"] = len(c)
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(mid("4.14_고정시간_기준선.csv"), index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
