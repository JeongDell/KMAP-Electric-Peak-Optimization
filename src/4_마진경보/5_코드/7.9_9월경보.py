"""7.9 최종 경보 규칙(위험한 날 + 위험 시간 3시간)을 9월(시험)에 한 번 적용한다.

규칙은 4.오차분석/5_코드/4.10e_위험한날_판별.py와 같다(딥러닝 대표, 손실 C, 24h 모델 08:00 발행 하나).
- 요금 시간대 = 일요일·공휴일이 아닌 날 09:00~22:45 칸. 7/13·7/15 제외(7월 기준선 계산 때).
- S = 그날 요금 시간대 예측 ÷ b_origin 의 최댓값. 기준선 T = 7월 위험한 날(피크 근처 칸이 있는 날) S의 하위 10% 지점.
- 위험한 날 = S ≥ T 또는 재가동 주(생산량 0인 날 3일 이상 연속 뒤, 생산이 있는 첫 5일).
- 위험 시간 = 그날 예측 ÷ b 상위 12칸(3시간). 피크 근처 = 실제 ≥ 0.9 × b_origin.
- 7월로 T를 정한 뒤 8월 결과를 재현하고(4.11과 대조), 9월을 한 번 채점한다.
출력: 6_중간산출물/csv/7.9_9월경보_날별.csv, 로그/7.9_9월경보.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent                                 # 제출 폴더 최상위
import sys; sys.path.insert(0, str(ROOT)); import paths as P   # 경로는 paths.py 한곳에서 정한다
MID = CH / "6_중간산출물"
SRC = P.DL_PRED
LOG = []


def mid(name):
    sub = {".csv": "csv", ".txt": "로그"}[Path(name).suffix]
    (MID / sub).mkdir(parents=True, exist_ok=True)
    return MID / sub / name


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


f = pd.read_csv(P.MODEL_INPUT, encoding="utf-8-sig")
f["ts"] = pd.to_datetime(f["timestamp"])
day = f.groupby(f["ts"].dt.normalize()).agg(day=("day", "first"), prod=("생산량", "sum"))
HOL = set(pd.to_datetime(["2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-03-01", "2021-05-05", "2021-05-19",
                          "2021-06-06", "2021-08-15", "2021-08-16"]))
offday = lambda d: day.loc[d, "day"] == 7 or d in HOL

# 재가동 주: 생산량 0인 날이 3일 이상 이어진 뒤 생산이 있는 첫 5일
restart = set()
zero = (day["prod"] == 0).to_numpy()
idx = day.index
i = 0
while i < len(idx):
    if zero[i]:
        j = i
        while j < len(idx) and zero[j]:
            j += 1
        if j - i >= 3:
            k, cnt = j, 0
            while k < len(idx) and cnt < 5:
                if not zero[k]:
                    restart.add(idx[k])
                    cnt += 1
                k += 1
        i = j
    else:
        i += 1
log("재가동 주 날짜", sorted(d.strftime("%m/%d") for d in restart))


def prep(d):
    d = d[d["loss"] == "C"].copy()
    d["origin"], d["target_time"] = pd.to_datetime(d["origin"]), pd.to_datetime(d["target_time"])
    d = d[(d["origin"].dt.hour == 8) & (d["origin"].dt.minute == 0)]
    d = d[d["target_time"].dt.normalize() == d["origin"].dt.normalize()]
    tn = d["target_time"].dt.normalize()
    d = d[~tn.map(offday) & (d["target_time"].dt.hour >= 9) & (d["target_time"].dt.hour < 23)].copy()
    d = d[~d["target_time"].dt.strftime("%m-%d").isin(["07-13", "07-15"])]
    d["s"] = d["yhat"] / d["b_origin"]
    d["근처"] = d["y"] >= 0.9 * d["b_origin"]
    d["dday"] = d["target_time"].dt.normalize()
    return d


a = pd.read_csv(SRC / "예측_7_8월" / "24h.csv", low_memory=False)
a = prep(a[(a["model"] == "ENSEMBLE") & a["대표"].notna()])
b9 = prep(pd.read_csv(SRC / "예측_9월" / "24h.csv", low_memory=False))
d = pd.concat([a, b9])
dd = d.groupby("dday").agg(S=("s", "max"), 위험=("근처", "any"), 달=("period", "first"))
T = dd.loc[(dd["달"] == "7월") & dd["위험"], "S"].quantile(0.10)
log("기준선 T(7월로만)", round(T, 3))
top = d.sort_values("s", ascending=False).groupby("dday").head(12)
dd["잡음"] = top.groupby("dday")["근처"].any().reindex(dd.index).fillna(False)
dd["재가동"] = dd.index.isin(restart)
dd["알림"] = (dd["S"] >= T) | dd["재가동"]
dd["피크잡음"] = dd["위험"] & dd["알림"] & dd["잡음"]
for per in ["7월", "8월", "9월"]:
    x = dd[dd["달"] == per]
    log(per, "요금 날", len(x), "| 위험한 날(피크 근처 있음)", int(x["위험"].sum()), "| 알린 날", int(x["알림"].sum()),
        "| 그중 재가동", int((x["알림"] & x["재가동"]).sum()), "| 피크 날 중 위험 시간으로 잡은 날", int(x["피크잡음"].sum()))
x9 = dd[dd["달"] == "9월"]
log("9월 날별\n" + x9.assign(S=x9["S"].round(3)).to_string())
# 9월 알린 날 위험 시간 칸 중 실제 피크 근처 비율
t9 = top[top["period"] == "9월"]
t9 = t9[t9["dday"].isin(x9.index[x9["알림"]])]
log("9월 위험 시간 칸 적중률", f"{t9['근처'].mean():.1%}", "(", int(t9["근처"].sum()), "/", len(t9), ")")
dd.to_csv(mid("7.9_9월경보_날별.csv"), encoding="utf-8-sig")
mid("7.9_9월경보.txt").write_text("\n".join(LOG), encoding="utf-8")
