"""7.19 위험한 날 + 위험 시간 경보를 2026년 요금 체계로 다시 검증한다(사용자 피드백 7.18 28번).

7.9(2021년 체계: 08시 발행, 09~22시대 칸)와 규칙은 같고, 시각만 2026년 체계에 맞춘다.
- 발행 시각 07:00 (08시대가 요금 산정에 들어가므로 그 전에 알림)
- 요금 산정 칸 = 일요일·공휴일이 아닌 날 08:00~21:45 (5.0_피크준비.py의 요금칸_2026과 같은 정의)
- S = 그날 요금 칸 예측 ÷ b 의 최댓값, 기준값 T = 7월 피크 근접 날 S의 하위 10% 지점(7월로만 정함)
- 위험한 날 = S ≥ T 또는 재가동 기간, 위험 시간 = 그날 예측 ÷ b 상위 12칸(3시간), 피크 근접 = 실제 ≥ 0.9 × b
- 7/13·7/15 제외, 딥러닝 앙상블(피크 가중 손실) 24시간 모델
출력: 6_중간산출물/csv/7.19_2026체계경보_날별.csv, 로그/7.19_2026체계경보.txt
"""
from pathlib import Path
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

ISSUE, H0, H1 = 7, 8, 22  # 07시 발행, 08:00~21:45 칸


def prep(d):
    d = d[d["loss"] == "C"].copy()
    d["origin"], d["target_time"] = pd.to_datetime(d["origin"]), pd.to_datetime(d["target_time"])
    d = d[(d["origin"].dt.hour == ISSUE) & (d["origin"].dt.minute == 0)]
    d = d[d["target_time"].dt.normalize() == d["origin"].dt.normalize()]
    tn = d["target_time"].dt.normalize()
    d = d[~tn.map(offday) & (d["target_time"].dt.hour >= H0) & (d["target_time"].dt.hour < H1)].copy()
    d = d[~d["target_time"].dt.strftime("%m-%d").isin(["07-13", "07-15"])]
    d["s"] = d["yhat"] / d["b_origin"]
    d["근처"] = d["y"] >= 0.9 * d["b_origin"]
    d["dday"] = d["target_time"].dt.normalize()
    return d


a = pd.read_csv(SRC / "예측_7_8월" / "24h.csv", low_memory=False)
a = prep(a[(a["model"] == "ENSEMBLE") & a["대표"].notna()])
b9 = prep(pd.read_csv(SRC / "예측_9월" / "24h.csv", low_memory=False))
d = pd.concat([a, b9])
# 예측 경로가 데이터 끝을 넘는 날(그날 칸이 다 차지 않은 날)은 뺀다(7.9와 같은 처리)
full = d.groupby("dday")["target_time"].count()
d = d[d["dday"].map(full) == (H1 - H0) * 4]
dd = d.groupby("dday").agg(S=("s", "max"), 위험=("근처", "any"), 달=("period", "first"))
T = dd.loc[(dd["달"] == "7월") & dd["위험"], "S"].quantile(0.10)
log("2026년 체계(07시 발행, 08:00~21:45) 기준값 T(7월로만)", round(T, 3))
top = d.sort_values("s", ascending=False).groupby("dday").head(12)
dd["잡음"] = top.groupby("dday")["근처"].any().reindex(dd.index).fillna(False)
dd["재가동"] = dd.index.isin(restart)
dd["알림"] = (dd["S"] >= T) | dd["재가동"]
dd["피크잡음"] = dd["위험"] & dd["알림"] & dd["잡음"]
dd["08시대피크"] = d[d["근처"] & (d["target_time"].dt.hour == 8)].groupby("dday").size().reindex(dd.index).fillna(0) > 0
for per in ["7월", "8월", "9월"]:
    x = dd[dd["달"] == per]
    t = top[(top["period"] == per) & top["dday"].isin(x.index[x["알림"]])]
    log(per, "요금 산정일", len(x), "| 피크 근접 날", int(x["위험"].sum()), "| 알린 날", int(x["알림"].sum()),
        "| 위험 시간으로 잡은 날", int(x["피크잡음"].sum()), "| 오경보", int((x["알림"] & ~x["위험"]).sum()),
        "| 미탐지(알리지 않음)", int((x["위험"] & ~x["알림"]).sum()), "| 미탐지(위험 시간 밖)", int((x["위험"] & x["알림"] & ~x["잡음"]).sum()),
        "| 08시대 피크 근접 있는 날", int(x["08시대피크"].sum()),
        f"| 위험 시간 칸 적중 {t['근처'].mean():.1%} ({int(t['근처'].sum())}/{len(t)})")
dd.to_csv(mid("7.19_2026체계경보_날별.csv"), encoding="utf-8-sig")
mid("7.19_2026체계경보.txt").write_text("\n".join(LOG), encoding="utf-8")
