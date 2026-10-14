"""7.8 최종 모델(딥러닝 앙상블)과 기준선 4개를 9월(시험)에 한 번 채점한다.

채점식은 동수 코드(core/train.py risk_loss, core/peak.py weight)와 같다.
  공통 C = 0.5·L_mean + 0.3·L_CVaR + 0.2·L_peak
  L_mean = 블록(1h~24h는 4칸=1시간, 4d는 96칸=1일) RMSE의 평균, L_CVaR = 가장 나쁜 20% 블록 RMSE 평균,
  L_peak = √(모든 칸 평균 of w·max(0, y−ŷ)²), w = 1 + clip((y/b − 0.8)/0.2, 0, 1), b = 발행 시각의 b_origin
먼저 7·8월 파일(기준선 포함)로 동수 표의 7월 값을 재현해 채점식이 같음을 확인한 뒤 9월을 채점한다.
기준선: 직전값(발행 직전 칸), 어제(대상 시각 −1일), 지난주(−7일), 비슷한날(같은 날 종류 4범주의 최근 4일 같은 시각 중앙값, 4주 안).
  비슷한날 정의는 7월 파일과 대조해 추정한 것(표본 97% 일치) — 9월 값은 참고.
입력: 팀원 작업물/보낼것/②예측_7_8월, ③예측_9월_대표_마지막에열기, 최종 15분 파일
출력: 6_중간산출물/csv/7.8_9월채점.csv, 로그/7.8_9월채점.txt
실행: python 7.8_9월채점.py
"""
from pathlib import Path
import math
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


H = {"1h": 4, "4h": 16, "8h": 32, "12h": 48, "24h": 96, "4d": 384}
BLOCK = {"1h": 4, "4h": 4, "8h": 4, "12h": 4, "24h": 4, "4d": 96}


def common_c(y, p, b, block):
    """y, p, b: (N, Hh) 배열. 동수 risk_loss(kind='C', mask=전체)와 같은 계산."""
    N, Hh = y.shape
    e2 = (y - p) ** 2
    R = np.sqrt(e2.reshape(N, Hh // block, block).mean(-1) + 1e-8).ravel()
    l_mean = R.mean()
    k = max(1, math.ceil(0.2 * R.size))
    l_cvar = np.sort(R)[-k:].mean()
    w = 1 + np.clip((y / b - 0.8) / 0.2, 0, 1)
    l_peak = math.sqrt((w * np.maximum(0, y - p) ** 2).mean() + 1e-8)
    return 0.5 * l_mean + 0.3 * l_cvar + 0.2 * l_peak


def to_mat(d, g):
    d = d.sort_values(["origin", "lead"])
    n = d["origin"].nunique()
    shp = (n, H[g])
    return d["y"].to_numpy().reshape(shp), d["yhat"].to_numpy().reshape(shp), d["b_origin"].to_numpy().reshape(shp), d


f = pd.read_csv(P.MODEL_INPUT, encoding="utf-8-sig")
f["ts"] = pd.to_datetime(f["timestamp"])
P = f.set_index("ts")["power_15min"]
day = f.groupby(f["ts"].dt.normalize()).first()
hol = pd.to_datetime(["2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-03-01", "2021-05-05", "2021-05-19",
                      "2021-06-06", "2021-08-15", "2021-08-16"] + [f"2021-08-0{i}" for i in range(2, 7)])
D4 = pd.Series(np.where(day.index.isin(hol), "H", np.where(day["day"] == 6, "S", np.where(day["day"] == 7, "U", "W"))), index=day.index)


def similar(t, o):
    d0 = t.normalize()
    vals = []
    for k in range(1, 29):
        d = d0 - pd.Timedelta(days=k)
        tt = t - pd.Timedelta(days=k)
        if d not in D4.index:
            break
        if D4[d] == D4[d0] and tt < o:
            vals.append(P[tt])
        if len(vals) == 4:
            break
    return np.median(vals) if vals else np.nan


def baselines(d):
    o = pd.to_datetime(d["origin"])
    t = pd.to_datetime(d["target_time"])
    out = {}
    out["기준선 직전값"] = P.reindex(o - pd.Timedelta(minutes=15)).to_numpy()
    out["기준선 어제"] = P.reindex(t - pd.Timedelta(days=1)).to_numpy()
    out["기준선 지난주"] = P.reindex(t - pd.Timedelta(days=7)).to_numpy()
    out["기준선 비슷한날"] = np.array([similar(a, b) for a, b in zip(t, o)])
    return out


rows = []
log("## 1. 7·8월 재현 (파일의 기준선 예측 그대로)")
for g in H:
    a = pd.read_csv(SRC / "예측_7_8월" / f"{g}.csv", low_memory=False)
    for per in ["7월", "8월"]:
        for (m, l), d in a[a["period"] == per].groupby(["model", "loss"]):
            if not (m.startswith("기준선") or m == "ENSEMBLE"):
                continue
            y, p, b, _ = to_mat(d, g)
            c = common_c(y, p, b, BLOCK[g])
            rmse = float(np.sqrt(((y - p) ** 2).mean()))
            rows.append({"구간": g, "달": per, "모델": m, "손실": l, "공통C": round(c, 2), "RMSE": round(rmse, 2), "출처": "파일"})
    t7 = pd.DataFrame([r for r in rows if r["구간"] == g and r["달"] == "7월"])
    log(g, "7월", t7[["모델", "손실", "공통C"]].to_dict("records"))

# 비슷한날 정의 확인: 7월 1h 파일 값과 재구성 값 비교
a = pd.read_csv(SRC / "예측_7_8월" / "1h.csv", low_memory=False)
s = a[(a["model"] == "기준선 비슷한날") & (a["period"] == "7월")]
rec = baselines(s)["기준선 비슷한날"]
log("비슷한날 재구성 일치율(7월 1h)", round(float(np.isclose(rec, s["yhat"]).mean()), 3))
for nm in ["기준선 직전값", "기준선 어제", "기준선 지난주"]:
    s2 = a[(a["model"] == nm) & (a["period"] == "7월")]
    log(nm, "재구성 일치율(7월 1h)", round(float(np.isclose(baselines(s2)[nm], s2["yhat"]).mean()), 3))

log("\n## 2. 9월 시험 (최종 모델 + 재구성 기준선)")
for g in H:
    b9 = pd.read_csv(SRC / "예측_9월" / f"{g}.csv", low_memory=False)
    for l, d in b9.groupby("loss"):
        y, p, b, ds = to_mat(d, g)
        rows.append({"구간": g, "달": "9월", "모델": "ENSEMBLE", "손실": l, "공통C": round(common_c(y, p, b, BLOCK[g]), 2),
                     "RMSE": round(float(np.sqrt(((y - p) ** 2).mean())), 2), "출처": "9월 파일"})
    d = b9[b9["loss"] == "C"].sort_values(["origin", "lead"])
    bl = baselines(d)
    y = d["y"].to_numpy().reshape(-1, H[g])
    b = d["b_origin"].to_numpy().reshape(-1, H[g])
    for nm, v in bl.items():
        p = v.reshape(-1, H[g])
        if np.isnan(p).any():
            log(g, nm, "NaN 칸", int(np.isnan(p).sum()), "→ 그 경로 제외 없이 채점 불가, 건너뜀")
            continue
        rows.append({"구간": g, "달": "9월", "모델": nm, "손실": "-", "공통C": round(common_c(y, p, b, BLOCK[g]), 2),
                     "RMSE": round(float(np.sqrt(((y - p) ** 2).mean())), 2), "출처": "재구성"})
    t9 = pd.DataFrame([r for r in rows if r["구간"] == g and r["달"] == "9월"])
    log(g, "9월", t9[["모델", "손실", "공통C", "RMSE"]].to_dict("records"), "발행 수", d["origin"].nunique())

out = pd.DataFrame(rows)
out.to_csv(mid("7.8_9월채점.csv"), index=False, encoding="utf-8-sig")

log("\n## 3. 요약: 최종 모델 vs 가장 좋은 기준선 (공통 C)")
final = {"1h": "C", "4h": "C", "8h": "C", "12h": "C", "24h": "C", "4d": "A"}
for per in ["7월", "8월", "9월"]:
    for g in H:
        t = out[(out["구간"] == g) & (out["달"] == per)]
        fm = t[(t["모델"] == "ENSEMBLE") & (t["손실"] == final[g])]["공통C"].iloc[0]
        bl = t[t["모델"].str.startswith("기준선")]
        bb = bl.loc[bl["공통C"].idxmin()]
        log(per, g, "최종", fm, "| 최고 기준선", bb["모델"], bb["공통C"], "| 차이", f"{(fm / bb['공통C'] - 1) * 100:+.1f}%")
mid("7.8_9월채점.txt").write_text("\n".join(LOG), encoding="utf-8")
