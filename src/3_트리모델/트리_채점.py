"""트리 앙상블(RF·XGBoost·LightGBM·CatBoost 단순 평균, 피크 가중 손실 C)의 저장 예측을 7·8월 종합 오차 지표 C로 채점한다.

학습은 하지 않는다. 채점식은 4_마진경보/5_코드/7.8_9월채점.py의 common_c와 같다(딥러닝 risk_loss kind='C'와 같은 계산).
입력: 3_트리모델/저장예측/(구간).csv
출력: 3_트리모델/트리_채점.csv
실행: python 트리_채점.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths as P  # noqa: E402

H = {"1h": 4, "4h": 16, "8h": 32, "12h": 48, "24h": 96, "4d": 384}
BLOCK = {"1h": 4, "4h": 4, "8h": 4, "12h": 4, "24h": 4, "4d": 96}


def common_c(y, p, b, block):
    """y, p, b: (N, Hh) 배열. 7.8_9월채점.py와 같은 계산."""
    N, Hh = y.shape
    e2 = (y - p) ** 2
    R = np.sqrt(e2.reshape(N, Hh // block, block).mean(-1) + 1e-8).ravel()
    l_mean = R.mean()
    k = max(1, math.ceil(0.2 * R.size))
    l_cvar = np.sort(R)[-k:].mean()
    w = 1 + np.clip((y / b - 0.8) / 0.2, 0, 1)
    l_peak = math.sqrt((w * np.maximum(0, y - p) ** 2).mean() + 1e-8)
    return 0.5 * l_mean + 0.3 * l_cvar + 0.2 * l_peak


rows = []
for g in H:
    a = pd.read_csv(P.TREE_PRED / f"{g}.csv")
    for per, d in a.groupby("period"):
        d = d.sort_values(["origin", "lead"])
        shp = (d["origin"].nunique(), H[g])
        y, p, b = (d[c].to_numpy(float).reshape(shp) for c in ["y", "yhat", "b_origin"])
        rows.append({"구간": g, "달": per, "모델": "트리 앙상블", "손실": "C",
                     "공통C": round(common_c(y, p, b, BLOCK[g]), 2), "RMSE": round(float(np.sqrt(((y - p) ** 2).mean())), 2)})
out = pd.DataFrame(rows)
out.to_csv(HERE / "트리_채점.csv", index=False, encoding="utf-8-sig")
print(out.pivot(index="구간", columns="달", values="공통C").reindex(list(H)).to_string())
