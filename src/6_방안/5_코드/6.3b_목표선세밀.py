"""6.3b 목표선 세밀 탐색 (결과를 본 뒤 보탠 분석, 독립 검증 지적): 6.3과 같은 탐욕법으로 목표선 P를 222~190kW에서 1kW씩 바꿔,
옮길 수 있는 거리 k마다 '옮긴 뒤 ★ 칸 최고(= 새 요금 기준)'가 가장 낮아지는 P를 찾는다. f = 0.1 고정(6.3에서 f는 거의 무관).
주의: 1~8월 데이터에 맞춰 고른 값이라 실제 운영의 목표선으로 그대로 쓰기보다 '어디까지 내려갈 수 있나'의 참고로 쓴다.
입력: 진행상황/5.피크분석/3_최종데이터/5.0_피크준비.csv
산출: 3_최종데이터/6.3b_목표선세밀.csv, 로그
실행: python 6.3b_목표선세밀.py
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent                                 # 제출 폴더 최상위
import sys; sys.path.insert(0, str(ROOT)); import paths as P   # 경로는 paths.py 한곳에서 정한다
FINAL = CH / "3_최종데이터"
MID = CH / "6_중간산출물"
SRC5 = P.PEAK_FINAL
BASE = {"선택I": 7220, "선택II": 8320, "선택III": 9810}


def mid(name):
    sub = {".csv": "csv", ".json": "json", ".txt": "로그", ".png": "그림", ".html": "템플릿"}[Path(name).suffix]
    (MID / sub).mkdir(parents=True, exist_ok=True)
    return MID / sub / name


# 6.3의 shift_day를 그대로 가져온다(같은 규칙)
src = (HERE / "6.3_앞뒤칸이동.py").read_text(encoding="utf-8")
ns = {"np": np}
exec(src[src.index("def shift_day"):src.index("rows, unsolved, movelog")], ns)
shift_day = ns["shift_day"]

df = pd.read_csv(SRC5 / "5.0_피크준비.csv", encoding="utf-8-sig", parse_dates=["시각"]).sort_values("시각").reset_index(drop=True)
days = {d: g.index.values for d, g in df.groupby("날짜")}
rows = []
for ver, bill_col in [("2026", "요금칸_2026"), ("2021", "요금칸_2021")]:
    billall = df[bill_col].values
    star = (df[bill_col] & df["여름겨울달"]).values
    for k in [1, 2, 4]:
        for P in range(222, 189, -1):
            x_all = df["전력"].values.astype(float).copy()
            nd = sd = 0
            for d, idx in days.items():
                y = x_all[idx].copy(); b = billall[idx]
                if not (b & (np.nan_to_num(y) > P)).any():
                    continue
                x, _ = shift_day(y, b, P, k, 0.1)
                x_all[idx] = x
                nd += 1; sd += int(np.nanmax(np.where(b, x, np.nan)) <= P + 1e-9)
            newmax = float(np.nanmax(np.where(star, x_all, np.nan)))
            r = {"판": ver, "k": k, "P": P, "막아야할날": nd, "막은날": sd, "새기준_별최고": newmax}
            for plan, u in BASE.items():
                r[f"1년절감_{plan}"] = round(max(0.0, 222 - newmax) * u * 12)
            rows.append(r)
tab = pd.DataFrame(rows)
tab.to_csv(FINAL / "6.3b_목표선세밀.csv", index=False, encoding="utf-8-sig")
best = tab.sort_values(["판", "k", "새기준_별최고", "P"], ascending=[True, True, True, False]).groupby(["판", "k"]).head(1)
lines = ["6.3b 목표선 1kW 단위 탐색 (f = 0.1, 6.3과 같은 탐욕법, 결과를 본 뒤 보탠 분석)", "",
         "[거리별로 새 기준이 가장 낮은 P]", best.to_string(index=False), ""]
for ver in ["2026", "2021"]:
    for k in [1, 2, 4]:
        t = tab[(tab["판"] == ver) & (tab["k"] == k)]
        lines.append(f"[{ver} · k={k}] P별 새 기준: " + ", ".join(f"{p}→{m:g}" for p, m in zip(t["P"], t["새기준_별최고"])))
txt = "\n".join(lines)
print(txt)
mid("6.3b_목표선세밀_log.txt").write_text(txt, encoding="utf-8")
