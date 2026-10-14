"""6.3 앞뒤 칸 이동 실험: 목표선 P를 넘은 요금 칸의 초과분을 앞뒤 k칸 안의 빈자리로 옮겼다고 치고, 그날 요금 칸 최고가 P 이하가 되는지 본다.
조건(6.0 설계, 계산 전 고정):
- 판: 2026(기본) · 2021, P: 212 · 202 · 196kW, k: 1 · 2 · 4칸(15 · 30 · 60분), f: 0.1 · 0.2 · 0.3(한 칸에서 덜 수 있는 최대 = 원래 전력 × f)
- 대상: 그 판의 요금 칸(★ + 봄 달) 중 전력 > P인 칸
- 받는 칸: 같은 날, 같은 판의 요금 칸, P까지 남은 만큼만(받는 칸이 P를 넘지 않게). 밤 칸으로 미루는 것은 하지 않는다.
- 순서(탐욕법): 그날 P를 넘은 칸을 원래 전력이 높은 순으로. 각 칸에서 덜 양 = min(전력 − P, 원래 × f).
  가까운 칸부터(거리 1 → k), 같은 거리면 뒤(늦추기) 먼저 채운다. 다 못 옮기면 옮긴 만큼만 덜어낸다.
- 하루 총 전력량은 그대로(옮긴 만큼 받는 칸이 오른다). 옮겨도 다른 칸 모양은 그대로라는 가정.
산출: 3_최종데이터/6.3_이동결과.csv(조건마다 한 줄), 6_중간산출물/csv/6.3_못막은날.csv, 6.3_이동기록.csv(전력량요금 계산용), 6.3_칸결과_<판>_P<P>_k<k>_f<f>.csv는 만들지 않음
실행: python 6.3_앞뒤칸이동.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent                                 # 제출 폴더 최상위
import sys; sys.path.insert(0, str(ROOT)); import paths as P   # 경로는 paths.py 한곳에서 정한다
FINAL = CH / "3_최종데이터"
MID = CH / "6_중간산출물"
SRC5 = P.PEAK_FINAL
P_LIST, K_LIST, F_LIST = [212, 202, 196], [1, 2, 4], [0.1, 0.2, 0.3]
RESTART = set(range(20210809, 20210814))


def mid(name):
    sub = {".csv": "csv", ".json": "json", ".txt": "로그", ".png": "그림", ".html": "템플릿"}[Path(name).suffix]
    (MID / sub).mkdir(parents=True, exist_ok=True)
    return MID / sub / name


df = pd.read_csv(SRC5 / "5.0_피크준비.csv", encoding="utf-8-sig", parse_dates=["시각"])
df = df.sort_values("시각").reset_index(drop=True)
days = {d: g.index.values for d, g in df.groupby("날짜")}


def monthly_basis(power, bill):
    M = power.where(bill).groupby(df["월"]).max().reindex(range(1, 9))
    R, season = {}, -np.inf
    for m in range(1, 9):
        R[m] = max(M[m], season)
        if m in (1, 2, 7, 8):
            season = max(season, M[m])
    return pd.Series(R)


def shift_day(y, bill, P, k, f):
    """y: 원래 전력(96), bill: 요금 칸 여부. 반환: 옮긴 뒤 전력, 이동 기록[(보낸 칸, 받은 칸, kW)]"""
    x = y.copy()
    moves = []
    order = [t for t in np.argsort(-np.nan_to_num(y, nan=-1)) if bill[t] and not np.isnan(y[t]) and y[t] > P]
    for t in order:
        need = x[t] - P
        if need <= 0:
            continue
        plan = min(need, y[t] * f)
        moved = 0.0
        for dist in range(1, k + 1):
            for j in (t + dist, t - dist):
                if moved >= plan - 1e-9:
                    break
                if 0 <= j < len(x) and bill[j] and not np.isnan(x[j]):
                    room = P - x[j]
                    if room > 1e-9:
                        amt = min(room, plan - moved)
                        x[j] += amt
                        moved += amt
                        moves.append((t, j, amt))
        x[t] -= moved
    return x, moves


rows, unsolved, movelog = [], [], []
for ver, bill_col in [("2026", "요금칸_2026"), ("2021", "요금칸_2021")]:
    billall = df[bill_col].values
    R0 = monthly_basis(df["전력"], df[bill_col])
    for P in P_LIST:
        for k in K_LIST:
            for f in F_LIST:
                x_all = df["전력"].values.astype(float).copy()
                need_days = solved_days = need_cells = solved_cells = 0
                moved_kw = 0.0
                for d, idx in days.items():
                    y = df.loc[idx, "전력"].values.astype(float)
                    b = billall[idx]
                    over = b & (np.nan_to_num(y) > P)
                    if not over.any():
                        continue
                    x, moves = shift_day(y, b, P, k, f)
                    x_all[idx] = x
                    need_days += 1
                    need_cells += int(over.sum())
                    solved_cells += int((over & (x <= P + 1e-9)).sum())
                    ok = np.nanmax(np.where(b, x, np.nan)) <= P + 1e-9
                    solved_days += int(ok)
                    moved_kw += sum(m[2] for m in moves)
                    if not ok:
                        unsolved.append({"판": ver, "P": P, "k": k, "f": f, "날짜": d, "월": int(str(d)[4:6]),
                                         "원래최고": float(np.nanmax(np.where(b, y, np.nan))), "옮긴뒤최고": round(float(np.nanmax(np.where(b, x, np.nan))), 2),
                                         "넘은칸": int(over.sum()), "재가동주": d in RESTART})
                    for t, j, a in moves:
                        movelog.append({"판": ver, "P": P, "k": k, "f": f, "보낸칸": int(idx[t]), "받은칸": int(idx[j]), "kW": a})
                xs = pd.Series(x_all)
                star = df[bill_col] & df["여름겨울달"]
                spring = df[bill_col] & df["월"].between(3, 6)
                R1 = monthly_basis(xs, df[bill_col])
                rows.append({"판": ver, "P": P, "k": k, "f": f, "막아야할날": need_days, "막은날": solved_days,
                             "막은날비율": round(solved_days / need_days, 3) if need_days else None,
                             "막아야할칸": need_cells, "막은칸": solved_cells,
                             "막은칸비율": round(solved_cells / need_cells, 3) if need_cells else None,
                             "옮긴전력량_kWh": round(moved_kw * 0.25, 2),
                             "남은최고_별": float(xs[star].max()), "남은최고_봄": float(xs[spring].max()),
                             "요금기준합_감소_8개월_kW": round(float((R0 - R1).sum()), 2)})
res = pd.DataFrame(rows)
res.to_csv(FINAL / "6.3_이동결과.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(unsolved).to_csv(mid("6.3_못막은날.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(movelog).to_csv(mid("6.3_이동기록.csv"), index=False, encoding="utf-8-sig")

# 추천 목표(미리 정한 규칙): 막은 칸 비율 >= 0.9인 가장 낮은 P (판 · k · f마다)
rec = []
for (ver, k, f), g in res.groupby(["판", "k", "f"]):
    ok = g[g["막은칸비율"] >= 0.9]
    rec.append({"판": ver, "k": k, "f": f, "추천P": int(ok["P"].min()) if len(ok) else None})
rec = pd.DataFrame(rec)
rec.to_csv(mid("6.3_추천목표.csv"), index=False, encoding="utf-8-sig")

lines = ["6.3 앞뒤 칸 이동 (탐욕법, 9월 미사용)", ""]
for ver in ["2026", "2021"]:
    t = res[res["판"] == ver]
    lines.append(f"[{ver} 규칙]")
    lines.append(t.drop(columns="판").to_string(index=False))
    lines.append("")
    piv = rec[rec["판"] == ver].pivot(index="k", columns="f", values="추천P")
    lines.append(f"  추천 목표(막은 칸 ≥ 90%인 가장 낮은 P, 행 k · 열 f):\n{piv.to_string()}")
    lines.append("")
u = pd.DataFrame(unsolved)
if len(u):
    lines.append("[못 막은 날 — 2026 · P=202]")
    lines.append(u[(u["판"] == "2026") & (u["P"] == 202)].to_string(index=False))
txt = "\n".join(lines)
print(txt)
mid("6.3_앞뒤칸이동_log.txt").write_text(txt, encoding="utf-8")
