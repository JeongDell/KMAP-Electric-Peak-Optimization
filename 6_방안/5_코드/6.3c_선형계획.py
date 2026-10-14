"""6.3c 선형계획 이동: 6.3(탐욕법)과 같은 조건에서, 넘는 칸만이 아니라 모든 요금 칸이 앞뒤 k칸 안으로 부하를 보낼 수 있게(연쇄 이동) 하고 최적해를 구한다.
조건(6.3과 같음): 같은 날 · 같은 판의 요금 칸끼리만, 거리 ≤ k칸, 한 칸에서 보낼 수 있는 양 ≤ 원래 전력 × f, 하루 총량 그대로, 다른 칸 모양 그대로라는 가정.
6.3과 다른 점: 보내는 칸이 '넘는 칸'으로 한정되지 않는다(옆 칸이 자기 부하를 한 칸 더 옆으로 비켜 줄 수 있다).
풀이(날마다, scipy HiGHS):
  1단계  그날 요금 칸 최고 Z를 최소화 → Z*(그날 낮출 수 있는 한계)
  2단계  목표 P마다: 그날 최고 > P인 날만, 모든 요금 칸 ≤ max(P, Z*)를 지키며 옮기는 양(kW 합)을 최소화
  - 최저 목표: ★ 칸 날들의 Z* 중 최대 = 그 k · f로 1년 기준을 낮출 수 있는 한계
요금: 기본요금 선택 I 7,220(기준, 가정) · II 8,320 · III 9,810원/kW · 월(6.4와 같음). 1년 유지 = (222 − 옮긴 뒤 ★ 최고) × 단가 × 12. 전력량요금 변화는 계산하지 않음(6.4에서 0 근처).
입력: 진행상황/5.피크분석/3_최종데이터/5.0_피크준비.csv, 3_최종데이터/6.3_이동결과.csv(탐욕법 비교)
산출: 3_최종데이터/6.3c_선형계획.csv, 로그
실행: python 6.3c_선형계획.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

HERE = Path(__file__).resolve().parent
CH = HERE.parent
ROOT = CH.parent                                 # 제출 폴더 최상위
import sys; sys.path.insert(0, str(ROOT)); import paths as P   # 경로는 paths.py 한곳에서 정한다
FINAL = CH / "3_최종데이터"
MID = CH / "6_중간산출물"
SRC5 = P.PEAK_FINAL
P_LIST, K_LIST, F_LIST = [212, 202, 196], [1, 2, 4], [0.1, 0.2, 0.3]
BASE, OLD = {"선택I": 7220, "선택II": 8320, "선택III": 9810}, 222


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


def first_year(old, new, bill):
    """작년이 2021년과 같았다고 칠 때(작년 9 · 12월은 새 최고 이하로 가정), 올해 1~8월 요금 기준 감소 합(kW · 월).
    요금 기준(m월) = max(그 달 최고, 올해 m월까지 ★ 달 최고, 작년 m월 이후 ★ 달 최고)."""
    Mo = old.where(bill).groupby(df["월"]).max().reindex(range(1, 9))
    Mn = new.where(bill).groupby(df["월"]).max().reindex(range(1, 9))
    W = (1, 2, 7, 8)
    tot = 0.0
    for m in range(1, 9):
        prior = max([Mo[j] for j in W if j > m], default=-np.inf)
        r_old = max([Mo[m], prior] + [Mo[j] for j in W if j <= m])
        r_new = max([Mn[m], prior] + [Mn[j] for j in W if j <= m])
        tot += r_old - r_new
    return round(float(tot), 1)


def solve_day(y, cells, k, f, cap=None):
    """cells: 요금 칸 위치(그날 안). cap=None → 최고 최소화, cap=값 → 모든 칸 ≤ cap에서 옮긴 양 최소화. 반환 (새 전력, 최고 또는 None)"""
    pairs = [(a, b) for a in range(len(cells)) for b in range(len(cells)) if a != b and abs(cells[a] - cells[b]) <= k]
    n, m = len(pairs), len(cells)
    nz = n + (1 if cap is None else 0)
    yc = y[cells]
    A_ub, b_ub = lil_matrix((2 * m, nz)), []
    for r in range(m):  # 보낸 양 ≤ f · y
        b_ub.append(f * yc[r])
    for i, (a, b) in enumerate(pairs):
        A_ub[a, i] = 1
        A_ub[m + a, i] -= 1   # x_a = y_a − out + in ≤ cap (또는 Z)
        A_ub[m + b, i] += 1
    for r in range(m):
        if cap is None:
            A_ub[m + r, n] = -1
            b_ub.append(-yc[r])
        else:
            b_ub.append(cap - yc[r])
    c = np.zeros(nz)
    if cap is None:
        c[n] = 1
    else:
        c[:n] = 1
    res = linprog(c, A_ub=A_ub.tocsr(), b_ub=b_ub, bounds=[(0, None)] * nz, method="highs")
    if res.status != 0:
        return None, None
    g = res.x[:n]
    x = yc.copy()
    for i, (a, b) in enumerate(pairs):
        x[a] -= g[i]
        x[b] += g[i]
    return x, (res.x[n] if cap is None else None)


greedy = pd.read_csv(FINAL / "6.3_이동결과.csv", encoding="utf-8-sig")
rows = []
for ver, bill_col in [("2026", "요금칸_2026"), ("2021", "요금칸_2021")]:
    bill = df[bill_col].values
    star_d = df.groupby("날짜")["여름겨울달"].first()
    R0 = monthly_basis(df["전력"], df[bill_col])
    for k in K_LIST:
        for f in F_LIST:
            zstar = {}
            for d, idx in days.items():
                b = bill[idx]
                y = df.loc[idx, "전력"].values.astype(float)
                cells = np.where(b & ~np.isnan(y))[0]
                if len(cells) == 0:
                    continue
                _, z = solve_day(y, cells, k, f)
                zstar[d] = (z, idx, cells, y)
            floor = max(v[0] for d, v in zstar.items() if star_d[d])
            for P in P_LIST + ["최저"]:
                target = floor if P == "최저" else P
                x_all = df["전력"].values.astype(float).copy()
                need = solved = 0
                moved = 0.0
                for d, (z, idx, cells, y) in zstar.items():
                    if y[cells].max() <= target + 1e-6:
                        continue
                    need += 1
                    cap = max(target, z) + 1e-6
                    x, _ = solve_day(y, cells, k, f, cap=cap)
                    x_all[idx[cells]] = x
                    solved += int(z <= target + 1e-6)
                    moved += float(np.clip(y[cells] - x, 0, None).sum())
                xs = pd.Series(x_all)
                star = df[bill_col] & df["여름겨울달"]
                left = float(xs[star].max())
                R1 = monthly_basis(xs, df[bill_col])
                gr = greedy[(greedy["판"] == int(ver)) & (greedy["k"] == k) & (greedy["f"] == f) & (greedy["P"] == P)] if P != "최저" else None
                rows.append({"판": ver, "k분": k * 15, "f": f, "목표P": P if P != "최저" else f"최저({floor:.1f})",
                             "막아야할날": need, "막은날": solved, "남은최고_별": round(left, 1),
                             "탐욕법_남은최고_별": float(gr["남은최고_별"].iloc[0]) if gr is not None and len(gr) else None,
                             "줄어든양_kWh": round(moved * 0.25, 1),
                             "요금기준합_감소_8개월_kW": round(float((R0 - R1).sum()), 1),
                             **{f"기본절감_1년_{n}": round((OLD - left) * u * 12) for n, u in BASE.items()},
                             "첫해_1~8월_감소_kW월": first_year(df["전력"], xs, df[bill_col])})
tab = pd.DataFrame(rows)
tab.to_csv(FINAL / "6.3c_선형계획.csv", index=False, encoding="utf-8-sig")
lines = ["6.3c 선형계획 이동 (연쇄 이동 허용, 9월 미사용, 기본요금 선택 I 기준)", ""]
for ver in ["2026", "2021"]:
    lines.append(f"[{ver} 규칙]")
    lines.append(tab[tab["판"] == ver].drop(columns="판").to_string(index=False))
    lines.append("")
txt = "\n".join(lines)
print(txt)
mid("6.3c_선형계획_log.txt").write_text(txt, encoding="utf-8")
