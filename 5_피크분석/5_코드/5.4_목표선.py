"""5.4 목표선 내려 보기: 요금 기준을 P까지 낮추려면 무엇을 얼마나 막아야 하고, 기본요금이 얼마 주나.
- 판: 2021(요금칸_2021, 별칸) / 2026(요금칸_2026, 별칸_2026). 작년 피크 X 없음("데이터만").
- 그달 요금 기준 R_m = max(그달 요금칸 최고, 앞 달들 중 여름겨울달 요금칸 최고)  (약관 제68조, 1~8월 안에서)
- P: 1~8월 별칸 최고에서 2kW씩 170까지.
- 막을 일: 별칸과 봄 달(3~6월) 요금칸을 따로 센다. kWh = Σ max(0, 전력-P) × 0.25
- 여유부족날수: 별칸에서 P를 넘은 날 중, 같은 날 같은 판 요금칸의 Σ max(0, P-전력)×0.25 < 그날 넘는 kWh 인 날
- 절감_8개월 = Σ_m (R_m - R'_m) × 단가, R' = 모든 요금칸을 min(전력,P)로 바꿔 다시 계산
- 절감_1년 = (현재 1~8월 최고 요금 기준 - P) × 단가 × 12
- 기본요금 단가(원/kW·월): 고압A 선택 I 7,220 / II 8,320 / III 9,810 (2021=2026, tariff_research.md)
산출: 3_최종데이터/5.4_목표선표.csv(두 판), 6_중간산출물/csv/5.4_달별요금기준.csv, 로그
실행: python 5.4_목표선.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CH = HERE.parent
FINAL = CH / "3_최종데이터"
MID = CH / "6_중간산출물"
PRICE = {"선택I": 7220, "선택II": 8320, "선택III": 9810}


def mid(name):
    sub = {".csv": "csv", ".json": "json", ".txt": "로그", ".png": "그림", ".html": "템플릿"}[Path(name).suffix]
    (MID / sub).mkdir(parents=True, exist_ok=True)
    return MID / sub / name


df = pd.read_csv(FINAL / "5.0_피크준비.csv", encoding="utf-8-sig", parse_dates=["시각"])
SPRING = df["월"].between(3, 6)


def monthly_basis(power, billmask):
    M = power.where(billmask).groupby(df["월"]).max().reindex(range(1, 9))
    R, seasonmax = {}, -np.inf
    for m in range(1, 9):
        R[m] = max(M[m], seasonmax)
        if m in (1, 2, 7, 8):
            seasonmax = max(seasonmax, M[m])
    return M, pd.Series(R)


def runs(flags):
    out, cur = [], 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            out.append(cur); cur = 0
    if cur:
        out.append(cur)
    return out


tables, mlines = [], []
for ver, bill, star in [("2021", "요금칸_2021", "별칸"), ("2026", "요금칸_2026", "별칸_2026")]:
    B, S = df[bill], df[star]
    SP = B & SPRING
    M, R = monthly_basis(df["전력"], B)
    mlines.append(pd.DataFrame({"판": ver, "월": range(1, 9), "그달요금칸최고": M.values, "요금기준": R.values}))
    Rmax = R.max()
    top = df.loc[S, "전력"].max()
    rows = []
    for P in np.arange(top, 169.9, -2.0):
        over = df["전력"] - P
        so = S & (over > 0)
        po = SP & (over > 0)
        # 연속 길이(별칸, 날 안에서)
        rl = []
        for d, g in df[S].groupby("날짜"):
            rl += runs((g["전력"] > P).tolist())
        # 여유 부족
        short = 0
        for d in df.loc[so, "날짜"].unique():
            g = df[df["날짜"] == d]
            need = (g.loc[g[star], "전력"] - P).clip(lower=0).sum() * 0.25
            room = (P - g.loc[g[bill], "전력"]).clip(lower=0).sum() * 0.25
            if room < need:
                short += 1
        capped = df["전력"].where(~B, np.minimum(df["전력"], P))
        _, R2 = monthly_basis(capped, B)
        dsum = float((R - R2).sum())
        save_kw = max(0.0, Rmax - P)
        row = {"판": ver, "P": P,
               "별_넘는날수": int(df.loc[so, "날짜"].nunique()), "별_넘는칸수": int(so.sum()),
               "별_넘는전력량_kWh": round(float(over[so].sum() * 0.25), 2),
               "봄_넘는날수": int(df.loc[po, "날짜"].nunique()), "봄_넘는칸수": int(po.sum()),
               "봄_넘는전력량_kWh": round(float(over[po].sum() * 0.25), 2),
               "최장연속_칸": max(rl) if rl else 0,
               "연속중앙값_칸": float(np.median(rl)) if rl else 0,
               "여유부족날수": short, "아끼는kW": save_kw, "요금기준kW합_8개월감소": dsum}
        for k, v in PRICE.items():
            row[f"절감_8개월_{k}"] = int(round(dsum * v))
            row[f"절감_1년_{k}"] = int(round(save_kw * v * 12))
        rows.append(row)
    tables.append(pd.DataFrame(rows))

tab = pd.concat(tables, ignore_index=True)
tab.to_csv(FINAL / "5.4_목표선표.csv", index=False, encoding="utf-8-sig")
mb = pd.concat(mlines, ignore_index=True)
mb.to_csv(mid("5.4_달별요금기준.csv"), index=False, encoding="utf-8-sig")

show = ["P", "별_넘는날수", "별_넘는칸수", "별_넘는전력량_kWh", "봄_넘는날수", "봄_넘는칸수", "봄_넘는전력량_kWh",
        "최장연속_칸", "연속중앙값_칸", "여유부족날수", "아끼는kW", "절감_8개월_선택II", "절감_1년_선택II"]
lines = ["5.4 목표선 (작년 피크 X 없음, 9월 미사용)", "", "[달별 현재 요금 기준]", mb.to_string(index=False), ""]
for ver in ["2021", "2026"]:
    t = tab[tab["판"] == ver]
    lines += [f"[{ver}판] (단가 표시는 선택II 8,320원/kW; 다른 단가는 csv)", t[show].to_string(index=False), ""]
    mono = all((t[c].diff().dropna() >= 0).all() for c in ["별_넘는날수", "별_넘는칸수", "별_넘는전력량_kWh"])
    first = t.iloc[0]
    lines.append(f"  점검: P=최고({first['P']})일 때 넘는 칸 {first['별_넘는칸수']}, 절감 {first['절감_1년_선택II']} / P를 내릴수록 줄어드는 곳 없음 = {mono}")
    lines.append("")
txt = "\n".join(lines)
print(txt)
mid("5.4_목표선_log.txt").write_text(txt, encoding="utf-8")
