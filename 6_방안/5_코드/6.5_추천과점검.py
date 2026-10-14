"""6.5 추천과 점검:
(1) 점검: 6.3 이동이 하루 총 전력량을 바꾸지 않았는지, 받는 칸이 P를 넘지 않았는지, 이동 거리가 k 이하인지 이동 기록으로 확인한다.
(2) 추천: 미리 정한 규칙(막은 칸 비율 ≥ 90%인 가장 낮은 P)과, 돈 기준(1년 유지 기본요금 절감이 가장 큰 P)을 나란히 보인다.
    돈 기준은 결과를 본 뒤 보탠 것이다. 기본요금은 1년 중 가장 높은 칸 하나로 정해지므로, 칸 비율이 높아도 못 막은 날이 있으면 그날 값이 기준이 된다.
(3) 그림용: 예시 날(7/19, 7/28)의 옮기기 전 · 후 곡선(2026 규칙, P=202, k=2 · 4, f=0.1).
입력: 3_최종데이터/6.3_이동결과.csv · 6.4_요금환산.csv, 6_중간산출물/csv/6.3_이동기록.csv, 5장 5.0_피크준비.csv
산출: 3_최종데이터/6.5_추천.csv, 6_중간산출물/csv/6.5_예시날.csv, 로그
실행: python 6.5_추천과점검.py
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


def mid(name):
    sub = {".csv": "csv", ".json": "json", ".txt": "로그", ".png": "그림", ".html": "템플릿"}[Path(name).suffix]
    (MID / sub).mkdir(parents=True, exist_ok=True)
    return MID / sub / name


df = pd.read_csv(SRC5 / "5.0_피크준비.csv", encoding="utf-8-sig", parse_dates=["시각"]).sort_values("시각").reset_index(drop=True)
fee = pd.read_csv(FINAL / "6.4_요금환산.csv", encoding="utf-8-sig"); fee["판"] = fee["판"].astype(str)
mv = pd.read_csv(MID / "csv" / "6.3_이동기록.csv", encoding="utf-8-sig"); mv["판"] = mv["판"].astype(str)
lines = ["6.5 추천과 점검", ""]

# (1) 점검
bad = []
for (ver, P, k, f), g in mv.groupby(["판", "P", "k", "f"]):
    x = df["전력"].values.astype(float).copy()
    np.add.at(x, g["받은칸"].values, g["kW"].values)
    np.add.at(x, g["보낸칸"].values, -g["kW"].values)
    dist = (g["받은칸"] - g["보낸칸"]).abs().max()
    recv_max = pd.Series(x)[g["받은칸"].unique()].max()
    same_day = (df.loc[g["받은칸"], "날짜"].values == df.loc[g["보낸칸"], "날짜"].values).all()
    day_diff = (pd.Series(np.nan_to_num(x)).groupby(df["날짜"]).sum() - pd.Series(np.nan_to_num(df["전력"].values)).groupby(df["날짜"]).sum()).abs().max()
    if dist > k or recv_max > P + 1e-6 or not same_day or day_diff > 1e-6:
        bad.append((ver, P, k, f, dist, recv_max, same_day, day_diff))
lines.append(f"[점검] 조건 {mv.groupby(['판','P','k','f']).ngroups}개 중 위반 {len(bad)}개 (하루 총량 보존 · 받는 칸 ≤ P · 거리 ≤ k · 같은 날) {bad[:3]}")

# (2) 추천
rows = []
for (ver, k, f), g in fee.groupby(["판", "k", "f"]):
    ok = g[g["막은칸비율"] >= 0.9]
    rule = int(ok["P"].min()) if len(ok) else None
    best = g.sort_values(["기본절감_1년_선택II", "P"], ascending=[False, False]).iloc[0]
    rows.append({"판": ver, "k": k, "f": f, "규칙추천P": rule,
                 "규칙추천_남은최고": float(g.loc[g["P"] == rule, "남은최고_별"].iloc[0]) if rule else None,
                 "규칙추천_1년절감_II": float(g.loc[g["P"] == rule, "기본절감_1년_선택II"].iloc[0]) if rule else None,
                 "돈기준P": int(best["P"]), "돈기준_남은최고": float(best["남은최고_별"]),
                 "돈기준_1년절감_I": float(best["기본절감_1년_선택I"]), "돈기준_1년절감_II": float(best["기본절감_1년_선택II"]),
                 "돈기준_1년절감_III": float(best["기본절감_1년_선택III"]), "돈기준_전력량변화_II": float(best["전력량변화_선택II"]),
                 "돈기준_막은날": f"{int(best['막은날'])}/{int(best['막아야할날'])}"})
rec = pd.DataFrame(rows)
rec.to_csv(FINAL / "6.5_추천.csv", index=False, encoding="utf-8-sig")
lines += ["", "[추천] 행 = 판 · k · f", rec.to_string(index=False)]

# (3) 예시 날
ex = []
for d in [20210719, 20210728]:
    idx = df.index[df["날짜"] == d].values
    for k in [2, 4]:
        g = mv[(mv["판"] == "2026") & (mv["P"] == 202) & (mv["k"] == k) & (mv["f"] == 0.1)]
        x = df["전력"].values.astype(float).copy()
        np.add.at(x, g["받은칸"].values, g["kW"].values)
        np.add.at(x, g["보낸칸"].values, -g["kW"].values)
        for i in idx:
            ex.append({"날짜": d, "k": k, "칸": int(i - idx[0]), "원래": df.loc[i, "전력"], "옮긴뒤": round(x[i], 2), "요금칸": bool(df.loc[i, "요금칸_2026"])})
pd.DataFrame(ex).to_csv(mid("6.5_예시날.csv"), index=False, encoding="utf-8-sig")
e = pd.DataFrame(ex)
for (d, k), g in e.groupby(["날짜", "k"]):
    lines.append(f"[예시] {d} k={k}: 원래 요금칸 최고 {g.loc[g['요금칸'],'원래'].max()} → 옮긴 뒤 {g.loc[g['요금칸'],'옮긴뒤'].max()}, 하루 합 {g['원래'].sum():.1f} → {g['옮긴뒤'].sum():.1f}")
txt = "\n".join(lines)
print(txt)
mid("6.5_추천과점검_log.txt").write_text(txt, encoding="utf-8")
