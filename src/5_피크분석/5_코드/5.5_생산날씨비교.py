"""5.5 생산 · 날씨 비교: 평일(월~금, 공휴일 아님) 정상 가동일 중 피크 근처 날 vs 아닌 날.
항목: 그날 생산량 합(시간마다 한 번), 피크 무렵 생산량(그날 최고 칸이 속한 시간과 앞뒤 1시간, 3시간 평균 —
      비교 그룹도 각자 그날 최고 칸 기준), 그날 최고기온, 휴일 다음 날(전날이 일요일·공휴일) 비율.
판정(미리 고정): 평균 차 >= 비교 그룹 표준편차의 절반 / 비율 차 >= 10%p, 모든 날·서로 다른 모양 두 기준에서 방향이 같고,
      비교 그룹 5일 이상일 때만 '차이 있음'.
나눔: 전체 / 여름겨울달(1·2·7·8) / 봄 달(3~6)
산출: 3_최종데이터/5.5_비교표.csv, 로그
실행: python 5.5_생산날씨비교.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CH = HERE.parent
FINAL = CH / "3_최종데이터"
MID = CH / "6_중간산출물"


def mid(name):
    sub = {".csv": "csv", ".json": "json", ".txt": "로그", ".png": "그림", ".html": "템플릿"}[Path(name).suffix]
    (MID / sub).mkdir(parents=True, exist_ok=True)
    return MID / sub / name


df = pd.read_csv(FINAL / "5.0_피크준비.csv", encoding="utf-8-sig", parse_dates=["시각"])
df["d"] = df["시각"].dt.normalize()
hours = df[df["분"] == 0]
rows = []
alld = df.groupby("d").first()
offday = set(alld.index[(alld["요일"] == 6) | (alld["공휴일"] == 1)])
for d, g in df.groupby("d"):
    f = g.iloc[0]
    if not (f["정상가동일"] and f["요일"] <= 4 and f["공휴일"] == 0):
        continue
    imax = g["전력"].idxmax()
    h = int(g.loc[imax, "시"])
    hp = hours[hours["d"] == d].set_index("시")["생산량"]
    rows.append({"날짜": int(f["날짜"]), "월": int(f["월"]), "피크근처날": bool(g["피크근처"].any()),
                 "생산량합": float(hp.sum()), "피크무렵생산량": float(hp.reindex([h - 1, h, h + 1]).mean()),
                 "최고기온": float(g["기온"].max()), "휴일다음날": (d - pd.Timedelta(days=1)) in offday,
                 "같은모양묶음": int(f["같은모양묶음"])})
day = pd.DataFrame(rows)

out, lines = [], ["5.5 생산 · 날씨 비교 (평일 정상 가동일, 9월 미사용)", ""]
for part, mask in [("전체", day["월"] > 0), ("여름겨울달", day["월"].isin([1, 2, 7, 8])), ("봄 달", day["월"].between(3, 6))]:
    sub = day[mask]
    for basis, x in [("모든 날", sub), ("서로 다른 모양", sub.drop_duplicates("같은모양묶음"))]:
        a, b = x[x["피크근처날"]], x[~x["피크근처날"]]
        for item in ["생산량합", "피크무렵생산량", "최고기온", "휴일다음날"]:
            ratio = item == "휴일다음날"
            va, vb = a[item].astype(float).mean(), b[item].astype(float).mean()
            sd = b[item].astype(float).std()
            big = (abs(va - vb) >= 0.10) if ratio else (abs(va - vb) >= 0.5 * sd if sd == sd else False)
            out.append({"나눔": part, "기준": basis, "항목": item, "피크근처날_n": len(a), "비교_n": len(b),
                        "피크근처날": round(va, 3), "비교": round(vb, 3), "비교_표준편차": None if ratio else round(sd, 2),
                        "차이": round(va - vb, 3), "문턱넘음": bool(big)})
tab = pd.DataFrame(out)
# 판정
judg = []
for (part, item), g in tab.groupby(["나눔", "항목"], sort=False):
    g = g.set_index("기준")
    ok_n = (g["비교_n"] >= 5).all()
    same = np.sign(g.loc["모든 날", "차이"]) == np.sign(g.loc["서로 다른 모양", "차이"])
    big = g["문턱넘음"].all()
    judg.append({"나눔": part, "항목": item, "판정": "차이 있음" if (ok_n and same and big) else ("판정 안 함(비교 5일 미만)" if not ok_n else "차이 없음")})
jd = pd.DataFrame(judg)
tab = tab.merge(jd, on=["나눔", "항목"])
tab.to_csv(FINAL / "5.5_비교표.csv", index=False, encoding="utf-8-sig")
lines.append(tab.to_string(index=False))
# 보조: 같은 달끼리 비교(달 구성 섞임 점검). 두 그룹이 각각 3일 이상인 달만, 달별 차이의 평균
lines += ["", "[보조] 같은 달 안에서 비교 (피크근처날 평균 - 비교 평균, 두 그룹 각 3일 이상인 달만)"]
aux = []
for m, g in day.groupby("월"):
    a, b = g[g["피크근처날"]], g[~g["피크근처날"]]
    r = {"월": m, "피크근처날_n": len(a), "비교_n": len(b)}
    for item in ["생산량합", "피크무렵생산량", "최고기온", "휴일다음날"]:
        r[item] = round(a[item].astype(float).mean() - b[item].astype(float).mean(), 3) if (len(a) >= 3 and len(b) >= 3) else None
    aux.append(r)
aux = pd.DataFrame(aux)
aux.to_csv(mid("5.5_같은달비교.csv"), index=False, encoding="utf-8-sig")
lines.append(aux.to_string(index=False))
ok = aux.dropna()
lines.append(f"쓸 수 있는 달 {ok['월'].tolist()} / 달별 차이 평균: " + ", ".join(f"{c} {ok[c].mean():.2f}(양수 달 {int((ok[c]>0).sum())}/{len(ok)})" for c in ["생산량합","피크무렵생산량","최고기온","휴일다음날"]))
lines.append("모든 날 기준 달별 피크근처날 / 평일 정상가동일: " + str(day.groupby('월')['피크근처날'].agg(['sum','count']).astype(int).values.tolist()))

# 보조 2 (결과를 본 뒤 추가, 독립 검증 D6): 피크 근처 판정(b)은 매달 다시 시작해 봄 달 피크 근처 날이 달 초에 몰린다.
# 달 초는 기온 · 생산량이 낮아 위 그룹 비교가 섞인다. 그래서 b를 쓰지 않고 '날 최고 kW' 자체를 같은 달 안에서 비교한다.
lines += ["", "[보조 2] b 없이: 같은 달 안에서 생산량(또는 최고기온) 많은 절반 vs 적은 절반의 날 최고 kW 차이 + 스피어만 상관 (평일 정상 가동일, 6일 이상인 달)"]
dm = df.groupby("날짜")["전력"].max()
day["날최고"] = day["날짜"].map(dm)
day["일"] = day["날짜"] % 100
aux2 = []
for m, g in day.groupby("월"):
    if len(g) < 6:
        continue
    r = {"월": m, "날수": len(g)}
    for item in ["생산량합", "최고기온"]:
        med = g[item].median()
        r[item + "_많은절반-적은절반_kW"] = round(g[g[item] > med]["날최고"].mean() - g[g[item] <= med]["날최고"].mean(), 1)
        r[item + "_상관"] = round(g["날최고"].corr(g[item], method="spearman"), 2)
    r["피크근처날_평균일"] = round(g.loc[g["피크근처날"], "일"].mean(), 1) if g["피크근처날"].any() else None
    r["아닌날_평균일"] = round(g.loc[~g["피크근처날"], "일"].mean(), 1) if (~g["피크근처날"]).any() else None
    aux2.append(r)
aux2 = pd.DataFrame(aux2)
aux2.to_csv(mid("5.5_b없이_같은달.csv"), index=False, encoding="utf-8-sig")
lines.append(aux2.to_string(index=False))
for item in ["생산량합", "최고기온"]:
    c = aux2[item + "_많은절반-적은절반_kW"]
    lines.append(f"{item}: 많은 절반의 날 최고가 더 높은 달 {int((c > 0).sum())}/{len(c)}, 차이 범위 {c.min()}~{c.max()}kW, 상관 양수 달 {int((aux2[item + '_상관'] > 0).sum())}/{len(c)}")

# 보조 3 (독립 검증 3차 C2 · '시각 · 달은 kW 자체로도'): 달 평균을 뺀 뒤 모든 달을 합친 상관, b 없는 시각 · 달 근거
from scipy.stats import spearmanr, pearsonr
dd = day.copy()
for col in ["날최고", "생산량합", "최고기온"]:
    dd[col + "_dm"] = dd[col] - dd.groupby("월")[col].transform("mean")
lines += ["", "[보조 3] 달 평균을 뺀 뒤 합친 상관 (평일 정상 가동일)"]
for item in ["생산량합", "최고기온"]:
    r, pv = pearsonr(dd["날최고_dm"], dd[item + "_dm"])
    lines.append(f"  날 최고 vs {item}: r = {r:.2f}, p = {pv:.2f}, n = {len(dd)}")
for col in ["일"]:
    out = [(m, round(g["날최고"].corr(g[col], method="spearman"), 2)) for m, g in day.groupby("월")]
    lines.append(f"  날 최고 vs 날짜(같은 달 스피어만): {out}")
lines.append(f"  달별 평균 최고기온: {day.groupby('월')['최고기온'].mean().round(1).to_dict()} / 달별 날 최고 평균: {day.groupby('월')['날최고'].mean().round(1).to_dict()}")
# b 없는 시각 근거: 평일 정상 가동일의 날 최고 시각, 196kW 이상 칸의 시각
opd = set(day["날짜"])
sub = df[df["날짜"].isin(opd)].copy()
imax = sub.loc[sub.groupby("날짜")["전력"].idxmax()]
lines.append(f"  [b 없는 시각] 평일 정상 가동일 {len(imax)}일의 날 최고 시각(같으면 이른 칸): {imax['시'].value_counts().sort_index().to_dict()}")
hi = sub[sub["전력"] >= 196]
lines.append(f"  [b 없는 시각] 196kW 이상 칸의 시: {hi['시'].value_counts().sort_index().to_dict()}")
allop = df[df["정상가동일"]].groupby("날짜").agg(월=("월", "first"), 최고=("전력", "max"))
lines.append(f"  [b 없는 달] 정상 가동일 중 날 최고 > 202kW인 날 수(모든 요일): {allop[allop['최고'] > 202].groupby('월').size().to_dict()}")
txt = "\n".join(lines)
print(txt)
mid("5.5_생산날씨비교_log.txt").write_text(txt, encoding="utf-8")
