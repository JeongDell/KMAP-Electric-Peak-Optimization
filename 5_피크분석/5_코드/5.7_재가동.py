"""5.7 휴무 뒤 재가동한 날 (사용자 제안 10/08).
- 휴무 = 그날 생산량 합이 0인 날이 3일 이상 이어진 묶음(달력 날 기준). 데이터 첫날(1/1)부터 시작한 묶음은 '데이터 시작'으로 표시.
  (처음 정한 '정상 가동일 아닌 날 3일 이상'은 금요일 생산이 있는데 전력만 낮은 2월 주말을 휴무로 잡아 바꿨다[독립 검증 10/08]. 처음 규칙의 결과는 로그 끝에 참고로 남긴다.)
- 재가동 날 = 휴무 뒤 정상 가동일 1~5일째. 다음 휴무가 먼저 시작하면 거기서 끊는다(겹쳐 세지 않음).
- 비교: 같은 달 정상 가동 평일(월~금, 공휴일 아님) 중 재가동 날이 아닌 날.
- 항목: 날 최고(kW), 날 최고 / b(그날 첫 칸 b), 피크 근처 칸 수, 유형(5.3), 아침 상승 칸 시각 · 상승폭(5.3과 같은 정의), 같은모양묶음(복사 여부).
- 함정: b는 매달 다시 시작하므로, 달 초 휴무 뒤에는 b가 낮아 '피크 근처'가 쉽게 된다 -> kW로도 본다.
산출: 3_최종데이터/5.7_재가동.csv, 6_중간산출물/csv/5.7_재가동곡선.csv(그림용 하루 곡선), 로그
실행: python 5.7_재가동.py
"""
from pathlib import Path
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
sh = pd.read_csv(FINAL / "5.3_피크모양.csv", encoding="utf-8-sig")
df["hm"] = df["시각"].dt.hour * 60 + df["시각"].dt.minute
day = df.groupby("날짜").agg(월=("월", "first"), 요일=("요일", "first"), 공휴일=("공휴일", "first"),
                            정상=("정상가동일", "first"), 날최고=("전력", "max"), b첫칸=("b", "first"),
                            피크칸=("피크근처", "sum"), 모양=("같은모양묶음", "first")).reset_index()
day["d"] = pd.to_datetime(day["날짜"].astype(str))
prod = df[df["분"] == 0].groupby("날짜")["생산량"].sum()
day["생산합"] = day["날짜"].map(prod)
old_breaks = []
_c = []
for _, r in day.iterrows():
    if not r["정상"]:
        _c.append(r["d"])
    else:
        if len(_c) >= 3:
            old_breaks.append((_c[0], _c[-1]))
        _c = []

# 휴무 묶음
breaks, cur = [], []
for _, r in day.iterrows():
    if r["생산합"] == 0:
        cur.append(r["d"])
    else:
        if len(cur) >= 3:
            breaks.append((cur[0], cur[-1]))
        cur = []
if len(cur) >= 3:
    breaks.append((cur[0], cur[-1]))


def morning(dn):
    g = df[df["날짜"] == dn].sort_values("시각").reset_index(drop=True)
    dd = g["전력"].diff()
    w = g[(g["hm"] >= 360) & (g["hm"] <= 645)].copy()
    w["up"] = dd.loc[w.index]
    w = w.dropna(subset=["up"])
    i = w["up"].idxmax()
    return f"{int(g.loc[i,'hm'])//60:02d}:{int(g.loc[i,'hm'])%60:02d}", float(w.loc[i, "up"])


rows, curves = [], []
opdays = day[day["정상"] & (day["생산합"] > 0)].reset_index(drop=True)
for k, (s, e) in enumerate(breaks, 1):
    nxt = breaks[k][0] if k < len(breaks) else pd.Timestamp("2100-01-01")
    after = opdays[(opdays["d"] > e) & (opdays["d"] < nxt)].head(5)
    tag = "데이터 시작" if s == day["d"].min() else ""
    for i, (_, r) in enumerate(after.iterrows(), 1):
        mt, mu = morning(r["날짜"])
        typ = sh.loc[sh["날짜"] == r["날짜"], "유형"]
        rows.append({"휴무": k, "휴무시작": s.date(), "휴무끝": e.date(), "휴무일수": (e - s).days + 1, "비고": tag,
                     "재가동일차": i, "날짜": r["날짜"], "요일": r["요일"], "날최고": r["날최고"], "b첫칸": r["b첫칸"],
                     "최고/b": round(r["날최고"] / r["b첫칸"], 3), "피크근처칸": int(r["피크칸"]),
                     "유형": typ.iloc[0] if len(typ) else "피크근처 아님", "아침상승칸": mt, "아침상승폭": mu, "같은모양묶음": r["모양"]})
re_ = pd.DataFrame(rows)
re_.to_csv(FINAL / "5.7_재가동.csv", index=False, encoding="utf-8-sig")

# 비교 그룹: 같은 달 정상 가동 평일, 재가동 날 제외
lines = ["5.7 휴무 뒤 재가동 (9월 미사용)", f"휴무(생산량 0인 날 3일 이상): {[(str(a.date()), str(b.date()), (b-a).days+1) for a,b in breaks]}", ""]
lines.append(re_.to_string(index=False))
lines.append(f"휴무 길이: {[(b-a).days+1 for a,b in breaks]} -> 9일 휴무는 1번(7/31~8/8), 나머지는 모두 3일")
lines.append("")
redays = set(re_["날짜"])
for k, g in re_.groupby("휴무"):
    months = [int(x) for x in g["날짜"].astype(str).str[4:6].astype(int).unique()]
    base = day[(day["월"].isin(months)) & day["정상"] & (day["요일"] <= 4) & (day["공휴일"] == 0) & (~day["날짜"].isin(redays))]
    mts = [morning(x)[1] for x in base["날짜"]]
    lines.append(f"[휴무 {k}: {g['휴무시작'].iloc[0]}~{g['휴무끝'].iloc[0]} {g['휴무일수'].iloc[0]}일 {g['비고'].iloc[0]}] 비교 = {list(months)}월 평소 평일 정상 가동일 {len(base)}일")
    lines.append(f"   재가동 1~5일 날최고 {g['날최고'].tolist()} 평균 {g['날최고'].mean():.1f} / 평소 평균 {base['날최고'].mean():.1f} (중앙값 {base['날최고'].median():.1f}, 범위 {base['날최고'].min()}~{base['날최고'].max()})")
    lines.append(f"   피크근처칸 재가동 {g['피크근처칸'].tolist()} / 평소 평균 {base['피크칸'].mean():.1f}")
    lines.append(f"   b첫칸 재가동 {g['b첫칸'].tolist()} / 평소 b첫칸 평균 {base['b첫칸'].mean():.1f}")
    lines.append(f"   아침상승폭 재가동 {g['아침상승폭'].round(1).tolist()} / 평소 평균 {sum(mts)/len(mts):.1f}")
    lines.append(f"   재가동 날이 다른 날의 복사인가(같은모양묶음 공유 날 수): {[int((day['모양']==m).sum())-1 for m in g['같은모양묶음']]}")
    for _, r in g.iterrows():
        c = df[df["날짜"] == r["날짜"]][["시각", "전력", "b"]].copy()
        c["휴무"] = k; c["구분"] = f"재가동 {r['재가동일차']}일째"
        curves.append(c)
    avg = df[df["날짜"].isin(base["날짜"])].groupby("hm")["전력"].mean().reset_index()
    avg["휴무"] = k; avg["구분"] = "평소 평일 평균"
    curves.append(avg)
pd.concat(curves, ignore_index=True).to_csv(mid("5.7_재가동곡선.csv"), index=False, encoding="utf-8-sig")
lines += ["", f"[참고] 처음 규칙(정상 가동일 아닌 날 3일 이상)의 휴무: {[(str(a.date()), str(b.date()), (b - a).days + 1) for a, b in old_breaks]}",
          "  -> 2월 금~일 3건은 금요일 생산이 있는(전력만 150 미만) 평범한 주말이라 규칙을 바꿨다."]
txt = "\n".join(lines)
print(txt)
mid("5.7_재가동_log.txt").write_text(txt, encoding="utf-8")
