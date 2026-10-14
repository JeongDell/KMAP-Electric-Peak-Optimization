"""5.1 사건 목록: 피크 근처 칸(전력 >= 0.9*b)이 하나라도 있는 날을 하루 한 줄로 정리한다.
- 입력: 3_최종데이터/5.0_피크준비.csv (9월 없음)
- 산출: 3_최종데이터/5.1_피크사건.csv, 6_중간산출물/csv/5.1_달별요약.csv, 로그
- 달별 요약은 모든 날 기준과 서로 다른 모양 기준(같은모양묶음당 1)을 나란히 둔다.
실행: python 5.1_사건목록.py
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
pk = df[df["피크근처"]]
rows = []
for d, g in df.groupby("날짜"):
    p = g[g["피크근처"]]
    if p.empty:
        continue
    imax = g["전력"].idxmax()  # 같은 값이면 가장 이른 칸
    rows.append({
        "날짜": d, "월": int(g["월"].iloc[0]), "요일": int(g["요일"].iloc[0]), "공휴일": int(g["공휴일"].iloc[0]),
        "정상가동일": bool(g["정상가동일"].iloc[0]),
        "피크근처_칸수": len(p),
        "최고값": g.loc[imax, "전력"], "최고시각": g.loc[imax, "시각"].strftime("%H:%M"),
        "처음시각": p["시각"].min().strftime("%H:%M"), "마지막시각": p["시각"].max().strftime("%H:%M"),
        "그날_b": p["b"].iloc[0],
        "기록갱신_칸수": int(g["기록갱신"].sum()),
        "별칸_피크수": int((p["별칸"]).sum()),
        "요금칸2021_피크수": int(p["요금칸_2021"].sum()),
        "요금칸2026_피크수": int(p["요금칸_2026"].sum()),
        "별칸2026_피크수": int(p["별칸_2026"].sum()),
        "같은모양묶음": int(g["같은모양묶음"].iloc[0]),
    })
ev = pd.DataFrame(rows)
ev.to_csv(FINAL / "5.1_피크사건.csv", index=False, encoding="utf-8-sig")

days = df.groupby("날짜").first()
summ = []
for m, g in days.groupby("월"):
    op = g[g["정상가동일"]]
    e = ev[ev["월"] == m]
    star = e[e["별칸_피크수"] > 0]
    summ.append({
        "월": m, "전체날": len(g), "정상가동일": len(op),
        "피크근처날": len(e), "피크근처날_정상가동": int(e["정상가동일"].sum()),
        "피크근처날_서로다른모양": e["같은모양묶음"].nunique(),
        "별피크날(2021)": len(star), "별피크날(2026)": int((e["별칸2026_피크수"] > 0).sum()),
        "요금칸피크날(2021)": int((e["요금칸2021_피크수"] > 0).sum()),
        "월최고": g["일최고"].max(),
    })
sm = pd.DataFrame(summ)
sm.to_csv(mid("5.1_달별요약.csv"), index=False, encoding="utf-8-sig")

c129 = df[(df["날짜"] == 20210129) & (df["시각"].dt.strftime("%H:%M") == "08:15")]
lines = [
    "5.1 사건 목록 (9월 미사용)",
    f"피크 근처 날 {len(ev)}일 (정상가동일인 날 {int(ev['정상가동일'].sum())}일, 서로 다른 모양 {ev['같은모양묶음'].nunique()}개)",
    f"정상가동일이 아닌데 피크 근처인 날: {ev.loc[~ev['정상가동일'], ['날짜','최고값','그날_b']].values.tolist()}",
    "", sm.to_string(index=False), "",
    "[확인할 것]",
    f"1/29 목록에 있나: {20210129 in ev['날짜'].values}",
    f"1/29 08:15 전력 {c129['전력'].iloc[0]}, 별칸(2021) {bool(c129['별칸'].iloc[0])}, 별칸_2026 {bool(c129['별칸_2026'].iloc[0])}",
    f"8월 정상가동일 {int(sm.loc[sm['월']==8,'정상가동일'].iloc[0])} (확인값 17)",
    f"피크 근처 날이 없는 달: {[m for m in range(1,9) if m not in ev['월'].values]}",
]
txt = "\n".join(lines)
print(txt)
mid("5.1_사건목록_log.txt").write_text(txt, encoding="utf-8")
