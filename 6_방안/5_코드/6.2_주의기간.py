"""6.2 주의 기간: '조심할 때'(달 · 시간)를 정하면 목표선 P를 넘는 요금 칸을 얼마나 덮는지, 그때 조심해야 할 시간이 얼마나 긴지 센다.
- 피크 정의(b)를 쓰지 않는다. 요금 칸(★ + 봄 달)의 실제 전력이 P를 넘은 칸이 대상이다(5.4와 같은 범위).
- 주의 시간: 2026 규칙 08~11시(A) · 13~15시(B), 2021 규칙 09~11시(A) · 13~15시(B) (5.2 덜어낼 시간대).
- 주의 달: 7월 / 여름 · 겨울 요금달(1 · 2 · 7 · 8월).
- 참고: 8월 휴무 뒤 첫 주(8/9~8/13)가 덮는 칸 수(참고 사례라 규칙 후보에는 넣지 않음).
- 조심할 시간 길이 = 그 달들의 정상 가동 요금일(평일 · 토요일) 수 × 시간 수.
입력: 진행상황/5.피크분석/3_최종데이터/5.0_피크준비.csv
산출: 3_최종데이터/6.2_주의기간.csv, 로그
실행: python 6.2_주의기간.py
"""
from pathlib import Path
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


df = pd.read_csv(SRC5 / "5.0_피크준비.csv", encoding="utf-8-sig", parse_dates=["시각"])
HOURS = {"2026": {"A": range(8, 12), "B": range(13, 16)}, "2021": {"A": range(9, 12), "B": range(13, 16)}}
MONTHS = {"7월": [7], "여름겨울달": [1, 2, 7, 8], "전체": list(range(1, 9))}
restart = df["날짜"].between(20210809, 20210813)
rows = []
for ver, bill_col in [("2026", "요금칸_2026"), ("2021", "요금칸_2021")]:
    bill = df[bill_col]
    opday = df.groupby("날짜")["정상가동일"].transform("first")
    for P in [212, 202, 196]:
        ex = bill & (df["전력"] > P)
        n_ex, d_ex = int(ex.sum()), int(df.loc[ex, "날짜"].nunique())
        for mname, months in MONTHS.items():
            for hname, hours in [("A", list(HOURS[ver]["A"])), ("A+B", list(HOURS[ver]["A"]) + list(HOURS[ver]["B"])), ("하루 종일", list(range(24)))]:
                win = df["월"].isin(months) & df["시"].isin(hours)
                cov = ex & win
                days = df.loc[bill & opday & df["월"].isin(months), "날짜"].nunique()
                rows.append({"판": ver, "P": P, "넘는칸": n_ex, "넘는날": d_ex, "주의달": mname, "주의시간": hname,
                             "덮는칸": int(cov.sum()), "덮는칸비율": round(cov.sum() / n_ex, 3) if n_ex else None,
                             "덮는날": int(df.loc[cov, "날짜"].nunique()),
                             "조심할날": int(days), "조심할시간_h": int(days * len(hours)) if hname != "하루 종일" else None})
        rows.append({"판": ver, "P": P, "넘는칸": n_ex, "넘는날": d_ex, "주의달": "참고: 8/9~8/13", "주의시간": "하루 종일",
                     "덮는칸": int((ex & restart).sum()), "덮는칸비율": round((ex & restart).sum() / n_ex, 3) if n_ex else None,
                     "덮는날": int(df.loc[ex & restart, "날짜"].nunique()), "조심할날": 5, "조심할시간_h": None})
tab = pd.DataFrame(rows)
tab.to_csv(FINAL / "6.2_주의기간.csv", index=False, encoding="utf-8-sig")
lines = ["6.2 주의 기간 (요금 칸 중 P를 넘은 칸 기준, b 미사용, 9월 미사용)", ""]
for ver in ["2026", "2021"]:
    for P in [212, 202, 196]:
        t = tab[(tab["판"] == ver) & (tab["P"] == P)]
        lines.append(f"[{ver} 규칙 · P={P}] 넘는 칸 {t['넘는칸'].iloc[0]} · 넘는 날 {t['넘는날'].iloc[0]}")
        lines.append(t[["주의달", "주의시간", "덮는칸", "덮는칸비율", "덮는날", "조심할날", "조심할시간_h"]].to_string(index=False))
        lines.append("")
txt = "\n".join(lines)
print(txt)
mid("6.2_주의기간_log.txt").write_text(txt, encoding="utf-8")
