"""1.6 생산량 두 버전: 7/13·7/15(전력은 가동 수준인데 생산량 0)의 생산량을 두 가지로 처리한다.

- 생산량_v1: 같은 요일 지난주·다음주 같은 시각 생산량의 평균으로 채움 (7/13 ← 7/6·7/20, 7/15 ← 7/8·7/22).
  다음 주 값을 쓰지만 네 날 모두 학습 구간(7/31 이전)이고, 생산량은 정답이 아니라 입력이다.
- 생산량_v2: NaN.
- f_생산량대체: 두 날 = 1 (두 버전 모두 남김).
복제일 여부는 이번 단계에서 고려하지 않는다(사용자 결정, 보류).
입력: 6_중간산출물/csv/1.5_전력.csv
출력: 6_중간산출물/csv/1.6_생산량.csv, 6_중간산출물/로그/1.6_log.txt
실행: python 1.6_생산량.py
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent          # 장 폴더/5_코드
CH = HERE.parent                                 # 장 폴더
ROOT = CH.parent                                 # 제출 폴더 최상위
import sys; sys.path.insert(0, str(ROOT)); import paths as P   # 경로는 paths.py 한곳에서 정한다
MID = CH / "6_중간산출물"                         # 중간 산출물 (csv/json/로그/그림/템플릿)
FINAL = CH / "3_최종데이터"                       # 다음 장에서 읽는 확정 데이터
REPORT = CH / "1_보고서"                          # HTML·PDF 보고서


def mid(name):
    """중간 산출물 경로. 확장자별 하위 폴더로 보낸다."""
    sub = {".csv": "csv", ".json": "json", ".txt": "로그", ".png": "그림", ".html": "템플릿"}[Path(name).suffix]
    (MID / sub).mkdir(parents=True, exist_ok=True)
    return MID / sub / name
LOG = []
BAD_DAYS = [20210713, 20210715]


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


q = pd.read_csv(mid("1.5_전력.csv"), encoding="utf-8-sig", parse_dates=["시각", "시각_시"])
bad = q["날짜"].isin(BAD_DAYS)
q["f_생산량대체"] = bad.astype(int)

d = q.drop_duplicates("날짜").set_index("날짜")
for b in BAD_DAYS:
    g = q[q["날짜"] == b]
    log(f"{b}({'월화수목금토일'[g['요일'].iloc[0]]}): 생산량 합 {g.drop_duplicates('시각_시')['생산량_시간당'].sum()}, "
        f"전력 일평균 {g['전력'].mean():.1f}")

s = q.set_index("시각")["생산량_시간당"]
prev_w = s.reindex(q["시각"] - pd.Timedelta(days=7)).values
next_w = s.reindex(q["시각"] + pd.Timedelta(days=7)).values
fill = (pd.Series(prev_w, index=q.index) + pd.Series(next_w, index=q.index)) / 2

for b in BAD_DAYS:
    for lab, off in [("지난주", -7), ("다음주", 7)]:
        t = pd.Timestamp(str(b)) + pd.Timedelta(days=off)
        g = q[q["시각"].dt.normalize() == t].drop_duplicates("시각_시")
        log(f"  {lab} {t.date()}({'월화수목금토일'[t.dayofweek]}): 생산량 합 {g['생산량_시간당'].sum()}, "
            f"생산 0인 시간 {int((g['생산량_시간당'] == 0).sum())}/24, 전력 일평균 {q.loc[q['시각'].dt.normalize() == t, '전력'].mean():.1f}")

q["생산량_v1"] = q["생산량_시간당"].astype(float).where(~bad, fill)
q["생산량_v2"] = q["생산량_시간당"].astype(float).mask(bad)
assert q.loc[bad, "생산량_v1"].notna().all()
for b in BAD_DAYS:
    g = q[q["날짜"] == b].drop_duplicates("시각_시")
    log(f"{b} v1 채운 생산량 합 {g['생산량_v1'].sum():.1f}, 시간별 범위 {g['생산량_v1'].min():.1f}~{g['생산량_v1'].max():.1f}")
log("v2 NaN 15분 칸:", int(q["생산량_v2"].isna().sum()))

q = q.drop(columns=["생산량_시간당"])
q.to_csv(mid("1.6_생산량.csv"), index=False, encoding="utf-8-sig")
log("저장", q.shape)
(mid("1.6_log.txt")).write_text("\n".join(LOG), encoding="utf-8")
