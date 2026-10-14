"""1.5 전력 0 처리: 전력이 0인 15분 칸을 정답에서는 비우고(NaN), 입력용 칸은 과거 값으로만 채운다.

- 전력: 0 → NaN. 채점에서 제외할 정답 칸.
- f_전력0: 원래 0이었던 칸 = 1.
- 전력_채움: 전력과 같되 NaN 칸만 '1주 전 같은 시각' 값으로 채움. 시차 입력을 만들 때 쓴다.
  다음 주 값은 쓰지 않는다(예: 9/4 18:00의 1주 전 = 8/28 18:00에 9/4 값이 섞이면 정답 누수).
입력: 6_중간산출물/csv/1.4_달력.csv
출력: 6_중간산출물/csv/1.5_전력.csv, 6_중간산출물/로그/1.5_log.txt
실행: python 1.5_전력0처리.py
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


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


q = pd.read_csv(mid("1.4_달력.csv"), encoding="utf-8-sig", parse_dates=["시각", "시각_시"])
zero = q["전력"] == 0
q["f_전력0"] = zero.astype(int)
log("전력 0 칸:", int(zero.sum()), "| 0 아닌 칸 최솟값:", int(q.loc[~zero, "전력"].min()))
z = q.loc[zero, "시각"]
runs = (z.diff() != pd.Timedelta(minutes=15)).cumsum()
for _, g in z.groupby(runs.values):
    log(f"  {g.iloc[0]} ~ {g.iloc[-1]} ({len(g)}칸, 시작 시각 기준)")

q["전력"] = q["전력"].astype(float).mask(zero)

s = q.set_index("시각")["전력"]
week_ago = s.reindex(s.index - pd.Timedelta(days=7)).values
fill = pd.Series(week_ago, index=q.index)
q["전력_채움"] = q["전력"].fillna(fill)
assert q["전력_채움"].notna().all(), "1주 전 값도 비어 있는 칸이 있음"
log("전력_채움: 1주 전 같은 시각으로 채운 칸", int(zero.sum()), "| 채운 값 범위",
    q.loc[zero, "전력_채움"].min(), "~", q.loc[zero, "전력_채움"].max())

q.to_csv(mid("1.5_전력.csv"), index=False, encoding="utf-8-sig")
log("저장", q.shape)
(mid("1.5_log.txt")).write_text("\n".join(LOG), encoding="utf-8")
