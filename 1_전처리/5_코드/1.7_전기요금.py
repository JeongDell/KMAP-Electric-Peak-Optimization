"""1.7 전기요금(계절) 교환: 겨울과 봄·가을 값이 뒤바뀐 것을 바로잡는다.

2021 산업용(을) 고압A 선택 I 최대부하 단가(5번_분석/data/tariff_research.md 2-3, 원문 확인):
  여름(6~8월) 191.6 / 봄·가을(3~5, 9~10월) 109.8 / 겨울(11~2월) 167.2
원본: 1~2월 109.8, 3~5·9월 167.2, 6~8월 191.6 → 1~2월 ↔ 3~5·9월만 맞바꾼다. 6~8월은 그대로.
이 칸은 달마다 값 하나(시간대 구분 없음)라 참고용이다. 예측 입력에서는 뺀다. 시간대별 요금은 요금 단계에서 요금표로 만든다.
입력: 6_중간산출물/csv/1.6_생산량.csv
출력: 6_중간산출물/csv/1.7_요금.csv, 6_중간산출물/로그/1.7_log.txt
실행: python 1.7_전기요금.py
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
KEPCO_2021 = {**{m: 191.6 for m in (6, 7, 8)}, **{m: 109.8 for m in (3, 4, 5, 9, 10)},
              **{m: 167.2 for m in (11, 12, 1, 2)}}


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


q = pd.read_csv(mid("1.6_생산량.csv"), encoding="utf-8-sig", parse_dates=["시각", "시각_시"])
month = q["시각"].dt.month
log("교환 전 월별 값:", q.groupby(month)["전기요금(계절)"].unique().apply(list).to_dict())

swap = {109.8: 167.2, 167.2: 109.8}
q["전기요금_계절"] = q["전기요금(계절)"].map(lambda v: swap.get(v, v))
log("교환 후 월별 값:", q.groupby(month)["전기요금_계절"].unique().apply(list).to_dict())
assert (q["전기요금_계절"] == month.map(KEPCO_2021)).all(), "교환 후 값이 한전 2021 계절 단가와 다름"
log("교환 후 = 한전 2021 선택 I 최대부하 계절 단가: 전 줄 일치")
log("바뀐 15분 줄:", int((q["전기요금(계절)"] != q["전기요금_계절"]).sum()))

q = q.drop(columns=["전기요금(계절)"])
q.to_csv(mid("1.7_요금.csv"), index=False, encoding="utf-8-sig")
log("저장", q.shape)
(mid("1.7_log.txt")).write_text("\n".join(LOG), encoding="utf-8")
