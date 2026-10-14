"""1.4 달력·주야 칸: 요일, 공휴일, 요일 구분(평일/토/일·공휴일), 주야(is_night)를 만든다.

- 요일: 0=월 ~ 6=일.
- 공휴일: 2021 관공서 공휴일 중 데이터 기간(1/1~9/14). 대체공휴일 8/16 포함.
  (요금 약관이 대체공휴일을 공휴일로 보는지는 요금 단계에서 원문 확인)
- daytype: 평일 / 토 / 일공휴. 토요일인 공휴일(2/13)은 일공휴. 숫자·원핫 변환은 1.8에서 파일별로.
- is_night: 18:00~08:59 = 1, 09:00~17:59 = 0. 인건비 칸(1.5/1.0)과 같은 정보.
입력: 6_중간산출물/csv/1.3_기상.csv
출력: 6_중간산출물/csv/1.4_달력.csv, 6_중간산출물/로그/1.4_log.txt
실행: python 1.4_달력주야.py
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


HOLIDAYS_2021 = {
    "2021-01-01": "신정", "2021-02-11": "설 연휴", "2021-02-12": "설날", "2021-02-13": "설 연휴",
    "2021-03-01": "삼일절", "2021-05-05": "어린이날", "2021-05-19": "부처님오신날",
    "2021-06-06": "현충일", "2021-08-15": "광복절", "2021-08-16": "대체공휴일(광복절)",
}

q = pd.read_csv(mid("1.3_기상.csv"), encoding="utf-8-sig", parse_dates=["시각", "시각_시"])
day = q["시각"].dt.normalize()
q["요일"] = q["시각"].dt.dayofweek
q["공휴일"] = day.isin(pd.to_datetime(list(HOLIDAYS_2021))).astype(int)
q["daytype"] = "평일"
q.loc[q["요일"] == 5, "daytype"] = "토"
q.loc[(q["요일"] == 6) | (q["공휴일"] == 1), "daytype"] = "일공휴"
q["is_night"] = (~q["시"].between(9, 17)).astype(int)

assert ((q["is_night"] == 1) == (q["인건비"] == 1.5)).all(), "is_night와 인건비가 다름"
log("is_night = (인건비 1.5) 전 줄 일치: True")

d = q.drop_duplicates("날짜")
hol = d[d["공휴일"] == 1]
log("공휴일", len(hol), "일:", [(str(x.date()), w) for x, w in zip(hol["시각"].dt.normalize(), hol["요일"].map(dict(enumerate("월화수목금토일"))))])
log("daytype 날 수:", d["daytype"].value_counts().to_dict())
log("평일 공휴일(월~금):", int(((hol["요일"] < 5)).sum()), "일")

q.to_csv(mid("1.4_달력.csv"), index=False, encoding="utf-8-sig")
log("저장", q.shape)
(mid("1.4_log.txt")).write_text("\n".join(LOG), encoding="utf-8")
