"""1.2 15분 변환: 시간 단위 한 줄(15·30·45·60분 네 칸)을 15분 간격 네 줄로 펼친다.

시각 표기 = 구간 시작. 10시 줄의 15분 칸 → 10:00(10:00~10:15), 30분 → 10:15, 45분 → 10:30, 60분 → 10:45.
같은 시간 안 네 줄은 생산량·기상·인건비·요금 값이 같다(복사). 생산량은 4로 나누지 않는다(시간당 값).
정리하는 칸: 평균(네 칸에서 재계산 가능), day·d·m(날짜에서 재계산), 공장인원(전력이 섞인 값 → 누수).
입력: 6_중간산출물/csv/1.1_시간복원.csv
출력: 6_중간산출물/csv/1.2_15분.csv (24,672줄), 6_중간산출물/로그/1.2_log.txt
실행: python 1.2_15분변환.py
"""
from pathlib import Path
import numpy as np
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


h = pd.read_csv(mid("1.1_시간복원.csv"), encoding="utf-8-sig", parse_dates=["시각"])
log("입력", h.shape)

# 평균 칸 규칙 확인 후 정리 (평균 = floor(네 칸 평균 + 0.5), 1.0.1)
Q = h[["15분", "30분", "45분", "60분"]]
log("평균 = floor(네 칸 평균+0.5) 일치:", bool((h["평균"] == np.floor(Q.mean(axis=1) + 0.5)).all()))

keep = ["시각", "날짜", "시", "생산량", "기온", "풍속", "습도", "강수량", "전기요금(계절)", "인건비", "f_시간복원"]
parts = []
for i, c in enumerate(["15분", "30분", "45분", "60분"]):
    p = h[keep].copy()
    p["시각_시"] = p["시각"]
    p["시각"] = p["시각"] + pd.Timedelta(minutes=15 * i)
    p["분"] = 15 * i
    p["전력"] = h[c].values
    parts.append(p)
q = pd.concat(parts).sort_values("시각").reset_index(drop=True)
q = q.rename(columns={"생산량": "생산량_시간당"})
q = q[["시각", "시각_시", "날짜", "시", "분", "전력", "생산량_시간당", "기온", "풍속", "습도", "강수량",
       "전기요금(계절)", "인건비", "f_시간복원"]]

# 검사: 줄 수, 15분 간격 연속, 네 칸 합 보존
assert len(q) == len(h) * 4
assert (q["시각"].diff().dropna() == pd.Timedelta(minutes=15)).all(), "15분 간격이 끊김"
assert q["전력"].sum() == Q.values.sum()
log("출력", q.shape, "시작", q["시각"].min(), "끝", q["시각"].max(), "| 15분 간격 연속, 전력 합 보존")

q.to_csv(mid("1.2_15분.csv"), index=False, encoding="utf-8-sig")
(mid("1.2_log.txt")).write_text("\n".join(LOG), encoding="utf-8")
