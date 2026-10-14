"""5.0 피크분석 공통 준비: 확정 데이터에 5장 정의 칸을 붙여 한 파일로 저장한다.
- 데이터: 1장 확정 전처리_v1_label.csv. 9월은 읽자마자 자른다(9/1 00:00 이전만).
- 정의는 5.0 실행 계획(1_보고서/5.0_피크분석계획.html) 7쪽 그대로.
  b        = max(170, 같은 달 그 칸 직전까지 전력 최고). 달의 첫 칸은 170. NaN 칸은 최고 계산에서 건너뛴다.
  피크근처 = 전력 >= 0.9*b,  기록갱신 = 전력 > b
  정상가동일 = 그날 전력 최고 >= 150
  요금칸_2021 = 공휴일 0, 일요일 아님, 시 9~22 / 요금칸_2026 = 시 8~21
  여름겨울달 = 1,2,7,8월 / 별칸 = 요금칸_2021 & 여름겨울달 / 별칸_2026 = 요금칸_2026 & 여름겨울달
  같은모양묶음 = 하루 96칸 전력이 완전히 같은 날끼리 같은 번호(NaN끼리는 같은 값으로 봄)
실행: python 5.0_피크준비.py
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


def mid(name):
    sub = {".csv": "csv", ".json": "json", ".txt": "로그", ".png": "그림", ".html": "템플릿"}[Path(name).suffix]
    (MID / sub).mkdir(parents=True, exist_ok=True)
    return MID / sub / name


SRC = P.PRE_FINAL / "전처리_v1_label.csv"
cols = ["시각", "날짜", "시", "분", "전력", "생산량", "기온", "요일", "공휴일", "daytype"]
df = pd.read_csv(SRC, encoding="utf-8-sig", usecols=cols)
df["시각"] = pd.to_datetime(df["시각"])
df = df[df["시각"] < "2021-09-01"].sort_values("시각").reset_index(drop=True)
df["월"] = df["시각"].dt.month
df["일"] = df["시각"].dt.day

# b: 같은 달, 그 칸 직전까지의 최고 (NaN 건너뜀)
prev_max = df.groupby("월")["전력"].transform(lambda s: s.cummax().ffill().shift(1))
df["b"] = np.maximum(170.0, prev_max.fillna(0.0))
df["피크근처"] = (df["전력"] >= 0.9 * df["b"]).fillna(False)
df["기록갱신"] = (df["전력"] > df["b"]).fillna(False)

dmax = df.groupby("날짜")["전력"].max()
df["일최고"] = df["날짜"].map(dmax)
df["정상가동일"] = df["일최고"] >= 150

base = (df["공휴일"] == 0) & (df["요일"] != 6)
df["요금칸_2021"] = base & df["시"].between(9, 22)
df["요금칸_2026"] = base & df["시"].between(8, 21)
df["여름겨울달"] = df["월"].isin([1, 2, 7, 8])
df["별칸"] = df["요금칸_2021"] & df["여름겨울달"]
df["별칸_2026"] = df["요금칸_2026"] & df["여름겨울달"]

# 같은모양묶음
shape_id, seen = {}, {}
for d, g in df.groupby("날짜"):
    key = tuple(g["전력"].fillna(-1).tolist())
    if key not in seen:
        seen[key] = len(seen) + 1
    shape_id[d] = seen[key]
df["같은모양묶음"] = df["날짜"].map(shape_id)

out = FINAL / "5.0_피크준비.csv"
FINAL.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False, encoding="utf-8-sig")

day_n = df.groupby("날짜").size()
lines = [
    "5.0 피크준비 (전처리_v1_label.csv, 2021-01-01~2021-08-31, 9월 미사용)",
    f"칸 {len(df)} / 날 {df['날짜'].nunique()} / 하루 96칸 아닌 날 {int((day_n != 96).sum())} / 전력 NaN {int(df['전력'].isna().sum())}칸",
    f"정상가동일 {int(df.groupby('날짜')['정상가동일'].first().sum())}일 / 서로 다른 모양 {len(seen)}개",
    f"달별 정상가동일: {df.groupby(["월", "날짜"])["정상가동일"].first().groupby(level=0).sum().astype(int).to_dict()}",
    f"피크근처 칸 {int(df['피크근처'].sum())} / 기록갱신 칸 {int(df['기록갱신'].sum())} / 별칸 {int(df['별칸'].sum())} / 별칸_2026 {int(df['별칸_2026'].sum())}",
    f"공휴일 날짜: {sorted(df.loc[df['공휴일']==1,'날짜'].unique().tolist())}",
    f"b 범위: {df['b'].min()}~{df['b'].max()}",
]
txt = "\n".join(lines)
print(txt)
mid("5.0_피크준비_log.txt").write_text(txt, encoding="utf-8")
