"""1.8 최종 파일 4개 저장 + 자동 검사.

파일 = 생산량 버전(v1 채움 / v2 NaN) × 요일 구분 표기(label 숫자 / onehot 세 칸)
  - label : daytype = 0(일공휴) / 1(토) / 2(평일). 가동 수준 순서. 나무 모델용.
  - onehot: daytype_평일 / daytype_토 / daytype_일공휴 (해당 칸만 1). 선형·딥러닝용.
복제일은 고려하지 않는다(보류). 스케일링은 3장.
입력: 6_중간산출물/csv/1.7_요금.csv
출력: 3_최종데이터/전처리_v1_label.csv, 전처리_v1_onehot.csv, 전처리_v2_label.csv, 전처리_v2_onehot.csv,
      3_최종데이터/1.8_칸설명.csv, 6_중간산출물/로그/1.8_log.txt
실행: python 1.8_파일저장.py
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
DAYTYPE_CODE = {"일공휴": 0, "토": 1, "평일": 2}


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


q = pd.read_csv(mid("1.7_요금.csv"), encoding="utf-8-sig", parse_dates=["시각", "시각_시"])

# ---------- 자동 검사 ----------
checks = {
    "줄 수 24,672": len(q) == 24672,
    "15분 간격 연속": bool((q["시각"].diff().dropna() == pd.Timedelta(minutes=15)).all()),
    "기간 1/1 00:00 ~ 9/14 23:45": (q["시각"].min(), q["시각"].max()) == (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-09-14 23:45")),
    "전력에 0 없음": bool((q["전력"] != 0).all()),
    "전력 NaN = f_전력0 (74칸)": bool((q["전력"].isna() == (q["f_전력0"] == 1)).all()) and int(q["f_전력0"].sum()) == 74,
    "전력_채움 결측 없음": bool(q["전력_채움"].notna().all()),
    "생산량_v1 결측 없음": bool(q["생산량_v1"].notna().all()),
    "생산량_v2 NaN = f_생산량대체 (192칸)": bool((q["생산량_v2"].isna() == (q["f_생산량대체"] == 1)).all()) and int(q["f_생산량대체"].sum()) == 192,
    "기상 결측 없음": bool(q[["기온", "풍속", "습도", "강수량"]].notna().all().all()),
    "강수량 음수 없음": bool((q["강수량"] >= 0).all()),
    "시간 안 네 줄 생산량·기상 동일": bool((q.groupby("시각_시")[["생산량_v1", "기온", "강수량"]].nunique() <= 1).all().all()),
    "is_night = 인건비 1.5": bool(((q["is_night"] == 1) == (q["인건비"] == 1.5)).all()),
}
for k, v in checks.items():
    log(("통과 " if v else "실패 ") + k)
assert all(checks.values()), "자동 검사 실패"

common = ["시각", "시각_시", "날짜", "시", "분", "전력", "f_전력0", "전력_채움",
          "기온", "풍속", "습도", "강수량", "요일", "공휴일", "is_night", "인건비", "전기요금_계절", "f_시간복원", "f_생산량대체"]

for ver in ["v1", "v2"]:
    base = q[common[:8] + [f"생산량_{ver}"] + common[8:]].rename(columns={f"생산량_{ver}": "생산량"})
    lab = base.copy()
    lab.insert(lab.columns.get_loc("공휴일") + 1, "daytype", q["daytype"].map(DAYTYPE_CODE))
    one = base.copy()
    pos = one.columns.get_loc("공휴일") + 1
    for i, name in enumerate(["평일", "토", "일공휴"]):
        one.insert(pos + i, f"daytype_{name}", (q["daytype"] == name).astype(int))
    assert (one[["daytype_평일", "daytype_토", "daytype_일공휴"]].sum(axis=1) == 1).all()
    for tag, df in [("label", lab), ("onehot", one)]:
        path = FINAL / f"전처리_{ver}_{tag}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        log(f"저장 {path.name} {df.shape} 생산량 NaN {int(df['생산량'].isna().sum())}")

desc = [
    ("시각", "15분 구간 시작 시각 (10:00 = 10:00~10:15)"),
    ("시각_시", "그 줄이 속한 시간의 시작 (10:15 → 10:00)"),
    ("날짜", "YYYYMMDD"), ("시", "0~23"), ("분", "0/15/30/45"),
    ("전력", "15분 최대수요전력(kW). 원래 0인 74칸은 NaN → 정답·채점에서 제외"),
    ("f_전력0", "원래 전력 0이었던 칸 = 1"),
    ("전력_채움", "전력의 NaN만 1주 전 같은 시각 값으로 채움. 시차 입력용(과거 값만)"),
    ("생산량", "시간당 생산량. 같은 시간 네 줄 동일, 4로 나누지 않음. v1: 7/13·7/15를 지난주·다음주 같은 시각 평균으로 채움 / v2: 7/13·7/15 NaN"),
    ("기온·풍속·습도", "기상청 울산(152) HH:00 관측값"),
    ("강수량", "기상청 (HH-1):00~HH:00 1시간 강수량(mm), 빈칸=0"),
    ("요일", "0=월 ~ 6=일"), ("공휴일", "2021 관공서 공휴일(대체공휴일 8/16 포함) = 1"),
    ("daytype (label)", "0=일요일·공휴일, 1=토요일, 2=평일"),
    ("daytype_평일/토/일공휴 (onehot)", "해당 칸만 1"),
    ("is_night", "18:00~08:59 = 1, 09:00~17:59 = 0 (인건비와 같은 정보)"),
    ("인건비", "주간 1.0 / 야간 1.5 (7/13·7/15 복원 시각으로 재계산)"),
    ("전기요금_계절", "2021 선택 I 최대부하 계절 단가(1~2월↔3~5·9월 교환 후). 참고용, 예측 입력 제외"),
    ("f_시간복원", "7/13·7/15 시간 칸을 줄 순서로 복원한 줄 = 1"),
    ("f_생산량대체", "7/13·7/15 생산량 처리 대상 = 1"),
]
pd.DataFrame(desc, columns=["칸", "뜻"]).to_csv(FINAL / "1.8_칸설명.csv", index=False, encoding="utf-8-sig")
(mid("1.8_log.txt")).write_text("\n".join(LOG), encoding="utf-8")
