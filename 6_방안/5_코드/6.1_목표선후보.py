"""6.1 목표선 후보: 5장 5.4 목표선 표에서 후보 P(212 · 202 · 196kW)를 두 요금 규칙(2026 기본, 2021) · 단가 3종으로 정리한다.
- 입력: 진행상황/5.피크분석/3_최종데이터/5.4_목표선표.csv (계산하지 않고 고르기만 한다)
- 산출: 3_최종데이터/6.1_목표선후보.csv, 로그
실행: python 6.1_목표선후보.py
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
P_LIST = [212, 202, 196]


def mid(name):
    sub = {".csv": "csv", ".json": "json", ".txt": "로그", ".png": "그림", ".html": "템플릿"}[Path(name).suffix]
    (MID / sub).mkdir(parents=True, exist_ok=True)
    return MID / sub / name


tl = pd.read_csv(SRC5 / "5.4_목표선표.csv", encoding="utf-8-sig")
tl["판"] = tl["판"].astype(str)
cand = tl[tl["P"].isin(P_LIST)].copy()
cand["판"] = pd.Categorical(cand["판"], ["2026", "2021"], ordered=True)
cand = cand.sort_values(["판", "P"], ascending=[True, False])
cols = ["판", "P", "별_넘는날수", "별_넘는칸수", "별_넘는전력량_kWh", "봄_넘는날수", "봄_넘는칸수", "봄_넘는전력량_kWh",
        "최장연속_칸", "아끼는kW", "절감_1년_선택I", "절감_1년_선택II", "절감_1년_선택III", "절감_8개월_선택II"]
cand[cols].to_csv(FINAL / "6.1_목표선후보.csv", index=False, encoding="utf-8-sig")
txt = "6.1 목표선 후보 (5.4 표에서 고름, 작년 피크 X 없음)\n\n" + cand[cols].to_string(index=False)
print(txt)
mid("6.1_목표선후보_log.txt").write_text(txt, encoding="utf-8")
