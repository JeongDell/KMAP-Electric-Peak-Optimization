"""1.1 시간·인건비 복원: 7/13·7/15의 깨진 시간 칸을 줄 순서(0~23시)로 고치고 인건비를 다시 계산한다.

근거: 하루 24줄이 시간순으로 들어 있고(1.0.1), 인건비는 정상 6,120줄에서 시간 규칙(9~17시=1.0)을 100% 따른다(1.0.4).
검증: 복원한 시각의 기온이 기상청 울산(152) 같은 시각 기온과 맞는지 본다.
입력: 원본 zip, 기상청 파일(검증용)
출력: 6_중간산출물/csv/1.1_시간복원.csv (시간 단위 6,168줄), 6_중간산출물/로그/1.1_log.txt
실행: python 1.1_시간복원.py
"""
import io, zipfile
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
ZIP = P.RAW_ZIP
KMA = P.KMA
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


def load_raw():
    z = zipfile.ZipFile(ZIP)
    name = [n for n in z.namelist() if n.endswith(".csv")][0]
    return pd.read_csv(io.BytesIO(z.read(name)), encoding="utf-8-sig")


df = load_raw()
log("원본", df.shape)
assert (df.groupby("날짜").size() == 24).all(), "하루 24줄이 아닌 날이 있음"

# ---------- 시각 복원 ----------
pos = df.groupby("날짜").cumcount()                     # 날짜 안 줄 순서 = 0~23시
bad = df["시간"] != pos
log("시간 칸이 줄 순서와 다른 줄:", int(bad.sum()), "날짜:", sorted(df.loc[bad, "날짜"].unique().tolist()))
df["f_시간복원"] = bad.astype(int)
df["시"] = pos
df["시각"] = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d") + pd.to_timedelta(df["시"], unit="h")

# ---------- 인건비 재계산 ----------
rule = np.where(df["시"].between(9, 17), 1.0, 1.5)
ok = df.loc[~bad, "인건비"].values == rule[~bad.values]
log(f"정상 줄 인건비 규칙 일치: {ok.sum()}/{len(ok)}")
changed = int((df["인건비"].values != rule).sum())
df["인건비"] = rule
log("인건비가 바뀐 줄:", changed)

# ---------- 검증: 기상청 기온과 비교 ----------
k = pd.read_csv(KMA, sep="\t", encoding="cp949")
k["시각"] = pd.to_datetime(k["일시"])
# 5/22 이후 원본 날씨는 기상청 '한 시간 뒤' 값과 같으므로(1.3에서 확인) 같은 시각·한 시간 뒤 둘 다 본다
kt = k.set_index("시각")["기온(°C)"]
g = df.loc[bad, ["날짜", "시각", "기온"]]
for d, gg in g.groupby("날짜"):
    same = np.isclose(gg["기온"].values, kt.reindex(gg["시각"]).values).sum()
    next_h = np.isclose(gg["기온"].values, kt.reindex(gg["시각"] + pd.Timedelta(hours=1)).values).sum()
    log(f"{d} 복원 시각 기온 일치: 기상청 같은 시각 {same}/24, 한 시간 뒤 {next_h}/24")
    assert next_h == 24 or same == 24, "복원 순서가 기상청 흐름과 맞지 않음"

cols = ["시각", "날짜", "시", "15분", "30분", "45분", "60분", "평균", "생산량", "기온", "풍속", "습도", "강수량",
        "전기요금(계절)", "day", "d", "m", "공장인원", "인건비", "f_시간복원"]
df[cols].to_csv(mid("1.1_시간복원.csv"), index=False, encoding="utf-8-sig")
log("저장", mid("1.1_시간복원.csv").relative_to(ROOT), df[cols].shape)
(mid("1.1_log.txt")).write_text("\n".join(LOG), encoding="utf-8")
