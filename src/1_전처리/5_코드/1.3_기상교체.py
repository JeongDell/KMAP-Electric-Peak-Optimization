"""1.3 기상 교체: 기온·풍속·습도·강수량을 기상청 울산(지점 152) 시간자료로 바꾼다.

붙이는 법: 기상청 HH:00 값을 HH시 네 줄(HH:00~HH:45)에 붙인다.
 - 기온·풍속·습도 = HH:00 관측값, 강수량 = (HH-1):00~HH:00 한 시간 동안 내린 양.
 - 둘 다 HH:00 시작 시점에 이미 관측된 값이다.
강수량: 기상청 파일은 이미 1시간 강수량이다(8/21 시간값 합 122.4 = 원본 누적 마지막 값). 차분하지 않는다. 빈칸 = 비 안 옴 → 0.
기존 값과의 관계를 함께 기록한다(같은 시각 / 한 시간 뒤 값과의 일치).
입력: 6_중간산출물/csv/1.2_15분.csv, 기상청 파일 ulsan_20210101-20210914.xls
출력: 6_중간산출물/csv/1.3_기상.csv, 6_중간산출물/csv/1.3_기존날씨비교.csv(날짜별), 6_중간산출물/로그/1.3_log.txt
실행: python 1.3_기상교체.py
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
KMA = P.KMA
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


q = pd.read_csv(mid("1.2_15분.csv"), encoding="utf-8-sig", parse_dates=["시각", "시각_시"])
k = pd.read_csv(KMA, sep="\t", encoding="cp949")      # 확장자는 xls지만 탭 구분 텍스트
k["시각_시"] = pd.to_datetime(k["일시"])
log("기상청", k.shape, "지점", k["지점"].unique().tolist(), k["지점명"].unique().tolist(),
    k["시각_시"].min(), "~", k["시각_시"].max())
full = pd.date_range(k["시각_시"].min(), k["시각_시"].max(), freq="h")
assert len(full.difference(k["시각_시"])) == 0, "기상청 빠진 시각 있음"
log("기상청 결측:", k[["기온(°C)", "풍속(m/s)", "습도(%)", "강수량(mm)"]].isna().sum().to_dict())

# ---------- 강수량 형식 확인: 시간값 하루 합 = 원본 하루 누적 최댓값? ----------
k["날짜"] = k["시각_시"].dt.strftime("%Y%m%d").astype(int)
h = q.drop_duplicates("시각_시")                      # 시간 단위로 비교
day_sum = k.groupby("날짜")["강수량(mm)"].sum()
day_max = h.groupby("날짜")["강수량"].max()
c = pd.concat([day_sum, day_max], axis=1, keys=["기상청_합", "원본_최대"]).dropna()
log(f"강수 하루 합(기상청) = 하루 최대(원본 누적) 일치: {int(np.isclose(c['기상청_합'], c['원본_최대']).sum())}/{len(c)}일",
    "| 8/21:", c.loc[20210821].tolist())

# ---------- 기존 날씨와 비교 (같은 시각 vs 한 시간 뒤) ----------
kk = k.set_index("시각_시")
rows = []
for sh in [0, 1]:
    ref = kk.reindex(h["시각_시"] + pd.Timedelta(hours=sh))
    eq = np.isclose(h["기온"].values, ref["기온(°C)"].values) & np.isclose(h["습도"].values, ref["습도(%)"].values)
    rows.append(pd.Series(eq, index=h["날짜"].values).groupby(level=0).sum().rename(f"일치_{sh}h"))
cmp_ = pd.concat(rows, axis=1)
cmp_["판정"] = np.where(cmp_["일치_0h"] >= cmp_["일치_1h"], "같은 시각", "한 시간 뒤")
cmp_.to_csv(mid("1.3_기존날씨비교.csv"), encoding="utf-8-sig")
log("기존 날씨(기온·습도) 판정 날 수:", cmp_["판정"].value_counts().to_dict())
for lab, g in cmp_.groupby("판정"):
    log(f"  {lab}: {g.index.min()} ~ {g.index.max()}, {len(g)}일, 그 기준 일치 24/24인 날 {int((g.max(axis=1, numeric_only=True) == 24).sum())}")
log("  판정 날 수 최소 일치:", int(cmp_[["일치_0h", "일치_1h"]].max(axis=1).min()), "/24")

# ---------- 교체 ----------
w = k[["시각_시", "기온(°C)", "풍속(m/s)", "습도(%)", "강수량(mm)"]].rename(columns={
    "기온(°C)": "기온", "풍속(m/s)": "풍속", "습도(%)": "습도", "강수량(mm)": "강수량"})
w["강수량"] = w["강수량"].fillna(0.0)
before = q[["기온", "풍속", "습도", "강수량"]].copy()
q = q.drop(columns=["기온", "풍속", "습도", "강수량"]).merge(w, on="시각_시", how="left")
assert len(q) == 24672 and q[["기온", "풍속", "습도", "강수량"]].notna().all().all()
for c_ in ["기온", "풍속", "습도"]:
    diff = ~np.isclose(before[c_].values, q[c_].values, equal_nan=False)
    log(f"{c_}: 값이 바뀐 15분 줄 {int(diff.sum())}/24672 (원본 결측 {int(before[c_].isna().sum())} 포함, 값이 다른 줄 {int((diff & before[c_].notna().values).sum())})")
log("원본 결측(풍속·강수량)은 교체로 사라짐:", before.isna().sum().to_dict(), "→", q[["기온", "풍속", "습도", "강수량"]].isna().sum().to_dict())
log("강수량 > 0 시간 수:", int((w["강수량"] > 0).sum()), "최대", w["강수량"].max())

q.to_csv(mid("1.3_기상.csv"), index=False, encoding="utf-8-sig")
log("저장", q.shape)
(mid("1.3_log.txt")).write_text("\n".join(LOG), encoding="utf-8")
