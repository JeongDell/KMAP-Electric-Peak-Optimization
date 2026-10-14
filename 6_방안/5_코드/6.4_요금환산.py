"""6.4 요금 환산: 6.3의 각 조건에 대해 기본요금 감소와 전력량요금 변화를 계산한다.
- 기본요금(원/kW · 월, 고압A): 선택 I 7,220 / II 8,320 / III 9,810 (2021 = 2026, 5번_분석/data/tariff_research.md 1장)
  · 1년 유지: (222 − 옮긴 뒤 ★ 칸 최고) × 단가 × 12
  · 2021년 1~8월 실제였다면: Σ(달별 요금 기준 감소) × 단가 (6.3의 요금기준합_감소_8개월_kW)
- 전력량요금 변화: Σ 옮긴 kWh × (받은 칸 단가 − 보낸 칸 단가), 1~8월 이동분. 단가 · 시간대는 tariff_research.md 2 · 3장 원문 값.
  · 계절: 여름 6~8월, 봄 · 가을 3~5월, 겨울 1 · 2월
  · 토요일: 최대부하 사용량은 중간부하 단가. 2026 봄 · 가을 토요일 11~14시는 전력량요금 50% 할인
  · 받는 칸 · 보내는 칸은 모두 요금 칸(평일 · 토요일 낮)이라 일요일 · 공휴일 규칙은 쓰이지 않는다
- 기후환경요금 · 연료비조정요금은 옮겨도 양이 같아 차이 0이므로 뺀다.
입력: 3_최종데이터/6.3_이동결과.csv, 6_중간산출물/csv/6.3_이동기록.csv, 진행상황/5.피크분석/3_최종데이터/5.0_피크준비.csv
산출: 3_최종데이터/6.4_요금환산.csv, 로그
실행: python 6.4_요금환산.py
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
BASE = {"선택I": 7220, "선택II": 8320, "선택III": 9810}
# 전력량요금 단가(원/kWh): [판][요금제][계절] = (경부하, 중간부하, 최대부하)
ENERGY = {
    "2021": {"선택I": {"여름": (56.6, 109.5, 191.6), "봄가을": (56.6, 79.1, 109.8), "겨울": (63.6, 109.7, 167.2)},
             "선택II": {"여름": (51.1, 104.0, 186.1), "봄가을": (51.1, 73.6, 104.3), "겨울": (58.1, 104.2, 161.7)},
             "선택III": {"여름": (50.2, 103.4, 173.7), "봄가을": (50.2, 72.3, 96.0), "겨울": (57.5, 103.6, 150.5)}},
    "2026": {"선택I": {"여름": (121.5, 169.3, 234.5), "봄가을": (121.5, 138.9, 156.4), "겨울": (128.5, 169.5, 210.1)},
             "선택II": {"여름": (116.0, 163.8, 229.0), "봄가을": (116.0, 133.4, 150.9), "겨울": (123.0, 164.0, 204.6)},
             "선택III": {"여름": (115.1, 163.2, 216.6), "봄가을": (115.1, 132.1, 142.6), "겨울": (122.4, 163.4, 193.4)}},
}


def mid(name):
    sub = {".csv": "csv", ".json": "json", ".txt": "로그", ".png": "그림", ".html": "템플릿"}[Path(name).suffix]
    (MID / sub).mkdir(parents=True, exist_ok=True)
    return MID / sub / name


def season(m):
    return "여름" if m in (6, 7, 8) else ("겨울" if m in (11, 12, 1, 2) else "봄가을")


def zone(ver, m, h, dow):
    """평일 · 토요일 낮 칸의 시간대(0=경, 1=중, 2=최). 토요일은 최대 → 중간."""
    w = season(m) == "겨울"
    if ver == "2021":
        if h >= 23 or h < 9: z = 0
        elif not w: z = 2 if (10 <= h < 12 or 13 <= h < 17) else 1
        else: z = 2 if (10 <= h < 12 or 17 <= h < 20 or 22 <= h < 23) else 1
    else:
        if h >= 22 or h < 8: z = 0
        elif not w: z = 2 if 15 <= h < 21 else 1
        else: z = 2 if (9 <= h < 12 or 16 <= h < 19) else 1
    if dow == 5 and z == 2:
        z = 1
    return z


def price(ver, plan, m, h, dow):
    p = ENERGY[ver][plan][season(m)][zone(ver, m, h, dow)]
    if ver == "2026" and dow == 5 and season(m) == "봄가을" and 11 <= h < 14:
        p *= 0.5
    return p


df = pd.read_csv(SRC5 / "5.0_피크준비.csv", encoding="utf-8-sig", parse_dates=["시각"]).sort_values("시각").reset_index(drop=True)
res = pd.read_csv(FINAL / "6.3_이동결과.csv", encoding="utf-8-sig")
mv = pd.read_csv(MID / "csv" / "6.3_이동기록.csv", encoding="utf-8-sig")
res["판"] = res["판"].astype(str); mv["판"] = mv["판"].astype(str)
for side, col in [("보낸", "보낸칸"), ("받은", "받은칸")]:
    mv[side + "_월"] = df.loc[mv[col], "월"].values
    mv[side + "_시"] = df.loc[mv[col], "시"].values
    mv[side + "_요일"] = df.loc[mv[col], "요일"].values
for plan in BASE:
    mv["차_" + plan] = [
        (price(v, plan, rm, rh, rd) - price(v, plan, sm, sh, sd)) * kw * 0.25
        for v, kw, sm, sh, sd, rm, rh, rd in zip(mv["판"], mv["kW"], mv["보낸_월"], mv["보낸_시"], mv["보낸_요일"], mv["받은_월"], mv["받은_시"], mv["받은_요일"])]
mv["같은시간대"] = [zone(v, sm, sh, sd) == zone(v, rm, rh, rd) for v, sm, sh, sd, rm, rh, rd in
                   zip(mv["판"], mv["보낸_월"], mv["보낸_시"], mv["보낸_요일"], mv["받은_월"], mv["받은_시"], mv["받은_요일"])]
mv["같은시간대_kW"] = mv["kW"] * mv["같은시간대"]
g = mv.groupby(["판", "P", "k", "f"]).agg(**{f"전력량변화_{p}": ("차_" + p, "sum") for p in BASE},
                                         이동건수=("kW", "size"), 같은시간대_건수비율=("같은시간대", "mean"),
                                         같은시간대_kW=("같은시간대_kW", "sum"), 옮긴kW합=("kW", "sum")).reset_index()
g["같은시간대비율"] = g["같은시간대_kW"] / g["옮긴kW합"]  # 옮긴 전력량(kWh) 기준
g = g.drop(columns=["같은시간대_kW", "옮긴kW합"])
out = res.merge(g, on=["판", "P", "k", "f"], how="left")
for plan, unit in BASE.items():
    out[f"기본절감_1년_{plan}"] = ((222 - out["남은최고_별"]).clip(lower=0) * unit * 12).round(0)
    out[f"기본절감_8개월_{plan}"] = (out["요금기준합_감소_8개월_kW"] * unit).round(0)
    out[f"전력량변화_{plan}"] = out[f"전력량변화_{plan}"].round(0)
out["같은시간대비율"] = out["같은시간대비율"].round(3)
out["같은시간대_건수비율"] = out["같은시간대_건수비율"].round(3)
out.to_csv(FINAL / "6.4_요금환산.csv", index=False, encoding="utf-8-sig")

show = ["판", "P", "k", "f", "막은칸비율", "남은최고_별", "옮긴전력량_kWh", "같은시간대비율",
        "기본절감_1년_선택II", "기본절감_8개월_선택II", "전력량변화_선택II"]
lines = ["6.4 요금 환산 (선택 II 표시, I · III은 csv). 전력량변화 양수 = 전력량요금 증가(1~8월 이동분)", ""]
for ver in ["2026", "2021"]:
    lines.append(f"[{ver} 규칙]")
    lines.append(out[out["판"] == ver][show].to_string(index=False))
    lines.append("")
lines.append(f"전력량요금 변화 범위(모든 조건, 선택 II): {out['전력량변화_선택II'].min()} ~ {out['전력량변화_선택II'].max()}원")
for v in ["2026", "2021"]:
    o = out[out["판"] == v]
    lines.append(f"[{v}] 같은 시간대 안에서 옮긴 비율(kWh 기준) {o['같은시간대비율'].min():.3f}~{o['같은시간대비율'].max():.3f}, (건수 기준) {o['같은시간대_건수비율'].min():.3f}~{o['같은시간대_건수비율'].max():.3f}, 전력량 변화 {o['전력량변화_선택II'].min()}~{o['전력량변화_선택II'].max()}원")
txt = "\n".join(lines)
print(txt)
mid("6.4_요금환산_log.txt").write_text(txt, encoding="utf-8")
