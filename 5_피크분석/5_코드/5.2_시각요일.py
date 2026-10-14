"""5.2 언제: 피크 근처 칸이 나온 시각(1시간 단위) x 요일 유형 표. 칸 값 = 그 시각에 피크 근처 칸이 나온 날 수
(같은 날 같은 시간에 여러 칸이어도 1).
- 판: 전체 / 별칸(2021) / 별칸_2026
- 덜어낼 시간대 규칙(미리 정함): 별 날 수 합의 절반 이상을 덮는 가장 짧은 연속 시각. 같은 길이면 덮는 양이 큰 쪽, 그래도 같으면 이른 쪽.
- 산출: 6_중간산출물/csv/5.2_시각요일표.csv, 로그
실행: python 5.2_시각요일.py
"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
CH = HERE.parent
FINAL = CH / "3_최종데이터"
MID = CH / "6_중간산출물"


def mid(name):
    sub = {".csv": "csv", ".json": "json", ".txt": "로그", ".png": "그림", ".html": "템플릿"}[Path(name).suffix]
    (MID / sub).mkdir(parents=True, exist_ok=True)
    return MID / sub / name


df = pd.read_csv(FINAL / "5.0_피크준비.csv", encoding="utf-8-sig", parse_dates=["시각"])
df["요일유형"] = "평일"
df.loc[df["요일"] == 5, "요일유형"] = "토요일"
df.loc[(df["요일"] == 6) | (df["공휴일"] == 1), "요일유형"] = "일요일·공휴일"

tabs = []
for name, mask in [("전체", df["피크근처"]), ("별칸2021", df["피크근처"] & df["별칸"]), ("별칸2026", df["피크근처"] & df["별칸_2026"])]:
    p = df[mask].drop_duplicates(["날짜", "시"])
    t = p.groupby(["요일유형", "시"]).size().unstack(fill_value=0).reindex(columns=range(24), fill_value=0)
    t = t.reindex(["평일", "토요일", "일요일·공휴일"]).fillna(0).astype(int)
    t.insert(0, "판", name)
    tabs.append(t.reset_index())
tab = pd.concat(tabs, ignore_index=True)
tab.to_csv(mid("5.2_시각요일표.csv"), index=False, encoding="utf-8-sig")


def window(counts):
    tot = counts.sum()
    best = None
    for L in range(1, 25):
        for s in range(0, 25 - L):
            cov = counts[s:s + L].sum()
            if cov * 2 >= tot:
                if best is None or cov > best[2]:
                    best = (s, s + L - 1, cov)
        if best:
            return best, tot
    return best, tot


lines = ["5.2 시각 x 요일 유형 (값 = 피크 근처 칸이 나온 날 수)", ""]
for name in ["전체", "별칸2021", "별칸2026"]:
    t = tab[tab["판"] == name].set_index("요일유형").drop(columns="판")
    lines.append(f"[{name}]")
    lines.append(t.to_string())
    allc = t.sum(axis=0).values
    (s, e, cov), tot = window(allc)
    lines.append(f"  덜어낼 시간대: {s:02d}:00~{e:02d}:59 ({e-s+1}시간), 덮는 날-시간 {cov}/{tot} = {cov/tot:.0%}")
    lines.append("")
txt = "\n".join(lines)
print(txt)
mid("5.2_시각요일_log.txt").write_text(txt, encoding="utf-8")
