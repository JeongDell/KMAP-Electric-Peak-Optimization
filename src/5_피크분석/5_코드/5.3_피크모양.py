"""5.3 피크 모양: 피크 근처 날마다 가장 긴 연속 길이 · 튄 높이를 재고, 1~6월로 정한 경계로 유형을 나눈다.
경계 방법(미리 고정): 1~6월 피크 근처 날(같은모양묶음당 하루) 의 '가장 긴 연속 길이' 막대(폭 1칸).
  봉우리 = 양옆 막대보다 높은 막대(끝 막대는 바깥을 0으로 봄). 봉우리 2개 이상 -> 가장 높은 두 봉우리 사이 가장 낮은 막대
  (같으면 가장 짧은 길이). 1개 -> 중앙값. 경계 이하 = 한 번 튀는 형, 초과 = 오래 높은 형.
확인할 사실(최고 칸 = 그날 전력 최고, 같으면 가장 이른 칸):
  아침 상승 칸 = 06:00~10:45 칸 중 직전 칸 대비 상승이 가장 큰 칸(같으면 이른 칸). 최고 칸이 [상승 칸, 상승 칸+3칸] 안이면 '아침직후'
  점심 상승 칸 = 12:00~14:45 같은 방식 -> '점심직후'
  생산 시작 = 06:00~10:45 사이 생산량이 직전 시간 0에서 양수로 바뀐 첫 시간 H. 최고 칸이 [H:00, H+1:00) 안이면 '생산직후', 없으면 '해당없음'
산출: 3_최종데이터/5.3_피크모양.csv, 6_중간산출물/csv/5.3_연속길이분포.csv, 로그
실행: python 5.3_피크모양.py
"""
from pathlib import Path
import numpy as np
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
ev = pd.read_csv(FINAL / "5.1_피크사건.csv", encoding="utf-8-sig")
df["hm"] = df["시각"].dt.hour * 60 + df["시각"].dt.minute


def longest_run(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def rise_cell(g, lo, hi):
    d = g["전력"].diff()
    w = g[(g["hm"] >= lo) & (g["hm"] <= hi)].copy()
    w["상승"] = d.loc[w.index]
    w = w.dropna(subset=["상승"])
    if w.empty:
        return None, None
    i = w["상승"].idxmax()
    return int(g.loc[i, "hm"]), float(w.loc[i, "상승"])


rows = []
for _, e in ev.iterrows():
    g = df[df["날짜"] == e["날짜"]].sort_values("시각").reset_index(drop=True)
    imax = g["전력"].idxmax()
    mx_hm = int(g.loc[imax, "hm"])
    hr = g[g["시"] == g.loc[imax, "시"]]["전력"]
    morning, mrise = rise_cell(g, 6 * 60, 10 * 60 + 45)
    lunch, lrise = rise_cell(g, 12 * 60, 14 * 60 + 45)
    hp = g[g["분"] == 0][["시", "생산량"]].set_index("시")["생산량"]
    start = None
    for h in range(6, 11):
        if h in hp.index and h - 1 in hp.index and hp[h - 1] == 0 and hp[h] > 0:
            start = h
            break
    rows.append({
        "날짜": e["날짜"], "월": e["월"],
        "최고칸": f"{mx_hm//60:02d}:{mx_hm%60:02d}", "최고값": g.loc[imax, "전력"],
        "가장긴연속_칸": longest_run(g["피크근처"].tolist()),
        "튄높이_kW": round(hr.max() - hr.mean(), 2),
        "아침상승칸": f"{morning//60:02d}:{morning%60:02d}", "아침상승폭": mrise,
        "아침직후": morning <= mx_hm <= morning + 45,
        "점심상승칸": f"{lunch//60:02d}:{lunch%60:02d}", "점심상승폭": lrise,
        "점심직후": lunch <= mx_hm <= lunch + 45,
        "생산시작시": start if start is not None else "",
        "생산직후": ("해당없음" if start is None else bool(start * 60 <= mx_hm < start * 60 + 60)),
        "별피크날": e["별칸_피크수"] > 0, "같은모양묶음": e["같은모양묶음"],
    })
sh = pd.DataFrame(rows)

# 경계: 1~6월만, 같은모양묶음당 하루
train = sh[sh["월"] <= 6].drop_duplicates("같은모양묶음")
L = train["가장긴연속_칸"]
cnt = L.value_counts().reindex(range(1, L.max() + 1), fill_value=0)
peaks = [k for k in cnt.index if cnt[k] > cnt.get(k - 1, 0) and cnt[k] > cnt.get(k + 1, 0)]
if len(peaks) >= 2:
    top2 = sorted(sorted(peaks, key=lambda k: (-cnt[k], k))[:2])
    seg = cnt.loc[top2[0]:top2[1]]
    bound = int(seg[seg == seg.min()].index.min())
    how = f"봉우리 {peaks} 중 가장 높은 두 봉우리 {top2} 사이 최저 막대"
else:
    bound = float(L.median())
    how = f"봉우리 {peaks} 하나 이하 -> 중앙값"
cnt.rename("날수").to_frame().to_csv(mid("5.3_연속길이분포.csv"), encoding="utf-8-sig")
sh["유형"] = np.where(sh["가장긴연속_칸"] <= bound, "한 번 튀는 형", "오래 높은 형")
sh.to_csv(FINAL / "5.3_피크모양.csv", index=False, encoding="utf-8-sig")


def summary(x, label):
    t = x[x["유형"] == "한 번 튀는 형"]
    out = [f"[{label}] 날 {len(x)} (서로 다른 모양 {x['같은모양묶음'].nunique()}) / 오래 높은 형 {int((x['유형']=='오래 높은 형').sum())} / 한 번 튀는 형 {len(t)}"]
    for col in ["아침직후", "점심직후"]:
        out.append(f"   {col}: 전체 {int(x[col].sum())}/{len(x)}, 튀는 형 {int(t[col].sum())}/{len(t)}")
    pv = x["생산직후"].astype(str).value_counts().to_dict()
    out.append(f"   생산직후: {pv} / 튀는 형 {t['생산직후'].astype(str).value_counts().to_dict()}")
    out.append(f"   최고칸 상위: {x['최고칸'].value_counts().head(6).to_dict()}")
    out.append(f"   아침상승칸 상위: {x['아침상승칸'].value_counts().head(4).to_dict()} / 점심상승칸 상위: {x['점심상승칸'].value_counts().head(3).to_dict()}")
    return out


lines = [
    "5.3 피크 모양 (9월 미사용)",
    f"[경계 정하기 — 1~6월, 같은모양묶음당 하루 {len(train)}일] 연속 길이 막대: {cnt.to_dict()}",
    f"  방법: {how} -> 경계 {bound}칸 ({bound*15}분). 이하 = 한 번 튀는 형",
    "",
]
lines += summary(sh[sh["월"] <= 6], "1~6월(경계 정한 기간, 모든 날)")
lines += summary(sh[sh["월"] <= 6].drop_duplicates("같은모양묶음"), "1~6월(서로 다른 모양 기준)")
lines += summary(sh[sh["월"] >= 7], "7~8월(적용)")
lines += summary(sh[sh["월"] >= 7].drop_duplicates("같은모양묶음"), "7~8월(서로 다른 모양 기준)")
lines += summary(sh[sh["별피크날"]], "★ 피크날(2021)")
lines += summary(sh[sh["별피크날"]].drop_duplicates("같은모양묶음"), "★ 피크날(2021, 서로 다른 모양)")
lines.append(f"튄 높이 kW: 오래 높은 형 중앙값 {sh.loc[sh['유형']=='오래 높은 형','튄높이_kW'].median()}, 튀는 형 중앙값 {sh.loc[sh['유형']=='한 번 튀는 형','튄높이_kW'].median()}")
# 민감도: 끝 막대를 봉우리로 치지 않는 해석(양옆이 모두 있는 막대만 봉우리)
inner = [k for k in cnt.index if (k - 1) in cnt.index and (k + 1) in cnt.index and cnt[k] > cnt[k - 1] and cnt[k] > cnt[k + 1]]
if len(inner) >= 2:
    t2 = sorted(sorted(inner, key=lambda k: (-cnt[k], k))[:2])
    sg = cnt.loc[t2[0]:t2[1]]
    alt = int(sg[sg == sg.min()].index.min())
else:
    alt = float(L.median())
lines += ["", f"[민감도] 끝 막대를 봉우리로 치지 않으면 봉우리 {inner} -> 경계 {alt}칸 ({alt*15}분)"]
alt_type = np.where(sh["가장긴연속_칸"] <= alt, "한 번 튀는 형", "오래 높은 형")
for lab, m in [("1~6월", sh["월"] <= 6), ("7~8월", sh["월"] >= 7), ("★ 피크날", sh["별피크날"])]:
    a = pd.Series(alt_type[m.values]).value_counts().to_dict()
    tt = sh[m.values & (alt_type == "한 번 튀는 형")]
    lines.append(f"   {lab}: {a} / 튀는 형 아침직후 {int(tt['아침직후'].sum())}/{len(tt)}, 점심직후 {int(tt['점심직후'].sum())}/{len(tt)}, 생산직후 {int((tt['생산직후'] == True).sum())}/{len(tt)}")
lines.append(f"   모든 피크 근처 날(유형 무관): 아침직후 {int(sh['아침직후'].sum())}/{len(sh)}, 점심직후 {int(sh['점심직후'].sum())}/{len(sh)}")
lines.append("   참고: 연속 길이는 날 안에서만 센다(자정에서 끊김). 7/13 · 7/15 23시대에 피크 근처 칸이 있다.")
txt = "\n".join(lines)
print(txt)
mid("5.3_피크모양_log.txt").write_text(txt, encoding="utf-8")
