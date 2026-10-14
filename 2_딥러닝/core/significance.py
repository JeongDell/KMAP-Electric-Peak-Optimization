"""최고 선정 · 유의성 검정 — RF '전체 C 직접 학습·공통 연속형2' 설계와 같은 기준 (config.py의 최고 선정 · 유의성 검정 설명).
학습 없이 저장된 예측(preds/)으로 계산 (몇 분). main.py 마지막에 자동 실행 (다시 F5 하면 끝난 학습은 건너뛰고 이것만 다시 계산).

후보 = 모델별 최고 조합(best_params.csv)의 시드 평균 예측('모델 loss'), 같은 loss 세 모델 평균('ENSEMBLE loss'),
  기준선('Baseline' = 지난주 같은 시각 값, 선정 후보는 아님). 채점 칸 = SELECT_EVAL(전체 = 모든 칸), 피크 기준·가중치 = 연속형2.
기간 역할: 고르기 = config.FOLDS(7월), 확인 = CONFIRM_MONTHS(마진 확정 뒤 8월), 시험 = TEST_MONTH(9월, RUN_SEPTEMBER일 때).
고르는 건 그룹 단위, 채점·검정은 평가 구간(config.GROUPS의 report)마다. 지금은 그룹 = 구간 (1h · 4h · 8h · 12h · 24h · 4d를 따로 학습).
0) 최고 선정 → representatives.csv: 구간마다 A 최고 · C 최고 = 그 손실로 학습한 후보 중 고르기 월 공통 C 평균이 가장 낮은 것.
   하나로 고르지 않고 둘 다 남긴다 (비교용, 2026-10-05 사용자 결정). 참고로 A·C를 합친 1등과 달별·고르기 평균 C도 적는다
검정하는 달 = test_months(): 고르기에 안 쓴 달만 — 확인 월(있고 DM_ON_CONFIRM일 때), 9월(RUN_SEPTEMBER일 때).
  고르기 월(7월)로는 검정하지 않는다 (설정을 고른 달이라 낙관적). 9월 전 첫 실행에서는 최고 선정만 하고 검정은 건너뛴다.
1) 같은 loss 안 모델끼리 → dm_results.csv: 검정 달마다 따로, 구간마다 후보를 모두 짝지어 C 잣대로 검정. 짝마다 C가 낮은 쪽이 '모델1'
2) A 최고 vs C 최고 비교 (고르기 월로 고른 것) → ac_comparison.csv(검정), ac_decision.csv(비교 결과), 검정 달·구간마다.
   잣대 = C, MSE. 결과는 비교로만 적고 다시 고르지 않는다.
   p값은 기간마다 (구간 × 잣대) Holm 보정. 9월은 14일뿐이라 독립에 가까운 경로가 적은 구간(4d = 11개)은 '표본 적음'.

방법 (DM): 예측 시작 시점(경로)마다 잣대 값을 내고, 두 후보의 차이 계열로 검정.
  잣대 C = 그 경로 하나로 잰 C (core/train.py의 path_c), MSE = 그 경로 채점 칸의 제곱오차 평균.
  경로가 겹쳐 이웃 시점의 차이가 연관되므로 Newey-West 분산(시차 = 겹치는 시작 시점 수 − 1),
  HLN 소표본 보정, t 분포(자유도 n − 1, scipy가 없으면 정규 근사). 양측 검정, 유의수준 config.DM_ALPHA.
"""
import itertools
import math

import numpy as np
import pandas as pd
import torch

import config as C
from core import data as D, peak as PK, pipeline as P, train as T

OUT = C.RESULT_DIR / "significance"
YARDSTICKS = ["C", "MSE"]
MIN_EFFECTIVE = 10   # 경로 수 / (시차 + 1) — 서로 겹치지 않는 경로 수가 이보다 적으면 '표본 적음'


def p_two_sided(stat, dof):
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(stat), dof))
    except ImportError:
        return float(math.erfc(abs(stat) / math.sqrt(2)))


def dm_test(d, lag):
    """손실 차이 계열 d (모델1 − 모델2) → (DM 통계량, p값). 음수 = 모델1이 더 정확."""
    n = len(d)
    c = d - d.mean()
    gamma = [float((c[k:] * c[:n - k]).sum() / n) for k in range(lag + 1)]
    if gamma[0] == 0:
        return np.nan, np.nan
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    var = var if var > 0 else gamma[0]
    h = lag + 1
    stat = d.mean() / math.sqrt(var / n) * math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    return stat, p_two_sided(stat, n - 1)


def holm(p):
    """Holm 보정 p값 (NaN은 1로)."""
    p = np.where(np.isnan(np.asarray(p, float)), 1.0, np.asarray(p, float))
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def lag_of(g, h):
    """h칸 구간에서 경로가 겹치는 이웃 시작 시점 수."""
    return -(-h // C.GROUPS[g]["eval_stride"]) - 1


def candidate_preds(df, best, month, g):
    """그 달·그룹의 후보 예측 {이름: (예측 시작 시점, 칸) kW}과 (fold, 예측 시작 칸 t0, 정답 칸 fi, 칸별 피크 기준 b).
    이름 = '모델 loss', 'ENSEMBLE loss', 'Baseline'. 시드 예측 파일이 모두 있는 후보만 (시드 평균)."""
    G = C.GROUPS[g]
    H = G["horizon"]
    val_start, val_end, _ = P.split(df, month)
    fold = D.Fold(df, val_start, P.DEVICE)
    t0 = D.origins(df, val_start, val_end, H, G["eval_stride"])
    fi, thr = P.targets(fold, t0, H)
    preds = {"Baseline": fold.lag_kw[fi].cpu().numpy()}
    by_loss = {}
    for _, b in best[best["group"] == g].iterrows():
        paths = [P.pred_path(g, b["model"], b["loss"], b["params"], month, s) for s in C.SEEDS]
        if all(p.exists() for p in paths):
            p = np.mean([np.load(x) for x in paths], axis=0)
            preds[f"{b['model']} {b['loss']}"] = p
            by_loss.setdefault(b["loss"], []).append(p)
    for loss, ps in by_loss.items():
        if len(ps) == len(C.MODELS_TO_RUN) and len(ps) > 1:
            preds[f"ENSEMBLE {loss}"] = np.mean(ps, axis=0)
    return preds, fold, t0, fi, thr


def month_data(df, best, month, g):
    """그 달·그룹 → {구간: {후보 이름: {"월C": 그 달 전체 경로로 잰 C, "C": 경로별 C, "MSE": 경로별 MSE}}}.
    채점 칸이 없는 경로는 뺀다 (후보끼리 같은 경로)."""
    G = C.GROUPS[g]
    preds, fold, _, fi, thr = candidate_preds(df, best, month, g)
    y, m, b = fold.y_kw[fi].cpu(), fold.score[fi].cpu(), thr.cpu()
    w = PK.weight(y, b)
    out = {}
    for seg, h in G["report"].items():
        ys, ms, ws = y[:, :h], m[:, :h], w[:, :h]
        keep = ms.any(1)
        out[seg] = {}
        for name, p in preds.items():
            ps = torch.as_tensor(p[:, :h], dtype=torch.float32)
            out[seg][name] = {
                "월C": float(T.risk_loss(ps, ys, "C", ws, G["block"], ms)),
                "C": T.path_c(ps, ys, ws, ms, G["block"])[keep].numpy(),
                "MSE": ((((ps - ys) ** 2) * ms).sum(1) / ms.sum(1).clamp(min=1))[keep].numpy()}
    return out


def month_c(data, name, seg, months):
    """월 동일 평균 C (각 달 C의 단순 평균)."""
    return float(np.mean([data[mo][seg][name]["월C"] for mo in months]))


def joined(data, name, seg, months, ys):
    return np.concatenate([data[mo][seg][name][ys] for mo in months])


def names_in(data, seg, months, loss=None):
    """months 내내 있는 후보 이름 (Baseline 제외, loss를 주면 그 loss만)."""
    return [n for n in data[months[0]][seg] if n != "Baseline" and (loss is None or n.endswith(f" {loss}"))
            and all(n in data[mo][seg] for mo in months)]


def full_seg(g):
    """그룹 전체 구간 이름 (report의 마지막 = 모델이 학습한 전체 예측 구간)."""
    return list(C.GROUPS[g]["report"])[-1]


def pick(data, months, g, loss=None):
    """그룹 전체 구간에서 months(고르기 월)의 C가 가장 낮은 후보 (loss를 주면 그 loss 안에서 = 'loss 최고'). 없으면 None."""
    seg = full_seg(g)
    names = names_in(data, seg, months, loss)
    return min(names, key=lambda n: month_c(data, n, seg, months)) if names else None


def role(month):
    """달 → 역할 이름 (고르기 / 확인 / 시험)."""
    return "고르기" if month in C.FOLDS else "확인" if month in C.CONFIRM_MONTHS else "시험"


def test_months():
    """검정하는 달 (고르기 월은 넣지 않음): DM_ON_CONFIRM이면 확인 월, RUN_SEPTEMBER면 9월."""
    return (list(C.CONFIRM_MONTHS) if C.DM_ON_CONFIRM else []) + ([C.TEST_MONTH] if C.RUN_SEPTEMBER else [])


def representatives(data, g):
    """구간마다 A 최고 · C 최고 (고르기 월 C로 고름, 둘 다 남김)와 참고 1등(A·C 합쳐), 달별 C → 표 행."""
    picks = {f"{loss} 최고": pick(data, C.FOLDS, g, loss) for loss in C.LOSSES_TO_RUN}
    picks["참고 1등(A·C 합쳐)"] = pick(data, C.FOLDS, g)
    rows = []
    for seg in C.GROUPS[g]["report"]:
        row = {"group": g, "구간": seg}
        for col, n in picks.items():
            row[col] = n
            row[f"{col} C(고르기 평균)"] = month_c(data, n, seg, C.FOLDS) if n else np.nan
            for mo in data:
                row[f"{col} C({role(mo)} {mo})"] = data[mo][seg][n]["월C"] if n in data[mo][seg] else np.nan
        for mo in data:
            row[f"Baseline C({role(mo)} {mo})"] = data[mo][seg]["Baseline"]["월C"]
        rows.append(row)
    return rows


def within_loss(data, months, g, period):
    """구간·loss마다 후보(+ Baseline)를 모두 짝지어 C 잣대로 검정."""
    rows = []
    for seg, h in C.GROUPS[g]["report"].items():
        for loss in C.LOSSES_TO_RUN:
            names = names_in(data, seg, months, loss) + ["Baseline"]
            if len(names) < 2:
                print(f"{period} {g} {seg} 손실 {loss}: 시드 예측 파일이 모두 있는 모델이 없어 건너뜀", flush=True)
                continue
            names.sort(key=lambda n: month_c(data, n, seg, months))   # C 낮은 순 → 짝의 모델1 = 더 나은 쪽
            for a, b in itertools.combinations(names, 2):
                la, lb = joined(data, a, seg, months, "C"), joined(data, b, seg, months, "C")
                stat, p = dm_test(la - lb, lag_of(g, h))
                verdict = ("차이 유의하지 않음" if not p < C.DM_ALPHA
                           else "모델1이 유의하게 나음" if stat < 0 else "모델2가 유의하게 나음")
                rows.append({"period": period, "group": g, "구간": seg, "loss": loss, "모델1": a, "모델2": b,
                             "C_1": month_c(data, a, seg, months), "C_2": month_c(data, b, seg, months),
                             "DM": stat, "p값": p, "판정": verdict, "경로 수": len(la)})
    return rows


def a_vs_c(data, plan, g, period):
    """plan = [(검정 달, 고르는 달들)] → A 최고 vs C 최고(그룹 단위로 고름)를 구간 · 잣대마다 검정한 DM 행
    (DM < 0 = A가 나음, > 0 = C가 나음). 고르는 달과 검정 달은 겹치지 않게 부른다."""
    picks = [(mo, pick(data, sel, g, "A"), pick(data, sel, g, "C")) for mo, sel in plan]
    seg_full = full_seg(g)
    if any(a is None or c is None or a not in data[mo][seg_full] or c not in data[mo][seg_full] for mo, a, c in picks):
        print(f"{period} {g}: A 또는 C 최고의 예측 파일이 없어 A vs C 비교를 건너뜀", flush=True)
        return []

    def label(k):   # 검정 달마다 고른 최고가 같으면 이름 하나, 다르면 '달: 이름'
        names = [x[k] for x in picks]
        return names[0] if len(set(names)) == 1 else ", ".join(f"{x[0]}: {x[k]}" for x in picks)

    rows = []
    for seg, h in C.GROUPS[g]["report"].items():
        lag = lag_of(g, h)
        for ys in YARDSTICKS:
            la = np.concatenate([data[mo][seg][a][ys] for mo, a, _ in picks])
            lc = np.concatenate([data[mo][seg][c][ys] for mo, _, c in picks])
            stat, p = dm_test(la - lc, lag)
            if ys == "C":   # 표시 값 = 월 C (검정 달마다 그 달에 쓴 최고의 C, 달끼리 단순 평균)
                va = float(np.mean([data[mo][seg][a]["월C"] for mo, a, _ in picks]))
                vc = float(np.mean([data[mo][seg][c]["월C"] for mo, _, c in picks]))
            else:           # 표시 값 = RMSE
                va, vc = math.sqrt(la.mean()), math.sqrt(lc.mean())
            rows.append({"period": period, "group": g, "구간": seg, "잣대": ys, "A 최고": label(1), "C 최고": label(2),
                         "A 값": va, "C 값": vc, "DM": stat, "p값": p, "경로 수": len(la),
                         "표본": "적음 — 해석 보류" if len(la) / (lag + 1) < MIN_EFFECTIVE else ""})
    return rows


def decide(part):
    """한 기간·구간의 두 잣대 검정 → 비교 결과 문장. 고르는 규칙이 아니라 'C 손실이 효과가 있었나'의 보고 (다시 고르지 않음)."""
    r = part.set_index("잣대")["판정"]
    c_better, a_better = r["C"] == "C가 유의하게 나음", r["C"] == "A가 유의하게 나음"
    mse_worse = r["MSE"] == "A가 유의하게 나음"
    if c_better and not mse_worse:
        text = "C가 나음 — C 잣대에서 유의하게 낫고, MSE 손해는 유의하지 않음"
    elif c_better:
        text = "맞바꿈 — C 잣대는 C가 유의하게 낫고, MSE는 A가 유의하게 나음"
    elif a_better:
        text = "A가 나음 — C 잣대에서도 A가 유의하게 나음"
    elif mse_worse:
        text = "C 잣대는 차이 없음, MSE는 A가 유의하게 나음"
    else:
        text = "차이 유의하지 않음"
    return text + (" (표본 적음 — 해석 보류)" if (part["표본"] != "").any() else "")


def show(table, title):
    print(f"\n===== {title} =====")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def run(df):
    best_file = C.RESULT_DIR / "best_params.csv"
    if not best_file.exists():
        print("best_params.csv가 없어 대표 선정·유의성 검정을 건너뜀 (main.py의 그리드 서치가 먼저 끝나야 함)", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    best = pd.read_csv(best_file, dtype={"params": str}, encoding="utf-8-sig")
    folds, confirm = list(C.FOLDS), list(C.CONFIRM_MONTHS)
    months = folds + confirm + ([C.TEST_MONTH] if C.RUN_SEPTEMBER else [])
    do_ac = {"A", "C"} <= set(C.LOSSES_TO_RUN)
    reps, within, ac = [], [], []
    for g in C.GROUPS_TO_RUN:
        data = {mo: month_data(df, best, mo, g) for mo in months}
        reps += representatives(data, g)
        for mo in test_months():   # 고르기에 안 쓴 달마다 따로 검정
            within += within_loss(data, [mo], g, f"{role(mo)} {mo}")
            if do_ac:              # A 최고 vs C 최고: 고르기 월로 고른 것끼리
                ac += a_vs_c(data, [(mo, folds)], g, f"{role(mo)} {mo}")
    rep = pd.DataFrame(reps)
    rep.to_csv(OUT / "representatives.csv", index=False, encoding="utf-8-sig")
    show(rep, f"구간별 A 최고 · C 최고 (고르기 월 {', '.join(folds)}의 공통 C 평균 최저, 연속형2, {C.SELECT_EVAL}, 둘 다 남김)와 "
              "달별 C — 고르기 월이 아닌 달(확인·시험)은 고르는 데 안 씀")
    if within:
        res = pd.DataFrame(within)
        res.to_csv(OUT / "dm_results.csv", index=False, encoding="utf-8-sig")
        for (period, g, seg), part in res.groupby(["period", "group", "구간"], sort=False):
            show(part.drop(columns=["period", "group", "구간"]),
                 f"같은 loss 안 DM 검정 (C 잣대) — {period}, {g} {seg} (C_1·C_2 = 월 동일 평균 C)")
    if not ac:
        if not test_months():
            print("\n검정은 고르기에 안 쓴 달로만 한다 — 지금은 9월뿐이라 RUN_SEPTEMBER = True 단계에서 실행됨 (최고 선정은 위 표)",
                  flush=True)
        return
    res = pd.DataFrame(ac)
    res["p_Holm"] = res.groupby("period")["p값"].transform(lambda p: holm(p.to_numpy()))
    res["판정"] = np.select([(res["p_Holm"] < C.DM_ALPHA) & (res["DM"] > 0), (res["p_Holm"] < C.DM_ALPHA) & (res["DM"] < 0)],
                          ["C가 유의하게 나음", "A가 유의하게 나음"], "차이 유의하지 않음")
    res = res[[c for c in res.columns if c != "표본"] + ["표본"]]
    res.to_csv(OUT / "ac_comparison.csv", index=False, encoding="utf-8-sig")
    dec = pd.DataFrame([{"period": period, "group": g, "구간": seg, "A 최고": part["A 최고"].iloc[0],
                         "C 최고": part["C 최고"].iloc[0], "비교 결과": decide(part)}
                        for (period, g, seg), part in res.groupby(["period", "group", "구간"], sort=False)])
    dec.to_csv(OUT / "ac_decision.csv", index=False, encoding="utf-8-sig")
    for period, part in res.groupby("period", sort=False):
        show(part.drop(columns="period"), f"A 최고 vs C 최고 — {period} (값 = 잣대 C면 월 C, MSE면 RMSE kW, "
                                          "DM > 0 = C가 나음, p_Holm = 기간 안 Holm 보정)")
    show(dec, "A 최고 vs C 최고 비교 결과 (C 손실의 효과 보고 — 이 결과로 다시 고르지 않음)")
