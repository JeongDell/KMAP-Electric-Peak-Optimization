"""실험 엔진 — 그리드 서치, 두 단계 학습(train_eval), 시드 반복·앙상블, 9월 테스트, 결과 표. 결과 파일 설명은 main.py 맨 위.
모델 비교는 같은 loss끼리만 한다 (표·출력 모두 그룹 → horizon → loss → 모델 순서, Baseline은 loss마다 참고로)."""
import hashlib
import itertools
import json
import time

import numpy as np
import pandas as pd
import torch

import config as C
from core import data as D, peak as PK, train as T
from models import MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID_FILE = C.RESULT_DIR / "grid_results.csv"
CURVE_FILE = C.RESULT_DIR / "curves.csv"
SEED_FILE = C.RESULT_DIR / "seed_results.csv"
PRED_DIR = C.RESULT_DIR / "preds"
KEY = ["group", "model", "loss", "params", "fold"]
INFO = KEY + ["best_epoch", "train_RMSE", "seconds"]


def param_list(grid):
    """그리드의 모든 조합. GRID_SEARCH = False면 각 후보의 첫 값만."""
    if not C.GRID_SEARCH:
        return [{k: v[0] for k, v in grid.items()}]
    return [dict(zip(grid, combo)) for combo in itertools.product(*grid.values())]


def check_settings(df):
    """긴 학습 전에 설정 점검: loss 이름, 그리드 설정 이름(모델·학습이 아는 이름인지, 필수 이름이 있는지), lookback 상한."""
    bad = [l for l in C.LOSSES_TO_RUN if l not in T.LOSSES]
    if bad:
        raise SystemExit(f"LOSSES_TO_RUN 확인: {bad} — 쓸 수 있는 loss는 {T.LOSSES}")
    for name in (C.SELECT_EVAL, C.CHECK_EVAL):
        if name not in C.EVAL_SETS:
            raise SystemExit(f"SELECT_EVAL·CHECK_EVAL 확인: '{name}'이 EVAL_SETS에 없음 ({list(C.EVAL_SETS)})")
    for name, rule in C.EVAL_SETS.items():
        if set(rule) != {"copied", "prod_fix", "outage"}:
            raise SystemExit(f"EVAL_SETS['{name}'] 확인 — copied·prod_fix·outage 세 키가 있어야 함")
    max_lb = int(df["ts"].searchsorted(pd.Timestamp(C.TRAIN_FIRST_ORIGIN))) - 672   # 과거 블록 첫 칸의 지난주 값이 있어야 함
    for name in C.MODELS_TO_RUN:
        grid, allowed = MODELS[name].GRID, {**T.TRAIN_PARAMS, **MODELS[name].PARAMS}
        unknown = [k for k in grid if k not in allowed]
        missing = [k for k, v in allowed.items() if v is None and k not in grid]
        empty = [k for k, v in grid.items() if not v]
        if unknown or missing or empty:
            raise SystemExit(f"config.GRIDS['{name}'] 확인 — 모르는 이름 {unknown}, 빠진 필수 이름 {missing}, 후보가 빈 이름 {empty}\n"
                             f"  쓸 수 있는 이름 (기본값, None = 필수): {allowed}")
        if max(grid["lookback"]) > max_lb:
            raise SystemExit(f"config.GRIDS['{name}']의 lookback 최대 {max(grid['lookback'])} > {max_lb}칸 "
                             "→ lookback을 줄이거나 TRAIN_FIRST_ORIGIN을 늦추기")
    for g in C.GROUPS_TO_RUN:
        G = C.GROUPS[g]
        if any(h % G["block"] for h in [G["horizon"], *G["report"].values()]):
            raise SystemExit(f"config.GROUPS['{g}'] 확인 — horizon·report 칸 수가 block {G['block']}의 배수여야 함")
    if not 0 <= C.PEAK_WEIGHT["START"] < 1 or C.PEAK_WEIGHT["W_MAX"] < 1:
        raise SystemExit(f"config.PEAK_WEIGHT 확인: {C.PEAK_WEIGHT} — 0 ≤ START < 1, W_MAX ≥ 1")


def describe_grid():
    """학습 횟수를 미리 보여준다 (처음부터 돌릴 때 기준: 그리드 서치 + 시드 반복 + 최종 모델). 끝난 학습은 실제로는 건너뛴다."""
    lines, total = [], 0
    for name in C.MODELS_TO_RUN:
        grid = MODELS[name].GRID if C.GRID_SEARCH else {k: v[:1] for k, v in MODELS[name].GRID.items()}
        n = len(param_list(MODELS[name].GRID))
        total += n
        lines.append(f"  {name:6s}: " + " × ".join(f"{k} {len(v)}" for k, v in grid.items()) + f" = {n}")
    nl, ng, nf, nk = len(C.LOSSES_TO_RUN), len(C.GROUPS_TO_RUN), len(C.FOLDS), len(C.CONFIRM_MONTHS)
    best = len(C.MODELS_TO_RUN) * nl * ng
    extra = len([s for s in C.SEEDS if s != C.SEED])
    seeds = best * nf * extra                         # 고르기 월: 그리드에서 한 시드는 빼고
    final = best * len(C.SEEDS) if nk else 0          # 최종 모델: 확인 월 직전까지 학습 → 확인 월·9월 예측
    one = total * nl * ng * nf + seeds + final
    nc = len(getattr(C, "CONDITIONS_TO_RUN", [None]))
    print("학습 횟수 (처음부터 돌릴 때, 한 번 = 최고 에폭 찾기 + 다시 학습의 두 단계)\n" + "\n".join(lines)
          + f"\n  그리드 서치: 조합 {total} × loss {nl} × 그룹 {ng} × 고르기 월 {nf} = {total * nl * ng * nf:,}번"
          + f"\n  시드 반복  : 최고 조합 {best}개 (모델 {len(C.MODELS_TO_RUN)} × loss {nl} × 그룹 {ng}) × "
          + f"고르기 월 {nf} × 추가 시드 {extra} = {seeds:,}번"
          + (f"\n  최종 모델  : {best} × 시드 {len(C.SEEDS)} = {final:,}번 ({C.CONFIRM_MONTHS[0]} 직전까지 학습 → "
             f"{C.CONFIRM_MONTHS[0]} 확인 · {C.TEST_MONTH} 시험 예측)" if nk else "")
          + (f"\n  미래 정보 조건 {nc}개({', '.join(C.CONDITIONS_TO_RUN)})를 각각 처음부터 → 총 {one * nc:,}번" if nc > 1 else ""),
          flush=True)


def read_csv(path, keys):
    if not path.exists():
        return pd.DataFrame(columns=keys + ["eval_set"])
    return pd.read_csv(path, dtype={k: str for k in keys + ["eval_set"]}, encoding="utf-8-sig")


def load_results():
    return read_csv(GRID_FILE, KEY).drop_duplicates(KEY + ["eval_set", "horizon"], keep="last") if GRID_FILE.exists() else read_csv(GRID_FILE, KEY)


def append_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


def window_index(t0, horizon):
    return torch.as_tensor(t0, device=DEVICE)[:, None] + torch.arange(horizon, device=DEVICE)


def targets(fold, t0, horizon):
    """평가 창 t0 → (정답 칸 번호 (N, H), 칸별 피크 기준 b kW (N, H) — 각 창의 원점까지 관측값만 사용)."""
    fi = window_index(t0, horizon)
    return fi, PK.threshold(fold, t0, fi)


def score(pred_kw, fold, fi, peak, G):
    """같은 예측을 평가 세트마다 채점한 지표 행 목록 (G = config.GROUPS의 그룹 설정)."""
    return [{"eval_set": name, **r} for name, m in fold.eval.items()
            for r in T.report(pred_kw, fold.y_kw[fi], m[fi], peak, G["report"], G["block"])]


def pred_path(group, model, loss, params_json, fold, seed):
    """검증 예측 저장 경로 (설정값은 짧은 해시로 구분)."""
    h = hashlib.md5(params_json.encode()).hexdigest()[:8]
    return PRED_DIR / f"{group}_{model}_loss{loss}_{h}_{fold}_seed{seed}.npy"


def split(df, month):
    """검증 월 → (검증 시작, 검증 끝, 조기 종료 확인 시작)."""
    val_start = pd.Timestamp(month + "-01")
    return val_start, val_start + pd.offsets.MonthBegin(1), D.es_start(df, val_start, C.ES_DAYS)


def windows(df, fold, G, val_start, val_end, es_start):
    """(최고 에폭 찾기 학습 창, 조기 종료 확인 창, 다시 학습 창, 검증 창)."""
    H, s = G["horizon"], G["train_stride"]
    return (D.origins(df, C.TRAIN_FIRST_ORIGIN, es_start, H, s),
            D.has_targets(fold.check_np, D.origins(df, es_start, val_start, H, s), H),
            D.origins(df, C.TRAIN_FIRST_ORIGIN, val_start, H, s),
            D.origins(df, val_start, val_end, H, G["eval_stride"]))


def train_eval(fold, mod, params, loss, G, win, seed, save_to=None, fold_es=None):
    """두 단계 학습 → (평가 월 예측 kW, 최고 에폭, 학습 곡선, 학습 구간 RMSE). 검증·9월 테스트 공통. G = 그룹 설정.
    ① 조기 종료 확인(직전, 복사되지 않은 날 14일)으로 최고 에폭 찾기 — fold_es(정규화 = 확인 구간 시작 전까지) 사용
    ② 같은 시드로 처음부터 평가 월 직전까지 전체를 그 에폭만큼 — fold(정규화 = 평가 월 직전까지) 사용.
    fold_es가 없으면 ①도 fold. save_to가 있으면 ② 모델 가중치를 저장."""
    tr_t0, es_t0, all_t0, val_t0 = win
    H = G["horizon"]
    spec = T.make_spec(fold, mod, params["lookback"], H, G["block"])   # 열 순서는 두 fold가 같음
    torch.manual_seed(seed)
    model = mod.build(params, H).to(DEVICE)
    best_ep, history = T.fit(model, fold_es if fold_es is not None else fold, spec, loss, params, tr_t0, es_t0, seed=seed)
    torch.manual_seed(seed)
    model = mod.build(params, H).to(DEVICE)
    T.fit(model, fold, spec, loss, params, all_t0, epochs=max(1, best_ep), seed=seed)
    if save_to is not None:
        torch.save(model.state_dict(), save_to)
    pred, _ = T.predict(model, fold, spec, val_t0)
    return fold.to_kw(pred), best_ep, history, T.train_rmse(model, fold, spec, all_t0)


def left_time(spent, remaining):
    left = remaining * np.mean(spent)
    return f"남은 예상 약 {left // 3600:.0f}시간 {left % 3600 // 60:.0f}분"


def run_grid(df):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    done = set(map(tuple, load_results()[KEY].to_numpy()))
    n_params = sum(len(param_list(MODELS[m].GRID)) for m in C.MODELS_TO_RUN)
    jobs = n_params * len(C.LOSSES_TO_RUN) * len(C.GROUPS_TO_RUN) * len(C.FOLDS)
    n, spent = 0, []
    for month in C.FOLDS:
        val_start, val_end, es_start = split(df, month)
        print(f"{month} 검증: ① 학습 {C.TRAIN_FIRST_ORIGIN} ~ {es_start.date()} 전날 + 조기 종료 확인 {es_start.date()} ~ "
              f"{val_start.date()} 전날 → ② 다시 학습 {C.TRAIN_FIRST_ORIGIN} ~ {val_start.date()} 전날", flush=True)
        fold = D.Fold(df, val_start, DEVICE)       # ② 다시 학습·채점: 정규화 기준 = 검증 월 이전
        fold_es = D.Fold(df, es_start, DEVICE)     # ① 최고 에폭 찾기: 정규화 기준 = 확인 구간 이전
        for g in C.GROUPS_TO_RUN:
            G = C.GROUPS[g]
            H = G["horizon"]
            last_h = list(G["report"])[-1]
            win = windows(df, fold, G, val_start, val_end, es_start)
            fi, peak = targets(fold, win[3], H)
            info = {"group": g, "model": "Baseline", "loss": "-", "params": "{}", "fold": month,
                    "best_epoch": np.nan, "train_RMSE": np.nan, "seconds": 0.0}
            if tuple(info[k] for k in KEY) not in done:
                append_csv(GRID_FILE, [{**info, **r} for r in score(fold.lag_kw[fi], fold, fi, peak, G)])
            for name in C.MODELS_TO_RUN:
                mod = MODELS[name]
                for params in param_list(mod.GRID):
                    for loss in C.LOSSES_TO_RUN:
                        n += 1
                        info = {"group": g, "model": name, "loss": loss, "params": json.dumps(params), "fold": month}
                        if tuple(info[k] for k in KEY) in done and pred_path(g, name, loss, info["params"], month, C.SEED).exists():
                            continue
                        start = time.time()
                        pred_kw, best_ep, history, tr_rmse = train_eval(fold, mod, params, loss, G, win, C.SEED,
                                                                        fold_es=fold_es)
                        np.save(pred_path(g, name, loss, info["params"], month, C.SEED), pred_kw.cpu().numpy())
                        rows = score(pred_kw, fold, fi, peak, G)
                        info.update(best_epoch=best_ep, train_RMSE=tr_rmse, seconds=round(time.time() - start, 1))
                        append_csv(CURVE_FILE, [{**{k: info[k] for k in KEY}, **h} for h in history])
                        append_csv(GRID_FILE, [{**{k: info[k] for k in INFO}, **r} for r in rows])   # '끝난 조합' 표시라 마지막에
                        spent.append(info["seconds"])
                        val_rmse = next(r["RMSE"] for r in rows if r["eval_set"] == C.SELECT_EVAL and r["horizon"] == last_h)
                        print(f"[{n}/{jobs}] {month} {g} {name} 손실{loss} {params} → 최고 에폭 {best_ep}, "
                              f"학습 RMSE {tr_rmse:.2f} / 검증 RMSE {val_rmse:.2f}, {info['seconds']:.0f}초, "
                              f"{left_time(spent, jobs - n)}", flush=True)


def ordered(table):
    """(평가 세트 →) (검증 월 →) 그룹 → horizon → loss → 모델 순서로 정렬 (같은 loss끼리 붙도록)."""
    h_rank = {(g, h): i for g, G in C.GROUPS.items() for i, h in enumerate(G["report"])}
    keys = {}
    if "eval_set" in table:
        keys["e"] = table["eval_set"].map({e: i for i, e in enumerate(C.EVAL_SETS)})
    if "fold" in table:
        keys["f"] = table["fold"]
    keys["g"] = table["group"].map({g: i for i, g in enumerate(C.GROUPS)})
    keys["h"] = [h_rank[(g, h)] for g, h in zip(table["group"], table["horizon"])]
    keys["l"] = table["loss"]
    keys["m"] = table["model"].map({m: i for i, m in enumerate(["Baseline", *MODELS, "ENSEMBLE"])})
    keys = pd.DataFrame(keys, index=table.index)
    return table.loc[keys.sort_values(list(keys.columns)).index].reset_index(drop=True)


def summarize(res):
    """(그룹, 모델, loss)별 최적 조합 = 고르기 월(FOLDS) SELECT_METRIC(SELECT_EVAL 기준, 달이 여럿이면 평균)이 가장 작은 조합
    (SELECT_METRIC = None이면 '그 loss 값') → (최적 조합, 고르기 월 비교표, 월별 비교표).
    고르기 월이 아닌 달의 그리드 줄(예: 원본에서 넘어온 8월)은 읽지 않는다."""
    res = res[res["fold"].isin(C.FOLDS) & res["group"].isin(C.GROUPS_TO_RUN)
              & res["model"].isin(["Baseline", *C.MODELS_TO_RUN]) & res["loss"].isin(["-", *C.LOSSES_TO_RUN])]
    valid = {model: {json.dumps(params) for params in param_list(MODELS[model].GRID)} for model in C.MODELS_TO_RUN}
    res = res[[model == "Baseline" or params in valid.get(model, set()) for model, params in zip(res["model"], res["params"])]]
    sel = res[res["eval_set"] == C.SELECT_EVAL]
    models = sel[sel["model"] != "Baseline"]
    full_h = {g: list(G["report"])[-1] for g, G in C.GROUPS.items()}   # 전체 horizon 행으로 선택
    full = models[models["horizon"] == models["group"].map(full_h)].copy()
    metric = getattr(C, "SELECT_METRIC", None)
    full["score"] = [r[metric] if metric else r["Loss" + r["loss"]] for _, r in full.iterrows()]
    full = full[full["score"].notna()]   # 학습이 발산해 점수가 NaN인 달은 빼고 → 고르기 월이 다 있는 조합만 남음
    sc = (full.groupby(["group", "model", "loss", "params"])
          .agg(score=("score", "mean"), n_folds=("fold", "nunique"), best_epoch=("best_epoch", "median"))
          .reset_index())
    sc = sc[sc["n_folds"] == len(C.FOLDS)]
    best = sc.loc[sc.groupby(["group", "model", "loss"])["score"].idxmin()].reset_index(drop=True)
    keep = pd.concat([res.merge(best[["group", "model", "loss", "params"]]), res[res["model"] == "Baseline"]])
    cols = ["MAE", "RMSE", "train_RMSE"] + T.METRICS[2:]   # train_RMSE = 전체 예측 구간 기준 → 전체 horizon 행의 RMSE와 비교
    by = ["eval_set", "group", "horizon", "model", "loss", "params"]
    comp = keep.groupby(by)[cols].mean().reset_index()
    comp_month = keep.groupby(by + ["fold"])[cols].mean().reset_index()
    return best, ordered(comp), ordered(comp_month)


def pyplot():
    """한글 글꼴을 맞춘 matplotlib.pyplot. matplotlib이 없으면 None (그림만 건너뜀, 숫자는 csv에 있음)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import font_manager, pyplot as plt
    except ImportError:
        print("matplotlib이 없어 그림은 건너뜀 (숫자는 csv에 있음)", flush=True)
        return None
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    for f in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"]:   # 한글 글꼴이 있으면 사용
        if f in fonts:
            plt.rcParams["font.family"] = f
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_curves(best):
    """최적 조합들의 학습 곡선 (검증 월별 학습 손실·확인 손실, 점선 = 확인 손실 최저 epoch) → curves/*.png"""
    if best.empty or not CURVE_FILE.exists():
        return
    plt = pyplot()
    if plt is None:
        return
    curves = pd.read_csv(CURVE_FILE, dtype={k: str for k in KEY}, encoding="utf-8-sig")
    curves = curves.drop_duplicates(KEY + ["epoch"], keep="last")   # 중간에 끊겨 다시 학습한 조합은 마지막 기록만
    out = C.RESULT_DIR / "curves"
    out.mkdir(exist_ok=True)
    for _, b in best.iterrows():
        part = curves.merge(pd.DataFrame([b[["group", "model", "loss", "params"]]]))
        fig, axes = plt.subplots(1, len(C.FOLDS), figsize=(5 * len(C.FOLDS), 3.5), sharey=True, squeeze=False)
        for ax, (month, c) in zip(axes[0], part.groupby("fold")):
            ax.plot(c["epoch"], c["train_loss"], label="학습 구간")
            ax.plot(c["epoch"], c["es_loss"], label="확인 구간")
            if c["es_loss"].notna().any():
                ax.axvline(c.loc[c["es_loss"].idxmin(), "epoch"], color="gray", ls="--", lw=1)
            ax.set_title(f"{month} 검증")
            ax.set_xlabel("에폭")
        axes[0][0].set_ylabel(f"손실{b['loss']} (kW)")
        axes[0][0].legend()
        fig.suptitle(f"{b['group']} {b['model']} 손실{b['loss']} {b['params']}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{b['group']}_{b['model']}_loss{b['loss']}.png", dpi=100)
        plt.close(fig)


def show(table, title):
    """그룹 · horizon · loss별 블록으로 출력 — 모델은 같은 loss끼리만 비교 (Baseline은 블록마다 참고로 아래에)."""
    print(f"\n===== {title} =====")
    for (g, h), part in table.groupby(["group", "horizon"], sort=False):
        base = part[part["model"] == "Baseline"]
        for loss in [l for l in C.LOSSES_TO_RUN if (part["loss"] == l).any()]:
            print(f"\n[{g} {h} · 손실 {loss}]")
            block = pd.concat([part[part["loss"] == loss], base]).drop(columns=["group", "horizon", "loss"])
            print(block.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def side_by_side(table, column, out_name, title):
    """RMSE를 column(평가 세트 또는 검증 월)끼리 나란히 놓은 표 → 저장 + 출력."""
    if table.empty:
        return
    wide = table.pivot_table(index=["group", "horizon", "model", "loss"], columns=column, values="RMSE", sort=False)
    wide = wide.reset_index()
    wide.to_csv(C.RESULT_DIR / out_name, index=False, encoding="utf-8-sig")
    show(wide, title)


def run_seeds(df, best, months, seed_file, model_prefix=None):
    """best의 조합들을 months(고르기·확인 월 또는 9월)마다 시드 SEEDS로 두 단계 학습 → seed_file에 시드별 지표.
    예측 파일이 이미 있으면(고르기 월 그리드에서 한 시드 등) 다시 학습하지 않는다. 확인 월은 그리드가 없어 시드 전부 새로 학습.
    model_prefix가 있으면 모델 가중치도 저장."""
    if best.empty:
        return
    seed_key = KEY + ["seed"]
    done = set(map(tuple, read_csv(seed_file, seed_key)[seed_key].astype(str).to_numpy()))
    if model_prefix:
        (C.RESULT_DIR / "models").mkdir(exist_ok=True)
    jobs = len(best) * len(months) * len(C.SEEDS)
    n, spent = 0, []
    for month in months:
        val_start, val_end, es_start = split(df, month)
        fold = D.Fold(df, val_start, DEVICE)       # ② 다시 학습·채점: 정규화 기준 = 평가 월 이전
        fold_es = D.Fold(df, es_start, DEVICE)     # ① 최고 에폭 찾기: 정규화 기준 = 확인 구간 이전
        for g, part in best.groupby("group"):
            G = C.GROUPS[g]
            H = G["horizon"]
            win = windows(df, fold, G, val_start, val_end, es_start)
            fi, peak = targets(fold, win[3], H)
            for _, b in part.iterrows():
                mod, params = MODELS[b["model"]], json.loads(b["params"])
                for seed in C.SEEDS:
                    n += 1
                    info = {"group": g, "model": b["model"], "loss": b["loss"], "params": b["params"], "fold": month,
                            "seed": str(seed)}
                    path = pred_path(g, b["model"], b["loss"], b["params"], month, seed)
                    if tuple(info[k] for k in seed_key) in done and path.exists():
                        continue
                    start, best_ep = time.time(), np.nan
                    if not path.exists():
                        save_to = (C.RESULT_DIR / "models" / f"{model_prefix}{path.stem}.pt") if model_prefix else None
                        pred_kw, best_ep, _, _ = train_eval(fold, mod, params, b["loss"], G, win, seed, save_to,
                                                            fold_es=fold_es)
                        np.save(path, pred_kw.cpu().numpy())
                        spent.append(time.time() - start)
                        print(f"[시드 {n}/{jobs}] {month} {g} {b['model']} 손실{b['loss']} 시드 {seed} → 최고 에폭 {best_ep}, "
                              f"{spent[-1]:.0f}초, {left_time(spent, jobs - n)}", flush=True)
                    pred_kw = torch.as_tensor(np.load(path), device=DEVICE)
                    append_csv(seed_file, [{**info, "best_epoch": best_ep, **r}
                                           for r in score(pred_kw, fold, fi, peak, G)])


def summarize_seeds(df, best, months, seed_file, prefix, period):
    """months 평균으로 (1) 시드별 RMSE 평균 ± 표준편차, (2) 시드 평균 앙상블·세 모델 평균 앙상블·Baseline의 전체 지표 (SELECT_EVAL).
    저장: {prefix}seed_summary.csv, {prefix}ensemble.csv. 반환: (2) 표."""
    if best.empty or not seed_file.exists():
        return None
    seeds = read_csv(seed_file, KEY + ["seed"])
    seeds = seeds[(seeds["eval_set"] == C.SELECT_EVAL) & seeds["fold"].isin(months)]
    per_seed = seeds.groupby(["group", "horizon", "model", "loss", "seed"])["RMSE"].mean()   # 평가 월 평균
    stat = per_seed.groupby(["group", "horizon", "model", "loss"]).agg(["mean", "std", "count"])
    stat.columns = ["RMSE_시드평균", "RMSE_시드표준편차", "시드수"]
    ens = []   # Baseline, 시드 평균 예측, 세 모델 평균 예측을 채점
    for month in months:
        val_start, val_end, _ = split(df, month)
        fold = D.Fold(df, val_start, DEVICE)
        for g, part in best.groupby("group"):
            G = C.GROUPS[g]
            fi, peak = targets(fold, D.origins(df, val_start, val_end, G["horizon"], G["eval_stride"]), G["horizon"])
            y_kw, mask = fold.y_kw[fi], fold.score[fi]
            ens += [{"group": g, "model": "Baseline", "loss": "-", "fold": month, **r}
                    for r in T.report(fold.lag_kw[fi], y_kw, mask, peak, G["report"], G["block"])]
            by_loss = {}
            for _, b in part.iterrows():
                paths = [pred_path(g, b["model"], b["loss"], b["params"], month, s) for s in C.SEEDS]
                if not all(p.exists() for p in paths):
                    continue
                p = torch.as_tensor(np.mean([np.load(x) for x in paths], axis=0), device=DEVICE)
                by_loss.setdefault(b["loss"], []).append(p)
                ens += [{"group": g, "model": b["model"], "loss": b["loss"], "fold": month, **r}
                        for r in T.report(p, y_kw, mask, peak, G["report"], G["block"])]
            for loss, ps in by_loss.items():
                if len(ps) == len(C.MODELS_TO_RUN) and len(ps) > 1:
                    p = torch.stack(ps).mean(0)
                    ens += [{"group": g, "model": "ENSEMBLE", "loss": loss, "fold": month, **r}
                            for r in T.report(p, y_kw, mask, peak, G["report"], G["block"])]
    full = ordered(pd.DataFrame(ens).groupby(["group", "horizon", "model", "loss"])[T.METRICS].mean().reset_index())
    full.to_csv(C.RESULT_DIR / f"{prefix}ensemble.csv", index=False, encoding="utf-8-sig")
    stat = stat.join(full.set_index(["group", "horizon", "model", "loss"])["RMSE"].rename("RMSE_앙상블"), how="outer")
    out = ordered(stat.reset_index())
    out.to_csv(C.RESULT_DIR / f"{prefix}seed_summary.csv", index=False, encoding="utf-8-sig")
    show(out, f"시드 {len(C.SEEDS)}개 반복 + 앙상블 ({C.SELECT_EVAL}, {period}) — RMSE_앙상블: 모델 줄 = 시드 평균 예측, "
              "ENSEMBLE 줄 = 같은 loss 세 모델(시드 평균) 평균 예측, Baseline 줄 = 기준선")
    return full


def final_seed_file():
    """최종 모델(run_final)의 시드별 지표 파일 — 현재 조건의 결과 폴더."""
    return C.RESULT_DIR / "final_seed_results.csv"


def predict_saved(fold, mod, params, G, weights, t0):
    """저장한 가중치로 발행 시각 t0를 예측 (kW). 정규화는 학습 때의 fold 그대로 (기준과 모델을 한 쌍으로)."""
    spec = T.make_spec(fold, mod, params["lookback"], G["horizon"], G["block"])
    model = mod.build(params, G["horizon"]).to(DEVICE)
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    pred, _ = T.predict(model, fold, spec, t0)
    return fold.to_kw(pred)


def run_final(df, best, months=None, reuse_predictions=False):
    """8월 직전까지 재학습한 모델로 요청한 달만 예측·평가한다. 테스트에서 재학습하지 않는다."""
    months = list(C.CONFIRM_MONTHS if months is None else months)
    if best.empty or not C.CONFIRM_MONTHS or not months:
        raise ValueError('최종 모델 설정과 평가 월이 필요합니다.')
    confirm = C.CONFIRM_MONTHS[0]
    val_start, val_end, es_start = split(df, confirm)
    fold = D.Fold(df, val_start, DEVICE)
    fold_es = D.Fold(df, es_start, DEVICE)
    model_dir = C.RESULT_DIR / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for group, part in best.groupby('group'):
        spec = C.GROUPS[group]
        win = windows(df, fold, spec, val_start, val_end, es_start)
        origins = {mo: D.origins(df, mo + '-01', pd.Timestamp(mo + '-01') + pd.offsets.MonthBegin(1),
                                spec['horizon'], spec['eval_stride']) for mo in months}
        for _, selected in part.iterrows():
            module = MODELS[selected['model']]
            params = json.loads(selected['params'])
            for seed in C.SEEDS:
                confirm_path = pred_path(group, selected['model'], selected['loss'], selected['params'], confirm, seed)
                weights = model_dir / ('final_' + confirm_path.stem + '.pt')
                epoch = np.nan
                if not weights.exists():
                    if confirm not in months:
                        raise FileNotFoundError(f'테스트 전에 최종 모델 학습이 완료되어야 합니다: {weights.name}')
                    started = time.time()
                    pred, epoch, _, _ = train_eval(fold, module, params, selected['loss'], spec, win,
                                                  seed, weights, fold_es=fold_es)
                    np.save(confirm_path, pred.cpu().numpy())
                    print(f'[재학습] {group} {selected["model"]} {selected["loss"]} seed={seed}, '
                          f'최고 epoch={epoch}, {time.time()-started:.1f}초', flush=True)
                for month in months:
                    path = pred_path(group, selected['model'], selected['loss'], selected['params'], month, seed)
                    expected_shape = (len(origins[month]), spec['horizon'])
                    cached = None
                    if reuse_predictions and path.exists():
                        cached = np.load(path, allow_pickle=False)
                        if cached.shape != expected_shape or not np.isfinite(cached).all():
                            raise ValueError(f'저장된 예측 배열이 현재 설정과 맞지 않습니다: {path.name}')
                    if cached is None:
                        if month == confirm and not np.isnan(epoch):
                            cached = np.load(confirm_path, allow_pickle=False)
                        else:
                            cached = predict_saved(fold, module, params, spec, weights, origins[month]).cpu().numpy()
                            np.save(path, cached)
                    fi, threshold = targets(fold, origins[month], spec['horizon'])
                    info = {'group': group, 'model': selected['model'], 'loss': selected['loss'],
                            'params': selected['params'], 'fold': month, 'seed': str(seed), 'best_epoch': epoch}
                    rows.extend({**info, **result} for result in score(torch.as_tensor(cached, device=DEVICE), fold, fi, threshold, spec))
    path = final_seed_file()
    updated = pd.DataFrame(rows)
    keys = KEY + ['seed', 'eval_set', 'horizon']
    if path.exists():
        previous = read_csv(path, KEY + ['seed'])
        model_keys = [key for key in KEY if key != 'fold'] + ['seed']
        epochs = previous.dropna(subset=['best_epoch']).drop_duplicates(model_keys).set_index(model_keys)['best_epoch']
        inherited = epochs.reindex(pd.MultiIndex.from_frame(updated[model_keys])).to_numpy()
        updated['best_epoch'] = updated['best_epoch'].fillna(pd.Series(inherited, index=updated.index))
        # Keep the untouched month's record, replace scores for the prediction arrays just evaluated.
        matched = pd.MultiIndex.from_frame(previous[keys]).isin(pd.MultiIndex.from_frame(updated[keys]))
        updated = pd.concat([previous.loc[~matched], updated], ignore_index=True)
    updated.to_csv(path, index=False, encoding='utf-8-sig')
