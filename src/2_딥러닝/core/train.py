"""세 모델 공통: 배치 만들기, loss(A/C), 조기 종료 학습, 예측, 지표."""
import copy
import math

import numpy as np
import torch

import config as C
from core import peak as PK

EPS = 1e-8
LOSSES = ("A", "C")
# 그리드(config.GRIDS)에 넣을 수 있는 학습 설정: 기본값 (None = 그리드에 꼭 있어야 함)
TRAIN_PARAMS = {"lookback": None, "lr": None, "batch_size": C.BATCH_SIZE, "weight_decay": 0.0}
METRICS = ["MAE", "RMSE", "WinRMSE_mean", "WinRMSE_worst20", "WinRMSE_max", "WinRMSE_std",
           "Peak_n", "Peak_under", "Peak_recall", "Peak_precision", "Peak_F1", "LossA", "LossC"]


def make_spec(fold, model_module, lookback, horizon, block):
    """모델 파일의 PAST_COLS·FUTURE_COLS를 열 번호로 바꾼 입력 설정 + 손실 블록 크기 (config.GROUPS의 block)."""
    dev = fold.X.device
    return {
        "past_off": torch.arange(-lookback, 0, device=dev),
        "fut_off": torch.arange(0, horizon, device=dev),
        "past_idx": torch.tensor([fold.cols.index(c) for c in model_module.PAST_COLS], device=dev),
        "fut_idx": torch.tensor([fold.cols.index(c) for c in model_module.FUTURE_COLS], device=dev),
        "block": block,
    }


def make_batch(fold, t0, spec):
    """예측 시작 칸 t0 → 과거 블록 (B, L, 과거 피처), 미래 블록 (B, H, 미래 피처), 정답 칸 번호 (B, H)."""
    t0 = torch.as_tensor(t0, device=fold.X.device)
    pi = t0[:, None] + spec["past_off"]
    fi = t0[:, None] + spec["fut_off"]
    return fold.X[pi][:, :, spec["past_idx"]], fold.X[fi][:, :, spec["fut_idx"]], fi


def risk_loss(pred, y, kind, w, block, mask=None):
    """A = RMSE / C = 0.5·L_mean + 0.3·L_CVaR + 0.2·L_peak.

    block = 예측 구간을 자르는 단위 (단기·중기 1시간 = 4칸, 장기 1일 = 96칸), R = 블록별 RMSE.
    L_mean = R 평균, L_CVaR = 가장 나쁜 20% R 평균, L_peak = √(전체 칸 평균 of w·u²), u = max(0, 실제 − 예측),
    w = 칸별 피크 가중치 (core/peak.py, 연속형2. A에서는 안 씀).
    세 항 모두 RMSE 단위라 정규화 단위로 계산해도 kW 기준과 순서가 같다.
    mask가 있으면 False 칸은 빼고 계산 (학습 = mask 없이 전부, 평가 = 평가 세트의 채점 칸).
    """
    m = torch.ones_like(y) if mask is None else mask.float()
    e2 = (y - pred) ** 2 * m
    n = m.sum().clamp(min=1)
    if kind == "A":
        return torch.sqrt(e2.sum() / n + EPS)
    if kind != "C":
        raise ValueError(f"loss는 {LOSSES} 중 하나: {kind}")
    B, H = pred.shape
    cnt = m.reshape(B, H // block, block).sum(-1)
    ok = cnt > 0
    if not ok.any():
        return pred.sum() * 0
    R = torch.sqrt(e2.reshape(B, H // block, block).sum(-1)[ok] / cnt[ok] + EPS)
    l_mean = R.mean()
    l_cvar = torch.topk(R, max(1, math.ceil(C.CVAR_FRACTION * R.numel()))).values.mean()
    l_peak = torch.sqrt((w * torch.relu(y - pred) ** 2 * m).sum() / n + EPS)
    return 0.5 * l_mean + 0.3 * l_cvar + 0.2 * l_peak


def path_c(pred, y, w, mask, block):
    """경로(행 = 한 원점에서 낸 예측 구간)마다의 C (N,) — risk_loss(kind="C")를 행 하나씩 잰 값과 같다 (유의성 검정용).
    채점 칸이 없는 행은 NaN."""
    m = mask.float()
    N, H = pred.shape
    cnt = m.reshape(N, H // block, block).sum(-1)
    ok = cnt > 0
    R = torch.sqrt(((y - pred) ** 2 * m).reshape(N, H // block, block).sum(-1) / cnt.clamp(min=1) + EPS)
    nv = ok.sum(1)
    l_mean = (R * ok).sum(1) / nv.clamp(min=1)
    k = torch.ceil(nv.double() * C.CVAR_FRACTION).clamp(min=1).long()   # risk_loss의 math.ceil(0.2 × 블록 수)와 같게
    top = R.masked_fill(~ok, -1.0).sort(1, descending=True).values   # 유효 블록이 앞에 (k ≤ 유효 블록 수)
    l_cvar = top.clamp(min=0).cumsum(1).gather(1, (k - 1)[:, None])[:, 0] / k
    n = m.sum(1)
    l_peak = torch.sqrt((w * torch.relu(y - pred) ** 2 * m).sum(1) / n.clamp(min=1) + EPS)
    return torch.where(n > 0, 0.5 * l_mean + 0.3 * l_cvar + 0.2 * l_peak, torch.full_like(n, float("nan")))


def fit(model, fold, spec, kind, params, train_t0, es_t0=None, epochs=None, seed=C.SEED):
    """학습 (모든 칸을 정답으로 사용). params = 그리드 조합 (lr·batch_size·weight_decay, 없으면 TRAIN_PARAMS 기본값).
    seed = 배치 섞는 순서·학습 곡선 표본의 난수 (초기 가중치·드롭아웃은 호출 전 torch.manual_seed로 정함).
    es_t0가 있으면 매 epoch 그 구간에서 같은 loss를 확인 → 좋아지면 가중치 저장,
    PATIENCE번 연속 개선이 없으면 멈추고 저장한 최고 가중치로 되돌린다.
    es_t0가 없으면(검증 직전까지 다시 학습) epochs만큼 학습.

    과적합 확인용으로 매 epoch 학습 손실(학습 창 중 고정 표본 PROBE_SIZE개)과 확인 구간 손실을 같은 방식으로 잰다
    (둘 다 복사된 날을 뺀 칸 = CHECK_EVAL = 중간 평가).
    반환: (최고 epoch, epoch별 기록 [{"epoch", "train_loss", "es_loss"}] — 손실은 kW 단위)
    """
    p = {**TRAIN_PARAMS, **params}
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    rng = np.random.default_rng(seed)
    probe = rng.choice(train_t0, size=min(len(train_t0), C.PROBE_SIZE), replace=False)
    n_batch = max(1, len(train_t0) // p["batch_size"])
    best, best_state, best_ep, wait, history = math.inf, None, 0, 0, []
    for ep in range(1, (epochs or C.MAX_EPOCHS) + 1):
        model.train()
        for idx in np.array_split(rng.permutation(train_t0), n_batch):
            past, fut, fi = make_batch(fold, idx, spec)
            loss = risk_loss(model(past, fut), fold.y[fi], kind, PK.weights(fold, idx, fi), spec["block"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
        val = eval_loss(model, fold, spec, es_t0, kind) if es_t0 is not None else math.nan
        history.append({"epoch": ep, "train_loss": eval_loss(model, fold, spec, probe, kind) * fold.sd,
                        "es_loss": val * fold.sd})
        if es_t0 is None:
            best_ep = ep
            continue
        if val < best:
            best, best_state, best_ep, wait = val, copy.deepcopy(model.state_dict()), ep, 0
        else:
            wait += 1
            if wait >= C.PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_ep, history


def eval_loss(model, fold, spec, t0, kind):
    """t0 창 전체에 대한 loss (정규화 단위, 복사된 날을 뺀 칸 = CHECK_EVAL = 중간 평가)."""
    pred, fi = predict(model, fold, spec, t0)
    return float(risk_loss(pred, fold.y[fi], kind, PK.weights(fold, t0, fi), spec["block"], fold.check[fi]))


def train_rmse(model, fold, spec, t0):
    """학습 구간 전체 RMSE (kW, 복사된 날을 뺀 칸 = CHECK_EVAL) — 과적합 판단용. 학습 구간(2~6월)은 복사된 날이 많아
    그 칸을 넣으면 외운 값이 섞여 학습 RMSE가 실제보다 좋아 보인다 (검증 7·8월은 복사된 날 1일뿐이라 '전체'와 거의 같음)."""
    pred, fi = predict(model, fold, spec, t0)
    e = (fold.to_kw(pred) - fold.y_kw[fi])[fold.check[fi]]
    return float(torch.sqrt((e ** 2).mean()))


@torch.no_grad()
def predict(model, fold, spec, t0, batch=1024):
    """정규화 단위 예측 (N, H)과 정답 칸 번호 (N, H)."""
    model.eval()
    preds, fis = [], []
    for idx in np.array_split(t0, max(1, math.ceil(len(t0) / batch))):
        past, fut, fi = make_batch(fold, idx, spec)
        preds.append(model(past, fut))
        fis.append(fi)
    return torch.cat(preds), torch.cat(fis)


def report(pred_kw, y_kw, mask, peak_kw, horizons, block):
    """kW 단위 (N, H) → 평가 구간(앞부분 h칸)별 지표 dict 목록. 학습 loss와 상관없이 모든 지표를 같은 방식으로 잰다.
    WinRMSE_* = 손실 블록(block)별 RMSE의 통계. 피크 칸 = 실제 ≥ 피크 기준 b(peak_kw, core/peak.py), 피크 경보 = 예측 ≥ b.
    Peak_n = 채점한 피크 칸 수 (적으면 Peak_recall·precision이 크게 흔들림)."""
    pred_kw, y_kw, mask, peak_kw = (x.cpu() for x in (pred_kw, y_kw, mask, peak_kw))
    rows = []
    for name, h in horizons.items():
        p, y, m, thr = pred_kw[:, :h], y_kw[:, :h], mask[:, :h], peak_kw[:, :h]
        e = (y - p)[m].numpy()
        N = p.shape[0]
        cw = m.reshape(N, h // block, block).sum(-1)
        e2w = ((y - p) ** 2 * m).reshape(N, h // block, block).sum(-1)
        R = torch.sqrt(e2w[cw > 0] / cw[cw > 0]).numpy()
        k = max(1, math.ceil(C.CVAR_FRACTION * len(R)))
        peak, hit = (y >= thr) & m, (p >= thr) & m
        tp, n_peak, n_hit = int((peak & hit).sum()), int(peak.sum()), int(hit.sum())
        rows.append({
            "horizon": name, "MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "WinRMSE_mean": R.mean(), "WinRMSE_worst20": np.sort(R)[::-1][:k].mean(),
            "WinRMSE_max": R.max(), "WinRMSE_std": R.std(),
            "Peak_n": n_peak,
            "Peak_under": float(torch.relu(y - p)[peak].mean()) if n_peak else np.nan,   # 피크 칸 과소예측 평균
            "Peak_recall": tp / n_peak if n_peak else np.nan,
            "Peak_precision": tp / n_hit if n_hit else np.nan,
            "Peak_F1": 2 * tp / (n_peak + n_hit) if n_peak + n_hit else np.nan,
            **{f"Loss{kind}": float(risk_loss(p, y, kind, PK.weight(y, thr), block, m)) for kind in LOSSES},
        })
    return rows
