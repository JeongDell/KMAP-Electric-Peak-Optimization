"""피크 기준 b와 L_peak 가중치 w — RF '전체 C 직접 학습·공통 연속형2' 설계와 같은 정의 (config.py의 loss 설명).

b[원점] = max(PEAK_FLOOR, 원점까지 이번 달 관측 최대). 원점 = 예측 시작 칸 t0의 시작 시각, 관측 = t0 − 1 칸까지,
  이번 달 = t0가 속한 달 → 그 달 첫 원점(1일 00:00)은 이번 달 관측이 없어 PEAK_FLOOR (t0 − 1이 전달 칸이라 전달 최대를 쓰지 않음).
  그 원점에서 낸 경로 전체(모든 몇 칸 앞)에 같은 b — 경로 중간에 달이 바뀌어도 그대로. 미래 값은 쓰지 않는다.
  보간한 전력 칸(config.OUTAGES)도 데이터에 있는 값 그대로 최댓값 계산에 넣는다.
w = 1 + (W_MAX − 1) × clip((y / b − START) / (1 − START), 0, 1), y = 대상 칸의 실제 전력 (연속형2: START 0.8, W_MAX 2).
"""
import numpy as np
import torch

import config as C


def month_id(df):
    """칸별 달 번호 (연 × 12 + 월)."""
    ts = df["ts"]
    return (ts.dt.year * 12 + ts.dt.month - 1).to_numpy()


def running_max(df):
    """칸마다 '그 달 첫 칸 ~ 그 칸'의 전력 최댓값 (kW, 보간한 칸도 그대로 포함)."""
    return df["power"].groupby(month_id(df)).cummax().to_numpy(np.float32)


def threshold(fold, t0, fi):
    """예측 시작 칸 t0 (B,) · 정답 칸 번호 fi (B, H) → 칸별 피크 기준 b (B, H), kW. 한 경로 안에서는 모두 같은 값."""
    t0 = torch.as_tensor(t0, device=fi.device)
    last = t0 - 1                                       # 마지막으로 관측된 칸
    known = fold.run_max[last].masked_fill(fold.month[last] != fold.month[t0], -np.inf)   # 이번 달(t0의 달) 관측만
    return known.clamp(min=C.PEAK_FLOOR)[:, None].expand(fi.shape)


def weight(y, b):
    """실제 y · 피크 기준 b (같은 모양, kW) → L_peak 가중치 w (1 ~ W_MAX)."""
    start, w_max = C.PEAK_WEIGHT["START"], C.PEAK_WEIGHT["W_MAX"]
    return 1.0 + (w_max - 1.0) * ((y / b - start) / (1.0 - start)).clamp(0.0, 1.0)


def weights(fold, t0, fi):
    """창들의 L_peak 가중치 (B, H) — 학습·조기 종료 확인용."""
    return weight(fold.y_kw[fi], threshold(fold, t0, fi))
