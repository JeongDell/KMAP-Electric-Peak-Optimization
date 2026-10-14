"""N-HiTS: 과거·미래 블록을 펼쳐 MLP 블록 3단(굵은 → 세밀한 해상도)으로 H칸 출력.

각 블록은 과거를 풀링해 보고, 예측을 적은 점으로 낸 뒤 보간한다 (중기 기준 6시간·1시간·15분 해상도).
앞 블록이 설명한 과거 전력(backcast)을 빼고 다음 블록이 나머지를 맡는다.

입력 (모델 특성 반영)
- 과거 블록, 칸마다 6개 = 관측만
  펼친 입력이라 위치가 고정 → 과거 칸의 시각·요일은 미래 블록 첫 칸 달력으로 정해져 과거 달력은 중복.
  어제·지난주 값도 lookback 안에 그 위치로 이미 들어 있다 (672칸이면 지난주 전체).
- 미래 블록, 칸마다 15개 = 달력 14 (요일 7 · 날 종류 4 · 야간 · 시각 sin·cos) + 지난주 같은 칸 전력 (세 모델 공통)
  중기·장기는 1시간(4칸) 평균으로 줄여 넣는다 (펼친 입력이 너무 커지지 않게). 단기 4칸은 그대로.
- 그리드 후보 값은 config.GRIDS, 넣을 수 있는 이름은 아래 PARAMS (+ 학습 설정 core/train.py의 TRAIN_PARAMS)
"""
import torch
from torch import nn
from torch.nn import functional as F

import config as C
from core.data import FUTURE, OBS

NAME = "NHITS"
PAST_COLS = OBS           # 0번 열 = power (backcast 대상)
FUTURE_COLS = FUTURE
GRID = C.GRIDS[NAME]
# 그리드에 넣을 수 있는 모델 설정: 기본값 (None = 그리드에 꼭 있어야 함). 과거 길이는 학습 설정 lookback
PARAMS = {"units": None, "dropout": 0.0}
POOL = [8, 4, 1]          # 블록별 과거 풀링 크기
FREQ = [24, 4, 1]         # 블록별 출력 점 간격: H/24, H/4, H개 점을 보간
FUT_POOL = 4              # 중기·장기 미래 블록 평균 단위 (4칸 = 1시간)


class Block(nn.Module):
    def __init__(self, lookback, horizon, n_past, n_fut_in, pool, freq, units, dropout):
        super().__init__()
        self.lookback, self.horizon = lookback, horizon
        self.pool = nn.MaxPool1d(pool, pool, ceil_mode=True)
        n_in = -(-lookback // pool) * n_past + n_fut_in
        self.n_back, self.n_fore = max(1, lookback // freq), max(1, horizon // freq)
        self.mlp = nn.Sequential(nn.Linear(n_in, units), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(units, units), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(units, self.n_back + self.n_fore))

    def forward(self, past, fut):
        x = self.pool(past.transpose(1, 2)).flatten(1)
        theta = self.mlp(torch.cat([x, fut], dim=1))
        back = F.interpolate(theta[:, None, :self.n_back], size=self.lookback, mode="linear").squeeze(1)
        fore = F.interpolate(theta[:, None, self.n_back:], size=self.horizon, mode="linear").squeeze(1)
        return back, fore


class Net(nn.Module):
    def __init__(self, lookback, horizon, n_past, n_future, units, dropout):
        super().__init__()
        fp = 1 if horizon <= 4 else FUT_POOL
        self.fut_pool = nn.AvgPool1d(fp, fp, ceil_mode=True)
        n_fut_in = -(-horizon // fp) * n_future
        self.blocks = nn.ModuleList(Block(lookback, horizon, n_past, n_fut_in, p, f, units, dropout)
                                    for p, f in zip(POOL, FREQ))

    def forward(self, past, future):
        fut = self.fut_pool(future.transpose(1, 2)).flatten(1)
        resid, others = past[:, :, 0], past[:, :, 1:]
        out = 0
        for block in self.blocks:
            back, fore = block(torch.cat([resid[:, :, None], others], dim=2), fut)
            resid = resid - back
            out = out + fore
        return out


def build(params, horizon):
    p = {**PARAMS, **params}
    return Net(p["lookback"], horizon, len(PAST_COLS), len(FUTURE_COLS), p["units"], p["dropout"])
