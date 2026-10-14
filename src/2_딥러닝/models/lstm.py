"""LSTM: 과거 블록을 순서대로 읽어 요약(마지막 상태)을 만들고, 미래 칸마다 같은 작은 층으로 1칸씩 출력.

입력 (모델 특성 반영)
- 과거 블록, 칸마다 22개 = 관측 6 + 달력 14 (요일 7 · 날 종류 4 · 야간 · 시각 sin·cos) + 어제·지난주 같은 칸 전력 2
  RNN과 똑같이 둬서 RNN과의 차이가 셀 구조(게이트)에서만 나오게 한다.
- 미래 블록, 칸마다 15개 = 달력 14 + 지난주 같은 칸 전력 (세 모델 공통)
- 출력층: RNN과 같음 (미래 칸마다 [과거 요약 + 그 칸 미래 피처 + 몇 번째 칸인지] → 공유 층)
- 과거 길이: RNN보다 길게 (2일, 1일). 게이트 덕분에 더 오래 기억할 수 있다.
- 그리드 후보 값은 config.GRIDS, 넣을 수 있는 이름은 아래 PARAMS (+ 학습 설정 core/train.py의 TRAIN_PARAMS)
"""
import torch
from torch import nn

import config as C
from core.data import CAL, FUTURE, LAGS, OBS

NAME = "LSTM"
PAST_COLS = OBS + CAL + LAGS
FUTURE_COLS = FUTURE
GRID = C.GRIDS[NAME]
# 그리드에 넣을 수 있는 모델 설정: 기본값 (None = 그리드에 꼭 있어야 함)
PARAMS = {"hidden": None, "layers": 1, "dropout": 0.0, "head_units": 64}


class Net(nn.Module):
    def __init__(self, n_past, n_future, horizon, p):
        super().__init__()
        self.lstm = nn.LSTM(n_past, p["hidden"], num_layers=p["layers"], batch_first=True,
                            dropout=p["dropout"] if p["layers"] > 1 else 0.0)   # 층 사이 드롭아웃 (1층이면 없음)
        self.register_buffer("step", torch.linspace(0, 1, horizon))   # 몇 번째 미래 칸인지
        self.head = nn.Sequential(nn.Linear(p["hidden"] + n_future + 1, p["head_units"]), nn.ReLU(),
                                  nn.Linear(p["head_units"], 1))

    def forward(self, past, future):
        _, (h, _) = self.lstm(past)   # 마지막 상태 (층 수, B, hidden) → 마지막 층을 씀
        B, H, _ = future.shape
        x = torch.cat([h[-1][:, None, :].expand(B, H, -1), future, self.step.expand(B, H)[..., None]], dim=2)
        return self.head(x).squeeze(-1)


def build(params, horizon):
    return Net(len(PAST_COLS), len(FUTURE_COLS), horizon, {**PARAMS, **params})
