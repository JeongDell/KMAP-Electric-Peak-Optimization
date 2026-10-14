"""SimpleRNN: 과거 블록을 순서대로 읽어 요약(마지막 상태)을 만들고, 미래 칸마다 같은 작은 층으로 1칸씩 출력.

입력 (모델 특성 반영)
- 과거 블록, 칸마다 22개 = 관측 6 + 달력 14 (요일 7 · 날 종류 4 · 야간 · 시각 sin·cos) + 어제·지난주 같은 칸 전력 2
  기억이 짧아(기울기 소실) 긴 과거를 직접 못 보므로, 어제·지난주 값을 칸마다 붙여 짧은 창 안에서 비교하게 한다.
  달력은 순서대로 읽을 때 각 칸이 몇 시·무슨 요일인지 알려준다.
- 미래 블록, 칸마다 15개 = 달력 14 + 지난주 같은 칸 전력 (세 모델 공통)
- 출력층: 미래 칸마다 [과거 요약 + 그 칸 미래 피처 15개 + 몇 번째 칸인지(0~1)] → 모든 칸이 같은 층을 공유.
  예측 길이(4~384칸)와 상관없이 출력층 가중치 수가 같다 (데이터 대비 모델이 커지지 않게).
- 그리드 후보 값은 config.GRIDS, 넣을 수 있는 이름은 아래 PARAMS (+ 학습 설정 core/train.py의 TRAIN_PARAMS)
"""
import torch
from torch import nn

import config as C
from core.data import CAL, FUTURE, LAGS, OBS

NAME = "RNN"
PAST_COLS = OBS + CAL + LAGS
FUTURE_COLS = FUTURE
GRID = C.GRIDS[NAME]
# 그리드에 넣을 수 있는 모델 설정: 기본값 (None = 그리드에 꼭 있어야 함)
PARAMS = {"hidden": None, "layers": 1, "dropout": 0.0, "head_units": 64}


class Net(nn.Module):
    def __init__(self, n_past, n_future, horizon, p):
        super().__init__()
        self.rnn = nn.RNN(n_past, p["hidden"], num_layers=p["layers"], batch_first=True,
                          dropout=p["dropout"] if p["layers"] > 1 else 0.0)   # 층 사이 드롭아웃 (1층이면 없음)
        self.register_buffer("step", torch.linspace(0, 1, horizon))   # 몇 번째 미래 칸인지
        self.head = nn.Sequential(nn.Linear(p["hidden"] + n_future + 1, p["head_units"]), nn.ReLU(),
                                  nn.Linear(p["head_units"], 1))

    def forward(self, past, future):
        _, h = self.rnn(past)   # 마지막 상태 (층 수, B, hidden) → 마지막 층을 씀
        B, H, _ = future.shape
        x = torch.cat([h[-1][:, None, :].expand(B, H, -1), future, self.step.expand(B, H)[..., None]], dim=2)
        return self.head(x).squeeze(-1)


def build(params, horizon):
    return Net(len(PAST_COLS), len(FUTURE_COLS), horizon, {**PARAMS, **params})
