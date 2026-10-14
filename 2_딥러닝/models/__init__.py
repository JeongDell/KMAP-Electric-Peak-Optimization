"""예측 모델 — 파일마다 NAME, PAST_COLS, FUTURE_COLS, GRID(= config.GRIDS), PARAMS(그리드에 넣을 수 있는 모델 설정: 기본값),
build(params, horizon). MODELS = 이름 → 모델 파일 (순서 = 결과 표의 모델 순서)."""
from models import lstm, nhits, rnn

MODELS = {m.NAME: m for m in (rnn, lstm, nhits)}
