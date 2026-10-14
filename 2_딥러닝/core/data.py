"""데이터 읽기 → 피처 → 폴드별 정규화 텐서 → 예측 시작 칸(origin) 목록.

CSV 열 중 학습에 쓰는 것: power_15min(대상·과거 입력), 생산량·기온·습도·풍속·강수량(과거 관측), day(요일), day_type(날 종류),
  is_night, timestamp(하루 중 시각 sin·cos).
쓰지 않는 것과 이유:
  m(월)·d(일)·날짜·timestamp_end — 학습은 1~8월뿐이라 9월 값을 본 적이 없고(월), 날짜 자체는 외우기만 하는 식별자. 계절은 기온이 대신함
  전기요금(계절) — 하루·시간대와 상관없이 달마다 값 하나(1~2월 167.2, 3~5·9월 109.8, 6~8월 191.6) = 계절 표시일 뿐
  시간 — timestamp의 tod_sin·tod_cos(15분 단위)와 같은 정보 / 공장인원 — 그날이 돼야 아는 실측값 / 인건비 — 1년 내내 1.5
"""
import numpy as np
import pandas as pd
import torch

import config as C
from core import peak as PK

DAY_TYPES = {0: "평일", 1: "토요일", 2: "일요일", 3: "공휴일"}   # CSV day_type 값 (공휴일 = 주말과 겹친 공휴일·대체휴일 포함)

# 피처 이름 (모델 파일은 여기서 골라 쓴다)
OBS = ["power", "production", "temp", "humidity", "wind", "rain"]   # 과거에만 아는 관측값 (power = 예측 대상)
CAL = ([f"dow_{i}" for i in range(1, 8)] + [f"daytype_{k}" for k in DAY_TYPES]   # 미래도 미리 아는 달력: 요일 7 + 날 종류 4
       + ["is_night", "tod_sin", "tod_cos"])                                     # (one-hot) + 야간 + 하루 중 시각
LAGS = ["power_lag96", "power_lag672"]   # 어제·지난주 같은 칸 전력
FUTURE = CAL + ["power_lag672"]          # 미래 블록 (세 모델 공통)


def load():
    """CSV → 피처 DataFrame (정규화 전). 0번 행 = 2021-01-01 00:00, 15분 간격."""
    raw = pd.read_csv(C.DATA_PATH, encoding="utf-8-sig")
    ts = pd.to_datetime(raw["timestamp"])
    if not (ts.diff().iloc[1:] == pd.Timedelta(minutes=15)).all():
        raise ValueError("timestamp가 15분 간격으로 이어지지 않음")
    df = pd.DataFrame({"ts": ts, "power": raw["power_15min"].astype(float)})
    df["production"] = np.log1p(raw["생산량"].astype(float))   # 43%가 0이고 한쪽으로 치우침
    df["temp"] = raw["기온"].astype(float)
    df["humidity"] = raw["습도"].astype(float)
    df["wind"] = raw["풍속"].astype(float)
    df["rain"] = np.log1p(raw["강수량"].astype(float))         # 92%가 0
    for i in range(1, 8):                                      # 요일 one-hot (day: 1=월 ~ 7=일)
        df[f"dow_{i}"] = (raw["day"] == i).astype(float)
    if "day_type" not in raw:
        raise ValueError("CSV에 day_type 열이 없음 (0 평일, 1 토요일, 2 일요일, 3 공휴일)")
    bad = sorted(set(raw["day_type"].unique()) - set(DAY_TYPES))
    if bad:
        raise ValueError(f"day_type에 모르는 값 {bad} — 0 평일, 1 토요일, 2 일요일, 3 공휴일만")
    for k in DAY_TYPES:                                        # 날 종류 one-hot (순서 없는 범주라 숫자 하나로 넣지 않음)
        df[f"daytype_{k}"] = (raw["day_type"] == k).astype(float)
    df["is_night"] = raw["is_night"].astype(float)
    slot = ts.dt.hour * 4 + ts.dt.minute // 15                 # 하루 중 칸 번호 0~95
    df["tod_sin"] = np.sin(2 * np.pi * slot / 96)
    df["tod_cos"] = np.cos(2 * np.pi * slot / 96)
    df["power_lag96"] = df["power"].shift(96)
    df["power_lag672"] = df["power"].shift(672)
    if df[OBS + CAL].isna().any().any():
        raise ValueError("입력에 결측이 있음 (v2 파일은 생산량 NaN을 먼저 채워야 함)")
    # 특별한 칸·날 표시 (학습·입력에는 모두 사용, 채점할지는 config.EVAL_SETS가 정함)
    df["outage"] = False
    for a, b in C.OUTAGES:
        df.loc[df["ts"].between(pd.Timestamp(a), pd.Timestamp(b)), "outage"] = True
    df["prod_fix"] = df["ts"].dt.normalize().isin(pd.to_datetime(C.PROD_FIX_DAYS))
    df["copied"] = copied_days(df)
    return df


def eval_mask(df, rule):
    """평가 세트 rule = {"copied": 복사된 날, "prod_fix": 생산량 보간일, "outage": 보간한 전력 칸 — 채점할지} → 채점하는 칸 True."""
    m = np.ones(len(df), bool)
    for k in ("copied", "prod_fix", "outage"):
        if not rule[k]:
            m &= ~df[k].to_numpy()
    return m


def copied_days(df):
    """하루 96칸 전력이 앞의 어떤 날과 완전히 같은 날(복사본)이면 True. 처음 나온 날(원본)은 False.
    9월(테스트)은 들여다보지 않는다."""
    day = df["ts"].dt.normalize()
    seen, dup = set(), set()
    for d, power in df["power"].groupby(day):
        if d >= pd.Timestamp(C.TEST_MONTH + "-01"):
            break
        key = tuple(power)
        if key in seen:
            dup.add(d)
        else:
            seen.add(key)
    return day.isin(dup).to_numpy()


class Fold:
    """fit_end 이전 행으로만 z-score 평균·표준편차를 구해 전체를 변환한 텐서 묶음.
    eval = 평가 세트별 채점 칸, score = 성능 채점 칸 (SELECT_EVAL = 전체),
    check = 중간 평가 칸 (CHECK_EVAL = 복사날제외: 조기 종료 확인·학습 곡선·train_RMSE),
    run_max·month = 피크 기준용 칸별 '그 달 처음 ~ 그 칸' 최댓값·달 번호 (core/peak.py)."""

    def __init__(self, df, fit_end, device):
        fit = df["ts"] < pd.Timestamp(fit_end)
        mean, std = df.loc[fit, OBS].mean(), df.loc[fit, OBS].std()
        for c in LAGS:   # 어제·지난주 전력은 전력과 같은 기준으로
            mean[c], std[c] = mean["power"], std["power"]
        z = (df[OBS + LAGS] - mean) / std
        feats = pd.concat([z, df[CAL]], axis=1).fillna(0.0)   # lag 앞부분 NaN은 TRAIN_FIRST_ORIGIN 이전이라 안 쓰임
        self.cols = list(feats.columns)
        self.X = torch.tensor(feats.to_numpy(np.float32), device=device)
        self.y = self.X[:, self.cols.index("power")]
        self.eval = {name: torch.tensor(eval_mask(df, rule), device=device) for name, rule in C.EVAL_SETS.items()}
        self.score = self.eval[C.SELECT_EVAL]
        self.check_np = eval_mask(df, C.EVAL_SETS[C.CHECK_EVAL])
        self.check = self.eval[C.CHECK_EVAL]
        self.y_kw = torch.tensor(df["power"].to_numpy(np.float32), device=device)
        self.lag_kw = torch.tensor(df["power_lag672"].fillna(0.0).to_numpy(np.float32), device=device)
        self.run_max = torch.tensor(PK.running_max(df), device=device)
        self.month = torch.tensor(PK.month_id(df), device=device)
        self.mu, self.sd = float(mean["power"]), float(std["power"])

    def to_kw(self, z):
        return z * self.sd + self.mu


def origins(df, start, end, horizon, stride):
    """예측 시작 칸 번호: start ≤ 첫 칸, 마지막 칸 < end, 칸 번호가 stride의 배수 (4 = 정각, 96 = 00시)."""
    i0 = int(df["ts"].searchsorted(pd.Timestamp(start)))
    i1 = int(df["ts"].searchsorted(pd.Timestamp(end)))
    t0 = np.arange(i0, i1 - horizon + 1)
    return t0[t0 % stride == 0]


def has_targets(mask, t0, horizon):
    """예측 구간에 채점 칸(mask = Fold.check_np)이 하나라도 있는 창만 남긴다 (조기 종료 확인 창용)."""
    c = np.concatenate([[0], np.cumsum(mask)])
    return t0[c[t0 + horizon] - c[t0] > 0]


def es_start(df, val_start, n_days):
    """조기 종료 구간 시작일: 검증 월 직전에 복사되지 않은 날이 n_days개 들어가도록 거슬러 올라간 날."""
    before = df[df["ts"] < pd.Timestamp(val_start)]
    copied = before.groupby(before["ts"].dt.normalize())["copied"].first()
    return copied.index[~copied.to_numpy()][-n_days]
