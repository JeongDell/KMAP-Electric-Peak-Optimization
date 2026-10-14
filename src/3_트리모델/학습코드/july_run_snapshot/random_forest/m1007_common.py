"""2026-10-07 회의 결정 반영 재실험(7월 검증까지)의 공통 규칙.

- 선정 달: 7월만(학습 = 7/1 이전에 정답 경로가 끝나는 원점). 8·9월은 이번 범위에서 예측하지 않는다.
- 복사된 날: 하루 96칸이 더 이른 날과 완전히 같은 날(팀원 nn_fianal 정의). 부스팅 나무 수 확인 구간에 사용.
- 공통 C: full_c_loss.loss_gradient, 평가 가중치 w = 1+clip((y/b-0.8)/0.2,0,1), 위험창 4칸(4d는 96칸).
- 지시서: docs/meeting_20261007_rerun_plan.md
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
KST = ZoneInfo('Asia/Seoul')
SOURCE = ROOT/'data/processed/preprocessed_20261002/timeseries_15min_v1_previous_week.csv'
PROFILES = ['short_1h', 'medium_4h', 'medium_8h', 'medium_12h', 'medium_24h', 'long_4d']
HORIZON = dict(short_1h=4, medium_4h=16, medium_8h=32, medium_12h=48, medium_24h=96, long_4d=384)
WIDTH = {p: (96 if p == 'long_4d' else 4) for p in PROFILES}
FEATURE_ORDER = ['F1', 'F2', 'F3']
JULY = (pd.Timestamp('2021-07-01'), pd.Timestamp('2021-08-01'))
DATA_END = pd.Timestamp('2021-08-01')  # timestamp_end 기준 마지막 사용 칸(7/31 23:45~24:00)
HOLDOUT_ORIGINAL_DAYS = 14
DECISION_DOC = 'docs/meeting_20261007_rerun_plan.md'


def now():
    return datetime.now(KST).isoformat()


def run_id(tag):
    return datetime.now(KST).strftime('%Y%m%dT%H%M%S_KST_') + hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()[:6] + '_' + tag


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def copied_days(power_by_start):
    """power_by_start: 시작 시각 index의 15분 전력. 96칸이 더 이른 날과 같으면 복사된 날."""
    seen, copied = set(), []
    for day, values in power_by_start.groupby(power_by_start.index.normalize()):
        if len(values) != 96:
            continue
        key = tuple(np.round(values.to_numpy(float), 6))
        if key in seen:
            copied.append(day)
        else:
            seen.add(key)
    return pd.DatetimeIndex(copied)


def holdout_start(train_end, copied, n=HOLDOUT_ORIGINAL_DAYS):
    """train_end(배타) 직전부터 거꾸로 복사 안 된 날 n일이 들어가는 첫 날짜."""
    day = pd.Timestamp(train_end).normalize() - pd.Timedelta(days=1)
    found = 0
    copied = set(pd.DatetimeIndex(copied))
    while found < n:
        if day not in copied:
            found += 1
        if found == n:
            return day
        day -= pd.Timedelta(days=1)


def all_copied_target(origins, horizon, copied):
    """원점별 정답 칸(시작 시각 기준 날짜)이 전부 복사된 날이면 True."""
    starts = pd.DatetimeIndex(origins).to_numpy()[:, None] + np.arange(horizon)[None, :]*np.timedelta64(15, 'm')
    days = pd.DatetimeIndex(starts.ravel()).normalize()
    return np.asarray(days.isin(copied)).reshape(len(origins), horizon).all(axis=1)


def july_split(origins, horizon, start=JULY[0], end=JULY[1]):
    """학습 = 마지막 정답 칸 종료 ≤ 7/1 00:00, 평가 = 원점 ≥ 7/1 이고 마지막 정답 칸 종료 ≤ 8/1 00:00.
    기존 코드(종료 < 경계)는 7/31 23:45~24:00 칸으로 끝나는 원점을 빼서 팀 발행 시각(4d 7월 28개)과 1개 달랐다."""
    idx = pd.DatetimeIndex(origins)
    last = idx + pd.Timedelta(minutes=15*horizon)
    ti = np.flatnonzero(last <= start)
    vi = np.flatnonzero((idx >= start) & (last <= end))
    if not len(ti) or not len(vi):
        raise ValueError('빈 분할')
    return ti, vi


def eval_weights(y, b):
    return 1 + np.clip((np.asarray(y, float)/np.asarray(b, float)[:, None] - .8)/.2, 0, 1)


def score(y, p, b, profile):
    import full_c_loss
    m = full_c_loss.loss_gradient(y, p, eval_weights(y, b), WIDTH[profile])[0]
    return {k: float(m[k]) for k in ['Loss_C', 'RMSE', 'MAE', 'L_mean', 'L_CVaR', 'L_peak', 'under_RMSE', 'bias']}


def long_predictions(origins, y, p, b, **meta):
    """회의 양식: 한 줄 = 한 예측 칸."""
    origins = pd.DatetimeIndex(origins)
    n, h = y.shape
    lead = np.tile(np.arange(1, h + 1), n)
    origin = np.repeat(origins, h)
    frame = pd.DataFrame(dict(origin=origin, lead=lead,
                              target_time=origin + pd.to_timedelta((lead - 1)*15, unit='min'),
                              y=y.ravel(), yhat=p.ravel(), b_origin=np.repeat(b, h)))
    for key, value in reversed(list(meta.items())):
        frame.insert(0, key, value)
    return frame


def write_record(path, fields, table=None):
    """AGENTS.md 8.2 항목 순서의 실험 기록."""
    keys = ['상태', '목적과 가설', '시작·종료 시각 및 시간대', '모델·라이브러리 버전', '데이터 위치·버전 또는 해시·사용 범위',
            '목표값·단위·예측 구간', '학습·검증·테스트 분할과 누수 점검', '전처리·특성·하이퍼파라미터·seed',
            '이전 또는 기준 실험 링크와 변경 사항', '실행 명령·환경·코드 버전 또는 코드 식별 정보',
            '실제 실행 시간·자원 사용량(측정한 경우)', '평가 지표·구간별 또는 fold별 결과', '기준모델 대비 변화와 비교 가능 여부',
            '오차가 큰 조건·해석·한계', '오류·경고·실패 또는 중단 원인', '예측·모델·표·그래프·로그의 저장 경로', '결론과 다음 실험']
    lines = [f"# 실험 {Path(path).stem}", '']
    lines += [f"- {k}: {fields.get(k, '미기록')}" for k in keys]
    if table is not None and len(table):
        lines += ['', pd.DataFrame(table).round(4).to_markdown(index=False)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
