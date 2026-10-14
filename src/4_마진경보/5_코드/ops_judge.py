"""운영 판정 v2 — 단독 실행판 (팀 공유용)

원본: notebooks/_shared/m1007_ops_judge.py (ML, 2026-10-07 확정 · 10-08 p 정의 정정)
기준 문서: docs/meeting_20261007_rerun_plan.md "운영 판정 합격선 추가"
원본과 판정 로직은 같고, 프로젝트 내부 로더(full_c_data, m1007_common) 의존만 뺐다.
달력(전력·생산량)은 build_calendar()에 직접 넘긴다.
(프로젝트로 가져온 사본, 2026-10-08. 로직 수정 없음)

정의
  피크 근처 = y ≥ 0.9·b(발행 때). 맞힐 시각이 어느 발행에서든 피크 근처면 피크 근처 시각.
  가동 = 비가동이 아닌 15분 칸. 비가동 = 생산량 0이 4시간(16칸) 이상 이어진 구간(1~3시간 0은 짧은 정지로 가동).
  가동일 = 가동 칸이 있는 날(칸 시작 시각 기준). 복사된 날(96칸 전력이 더 이른 날과 같은 날)은 가동에서 뺀다.
  ① 날 재현율 = 피크 근처 시각이 있는 날 중 그 시각을 어떤 발행이든 경보로 짚은 날 (날짜 = target_time 날짜). 합격 ≥ 80%
  ② 날별 헛경보 = 대상 기준 묶음(경보 맞힐 시각을 이어 60분 넘게 끊기면 다른 묶음, 묶음 안 피크 근처 시각 없음 = 헛경보)을
     묶음 첫 칸 날짜에 센다. 합격 = 가동일의 80% 이상에서 그날 헛경보 ≤ 3
  ③ 헛경보 시간 비율 = 경보가 켜졌는데 피크 근처가 아닌 가동 칸 수 ÷ 가동 칸 수 (15분 칸 단위, 묶음 사이 빈 시간 미포함)
     합격 = ≤ p + 0.10. p = 평가 달 가동 칸 중, 이 경보 칸들에서 피크 근처(위 정의)인 시각의 비율 → 구간마다 다름
  운영판정_합격 = ① 그리고 ② 그리고 ③. 여러 달은 counts를 더해 pooled()로 다시 계산.
  판정 시간대: 하루 전체(밤·일요일·공휴일 포함). 요금 시간대 제한 없음.
"""
import numpy as np
import pandas as pd

NEAR, GAP, OFF_SLOTS = .9, pd.Timedelta(minutes=60), 16
MAX_FALSE_PER_DAY, DAY_SHARE, SLACK, PASS_RECALL = 3, .8, .10, .8
SLOT = pd.Timedelta(minutes=15)
_CAL = None


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


def build_calendar(start, power, production, end=None):
    """start: 칸 시작 시각, power: 15분 전력(kW), production: 생산량. end(배타, 칸 시작 기준)가 있으면 그 전까지만."""
    global _CAL
    start = pd.DatetimeIndex(pd.to_datetime(start)).as_unit('ns')
    cal = pd.DataFrame({'y': np.asarray(power, float), 'prod': np.asarray(production, float)}, index=start)
    if end is not None:
        cal = cal[cal.index < pd.Timestamp(end)]
    if cal.index.has_duplicates or not cal.index.is_monotonic_increasing:
        raise ValueError('달력 시각이 중복되거나 정렬되지 않음')
    if not np.isfinite(cal[['y', 'prod']].to_numpy()).all():
        raise ValueError('달력 전력·생산량에 결측/무한값')
    cal['day'], cal['month'] = cal.index.normalize(), cal.index.to_period('M')
    zero = cal['prod'] == 0
    run = (zero != zero.shift()).cumsum()
    off = zero & (cal.groupby(run)['prod'].transform('size') >= OFF_SLOTS)
    cal['copied'] = cal.day.isin(copied_days(cal.y))
    cal['op'] = ~off & ~cal.copied
    prior = cal.groupby('month')['y'].transform(lambda s: s.shift().cummax())
    cal['near_cal'] = cal.y >= NEAR*np.maximum(170., prior.fillna(0.))
    _CAL = cal
    return cal


def calendar():
    if _CAL is None:
        raise RuntimeError('build_calendar()를 먼저 호출해야 함')
    return _CAL


def start_times(a):
    """target_time을 칸 시작 시각(ns)으로. 파일마다 시작/끝 표기가 달라 달력 전력과 y가 맞는 쪽을 고른다."""
    tt = pd.to_datetime(a['target_time'], format='mixed') if a['target_time'].dtype == object else a['target_time']
    tt = pd.Series(pd.DatetimeIndex(tt).as_unit('ns'), index=a.index)
    cal = calendar()
    u = a.assign(_t=tt).drop_duplicates('_t')
    gaps = {}
    for shift in [pd.Timedelta(0), -SLOT]:
        got = cal.y.reindex(pd.DatetimeIndex(u._t + shift))
        gaps[shift] = float(np.nanmax(np.abs(got.to_numpy() - u.y.to_numpy(float)))) if got.notna().any() else np.inf
    shift = min(gaps, key=gaps.get)
    if gaps[shift] > 1e-6:
        raise ValueError(f'target_time 표기 판별 실패: 시작 가정 오차 {gaps[pd.Timedelta(0)]}, 끝 가정 오차 {gaps[-SLOT]}')
    return tt + shift


def judge(a, alarm):
    """a: target_time, y, b 열(b는 발행 때 b; b_loss·b_origin 이름도 받음). alarm: 행별 불리언. 반환: 지표 + 합칠 수 있는 counts."""
    b = a['b'] if 'b' in a else a['b_loss'] if 'b_loss' in a else a['b_origin']
    tt = start_times(a)
    alarm = np.asarray(alarm, bool)
    near_row = (a['y'].to_numpy(float) >= NEAR*b.to_numpy(float))
    ev = pd.DatetimeIndex(tt[near_row].drop_duplicates())
    caught = pd.DatetimeIndex(tt[near_row & alarm].drop_duplicates())
    ev_days, caught_days = ev.normalize().unique(), caught.normalize().unique()

    cal = calendar()
    months = sorted(pd.DatetimeIndex(tt.drop_duplicates()).to_period('M').unique())
    scope = cal[cal.month.isin(months)]
    covered = pd.DatetimeIndex(tt.drop_duplicates()).normalize().unique()
    partial = len(covered) < .9*scope.day.nunique()
    op = scope[scope.op]
    op_days = op.day.unique()

    times = pd.DatetimeIndex(tt[alarm].drop_duplicates().sort_values())
    in_ev = times.isin(ev)
    t_op = scope.op.reindex(times).fillna(False).to_numpy(bool)
    false_slots = int((t_op & ~in_ev).sum())
    per_day = pd.Series(0, index=pd.DatetimeIndex(op_days), dtype=int)
    n_bundles = n_false = n_false_off = 0
    if len(times):
        gid = np.r_[0, np.cumsum(np.diff(times.asi8) > GAP.value)]
        hit = pd.Series(in_ev).groupby(gid).any()
        first = pd.Series(times).groupby(gid).first()
        n_bundles, n_false = len(hit), int((~hit).sum())
        for d in first[~hit].dt.normalize():
            if d in per_day.index:
                per_day[d] += 1
            else:
                n_false_off += 1
    ok_days = int((per_day <= MAX_FALSE_PER_DAY).sum())
    counts = dict(피크근처_날=len(ev_days), 잡은_날=len(caught_days), 가동칸=len(op), 피크근처_가동칸=int(op.index.isin(ev).sum()),
                  피크근처_가동칸_달력=int(op.near_cal.sum()),
                  헛경보_가동칸=false_slots, 경보_가동칸=int(t_op.sum()), 경보_비가동칸=int((~t_op).sum()),
                  가동일=len(op_days), 헛경보3이하_가동일=ok_days, 경보묶음=n_bundles, 헛경보묶음=n_false, 헛경보묶음_비가동일=n_false_off,
                  최악일_헛경보=int(per_day.max()) if len(per_day) else 0,
                  최악일=str(per_day.idxmax().date()) if len(per_day) and per_day.max() > 0 else '')
    return dict(평가달=','.join(str(m) for m in months), 부분기간=partial, **counts, **metrics(counts))


def metrics(c):
    recall = c['잡은_날']/c['피크근처_날'] if c['피크근처_날'] else np.nan
    share = c['헛경보3이하_가동일']/c['가동일'] if c['가동일'] else np.nan
    fp = c['헛경보_가동칸']/c['가동칸'] if c['가동칸'] else np.nan
    p = c['피크근처_가동칸']/c['가동칸'] if c['가동칸'] else np.nan
    p_cal = c['피크근처_가동칸_달력']/c['가동칸'] if c['가동칸'] else np.nan
    limit = p + SLACK
    ok = [recall >= PASS_RECALL, share >= DAY_SHARE, fp <= limit]
    return dict(날재현율=recall, 헛경보3이하_가동일비율=share, 헛경보시간비율=fp, p=p, 헛경보시간_기준=limit, p_달력=p_cal,
                경보시간비율_가동=c['경보_가동칸']/c['가동칸'] if c['가동칸'] else np.nan, 비가동_경보시간_h=c['경보_비가동칸']*.25,
                합격_재현율=ok[0], 합격_날별헛경보=ok[1], 합격_헛경보시간=ok[2], 운영판정_합격=all(ok))


COUNT_KEYS = ['피크근처_날', '잡은_날', '가동칸', '피크근처_가동칸', '피크근처_가동칸_달력', '헛경보_가동칸', '경보_가동칸', '경보_비가동칸',
              '가동일', '헛경보3이하_가동일', '경보묶음', '헛경보묶음', '헛경보묶음_비가동일']


def pooled(rows):
    """같은 후보의 여러 달 결과(judge 반환)를 counts 합으로 다시 판정."""
    rows = list(rows)
    c = {k: sum(r[k] for r in rows) for k in COUNT_KEYS}
    worst = max(rows, key=lambda r: r['최악일_헛경보'])
    return dict(평가달=','.join(r['평가달'] for r in rows), 부분기간=any(r['부분기간'] for r in rows), **c,
                최악일_헛경보=worst['최악일_헛경보'], 최악일=worst['최악일'], **metrics(c))
