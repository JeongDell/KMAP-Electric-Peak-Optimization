"""가동 시간 기준 경보 채점 — ML(우리)·DL 8월, 같은 마진 코드·같은 지표. 합격선은 정하지 않는다.

마진: DL shared/⑤코드/margin.py 함수(prepare·analyze·margin_table·apply_margin)를 수정 없이 사용(7월로 만들고 8월 적용).
지표(사용자 확정 2026-10-07):
  가동 = 맞힐 칸이 속한 시간의 생산량 > 0. 가동 시간 = 8월 전체 15분 칸 중 가동 칸 × 0.25h(예측과 무관한 달력 기준)
  피크 근처 = y ≥ 0.9·b(발행 때) (팀 정의). 경보 = ŷ ≥ (0.9 − m)·b (게이트 변형은 ŷ ≥ 0.9b 또는 (p ≥ τ 이고 ŷ ≥ (0.9 − m)b))
  ① 날 재현율 = 피크 근처 날 중 그날 피크 근처 칸을 어떤 발행이든 경보로 짚은 날
  ② 헛경보 시간 비율 = 헛경보 묶음(경보 시각을 이어 60분 넘게 끊기면 다른 묶음, 묶음 안에 피크 근처 시각 없음)의
     가동 칸 수 × 0.25h ÷ 가동 시간
  ③ 참고: 가동 시간 100시간당 경보 횟수(경보 시각을 60분마다 다시 셈), 비가동 시간 경보 시간, 경보 시간 비율,
     미리 알린 시간(피크 근처 칸마다 그 칸을 경보로 짚은 가장 이른 발행 → 맞힐 칸 시작까지, 중앙값·평균)
  곡선: 기본 마진 m을 0~0.6로 바꿔 ①·② 맞바꿈(설명용, 선택에 쓰지 않음)
사용: python -B -X utf8 notebooks/_shared/m1007_ops_score.py --out <dir>
"""
import argparse
import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import m1007_common as mc  # noqa: E402

ROOT = mc.ROOT
TEAM_CODE = ROOT/'shared'/'⑤코드'
NEAR = .9
GAP = pd.Timedelta(minutes=60)
ML_JULY = ROOT/'reports/20261007T094057_KST_d80d75_m1007_july_selection/predictions_july.csv.gz'
ML_AUG = ROOT/'reports/20261007T102741_KST_m1007_august_margin/predictions_august.csv.gz'
GATE = {'short_1h': '20261007T161906_KST_m1007_margin_gate', 'medium_4h': '20261007T161906_KST_m1007_margin_gate',
        'medium_8h': '20261007T161906_KST_m1007_margin_gate', 'medium_12h': '20261007T163129_KST_m1007_margin_gate_long',
        'medium_24h': '20261007T163129_KST_m1007_margin_gate_long', 'long_4d': '20261007T163129_KST_m1007_margin_gate_long'}
SHORT = {'short_1h': '1h', 'medium_4h': '4h', 'medium_8h': '8h', 'medium_12h': '12h', 'medium_24h': '24h', 'long_4d': '4d'}
LONG = {v: k for k, v in SHORT.items()}


def team_module():
    fake = types.ModuleType('config')
    fake.FOLDS, fake.CONFIRM_MONTH, fake.TEST_MONTH = ['2021-07'], '2021-08', '2021-09'
    fake.RUN_SEPTEMBER, fake.DATA_PATH, fake.RESULT_DIR = False, mc.SOURCE, ROOT/'tmp'
    sys.modules['config'] = fake
    sys.path.insert(0, str(TEAM_CODE))
    return importlib.import_module('margin')


def load_ml():
    parts = []
    for path in [ML_JULY, ML_AUG]:
        p = pd.read_csv(path, usecols=['model', 'horizon', 'origin', 'lead', 'target_time', 'y', 'yhat', 'b_origin'])
        p = p.loc[p.model == '단순평균 앙상블'].copy()
        parts.append(p)
    d = pd.concat(parts, ignore_index=True)
    d = d.assign(horizon=d.horizon.map(SHORT), loss='C', 후보='ML 4모델 평균', 출처='ML')
    return d.rename(columns={'b_origin': 'b_loss'})


def load_dl():
    parts = []
    for h in SHORT.values():
        p = pd.read_csv(TEAM_CODE.parent/'②예측_7_8월'/f'{h}.csv', low_memory=False)
        p = p.loc[p['대표'].notna() & p.period.isin(['7월', '8월'])].copy()
        parts.append(p.assign(후보=p.model + ' ' + p.loss, 출처='DL'))
    d = pd.concat(parts, ignore_index=True)
    return d.rename(columns={'b_origin': 'b_loss'})[['horizon', 'loss', '후보', '출처', 'origin', 'lead', 'target_time', 'y', 'yhat', 'b_loss']]


def operating_slots(month='2021-08'):
    import full_c_data as fd
    dev = fd.load_frame('2021-09-01 00:01').set_index('timestamp')
    slots = dev.loc[dev.index.strftime('%Y-%m') == month]
    prod = slots['생산량'] > 0
    return prod, dev['생산량']


def ops_score(a, alarm, op_slots):
    """a: 한 달·한 구간·한 모델의 칸(origin, target_time, y, yhat, b_loss). alarm: 칸별 불리언."""
    a = a.assign(alarm=np.asarray(alarm, bool), near=a.y >= NEAR*a.b_loss)
    ev = pd.DatetimeIndex(a.loc[a.near, 'target_time'].drop_duplicates().sort_values())
    hit_rows = a.loc[a.near & a.alarm]
    caught = pd.DatetimeIndex(hit_rows.target_time.drop_duplicates())
    ev_days = ev.normalize().unique()
    caught_days = caught.normalize().unique()
    times = pd.DatetimeIndex(a.loc[a.alarm, 'target_time'].drop_duplicates().sort_values())
    op_hours = op_slots.sum()*.25
    is_op = op_slots.reindex(times).fillna(False).to_numpy(bool)
    if len(times):
        gid = np.r_[0, np.cumsum(np.diff(times.asi8) > GAP.value)]
        ev_set = set(ev)
        true_group = pd.Series([t in ev_set for t in times]).groupby(gid).any()
        false_slot = ~true_group.reindex(gid).to_numpy()
        n_groups = int(gid.max() + 1)
        # 회의 문서 그대로: 같은 경보는 60분 안에 한 번만 → 60분이 지나 계속 켜져 있으면 다시 셈
        renotify, start = 0, None
        for t in times:
            if start is None or t - start >= GAP:
                renotify += 1
                start = t
    else:
        false_slot = np.zeros(0, bool)
        n_groups = renotify = 0
    lead = (hit_rows.groupby('target_time').apply(lambda g: (g.target_time.iloc[0] - g.origin.min()).total_seconds()/3600)
            if len(hit_rows) else pd.Series(dtype=float))
    return dict(피크근처_날=len(ev_days), 잡은_날=len(caught_days), 날재현율=len(caught_days)/len(ev_days) if len(ev_days) else np.nan,
                피크근처_칸시각=len(ev), 잡은_칸시각=len(caught), 칸재현율=len(caught)/len(ev) if len(ev) else np.nan,
                가동시간_h=op_hours, 경보시간_h=len(times)*.25, 경보시간_가동중_h=is_op.sum()*.25, 경보시간_비가동_h=(~is_op).sum()*.25,
                헛경보시간_가동중_h=(false_slot & is_op).sum()*.25,
                헛경보시간비율_가동=(false_slot & is_op).sum()*.25/op_hours,
                경보시간비율_가동=is_op.sum()*.25/op_hours,
                경보묶음=n_groups, 헛경보묶음=int(pd.Series(false_slot).groupby(gid).all().sum()) if len(times) else 0,
                경보횟수_60분재통지=renotify, 경보횟수_가동100h당=renotify/op_hours*100,
                미리알림_중앙값_h=float(lead.median()) if len(lead) else np.nan, 미리알림_평균_h=float(lead.mean()) if len(lead) else np.nan)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    team = team_module()
    cond, _ = team.raw_table()
    op_slots, _ = operating_slots()
    reps = pd.concat([load_ml(), load_dl()], ignore_index=True)
    for c in ['origin', 'target_time']:  # 자정이 날짜만 기록된 파일이 있어 혼합 형식으로 읽는다
        reps[c] = pd.to_datetime(reps[c], format='mixed')
    reps = team.prepare(reps, cond)  # r·근처·조건 열(DL 정의)
    tau1 = pd.concat([pd.read_csv(ROOT/'reports'/f/'gate_auc.csv') for f in set(GATE.values())]).set_index('horizon')['tau']
    tau3 = pd.read_csv(ROOT/'reports/20261007T170042_KST_m1007_gate_tau3/tau3_selection_oof.csv').set_index('horizon')['tau']
    rows, curve = [], []
    for (src, h, loss), part in reps.groupby(['출처', 'horizon', 'loss'], sort=False):
        jul = part[part['달'] == '2021-07']
        aug = part[part['달'] == '2021-08'].copy()
        near = jul[jul['근처']]
        base = near['r'].quantile(team.BASE_Q)
        table, chosen = team.analyze(near, base)
        mtab = team.margin_table(near, chosen, base)
        m_cond = team.apply_margin(aug, mtab, base, near, chosen)
        variants = {'마진 없음': aug.yhat >= NEAR*aug.b_loss,
                    '기본 마진': aug.yhat >= (NEAR - base)*aug.b_loss,
                    '조건별 마진': aug.yhat >= (NEAR - m_cond)*aug.b_loss}
        if src == 'ML':
            g = pd.read_csv(ROOT/'reports'/GATE[LONG[h]]/f'gate_cells_aug_{LONG[h]}.csv.gz', usecols=['origin', 'lead', 'gate_p'])
            g['origin'] = pd.to_datetime(g.origin, format='mixed')
            p = aug.merge(g, on=['origin', 'lead'], how='left', validate='one_to_one').gate_p.to_numpy()
            if np.isnan(p).any():
                raise ValueError(f'{h}: 게이트 확률 매칭 실패')
            plain, wide = (aug.yhat >= NEAR*aug.b_loss).to_numpy(), (aug.yhat >= (NEAR - base)*aug.b_loss).to_numpy()
            wide_c = (aug.yhat >= (NEAR - m_cond)*aug.b_loss).to_numpy()
            variants.update({'게이트+기본 마진(처음 τ)': plain | ((p >= tau1[LONG[h]]) & wide),
                             '게이트+기본 마진(τ 3번 규칙)': plain | ((p >= tau3[LONG[h]]) & wide),
                             '게이트+조건별 마진(처음 τ)': plain | ((p >= tau1[LONG[h]]) & wide_c)})
        for name, alarm in variants.items():
            rows.append(dict(출처=src, 구간=h, 손실=loss, 방식=name, 기본마진=base, 고른조건=' + '.join(chosen) or '-',
                             **ops_score(aug, alarm, op_slots)))
        for m in np.round(np.arange(0, .61, .02), 2):
            s = ops_score(aug, aug.yhat >= (NEAR - m)*aug.b_loss, op_slots)
            curve.append(dict(출처=src, 구간=h, 손실=loss, 마진=m, 날재현율=s['날재현율'], 헛경보시간비율_가동=s['헛경보시간비율_가동'],
                              경보시간비율_가동=s['경보시간비율_가동']))
        print(mc.now(), src, h, loss, flush=True)
    pd.DataFrame(rows).to_csv(out/'ops_score_august.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    pd.DataFrame(curve).to_csv(out/'tradeoff_curve_august.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    print(out)


if __name__ == '__main__':
    main()
