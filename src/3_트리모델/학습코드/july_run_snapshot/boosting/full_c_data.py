"""기존 F 피처를 유지하며 전체 미래 경로를 시간순으로 구성한다."""
import numpy as np
import pandas as pd
import boosting_common as bc
import performance_profiles as pp
from full_c_loss import thresholds, weights

PROFILES={'short_1h':(4,1,4,7),'medium_4h':(16,4,4,14),'medium_8h':(32,4,4,14),
          'medium_12h':(48,4,4,14),'medium_24h':(96,4,4,14),'long_4d':(384,96,96,28),
          'long_3d':(288,96,96,28)}
# 2026-10-06 사용자 확정 구간(RF와 동일): 1h·4/8/12/24h·4d. long_3d는 이전 run 재현용으로만 유지.
ACTIVE_PROFILES=['short_1h','medium_4h','medium_8h','medium_12h','medium_24h','long_4d']
FOLDS={'m07':('2021-07-01','2021-08-01'),'m08':('2021-08-01','2021-09-01')}
REPAIR_SOURCE=bc.ROOT/'data/processed/timeseries_all.csv'


def repair_flags(df):
    """RF와 동일한 종료시각 키로 기존 0전력의 전주 대체 여부를 확인한다."""
    old=pd.read_csv(REPAIR_SOURCE,usecols=['timestamp','power_15min'],parse_dates=['timestamp']).set_index('timestamp').power_15min
    if old.index.has_duplicates:raise ValueError('보정 대조 키 중복')
    prior=old.reindex(df.index)
    if prior.isna().any():raise ValueError('보정 대조 매칭 실패')
    repaired=df.power_15min.ne(prior)
    if not prior[repaired].eq(0).all():raise ValueError('기존0값 보정 외 전력 변경')
    np.testing.assert_array_equal(df.power_15min[repaired],old.reindex(df.index[repaired]-pd.Timedelta(days=7)))
    return repaired


def load_frame(end='2021-09-01'):
    df=pd.read_csv(bc.rf.SOURCE,parse_dates=['timestamp','timestamp_end']).set_index('timestamp_end',drop=False)
    bc.rf.validate_frame(df)
    if not np.isfinite(df[['power_15min',*bc.rf.OBSERVED]].to_numpy(float)).all():raise ValueError('원자료 유한 값 필요')
    if not df.groupby(df.timestamp.dt.floor('h'))[bc.rf.OBSERVED].nunique().eq(1).all().all():raise ValueError('시간별 외생 관측 일관성 오류')
    frame=df.loc[df.index<pd.Timestamp(end)].copy()
    frame.attrs['repaired_power']=repair_flags(frame)
    return frame


class Dataset:
    def __init__(self,df,profile,feature='F2',scenario='observed'):
        if profile not in PROFILES or feature not in ['F1','F2','F3'] or scenario not in ['observed','future']:
            raise ValueError('profile/F/scenario 확인 필요')
        self.df=df;self.profile=profile;self.feature=feature;self.scenario=scenario
        self.horizon,cadence,self.width,support=PROFILES[profile];self.direct=profile=='short_1h'
        if self.direct:
            designs=bc.build_designs(df);design=designs[f'{scenario}_{feature}']
            self.base=design['X'];self.groups=None
            self.observed_production=design['full_features']['observed_생산량']
            valid=np.isfinite(design['full_features'][design['support']].to_numpy()).all(axis=1)
        else:
            self.base,self.groups=pp.base_features(df,profile)
            self.observed_production=self.base['observed_생산량']
            valid=np.isfinite(self.base[self.groups['F3']].to_numpy()).all(axis=1)
        H=self.horizon
        windows=np.lib.stride_tricks.sliding_window_view(df.power_15min.to_numpy(float),H+1)[:,1:]
        origins=df.index[:-H]
        valid=valid[:-H]&np.isfinite(windows).all(axis=1)
        valid&=(origins.hour*60+origins.minute)%(cadence*15)==0
        valid&=origins>=df.index[0]+pd.Timedelta(days=support)
        self.origins=origins[valid];self.y=windows[valid].copy()
        flags=df.attrs.get('repaired_power')
        flags=repair_flags(df) if flags is None else flags.reindex(df.index)
        if flags.isna().any():raise ValueError('보정 마스크 키 누락')
        repaired=flags.to_numpy(bool)
        self.has_repaired_target=np.lib.stride_tricks.sliding_window_view(repaired,H+1)[:,1:].any(axis=1)[valid]
        self.b=thresholds(df.power_15min).reindex(self.origins).to_numpy()
        self.w_train=weights(self.y,self.b,self.origins,training=True)
        self.w_eval=weights(self.y,self.b)
        if self.origins.has_duplicates or not len(self.origins):raise ValueError('원점 없음/중복')

    def split(self,fold):
        if fold=='september':start,end=pd.Timestamp('2021-09-01'),pd.Timestamp('2021-09-15')
        else:start,end=map(pd.Timestamp,FOLDS[fold])
        last=self.origins+pd.Timedelta(minutes=15*self.horizon)
        ti=np.flatnonzero(last<start);vi=np.flatnonzero((self.origins>=start)&(last<end))
        if not len(ti) or not len(vi) or np.intersect1d(ti,vi).size:raise ValueError('빈 분할/원점 중복')
        return ti,vi

    def inputs(self,ids):
        origins=self.origins[ids]
        if self.direct:return self.base.loc[origins].astype(np.float32)
        x=pp.build_inputs(self.df,origins,self.profile,self.feature,self.scenario,base=self.base,groups=self.groups)
        expected=np.tile(np.arange(1,self.horizon+1),len(origins))
        if len(x)!=len(origins)*self.horizon or not np.array_equal(x.index.get_level_values('horizon_step'),expected):
            raise ValueError('전체 H 입력 또는 원점/lead 순서 불일치')
        return x
