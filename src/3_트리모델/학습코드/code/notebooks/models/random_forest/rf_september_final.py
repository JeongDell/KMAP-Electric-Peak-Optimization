"""기존 RF 후보 동결 → 1~8월 재학습 → 9월 한 번 평가. 기존 파일 미변경."""
from pathlib import Path
from datetime import datetime
from time import perf_counter
from importlib.metadata import version
from uuid import uuid4
import gc,json,platform,sys,traceback,warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import rf_current_data as common
import rf_grid_search as grid
import rf_future_proxy as proxy
import rf_august_confirmation as august
import rf_medium_long as ml
import rf_input_windows as iw
import boosting_common as bc
import performance_profiles as pp

ROOT=common.ROOT
START=pd.Timestamp('2021-09-01'); END=pd.Timestamp('2021-09-15')
RUNS=ROOT/'artifacts/random_forest/runs'
FEATURE_RUN=RUNS/'20261003T131206_KST_cbba11'
GRID_RUN=RUNS/'20261003T160916_KST_c2f466_rf_grid'
F123_RUN=RUNS/'20261003T165826_KST_d639f0_f123'
WINDOW_RUN=RUNS/'20261003T213322_KST_e06584_rf_input_windows'
PERFORMANCE_RUN=ROOT/'artifacts/performance/runs/20261003T182425_KST_e84d50_performance'
GROUPS=['short_core','short_weekly']+[p+'_native' for p in iw.WINDOWS]+[p+'_pooled' for p in iw.WINDOWS]

def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def spec_for(group):
    profile=group.removesuffix('_native').removesuffix('_pooled') if not group.startswith('short_') else 'short_1h'
    return profile,dict(horizon_steps=4,cadence_steps=1,loss_steps=4) if profile=='short_1h' else dict(ml.PROFILES[profile])

def frozen_manifest():
    """실측 9월을 읽기 전 기존 설정과 출처만 수집. 선정·정렬에 테스트 점수 없음."""
    selected=read(GRID_RUN/'selected_config.json'); windows=read(WINDOW_RUN/'selection.json')
    for key,path in [('source_sha256',common.SOURCE),('shared_code_sha256',common.__file__),('proxy_code_sha256',proxy.__file__),('code_sha256',grid.__file__)]:
        if common.digest(path)!=selected[key]:raise ValueError(f'기존 동결 파일 불일치:{key}')
    candidates=[]
    def add(group,name,kind,design,params,source,scenario='future',role='historical',aliases=None):
        profile,spec=spec_for(group)
        c=read(source) if source else None
        if c and c.get('source_sha256',selected['source_sha256'])!=selected['source_sha256']:raise ValueError(f'기존 데이터 버전 불일치:{source}')
        candidates.append(dict(group=group,profile=profile,spec=spec,candidate=name,kind=kind,design=design,
            params={**params,'n_jobs':4} if kind=='random_forest' else {},scenario_key=scenario,
            scenario=bc.SCENARIOS.get(scenario,'power_only'),role=role,reference=str(source) if source else None,
            reference_sha256=common.digest(source) if source else None,aliases=aliases or []))
    for folder in sorted(GRID_RUN.glob('c[0-9][0-9][0-9]')):
        c=read(folder/'config.json')
        if read(folder/'status.json')['status']!='완료':raise ValueError(f'미완료 그리드:{folder}')
        add('short_core','grid_'+folder.name,'random_forest','grid_'+str(c['spec']['window_hours']),c['rf_params'],folder/'config.json',
            role='cv_primary' if folder.name==selected['candidate'] else ('original_reference' if folder.name=='c001' else 'historical'))
    for feature in ['F0','F1','F2']:
        path=FEATURE_RUN/feature/'config.json';c=read(path)
        add('short_core','observed_'+feature+'_core','random_forest',feature,c['rf_params'],path,'observed')
    for scenario in ['observed','future']:
        for feature in ['F1','F2','F3']:
            path=F123_RUN/f'{scenario}_{feature}'/'config.json';c=read(path)
            aliases=[]
            if scenario=='observed' and feature in ['F2','F3']:
                aliases=[str(FEATURE_RUN/('F2_week_support' if feature=='F2' else 'F3')/'config.json')]
            add('short_weekly',f'{scenario}_{feature}_original','random_forest',f'{scenario}_{feature}',c['params'],path,scenario,aliases=aliases)
    for path in sorted(PERFORMANCE_RUN.glob('short_1h/random_forest/*/config.json')):
        c=read(path)
        add('short_weekly',c['candidate'],'random_forest',f"{c['scenario_key']}_{c['feature_set']}",c['params'],path,c['scenario_key'])
    for profile,values in iw.WINDOWS.items():
        for w in values:
            name=f'w{w:03d}h';path=WINDOW_RUN/profile/name/'config.json';c=read(path)
            aliases=[str(RUNS/'20261003T181854_KST_b27d86_rf_medium_long'/profile/'RF_selected/config.json')] if w==1 else []
            add(profile+'_native',name,'random_forest',name,c['params'],path,
                role='cv_primary' if name==windows['selected'][profile]['candidate'] else ('window_reference' if w==1 else 'historical'),aliases=aliases)
        for path in sorted(PERFORMANCE_RUN.glob(profile+'/random_forest/*/config.json')):
            c=read(path)
            add(profile+'_pooled',c['candidate'],'random_forest',c['feature_set'],c['params'],path,c['scenario_key'])
    for group in GROUPS:
        for name in ['persistence','daily','weekly']:add(group,name,name,name,{},None,scenario='power_only',role='naive_reference')
    identifiers=[(c['group'],c['candidate']) for c in candidates]
    if len(identifiers)!=len(set(identifiers)):raise ValueError('후보 식별키 중복')
    return dict(frozen_at=datetime.now(common.KST).isoformat(),selection_rule='기존4~7월 CV 선정 유지, 9월 재선정 없음',
        september_used_for_selection=False,august_reselection=False,short_primary='grid_'+selected['candidate'],
        native_primary={p:s['candidate'] for p,s in windows['selected'].items()},
        selection_sources={str(p):common.digest(p) for p in [GRID_RUN/'selected_config.json',WINDOW_RUN/'selection.json']},
        n_rf_candidates=sum(c['kind']=='random_forest' for c in candidates),n_records=len(candidates),candidates=candidates)

def load_final_data():
    raw=pd.read_csv(common.SOURCE,parse_dates=['timestamp','timestamp_end'])
    df=raw.loc[raw.timestamp_end<END].set_index('timestamp_end',drop=False).copy()
    common.validate_frame(df)
    if not np.isfinite(df[['power_15min',*common.OBSERVED]].to_numpy(float)).all():raise ValueError('최종 데이터 결측/무한값')
    if not df.groupby(df.timestamp.dt.floor('h'))[common.OBSERVED].nunique().eq(1).all().all():raise ValueError('시간별 외생 입력 반복 불일치')
    if not df.day_type.isin([0,1,2,3]).all():raise ValueError('day_type 범위 오류')
    old=pd.read_csv(august.LEGACY,usecols=['timestamp','power_15min'],parse_dates=['timestamp']).set_index('timestamp').power_15min
    if old.index.has_duplicates:raise ValueError('보정 대조 키 중복')
    prior=old.reindex(df.index)
    if prior.isna().any():raise ValueError('보정 대조 매칭 실패')
    repair=df.power_15min.ne(prior)
    if not prior[repair].eq(0).all():raise ValueError('기존0값 보정 외 전력 변경')
    np.testing.assert_array_equal(df.power_15min[repair],old.reindex(df.index[repair]-pd.Timedelta(days=7)))
    audit=dict(raw_rows=len(raw),raw_columns=len(raw.columns),used_rows=len(df),start_min=str(raw.timestamp.min()),
        end_max=str(raw.timestamp_end.max()),strict_end=str(END),excluded_end_boundary_rows=int((raw.timestamp_end==END).sum()),
        duplicate_timestamp_end=int(raw.timestamp_end.duplicated().sum()),duplicate_full_rows=int(raw.duplicated().sum()),
        dtypes=raw.dtypes.astype(str).to_dict(),missing=raw.isna().sum().to_dict(),
        repaired_power_before_september=int(repair[df.index<START].sum()),repaired_power_in_september=int(repair[df.index>=START].sum()),
        used_september_rows=int((df.index>=START).sum()),timezone='Asia/Seoul assumed local naive timestamps; no arrival-time column',
        zero_error_future_exogenous=True,power_availability='구간 종료 직후 관측 가능 가정')
    return df,repair,audit

def repeat_baseline(df,origins,H,kind):return ml.baseline(df,origins,H,kind)

def prepare_group(df,repair,group):
    if group not in GROUPS:raise ValueError(f'지원하지 않는 비교군:{group}')
    profile,spec=spec_for(group);H=spec['horizon_steps'];df=df.loc[df.index<END].copy()
    repair=repair.reindex(df.index)
    if repair.isna().any():raise ValueError('보정 마스크 키 누락')
    designs={};base=None;groups=None
    if group=='short_core':
        f,y,g=common.build_features(df)
        designs.update({k:f[g[k]] for k in ['F0','F1','F2']})
        for w in [1,4,24]:
            x,yy,cols=grid.build_window_features(df,w);pd.testing.assert_frame_equal(yy,y)
            designs['grid_'+str(w)]=x[cols]
        max_rolling=96;lags={0,4,96}
    elif group=='short_weekly':
        d=bc.build_designs(df);designs={k:v['X'] for k,v in d.items()};y=d['observed_F3']['y']
        max_rolling=96;lags={0,1,2,4,96,672}
        lags.update(96-h for h in range(1,5));lags.update(672-h for h in range(1,5))
    elif group.endswith('_native'):
        idx=df.index[(df.index.hour*60+df.index.minute)%(spec['cadence_steps']*15)==0]
        for w in iw.WINDOWS[profile]:
            x,yy,_=iw.build_features(df,H,w,idx);designs[f'w{w:03d}h']=x;y=yy
        max_rolling=max(iw.WINDOWS[profile])*4;lags={0,4,96}
    else:
        base,groups=pp.base_features(df,profile)
        idx=df.index[(df.index.hour*60+df.index.minute)%(spec['cadence_steps']*15)==0]
        designs={'support':base.reindex(idx)[groups['F3']]}
        y=pd.DataFrame({f'h{h}':df.power_15min.shift(-h).reindex(idx) for h in range(1,H+1)},index=idx)
        max_rolling=1344 if profile=='long_3d' else 96;lags={0,1,2,4,96,672}
        if profile=='long_3d':lags.add(1344);lags.update(1344-h for h in range(1,H+1))
    idx=y.index;complete=np.isfinite(y.to_numpy()).all(axis=1)
    for x in designs.values():complete &= np.isfinite(x.reindex(idx).to_numpy()).all(axis=1)
    cadence=(idx.hour*60+idx.minute)%(spec['cadence_steps']*15)==0
    # Future external values are finite in load_final_data; every target must be complete.
    train=complete&cadence&(idx+pd.Timedelta(minutes=H*15)<START)
    valid=complete&cadence&(idx>=START)&(idx+pd.Timedelta(minutes=H*15)<END)
    for name in ['persistence','daily','weekly']:valid &= np.isfinite(repeat_baseline(df,idx,H,name)).all(axis=1)
    orig=idx[valid];train_orig=idx[train]
    if not len(orig) or not len(train_orig):raise ValueError(f'{group}: 학습/평가 원점 없음')
    rf=repair.rolling(max_rolling,min_periods=1).max().astype(bool)
    for lag in lags:rf |= repair.shift(lag,fill_value=False)
    tags={'has_repaired_rf_input':rf}
    target=pd.Series(False,index=df.index)
    for h in range(1,H+1):target |= repair.shift(-h,fill_value=False)
    tags['has_repaired_target']=target
    for name,period in [('daily',96),('weekly',672)]:
        flag=pd.Series(False,index=df.index)
        for lag in set(period-(np.arange(H)%period+1)):flag |= repair.shift(int(lag),fill_value=False)
        tags['has_repaired_'+name+'_input']=flag
    tags=pd.DataFrame(tags,index=df.index)
    tags['has_repaired_any_input']=tags.has_repaired_rf_input|tags.has_repaired_daily_input|tags.has_repaired_weekly_input
    masks=tags.reindex(orig).copy();masks['keep_full']=True
    masks['keep_clean_targets']=~masks.has_repaired_target
    masks['keep_clean_targets_inputs']=~masks.has_repaired_target&~masks.has_repaired_any_input
    return dict(group=group,profile=profile,spec=spec,df=df,repair=repair,designs=designs,targets=y,
        train_origins=train_orig,valid_origins=orig,masks=masks,base=base,groups=groups,max_rolling_steps=max_rolling)

def prediction_rows(df,origins,actual,predicted):
    H=actual.shape[1];ends=origins.repeat(H)+pd.to_timedelta(np.tile(np.arange(1,H+1)*15,len(origins)),unit='min')
    f=pd.DataFrame(dict(origin=origins.repeat(H),target_end=ends,target_start=ends-pd.Timedelta(minutes=15),
        horizon_minutes=np.tile(np.arange(1,H+1)*15,len(origins)),actual=actual.ravel(),predicted=predicted.ravel()))
    f['is_night']=((f.target_start.dt.hour>=18)|(f.target_start.dt.hour<9)).astype(int)
    f['is_workday']=df.day_type.reindex(ends).eq(0).to_numpy().astype(int)
    f['production']=df['생산량'].reindex(ends).to_numpy()
    return f

def align_paths(frame,origins,H):
    key=['origin','horizon_minutes'];f=frame.copy()
    for c in ['origin','target_end']:f[c]=pd.to_datetime(f[c])
    if f.duplicated(key).any():raise ValueError('발행 예측 키 중복')
    expected=pd.MultiIndex.from_product([origins,np.arange(1,H+1)*15],names=key)
    if len(f)!=len(expected):raise ValueError('예측 셀 수 불일치')
    f=f.set_index(key).reindex(expected)
    end=expected.get_level_values(0)+pd.to_timedelta(expected.get_level_values(1),unit='min')
    if not np.array_equal(f.target_end.to_numpy(),end.to_numpy()) or not np.isfinite(f[['actual','predicted']].to_numpy()).all():
        raise ValueError('예측 목표시점 대응/유한값 오류')
    return f.actual.to_numpy().reshape(-1,H),f.predicted.to_numpy().reshape(-1,H)

def score_paths(a,p,W):return ml.losses(a,p,W)

def evaluation_tables(c,data,frame):
    H=c['spec']['horizon_steps'];W=c['spec']['loss_steps'];orig=data['valid_origins']
    a,p=align_paths(frame,orig,H);out={k:[] for k in ['metrics','horizons','conditions','peaks','bands','dates','direction']}
    for scope in ['full','clean_targets','clean_targets_inputs']:
        keep=data['masks']['keep_'+scope].to_numpy(bool);aa,pr=a[keep],p[keep]
        tag={k:c[k] for k in ['group','profile','candidate','scenario','role']};tag['scope']=scope
        if not keep.any():
            if scope=='full':raise ValueError('전체 평가 원점이 없습니다.')
            out['metrics'].append(dict(**tag,n_train=len(data['train_origins']) if c['kind']=='random_forest' else 0,
                n_comparison_train=len(data['train_origins']),n_excluded_origins=len(orig),n_unique_target_times=0,
                n_origins=0,n_predictions=0,n_loss_windows=0,tail_windows=0,
                **{key:np.nan for key in grid.METRICS},scope_status='평가 불가',reason='공통 보정 제외 후 유효 경로0개'))
            continue
        part=frame.loc[np.repeat(keep,H)]
        out['metrics'].append(dict(**tag,n_train=len(data['train_origins']) if c['kind']=='random_forest' else 0,
            n_comparison_train=len(data['train_origins']),n_excluded_origins=len(orig)-int(keep.sum()),
            n_unique_target_times=part.target_end.nunique(),**score_paths(aa,pr,W)))
        for h in range(H):
            e=aa[:,h]-pr[:,h];out['horizons'].append(dict(**tag,horizon_minutes=15*(h+1),n_origins=len(aa),
                Loss_A=np.sqrt(np.mean(e**2)),MAE=np.abs(e).mean(),L_peak=np.sqrt(np.mean(np.maximum(0,e)**2))))
        for dim,vals in [('is_night',[0,1]),('is_workday',[0,1]),('weekday',range(7)),('production_active',[0,1])]:
            for value in vals:
                mask=(part.target_start.dt.dayofweek.eq(value) if dim=='weekday' else (part.production.gt(0).eq(bool(value)) if dim=='production_active' else part[dim].eq(value)))
                chosen=part.loc[mask]
                if len(chosen):
                    e=chosen.actual.to_numpy()-chosen.predicted.to_numpy()
                    out['conditions'].append(dict(**tag,dimension=dim,condition=int(value),n_predictions=len(chosen),
                        n_unique_target_times=chosen.target_end.nunique(),Loss_A=np.sqrt(np.mean(e**2)),MAE=np.abs(e).mean(),
                        L_peak=np.sqrt(np.mean(np.maximum(0,e)**2)),mean_signed_error=-e.mean()))
        for grouping,width in [('path',H),('loss_window',W)]:
            out['peaks'].append(dict(**tag,grouping=grouping,window_minutes=15*width,**ml.peak_scores(aa.reshape(-1,width),pr.reshape(-1,width))))
        width=96 if H==288 else (16 if H>4 else 4)
        for begin in range(0,H,width):
            e=aa[:,begin:begin+width]-pr[:,begin:begin+width]
            out['bands'].append(dict(**tag,lead_start_minutes=15*begin,lead_end_minutes=15*min(H,begin+width),n_predictions=e.size,
                Loss_A=np.sqrt(np.mean(e**2)),MAE=np.abs(e).mean(),L_peak=np.sqrt(np.mean(np.maximum(0,e)**2))))
        for date,chosen in part.groupby(part.target_start.dt.date):
            e=chosen.actual.to_numpy()-chosen.predicted.to_numpy()
            out['dates'].append(dict(**tag,date=str(date),n_predictions=len(chosen),n_unique_target_times=chosen.target_end.nunique(),
                Loss_A=np.sqrt(np.mean(e**2)),MAE=np.abs(e).mean(),L_peak=np.sqrt(np.mean(np.maximum(0,e)**2)),mean_signed_error=-e.mean()))
        e=pr-aa
        out['direction'].append(dict(**tag,n_predictions=e.size,mean_signed_error=e.mean(),over_RMSE=np.sqrt(np.mean(np.maximum(0,e)**2)),
            under_RMSE=np.sqrt(np.mean(np.maximum(0,-e)**2)),over_fraction=np.mean(e>0),under_fraction=np.mean(e<0)))
    return {k:pd.DataFrame(v) for k,v in out.items()}

def write_record(suite,c,status,started,ended=None,elapsed=None,error=None,fit=0):
    folder=suite/c['group']/c['candidate'];rid=suite.name+'_'+c['group']+'_'+c['candidate']
    path=ROOT/'experiments/random_forest'/f'{rid}.md';path.parent.mkdir(exist_ok=True)
    m=pd.read_csv(folder/'metrics.csv') if (folder/'metrics.csv').exists() else None
    table=common.md_table(m[['scope','n_origins','n_predictions','Loss_A','L_peak','Loss_B','Loss_C']]) if m is not None else '미평가.'
    cfg=read(suite/'suite_config.json')
    path.write_text(f'''# 실험 {rid}

- 상태: {status}
- 목적과 가설: 기존 {c['group']}/{c['candidate']} 설정의 9월 최종 성능을 확인. 테스트로 선정하지 않음. 역할={c['role']}.
- 시작·종료 시각 및 시간대: {started} / {ended or '미완료'}, Asia/Seoul.
- 모델·라이브러리 버전: {c['kind']}; Python {platform.python_version()}; {cfg['versions']}.
- 데이터 위치·버전 또는 해시·사용 범위: {common.SOURCE}, SHA256 {cfg['source_sha256']}; 보정 대조 {august.LEGACY}, SHA256 {cfg['legacy_sha256']}; 학습1~8월, 평가9월1~14일 strict 경계. 원본 미변경.
- 목표값·단위·예측 구간: power_15min kW; H={c['spec']['horizon_steps']*15}분,15분 원경로, 원점 갱신{c['spec']['cadence_steps']*15}분.
- 학습·검증·테스트 분할과 누수 점검: 학습 전체 정답 끝<2021-09-01; 테스트 원점>=2021-09-01 및 전체 경로 끝<2021-09-15. 후보 공통 학습·평가 지원 및 보정 마스크는 group/split.json·origin_masks.csv. 월내 모델 고정, 발행경로 미래 전력 재주입 없음. 대표 선정은 기존CV만,9월 재선정 없음.
- 전처리·특성·하이퍼파라미터·seed: 설계={c['design']}; 파라미터={c['params']}; scaler 없음, 원단위. 시나리오={c['scenario']}; future는사후 실측 외생=예보오차0. pooled는 최대60000행 원점별균등무복원 seed42. CPU worker만4로 통일.
- 이전 또는 기준 실험 링크와 변경 사항: 출처 {c['reference']}, SHA256 {c['reference_sha256']}; 중복 과거 출처={c['aliases']}. 설정 유지하고 학습기간만1~8월로 확장. 기준선은같은group시간키. [전체 과정 최종 보고서](../../notebooks/models/random_forest/RF_FINAL_REPORT_{suite.name}.md).
- 실행 명령·환경·코드 버전 또는 코드 식별 정보: anaconda3/python.exe -B -X utf8 rf_september_final.py; Git없는workspace. 실제 인터프리터={sys.executable}; code/선정파일 해시는suite_config.json·frozen_manifest.json·code/.
- 실제 실행 시간·자원 사용량(측정한 경우): {elapsed if elapsed is not None else '미완료'}초; 완료RF fit={fit}; 순차,전체5400초 예산. fold.json에 process_peak_rss_bytes(측정가능시).
- 평가 지표·구간별 또는 fold별 결과: Loss_A전체RMSE, L_mean은원점내 W={c['spec']['loss_steps']*15}분 블록RMSE 평균, L_CVaR전체 최악ceil20% 블록평균, L_peak=sqrt(mean(max(0,y-p)^2)),w1. B=.6mean+.4CVaR,C=.5mean+.3CVaR+.2peak. 학습은 squared_error,설계손실은평가만.
- 기준모델 대비 변화와 비교 가능 여부: 동일group/scope시간키만비교. 관측/미래시나리오·native/pooled·학습지원 차이를 구분. 테스트 최고 후보를 새선정값으로 취급하지 않음.
- 오차가 큰 조건·해석·한계: horizon/condition/date/lead/peak 표. 9월14일,장기11개중첩경로; 독립표본이나 계절일반화 근거아님. 보정전력은참값복구가 아님. 관측도착지연 자료없음. L_peak는전체과소예측항이지임계피크경보 지표가아님.
- 오류·경고·실패 또는 중단 원인: {error or '없음'}; Python경고는fold.json.
- 예측·모델·표·그래프·로그의 저장 경로: {folder}; predictions.csv.gz, model.joblib(RF),config/status/fold JSON,7개평가표,run.log. 원점/train/sampling키는group폴더.
- 결론과 다음 실험: 아래는실제 평가값. 설정동결후평가이며 추가튜닝없음. 전체보고서에서 기존단기선정·중장기선정및1h참조를보고.

{table}
''',encoding='utf-8')
    dump(folder/'status.json',dict(status=status,started=started,ended=ended,elapsed_seconds=elapsed,error=error,n_rf_fits=fit))
    return path

def run_final(budget_seconds=5400,resume_suite=None):
    before=perf_counter();previous_elapsed=0.
    if resume_suite:
        suite=Path(resume_suite);cfg=read(suite/'suite_config.json');manifest=read(suite/'frozen_manifest.json')
        if cfg['status'] not in ['실패','중단']:raise ValueError('실패/중단 실행만 이어서 평가합니다.')
        if common.digest(suite/'frozen_manifest.json')!=cfg['manifest_sha256']:raise ValueError('동결 설정 변경')
        for p,sha in {**cfg['protected_code_sha256'],**manifest['selection_sources'],cfg['source']:cfg['source_sha256'],cfg['legacy']:cfg['legacy_sha256']}.items():
            if common.digest(p)!=sha:raise ValueError(f'재개 시 기존 파일 변경:{p}')
        previous_elapsed=cfg['elapsed_seconds'];history=suite/'execution_history';history.mkdir(exist_ok=True)
        phase=len(list(history.glob('suite_phase_*.json')))+1
        dump(history/f'suite_phase_{phase}.json',cfg)
        revision=suite/'code'/f'rf_september_final_revision_{phase}.py';revision.write_bytes(Path(__file__).read_bytes())
        cfg.setdefault('execution_revisions',[]).append(dict(resumed_at=datetime.now(common.KST).isoformat(),code_path=str(revision),code_sha256=common.digest(revision),
            reason='빈 보정 민감도 범위는 표본0·NaN 평가불가로 보고. 동결 설정/예측/학습 변경 없음.',previous_status=cfg['status'],previous_error=cfg.get('error')))
        cfg['status']='실행 중';cfg.pop('error',None);dump(suite/'suite_config.json',cfg)
    else:
        suite=RUNS/(datetime.now(common.KST).strftime('%Y%m%dT%H%M%S_KST')+'_'+uuid4().hex[:6]+'_rf_september_final')
        suite.mkdir(parents=True,exist_ok=False)
        manifest=frozen_manifest();dump(suite/'frozen_manifest.json',manifest)
        protected=[common.__file__,grid.__file__,proxy.__file__,august.__file__,ml.__file__,iw.__file__,bc.__file__,pp.__file__]
        cfg=dict(suite_id=suite.name,status='실행 중',started=datetime.now(common.KST).isoformat(),budget_seconds=budget_seconds,
            source=str(common.SOURCE),source_sha256=common.digest(common.SOURCE),legacy=str(august.LEGACY),legacy_sha256=common.digest(august.LEGACY),
            manifest_sha256=common.digest(suite/'frozen_manifest.json'),code_sha256=common.digest(__file__),
            protected_code_sha256={str(p):common.digest(p) for p in protected},versions={p:version(p) for p in ['numpy','pandas','scikit-learn','joblib','nbclient','nbformat','matplotlib']},
            interpreter=sys.executable,september_used_for_selection=False,tuning_after_test=False,n_planned_rf_fits=manifest['n_rf_candidates'],n_planned_records=manifest['n_records'])
        dump(suite/'suite_config.json',cfg);code=suite/'code';code.mkdir()
        for p in [__file__,*protected]:(code/Path(p).name).write_bytes(Path(p).read_bytes())
    print(f'SUITE {suite}\nFROZEN RF={manifest["n_rf_candidates"]}, RECORDS={manifest["n_records"]}',flush=True)
    # This is the first numerical read of September in this final execution.
    df,repair,audit=load_final_data();dump(suite/'data_audit.json',audit)
    df.loc[repair,['timestamp','timestamp_end','power_15min']].to_csv(suite/'repaired_power_times.csv',index=False,encoding='utf-8-sig')
    results=[];records=[];completed=len(list(suite.glob('*/*/model.joblib'))) if resume_suite else 0
    try:
        for group in GROUPS:
            data=prepare_group(df,repair,group);orig=data['valid_origins'];train=data['train_origins'];H=data['spec']['horizon_steps']
            folder=suite/group;folder.mkdir(exist_ok=bool(resume_suite))
            if (folder/'origin_masks.csv').exists():
                old=pd.read_csv(folder/'origin_masks.csv',parse_dates=['origin']).set_index('origin')
                pd.testing.assert_frame_equal(old,data['masks'].rename_axis('origin'),check_freq=False)
                np.testing.assert_array_equal(pd.read_csv(folder/'train_origins.csv',parse_dates=['origin']).origin,train)
            else:
                data['masks'].to_csv(folder/'origin_masks.csv',index_label='origin',encoding='utf-8-sig')
                pd.DataFrame({'origin':train}).to_csv(folder/'train_origins.csv',index=False,encoding='utf-8-sig')
            dump(folder/'split.json',dict(group=group,profile=data['profile'],spec=data['spec'],n_train=len(train),n_valid=len(orig),
                train_first=str(train[0]),train_last=str(train[-1]),train_label_max=str(train[-1]+pd.Timedelta(minutes=H*15)),
                valid_first=str(orig[0]),valid_last=str(orig[-1]),valid_label_max=str(orig[-1]+pd.Timedelta(minutes=H*15)),
                max_rolling_steps=data['max_rolling_steps'],scopes={k:int(data['masks'][k].sum()) for k in data['masks'] if k.startswith('keep_')}))
            print(f'SPLIT {group}: train={len(train)}, test={len(orig)}, H={H}',flush=True)
            if group.endswith('_pooled'):
                leads=pp.training_pairs(train,H);np.save(folder/'training_leads.npy',leads)
                keys=pd.DataFrame(dict(origin=train.repeat(leads.shape[1]),horizon_step=leads.ravel()))
                keys.to_csv(folder/'training_pairs.csv.gz',index=False,compression='gzip',encoding='utf-8-sig')
            for c in [c for c in manifest['candidates'] if c['group']==group]:
                dest=folder/c['candidate'];rec=ROOT/'experiments/random_forest'/f'{suite.name}_{group}_{c["candidate"]}.md'
                if dest.exists() and read(dest/'status.json')['status']=='완료':
                    records.append(rec);results.append(pd.read_csv(dest/'metrics.csv').iloc[0].to_dict());continue
                recovering=dest.exists();old_status=read(dest/'status.json') if recovering else None
                if recovering:
                    if read(dest/'config.json')!=c:raise ValueError('재개 후보 동결 설정 불일치')
                    history=suite/'execution_history'/f'{group}_{c["candidate"]}';history.mkdir(exist_ok=False)
                    for p in [dest/'status.json',dest/'run.log',rec]:(history/p.name).write_bytes(p.read_bytes())
                    dump(history/'saved_predictions_and_model_sha256.json',{p.name:common.digest(p) for p in [dest/'model.joblib',dest/'predictions.csv.gz'] if p.exists()})
                else:dest.mkdir(exist_ok=False)
                t0=perf_counter();started=old_status['started'] if old_status else datetime.now(common.KST).isoformat()
                dump(dest/'config.json',c);rec=write_record(suite,c,'실행 중',started);records.append(rec);fit=0
                try:
                    if previous_elapsed+perf_counter()-before>=budget_seconds:raise TimeoutError('최종 평가 학습 예산 소진')
                    a=data['targets'].reindex(orig).to_numpy(float);warnings_seen=[];n_scalar=len(train)
                    restored_model=recovering and (dest/'model.joblib').exists() and (dest/'predictions.csv.gz').exists()
                    if restored_model:
                        frame=pd.read_csv(dest/'predictions.csv.gz',parse_dates=['origin','target_start','target_end'])
                        saved_a,p=align_paths(frame,orig,H);np.testing.assert_array_equal(a,saved_a)
                        model=joblib.load(dest/'model.joblib')
                        for key,value in c['params'].items():
                            if model.get_params()[key]!=value:raise ValueError('저장 RF 설정 불일치')
                        features=list(model.feature_names_in_);fit=1;del model;gc.collect()
                    elif c['kind']=='random_forest':
                        if group.endswith('_pooled'):
                            xt=pp.build_inputs(df,train,data['profile'],c['design'],c['scenario_key'],leads,data['base'],data['groups'])
                            xv=pp.build_inputs(df,orig,data['profile'],c['design'],c['scenario_key'],base=data['base'],groups=data['groups'])
                            yt=pd.Series(np.take_along_axis(data['targets'].reindex(train).to_numpy(),leads-1,axis=1).ravel(),index=xt.index);n_scalar=len(xt)
                        else:
                            xt=data['designs'][c['design']].reindex(train).astype(np.float32)
                            xv=data['designs'][c['design']].reindex(orig).astype(np.float32)
                            yt=data['targets'].reindex(train)
                        if not xt.index.equals(yt.index) or list(xt.columns)!=list(xv.columns):raise ValueError('X/y/열 대응 불일치')
                        if not np.isfinite(xt.to_numpy()).all() or not np.isfinite(yt.to_numpy()).all():raise ValueError('학습 입력 결측')
                        model=RandomForestRegressor(**c['params'])
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always');model.fit(xt,yt);p=model.predict(xv).reshape(len(orig),H)
                        warnings_seen=[str(w.message) for w in caught];fit=1;completed+=1
                        joblib.dump(model,dest/'model.joblib',compress=3);features=list(xt.columns)
                        del model,xt,xv,yt;gc.collect()
                    else:p=repeat_baseline(df,orig,H,c['kind']);features=[]
                    if not restored_model:
                        frame=prediction_rows(df,orig,a,p);frame['actual_was_repaired']=repair.reindex(frame.target_end).to_numpy()
                        frame.assign(group=group,profile=c['profile'],candidate=c['candidate'],scenario=c['scenario']).to_csv(dest/'predictions.csv.gz',index=False,compression='gzip',encoding='utf-8-sig')
                    tables=evaluation_tables(c,data,frame)
                    for name,table in tables.items():table.to_csv(dest/(name+'.csv'),index=False,encoding='utf-8-sig')
                    try:
                        import psutil
                        peak_rss=getattr(psutil.Process().memory_info(),'peak_wset',None)
                    except ImportError:peak_rss=None
                    elapsed=perf_counter()-t0+(old_status.get('elapsed_seconds',0) if restored_model else 0)
                    dump(dest/'fold.json',dict(features=features,n_train=len(train),n_train_scalar_rows=n_scalar,n_valid=len(orig),
                        elapsed_seconds=elapsed,process_peak_rss_bytes=peak_rss,warnings=warnings_seen,restored_existing_model=restored_model))
                    write_record(suite,c,'완료',started,datetime.now(common.KST).isoformat(),elapsed,fit=fit)
                    if recovering:
                        with rec.open('a',encoding='utf-8') as note:note.write(f'\n## 실행 이력\n\n첫 평가 중 {old_status.get("error")}로 중단. 저장 모델·예측을 그대로 보존해 점수 보고만 복구했다. [첫 시도 기록](../../artifacts/random_forest/runs/{suite.name}/execution_history/{group}_{c["candidate"]}/{rec.name}). 추가 모델 적합0회, 동결 설정 변경 없음.\n')
                    row=tables['metrics'].iloc[0].to_dict();results.append(row)
                    line=f'DONE {group}/{c["candidate"]}: RMSE={row["Loss_A"]:.4f}, peak={row["L_peak"]:.4f}, {elapsed:.1f}s'
                    with (dest/'run.log').open('a',encoding='utf-8') as log:log.write(line+(' (기존모델복구)' if restored_model else '')+'\n')
                    print(line,flush=True)
                    pd.DataFrame(results).to_csv(suite/'progress.csv',index=False,encoding='utf-8-sig')
                except Exception as exc:
                    write_record(suite,c,'중단' if isinstance(exc,TimeoutError) else '실패',started,datetime.now(common.KST).isoformat(),perf_counter()-t0,error=f'{type(exc).__name__}: {exc}',fit=fit)
                    (dest/'run.log').write_text(traceback.format_exc(),encoding='utf-8');raise
            del data;gc.collect()
        if common.digest(suite/'frozen_manifest.json')!=cfg['manifest_sha256']:raise ValueError('평가 중 동결 manifest 변경')
        for p,sha in {**manifest['selection_sources'],**cfg['protected_code_sha256']}.items():
            if common.digest(p)!=sha:raise ValueError(f'평가 중 기존 파일 변경:{p}')
        cfg.update(status='완료',ended=datetime.now(common.KST).isoformat(),elapsed_seconds=previous_elapsed+perf_counter()-before,n_completed_rf_fits=completed,n_completed_records=len(records))
    except Exception as exc:
        cfg.update(status='중단' if isinstance(exc,TimeoutError) else '실패',ended=datetime.now(common.KST).isoformat(),elapsed_seconds=previous_elapsed+perf_counter()-before,n_completed_rf_fits=completed,n_completed_records=len(results),error=f'{type(exc).__name__}: {exc}')
        dump(suite/'suite_config.json',cfg);raise
    dump(suite/'suite_config.json',cfg)
    for name in ['metrics','horizons','conditions','peaks','bands','dates','direction']:
        pd.concat([pd.read_csv(suite/c['group']/c['candidate']/(name+'.csv')) for c in manifest['candidates']],ignore_index=True).to_csv(suite/('comparison_'+name+'.csv'),index=False,encoding='utf-8-sig')
    with (ROOT/'experiments/README.md').open('a',encoding='utf-8') as out:
        out.write(f'\n## {suite.name}: RF 9월 동결 최종 평가\n\n1~8월 재학습, 9월1~14일 최종 평가. 테스트로 재선정하지 않음.\n\n')
        for c in manifest['candidates']:
            rec=ROOT/'experiments/random_forest'/f'{suite.name}_{c["group"]}_{c["candidate"]}.md'
            out.write(f'- [{c["group"]}/{c["candidate"]}]({rec.relative_to(ROOT/"experiments").as_posix()}): 완료\n')
    dump(suite/'TRAINING_COMPLETE.json',dict(ended=cfg['ended'],manifest_sha256=cfg['manifest_sha256'],n_fits=completed,n_records=len(records)))
    print(f'FINAL_COMPLETED {suite} RF={completed}, RECORDS={len(records)}, seconds={cfg["elapsed_seconds"]:.1f}',flush=True)
    return suite

if __name__=='__main__':run_final(resume_suite=sys.argv[1] if len(sys.argv)>1 else None)
