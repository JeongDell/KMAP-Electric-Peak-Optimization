"""7·8월 검증 전용 전체 C 직접 RF 실험. 가중치 정책은 명시 인자로 요구한다."""
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import argparse
import gc
import json
import os
import platform
import shutil
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits

import rf_direct_c as direct
import rf_peak_probability as previous

ROOT=previous.ROOT
SOURCE_RUN=ROOT/'artifacts/random_forest/runs/20261004T041903_KST_8bcf48_rf_peak_probability'
FOLDS=['m07','m08']
FAMILIES=['rf_mse','c_leaf_d3','c_split_d3','c_leaf_d10','c_split_d10']
ACTIVE_PROFILES=['short_1h','medium_4h','medium_8h','medium_12h','medium_24h','long_4d']
EXTRA_PROFILES={'medium_24h':dict(horizon_steps=96,cadence_steps=4,loss_steps=4),
    'long_4d':dict(horizon_steps=384,cadence_steps=96,loss_steps=96)}
previous.PROFILES.update(EXTRA_PROFILES)


def weight_conditions(policy):
    methods=['binary','continuous'] if policy=='methods_no_rf' else ['binary','continuous','prob_cont']
    return [dict(mode='none',cap=1)]+[dict(mode=m,cap=c) for c in [2,4] for m in methods]


def common_evaluation_weights(y,b,reference,mode):
    y=np.asarray(y,float);b=np.asarray(b,float)
    if mode=='continuous':return 1+np.clip((y/b[:,None]-.8)/.2,0,1)
    return previous.risk.cell_weights(y,b,reference,2)


def support_reference(data):
    """과거 비교와 동일한 지원 원점 mask를 달력으로 만들고 RF 확률값은 쓰지 않는다."""
    origins=data['origins'];h=data['spec']['horizon_steps']
    end=origins.to_period('M').to_timestamp()+pd.offsets.MonthBegin(1)
    known=(origins>=pd.Timestamp('2021-02-01'))&(origins+pd.Timedelta(minutes=15*h)<end)
    marker=np.full(data['targets'].shape,np.nan);marker[known]=0.
    return marker


def evaluate_candidate(config,data,vi,prediction,out,reference,tag):
    actual=data['targets'].iloc[vi].to_numpy();bv=data['threshold'].iloc[vi].to_numpy()
    rows,reliability,horizons,conditions,_=previous.evaluate(data,vi,prediction,out,
        previous.risk.MonotoneCalibrator(),reference[vi],tag)
    eval_w=common_evaluation_weights(actual,bv,reference[vi],config.get('eval_mode','prob_cont'))
    for row in rows:
        scope=row['scope'];keep=np.ones(len(actual),bool) if scope=='full' else (~data['has_repaired_target'][vi] if scope=='clean_targets'
            else ~(data['has_repaired_target'][vi]|data['has_repaired_input'][vi]))
        if not keep.any():continue
        row.update(previous.risk.point_metrics(actual[keep],prediction[keep],eval_w[keep],data['spec']['loss_steps']))
        sensitivity=previous.risk.point_metrics(actual[keep],prediction[keep],1+3*(eval_w[keep]-1),data['spec']['loss_steps'])
        row.update(Loss_C_cap4=sensitivity['Loss_C'],L_peak_cap4=sensitivity['L_peak'])
        for evaluation_mode in ['binary','continuous']:
            for evaluation_cap in [2,4]:
                ew,_=training_weights(actual[keep],bv[keep],reference[vi][keep],evaluation_cap,evaluation_mode)
                score=previous.risk.point_metrics(actual[keep],prediction[keep],ew,data['spec']['loss_steps'])
                row[f'Loss_C_{evaluation_mode}{evaluation_cap}']=score['Loss_C']
        event=actual[keep]>bv[keep,None];estimated=prediction[keep]>=bv[keep,None]
        positive=int(event.sum());negative=int((~event).sum())
        row.update(power_peak_recall=float((event&estimated).sum()/positive) if positive else np.nan,
            power_peak_false_positive_rate=float((~event&estimated).sum()/negative) if negative else np.nan,
            n_power_false_peak=int((~event&estimated).sum()),n_power_missed_peak=int((event&~estimated).sum()),
            n_predicted_power_peak=int(estimated.sum()),n_actual_power_peak=positive,
            power_peak_under_RMSE=float(np.sqrt(np.mean(np.maximum(actual[keep]-prediction[keep],0)[event]**2))) if positive else np.nan)
    return rows,reliability,horizons,conditions,eval_w


def saved_outputs(folder,h):
    frame=pd.read_csv(folder/'predictions.csv.gz');paths=pd.read_csv(folder/'path_predictions.csv.gz')
    n=len(paths)
    out=dict(p_point=frame.p_peak_raw.to_numpy().reshape(n,h),p_period=paths.p_period_raw.to_numpy(),
        p_initial=frame.p_initial_q90.to_numpy().reshape(n,h),p_initial_period=paths.p_initial_period.to_numpy(),
        quantiles=np.stack([frame[f'Q{int(q*100):02}'].to_numpy().reshape(n,h) for q in previous.risk.QUANTILES],axis=-1))
    return frame.predicted.to_numpy().reshape(n,h),out


def training_weights(y,b,reference,cap,mode='prob_cont'):
    y=np.asarray(y,float);p=np.asarray(reference,float)
    if y.ndim!=2 or not y.size or not np.isfinite(y).all() or p.shape!=y.shape:
        raise ValueError('유한 (N,H) 정답과 같은 OOF 확률 shape 필요')
    if np.isinf(p).any() or ((p[np.isfinite(p)]<0)|(p[np.isfinite(p)]>1)).any():
        raise ValueError('기준 확률은0~1 또는미지원NaN이어야 합니다.')
    known=np.isfinite(p).all(axis=1);w=np.ones_like(y)
    if mode not in ['none','binary','continuous','prob_cont'] or not np.isfinite(cap) or cap<1:
        raise ValueError('가중치 방식과 상한 확인 필요')
    if known.any() and mode!='none':
        a=y[known];threshold=np.asarray(b,float)[known,None]
        if not np.isfinite(threshold).all() or (threshold<=0).any():raise ValueError('양수 피크 기준 필요')
        g=np.clip((a/threshold-.8)/.2,0,1)
        factor=(a>threshold).astype(float) if mode=='binary' else g
        if mode=='prob_cont':factor*=((1+p[known])/2)
        w[known]=1+(cap-1)*factor
    return w,known


def split_indices(origins,horizon,start,end):
    train,valid=previous.risk.time_masks(origins,horizon,start,end)
    ti=np.flatnonzero(train);vi=np.flatnonzero(valid)
    if not len(ti) or not len(vi):raise ValueError('학습/검증 구간이 비어 있습니다.')
    if np.intersect1d(ti,vi).size:raise ValueError('학습/검증 원점 중복')
    if not (pd.DatetimeIndex(origins)[ti]+pd.Timedelta(minutes=15*horizon)<start).all():
        raise ValueError('학습 미래 목표가 검증 구간과 중첩됩니다.')
    return ti,vi


def candidate_name(family,cap,policy,mode='prob_cont'):
    if policy in ['methods','methods_no_rf']:return ('w1' if mode=='none' else f'{mode}{cap}')+'_'+family
    return f'cap{cap}_{family}' if policy=='tune' else family


def select_candidates(metrics):
    eligible=metrics.loc[metrics.phase.eq('cv')&metrics.scope.eq('full')&metrics.fold.isin(FOLDS)].copy()
    if eligible.empty:raise ValueError('7·8월 검증 결과가 필요합니다.')
    if eligible.duplicated(['profile','candidate','fold']).any():raise ValueError('선정 검증키 중복')
    selected={}
    for profile,rows in eligible.groupby('profile',sort=False):
        choices=[]
        for candidate,group in rows.groupby('candidate',sort=False):
            if set(group.fold)!=set(FOLDS) or not np.isfinite(group.Loss_C).all():
                raise ValueError(f'{profile}/{candidate}:7·8월 모두 유한 C가 필요합니다.')
            family=next((f for f in FAMILIES if candidate.endswith(f)),candidate)
            choices.append(dict(candidate=candidate,cv_mean_C=float(group.Loss_C.mean()),
                cv_mean_RMSE=float(group.RMSE.mean()),priority=FAMILIES.index(family) if family in FAMILIES else 99))
        choices.sort(key=lambda r:(r['cv_mean_C'],r['priority'],r['candidate']))
        selected[profile]=choices[0]
    return selected


def load_reference(data,source=SOURCE_RUN):
    """과거 cap1 OOF의 원점 ID와 학습 종료를 검증하여 대응시킨다."""
    n=len(data['origins']);h=data['spec']['horizon_steps'];reference=np.full((n,h),np.nan);files=[]
    base=source/'reference'/data['profile']/'reference_cap1' if data['profile'] in EXTRA_PROFILES else source/'cv'/data['profile']/'cap1'
    for folder in sorted(base.glob('m[0-9][0-9]')):
        config=previous.read(folder/'fold_config.json');start=pd.Timestamp(config['start'])
        if pd.Timestamp(config['train_label_max'])>=start:raise ValueError(f'과거 OOF 누수:{folder}')
        with np.load(folder/'oof.npz',allow_pickle=False) as saved:
            origins=pd.to_datetime(saved['origins']);p=saved['p_point'].copy()
            if int(saved['horizon'])!=h or p.shape!=(len(origins),h):raise ValueError(f'과거 OOF schema:{folder}')
        positions=data['origins'].get_indexer(origins)
        if (positions<0).any() or origins.has_duplicates or not (origins>=start).all():
            raise ValueError(f'과거 OOF 원점 대응:{folder}')
        if np.isfinite(reference[positions]).any():raise ValueError('OOF 원점 중복')
        reference[positions]=p
        files.append(dict(path=str(folder/'oof.npz'),sha256=previous.common.digest(folder/'oof.npz'),
            train_label_max=config['train_label_max'],start=config['start']))
    return reference,files


def generate_reference(suite,config,data,record,months,deadline):
    """새 구간의 과거 시간순 cap1 RF 확률을 만들며9월은선정후만호출한다."""
    h=data['spec']['horizon_steps']
    for month in months:
        if perf_counter()>=deadline:raise TimeoutError('기준 OOF 학습 중 전체 예산 소진')
        start=pd.Timestamp(2021,month,1);end=start+pd.offsets.MonthBegin(1)
        if month==9:end=pd.Timestamp('2021-09-15')
        ti,vi=split_indices(data['origins'],h,start,end)
        x=data['features'].iloc[ti].to_numpy(np.float32);y=data['targets'].iloc[ti].to_numpy(float)
        xv=data['features'].iloc[vi].to_numpy(np.float32);actual=data['targets'].iloc[vi].to_numpy(float)
        b=data['threshold'].iloc[vi].to_numpy();before=perf_counter()
        fit_started=datetime.now(previous.common.KST).isoformat()
        forest=RandomForestRegressor(**config['rf_params']).fit(x,y);config['n_new_rf_fits']+=1
        distribution=previous.risk.LeafPathDistribution(forest,x,y);out=distribution.predict(xv,b,initial=config['initial_peak_kw'])
        folder=record['folder']/f'm{month:02}';folder.mkdir(exist_ok=False)
        previous.save_oof(folder,dict(origins=data['origins'][vi],horizon=h,p_point=out['p_point'],p_period=out['p_period'],
            events=actual>b[:,None],clean_targets=~data['has_repaired_target'][vi]))
        fit=dict(fold=f'm{month:02}',fit_started=fit_started,start=str(start),end=str(end),n_train=len(ti),n_valid=len(vi),
            train_label_max=str(data['origins'][ti[-1]]+pd.Timedelta(minutes=15*h)),elapsed_seconds=perf_counter()-before,
            reference_role=True,n_rf_fits=1,features=list(data['features']))
        record['fits'].append(fit);previous.dump(folder/'fold_config.json',fit)
        joblib.dump(dict(forest=forest,features=list(data['features']),spec=data['spec'],training_targets=y,training_features=x),
            folder/'reference_model.joblib',compress=3)
        metric=previous.risk.point_metrics(actual,forest.predict(xv),np.ones_like(actual),data['spec']['loss_steps'])
        record['metrics'].append(dict(fold=f'm{month:02}',scope='reference_only_not_selection',n_origins=len(vi),**metric,
            point_Brier_raw=float(np.mean((out['p_point']-(actual>b[:,None]))**2))))
        write_record(record,config);previous.dump(suite/'suite_config.json',config)
        previous.log(suite,f"REFERENCE {data['profile']}/m{month:02} train={len(ti)} valid={len(vi)} seconds={fit['elapsed_seconds']:.1f}")
        del forest,distribution;gc.collect()


def write_record(record,config):
    metrics=pd.DataFrame(record['metrics']);fits=record['fits']
    no_rf=config.get('eval_mode')=='continuous'
    weight_note=('달력 지원 원점(2월 이후·마지막 목표가 다음 월 이전)에 추가 w 적용, 1월·월경계 w=1. '
        '주평가 공통 순수 continuous cap2, w=1+g; RF 확률항 제거. 가중치용 기준 RF 추가 학습 없음.' if no_rf else
        '모든방식OOF미지원원점1로비교지원동일. 주평가공통prob_cont cap2. '
        '학습/검증의기준확률은각각과거/현재fold의 cap1 RF.')
    resource_note=f"조건 프로세스 {config.get('parallel_workers',1)}개×RF 스레드 {config.get('rf_threads_per_worker',4)}개, BLAS1thread"
    table=previous.common.md_table(metrics[['fold','scope','n_origins','Loss_C','L_mean','L_CVaR','L_peak','RMSE','point_Brier_raw']]) if len(metrics) else '실제 평가 결과 없음.'
    record['md'].write_text(f'''# 실험 {record['id']}

- 상태: {record['status']}
- 목적과 가설: {record['family']}, 가중치{record['weight_mode']} / 전체 C 직접 학습이 동일 용량 RF의 과소예측과 최악 구간 오차를 개선하는지 확인.
- 시작·종료 시각 및 시간대: {record['started']} / {record.get('ended','미종료')}, KST(UTC+9).
- 모델·라이브러리 버전: {config['versions']}; Python {config['python']}.
- 데이터 위치·버전 또는 해시·사용 범위: {previous.common.SOURCE}; SHA256 {config['source_sha256']};1월부터 각 검증 월 이전까지만 학습.
- 목표값·단위·예측 구간: power_15min, kW; {record['profile']};15분 미래 벡터. 관측 종료시각 기준.
- 학습·검증·테스트 분할과 누수 점검: 7월=1~6월학습,8월=1~7월학습.9월=설정동결후1~8월학습 재평가. phase={record['phase']}; 실제 fit 범위는 아래 JSON. 전체q90에9월포함/기존열람으로 독립 테스트아님.
- 전처리·특성·하이퍼파라미터·seed: 기존 native 미래 생산·기상 실측=예보 오차0. 입력창 {config['windows'][record['profile']]}h; RF {config['rf_params']}; 변경범위 {record['delta']}kW; 학습cap={record['cap']}; 방식={record['weight_mode']}; 정책={config['weight_policy']}; 셀별 w 고정, {weight_note} C=.5mean+.3topceil20%+.2weighted-under-root. 분할후부모bound상속. 각공동최적화{config['optimizer_seconds']}초,각구조탐색{config['split_seconds']}초한계.
- 이전 또는 기준 실험 링크와 변경 사항: [기존 RF 확률 보고서](../../notebooks/models/random_forest/PEAK_PROBABILITY_RESULTS_{SOURCE_RUN.name}.md). 신규 RF64trees/64leaves/minleaf20 공통 용량; 기존500tree결과는별도참고.
- 실행 명령·환경·코드 버전 또는 코드 식별 정보: {config['command']}; 코드SHA {config['code_hashes']}.
- 실제 실행 시간·자원 사용량(측정한 경우): fit elapsed_seconds 합 {sum(f.get('elapsed_seconds',0) for f in fits):.3f}초; {resource_note}; 메모리 피크 미측정.
- 평가 지표·구간별 또는 fold별 결과: 아래 실제 결과. 확률은 raw leaf 경로질량; 확률보정·경보정책 새선정없음.
- 기준모델 대비 변화와 비교 가능 여부: 같은profile/월/용량/입력/학습cap의 rf_mse와 비교. 기존500trees는용량과학습cap이다를수있어손실학습효과로해석불가.
- 오차가 큰 조건·해석·한계: condition_metrics.csv와horizon_metrics.csv. 중첩 원점 독립표본아님. 정확한C목표와수치적전역최적보장은별개. solver메시지를아래fit에기록.
- 오류·경고·실패 또는 중단 원인: {record.get('error','실행 중' if record['status']=='실행 중' else '치명 오류 없음; 최적화 수렴은 fit/optimizer.json 참조')}.
- 예측·모델·표·그래프·로그의 저장 경로: {record['folder']}; 전체run {config['run_id']}.
- 결론과 다음 실험: 7·8월 공통 C 월동일평균으로 선택.9월로 재튜닝하지 않는다. 실제 결과/solver제한을 보고서에 정리.

{table}

## 실제 적합·최적화 정보

```json
{json.dumps(fits,ensure_ascii=False,indent=2,default=str)}
```
''',encoding='utf-8')


def create_record(suite,config,profile,candidate,family,cap,phase,mode='prob_cont'):
    folder=suite/phase/profile/candidate;folder.mkdir(parents=True,exist_ok=False)
    identity=f'{suite.name}_{phase}_{profile}_{candidate}'
    record=dict(id=identity,md=ROOT/'experiments/random_forest'/f'{identity}.md',folder=folder,
        started=datetime.now(previous.common.KST).isoformat(),status='실행 중',
        phase=phase,profile=profile,candidate=candidate,family=family,cap=cap,weight_mode=mode,
        delta=int(family.rsplit('d',1)[1]) if family!='rf_mse' else None,metrics=[],fits=[])
    record['md'].parent.mkdir(exist_ok=True,parents=True);write_record(record,config)
    return record


def resume_folders(cache,profile,cap,mode,phase,fold,policy):
    """동일 조건의 다섯 후보가 모두 저장된 CV 그룹만 재사용한다."""
    if cache is None or phase!='cv' or cache['tables']['metrics'].empty:return None
    found={};needed=['model_bundle.joblib','fold_config.json','optimizer.json','training_inputs.npz',
        'predictions.csv.gz','path_predictions.csv.gz','training_curve.csv']
    for family in FAMILIES:
        name=candidate_name(family,cap,policy,mode);folder=cache['suite']/phase/profile/name/fold
        if any(not (folder/filename).is_file() for filename in needed):return None
        for key in ['metrics','training_curves','reliability','horizon_metrics','condition_metrics']:
            frame=cache['tables'][key]
            rows=frame.loc[frame.profile.eq(profile)&frame.candidate.eq(name)&frame.phase.eq(phase)&frame.fold.eq(fold)]
            if rows.empty or (key=='metrics' and (len(rows)!=3 or set(rows.scope)!=set(['full','clean_targets','clean_targets_inputs']))):return None
        info=previous.read(folder/'fold_config.json')
        if info['training_cap']!=cap or info['weight_mode']!=mode:return None
        found[family]=folder
    return found


def reuse_condition(suite,config,data,reference,ti,vi,records,fold,tables,cache,cap,mode):
    found=resume_folders(cache,data['profile'],cap,mode,'cv',fold,config['weight_policy'])
    if found is None:return False
    with np.load(found['rf_mse']/'training_inputs.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['origins'],data['origins'][ti].asi8)
        np.testing.assert_array_equal(saved['origins_valid'],data['origins'][vi].asi8)
        np.testing.assert_allclose(saved['x'],data['features'].iloc[ti].to_numpy(np.float32))
        np.testing.assert_allclose(saved['x_valid'],data['features'].iloc[vi].to_numpy(np.float32))
        np.testing.assert_allclose(saved['y'],data['targets'].iloc[ti].to_numpy())
        if config['weight_policy']=='methods_no_rf':
            np.testing.assert_array_equal(np.isfinite(saved['reference']).all(axis=1),np.isfinite(reference[ti]).all(axis=1))
        else:
            np.testing.assert_allclose(saved['reference'],reference[ti],equal_nan=True)
            np.testing.assert_allclose(saved['reference_valid'],reference[vi])
        expected,_=training_weights(saved['y'],saved['b'],reference[ti],cap,mode)
        np.testing.assert_allclose(saved['w'],expected)
    for family,source in found.items():
        name=source.parent.name;record=records[(data['profile'],name)];folder=record['folder']/fold;before=perf_counter()
        shutil.copytree(source,folder)
        info=previous.read(folder/'fold_config.json')
        info.update(reused_from=str(source),reused_model_sha256=previous.common.digest(source/'model_bundle.joblib'),
            original_elapsed_seconds=info['elapsed_seconds'],elapsed_seconds=perf_counter()-before,
            original_n_rf_fits=info['n_rf_fits'],n_rf_fits=0,n_reused_rf_fits=1 if family=='rf_mse' else 0,
            reused_at=datetime.now(previous.common.KST).isoformat())
        previous.dump(folder/'fold_config.json',info);record['fits'].append(info)
        for key in ['training_curves','split_proposals']:
            frame=cache['tables'][key]
            rows=frame.loc[frame.profile.eq(data['profile'])&frame.candidate.eq(name)&frame.phase.eq('cv')&frame.fold.eq(fold)].to_dict('records')
            tables[key].extend(rows)
        prediction,out=saved_outputs(folder,data['spec']['horizon_steps'])
        tag=dict(profile=data['profile'],candidate=name,family=family,weight_mode=mode,train_cap=cap,fold=fold,phase='cv')
        rows,reliability,horizons,conditions,eval_w=evaluate_candidate(config,data,vi,prediction,out,reference,tag)
        for key,value in [('metrics',rows),('reliability',reliability),('horizon_metrics',horizons),('condition_metrics',conditions)]:tables[key].extend(value)
        record['metrics'].extend(rows)
        previous.export_predictions(folder,data,vi,data['targets'].iloc[vi].to_numpy(),prediction,out,previous.risk.MonotoneCalibrator(),eval_w)
        write_record(record,config);config['n_reused_models']+=1
    config['n_reused_rf_fits']+=1
    for key,value in tables.items():
        if value:pd.DataFrame(value).to_csv(suite/f'{key}.csv',index=False,encoding='utf-8-sig')
    previous.dump(suite/'suite_config.json',config)
    previous.log(suite,f'REUSED cv/{data["profile"]}/{fold}/{mode}{cap} models=5 source={cache["suite"].name}')
    return True


def fit_and_save(suite,config,data,reference,ti,vi,start,end,records,phase,fold,deadline,tables,resume_cache=None):
    x=data['features'].iloc[ti].to_numpy(np.float32);y=data['targets'].iloc[ti].to_numpy(float)
    xv=data['features'].iloc[vi].to_numpy(np.float32);actual=data['targets'].iloc[vi].to_numpy(float)
    b=data['threshold'].iloc[ti].to_numpy();bv=data['threshold'].iloc[vi].to_numpy()
    if not np.isfinite(reference[vi]).all():raise ValueError('현재 검증 원점의 기준 cap1 확률 누락')
    for condition in config['weight_conditions'][data['profile']]:
        cap=condition['cap'];mode=condition['mode']
        if perf_counter()>=deadline:raise TimeoutError('전체4시간 학습 예산 소진')
        if phase=='cv' and reuse_condition(suite,config,data,reference,ti,vi,records,fold,tables,resume_cache,cap,mode):continue
        w,known=training_weights(y,b,reference[ti],cap,mode)
        sw=w.mean(axis=1);base_before=perf_counter()
        forest=RandomForestRegressor(**config['rf_params']).fit(x,y,sample_weight=sw)
        base=direct.ForestCModel.from_forest(forest);base_seconds=perf_counter()-base_before
        config['n_new_rf_fits']+=1
        models={'rf_mse':base};optimizers={};base_distribution=previous.risk.LeafPathDistribution(base,x,y)
        base_out=base_distribution.predict(xv,bv,quantiles=True,initial=config['initial_peak_kw'])
        for family in FAMILIES:
            if perf_counter()>=deadline:raise TimeoutError('전체4시간 학습 예산 소진')
            name=candidate_name(family,cap,config['weight_policy'],mode);record=records[(data['profile'],name)]
            folder=record['folder']/fold;folder.mkdir(exist_ok=False)
            fit_started=datetime.now(previous.common.KST).isoformat()
            candidate_before=perf_counter();iterations=[];proposal_rows=[]
            if family=='rf_mse':model=base;out=base_out;solver=[]
            elif family.startswith('c_leaf'):
                delta=int(family.rsplit('d',1)[1]);model=deepcopy(base)
                optimizer=direct.optimize_leaves(model,x,y,w,data['spec']['loss_steps'],delta=delta,
                    max_iter=config['max_iter'],deadline=min(deadline,perf_counter()+config['optimizer_seconds']))
                optimizers[family]=optimizer;models[family]=model
                iterations.extend(dict(stage='joint_leaf',**r) for r in optimizer['curve']);solver=[{k:v for k,v in optimizer.items() if k!='curve'}]
                out=base_out
            else:
                leaf=family.replace('c_split','c_leaf');model=deepcopy(models[leaf])
                optimizer=direct.refine_splits(model,x,y,w,data['spec']['loss_steps'],min_leaf=config['rf_params']['min_samples_leaf'],
                    max_splits=config['max_splits'],proposals=config['proposals'],max_iter=config['split_iter'],seed=42,
                    deadline=min(deadline,perf_counter()+config['split_seconds']))
                iterations.extend(dict(stage='inherited_joint_leaf',**r) for r in optimizers[leaf]['curve'])
                proposal_rows=optimizer['proposals'];solver=[]
                for item in optimizer['joint_optimizations']:
                    iterations.extend(dict(stage=f"split_joint_{item['round']}",**r) for r in item['curve'])
                    solver.append({k:v for k,v in item.items() if k!='curve'})
                previous.dump(folder/'split_search.json',optimizer)
                distribution=previous.risk.LeafPathDistribution(model,x,y)
                out=distribution.predict(xv,bv,quantiles=True,initial=config['initial_peak_kw'])
            prediction=model.predict(xv);train_score,_=direct.c_value_gradient(y,model.predict(x),w,data['spec']['loss_steps'])
            tag=dict(profile=data['profile'],candidate=name,family=family,weight_mode=mode,train_cap=cap,fold=fold,phase=phase)
            rows,reliability,horizons,conditions,eval_w=evaluate_candidate(config,data,vi,prediction,out,reference,tag)
            tables['metrics'].extend(rows);record['metrics'].extend(rows)
            tables['reliability'].extend(reliability);tables['horizon_metrics'].extend(horizons);tables['condition_metrics'].extend(conditions)
            baseline_train,_=direct.c_value_gradient(y,base.predict(x),w,data['spec']['loss_steps'])
            if not iterations:iterations=[dict(stage='mse_postfit',iteration=0,seconds=base_seconds,**train_score)]
            for item in iterations:tables['training_curves'].append(dict(**tag,**item))
            for item in proposal_rows:tables['split_proposals'].append(dict(**tag,**item))
            pd.DataFrame(iterations).to_csv(folder/'training_curve.csv',index=False,encoding='utf-8-sig')
            pd.DataFrame(proposal_rows).to_csv(folder/'split_proposals.csv',index=False,encoding='utf-8-sig')
            previous.dump(folder/'optimizer.json',dict(solvers=solver,training_score=train_score,baseline_training_score=baseline_train,
                exact_C_objective=family!='rf_mse',structure_C_refined=family.startswith('c_split'),
                n_splits=len(model.splits),n_leaf_parameters=int(model.theta.size),
                numerical_global_optimum_certified=False,raw_probability_only=True))
            previous.export_predictions(folder,data,vi,actual,prediction,out,previous.risk.MonotoneCalibrator(),eval_w)
            np.savez_compressed(folder/'training_inputs.npz',x=x,y=y,w=w,b=b,reference=reference[ti],known_oof=known,
                origins=data['origins'][ti].asi8,x_valid=xv,origins_valid=data['origins'][vi].asi8,reference_valid=reference[vi])
            fit=dict(fold=fold,fit_started=fit_started,n_train=len(ti),n_valid=len(vi),n_supported_oof=int(known.sum()),
                training_weight_min=float(w.min()),training_weight_max=float(w.max()),training_weight_mean=float(w.mean()),
                start=str(start),end=str(end),train_label_max=str(data['origins'][ti[-1]]+pd.Timedelta(minutes=15*y.shape[1])),
                valid_label_max=str(data['origins'][vi[-1]]+pd.Timedelta(minutes=15*y.shape[1])),
                elapsed_seconds=perf_counter()-candidate_before+(base_seconds if family=='rf_mse' else 0),
                n_rf_fits=1 if family=='rf_mse' else 0,train_C=train_score['Loss_C'],baseline_train_C=baseline_train['Loss_C'],
                n_splits=len(model.splits),solver_status=[s['solver_status'] for s in solver],
                worker_pid=config.get('worker_pid',os.getpid()),parallel_workers=config.get('parallel_workers',1))
            record['fits'].append(fit);previous.dump(folder/'fold_config.json',dict(**fit,features=list(data['features']),spec=data['spec'],
                window_hours=data['window'],training_cap=cap,weight_mode=mode,selection_sha256=config.get('selection_sha256')))
            joblib.dump(dict(model=model,features=list(data['features']),profile=data['profile'],spec=data['spec'],
                context_window_hours=data['window'],initial_peak_kw=config['initial_peak_kw'],threshold_mode='dynamic',
                training_targets=y,training_features=x,raw_probability_only=True),folder/'model_bundle.joblib',compress=3)
            write_record(record,config)
            for key,value in tables.items():
                if value:pd.DataFrame(value).to_csv(suite/f'{key}.csv',index=False,encoding='utf-8-sig')
            previous.log(suite,f"SCORE {phase}/{data['profile']}/{fold}/{name} trainC={train_score['Loss_C']:.4f} validC={rows[0]['Loss_C']:.4f} peak={rows[0]['L_peak']:.4f} splits={len(model.splits)} seconds={fit['elapsed_seconds']:.1f}")
            previous.dump(suite/'suite_config.json',config)
            if family.startswith('c_split'):del model,distribution
        del forest,base,models,base_distribution,base_out;gc.collect()


def condition_worker(suite,config,data,reference,ti,vi,start,end,records,phase,fold,deadline,condition,resume_cache):
    """조건마다 독립 프로세스·폴더·테이블로 계산해 공유 쓰기를 피한다."""
    cfg=deepcopy(config);cfg['weight_conditions']={data['profile']:[condition]}
    cfg.update(n_new_rf_fits=0,n_reused_rf_fits=0,n_reused_models=0,worker_pid=os.getpid())
    name=condition['mode']+str(condition['cap']);work=Path(suite)/'workers'/phase/data['profile']/fold/name
    work.mkdir(parents=True,exist_ok=False)
    selected={(p,c):deepcopy(rec) for (p,c),rec in records.items() if p==data['profile'] and
        c in [candidate_name(f,condition['cap'],cfg['weight_policy'],condition['mode']) for f in FAMILIES]}
    tables={key:[] for key in ['metrics','training_curves','split_proposals','reliability','condition_metrics','horizon_metrics']}
    error=None;error_type=None
    try:
        with threadpool_limits(limits=1):
            fit_and_save(work,cfg,data,reference,ti,vi,start,end,selected,phase,fold,deadline,tables,resume_cache)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';error_type=type(exc).__name__
        (work/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
    cfg.update(status='조건 계산 실패' if error else '조건 계산 완료',error=error,
        ended=datetime.now(previous.common.KST).isoformat());previous.dump(work/'suite_config.json',cfg)
    return dict(tables=tables,records=selected,n_new_rf_fits=cfg['n_new_rf_fits'],n_reused_rf_fits=cfg['n_reused_rf_fits'],
        n_reused_models=cfg['n_reused_models'],worker_pid=os.getpid(),condition=condition,worker_folder=str(work),
        error=error,error_type=error_type)


def fit_fold(suite,config,data,reference,ti,vi,start,end,records,phase,fold,deadline,tables,resume_cache=None):
    if config['parallel_workers']==1:
        return fit_and_save(suite,config,data,reference,ti,vi,start,end,records,phase,fold,deadline,tables,resume_cache)
    jobs=(joblib.delayed(condition_worker)(suite,config,data,reference,ti,vi,start,end,records,phase,fold,deadline,c,resume_cache)
        for c in config['weight_conditions'][data['profile']])
    failures=[]
    with joblib.parallel_config(backend='loky',inner_max_num_threads=1):
        results=joblib.Parallel(n_jobs=config['parallel_workers'],return_as='generator_unordered',pre_dispatch=config['parallel_workers'])(jobs)
        for result in results:
            for key,value in result['tables'].items():tables[key].extend(value)
            for key,value in result['records'].items():records[key].clear();records[key].update(value)
            for key in ['n_new_rf_fits','n_reused_rf_fits','n_reused_models']:config[key]+=result[key]
            config.setdefault('parallel_jobs',[]).append(dict(phase=phase,profile=data['profile'],fold=fold,
                worker_pid=result['worker_pid'],condition=result['condition'],folder=result['worker_folder']))
            for key,value in tables.items():
                if value:pd.DataFrame(value).to_csv(suite/f'{key}.csv',index=False,encoding='utf-8-sig')
            previous.dump(suite/'suite_config.json',config)
            if result.get('error'):failures.append(result)
            label='CONDITION_FAILED' if result.get('error') else 'CONDITION_DONE'
            previous.log(suite,f"{label} {phase}/{data['profile']}/{fold}/{result['condition']} worker_pid={result['worker_pid']}")
    if failures:
        message='병렬 조건 계산 실패: '+'; '.join(item['error'] for item in failures)
        if any(item.get('error_type')=='TimeoutError' for item in failures):raise TimeoutError(message)
        raise RuntimeError(message)


def preserve_failure_record(record,config):
    """worker가 마지막으로 저장한 부분 결과를 부모의 이전 상태로 덮지 않는다."""
    path=record['md']
    if not path.exists():return write_record(record,config)
    lines=path.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('- 상태:'):
            lines[i]=f"- 상태: {record['status']}";break
    lines.extend(['','## 실행 종료 기록',f"종료 {record['ended']}; {record['error']}",
        '마지막 worker가 저장한 실제 부분 결과를 보존했다.'])
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def load_final_reference(data,reference):
    path=SOURCE_RUN/'final'/data['profile']/'cap1/september/predictions.csv.gz'
    frame=pd.read_csv(path,parse_dates=['origin','target_end'])
    if frame.duplicated(['origin','target_end']).any():raise ValueError('기존 최종 기준 예측 키 중복')
    h=data['spec']['horizon_steps'];origins=pd.DatetimeIndex(frame.origin.unique())
    position=data['origins'].get_indexer(origins)
    expected=origins.to_numpy()[:,None]+np.arange(1,h+1)[None,:]*np.timedelta64(15,'m')
    if (position<0).any() or len(frame)!=len(origins)*h or not np.array_equal(frame.target_end.to_numpy(),expected.ravel()):
        raise ValueError('기존9월기준확률순서/목표시각불일치')
    np.testing.assert_allclose(frame.actual.to_numpy().reshape(-1,h),data['targets'].iloc[position].to_numpy())
    np.testing.assert_allclose(frame.peak_threshold.to_numpy().reshape(-1,h)[:,0],data['threshold'].iloc[position].to_numpy())
    reference[position]=frame.p_peak_raw.to_numpy().reshape(-1,h)
    return dict(path=str(path),sha256=previous.common.digest(path))


def run_suite(weight_policy,profiles=None,budget_seconds=14400,smoke=False,trees=64,leaf_nodes=64,max_iter=120,max_splits=4,proposals=12,resume_from=None,workers=1):
    if weight_policy not in ['common2','legacy','tune','methods','methods_no_rf']:raise ValueError('사용자가 정한 명시적인 가중치 정책 필요')
    if workers<1 or workers*4>(os.cpu_count() or 1):raise ValueError('조건 프로세스×RF4스레드가 논리CPU수를 초과합니다.')
    before=perf_counter();deadline=before+budget_seconds;started=datetime.now(previous.common.KST).isoformat()
    identity=datetime.now(previous.common.KST).strftime('%Y%m%dT%H%M%S_KST_')+uuid4().hex[:6]+'_rf_direct_c'+('_smoke' if smoke else '')
    suite=ROOT/'artifacts/random_forest/runs'/identity;suite.mkdir(parents=True,exist_ok=False)
    source=previous.read(SOURCE_RUN/'suite_config.json');old_selection=previous.read(SOURCE_RUN/'selection.json')['selected']
    profiles=profiles or ACTIVE_PROFILES
    if 'long_3d' in profiles:raise ValueError('사용자 지시로 활성 실험에서 장기3일을 제외했습니다.')
    train_caps={p:([1,2,4] if weight_policy=='tune' else [2 if weight_policy=='common2' else old_selection.get(p,{'cap':2})['cap']]) for p in profiles}
    conditions={p:(weight_conditions(weight_policy) if weight_policy in ['methods','methods_no_rf'] else
        [dict(mode='prob_cont',cap=c) for c in train_caps[p]]) for p in profiles}
    train_caps={p:sorted(set(c['cap'] for c in conditions[p])) for p in profiles}
    config=dict(run_id=identity,status='실행 중',started=started,source_run=str(SOURCE_RUN),source_sha256=previous.common.digest(previous.common.SOURCE),
        python=platform.python_version(),versions={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib','nbformat','nbclient','matplotlib']},
        weight_policy=weight_policy,train_caps=train_caps,weight_conditions=conditions,eval_cap=2,
        eval_mode='continuous' if weight_policy=='methods_no_rf' else 'prob_cont',reference_sources={},
        reference_probability='calendar support marker; RF numeric probability excluded' if weight_policy=='methods_no_rf' else 'previous cap1 RF OOF frozen',
        parallel_workers=workers,rf_threads_per_worker=4,logical_cpus=os.cpu_count(),
        training_weight_support='Feb onward & last target before next month; same support as prior comparison',
        windows={p:{**source['windows'],'medium_24h':48,'long_4d':24}[p] for p in profiles},profiles=profiles,validation_folds=FOLDS,smoke_only=smoke,budget_seconds=budget_seconds,
        rf_params=dict(n_estimators=trees,max_leaf_nodes=leaf_nodes,min_samples_leaf=20,max_features=source['rf_params']['max_features'],
            bootstrap=True,n_jobs=4,random_state=42,criterion='squared_error'),max_iter=max_iter,max_splits=max_splits,proposals=proposals,
        optimizer_seconds=16 if weight_policy in ['methods','methods_no_rf'] else 600,split_seconds=20 if weight_policy in ['methods','methods_no_rf'] else 600,
        split_iter=60,initial_peak_kw=source['initial_peak_kw'],n_new_rf_fits=0,n_reused_rf_fits=0,n_reused_models=0,
        code_hashes={Path(p).name:previous.common.digest(p) for p in [__file__,direct.__file__,previous.__file__,previous.risk.__file__,previous.common.__file__,previous.grid.__file__,previous.iw.__file__,previous.ml.__file__,previous.final.__file__]},
        command=f'python -B -X utf8 notebooks/models/random_forest/rf_direct_c_experiments.py --weight-policy {weight_policy}'+(' --smoke' if smoke else ''),
        global_q90_includes_september=True,independent_test=False,future_exogenous='actual_as_zero_error_forecast',raw_probability_only=True,
        exact_C_objective=True,numerical_global_optimum_certified=False)
    if config['source_sha256']!=source['source_sha256']:raise ValueError('기존 전처리 데이터 해시가 달라졌습니다.')
    config['command']+=f' --profiles {" ".join(profiles)} --trees {trees} --leaf-nodes {leaf_nodes} --max-iter {max_iter} --max-splits {max_splits} --proposals {proposals} --budget-seconds {budget_seconds}'
    config['command']+=f' --workers {workers}'
    resume_cache=None
    if resume_from:
        resume_path=Path(resume_from).resolve()
        if not resume_path.is_relative_to(ROOT/'artifacts/random_forest/runs'):raise ValueError('프로젝트 내부 신규 실험만 재사용 가능')
        prior=previous.read(resume_path/'suite_config.json')
        for key in ['rf_params','source_sha256','weight_policy','optimizer_seconds','split_seconds','max_iter','max_splits','proposals']:
            compatible=key=='weight_policy' and config[key]=='methods_no_rf' and prior[key]=='methods'
            if config[key]!=prior[key] and not compatible:raise ValueError(f'재사용 설정 불일치:{key}')
        if prior['smoke_only']:raise ValueError('smoke는 본 실험 학습에 재사용하지 않습니다.')
        resume_cache=dict(suite=resume_path,tables={key:pd.read_csv(resume_path/f'{key}.csv') for key in
            ['metrics','training_curves','split_proposals','reliability','condition_metrics','horizon_metrics']})
        config.update(resume_from=str(resume_path),resume_source_config_sha256=previous.common.digest(resume_path/'suite_config.json'),
            reused_source_code_hashes=prior['code_hashes'],prior_active_seconds=prior.get('elapsed_seconds',0),
            excluded_profiles=['long_3d'])
        config['command']+=f' --resume-from "{resume_path}"'
    previous.dump(suite/'suite_config.json',config);snapshot=suite/'code_snapshot';snapshot.mkdir()
    for path in [__file__,direct.__file__,previous.__file__,previous.risk.__file__,previous.common.__file__,previous.grid.__file__,previous.iw.__file__,previous.ml.__file__,previous.final.__file__]:
        original=Path(path);(snapshot/original.name).write_bytes(original.read_bytes())
    all_records=[];reference_records={};tables={key:[] for key in ['metrics','training_curves','split_proposals','reliability','condition_metrics','horizon_metrics']}
    def make_records(phase):
        records={}
        for profile in profiles:
            for condition in conditions[profile]:
                cap=condition['cap'];mode=condition['mode']
                for family in FAMILIES:
                    candidate=candidate_name(family,cap,weight_policy,mode)
                    rec=create_record(suite,config,profile,candidate,family,cap,phase,mode);all_records.append(rec);records[(profile,candidate)]=rec
        previous.dump(suite/'records.json',[dict(id=r['id'],path=str(r['md']),folder=str(r['folder'])) for r in all_records])
        return records
    try:
        df,repair,audit=previous.final.load_final_data();config['data_audit']=audit
        previous.log(suite,f'START validation=July,August policy={weight_policy} trees={trees} max_leaf_nodes={leaf_nodes}')
        records=make_records('cv')
        with threadpool_limits(limits=1):
            for profile in profiles:
                data=previous.build_profile(df,repair,profile,config['windows'][profile],config['initial_peak_kw'],'dynamic')
                if profile in EXTRA_PROFILES and weight_policy!='methods_no_rf':
                    record=create_record(suite,config,profile,'reference_cap1','rf_mse',1,'reference','none')
                    reference_records[profile]=record;all_records.append(record)
                    generate_reference(suite,config,data,record,range(2,9),deadline)
                    previous.dump(suite/'records.json',[dict(id=r['id'],path=str(r['md']),folder=str(r['folder'])) for r in all_records])
                reference,files=(support_reference(data),[]) if weight_policy=='methods_no_rf' else load_reference(data,suite if profile in EXTRA_PROFILES else SOURCE_RUN)
                config.setdefault('reference_sources',{})[profile]=files
                for month in ([7] if smoke else [7,8]):
                    start=pd.Timestamp(2021,month,1);end=start+pd.offsets.MonthBegin(1)
                    ti,vi=split_indices(data['origins'],data['spec']['horizon_steps'],start,end)
                    fit_fold(suite,config,data,reference,ti,vi,start,end,records,'cv',f'm{month:02}',deadline,tables,resume_cache)
                for rec in records.values():
                    if rec['profile']==profile:
                        rec.update(status='완료' if not smoke else '스모크 완료',ended=datetime.now(previous.common.KST).isoformat());write_record(rec,config)
                del data,reference;gc.collect()
            if not smoke:
                selection=select_candidates(pd.DataFrame(tables['metrics']))
                previous.dump(suite/'selection.json',dict(selected=selection,folds=FOLDS,
                    selected_at=datetime.now(previous.common.KST).isoformat(),selection_rule='7·8월공통C월동일평균최소;동률단순모델우선',
                    september_scores_read=False,other_validation_months_used=False))
                config['selection_sha256']=previous.common.digest(suite/'selection.json');config['selected']=selection
                previous.dump(suite/'suite_config.json',config);previous.log(suite,'SELECTION_FROZEN '+json.dumps(selection,ensure_ascii=False))
                records=make_records('final')
                for profile in profiles:
                    data=previous.build_profile(df,repair,profile,config['windows'][profile],config['initial_peak_kw'],'dynamic')
                    if weight_policy=='methods_no_rf':reference=support_reference(data)
                    elif profile in EXTRA_PROFILES:
                        generate_reference(suite,config,data,reference_records[profile],[9],deadline)
                        reference,_=load_reference(data,suite)
                    else:
                        reference,_=load_reference(data);config.setdefault('final_reference_sources',{})[profile]=load_final_reference(data,reference)
                    start=pd.Timestamp('2021-09-01');end=pd.Timestamp('2021-09-15')
                    ti,vi=split_indices(data['origins'],data['spec']['horizon_steps'],start,end)
                    fit_fold(suite,config,data,reference,ti,vi,start,end,records,'final','september',deadline,tables)
                    for rec in records.values():
                        if rec['profile']==profile:rec.update(status='완료',ended=datetime.now(previous.common.KST).isoformat());write_record(rec,config)
                    del data,reference;gc.collect()
            for record in reference_records.values():
                record.update(status='완료' if not smoke else '스모크 완료',ended=datetime.now(previous.common.KST).isoformat());write_record(record,config)
        config.update(status='스모크 완료' if smoke else '학습 완료·검증 대기',ended=datetime.now(previous.common.KST).isoformat(),
            elapsed_seconds=perf_counter()-before,n_experiment_records=len(all_records))
    except Exception as exc:
        config.update(status='중단' if isinstance(exc,TimeoutError) else '실패',error=f'{type(exc).__name__}: {exc}',
            ended=datetime.now(previous.common.KST).isoformat(),elapsed_seconds=perf_counter()-before,n_experiment_records=len(all_records))
        for rec in all_records:
            if rec['status']=='실행 중':
                rec.update(status=config['status'],error=config['error'],ended=config['ended']);preserve_failure_record(rec,config)
        previous.dump(suite/'suite_config.json',config);(suite/'error.log').write_text(traceback.format_exc(),encoding='utf-8');raise
    previous.dump(suite/'suite_config.json',config)
    with (ROOT/'experiments/README.md').open('a',encoding='utf-8') as stream:
        stream.write(f'\n## {identity}: RF 전체 C 직접 학습 ({config["status"]})\n\n검증7·8월만,가중치정책{weight_policy}.\n\n')
        for r in all_records:stream.write(f'- [{r["id"]}](random_forest/{r["md"].name}): {r["status"]}\n')
    previous.log(suite,f'FINISHED status={config["status"]} new_rf_fits={config["n_new_rf_fits"]} seconds={config["elapsed_seconds"]:.1f}')
    print('SUITE',suite,flush=True);return suite


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--weight-policy',required=True,choices=['common2','legacy','tune','methods','methods_no_rf'])
    parser.add_argument('--profiles',nargs='+',choices=ACTIVE_PROFILES);parser.add_argument('--budget-seconds',type=int,default=14400)
    parser.add_argument('--resume-from')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--trees',type=int,default=64);parser.add_argument('--leaf-nodes',type=int,default=64)
    parser.add_argument('--max-iter',type=int,default=120);parser.add_argument('--max-splits',type=int,default=4);parser.add_argument('--proposals',type=int,default=12)
    run_suite(**vars(parser.parse_args()))
