"""15분 전력 모델의 예측 구간·월별 검증·평가 공통 계약."""
from dataclasses import dataclass
import json
import hashlib
from importlib.metadata import version, PackageNotFoundError
from time import perf_counter
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
import numpy as np
import pandas as pd

# horizon_steps: 15분 미래 경로 길이 / refresh_steps: 원점 갱신 주기
# output_steps: 보고용 평균 집계 길이. 15분 원경로는 항상 보존한다.
PROFILES = {
    "short_1h": dict(label="단기 1시간", horizon_steps=4, refresh_steps=1, output_steps=1),
    "medium_4h": dict(label="중기 4시간", horizon_steps=16, refresh_steps=4, output_steps=1),
    "medium_8h": dict(label="중기 8시간", horizon_steps=32, refresh_steps=4, output_steps=4),
    "medium_12h": dict(label="중기 12시간", horizon_steps=48, refresh_steps=4, output_steps=4),
    "daily_24h": dict(label="일간 24시간", horizon_steps=96, refresh_steps=4, output_steps=4),
    "long_3d": dict(label="장기 3일", horizon_steps=288, refresh_steps=96, output_steps=4),
    "long_7d": dict(label="장기 7일", horizon_steps=672, refresh_steps=96, output_steps=4),
}
# 시작 포함 / 종료 미포함. 모든 fold는 해당 월 이전 전체 자료로 학습한다.
FOLDS = {
    "apr": ("2021-04-01", "2021-05-01"),
    "may": ("2021-05-01", "2021-06-01"),
    "jun": ("2021-06-01", "2021-07-01"),
    "jul": ("2021-07-01", "2021-08-01"),
}
TARGET = "power_15min"

@dataclass
class FoldData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    train_origins: pd.DatetimeIndex
    valid_origins: pd.DatetimeIndex
    train_end: pd.Timestamp
    valid_end: pd.Timestamp
    horizon: int
    refresh: int


def check_frame(df):
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans or df.index.has_duplicates:
        raise ValueError("시각은 결측/중복 없는 DatetimeIndex여야 합니다.")
    if not df.index.is_monotonic_increasing or not df.index.to_series().diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("시간축은 오름차순의 연속 15분 간격이어야 합니다.")
    if TARGET not in df: raise ValueError(f"필수 목표 열: {TARGET}")


def prepare_fold(df, profile_name, fold_name, lookback=96):
    check_frame(df)
    if profile_name not in PROFILES or fold_name not in FOLDS:
        raise ValueError("지원하는 PROFILES/FOLDS 키를 선택하세요.")
    if not isinstance(lookback, int) or lookback < 1: raise ValueError("lookback은 양의 정수입니다.")
    spec = PROFILES[profile_name]
    horizon, refresh = spec["horizon_steps"], spec["refresh_steps"]
    start, end = map(pd.Timestamp, FOLDS[fold_name])
    # 최종 테스트 및 해당 검증월 이후 값은 표본 생성에도 사용하지 않는다.
    frame = df.loc[df.index < end]
    values = frame[TARGET].to_numpy(dtype=np.float32)
    if len(values) < lookback+horizon: raise ValueError("입력/예측 길이에 비해 자료가 부족합니다.")
    windows = np.lib.stride_tricks.sliding_window_view(values, lookback+horizon)
    origins = frame.index[lookback-1:len(frame)-horizon]
    ends = origins + pd.Timedelta(minutes=15*horizon)
    # 달력 정렬: 중기는 정시, 장기는 자정에만 새 예측을 만든다.
    cadence = ((origins.hour*60 + origins.minute) % (15*refresh) == 0)
    complete = np.isfinite(windows).all(axis=1)
    train = np.asarray(cadence & complete & (ends < start))
    valid = np.asarray(cadence & complete & (origins >= start) & (ends < end))
    if not train.any() or not valid.any(): raise ValueError("학습/검증의 유효 표본이 없습니다.")
    return FoldData(
        windows[train,:lookback,None].copy(), windows[train,lookback:].copy(),
        windows[valid,:lookback,None].copy(), windows[valid,lookback:].copy(),
        origins[train], origins[valid], start, end, horizon, refresh,
    )


def aggregate_path(path, steps):
    path = np.asarray(path)
    if path.ndim != 2 or not isinstance(steps,int) or steps<1 or path.shape[1]%steps:
        raise ValueError("집계 길이는 미래 경로 길이를 나누는 양의 정수여야 합니다.")
    return path.reshape(len(path),-1,steps).mean(axis=2)


def baseline_predictions(df, origins, horizon, method):
    power=df[TARGET]
    if method == "persistence":
        return np.repeat(power.reindex(origins).to_numpy()[:,None],horizon,axis=1)
    if method not in ("daily","weekly"): raise ValueError("지원하지 않는 기준선입니다.")
    period={"daily":96,"weekly":672}[method]
    offsets=(np.arange(1,horizon+1)-1)%period+1-period
    stamps=origins.to_numpy()[:,None]+offsets[None,:]*np.timedelta64(15,"m")
    return power.reindex(pd.DatetimeIndex(stamps.ravel())).to_numpy().reshape(len(origins),horizon)


def score_predictions(actual, predicted, resolution_minutes):
    a,p=np.asarray(actual,dtype=float),np.asarray(predicted,dtype=float)
    if a.ndim!=2 or a.shape!=p.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("정답/예측은 같은 크기의 유한한 2차원 배열이어야 합니다.")
    error=a-p
    return pd.DataFrame({"horizon_minutes":resolution_minutes*np.arange(1,a.shape[1]+1),
                         "n":len(a),"RMSE":np.sqrt((error**2).mean(axis=0)),"MAE":np.abs(error).mean(axis=0)})


def prediction_frame(actual, predicted, origins, minutes):
    stamps=origins.to_numpy()[:,None]+np.arange(1,actual.shape[1]+1)[None,:]*np.timedelta64(minutes,"m")
    return pd.DataFrame({"origin":np.repeat(origins,actual.shape[1]),"target_timestamp":stamps.ravel(),
                         "horizon_minutes":np.tile(minutes*np.arange(1,actual.shape[1]+1),len(origins)),
                         "y_true":actual.ravel(),"y_pred":predicted.ravel()})


def fit_predict(model_name, factory, data, df, options):
    """fold마다 새 트리 모델 생성. 제거한 딥러닝 모델은 실행하지 않는다."""
    if model_name not in ("random_forest", "lightgbm", "catboost"):
        raise ValueError(f"지원하지 않는 모델: {model_name}")
    try:
        model=factory(data.X_train.shape[1],data.horizon)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"{model_name} 의존성 '{exc.name}'이 없습니다. 노트북 상단 설치 안내를 확인하세요.") from exc
    model.fit(data.X_train[:,:,0],data.y_train)
    return np.asarray(model.predict(data.X_valid[:,:,0]))


def run_experiments(df, model_name, factory, profile_names, fold_names, lookback=96,
                    train_options=None, output_steps=None, output_dir=None):
    if not profile_names or not fold_names: raise ValueError("예측 구간과 검증 월을 선택하세요.")
    options=dict(train_options or {})
    metrics,records,summary=[],[],[]
    for profile in profile_names:
        spec=PROFILES[profile]
        block=(output_steps or {}).get(profile,spec["output_steps"])
        if block not in (1,4,96): raise ValueError("출력 집계는 1(15분), 4(시간 평균), 96(일 평균) 중 선택하세요.")
        aggregate_path(np.zeros((1,spec["horizon_steps"])),block)
        for fold in fold_names:
            data=prepare_fold(df,profile,fold,lookback)
            started=perf_counter()
            if model_name=="baseline":
                outputs={name:baseline_predictions(df,data.valid_origins,data.horizon,name)
                         for name in ("persistence","daily","weekly")}
                shared=np.logical_and.reduce([np.isfinite(p).all(axis=1) for p in outputs.values()])
            else:
                outputs={model_name:fit_predict(model_name,factory,data,df,options)}
                shared=np.ones(len(data.y_valid),dtype=bool)
            elapsed=perf_counter()-started
            for name,pred in outputs.items():
                if pred.shape!=data.y_valid.shape: raise ValueError("예측과 공통 정답의 shape 불일치")
                actual,pred=data.y_valid[shared],pred[shared]
                orig=data.valid_origins[shared]
                for steps in sorted({1,block}):
                    a,p=aggregate_path(actual,steps),aggregate_path(pred,steps)
                    tag=dict(model=name,profile=profile,fold=fold,resolution_minutes=15*steps,
                             aggregation="point" if steps==1 else "mean")
                    metrics.append(score_predictions(a,p,15*steps).assign(**tag))
                    records.append(prediction_frame(a,p,orig,15*steps).assign(**tag))
                    summary.append(dict(**tag,n_train=len(data.y_train),n_valid=len(a),
                        RMSE=float(np.sqrt(np.mean((a.astype(float)-p)**2))),
                        MAE=float(np.mean(np.abs(a-p))),fit_predict_seconds=elapsed))
            print(f"{profile} / {fold}: train={len(data.y_train)}, valid={int(shared.sum())}, {elapsed:.1f}s")
    result={"metrics":pd.concat(metrics,ignore_index=True),"predictions":pd.concat(records,ignore_index=True),
            "summary":pd.DataFrame(summary)}
    if output_dir is not None:
        folder=Path(output_dir)/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid4().hex[:8])
        folder.mkdir(parents=True,exist_ok=False)
        for key,frame in result.items(): frame.to_csv(folder/f"{key}.csv",index=False,encoding="utf-8-sig")
        versions={}
        for package in ("numpy","pandas","scikit-learn","lightgbm","catboost"):
            try: versions[package]=version(package)
            except PackageNotFoundError: pass
        data_hash=hashlib.sha256(pd.util.hash_pandas_object(df[[TARGET]],index=True).values.tobytes()).hexdigest()
        config=dict(data_target_index_sha256=data_hash, package_versions=versions, model=model_name,profiles=profile_names,folds=fold_names,lookback=lookback,
                    train_options=options,output_steps=output_steps,profile_definitions=PROFILES,fold_definitions=FOLDS,
                    note="월별 검증 결과. 최종 테스트 아님. 시간/일 출력은 전력 평균 제안값.")
        (folder/"config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
        result["output_dir"]=folder
    return result
