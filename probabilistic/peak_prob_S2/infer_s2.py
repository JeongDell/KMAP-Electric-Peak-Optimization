"""피크 초과확률 S2 추론 (학습 없음, 저장 모델만 사용).

S2 = 0.5 × LightGBM v2 분류기(seed 5개 평균) + 0.5 × 잔차 경로 분포 기간 확률

사건: 한 발행 원점의 예측 경로(H칸) 중 한 번이라도 실제 전력 y > b_origin (엄격한 >).
b_origin = max(170 kW, 그달 발행 전까지 관측된 최대 전력).

입력 (원점 N개)
- features : DataFrame, 열 = 모델 파일의 features 목록(model/input_schema.json 참고). 발행 시점 정보만.
- yhat     : (N, H) 전력 점예측 경로 kW (트리 4모델 m1007 단순평균, A판)
- b_origin : (N,) kW

제공 구간: short_1h, medium_4h, medium_8h, medium_12h(경계). 24h·4d는 검증 기준 미달이라 미제공
(파일은 있으나 기본으로 거절, --allow-unprovided로만 실행).

사용 예:
    python infer_s2.py --profile short_1h --features feats.csv --yhat yhat.npy --b b.npy --out p.csv
    python infer_s2.py --selftest            # 저장 모델 로드·형상 점검 (합성 입력)
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
import residual_peak_distribution as rpd  # noqa: E402

MODEL_DIR = HERE / "model"
HORIZON = {"short_1h": 4, "medium_4h": 16, "medium_8h": 32, "medium_12h": 48, "medium_24h": 96, "long_4d": 384}
PROVIDED = {"short_1h", "medium_4h", "medium_8h", "medium_12h"}
WEIGHT = 0.5


def load(profile):
    bundle = joblib.load(MODEL_DIR / f"model_v2_{profile}.joblib")
    pool = np.load(MODEL_DIR / f"{profile}_residual_pool.npz")
    dist = rpd.ResidualPathDistribution().fit(pool["residual"], pd.to_datetime(pool["origins"]))
    return bundle, dist


def predict(profile, features, yhat, b_origin, allow_unprovided=False):
    if profile not in HORIZON:
        raise ValueError(f"profile은 {list(HORIZON)} 중 하나")
    if profile not in PROVIDED and not allow_unprovided:
        raise ValueError(f"{profile}: 검증 기준 미달로 미제공 구간 (--allow-unprovided로만 실행)")
    bundle, dist = load(profile)
    feats = bundle["features"]
    missing = [c for c in feats if c not in features.columns]
    if missing:
        raise ValueError(f"입력 특성 누락: {missing}")
    X = features[feats].astype(float)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("특성에 결측·무한값이 있음")
    yhat = np.asarray(yhat, float)
    b = np.asarray(b_origin, float)
    if yhat.shape != (len(X), HORIZON[profile]) or b.shape != (len(X),):
        raise ValueError(f"yhat (N,{HORIZON[profile]}), b (N,) 필요 — 받은 값 {yhat.shape}, {b.shape}")
    p_clf = np.mean([m.predict_proba(X)[:, 1] for m in bundle["models"]], axis=0)
    p_res = dist.predict(yhat, b)["p_any_raw"]
    return pd.DataFrame({"p_lgbm_v2": p_clf, "p_residual": p_res,
                         "p_S2": WEIGHT * p_clf + (1 - WEIGHT) * p_res}, index=features.index)


def selftest():
    schema = json.loads((MODEL_DIR / "input_schema.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    for profile, h in HORIZON.items():
        bundle, dist = load(profile)
        n = 8
        X = pd.DataFrame(rng.uniform(0, 1, (n, len(bundle["features"]))), columns=bundle["features"])
        X["b"] = 170.0
        X["b170"] = 1.0
        yhat = rng.uniform(100, 200, (n, h))
        out = predict(profile, X, yhat, np.full(n, 170.0), allow_unprovided=True)
        assert out.shape == (n, 3) and ((out >= 0) & (out <= 1)).all().all()
        print(f"{profile:<10} models={len(bundle['models'])} pool={dist.n_paths:>5}x{dist.horizon:<3} "
              f"features={len(bundle['features'])} 제공={'O' if profile in PROVIDED else 'X'} ok")
    assert set(schema["v2_features_base"]) <= set(load("short_1h")[0]["features"])
    print("selftest 통과 (합성 입력: 값의 의미는 없음, 로드·형상·범위만 확인)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile")
    ap.add_argument("--features", help="CSV, index_col=0 (원점)")
    ap.add_argument("--yhat", help=".npy (N,H) kW")
    ap.add_argument("--b", help=".npy (N,) kW")
    ap.add_argument("--out", default="p_s2.csv")
    ap.add_argument("--allow-unprovided", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    feats = pd.read_csv(a.features, index_col=0)
    out = predict(a.profile, feats, np.load(a.yhat), np.load(a.b), a.allow_unprovided)
    out.to_csv(a.out, encoding="utf-8-sig")
    print(f"저장: {a.out} ({len(out)}행)")


if __name__ == "__main__":
    main()
