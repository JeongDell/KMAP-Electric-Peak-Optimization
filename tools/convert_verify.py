"""2_딥러닝 최종 가중치(.pt) → safetensors 변환 + 9월 저장 예측 재현 검증.

사용: python convert_verify.py <2_딥러닝 폴더> <출력 폴더>
- 원본 .pt / 변환 .safetensors 각각으로 9월 예측을 다시 만들어 저장 예측(preds/*_2021-09_*.npy)과 비교한다.
- 출력: <out>/dl/<group>/<model>_loss<L>/seed<s>.safetensors, config.json, normalization.json, verify.csv
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file

SRC = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(SRC))

import config as C  # noqa: E402
from core import conditions as CD, data as D, pipeline as P, train as T  # noqa: E402
from models import MODELS  # noqa: E402

torch.set_num_threads(4)
DEV = torch.device("cpu")
CD.use("다앎")
RES = C.RESULT_DIR                       # results/future_known
MODEL_DIR, PRED_DIR = RES / "models", RES / "preds"
CONFIRM, TEST = C.CONFIRM_MONTHS[0], C.TEST_MONTH

df = D.load()
val_start, _, _ = P.split(df, CONFIRM)
fold = D.Fold(df, val_start, DEV)

# 정규화 통계 (Fold와 같은 계산을 다시 해서 저장)
fit = df["ts"] < pd.Timestamp(val_start)
mean, std = df.loc[fit, D.OBS].mean(), df.loc[fit, D.OBS].std()
norm = {
    "fit_rows": f"timestamp < {val_start}",
    "obs_mean": {k: float(v) for k, v in mean.items()},
    "obs_std": {k: float(v) for k, v in std.items()},
    "lags_use_power_stats": D.LAGS,
    "feature_order": fold.cols,
    "transforms": {"production": "log1p(생산량)", "rain": "log1p(강수량)",
                   "power_lag96": "power.shift(96)", "power_lag672": "power.shift(672)",
                   "tod_sin/cos": "15분 칸 번호/96", "calendar": "one-hot, 정규화 안 함"},
    "output": "kW = z * obs_std.power + obs_mean.power",
}
assert abs(norm["obs_mean"]["power"] - fold.mu) < 1e-9 and abs(norm["obs_std"]["power"] - fold.sd) < 1e-9
(OUT / "dl").mkdir(parents=True, exist_ok=True)
(OUT / "dl" / "normalization.json").write_text(json.dumps(norm, ensure_ascii=False, indent=2), encoding="utf-8")

best = pd.read_csv(RES / "best_params.csv", encoding="utf-8-sig")
rows, configs = [], []
with torch.no_grad():
    for _, b in best.iterrows():
        group, name, loss, pjson = b["group"], b["model"], b["loss"], b["params"]
        G, mod, params = C.GROUPS[group], MODELS[name], json.loads(pjson)
        h = hashlib.md5(pjson.encode()).hexdigest()[:8]
        t0 = D.origins(df, TEST + "-01", pd.Timestamp(TEST + "-01") + pd.offsets.MonthBegin(1), G["horizon"], G["eval_stride"])
        spec = T.make_spec(fold, mod, params["lookback"], G["horizon"], G["block"])
        dest = OUT / "dl" / group / f"{name}_loss{loss}"
        dest.mkdir(parents=True, exist_ok=True)
        files = {}
        for seed in C.SEEDS:
            stem = f"{group}_{name}_loss{loss}_{h}_{CONFIRM}_seed{seed}"
            pt = MODEL_DIR / f"final_{stem}.pt"
            saved = np.load(PRED_DIR / f"{group}_{name}_loss{loss}_{h}_{TEST}_seed{seed}.npy")
            state = torch.load(pt, map_location="cpu", weights_only=True)
            st_path = dest / f"seed{seed}.safetensors"
            save_file({k: v.contiguous() for k, v in state.items()}, st_path,
                      metadata={"group": group, "model": name, "loss": loss, "seed": str(seed), "source": pt.name})
            res = {}
            for kind, sd in [("pt", state), ("safetensors", load_file(st_path))]:
                model = mod.build(params, G["horizon"]).to(DEV)
                model.load_state_dict(sd)
                pred, _ = T.predict(model, fold, spec, t0)
                res[kind] = fold.to_kw(pred).numpy()
            rows.append({"group": group, "model": name, "loss": loss, "seed": seed, "n_origins": len(t0),
                         "shape_ok": res["pt"].shape == saved.shape,
                         "maxabs_pt_vs_saved_kW": float(np.abs(res["pt"] - saved).max()),
                         "maxabs_st_vs_pt_kW": float(np.abs(res["safetensors"] - res["pt"]).max()),
                         "source_sha256": hashlib.sha256(pt.read_bytes()).hexdigest()})
            files[str(seed)] = st_path.name
        cfg = {"group": group, "model": name, "loss": loss, "params": params, "params_hash": h,
               "horizon_steps": G["horizon"], "block": G["block"], "eval_stride": G["eval_stride"],
               "past_cols": list(mod.PAST_COLS), "future_cols": list(mod.FUTURE_COLS),
               "condition": "다앎 (미래 생산량·기상 실제값을 입력)", "trained_on": f"2021-01-15 ~ {val_start:%Y-%m-%d} 직전",
               "seeds": files, "ensemble": "같은 loss의 RNN·LSTM·NHITS 각 seed 3개 평균 → 세 모델 평균"}
        (dest / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{group:>4} {name:<5} {loss}  max|pt-saved|={max(r['maxabs_pt_vs_saved_kW'] for r in rows[-3:]):.2e}", flush=True)

v = pd.DataFrame(rows)
v.to_csv(OUT / "verify.csv", index=False, encoding="utf-8-sig")
print("\n파일", len(v), "| shape 불일치", int((~v.shape_ok).sum()),
      "| max|pt-saved| kW", v.maxabs_pt_vs_saved_kW.max(), "| max|st-pt| kW", v.maxabs_st_vs_pt_kW.max())
