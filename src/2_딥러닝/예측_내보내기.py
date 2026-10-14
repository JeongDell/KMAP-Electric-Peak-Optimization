"""딥러닝 저장 예측(results/future_known/preds)에서 마진·경보 단계의 입력 CSV를 만든다. 학습은 하지 않는다.

- 예측_7_8월/(구간).csv: 대표(ENSEMBLE A·C) + 기준선 4개, 7·8월
- 예측_9월/(구간).csv: 대표(ENSEMBLE A·C), 9월 (= test_predictions/와 같은 값)
대표는 main.py와 같은 규칙(7월 공통 C 1위)으로 다시 고른다. 열 형식은 test_predictions/와 같다.
실행: python 예측_내보내기.py  (main.py --resume을 먼저 실행한 뒤)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

C.RESULT_DIR = C.BASE_DIR / "results"
C.RUN_SEPTEMBER = True

import pandas as pd  # noqa: E402
from core import conditions as CD, data as D, pipeline as P, report as R  # noqa: E402

OUT = HERE / "예측_내보내기"
COLS = ["model", "loss", "horizon", "seed", "period", "origin", "lead", "target_time", "y", "yhat", "b_origin", "대표"]
FMT = dict(index=False, encoding="utf-8-sig", float_format="%.3f", date_format="%Y-%m-%d %H:%M:%S")


def main():
    for s in [sys.stdout, sys.stderr]:
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8")
    df = D.load()
    CD.check_settings()
    CD.use("다앎")
    P.check_settings(df)
    best = P.summarize(P.load_results())[0]
    months = C.FOLDS + C.CONFIRM_MONTHS + [C.TEST_MONTH]
    _, pred, picks = R.collect(df, best, months)
    print("대표:", {f"{g} {l}": n for (g, l), n in picks.items()})
    pred = pred.rename(columns={"b_loss": "b_origin"})

    rate = R.rate_before(df)
    base_rows = []
    for g in C.GROUPS_TO_RUN:
        for month in C.FOLDS + C.CONFIRM_MONTHS:
            role = "고르기" if month in C.FOLDS else "확인"
            blk = R.month_block(df, best, g, month, role)
            frame = R.path_frame(df, blk, rate)
            for name, p in blk["preds"].items():
                if not name.startswith("기준선"):
                    continue
                part = frame[["origin", "lead", "target_time", "y", "b_loss"]].rename(columns={"b_loss": "b_origin"})
                part = part.assign(yhat=p.ravel(), model=name, loss="-", horizon=g, seed="-",
                                   period=str(int(month[-2:])) + "월", 대표=None)
                for c in ["y", "yhat", "b_origin"]:
                    part[c] = part[c].map(lambda v: float(f"{v:.3f}"))
                base_rows.append(part)
    base = pd.concat(base_rows, ignore_index=True)

    for sub in ["예측_7_8월", "예측_9월"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for g in C.GROUPS_TO_RUN:
        rep = pred[pred["horizon"].eq(g)]
        jul_aug = pd.concat([rep[rep["period"].isin(["7월", "8월"])][COLS], base[base["horizon"].eq(g)][COLS]],
                            ignore_index=True)
        jul_aug.to_csv(OUT / "예측_7_8월" / f"{g}.csv", **FMT)
        sep = rep[rep["period"].eq("9월")][COLS]
        sep.to_csv(OUT / "예측_9월" / f"{g}.csv", **FMT)
        print(f"{g}: 7·8월 {len(jul_aug):,}줄, 9월 {len(sep):,}줄")


if __name__ == "__main__":
    main()
