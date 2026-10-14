"""미래 입력 조건과 결과 경로를 함께 설정한다."""
import config as C
from core import data as D, pipeline as P, significance as S
from models import MODELS

BASE = C.RESULT_DIR

def result_dir(name):
    sub = C.CONDITIONS[name]["dir"]
    return BASE / sub if sub else BASE


def future_cols(name):
    return D.FUTURE + list(C.CONDITIONS[name]["future"])


def use(name):
    """현재 조건을 name으로 → 그 결과 폴더."""
    out = result_dir(name)
    C.RESULT_DIR = out
    P.GRID_FILE, P.CURVE_FILE = out / "grid_results.csv", out / "curves.csv"
    P.SEED_FILE, P.PRED_DIR = out / "seed_results.csv", out / "preds"
    S.OUT = out / "significance"
    for mod in MODELS.values():
        mod.FUTURE_COLS = future_cols(name)
    return out


def check_settings():
    """긴 학습 전에 조건 설정 점검: 이름, 미래에 넣을 열(과거 관측 열 중 전력 제외), 결과 폴더 겹침."""
    for name in C.CONDITIONS_TO_RUN:
        if name not in C.CONDITIONS:
            raise SystemExit(f"CONDITIONS_TO_RUN 확인: '{name}'이 CONDITIONS에 없음 ({list(C.CONDITIONS)})")
        bad = [c for c in C.CONDITIONS[name]["future"] if c not in D.OBS or c == "power"]
        if bad:
            raise SystemExit(f"CONDITIONS['{name}']['future'] 확인: {bad} — 넣을 수 있는 열 {[c for c in D.OBS if c != 'power']}")
    dirs = [result_dir(n) for n in C.CONDITIONS_TO_RUN]
    if len(set(dirs)) != len(dirs):
        raise SystemExit("CONDITIONS의 dir이 겹침 — 조건마다 결과 폴더가 달라야 함")


