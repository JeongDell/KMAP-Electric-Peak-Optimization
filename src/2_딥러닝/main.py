"""python main.py: 학습 → 선정 검증 → 재학습 → 확인 검증 → 테스트 → 제출 CSV 저장."""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import config as C


def cache_signature():
    keys = ['GROUPS_TO_RUN', 'MODELS_TO_RUN', 'LOSSES_TO_RUN', 'GROUPS', 'GRIDS', 'GRID_SEARCH',
            'FOLDS', 'CONFIRM_MONTHS', 'TEST_MONTH', 'TRAIN_FIRST_ORIGIN', 'ES_DAYS', 'EVAL_SETS',
            'SELECT_EVAL', 'CHECK_EVAL', 'SELECT_METRIC', 'SEED', 'SEEDS', 'BATCH_SIZE', 'MAX_EPOCHS',
            'PATIENCE', 'PROBE_SIZE', 'GRAD_CLIP', 'CVAR_FRACTION', 'PEAK_FLOOR', 'PEAK_WEIGHT',
            'CONDITIONS', 'CONDITIONS_TO_RUN', 'OUTAGES', 'PROD_FIX_DAYS']
    code = Path(__file__).resolve().parent
    files = [C.DATA_PATH, code / 'core/train.py', code / 'core/data.py', code / 'core/peak.py',
             *sorted((code / 'models').glob('*.py'))]
    signature = {'settings': {key: getattr(C, key) for key in keys},
                 'inputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    return json.loads(json.dumps(signature, ensure_ascii=False))


def main(argv=None):
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', nargs='?', const=str(C.BASE_DIR / 'results'),
                        help='저장된 학습 결과로 이어서 실행. 경로를 생략하면 포함된 results 사용')
    parser.add_argument('--output-dir', help='새 학습 결과를 저장할 폴더 (기본: results/run_실행시각)')
    args = parser.parse_args(argv)
    if args.resume and args.output_dir:
        parser.error('--resume과 --output-dir은 함께 사용할 수 없습니다.')
    if args.resume:
        base = Path(args.resume).resolve()
        if not base.is_dir():
            parser.error('재개할 결과 폴더가 없습니다.')
    else:
        base = Path(args.output_dir).resolve() if args.output_dir else C.BASE_DIR / 'results' / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
        if base.exists() and any(base.iterdir()):
            parser.error('새 학습은 빈 결과 폴더를 사용해야 합니다. 이어서 실행하려면 --resume을 사용하세요.')
    signature = cache_signature()
    manifest = base / 'run_settings.json'
    if args.resume:
        if not manifest.exists() or json.loads(manifest.read_text(encoding='utf-8')) != signature:
            raise SystemExit('저장된 결과와 현재 데이터·학습 설정이 다릅니다. python main.py로 새로 학습하세요.')
    base.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding='utf-8')
    C.RESULT_DIR = base
    C.RUN_SEPTEMBER = True

    import torch
    import pandas as pd
    from core import conditions as CD, data as D, pipeline as P, report as R, margin as M, significance as S
    if P.DEVICE.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    if C.CONDITIONS_TO_RUN != ['다앎']:
        raise SystemExit('이 제출본은 다앎 조건으로 구성되어 있습니다. CONDITIONS_TO_RUN을 확인하세요.')
    df = D.load()
    if (C.FOLDS, C.CONFIRM_MONTHS, C.TEST_MONTH) != (['2021-07'], ['2021-08'], '2021-09'):
        raise ValueError('이 제출본의 기간 분할은 7월 선정·8월 확인·9월 테스트입니다.')
    if C.SEED not in C.SEEDS or len(set(C.SEEDS)) != len(C.SEEDS):
        raise ValueError('SEEDS에는 그리드 학습 시드 SEED를 포함하고 중복을 제거해야 합니다.')
    CD.check_settings()
    CD.use('다앎')
    P.check_settings(df)
    for month in C.FOLDS + C.CONFIRM_MONTHS + [C.TEST_MONTH]:
        for group in C.GROUPS_TO_RUN:
            G = C.GROUPS[group]
            if not len(D.origins(df, month + '-01', pd.Timestamp(month + '-01') + pd.offsets.MonthBegin(1), G['horizon'], G['eval_stride'])):
                raise ValueError(f'{month} {group}: 평가할 데이터가 없습니다.')
    C.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'장치: {P.DEVICE} | {"저장된 결과로 재개" if args.resume else "새 학습"} | 결과: {C.RESULT_DIR}', flush=True)
    P.describe_grid()

    print('\n[1/6] 모델 후보 학습 → 7월 검증 및 설정 선정', flush=True)
    P.run_grid(df)
    best, comparison, by_month = P.summarize(P.load_results())
    for table, name in [(best, 'best_params.csv'), (comparison, 'comparison.csv'), (by_month, 'comparison_by_month.csv')]:
        table.to_csv(C.RESULT_DIR / name, index=False, encoding='utf-8-sig')
    P.run_seeds(df, best, C.FOLDS, P.SEED_FILE)
    P.summarize_seeds(df, best, C.FOLDS, P.SEED_FILE, '', '7월 선정 검증')

    print('\n[2/6] 선정된 모델을 1~7월 데이터로 재학습', flush=True)
    P.run_final(df, best, C.CONFIRM_MONTHS, reuse_predictions=bool(args.resume))
    print('\n[3/6] 8월 검증 → 대표 및 안전마진 확정', flush=True)
    P.summarize_seeds(df, best, C.CONFIRM_MONTHS, P.final_seed_file(), 'confirm_', '8월 확인 검증')
    phases = C.FOLDS + C.CONFIRM_MONTHS
    performance, predictions, picks = R.collect(df, best, phases)
    M.run(predictions, include_test=False)
    frozen = (C.RESULT_DIR / 'margin/②마진표.csv').read_bytes()

    print('\n[4/6] 확정한 모델과 마진으로 9월 테스트', flush=True)
    P.run_final(df, best, [C.TEST_MONTH], reuse_predictions=bool(args.resume))
    P.summarize_seeds(df, best, [C.TEST_MONTH], P.final_seed_file(), 'test_', '9월 테스트')
    performance, predictions, final_picks = R.collect(df, best, phases + [C.TEST_MONTH], picks=picks)
    if final_picks != picks:
        raise RuntimeError('테스트에서 선정된 대표가 변경되었습니다.')
    M.run(predictions, include_test=True)
    if (C.RESULT_DIR / 'margin/②마진표.csv').read_bytes() != frozen:
        raise RuntimeError('테스트에서 안전마진이 변경되었습니다.')
    S.run(df)

    print('\n[5/6] 제출용 테스트 예측 및 성능 결과 저장', flush=True)
    staged = R.export(df, best, performance, predictions, picks)
    print('\n[6/6] 제출 결과 누락·중복 및 예측값 검사', flush=True)
    R.verify(df, best, picks, staged)
    R.publish(staged)
    print(f'\n완료: {C.BASE_DIR / "test_predictions"}\n성능 결과: {C.BASE_DIR / "results_summary.csv"}\n상세 결과: {C.RESULT_DIR}', flush=True)


if __name__ == '__main__':
    main()
