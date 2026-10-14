"""실험 설정. 실행: python main.py. 기본 실행은 새 학습부터 테스트까지 완료한다."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'timeseries_15min_v1_previous_week.csv'
RESULT_DIR = BASE_DIR / 'results'

GROUPS_TO_RUN = ['1h', '4h', '8h', '12h', '24h', '4d']
MODELS_TO_RUN = ['RNN', 'LSTM', 'NHITS']
LOSSES_TO_RUN = ['A', 'C']
GRID_SEARCH = True
RUN_SEPTEMBER = True

OUTAGES = [('2021-08-28 17:30', '2021-08-29 11:15'), ('2021-09-08 12:00', '2021-09-08 12:15')]
PROD_FIX_DAYS = ['2021-07-13', '2021-07-15']
EVAL_SETS = {'전체': {'copied': True, 'prod_fix': True, 'outage': True},
 '복사날제외': {'copied': False, 'prod_fix': True, 'outage': True}}
SELECT_EVAL = '전체'
CHECK_EVAL = '복사날제외'

GROUPS = {'1h': {'horizon': 4, 'train_stride': 1, 'eval_stride': 1, 'block': 4, 'report': {'1h': 4}},
 '4h': {'horizon': 16, 'train_stride': 4, 'eval_stride': 4, 'block': 4, 'report': {'4h': 16}},
 '8h': {'horizon': 32, 'train_stride': 4, 'eval_stride': 4, 'block': 4, 'report': {'8h': 32}},
 '12h': {'horizon': 48,
         'train_stride': 4,
         'eval_stride': 4,
         'block': 4,
         'report': {'12h': 48}},
 '24h': {'horizon': 96,
         'train_stride': 4,
         'eval_stride': 4,
         'block': 4,
         'report': {'24h': 96}},
 '4d': {'horizon': 384,
        'train_stride': 4,
        'eval_stride': 96,
        'block': 96,
        'report': {'4d': 384}}}

GRIDS = {'RNN': {'lookback': [48, 24, 96], 'hidden': [64, 128, 256, 512], 'lr': [0.001, 0.0003]},
 'LSTM': {'lookback': [192, 384], 'hidden': [32, 64, 128, 256], 'lr': [0.001, 0.0003]},
 'NHITS': {'lookback': [96, 48],
           'units': [64, 128, 256],
           'dropout': [0.2, 0.4],
           'lr': [0.001, 0.0003]}}

SEEDS = [42, 43, 44]

FOLDS = ['2021-07']
CONFIRM_MONTHS = ['2021-08']
TEST_MONTH = '2021-09'
SELECT_METRIC = 'LossC'
TRAIN_FIRST_ORIGIN = '2021-01-15'
ES_DAYS = 14

CONDITIONS = {'모름': {'future': [], 'dir': '', 'label': '미래 정보 모름 (운영 조건)'},
 '다앎': {'future': ['production', 'temp', 'humidity', 'wind', 'rain'],
        'dir': 'future_known',
        'label': '미래 정보 다 앎 (실제값 상한)'}}
CONDITIONS_TO_RUN = ['다앎']

CVAR_FRACTION = 0.2
PEAK_FLOOR = 170.0
PEAK_WEIGHT = {'START': 0.8, 'W_MAX': 2.0}

BATCH_SIZE = 128
MAX_EPOCHS = 200
PATIENCE = 10
PROBE_SIZE = 1000
GRAD_CLIP = 1.0

SEED = 42
DM_ALPHA = 0.05
DM_ON_CONFIRM = True
