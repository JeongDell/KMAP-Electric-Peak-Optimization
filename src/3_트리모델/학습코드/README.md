# 트리 모델 4종 최종 학습 코드 (10/7 재실험, m1007)

최신 ML 앙상블 대표(**RF·XGBoost·LightGBM·CatBoost, C 손실, 각 25% 단순평균**)를 만든 학습 코드를 **수정 없이 복사**해 한곳에 모았다. 코드 정리용 묶음이라 새 학습이나 실행 결과는 없다. 원본 데이터와 모델 바이너리는 복사하지 않았다.

| 단계 | 학습 범위 | 실행 run | 진입 스크립트 |
|---|---|---|---|
| 7월 선정 | 1\~6월 학습 → 7월 C로 후보 선정 | RF `20261007T081917_KST_f4847a_rf_m1007` · 부스팅 `20261007T090243_KST_m1007_boost` | `run_rf_m1007.py`, `run_boosting_m1007.py` |
| 8월 재학습 | 7월에 고른 설정 그대로 1\~7월 재학습 → 8월 예측 | `20261007T102741_KST_m1007_august_margin` | `m1007_august.py` |

이 두 run의 예측이 [7·8월 예측 교환 묶음](../ensemble/20261008T011059_KST_jul_aug_delivery/README.md)의 구성원 예측이다. 구간·모델별 선정 설정은 [학습 메타데이터](../ensemble/20261008T011059_KST_jul_aug_delivery/model/training_provenance.csv)에 있다.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| [code/](code/) | 실행용 코드 26개. 프로젝트와 같은 `notebooks/_shared`, `notebooks/models/random_forest` 구조 유지 |
| [july_run_snapshot/](july_run_snapshot/) | 7월 선정 run이 실행 당시 저장한 코드 사본(RF 7개, 부스팅 10개). 기록 해시와 모두 일치 |
| [configs/](configs/) | 모델별 7월 run의 `config.json`(후보·고정 설정·버전·코드 해시)과 `selection.json`(구간별 선정 결과) |
| [manifest.csv](manifest.csv) | 파일별 원위치, SHA256, 원본 수정 시각, run 기록 해시와의 일치 여부 |
| [requirements.txt](requirements.txt) | 프로젝트 의존성 파일 사본 |

`code/`에는 진입 스크립트 3개가 import하는 프로젝트 내부 모듈을 모두 넣었다. run 설정의 `code_sha256`에 없는 간접 import 모듈(`forecast_common`, `m1007_july_report`, `rf_september_final` 등 10개)와 8월 스크립트 `m1007_august.py`도 포함했다.

## 코드 버전 확인 결과

- **7월 선정 run**: `july_run_snapshot/`은 기록 해시와 17/17 일치한다. `code/`의 작업본도 `run_boosting_m1007.py` 하나를 빼면 해시가 같다.
- `code/notebooks/_shared/run_boosting_m1007.py`는 7월 run 이후(10/7 10:26) `fit_two_stage(..., period=mc.JULY)` 인자 하나만 추가됐다. 기본값이면 7월 동작이 같고, 8월 재학습은 이 버전을 쓴다(8월 run 시작 10:27보다 먼저 수정됨).
- **8월 재학습 run**: `m1007_august.py`는 실행 후에도 수정됐다(원본 수정 시각 10/7 12:30). 기록된 해시(RF `8998895a…`, 부스팅 `fce81af8…`)와 현재 파일(`ede36c5b…`)이 다르고, 실행 당시 사본은 프로젝트 안에서 찾지 못했다. 따라서 **8월 학습 코드는 현재 작업본이며 실행 당시와 완전히 같다고 보장하지 못한다.**
- 나머지 간접 import 모듈은 수정 시각이 모두 각 run 시작 전이다. 이는 수정 시각에 근거한 추정이며 해시 기록으로 확인한 것은 아니다.
- 검증: 복사본 43개 `.py` 구문 확인, 진입 스크립트 3개 `--help`(전체 import 포함) 실행 성공. 학습 재실행은 하지 않았다.

## 실행 방법

코드는 `ROOT = Path(__file__).resolve().parents[2]`를 기준으로 데이터를 찾는다. 데이터를 복사하지 않았으므로 **프로젝트 루트에 있는 원래 위치의 코드로 실행**하거나, `code/` 아래 같은 구조에 데이터를 둔다.

- 입력: `data/processed/preprocessed_20261002/timeseries_15min_v1_previous_week.csv` (SHA256 `d12b495c…74ea5e`)
- 실행 환경: Python 3.13.5, numpy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, xgboost 3.4.1 (나머지 버전은 `configs/*/config.json`의 `versions`)

```bash
python -u -B -X utf8 notebooks/models/random_forest/run_rf_m1007.py --workers 2
```

```bash
python -u -B -X utf8 notebooks/_shared/run_boosting_m1007.py --library xgboost --run-id <run_id>
```

`--library`를 `lightgbm`·`catboost`로 바꿔 3번 실행한다. 8월 재학습은 7월 run 결과 폴더를 입력으로 받는다.

```bash
python -B -X utf8 notebooks/_shared/m1007_august.py rf --rf-run <RF 7월 run 폴더> --out <출력 폴더>
```

```bash
python -B -X utf8 notebooks/_shared/m1007_august.py boost --library xgboost --boost-run <부스팅 7월 run_id> --out <출력 폴더>
```

주의: 이 스크립트들은 실행할 때 `experiments/`, `artifacts/`, `final/README.md`, `experiments/README.md`에 새 기록을 쓴다. 재현 실행은 프로젝트 사본에서 하는 것이 안전하다.

## 공통 설정 요약

- 예측 구간: 1h·4h·8h·12h·24h·4d (15분 칸), 목표 `power_15min`(kW), 학습 손실 C, seed 42
- 특성: F1/F2/F3 중 구간별 선정, future 시나리오(미래 생산·기상은 **사후 실측**을 예보로 가정), 월 칸 제거
- 부스팅 나무 수: 학습 끝의 복사되지 않은 날 14일로 최적 round(최대 60)를 정한 뒤 전체 재학습
- 선정 기준: 7월 공통 C 최소. 8월·9월 성적으로 다시 고르지 않음

원실험 기록: [RF 7월](../../experiments/random_forest/20261007T081917_KST_f4847a_rf_m1007/) · [XGBoost 7월](../../experiments/xgboost/20261007T090243_KST_m1007_boost/) · [LightGBM 7월](../../experiments/lightgbm/20261007T090243_KST_m1007_boost/) · [CatBoost 7월](../../experiments/catboost/20261007T090243_KST_m1007_boost/) · [8월 재학습 기록](../../reports/20261007T102741_KST_m1007_august_margin/records/) · [재실험 결정 문서](../../docs/meeting_20261007_rerun_plan.md)
