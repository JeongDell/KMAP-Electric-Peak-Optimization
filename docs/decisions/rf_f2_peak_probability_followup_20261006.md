# F2 RF 변경 구간 피크 확률 후속 실행

- 사용자 지시: 피크 확률 모델도 바뀐 부분에 한해서 실행.
- 범위: `20261006T171055_KST_e7e556_rf_integrated_grid`의 medium_8h / medium_12h / medium_24h, 피처 F2, 48시간 입력 통계 창. 이 세 구간의 F2 강제 튜닝과 선정 RF 설정 변경에 따른 확률 후속 작업이다.
- 목적: 원점까지 관측한 이번 달 최대(최소 170 kW)를 향후 초과할 시점별·기간별 확률과 원시 경로 분위수를 제공한다. 운영 경보 비용·tau는 이번 범위에서 선정하지 않는다.
- 데이터: 기존 15분 power_15min kW, 2021-01-01\~2021-09-14, `data/processed/preprocessed_20261002/timeseries_15min_v1_previous_week.csv`, SHA256 `d12b495c8ce243dcbd801432d2f02c585df3c8e4da981175c0e662b08674ea5e`.
- 출력: 8시간 32점, 12시간 48점, 24시간 96점. 원점마다 임계값을 고정하고 미래 실제 전력을 중간 입력으로 쓰지 않는다.
- 학습: 2\~6월 expanding 과거 OOF RF 5개 × 3구간 = 신규 15개. 7·8월 및 9월의 C 선정 RF는 각 3구간 × 3개 = 기존 9개 재사용. seed 42, 원 RF 설정·C 잎 조정 방법 그대로.
- 확률 후보: RF 원시 조건부 경로 분포(raw), 과거 OOF sigmoid 보정(calibrated), 같은 학습 경로의 무조건 분포(history 기준선).
- 주 지표: 7·8월 full의 월 동일 평균 `0.5*(point Brier + period Brier)`, 작을수록 좋음. raw/calibrated 선택을 동결한 후 9월 평가. logloss·reliability·분위수 coverage/Pinball·조건별 오차·사건 수 병기.
- 누수 점검: 보정 및 RF 학습 전체 경로 끝 < 평가월 시작. 평가 경로 끝 < 다음 달 시작(9월 < 9월 15일). 월 최대는 원점까지의 같은 달 관측값만 사용. 저장 모델 추론·예측 키·Brier·확률 관계·전체 경로 표본을 독립 검증한다.
- 해석: 원 피처 비교는 이 세 구간에 F1을 선정했고 이번 F2는 추가 비교 시나리오다. 동일 7·8월 C 튜닝에 조건부인 확률 선정이며 nested CV가 아니다. 전체 q90·기열람 이력으로 9월은 탐색적 평가다. 미래 생산·기상은 사후 실측을 오차 0 예보로 사용하는 조건.
- 자원: 지역 CPU 2구간 병렬, RF 4스레드, BLAS 1, 이 후속 실행 최대 14,400초. 새 run_id로 기록하고 기존 모델·완료 묶음은 보존한다.
- 사전 검증: 원 RF 완료·독립 검증·final 보관, 데이터·피처·학습 코드 해시 확인 통과. `test_rf_peak_model.py` 7개 테스트 통과.
- 실행: `./.venv_boosting/Scripts/python.exe -B -X utf8 notebooks/models/random_forest/rf_peak_model_experiment.py --tuning-run artifacts/random_forest/runs/20261006T171055_KST_e7e556_rf_integrated_grid --budget-seconds 14400`.
- 보관: 새 `experiments/random_forest/<run_id>/records/`에 21개 개별 기록(3후보 × CV/9월 × 3구간 + warm OOF 3개). 독립 검증 후 `final/random_forest/<run_id>/`와 두 인덱스를 갱신한다.
- 노트북: 기존 커널 응답 지연을 피하기 위해 저장 결과를 새 IPython 프로세스에서 순차 실행하고 실행 방식·출력을 기록한다. Jupyter 커널 통신 성공과 구분한다.
