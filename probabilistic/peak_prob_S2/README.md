# 피크 초과확률 모형 S2 (부가 출력, 실험적)

전력 점예측에 덧붙이는 **"이번 예측 구간 안에 이번 달 최고(b)를 넘을 확률"** 출력이다. 경보 판단은 본 프로젝트의 주 경로(예측 + 마진, `src/4_마진경보`)가 맡고, 이 확률은 참고 정보로만 쓴다.

> 이 모형의 전력 예측 입력은 **트리 4모델(RF·XGBoost·LightGBM·CatBoost) 단순평균(m1007)** 이다. 최종 제출 모델(딥러닝 앙상블, `src/2_딥러닝`)과 다르다.

## 정의

- 사건: 발행 원점의 예측 경로(H칸) 중 한 번이라도 `y > b_origin` (엄격한 >)
- `b_origin = max(170 kW, 그달 발행 전까지 관측된 최대 전력)`, 매달 초기화
- **S2 = 0.5 × LightGBM v2 분류기(seed 5개 평균) + 0.5 × 잔차 경로 분포 확률**
  - 잔차 경로 분포: 과거 예측 오차 경로 `e = y − ŷ`(학습 직전 2달, 시간순 OOF)를 현재 예측 경로에 더해 만든 경험분포에서 `P(any(ŷ + e > b))`
  - 비중 0.5는 고정값(튜닝하지 않음)

## 성능 (검증 = 5\~8월 달별 굴리기, 주 지표 = 새 피크 AUC)

"새 피크" = `b_origin ≠ 170` 원점(월초 쉬운 사건 제외). 제공 기준: 검증 새 피크 AUC ≥ 0.70 그리고 AP ≥ 2 × 발생률.

| 구간 | 현행 잔차 | v2 LightGBM | S1 분리 | **S2** | 9월 AUC(참고) | 판정 |
|---|---|---|---|---|---|---|
| 1h | 0.907 | 0.835 | 0.825 | **0.900** | 0.951 | 제공 |
| 4h | 0.844 | 0.809 | 0.735 | **0.863** | 0.825 | 제공 |
| 8h | 0.793 | 0.622 | 0.700 | **0.797** | 0.763 | 제공 |
| 12h | 0.759 | 0.492 | 0.693 | **0.752** | 0.676 | 제공(경계) |
| 24h | 0.622 | 0.398 | 0.510 | 0.610 | 0.522 | 미제공 |
| 4d | 0.526 | 0.443 | 0.443 | 0.496 | 0.562 | 미제공 |

S2를 고른 이유: 새 피크 구분력은 현행 잔차와 비슷하면서, 8/9 휴무 뒤 재가동을 4h\~24h 앞에서 0.33\~0.46으로 표시했다(현행 0.06\~0.13, `results/hits_key_days.csv`).

## 한계 (반드시 함께 읽기)

- **주 지표를 결과를 본 뒤 바꿨다.** 처음에는 Brier였으나, v2가 24h Brier를 0.074 → 0.046으로 줄이면서 새 피크 AUC는 0.62 → 0.40으로 떨어져 "새 피크 AUC"로 바꿨다. S2 설계도 결과 열람 후 결정이다.
- **9월은 준독립 평가다.** 이전 실험에서 9월 결과를 본 이력이 있다.
- **확률을 경보 임계값으로 쓰지 않는다.** 검증에서 헛경보 날 20%에 맞춘 임계값이 9월에는 43\~64%가 됐다.
- 사건 날이 달마다 1\~2일뿐이다. 월초 b=170 사건(7/1·9/1)은 모두 맞혔지만, 월 최대 갱신(7/19·9/10)은 어느 모델도 평일과 구분하지 못했다.
- 미래 생산량·기상은 실제값을 예보로 쓴 "다앎" 가정이다.

## 폴더

| 경로 | 내용 |
|---|---|
| `infer_s2.py` | **추론 전용 스크립트**(학습 없음). 저장 모델 + 잔차 풀로 S2 확률 계산. `--selftest`로 로드·형상 점검 |
| `model/model_v2_<구간>.joblib` | LightGBM 분류기 seed 5개 + 특성 목록 + 설정 (`{"models", "features", "profile", "config", "train"}`) |
| `model/<구간>_residual_pool.npz` | 잔차 풀 `residual (N,H)` kW, `origins (N,)` ns. 원본 풀에서 **실측 전력(`historical_y`)과 b 열을 빼고** 다시 저장한 판 |
| `model/input_schema.json`, `v2_selection.json`, `provision_S2.json` | 특성 이름, 구간별 선정 설정, 제공 판정 |
| `code/` | 학습·평가 코드 사본(1·2·3차 실험). 원래 프로젝트 폴더 구조와 중간 산출물(게이트 OOF 등)을 읽으므로 **이 저장소만으로는 재학습되지 않는다** |
| `records/`, `results/` | 실험 기록(1차 M1\~M3, 2차 v2 설정 탐색, 3차 S1·S2)과 점수표, 최종 보고서 `results/final_report.md` |
| `BUNDLE_README.md` | 원 실험 묶음의 README |

## 특성 (원점 단위, 발행 시점 정보만)

`code/m1007_gate.py`(칸 단위)와 `code/m1007_rare_peak_models.py::origin_table`(원점 집계)에서 만든다. 4h는 `plus` 판으로 5개가 더 붙는다(`code/m1007_rare_peak_v2.py::peak_features`).

| 특성 | 정의 |
|---|---|
| `yhat_b_max`, `yhat_b_mean` | 4모델 평균 예측 ŷ/b의 경로 최대·평균 |
| `maxmember_b_max` | 칸별 4모델 중 최대 예측/b의 경로 최대 |
| `spread_b_mean` | 칸별 4모델 표준편차/b의 경로 평균 |
| `latest_b`, `max1h_b`, `max24h_b` | 원점의 전력, 직전 1시간·24시간 최대 전력 / b |
| `b`, `b170` | b_origin(kW), b = 170 여부 |
| `day_of_month`, `hour`, `weekday` | 원점의 일(+시/24), 시각, 요일 |
| `op_ratio`, `prod_max`, `prevday_zero` | 경로 중 평일 칸 비율, 경로 시간대 계획 생산량 최대, 전날 생산 0 여부 |
| (4h plus) `yday_max_b`, `recent5_opmax_b`, `days_since_b_up`, `ramp1h_b`, `prev_month_max_b` | 어제 최대/b, 최근 가동 5일 최대 중앙값/b, b가 마지막으로 오른 뒤 일수, 1시간 전 대비 증가/b, 지난달 최대/b |

## 실행

```bash
pip install -r requirements.txt
python infer_s2.py --selftest
python infer_s2.py --profile short_1h --features feats.csv --yhat yhat.npy --b b.npy --out p_s2.csv
```

`--selftest`는 합성 입력으로 6개 구간의 로드·형상·확률 범위만 확인한다(값의 의미는 없음). 확인 환경: Python 3.13, lightgbm 4.7.0, scikit-learn 1.6.1, joblib 1.6.0, numpy 2.5, pandas 2.3.3. 원 학습 환경은 `code/environment.txt`(Python 3.13.5).
