# Loss C와 피크 확률 실험의 근거

> 2026-10-04 현재 결정: 새 부스팅 학습·주 평가에서 RF 참조 확률 인자를 제거했다. 아래 문헌·과거 수식은 근거 이력이며 RF 인자 재실험을 요구하지 않는다. [현재 방향](<<PROJECT_ROOT>/notebooks/models/boosting/README.md>).

기존 네 모델의 학습 파라미터 탐색 근거는 [전력 예측 튜닝 연구 검토](2026-10-03-load-forecasting-tuning-evidence.md)를 유지한다. 아래 자료는 새 확률 실험과 평가 절차의 근거다. 논문의 사례가 이 공장의 특정 상한·입력 창·파라미터를 입증한 것으로 표현하지 않는다.

1. [Meinshausen, Quantile Regression Forests, JMLR 2006](https://www.jmlr.org/papers/v7/meinshausen06a.html). 잎에 속한 실제 학습 목표값의 조건부 분포를 이용하는 RF 분포 확장의 근거다. 참조 채팅의 native 다변량 미래 경로 확장은 원 scalar 이론과 구분한다. 나무 평균 점 예측의 빈도를 사건 확률로 계산하지 않는다.
2. [scikit-learn 확률 보정 문서](https://scikit-learn.org/stable/modules/calibration.html). 확률 보정에 독립된 적합/보정 데이터가 필요하고 sigmoid와 isotonic 방법을 설명한다. 적은 피크 양성 표본에서 유연한 isotonic 탐색을 추가하기보다 고정 2계수 단조 sigmoid를 과거 28일 보정 구간에서만 학습한다. 양성/음성 둘 다 없으면 Jeffreys 평활 상수 확률을 기록한다. 이 작은 표본 대체 규칙은 프로젝트의 명시적 구현 선택이다.
3. 같은 문서의 Brier/log loss는 신뢰도와 구별력 등을 함께 평가하므로 Brier만으로 보정이 완벽하다고 주장하지 않는다. 확률 구간별 평균예측·실제사건빈도·표본 수를 함께 보고한다.
4. [XGBoost 공식 파라미터](https://xgboost.readthedocs.io/en/stable/parameter.html): binary:logistic은 이진 확률 출력 목적이다. reg:squarederror 점 예측과 분류를 별도 실험으로 기록한다.
5. [LightGBM 공식 파라미터](https://lightgbm.readthedocs.io/en/latest/Parameters.html): binary objective를 사용한다. 클래스 불균형을 자동 재가중하는 is_unbalance/scale_pos_weight는 개별 확률 추정을 악화시킬 수 있다고 명시되어 있어 사용하지 않는다.
6. [CatBoost 이진 목적](https://catboost.ai/docs/en/concepts/loss-functions-classification), [predict_proba](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_predict_proba): Logloss 목적의 실제 분류기를 학습하고 positive class probability를 저장한다.
7. [FPP3 시계열 교차검증](https://otexts.com/fpp3/tscv.html): 과거로 적합하고 이후 기간에서 평가하는 원칙을 4\~8월 expanding 5-fold에 적용한다. 시간 순서를 무시한 무작위 KFold를 사용하지 않는다.

확률 분류기의 small/regular 설정은 검증할 소수 후보로 사전 정의한 CPU 예산 선택이다. small/regular의 깊이·leaf·정규화가 연구로 입증된 최적값이라는 뜻은 아니다. 선택은 실제 완료 5-fold의 월별 동일가중 Brier로 한다. 구간 사건은 실제 경로 max > B로 정의하며, 시점 독립성 곱은 사용하지 않는다.

Loss C의 공통 가중치는 참조 채팅이 채택한 식을 그대로 적용한다.

```text
g = clip((actual/B - 0.8)/0.2, 0, 1)
w = 1 + (cap-1)*g*(1+p_reference)/2
C = .5 L_mean + .3 L_CVaR + .2 sqrt(mean(w*max(actual-predicted,0)^2))
```

대표 선정 cap=2는 후보마다 같은 평가 기준이다. cap1/2/4의 C 숫자 자체를 서로 최소화해 cap를 정하지 않는다. p_reference는 동일 무가중 native RF/F2/미래 외생 오차0 조건의 해당 월 OOF raw 확률이다. C 점 모델 후보 자신의 확률에 따라 평가 가중치를 바꾸지 않는다.

초기 전체 q90=170kW는 사용자 지정 탐색 기준으로9월까지 포함한다. 9월은 과거 프로젝트에서 이미 평가했으므로 이번에도 새 blind test라고 표현하지 않는다. 9월 점수로 모델·입력 창·C·확률 보정을 고르지 않는다.
