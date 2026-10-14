# 연구 근거 기반 모델 튜닝과 중·장기 참조 윈도우

작성: 2026-10-03, Asia/Seoul. 사용자 요청: RF·XGBoost·LightGBM·CatBoost를 연구 근거로 튜닝하고, 중기4/8/12h·장기3d의 참조 윈도우를 검토해 실제 학습한다. 8월 검증 후 모델/설정을 동결하고 9월 제공분으로 최종 결과·보고서를 작성한다. CPU 학습 예산60분을 사용자가 선택했다.

## 원문에서 확인한 근거

1. **Dudek (2022), A Comprehensive Study of Random Forest for Short-Term Load Forecasting, Energies15,7547.** [저자 소속기관 원문](https://gdudek.el.pcz.pl/images/files/Energies_22.pdf), [DOI](https://doi.org/10.3390/en15207547). 4개 국가의 시간별 부하·하루 경로 예측. §2는 전일·전주·21일 패턴을 비교하고 §5.2는 트리 수·leaf 최소 표본 수·분할 특성 수를 비교한다. §4의 n/3 특성 비율 및 leaf5는 출발점이고 최선값은 데이터에 따라 달랐다. **적용:** leaf1/5/20 및 feature1/3/0.7/1.0의 후보를 포함하고 무조건 트리 수만 증가시키지 않는다. 국가 집계와 이 프로젝트의 단일 공장 부하는 다르므로 논문의 수치나 우승 모델을 이전하지 않는다.
2. **Hakyemez & Adar (2024), Testing the Efficacy of Hyperparameter Optimization Algorithms in Short-Term Load Forecasting, arXiv:2410.15047.** [원문](https://arxiv.org/html/2410.15047v1). 시간별 Panama 수요·기상, 다음1h XGBoost. TableIII에서 depth3\~10, trees100\~1000, 학습률 및 row/column sampling, min_child_weight를 탐색했다. 50회 HPO와 실행량을 비교한 프리프린트다. **적용:** depth3/4/6, rounds200\~600, lr.03/.05/.08, sampling.8\~1.0, child weight3/10/20를 제한적으로 조합한다. 그 연구의 A100 GPU·랜덤CV·minmax 전처리를 복제하지 않고 기존 시간순 원단위 평가를 유지한다.
3. **Muqtadir et al. (2025), Nowcasting the next hour of residential load using boosting ensemble machines, Scientific Reports15,7157.** [원문](https://www.nature.com/articles/s41598-025-91767-6). 건물 smart-meter 다음1h의 XGB·LGBM·CatBoost 비교 및 앙상블 연구. XGB 학습률/rounds/정규화 조합 탐색을 설명한다. **적용:** 세 boosting 모델을 같은 원점과 목표로 각각 검증한다. 주거·여러 건물 자료이며 산업 부하나 본 프로젝트의 3일 성능을 입증하지 않는다. 원문의 classifier 표현·일반 GridSearchCV를 이 프로젝트에 그대로 적용하지 않는다.
4. **Research on Ultra-Short-Term Load Forecasting Based on Real-Time Electricity Price and Window-Based XGBoost Model (2022), Energies15,7367.** [원문](https://psecommunity.org/wp-content/plugins/wpor/includes/file/2302/LAPSE-2023.9456-1v1.pdf), [DOI](https://doi.org/10.3390/en15197367). 창으로 구성한 과거 부하를 XGB 입력으로 이용하는 초단기 예측 연구. **적용:** 예측 시점까지 완료된 전력만 window 통계에 포함하고 다음 경로의 실제 전력을 중간에 입력하지 않는다. 실시간 가격 설명변수는 이 데이터에 없어 추가하지 않는다.
5. **Wang et al. (2021), Short-Term Load Forecasting for Industrial Customers Based on TCN-LightGBM.** [저자 원문 레코드](https://authors.library.caltech.edu/records/p17n6-54p35). 개별 산업 부하의 변동성과 고정 길이 sliding window의 전력 특성 재구성을 설명한다. **적용:** 산업 부하에서도 입력 참조 길이를 검증해야 한다는 근거로만 사용한다. TCN 새 모델이나 논문의 개선율은 이 실험에 도입하지 않는다.

## 공식 구현 문서와 모델별 탐색 축

- [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html): depth/child weight/row·column sampling/alpha·lambda는 모델 복잡도와 정규화 제어. 설치3.4.1의 실제 sklearn API로 실행한다.
- [LightGBM tuning](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html): leaf-wise 복잡도는 num_leaves와 min_data_in_leaf를 함께 제어하며 row subsample은 subsample_freq>0일 때 작동한다. leaf15/31/63, 최소 leaf 표본20/40/80, lambda1/5/10, sampling.8\~1.0 후보. lr와rounds를 짝지어 탐색한다.
- [CatBoost tuning](https://catboost.ai/docs/en/concepts/parameter-tuning): depth4\~10, learning_rate, l2_leaf_reg, random_strength, Bayesian bagging_temperature를 조정한다. 이 CPU 예산에서는 depth4/6/8, L2=3/10/20, randomness.5/1 및 rounds200\~600 후보. 기존 SymmetricTree에서 min_data_in_leaf는 유효 튜닝 축이 아니므로 사용하지 않는다. 범주 전처리는 기존과 같게 두어 파라미터 효과를 분리한다.

## 본 프로젝트에서의 설계 판단 — 논문 최적값이 아님

- 기존15분 kW·단기1h/중기4/8/12h·장기3d·관측전용/미래실측대체 조건·F1/F2/F3·seed42 유지. 주 지표는 월별 동일 가중 RMSE; 위험window1h/1d의CVaR와 과소예측·피크 시점도 보조 보고.
- **참조 윈도우**는 과거 전력 통계를 계산하는 길이이며 예측 H나 학습 데이터 기간과 다르다. 누적 expanding 학습은 유지한다. 기존 시차/1h·4h·24h 통계·전일cycle·전주 패턴은 유지하고, F3에 context mean/max/std/lag/current-minus-mean을 추가한다. 중기1/7/14일, 장기7/14/28일. 해당 profile의 최대 window가 유효한 원점으로 F/모델/window 전체를 맞춘다. F1/F2는 window 특성이 없어 중복 학습하지 않는다. 기본 context없음도 후보로 포함한다. 14·28일은 주기 배수의 프로젝트 후보이며 논문에서 보장한 최적값이 아니다.
- 시간 budget 때문에 순차 탐색: 기존 선정F에서 모델별3개 새 파라미터 조합을6·7월로 선별 → 선별 파라미터의 F1/F2/F3와 F3 context길이를 비교 → 기존 기준과 유망 후보를4\~7월 전부 검증. 파라미터×F×window 모든 Cartesian 조합을 최적화한 결과로 주장하지 않는다. 모델별/구간별 순서 편향을 줄여 후보번호를 번갈아 실행한다.
- 최대60분 새fit 시작 예산을 튜닝·월별검증·8/9월 고정평가에 배분한다. 사전에 정한 후보 수와 시간 상한을 둘 다 적용하고 예산 제한으로 시작 못 한 후보는 미실행으로 기록한다. 각 모델에서 최소한 기준+새 후보를 확보하고 종료 평가 실행량을 먼저 확보한다. 실행 중 fit과 산출물 저장은 완료할 수 있다.
- 8월은 이미 이전 실험에서 본 추가 검증월이다. 4\~7월로 모델별 설정을 선택하고 8월에는 파라미터/window를 다시 탐색하지 않는다. 8월 RMSE로 사전에 정한 최종 모델 선택 규칙만 적용한 뒤 선택과코드 hash를 동결한다. 9월 target를 읽기 전에 동결 manifest를 저장한다. 9월으로 추가튜닝·window선택·모델교체하지 않는다.
- 9월 제공 시각은2021-09-01 00:00\~2021-09-15 00:00(9/1\~14의1344개15분 구간). 월 전체 성능이 아니며 길이H만큼 말단 평가 원점이 줄어든다. 3일 원점은 중첩하고 적어 불확실성이 크다. 실제 생산·날씨 예보자료는 없어 future 조건은 zero-error actual proxy로 유지한다.
- 원본/이전 결과는 보존. 모든 실제 후보에 독립MD·설정·시각키예측·해시·fold별 점수·시간/관측RSS·저장모델 검증·실험인덱스를 남긴다. 보고서는 프로젝트 폴더에 저장하고 이 대화에서 파일 링크로 제공한다.
