# 전체 Loss C 부스팅 학습: 근거와 적용 한계

- 확인일: 2026-10-04, Asia/Seoul.
- 상태: 공식 API 문서와 원 논문의 설명을 확인한 **설계 근거**. 새 모델 학습 및 성능 실험은 실행하지 않았다.
- 연결 문서: [실험 설계](<<PROJECT_ROOT>/docs/superpowers/specs/2026-10-04-boosting-full-c-training-design.md>).
- 근거 수준을 구분한다: 아래 1\~4는 출처에서 확인한 설명, 5\~6은 프로젝트 코드 확인 및 이에 기반한 설계 판단이다.

## 1. XGBoost 사용자 정의 목적의 제약

공식 문서는 사용자 정의 gradient/Hessian으로 학습할 수 있음을 설명한다. 정상 동작의 조건으로 매끄러운 이계 미분, 관측 행에 대한 가산성, convex 목적 및 다중 출력에서의 Hessian 구조를 든다. 문서는 Hessian 근사 등의 우회가 가능함도 설명한다.

[Advanced Usage of Custom Objectives, XGBoost 공식 문서](https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html)

**프로젝트 적용 판단:** C의 구간 RMSE는 여러 셀을 함께 사용하고, 상위 20% 구간 평균은 현재 전체 구간 순위에 의존한다. 따라서 단순 행별 손실의 정확한 Hessian으로 표현할 수 없다. 전체 배열에서 C gradient를 계산하고 대각 Hessian 대리값을 제공하는 연구용 구현은 가능성을 검토할 수 있으나, 정확한 C Hessian을 사용한 기본 XGBoost 학습이라고 설명할 수는 없다.

확인한 stable 문서는 3.4.2이고 설치 버전은 3.4.1이다. 실제 구현 전에 설치 버전의 API·base_score·1tree 동작을 확인해야 한다.

## 2. LightGBM의 gradient/Hessian 반환 형식

공식 train API는 사용자 정의 objective가 예측 배열과 Dataset을 받아 샘플별 1차·2차 미분 배열을 반환하도록 설명한다. 사용자 정의 평가 함수와 학습 objective는 다른 인자다.

[lightgbm.train, LightGBM 공식 문서](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html)

**프로젝트 적용 판단:** 전체 학습 배열을 이용해 C gradient를 계산할 여지는 있지만, 반환 형식만으로 구간 간 결합을 포함한 전체 Hessian을 전달하는 것은 아니다. metric에 C를 추가한 것만으로 학습 목적이 C가 되지 않는다. XGB/LGB만 네이티브 우회 방식으로 학습하고 CatBoost는 다른 손실을 쓰는 혼합 실험은 세 모델이 같은 방식으로 전체 C를 학습했다는 주장에 부적합하다.

## 3. CatBoost 콜백의 처리 범위

공식 사용자 정의 목적 예제의 calc_ders_range는 approxes/targets/weights를 받고 예측값에 대한 음의 1차·2차 미분 쌍을 반환한다. 예제는 전달되는 배열을 현재 처리 중인 데이터 부분집합으로 설명한다. 사용자 정의 metric은 별도 인터페이스다.

[Usage examples: User-defined loss function, CatBoost 공식 문서](https://catboost.ai/docs/en/concepts/python-usages-examples#user-defined-loss-function)

**프로젝트 적용 판단:** 해당 콜백에서 항상 전체 원점의 모든 lead와 전체 fold의 위험 구간이 전달된다고 전제할 수 없다. 그 부분집합에서 top20%를 구하면 전체 C와 다르다. 표준 공개 콜백만으로 전체 C의 정확한 경로·순위를 처리한다고 선언하는 설계는 채택하지 않았다. 이는 CatBoost의 다른 모든 확장 방법이 불가능하다는 주장이 아니다.

## 4. 함수 공간 부스팅·비매끄러운 목적에 대한 원 논문

Grubb와 Bagnell의 연구는 일반적인 convex 목적의 부스팅을 분석한다. 매끄러운 경우의 기존 gradient boosting 결과를 비매끄러운 목적에 그대로 확장할 수 없고, 이를 위한 별도 알고리즘이 필요함을 설명한다.

[Generalized Boosting Algorithms for Convex Optimization, arXiv:1105.2054v2 / ICML 2011](https://arxiv.org/abs/1105.2054v2)

**프로젝트 적용 판단:** 기저 학습기가 전체 목적의 gradient/부분경사 방향을 근사하도록 하는 외부 루프는 연구 흐름에 근거한 설계다. 다만 제안한 “부분경사 → 1tree 근사 → 실제 C 감소 스텝만 채택”이 이 논문의 특정 알고리즘이나 그 수렴 보장을 그대로 구현한 것은 아니다. 하드 top-K 동률 및 norm 영점에서 정체 가능성을 명시하고 실제 C 감소·거부 로그를 필수로 남긴다. 원래 C가 내려간다는 관측과 전역 최적해 보장은 구분한다.

이 연구가 현재 KAMP 데이터에서 C가 RMSE보다 성능이 좋다고 증명하는 것은 아니다. 그 결론은 동일 자료·같은 외부 학습기·동일 5-fold의 실제 비교로 검증해야 한다.

## 5. 프로젝트에서 확인한 사실

| 사실 | 확인 위치 |
|---|---|
| C=.5 mean+.3 tail+.2 peak, top-K=ceil(.2 M), peak=sqrt(mean(w × positive_under_error²)) | [rf_current_data.py](<<PROJECT_ROOT>/notebooks/models/random_forest/rf_current_data.py:138>) |
| 단·중기 위험 구간 4셀, 장기 96셀, B의 월중 최댓값 갱신 및 연속 가중치 | [loss_c_fivefold.py](<<PROJECT_ROOT>/artifacts/performance/runs/20261004T040425_KST_cffc42_loss_c_fivefold/code/frozen_final/loss_c_fivefold.py:33>) |
| 이전 부스팅의 학습 가중치는 lead별 sample_weight이고 원래 목적은 제곱오차 | [weighted_tree_fivefold.py](<<PROJECT_ROOT>/artifacts/performance/runs/20261004T040425_KST_cffc42_loss_c_fivefold/code/frozen_final/weighted_tree_fivefold.py:24>) |
| 중·장기에서 최대 60,000행 한도로 원점당 일부 lead만 선택 | [performance_profiles.py](<<PROJECT_ROOT>/notebooks/_shared/performance_profiles.py>) |
| 434개 완료 점 예측 후보와 340개 가중치 파일의 추적 감사 통과; 현재 학습은 전체 C 직접 학습이 아님 | [L_peak 추적 보고서](<<PROJECT_ROOT>/reports/l_peak_trace_20261004T134256_KST/L_PEAK_TRAINING_TRACE.md>) |

## 6. 검증되지 않은 부분과 확인 방법

- 세 라이브러리의 fresh 1tree 반복 비용: 실제 파일럿에서 wall time·fit 수·예측 비용·peak RSS 측정.
- 전체 C 부분경사의 수치 정확성: 독립 autodiff/유한차분 fixture와 기존 평가기 일치 검사. 설계 시 수식만 대조했으며 새 검증 코드는 아직 실행하지 않았다.
- 전체 C 직접 학습의 실제 진행 가능성: 모든 라운드에서 C_before/C_after·gradient·λ·거부 원인 확인. 비매끄러운 지점에서 정체한 실행도 공개.
- 원래 모델 대비 성능 개선: 새로운 완전 경로 자료에서 A/C를 같은 외부 루프로 학습한 비교가 필요. 이전 sparse lead 모델과의 차이를 손실 변경만의 효과로 귀속하지 않음.
- 피크 확률·알림의 개선: 별도 확률 추정·보정·같은 정책 적용 실험 필요. C 회귀만으로 확률을 학습했다고 하지 않음.
- 9월 일반화: 기존 결과 노출 및 전체 q90로 정한 초기 B 한계를 유지. 신규 미래 자료에 대한 독립 테스트는 아직 없음.

이 문서의 권장 파라미터는 문헌에서 해당 데이터의 최적값을 가져온 것이 아니다. 기저 트리의 복잡도를 고정하고 C의 학습 참여 여부를 먼저 검증하기 위한 출발 설정이다.