# 피크 확률 S2 — 20261008T104516_KST_9e4a00_m1007_rare_peak_v3

상태: 완료. 선정 = S2(0.5 × v2 LightGBM 분류기 + 0.5 × 현행 잔차 분포). 제공 = 1h·4h·8h·12h(12h 경계), 미제공 = 24h·4d. 9월은 준독립 확인.

| 구간 | 검증 새 피크 AUC | 판정 |
|---|---|---|
| 1h | 0.900 | 제공 |
| 4h | 0.863 | 제공 |
| 8h | 0.797 | 제공 |
| 12h | 0.752 | 제공(9월 0.676) |
| 24h | 0.610 | 미제공 |
| 4d | 0.496 | 미제공 |

- [최종 보고서](results/final_report.md) · [실험 요약](records/summary.csv) · [제공 판정](model/provision_S2.json)
- 기록: [1차](records/v1/) · [2차](records/v2/) · [3차·재채점·적중 확인](records/v3/)
- 모델: model/model_v2_*.joblib(구간별 seed 5개), model/*_residual_pool.npz(잔차 풀), [입력 스키마](model/input_schema.json), [v2 선정](model/v2_selection.json)
- 코드: code/*.py 사본, [소스 해시](code/source_sha256.json), [환경](code/environment.txt), [재현 명령](code/reproduce.md)
- 결과: [3차 점수](results/v3_scores.csv), [새 기준 재채점](results/new_criteria_scores.csv), [주요 사건일](results/hits_key_days.csv), [추론 점검](results/inference_check.json)

원천 데이터·전력 모델·OOF 예측은 프로젝트 경로에 그대로 두고 경로·해시로 연결했다. 결정 이력: docs/meeting_20261007_rerun_plan.md.
