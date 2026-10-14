# 딥러닝·미사용 코드 정리

2026-10-04 사용자 요청으로 딥러닝을 현재 작업 범위에서 제외했다. 이전 딥러닝 실험이 모두 잘못되었다는 뜻은 아니다. 새 모델 성능 실험은 하지 않았다.

- 제거: DLinear·GRU·PatchTST·LSTM·NHITS·Simple RNN 전용 노트북/코드, neural 학습·복구·보고·검증 실행기와 전용 테스트, 오래된 실행 포인터·계획·로그·설치 스크립트.
- 공통 모듈: forecast_common.py의 딥러닝 학습 분기와 딥러닝 패키지 기록만 제거했다. 시간순 분할·경로 집계·RF·기준모델 기능은 유지한다.
- 보존: RF·부스팅 관련 코드, 원본·전처리·F2/F3 데이터, 설치된 Python 환경 및 기존 artifacts의 학습 결과. 보관된 결과를 현재 실행 대상이라고 소개하지 않는다.
- [개별 실험 원기록](experiment_records.zip): DLinear 35개, GRU 35개, PatchTST 36개, 총 106개 MD를 원래 바이트로 압축 보존했다.
- [정리 manifest](manifest.json): 삭제 경로·해시·보호 대상·검증 결과.
- 구식 RF 경보 진단 모듈과 테스트 2개는 자동 승인 검토에서 복구 불가능한 삭제가 거절되어, [복구 가능한 보관](retained_rf_diagnostics/moves.json)으로 처리했다. 파일 내용과 해시는 유지했고 현재 RF 경보 정책 코드는 수정하지 않았다. 원위치와 보관 위치가 moves.json에 있으며 원위치로 이동하면 복구할 수 있다.

현재 모델 작업은 [RF](../../notebooks/models/random_forest/RF_SCENARIO_OVERVIEW.md), [부스팅](../../notebooks/models/boosting/README.md), 기준모델에 한정한다. tests의 딥러닝 제거 검증은 모델을 학습하거나 딥러닝 패키지를 로딩하지 않는다.

검증: 관련 테스트 21개 통과, 원본/전처리 데이터 148개 변경 없음, 남은 실행 코드의 딥러닝 모듈 참조 없음, 보관 원기록 106개 해시 일치 및 현재 문서 링크 23개 확인. RF 보고 코드 1개는 정리 밖 동시 변경을 감지해 그대로 두었다. [검증 결과](verification.json).
