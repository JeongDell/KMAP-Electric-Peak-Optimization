# 8월 재학습 실행 당시 `m1007_august.py` 복원본

복원일: 2026-10-08. 원본 작업 파일(`notebooks/_shared/m1007_august.py`)은 그대로 두었다.

| 파일 | SHA256 | 대응 run 기록 |
| --- | --- | --- |
| [m1007_august_rf_run.py](m1007_august_rf_run.py) | `8998895acb85ecdec1e872e2817fd59c70a51d497bddfdc083f7d3445fe592dd` | RF 6개 기록(`*_c_leaf_d10_august.md`, 10:27:54~10:31:18 KST 작성) |
| [m1007_august_boosting_run.py](m1007_august_boosting_run.py) | `fce81af88c8fb28f38717f6b45112b7554be470d11e2585130fe3694afc332ad` | 부스팅 18개 기록(xgboost·lightgbm·catboost, 10:43:31 KST 작성) |

두 파일 모두 해시가 기록값과 **바이트 단위로 일치**한다.

## 복원 방법

실행 기록에 남은 이 파일의 수정 이력을 시간순으로 다시 적용해 복원했다. 같은 방법으로 마지막까지 적용하면 현재 작업본 해시(`ede36c5b…`)가 그대로 재현되므로 복원 방법도 검증된다.

| 시각(KST, 10/07) | 변경 | 결과 판 |
| --- | --- | --- |
| 10:27:28 | 파일 최초 작성(Write) | |
| 10:27:39 | 문서 문자열 1줄 수정(9월 행 제외 설명, PowerShell) | **RF 실행판 `8998895a`** |
| 10:27:41~ | RF 8월 재학습 실행. 학습·기록 후 마진 평가 단계에서 `tab.axes` 버그(pandas 속성으로 해석됨)로 `KeyError: False` 발생 | |
| 10:40:14 | `tab.axes` → `tab['axes']` 3곳 수정(PowerShell) | |
| 10:43:20 | 실험 ID 앞에 라이브러리 이름 추가(`{lib}_…`) | **부스팅 실행판 `fce81af8`** |
| 12:27~12:30 | `evaluate_alarm`에 `near`·`counting` 선택 인자와 `episodes_run` 추가 | 현재 작업본 `ede36c5b` |

## 실행 판끼리의 차이와 영향

- **RF판 → 부스팅판**: 실험 ID 이름 규칙, 마진 평가의 열 접근 버그 수정. 모델 학습 부분은 같다.
- **부스팅판 → 현재 작업본**: `evaluate_alarm(..., near=.9, counting='window60')`에 기본값이 붙은 선택 인자만 추가됐다. 스크립트 안의 호출(`evaluate_alarm(part, m, near_scope, days)`)은 기본값을 쓰므로 8월 평가 동작은 같다. 새 인자는 다른 스크립트(마진 변형 비교)에서 사용한다.
- 결론: 현재 작업본으로 실행해도 8월 학습·평가 동작은 실행 당시와 같다. 이는 코드 비교로 판단한 것이며, 재실행으로 예측값을 대조하지는 않았다.
