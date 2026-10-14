"""제출 폴더의 모든 경로를 여기서 정한다. 각 단계 코드는 `import paths as P`로 이 값을 읽는다.

폴더를 옮겨도 이 파일 위치를 기준으로 경로가 정해지므로 고칠 곳이 없다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent            # 제출 폴더 최상위

# 입력 데이터
RAW_ZIP = ROOT / "data" / "원본" / "5. 자원 최적화 AI 데이터셋.zip"    # 주최측 제공 원자료
KMA = ROOT / "data" / "외부" / "ulsan_20210101-20210914.xls"           # 외부 데이터: 기상청 울산(152) 시간자료

# 단계별 결과
PRE_FINAL = ROOT / "1_전처리" / "3_최종데이터"     # 전처리 확정 데이터 (뒤 단계 입력)
DL = ROOT / "2_딥러닝"                              # 딥러닝 최종 코드·가중치·저장 예측
MODEL_INPUT = DL / "data" / "timeseries_15min_v1_previous_week.csv"   # 모델 입력(팀 공통 15분 파일, 트리·딥러닝 공통)
DL_PRED = DL / "예측_내보내기"                      # 마진·경보 단계 입력 (예측_7_8월/, 예측_9월/)
TREE_PRED = ROOT / "3_트리모델" / "저장예측"        # 트리 앙상블 7·8월 저장 예측
PEAK_FINAL = ROOT / "5_피크분석" / "3_최종데이터"   # 피크 분석 확정 표 (방안 단계 입력)
RESULT = ROOT / "결과"                              # 제출 예측결과·요약표
