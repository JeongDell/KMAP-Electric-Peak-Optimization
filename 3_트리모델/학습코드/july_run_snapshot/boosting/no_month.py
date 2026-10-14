"""월(month) 입력 제거 시나리오. 기존 피처 코드는 수정하지 않고 실행 시점에 열만 걸러낸다.

근거: 시간순 확장 분할(7월 검증=1~6월 학습, 8월=1~7월, 9월=1~8월)에서 평가 월의 month 값은
학습에 한 번도 나오지 않아 트리 모델이 외삽하지 못한다. 다른 달력 변수(요일·분기·야간·근무일)는 유지한다.
제거 대상 열: RF·부스팅 단기 `h<k>_month`, 부스팅 pooled `target_month`. 정확히 `month`로 끝나는 열만.
"""
import hashlib
import re
from pathlib import Path

MONTH_PATTERN = re.compile(r'(^|_)month$')
SCENARIO = 'no_month_v1'


def is_month(column):
    return bool(MONTH_PATTERN.search(str(column)))


def drop_month(frame):
    columns = [c for c in frame.columns if is_month(c)]
    if not columns:
        raise ValueError('month 열이 없음: 제거 대상 규칙 또는 입력 구조 확인 필요')
    out = frame.drop(columns=columns)
    if any(is_month(c) for c in out.columns):
        raise ValueError('month 열 잔존')
    return out


def metadata():
    path = Path(__file__)
    return dict(feature_scenario=SCENARIO, excluded_feature_pattern=MONTH_PATTERN.pattern,
                excluded_feature_reason='시간순 분할에서 평가 월 값이 학습에 없어 외삽 불가',
                feature_filter_code=str(path), feature_filter_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def patch_rf():
    """RF 피처 비교·그리드·질의 피처가 공통으로 거치는 _feature_frames 결과에서 month 열을 제거한다.
    피처는 부모 프로세스에서 만든 뒤 작업자에 전달되므로 부모에서 한 번 적용하면 된다."""
    import rf_feature_sequence as fs
    if getattr(fs, '_no_month_patched', False):
        return fs
    original = fs._feature_frames

    def frames_without_month(*args, **kwargs):
        frames, masks = original(*args, **kwargs)
        return {key: drop_month(x) for key, x in frames.items()}, masks

    fs._feature_frames = frames_without_month
    fs._no_month_patched = True
    return fs


def patch_boosting():
    """부스팅 학습·저장·9월 평가가 모두 거치는 Dataset.inputs에서 month 열을 제거한다."""
    import full_c_data as fd
    if getattr(fd.Dataset, '_no_month_patched', False):
        return fd
    original = fd.Dataset.inputs

    def inputs_without_month(self, ids):
        return drop_month(original(self, ids))

    fd.Dataset.inputs = inputs_without_month
    fd.Dataset._no_month_patched = True
    return fd
