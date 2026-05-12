"""
time_estimator.py — 처리 시간 예측 모듈
=========================================
VideoInfo와 활성화된 모듈 목록을 받아,
모듈별·합산 예상 처리 시간을 계산한다.

반환 데이터:
    TimeEstimate (dataclass)
        .per_module     : {모듈명: 예상 초} 딕셔너리
        .total_sec      : 전체 합산 예상 시간 (초)
        .total_str      : 사람이 읽기 좋은 형식 ("약 52분")
        .summary_lines  : UI 표시용 줄별 문자열 리스트

사용 예:
    from core.video_inspector import inspect_video
    from core.time_estimator import estimate_time
    from config import DEFAULT_MODULES

    info = inspect_video("game_vod.mp4")
    est = estimate_time(info, DEFAULT_MODULES)
    print(est.total_str)           # "약 52분"
    for line in est.summary_lines:
        print(line)
"""

from dataclasses import dataclass, field

from config import TIME_CONSTANTS, FIXED_MODULES
from core.video_inspector import VideoInfo


# ─────────────────────────────────────────────
# 모듈 표시 이름 (UI용 한글 레이블)
# ─────────────────────────────────────────────

MODULE_LABELS = {
    "ffmpeg_extract":   "오디오 추출 (ffmpeg)",
    "whisper":          "음성 전사 (Whisper)",
    "energy_detector":  "에너지 분석 (librosa)",
    "emotion_analyzer": "감정 분류 (KoELECTRA)",
    "visual_analyzer":  "시각 보정 (Qwen2-VL)",
    "speaker_separator":"화자 분리 (pyannote)",
    "xml_export":       "XML 생성",
}

# 필수 모듈은 항상 ON이므로 UI에서 따로 표시
REQUIRED_LABEL = "필수"
OPTIONAL_LABEL = "선택"


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class TimeEstimate:
    """모듈별 + 전체 예상 처리 시간 컨테이너."""

    per_module: dict[str, float]        # {모듈명: 예상 초}
    total_sec: float

    # 자동 계산
    total_str: str = field(init=False)
    summary_lines: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.total_str = _format_duration(self.total_sec)
        self.summary_lines = _build_summary_lines(self.per_module)

    def module_str(self, module: str) -> str:
        """단일 모듈의 예상 시간 문자열 반환."""
        sec = self.per_module.get(module, 0)
        return _format_duration(sec)


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

def estimate_time(
    video_info: VideoInfo,
    active_modules: dict[str, bool],
) -> TimeEstimate:
    """
    영상 정보와 활성 모듈 목록으로 처리 시간을 추정한다.

    Args:
        video_info  : inspect_video() 반환값
        active_modules: {모듈명: bool} — True면 활성화

    Returns:
        TimeEstimate 인스턴스

    Notes:
        - FIXED_MODULES(필수 모듈)는 active_modules 값에 관계없이 항상 포함된다.
        - visual_analyzer는 전체 길이가 아닌 후보 구간만 처리하므로
          실제 시간은 더 짧을 수 있다. 계산 시 50% 보정을 적용한다.
    """
    duration_min = video_info.duration_min
    per_module: dict[str, float] = {}

    for module, constant in TIME_CONSTANTS.items():
        is_active = module in FIXED_MODULES or active_modules.get(module, False)
        if not is_active:
            continue

        if module == "xml_export":
            # XML 생성은 영상 길이와 무관한 고정 비용
            estimated_sec = constant
        elif module == "visual_analyzer":
            # 후보 구간(전체의 약 30~50%)만 처리 → 50% 보정
            estimated_sec = constant * duration_min * 0.5
        else:
            estimated_sec = constant * duration_min

        per_module[module] = round(estimated_sec)

    total_sec = sum(per_module.values())
    return TimeEstimate(per_module=per_module, total_sec=total_sec)


# ─────────────────────────────────────────────
# 내부 함수
# ─────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    """
    초를 UI 표시용 짧은 문자열로 변환한다.

    예: 3120 → "약 52분"
         45 → "약 45초"
         5400 → "약 1시간 30분"
    """
    total = int(seconds)
    if total < 60:
        return f"약 {total}초"

    hours, remainder = divmod(total, 3600)
    minutes = round(remainder / 60)

    if hours == 0:
        return f"약 {minutes}분"
    if minutes == 0:
        return f"약 {hours}시간"
    return f"약 {hours}시간 {minutes}분"


def _build_summary_lines(per_module: dict[str, float]) -> list[str]:
    """
    UI 표시용 줄별 문자열을 생성한다.

    예:
        [✅] 오디오 추출 (ffmpeg)    필수   ~1분
        [✅] 음성 전사 (Whisper)     필수   ~8분
        [✅] 에너지 분석 (librosa)   선택   ~3분
    """
    lines = []
    for module, sec in per_module.items():
        label = MODULE_LABELS.get(module, module)
        tag = REQUIRED_LABEL if module in FIXED_MODULES else OPTIONAL_LABEL
        time_str = _format_duration(sec)
        lines.append(f"[✅] {label:<28} {tag:<4}  {time_str}")
    return lines


# ─────────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from core.video_inspector import inspect_video
    from config import DEFAULT_MODULES

    if len(sys.argv) < 2:
        print("사용법: python -m core.time_estimator <영상파일경로>")
        sys.exit(1)

    try:
        info = inspect_video(sys.argv[1])
        est = estimate_time(info, DEFAULT_MODULES)

        print(f"\n⏱️  예상 처리 시간: {est.total_str}")
        print("─" * 50)
        for line in est.summary_lines:
            print(f"  {line}")

    except Exception as e:
        print(f"\n❌ 오류: {e}")
        sys.exit(1)
