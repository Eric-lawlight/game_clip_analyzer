"""
config.py — 전역 설정 파일
============================
모든 모듈이 이 파일을 단일 진실의 원천(Single Source of Truth)으로 사용한다.
상수를 변경할 때 이 파일만 수정하면 전체에 반영된다.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent
CACHE_DIR = ROOT_DIR / ".cache"          # 체크포인트 JSON 저장 위치
OUTPUT_DIR = ROOT_DIR / "output_files"   # XML 출력 위치
SAMPLE_DIR = ROOT_DIR / "tests" / "sample_clips"

# 디렉터리 자동 생성 (없으면 만든다)
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 모델 이름
# ─────────────────────────────────────────────

WHISPER_MODEL = "large-v3"
KOELECTRA_MODEL = "snunlp/KR-FinBert-SC"   # 감정 분류용, 추후 교체 가능
QWEN_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"


# ─────────────────────────────────────────────
# 처리 속도 상수 (time_estimator.py 전용)
# ── 단위: 초/분(영상) → 즉, 영상 1분당 처리에 걸리는 초
# ── 기준: RTX 3060 12GB
# ─────────────────────────────────────────────

TIME_CONSTANTS = {
    "ffmpeg_extract":   0.9,    # 오디오 추출: 영상 1분당 ~0.9초
    "whisper":          8.0,    # Whisper large-v3: 영상 1분당 ~8초
    "energy_detector":  2.5,    # librosa 분석: 영상 1분당 ~2.5초
    "emotion_analyzer": 5.5,    # KoELECTRA: 영상 1분당 ~5.5초
    "visual_analyzer":  10.0,   # Qwen2-VL (후보 구간만): 영상 1분당 ~10초
    "speaker_separator":9.0,    # pyannote: 영상 1분당 ~9초
    "xml_export":       0.5,    # XML 생성: 고정값(분 무관)
}

# 기본 활성화 모듈 (사용자가 UI에서 변경 가능)
DEFAULT_MODULES = {
    "ffmpeg_extract":   True,   # 필수 (변경 불가)
    "whisper":          True,   # 필수 (변경 불가)
    "energy_detector":  True,   # 권장
    "emotion_analyzer": True,   # 권장
    "visual_analyzer":  True,   # 권장
    "speaker_separator":False,  # 선택 (협방송 시 ON)
    "xml_export":       True,   # 필수 (변경 불가)
}

# 변경 불가 모듈 (UI에서 토글 비활성화)
FIXED_MODULES = {"ffmpeg_extract", "whisper", "xml_export"}


# ─────────────────────────────────────────────
# 앙상블 가중치 (scorer.py 전용)
# ─────────────────────────────────────────────

SCORE_WEIGHTS = {
    "audio":   0.35,
    "emotion": 0.30,
    "hook":    0.20,   # 구간 첫 3초 임팩트
    "visual":  0.15,   # Qwen 시각 보정 (비활성 시 audio로 재배분)
}

# 라벨 분류 임계값
LABEL_THRESHOLDS = {
    "green":  {"emotion": 0.8, "audio_energy": 0.75},
    "yellow": {"emotion": 0.6},
    "red":    {"silence_sec": 15, "energy": 0.2},
}


# ─────────────────────────────────────────────
# 오디오 / 영상 처리 설정
# ─────────────────────────────────────────────

AUDIO_SAMPLE_RATE = 16_000      # Whisper 요구 샘플레이트 (Hz)
AUDIO_CHANNELS = 1              # mono
CHUNK_DURATION_MIN = 30         # 청크 단위 (분) — Whisper OOM 방지
FRAME_SAMPLE_INTERVAL = 2       # 시각 분석: N초당 1프레임
VISUAL_CANDIDATE_PADDING = 5    # 후보 구간 앞뒤 ±N초 확장
VISUAL_DOWNSCALE_HEIGHT = 480   # Qwen 입력 전 다운샘플 해상도


# ─────────────────────────────────────────────
# 한국어 감탄사 패턴 (exclamation_detector.py 전용)
# ─────────────────────────────────────────────

EXCLAMATION_PATTERNS = {
    "excitement": [
        r'야+\s*[!！]', r'와+[!！]', r'오+[!！]', r'대박',
        r'클리어', r'잡았', r'성공', r'이겼', r'크리티컬',
    ],
    "failure": [
        r'아+[!！씨]', r'왜[!！]?', r'어떻게', r'죽었', r'망했',
        r'ㅅㅂ', r'진짜[!！]?',
    ],
    "laughter": [
        r'ㅋ{5,}', r'하하+', r'헐', r'미쳤',
    ],
    "cut_signal": [   # 제거 대상 신호
        r'로딩', r'잠깐만', r'화장실', r'잠시만요',
    ],
}


# ─────────────────────────────────────────────
# ffprobe 설정
# ─────────────────────────────────────────────

FFPROBE_TIMEOUT = 30            # ffprobe 응답 대기 최대 시간 (초)
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".ts"
}
UNSUPPORTED_CODECS = {
    # 이 코덱은 ffmpeg 재인코딩 없이 처리 불가 → 사용자에게 안내
    "hevc_qsv", "av1",
}
