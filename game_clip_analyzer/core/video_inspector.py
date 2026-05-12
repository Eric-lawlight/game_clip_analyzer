"""
video_inspector.py — 영상 사전 분석 모듈
==========================================
ffprobe를 사용해 영상 파일의 메타데이터를 파악한다.
이후 모든 모듈이 이 결과를 기반으로 동작하므로, 파이프라인의 첫 번째 단계다.

반환 데이터:
    VideoInfo (dataclass)
        .path           : 영상 파일 절대 경로
        .duration_sec   : 총 길이 (초, float)
        .duration_str   : 사람이 읽기 좋은 형식 ("2시간 14분 33초")
        .fps            : 프레임레이트 (float)
        .fps_int        : XML 타임베이스용 정수 fps
        .codec          : 영상 코덱 이름 ("h264", "hevc" 등)
        .resolution     : (width, height) 튜플
        .has_audio      : 오디오 스트림 존재 여부
        .file_size_mb   : 파일 크기 (MB)
        .is_supported   : 처리 가능한 코덱인지 여부
        .warnings       : 경고 메시지 리스트

사용 예:
    from core.video_inspector import inspect_video

    info = inspect_video("game_vod.mp4")
    print(info.duration_str)   # "2시간 14분 33초"
    print(info.fps)            # 60.0
    print(info.warnings)       # ["HEVC 코덱: ffmpeg 재인코딩 필요"]
"""

import json
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

from config import (
    FFPROBE_TIMEOUT,
    SUPPORTED_VIDEO_EXTENSIONS,
    UNSUPPORTED_CODECS,
)


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class VideoInfo:
    """ffprobe로 추출한 영상 메타데이터 컨테이너."""

    path: Path
    duration_sec: float
    fps: float
    codec: str
    resolution: tuple[int, int]         # (width, height)
    has_audio: bool
    file_size_mb: float

    # 자동 계산 필드
    duration_str: str = field(init=False)
    fps_int: int = field(init=False)
    is_supported: bool = field(init=False)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.duration_str = _format_duration(self.duration_sec)
        self.fps_int = round(self.fps)
        self.is_supported = self.codec not in UNSUPPORTED_CODECS

        if not self.is_supported:
            self.warnings.append(
                f"⚠️  {self.codec.upper()} 코덱: ffmpeg 재인코딩 후 사용 권장"
            )
        if not self.has_audio:
            self.warnings.append("⚠️  오디오 스트림 없음: 음성 분석 불가")
        if self.duration_sec > 3 * 3600:
            self.warnings.append(
                "ℹ️  3시간 초과 영상: 30분 단위 청크 처리가 자동 적용됩니다"
            )

    @property
    def duration_min(self) -> float:
        """분 단위 길이 (time_estimator에서 사용)."""
        return self.duration_sec / 60

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    def summary(self) -> str:
        """UI 표시용 한 줄 요약."""
        w, h = self.resolution
        return (
            f"길이: {self.duration_str}  /  "
            f"fps: {self.fps_int}  /  "
            f"코덱: {self.codec.upper()}  /  "
            f"해상도: {w}×{h}  /  "
            f"크기: {self.file_size_mb:.1f} MB"
        )


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

def inspect_video(video_path: str | Path) -> VideoInfo:
    """
    영상 파일을 ffprobe로 분석하고 VideoInfo를 반환한다.

    Args:
        video_path: 분석할 영상 파일 경로

    Returns:
        VideoInfo 인스턴스

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        ValueError: 지원하지 않는 확장자이거나 ffprobe 파싱 실패 시
        RuntimeError: ffprobe 실행 실패 시
    """
    path = _validate_path(video_path)
    raw = _run_ffprobe(path)
    return _parse_ffprobe_output(path, raw)


# ─────────────────────────────────────────────
# 내부 함수
# ─────────────────────────────────────────────

def _validate_path(video_path: str | Path) -> Path:
    """경로 유효성 검사."""
    path = Path(video_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"지원하지 않는 확장자: {path.suffix}\n"
            f"지원 형식: {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}"
        )

    return path


def _run_ffprobe(path: Path) -> dict:
    """ffprobe를 실행하고 JSON 결과를 반환한다."""
    cmd = [
        "ffprobe",
        "-v", "quiet",                  # 불필요한 로그 숨김
        "-print_format", "json",        # JSON 출력
        "-show_streams",                # 스트림 정보 포함
        "-show_format",                 # 포맷(전체 길이, 파일크기) 포함
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffprobe를 찾을 수 없습니다.\n"
            "ffmpeg를 설치하세요: https://ffmpeg.org/download.html"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"ffprobe가 {FFPROBE_TIMEOUT}초 내에 응답하지 않았습니다."
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe 실행 실패 (returncode={result.returncode})\n"
            f"stderr: {result.stderr.strip()}"
        )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise ValueError(f"ffprobe 출력 파싱 실패: {e}")


def _parse_ffprobe_output(path: Path, raw: dict) -> VideoInfo:
    """ffprobe JSON을 VideoInfo로 변환한다."""
    streams = raw.get("streams", [])
    fmt = raw.get("format", {})

    # 영상 스트림 추출
    video_stream = _find_stream(streams, "video")
    if video_stream is None:
        raise ValueError(f"영상 스트림을 찾을 수 없습니다: {path.name}")

    # 오디오 스트림 존재 여부
    audio_stream = _find_stream(streams, "audio")
    has_audio = audio_stream is not None

    # 길이: 스트림 > 포맷 순으로 우선 적용
    duration_sec = _extract_duration(video_stream, fmt)

    # FPS 파싱 ("60/1", "30000/1001" 형식 처리)
    fps = _parse_fps(
        video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate", "30/1")
    )

    # 해상도
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    # 코덱
    codec = video_stream.get("codec_name", "unknown").lower()

    # 파일 크기
    file_size_mb = int(fmt.get("size", 0)) / (1024 * 1024)

    return VideoInfo(
        path=path,
        duration_sec=duration_sec,
        fps=fps,
        codec=codec,
        resolution=(width, height),
        has_audio=has_audio,
        file_size_mb=file_size_mb,
    )


def _find_stream(streams: list[dict], codec_type: str) -> Optional[dict]:
    """스트림 목록에서 특정 타입의 첫 번째 스트림을 반환한다."""
    for stream in streams:
        if stream.get("codec_type") == codec_type:
            return stream
    return None


def _extract_duration(video_stream: dict, fmt: dict) -> float:
    """가능한 소스에서 영상 길이(초)를 추출한다."""
    # 1순위: 영상 스트림 duration
    if "duration" in video_stream:
        return float(video_stream["duration"])

    # 2순위: 포맷 duration
    if "duration" in fmt:
        return float(fmt["duration"])

    # 3순위: nb_frames / fps로 추정
    nb_frames = video_stream.get("nb_frames")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if nb_frames:
        fps = _parse_fps(fps_str)
        return int(nb_frames) / fps if fps > 0 else 0.0

    raise ValueError("영상 길이를 파악할 수 없습니다.")


def _parse_fps(fps_str: str) -> float:
    """
    ffprobe fps 문자열을 float으로 변환한다.

    예: "60/1" → 60.0, "30000/1001" → 29.97, "25/1" → 25.0
    """
    try:
        frac = Fraction(fps_str)
        return float(frac)
    except (ValueError, ZeroDivisionError):
        return 30.0  # 파싱 실패 시 기본값


def _format_duration(seconds: float) -> str:
    """
    초를 사람이 읽기 좋은 문자열로 변환한다.

    예: 8073.0 → "2시간 14분 33초"
         90.0 → "1분 30초"
         45.0 → "45초"
    """
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours}시간")
    if minutes:
        parts.append(f"{minutes}분")
    parts.append(f"{secs}초")

    return " ".join(parts)


# ─────────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python video_inspector.py <영상파일경로>")
        sys.exit(1)

    try:
        info = inspect_video(sys.argv[1])
        print("\n📋 영상 분석 결과")
        print("─" * 50)
        print(f"  파일명   : {info.path.name}")
        print(f"  {info.summary()}")
        print(f"  오디오   : {'있음' if info.has_audio else '없음'}")
        print(f"  지원여부 : {'✅ 정상' if info.is_supported else '⚠️  재인코딩 필요'}")

        if info.warnings:
            print("\n⚠️  경고")
            for w in info.warnings:
                print(f"  {w}")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ 오류: {e}")
        sys.exit(1)
