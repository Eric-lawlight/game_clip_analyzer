"""
audio_extractor.py — 오디오 추출 모듈
========================================
ffmpeg를 사용해 영상에서 오디오 스트림을 추출한다.
Whisper와 librosa가 요구하는 형식(16kHz mono WAV)으로 출력한다.

반환 데이터:
    AudioInfo (dataclass)
        .path           : 추출된 오디오 파일 경로
        .duration_sec   : 오디오 길이 (초)
        .sample_rate    : 샘플레이트 (기본 16000)
        .channels       : 채널 수 (기본 1 = mono)
        .file_size_mb   : 파일 크기 (MB)

사용 예:
    from pipeline.audio_extractor import extract_audio
    from core.video_inspector import inspect_video

    info = inspect_video("game_vod.mp4")
    audio = extract_audio(info)
    print(audio.path)   # .cache/game_vod_audio.wav
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

from config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    CACHE_DIR,
)
from core.video_inspector import VideoInfo


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class AudioInfo:
    """추출된 오디오 파일 메타데이터."""

    path: Path
    duration_sec: float
    sample_rate: int
    channels: int
    file_size_mb: float

    @property
    def duration_min(self) -> float:
        return self.duration_sec / 60


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

def extract_audio(
    video_info: VideoInfo,
    output_path: Path | None = None,
    force: bool = False,
) -> AudioInfo:
    """
    영상에서 오디오를 추출한다.

    Args:
        video_info  : inspect_video() 반환값
        output_path : 출력 경로. None이면 .cache/{stem}_audio.wav 자동 지정
        force       : True면 캐시가 있어도 재추출

    Returns:
        AudioInfo 인스턴스

    Raises:
        ValueError: 영상에 오디오 스트림이 없을 때
        RuntimeError: ffmpeg 실행 실패 시
    """
    if not video_info.has_audio:
        raise ValueError(
            f"오디오 스트림이 없습니다: {video_info.path.name}\n"
            "음성 분석 모듈을 비활성화하고 시각 분석만 진행하세요."
        )

    out_path = output_path or _default_output_path(video_info.path)

    # 캐시 히트: 이미 추출된 파일이 있으면 재사용
    if out_path.exists() and not force:
        print(f"  ♻️  캐시 사용: {out_path.name}")
        return _make_audio_info(out_path, video_info.duration_sec)

    print(f"  🎵 오디오 추출 중: {video_info.path.name} → {out_path.name}")
    _run_ffmpeg(video_info.path, out_path)
    print(f"  ✅ 오디오 추출 완료: {out_path.name}")

    return _make_audio_info(out_path, video_info.duration_sec)


# ─────────────────────────────────────────────
# 내부 함수
# ─────────────────────────────────────────────

def _default_output_path(video_path: Path) -> Path:
    """캐시 디렉터리 아래 기본 출력 경로를 반환한다."""
    return CACHE_DIR / f"{video_path.stem}_audio.wav"


def _run_ffmpeg(video_path: Path, out_path: Path) -> None:
    """
    ffmpeg를 실행해 오디오를 추출한다.

    옵션 설명:
        -vn          : 영상 스트림 무시
        -acodec pcm_s16le : WAV 16-bit PCM (Whisper / librosa 호환)
        -ar {rate}   : 샘플레이트 16000Hz
        -ac {ch}     : 채널 1 (mono)
    """
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", str(AUDIO_CHANNELS),
        "-y",           # 이미 존재하면 덮어쓰기
        str(out_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg를 찾을 수 없습니다.\n"
            "설치: https://ffmpeg.org/download.html"
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 실행 실패 (returncode={result.returncode})\n"
            f"stderr: {result.stderr[-500:]}"  # 마지막 500자만
        )


def _make_audio_info(path: Path, duration_sec: float) -> AudioInfo:
    """AudioInfo를 생성한다."""
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    return AudioInfo(
        path=path,
        duration_sec=duration_sec,
        sample_rate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        file_size_mb=size_mb,
    )


# ─────────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from core.video_inspector import inspect_video

    if len(sys.argv) < 2:
        print("사용법: python -m pipeline.audio_extractor <영상파일경로>")
        sys.exit(1)

    try:
        info = inspect_video(sys.argv[1])
        audio = extract_audio(info, force=True)

        print(f"\n🎵 추출 결과")
        print(f"  경로       : {audio.path}")
        print(f"  길이       : {audio.duration_sec:.1f}초")
        print(f"  샘플레이트 : {audio.sample_rate} Hz")
        print(f"  채널       : {audio.channels} (mono)")
        print(f"  크기       : {audio.file_size_mb:.1f} MB")

    except Exception as e:
        print(f"\n❌ 오류: {e}")
        sys.exit(1)
