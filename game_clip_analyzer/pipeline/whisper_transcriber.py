"""
whisper_transcriber.py — 음성 전사 모듈
=========================================
Whisper large-v3으로 오디오를 전사하고 타임코드 세그먼트를 반환한다.
CUDA OOM 방지를 위해 30분 단위 청크로 분할 처리한다.
청크 완료 시마다 JSON 캐시에 저장해 중단 후 재개를 지원한다.

반환 데이터:
    TranscriptSegment (dataclass) — 발화 1개 단위
        .start      : 시작 시간 (초)
        .end        : 종료 시간 (초)
        .text       : 전사 텍스트
        .language   : 감지된 언어 코드 ("ko", "en" 등)
        .confidence : 평균 토큰 확률 (0~1, 낮을수록 불확실)

    TranscriptResult (dataclass)
        .segments       : List[TranscriptSegment]
        .language       : 전체 주 언어
        .total_duration : 전체 길이 (초)

사용 예:
    from pipeline.whisper_transcriber import transcribe
    from pipeline.audio_extractor import extract_audio
    from core.video_inspector import inspect_video

    info = inspect_video("game_vod.mp4")
    audio = extract_audio(info)
    result = transcribe(audio, progress_cb=print)

    for seg in result.segments:
        print(f"[{seg.start:.1f}s] {seg.text}")
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from config import (
    WHISPER_MODEL,
    CHUNK_DURATION_MIN,
    CACHE_DIR,
)
from pipeline.audio_extractor import AudioInfo


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class TranscriptSegment:
    """Whisper 세그먼트 1개 (단어 단위 아닌 구절 단위)."""

    start: float            # 초
    end: float              # 초
    text: str
    language: str = "ko"
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def text_clean(self) -> str:
        """앞뒤 공백 제거 + 특수문자 정리."""
        return self.text.strip()


@dataclass
class TranscriptResult:
    """전체 전사 결과."""

    segments: list[TranscriptSegment]
    language: str
    total_duration: float

    # 편의 프로퍼티
    @property
    def full_text(self) -> str:
        """전체 전사 텍스트 (공백 구분)."""
        return " ".join(s.text_clean for s in self.segments)

    @property
    def segment_count(self) -> int:
        return len(self.segments)


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

def transcribe(
    audio_info: AudioInfo,
    progress_cb: Optional[Callable[[str], None]] = None,
    force: bool = False,
) -> TranscriptResult:
    """
    오디오를 Whisper로 전사한다.

    Args:
        audio_info  : extract_audio() 반환값
        progress_cb : 진행 상황 콜백 함수. 문자열 메시지를 전달받는다.
        force       : True면 캐시가 있어도 재전사

    Returns:
        TranscriptResult

    Raises:
        RuntimeError: Whisper 로딩 또는 전사 실패 시
    """
    log = progress_cb or _noop

    cache_path = _cache_path(audio_info.path)

    # 전체 캐시 히트
    if cache_path.exists() and not force:
        log(f"  ♻️  전사 캐시 사용: {cache_path.name}")
        return _load_from_cache(cache_path)

    log("  🤖 Whisper 모델 로딩 중...")
    model = _load_whisper_model()
    log(f"  ✅ 모델 로딩 완료 ({WHISPER_MODEL})")

    # 청크 분할
    chunks = _split_into_chunks(audio_info)
    total_chunks = len(chunks)
    all_segments: list[TranscriptSegment] = []
    detected_language = "ko"

    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        chunk_cache = _chunk_cache_path(audio_info.path, i)

        if chunk_cache.exists() and not force:
            log(f"  ♻️  청크 {i}/{total_chunks} 캐시 사용")
            chunk_segs = _load_chunk_cache(chunk_cache)
        else:
            log(f"  🔄 청크 {i}/{total_chunks} 전사 중... "
                f"({_fmt_time(chunk_start)} ~ {_fmt_time(chunk_end)})")
            chunk_segs, detected_language = _transcribe_chunk(
                model=model,
                audio_path=audio_info.path,
                start_sec=chunk_start,
                end_sec=chunk_end,
                time_offset=chunk_start,
            )
            _save_chunk_cache(chunk_cache, chunk_segs)
            log(f"  ✅ 청크 {i}/{total_chunks} 완료 "
                f"({len(chunk_segs)}개 세그먼트)")

        all_segments.extend(chunk_segs)

    result = TranscriptResult(
        segments=all_segments,
        language=detected_language,
        total_duration=audio_info.duration_sec,
    )
    _save_to_cache(cache_path, result)
    log(f"  ✅ 전사 완료: 총 {result.segment_count}개 세그먼트")

    return result


# ─────────────────────────────────────────────
# 청크 분할
# ─────────────────────────────────────────────

def _split_into_chunks(audio_info: AudioInfo) -> list[tuple[float, float]]:
    """
    오디오를 CHUNK_DURATION_MIN 단위로 분할한다.

    Returns:
        [(start_sec, end_sec), ...] 리스트
    """
    chunk_sec = CHUNK_DURATION_MIN * 60
    total = audio_info.duration_sec
    n_chunks = math.ceil(total / chunk_sec)

    chunks = []
    for i in range(n_chunks):
        start = i * chunk_sec
        end = min(start + chunk_sec, total)
        chunks.append((start, end))

    return chunks


# ─────────────────────────────────────────────
# Whisper 래퍼
# ─────────────────────────────────────────────

def _load_whisper_model():
    """Whisper 모델을 GPU에 로딩한다."""
    try:
        import whisper
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("  ⚠️  GPU를 찾을 수 없습니다. CPU로 실행합니다 (매우 느림)")

        model = whisper.load_model(WHISPER_MODEL, device=device)
        return model

    except ImportError:
        raise RuntimeError(
            "openai-whisper가 설치되지 않았습니다.\n"
            "pip install openai-whisper"
        )


def _transcribe_chunk(
    model,
    audio_path: Path,
    start_sec: float,
    end_sec: float,
    time_offset: float,
) -> tuple[list[TranscriptSegment], str]:
    """
    오디오의 특정 구간을 Whisper로 전사한다.

    Args:
        model       : 로딩된 Whisper 모델
        audio_path  : 오디오 WAV 파일 경로
        start_sec   : 청크 시작 (초)
        end_sec     : 청크 종료 (초)
        time_offset : 세그먼트 타임코드에 더할 오프셋 (초)

    Returns:
        (세그먼트 리스트, 감지된 언어)
    """
    import whisper

    # 청크 구간만 로딩 (전체 오디오를 메모리에 올리지 않는다)
    audio = whisper.load_audio(str(audio_path))
    sr = 16_000
    start_frame = int(start_sec * sr)
    end_frame = int(end_sec * sr)
    audio_chunk = audio[start_frame:end_frame]

    result = model.transcribe(
        audio_chunk,
        language=None,          # 자동 감지
        task="transcribe",
        word_timestamps=False,  # 세그먼트 단위면 충분
        fp16=True,              # GPU 메모리 절약
        verbose=False,
    )

    language = result.get("language", "ko")
    segments = []

    for seg in result.get("segments", []):
        # 신뢰도: avg_logprob → 확률로 변환 (대략적)
        avg_logprob = seg.get("avg_logprob", 0.0)
        confidence = min(1.0, max(0.0, (avg_logprob + 1.0)))

        segments.append(TranscriptSegment(
            start=seg["start"] + time_offset,
            end=seg["end"] + time_offset,
            text=seg["text"],
            language=language,
            confidence=confidence,
        ))

    return segments, language


# ─────────────────────────────────────────────
# 캐시 관리
# ─────────────────────────────────────────────

def _cache_path(audio_path: Path) -> Path:
    return CACHE_DIR / f"{audio_path.stem}_transcript.json"


def _chunk_cache_path(audio_path: Path, chunk_idx: int) -> Path:
    return CACHE_DIR / f"{audio_path.stem}_chunk_{chunk_idx:03d}.json"


def _save_to_cache(cache_path: Path, result: TranscriptResult) -> None:
    data = {
        "language": result.language,
        "total_duration": result.total_duration,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "language": s.language,
                "confidence": s.confidence,
            }
            for s in result.segments
        ],
    }
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _load_from_cache(cache_path: Path) -> TranscriptResult:
    data = json.loads(cache_path.read_text())
    segments = [
        TranscriptSegment(**seg) for seg in data["segments"]
    ]
    return TranscriptResult(
        segments=segments,
        language=data["language"],
        total_duration=data["total_duration"],
    )


def _save_chunk_cache(cache_path: Path, segments: list[TranscriptSegment]) -> None:
    data = [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "language": s.language,
            "confidence": s.confidence,
        }
        for s in segments
    ]
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _load_chunk_cache(cache_path: Path) -> list[TranscriptSegment]:
    data = json.loads(cache_path.read_text())
    return [TranscriptSegment(**seg) for seg in data]


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """초 → "MM:SS" 형식."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _noop(msg: str) -> None:
    pass


# ─────────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from core.video_inspector import inspect_video
    from pipeline.audio_extractor import extract_audio

    if len(sys.argv) < 2:
        print("사용법: python -m pipeline.whisper_transcriber <영상파일경로>")
        sys.exit(1)

    try:
        info = inspect_video(sys.argv[1])
        audio = extract_audio(info)
        result = transcribe(audio, progress_cb=print)

        print(f"\n📝 전사 결과 (첫 10개 세그먼트)")
        print("─" * 60)
        for seg in result.segments[:10]:
            print(f"  [{_fmt_time(seg.start)} ~ {_fmt_time(seg.end)}]  {seg.text_clean}")
        print(f"\n  총 {result.segment_count}개 세그먼트 / 언어: {result.language}")

    except Exception as e:
        print(f"\n❌ 오류: {e}")
        sys.exit(1)
