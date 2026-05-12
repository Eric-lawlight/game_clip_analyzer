"""
speaker_separator.py — 화자 분리 모듈 (pyannote)
==================================================
협방송 시 여러 화자의 발화를 분리하고,
실況자 목소리만 감정 분석 대상으로 필터링한다.

사용 시나리오:
    - 협방송: 실況자 + 게스트 2~4명
    - NPC 음성 / 게임 내 대사 혼입
    - 디스코드 통화 파트너 음성

동작 방식:
    1. pyannote로 전체 오디오 화자 분리 → 화자별 발화 구간 추출
    2. 실況자 화자 ID 결정:
       - 기준 샘플(30초)을 등록한 경우: 유사도 비교로 자동 매핑
       - 미등록: 총 발화 시간이 가장 긴 화자를 실況자로 간주
    3. 실況자 발화가 아닌 세그먼트에 is_host=False 태그
    4. emotion_analyzer, exclamation_detector가 is_host=True 세그먼트만 처리

반환 데이터:
    SpeakerSegment (dataclass)
        .start, .end    : 발화 구간 (초)
        .speaker_id     : "SPEAKER_00" 형식
        .is_host        : 실況자 여부

    SpeakerResult (dataclass)
        .segments               : List[SpeakerSegment]
        .host_id                : 실況자 화자 ID
        .filter_transcript(t)   : 실況자 발화만 포함한 TranscriptResult 반환

사용 예:
    from pipeline.speaker_separator import separate_speakers
    result = separate_speakers(audio_info, progress_cb=print)
    host_transcript = result.filter_transcript(transcript)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from config import PYANNOTE_MODEL, CACHE_DIR
from pipeline.audio_extractor import AudioInfo
from pipeline.whisper_transcriber import TranscriptResult, TranscriptSegment


# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────

# 실況자 판정 기준: 이 비율 이상 발화하면 실況자 후보
_HOST_MIN_SPEAKING_RATIO = 0.30

# 화자 발화가 겹치는 허용 오차 (초)
_OVERLAP_TOLERANCE = 0.1


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class SpeakerSegment:
    """화자 분리 결과 단일 발화 구간."""

    start: float
    end: float
    speaker_id: str     # "SPEAKER_00", "SPEAKER_01" 등
    is_host: bool = False

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, start: float, end: float) -> bool:
        """이 구간이 [start, end]와 겹치는지 확인한다."""
        return self.start < end and self.end > start


@dataclass
class SpeakerResult:
    """전체 화자 분리 결과."""

    segments: list[SpeakerSegment]
    host_id: str            # 실況자 화자 ID
    speaker_ids: list[str]  # 감지된 전체 화자 ID 목록

    @property
    def host_segments(self) -> list[SpeakerSegment]:
        return [s for s in self.segments if s.is_host]

    @property
    def host_speaking_ratio(self) -> float:
        """실況자 발화 비율 (전체 발화 중)."""
        total = sum(s.duration for s in self.segments)
        host = sum(s.duration for s in self.host_segments)
        return host / total if total > 0 else 0.0

    def filter_transcript(
        self,
        transcript: TranscriptResult,
    ) -> TranscriptResult:
        """
        실況자 발화 구간과 겹치는 세그먼트만 남긴 TranscriptResult를 반환한다.
        겹침 기준: 세그먼트의 중심 시각이 host_segment 내에 있으면 포함.
        """
        host_segs = self.host_segments
        filtered: list[TranscriptSegment] = []

        for seg in transcript.segments:
            center = (seg.start + seg.end) / 2
            is_host_speech = any(
                hs.start - _OVERLAP_TOLERANCE <= center <= hs.end + _OVERLAP_TOLERANCE
                for hs in host_segs
            )
            if is_host_speech:
                filtered.append(seg)

        return TranscriptResult(
            segments=filtered,
            language=transcript.language,
            total_duration=transcript.total_duration,
        )

    def speaking_stats(self) -> dict[str, float]:
        """화자별 총 발화 시간(초) 딕셔너리를 반환한다."""
        stats: dict[str, float] = {}
        for seg in self.segments:
            stats[seg.speaker_id] = stats.get(seg.speaker_id, 0.0) + seg.duration
        return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

def separate_speakers(
    audio_info: AudioInfo,
    host_sample_path: Optional[Path] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    force: bool = False,
) -> SpeakerResult:
    """
    오디오에서 화자를 분리하고 실況자를 식별한다.

    Args:
        audio_info      : extract_audio() 반환값
        host_sample_path: 실況자 목소리 샘플 WAV (없으면 자동 감지)
        progress_cb     : 진행 상황 콜백
        force           : 캐시 무시 여부

    Returns:
        SpeakerResult
    """
    log = progress_cb or _noop
    cache_path = _cache_path(audio_info.path)

    if cache_path.exists() and not force:
        log("  ♻️  화자 분리 캐시 사용")
        return _load_from_cache(cache_path)

    log("  🎙️  pyannote 모델 로딩 중...")
    pipeline = _load_pyannote_pipeline()
    log("  ✅ 모델 로딩 완료")

    log("  🔄 화자 분리 실행 중... (시간이 걸립니다)")
    raw_segments = _run_diarization(pipeline, audio_info.path)
    log(f"  ✅ 화자 분리 완료: {len(set(s['speaker'] for s in raw_segments))}명 감지")

    # 실況자 화자 ID 결정
    host_id = _identify_host(
        raw_segments=raw_segments,
        pipeline=pipeline,
        host_sample_path=host_sample_path,
        audio_path=audio_info.path,
        total_duration=audio_info.duration_sec,
    )
    log(f"  👤 실況자 화자 ID: {host_id}")

    # SpeakerSegment 변환
    speaker_ids = sorted(set(s["speaker"] for s in raw_segments))
    segments = [
        SpeakerSegment(
            start=s["start"],
            end=s["end"],
            speaker_id=s["speaker"],
            is_host=(s["speaker"] == host_id),
        )
        for s in raw_segments
    ]

    result = SpeakerResult(
        segments=segments,
        host_id=host_id,
        speaker_ids=speaker_ids,
    )

    stats = result.speaking_stats()
    for spk_id, duration in stats.items():
        marker = " ← 실況자" if spk_id == host_id else ""
        log(f"  　{spk_id}: {duration:.1f}초{marker}")

    _save_to_cache(cache_path, result)
    return result


# ─────────────────────────────────────────────
# pyannote 래퍼
# ─────────────────────────────────────────────

def _load_pyannote_pipeline():
    """pyannote 화자 분리 파이프라인을 로딩한다."""
    try:
        from pyannote.audio import Pipeline
        import torch

        # HuggingFace 토큰 필요 (pyannote는 라이선스 동의 필요)
        # 환경변수 HUGGINGFACE_TOKEN 또는 ~/.cache/huggingface/token 에서 읽음
        pipeline = Pipeline.from_pretrained(
            PYANNOTE_MODEL,
            use_auth_token=True,
        )
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
        return pipeline

    except ImportError:
        raise RuntimeError(
            "pyannote.audio가 설치되지 않았습니다.\n"
            "pip install pyannote.audio\n"
            "또한 HuggingFace에서 pyannote/speaker-diarization-3.1 라이선스 동의 필요:\n"
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )


def _run_diarization(pipeline, audio_path: Path) -> list[dict]:
    """
    pyannote로 화자 분리를 실행하고 세그먼트 리스트를 반환한다.

    Returns:
        [{"start": float, "end": float, "speaker": str}, ...]
    """
    diarization = pipeline(str(audio_path))
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })
    return segments


def _identify_host(
    raw_segments: list[dict],
    pipeline,
    host_sample_path: Optional[Path],
    audio_path: Path,
    total_duration: float,
) -> str:
    """
    실況자 화자 ID를 결정한다.

    샘플이 있으면 음성 유사도 비교, 없으면 최다 발화 화자로 간주.
    """
    if host_sample_path and host_sample_path.exists():
        return _identify_host_by_sample(
            pipeline=pipeline,
            raw_segments=raw_segments,
            sample_path=host_sample_path,
            audio_path=audio_path,
        )

    # fallback: 최다 발화 시간 화자
    return _identify_host_by_duration(raw_segments)


def _identify_host_by_duration(raw_segments: list[dict]) -> str:
    """발화 시간이 가장 긴 화자를 실況자로 반환한다."""
    durations: dict[str, float] = {}
    for seg in raw_segments:
        spk = seg["speaker"]
        durations[spk] = durations.get(spk, 0.0) + (seg["end"] - seg["start"])
    return max(durations, key=lambda k: durations[k])


def _identify_host_by_sample(
    pipeline,
    raw_segments: list[dict],
    sample_path: Path,
    audio_path: Path,
) -> str:
    """
    샘플 오디오와의 화자 임베딩 유사도로 실況자를 식별한다.

    pyannote의 SpeakerEmbedding 모델 사용.
    유사도: 코사인 유사도 (높을수록 같은 화자)
    """
    try:
        from pyannote.audio import Model, Inference
        from pyannote.core import Segment
        import torch
        import numpy as np

        # 화자 임베딩 모델
        embedding_model = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=True,
        )
        inference = Inference(embedding_model, window="whole")

        # 샘플 임베딩
        sample_embedding = inference(str(sample_path))

        # 화자별 대표 임베딩 계산
        speakers = set(s["speaker"] for s in raw_segments)
        similarities: dict[str, float] = {}

        for speaker in speakers:
            # 해당 화자의 첫 30초 발화 세그먼트 추출
            spk_segs = [s for s in raw_segments if s["speaker"] == speaker][:3]
            if not spk_segs:
                continue
            # 첫 번째 세그먼트 임베딩
            seg = spk_segs[0]
            spk_embedding = inference({
                "uri": str(audio_path),
                "audio": str(audio_path),
                "channel": 0,
            }, Segment(seg["start"], seg["end"]))

            # 코사인 유사도
            cos_sim = np.dot(sample_embedding, spk_embedding) / (
                np.linalg.norm(sample_embedding) * np.linalg.norm(spk_embedding) + 1e-8
            )
            similarities[speaker] = float(cos_sim)

        return max(similarities, key=lambda k: similarities[k])

    except Exception:
        # 유사도 방식 실패 시 최다 발화로 fallback
        return _identify_host_by_duration(raw_segments)


# ─────────────────────────────────────────────
# 캐시 관리
# ─────────────────────────────────────────────

def _cache_path(audio_path: Path) -> Path:
    return CACHE_DIR / f"{audio_path.stem}_speakers.json"


def _save_to_cache(cache_path: Path, result: SpeakerResult) -> None:
    data = {
        "host_id": result.host_id,
        "speaker_ids": result.speaker_ids,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "speaker_id": s.speaker_id,
                "is_host": s.is_host,
            }
            for s in result.segments
        ],
    }
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _load_from_cache(cache_path: Path) -> SpeakerResult:
    data = json.loads(cache_path.read_text())
    segments = [SpeakerSegment(**s) for s in data["segments"]]
    return SpeakerResult(
        segments=segments,
        host_id=data["host_id"],
        speaker_ids=data["speaker_ids"],
    )


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────

def _noop(_: str) -> None:
    pass


# ─────────────────────────────────────────────
# 단독 실행 테스트 (더미 데이터)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from pipeline.whisper_transcriber import TranscriptSegment, TranscriptResult

    # 더미 화자 분리 결과
    dummy_segments = [
        SpeakerSegment(start=0.0,  end=5.0,  speaker_id="SPEAKER_00", is_host=True),
        SpeakerSegment(start=5.0,  end=8.0,  speaker_id="SPEAKER_01", is_host=False),
        SpeakerSegment(start=8.0,  end=20.0, speaker_id="SPEAKER_00", is_host=True),
        SpeakerSegment(start=20.0, end=25.0, speaker_id="SPEAKER_01", is_host=False),
        SpeakerSegment(start=25.0, end=60.0, speaker_id="SPEAKER_00", is_host=True),
    ]
    result = SpeakerResult(
        segments=dummy_segments,
        host_id="SPEAKER_00",
        speaker_ids=["SPEAKER_00", "SPEAKER_01"],
    )

    print("\n🎙️  화자 분리 결과 (더미)")
    print("─" * 55)
    print(f"  실況자 ID: {result.host_id}")
    print(f"  발화 비율: {result.host_speaking_ratio:.1%}")
    print(f"\n  화자별 발화 시간:")
    for spk, dur in result.speaking_stats().items():
        marker = " ← 실況자" if spk == result.host_id else ""
        print(f"    {spk}: {dur:.1f}초{marker}")

    # 필터링 테스트
    dummy_transcript = TranscriptResult(
        segments=[
            TranscriptSegment(start=2.0,  end=4.0,  text="야 대박이다"),        # host
            TranscriptSegment(start=6.0,  end=7.5,  text="저도 봤어요"),        # guest
            TranscriptSegment(start=10.0, end=18.0, text="지금 보스 잡았어"),   # host
            TranscriptSegment(start=22.0, end=24.0, text="진짜요? 대단해요"),   # guest
            TranscriptSegment(start=30.0, end=35.0, text="ㅋㅋㅋ 미쳤다"),     # host
        ],
        language="ko",
        total_duration=60.0,
    )

    filtered = result.filter_transcript(dummy_transcript)
    print(f"\n  전체 세그먼트: {len(dummy_transcript.segments)}개")
    print(f"  실況자 발화만: {len(filtered.segments)}개")
    print(f"\n  필터링 결과:")
    for seg in filtered.segments:
        print(f"    [{seg.start:.1f}s] \"{seg.text}\"")
