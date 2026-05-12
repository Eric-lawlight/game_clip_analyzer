"""
scorer.py — 앙상블 스코어링 모듈
===================================
각 분석 모듈의 독립 점수를 받아 구간별 최종 점수를 계산한다.

설계 원칙:
    - 각 모듈 점수는 0~1 범위로 독립 유지
    - 비활성 모듈 점수는 제외하고 나머지 가중치를 재배분
    - 하나의 모듈이 틀려도 전체 결과가 붕괴되지 않음 (앙상블)

점수 구성:
    audio_score   = RMS 에너지 피크 × 지속시간 보정
    emotion_score = 감탄사 강도 × 유형 가중치  (KoELECTRA 없을 때 대체)
    hook_score    = 구간 첫 3초 에너지 임팩트
    visual_score  = Qwen 판정 결과 (선택, 없으면 audio로 흡수)

반환 데이터:
    ScoredSegment (dataclass)
        .start, .end        : 구간 타임코드 (초)
        .audio_score        : 0~1
        .emotion_score      : 0~1
        .hook_score         : 0~1
        .visual_score       : 0~1 (없으면 None)
        .final_score        : 가중 합산 (0~1)
        .transcript_text    : 해당 구간 전사 텍스트
        .exclamation_hits   : 감지된 감탄사 리스트

사용 예:
    from scoring.scorer import score_segments
    scored = score_segments(energy_result, exclamation_result, transcript_result)
    for seg in scored:
        print(f"[{seg.start:.1f}s]  최종 점수: {seg.final_score:.2f}")
"""

from dataclasses import dataclass, field
from typing import Optional

from config import SCORE_WEIGHTS, LABEL_THRESHOLDS
from pipeline.energy_detector import EnergyResult, PeakSegment, SilenceSegment
from pipeline.whisper_transcriber import TranscriptResult
from scoring.exclamation_detector import ExclamationResult, ExclamationHit


# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────

# hook_score: 구간 첫 N초
_HOOK_WINDOW_SEC = 3.0

# 구간 최소 길이 (초) — 너무 짧은 구간은 무시
_MIN_SEGMENT_DURATION = 2.0

# 피크 구간 병합: 이 간격 이내면 하나로 합친다 (초)
_MERGE_GAP_SEC = 5.0

# excitement 와 failure 동시 히트 시 excitement 우선 가중치
_CONFLICT_EXCITEMENT_WEIGHT = 0.7
_CONFLICT_FAILURE_WEIGHT = 0.3


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class ScoredSegment:
    """점수가 계산된 구간 단위."""

    start: float
    end: float
    audio_score: float
    emotion_score: float
    hook_score: float
    visual_score: Optional[float]       # None이면 Qwen 비활성
    final_score: float
    transcript_text: str = ""
    exclamation_hits: list[ExclamationHit] = field(default_factory=list)

    # 침묵 구간 플래그 (labeler용)
    is_silence: bool = False
    silence_duration: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def dominant_exclamation(self) -> Optional[str]:
        """가장 높은 점수의 감탄사 유형을 반환한다."""
        if not self.exclamation_hits:
            return None
        pos = [h for h in self.exclamation_hits if h.is_positive]
        if not pos:
            return None
        return max(pos, key=lambda h: h.score).pattern_type

    def __repr__(self) -> str:
        return (
            f"ScoredSegment({self.start:.1f}~{self.end:.1f}s "
            f"final={self.final_score:.2f})"
        )


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

def score_segments(
    energy: EnergyResult,
    exclamation: ExclamationResult,
    transcript: TranscriptResult,
    visual_scores: Optional[dict[tuple[float, float], float]] = None,
    active_modules: Optional[dict[str, bool]] = None,
    emotion_result=None,   # pipeline.emotion_analyzer.EmotionResult | None
) -> list[ScoredSegment]:
    """
    모든 분석 결과를 받아 구간별 최종 점수를 계산한다.

    Args:
        energy          : detect_energy() 반환값
        exclamation     : detect_exclamations() 반환값
        transcript      : transcribe() 반환값
        visual_scores   : {(start, end): score} — Qwen 결과 (없으면 None)
        active_modules  : 활성 모듈 딕셔너리 (가중치 재배분에 사용)

    Returns:
        List[ScoredSegment] — final_score 내림차순 정렬
    """
    modules = active_modules or {}
    weights = _compute_weights(
        has_visual=visual_scores is not None,
        has_emotion="emotion_analyzer" in modules and modules.get("emotion_analyzer"),
    )

    # ── 1. 후보 구간 수집 ──────────────────────
    candidate_intervals = _collect_candidates(energy, exclamation, transcript)

    # ── 2. 구간별 점수 계산 ────────────────────
    scored: list[ScoredSegment] = []
    for start, end in candidate_intervals:
        seg = _score_one(
            start=start,
            end=end,
            energy=energy,
            exclamation=exclamation,
            transcript=transcript,
            visual_scores=visual_scores,
            weights=weights,
            emotion_result=emotion_result,
        )
        scored.append(seg)

    # ── 3. 침묵 구간 추가 (Red 마커용) ────────
    for silence in energy.red_silences:
        scored.append(ScoredSegment(
            start=silence.start,
            end=silence.end,
            audio_score=0.0,
            emotion_score=0.0,
            hook_score=0.0,
            visual_score=None,
            final_score=0.0,
            transcript_text="[침묵 구간]",
            is_silence=True,
            silence_duration=silence.duration,
        ))

    # ── 4. final_score 내림차순 정렬 ──────────
    scored.sort(key=lambda s: s.final_score, reverse=True)
    return scored


# ─────────────────────────────────────────────
# 후보 구간 수집
# ─────────────────────────────────────────────

def _collect_candidates(
    energy: EnergyResult,
    exclamation: ExclamationResult,
    transcript: TranscriptResult,
) -> list[tuple[float, float]]:
    """
    에너지 피크 + 감탄사 발생 구간을 합쳐 중복 제거 후 반환한다.
    인접 구간(gap < _MERGE_GAP_SEC)은 병합한다.
    """
    intervals: list[tuple[float, float]] = []

    # 에너지 피크 구간
    for peak in energy.peak_segments:
        intervals.append((peak.start, peak.end))

    # 감탄사 발생 구간 (세그먼트 타임코드 기준)
    for hit in exclamation.positive_hits:
        seg = hit.segment
        intervals.append((seg.start, seg.end))

    # 정렬 후 병합
    intervals.sort(key=lambda x: x[0])
    merged: list[tuple[float, float]] = []

    for start, end in intervals:
        if merged and (start - merged[-1][1]) <= _MERGE_GAP_SEC:
            # 직전 구간과 병합
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # 최소 길이 필터
    return [(s, e) for s, e in merged if (e - s) >= _MIN_SEGMENT_DURATION]


# ─────────────────────────────────────────────
# 단일 구간 점수 계산
# ─────────────────────────────────────────────

def _score_one(
    start: float,
    end: float,
    energy: EnergyResult,
    exclamation: ExclamationResult,
    transcript: TranscriptResult,
    visual_scores: Optional[dict],
    weights: dict[str, float],
    emotion_result=None,
) -> ScoredSegment:

    # audio_score: 구간 평균 에너지 × 지속시간 보정
    avg_e = energy.avg_energy_in(start, end)
    duration_factor = min(1.0, (end - start) / 10.0)   # 10초 기준 포화
    audio_score = avg_e * (0.7 + 0.3 * duration_factor)

    # emotion_score: KoELECTRA 결과 우선, 없으면 감탄사 패턴으로 대체
    if emotion_result is not None:
        emotion_score = emotion_result.intensity_at(start, end)
        # KoELECTRA와 감탄사 패턴을 7:3으로 혼합
        pattern_score = _compute_emotion_score(exclamation, start, end)
        emotion_score = emotion_score * 0.7 + pattern_score * 0.3
    else:
        emotion_score = _compute_emotion_score(exclamation, start, end)

    # hook_score: 구간 첫 3초 에너지
    hook_end = min(start + _HOOK_WINDOW_SEC, end)
    hook_score = energy.avg_energy_in(start, hook_end)

    # visual_score: Qwen 결과 조회
    visual_score = _lookup_visual(visual_scores, start, end)

    # 전사 텍스트 수집
    texts = [
        s.text_clean for s in transcript.segments
        if start <= s.start <= end and s.text_clean
    ]
    transcript_text = " ".join(texts)

    # 감탄사 히트 수집
    exc_hits = exclamation.hits_in_range(start, end)

    # 최종 점수
    final_score = _weighted_sum(
        audio=audio_score,
        emotion=emotion_score,
        hook=hook_score,
        visual=visual_score,
        weights=weights,
    )

    return ScoredSegment(
        start=start,
        end=end,
        audio_score=round(audio_score, 4),
        emotion_score=round(emotion_score, 4),
        hook_score=round(hook_score, 4),
        visual_score=round(visual_score, 4) if visual_score is not None else None,
        final_score=round(final_score, 4),
        transcript_text=transcript_text,
        exclamation_hits=exc_hits,
    )


def _compute_emotion_score(
    exclamation: ExclamationResult,
    start: float,
    end: float,
) -> float:
    """
    구간 내 감탄사 히트를 바탕으로 emotion_score를 산출한다.
    excitement + failure 동시 히트 시 excitement 우선.
    cut_signal은 감점 처리.
    """
    hits = exclamation.hits_in_range(start, end)
    if not hits:
        return 0.0

    type_scores: dict[str, float] = {}
    for hit in hits:
        t = hit.pattern_type
        type_scores[t] = max(type_scores.get(t, 0.0), hit.score)

    score = 0.0

    # 충돌 해소: excitement + failure 동시 → 가중 평균
    if "excitement" in type_scores and "failure" in type_scores:
        score += (
            type_scores["excitement"] * _CONFLICT_EXCITEMENT_WEIGHT
            + type_scores["failure"] * _CONFLICT_FAILURE_WEIGHT
        )
    elif "excitement" in type_scores:
        score += type_scores["excitement"]
    elif "failure" in type_scores:
        score += type_scores["failure"] * 0.9  # 실패도 강조 후보

    if "laughter" in type_scores:
        score = max(score, type_scores["laughter"] * 0.85)

    if "cut_signal" in type_scores:
        score += type_scores["cut_signal"]  # 음수값이므로 감점

    return max(0.0, min(1.0, score))


def _lookup_visual(
    visual_scores: Optional[dict],
    start: float,
    end: float,
) -> Optional[float]:
    """visual_scores 딕셔너리에서 구간과 겹치는 점수를 찾는다."""
    if visual_scores is None:
        return None

    overlaps = []
    for (vs, ve), score in visual_scores.items():
        # 겹침 조건
        if vs < end and ve > start:
            overlaps.append(score)

    if not overlaps:
        return None

    return sum(overlaps) / len(overlaps)  # 평균


def _weighted_sum(
    audio: float,
    emotion: float,
    hook: float,
    visual: Optional[float],
    weights: dict[str, float],
) -> float:
    """가중합산 최종 점수 계산."""
    total = (
        audio   * weights["audio"]
        + emotion * weights["emotion"]
        + hook    * weights["hook"]
    )
    if visual is not None:
        total += visual * weights["visual"]

    return max(0.0, min(1.0, total))


# ─────────────────────────────────────────────
# 가중치 재배분
# ─────────────────────────────────────────────

def _compute_weights(
    has_visual: bool,
    has_emotion: bool,
) -> dict[str, float]:
    """
    비활성 모듈의 가중치를 audio로 재배분한다.
    """
    w = dict(SCORE_WEIGHTS)  # 복사

    freed = 0.0
    if not has_visual:
        freed += w.pop("visual", 0.0)
        w["visual"] = 0.0
    if not has_emotion:
        freed += w.pop("emotion", 0.0)
        w["emotion"] = 0.0

    if freed > 0:
        w["audio"] = w.get("audio", 0.0) + freed

    # 정규화 (합산이 1이 되도록)
    total = sum(v for v in w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}

    return w


# ─────────────────────────────────────────────
# 단독 실행 테스트 (더미 데이터)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from pipeline.energy_detector import EnergyResult, PeakSegment, SilenceSegment, PitchSpike
    from pipeline.whisper_transcriber import TranscriptResult, TranscriptSegment
    from scoring.exclamation_detector import detect_exclamations

    # 더미 에너지 데이터 (60초 영상)
    frame_times = np.linspace(0, 60, 1000)
    rms = np.random.uniform(0.1, 0.4, 1000)
    # 20~25초 구간 피크
    rms[333:416] = np.random.uniform(0.7, 0.95, 83)
    # 45~60초 침묵
    rms[750:] = np.random.uniform(0.01, 0.03, 250)

    energy = EnergyResult(
        frame_times=frame_times,
        rms_curve=rms,
        peak_segments=[PeakSegment(start=20.0, end=25.0, peak_energy=0.92)],
        silence_segments=[SilenceSegment(start=45.0, end=60.0)],
        pitch_spikes=[PitchSpike(time=21.0, pitch_hz=320.0)],
        audio_path=None,
    )

    # 더미 전사 데이터
    transcript = TranscriptResult(
        segments=[
            TranscriptSegment(start=20.0, end=22.0, text="야!! 대박이다 진짜"),
            TranscriptSegment(start=22.0, end=24.0, text="이겼어 클리어!!"),
            TranscriptSegment(start=30.0, end=32.0, text="다음 스테이지 가보자"),
        ],
        language="ko",
        total_duration=60.0,
    )
    exclamation = detect_exclamations(transcript)

    scored = score_segments(
        energy=energy,
        exclamation=exclamation,
        transcript=transcript,
        active_modules={"emotion_analyzer": True},
    )

    print("\n🏆 스코어링 결과")
    print("─" * 65)
    for s in scored:
        tag = "🔴 침묵" if s.is_silence else "    "
        print(
            f"  {tag} [{s.start:.1f}~{s.end:.1f}s] "
            f"final={s.final_score:.3f}  "
            f"audio={s.audio_score:.2f}  "
            f"emo={s.emotion_score:.2f}  "
            f"hook={s.hook_score:.2f}"
        )
        if s.transcript_text:
            print(f"         \"{s.transcript_text[:60]}\"")
