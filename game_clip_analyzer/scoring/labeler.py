"""
labeler.py — 라벨 분류 모듈
==============================
ScoredSegment 리스트를 받아 각 구간에 색깔 라벨을 부여한다.
라벨은 xml_exporter.py에서 마커 이름으로 사용된다.

라벨 우선순위:
    Red    > Green > Yellow > Orange > Blue
    (침묵/로딩이 있으면 다른 점수와 무관하게 Red)

반환 데이터:
    LabeledSegment (dataclass)
        .scored         : 원본 ScoredSegment
        .label          : "Green" | "Yellow" | "Blue" | "Orange" | "Red"
        .label_reason   : 분류 이유 (마커 코멘트에 포함)
        .marker_name    : "[Green] 하이라이트" 형식
        .marker_comment : "보스 처치 반응 | 흥분도 94 | ..." 형식

사용 예:
    from scoring.labeler import label_segments
    labeled = label_segments(scored_segments)
    for seg in labeled:
        print(seg.marker_name, seg.marker_comment)
"""

from dataclasses import dataclass, field

from config import LABEL_THRESHOLDS
from scoring.scorer import ScoredSegment


# ─────────────────────────────────────────────
# 라벨 메타데이터
# ─────────────────────────────────────────────

LABEL_META = {
    "Green": {
        "name": "하이라이트",
        "description": "최우선 편집 구간. 감정 폭발 + 음량 피크",
        "priority": 2,
    },
    "Yellow": {
        "name": "강조 후보",
        "description": "재미있는 장면, 웃음, 놀람 등",
        "priority": 3,
    },
    "Blue": {
        "name": "정상 플레이",
        "description": "편집 여부 검토 필요한 일반 구간",
        "priority": 5,
    },
    "Orange": {
        "name": "전환점",
        "description": "자연스러운 편집 컷 포인트",
        "priority": 4,
    },
    "Red": {
        "name": "컷 권장",
        "description": "제거 권장 구간",
        "priority": 1,
    },
}

# 감탄사 유형 → 마커 설명
_EXCLAMATION_DESCRIPTIONS = {
    "excitement": "흥분/성공 반응",
    "failure":    "실패/당황 반응",
    "laughter":   "웃음 반응",
    "cut_signal": "컷 신호 감지",
}


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class LabeledSegment:
    """라벨이 부여된 구간."""

    scored: ScoredSegment
    label: str              # "Green" | "Yellow" | "Blue" | "Orange" | "Red"
    label_reason: str       # 분류 근거
    marker_name: str        # "[Green] 하이라이트"
    marker_comment: str     # 마커 코멘트 전문

    @property
    def start(self) -> float:
        return self.scored.start

    @property
    def end(self) -> float:
        return self.scored.end

    @property
    def final_score(self) -> float:
        return self.scored.final_score

    @property
    def label_name(self) -> str:
        return LABEL_META[self.label]["name"]

    def __repr__(self) -> str:
        return (
            f"LabeledSegment({self.start:.1f}~{self.end:.1f}s "
            f"[{self.label}] score={self.final_score:.2f})"
        )


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

def label_segments(
    scored: list[ScoredSegment],
    user_context: str = "",
) -> list[LabeledSegment]:
    """
    ScoredSegment 리스트에 라벨을 부여한다.

    Args:
        scored       : score_segments() 반환값
        user_context : 게임명 + 하이라이트 기준 (마커 코멘트에 포함)

    Returns:
        List[LabeledSegment] — 타임코드(start) 오름차순 정렬
    """
    labeled = [_label_one(seg, user_context) for seg in scored]

    # 타임코드 순 정렬 (XML 마커는 시간 순서대로 삽입)
    labeled.sort(key=lambda s: s.start)
    return labeled


# ─────────────────────────────────────────────
# 단일 구간 라벨 판정
# ─────────────────────────────────────────────

def _label_one(seg: ScoredSegment, user_context: str) -> LabeledSegment:
    """한 구간에 라벨을 부여하고 LabeledSegment를 반환한다."""
    label, reason = _determine_label(seg)
    marker_name = f"[{label}] {LABEL_META[label]['name']}"
    marker_comment = _build_comment(seg, label, reason, user_context)

    return LabeledSegment(
        scored=seg,
        label=label,
        label_reason=reason,
        marker_name=marker_name,
        marker_comment=marker_comment,
    )


def _determine_label(seg: ScoredSegment) -> tuple[str, str]:
    """
    라벨 판정 로직.
    우선순위: Red(침묵/저에너지) > Green(고흥분) > Yellow(중간) > Orange(전환) > Blue

    Returns:
        (label, reason)
    """
    thresholds = LABEL_THRESHOLDS

    # ── Red: 침묵 / 에너지 극저 ──────────────
    if seg.is_silence:
        reason = f"침묵 {seg.silence_duration:.0f}초"
        return "Red", reason

    if seg.audio_score < thresholds["red"]["energy"]:
        reason = f"에너지 낮음 ({seg.audio_score:.2f})"
        return "Red", reason

    # cut_signal 감탄사가 있고 에너지도 낮으면 Red
    if (seg.dominant_exclamation == "cut_signal"
            and seg.audio_score < 0.4):
        reason = "컷 신호 감지 (로딩/이석)"
        return "Red", reason

    # ── Green: 높은 흥분 + 에너지 피크 ───────
    green_t = thresholds["green"]
    if (seg.emotion_score >= green_t["emotion"]
            and seg.audio_score >= green_t["audio_energy"]):
        exc_type = _exclamation_desc(seg)
        reason = f"감정 {seg.emotion_score:.2f} + 에너지 {seg.audio_score:.2f}"
        if exc_type:
            reason = f"{exc_type} | {reason}"
        return "Green", reason

    # final_score가 매우 높으면 Green (emotion 모듈 비활성 시 대체)
    if seg.final_score >= 0.82:
        reason = f"종합 점수 높음 ({seg.final_score:.2f})"
        return "Green", reason

    # ── Yellow: 중간 흥분 / 감탄사 감지 ─────
    yellow_t = thresholds["yellow"]
    if seg.emotion_score >= yellow_t["emotion"]:
        exc_type = _exclamation_desc(seg)
        reason = f"감정 {seg.emotion_score:.2f}"
        if exc_type:
            reason = f"{exc_type} | {reason}"
        return "Yellow", reason

    if seg.final_score >= 0.55:
        reason = f"종합 점수 중상 ({seg.final_score:.2f})"
        return "Yellow", reason

    if seg.dominant_exclamation in ("excitement", "laughter", "failure"):
        desc = _exclamation_desc(seg)
        return "Yellow", desc or "감탄사 감지"

    # ── Orange: 장면 전환 / 피치 스파이크 ────
    # (현재는 화자 분리 미구현 → 피치 스파이크만 사용)
    if seg.hook_score >= 0.6 and seg.audio_score < 0.55:
        reason = "도입부 임팩트 감지 (전환점)"
        return "Orange", reason

    # ── Blue: 기본 (일반 플레이) ──────────────
    return "Blue", f"일반 구간 (점수: {seg.final_score:.2f})"


def _exclamation_desc(seg: ScoredSegment) -> str:
    """감탄사 유형을 한글 설명으로 반환한다."""
    exc = seg.dominant_exclamation
    if exc is None:
        return ""
    return _EXCLAMATION_DESCRIPTIONS.get(exc, exc)


def _build_comment(
    seg: ScoredSegment,
    label: str,
    reason: str,
    user_context: str,
) -> str:
    """
    마커 코멘트 문자열 생성.
    Premiere Pro 마커 코멘트에 표시되는 텍스트.

    형식:
        흥분/성공 반응 | 흥분도 94 | "야 이게 뭐야!!"
    """
    parts = []

    if reason:
        parts.append(reason)

    # 흥분도: final_score → 0~100 정수
    excitement_pct = int(seg.final_score * 100)
    parts.append(f"흥분도 {excitement_pct}")

    # 전사 텍스트 (앞 40자)
    text = seg.transcript_text.strip()
    if text:
        trimmed = text[:40] + ("..." if len(text) > 40 else "")
        parts.append(f'"{trimmed}"')

    # 게임 컨텍스트
    if user_context:
        parts.append(f"[{user_context}]")

    return " | ".join(parts)


# ─────────────────────────────────────────────
# 통계 요약 (선택적 출력용)
# ─────────────────────────────────────────────

def summarize_labels(labeled: list[LabeledSegment]) -> str:
    """라벨 분포 요약 문자열을 반환한다."""
    from collections import Counter
    counts = Counter(s.label for s in labeled)
    lines = ["📊 라벨 분포"]
    for label in ["Green", "Yellow", "Orange", "Blue", "Red"]:
        n = counts.get(label, 0)
        bar = "█" * n
        lines.append(f"  {label:6s}: {n:3d}개  {bar}")
    lines.append(f"  합계  : {len(labeled):3d}개")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from pipeline.energy_detector import EnergyResult, PeakSegment, SilenceSegment, PitchSpike
    from pipeline.whisper_transcriber import TranscriptResult, TranscriptSegment
    from scoring.exclamation_detector import detect_exclamations
    from scoring.scorer import score_segments

    # 더미 데이터
    frame_times = np.linspace(0, 90, 1500)
    rms = np.random.uniform(0.1, 0.3, 1500)
    rms[300:450] = np.random.uniform(0.75, 0.95, 150)   # 18~27s 피크
    rms[600:750] = np.random.uniform(0.60, 0.80, 150)   # 36~45s 중간
    rms[1200:]   = np.random.uniform(0.01, 0.02, 300)   # 72~90s 침묵

    energy = EnergyResult(
        frame_times=frame_times,
        rms_curve=rms,
        peak_segments=[
            PeakSegment(start=18.0, end=27.0, peak_energy=0.94),
            PeakSegment(start=36.0, end=45.0, peak_energy=0.72),
        ],
        silence_segments=[SilenceSegment(start=72.0, end=90.0)],
        pitch_spikes=[PitchSpike(time=19.0, pitch_hz=350.0)],
        audio_path=None,
    )

    transcript = TranscriptResult(
        segments=[
            TranscriptSegment(start=18.0, end=21.0, text="야!!! 대박 이겼어 클리어!!!"),
            TranscriptSegment(start=36.0, end=39.0, text="ㅋㅋㅋㅋㅋ 헐 미쳤다"),
            TranscriptSegment(start=55.0, end=57.0, text="잠깐만요 화장실 다녀올게요"),
        ],
        language="ko",
        total_duration=90.0,
    )
    exclamation = detect_exclamations(transcript)
    scored = score_segments(energy, exclamation, transcript,
                            active_modules={"emotion_analyzer": True})
    labeled = label_segments(scored, user_context="다크소울 3")

    print("\n🏷️  라벨 분류 결과")
    print("─" * 70)
    for s in labeled:
        print(
            f"  [{s.label:6s}] {s.start:.1f}~{s.end:.1f}s  "
            f"score={s.final_score:.2f}"
        )
        print(f"           마커명  : {s.marker_name}")
        print(f"           코멘트  : {s.marker_comment}")
        print()

    print(summarize_labels(labeled))
