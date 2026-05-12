"""
exclamation_detector.py — 한국어 감탄사 감지 모듈
===================================================
Whisper 전사 결과의 텍스트에서 감탄사 패턴을 감지한다.
정규식 기반으로 동작하며 외부 모델 없이 CPU만 사용한다.

감탄사 유형:
    excitement  : 흥분/성공 ("야!!", "대박", "이겼어")
    failure     : 실패/당황 ("아씨", "죽었다", "망했어")
    laughter    : 웃음     ("ㅋㅋㅋㅋㅋ", "헐", "미쳤다")
    cut_signal  : 컷 대상  ("잠깐만", "로딩", "화장실")

반환 데이터:
    ExclamationHit (dataclass)
        .segment    : 감지된 TranscriptSegment
        .pattern_type : "excitement" | "failure" | "laughter" | "cut_signal"
        .matched_text : 매칭된 텍스트
        .score      : 0~1 (패턴 수 × 가중치)

    ExclamationResult (dataclass)
        .hits       : List[ExclamationHit]
        .by_type    : {pattern_type: [hit, ...]}

사용 예:
    from scoring.exclamation_detector import detect_exclamations
    from pipeline.whisper_transcriber import TranscriptResult

    exc_result = detect_exclamations(transcript)
    for hit in exc_result.hits:
        print(f"[{hit.segment.start:.1f}s] {hit.pattern_type}: {hit.matched_text}")
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

from config import EXCLAMATION_PATTERNS
from pipeline.whisper_transcriber import TranscriptResult, TranscriptSegment


# ─────────────────────────────────────────────
# 패턴 가중치 (pattern_type → score 기여도)
# ─────────────────────────────────────────────

_TYPE_WEIGHTS = {
    "excitement": 1.0,
    "failure":    0.8,   # 실패 반응도 하이라이트 후보
    "laughter":   0.9,
    "cut_signal": -0.5,  # 네거티브: scorer에서 감점 신호
}

# 한 세그먼트에서 여러 패턴이 매칭될 경우 추가 점수
_MULTI_MATCH_BONUS = 0.2


# ─────────────────────────────────────────────
# 컴파일된 패턴 캐시 (모듈 로딩 시 1회만 컴파일)
# ─────────────────────────────────────────────

_COMPILED: dict[str, list[re.Pattern]] = {
    ptype: [re.compile(p, re.IGNORECASE) for p in patterns]
    for ptype, patterns in EXCLAMATION_PATTERNS.items()
}


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class ExclamationHit:
    """감탄사 감지 결과 1건."""

    segment: TranscriptSegment
    pattern_type: str           # "excitement" | "failure" | "laughter" | "cut_signal"
    matched_texts: list[str]    # 매칭된 텍스트 목록 (중복 제거)
    score: float                # -1~1

    @property
    def is_positive(self) -> bool:
        return self.score > 0

    @property
    def time(self) -> float:
        return self.segment.start


@dataclass
class ExclamationResult:
    """전체 감탄사 감지 결과."""

    hits: list[ExclamationHit]

    @property
    def by_type(self) -> dict[str, list[ExclamationHit]]:
        result: dict[str, list[ExclamationHit]] = defaultdict(list)
        for hit in self.hits:
            result[hit.pattern_type].append(hit)
        return dict(result)

    @property
    def positive_hits(self) -> list[ExclamationHit]:
        return [h for h in self.hits if h.is_positive]

    @property
    def cut_signals(self) -> list[ExclamationHit]:
        return [h for h in self.hits if h.pattern_type == "cut_signal"]

    def hits_in_range(self, start: float, end: float) -> list[ExclamationHit]:
        """특정 시간 범위 내의 히트만 반환한다."""
        return [h for h in self.hits if start <= h.time <= end]

    def score_at(self, start: float, end: float) -> float:
        """특정 구간의 감탄사 점수 합산을 반환한다 (scorer 전용)."""
        hits = self.hits_in_range(start, end)
        if not hits:
            return 0.0
        total = sum(h.score for h in hits)
        return max(-1.0, min(1.0, total))


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

def detect_exclamations(transcript: TranscriptResult) -> ExclamationResult:
    """
    전사 결과에서 감탄사 패턴을 감지한다.

    Args:
        transcript: transcribe() 반환값

    Returns:
        ExclamationResult
    """
    hits: list[ExclamationHit] = []

    for segment in transcript.segments:
        text = segment.text_clean
        if not text:
            continue

        seg_hits = _scan_segment(segment, text)
        hits.extend(seg_hits)

    return ExclamationResult(hits=hits)


# ─────────────────────────────────────────────
# 내부 함수
# ─────────────────────────────────────────────

def _scan_segment(
    segment: TranscriptSegment,
    text: str,
) -> list[ExclamationHit]:
    """
    하나의 세그먼트 텍스트에서 모든 패턴 유형을 검사한다.
    같은 유형의 여러 패턴이 매칭되면 1개의 히트로 병합한다.
    """
    hits: list[ExclamationHit] = []

    for ptype, patterns in _COMPILED.items():
        matched_texts: list[str] = []

        for pattern in patterns:
            match = pattern.search(text)
            if match:
                matched_texts.append(match.group(0))

        if not matched_texts:
            continue

        # 점수 계산
        base_weight = _TYPE_WEIGHTS.get(ptype, 0.5)
        bonus = _MULTI_MATCH_BONUS * (len(matched_texts) - 1)
        score = base_weight + bonus
        score = max(-1.0, min(1.0, score))

        hits.append(ExclamationHit(
            segment=segment,
            pattern_type=ptype,
            matched_texts=list(dict.fromkeys(matched_texts)),  # 순서 유지 중복 제거
            score=score,
        ))

    return hits


# ─────────────────────────────────────────────
# 단독 실행 테스트 (텍스트 직접 입력)
# ─────────────────────────────────────────────

def _demo_with_text(texts: list[str]) -> None:
    """텍스트 리스트를 직접 넣어 패턴 감지 결과를 출력한다."""
    from pipeline.whisper_transcriber import TranscriptSegment, TranscriptResult

    segments = [
        TranscriptSegment(start=i * 5.0, end=(i + 1) * 5.0, text=t)
        for i, t in enumerate(texts)
    ]
    transcript = TranscriptResult(
        segments=segments,
        language="ko",
        total_duration=len(texts) * 5.0,
    )
    result = detect_exclamations(transcript)

    print(f"\n🎯 감탄사 감지 결과: {len(result.hits)}건")
    print("─" * 60)
    for hit in result.hits:
        sign = "+" if hit.is_positive else ""
        print(
            f"  [{hit.time:.1f}s] [{hit.pattern_type:10s}] "
            f"score={sign}{hit.score:.2f}  "
            f"매칭: {', '.join(hit.matched_texts)}"
        )
        print(f"         텍스트: \"{hit.segment.text_clean}\"")


if __name__ == "__main__":
    # 샘플 텍스트로 패턴 감지 검증
    sample_texts = [
        "야!! 이게 뭐야 대박이다 진짜",
        "아씨 또 죽었어 왜 이래",
        "ㅋㅋㅋㅋㅋ 헐 미쳤다 진짜",
        "잠깐만요 화장실 다녀올게요",
        "지금 로딩 중이에요 잠시만",
        "이겼다! 클리어! 와아아아아!",
        "일반 플레이 중입니다 별 일 없어요",
    ]
    _demo_with_text(sample_texts)
