"""
emotion_analyzer.py — 한국어 감정 분류 모듈
=============================================
KoELECTRA 기반 모델로 Whisper 전사 세그먼트별 감정을 분류한다.
exclamation_detector의 패턴 매칭을 보완하는 역할이다.
(패턴은 빠르고 명확한 경우를 잡고, KoELECTRA는 맥락 의존 케이스를 잡는다)

감정 레이블:
    excitement  : 흥분/성공
    calm        : 평온 (일반 플레이)
    disappointment : 실망/실패
    laughter    : 웃음
    surprise    : 당황/놀람

반환 데이터:
    EmotionScore (dataclass) — 세그먼트 1개의 감정 분류 결과
        .segment        : 원본 TranscriptSegment
        .label          : 주 감정 레이블
        .scores         : {감정: 확률} 딕셔너리
        .intensity      : 주 감정의 확률 (0~1)
        .is_highlight_emotion : excitement or laughter or surprise

    EmotionResult (dataclass)
        .scored_segments : List[EmotionScore]
        .intensity_at(start, end) : 구간 내 평균 감정 강도

사용 예:
    from pipeline.emotion_analyzer import analyze_emotions
    result = analyze_emotions(transcript, progress_cb=print)
    for es in result.scored_segments:
        print(f"[{es.segment.start:.1f}s] {es.label} ({es.intensity:.2f})")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from config import KOELECTRA_MODEL, CACHE_DIR
from pipeline.whisper_transcriber import TranscriptResult, TranscriptSegment


# ─────────────────────────────────────────────
# 감정 레이블 정의
# ─────────────────────────────────────────────

# 모델 출력 레이블 → 내부 레이블 매핑
# 실제 모델에 따라 조정 필요
_LABEL_MAP = {
    # snunlp/KR-FinBert-SC 계열
    "positive":  "excitement",
    "neutral":   "calm",
    "negative":  "disappointment",
    # 감정 특화 모델 레이블 (사용 모델에 따라 추가)
    "joy":       "laughter",
    "surprise":  "surprise",
    "anger":     "disappointment",
    "fear":      "surprise",
    "sadness":   "disappointment",
    # 한국어 직접 레이블
    "기쁨":      "excitement",
    "놀람":      "surprise",
    "분노":      "disappointment",
    "슬픔":      "disappointment",
    "중립":      "calm",
}

# 하이라이트 감정 집합
_HIGHLIGHT_EMOTIONS = {"excitement", "laughter", "surprise"}

# 배치 크기 (VRAM 절약)
_BATCH_SIZE = 16


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class EmotionScore:
    """세그먼트 1개의 감정 분류 결과."""

    segment: TranscriptSegment
    label: str              # 주 감정 레이블
    scores: dict[str, float]  # {레이블: 확률}
    intensity: float        # 주 감정 확률 (0~1)

    @property
    def is_highlight_emotion(self) -> bool:
        return self.label in _HIGHLIGHT_EMOTIONS

    @property
    def is_calm(self) -> bool:
        return self.label == "calm"

    @property
    def start(self) -> float:
        return self.segment.start

    @property
    def end(self) -> float:
        return self.segment.end


@dataclass
class EmotionResult:
    """전체 감정 분류 결과."""

    scored_segments: list[EmotionScore]

    def intensity_at(self, start: float, end: float) -> float:
        """
        구간 내 하이라이트 감정 세그먼트의 평균 강도를 반환한다.
        하이라이트 감정이 없으면 0.0.
        """
        in_range = [
            es for es in self.scored_segments
            if start <= es.start <= end and es.is_highlight_emotion
        ]
        if not in_range:
            return 0.0
        return sum(es.intensity for es in in_range) / len(in_range)

    def dominant_emotion_in(self, start: float, end: float) -> Optional[str]:
        """구간 내 가장 강한 감정 레이블을 반환한다."""
        in_range = [
            es for es in self.scored_segments
            if start <= es.start <= end
        ]
        if not in_range:
            return None
        return max(in_range, key=lambda es: es.intensity).label

    def highlight_segments(
        self,
        min_intensity: float = 0.6,
    ) -> list[EmotionScore]:
        """하이라이트 감정 + 최소 강도 이상의 세그먼트만 반환한다."""
        return [
            es for es in self.scored_segments
            if es.is_highlight_emotion and es.intensity >= min_intensity
        ]


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

def analyze_emotions(
    transcript: TranscriptResult,
    progress_cb: Optional[Callable[[str], None]] = None,
    force: bool = False,
) -> EmotionResult:
    """
    전사 결과의 각 세그먼트에 감정을 분류한다.

    Args:
        transcript  : transcribe() 반환값
        progress_cb : 진행 상황 콜백
        force       : 캐시 무시 여부

    Returns:
        EmotionResult
    """
    log = progress_cb or _noop
    cache_path = _cache_path(transcript)

    if cache_path.exists() and not force:
        log(f"  ♻️  감정 분석 캐시 사용")
        return _load_from_cache(cache_path, transcript)

    log("  🧠 KoELECTRA 모델 로딩 중...")
    pipeline = _load_pipeline()
    log(f"  ✅ 모델 로딩 완료")

    # 텍스트가 있는 세그먼트만 분류
    valid_segs = [s for s in transcript.segments if s.text_clean]
    total = len(valid_segs)
    log(f"  🔄 감정 분류 중... ({total}개 세그먼트)")

    emotion_scores: list[EmotionScore] = []

    # 배치 처리
    for i in range(0, total, _BATCH_SIZE):
        batch = valid_segs[i : i + _BATCH_SIZE]
        texts = [s.text_clean for s in batch]
        results = _classify_batch(pipeline, texts)

        for seg, res in zip(batch, results):
            label = _normalize_label(res["label"])
            score = float(res["score"])

            # all_scores가 있으면 활용, 없으면 주 감정만
            all_scores = _extract_all_scores(res, label, score)

            emotion_scores.append(EmotionScore(
                segment=seg,
                label=label,
                scores=all_scores,
                intensity=score,
            ))

        done = min(i + _BATCH_SIZE, total)
        if done % 64 == 0 or done == total:
            log(f"  　{done}/{total} 완료")

    # 텍스트 없는 세그먼트는 calm으로 채움
    scored_map = {es.segment.start: es for es in emotion_scores}
    full_scores: list[EmotionScore] = []
    for seg in transcript.segments:
        if seg.start in scored_map:
            full_scores.append(scored_map[seg.start])
        else:
            full_scores.append(EmotionScore(
                segment=seg,
                label="calm",
                scores={"calm": 1.0},
                intensity=1.0,
            ))

    result = EmotionResult(scored_segments=full_scores)
    _save_to_cache(cache_path, result)

    highlight_count = len(result.highlight_segments())
    log(f"  ✅ 감정 분류 완료: 하이라이트 감정 {highlight_count}개 감지")
    return result


# ─────────────────────────────────────────────
# 모델 로딩 / 추론
# ─────────────────────────────────────────────

def _load_pipeline():
    """
    HuggingFace transformers 감정 분류 파이프라인을 로딩한다.
    VRAM이 있으면 GPU 사용.
    """
    try:
        from transformers import pipeline as hf_pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1  # 0=GPU, -1=CPU

        clf = hf_pipeline(
            "text-classification",
            model=KOELECTRA_MODEL,
            device=device,
            top_k=None,          # 모든 레이블 확률 반환
            truncation=True,
            max_length=512,
        )
        return clf

    except ImportError:
        raise RuntimeError(
            "transformers 또는 torch가 설치되지 않았습니다.\n"
            "pip install transformers torch"
        )


def _classify_batch(pipeline, texts: list[str]) -> list[dict]:
    """배치 텍스트를 분류한다."""
    try:
        results = pipeline(texts)
        # top_k=None이면 결과가 List[List[dict]] 형태
        # 각 항목에서 최고 점수 항목을 뽑거나 전체를 반환
        processed = []
        for item in results:
            if isinstance(item, list):
                # 최고 점수 레이블 추출 + 전체 점수 보존
                best = max(item, key=lambda x: x["score"])
                best["_all"] = item
                processed.append(best)
            else:
                processed.append(item)
        return processed
    except Exception as e:
        # 추론 실패 시 neutral로 fallback
        return [{"label": "neutral", "score": 0.5} for _ in texts]


def _normalize_label(raw_label: str) -> str:
    """모델 출력 레이블을 내부 레이블로 변환한다."""
    return _LABEL_MAP.get(raw_label.lower(), "calm")


def _extract_all_scores(res: dict, primary_label: str, primary_score: float) -> dict:
    """모든 레이블 확률을 딕셔너리로 반환한다."""
    if "_all" in res:
        return {
            _normalize_label(item["label"]): item["score"]
            for item in res["_all"]
        }
    return {primary_label: primary_score}


# ─────────────────────────────────────────────
# 캐시 관리
# ─────────────────────────────────────────────

def _cache_path(transcript: TranscriptResult) -> Path:
    # transcript의 총 길이를 키로 사용 (파일명 없음)
    key = int(transcript.total_duration)
    return CACHE_DIR / f"emotion_{key}s.json"


def _save_to_cache(cache_path: Path, result: EmotionResult) -> None:
    data = [
        {
            "start": es.segment.start,
            "end": es.segment.end,
            "text": es.segment.text,
            "label": es.label,
            "scores": es.scores,
            "intensity": es.intensity,
        }
        for es in result.scored_segments
    ]
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _load_from_cache(
    cache_path: Path,
    transcript: TranscriptResult,
) -> EmotionResult:
    data = json.loads(cache_path.read_text())
    seg_map = {s.start: s for s in transcript.segments}
    scored = []
    for item in data:
        seg = seg_map.get(item["start"]) or TranscriptSegment(
            start=item["start"],
            end=item["end"],
            text=item["text"],
        )
        scored.append(EmotionScore(
            segment=seg,
            label=item["label"],
            scores=item["scores"],
            intensity=item["intensity"],
        ))
    return EmotionResult(scored_segments=scored)


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────

def _noop(_: str) -> None:
    pass


# ─────────────────────────────────────────────
# 단독 실행 테스트 (모델 없이 더미 데이터)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from pipeline.whisper_transcriber import TranscriptSegment, TranscriptResult

    # 더미 데이터로 데이터 구조 검증
    segments = [
        TranscriptSegment(start=0.0,  end=3.0,  text="야!! 대박이다 이겼어"),
        TranscriptSegment(start=3.0,  end=6.0,  text="일반 플레이 중입니다"),
        TranscriptSegment(start=6.0,  end=9.0,  text="ㅋㅋㅋㅋㅋ 미쳤다"),
        TranscriptSegment(start=9.0,  end=12.0, text="아씨 또 죽었어"),
        TranscriptSegment(start=12.0, end=15.0, text="헐 이게 뭐야"),
    ]
    dummy_result = EmotionResult(scored_segments=[
        EmotionScore(seg, lbl, {lbl: sc}, sc)
        for seg, lbl, sc in zip(
            segments,
            ["excitement", "calm", "laughter", "disappointment", "surprise"],
            [0.92, 0.88, 0.91, 0.85, 0.79],
        )
    ])

    print("\n🧠 감정 분류 결과 (더미)")
    print("─" * 55)
    for es in dummy_result.scored_segments:
        flag = "⭐" if es.is_highlight_emotion else "  "
        print(
            f"  {flag} [{es.start:.1f}s] "
            f"{es.label:15s} {es.intensity:.2f}  "
            f"\"{es.segment.text_clean}\""
        )

    print(f"\n  하이라이트 감정 구간: {len(dummy_result.highlight_segments())}개")
    print(f"  10~15s 구간 감정 강도: {dummy_result.intensity_at(10, 15):.2f}")
