"""
visual_analyzer.py — 시각 보정 모듈 (Qwen2-VL 2B)
====================================================
에너지/감정 분석으로 추린 후보 구간의 프레임을 Qwen2-VL로 분석해
오디오 판정을 보정하거나 확정한다.

2패스 구조:
    1패스 (audio): 전체 영상 오디오 분석 → 후보 구간 선정
    2패스 (visual): 후보 구간 ±5초의 프레임만 Qwen으로 분석 (이 모듈)

Qwen에게 묻는 것:
    1. "이 장면은 무엇을 하고 있는가?" → 맥락 이해
    2. "로딩/죽음 화면인가?" → Red 판정 보정
    3. "전투/긴장/승리 장면인가?" → Green/Yellow 강화

VRAM 관리:
    - 4bit 양자화(bitsandbytes) 사용 → 약 3.5GB
    - 분석 완료 후 모델 언로드 (del model + torch.cuda.empty_cache())
    - 영상 프레임: 480p 다운샘플 후 2초당 1프레임 샘플링

반환 데이터:
    VisualJudgment (dataclass) — 구간 1개 판정
        .start, .end    : 구간 타임코드
        .scene_type     : "gameplay" | "loading" | "death" | "victory" | "cutscene" | "unknown"
        .score_delta    : 점수 조정값 (-0.3 ~ +0.3)
        .description    : Qwen의 장면 설명
        .frames_analyzed: 분석한 프레임 수

    VisualResult (dataclass)
        .judgments      : List[VisualJudgment]
        .as_score_dict  : {(start, end): score_delta} — scorer.py용

사용 예:
    from pipeline.visual_analyzer import analyze_visuals
    from scoring.scorer import ScoredSegment

    candidates = [(24.0, 36.0), (54.0, 66.0)]
    result = analyze_visuals(video_info, candidates, progress_cb=print)
    print(result.as_score_dict)
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from config import (
    QWEN_MODEL,
    CACHE_DIR,
    FRAME_SAMPLE_INTERVAL,
    VISUAL_CANDIDATE_PADDING,
    VISUAL_DOWNSCALE_HEIGHT,
)
from core.video_inspector import VideoInfo


# ─────────────────────────────────────────────
# 장면 유형 정의
# ─────────────────────────────────────────────

# score_delta: 각 장면 유형이 final_score에 미치는 조정값
_SCENE_SCORE_DELTA = {
    "victory":  +0.25,   # 승리/클리어 → Green 강화
    "combat":   +0.15,   # 전투/긴장 → 하이라이트 후보
    "gameplay": +0.05,   # 일반 게임플레이 → 소폭 상승
    "cutscene": +0.00,   # 컷씬 → 중립
    "unknown":  +0.00,   # 판단 불가 → 중립
    "death":    -0.10,   # 캐릭터 죽음 화면 → 소폭 하락
    "loading":  -0.30,   # 로딩 화면 → Red 강화
}

# Qwen에게 전달할 프롬프트 템플릿
_SCENE_PROMPT = """이 게임 실황 캡처 이미지를 보고 아래 형식으로만 답하세요.

분류 기준:
- victory: 보스 처치, 스테이지 클리어, 승리 연출
- combat: 전투 중, 긴장 상황, 액션 장면
- gameplay: 일반 게임플레이, 탐험, 이동
- cutscene: 컷씬, 대화, 스토리 연출
- loading: 로딩 화면, 검은 화면, 초기화 화면
- death: 죽음/게임오버 화면, 페널티 연출
- unknown: 판단 불가

반드시 아래 JSON 형식만 출력하세요:
{"scene_type": "<위 분류 중 하나>", "description": "<한 문장 설명>", "confidence": <0.0~1.0>}"""


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class VisualJudgment:
    """후보 구간 1개의 시각 판정 결과."""

    start: float
    end: float
    scene_type: str         # "gameplay" | "loading" | "death" | "victory" | ...
    score_delta: float      # -0.3 ~ +0.3
    description: str
    frames_analyzed: int
    confidence: float = 1.0

    @property
    def is_cut_candidate(self) -> bool:
        """로딩/죽음 화면 → 컷 후보."""
        return self.scene_type in ("loading", "death")


@dataclass
class VisualResult:
    """전체 시각 분석 결과."""

    judgments: list[VisualJudgment]

    @property
    def as_score_dict(self) -> dict[tuple[float, float], float]:
        """scorer.py의 visual_scores 형식으로 변환한다."""
        return {
            (j.start, j.end): j.score_delta
            for j in self.judgments
        }

    @property
    def cut_candidates(self) -> list[VisualJudgment]:
        return [j for j in self.judgments if j.is_cut_candidate]


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

def analyze_visuals(
    video_info: VideoInfo,
    candidate_intervals: list[tuple[float, float]],
    progress_cb: Optional[Callable[[str], None]] = None,
    force: bool = False,
) -> VisualResult:
    """
    후보 구간 프레임을 Qwen2-VL로 분석한다.

    Args:
        video_info          : inspect_video() 반환값
        candidate_intervals : [(start, end), ...] — scorer의 후보 구간
        progress_cb         : 진행 상황 콜백
        force               : 캐시 무시 여부

    Returns:
        VisualResult
    """
    log = progress_cb or _noop

    cache_path = _cache_path(video_info.path)
    if cache_path.exists() and not force:
        log(f"  ♻️  시각 분석 캐시 사용")
        return _load_from_cache(cache_path)

    # 패딩 적용 (후보 구간 ±VISUAL_CANDIDATE_PADDING초 확장)
    padded = _apply_padding(candidate_intervals, video_info.duration_sec)

    log(f"  🎬 Qwen2-VL 모델 로딩 중... (4bit 양자화)")
    model, processor = _load_qwen_model()
    log(f"  ✅ 모델 로딩 완료")

    judgments: list[VisualJudgment] = []
    total = len(padded)

    for i, (start, end) in enumerate(padded, start=1):
        log(f"  🔍 구간 {i}/{total} 분석 중... ({start:.1f}s ~ {end:.1f}s)")

        # 프레임 추출
        frames = _extract_frames(video_info.path, start, end)
        if not frames:
            log(f"  　⚠️  프레임 추출 실패, 스킵")
            continue

        # Qwen 추론
        judgment = _analyze_frames(
            model=model,
            processor=processor,
            frames=frames,
            start=start,
            end=end,
        )
        judgments.append(judgment)
        log(f"  　→ {judgment.scene_type}  delta={judgment.score_delta:+.2f}  \"{judgment.description}\"")

        # 프레임 임시파일 정리
        for f in frames:
            Path(f).unlink(missing_ok=True)

    # 모델 언로드 (VRAM 반환)
    log("  🧹 Qwen2-VL 언로드 중...")
    _unload_model(model)
    log("  ✅ VRAM 반환 완료")

    result = VisualResult(judgments=judgments)
    _save_to_cache(cache_path, result)
    log(f"  ✅ 시각 분석 완료: {len(judgments)}개 구간")
    return result


# ─────────────────────────────────────────────
# 프레임 추출 (ffmpeg)
# ─────────────────────────────────────────────

def _apply_padding(
    intervals: list[tuple[float, float]],
    max_duration: float,
) -> list[tuple[float, float]]:
    """후보 구간에 패딩을 적용하고 영상 범위를 벗어나지 않도록 클램핑한다."""
    padded = []
    for start, end in intervals:
        s = max(0.0, start - VISUAL_CANDIDATE_PADDING)
        e = min(max_duration, end + VISUAL_CANDIDATE_PADDING)
        padded.append((s, e))
    return padded


def _extract_frames(
    video_path: Path,
    start: float,
    end: float,
) -> list[str]:
    """
    ffmpeg로 구간 내 프레임을 추출한다.

    - 480p 다운샘플
    - FRAME_SAMPLE_INTERVAL초당 1프레임
    - 임시 디렉터리에 JPEG 저장

    Returns:
        추출된 JPEG 파일 경로 리스트
    """
    with tempfile.TemporaryDirectory(delete=False) as tmpdir:
        out_pattern = str(Path(tmpdir) / "frame_%04d.jpg")

        cmd = [
            "ffmpeg",
            "-ss", str(start),
            "-to", str(end),
            "-i", str(video_path),
            "-vf", (
                f"scale=-2:{VISUAL_DOWNSCALE_HEIGHT},"
                f"fps=1/{FRAME_SAMPLE_INTERVAL}"   # N초당 1프레임
            ),
            "-q:v", "3",   # JPEG 품질 (1=최고, 31=최저)
            "-y",
            out_pattern,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=60,
            )
            if result.returncode != 0:
                return []
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

        frames = sorted(Path(tmpdir).glob("frame_*.jpg"))
        return [str(f) for f in frames]


# ─────────────────────────────────────────────
# Qwen2-VL 모델 로딩 / 추론
# ─────────────────────────────────────────────

def _load_qwen_model():
    """Qwen2-VL 2B를 4bit 양자화로 로딩한다."""
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from transformers import BitsAndBytesConfig
        import torch

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(QWEN_MODEL)
        return model, processor

    except ImportError as e:
        raise RuntimeError(
            f"Qwen2-VL 의존성 누락: {e}\n"
            "pip install transformers bitsandbytes qwen-vl-utils"
        )


def _analyze_frames(
    model,
    processor,
    frames: list[str],
    start: float,
    end: float,
) -> VisualJudgment:
    """
    추출된 프레임들을 Qwen2-VL로 분석한다.
    여러 프레임의 결과를 다수결로 합산한다.
    """
    from PIL import Image

    scene_votes: dict[str, float] = {}
    descriptions: list[str] = []

    for frame_path in frames:
        try:
            image = Image.open(frame_path).convert("RGB")
            judgment = _query_single_frame(model, processor, image)
            scene_type = judgment.get("scene_type", "unknown")
            confidence = float(judgment.get("confidence", 0.5))
            description = judgment.get("description", "")

            scene_votes[scene_type] = (
                scene_votes.get(scene_type, 0.0) + confidence
            )
            if description:
                descriptions.append(description)

        except Exception:
            continue

    # 다수결: 가장 높은 누적 confidence의 scene_type
    if not scene_votes:
        scene_type = "unknown"
        confidence = 0.0
    else:
        scene_type = max(scene_votes, key=lambda k: scene_votes[k])
        total_conf = sum(scene_votes.values())
        confidence = scene_votes[scene_type] / total_conf if total_conf > 0 else 0.0

    description = descriptions[0] if descriptions else ""
    score_delta = _SCENE_SCORE_DELTA.get(scene_type, 0.0) * confidence

    return VisualJudgment(
        start=start,
        end=end,
        scene_type=scene_type,
        score_delta=round(score_delta, 3),
        description=description,
        frames_analyzed=len(frames),
        confidence=round(confidence, 3),
    )


def _query_single_frame(model, processor, image) -> dict:
    """단일 프레임에 대해 Qwen2-VL에 질의한다."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _SCENE_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    # 입력 토큰 제거
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]

    return _parse_qwen_response(response)


def _parse_qwen_response(response: str) -> dict:
    """Qwen 응답 JSON을 파싱한다. 실패 시 unknown으로 fallback."""
    # 중괄호 범위 추출
    start = response.find("{")
    end = response.rfind("}") + 1
    if start == -1 or end == 0:
        return {"scene_type": "unknown", "description": "", "confidence": 0.3}

    try:
        data = json.loads(response[start:end])
        # 유효하지 않은 scene_type 처리
        if data.get("scene_type") not in _SCENE_SCORE_DELTA:
            data["scene_type"] = "unknown"
        return data
    except json.JSONDecodeError:
        return {"scene_type": "unknown", "description": response[:50], "confidence": 0.3}


def _unload_model(model) -> None:
    """모델을 메모리에서 해제한다."""
    try:
        import torch
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass


# ─────────────────────────────────────────────
# 캐시 관리
# ─────────────────────────────────────────────

def _cache_path(video_path: Path) -> Path:
    return CACHE_DIR / f"{video_path.stem}_visual.json"


def _save_to_cache(cache_path: Path, result: VisualResult) -> None:
    data = [
        {
            "start": j.start,
            "end": j.end,
            "scene_type": j.scene_type,
            "score_delta": j.score_delta,
            "description": j.description,
            "frames_analyzed": j.frames_analyzed,
            "confidence": j.confidence,
        }
        for j in result.judgments
    ]
    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def _load_from_cache(cache_path: Path) -> VisualResult:
    data = json.loads(cache_path.read_text())
    return VisualResult(judgments=[VisualJudgment(**item) for item in data])


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────

def _noop(_: str) -> None:
    pass


# ─────────────────────────────────────────────
# 단독 실행 테스트 (더미 데이터)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 캐시 파싱 / 데이터 구조 검증만 수행 (모델 없이)
    dummy_judgments = [
        VisualJudgment(
            start=24.0, end=36.0,
            scene_type="victory",
            score_delta=+0.25,
            description="보스 처치 승리 연출이 화면에 표시되고 있음",
            frames_analyzed=6,
            confidence=0.91,
        ),
        VisualJudgment(
            start=54.0, end=66.0,
            scene_type="gameplay",
            score_delta=+0.05,
            description="일반 게임플레이 중, 캐릭터가 지형을 탐색 중",
            frames_analyzed=6,
            confidence=0.78,
        ),
        VisualJudgment(
            start=96.0, end=120.0,
            scene_type="loading",
            score_delta=-0.30,
            description="로딩 화면, 진행바가 표시됨",
            frames_analyzed=6,
            confidence=0.97,
        ),
    ]
    result = VisualResult(judgments=dummy_judgments)

    print("\n🎬 시각 분석 결과 (더미)")
    print("─" * 65)
    for j in result.judgments:
        tag = "🔴" if j.is_cut_candidate else "✅"
        print(
            f"  {tag} [{j.start:.1f}~{j.end:.1f}s]  "
            f"{j.scene_type:10s}  delta={j.score_delta:+.2f}  "
            f"conf={j.confidence:.2f}"
        )
        print(f"       \"{j.description}\"")

    print(f"\n  컷 후보: {len(result.cut_candidates)}개")
    print(f"  scorer용 score_dict: {result.as_score_dict}")
