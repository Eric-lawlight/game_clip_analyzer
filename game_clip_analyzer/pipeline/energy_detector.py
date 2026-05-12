"""
energy_detector.py — 오디오 에너지 분석 모듈
==============================================
librosa를 사용해 오디오의 에너지 곡선, 침묵 구간, 피치 변화를 분석한다.
scorer.py의 audio_score 계산에 필요한 원시 데이터를 제공한다.

분석 항목:
    1. RMS 에너지 곡선  — 프레임별 음량 (0~1 정규화)
    2. 에너지 피크 구간  — 에너지가 급격히 높아지는 순간
    3. 침묵 구간         — 에너지가 임계값 이하로 지속되는 구간
    4. 피치 변화         — 실況자가 목소리를 높이는 순간

반환 데이터:
    EnergyResult (dataclass)
        .frame_times    : 프레임 타임코드 배열 (초)
        .rms_curve      : RMS 에너지 배열 (0~1)
        .peak_segments  : [(start, end, peak_energy), ...]
        .silence_segments: [(start, end, duration), ...]
        .pitch_spikes   : [(time, pitch_hz), ...]  — 급격한 피치 상승 시점

사용 예:
    from pipeline.energy_detector import detect_energy
    from pipeline.audio_extractor import AudioInfo

    result = detect_energy(audio_info)
    for start, end, energy in result.peak_segments:
        print(f"[{start:.1f}s ~ {end:.1f}s]  에너지 피크: {energy:.2f}")
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from config import AUDIO_SAMPLE_RATE, CACHE_DIR
from pipeline.audio_extractor import AudioInfo


# ─────────────────────────────────────────────
# 분석 파라미터
# ─────────────────────────────────────────────

# RMS 프레임 크기: 512 샘플 = 약 32ms @ 16kHz
_HOP_LENGTH = 512

# 에너지 피크 판정 임계값 (정규화 0~1 기준)
_PEAK_THRESHOLD = 0.65

# 침묵 판정 임계값
_SILENCE_THRESHOLD = 0.05

# 침묵 최소 지속 시간 (초) — 이보다 짧으면 무시
_SILENCE_MIN_DURATION = 3.0

# Red 마커용 침묵 기준 (초) — config.py LABEL_THRESHOLDS와 동일
_SILENCE_RED_DURATION = 15.0

# 피치 스파이크: 직전 프레임 대비 이 비율 이상 상승 시 감지
_PITCH_SPIKE_RATIO = 1.4


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class SilenceSegment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_red(self) -> bool:
        """Red 마커 기준(15초+) 충족 여부."""
        return self.duration >= _SILENCE_RED_DURATION


@dataclass
class PeakSegment:
    start: float
    end: float
    peak_energy: float  # 구간 내 최대 에너지 (0~1)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class PitchSpike:
    time: float         # 스파이크 발생 시점 (초)
    pitch_hz: float     # 해당 시점 피치


@dataclass
class EnergyResult:
    """에너지 분석 전체 결과."""

    frame_times: np.ndarray         # 프레임별 타임코드
    rms_curve: np.ndarray           # 정규화된 RMS (0~1)
    peak_segments: list[PeakSegment]
    silence_segments: list[SilenceSegment]
    pitch_spikes: list[PitchSpike]
    audio_path: Path

    @property
    def red_silences(self) -> list[SilenceSegment]:
        """Red 마커 대상 침묵 구간만 반환."""
        return [s for s in self.silence_segments if s.is_red]

    def energy_at(self, time_sec: float) -> float:
        """특정 시점의 RMS 에너지를 반환한다 (보간)."""
        if len(self.frame_times) == 0:
            return 0.0
        idx = int(np.searchsorted(self.frame_times, time_sec))
        idx = min(idx, len(self.rms_curve) - 1)
        return float(self.rms_curve[idx])

    def avg_energy_in(self, start: float, end: float) -> float:
        """구간 내 평균 RMS 에너지를 반환한다."""
        mask = (self.frame_times >= start) & (self.frame_times <= end)
        if not mask.any():
            return 0.0
        return float(self.rms_curve[mask].mean())


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

def detect_energy(
    audio_info: AudioInfo,
    progress_cb: Optional[Callable[[str], None]] = None,
    force: bool = False,
) -> EnergyResult:
    """
    오디오 에너지를 분석한다.

    Args:
        audio_info  : extract_audio() 반환값
        progress_cb : 진행 상황 콜백
        force       : 캐시 무시 여부

    Returns:
        EnergyResult
    """
    log = progress_cb or _noop
    cache_path = _cache_path(audio_info.path)

    if cache_path.exists() and not force:
        log(f"  ♻️  에너지 분석 캐시 사용: {cache_path.name}")
        return _load_from_cache(cache_path, audio_info.path)

    log("  📊 오디오 에너지 분석 중...")
    y, sr = _load_audio(audio_info.path)

    # 1. RMS 에너지 곡선
    log("  　├─ RMS 에너지 계산")
    rms_raw = _compute_rms(y, sr)
    rms_norm, frame_times = rms_raw

    # 2. 피크 구간
    log("  　├─ 에너지 피크 감지")
    peaks = _find_peak_segments(rms_norm, frame_times)

    # 3. 침묵 구간
    log("  　├─ 침묵 구간 감지")
    silences = _find_silence_segments(rms_norm, frame_times)

    # 4. 피치 스파이크
    log("  　└─ 피치 변화 감지")
    pitch_spikes = _find_pitch_spikes(y, sr, frame_times)

    result = EnergyResult(
        frame_times=frame_times,
        rms_curve=rms_norm,
        peak_segments=peaks,
        silence_segments=silences,
        pitch_spikes=pitch_spikes,
        audio_path=audio_info.path,
    )

    _save_to_cache(cache_path, result)
    log(
        f"  ✅ 에너지 분석 완료: "
        f"피크 {len(peaks)}개 / 침묵 {len(silences)}개 "
        f"(Red 대상: {len(result.red_silences)}개)"
    )
    return result


# ─────────────────────────────────────────────
# 분석 내부 함수
# ─────────────────────────────────────────────

def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    """librosa로 오디오를 로딩한다."""
    try:
        import librosa
        y, sr = librosa.load(str(path), sr=AUDIO_SAMPLE_RATE, mono=True)
        return y, sr
    except ImportError:
        raise RuntimeError(
            "librosa가 설치되지 않았습니다.\n"
            "pip install librosa soundfile"
        )


def _compute_rms(
    y: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RMS 에너지를 프레임별로 계산하고 0~1로 정규화한다.

    Returns:
        (rms_normalized, frame_times)
    """
    import librosa

    rms = librosa.feature.rms(y=y, hop_length=_HOP_LENGTH)[0]
    frame_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=_HOP_LENGTH
    )

    # 0~1 정규화 (최대값으로 나눔, 최대값이 0이면 그대로)
    max_val = rms.max()
    rms_norm = rms / max_val if max_val > 0 else rms

    return rms_norm, frame_times


def _find_peak_segments(
    rms: np.ndarray,
    times: np.ndarray,
) -> list[PeakSegment]:
    """
    에너지 임계값(_PEAK_THRESHOLD) 이상인 연속 구간을 찾는다.
    인접한 피크 구간은 병합한다 (gap < 2초).
    """
    above = rms >= _PEAK_THRESHOLD
    segments: list[PeakSegment] = []
    in_peak = False
    seg_start = 0.0
    seg_max = 0.0

    for i, (t, is_above) in enumerate(zip(times, above)):
        if is_above and not in_peak:
            in_peak = True
            seg_start = float(t)
            seg_max = float(rms[i])
        elif is_above and in_peak:
            seg_max = max(seg_max, float(rms[i]))
        elif not is_above and in_peak:
            in_peak = False
            seg_end = float(t)
            # 인접 병합: 직전 구간과 간격 < 2초이면 합친다
            if segments and (seg_start - segments[-1].end) < 2.0:
                prev = segments.pop()
                seg_start = prev.start
                seg_max = max(prev.peak_energy, seg_max)
            segments.append(PeakSegment(
                start=seg_start,
                end=seg_end,
                peak_energy=seg_max,
            ))

    # 마지막 구간이 열려 있으면 닫는다
    if in_peak and len(times) > 0:
        segments.append(PeakSegment(
            start=seg_start,
            end=float(times[-1]),
            peak_energy=seg_max,
        ))

    return segments


def _find_silence_segments(
    rms: np.ndarray,
    times: np.ndarray,
) -> list[SilenceSegment]:
    """
    에너지가 _SILENCE_THRESHOLD 이하로 _SILENCE_MIN_DURATION 이상 지속되는 구간.
    """
    below = rms <= _SILENCE_THRESHOLD
    segments: list[SilenceSegment] = []
    in_silence = False
    seg_start = 0.0

    for t, is_below in zip(times, below):
        if is_below and not in_silence:
            in_silence = True
            seg_start = float(t)
        elif not is_below and in_silence:
            in_silence = False
            seg_end = float(t)
            duration = seg_end - seg_start
            if duration >= _SILENCE_MIN_DURATION:
                segments.append(SilenceSegment(start=seg_start, end=seg_end))

    if in_silence and len(times) > 0:
        seg_end = float(times[-1])
        if (seg_end - seg_start) >= _SILENCE_MIN_DURATION:
            segments.append(SilenceSegment(start=seg_start, end=seg_end))

    return segments


def _find_pitch_spikes(
    y: np.ndarray,
    sr: int,
    frame_times: np.ndarray,
) -> list[PitchSpike]:
    """
    피치가 직전 프레임 대비 _PITCH_SPIKE_RATIO 이상 급격히 상승하는 시점 감지.
    음성이 없는 구간(0Hz)은 제외한다.
    """
    try:
        import librosa
    except ImportError:
        return []

    # 피치 추정 (pyin: 보다 안정적, 느림 / yin: 빠름)
    # yin 사용 (속도 우선)
    f0 = librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz (남성 하한)
        fmax=librosa.note_to_hz("C6"),   # ~1047 Hz (여성 상한)
        hop_length=_HOP_LENGTH,
    )

    # frame_times 길이에 맞게 자르기
    f0 = f0[: len(frame_times)]

    spikes: list[PitchSpike] = []
    prev_f0 = 0.0

    for i, (t, pitch) in enumerate(zip(frame_times, f0)):
        if pitch <= 0:
            prev_f0 = 0.0
            continue
        if prev_f0 > 0 and pitch / prev_f0 >= _PITCH_SPIKE_RATIO:
            spikes.append(PitchSpike(time=float(t), pitch_hz=float(pitch)))
        prev_f0 = pitch

    return spikes


# ─────────────────────────────────────────────
# 캐시 관리
# ─────────────────────────────────────────────

def _cache_path(audio_path: Path) -> Path:
    return CACHE_DIR / f"{audio_path.stem}_energy.json"


def _save_to_cache(cache_path: Path, result: EnergyResult) -> None:
    data = {
        "frame_times": result.frame_times.tolist(),
        "rms_curve": result.rms_curve.tolist(),
        "peak_segments": [
            {"start": p.start, "end": p.end, "peak_energy": p.peak_energy}
            for p in result.peak_segments
        ],
        "silence_segments": [
            {"start": s.start, "end": s.end}
            for s in result.silence_segments
        ],
        "pitch_spikes": [
            {"time": ps.time, "pitch_hz": ps.pitch_hz}
            for ps in result.pitch_spikes
        ],
    }
    cache_path.write_text(json.dumps(data, ensure_ascii=False))


def _load_from_cache(cache_path: Path, audio_path: Path) -> EnergyResult:
    data = json.loads(cache_path.read_text())
    return EnergyResult(
        frame_times=np.array(data["frame_times"]),
        rms_curve=np.array(data["rms_curve"]),
        peak_segments=[PeakSegment(**p) for p in data["peak_segments"]],
        silence_segments=[SilenceSegment(**s) for s in data["silence_segments"]],
        pitch_spikes=[PitchSpike(**ps) for ps in data["pitch_spikes"]],
        audio_path=audio_path,
    )


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────

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
        print("사용법: python -m pipeline.energy_detector <영상파일경로>")
        sys.exit(1)

    try:
        info = inspect_video(sys.argv[1])
        audio = extract_audio(info)
        result = detect_energy(audio, progress_cb=print, force=True)

        print(f"\n📊 에너지 분석 결과")
        print("─" * 50)
        print(f"  에너지 피크: {len(result.peak_segments)}개")
        for p in result.peak_segments[:5]:
            print(f"    [{p.start:.1f}s ~ {p.end:.1f}s]  피크: {p.peak_energy:.2f}")

        print(f"\n  침묵 구간: {len(result.silence_segments)}개")
        for s in result.silence_segments[:5]:
            tag = "🔴 Red" if s.is_red else "   "
            print(f"    {tag} [{s.start:.1f}s ~ {s.end:.1f}s]  {s.duration:.1f}초")

        print(f"\n  피치 스파이크: {len(result.pitch_spikes)}개")
        for ps in result.pitch_spikes[:5]:
            print(f"    [{ps.time:.1f}s]  {ps.pitch_hz:.0f} Hz")

    except Exception as e:
        print(f"\n❌ 오류: {e}")
        sys.exit(1)
