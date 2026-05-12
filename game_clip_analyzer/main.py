"""
main.py — 진입점
==================
사용자 입력 → 파이프라인 실행 → XML 출력의 전체 흐름을 조율한다.

실행 방법:
    python main.py                          # 대화형 UI
    python main.py game_vod.mp4             # 영상 경로 직접 전달
    python main.py game_vod.mp4 --no-ui    # UI 없이 기본 설정으로 실행
"""

import sys
import time
from pathlib import Path


def main() -> int:
    """
    메인 진입점.

    Returns:
        종료 코드 (0 = 성공, 1 = 오류)
    """
    args = _parse_args()

    # ── Step 0: 영상 사전 분석 ─────────────────
    print("\n🎮 게임 실황 편집점 분석기")
    print("─" * 50)

    try:
        from core.video_inspector import inspect_video
        print(f"\n📁 영상 분석 중: {args['video_path']}")
        video_info = inspect_video(args["video_path"])
        print(f"  ✅ {video_info.summary()}")
        if video_info.warnings:
            for w in video_info.warnings:
                print(f"  {w}")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ 영상 분석 실패: {e}")
        return 1

    # ── Step 1: 모듈 선택 UI ───────────────────
    if args["use_ui"]:
        try:
            from ui.module_selector import run_module_selector
            result = run_module_selector(video_info)
            if not result.confirmed:
                print("취소했습니다.")
                return 0
            active_modules = result.active_modules
            context = result.context
        except KeyboardInterrupt:
            print("\n\n취소했습니다.")
            return 0
    else:
        from config import DEFAULT_MODULES
        from ui.module_selector import UserContext
        active_modules = DEFAULT_MODULES.copy()
        context = UserContext()
        print(f"\n⚙️  기본 모듈 설정으로 실행합니다.")

    user_context_str = context.game_name if args["use_ui"] else ""

    # ── Step 2: 오디오 추출 ────────────────────
    print("\n[1/5] 오디오 추출")
    try:
        from pipeline.audio_extractor import extract_audio
        audio_info = extract_audio(video_info)
    except (ValueError, RuntimeError) as e:
        print(f"  ❌ {e}")
        return 1

    # ── Step 3: Whisper 전사 ───────────────────
    print("\n[2/5] 음성 전사 (Whisper)")
    try:
        from pipeline.whisper_transcriber import transcribe
        transcript = transcribe(audio_info, progress_cb=print)
    except RuntimeError as e:
        print(f"  ❌ {e}")
        return 1

    # ── Step 4: 에너지 분석 ────────────────────
    print("\n[3/5] 오디오 에너지 분석")
    try:
        from pipeline.energy_detector import detect_energy
        energy = detect_energy(audio_info, progress_cb=print)
    except RuntimeError as e:
        print(f"  ❌ {e}")
        return 1

    # ── Step 4.5: 화자 분리 (협방송 시) ──────────
    if active_modules.get("speaker_separator"):
        print("\n[3.5/5] 화자 분리 (pyannote)")
        try:
            from pipeline.speaker_separator import separate_speakers
            speaker_result = separate_speakers(audio_info, progress_cb=print)
            transcript = speaker_result.filter_transcript(transcript)
            print(f"  ✅ 실況자 발화 필터링: {len(transcript.segments)}개 세그먼트")
        except RuntimeError as e:
            print(f"  ⚠️  화자 분리 실패 (계속 진행): {e}")

    # ── Step 5: 감탄사 감지 ────────────────────
    print("\n[4/5] 감탄사 패턴 감지")
    from scoring.exclamation_detector import detect_exclamations
    exclamation = detect_exclamations(transcript)
    print(f"  ✅ 감지 완료: {len(exclamation.hits)}건 "
          f"(긍정 {len(exclamation.positive_hits)}건 / "
          f"컷신호 {len(exclamation.cut_signals)}건)")

    # ── Step 5.5: 감정 분류 (KoELECTRA) ──────────
    emotion_result = None
    if active_modules.get("emotion_analyzer"):
        print("\n[4.5/5] 감정 분류 (KoELECTRA)")
        try:
            from pipeline.emotion_analyzer import analyze_emotions
            emotion_result = analyze_emotions(transcript, progress_cb=print)
        except RuntimeError as e:
            print(f"  ⚠️  감정 분류 실패 (계속 진행): {e}")

    # ── Step 6: 스코어링 (1차) ─────────────────
    print("\n[5/5] 스코어링 + 라벨 분류")
    from scoring.scorer import score_segments
    from scoring.labeler import label_segments, summarize_labels

    scored = score_segments(
        energy=energy,
        exclamation=exclamation,
        transcript=transcript,
        visual_scores=None,
        active_modules=active_modules,
        emotion_result=emotion_result,
    )

    # ── Step 6.5: 시각 보정 (Qwen2-VL) ──────────
    if active_modules.get("visual_analyzer"):
        print("\n  🎬 시각 보정 (Qwen2-VL) 실행 중...")
        try:
            from pipeline.visual_analyzer import analyze_visuals
            candidates = [(s.start, s.end) for s in scored if not s.is_silence]
            visual_result = analyze_visuals(
                video_info=video_info,
                candidate_intervals=candidates,
                progress_cb=print,
            )
            # visual_score 반영 후 재스코어링
            scored = score_segments(
                energy=energy,
                exclamation=exclamation,
                transcript=transcript,
                visual_scores=visual_result.as_score_dict,
                active_modules=active_modules,
                emotion_result=emotion_result,
            )
        except RuntimeError as e:
            print(f"  ⚠️  시각 보정 실패 (계속 진행): {e}")

    labeled = label_segments(scored, user_context=user_context_str)

    print(f"\n{summarize_labels(labeled)}")

    # ── Step 7: XML 출력 ───────────────────────
    print("\n[출력] XML 생성")
    from output.xml_exporter import export_xml

    try:
        out_path = export_xml(
            labeled=labeled,
            video_info=video_info,
        )
        print(f"\n  🎉 완료! Premiere Pro에서 임포트하세요:")
        print(f"     파일 > 가져오기 > {out_path}")
    except Exception as e:
        print(f"  ❌ XML 생성 실패: {e}")
        return 1

    return 0


# ─────────────────────────────────────────────
# CLI 인자 파싱
# ─────────────────────────────────────────────

def _parse_args() -> dict:
    """간단한 CLI 인자 파싱 (argparse 없이)."""
    video_path = None
    use_ui = True

    for arg in sys.argv[1:]:
        if arg == "--no-ui":
            use_ui = False
        elif not arg.startswith("-"):
            video_path = arg

    if video_path is None:
        if use_ui:
            video_path = _ask_video_path()
        else:
            print("오류: 영상 파일 경로를 입력하세요.")
            print("사용법: python main.py <영상파일> [--no-ui]")
            sys.exit(1)

    return {"video_path": video_path, "use_ui": use_ui}


def _ask_video_path() -> str:
    """영상 파일 경로를 직접 입력받는다."""
    print("\n영상 파일 경로를 입력하세요 (드래그 앤 드롭 가능):")
    path = input("  > ").strip().strip('"').strip("'")
    if not path:
        print("경로가 비어 있습니다.")
        sys.exit(1)
    return path


if __name__ == "__main__":
    sys.exit(main())
