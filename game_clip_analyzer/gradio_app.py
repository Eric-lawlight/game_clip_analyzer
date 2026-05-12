"""
gradio_app.py — 브라우저 기반 웹 UI
======================================
기존 파이프라인(main.py)을 그대로 재사용하고,
Gradio로 브라우저 인터페이스만 덧씌운다.

실행:
    python gradio_app.py
    → http://localhost:7860 자동 오픈

특징:
    - 영상 파일 드래그 앤 드롭 업로드
    - 영상 업로드 즉시 메타데이터 표시 + 예상 시간 자동 갱신
    - 모듈 체크박스 토글 → 예상 시간 실시간 업데이트
    - 분석 진행 로그 실시간 스트리밍
    - 완료 시 XML 다운로드 버튼 자동 노출
    - 협방송 ON → 화자 분리 자동 활성화
"""

import sys
import time
import traceback
from pathlib import Path

import gradio as gr

# ─────────────────────────────────────────────
# 커스텀 CSS — 게임 실況 툴 분위기
# ─────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Noto+Sans+KR:wght@400;500;700&display=swap');

:root {
    --bg-deep:    #0d0f14;
    --bg-card:    #151820;
    --bg-input:   #1c2030;
    --border:     #2a3045;
    --border-hi:  #3d4f7a;
    --accent:     #4f8ef7;
    --accent-dim: #1e3a6e;
    --green:      #3ddc84;
    --yellow:     #f5c542;
    --red:        #f45b69;
    --text-main:  #e8ecf5;
    --text-muted: #6b7a99;
    --mono:       'JetBrains Mono', monospace;
    --sans:       'Noto Sans KR', sans-serif;
}

body, .gradio-container {
    background: var(--bg-deep) !important;
    font-family: var(--sans) !important;
    color: var(--text-main) !important;
}

/* 헤더 */
.app-header {
    text-align: center;
    padding: 32px 0 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.app-header h1 {
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-main);
    margin: 0 0 6px;
}
.app-header p {
    color: var(--text-muted);
    font-size: 0.88rem;
    margin: 0;
    font-family: var(--mono);
}

/* 섹션 라벨 */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    font-family: var(--mono);
    margin-bottom: 10px !important;
}

/* 카드 */
.card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 18px !important;
}

/* 비디오 정보 박스 */
.info-box {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    font-family: var(--mono);
    font-size: 0.82rem;
    color: var(--text-main);
    min-height: 60px;
    line-height: 1.7;
}

/* 예상 시간 뱃지 */
.time-badge {
    display: inline-block;
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 20px;
    padding: 3px 14px;
    font-family: var(--mono);
    font-size: 0.82rem;
    font-weight: 600;
}

/* 로그 박스 */
.log-box textarea {
    background: #080a0f !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: #a8c0e8 !important;
    font-family: var(--mono) !important;
    font-size: 0.80rem !important;
    line-height: 1.6 !important;
}

/* 버튼 */
.btn-primary {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 12px 0 !important;
    transition: opacity 0.15s !important;
}
.btn-primary:hover { opacity: 0.85 !important; }
.btn-primary:disabled { opacity: 0.4 !important; cursor: not-allowed !important; }

.btn-download {
    background: var(--green) !important;
    color: #0d1a10 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

/* Gradio 기본 요소 오버라이드 */
.gr-input, input[type=text], textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    border-radius: 8px !important;
}
.gr-input:focus, input:focus, textarea:focus {
    border-color: var(--border-hi) !important;
    outline: none !important;
}
label { color: var(--text-muted) !important; font-size: 0.82rem !important; }
.gr-checkbox label { color: var(--text-main) !important; font-size: 0.88rem !important; }

/* 업로드 영역 */
.gr-file-upload {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-hi) !important;
    border-radius: 10px !important;
}

/* 상태 색상 */
.status-ok   { color: var(--green)  !important; }
.status-warn { color: var(--yellow) !important; }
.status-err  { color: var(--red)    !important; }

/* 구분선 */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }
"""

# ─────────────────────────────────────────────
# 상태 변수 (세션 간 공유 금지 — Gradio State 사용)
# ─────────────────────────────────────────────

def _empty_state() -> dict:
    return {
        "video_info": None,
        "active_modules": None,
        "output_xml_path": None,
    }


# ─────────────────────────────────────────────
# 이벤트 핸들러
# ─────────────────────────────────────────────

def on_video_upload(video_path: str, state: dict):
    """
    영상 업로드 시 호출.
    VideoInfo 파악 → 메타데이터 + 예상 시간 표시.
    """
    if not video_path:
        return state, "영상을 업로드하면 정보가 여기 표시됩니다.", "—"

    try:
        from core.video_inspector import inspect_video
        from core.time_estimator import estimate_time
        from config import DEFAULT_MODULES

        info = inspect_video(video_path)
        state["video_info"] = info

        est = estimate_time(info, DEFAULT_MODULES)

        # 경고 있으면 표시
        warn_lines = "\n".join(info.warnings) if info.warnings else ""
        warn_block = f"\n{warn_lines}" if warn_lines else ""

        info_text = (
            f"📁  {info.path.name}\n"
            f"⏱️  {info.duration_str}\n"
            f"🎞️  {info.fps_int} fps  /  {info.codec.upper()}  /  "
            f"{info.width}×{info.height}\n"
            f"💾  {info.file_size_mb:.1f} MB"
            f"{warn_block}"
        )
        time_text = est.total_str

        return state, info_text, time_text

    except Exception as e:
        return state, f"❌ 오류: {e}", "—"


def on_module_change(
    state: dict,
    use_energy: bool,
    use_emotion: bool,
    use_visual: bool,
    use_speaker: bool,
):
    """모듈 체크박스 변경 시 예상 시간 재계산."""
    info = state.get("video_info")
    if info is None:
        return "영상을 먼저 업로드하세요"

    from core.time_estimator import estimate_time

    modules = {
        "ffmpeg_extract":   True,
        "whisper":          True,
        "energy_detector":  use_energy,
        "emotion_analyzer": use_emotion,
        "visual_analyzer":  use_visual,
        "speaker_separator":use_speaker,
        "xml_export":       True,
    }
    est = estimate_time(info, modules)
    return est.total_str


def on_collab_change(is_collab: bool, use_speaker: bool):
    """협방송 ON → 화자 분리 자동 체크."""
    if is_collab:
        return gr.update(value=True)
    return gr.update(value=use_speaker)


def run_analysis(
    state: dict,
    game_name: str,
    highlight_criteria: str,
    cut_criteria: str,
    is_collab: bool,
    use_energy: bool,
    use_emotion: bool,
    use_visual: bool,
    use_speaker: bool,
):
    """
    분석 파이프라인을 실행한다.
    generator 함수 → Gradio가 yield 마다 로그 박스를 갱신한다.

    Yields:
        (log_str, download_component_update, state)
    """
    info = state.get("video_info")
    if info is None:
        yield "❌ 영상을 먼저 업로드하세요.", gr.update(visible=False), state
        return

    active_modules = {
        "ffmpeg_extract":   True,
        "whisper":          True,
        "energy_detector":  use_energy,
        "emotion_analyzer": use_emotion,
        "visual_analyzer":  use_visual,
        "speaker_separator":use_speaker and is_collab,
        "xml_export":       True,
    }
    state["active_modules"] = active_modules
    user_context = game_name.strip()

    logs: list[str] = []

    def log(msg: str):
        logs.append(msg)

    def current_log() -> str:
        return "\n".join(logs)

    try:
        # ── 1. 오디오 추출 ──────────────────────
        log("━━━ [1/6] 오디오 추출 ━━━")
        yield current_log(), gr.update(visible=False), state

        from pipeline.audio_extractor import extract_audio
        audio_info = extract_audio(info)
        log(f"  ✅ 완료: {audio_info.path.name}  ({audio_info.file_size_mb:.1f} MB)")
        yield current_log(), gr.update(visible=False), state

        # ── 2. Whisper 전사 ──────────────────────
        log("\n━━━ [2/6] 음성 전사 (Whisper) ━━━")
        yield current_log(), gr.update(visible=False), state

        from pipeline.whisper_transcriber import transcribe

        def whisper_log(msg):
            log(msg)

        transcript = transcribe(audio_info, progress_cb=whisper_log)
        log(f"  ✅ 완료: {transcript.segment_count}개 세그먼트  /  언어: {transcript.language}")
        yield current_log(), gr.update(visible=False), state

        # ── 3. 화자 분리 (선택) ──────────────────
        if active_modules.get("speaker_separator"):
            log("\n━━━ [3/6] 화자 분리 (pyannote) ━━━")
            yield current_log(), gr.update(visible=False), state
            try:
                from pipeline.speaker_separator import separate_speakers

                def spk_log(msg):
                    log(msg)

                speaker_result = separate_speakers(audio_info, progress_cb=spk_log)
                transcript = speaker_result.filter_transcript(transcript)
                log(f"  ✅ 실況자 발화 필터링: {transcript.segment_count}개 세그먼트")
            except RuntimeError as e:
                log(f"  ⚠️  화자 분리 실패 (계속 진행): {e}")
            yield current_log(), gr.update(visible=False), state

        # ── 4. 에너지 분석 ──────────────────────
        log("\n━━━ [4/6] 오디오 에너지 분석 ━━━")
        yield current_log(), gr.update(visible=False), state

        from pipeline.energy_detector import detect_energy

        def energy_log(msg):
            log(msg)

        energy = detect_energy(audio_info, progress_cb=energy_log)
        log(f"  ✅ 완료: 피크 {len(energy.peak_segments)}개 / 침묵 {len(energy.silence_segments)}개")
        yield current_log(), gr.update(visible=False), state

        # ── 5. 감탄사 감지 ──────────────────────
        log("\n━━━ [5/6] 스코어링 ━━━")
        yield current_log(), gr.update(visible=False), state

        from scoring.exclamation_detector import detect_exclamations
        exclamation = detect_exclamations(transcript)
        log(f"  감탄사: {len(exclamation.hits)}건 감지")
        yield current_log(), gr.update(visible=False), state

        # ── 5.5. 감정 분류 (선택) ───────────────
        emotion_result = None
        if active_modules.get("emotion_analyzer"):
            log("  KoELECTRA 감정 분류 중...")
            yield current_log(), gr.update(visible=False), state
            try:
                from pipeline.emotion_analyzer import analyze_emotions

                def emo_log(msg):
                    log(msg)

                emotion_result = analyze_emotions(transcript, progress_cb=emo_log)
                log(f"  ✅ 감정 분류 완료")
            except RuntimeError as e:
                log(f"  ⚠️  감정 분류 실패 (계속 진행): {e}")
            yield current_log(), gr.update(visible=False), state

        # ── 6. 스코어링 ─────────────────────────
        from scoring.scorer import score_segments
        scored = score_segments(
            energy=energy,
            exclamation=exclamation,
            transcript=transcript,
            visual_scores=None,
            active_modules=active_modules,
            emotion_result=emotion_result,
        )

        # ── 6.5. 시각 보정 (선택) ───────────────
        if active_modules.get("visual_analyzer"):
            log("  Qwen2-VL 시각 보정 중...")
            yield current_log(), gr.update(visible=False), state
            try:
                from pipeline.visual_analyzer import analyze_visuals

                def vis_log(msg):
                    log(msg)

                candidates = [(s.start, s.end) for s in scored if not s.is_silence]
                visual_result = analyze_visuals(info, candidates, progress_cb=vis_log)
                scored = score_segments(
                    energy=energy,
                    exclamation=exclamation,
                    transcript=transcript,
                    visual_scores=visual_result.as_score_dict,
                    active_modules=active_modules,
                    emotion_result=emotion_result,
                )
                log(f"  ✅ 시각 보정 완료")
            except RuntimeError as e:
                log(f"  ⚠️  시각 보정 실패 (계속 진행): {e}")
            yield current_log(), gr.update(visible=False), state

        # 라벨 분류
        from scoring.labeler import label_segments, summarize_labels
        labeled = label_segments(scored, user_context=user_context)

        # 라벨 분포 출력
        log(f"\n{summarize_labels(labeled)}")
        yield current_log(), gr.update(visible=False), state

        # ── 7. XML 출력 ─────────────────────────
        log("\n━━━ [6/6] XML 생성 ━━━")
        yield current_log(), gr.update(visible=False), state

        from output.xml_exporter import export_xml
        xml_path = export_xml(labeled, info)
        state["output_xml_path"] = str(xml_path)

        log(f"  ✅ 저장 완료: {xml_path.name}")
        log(f"\n{'━'*40}")
        log(f"🎉 분석 완료!")
        log(f"   Premiere Pro에서: 파일 > 가져오기 > {xml_path.name}")

        yield current_log(), gr.update(visible=True, value=str(xml_path)), state

    except Exception as e:
        tb = traceback.format_exc()
        log(f"\n❌ 오류 발생:\n{e}\n\n{tb}")
        yield current_log(), gr.update(visible=False), state


# ─────────────────────────────────────────────
# UI 레이아웃
# ─────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="게임 실況 편집점 분석기") as app:

        state = gr.State(_empty_state())

        # ── 헤더 ──────────────────────────────
        gr.HTML("""
        <div class="app-header">
            <h1>🎮 게임 실황 편집점 분석기</h1>
            <p>로컬 AI로 영상을 분석해 Premiere Pro 마커 XML을 자동 생성합니다</p>
        </div>
        """)

        with gr.Row(equal_height=False):

            # ── 왼쪽 패널: 설정 ───────────────
            with gr.Column(scale=4, min_width=340):

                # 영상 업로드
                gr.HTML('<p class="section-label">📁 영상 파일</p>')
                video_upload = gr.File(
                    label="영상 파일 (드래그 앤 드롭)",
                    file_types=[".mp4", ".mov", ".mkv", ".avi", ".webm", ".ts"],
                    elem_classes=["card"],
                )
                video_info_box = gr.HTML(
                    '<div class="info-box">영상을 업로드하면 정보가 여기 표시됩니다.</div>'
                )

                gr.HTML("<hr>")

                # 게임 정보
                gr.HTML('<p class="section-label">🎲 게임 정보 (선택 — 정확도 향상)</p>')
                with gr.Group(elem_classes=["card"]):
                    game_name = gr.Textbox(
                        label="게임명",
                        placeholder="예: 다크소울 3, 엘든링, 발로란트",
                        max_lines=1,
                    )
                    highlight_criteria = gr.Textbox(
                        label="하이라이트로 표시할 것",
                        placeholder="예: 보스 처치, 첫 클리어, 화려한 플레이, 웃긴 실수",
                        max_lines=2,
                    )
                    cut_criteria = gr.Textbox(
                        label="컷 권장 구간",
                        placeholder="예: 로딩화면, 5분 이상 침묵, 인벤토리 정리",
                        max_lines=2,
                    )
                    is_collab = gr.Checkbox(
                        label="👥 협방송 (화자 분리 자동 활성화)",
                        value=False,
                    )

                gr.HTML("<hr>")

                # 모듈 선택
                gr.HTML('<p class="section-label">⚙️ 분석 모듈</p>')
                with gr.Group(elem_classes=["card"]):
                    use_energy = gr.Checkbox(
                        label="📊 에너지 분석 (librosa)  — 권장",
                        value=True,
                    )
                    use_emotion = gr.Checkbox(
                        label="🧠 감정 분류 (KoELECTRA)  — 권장",
                        value=True,
                    )
                    use_visual = gr.Checkbox(
                        label="🎬 시각 보정 (Qwen2-VL)   — 권장 / VRAM 3.5GB",
                        value=True,
                    )
                    use_speaker = gr.Checkbox(
                        label="🎙️ 화자 분리 (pyannote)   — 협방송 시",
                        value=False,
                    )

                    gr.HTML('<div style="margin-top:12px; padding-top:12px; border-top:1px solid var(--border)">')
                    time_estimate = gr.HTML(
                        '<div>예상 시간: <span class="time-badge">—</span></div>'
                    )
                    gr.HTML('</div>')

                gr.HTML("<hr>")

                # 실행 버튼
                run_btn = gr.Button(
                    "🚀  분석 시작",
                    variant="primary",
                    elem_classes=["btn-primary"],
                )

            # ── 오른쪽 패널: 진행 상황 ────────
            with gr.Column(scale=6, min_width=420):
                gr.HTML('<p class="section-label">📋 분석 로그</p>')
                log_box = gr.Textbox(
                    label="",
                    lines=28,
                    max_lines=60,
                    interactive=False,
                    placeholder="분석 시작 버튼을 누르면 진행 상황이 여기 표시됩니다.",
                    elem_classes=["log-box"],
                )

                download_btn = gr.File(
                    label="⬇️  XML 다운로드 (Premiere Pro용)",
                    visible=False,
                    elem_classes=["btn-download"],
                )

        # ─────────────────────────────────────
        # 이벤트 바인딩
        # ─────────────────────────────────────

        # 영상 업로드 → 정보 표시 + 예상 시간
        def _on_upload(video, state):
            if video is None:
                return state, '<div class="info-box">영상을 업로드하면 정보가 여기 표시됩니다.</div>', '예상 시간: <span class="time-badge">—</span>'
            new_state, info_text, time_text = on_video_upload(video, state)
            info_html = f'<div class="info-box">{info_text.replace(chr(10), "<br>")}</div>'
            time_html = f'예상 시간: <span class="time-badge">{time_text}</span>'
            return new_state, info_html, time_html

        video_upload.change(
            fn=_on_upload,
            inputs=[video_upload, state],
            outputs=[state, video_info_box, time_estimate],
        )

        # 모듈 토글 → 예상 시간 업데이트
        def _on_module_change(state, e, emo, vis, spk):
            t = on_module_change(state, e, emo, vis, spk)
            return f'예상 시간: <span class="time-badge">{t}</span>'

        for chk in [use_energy, use_emotion, use_visual, use_speaker]:
            chk.change(
                fn=_on_module_change,
                inputs=[state, use_energy, use_emotion, use_visual, use_speaker],
                outputs=[time_estimate],
            )

        # 협방송 ON → 화자 분리 자동 체크
        is_collab.change(
            fn=on_collab_change,
            inputs=[is_collab, use_speaker],
            outputs=[use_speaker],
        )

        # 분석 시작 (generator → 스트리밍 업데이트)
        run_btn.click(
            fn=run_analysis,
            inputs=[
                state,
                game_name, highlight_criteria, cut_criteria,
                is_collab,
                use_energy, use_emotion, use_visual, use_speaker,
            ],
            outputs=[log_box, download_btn, state],
        )

    return app


# ─────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",   # 로컬 네트워크 내 다른 기기에서도 접속 가능
        server_port=7860,
        inbrowser=True,          # 자동으로 브라우저 탭 열기
        show_error=True,
    )
