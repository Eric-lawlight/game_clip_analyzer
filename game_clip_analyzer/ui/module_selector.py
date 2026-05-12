"""
module_selector.py — 모듈 선택 UI
=====================================
영상 분석 전에 사용자가 각 모듈을 ON/OFF 하고
전체 예상 처리 시간을 확인하는 터미널 UI.

설계 원칙:
- PyQt/tkinter 없이 표준 라이브러리만 사용 (Sprint 1 단계)
- Sprint 4에서 GUI로 교체할 때 이 파일만 수정하면 된다
- 반환값(active_modules, context)은 GUI 버전과 동일한 구조 유지

반환 데이터:
    SelectionResult (dataclass)
        .active_modules : {모듈명: bool} — 사용자가 설정한 ON/OFF
        .context        : UserContext — 게임명, 컨텍스트, 협방송 여부
        .confirmed      : 사용자가 '분석 시작'을 선택했는지 여부

사용 예:
    from core.video_inspector import inspect_video
    from core.time_estimator import estimate_time
    from ui.module_selector import run_module_selector

    info = inspect_video("game_vod.mp4")
    result = run_module_selector(info)

    if result.confirmed:
        est = estimate_time(info, result.active_modules)
        # → 파이프라인 실행
"""

import copy
from dataclasses import dataclass, field

from config import DEFAULT_MODULES, FIXED_MODULES
from core.video_inspector import VideoInfo
from core.time_estimator import estimate_time, MODULE_LABELS


# ─────────────────────────────────────────────
# 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class UserContext:
    """사용자가 입력한 게임/편집 컨텍스트."""

    game_name: str = ""
    highlight_criteria: str = ""    # 하이라이트로 표시할 것
    cut_criteria: str = ""          # 컷 권장 구간
    has_collab: bool = False        # 협방송 여부


@dataclass
class SelectionResult:
    """module_selector 최종 반환값."""

    active_modules: dict[str, bool]
    context: UserContext
    confirmed: bool = False


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

def run_module_selector(video_info: VideoInfo) -> SelectionResult:
    """
    터미널 대화형 UI를 실행하고 SelectionResult를 반환한다.

    Args:
        video_info: inspect_video() 반환값

    Returns:
        SelectionResult — confirmed=False면 사용자가 취소한 것
    """
    active_modules = copy.deepcopy(DEFAULT_MODULES)
    context = UserContext()

    _clear()
    _print_header()

    # ── 1. 영상 정보 표시 ──────────────────────
    _print_section("📁 영상 파일")
    print(f"  {video_info.path.name}")
    print(f"  → {video_info.summary()}")
    if video_info.warnings:
        for w in video_info.warnings:
            print(f"  {w}")

    # ── 2. 게임 컨텍스트 입력 ─────────────────
    _print_section("🎲 게임 정보  (선택 입력 — 정확도 향상에 도움)")
    context.game_name = _ask(
        "게임명", placeholder="다크소울 3"
    )
    context.highlight_criteria = _ask(
        "하이라이트로 표시할 것",
        placeholder="보스 처치, 첫 클리어, 화려한 플레이, 웃긴 실수",
    )
    context.cut_criteria = _ask(
        "컷 권장 구간",
        placeholder="로딩화면, 5분 이상 침묵, 인벤토리 정리",
    )

    collab_input = _ask("협방송 여부 (y = 있음)", placeholder="n")
    context.has_collab = collab_input.strip().lower() in ("y", "yes", "예", "있음")

    if context.has_collab:
        active_modules["speaker_separator"] = True
        print("  ✅ 화자 분리 (pyannote) 자동 활성화")

    # ── 3. 모듈 ON/OFF ────────────────────────
    _print_section("⚙️  분석 모듈 선택")
    active_modules = _module_toggle_ui(active_modules, video_info)

    # ── 4. 최종 확인 ──────────────────────────
    est = estimate_time(video_info, active_modules)
    _print_section(f"🚀 분석 시작  (예상 총 소요: {est.total_str})")

    confirm = _ask("분석을 시작할까요? (y/n)", placeholder="y")
    confirmed = confirm.strip().lower() in ("y", "yes", "예", "")

    if confirmed:
        print("\n  분석을 시작합니다...\n")
    else:
        print("\n  취소했습니다.\n")

    return SelectionResult(
        active_modules=active_modules,
        context=context,
        confirmed=confirmed,
    )


# ─────────────────────────────────────────────
# 모듈 토글 UI
# ─────────────────────────────────────────────

# ON/OFF 가능한 모듈 목록 (필수 제외)
_TOGGLEABLE_MODULES = [
    m for m in DEFAULT_MODULES if m not in FIXED_MODULES
]

_MODULE_DESCRIPTIONS = {
    "energy_detector":  "librosa — 오디오 에너지/침묵/피치 분석",
    "emotion_analyzer": "KoELECTRA — 한국어 발화 감정 분류",
    "visual_analyzer":  "Qwen2-VL 2B — 후보 구간 장면 이해",
    "speaker_separator":"pyannote — 협방송 화자 분리",
}


def _module_toggle_ui(
    active_modules: dict[str, bool],
    video_info: VideoInfo,
) -> dict[str, bool]:
    """
    토글 가능한 모듈 목록을 표시하고 사용자 입력을 받는다.

    번호를 입력하면 ON/OFF 토글.
    빈 엔터를 누르면 현재 상태 그대로 확정.
    """
    while True:
        est = estimate_time(video_info, active_modules)
        _print_module_table(active_modules, est)

        print(
            "\n  번호를 입력해 ON/OFF 전환  |  빈 엔터 = 확정\n"
            "  예) 1 → energy_detector 토글"
        )
        raw = input("  > ").strip()

        if raw == "":
            break

        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(_TOGGLEABLE_MODULES):
                module = _TOGGLEABLE_MODULES[idx]
                active_modules[module] = not active_modules[module]
                status = "ON ✅" if active_modules[module] else "OFF ⬜"
                print(f"\n  → {MODULE_LABELS[module]} {status}\n")
            else:
                print(f"\n  ⚠️  {raw}번 항목이 없습니다.\n")
        else:
            print("\n  ⚠️  숫자를 입력하거나 엔터를 누르세요.\n")

    return active_modules


def _print_module_table(
    active_modules: dict[str, bool],
    est,
) -> None:
    """모듈 목록 테이블 출력."""
    print()
    # 필수 모듈 표시
    for module in FIXED_MODULES:
        if module not in MODULE_LABELS:
            continue
        time_str = est.module_str(module)
        label = MODULE_LABELS[module]
        print(f"  [✅] (필수) {label:<30} {time_str}")

    print()

    # 토글 가능 모듈
    for i, module in enumerate(_TOGGLEABLE_MODULES, start=1):
        state = "✅" if active_modules.get(module) else "⬜"
        label = MODULE_LABELS.get(module, module)
        desc = _MODULE_DESCRIPTIONS.get(module, "")
        time_str = est.module_str(module) if active_modules.get(module) else "—"
        print(f"  [{state}] {i}. {label:<30} {time_str}")
        if desc:
            print(f"            {desc}")

    print()
    print(f"  {'─'*48}")
    print(f"  예상 총 소요 시간: {est.total_str}")


# ─────────────────────────────────────────────
# UI 헬퍼
# ─────────────────────────────────────────────

def _clear() -> None:
    """터미널 화면을 지운다 (가능한 경우)."""
    import os
    os.system("cls" if os.name == "nt" else "clear")


def _print_header() -> None:
    width = 52
    print("┌" + "─" * width + "┐")
    print(f"│{'🎮 게임 실황 편집점 분석기 v0.1':^{width}}│")
    print("└" + "─" * width + "┘")
    print()


def _print_section(title: str) -> None:
    print(f"\n  {title}")
    print(f"  {'─' * 46}")


def _ask(label: str, placeholder: str = "") -> str:
    """한 줄 입력 받기. 빈 입력이면 placeholder 반환."""
    hint = f"  (예: {placeholder})" if placeholder else ""
    raw = input(f"  {label}{hint}: ").strip()
    return raw if raw else placeholder


# ─────────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from core.video_inspector import inspect_video

    if len(sys.argv) < 2:
        print("사용법: python -m ui.module_selector <영상파일경로>")
        sys.exit(1)

    try:
        info = inspect_video(sys.argv[1])
        result = run_module_selector(info)

        print("\n─── 선택 결과 ───")
        print(f"confirmed      : {result.confirmed}")
        print(f"game_name      : {result.context.game_name}")
        print(f"has_collab     : {result.context.has_collab}")
        print(f"active_modules : {result.active_modules}")

    except Exception as e:
        print(f"\n❌ 오류: {e}")
        sys.exit(1)
