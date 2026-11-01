"""macOS 동작 실행.

동작은 {"type": ..., 추가 인자} 형태의 dict로 설정 파일에 저장된다.
새 동작을 추가하려면 ACTION_TYPES에 이름을 등록하고 perform()/describe()에 분기를 추가한다.
"""

from __future__ import annotations

import logging
import subprocess

from kkaddak import keys

log = logging.getLogger(__name__)

ACTION_TYPES: dict[str, str] = {
    "mission_control": "Mission Control",
    "key": "키 입력",
    "none": "동작 없음",
}

Action = dict  # {"type": "key", "key": "cmd+s"} 등


def validate(action: Action) -> Action:
    """설정 파일에서 읽은 동작을 검증하고 정규화. 잘못되면 ValueError."""
    t = action.get("type")
    if t not in ACTION_TYPES:
        raise ValueError(f"알 수 없는 동작 종류: {t!r}")
    if t == "key":
        return {"type": "key", "key": keys.parse(str(action.get("key", ""))).spec}
    return {"type": t}


def describe(action: Action) -> str:
    if action["type"] == "key":
        return f"키 입력 {keys.parse(action['key']).label}"
    return ACTION_TYPES[action["type"]]


def perform(action: Action, dry_run: bool = False) -> None:
    t = action["type"]
    if t == "none":
        return
    if dry_run:
        log.info("[dry-run] %s", describe(action))
        return
    if t == "mission_control":
        mission_control()
    elif t == "key":
        press(keys.parse(action["key"]))


def mission_control() -> None:
    # 단축키(Ctrl+↑) 대신 시스템 앱을 실행해 사용자 단축키 설정·접근성 권한에 의존하지 않는다.
    # Popen으로 띄워 카메라 루프를 막지 않는다
    subprocess.Popen(["open", "-a", "Mission Control"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log.info("Mission Control 실행")


# ---- 키 입력 (손쉬운 사용 권한 필요) ----

_MODIFIER_FLAGS: dict[str, int] | None = None
blocked_key_presses = 0  # 권한이 없어 보내지 못한 키 입력 횟수 (UI가 안내 여부 판단에 사용)


def has_key_permission() -> bool:
    import Quartz

    return bool(Quartz.CGPreflightPostEventAccess())


def request_key_permission() -> bool:
    """권한이 없으면 시스템 허용 요청 대화상자를 띄운다. 현재 허용 여부를 반환."""
    import Quartz

    return bool(Quartz.CGRequestPostEventAccess())


def press(combo: keys.KeyCombo) -> None:
    import Quartz

    global _MODIFIER_FLAGS, blocked_key_presses
    if _MODIFIER_FLAGS is None:
        _MODIFIER_FLAGS = {
            "ctrl": Quartz.kCGEventFlagMaskControl,
            "alt": Quartz.kCGEventFlagMaskAlternate,
            "shift": Quartz.kCGEventFlagMaskShift,
            "cmd": Quartz.kCGEventFlagMaskCommand,
        }
    if not has_key_permission():
        blocked_key_presses += 1
        log.warning("손쉬운 사용 권한이 없어 키 입력을 보낼 수 없습니다: %s", combo.label)
        return

    flags = 0
    for m in combo.modifiers:
        flags |= _MODIFIER_FLAGS[m]
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    for down in (True, False):
        ev = Quartz.CGEventCreateKeyboardEvent(src, combo.keycode, down)
        Quartz.CGEventSetFlags(ev, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)
    log.info("키 입력 %s", combo.label)
