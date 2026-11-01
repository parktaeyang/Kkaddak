"""제스처 목록과 제스처→동작 매핑 설정.

새 제스처를 추가하려면 GESTURES에 등록하고 engine.py에서 해당 id로 이벤트를 내보낸다.
설정은 ~/Library/Application Support/kkaddak/config.json 에 저장된다.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from kkaddak import actions

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
CONFIG_PATH = Path.home() / "Library" / "Application Support" / "kkaddak" / "config.json"

# id → 메뉴에 보여줄 이름 (메뉴 표시 순서)
GESTURES: dict[str, str] = {
    "palm": "손바닥 들기",
    "push_right": "오른손 주먹 밀기",
    "push_left": "왼손 주먹 밀기",
    "point_left": "왼쪽 가리키기",
    "point_right": "오른쪽 가리키기",
}

DEFAULT_BINDINGS: dict[str, actions.Action] = {
    "palm": {"type": "mission_control"},
    "push_right": {"type": "key", "key": "s"},
    "push_left": {"type": "key", "key": "a"},
    "point_left": {"type": "key", "key": "left"},
    "point_right": {"type": "key", "key": "right"},
}

Bindings = dict[str, actions.Action]


def load(path: Path = CONFIG_PATH) -> Bindings:
    """설정을 읽어 기본값과 합친다. 파일이 없거나 항목이 잘못되면 해당 항목만 기본값을 쓴다."""
    bindings = {g: dict(a) for g, a in DEFAULT_BINDINGS.items()}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return bindings
    except (OSError, json.JSONDecodeError) as e:
        log.warning("설정 파일을 읽지 못해 기본값을 사용합니다: %s", e)
        return bindings

    for gesture, action in (data.get("bindings") or {}).items():
        if gesture not in GESTURES:
            log.warning("알 수 없는 제스처 설정을 무시합니다: %s", gesture)
            continue
        try:
            bindings[gesture] = actions.validate(action)
        except (ValueError, AttributeError) as e:
            log.warning("잘못된 설정(%s)을 기본값으로 대체합니다: %s", gesture, e)
    return bindings


def save(bindings: Bindings, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"version": SCHEMA_VERSION, "bindings": bindings}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)  # 쓰는 도중 종료돼도 기존 파일이 깨지지 않게
