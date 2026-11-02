"""키 조합 문자열 파싱. 예: "s", "shift+s", "cmd+space", "ctrl+up", "f5"

물리 키(가상 키코드, ANSI 배열)를 누르는 방식이라 게임·단축키에서도 동작한다.
대신 한글 입력 상태에서 "s"를 누르면 실제 키보드처럼 "ㄴ"이 입력된다.
"""

from __future__ import annotations

from dataclasses import dataclass

KEYCODES: dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26,
    "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35,
    "return": 36, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43,
    "/": 44, "n": 45, "m": 46, ".": 47, "tab": 48, "space": 49, "`": 50,
    "delete": 51, "escape": 53,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100,
    "f9": 101, "f10": 109, "f11": 103, "f12": 111,
    "home": 115, "pageup": 116, "forwarddelete": 117, "end": 119, "pagedown": 121,
    "left": 123, "right": 124, "down": 125, "up": 126,
}  # fmt: skip

KEY_ALIASES = {"esc": "escape", "enter": "return", "backspace": "delete", "del": "forwarddelete"}

# 표시 순서는 macOS 관례(⌃⌥⇧⌘)를 따른다
MODIFIERS = ("ctrl", "alt", "shift", "cmd")
MODIFIER_ALIASES = {
    "control": "ctrl", "⌃": "ctrl",
    "option": "alt", "opt": "alt", "⌥": "alt",
    "⇧": "shift",
    "command": "cmd", "⌘": "cmd",
}  # fmt: skip
MODIFIER_SYMBOLS = {"ctrl": "⌃", "alt": "⌥", "shift": "⇧", "cmd": "⌘"}
KEY_SYMBOLS = {
    "return": "↩", "tab": "⇥", "space": "Space", "delete": "⌫", "escape": "Esc", "forwarddelete": "⌦",
    "left": "←", "right": "→", "up": "↑", "down": "↓", "pageup": "PageUp", "pagedown": "PageDown",
}  # fmt: skip


class KeySpecError(ValueError):
    pass


@dataclass(frozen=True)
class KeyCombo:
    key: str
    modifiers: frozenset[str]

    @property
    def keycode(self) -> int:
        return KEYCODES[self.key]

    @property
    def spec(self) -> str:
        """설정 파일에 저장하는 정규화된 문자열."""
        return "+".join([m for m in MODIFIERS if m in self.modifiers] + [self.key])

    @property
    def label(self) -> str:
        """메뉴에 보여줄 문자열. 예: ⇧⌘S"""
        mods = "".join(MODIFIER_SYMBOLS[m] for m in MODIFIERS if m in self.modifiers)
        return mods + KEY_SYMBOLS.get(self.key, self.key.upper())


def parse(spec: str) -> KeyCombo:
    parts = [p.strip().lower() for p in spec.replace(" ", "").split("+")]
    if not parts or not parts[-1]:
        raise KeySpecError(f"키가 비어 있습니다: {spec!r}")
    *mod_parts, key = parts
    key = KEY_ALIASES.get(key, key)
    if key not in KEYCODES:
        raise KeySpecError(f"지원하지 않는 키입니다: {key!r}")
    mods = set()
    for m in mod_parts:
        m = MODIFIER_ALIASES.get(m, m)
        if m not in MODIFIERS:
            raise KeySpecError(f"지원하지 않는 보조키입니다: {m!r}")
        mods.add(m)
    return KeyCombo(key, frozenset(mods))
