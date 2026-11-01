import pytest

from kkaddak.keys import KeySpecError, parse


@pytest.mark.parametrize(
    "spec, key, mods, label",
    [
        ("s", "s", set(), "S"),
        ("S", "s", set(), "S"),
        ("shift+s", "s", {"shift"}, "⇧S"),
        ("Command + Shift + 4", "4", {"cmd", "shift"}, "⇧⌘4"),
        ("ctrl+up", "up", {"ctrl"}, "⌃↑"),
        ("opt+esc", "escape", {"alt"}, "⌥Esc"),
        ("cmd+space", "space", {"cmd"}, "⌘Space"),
        ("f5", "f5", set(), "F5"),
    ],
)
def test_parse(spec, key, mods, label):
    combo = parse(spec)
    assert combo.key == key
    assert combo.modifiers == frozenset(mods)
    assert combo.label == label
    assert parse(combo.spec) == combo  # 저장 형식으로 왕복 가능


def test_spec_is_normalized():
    assert parse("CMD + Shift + S").spec == "shift+cmd+s"


@pytest.mark.parametrize("spec", ["", "cmd+", "ㄴ", "hyper+s", "cmd+shift"])
def test_invalid(spec):
    with pytest.raises(KeySpecError):
        parse(spec)
