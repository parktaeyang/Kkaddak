import json

from kkaddak import config


def test_missing_file_uses_defaults(tmp_path):
    assert config.load(tmp_path / "none.json") == config.DEFAULT_BINDINGS


def test_defaults():
    d = config.DEFAULT_BINDINGS
    assert d["palm"] == {"type": "mission_control"}
    assert d["push_right"] == {"type": "key", "key": "s"}
    assert d["push_left"] == {"type": "key", "key": "a"}


def test_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    bindings = dict(config.DEFAULT_BINDINGS, push_right={"type": "key", "key": "shift+cmd+s"}, palm={"type": "none"})
    config.save(bindings, path)
    assert config.load(path) == bindings


def test_invalid_entries_fall_back(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "version": 1,
        "bindings": {
            "palm": {"type": "explode"},
            "push_left": {"type": "key", "key": "nope"},
            "push_right": {"type": "key", "key": "Shift+D"},
            "unknown_gesture": {"type": "none"},
        },
    }))
    b = config.load(path)
    assert b["palm"] == config.DEFAULT_BINDINGS["palm"]
    assert b["push_left"] == config.DEFAULT_BINDINGS["push_left"]
    assert b["push_right"] == {"type": "key", "key": "shift+d"}
    assert "unknown_gesture" not in b


def test_corrupt_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json")
    assert config.load(path) == config.DEFAULT_BINDINGS


def test_gestures_have_defaults():
    assert set(config.GESTURES) == set(config.DEFAULT_BINDINGS)
