import numpy as np

from kkaddak.engine import HandObs, dedupe_hands
from kkaddak.fingers import classify
from tests.handgen import FIST


def hand(label, score, x0):
    pts = np.zeros((21, 3))
    pts[:, 0] = np.linspace(x0, x0 + 0.2, 21)
    pts[:, 1] = np.linspace(0.3, 0.6, 21)
    return HandObs(label, score, [tuple(p) for p in pts], classify(FIST), 100.0, None)


def test_duplicate_detection_keeps_confident_one():
    kept = dedupe_hands([hand("Left", 0.6, 0.30), hand("Right", 0.9, 0.31)])
    assert [h.label for h in kept] == ["Right"]


def test_two_separate_hands_kept():
    kept = dedupe_hands([hand("Left", 0.9, 0.1), hand("Right", 0.9, 0.6)])
    assert sorted(h.label for h in kept) == ["Left", "Right"]


# ---- 가리키는 동안 키 누르고 있기 ----

import pytest  # noqa: E402

from kkaddak import actions  # noqa: E402
from kkaddak.config import DEFAULT_BINDINGS  # noqa: E402
from kkaddak.engine import POINT_HOLD, POINT_RELEASE_MS, Engine  # noqa: E402


@pytest.fixture
def keylog(monkeypatch):
    log: list[tuple[str, str]] = []
    monkeypatch.setattr(actions, "key_down", lambda c: log.append(("down", c.spec)) or True)
    monkeypatch.setattr(actions, "key_up", lambda c: log.append(("up", c.spec)))
    monkeypatch.setattr(actions, "perform", lambda a, dry_run=False: log.append(("perform", a["type"])))
    return log


def pointing(direction):
    h = hand("Right", 0.9, 0.3)
    return HandObs(h.label, h.score, h.image_pts, h.state, h.size_px, direction)


def run(engine, frames, start=0, step_ms=33):
    """frames: 프레임마다 가리키는 방향(None이면 손 없음). 마지막 시각을 반환."""
    t = start
    for direction in frames:
        hands = [pointing(direction)] if direction else []
        engine._dispatch(engine._update_pointing(t, hands, engine.bindings), engine.bindings)
        t += step_ms
    return t


def frames_for(ms, direction, step_ms=33):
    return [direction] * (ms // step_ms + 2)  # ms 이상 지나도록


def test_point_holds_key_until_pointing_stops(keylog):
    e = Engine()
    t = run(e, frames_for(POINT_HOLD.hold_ms, "left"))
    assert keylog == [("down", "left")]
    t = run(e, frames_for(1000, "left"), start=t)  # 계속 가리키는 동안은 누른 채
    assert keylog == [("down", "left")]
    run(e, frames_for(POINT_RELEASE_MS + 50, None), start=t)
    assert keylog == [("down", "left"), ("up", "left")]


def test_brief_dropout_does_not_release(keylog):
    e = Engine()
    t = run(e, frames_for(POINT_HOLD.hold_ms, "right"))
    t = run(e, [None] * 2, start=t)  # 66ms 인식 끊김
    run(e, frames_for(300, "right"), start=t)
    assert keylog == [("down", "right")]


def test_release_keys_on_close(keylog):
    e = Engine()
    run(e, frames_for(POINT_HOLD.hold_ms, "left"))
    e.close()
    assert keylog == [("down", "left"), ("up", "left")]


def test_binding_change_releases_held_key(keylog):
    e = Engine()
    t = run(e, frames_for(POINT_HOLD.hold_ms, "left"))
    e.set_bindings({**DEFAULT_BINDINGS, "point_left": {"type": "key", "key": "a"}})
    run(e, ["left"], start=t)
    assert keylog == [("down", "left"), ("up", "left")]


def test_non_key_action_on_point_runs_once(keylog):
    e = Engine(bindings={**DEFAULT_BINDINGS, "point_left": {"type": "mission_control"}})
    t = run(e, frames_for(POINT_HOLD.hold_ms, "left"))
    run(e, frames_for(POINT_RELEASE_MS + 50, None), start=t)
    assert keylog == [("perform", "mission_control")]
