from kkaddak.gesture import HoldConfig, HoldTrigger, Phase

CFG = HoldConfig(hold_ms=500, dropout_ms=150, release_ms=400, cooldown_ms=1500)
FRAME = 33


def run(trigger: HoldTrigger, start: int, end: int, active: bool) -> list[int]:
    """start~end 구간을 30fps로 흘리고 발동한 시각 목록을 반환."""
    fired = []
    for t in range(start, end, FRAME):
        if trigger.update(t, active):
            fired.append(t)
    return fired


def test_fires_once_after_hold():
    tr = HoldTrigger(CFG)
    fired = run(tr, 0, 3000, True)
    assert len(fired) == 1
    assert 500 <= fired[0] < 500 + FRAME
    assert tr.phase is Phase.LATCHED


def test_short_hold_does_not_fire():
    tr = HoldTrigger(CFG)
    assert run(tr, 0, 400, True) == []
    assert run(tr, 400, 1000, False) == []
    assert tr.phase is Phase.ARMED


def test_brief_dropout_tolerated():
    tr = HoldTrigger(CFG)
    run(tr, 0, 300, True)
    run(tr, 300, 400, False)  # 100ms 끊김 < dropout_ms
    assert len(run(tr, 400, 800, True)) == 1


def test_long_dropout_resets_hold():
    tr = HoldTrigger(CFG)
    run(tr, 0, 400, True)
    run(tr, 400, 700, False)  # 300ms 끊김 > dropout_ms
    assert tr.phase is Phase.ARMED
    assert run(tr, 700, 1100, True) == []  # 처음부터 다시 500ms 필요


def test_rearm_requires_release_and_cooldown():
    tr = HoldTrigger(CFG)
    fired_at = run(tr, 0, 1000, True)[0]
    run(tr, 1000, 1500, False)  # 자세는 풀었지만 쿨다운(1500ms) 전
    assert tr.phase is Phase.LATCHED
    run(tr, 1500, 2200, False)
    assert tr.phase is Phase.ARMED
    second = run(tr, 2200, 3000, True)
    assert len(second) == 1 and second[0] - fired_at >= CFG.cooldown_ms


def test_flicker_while_latched_does_not_rearm():
    # 손바닥을 유지하는 중 인식이 잠깐씩 끊겨도 재발동하지 않음
    tr = HoldTrigger(CFG)
    fired = run(tr, 0, 600, True)
    for t in range(600, 5000, 300):
        fired += run(tr, t, t + 200, True)
        fired += run(tr, t + 200, t + 300, False)
    assert len(fired) == 1


def test_reset_keeps_latch():
    tr = HoldTrigger(CFG)
    run(tr, 0, 600, True)
    tr.reset()
    assert tr.phase is Phase.LATCHED
