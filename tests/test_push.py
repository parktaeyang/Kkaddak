from kkaddak.gesture import PushConfig, PushDetector

CFG = PushConfig(window_ms=400, grow_ratio=1.3, min_span_ms=80, dropout_ms=150, cooldown_ms=700, retract_ratio=0.85)
FRAME = 33


def feed(det, start, sizes):
    """30fps로 크기 시퀀스를 넣고 (발동 횟수, 다음 시각)을 반환."""
    fired, t = 0, start
    for s in sizes:
        fired += det.update(t, s)
        t += FRAME
    return fired, t


def ramp(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def test_push_fires_once():
    det = PushDetector(CFG)
    fired, t = feed(det, 0, [100] * 10 + ramp(100, 150, 8) + [150] * 30)
    assert fired == 1


def test_slow_approach_does_not_fire():
    # 3초에 걸쳐 천천히 다가오는 건 밀기가 아님
    det = PushDetector(CFG)
    fired, _ = feed(det, 0, ramp(100, 160, 90))
    assert fired == 0


def test_jitter_does_not_fire():
    det = PushDetector(CFG)
    fired, _ = feed(det, 0, [100, 110, 95, 105, 98, 112] * 10)
    assert fired == 0


def test_not_fist_resets():
    det = PushDetector(CFG)
    feed(det, 0, [100] * 5)
    fired, _ = feed(det, 165, [None] * 10 + [130] * 5)  # 주먹이 끊긴 뒤 큰 크기로 다시 나타남
    assert fired == 0


def test_rearm_after_retract():
    det = PushDetector(CFG)
    sizes = [100] * 5 + ramp(100, 150, 6) + [150] * 25 + ramp(150, 100, 8) + [100] * 5 + ramp(100, 150, 6)
    fired, _ = feed(det, 0, sizes)
    assert fired == 2


def test_no_rearm_without_retract():
    # 밀고 나서 계속 앞으로 더 밀어도 재발동하지 않음
    det = PushDetector(CFG)
    fired, _ = feed(det, 0, [100] * 5 + ramp(100, 150, 6) + ramp(150, 260, 60))
    assert fired == 1


def test_rearm_after_fist_released():
    det = PushDetector(CFG)
    fired, t = feed(det, 0, [100] * 5 + ramp(100, 150, 6))
    assert fired == 1
    fired, t = feed(det, t, [None] * 25)  # 쿨다운 넘게 주먹을 풂
    fired, _ = feed(det, t, [100] * 5 + ramp(100, 150, 6))
    assert fired == 1
